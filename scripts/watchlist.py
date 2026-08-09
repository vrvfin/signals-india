r"""
watchlist.py — the SHARED "names that matter" set for the tech-page enrichment
phases (docs, news, reasons, narrative). Phase 1 of the enrichment plan
(.claude/plans/tech-page-enrichment.md).

WL = (top-N by aggregated signal)  ∪  (PF holdings)

    build_watchlist(drive, top_n=500) -> DataFrame
        [isin, symbol, name, in_signal, signal_rank, signal_score, in_pf, source]

Run directly for a READ-ONLY dry-run: builds WL and cross-references the ONE
processing_queue (rule 7) to show, per doc_type, what is already done/pending vs
what the daily watchlist pull WOULD fetch. No Screener calls, no Drive writes.

    python scripts/watchlist.py                  # full dry-run report
    python scripts/watchlist.py --top 500        # change signal cutoff
    python scripts/watchlist.py --show 40         # list more "would-fetch" names
"""
from __future__ import annotations

import argparse
import io
import os
import sys

_SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

import pandas as pd

# Shared Drive helpers (rule 4 — never raw Drive calls in new scripts).
from _extractor_base import (get_drive, get_or_create_subfolder, find_file,
                             download_bytes, log)

# doc_types the watchlist pull is responsible for (mirror ingest_company_docs FEEDS)
DOC_TYPES = ["concall", "results", "presentation", "rating", "annual_report"]


def _folder(drive, parts: str) -> str:
    fid = os.environ["GDRIVE_FOLDER_ID"]
    for p in parts.split("/"):
        fid = get_or_create_subfolder(drive, fid, p)
    return fid


def _clean(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip()


def _read_csv(drive, folder_id: str, name: str) -> pd.DataFrame:
    fid = find_file(drive, folder_id, name)
    if not fid:
        return pd.DataFrame()
    return pd.read_csv(io.BytesIO(download_bytes(drive, fid)))


def _read_parquet(drive, folder_id: str, name: str) -> pd.DataFrame:
    fid = find_file(drive, folder_id, name)
    if not fid:
        return pd.DataFrame()
    return pd.read_parquet(io.BytesIO(download_bytes(drive, fid)))


# ---------------------------------------------------------------- signal half

def _top_signal_symbols(drive, top_n: int) -> pd.DataFrame:
    """Top-N distinct symbols from signals/aggregated/latest.csv, ranked the same
    way aggregate_signals ranks: n_strategies desc, then composite_score desc."""
    sig_fid = _folder(drive, "signals/aggregated")
    df = _read_csv(drive, sig_fid, "latest.csv")
    if df.empty or "symbol" not in df.columns:
        log("  WARNING: signals/aggregated/latest.csv missing or empty — signal half empty")
        return pd.DataFrame(columns=["symbol", "signal_score"])
    score = "composite_score" if "composite_score" in df.columns else "score"
    nstr = "n_strategies" if "n_strategies" in df.columns else None
    agg = {score: "max"}
    if nstr:
        agg[nstr] = "max"
    per = df.groupby(_clean(df["symbol"]).str.upper()).agg(agg).reset_index()
    per = per.rename(columns={"symbol": "symbol"})
    per.columns = ["symbol"] + list(per.columns[1:])
    sort_cols = ([nstr] if nstr else []) + [score]
    per = per.sort_values(sort_cols, ascending=[False] * len(sort_cols))
    per = per.head(top_n).reset_index(drop=True)
    per["signal_rank"] = per.index + 1
    per = per.rename(columns={score: "signal_score"})
    return per[["symbol", "signal_rank", "signal_score"]]


# -------------------------------------------------------------------- PF half

def _pf_isins(drive) -> list[str]:
    """ISINs from the latest Screener/broker export in the Drive portfolio/ folder.
    Faithful re-read of app.py's _read_portfolio_table (header row carries 'ISIN'),
    kept standalone so this CI helper never imports the Streamlit app."""
    pf_fid = _folder(drive, "portfolio")
    files = drive.files().list(
        q=f"'{pf_fid}' in parents and trashed=false",
        fields="files(id, name, modifiedTime)", orderBy="modifiedTime desc",
    ).execute().get("files", [])
    target = next((f for f in files
                   if f["name"].lower().endswith((".xls", ".xlsx", ".csv"))), None)
    if not target:
        log("  note: no portfolio export in portfolio/ — PF half empty")
        return []
    raw = download_bytes(drive, target["id"])
    fn = target["name"].lower()
    try:
        if fn.endswith(".csv"):
            raw_df = pd.read_csv(io.BytesIO(raw), header=None)
            reader = lambda h: pd.read_csv(io.BytesIO(raw), header=h)
        else:
            engine = "xlrd" if fn.endswith(".xls") else "openpyxl"
            raw_df = pd.read_excel(io.BytesIO(raw), engine=engine, header=None)
            reader = lambda h: pd.read_excel(io.BytesIO(raw), engine=engine, header=h)
        header_row = next(
            (i for i, row in raw_df.iterrows()
             if any(str(v).strip().upper() == "ISIN" for v in row.dropna())), None)
        if header_row is None:
            log(f"  WARNING: no 'ISIN' header in {target['name']} — PF half empty")
            return []
        df = reader(header_row).dropna(subset=["ISIN"])
        log(f"  PF export: {target['name']} -> {len(df)} holdings")
        return _clean(df["ISIN"]).str.upper().tolist()
    except Exception as e:
        log(f"  WARNING: could not parse PF export ({str(e)[:80]}) — PF half empty")
        return []


# ------------------------------------------------------------------ assemble

def build_watchlist(drive, top_n: int = 500) -> pd.DataFrame:
    """WL = top-N signal symbols ∪ PF holdings, resolved to [isin, symbol, name].
    Source columns flag where each name came from. Deduped by isin (symbol fallback)."""
    ml = _read_csv(drive, _folder(drive, "universe"), "master_list.csv").fillna("")

    # symbol -> (isin, name) and isin -> (symbol, name) maps from master_list
    sym2 = {}
    isin2 = {}
    if not ml.empty:
        for _, r in ml.iterrows():
            sym = str(r.get("symbol", "")).strip().upper()
            isin = str(r.get("isin", "")).strip().upper()
            name = str(r.get("name", "")).strip()
            if sym:
                sym2[sym] = (isin, name)
            if isin:
                isin2[isin] = (sym, name)

    sig = _top_signal_symbols(drive, top_n)
    pf_isins = _pf_isins(drive)

    rows: dict[str, dict] = {}  # key -> row

    def _key(isin, sym):
        return isin if isin else f"SYM:{sym}"

    for _, r in sig.iterrows():
        sym = r["symbol"]
        isin, name = sym2.get(sym, ("", ""))
        k = _key(isin, sym)
        rows.setdefault(k, {"isin": isin, "symbol": sym, "name": name,
                            "in_signal": False, "signal_rank": None,
                            "signal_score": None, "in_pf": False})
        rows[k].update(in_signal=True, signal_rank=int(r["signal_rank"]),
                       signal_score=round(float(r["signal_score"]), 2))

    for isin in pf_isins:
        sym, name = isin2.get(isin, ("", ""))
        k = _key(isin, sym)
        rows.setdefault(k, {"isin": isin, "symbol": sym, "name": name,
                            "in_signal": False, "signal_rank": None,
                            "signal_score": None, "in_pf": False})
        rows[k]["in_pf"] = True

    wl = pd.DataFrame(rows.values())
    if wl.empty:
        return wl
    wl["source"] = wl.apply(
        lambda r: "both" if r["in_signal"] and r["in_pf"]
        else ("signal" if r["in_signal"] else "pf"), axis=1)
    return wl.sort_values(["signal_rank"], na_position="last").reset_index(drop=True)


# ---------------------------------------------------------------- dry-run CLI

def _dry_run(top_n: int, show: int) -> None:
    drive = get_drive()
    log(f"Building watchlist (top {top_n} signal + PF)...")
    wl = build_watchlist(drive, top_n)
    if wl.empty:
        log("Watchlist is EMPTY — check signals/aggregated/latest.csv and portfolio/.")
        return

    n_both = int((wl["source"] == "both").sum())
    n_sig = int((wl["source"] == "signal").sum())
    n_pf = int((wl["source"] == "pf").sum())
    n_isin = int((_clean(wl["isin"]) != "").sum())
    print("\n" + "=" * 70)
    print(f"WATCHLIST: {len(wl)} names  (signal-only={n_sig}  pf-only={n_pf}  both={n_both})")
    print(f"  resolved to ISIN: {n_isin}/{len(wl)}   missing ISIN: {len(wl) - n_isin}")
    print("=" * 70)

    # Cross-reference the ONE processing_queue for doc coverage.
    idx = _folder(drive, "company_repo/_index")
    q = _read_parquet(drive, idx, "processing_queue.parquet")
    wl_isin = set(_clean(wl["isin"])) - {""}
    wl_sym = set(_clean(wl["symbol"]).str.upper()) - {""}

    print("\nDoc coverage of the watchlist (what the daily pull would target):")
    print(f"  {'doc_type':<14}{'WL done':>9}{'WL pending':>12}{'MISSING':>9}")
    gaps: dict[str, list] = {}
    for dt in DOC_TYPES:
        done_isin = pending_isin = set()
        if not q.empty and {"doc_type", "status"}.issubset(q.columns):
            sub = q[q["doc_type"] == dt]
            def _names(frame):
                a = set(_clean(frame.get("isin", pd.Series(dtype=str)))) & wl_isin
                b = set(_clean(frame.get("symbol", pd.Series(dtype=str))).str.upper()) & wl_sym
                return a, b
            di, ds = _names(sub[sub["status"] == "done"])
            pi, ps = _names(sub[sub["status"] == "pending"])
            done_isin, done_sym = di, ds
            pending_isin, pending_sym = pi, ps
        else:
            done_sym = pending_sym = set()
        # a WL name is "covered" for dt if it has a done OR pending row (by isin or sym)
        covered_isin = (done_isin | pending_isin)
        covered_sym = (done_sym | pending_sym)
        gap_rows = wl[~(
            wl["isin"].astype(str).str.strip().isin(covered_isin) |
            wl["symbol"].astype(str).str.upper().isin(covered_sym))]
        gaps[dt] = gap_rows
        print(f"  {dt:<14}{len(done_isin | done_sym):>9}"
              f"{len(pending_isin | pending_sym):>12}{len(gap_rows):>9}")

    # Show the concrete "would fetch" list for concall (the richest narrative source).
    head_dt = "concall"
    g = gaps[head_dt]
    print(f"\nWould FETCH {head_dt} for {len(g)} WL names with no done/pending row. "
          f"First {min(show, len(g))}:")
    for _, r in g.head(show).iterrows():
        tag = r["source"]
        rank = f"#{int(r['signal_rank'])}" if pd.notna(r["signal_rank"]) else "  -"
        print(f"  {rank:>5} [{tag:<6}] {str(r['symbol'])[:14]:<14} "
              f"{str(r['isin']):<14} {str(r['name'])[:36]}")
    print("\n(DRY RUN — read-only. No Screener calls, no Drive writes.)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=500, help="Top-N by signal (default 500).")
    ap.add_argument("--show", type=int, default=25, help="How many would-fetch names to list.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Default behaviour; present for symmetry with other scripts.")
    args = ap.parse_args()
    _dry_run(args.top, args.show)


if __name__ == "__main__":
    main()
