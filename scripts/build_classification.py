r"""
build_classification.py — T9: company -> peer -> subsector -> sector ->
segment -> industry mapping (user spec 2026-06-12; revisable over time).

The master is company_repo/_index/company_classification.csv on Drive. Levels:
  segment       <- mcap_segment (Largecap/Midcap/Smallcap/Microcap)  [deterministic]
  macro_sector  <- NSE index-list "Industry" column, by ISIN          [authoritative, free]
  sector        <- Gemini refine (NSE basic-industry level granularity)
  industry      <- Gemini refine (finer than sector)
  subsector     <- Gemini (opinion layer)
  peer_group    <- Gemini concise label; `peers` = same-label symbols
  locked        <- 1 once the user edits a row (never auto-overwritten)

Deterministic seed needs no key. --with-gemini fills sector/industry/subsector/
peer_group in batches (BACKFILL pool), capped, stalest-first, resumable — runs
in the 4h backfill slot. Respects locked rows.

Usage:
    python scripts/build_classification.py --dry-run        # seed coverage, no write
    python scripts/build_classification.py                  # write deterministic seed
    python scripts/build_classification.py --with-gemini --limit 300
"""
from __future__ import annotations

import argparse
import io
import os
import sys
from datetime import datetime

_SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(_SCRIPTS_DIR), ".env"))

from _extractor_base import (get_drive, get_or_create_subfolder, find_file,
                             download_bytes, upload_bytes, log)

CLASS_COLS = ["isin", "symbol", "name", "segment", "macro_sector", "sector",
              "industry", "subsector", "peer_group", "peers", "source",
              "confidence", "locked", "updated_at"]

UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36",
      "Accept": "*/*", "Accept-Language": "en-US,en;q=0.9"}

# NSE index CSVs carrying the Industry (macro-sector) column, by ISIN. Union
# maximises coverage of the liquid universe; the rest get Gemini/DATA_MISSING.
NSE_LISTS = [
    "https://nsearchives.nseindia.com/content/indices/ind_niftytotalmarket_list.csv",
    "https://nsearchives.nseindia.com/content/indices/ind_niftymicrocap250_list.csv",
    "https://nsearchives.nseindia.com/content/indices/ind_niftysmallcap250list.csv",
    "https://nsearchives.nseindia.com/content/indices/ind_niftymidcap150list.csv",
    "https://nsearchives.nseindia.com/content/indices/ind_nifty500list.csv",
]


def fetch_nse_macro_sector() -> dict[str, str]:
    """{ISIN -> macro_sector} unioned across the NSE index lists. Fail-soft."""
    out: dict[str, str] = {}
    for url in NSE_LISTS:
        try:
            r = requests.get(url, headers=UA, timeout=25)
            if r.status_code != 200:
                continue
            df = pd.read_csv(io.StringIO(r.text))
            ic = next((c for c in df.columns if c.lower() == "industry"), None)
            isc = next((c for c in df.columns if "isin" in c.lower()), None)
            if not ic or not isc:
                continue
            for _, r2 in df.iterrows():
                isin = str(r2[isc]).strip().upper()
                if isin and isin not in out:
                    out[str(isin)] = str(r2[ic]).strip()
        except Exception as e:
            log(f"  NSE list failed ({url.split('/')[-1]}): {str(e)[:50]}")
    log(f"NSE macro-sector map: {len(out)} ISINs")
    return out


# ------------------------------------------------------------------ #
#  Storage                                                            #
# ------------------------------------------------------------------ #

def _read_csv(drive, index_id, name):
    fid = find_file(drive, index_id, name)
    if not fid:
        return None
    try:
        return pd.read_csv(io.BytesIO(download_bytes(drive, fid)))
    except Exception:
        return None


def _read_parquet(drive, index_id, name):
    fid = find_file(drive, index_id, name)
    if not fid:
        return None
    try:
        return pd.read_parquet(io.BytesIO(download_bytes(drive, fid)))
    except Exception:
        return None


# ------------------------------------------------------------------ #
#  Gemini refine (finer + opinion levels)                            #
# ------------------------------------------------------------------ #

GEM_PROMPT = """You are classifying Indian listed companies for an equity
research system. For EACH company below assign a 4-level taxonomy. Use concise,
CONSISTENT labels so peers collapse together (e.g. all condom/contraceptive
makers share peer_group "Contraceptives"; all listed AMCs share "Asset
Management").

Reply EXACTLY one line per company, pipe-delimited, no header, no extra text:
SYMBOL|sector|industry|subsector|peer_group

Where sector is broad (e.g. Healthcare), industry narrower (Pharmaceuticals),
subsector narrower still (API / Formulations / CDMO), peer_group the tightest
comparable set (a few words). If unsure, use your best judgement; never leave
a field blank.

Companies (symbol — name — NSE macro-sector — one-line business if known):
{rows}
"""


def _build_pool():
    from gemini_pool import BucketPool, load_keys
    keys = load_keys(os.environ, prefix="BACKFILL_GEMINI_KEY")
    for p in ("GEMINI_API_KEY",):
        keys += [k for k in load_keys(os.environ, prefix=p) if k not in keys]
    if not keys:
        return None
    return BucketPool(keys, ["gemini-2.5-flash-lite", "gemini-2.0-flash-lite"],
                      inter_call_s=3.0, logger=log)


def gemini_refine(pool, batch: list[dict]) -> dict[str, tuple]:
    rows = "\n".join(
        f"{b['symbol']} — {b['name']} — {b.get('macro_sector', '?')} — "
        f"{b.get('hint', '')[:120]}" for b in batch)
    try:
        text, _ = pool.call_text(GEM_PROMPT.format(rows=rows))
    except Exception as e:
        log(f"  gemini batch failed ({str(e)[:60]})")
        return {}
    out = {}
    for line in text.strip().splitlines():
        parts = [p.strip() for p in line.split("|")]
        if len(parts) >= 5 and parts[0]:
            out[parts[0].upper()] = (parts[1], parts[2], parts[3], parts[4])
    return out


def load_classification(drive, root):
    """The classification frame (cleaned), or None. Shared accessor."""
    idx = get_or_create_subfolder(
        drive, get_or_create_subfolder(drive, root, "company_repo"), "_index")
    df = _read_csv(drive, idx, "company_classification.csv")
    return df.fillna("") if df is not None else None


def classification_block(drive, root, isin: str, sym: str = "") -> str:
    """One text block (deep dive / Ask) — the company's place in the taxonomy
    plus its peers. 'DATA_MISSING' if unclassified."""
    df = load_classification(drive, root)
    if df is None or df.empty:
        return "DATA_MISSING"
    row = df[df["isin"].astype(str) == isin]
    if row.empty and sym:
        row = df[df["symbol"].astype(str).str.upper() == sym.upper()]
    if row.empty:
        return "DATA_MISSING"
    r = row.iloc[0]
    peers = str(r.get("peers", "")).strip()
    return (f"segment={r.get('segment','?')} | macro_sector={r.get('macro_sector','?')}"
            f" | sector={r.get('sector','?')} | industry={r.get('industry','?')}"
            f" | subsector={r.get('subsector','?')} | "
            f"peer_group={r.get('peer_group','?')}"
            + (f"\npeers: {peers[:300]}" if peers and peers.lower() != "nan"
               else ""))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--with-gemini", action="store_true",
                    help="Fill sector/industry/subsector/peer_group (batched).")
    ap.add_argument("--limit", type=int, default=300,
                    help="Max companies refined this run (stalest-first).")
    ap.add_argument("--batch", type=int, default=25)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    drive = get_drive()
    root = os.environ["GDRIVE_FOLDER_ID"]
    repo_id = get_or_create_subfolder(drive, root, "company_repo")
    index_id = get_or_create_subfolder(drive, repo_id, "_index")
    today = datetime.now().strftime("%Y-%m-%d")

    uni = _read_csv(drive, index_id, "company_universe.csv")
    if uni is None or uni.empty:
        log("company_universe.csv missing — cannot build.")
        return
    sym_col = "nse_symbol" if "nse_symbol" in uni.columns else "symbol"

    # segment from market_cap.csv (symbol-keyed)
    seg = {}
    mc = _read_csv(drive, get_or_create_subfolder(drive, root, "universe"),
                   "market_cap.csv")
    if mc is not None and "mcap_segment" in mc.columns:
        seg = {str(r["symbol"]).upper(): str(r["mcap_segment"])
               for _, r in mc.iterrows()}

    macro = fetch_nse_macro_sector()

    prev = _read_csv(drive, index_id, "company_classification.csv")
    prev_by_isin = {}
    if prev is not None and not prev.empty:
        prev = prev.fillna("")            # empty CSV cells -> "" not NaN/"nan"
        prev_by_isin = {str(r["isin"]): r.to_dict() for _, r in prev.iterrows()}

    def _s(v) -> str:                     # normalise to a clean string
        s = str(v).strip()
        return "" if s.lower() in ("nan", "none") else s

    rows = []
    for _, r in uni.iterrows():
        isin = str(r.get("isin", "")).strip()
        sym = str(r.get(sym_col, "")).strip().upper()
        if not isin or not sym or sym == "NAN":
            continue
        pv = prev_by_isin.get(isin, {})
        locked = str(pv.get("locked", "0")) in ("1", "1.0", "True", "true")
        if locked:
            rows.append({c: pv.get(c) for c in CLASS_COLS})   # never touch
            continue
        rows.append({
            "isin": isin, "symbol": sym, "name": str(r.get("name", "")).strip(),
            "segment": seg.get(sym) or _s(pv.get("segment")),
            "macro_sector": macro.get(isin) or _s(pv.get("macro_sector")),
            "sector": _s(pv.get("sector")), "industry": _s(pv.get("industry")),
            "subsector": _s(pv.get("subsector")),
            "peer_group": _s(pv.get("peer_group")), "peers": _s(pv.get("peers")),
            "source": _s(pv.get("source")) or "nse+mcap",
            "confidence": _s(pv.get("confidence")),
            "locked": 0, "updated_at": _s(pv.get("updated_at")) or today,
        })
    df = pd.DataFrame(rows, columns=CLASS_COLS)
    n_macro = int((df["macro_sector"].astype(str).str.len() > 0).sum())
    n_peer = int((df["peer_group"].astype(str).str.len() > 0).sum())
    log(f"universe {len(df)}: macro_sector {n_macro}, peer_group {n_peer}, "
        f"segment {int((df['segment'].astype(str).str.len()>0).sum())}")

    if args.with_gemini:
        pool = _build_pool()
        if pool is None:
            log("no Gemini keys — skipping refine.")
        else:
            todo = df[(df["peer_group"].astype(str).str.len() == 0)
                      & (df["locked"] == 0)]
            todo = todo.sort_values("updated_at").head(args.limit)
            log(f"refining {len(todo)} companies (batch {args.batch})…")
            idx_by_sym = {r["symbol"]: i for i, r in df.iterrows()}
            for i in range(0, len(todo), args.batch):
                chunk = todo.iloc[i:i + args.batch]
                batch = [{"symbol": r["symbol"], "name": r["name"],
                          "macro_sector": r["macro_sector"]}
                         for _, r in chunk.iterrows()]
                res = gemini_refine(pool, batch)
                for sym, (sec, ind, sub, peer) in res.items():
                    j = idx_by_sym.get(sym)
                    if j is None:
                        continue
                    df.at[j, "sector"] = sec
                    df.at[j, "industry"] = ind
                    df.at[j, "subsector"] = sub
                    df.at[j, "peer_group"] = peer
                    df.at[j, "source"] = "nse+mcap+gemini"
                    df.at[j, "updated_at"] = today

    # peers = same peer_group symbols (computed every run)
    grp = df.groupby(df["peer_group"].astype(str))
    for g, members in grp:
        if not g or g == "nan":
            continue
        syms = list(members["symbol"])
        for j in members.index:
            df.at[j, "peers"] = ",".join(s for s in syms
                                         if s != df.at[j, "symbol"])[:500]

    if args.dry_run:
        log("DRY-RUN — sample:")
        cols = ["symbol", "segment", "macro_sector", "sector", "peer_group"]
        print(df[df["macro_sector"].astype(str).str.len() > 0][cols].head(15)
              .to_string(index=False))
        return

    upload_bytes(drive, index_id, "company_classification.csv",
                 df.to_csv(index=False).encode("utf-8"), "text/csv",
                 existing_id=find_file(drive, index_id,
                                       "company_classification.csv"))
    buf = io.BytesIO()
    df.to_parquet(buf, index=False)
    upload_bytes(drive, index_id, "company_classification.parquet",
                 buf.getvalue(), "application/octet-stream",
                 existing_id=find_file(drive, index_id,
                                       "company_classification.parquet"))
    log(f"wrote company_classification.csv/.parquet ({len(df)} rows).")


if __name__ == "__main__":
    main()
