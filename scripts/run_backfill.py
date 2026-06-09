r"""
run_backfill.py — Phase 3 / T1 orchestrator: priority rolling concall backfill.

Drives the existing, per-company fetch+enqueue primitive
(`backfill_company_docs.backfill`) across the universe in PRIORITY ORDER, so the
most relevant names get their history first even though the full universe takes
weeks at free-tier. It only fetches PDFs and appends `status=pending` rows to the
SAME `processing_queue.parquet` that Phase 2 uses — extraction is then done by:

    python scripts/extract_concall.py --backfill

(which uses the dedicated BACKFILL_GEMINI_KEY* pool, processes oldest->newest so
GF_TRACK accrues, and routes its digest to daily_backfill_*.md). This script
spends NO Gemini quota — it is Screener/BSE scraping + Drive writes only.

Priority order (highest first):
  1. Explicit --symbols / --token (process just those).
  2. Strong names — signals/aggregated/conviction.csv (>=2 strategies agree),
     already sorted by conviction.
  3. Long tail — remaining universe by market cap desc (universe/market_cap.csv),
     else master_list order.
  (Portfolio-holdings-first is intended but its source file is not yet confirmed;
   layer it in once that path is known — see TODO below.)

Usage:
    python scripts/run_backfill.py --dry-run                 # list plan, no writes
    python scripts/run_backfill.py --symbols TCS,INFY
    python scripts/run_backfill.py --token "venus remedies"
    python scripts/run_backfill.py --quarters 4 --max-companies 50
    python scripts/run_backfill.py --quarters 8 --start 50 --max-companies 50
"""
from __future__ import annotations

import argparse
import io
import os
import sys
import time
from datetime import datetime
from pathlib import Path

# Ensure scripts/ on sys.path whether run from repo root or scripts/.
_SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

import pandas as pd
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(_SCRIPTS_DIR), ".env"))

# Reuse Phase 2 Drive helpers + the per-company backfill primitive verbatim, so
# the Drive layout and queue schema stay identical to the live pipeline.
import backfill_company_docs as bcd
from ingest_company_docs import (
    get_drive, get_or_create_subfolder, find_file, download_bytes,
)
from _extractor_base import load_portfolio_isins   # T1-finish: portfolio-first tier


def log(msg: str) -> None:
    print(f"[{datetime.now():%H:%M:%S}] {msg}")


# --------------------------------------------------------------------------- #
#  Drive CSV readers (reuse ingest helpers; no new auth path)
# --------------------------------------------------------------------------- #
def _read_csv_from(drive, parent_id, *path) -> pd.DataFrame:
    """Read a CSV at <parent_id>/<path...>; empty DataFrame if any hop missing."""
    fid = parent_id
    for name in path[:-1]:
        fid = get_or_create_subfolder(drive, fid, name)
    file_id = find_file(drive, fid, path[-1])
    if not file_id:
        return pd.DataFrame()
    try:
        return pd.read_csv(io.BytesIO(download_bytes(drive, file_id)))
    except Exception as e:
        log(f"  WARNING: could not read {'/'.join(path)} ({str(e)[:80]})")
        return pd.DataFrame()


def _load_universe_df(drive, root_id) -> pd.DataFrame:
    """Full listed universe — mirror company_deep_report: prefer the Phase-2
    `company_repo/_index/company_universe.csv` (NSE+SME+BSE; column `nse_symbol`),
    fall back to `universe/master_list.csv` (Phase-1, NSE-only, column `symbol`).
    Normalises to a `symbol` + `name` column so the rest of the code is uniform.
    Using _index (which Phase 2 reads/writes every CI run) fixes the CI path issue."""
    df = _read_csv_from(drive, root_id, "company_repo", "_index", "company_universe.csv")
    if df.empty:
        df = _read_csv_from(drive, root_id, "universe", "master_list.csv")
    if df.empty:
        return df
    if "symbol" not in df.columns and "nse_symbol" in df.columns:
        df = df.rename(columns={"nse_symbol": "symbol"})
    if "name" not in df.columns:
        for c in ("company", "company_name"):
            if c in df.columns:
                df = df.rename(columns={c: "name"}); break
    return df


# --------------------------------------------------------------------------- #
#  Build the priority-ordered company list
# --------------------------------------------------------------------------- #
def build_company_order(drive, root_id) -> list[dict]:
    """Return ordered list of {symbol, isin, name}:
    portfolio holdings first → strong (conviction) names → market-cap tail."""
    universe = _load_universe_df(drive, root_id)
    if universe.empty or "symbol" not in universe.columns:
        log("  ERROR: no usable universe (company_universe.csv / master_list.csv).")
        return []
    universe["symbol"] = universe["symbol"].astype(str).str.strip()
    universe = universe[~universe["symbol"].str.lower().isin(["", "nan", "none"])]
    if "isin" not in universe.columns:
        universe["isin"] = ""
    if "name" not in universe.columns:
        universe["name"] = universe["symbol"]
    sym2row = {r["symbol"]: {"symbol": r["symbol"],
                             "isin": str(r.get("isin") or ""),
                             "name": str(r.get("name") or r["symbol"])}
               for _, r in universe.iterrows()}
    isin2row = {row["isin"]: row for row in sym2row.values() if row["isin"]}

    ordered: list[dict] = []
    seen: set[str] = set()

    # 1) Portfolio holdings first — reuse the shared portfolio reader (the same one
    #    the Phase 2 extractors use). Returns None when there's no portfolio file.
    pf_isins = load_portfolio_isins(drive, root_id)
    if pf_isins:
        for isin in pf_isins:
            row = isin2row.get(str(isin).strip())
            if row and row["symbol"] not in seen:
                ordered.append(row); seen.add(row["symbol"])
        log(f"  Portfolio names queued first: {len(ordered)}")
    else:
        log("  No portfolio file — skipping portfolio tier.")

    # 2) Strong names from conviction.csv (already conviction-sorted), de-duped.
    n_pf = len(ordered)
    conviction = _read_csv_from(drive, root_id, "signals", "aggregated", "conviction.csv")
    if not conviction.empty and "symbol" in conviction.columns:
        for sym in conviction["symbol"].astype(str).str.strip():
            if sym in sym2row and sym not in seen:
                ordered.append(sym2row[sym]); seen.add(sym)
        log(f"  Strong (conviction) names queued next: {len(ordered) - n_pf}")
    else:
        log("  No conviction.csv yet — skipping strong tier.")

    # 3) Long tail: remaining universe by market cap desc, else master_list order.
    mcap = _read_csv_from(drive, root_id, "universe", "market_cap.csv")
    remaining = [sym2row[s] for s in sym2row if s not in seen]
    if not mcap.empty and {"symbol", "market_cap_cr"}.issubset(mcap.columns):
        cap = dict(zip(mcap["symbol"].astype(str).str.strip(),
                       pd.to_numeric(mcap["market_cap_cr"], errors="coerce").fillna(0)))
        remaining.sort(key=lambda r: cap.get(r["symbol"], 0.0), reverse=True)
        log(f"  Long tail ordered by market cap: {len(remaining)}")
    else:
        log(f"  No market_cap.csv — long tail in master_list order: {len(remaining)}")

    return ordered + remaining


def resolve_explicit(drive, root_id, symbols: list[str], token: str) -> list[dict]:
    """Resolve explicit --symbols / --token to {symbol, isin, name} rows."""
    out: list[dict] = []
    universe = _load_universe_df(drive, root_id)
    by_sym = {}
    if not universe.empty and "symbol" in universe.columns:
        for _, r in universe.iterrows():
            by_sym[str(r["symbol"]).strip().upper()] = {
                "symbol": str(r["symbol"]).strip(),
                "isin": str(r.get("isin") or ""),
                "name": str(r.get("name") or r["symbol"]),
            }
    for s in symbols:
        s = s.strip()
        if not s:
            continue
        row = by_sym.get(s.upper(), {"symbol": s.upper(), "isin": "", "name": s.upper()})
        out.append(row)
    if token:
        # reuse the deep-dive resolver (name / NSE / BSE / ISIN -> symbol+isin)
        try:
            import company_deep_report as cdr
            svc = cdr.drive_service(); root = os.environ["GDRIVE_FOLDER_ID"]
            uni = cdr._read_csv(svc, cdr.DRIVE["universe"], root)
            r_isin, r_symbol, r_name, _ = cdr.resolve_isin(token, uni)
            if r_symbol:
                out.append({"symbol": r_symbol.upper(), "isin": r_isin or "",
                            "name": r_name or r_symbol})
        except Exception as e:
            log(f"  token resolution failed for '{token}': {str(e)[:80]}")
    return out


# --------------------------------------------------------------------------- #
#  Main
# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--symbols", default="",
                    help="Comma list of NSE symbols to backfill (skips priority list).")
    ap.add_argument("--token", default="",
                    help="Single company name / NSE / BSE / ISIN (resolved via universe).")
    ap.add_argument("--quarters", type=int, default=4,
                    help="Newest N concalls per company (default 4 = ~last 4 quarters).")
    ap.add_argument("--types", default="concall",
                    help="Doc types to fetch (default concall). "
                         "annual_report,rating also valid.")
    ap.add_argument("--max-companies", type=int, default=0,
                    help="Process at most N companies this run (0 = all).")
    ap.add_argument("--start", type=int, default=0,
                    help="Skip the first N companies in the priority list (resume).")
    ap.add_argument("--sleep", type=float, default=1.0,
                    help="Seconds to sleep between companies (polite to Screener).")
    ap.add_argument("--dry-run", action="store_true",
                    help="List the plan and per-company docs found; no Drive writes.")
    args = ap.parse_args()

    want_types = {t.strip() for t in args.types.split(",") if t.strip()}

    print("Phase 3 / T1 — priority rolling concall backfill")
    print("-" * 60)

    drive = repo_id = index_id = None
    if not args.dry_run:
        drive = get_drive()
        root_id = os.environ["GDRIVE_FOLDER_ID"]
        repo_id = get_or_create_subfolder(drive, root_id, "company_repo")
        index_id = get_or_create_subfolder(drive, repo_id, "_index")
    else:
        # dry-run still needs Drive to read the universe / conviction lists
        drive = get_drive()
        root_id = os.environ["GDRIVE_FOLDER_ID"]

    # Build the ordered company list
    explicit = [s for s in args.symbols.split(",") if s.strip()]
    if explicit or args.token:
        companies = resolve_explicit(drive, root_id, explicit, args.token)
    else:
        companies = build_company_order(drive, root_id)

    if not companies:
        sys.exit("No companies resolved — check universe/master_list.csv.")

    # Apply resume offset + cap
    companies = companies[args.start:]
    if args.max_companies:
        companies = companies[: args.max_companies]

    log(f"Companies this run: {len(companies)} "
        f"(start={args.start}, quarters={args.quarters}, types={sorted(want_types)})")

    totals = {"companies": 0, "found": 0, "new": 0, "downloaded": 0,
              "dup": 0, "download_fail": 0, "errors": 0}

    for i, co in enumerate(companies, 1):
        sym, isin, name = co["symbol"], co["isin"], co["name"]
        log(f"[{i}/{len(companies)}] {sym}  ({name[:40]})  ISIN={isin or '?'}")
        try:
            counts = bcd.backfill(
                symbol=sym, isin=isin, want_types=want_types,
                max_docs=args.quarters, dry_run=args.dry_run,
                drive=drive, repo_id=repo_id, index_id=index_id,
            )
        except Exception as e:
            log(f"  ! backfill error for {sym}: {str(e)[:100]}")
            totals["errors"] += 1
            continue
        totals["companies"] += 1
        for k in ("found", "new", "downloaded", "dup", "download_fail"):
            totals[k] += int(counts.get(k, 0))
        if args.sleep and i < len(companies):
            time.sleep(args.sleep)

    print("-" * 60)
    print(f"Companies processed : {totals['companies']}")
    print(f"Docs found          : {totals['found']}")
    if not args.dry_run:
        print(f"Already queued (dup): {totals['dup']}")
        print(f"New queued          : {totals['new']}")
        print(f"  PDFs downloaded   : {totals['downloaded']}")
        print(f"  download failures : {totals['download_fail']}")
    print(f"Company errors      : {totals['errors']}")
    print()
    print("Next: process the queued backfill concalls with the dedicated key pool:")
    print("    python scripts/extract_concall.py --backfill")


if __name__ == "__main__":
    main()
