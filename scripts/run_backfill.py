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
import atexit
import io
import os
import sys
import time
from datetime import datetime, timedelta, date
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
    load_queue, save_queue,
)
from _extractor_base import (
    load_portfolio_isins,          # T1-finish: portfolio-first tier
    acquire_lock, release_lock,    # T12: shared _extract.lock (Phase-2 safety)
)
from backfill_coverage import (    # T12: window-aware derived coverage view
    build_coverage, coverage_lookup, save_coverage,
)

# T12: doc types the per-company Screener #documents primitive can actually fetch
# (SUBSECTION_TYPES in backfill_company_docs). results/presentation are NOT on that
# page — the live results scraper / presentation feed own them, so backfill never
# fetches them here. Loop order: concall first (user priority), then AR, then rating.
TYPE_ORDER = ["concall", "annual_report", "rating"]
_LOCK_NAME = "_extract.lock"       # SAME lock extract_concall uses → cross-phase mutex
_LOCK_MAX_AGE_MIN = 360


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
    if "bse_code" not in df.columns:
        for c in ("scrip_code", "bsecode", "bse"):
            if c in df.columns:
                df = df.rename(columns={c: "bse_code"}); break
    return df


def _screener_token(row) -> str:
    """Screener-resolvable token for a universe row: NSE symbol if present, else
    BSE scrip code (Screener accepts /company/<bse_code>/). '' if neither."""
    nse = str(row.get("symbol") or "").strip()
    if nse and nse.lower() not in ("nan", "none"):
        return nse
    bse = str(row.get("bse_code") or "").strip()
    if bse and bse.replace(".", "").isdigit():
        return str(int(float(bse)))
    return ""


# --------------------------------------------------------------------------- #
#  Build the priority-ordered company list
# --------------------------------------------------------------------------- #
def build_company_order(drive, root_id) -> list[dict]:
    """Return ordered list of {symbol, isin, name}:
    portfolio holdings first → strong (conviction) names → market-cap tail."""
    universe = _load_universe_df(drive, root_id)
    if universe.empty:
        log("  ERROR: no usable universe (company_universe.csv / master_list.csv).")
        return []
    for c in ("isin", "symbol", "name"):
        if c not in universe.columns:
            universe[c] = ""
    # ALL listed companies: NSE by symbol, BSE-only by bse_code (Screener token).
    sym2row: dict[str, dict] = {}
    for _, r in universe.iterrows():
        tok = _screener_token(r)
        if not tok or tok in sym2row:
            continue
        sym2row[tok] = {"symbol": tok, "isin": str(r.get("isin") or ""),
                        "name": str(r.get("name") or tok)}
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
#  Timeframe + window-aware skip predicate (T12)
# --------------------------------------------------------------------------- #
_DAYS_PER_QUARTER = 92
# A request only counts as "deeper" (worth re-opening the page even if recently
# fetched) when its floor reaches more than this many days OLDER than the oldest
# doc we already hold. Guards against perpetual re-fetch when a company simply has
# no document as old as the rolling floor (its earliest is newer than the floor).
_DEEPER_MARGIN_DAYS = 180


def _since_for(doc_type: str, args) -> str | None:
    """Date floor (ISO) for this doc type given the timeframe args, or None for
    '--all' (no floor). concall/results/presentation use --quarters; AR/rating use
    --years. An explicit --since overrides both."""
    if args.all:
        return None
    if args.since:
        return str(args.since)[:10]
    today = date.today()
    if doc_type in ("annual_report", "rating"):
        return (today - timedelta(days=365 * max(1, args.years))).isoformat()
    # concall / results / presentation
    return (today - timedelta(days=_DAYS_PER_QUARTER * max(1, args.quarters))).isoformat()


def _needs_fetch(cov_row: dict | None, since_floor: str | None,
                 fetch_all: bool, refetch_days: int) -> tuple[bool, str]:
    """Decide whether to re-open a company's Screener page for (doc_type).
    Returns (needs_fetch, reason).

    Gates, in order:
      • no coverage row            -> fetch (never seen)
      • genuinely DEEPER request   -> fetch (bypasses recency) — the floor reaches
        >180d older than our oldest held doc, so real older history is missing
      • checked within refetch_days-> SKIP (covered + fresh); also prevents churn
        when the company simply has no doc as old as the rolling floor
      • otherwise (stale)          -> fetch"""
    if fetch_all:
        return True, "all"                       # explicit full re-scan override
    if not cov_row:
        return True, "never-fetched"             # no coverage row yet

    # Deeper-window: requested floor reaches meaningfully older than our oldest doc.
    cov_earliest = str(cov_row.get("covered_earliest_date") or "")
    if since_floor and cov_earliest:
        gap_days = (pd.to_datetime(cov_earliest)
                    - pd.to_datetime(since_floor)).days
        if gap_days > _DEEPER_MARGIN_DAYS:
            return True, "deeper-window"

    # Recency: did we check this page within the refetch window?
    last = pd.to_datetime(cov_row.get("last_fetched_at"), errors="coerce")
    if pd.isna(last):
        return True, "stale-no-date"
    age_days = (datetime.now() - last.to_pydatetime()).days
    if age_days > refetch_days:
        return True, f"stale-{age_days}d"
    return False, "covered"


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
    ap.add_argument("--types", default="concall",
                    help="Doc types to fetch, comma list (default concall). "
                         "Valid: concall, annual_report, rating. Fetched in the "
                         "order concall -> annual_report -> rating.")
    ap.add_argument("--quarters", type=int, default=8,
                    help="Concall/results/presentation history depth in quarters "
                         "(default 8 = ~2 years). Maps to a --since date floor.")
    ap.add_argument("--years", type=int, default=5,
                    help="Annual-report/rating history depth in years (default 5).")
    ap.add_argument("--since", default="",
                    help="Explicit ISO date floor (YYYY-MM-DD); overrides "
                         "--quarters/--years for ALL types.")
    ap.add_argument("--all", action="store_true",
                    help="Full re-scan: no date floor, ignore coverage (fetch every "
                         "company's entire history). Manual/deep use only.")
    ap.add_argument("--max-companies", type=int, default=0,
                    help="Cap companies actually FETCHED this run (0 = no cap). "
                         "Covered companies self-skip and do NOT count against it.")
    ap.add_argument("--refetch-days", type=int, default=30,
                    help="Skip a covered company's page if fetched within this many "
                         "days (default 30). New quarters are caught by the live feed.")
    ap.add_argument("--sleep", type=float, default=1.0,
                    help="Seconds to sleep between companies (polite to Screener).")
    ap.add_argument("--no-lock", action="store_true",
                    help="Skip the shared _extract.lock (testing only).")
    ap.add_argument("--dry-run", action="store_true",
                    help="List the plan and per-company decisions; no Drive writes.")
    args = ap.parse_args()

    want_types = [t for t in TYPE_ORDER
                  if t in {x.strip() for x in args.types.split(",") if x.strip()}]
    if not want_types:
        sys.exit(f"--types must include one of {TYPE_ORDER} (got '{args.types}').")

    print("T12 — cursor-free, coverage-driven backfill")
    print("-" * 60)

    # Drive is needed in BOTH modes (to read the universe + queue for the plan).
    # company_repo/_index always exists; resolving it is effectively read-only.
    # Writes below are gated on `not args.dry_run`; dry-run never mutates Drive.
    drive = get_drive()
    root_id = os.environ["GDRIVE_FOLDER_ID"]
    repo_id = get_or_create_subfolder(drive, root_id, "company_repo")
    index_id = get_or_create_subfolder(drive, repo_id, "_index")

    # Phase-2 safety (T12): take the SAME _extract.lock the extractors use, so a
    # backfill fetch can never write the queue while Phase 2 is mid-extraction.
    # On contention exit cleanly — the next slot retries (cursor-free: no progress
    # lost). Released on process exit via atexit.
    if not args.dry_run and not args.no_lock:
        if not acquire_lock(drive, index_id, _LOCK_NAME, "run_backfill",
                            max_age_min=_LOCK_MAX_AGE_MIN):
            log("Another extraction/fetch holds _extract.lock — exiting cleanly.")
            sys.exit(0)
        atexit.register(release_lock, drive, index_id, _LOCK_NAME)

    # QUEUE MIGRATION (LIVE, unchanged): re-tag pre-source-tag blank-source pending
    # rows to 'backfill' so Phase 2's live extractor does not pick them up.
    if not args.dry_run:
        try:
            queue = load_queue(drive, index_id)
            if not queue.empty:
                if "source" not in queue.columns:
                    queue["source"] = ""
                blank_mask = (
                    (queue["source"].astype(str).str.strip() == "")
                    | (queue["source"].astype(str).str.lower() == "nan")
                    | (queue["source"].astype(str).str.lower() == "none")
                )
                pending_blank = blank_mask & (queue["status"] == "pending")
                n_retagged = int(pending_blank.sum())
                if n_retagged:
                    queue.loc[pending_blank, "source"] = "backfill"
                    save_queue(drive, index_id, queue)
                    log(f"Queue migration: re-tagged {n_retagged} pending rows "
                        f"(no source) -> 'backfill'.")
        except Exception as _e:
            log(f"  WARNING: queue migration step failed ({str(_e)[:80]}) — continuing.")

    # Build the ordered company list (priority: portfolio -> conviction -> tail).
    explicit = [s for s in args.symbols.split(",") if s.strip()]
    force_explicit = bool(explicit or args.token)   # named names always fetch
    if force_explicit:
        companies = resolve_explicit(drive, root_id, explicit, args.token)
    else:
        companies = build_company_order(drive, root_id)
    if not companies:
        sys.exit("No companies resolved — check universe/master_list.csv.")

    # Build the CURRENT coverage view from the live queue (read-only; the same in
    # dry-run). This is what makes the walk cursor-free: covered (company,doc_type)
    # pairs self-skip, so each run spends its --max-companies budget on the tail.
    cov_index: dict = {}
    try:
        cov_index = coverage_lookup(build_coverage(load_queue(drive, index_id)))
        log(f"Coverage view: {len(cov_index)} (company,doc_type) pairs known.")
    except Exception as _e:
        log(f"  WARNING: could not build coverage view ({str(_e)[:80]}) — "
            f"treating all as uncovered.")

    cap = args.max_companies or 0
    fetched = 0
    totals = {"fetched": 0, "found": 0, "new": 0, "downloaded": 0,
              "dup": 0, "download_fail": 0, "errors": 0,
              "skipped_covered": 0}

    log(f"Plan: types={want_types}, quarters={args.quarters}, years={args.years}, "
        f"since={args.since or '-'}, all={args.all}, "
        f"max-companies={cap or 'all'}, refetch-days={args.refetch_days}")

    stop = False
    for doc_type in want_types:
        if stop:
            break
        floor = _since_for(doc_type, args)
        log(f"=== {doc_type}: since-floor {floor or '(all history)'} ===")
        for co in companies:
            if cap and fetched >= cap:
                log(f"  reached --max-companies={cap} — stopping walk.")
                stop = True
                break
            sym, isin, name = co["symbol"], co["isin"], co["name"]
            key = isin if isin else sym
            if force_explicit:
                need, why = True, "explicit"     # user named it -> always fetch
            else:
                need, why = _needs_fetch(cov_index.get((key, doc_type)), floor,
                                         args.all, args.refetch_days)
            if not need:
                totals["skipped_covered"] += 1
                continue
            fetched += 1
            log(f"[fetch {fetched}{('/' + str(cap)) if cap else ''}] {doc_type} "
                f"{sym} ({name[:34]}) ISIN={isin or '?'}  [{why}]")
            try:
                counts = bcd.backfill(
                    symbol=sym, isin=isin, want_types={doc_type},
                    max_docs=0, since=floor, dry_run=args.dry_run,
                    drive=drive, repo_id=repo_id, index_id=index_id,
                )
            except Exception as e:
                log(f"  ! backfill error for {sym}: {str(e)[:100]}")
                totals["errors"] += 1
                continue
            totals["fetched"] += 1
            for k in ("found", "new", "downloaded", "dup", "download_fail"):
                totals[k] += int(counts.get(k, 0))
            if args.sleep:
                time.sleep(args.sleep)

    # Refresh the derived coverage ledger from the now-updated queue.
    if not args.dry_run and index_id and totals["fetched"]:
        try:
            save_coverage(drive, index_id, build_coverage(load_queue(drive, index_id)))
            log("Coverage ledger refreshed (backfill_coverage.parquet).")
        except Exception as _e:
            log(f"  WARNING: coverage refresh failed ({str(_e)[:80]}).")

    print("-" * 60)
    print(f"Companies fetched   : {totals['fetched']}  "
          f"(skipped, already covered: {totals['skipped_covered']})")
    print(f"Docs found          : {totals['found']}")
    if not args.dry_run:
        print(f"Already queued (dup): {totals['dup']}")
        print(f"New queued          : {totals['new']}")
        print(f"  PDFs downloaded   : {totals['downloaded']}")
        print(f"  download failures : {totals['download_fail']}")
    print(f"Company errors      : {totals['errors']}")
    print()
    print("Next: process the queued backfill docs with the dedicated key pool:")
    print("    python scripts/extract_concall.py --backfill")


if __name__ == "__main__":
    main()
