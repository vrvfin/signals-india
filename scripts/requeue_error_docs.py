r"""
requeue_error_docs.py — one-off recovery for stuck 'error' queue rows (user 2026-07-12).

WHY: the nightly backfill retries rows whose status is download_failed or expired
(backfill_company_docs.py retry_mask) but NEVER 'error' — so a doc whose EXTRACTION
failed (Gemini quota mid-run, bad parse, …) is stuck forever. 153 FY2025-26 annual
reports for PF companies are in this state; their raw PDFs are already cleaned up.

WHAT: flips matching 'error' rows to 'expired' — semantically true (the PDF is gone)
and exactly the status the EXISTING backfill retry machinery re-fetches + re-extracts
on its next nightly pass. No new pipeline; one bounded status flip under the shared
_extract.lock (CLAUDE.md rule 4/7 — the ONE global queue, lock before write).

Scope guards (all default-on):
  --doc-type annual_report        only this doc type
  --pf-only                       only portfolio companies (default true)
  --min-fy-year <current-1>       only recent FY-ends (skip 1997-era errors)
  --max 200                       hard cap per run

Usage:
    python scripts/requeue_error_docs.py --dry-run     # list what would flip
    python scripts/requeue_error_docs.py               # flip (asks the lock first)
"""
from __future__ import annotations

import argparse
import io
import os
import sys
from datetime import date, datetime

_SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

import pandas as pd
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(_SCRIPTS_DIR), ".env"))

from _extractor_base import (get_drive, get_or_create_subfolder, find_file,
                             download_bytes, save_parquet, acquire_lock,
                             release_lock, load_portfolio_isins, log)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--doc-type", default="annual_report")
    ap.add_argument("--pf-only", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--min-fy-year", type=int, default=date.today().year - 1,
                    help="Only rows whose announcement_date year >= this (FY-end).")
    # Most errored 'annual_report' rows are legacy DRHP/rights-issue PDFs misfiled
    # under the AR doc_type (SEBI blocks bot downloads -> they error forever).
    # Default: only requeue titles that are genuinely the AR / its BRSR annexe.
    ap.add_argument("--title-like",
                    default="Financial Year|Reg. 34|Annual Report|Business Responsibility",
                    help="Regex a row's title must match ('' = no title filter).")
    ap.add_argument("--title-not",
                    default="Secretarial|DRHP|Right Issue",
                    help="Regex that excludes a title ('' = exclude nothing).")
    ap.add_argument("--max", type=int, default=200, help="Hard cap per run.")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    print("Requeue stuck 'error' docs -> 'expired' (backfill re-fetches them)")
    print("-" * 60)
    drive = get_drive()
    root_id = os.environ["GDRIVE_FOLDER_ID"]
    repo_id = get_or_create_subfolder(drive, root_id, "company_repo")
    index_id = get_or_create_subfolder(drive, repo_id, "_index")

    qfid = find_file(drive, index_id, "processing_queue.parquet")
    if not qfid:
        sys.exit("processing_queue.parquet missing.")
    # FULL frame (never load_parquet here — a column subset would truncate the
    # queue schema on save; same rule as cleanup_company_docs.mark_expired_rows)
    q = pd.read_parquet(io.BytesIO(download_bytes(drive, qfid)))

    mask = (q["status"].astype(str) == "error") \
        & (q["doc_type"].astype(str) == args.doc_type)
    yr = pd.to_numeric(q["announcement_date"].astype(str).str[:4], errors="coerce")
    mask &= yr >= args.min_fy_year
    t = q["title"].astype(str)
    if args.title_like:
        mask &= t.str.contains(args.title_like, case=False, regex=True, na=False)
    if args.title_not:
        mask &= ~t.str.contains(args.title_not, case=False, regex=True, na=False)
    if args.pf_only:
        pf = load_portfolio_isins(drive, root_id) or set()
        mask &= q["isin"].astype(str).isin(pf)
    idx = q.index[mask][:args.max]
    sel = q.loc[idx]
    log(f"matching stuck rows: {int(mask.sum())} · flipping (cap {args.max}): {len(sel)}")
    if sel.empty:
        return
    per_co = sel.groupby(sel["symbol"].astype(str)).size().sort_values(ascending=False)
    log("by company (top 15):\n" + per_co.head(15).to_string())

    if args.dry_run:
        print("\nDRY RUN — no queue write. Sample rows:")
        cols = ["symbol", "doc_type", "announcement_date", "title", "status"]
        print(sel[cols].head(12).to_string(index=False))
        return

    if not acquire_lock(drive, index_id, "_extract.lock", "requeue"):
        sys.exit("queue busy (_extract.lock) — try again later.")
    try:
        # re-read under the lock so we never clobber a concurrent writer
        q = pd.read_parquet(io.BytesIO(download_bytes(
            drive, find_file(drive, index_id, "processing_queue.parquet"))))
        doc_ids = set(sel["doc_id"].astype(str))
        hit = (q["status"].astype(str) == "error") \
            & q["doc_id"].astype(str).isin(doc_ids)
        q.loc[hit, "status"] = "expired"
        save_parquet(drive, index_id, "processing_queue.parquet", q)
        log(f"queue updated: {int(hit.sum())} row(s) error -> expired "
            f"(backfill re-fetches on its next nightly pass).")
    finally:
        release_lock(drive, index_id, "_extract.lock")


if __name__ == "__main__":
    main()
