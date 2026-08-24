#!/usr/bin/env python3
"""requeue_pf_ratings — mark PF rating documents for re-extraction.

WHY. Ratings extracted before 2026-08-21 are unreliable. Measured on live PF data: 115 of
160 rating rows held something that is not a rating at all — the agency's own name
("ICRA", "CRISIL", "CareEdge") or DATA_MISSING — and 160 rows contained ZERO downgrades,
because the parser asserted "Reaffirmed" whenever it could not find the answer. Tatva
Chintan's genuine A-/Negative -> BBB+ downgrade was stored as "D, Reaffirmed" and mailed
to the reader in that form.

Four parser defects were fixed in 0f1e4a1 and verified against two live rationales, one
per agency. NO CODE CHANGE REPAIRS THE STORED ROWS — they have to be read again.

WHAT THIS DOES. Flips `status` to `pending` for PF rating rows so extract_rating picks
them up. Nothing else: no deletion, no re-fetch, no writes to any structured table.
`upsert_structured` dedupes on source_doc_id, so re-extraction REPLACES a document's rows
rather than duplicating them.

SAFE TO STOP HALFWAY. A row left `pending` is extracted by the next nightly run anyway, so
an interrupted pass costs nothing and needs no cleanup.

Usage:
    python scripts/requeue_pf_ratings.py --dry-run
    python scripts/requeue_pf_ratings.py --limit 100
    python scripts/requeue_pf_ratings.py --symbols TATVA,APLAPOLLO
"""
from __future__ import annotations

import argparse
import os
import sys

_D = os.path.dirname(os.path.abspath(__file__))
if _D not in sys.path:
    sys.path.insert(0, _D)

import pandas as pd
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(_D), ".env"))

from _extractor_base import (get_drive, get_or_create_subfolder, load_parquet, save_parquet,
                             QUEUE_COLS, acquire_lock, release_lock, log)


def select(queue: pd.DataFrame, pf_isins: set, before: str,
           symbols: set | None = None) -> pd.DataFrame:
    """PF rating rows already processed, whose extraction predates the parser fixes.

    `error` rows are deliberately EXCLUDED. This is a re-read of documents that were
    processed WRONGLY, not a retry of ones that failed outright. The 68 error rows carry
    no recorded reason and need their own diagnosis, not a blind re-run that would burn
    quota rediscovering the same failure.
    """
    if queue is None or queue.empty:
        return pd.DataFrame(columns=QUEUE_COLS)
    q = queue[(queue["doc_type"].astype(str) == "rating")
              & (queue["status"].astype(str) == "done")
              & (queue["isin"].astype(str).str.strip().isin(pf_isins))].copy()
    if before:
        pa = pd.to_datetime(q["processed_at"], errors="coerce")
        # A row with no processed_at is old by definition — it predates the field.
        q = q[pa.isna() | (pa < pd.Timestamp(before))]
    if symbols:
        q = q[q["symbol"].astype(str).str.upper().isin({s.upper() for s in symbols})]
    # Newest first: the current rating matters more than one from 2019, and a run stopped
    # by quota should have spent it on the ratings that are actually live.
    return q.sort_values("announcement_date", ascending=False)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--symbols", type=str, default="")
    ap.add_argument("--before", type=str, default="2026-08-21",
                    help="Re-queue rows processed before this date (the parser fixes).")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    if args.self_test:
        sys.exit(_self_test())

    drive = get_drive()
    fid = os.environ["GDRIVE_FOLDER_ID"]
    repo = get_or_create_subfolder(drive, fid, "company_repo")
    idx = get_or_create_subfolder(drive, repo, "_index")

    from daily_brief import load_pf
    pf_isins = {str(t[0]).strip() for t in load_pf(drive, fid, idx)}
    queue = load_parquet(drive, idx, "processing_queue.parquet", QUEUE_COLS)
    syms = {s.strip() for s in args.symbols.split(",") if s.strip()} or None

    work = select(queue, pf_isins, args.before, syms)
    log(f"PF holdings {len(pf_isins)} | rating rows to re-read: {len(work)} "
        f"across {work['symbol'].nunique() if len(work) else 0} companies")
    if work.empty:
        log("Nothing to re-queue.")
        return
    if args.limit:
        work = work.head(args.limit)

    if args.dry_run:
        for _, r in work.head(12).iterrows():
            log(f"   WOULD RE-READ {str(r['announcement_date'])[:10]} {str(r['symbol']):<12} "
                + str(r["title"])[:50].encode("ascii", "ignore").decode())
        if len(work) > 12:
            log(f"   ... and {len(work) - 12} more")
        log(f"DRY RUN — {len(work)} row(s) would be set to pending. No writes.")
        return

    if not acquire_lock(drive, idx, "_extract.lock", "requeue_ratings",
                        max_age_min=360, wait_min=10):
        log("Could not take the lock — nothing changed.")
        sys.exit(1)
    try:
        queue = load_parquet(drive, idx, "processing_queue.parquet", QUEUE_COLS)
        ids = set(work["doc_id"].astype(str))
        hit = queue["doc_id"].astype(str).isin(ids)
        queue.loc[hit, "status"] = "pending"
        queue.loc[hit, "processed_at"] = ""
        save_parquet(drive, idx, "processing_queue.parquet", queue)
        log(f"{hit.sum()} rating row(s) set to pending — run extract_rating to re-read them.")
    finally:
        release_lock(drive, idx, "_extract.lock")


def _self_test() -> int:
    ok = fail = 0

    def check(name, cond):
        nonlocal ok, fail
        if cond:
            ok += 1
        else:
            fail += 1
            print(f"  FAIL {name}")

    def R(**kw):
        base = {"doc_id": "d1", "isin": "INE1", "symbol": "AAA", "company_name": "A",
                "doc_type": "rating", "title": "Rating update", "key": "INE1",
                "announcement_date": "2026-06-01", "pdf_url": "http://x", "description": "",
                "drive_file_id": "f1", "status": "done", "discovered_at": "",
                "processed_at": "2026-07-14T00:00:00", "attempts": 0,
                "last_error": "", "last_attempt_at": ""}
        return {**base, **kw}

    PF = {"INE1", "INE2"}
    B = "2026-08-21"
    check("basic selection", len(select(pd.DataFrame([R()]), PF, B)) == 1)
    check("non-PF excluded", select(pd.DataFrame([R(isin="INE9")]), PF, B).empty)
    check("other doc types excluded",
          select(pd.DataFrame([R(doc_type="presentation")]), PF, B).empty)
    # error rows are a different problem and are NOT swept up here
    check("error rows excluded", select(pd.DataFrame([R(status="error")]), PF, B).empty)
    check("pending rows excluded", select(pd.DataFrame([R(status="pending")]), PF, B).empty)
    # a row already re-read under the new parser must not be read a third time
    check("row processed after the fix is left alone",
          select(pd.DataFrame([R(processed_at="2026-08-22T10:00:00")]), PF, B).empty)
    check("row processed before the fix is re-queued",
          len(select(pd.DataFrame([R(processed_at="2026-08-20T10:00:00")]), PF, B)) == 1)
    check("row with no processed_at counts as old",
          len(select(pd.DataFrame([R(processed_at="")]), PF, B)) == 1)
    check("unparseable processed_at counts as old",
          len(select(pd.DataFrame([R(processed_at="not a date")]), PF, B)) == 1)
    two = pd.DataFrame([R(doc_id="old", announcement_date="2021-01-01"),
                        R(doc_id="new", announcement_date="2026-06-01")])
    check("newest first", list(select(two, PF, B)["doc_id"]) == ["new", "old"])
    mix = pd.DataFrame([R(doc_id="a", symbol="AAA"),
                        R(doc_id="b", symbol="BBB", isin="INE2")])
    check("symbol filter", list(select(mix, PF, B, {"bbb"})["doc_id"]) == ["b"])
    check("no symbol filter keeps both", len(select(mix, PF, B)) == 2)
    check("empty queue safe", select(pd.DataFrame(), PF, B).empty)
    check("none queue safe", select(None, PF, B).empty)
    check("no cutoff takes everything done", len(select(two, PF, "")) == 2)

    print(f"requeue_pf_ratings self-test: {ok} passed, {fail} failed")
    return 1 if fail else 0


if __name__ == "__main__":
    main()
