#!/usr/bin/env python3
"""requeue_documents.py — re-queue named documents, optionally under a different origin.

WHY THIS EXISTS
  Sometimes a specific document has to be read again: its stored analysis is a failed
  generation, or it was processed by the wrong pipeline. requeue_thin_ars.py does this
  for annual reports it judges thin; this does it for documents named EXPLICITLY, of any
  type, and can move them between origins.

  Moving the origin is the point. `source` decides which extractor drains a row:

      "backfill"  -> extract_concall.py --backfill, writing daily_backfill_<date>.md
      anything    -> the live run, writing concall_<date>.md
      else

  So re-queueing a concall as "backfill" sends its re-read down the backfill path and
  keeps its output out of the live daily digest — which is what you want when you are
  repairing history rather than reporting today.

TWO SAFETY RULES THIS FOLLOWS
  1. load_queue / save_queue, NEVER load_parquet(QUEUE_COLS) + save_parquet.
     load_parquet ends in `return df[cols]` — it SLICES — so writing the result back
     erases every column the caller's list does not name. That erased four columns from
     the live queue on 2026-09-05. load_queue returns the frame unsliced.
  2. drive_file_id is CLEARED, because retention deletes the raw PDF two days after it
     is processed. Without clearing it the extractor would look for a file that is gone.
     Re-hydrate with pf_docs_sweep.py --hydrate before extracting.

Usage:
    python scripts/requeue_documents.py --doc-ids a,b,c --dry-run
    python scripts/requeue_documents.py --doc-ids a,b,c --as-origin backfill --live
    python scripts/requeue_documents.py --symbols AVALON --doc-type concall --dry-run
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

from _extractor_base import (get_drive, get_or_create_subfolder, load_queue,  # noqa: E402
                             save_queue, acquire_lock, release_lock, log)


def select(queue: pd.DataFrame, doc_ids: set, symbols: set,
           doc_type: str, since: str) -> pd.DataFrame:
    """Rows to re-queue. doc_ids wins; symbols+doc_type+since is the broader form."""
    if doc_ids:
        return queue[queue["doc_id"].astype(str).isin(doc_ids)]
    m = pd.Series(True, index=queue.index)
    if symbols:
        m &= queue["symbol"].astype(str).str.upper().isin({s.upper() for s in symbols})
    if doc_type:
        m &= queue["doc_type"].astype(str) == doc_type
    if since:
        m &= pd.to_datetime(queue["announcement_date"],
                            errors="coerce") >= pd.Timestamp(since)
    return queue[m]


def _self_test() -> int:
    passed, failed = 0, []

    def check(name, cond):
        nonlocal passed
        if cond:
            passed += 1
        else:
            failed.append(name)

    q = pd.DataFrame({
        "doc_id": ["a", "b", "c", "d"],
        "symbol": ["AVALON", "AVALON", "MOREPENLAB", "TCS"],
        "doc_type": ["concall", "annual_report", "concall", "concall"],
        "status": ["done", "done", "done", "pending"],
        "source": [None, None, None, "backfill"],
        "announcement_date": ["2026-08-11", "2026-03-31", "2026-08-01", "2026-01-01"],
        "drive_file_id": ["f1", "f2", "f3", "f4"],
    })
    check("doc_ids select exactly those rows",
          list(select(q, {"a", "c"}, set(), "", "")["doc_id"]) == ["a", "c"])
    check("symbol + type + since selects the right row",
          list(select(q, set(), {"avalon"}, "concall", "2026-06-01")["doc_id"]) == ["a"])
    check("a doc_type filter excludes other types",
          "b" not in list(select(q, set(), {"AVALON"}, "concall", "")["doc_id"]))
    check("no filters at all selects everything", len(select(q, set(), set(), "", "")) == 4)

    # the write itself
    sel = select(q, {"a", "c"}, set(), "", "")
    out = q.copy()
    out.loc[sel.index, "status"] = "pending"
    out.loc[sel.index, "source"] = "backfill"
    out.loc[sel.index, "drive_file_id"] = ""
    check("selected rows become pending", list(out.loc[sel.index, "status"]) ==
          ["pending", "pending"])
    check("selected rows take the new origin", list(out.loc[sel.index, "source"]) ==
          ["backfill", "backfill"])
    check("drive_file_id is cleared so the PDF is re-fetched",
          list(out.loc[sel.index, "drive_file_id"]) == ["", ""])
    check("untouched rows keep their status", out.loc[3, "status"] == "pending"
          and out.loc[1, "status"] == "done")
    check("untouched rows keep their file id", out.loc[1, "drive_file_id"] == "f2")
    check("no column is lost", list(out.columns) == list(q.columns))

    for name in failed:
        print(f"  FAIL  {name}")
    print(f"requeue_documents self-test: {passed} passed, {len(failed)} failed")
    return 1 if failed else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--doc-ids", default="", help="Comma-separated doc_id list.")
    ap.add_argument("--symbols", default="", help="Comma-separated NSE symbols.")
    ap.add_argument("--doc-type", default="", help="concall / annual_report / ...")
    ap.add_argument("--since", default="", help="Only rows announced on/after YYYY-MM-DD.")
    ap.add_argument("--as-origin", default="",
                    help="Set source to this ('backfill' routes the re-read down the "
                         "backfill path). Omit to leave source alone.")
    ap.add_argument("--limit", type=int, default=50, help="Refuse to touch more than N.")
    ap.add_argument("--live", action="store_true", help="Actually write.")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()

    if a.self_test:
        return _self_test()

    doc_ids = {s.strip() for s in a.doc_ids.split(",") if s.strip()}
    symbols = {s.strip() for s in a.symbols.split(",") if s.strip()}
    if not doc_ids and not symbols:
        log("Give --doc-ids or --symbols. Refusing to act on the whole queue.")
        return 1

    drive = get_drive()
    root = os.environ["GDRIVE_FOLDER_ID"]
    idx = get_or_create_subfolder(
        drive, get_or_create_subfolder(drive, root, "company_repo"), "_index")

    queue = load_queue(drive, idx)          # NEVER load_parquet(COLS) — see the docstring
    sel = select(queue, doc_ids, symbols, a.doc_type, a.since)
    log(f"matched {len(sel)} row(s)")
    for _, r in sel.iterrows():
        log(f"  {str(r['symbol']):12s} {str(r['doc_type']):14s} "
            f"{str(r['status']):11s} {str(r.get('announcement_date'))[:10]}  "
            f"source={str(r.get('source'))!r}  {str(r['doc_id'])[:22]}")
    if sel.empty:
        return 0
    if len(sel) > a.limit:
        log(f"refusing: {len(sel)} rows exceeds --limit {a.limit}")
        return 1
    if not a.live:
        log("DRY RUN — nothing written. Re-run with --live to apply.")
        return 0

    if not acquire_lock(drive, idx, "_extract.lock", "requeue_documents"):
        log("could not take _extract.lock — nothing written.")
        return 1
    try:
        before_cols = list(queue.columns)
        queue.loc[sel.index, "status"] = "pending"
        queue.loc[sel.index, "drive_file_id"] = ""
        if a.as_origin:
            queue.loc[sel.index, "source"] = a.as_origin.strip().lower()
        assert list(queue.columns) == before_cols, "a column would be lost"
        save_queue(drive, idx, queue)
        log(f"{len(sel)} row(s) set to pending, drive_file_id cleared"
            + (f", source={a.as_origin}" if a.as_origin else "")
            + " — run pf_docs_sweep.py --hydrate, then the extractor for that origin.")
    finally:
        release_lock(drive, idx, "_extract.lock")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
