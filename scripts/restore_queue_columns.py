#!/usr/bin/env python3
"""restore_queue_columns.py — put back columns a narrowing write erased from the queue.

WHAT HAPPENED, SO THE NEXT PERSON DOES NOT REPEAT IT
  `load_parquet(drive, idx, file, cols)` ends in `return df[cols]` — it SLICES. Paired
  with `save_parquet` back to the same file, every column the caller's list does not
  name is erased for every other pipeline that relies on it.

  On 2026-09-05 a manual repair tool was run from a checkout of `main`, whose QUEUE_COLS
  has 17 entries, against a live queue that had 21. Drive's revision history records it
  exactly:

      12:22:23 UTC   21 cols   6,759 rows
      12:40:56 UTC   17 cols   6,759 rows   <- backfill_process_date, source,
                                               period, content_sha256 erased
      13:04:50 UTC   18 cols   6,760 rows   <- source restored by a later writer

  `load_queue()` returns the frame unsliced and is the safe reader; the CI extractors
  use it, which is why CI never caused this.

WHAT THIS DOES
  Reads a named pre-loss Drive REVISION of processing_queue.parquet and copies the
  missing columns back onto the CURRENT file, matched on doc_id. It is a patch, not a
  rollback: every current row, status and newer document is preserved untouched, and a
  column that already exists is never overwritten.

  Run it in CI, not locally — the whole failure came from a local copy whose column list
  had drifted from the live file.

Usage:
    python scripts/restore_queue_columns.py --revision <id> --dry-run
    python scripts/restore_queue_columns.py --revision <id> --live
    python scripts/restore_queue_columns.py --list-revisions
"""
from __future__ import annotations

import argparse
import io
import os
import sys

_D = os.path.dirname(os.path.abspath(__file__))
if _D not in sys.path:
    sys.path.insert(0, _D)

import pandas as pd
from dotenv import load_dotenv
from googleapiclient.http import MediaIoBaseUpload

load_dotenv(os.path.join(os.path.dirname(_D), ".env"))

from _extractor_base import (get_drive, get_or_create_subfolder, find_file,  # noqa: E402
                             download_bytes, acquire_lock, release_lock, log)

QUEUE_FILE = "processing_queue.parquet"


def _index(drive):
    root = os.environ["GDRIVE_FOLDER_ID"]
    return get_or_create_subfolder(
        drive, get_or_create_subfolder(drive, root, "company_repo"), "_index")


def list_revisions(drive, fid) -> None:
    revs = drive.revisions().list(
        fileId=fid, fields="revisions(id,modifiedTime,size)").execute().get(
            "revisions", [])
    log(f"{len(revs)} revision(s) retained by Drive")
    for r in revs[-25:]:
        try:
            df = pd.read_parquet(io.BytesIO(drive.revisions().get_media(
                fileId=fid, revisionId=r["id"]).execute()))
            log(f"  {r.get('modifiedTime', '?')[:19]}  {len(df.columns):>2} cols  "
                f"{len(df):>7,} rows   {r['id']}")
        except Exception as e:
            log(f"  {r.get('modifiedTime', '?')[:19]}  unreadable ({str(e)[:40]})")


def restore(drive, fid, revision: str, live: bool) -> int:
    old = pd.read_parquet(io.BytesIO(drive.revisions().get_media(
        fileId=fid, revisionId=revision).execute()))
    cur = pd.read_parquet(io.BytesIO(download_bytes(drive, fid)))
    log(f"pre-loss revision : {len(old):,} rows, {len(old.columns)} cols")
    log(f"current file      : {len(cur):,} rows, {len(cur.columns)} cols")

    missing = [c for c in old.columns if c not in cur.columns]
    if not missing:
        log("nothing to restore — the current file already has every column.")
        return 0
    log(f"columns to restore: {missing}")

    src = old.set_index(old["doc_id"].astype(str))[missing]
    src = src[~src.index.duplicated(keep="last")]
    key = cur["doc_id"].astype(str)
    for c in missing:
        cur[c] = key.map(src[c])
        log(f"  {c:24s} -> {int(cur[c].notna().sum()):>6,} of {len(cur):,} rows recovered")

    order = [c for c in old.columns if c in cur.columns] + \
            [c for c in cur.columns if c not in old.columns]
    cur = cur[order]

    # A restore must never LOSE anything either — belt and braces.
    before = pd.read_parquet(io.BytesIO(download_bytes(drive, fid)))
    assert len(cur) == len(before), f"row count changed: {len(before)} -> {len(cur)}"
    for c in before.columns:
        assert c in cur.columns, f"restore would drop {c}"
    log(f"result: {len(cur):,} rows, {len(cur.columns)} cols")

    if not live:
        log("DRY RUN — nothing written.")
        return 0

    idx = _index(drive)
    if not acquire_lock(drive, idx, "_extract.lock", "restore_queue_columns"):
        log("could not take _extract.lock — nothing written.")
        return 1
    try:
        buf = io.BytesIO()
        cur.to_parquet(buf, index=False)
        buf.seek(0)
        drive.files().update(fileId=fid, media_body=MediaIoBaseUpload(
            buf, mimetype="application/octet-stream", resumable=False)).execute()
        log("RESTORED — columns back, rows and statuses untouched.")
    finally:
        release_lock(drive, idx, "_extract.lock")
    return 0


def _self_test() -> int:
    passed, failed = 0, []

    def check(name, cond):
        nonlocal passed
        if cond:
            passed += 1
        else:
            failed.append(name)

    old = pd.DataFrame({"doc_id": ["a", "b", "c"], "status": ["done"] * 3,
                        "period": ["FY25", "FY26", None],
                        "content_sha256": ["x", "y", "z"]})
    cur = pd.DataFrame({"doc_id": ["a", "b", "c", "d"],
                        "status": ["done", "pending", "done", "done"]})
    missing = [c for c in old.columns if c not in cur.columns]
    src = old.set_index(old["doc_id"].astype(str))[missing]
    key = cur["doc_id"].astype(str)
    for c in missing:
        cur[c] = key.map(src[c])
    check("missing columns are identified", missing == ["period", "content_sha256"])
    check("values land on the right rows", list(cur["period"])[:2] == ["FY25", "FY26"])
    check("a row absent from the revision gets NaN, not an error",
          pd.isna(cur["period"].iloc[3]))
    check("newer rows survive", len(cur) == 4)
    check("current statuses are untouched", list(cur["status"])[1] == "pending")

    for name in failed:
        print(f"  FAIL  {name}")
    print(f"restore_queue_columns self-test: {passed} passed, {len(failed)} failed")
    return 1 if failed else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--revision", default="",
                    help="Drive revision id of a pre-loss copy of the queue.")
    ap.add_argument("--list-revisions", action="store_true")
    ap.add_argument("--dry-run", action="store_true", default=True)
    ap.add_argument("--live", action="store_true",
                    help="Actually write. Without this nothing is changed.")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()

    if a.self_test:
        return _self_test()

    drive = get_drive()
    fid = find_file(drive, _index(drive), QUEUE_FILE)
    if not fid:
        log(f"{QUEUE_FILE} not found.")
        return 1
    if a.list_revisions:
        list_revisions(drive, fid)
        return 0
    if not a.revision:
        log("--revision is required (see --list-revisions).")
        return 1
    return restore(drive, fid, a.revision, a.live)


if __name__ == "__main__":
    raise SystemExit(main())
