"""
requeue_orphans.py — one-time queue hygiene (T12).

Two classes of dead rows accumulated before the "process-or-repull" discipline
existed; this resets them to `expired` so the coverage layer treats them as
NOT-covered and re-fetches them cleanly when their doc_type is next backfilled
(concall re-fetches on the next backfill slot; the other types when Stage B runs).

  1. ORPHANS — backfill rows for non-concall types (results / annual_report /
     rating / presentation) stuck `status='pending'` with a `drive_file_id`
     pointing to a PDF the 2-day cleanup already deleted. No extractor ever drained
     them (backfill only runs concall), so they sit pending forever and a download
     attempt would 404.
  2. STALE ERRORS — backfill `doc_type='concall'` rows `status='error'`, many from
     the old transient-misclassification (network drop -> FATAL) since fixed.
     Re-fetch re-downloads + re-extracts under the corrected classifier.

Writes are serialized via the shared `_extract.lock` (Phase-2 priority wait).
`--dry-run` (default) only reports counts. `--apply` performs the update.

Usage:
    python scripts/requeue_orphans.py                 # dry-run: print counts
    python scripts/requeue_orphans.py --apply         # perform the reset
"""
from __future__ import annotations

import argparse
import atexit
import os
from datetime import datetime

from dotenv import load_dotenv
from pathlib import Path

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from _extractor_base import (get_drive, get_or_create_subfolder,
                             acquire_lock, release_lock)
import ingest_company_docs as icd

_LOCK_NAME = "_extract.lock"
_PF_ONLY_TYPES = ("results", "annual_report", "rating", "presentation")


def _blank(s) -> bool:
    v = str(s).strip().lower()
    return v in ("", "nan", "none")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--apply", action="store_true",
                    help="Perform the reset (default is dry-run: report only).")
    args = ap.parse_args()
    dry = not args.apply

    drive = get_drive()
    folder = os.environ["GDRIVE_FOLDER_ID"]
    repo = get_or_create_subfolder(drive, folder, "company_repo")
    index_id = get_or_create_subfolder(drive, repo, "_index")

    if not dry:
        # Serialize with live Phase 2 / backfill writers (Phase-2 priority wait).
        if not acquire_lock(drive, index_id, _LOCK_NAME, "maintenance",
                            max_age_min=360, wait_min=15):
            print("Lock unavailable after wait — exiting cleanly (no change).")
            return
        atexit.register(release_lock, drive, index_id, _LOCK_NAME)

    q = icd.load_queue(drive, index_id)
    src = q["source"].astype(str).str.lower()
    is_bf = src.eq("backfill")
    has_fid = ~q["drive_file_id"].map(_blank)

    orphan_mask = (is_bf
                   & q["doc_type"].isin(_PF_ONLY_TYPES)
                   & q["status"].eq("pending")
                   & has_fid)
    err_mask = is_bf & q["doc_type"].eq("concall") & q["status"].eq("error")

    n_orphan = int(orphan_mask.sum())
    n_err = int(err_mask.sum())

    print("-" * 56)
    print("Orphans (backfill non-concall pending w/ dead PDF):")
    for dt in _PF_ONLY_TYPES:
        c = int((orphan_mask & q["doc_type"].eq(dt)).sum())
        if c:
            print(f"  {dt:<14}: {c}")
    print(f"  TOTAL orphans -> expired : {n_orphan}")
    print(f"Stale backfill concall errors -> expired : {n_err}")
    print("-" * 56)

    if dry:
        print("DRY RUN — no changes written. Re-run with --apply to perform.")
        return

    now = datetime.now().isoformat(timespec="seconds")
    q.loc[orphan_mask | err_mask, "status"] = "expired"
    q.loc[orphan_mask | err_mask, "processed_at"] = now
    icd.save_queue(drive, index_id, q)
    print(f"APPLIED: {n_orphan + n_err} rows set status='expired' "
          f"(re-fetched when their doc_type is next backfilled).")
    print("Output: company_repo/_index/processing_queue.parquet")


if __name__ == "__main__":
    main()
