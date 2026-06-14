"""
phase2_beacon.py — set/clear the Phase-2 priority beacon (T12).

Phase 2 (high priority) raises this beacon for the whole job so backfill (low
priority) steps aside from the shared _extract.lock. phase2.yml sets it as the
first step and clears it as a final always() step. The Phase-2 extractors also
refresh it each time they wait for the lock (keeps it fresh through a long run);
it goes stale on its own (~20 min) if Phase 2 crashes, so backfill is never
starved indefinitely.

Usage:
    python scripts/phase2_beacon.py set
    python scripts/phase2_beacon.py clear
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _extractor_base import (
    get_drive, get_or_create_subfolder,
    set_phase2_beacon, clear_phase2_beacon,
)


def main() -> None:
    action = sys.argv[1] if len(sys.argv) > 1 else ""
    if action not in ("set", "clear"):
        sys.exit("usage: phase2_beacon.py set|clear")
    drive = get_drive()
    root = os.environ["GDRIVE_FOLDER_ID"]
    index_id = get_or_create_subfolder(
        drive, get_or_create_subfolder(drive, root, "company_repo"), "_index")
    if action == "set":
        set_phase2_beacon(drive, index_id)
        print("Phase 2 beacon SET (backfill will step aside).")
    else:
        clear_phase2_beacon(drive, index_id)
        print("Phase 2 beacon CLEARED.")


if __name__ == "__main__":
    main()
