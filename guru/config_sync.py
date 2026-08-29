r"""
CONFIG SYNC — config_sync.py  (Project Guru)

Keeps the RULES and config on Google Drive (private) instead of in the public
git repo, so CI can run the screen without publishing the validated strategy.

Drive layout:  <root>/guru_config/
    rule_template.xlsx        all rules + clauses
    _chosen_rules.parquet     the validated rule selection
    universe_hist.parquet     symbol -> guru_key -> company name mapping
    INDIA_VIX.parquet         VIX series (regime clause)
    screen_snapshot.parquet   last run's stock list (for NEW detection)

Also syncs the SNAPSHOT both ways, so "which stocks are new?" works across CI
runs even though the runner is wiped each time.

Usage:
    python guru/config_sync.py --upload      # from local  -> Drive (run once, and
                                             #  again whenever rules change)
    python guru/config_sync.py --download    # Drive -> runner (CI does this)
    python guru/config_sync.py --push-snapshot   # after a CI screen, save state
"""
from __future__ import annotations
import argparse, io, os, sys
from datetime import datetime
import pandas as pd

GURU = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(GURU, "data")
BT = os.path.join(GURU, "backtest")
PROJ = os.path.join(os.path.dirname(GURU), "Project_Guru")
SCRIPTS = os.path.join(os.path.dirname(GURU), "scripts")
if SCRIPTS not in sys.path:
    sys.path.insert(0, SCRIPTS)
from pathlib import Path
from dotenv import load_dotenv
load_dotenv(Path(GURU).parent / ".env")
from _extractor_base import (get_drive, get_or_create_subfolder, find_file,
                             download_bytes, upload_bytes)

FOLDER = "guru_config"
# (local path, drive name, mimetype)
ITEMS = [
    (os.path.join(PROJ, "rule_template.xlsx"), "rule_template.xlsx",
     "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"),
    (os.path.join(BT, "_chosen_rules.parquet"), "_chosen_rules.parquet",
     "application/octet-stream"),
    (os.path.join(DATA, "universe_hist.parquet"), "universe_hist.parquet",
     "application/octet-stream"),
    (os.path.join(DATA, "macro_hist", "INDIA_VIX.parquet"), "INDIA_VIX.parquet",
     "application/octet-stream"),
]
SNAPSHOT = (os.path.join(BT, "screen_snapshot.parquet"), "screen_snapshot.parquet",
            "application/octet-stream")


def log(m): print(f"[{datetime.now().strftime('%H:%M:%S')}] {m}", flush=True)


def folder(d):
    return get_or_create_subfolder(d, os.environ["GDRIVE_FOLDER_ID"], FOLDER)


def put(d, fid, local, name, mime):
    if not os.path.exists(local):
        log(f"  SKIP (missing locally): {name}")
        return False
    with open(local, "rb") as fh:
        upload_bytes(d, fid, name, fh.read(), mime)
    log(f"  uploaded {name} ({os.path.getsize(local)/1024:.0f} KB)")
    return True


def get(d, fid, local, name):
    f = find_file(d, fid, name)
    if not f:
        log(f"  MISSING on Drive: {name}")
        return False
    os.makedirs(os.path.dirname(local), exist_ok=True)
    with open(local, "wb") as fh:
        fh.write(download_bytes(d, f))
    log(f"  downloaded {name} -> {os.path.relpath(local, os.path.dirname(GURU))}")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--upload", action="store_true")
    ap.add_argument("--download", action="store_true")
    ap.add_argument("--push-snapshot", action="store_true")
    args = ap.parse_args()
    d = get_drive()
    fid = folder(d)

    if args.upload:
        log(f"uploading config -> Drive/{FOLDER}/")
        for local, name, mime in ITEMS:
            put(d, fid, local, name, mime)
        put(d, fid, *SNAPSHOT)
        log("upload complete — CI can now fetch these; nothing rule-related "
            "needs to live in the public repo")
    elif args.download:
        log(f"downloading config from Drive/{FOLDER}/")
        ok = all(get(d, fid, local, name) for local, name, _ in ITEMS)
        get(d, fid, *SNAPSHOT[:2])          # optional; absent on first run
        if not ok:
            log("WARNING: some config missing — run --upload from your machine first")
            sys.exit(1)
    elif args.push_snapshot:
        put(d, fid, *SNAPSHOT)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
