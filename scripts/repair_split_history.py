"""
repair_split_history.py — ONE-OFF repair of split/bonus-stale OHLCV parquets.

Problem: ingest_ohlcv.py only APPENDS bars newer than the stored max date.
When a split/bonus goes ex, newly appended bars arrive on the new price scale
while stored history stays on the old scale — a fake cliff at the ex-date
(e.g. GOLDIAM 4:3 bonus -> fake -25%, VMARCIND 6:1 -> fake -83%).

Detection (per symbol with a stored parquet):
  fetch CHECK_PERIOD with auto_adjust=True; on overlapping dates compute
  ratio = stored_close / fresh_close. A trailing run of ratio==1 (the bars we
  appended AFTER the action) preceded by ratio far from 1 = stale scale.

Repair — RESCALE IN PLACE (never trusts Yahoo's history, never drops a bar):
  boundary = first date of the TRAILING in-sync run (where our own series
  switched to the new scale). Every stored bar before the boundary is divided
  by its segment's measured ratio (volume multiplied). Rows older than the
  overlap window get the oldest observed segment's ratio. We deliberately do
  NOT overwrite from a fresh 10y pull: Yahoo itself can lag restating SME
  history (VMARCIND's own Yahoo series still has the -83% cliff), so a fresh
  pull can import a broken series.

Post-repair validation: the junction at each rescaled segment boundary must
not retain a residual jump (>25%) — else the symbol is reported, not written.

NSE (.NS) symbols only — BSE-only names use the BSE EOD feed, not Yahoo.

Usage:
    python scripts/repair_split_history.py                  # DRY-RUN (report only)
    python scripts/repair_split_history.py --symbols GOLDIAM,VMARCIND --live
    python scripts/repair_split_history.py --live           # repair everything flagged
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

warnings.filterwarnings("ignore")

_SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

load_dotenv(Path(_SCRIPTS_DIR).parent / ".env")

import ingest_ohlcv as ing  # noqa: E402  (Drive helpers + fetch/normalize)

# Detection/rescale helpers are shared with the nightly guard — single source
# of truth lives in ingest_ohlcv.py (SCALE_TOL / JUNCTION_TOL there).
detect_drift = ing.detect_drift
rescale_in_place = ing.rescale_in_place
junction_ok = ing.junction_ok

BATCH = 25
CHECK_PERIOD = "6mo"


def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", action="store_true",
                    help="Actually write repairs to Drive (default: dry-run report)")
    ap.add_argument("--symbols", type=str, default="",
                    help="Comma-separated subset (e.g. GOLDIAM,VMARCIND)")
    args = ap.parse_args()

    mode = "LIVE" if args.live else "DRY-RUN"
    print(f"repair_split_history — {mode}")
    print("-" * 60)

    drive = ing.get_drive_service()
    fid = os.environ["GDRIVE_FOLDER_ID"]
    uni_id = ing.get_or_create_subfolder(drive, fid, "universe")
    uni = ing.download_csv(
        drive, ing.list_files_in_folder(drive, uni_id)["master_list.csv"])
    uni = uni[uni["exchange"].astype(str) == "NSE"]
    symbols = uni["symbol"].astype(str).tolist()
    if args.symbols:
        want = {s.strip().upper() for s in args.symbols.split(",")}
        symbols = [s for s in symbols if s.upper() in want]

    data_id = ing.get_or_create_subfolder(drive, fid, "data")
    ohlcv_id = ing.get_or_create_subfolder(drive, data_id, "ohlcv")
    existing = ing.list_files_in_folder(drive, ohlcv_id)
    symbols = [s for s in symbols if f"{s}.parquet" in existing]
    log(f"NSE symbols with a stored parquet: {len(symbols)}")

    flagged, repaired, skipped, errors = [], [], [], []
    batches = [symbols[i:i + BATCH] for i in range(0, len(symbols), BATCH)]
    for bi, batch in enumerate(batches, 1):
        fresh_map = ing.fetch_ohlcv_batch(batch, period=CHECK_PERIOD)
        for sym in batch:
            fresh = fresh_map.get(sym)
            if fresh is None or fresh.empty:
                continue
            try:
                stored = ing.download_parquet(drive, existing[f"{sym}.parquet"])
                stored["date"] = pd.to_datetime(stored["date"])
                stale, boundary, segments = detect_drift(stored, fresh)
                if not stale:
                    continue
                if boundary is None or not segments:
                    skipped.append((sym, "no in-sync tail to anchor rescale"))
                    log(f"  STALE {sym}: SKIP (no in-sync tail)")
                    continue
                desc = ", ".join(f"{f:.4f}@{d.date()}" for d, f in segments)
                flagged.append((sym, boundary, desc))
                log(f"  STALE {sym}: boundary={boundary.date()} "
                    f"segments=[{desc}] rows={len(stored)}")
                fixed = rescale_in_place(stored, boundary, segments)
                check_dates = [d for d, _ in segments] + [boundary]
                if not junction_ok(fixed, check_dates):
                    skipped.append((sym, "residual jump after rescale"))
                    log(f"    SKIP {sym}: residual jump at a junction — not written")
                    continue
                if args.live:
                    ing.upload_parquet(drive, ohlcv_id, f"{sym}.parquet",
                                       fixed, existing[f"{sym}.parquet"])
                    repaired.append(sym)
                    log(f"    repaired {sym}: rescaled in place ({len(fixed)} rows)")
            except Exception as e:
                errors.append((sym, str(e)[:120]))
                log(f"  ERROR {sym}: {str(e)[:120]}")
        log(f"batch {bi}/{len(batches)} done  flagged_so_far={len(flagged)}")
        if bi < len(batches):
            time.sleep(1)

    print()
    print("=" * 60)
    print(f"{mode} SUMMARY  checked={len(symbols)}  stale={len(flagged)}  "
          f"repaired={len(repaired)}  skipped={len(skipped)}  errors={len(errors)}")
    for sym, boundary, desc in flagged:
        print(f"  {sym:<14} new-scale from {boundary.date()}  factors [{desc}]")
    if skipped:
        print("skipped (need manual look):")
        for sym, why in skipped:
            print(f"  {sym}: {why}")
    if errors:
        print("errors:")
        for sym, e in errors:
            print(f"  {sym}: {e}")


if __name__ == "__main__":
    main()
