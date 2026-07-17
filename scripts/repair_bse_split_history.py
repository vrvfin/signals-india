"""
repair_bse_split_history.py — ONE-OFF repair of split/bonus cliffs in BSE-only
OHLCV parquets (data/ohlcv/, keys from company_universe BSE-exclusive rows).

BSE serves RAW prices and never restates history, and fetch_bse_eod_api.py used
to append blindly — so a bonus/split ex-date left a fake cliff in the stored
series (FREDUN 2:1 bonus 2026-07-16 -> fake -65%).

Detection: an internal day-move > JUMP_TOL within the last LOOKBACK_DAYS of the
stored series. Confirmation: BSE CorporateAction API must show an official
Bonus/Sub-division with ex-date within EXDATE_SLACK_DAYS of the jump — the
OFFICIAL ratio is used, never a price-implied factor. Repair: rescale bars
before the ex-date (OHLC / factor, volume * factor) + junction continuity check.
No official record = reported for manual look, never touched.

Usage:
    python scripts/repair_bse_split_history.py                    # DRY-RUN
    python scripts/repair_bse_split_history.py --keys FREDUN --live
    python scripts/repair_bse_split_history.py --live             # repair all confirmed
"""

from __future__ import annotations

import argparse
import io
import os
import sys
import time
import warnings
from datetime import datetime

import pandas as pd

warnings.filterwarnings("ignore")

_SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

import fetch_bse_eod_api as bse  # noqa: E402  (Drive helpers + guard helpers)
from _extractor_base import download_bytes, upload_bytes  # noqa: E402

JUMP_TOL = 0.30        # internal day-move that marks a suspect cliff
LOOKBACK_DAYS = 180    # only hunt cliffs in the recent window


def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def find_cliffs(df: pd.DataFrame) -> list[pd.Timestamp]:
    s = df.sort_values("date").reset_index(drop=True)
    recent = s[s["date"] >= s["date"].max() - pd.Timedelta(days=LOOKBACK_DAYS)]
    pct = recent["close"].pct_change()
    return list(recent.loc[pct.abs() > JUMP_TOL, "date"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", action="store_true")
    ap.add_argument("--keys", type=str, default="",
                    help="Comma-separated storage keys (e.g. FREDUN)")
    args = ap.parse_args()

    mode = "LIVE" if args.live else "DRY-RUN"
    print(f"repair_bse_split_history — {mode}")
    print("-" * 60)

    drive = bse.get_drive()
    uni = bse._bse_only(drive)
    uni["key"] = [bse._storage_key(r) for _, r in uni.iterrows()]
    if args.keys:
        want = {k.strip().upper() for k in args.keys.split(",")}
        uni = uni[uni["key"].str.upper().isin(want)]

    live_fid = bse._folder(drive, bse.OHLCV_LIVE)
    files = bse._list_folder(drive, live_fid)
    uni = uni[[f"{k}.parquet" in files for k in uni["key"]]]
    log(f"BSE-only names with a stored parquet: {len(uni)}")

    flagged, repaired, manual, errors = [], [], [], []
    for i, (_, row) in enumerate(uni.iterrows(), 1):
        key, code = row["key"], row["bse_code"]
        try:
            df = pd.read_parquet(io.BytesIO(
                download_bytes(drive, files[f"{key}.parquet"])))
            df["date"] = pd.to_datetime(df["date"])
            cliffs = find_cliffs(df)
            if not cliffs:
                continue
            for cliff in cliffs:
                hit = bse.corp_action_factor(code, cliff.date())
                if not hit:
                    manual.append((key, f"cliff {cliff.date()} but no official "
                                        f"corp action — manual look"))
                    log(f"  SUSPECT {key}: cliff {cliff.date()} — NO official "
                        f"record, skipping")
                    continue
                factor, exd = hit
                flagged.append((key, exd, factor))
                log(f"  CONFIRMED {key}: official factor {factor:g} ex {exd} "
                    f"(cliff {cliff.date()}, rows={len(df)})")
                fixed = bse.rescale_history(df, exd, factor)
                if not bse.junction_ok(fixed, exd):
                    manual.append((key, "residual jump after rescale"))
                    log(f"    SKIP {key}: residual jump after rescale")
                    continue
                if args.live:
                    upload_bytes(drive, live_fid, f"{key}.parquet",
                                 fixed.to_parquet(index=False),
                                 "application/octet-stream",
                                 existing_id=files[f"{key}.parquet"])
                    repaired.append(key)
                    log(f"    repaired {key}: rescaled /{factor:g} before {exd}")
                df = fixed  # in case of a second cliff in the same file
        except Exception as e:
            errors.append((key, str(e)[:120]))
            log(f"  ERROR {key}: {str(e)[:120]}")
        if i % 200 == 0:
            log(f"  ...{i}/{len(uni)} scanned, confirmed={len(flagged)}")

    print()
    print("=" * 60)
    print(f"{mode} SUMMARY  scanned={len(uni)}  confirmed={len(flagged)}  "
          f"repaired={len(repaired)}  manual={len(manual)}  errors={len(errors)}")
    for key, exd, factor in flagged:
        print(f"  {key:<14} /{factor:g} before {exd}")
    if manual:
        print("manual look:")
        for key, why in manual:
            print(f"  {key}: {why}")
    if errors:
        print("errors:")
        for key, e in errors[:20]:
            print(f"  {key}: {e}")


if __name__ == "__main__":
    main()
