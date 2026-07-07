r"""
build_signal_membership.py — per-symbol signal-appearance history ("tenure").

Reads the dated aggregated snapshots that aggregate_signals.py already writes
(signals/aggregated/<date>.csv — one row per select on that day) over the last
6 months and derives, per symbol, how long / how often it has been a "select"
and whether it just appeared or just dropped.

Writes:  signals/aggregated/membership.parquet   (one row per symbol)

Columns (schema-first — additive, safe to extend):
  symbol
  first_seen, last_seen        first/last date it was a select (in window)
  days_present                 HOW MANY TIMES it came (# snapshots it is in)
  window_snapshots             total snapshots in the window (the denominator)
  span_days                    calendar days first_seen..last_seen (+1)
  present_last_7d              appeared in the last 7 calendar days?
  add_tier                     mutually exclusive, by first_seen age in days:
                                 "fresh"  = 0      (first appeared today)
                                 "new"    = 1..7   (first appeared last 7 days)
                                 "recent" = 8..14  (first appeared last 14 days)
                                 ""       = older / established
  dropped_last_7d              present 7-14 days ago but ABSENT the last 7 days
                               (freshly fell off this week)

NOTE: history currently starts ~2026-05-19, so the "6-month" window is only as
deep as the snapshots that exist; it deepens automatically as days accumulate.

Usage:
  python scripts/build_signal_membership.py --dry-run   # counts only, no write
  python scripts/build_signal_membership.py             # write membership.parquet
"""
from __future__ import annotations

import argparse
import io
import os
import re
import sys
from datetime import datetime, timedelta

_SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

import pandas as pd
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(_SCRIPTS_DIR), ".env"))

from _extractor_base import (get_drive, get_or_create_subfolder, find_file,
                             download_bytes, save_parquet, log)

WINDOW_DAYS = 183          # ~6 months
DATE_RE = re.compile(r"^(\d{4}-\d{2}-\d{2})\.csv$")


def _list_dated(drive, folder_id) -> list[tuple[str, str]]:
    """Return [(date_str, file_id)] for every YYYY-MM-DD.csv snapshot."""
    out, tok = [], None
    while True:
        r = drive.files().list(
            q=f"'{folder_id}' in parents and trashed=false",
            fields="nextPageToken, files(id,name)",
            pageSize=1000, pageToken=tok).execute()
        for f in r.get("files", []):
            m = DATE_RE.match(f["name"])
            if m:
                out.append((m.group(1), f["id"]))
        tok = r.get("nextPageToken")
        if not tok:
            break
    return sorted(out)


def build(drive, dry_run: bool = False) -> pd.DataFrame:
    root = os.environ["GDRIVE_FOLDER_ID"]
    sig_id = get_or_create_subfolder(drive, root, "signals")
    agg_id = get_or_create_subfolder(drive, sig_id, "aggregated")

    dated = _list_dated(drive, agg_id)
    if not dated:
        log("No dated snapshots found — nothing to build.")
        return pd.DataFrame()

    today = datetime.strptime(dated[-1][0], "%Y-%m-%d").date()
    cutoff = today - timedelta(days=WINDOW_DAYS)
    dated = [(d, fid) for (d, fid) in dated
             if datetime.strptime(d, "%Y-%m-%d").date() >= cutoff]
    log(f"window: {dated[0][0]} .. {dated[-1][0]}  ({len(dated)} snapshots); "
        f"'today' = {today}")

    # symbol -> sorted list of dates present
    seen: dict[str, list] = {}
    for dstr, fid in dated:
        try:
            df = pd.read_csv(io.BytesIO(download_bytes(drive, fid)))
        except Exception as e:
            log(f"  {dstr}: read failed ({str(e)[:60]}) — skipped")
            continue
        if "symbol" not in df.columns:
            continue
        d = datetime.strptime(dstr, "%Y-%m-%d").date()
        for s in df["symbol"].dropna().astype(str).str.upper().unique():
            seen.setdefault(s, []).append(d)

    rows = []
    for sym, dates in seen.items():
        dates = sorted(set(dates))
        first, last = dates[0], dates[-1]
        age_first = (today - first).days
        age_last = (today - last).days
        present_7 = age_last <= 7
        # Addition tiers are mutually exclusive AND require the name to still be
        # an active select (present in last 7d) — a first-appeared-then-dropped
        # name is a DROP, not an addition, so it never lands in a "recent" tier.
        if age_first == 0:
            tier = "fresh"          # first appeared today
        elif age_first <= 7:
            tier = "new"            # first appeared in last 7 days (always active)
        elif age_first <= 14 and present_7:
            tier = "recent"         # first appeared 8-14 days ago, still active
        else:
            tier = ""
        dropped = (not present_7) and (age_last <= 14)   # def (a): freshly fell off
        rows.append({
            "symbol": sym,
            "first_seen": pd.Timestamp(first),
            "last_seen": pd.Timestamp(last),
            "days_present": len(dates),
            "window_snapshots": len(dated),
            "span_days": (last - first).days + 1,
            "present_last_7d": present_7,
            "add_tier": tier,
            "dropped_last_7d": dropped,
        })
    out = pd.DataFrame(rows).sort_values("symbol").reset_index(drop=True)

    n_fresh = (out["add_tier"] == "fresh").sum()
    n_new = (out["add_tier"] == "new").sum()
    n_recent = (out["add_tier"] == "recent").sum()
    n_drop = out["dropped_last_7d"].sum()
    log(f"symbols: {len(out)} | fresh={n_fresh} new={n_new} recent={n_recent} "
        f"dropped_last_7d={n_drop}")

    if dry_run:
        log("DRY-RUN — not writing membership.parquet")
        for tier in ("fresh", "new", "recent"):
            sample = out[out["add_tier"] == tier]["symbol"].head(12).tolist()
            log(f"  {tier:7s}: {sample}")
        log(f"  dropped: {out[out['dropped_last_7d']]['symbol'].head(12).tolist()}")
        return out

    save_parquet(drive, agg_id, "membership.parquet", out)
    log(f"Wrote signals/aggregated/membership.parquet ({len(out)} rows)")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    drive = get_drive()
    build(drive, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
