r"""
strategy_ipo_base.py — first-base breakouts in recently listed stocks.

WHY THIS IS ITS OWN STRATEGY
  Every other pattern strategy here needs history a new listing does not have.
  darvas wants the stock within 5% of a 52-WEEK high; minervini wants a 200-day
  average and a rising 200 SMA; qullamaggie wants a 3-month run plus a base. A
  stock listed six months ago fails all of them on technicalities, so the most
  reactive part of the market is invisible to the engine.

  It is also genuinely different in shape. IPO bases are SHORTER and DEEPER than
  classic ones — as few as five weeks against seven-plus, and corrections of
  20-50% against the usual 12-35%. Feeding them into darvas's tolerances
  (BOX_MAX_RANGE_PCT = 10) would reject essentially all of them. That is
  precisely why they get missed.

SOURCE OF THE RULES
  The generic, publicly-taught IPO-base construction (O'Neil / IBD lineage):
  a first base within roughly 3-12 months of listing, a buy point above the base
  high on expanded volume, and the listing-day range as the structural reference.
  It is NOT taken from any specific book, and no book's text is reproduced here.
  Every threshold is a module constant so a different rule set can be swapped in
  by editing the CONFIG block and nothing else.

MEASURED CANDIDATE POOL (2026-09-02)
  299 stocks listed 3-12 months ago; 231 with current features; 135 trading at
  least Rs 1cr/day; 89 of those within 25% of their high. So this scans ~89-135
  names and is expected to emit single digits to low tens — a small additive
  list, not more noise.

Zones:
  add   close breaks above the base high on volume >= BREAKOUT_VOL_MULT x avg
  buy   close is coiling within BUY_MAX_BELOW_HIGH_PCT of the base high, and
        above the listing-day low. NOT "anywhere inside the base": that fired on
        68% of the cohort in testing, which describes the cohort rather than
        selecting from it.
  stop  base low (structural), reported alongside the listing-day low

Output:
  signals/per_strategy/ipo_base/<date>.csv + latest.csv

Usage:
    python scripts/strategy_ipo_base.py
    python scripts/strategy_ipo_base.py --dry-run --limit 40
"""
from __future__ import annotations

import argparse
import io
import os
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd

_SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(_SCRIPTS_DIR), ".env"))

from _extractor_base import (get_drive, get_or_create_subfolder, find_file,
                             download_bytes, upload_bytes)
from strategy_common import base_quality_score

# ─────────────────────────────── CONFIG ─────────────────────────────────────
# Swap these for a different rule set; nothing else needs to change.
MIN_AGE_DAYS = 90        # below this there has been no time to form a base
MAX_AGE_DAYS = 365       # beyond a year it is no longer a FIRST base
BASE_MIN_DAYS = 25       # ~5 weeks. IPO bases are much shorter than classic ones
BASE_MAX_DAYS = 90
BASE_MIN_DEPTH_PCT = 8   # flatter than this is a drift, not a base
BASE_MAX_DEPTH_PCT = 50  # IPO bases correct far deeper than the usual 12-35%
BREAKOUT_VOL_MULT = 1.4  # "expanded volume" — 40% above average
# A stock "in a base" is only interesting NEAR THE PIVOT. Accepting anything
# inside a 50%-deep range made `buy` near-vacuous: a live dry-run fired on 34 of
# the first 50 candidates (68%), which is a description of the cohort, not a
# signal. Requiring the close within this % of the base high means "coiling
# under resistance", which is the setup people actually mean.
BUY_MAX_BELOW_HIGH_PCT = 15.0
MIN_TURNOVER_CR = 1.0    # tradeable; the engine has no liquidity floor otherwise
MIN_PRICE = 10.0

STRATEGY = "ipo_base"
OUT_COLS = ["symbol", "date", "strategy", "zone_type", "score", "entry", "stop",
            "days_since_listing", "base_days", "base_high", "base_low",
            "base_depth_pct", "pct_below_pivot", "listing_day_low", "vol_today_ratio",
            "avg_turnover_20d_cr", "reason"]


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def find_ipo_base(ohlcv: pd.DataFrame) -> dict | None:
    """Longest recent window that looks like a first base.

    Measured EXCLUDING today's bar, so base_high is resistance built BEFORE
    today — otherwise `close > base_high` could never be true and the breakout
    zone would be permanently dead. (Same trap darvas documents as audit #32.)
    """
    if len(ohlcv) < BASE_MIN_DAYS + 1:
        return None
    high = ohlcv["high"].astype(float).values
    low = ohlcv["low"].astype(float).values

    for n in range(min(BASE_MAX_DAYS, len(ohlcv) - 1), BASE_MIN_DAYS - 1, -1):
        h = high[-(n + 1):-1].max()
        l = low[-(n + 1):-1].min()
        if h <= 0:
            continue
        depth = (h - l) / h * 100          # % off the base high, the classic read
        if BASE_MIN_DEPTH_PCT <= depth <= BASE_MAX_DEPTH_PCT:
            return {"base_days": n, "base_high": h, "base_low": l,
                    "base_depth_pct": depth}
    return None


def ipo_signal(symbol: str, ohlcv: pd.DataFrame, feat: pd.Series,
               days_since_listing: int) -> dict | None:
    d = ohlcv.sort_values("date").reset_index(drop=True)
    if len(d) < BASE_MIN_DAYS + 2:
        return None
    base = find_ipo_base(d)
    if not base:
        return None

    last = d.iloc[-1]
    close = float(last["close"])
    vol20 = float(d["volume"].tail(20).mean())
    vol_ratio = float(last["volume"]) / vol20 if vol20 else 0.0
    listing_low = float(d["low"].iloc[0])
    hi, lo = base["base_high"], base["base_low"]

    if close > hi and vol_ratio >= BREAKOUT_VOL_MULT:
        zone = "add"
        reason = (f"Breaks first base at Rs {hi:.2f} on {vol_ratio:.1f}x volume; "
                  f"{base['base_days']}d base, {base['base_depth_pct']:.0f}% deep, "
                  f"listed {days_since_listing}d ago")
    elif (lo <= close <= hi and close > listing_low
          and (hi - close) / hi * 100 <= BUY_MAX_BELOW_HIGH_PCT):
        zone = "buy"
        reason = (f"Coiling {(hi-close)/hi*100:.1f}% under the pivot Rs {hi:.2f} "
                  f"in a {base['base_days']}d first base "
                  f"({base['base_depth_pct']:.0f}% deep), above the listing-day "
                  f"low Rs {listing_low:.2f}; listed {days_since_listing}d ago")
    else:
        # Below the base low, or already extended past the breakout — neither is
        # an entry.
        return None

    qual, parts = base_quality_score(
        base["base_depth_pct"], BASE_MAX_DEPTH_PCT, base["base_days"],
        BASE_MIN_DAYS, BASE_MAX_DAYS, vol_ratio, breakout=(zone == "add"))

    return {
        "symbol": symbol, "date": feat["date"], "strategy": STRATEGY,
        "zone_type": zone, "score": qual,
        "entry": round(close, 2),
        # Structural: the base low is where the setup is wrong. The listing-day
        # low is reported beside it as the deeper reference the pattern is built
        # on, not used as the stop.
        "stop": round(lo, 2),
        "days_since_listing": int(days_since_listing),
        "base_days": base["base_days"],
        "base_high": round(hi, 2), "base_low": round(lo, 2),
        "base_depth_pct": round(base["base_depth_pct"], 1),
        "listing_day_low": round(listing_low, 2),
        "pct_below_pivot": round((hi - close) / hi * 100, 2),
        "vol_today_ratio": round(vol_ratio, 2),
        "avg_turnover_20d_cr": round(float(feat.get("avg_turnover_20d_cr", np.nan)), 2),
        "base_tightness": round(parts["tightness"], 3),
        "base_duration": round(parts["duration"], 3),
        "base_confirmation": round(parts["confirmation"], 3),
        "reason": reason,
    }


def _upload_csv(drive, folder_id, name, df):
    upload_bytes(drive, folder_id, name, df.to_csv(index=False).encode(),
                 "text/csv")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true",
                    help="compute and report, write nothing to Drive")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    print("Stage 14 — IPO first-base signals")
    print("-" * 50)
    drive = get_drive()
    folder_id = os.environ["GDRIVE_FOLDER_ID"]

    fid = find_file(drive, get_or_create_subfolder(drive, folder_id, "features"),
                    "latest.parquet")
    if not fid:
        print("features/latest.parquet missing — run compute_features.py first.")
        return 1
    feats = pd.read_parquet(io.BytesIO(download_bytes(drive, fid)))
    feats["symbol"] = feats["symbol"].astype(str)
    log(f"features: {len(feats):,} symbols")

    # Listing dates: the gate this whole strategy rests on. build_listing_dates
    # refuses to date a name whose first bar falls on an ingest cliff, so an
    # absent date means "we do not know", never "listed that day".
    repo = get_or_create_subfolder(drive, folder_id, "company_repo")
    lfid = find_file(drive, get_or_create_subfolder(drive, repo, "_index"),
                     "listing_dates.parquet")
    if not lfid:
        print("company_repo/_index/listing_dates.parquet missing — run "
              "scripts/build_listing_dates.py --backfill first.")
        return 1
    ld = pd.read_parquet(io.BytesIO(download_bytes(drive, lfid)))
    sym_col = "symbol" if "symbol" in ld.columns else ld.columns[0]
    ld[sym_col] = ld[sym_col].astype(str)
    ld["listing_date"] = pd.to_datetime(ld.get("listing_date"), errors="coerce")
    ld = ld.dropna(subset=["listing_date"]).drop_duplicates(sym_col)
    log(f"listing dates: {len(ld):,} names carry one")

    asof = pd.to_datetime(feats["date"], errors="coerce").max()
    ld["days_since_listing"] = (asof - ld["listing_date"]).dt.days
    recent = ld[(ld["days_since_listing"] >= MIN_AGE_DAYS)
                & (ld["days_since_listing"] <= MAX_AGE_DAYS)]
    log(f"listed {MIN_AGE_DAYS}-{MAX_AGE_DAYS}d ago: {len(recent):,}")

    cand = feats.merge(recent[[sym_col, "days_since_listing"]],
                       left_on="symbol", right_on=sym_col, how="inner")
    cand = cand[(cand["close"] >= MIN_PRICE)
                & (pd.to_numeric(cand.get("avg_turnover_20d_cr"),
                                 errors="coerce") >= MIN_TURNOVER_CR)]
    log(f"with features, price >= Rs {MIN_PRICE:g} and turnover >= "
        f"Rs {MIN_TURNOVER_CR}cr/day: {len(cand):,}")
    if args.limit:
        cand = cand.head(args.limit)
    if cand.empty:
        print("No candidates.")
        return 0

    data_id = get_or_create_subfolder(drive, folder_id, "data")
    ohlcv_id = get_or_create_subfolder(drive, data_id, "ohlcv")

    signals, n_nobase, n_nofile = [], 0, 0
    t0 = time.time()
    for i, (_, feat) in enumerate(cand.iterrows(), 1):
        sym = str(feat["symbol"])
        pf = find_file(drive, ohlcv_id, f"{sym}.parquet")
        if not pf:
            n_nofile += 1
            continue
        try:
            px = pd.read_parquet(io.BytesIO(download_bytes(drive, pf)))
            px["date"] = pd.to_datetime(px["date"], errors="coerce")
            px = px.dropna(subset=["date"]).sort_values("date")
            sig = ipo_signal(sym, px, feat, feat["days_since_listing"])
            if sig:
                signals.append(sig)
            else:
                n_nobase += 1
        except Exception:
            n_nofile += 1
        if i % 25 == 0:
            log(f"  {i}/{len(cand)} scanned, {len(signals)} signals "
                f"({time.time()-t0:.0f}s)")

    out = pd.DataFrame(signals)
    log(f"SCANNED {len(cand):,} | signals {len(out):,} | "
        f"no valid base {n_nobase:,} | no price file {n_nofile:,}")
    if len(out):
        out = out.sort_values("score", ascending=False).reset_index(drop=True)
        print()
        print(out[["symbol", "zone_type", "score", "days_since_listing",
                   "base_days", "base_depth_pct", "vol_today_ratio",
                   "avg_turnover_20d_cr"]].head(20).to_string(index=False))

    if args.dry_run:
        print()
        log("DRY RUN — nothing written to Drive")
        return 0

    signals_id = get_or_create_subfolder(drive, folder_id, "signals")
    per_id = get_or_create_subfolder(drive, signals_id, "per_strategy")
    sub_id = get_or_create_subfolder(drive, per_id, STRATEGY)
    # Written even when empty, so the health check's staleness gate sees a fresh
    # file on a quiet day and the aggregator skips it cleanly.
    frame = out[[c for c in OUT_COLS if c in out.columns]] if len(out) \
        else pd.DataFrame(columns=OUT_COLS)
    _upload_csv(drive, sub_id, "latest.csv", frame)
    _upload_csv(drive, sub_id, f"{pd.Timestamp(asof).date()}.csv", frame)
    log(f"wrote signals/per_strategy/{STRATEGY}/ ({len(frame)} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
