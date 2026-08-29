r"""
P4c — compute_technical_metrics.py  (Project Guru, RESUMABLE, heavy)

Daily technical metric series per security from ohlcv_hist + regime. One output
row per trading day. Output: guru/data/metrics/technical/<guru_key>.parquet
(float32 to keep the store compact). Resumable per key via ledger.

Columns (matching rule_template vocabulary):
  returns   single_day_return_pct, monthly_return_pct, price_return_{1,2,3,6,9,12,18,24,36}m_pct
  MAs       price_vs_ma_{20,50,100,200}, price_to_{ma20,ma50,ma200,ema50}_ratio,
            ma_cross_{20_50,20_100,50_100,50_200,100_200}
  range     pct_from_52w_high, pct_from_52w_low, pct_from_all_time_high,
            pct_position_in_52w_range, drawdown_from_high_pct, recovery_pct_from_low
  vol/flow  volume_ratio_{5,10,20}d_avg, volatility_20d_pct, atr_expansion_ratio
  osc       rsi_14, bb_upper_break_{1,1.5,2,2.5}sd
  slope     price_slope_{5,10,20,30,50}d_pct
  other     consecutive_up_weeks, days_since_listing, beta_1y,
            rel_strength_vs_index_pct, momentum_score (raw; percentile = 2nd pass)
  gap (NaN) sector_rel_strength_pct  (needs sector map — not available)

momentum_score_percentile and sector_rel_strength_pct are cross-sectional / need
data we lack; the percentile is added by a later cross-sectional pass (P4c2) and
sector_rel_strength stays NaN. Documented, not silently dropped.

Usage:
    python guru/compute_technical_metrics.py --dry-run
    python guru/compute_technical_metrics.py --limit 30
    python guru/compute_technical_metrics.py            # full, resumes
    python guru/compute_technical_metrics.py --status
"""
from __future__ import annotations

import argparse
import glob
import os
from datetime import datetime

import numpy as np
import pandas as pd

GURU_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(GURU_DIR, "data")
OHLCV_DIR = os.path.join(DATA_DIR, "ohlcv_hist")
OUT_DIR = os.path.join(DATA_DIR, "metrics", "technical")
STATUS_DIR = os.path.join(DATA_DIR, "_dump_status")
LEDGER_PATH = os.path.join(STATUS_DIR, "tech_metrics_ledger.parquet")

RET_MONTHS = {"1m": 21, "2m": 42, "3m": 63, "6m": 126, "9m": 189,
              "12m": 252, "18m": 378, "24m": 504, "36m": 756}
MIN_ROWS = 30


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    rs = up / dn.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def slope_pct(close: pd.Series, n: int) -> pd.Series:
    """rolling linear-regression slope over n days, as %/day of mean level."""
    x = np.arange(n)
    xm = x.mean()
    denom = ((x - xm) ** 2).sum()

    def _s(vals):
        y = vals
        return ((x - xm) * (y - y.mean())).sum() / denom

    sl = close.rolling(n).apply(_s, raw=True)
    return sl / close * 100.0


def consecutive_up_weeks(df: pd.DataFrame) -> pd.Series:
    wk = df.set_index("date")["close"].resample("W-FRI").last().dropna()
    up = (wk.diff() > 0)
    run = up * (up.groupby((~up).cumsum()).cumcount() + 1)
    daily = run.reindex(df["date"], method="ffill")
    return pd.Series(daily.values, index=df.index)


def compute(df: pd.DataFrame, regime: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("date").reset_index(drop=True)
    close, vol, high, low = df["close"], df["volume"], df["high"], df["low"]
    o = pd.DataFrame({"date": df["date"]})
    o["close"] = close
    o["single_day_return_pct"] = close.pct_change() * 100
    o["monthly_return_pct"] = close.pct_change(21) * 100
    for lbl, n in RET_MONTHS.items():
        o[f"price_return_{lbl}_pct"] = close.pct_change(n) * 100
    for n in (20, 50, 100, 200):
        ma = close.rolling(n, min_periods=n // 2).mean()
        o[f"price_vs_ma_{n}"] = (close / ma - 1) * 100
        if n in (20, 50, 200):
            o[f"price_to_ma{n}_ratio"] = close / ma
    ema50 = close.ewm(span=50, adjust=False).mean()
    o["price_to_ema50_ratio"] = close / ema50
    ma20 = close.rolling(20, min_periods=10).mean()
    ma50 = close.rolling(50, min_periods=25).mean()
    ma100 = close.rolling(100, min_periods=50).mean()
    ma200 = close.rolling(200, min_periods=100).mean()
    for a, b, fa, fb in [("20", "50", ma20, ma50), ("20", "100", ma20, ma100),
                         ("50", "100", ma50, ma100), ("50", "200", ma50, ma200),
                         ("100", "200", ma100, ma200)]:
        o[f"ma_cross_{a}_{b}"] = (fa >= fb).astype("int8")
    roll_max = close.rolling(252, min_periods=20).max()
    roll_min = close.rolling(252, min_periods=20).min()
    o["pct_from_52w_high"] = (close / roll_max - 1) * 100
    o["pct_from_52w_low"] = (close / roll_min - 1) * 100
    o["pct_from_all_time_high"] = (close / close.expanding().max() - 1) * 100
    rng = (roll_max - roll_min)
    o["pct_position_in_52w_range"] = (close - roll_min) / rng.replace(0, np.nan) * 100
    o["drawdown_from_high_pct"] = (close / close.expanding().max() - 1) * 100
    o["recovery_pct_from_low"] = (close / roll_min - 1) * 100
    for n in (5, 10, 20):
        o[f"volume_ratio_{n}d_avg"] = vol / vol.rolling(n, min_periods=n // 2).mean()
    ret = close.pct_change()
    o["volatility_20d_pct"] = ret.rolling(20).std() * np.sqrt(252) * 100
    tr = pd.concat([(high - low), (high - close.shift()).abs(),
                    (low - close.shift()).abs()], axis=1).max(axis=1)
    atr = tr.rolling(14).mean()
    o["atr_expansion_ratio"] = atr / atr.shift(63)
    o["rsi_14"] = rsi(close)
    for sd in (1.0, 1.5, 2.0, 2.5):
        band = ma20 + sd * close.rolling(20).std()
        o[f"bb_upper_break_{('%g' % sd)}sd"] = (close > band).astype("int8")
    for n in (5, 10, 20, 30, 50):
        o[f"price_slope_{n}d_pct"] = slope_pct(close, n)
    o["consecutive_up_weeks"] = consecutive_up_weeks(df).values
    o["days_since_listing"] = (df["date"] - df["date"].iloc[0]).dt.days
    # index-relative (beta, rel strength) from regime nifty500
    r = regime[["date", "nifty500_close"]].copy()
    m = o[["date"]].merge(r, on="date", how="left")
    # ffill: a single date where the stock traded but the index row is missing
    # would otherwise inject a NaN return that poisons every 252d rolling window.
    m["nifty500_close"] = m["nifty500_close"].ffill()
    idx_ret = m["nifty500_close"].pct_change(fill_method=None)
    cov = ret.rolling(252).cov(idx_ret)
    var = idx_ret.rolling(252).var()
    o["beta_1y"] = (cov / var).values
    o["rel_strength_vs_index_pct"] = (close.pct_change(63) * 100
                                      - m["nifty500_close"].pct_change(63, fill_method=None).values * 100)
    o["momentum_score"] = (o["price_return_3m_pct"].fillna(0)
                           + o["price_return_6m_pct"].fillna(0)
                           + o["price_return_12m_pct"].fillna(0)) / 3
    o["sector_rel_strength_pct"] = np.nan   # documented gap (no sector map)
    # compact
    for c in o.columns:
        if c != "date" and o[c].dtype != "int8":
            o[c] = o[c].astype("float32")
    return o


def load_or_init_ledger(keys) -> pd.DataFrame:
    if os.path.exists(LEDGER_PATH):
        led = pd.read_parquet(LEDGER_PATH)
        new = [k for k in keys if k not in set(led["guru_key"])]
        if new:
            led = pd.concat([led, pd.DataFrame({"guru_key": new, "status": "pending",
                             "rows": 0, "error": ""})], ignore_index=True)
        return led
    return pd.DataFrame({"guru_key": keys, "status": "pending", "rows": 0, "error": ""})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--retry-errors", action="store_true")
    ap.add_argument("--status", action="store_true")
    args = ap.parse_args()

    keys = [os.path.basename(f)[:-8] for f in glob.glob(os.path.join(OHLCV_DIR, "*.parquet"))]
    led = load_or_init_ledger(keys)
    if args.status:
        print(f"{led['status'].value_counts().to_dict()} | rows: "
              f"{int(led['rows'].sum()):,}")
        return

    regime = pd.read_parquet(os.path.join(DATA_DIR, "metrics", "regime.parquet"))
    todo_mask = led["status"].eq("pending")
    if args.retry_errors:
        todo_mask |= led["status"].eq("error")
    todo = led[todo_mask]
    if args.limit:
        todo = todo.head(args.limit)
    log(f"technical metric compute: {len(todo)} of {len(led)} keys")

    if args.dry_run:
        log("DRY RUN — first 10:"); [print("  ", k) for k in todo["guru_key"].head(10)]
        return

    os.makedirs(OUT_DIR, exist_ok=True)
    n_done = n_skip = n_err = 0
    for i, (idx, row) in enumerate(todo.iterrows(), 1):
        k = row["guru_key"]
        try:
            df = pd.read_parquet(os.path.join(OHLCV_DIR, f"{k}.parquet"))
            if len(df) < MIN_ROWS:
                led.at[idx, "status"] = "skip"; n_skip += 1
            else:
                out = compute(df, regime)
                out.to_parquet(os.path.join(OUT_DIR, f"{k}.parquet"), index=False)
                led.at[idx, "status"] = "done"; led.at[idx, "rows"] = len(out); n_done += 1
            led.at[idx, "error"] = ""
        except Exception as e:
            led.at[idx, "status"] = "error"; led.at[idx, "error"] = str(e)[:200]; n_err += 1
        if i % 300 == 0:
            os.makedirs(STATUS_DIR, exist_ok=True)
            led.to_parquet(LEDGER_PATH, index=False)
            log(f"  {i}/{len(todo)} (done={n_done} skip={n_skip} err={n_err})")
    os.makedirs(STATUS_DIR, exist_ok=True)
    led.to_parquet(LEDGER_PATH, index=False)
    log(f"RUN COMPLETE: done={n_done} skip={n_skip} err={n_err}")


if __name__ == "__main__":
    main()
