r"""
P4a — compute_regime.py  (Project Guru)

Market-wide daily regime series from guru/data/macro_hist/. Small single file;
recomputed whole each run. Output: guru/data/metrics/regime.parquet
  date, nifty500_close, nifty500_ma200, nifty500_above_200dma (0/1),
  nifty500_ret_1m/3m/6m/12m_pct (for rel-strength/excess later), india_vix

Usage: python guru/compute_regime.py
"""
from __future__ import annotations

import os
from datetime import datetime

import pandas as pd

GURU_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(GURU_DIR, "data")
MACRO_DIR = os.path.join(DATA_DIR, "macro_hist")
OUT_DIR = os.path.join(DATA_DIR, "metrics")


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def main() -> None:
    n500 = pd.read_parquet(os.path.join(MACRO_DIR, "NIFTY_500.parquet"),
                           columns=["date", "close"]).rename(
        columns={"close": "nifty500_close"})
    n500 = n500.sort_values("date").reset_index(drop=True)
    n500["nifty500_ma200"] = n500["nifty500_close"].rolling(200, min_periods=100).mean()
    n500["nifty500_above_200dma"] = (
        n500["nifty500_close"] > n500["nifty500_ma200"]).astype("Int8")
    for m, d in [("1m", 21), ("3m", 63), ("6m", 126), ("12m", 252)]:
        n500[f"nifty500_ret_{m}_pct"] = (
            n500["nifty500_close"].pct_change(d) * 100).round(2)

    vix = pd.read_parquet(os.path.join(MACRO_DIR, "INDIA_VIX.parquet"),
                          columns=["date", "close"]).rename(
        columns={"close": "india_vix"})
    out = n500.merge(vix, on="date", how="left")
    out["india_vix"] = out["india_vix"].ffill()

    os.makedirs(OUT_DIR, exist_ok=True)
    out.to_parquet(os.path.join(OUT_DIR, "regime.parquet"), index=False)
    log(f"regime.parquet: {len(out)} days {out['date'].min().date()} -> "
        f"{out['date'].max().date()}")
    log(f"  bull days (above 200dma): {int(out['nifty500_above_200dma'].sum())} "
        f"| VIX coverage from {out.dropna(subset=['india_vix'])['date'].min().date()}")


if __name__ == "__main__":
    main()
