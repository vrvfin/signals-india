r"""
DAILY REFRESH — refresh_daily.py  (Project Guru)

Keeps guru/data/ current. The original dump scripts fetch 25 YEARS and mark every
security "done", so re-running them does nothing — that is why the store went 8
weeks stale. This script does INCREMENTAL top-ups instead.

Chain (each step resumable, each can run alone):
  --prices    fetch the last PERIOD of bars for every security, append only the
              NEW dates to its parquet (dedupe on date), update the ledger
  --technical recompute the ~50 technical metrics for securities whose price file
              actually changed (skips the rest — that is the whole speed-up)
  --regime    recompute the market regime series (fast, always safe)
  --all       prices -> technical -> regime

Quarterly fundamentals are NOT refreshed here — they move on results season, not
daily. Use the existing dump_nse_results_hist.py / dump_fundamentals_hist.py and
then merge_quarterly_unified.py when a new quarter lands.

Usage:
    python guru/refresh_daily.py --all
    python guru/refresh_daily.py --prices --limit 200     # pilot
    python guru/refresh_daily.py --status
"""
from __future__ import annotations
import argparse, glob, os, time
from datetime import datetime
import numpy as np, pandas as pd

GURU = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(GURU, "data")
OHLCV = os.path.join(DATA, "ohlcv_hist")
TECH = os.path.join(DATA, "metrics", "technical")
STATUS = os.path.join(DATA, "_dump_status")
LEDGER = os.path.join(STATUS, "ohlcv_ledger.parquet")
REFRESH_LOG = os.path.join(STATUS, "refresh_log.parquet")

PERIOD = "3mo"          # enough to cover a stale gap; overlap is deduped
BATCH = 40
PAUSE = 0.6


def log(m): print(f"[{datetime.now().strftime('%H:%M:%S')}] {m}", flush=True)


def refresh_prices(limit: int = 0) -> set:
    """append new bars to each security's parquet. Returns keys that CHANGED."""
    import yfinance as yf
    led = pd.read_parquet(LEDGER)
    work = led[led.status == "done"][["guru_key", "yf_ticker"]]
    if limit:
        work = work.head(limit)
    log(f"price refresh: {len(work):,} securities (period={PERIOD})")
    changed, added_total, failed = set(), 0, 0
    tick2key = dict(zip(work.yf_ticker, work.guru_key))
    tickers = work.yf_ticker.tolist()

    for bi in range(0, len(tickers), BATCH):
        batch = tickers[bi:bi + BATCH]
        try:
            raw = yf.download(batch, period=PERIOD, interval="1d",
                              group_by="ticker", auto_adjust=True,
                              threads=True, progress=False)
        except Exception as e:
            failed += len(batch)
            log(f"  batch {bi//BATCH} failed: {str(e)[:70]}")
            continue
        if raw is None or raw.empty:
            continue
        frames = ({batch[0]: raw} if len(batch) == 1
                  else {t: raw[t] for t in batch
                        if t in raw.columns.get_level_values(0)})
        for t, new in frames.items():
            gk = tick2key.get(t)
            if gk is None:
                continue
            new = new.dropna(subset=["Close"])
            if new.empty:
                continue
            new = new.reset_index()
            new.columns = [str(c).lower() for c in new.columns]
            new = new.rename(columns={"index": "date"})
            new["date"] = pd.to_datetime(new["date"]).dt.tz_localize(None).dt.normalize()
            new = new[["date", "open", "high", "low", "close", "volume"]]
            new["adj_basis"] = True
            fp = os.path.join(OHLCV, f"{gk}.parquet")
            if os.path.exists(fp):
                old = pd.read_parquet(fp)
                last = old["date"].max()
                add = new[new["date"] > last]
                if add.empty:
                    continue
                out = pd.concat([old, add], ignore_index=True)
                out = out.drop_duplicates(subset=["date"], keep="last").sort_values("date")
            else:
                out, add = new, new
            out.to_parquet(fp, index=False)
            changed.add(gk)
            added_total += len(add)
        if (bi // BATCH) % 25 == 0:
            log(f"  {bi + len(batch)}/{len(tickers)} | changed {len(changed)} | "
                f"+{added_total:,} bars")
        time.sleep(PAUSE)
    log(f"PRICES DONE: {len(changed):,} securities updated, +{added_total:,} bars, "
        f"{failed} fetch failures")
    return changed


def refresh_technical(keys: set | None = None, limit: int = 0):
    """recompute technical metrics ONLY for changed securities."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "ctm", os.path.join(GURU, "compute_technical_metrics.py"))
    ctm = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ctm)
    regime = pd.read_parquet(os.path.join(DATA, "metrics", "regime.parquet"))
    if keys is None:
        keys = {os.path.basename(f)[:-8] for f in glob.glob(os.path.join(OHLCV, "*.parquet"))}
    keys = sorted(keys)
    if limit:
        keys = keys[:limit]
    log(f"technical recompute: {len(keys):,} securities")
    ok = err = 0
    for i, gk in enumerate(keys, 1):
        try:
            df = pd.read_parquet(os.path.join(OHLCV, f"{gk}.parquet"))
            if len(df) < 30:
                continue
            out = ctm.compute(df, regime)
            out.to_parquet(os.path.join(TECH, f"{gk}.parquet"), index=False)
            ok += 1
        except Exception:
            err += 1
        if i % 500 == 0:
            log(f"  {i}/{len(keys)} (ok={ok} err={err})")
    log(f"TECHNICAL DONE: ok={ok} err={err}")


def refresh_regime():
    import subprocess, sys
    log("regime recompute…")
    subprocess.run([sys.executable, os.path.join(GURU, "compute_regime.py")],
                   check=False)


def show_status():
    fs = glob.glob(os.path.join(OHLCV, "*.parquet"))
    import random
    random.seed(0)
    mx = []
    for f in random.sample(fs, min(300, len(fs))):
        d = pd.read_parquet(f, columns=["date"])
        if len(d):
            mx.append(pd.to_datetime(d["date"]).max())
    s = pd.Series(mx)
    today = pd.Timestamp(datetime.now().date())
    print(f"today            : {today.date()}")
    print(f"latest price bar : {s.max().date()}  ({(today - s.max()).days} days old)")
    print(f"median security  : {s.median().date()}")
    print(f"securities       : {len(fs):,}")
    q = glob.glob(os.path.join(DATA, "metrics", "quarterly_unified", "*.parquet"))
    if q:
        qm = []
        for f in random.sample(q, min(200, len(q))):
            d = pd.read_parquet(f, columns=["period_end"])
            if len(d):
                qm.append(pd.to_datetime(d["period_end"]).max())
        print(f"latest quarter   : {pd.Series(qm).max().date()} "
              f"(refresh via NSE/Screener dumps when a new season lands)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prices", action="store_true")
    ap.add_argument("--technical", action="store_true")
    ap.add_argument("--regime", action="store_true")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    if args.status:
        show_status(); return
    changed = None
    if args.prices or args.all:
        changed = refresh_prices(args.limit)
        os.makedirs(STATUS, exist_ok=True)
        pd.DataFrame({"guru_key": sorted(changed),
                      "refreshed_at": datetime.now().isoformat(timespec="seconds")}
                     ).to_parquet(REFRESH_LOG, index=False)
    if args.regime or args.all:
        refresh_regime()
    if args.technical or args.all:
        if changed is None and os.path.exists(REFRESH_LOG):
            changed = set(pd.read_parquet(REFRESH_LOG)["guru_key"])
        refresh_technical(changed, args.limit)
    if not any([args.prices, args.technical, args.regime, args.all]):
        show_status()


if __name__ == "__main__":
    main()
