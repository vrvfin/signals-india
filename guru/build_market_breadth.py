r"""
MARKET BREADTH HISTORY -- build_market_breadth.py  (Project Guru, LOCAL)

One row per (trading day, universe) for the whole 25-year price history. Answers
the question a single "as of today" cross-section cannot: how does market breadth
actually LOOK in a bull phase versus a bear phase.

Source: guru/data/ohlcv_hist/*.parquet -- 5,062 securities, 18.6M daily bars,
2001-07-03 onward, split/dividend adjusted, INCLUDING ~692 delisted names. That
last part matters more than anything else here: breadth computed only on today's
survivors makes every past bear market look mild.

Output: guru/data/metrics/market_breadth.parquet  (long: date x universe)

Four universes, each a time-varying mask evaluated per (security, day) with no
lookahead:
    all         every security with a bar that day
    nse         security carries an NSE symbol in universe_hist.parquet (~2,365)
    liquid_nse  NSE + 20-day average turnover >= LIQUID_CR crore ON THAT DAY
    top500_nse  NSE + that day's top TOP_N by 20-day average turnover

WHY THE TURNOVER COHORTS ARE NSE-ONLY -- this is a correction, not a preference.
yfinance reports almost no volume for BSE-only tickers: measured over the last 40
bars, BSE-only names have 33.5% zero-volume bars and 31% of those files are more
than half zeros, against 0.4% for NSE names. So a turnover filter applied to the
whole store does not select "liquid securities", it silently selects "NSE
securities" and calls them liquid. Making the restriction EXPLICIT is the honest
version of what the filter was already doing, and it is also what makes these
cohorts reproducible from Phase 1's Drive feed (which sources BSE volume from the
BSE API, not yfinance, and would otherwise disagree by 8-19 percentage points).

`all` therefore carries one caveat of its own: its turnover_* metrics include
BSE names whose volume is missing, so treat updown_vol_ratio on `all` as
indicative only. Its share-of-universe metrics are unaffected -- membership there
does not depend on volume.

top500_nse is a TURNOVER-RANK PROXY for large-cap, not index membership. We have
no historical Nifty 500 constituent list, and market_cap_current.csv is today's
segment only -- applying it to 2008 would label a stock by what it later became.
The proxy is at least derived from that day's own data.

Data quality: guru/data/_flagged_price_jumps.parquet marks |1-day return| > 40%.
Non-finite returns (prev_close == 0) and suspected split-adjustment failures are
dropped from the return-based metrics; genuine large moves are KEPT, because
"how many stocks moved more than 10%" is exactly what this table is for. The
equal-weight index uses a winsorized mean (+/- EW_CLIP%) so one bad print cannot
bend a 25-year line.

Cost: three passes over the store, ~2 minutes total, ~250 MB peak. Cheap enough
that there is no incremental state to drift out of sync -- it always rebuilds.

Usage:
    python guru/build_market_breadth.py --dry-run     # counts only, no write
    python guru/build_market_breadth.py --limit 200   # pilot
    python guru/build_market_breadth.py               # full (skips if fresh)
    python guru/build_market_breadth.py --force
    python guru/build_market_breadth.py --status
    python guru/build_market_breadth.py --upload    # + publish to Drive
"""
from __future__ import annotations

import argparse
import glob
import os
import time
from datetime import datetime

import numpy as np
import pandas as pd

GURU = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(GURU, "data")
OHLCV_DIR = os.path.join(DATA, "ohlcv_hist")
METRICS_DIR = os.path.join(DATA, "metrics")
OUT_PATH = os.path.join(METRICS_DIR, "market_breadth.parquet")
UNIVERSE_PATH = os.path.join(DATA, "universe_hist.parquet")
JUMPS_PATH = os.path.join(DATA, "_flagged_price_jumps.parquet")

MAS = (10, 20, 50, 100, 200)      # moving averages to measure breadth against
MOVES = (3, 5, 10)                # single-day move thresholds, percent
WIN_52W = 252
WIN_1M, WIN_3M = 21, 63
TOP_N = 500                       # size of the top500 turnover cohort
LIQUID_CR = 1.0                   # Rs crore, 20d average turnover
MIN_ROWS = 2                      # need a prior bar for a return
EW_CLIP = 25.0                    # winsorise daily returns for the EW index
DD_STEP = 2.0                     # drawdown histogram bucket width, percent
DD_NB = 51                        # buckets: 0..-100% in DD_STEP, plus overflow
COVERAGE_MIN = 0.60               # drop a date whose universe coverage collapsed
UNIVERSES = ("all", "nse", "liquid_nse", "top500_nse")


def log(m: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {m}", flush=True)


# --------------------------------------------------------------- metric layout
def metric_names() -> list[str]:
    """Column order of the accumulator. Every entry must be SUMMABLE across
    securities -- that is what lets the whole thing stream instead of
    materialising an 18M-row matrix."""
    n = ["n_traded", "n_ret"]
    n += [f"n_ma{m}" for m in MAS]
    n += [f"above_{m}dma" for m in MAS]
    n += ["n_ma50_200", "ma50_above_ma200"]
    n += ["adv", "dec", "unch"]
    for t in MOVES:
        n += [f"up_{t}", f"dn_{t}"]
    n += ["n_52w", "new_52w_high", "new_52w_low"]
    n += ["n_1m", "new_1m_high", "n_3m", "new_3m_high"]
    n += ["turnover_cr", "turnover_up_cr", "turnover_dn_cr"]
    n += ["n_dd", "n_dd20", "sum_ret_clipped"]
    n += [f"_ddb{i}" for i in range(DD_NB)]
    return n


MET = metric_names()
MIDX = {k: i for i, k in enumerate(MET)}


def _ma20_turnover(close: np.ndarray, volume: np.ndarray) -> np.ndarray:
    """20-day average turnover in Rs crore. min_periods=10 so a young listing
    becomes eligible for the liquidity masks after two weeks, not a month."""
    to = pd.Series(close * volume / 1e7)
    return to.rolling(20, min_periods=10).mean().to_numpy(dtype="float32")


def _read(path: str, cols: list[str]) -> pd.DataFrame | None:
    try:
        d = pd.read_parquet(path, columns=cols)
    except Exception:
        return None
    if len(d) < MIN_ROWS:
        return None
    return d.sort_values("date")


# ------------------------------------------------------------------ pass 0 / 1
def collect_dates(files: list[str]) -> pd.DatetimeIndex:
    """Union of every trading date present in the store. Built from the DATA,
    not from trading_calendar.parquet -- that calendar stops at 2026-07-03 while
    refresh_daily.py keeps the price files current."""
    # np.concatenate + np.unique, NOT set.update: the latter turns 18.6M
    # datetime64 values into Python objects one at a time and costs ~2 minutes.
    chunks = []
    for i, f in enumerate(files, 1):
        d = _read(f, ["date"])
        if d is not None:
            chunks.append(d["date"].to_numpy())
        if i % 2000 == 0:
            log(f"  dates {i}/{len(files)}")
    if not chunks:
        return pd.DatetimeIndex([])
    return pd.DatetimeIndex(np.unique(np.concatenate(chunks)))


def turnover_cut(files: list[str], keys: list[str], dates: pd.DatetimeIndex,
                 nse_keys: set) -> np.ndarray:
    """Per-date turnover of the TOP_N-th most traded security -- the top500
    cutoff. Ranked among NSE names ONLY -- see the module docstring: BSE
    turnover from yfinance is mostly zero, so ranking the whole store by
    turnover just reproduces the NSE list with extra steps.

    NaN on days holding fewer than TOP_N securities, so early history reports
    no top500 cohort rather than a silently smaller one."""
    n_d, n_s = len(dates), len(files)
    mat = np.full((n_d, n_s), -np.inf, dtype="float32")
    for j, (f, key) in enumerate(zip(files, keys)):
        if key not in nse_keys:
            continue                      # stays -inf: never in the cohort
        d = _read(f, ["date", "close", "volume"])
        if d is None:
            continue
        di = dates.get_indexer(pd.DatetimeIndex(d["date"]))
        ok = di >= 0
        t20 = _ma20_turnover(d["close"].to_numpy("float64"),
                             d["volume"].to_numpy("float64"))
        mat[di[ok], j] = np.nan_to_num(t20[ok], nan=-np.inf, posinf=-np.inf)
        if (j + 1) % 2000 == 0:
            log(f"  turnover {j + 1}/{n_s}")
    if len(nse_keys & set(keys)) < TOP_N:
        return np.full(n_d, np.nan, dtype="float64")
    kth = n_s - TOP_N
    cut = np.partition(mat, kth, axis=1)[:, kth].astype("float64")
    del mat
    cut[~np.isfinite(cut)] = np.nan
    return cut


# ---------------------------------------------------------------------- pass 2
def accumulate(files: list[str], keys: list[str], dates: pd.DatetimeIndex,
               cut: np.ndarray, nse_keys: set, bad: dict) -> dict:
    """Stream every security, adding its per-day flags into one
    (n_dates, n_metrics) array per universe."""
    n_d, n_m = len(dates), len(MET)
    acc = {u: np.zeros((n_d, n_m), dtype="float64") for u in UNIVERSES}
    cut_ok = np.isfinite(cut)
    n_bad = 0

    for j, (f, key) in enumerate(zip(files, keys), 1):
        d = _read(f, ["date", "high", "low", "close", "volume"])
        if d is None:
            continue
        dt = pd.DatetimeIndex(d["date"])
        di = dates.get_indexer(dt)
        if (di < 0).any():
            keep = di >= 0
            d, dt, di = d[keep], dt[keep], di[keep]
            if len(d) < MIN_ROWS:
                continue
        close = d["close"].to_numpy("float64")
        high = d["high"].to_numpy("float64")
        low = d["low"].to_numpy("float64")
        vol = d["volume"].to_numpy("float64")
        cs, hs, ls = pd.Series(close), pd.Series(high), pd.Series(low)
        n = len(close)
        M = np.zeros((n, n_m), dtype="float64")

        M[:, MIDX["n_traded"]] = 1.0

        # ---- single-day return. prev_close <= 0 yields inf, which is a broken
        # bar, not a 5,000% day. Flagged split-adjustment failures go too.
        prev = np.r_[np.nan, close[:-1]]
        with np.errstate(divide="ignore", invalid="ignore"):
            ret = np.where(prev > 0, (close / prev - 1.0) * 100.0, np.nan)
        ret[~np.isfinite(ret)] = np.nan
        drop = bad.get(key)
        if drop is not None:
            m = np.isin(dt.to_numpy(), drop)
            n_bad += int(m.sum())
            ret[m] = np.nan
        rv = np.isfinite(ret)
        M[:, MIDX["n_ret"]] = rv
        M[:, MIDX["adv"]] = rv & (ret > 0)
        M[:, MIDX["dec"]] = rv & (ret < 0)
        M[:, MIDX["unch"]] = rv & (ret == 0)
        for t in MOVES:
            M[:, MIDX[f"up_{t}"]] = rv & (ret >= t)
            M[:, MIDX[f"dn_{t}"]] = rv & (ret <= -t)
        M[:, MIDX["sum_ret_clipped"]] = np.where(
            rv, np.clip(ret, -EW_CLIP, EW_CLIP), 0.0)

        # ---- moving averages. STRICT min_periods: a "200 DMA" computed from
        # 100 bars is not a 200 DMA. Each MA carries its own denominator so a
        # percentage is never taken over names that do not have that MA yet.
        ma = {}
        for w in MAS:
            a = cs.rolling(w, min_periods=w).mean().to_numpy()
            ma[w] = a
            ok = np.isfinite(a)
            M[:, MIDX[f"n_ma{w}"]] = ok
            M[:, MIDX[f"above_{w}dma"]] = ok & (close > a)
        both = np.isfinite(ma[50]) & np.isfinite(ma[200])
        M[:, MIDX["n_ma50_200"]] = both
        M[:, MIDX["ma50_above_ma200"]] = both & (ma[50] > ma[200])

        # ---- 52-week extremes on INTRADAY high/low, strict 252-bar window
        hi52 = hs.rolling(WIN_52W, min_periods=WIN_52W).max().to_numpy()
        lo52 = ls.rolling(WIN_52W, min_periods=WIN_52W).min().to_numpy()
        ok52 = np.isfinite(hi52) & np.isfinite(lo52)
        M[:, MIDX["n_52w"]] = ok52
        M[:, MIDX["new_52w_high"]] = ok52 & (high >= hi52)
        M[:, MIDX["new_52w_low"]] = ok52 & (low <= lo52)
        for w, tag in ((WIN_1M, "1m"), (WIN_3M, "3m")):
            r = hs.rolling(w, min_periods=w).max().to_numpy()
            ok = np.isfinite(r)
            M[:, MIDX[f"n_{tag}"]] = ok
            M[:, MIDX[f"new_{tag}_high"]] = ok & (high >= r)

        # ---- drawdown from own 52w closing high, plus a histogram so that the
        # median is recoverable from summed counts alone
        cmax = cs.rolling(WIN_52W, min_periods=WIN_52W).max().to_numpy()
        with np.errstate(divide="ignore", invalid="ignore"):
            dd = np.where(cmax > 0, (close / cmax - 1.0) * 100.0, np.nan)
        okdd = np.isfinite(dd)
        M[:, MIDX["n_dd"]] = okdd
        M[:, MIDX["n_dd20"]] = okdd & (dd <= -20.0)
        b = np.clip(np.floor(-np.where(okdd, dd, 0.0) / DD_STEP),
                    0, DD_NB - 1).astype("int64")
        rows = np.nonzero(okdd)[0]
        if rows.size:
            np.add.at(M, (rows, MIDX["_ddb0"] + b[rows]), 1.0)

        # ---- turnover split by direction: distribution shows up here before it
        # shows up in the price index
        to = close * vol / 1e7
        to = np.where(np.isfinite(to), to, 0.0)
        M[:, MIDX["turnover_cr"]] = to
        M[:, MIDX["turnover_up_cr"]] = np.where(rv & (ret > 0), to, 0.0)
        M[:, MIDX["turnover_dn_cr"]] = np.where(rv & (ret < 0), to, 0.0)

        # ---- universe masks, then one vectorised add per universe. di is
        # unique within a security, so += on fancy indices is well defined.
        t20 = _ma20_turnover(close, vol)
        masks = {
            "all": np.ones(n, dtype=bool),
            "nse": np.full(n, key in nse_keys, dtype=bool),
            "liquid_nse": (np.full(n, key in nse_keys, dtype=bool)
                           & np.isfinite(t20) & (t20 >= LIQUID_CR)),
            "top500_nse": (np.full(n, key in nse_keys, dtype=bool)
                           & np.isfinite(t20) & cut_ok[di] & (t20 >= cut[di])),
        }
        for u, msk in masks.items():
            if msk.any():
                acc[u][di[msk]] += M[msk]
        if j % 500 == 0:
            log(f"  breadth {j}/{len(files)}")
    log(f"  return-bars dropped as bad prints: {n_bad:,}")
    return acc


# -------------------------------------------------------------------- assemble
def _pct(num, den):
    den = np.asarray(den, dtype="float64")
    out = np.full(den.shape, np.nan)
    ok = den > 0
    out[ok] = np.asarray(num, dtype="float64")[ok] / den[ok] * 100.0
    return out


def _ema(x: np.ndarray, span: int) -> np.ndarray:
    return pd.Series(x).ewm(span=span, adjust=False).mean().to_numpy()


def _median_from_hist(h: np.ndarray) -> np.ndarray:
    """Median drawdown, interpolated inside its DD_STEP-wide bucket. Accurate to
    well under a percentage point, and unlike a true median it survives being
    accumulated one security at a time."""
    tot = h.sum(axis=1)
    out = np.full(len(h), np.nan)
    ok = tot > 0
    if not ok.any():
        return out
    hh = h[ok]
    cum = np.cumsum(hh, axis=1)
    half = tot[ok] / 2.0
    idx = (cum < half[:, None]).sum(axis=1).clip(0, DD_NB - 1)
    ar = np.arange(len(idx))
    before = np.where(idx > 0, cum[ar, (idx - 1).clip(0)], 0.0)
    inb = hh[ar, idx]
    frac = np.where(inb > 0, (half - before) / inb, 0.0)
    out[ok] = -((idx + frac) * DD_STEP)
    return out


def usable_days(acc: dict, dates: pd.DatetimeIndex) -> tuple[np.ndarray, np.ndarray]:
    """Which dates are real trading days, judged by how much of the universe
    actually printed a bar.

    The store contains ~271 phantom dates: exchange holidays (Diwali, Bakrid,
    Gandhi Jayanti) and stray Saturdays where 5 to 80 securities out of ~4,500
    carry a bar. Breadth over five names is noise, and it would spike every
    chart. The last date can also be partial -- on 2026-08-28 only the NSE names
    had refreshed, so the universe silently became NSE-only.

    The split is cleanly bimodal: real days sit above 0.93 of their trailing
    median, phantom days below 0.02. COVERAGE_MIN sits in the empty gap.

    Judged on the `all` universe and applied to every universe: if the broad
    universe's composition collapsed, no cohort drawn from it is comparable to
    the rest of the series that day."""
    n = pd.Series(acc["all"][:, MIDX["n_traded"]])
    med = n.shift(1).rolling(21, min_periods=5).median()
    ratio = (n / med).to_numpy()
    keep = ~(np.isfinite(ratio) & (ratio < COVERAGE_MIN))
    return keep, ratio


def assemble(acc: dict, dates: pd.DatetimeIndex, ratio: np.ndarray) -> pd.DataFrame:
    frames = []
    for u in UNIVERSES:
        a = acc[u]
        g = {k: a[:, MIDX[k]] for k in MET if not k.startswith("_ddb")}
        o = pd.DataFrame({"date": dates, "universe": u})
        for k, v in g.items():
            o[k] = v
        o["coverage_ratio"] = ratio
        for w in MAS:
            o[f"pct_above_{w}dma"] = _pct(g[f"above_{w}dma"], g[f"n_ma{w}"])
        o["pct_ma50_above_ma200"] = _pct(g["ma50_above_ma200"], g["n_ma50_200"])
        for t in MOVES:
            o[f"pct_up_{t}"] = _pct(g[f"up_{t}"], g["n_ret"])
            o[f"pct_dn_{t}"] = _pct(g[f"dn_{t}"], g["n_ret"])
        adv, dec = g["adv"], g["dec"]
        o["pct_advancing"] = _pct(adv, adv + dec)
        o["ad_net"] = adv - dec
        o["ad_line"] = np.cumsum(adv - dec)
        # Universe size runs from a few hundred names in 2001 to ~5,000 today,
        # so the RAW A/D line is not comparable across the sample. The
        # normalised one is, and it is what the page plots.
        o["ad_line_norm"] = np.cumsum(
            np.nan_to_num(_pct(adv - dec, g["n_ret"]) / 100.0))
        o["pct_new_52w_high"] = _pct(g["new_52w_high"], g["n_52w"])
        o["pct_new_52w_low"] = _pct(g["new_52w_low"], g["n_52w"])
        o["net_hl"] = g["new_52w_high"] - g["new_52w_low"]
        o["pct_net_hl"] = _pct(g["new_52w_high"] - g["new_52w_low"], g["n_52w"])
        o["pct_new_1m_high"] = _pct(g["new_1m_high"], g["n_1m"])
        o["pct_new_3m_high"] = _pct(g["new_3m_high"], g["n_3m"])
        den = g["turnover_up_cr"] + g["turnover_dn_cr"]
        o["pct_turnover_up"] = _pct(g["turnover_up_cr"], den)
        with np.errstate(divide="ignore", invalid="ignore"):
            o["updown_vol_ratio"] = np.where(
                g["turnover_dn_cr"] > 0,
                g["turnover_up_cr"] / g["turnover_dn_cr"], np.nan)
        o["pct_dd20"] = _pct(g["n_dd20"], g["n_dd"])
        o["median_dd_from_52w_high"] = _median_from_hist(
            a[:, MIDX["_ddb0"]:MIDX["_ddb0"] + DD_NB])
        # Equal-weight market: mean daily return of the cohort, chained. The
        # mean IS the equal-weighted index return; the +/-EW_CLIP winsorisation
        # stops one bad print from bending a 25-year line.
        # NB: a plain divide, not _pct -- sum_ret_clipped is already in percent,
        # so scaling by 100 again would put the mean at +/-2,500% and send the
        # chained index to inf.
        mr = np.divide(g["sum_ret_clipped"], g["n_ret"],
                       out=np.full(len(dates), np.nan, dtype="float64"),
                       where=g["n_ret"] > 0)
        o["mean_ret_1d"] = mr
        o["eqw_index"] = 100.0 * np.cumprod(1.0 + np.nan_to_num(mr) / 100.0)
        # Ratio-adjusted McClellan -- the raw-difference form is meaningless
        # when the universe quadruples over the sample.
        rana = np.nan_to_num(_pct(adv - dec, adv + dec) * 10.0)
        o["mcclellan_osc"] = _ema(rana, 19) - _ema(rana, 39)
        o["mcclellan_summation"] = np.cumsum(
            np.nan_to_num(o["mcclellan_osc"].to_numpy()))
        thrust = pd.Series(np.nan_to_num(_pct(adv, adv + dec) / 100.0)).ewm(
            span=10, adjust=False).mean().to_numpy()
        o["zweig_thrust"] = thrust
        low10 = pd.Series(thrust).rolling(10, min_periods=1).min().to_numpy()
        o["zweig_signal"] = (thrust > 0.615) & (low10 < 0.40)
        frames.append(o)
    out = pd.concat(frames, ignore_index=True)
    for c in out.columns:
        if c in ("date", "universe"):
            continue
        out[c] = out[c].astype(bool) if c == "zweig_signal" else out[c].astype("float32")
    return out


# ------------------------------------------------------------------------ main
def newest_source_mtime(files: list[str]) -> float:
    return max((os.path.getmtime(f) for f in files), default=0.0)


# Drive lands in company_repo/_index/, matching build_price_highs.py's
# price_highs.parquet. Imports are LAZY and the flag is opt-in, so a plain local
# build needs no credentials and cannot fail on a Drive outage.
DRIVE_TABLE = "market_breadth.parquet"


def upload_to_drive(df: pd.DataFrame) -> None:
    """Publish the seeded table. This is a ONE-TIME EXTRACTION uploaded for
    viewing, not a runtime dependency -- exactly the relationship
    build_price_highs.py has with this same guru store. Nothing in CI reads it
    yet; wiring a daily append is a separate, deliberate change."""
    import sys as _sys
    scripts = os.path.join(os.path.dirname(GURU), "scripts")
    if scripts not in _sys.path:
        _sys.path.insert(0, scripts)
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(GURU), ".env"))
    from _extractor_base import (get_drive, get_or_create_subfolder,
                                 save_parquet)
    drive = get_drive()
    fid = os.environ["GDRIVE_FOLDER_ID"]
    for part in ("company_repo", "_index"):
        fid = get_or_create_subfolder(drive, fid, part)
    save_parquet(drive, fid, DRIVE_TABLE, df)
    log(f"uploaded company_repo/_index/{DRIVE_TABLE} "
        f"({len(df):,} rows, {df['date'].min().date()} -> {df['date'].max().date()})")


def store_last_dates(files: list[str]) -> pd.Series:
    """Last bar date of every price file (one tiny column read each)."""
    out = {}
    for f in files:
        d = _read(f, ["date"])
        if d is not None and len(d):
            out[os.path.basename(f)[:-8]] = pd.Timestamp(d["date"].max()).normalize()
    return pd.Series(out, dtype="datetime64[ns]")


def show_status() -> None:
    if not os.path.exists(OUT_PATH):
        print(f"{OUT_PATH} -- not built yet")
        return
    d = pd.read_parquet(OUT_PATH)
    print(f"rows            : {len(d):,}")
    print(f"universes       : {sorted(d['universe'].unique())}")
    print(f"date range      : {d['date'].min().date()} -> {d['date'].max().date()}")
    print(f"trading days    : {d['date'].nunique():,}")
    files = glob.glob(os.path.join(OHLCV_DIR, "*.parquet"))
    # Two different questions, answered separately. "Rebuilt since the price
    # files last changed?" (file times) used to be printed as "fresh" on its
    # own — while the table quietly ended weeks before the newest bar, because
    # days where most of the store had no price yet are dropped by
    # COVERAGE_MIN. So also compare the CONTENT: table end vs price-store end.
    rebuilt = os.path.getmtime(OUT_PATH) >= newest_source_mtime(files)
    print(f"rebuilt since   : {'yes' if rebuilt else 'NO -- price files changed, rerun'}"
          f" (file times)")
    last = store_last_dates(files)
    t_end = pd.Timestamp(d["date"].max()).normalize()
    if not last.empty:
        newest = last.max()
        n_new = int((last >= newest).sum())
        recent = last[last >= newest - pd.Timedelta(days=60)]
        print(f"price store end : newest bar {newest.date()} in {n_new:,} of "
              f"{len(last):,} files")
        for dt, n in recent.value_counts().head(3).sort_index(ascending=False).items():
            print(f"                  {n:,} files end {dt.date()}")
        if t_end < newest:
            behind = int((last >= t_end + pd.Timedelta(days=1)).sum())
            print(f"vs price store  : TABLE ENDS {t_end.date()}, {(newest - t_end).days}d "
                  f"before the newest bar. Later days are dropped on purpose: under "
                  f"{COVERAGE_MIN:.0%} of securities have a price for them (only "
                  f"{behind:,} files go past {t_end.date()}). Fix = refresh the "
                  f"lagging part of the price store, then rebuild.")
        else:
            print("vs price store  : up to date (table reaches the newest bar)")
    last = d[d["date"].eq(d["date"].max()) & d["universe"].eq("all")].iloc[0]
    print(f"last day (all)  : {int(last['n_traded']):,} traded, "
          f"{last['pct_above_50dma']:.1f}% >50DMA, "
          f"{int(last['new_52w_high'])}H/{int(last['new_52w_low'])}L")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true",
                    help="report what would be built; write nothing")
    ap.add_argument("--limit", type=int, default=0, help="pilot on N securities")
    ap.add_argument("--force", action="store_true", help="rebuild even if fresh")
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--no-jump-filter", action="store_true",
                    help="keep suspected bad prints in the return metrics")
    ap.add_argument("--upload", action="store_true",
                    help="also publish the table to Drive "
                         "company_repo/_index/market_breadth.parquet")
    args = ap.parse_args()

    if args.status:
        show_status()
        return

    files = sorted(glob.glob(os.path.join(OHLCV_DIR, "*.parquet")))
    if not files:
        log(f"no price files in {OHLCV_DIR} -- nothing to do")
        return
    if args.limit:
        files = files[:args.limit]
    keys = [os.path.basename(f)[:-8] for f in files]

    if (not args.force and not args.limit and os.path.exists(OUT_PATH)
            and os.path.getmtime(OUT_PATH) >= newest_source_mtime(files)):
        log(f"{os.path.basename(OUT_PATH)} is newer than every price file -- "
            f"skipping rebuild (use --force to force one); table ends "
            f"{pd.read_parquet(OUT_PATH, columns=['date'])['date'].max().date()} "
            f"-- run --status to compare with the price store's end")
        if args.upload:
            upload_to_drive(pd.read_parquet(OUT_PATH))
        return

    nse_keys: set = set()
    if os.path.exists(UNIVERSE_PATH):
        u = pd.read_parquet(UNIVERSE_PATH, columns=["guru_key", "nse_symbol"])
        nse_keys = set(u.loc[u["nse_symbol"].notna(), "guru_key"])
    log(f"securities: {len(files):,} | with an NSE symbol: "
        f"{len(nse_keys & set(keys)):,}")

    bad: dict = {}
    if not args.no_jump_filter and os.path.exists(JUMPS_PATH):
        jd = pd.read_parquet(JUMPS_PATH)
        r = pd.to_numeric(jd["return"], errors="coerce")
        drop = jd[jd["suspected_split"].astype(bool) | ~np.isfinite(r)]
        bad = {k: g["date"].to_numpy() for k, g in drop.groupby("guru_key")}
        log(f"bad-print filter: {len(drop):,} bar-days over {len(bad):,} "
            f"securities (suspected splits + non-finite returns)")

    t0 = time.time()
    log("pass 0/3 -- collecting the trading-date index")
    dates = collect_dates(files)
    if len(dates) == 0:
        log("no dates found -- nothing to do")
        return
    log(f"  {len(dates):,} trading days, {dates[0].date()} -> {dates[-1].date()}")

    if args.dry_run:
        log(f"DRY-RUN -- would build {len(dates):,} days x {len(UNIVERSES)} "
            f"universes = {len(dates) * len(UNIVERSES):,} rows "
            f"from {len(files):,} securities.")
        log(f"DRY-RUN -- would write {OUT_PATH}. Nothing written.")
        return

    log("pass 1/3 -- per-day turnover cutoff for top500_nse (ranked within NSE)")
    cut = turnover_cut(files, keys, dates, nse_keys)
    log(f"  top500_nse cohort exists on {int(np.isfinite(cut).sum()):,} of "
        f"{len(dates):,} days")

    log("pass 2/3 -- streaming breadth flags into the accumulators")
    acc = accumulate(files, keys, dates, cut, nse_keys, bad)

    log("assembling")
    # Drop phantom/partial dates BEFORE assembling: the A/D line, the
    # equal-weight index and the McClellan summation are cumulative, so a
    # 5-security holiday left in here would be baked into every later value.
    keep, ratio = usable_days(acc, dates)
    n_drop = int((~keep).sum())
    if n_drop:
        dropped = dates[~keep]
        log(f"  dropping {n_drop} date(s) below {COVERAGE_MIN:.0%} universe "
            f"coverage (holidays / partial ingest); most recent: "
            f"{', '.join(str(x.date()) for x in dropped[-3:])}")
        dates = dates[keep]
        acc = {u: a[keep] for u, a in acc.items()}
        ratio = ratio[keep]
    out = assemble(acc, dates, ratio)
    os.makedirs(METRICS_DIR, exist_ok=True)
    out.to_parquet(OUT_PATH, index=False)
    log(f"wrote {OUT_PATH} -- {len(out):,} rows x {len(out.columns)} cols "
        f"({os.path.getsize(OUT_PATH)/1e6:.1f} MB) in {time.time()-t0:.0f}s")
    if args.upload:
        upload_to_drive(out)
    for u in UNIVERSES:
        s = out[out["universe"].eq(u)].iloc[-1]
        log(f"  {u:<7} last day: n={int(s['n_traded']):>5,} "
            f">50DMA {s['pct_above_50dma']:>5.1f}%  "
            f">200DMA {s['pct_above_200dma']:>5.1f}%  "
            f"{int(s['new_52w_high']):>3}H/{int(s['new_52w_low']):<3}L")


if __name__ == "__main__":
    main()
