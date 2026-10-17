r"""
tech_history.py — a stock's own price history, turned into "is today unusual,
has it happened before, and what happened next".

Pure functions only: no Drive, no network, no file I/O. build_pf_history.py
loads the data and renders; everything here is testable on synthetic series
(scripts/tests/test_tech_history.py).

Every fact follows one shape:
  * how rare today is      -> percentile of today's value in the stock's past
  * how often it happened  -> separate EPISODES (a 10-day run counts once)
  * what happened next     -> 5/20/60-trading-day forward return after each past
                              episode: median, % up, worst
An episode count under THIN_N is labelled thin and still shown — small samples
are never dropped (Guru rule).

Causality: every rolling window and every pattern looks only at bars up to and
including bar t, so the history used to judge "today" is the same history a
person would have had on that day.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

HORIZONS = (5, 20, 60)
MAS = (10, 20, 30, 50, 60, 200)
THIN_N = 5
JUMP_MAX = 0.40          # |1-day return| above this is bad data, not a move
MIN_BARS_HISTORY = 252   # below this: "too new" — no percentiles / outcomes
YEAR = 248               # NSE trading days per year (approx.)


# ───────────────────────────── series prep ──────────────────────────────────
def _norm(df: pd.DataFrame) -> pd.DataFrame:
    """date-sorted OHLCV with a DatetimeIndex-free 'date' column; one row/day."""
    if df is None or df.empty:
        return pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume"])
    d = df.copy()
    if "date" not in d.columns:
        d = d.reset_index().rename(columns={d.index.name or "index": "date"})
    d["date"] = pd.to_datetime(d["date"], errors="coerce").dt.tz_localize(None).dt.normalize()
    for c in ("open", "high", "low", "close", "volume"):
        if c not in d.columns:
            d[c] = np.nan
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d.dropna(subset=["date", "close"])
    d = d[d["close"] > 0]
    return (d.drop_duplicates("date", keep="last").sort_values("date")
             .reset_index(drop=True)[["date", "open", "high", "low", "close", "volume"]])


def join_series(local: pd.DataFrame, drive: pd.DataFrame, min_overlap: int = 20,
                tol: float = 0.02):
    """Deep local history + fresh Drive history -> (df, note).

    Drive wins on every shared date (same rule as build_market_state_html.
    _close_series). Local is used only for dates BEFORE Drive's first bar, so
    the scale that matters is the one at that junction: the median Drive/local
    close ratio over the first `min_overlap` shared days. If it is off by more
    than `tol` (a split/bonus one source has adjusted and the other has not),
    the local segment is rescaled to Drive's basis and the note says so.
    Under `min_overlap` shared days the junction cannot be checked, so the
    result is Drive only."""
    lo, dr = _norm(local), _norm(drive)
    if dr.empty and lo.empty:
        return lo, "no price data"
    if dr.empty:
        return lo, "local store only (no Drive prices)"
    if lo.empty:
        return dr, "Drive only (not in the local store)"
    ov = lo.merge(dr, on="date", suffixes=("_l", "_d"))
    if len(ov) < min_overlap:
        return dr, f"Drive only (only {len(ov)} days overlap the local store)"
    head = ov.head(min_overlap)
    ratio = float((head["close_d"] / head["close_l"]).median())
    pre = lo[lo["date"] < dr["date"].iloc[0]].copy()
    note = f"joined local+Drive at {dr['date'].iloc[0].date()}"
    if abs(ratio - 1) > tol:
        for c in ("open", "high", "low", "close"):
            pre[c] = pre[c] * ratio
        pre["volume"] = pre["volume"] / ratio
        note += f"; local rescaled x{ratio:.4f} (adjustment mismatch)"
    out = pd.concat([pre, dr], ignore_index=True)
    return out.sort_values("date").reset_index(drop=True), note


def returns(df: pd.DataFrame) -> pd.Series:
    """Close-to-close returns with bad-data bars (|r| > JUMP_MAX) set to NaN, so
    a missed split never counts as a 'big day'."""
    r = df["close"].pct_change()
    return r.where(r.abs() <= JUMP_MAX)


def sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n, min_periods=n).mean()


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    pc = df["close"].shift()
    tr = pd.concat([df["high"] - df["low"], (df["high"] - pc).abs(),
                    (df["low"] - pc).abs()], axis=1).max(axis=1)
    return tr.rolling(n, min_periods=n).mean()


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    rs = up / dn.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


# ─────────────────────────── counting machinery ─────────────────────────────
def run_lengths(state: pd.Series) -> pd.Series:
    """Length of the current run of equal values at each bar (1, 2, 3 ...).
    NaN state breaks a run and has NaN length."""
    s = state.astype("object").where(state.notna(), None)
    rid = (s != s.shift()).cumsum()
    ln = s.groupby(rid).cumcount() + 1
    return ln.where(state.notna()).astype(float)


def episodes(cond: pd.Series, cooldown: int = 5) -> np.ndarray:
    """Positions where `cond` turns True after at least `cooldown` bars of not
    being True — one position per episode, so a 10-day run counts once."""
    c = cond.fillna(False).astype(bool).to_numpy()
    out, last = [], -10**9
    for i, v in enumerate(c):
        if v:
            if i - last > cooldown:
                out.append(i)
            last = i
    return np.array(out, dtype=int)


def forward_stats(close: pd.Series, pos, horizons=HORIZONS) -> dict:
    """{h: {n, median, pct_up, worst}} of close[p+h]/close[p]-1 over past
    positions whose outcome is already known (p+h inside the series)."""
    c = close.to_numpy(dtype=float)
    out = {}
    for h in horizons:
        p = np.asarray([x for x in pos if x + h < len(c)], dtype=int)
        if len(p) == 0:
            out[h] = {"n": 0, "median": None, "pct_up": None, "worst": None}
            continue
        r = c[p + h] / c[p] - 1
        out[h] = {"n": int(len(r)), "median": float(np.median(r)),
                  "pct_up": float((r > 0).mean() * 100), "worst": float(r.min())}
    return out


def pct_rank(hist: pd.Series, cur) -> float | None:
    """Percentile (0-100) of `cur` within `hist` (NaNs ignored)."""
    h = pd.Series(hist).dropna()
    if cur is None or pd.isna(cur) or h.empty:
        return None
    return float((h < cur).mean() * 100 + (h == cur).mean() * 50)


def _fact(key, text, cur, pct, pos_past, close, side=None):
    """One fact row. `pos_past` are past episode starts (current one excluded)."""
    n = int(len(pos_past))
    return {"key": key, "text": text, "value": cur, "pct": pct, "n": n,
            "thin": n < THIN_N, "fwd": forward_stats(close, pos_past), "side": side}


# ───────────────────────────── 1) personality ───────────────────────────────
def personality(df: pd.DataFrame, nifty: pd.Series | None = None) -> dict:
    df = _norm(df)
    n = len(df)
    out = {"bars": n, "years": round(n / YEAR, 1),
           "first": df["date"].iloc[0].date() if n else None}
    if n < 30:
        return out
    r = returns(df)
    yrs = max(r.notna().sum() / YEAR, 1 / YEAR)
    out["pct_up_days"] = float((r > 0).sum() / r.notna().sum() * 100)
    out["median_abs_move"] = float(r.abs().median() * 100)
    out["vol_annual"] = float(r.std() * np.sqrt(YEAR) * 100)
    out["big_days_per_year"] = {f"{s}{k}": float(((r >= k / 100) if s == "+" else
                                                  (r <= -k / 100)).sum() / yrs)
                                for k in (3, 5, 10) for s in ("+", "-")}
    out["best_day"], out["worst_day"] = float(r.max() * 100), float(r.min() * 100)

    sign = np.sign(r).where(r.notna() & (r != 0))
    ln = run_lengths(sign)
    ends = ln[(sign != sign.shift(-1)) & sign.notna()]
    for lbl, v in (("up", 1), ("down", -1)):
        runs = ends[sign.loc[ends.index] == v]
        out[f"streak_{lbl}"] = {"median": float(runs.median()) if len(runs) else None,
                                "max": int(runs.max()) if len(runs) else 0,
                                "five_plus_per_year": float((runs >= 5).sum() / yrs)}

    pc = df["close"].shift()
    gu = df["open"] >= pc * 1.02
    gd = df["open"] <= pc * 0.98
    out["gaps"] = {
        "up_per_year": float(gu.sum() / yrs), "down_per_year": float(gd.sum() / yrs),
        "up_fill_pct": float((df["low"][gu] <= pc[gu]).mean() * 100) if gu.any() else None,
        "down_fill_pct": float((df["high"][gd] >= pc[gd]).mean() * 100) if gd.any() else None}

    a = atr(df) / df["close"] * 100
    out["atr_pct_now"] = float(a.iloc[-1]) if a.notna().any() else None
    out["atr_pct_median"] = float(a.median()) if a.notna().any() else None

    ma_stats = {}
    for m in MAS:
        if n < m + 20:
            continue
        st = (df["close"] > sma(df["close"], m)).where(sma(df["close"], m).notna())
        ln_m = run_lengths(st)
        e = ln_m[(st != st.shift(-1)) & st.notna()]
        ma_stats[m] = {
            "pct_time_above": float(st.dropna().mean() * 100),
            "median_stay_above": float(e[st.loc[e.index] == True].median()) if (st.loc[e.index] == True).any() else None,   # noqa: E712
            "median_stay_below": float(e[st.loc[e.index] == False].median()) if (st.loc[e.index] == False).any() else None}  # noqa: E712
    out["ma"] = ma_stats

    peak = df["close"].cummax()
    dd = df["close"] / peak - 1
    out["max_drawdown"] = float(dd.min() * 100)
    out["drawdown_now"] = float(dd.iloc[-1] * 100)
    seg = (df["close"] >= peak).cumsum()
    troughs, recov = [], []
    for _, g in dd.groupby(seg):
        t = g.min()
        if t <= -0.10:
            troughs.append(t * 100)
            ti = g.idxmin()
            if g.index[-1] < len(df) - 1:          # a new high followed
                recov.append(int(g.index[-1] + 1 - ti))
    out["corrections_10pct"] = {"count": len(troughs),
                                "median_depth": float(np.median(troughs)) if troughs else None,
                                "median_recovery_days": float(np.median(recov)) if recov else None}

    v = df["volume"]
    spike = v >= 2 * v.rolling(50, min_periods=20).median().shift()
    out["volume_spikes_per_year"] = float(spike.sum() / yrs)

    out["beta_1y"] = beta(df, nifty)
    return out


def beta(df: pd.DataFrame, idx: pd.Series | None, bars: int = YEAR) -> float | None:
    if idx is None or len(idx) == 0:
        return None
    s = df.set_index("date")["close"].pct_change()
    i = idx.pct_change()
    j = pd.concat([s, i], axis=1, join="inner").dropna().tail(bars)
    if len(j) < 60 or j.iloc[:, 1].var() == 0:
        return None
    return float(j.cov().iloc[0, 1] / j.iloc[:, 1].var())


# ───────────────────────────── 2) is now unusual ────────────────────────────
def unusual(df: pd.DataFrame) -> list[dict]:
    """Facts about TODAY, each judged against this stock's own past."""
    df = _norm(df)
    n = len(df)
    if n < MIN_BARS_HISTORY:
        return []
    c = df["close"]
    r = returns(df)
    facts = []
    last = n - 1

    # moves over 1 / 5 / 20 days: "moved this much or more, this way"
    bad = (c.pct_change().abs() > JUMP_MAX).astype(float)
    for k, lbl in ((1, "today"), (5, "5 days"), (20, "20 days")):
        # a window containing a bad-data bar is not a real k-day move
        rk = c.pct_change(k).where(bad.rolling(k, min_periods=1).max() == 0) if k > 1 else r
        cur = rk.iloc[-1]
        if pd.isna(cur) or cur == 0:
            continue
        hist = rk.iloc[:-1]
        cond = (rk >= cur) if cur > 0 else (rk <= cur)
        pos = episodes(cond.iloc[:-1], cooldown=max(k, 5))
        facts.append(_fact(f"move_{k}d", f"{'+' if cur > 0 else ''}{cur * 100:.1f}% "
                           + ("today" if k == 1 else f"over {lbl}"),
                           cur, pct_rank(hist, cur), pos, c, "bull" if cur > 0 else "bear"))

    # up/down closing streak
    sign = np.sign(r).where(r.notna() & (r != 0))
    ln = run_lengths(sign)
    if pd.notna(ln.iloc[-1]) and ln.iloc[-1] >= 2:
        L, sg = int(ln.iloc[-1]), sign.iloc[-1]
        start = last - L + 1
        reach = (ln == L) & (sign == sg)
        pos = np.array([p for p in np.flatnonzero(reach.to_numpy()) if p < start])
        ends = ln[(sign != sign.shift(-1)) & sign.notna() & (sign == sg)].iloc[:-1]
        facts.append(_fact("streak", f"{L} {'up' if sg > 0 else 'down'} closes in a row",
                           L, pct_rank(ends, L), pos, c, "bull" if sg > 0 else "bear")
                     | {"longer_before": int((ends > L).sum())})

    # how long above / below each average
    for m in MAS:
        ma = sma(c, m)
        if ma.notna().sum() < 60:
            continue
        st = (c > ma).where(ma.notna())
        ln_m = run_lengths(st)
        if pd.isna(ln_m.iloc[-1]):
            continue
        L, above = int(ln_m.iloc[-1]), bool(st.iloc[-1])
        start = last - L + 1
        reach = (ln_m == L) & (st == above)
        pos = np.array([p for p in np.flatnonzero(reach.to_numpy()) if p < start])
        ends = ln_m[(st != st.shift(-1)) & st.notna() & (st == above)]
        ends = ends[ends.index < start]
        facts.append(_fact(f"ma{m}_run", f"{L} days {'above' if above else 'below'} the {m}-day average",
                           L, pct_rank(ends, L), pos, c, "bull" if above else "bear")
                     | {"longer_before": int((ends > L).sum()), "ma": m})

    # stretch from the 50 / 200-day average, in ATRs
    a = atr(df)
    for m in (50, 200):
        ma = sma(c, m)
        x = (c - ma) / a
        if pd.isna(x.iloc[-1]):
            continue
        cur = float(x.iloc[-1])
        cond = (x >= cur) if cur > 0 else (x <= cur)
        pos = episodes(cond.iloc[:-1], cooldown=10)
        facts.append(_fact(f"stretch_{m}", f"{cur:+.1f} ATRs from the {m}-day average",
                           cur, pct_rank(x.iloc[:-1], cur), pos, c))

    # volume vs normal
    # vs the 50-day MEDIAN: one 20x block-deal day inflates a mean for 50 days
    # and makes every ordinary day look abnormally quiet
    v = df["volume"]
    vr = v / v.rolling(50, min_periods=20).median().shift()
    if pd.notna(vr.iloc[-1]) and vr.iloc[-1] > 0:
        cur = float(vr.iloc[-1])
        cond = (vr >= cur) if cur >= 1 else (vr <= cur)
        pos = episodes(cond.iloc[:-1], cooldown=5)
        facts.append(_fact("volume", f"volume {cur:.1f}x its 50-day normal (median)", cur,
                           pct_rank(vr.iloc[:-1], cur), pos, c))

    # volatility squeeze / expansion
    vol20 = r.rolling(20, min_periods=15).std()
    vol1y = r.rolling(YEAR, min_periods=120).std()
    vreg = vol20 / vol1y
    if pd.notna(vreg.iloc[-1]):
        cur = float(vreg.iloc[-1])
        cond = (vreg <= cur) if cur < 1 else (vreg >= cur)
        pos = episodes(cond.iloc[:-1], cooldown=20)
        facts.append(_fact("vol_regime", f"20-day volatility {cur:.2f}x its 1-year level"
                           + (" (squeeze)" if cur < 1 else " (expansion)"),
                           cur, pct_rank(vreg.iloc[:-1], cur), pos, c))

    # where in the 52-week range
    hi = df["high"].rolling(YEAR, min_periods=120).max()
    lo = df["low"].rolling(YEAR, min_periods=120).min()
    pos52 = (c - lo) / (hi - lo)
    if pd.notna(pos52.iloc[-1]):
        cur = float(pos52.iloc[-1])
        dh = float((c.iloc[-1] / hi.iloc[-1] - 1) * 100)
        cond = (pos52 >= cur) if cur >= 0.5 else (pos52 <= cur)
        pos = episodes(cond.iloc[:-1], cooldown=10)
        facts.append(_fact("range52", f"{dh:+.0f}% from the 52-week high "
                           f"({cur * 100:.0f}% up the 52-week range)",
                           cur, pct_rank(pos52.iloc[:-1], cur), pos, c))
    return facts


# ───────────────────────────── 3) patterns ──────────────────────────────────
def pattern_series(df: pd.DataFrame) -> dict:
    """{name: (side, bool Series)} over the FULL history, causal. A flag is
    'fired' when it is True on any of the last few bars; its past True bars
    give this stock's own track record after that flag."""
    df = _norm(df)
    c, v = df["close"], df["volume"]
    r = returns(df)
    ma20, ma50, ma200 = sma(c, 20), sma(c, 50), sma(c, 200)
    v50 = v.rolling(50, min_periods=20).mean().shift()
    P = {}

    def cross_up(a, b):
        return (a > b) & (a.shift() <= b.shift())

    P["golden_cross_50_200"] = ("bull", cross_up(ma50, ma200))
    P["death_cross_50_200"] = ("bear", cross_up(ma200, ma50))
    P["cross_up_20_50"] = ("bull", cross_up(ma20, ma50))
    P["cross_down_20_50"] = ("bear", cross_up(ma50, ma20))
    for m, ma in ((50, ma50), (200, ma200)):
        st = (c > ma).where(ma.notna())
        ln = run_lengths(st)
        P[f"reclaim_{m}"] = ("bull", (st == True) & (st.shift() == False) & (ln.shift() >= 20))   # noqa: E712
        P[f"lose_{m}"] = ("bear", (st == False) & (st.shift() == True) & (ln.shift() >= 20))      # noqa: E712

    hh = df["high"].rolling(20).max()
    ll = df["low"].rolling(20).min()
    up_struct = (hh > hh.shift(20)) & (ll > ll.shift(20))
    dn_struct = (hh < hh.shift(20)) & (ll < ll.shift(20))
    P["higher_highs_lows"] = ("bull", up_struct & ~up_struct.shift(fill_value=False))
    P["lower_highs_lows"] = ("bear", dn_struct & ~dn_struct.shift(fill_value=False))

    prior_hi = df["high"].rolling(YEAR, min_periods=120).max().shift()
    prior_lo = df["low"].rolling(YEAR, min_periods=120).min().shift()
    P["high_52w_on_volume"] = ("bull", (c > prior_hi) & (v >= 1.5 * v50))
    P["low_52w"] = ("bear", c < prior_lo)

    bw = (c.rolling(20).std() * 4) / ma20
    tight = bw <= bw.rolling(YEAR, min_periods=120).quantile(0.10)
    was_tight = tight.astype(float).rolling(5, min_periods=1).max().shift().fillna(0).astype(bool)
    up_b = ma20 + 2 * c.rolling(20).std()
    lo_b = ma20 - 2 * c.rolling(20).std()
    P["squeeze_break_up"] = ("bull", was_tight & (c > up_b))
    P["squeeze_break_down"] = ("bear", was_tight & (c < lo_b))

    down_vol = v.where(r < 0)
    P["pocket_pivot"] = ("bull", (r > 0) & (v > down_vol.rolling(10, min_periods=1).max().shift()))
    dist_day = (r <= -0.002) & (v > v.shift())
    P["distribution_5_in_25"] = ("bear", (dist_day.rolling(25).sum() >= 5)
                                 & (dist_day.rolling(25).sum().shift() < 5))

    rs = rsi(c)
    new_hi = c >= c.rolling(20).max()
    new_lo = c <= c.rolling(20).min()
    prev_hi_rsi = rs.where(new_hi).shift(20).rolling(20, min_periods=1).max()
    prev_lo_rsi = rs.where(new_lo).shift(20).rolling(20, min_periods=1).min()
    prev_hi_px = c.where(new_hi).shift(20).rolling(20, min_periods=1).max()
    prev_lo_px = c.where(new_lo).shift(20).rolling(20, min_periods=1).min()
    P["rsi_bear_divergence"] = ("bear", new_hi & (c > prev_hi_px) & (rs < prev_hi_rsi))
    P["rsi_bull_divergence"] = ("bull", new_lo & (c < prev_lo_px) & (rs > prev_lo_rsi))

    bo = c > prior_hi
    bo_level = prior_hi.where(bo).ffill(limit=15)
    recent_bo = bo.astype(float).rolling(15, min_periods=1).max().astype(bool) & ~bo
    P["failed_breakout"] = ("bear", recent_bo & (c < bo_level) & (c.shift() >= bo_level.shift()))

    ema21 = c.ewm(span=21, adjust=False).mean()
    st21 = (c > ema21)
    ln21 = run_lengths(st21)
    P["close_below_21ema_after_run"] = ("bear", (~st21) & st21.shift(fill_value=False)
                                        & (ln21.shift() >= 20))
    return {k: (s, b.fillna(False).astype(bool)) for k, (s, b) in P.items()}


def stage(df: pd.DataFrame) -> str | None:
    """Trend stage from the 30-week (150-day) average and its 20-day slope."""
    df = _norm(df)
    ma = sma(df["close"], 150)
    if ma.notna().sum() < 21:
        return None
    above = bool(df["close"].iloc[-1] > ma.iloc[-1])
    rising = bool(ma.iloc[-1] > ma.iloc[-21])
    return {(True, True): "Stage 2 uptrend", (True, False): "Stage 3 topping",
            (False, False): "Stage 4 downtrend", (False, True): "Stage 1 basing"}[(above, rising)]


def patterns(df: pd.DataFrame, recent: int = 10) -> dict:
    df = _norm(df)
    fired = []
    for name, (side, s) in pattern_series(df).items():
        hits = np.flatnonzero(s.to_numpy())
        if len(hits) == 0 or hits[-1] < len(df) - recent:
            continue
        last = int(hits[-1])
        past = episodes(s.iloc[:last], cooldown=10)
        fired.append({"key": name, "side": side,
                      "date": df["date"].iloc[last].date(),
                      "bars_ago": len(df) - 1 - last, "n": int(len(past)),
                      "thin": len(past) < THIN_N,
                      "fwd": forward_stats(df["close"], past)})
    bull = sum(1 for f in fired if f["side"] == "bull")
    bear = sum(1 for f in fired if f["side"] == "bear")
    return {"fired": sorted(fired, key=lambda f: f["bars_ago"]),
            "score": bull - bear, "bull": bull, "bear": bear, "stage": stage(df)}


# ───────────────────────────── 4) vs market ─────────────────────────────────
def vs_market(df: pd.DataFrame, idx: pd.Series, label: str) -> dict | None:
    """Relative strength of the stock against one index (close Series by date)."""
    df = _norm(df)
    if idx is None or len(idx) == 0 or df.empty:
        return None
    s = df.set_index("date")["close"]
    j = pd.concat([s.rename("s"), idx.rename("i")], axis=1, join="inner").dropna()
    if len(j) < 30:
        return None
    out = {"label": label, "bars": len(j)}
    ex = {}
    for k, lbl in ((21, "1M"), (63, "3M"), (126, "6M"), (252, "12M")):
        if len(j) > k:
            ex[lbl] = float((j["s"].iloc[-1] / j["s"].iloc[-1 - k]
                             - j["i"].iloc[-1] / j["i"].iloc[-1 - k]) * 100)
    out["excess"] = ex
    rsl = j["s"] / j["i"]
    w = rsl.tail(YEAR)
    out["rs_52w_high"] = bool(rsl.iloc[-1] >= w.max()) if len(w) >= 120 else None
    out["rs_52w_low"] = bool(rsl.iloc[-1] <= w.min()) if len(w) >= 120 else None
    out["rs_pct_in_year"] = pct_rank(w.iloc[:-1], rsl.iloc[-1]) if len(w) >= 120 else None

    sr, ir = j["s"].pct_change(), j["i"].pct_change()
    sr = sr.where(sr.abs() <= JUMP_MAX)
    b = beta(df, idx)
    out["beta"] = b
    if b is not None:
        for k in (20, 60):
            if len(j) > k:
                out[f"alpha_{k}d"] = float((sr.tail(k).sum() - b * ir.tail(k).sum()) * 100)
    yr = pd.concat([sr, ir], axis=1).tail(YEAR).dropna()
    yr.columns = ["s", "i"]
    down = yr[yr["i"] <= -0.01]
    out["on_index_down_1pct"] = ({"n": len(down), "stock_avg": float(down["s"].mean() * 100),
                                  "index_avg": float(down["i"].mean() * 100)}
                                 if len(down) else None)
    upd, dnd = yr[yr["i"] > 0], yr[yr["i"] < 0]
    out["up_capture"] = float(upd["s"].mean() / upd["i"].mean() * 100) if len(upd) > 20 else None
    out["down_capture"] = float(dnd["s"].mean() / dnd["i"].mean() * 100) if len(dnd) > 20 else None

    # Leading / Weakening / Lagging / Improving: is the relative-strength line
    # above its own 50-day average (trend), and is it rising over 10 days (now)?
    rs_ma = rsl.rolling(50, min_periods=50).mean()
    if rs_ma.notna().any() and len(rsl) > 11:
        lead = bool(rsl.iloc[-1] >= rs_ma.iloc[-1])
        rising = bool(rsl.iloc[-1] >= rsl.iloc[-11])
        out["quadrant"] = ("Leading" if lead and rising else "Weakening" if lead
                           else "Improving" if rising else "Lagging")
    return out


def market_context(idx500: pd.Series | None) -> str | None:
    if idx500 is None or len(idx500) < 200:
        return None
    ma = idx500.rolling(200).mean()
    return ("Nifty 500 above its 200-day average" if idx500.iloc[-1] > ma.iloc[-1]
            else "Nifty 500 below its 200-day average")


# ───────────────────────────── verdict ──────────────────────────────────────
def verdict(unusual_facts: list, pat: dict, rel: list, top: int = 3) -> list[dict]:
    """The most extreme facts, most extreme first: percentile at/beyond 90/10,
    flags fired in the last 10 sessions, and a relative-strength line at a
    52-week high or low."""
    cands = []
    for f in unusual_facts:
        p = f.get("pct")
        if p is not None and (p >= 90 or p <= 10):
            cands.append((abs(p - 50), f))
    for f in (pat or {}).get("fired", []):
        cands.append((45 - f["bars_ago"] * 0.1, f))
    for m in rel or []:
        if m and m.get("rs_52w_high"):
            cands.append((44, {"key": "rs_high", "text": f"relative strength vs {m['label']} at a 52-week HIGH", "side": "bull"}))
        if m and m.get("rs_52w_low"):
            cands.append((44, {"key": "rs_low", "text": f"relative strength vs {m['label']} at a 52-week LOW", "side": "bear"}))
    cands.sort(key=lambda x: -x[0])
    return [f for _, f in cands[:top]]
