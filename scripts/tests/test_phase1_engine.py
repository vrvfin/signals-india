r"""
test_phase1_engine.py — offline checks for the Phase 1 signal engine.

Plain python, no pytest: everything in this repo is a standalone script, and
these must run on a laptop with no Drive credentials and no network. Every check
here is pure computation on synthetic data, or against guru's research code.

WHY THIS EXISTS
  The live engine and the backtest that measured each signal family's edge are
  two separate implementations. If they drift, the measured lift no longer
  describes what the screen actually does — which is exactly the bug found in
  guru/daily_screen.py, where a 4-year window was labelled "all-time high".
  So the headline check is: do our metrics equal guru's, to float precision?

Run:
    python scripts/tests/test_phase1_engine.py
Exit code is the number of failures, so CI can gate on it.
"""
from __future__ import annotations

import importlib.util
import os
import sys
from datetime import datetime, date

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = os.path.dirname(HERE)
REPO = os.path.dirname(SCRIPTS)
# guru/ is the research pipeline; a read-only reference for metric definitions.
GURU_TECH = os.path.join(REPO, "guru", "compute_technical_metrics.py")

_FAILS: list[str] = []


def _load(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def check(label: str, ok: bool, extra: str = "") -> None:
    if not ok:
        _FAILS.append(label)
    print(f"  {'ok  ' if ok else 'FAIL'} {label}{('  ' + extra) if extra else ''}")


def section(title: str) -> None:
    print(f"\n=== {title} ===")


def synth(nyears: float, seed: int = 7) -> pd.DataFrame:
    """Synthetic daily OHLCV. Business days, so ~261 bars per CALENDAR year."""
    n = int(nyears * 252)
    rng = np.random.default_rng(seed)
    d = pd.bdate_range("2014-01-01", periods=n)
    c = 100 * np.exp(np.cumsum(rng.normal(0.0004, 0.015, n)))
    return pd.DataFrame({
        "date": d, "open": c,
        "high": c * (1 + abs(rng.normal(0, 0.008, n))),
        "low": c * (1 - abs(rng.normal(0, 0.008, n))),
        "close": c, "volume": rng.integers(1e4, 1e6, n)})


# ─────────────────────────────────────────────────────────────────────────────

def test_session_date(psc) -> None:
    section("pipeline_skip_check: which session a run belongs to")
    f = psc._expected_bar_date
    cases = [
        (datetime(2026, 8, 28, 17, 0), date(2026, 8, 28), "Fri 17:00 -> Fri"),
        (datetime(2026, 8, 28, 10, 0), date(2026, 8, 27), "Fri 10:00 -> Thu (pre-cutoff)"),
        (datetime(2026, 8, 29, 2, 30), date(2026, 8, 28), "Sat 02:30 late-cron -> Fri"),
        (datetime(2026, 8, 30, 18, 0), date(2026, 8, 28), "Sun -> Fri"),
        (datetime(2026, 8, 31, 16, 0), date(2026, 8, 31), "Mon 16:00 -> Mon"),
        (datetime(2026, 9, 1, 2, 42), date(2026, 8, 31), "Tue 02:42 late-cron -> Mon"),
    ]
    for now, want, label in cases:
        check(label, f(now) == want, f"got {f(now)}")
    # The 2026-08-28 regression: a 02:11 IST run processed Thursday's session,
    # then the real 16:00 IST run was suppressed as a same-calendar-day duplicate.
    want = f(datetime(2026, 8, 28, 16, 0))
    check("Aug-28 regression: the 4pm run is NOT skipped",
          date(2026, 8, 27) < want, f"last_bar 2026-08-27 < expected {want}")
    # 2026-09-03: a 12:36 IST test run picked up a PARTIAL intraday bar, stamped
    # it 2026-09-03, and the real 16:00 IST trigger then stood down 134 minutes
    # later. Same session is not the same as session FINISHED.
    cutoff = psc.PHASE1_SESSION_CUTOFF_IST_HOUR
    check("a pre-close run's bar is treated as partial", 12 < cutoff,
          f"12:36 IST < {cutoff}:00 cutoff, so it must not block the 4pm run")
    check("a post-close run's bar is treated as complete", 17 >= cutoff,
          f"17:30 IST >= {cutoff}:00 cutoff, so it still blocks a duplicate")


def test_bar_date_naming() -> None:
    section("dated snapshots named for the session, not the runner clock")

    def name(dates):
        bar = pd.to_datetime(pd.Series(dates), errors="coerce").max()
        return None if pd.isna(bar) else pd.Timestamp(bar).strftime("%Y-%m-%d")

    check("max bar wins", name(["2026-08-27", "2026-08-28"]) == "2026-08-28")
    check("stale rows do not win", name(["2026-08-21", "2026-08-28"]) == "2026-08-28")
    check("all-NaT signals the fallback", name([None, None]) is None)


def test_guru_fidelity(cf, gtm) -> None:
    section("metric fidelity vs guru/compute_technical_metrics.py")
    cols = ["price_slope_20d_pct", "price_slope_50d_pct", "consecutive_up_weeks",
            "atr_expansion_ratio", "price_vs_ma_20", "price_vs_ma_50",
            "price_vs_ma_200"]
    for seed in (7, 11, 23, 42, 99):
        df = synth(12, seed)
        mine = cf.compute_features_one("T", df)
        regime = pd.DataFrame({"date": df["date"], "nifty500_close": 100.0})
        theirs = gtm.compute(df.copy(), regime).iloc[-1]
        bad = [c for c in cols if abs(float(mine[c]) - float(theirs[c])) > 1e-4]
        check(f"seed {seed}: all {len(cols)} metrics match guru", not bad,
              str(bad) if bad else "")


def test_history_gating(cf) -> None:
    section("long-horizon highs are gated on real history")
    f12 = cf.compute_features_one("LONG", synth(12))
    check("history_years reported", 11.4 < f12["history_years"] < 11.8,
          f"= {f12['history_years']:.2f}")
    check("monotone 10y <= 5y <= 3y <= 1y",
          f12["pct_from_high_10y"] <= f12["pct_from_high_5y"] + 1e-9
          <= f12["pct_from_high_3y"] + 1e-9
          <= f12["dist_from_52w_high_pct"] + 1e-9)
    # NaN, never a silent fallback to a shorter window: less history otherwise
    # makes "near its high" trivially easy and floods the screen with new names.
    for yrs, expect in ((2, []), (4, [3]), (6, [3, 5]), (11, [3, 5, 10])):
        f = cf.compute_features_one("X", synth(yrs, 3))
        got = [y for y in (3, 5, 10) if pd.notna(f[f"pct_from_high_{y}y"])]
        check(f"{yrs}y history -> windows {expect}", got == expect, f"got {got}")
    f4 = cf.compute_features_one("X", synth(4, 3))
    check(f"4y stock fails the >={cf.ATH_MIN_YEARS}y ATH gate",
          f4["history_years"] < cf.ATH_MIN_YEARS)


def test_rs_alignment(cf) -> None:
    section("rs_vs_nifty500 differenced against each stock's OWN bar date")
    n = 400
    idx_dates = pd.bdate_range("2024-01-01", periods=n)
    rng = np.random.default_rng(5)
    idx_close = pd.Series(100 * np.exp(np.cumsum(rng.normal(0.0006, 0.011, n))))
    n500 = pd.DataFrame({"date": idx_dates, "close": idx_close})
    lookbacks = ["1m", "2m", "3m", "6m", "12m"]

    def frame(symbols, dates):
        d = {"symbol": symbols, "date": dates}
        for lb in lookbacks:
            d[f"return_{lb}_pct"] = [10.0] * len(symbols)
        return pd.DataFrame(d)

    out = cf.add_relative_strength(
        frame(["FRESH", "STALE"], [idx_dates[-1], idx_dates[-6]]), n500)
    exp_fresh = (idx_close.iloc[-1] / idx_close.iloc[-1 - 21] - 1) * 100
    exp_stale = (idx_close.iloc[-6] / idx_close.iloc[-6 - 21] - 1) * 100
    got_f = out.loc[out.symbol == "FRESH", "rs_vs_nifty500_1m_pct"].iloc[0]
    got_s = out.loc[out.symbol == "STALE", "rs_vs_nifty500_1m_pct"].iloc[0]
    check("fresh name uses its own window", abs(got_f - (10.0 - exp_fresh)) < 1e-6)
    check("stale name uses its own window", abs(got_s - (10.0 - exp_stale)) < 1e-6)
    check("the two differ (old code gave both one scalar)",
          abs(got_f - got_s) > 1e-6, f"delta {abs(got_f - got_s):.4f}pp")

    o2 = cf.add_relative_strength(
        frame(["HOLIDAY"], [idx_dates[-3] + pd.Timedelta(days=1)]), n500)
    check("date with no index row falls back to the prior session",
          pd.notna(o2["rs_vs_nifty500_1m_pct"].iloc[0]))
    o3 = cf.add_relative_strength(
        frame(["ANCIENT"], [pd.Timestamp("2020-01-02")]), n500)
    check("date before the index history -> NaN, not a bogus number",
          pd.isna(o3["rs_vs_nifty500_1m_pct"].iloc[0]))


def test_strategy_scoring() -> None:
    section("strategy scoring: margin and risk replace counts and extension")
    mv = _load("mv", os.path.join(SCRIPTS, "strategy_minervini.py"))
    cs = _load("cs", os.path.join(SCRIPTS, "strategy_canslim.py"))
    ma = _load("ma", os.path.join(SCRIPTS, "strategy_ma_respect.py"))

    # --- minervini: 8/8 only, ranked by distance to failing the template ---
    def mstock(sym, rs=90, dh=-5, dl=80, rising=True):
        return {"symbol": sym, "date": "2026-09-01", "close": 120.0,
                "sma_50": 110.0, "sma_200": 100.0, "ema_100": 105.0,
                "dist_from_52w_high_pct": dh, "dist_from_52w_low_pct": dl,
                "rs_rank_6m": rs, "200sma_rising": rising, "atr_14": 3.0,
                "return_6m_pct": 40.0}
    out = mv.minervini_signals(pd.DataFrame([
        mstock("ROBUST", rs=95, dh=-2), mstock("TIGHT_RS", rs=71, dh=-2),
        mstock("SEVEN", rs=60), mstock("NOTRISE", rs=95, rising=False)]))
    got = set(out["symbol"])
    check("minervini: 7-of-8 no longer emitted as 'hold'", "SEVEN" not in got)
    check("minervini: failing a boolean rule excludes", "NOTRISE" not in got)
    sc = dict(zip(out["symbol"], out["score"]))
    check("minervini: slack beats fragility", sc["ROBUST"] > sc["TIGHT_RS"],
          f"{sc['ROBUST']} vs {sc['TIGHT_RS']} (old score: 89.5 vs 87.1)")
    check("minervini: binding rule named",
          dict(zip(out["symbol"], out["binding_rule"]))["TIGHT_RS"] == "rs")

    # --- canslim: M is a day gate, C and A are required ---
    def crow(**kw):
        d = dict(q_eps_yoy_pct=40, ann_eps_yoy_pct=40,
                 dist_from_52w_high_pct=-5, promoter_holding_pct=60,
                 rs_rank_6m=90, roe_pct=25, market_health_score=50)
        d.update(kw)
        return pd.Series(d)
    hi, _ = cs.evaluate(crow(market_health_score=90))
    lo, _ = cs.evaluate(crow(market_health_score=10))
    check("canslim: market health no longer shifts the stock score",
          hi == lo == len(cs.STOCK_RULES), f"{hi} vs {lo}")
    _, checks = cs.evaluate(crow(q_eps_yoy_pct=2, ann_eps_yoy_pct=1))
    check("canslim: failing both earnings rules is now rejected",
          not all(checks[k] for k in cs.REQUIRED_RULES))
    s_rob, _ = cs.min_slack_score(cs.canslim_slacks(crow(q_eps_yoy_pct=80)))
    s_tgt, b = cs.min_slack_score(cs.canslim_slacks(crow(q_eps_yoy_pct=26)))
    check("canslim: margin separates two 6-of-6 stocks", s_rob > s_tgt,
          f"{s_rob} vs {s_tgt} (old score: both 69.9)")

    # --- ma_respect: the longest streak is no longer the best trade ---
    f = pd.DataFrame({
        "symbol": ["TIGHT", "STRETCHED"], "date": "2026-09-01",
        "close": [100.0, 100.0], "ema_20": [98.0, 85.0],
        "days_above_ema_20": [45, 250], "atr_14": [1.0, 1.0],
        "adr_pct_20": [3.0, 3.0], "dist_from_52w_high_pct": [-4.0, -2.0]})
    o = ma.ma_respect_signals(f, 20, 30, "ma_respect_20ema_30d")
    sc = dict(zip(o["symbol"], o["score"]))
    check("ma_respect: near stop beats long streak", sc["TIGHT"] > sc["STRETCHED"],
          f"{sc['TIGHT']} vs {sc['STRETCHED']} (old: 45 vs 250, wrong way round)")


def test_momentum_index_relative() -> None:
    section("momentum gates on index-relative strength")
    mo = _load("mo", os.path.join(SCRIPTS, "strategy_momentum.py"))
    n = 200
    rng = np.random.default_rng(3)
    raw = rng.uniform(-30, 60, n)
    base = pd.DataFrame({
        "symbol": [f"S{i}" for i in range(n)], "date": "2026-09-01",
        "close": 100.0, "atr_14": 2.0, "above_200sma": True,
        "return_6m_pct": raw,
        "rs_rank_6m": (pd.Series(raw).rank(pct=True) * 100).round(2),
        "rs_vs_nifty500_6m_pct": raw - 25.0,
        "dist_from_52w_high_pct": rng.uniform(-30, 0, n),
        "adr_pct_20": rng.uniform(2, 6, n)})
    out = mo.momentum_signals(base, "6m")
    buys = out[out["zone_type"].isin(["buy", "add"])]
    holds = out[out["zone_type"] == "hold"]
    check("variant tagged for the family collapse", set(out["variant"]) == {"6m"})
    check("buy/add strictly outrank hold on excess return",
          buys["rs_excess_pct"].min() > holds["rs_excess_pct"].max())
    # A universe that rose hard but LAGGED the index must not qualify.
    f2 = base.copy()
    raw2 = np.concatenate([rng.uniform(40, 60, 100), rng.uniform(-30, 60, 100)])
    f2["return_6m_pct"] = raw2
    f2["rs_rank_6m"] = (pd.Series(raw2).rank(pct=True) * 100).round(2)
    f2["rs_vs_nifty500_6m_pct"] = np.concatenate(
        [np.full(100, -5.0), rng.uniform(-40, 35, 100)])
    new = set(mo.momentum_signals(f2, "6m").query("zone_type in ['buy','add']")["symbol"])
    old = set(mo.momentum_signals(f2.drop(columns=["rs_vs_nifty500_6m_pct"]), "6m")
              .query("zone_type in ['buy','add']")["symbol"])
    check("index-relative picks different names to raw return", new != old,
          f"{len(old - new)} raw-gate names dropped as market-drift")
    check("raw-percentile fallback still emits when the index is missing",
          len(old) > 0)


def test_pead_anchor() -> None:
    section("pead anchors on the declared date, not the biggest gap")
    pe = _load("pe", os.path.join(SCRIPTS, "strategy_pead.py"))
    dates = pd.bdate_range("2026-07-01", periods=60)
    o = pd.DataFrame({"date": dates, "open": 100.0, "high": 101.0, "low": 99.0,
                      "close": 100.0, "volume": 100000.0})
    # Results day gaps DOWN 8%; two sessions later it rebounds +6%.
    o.loc[40, ["open", "close"]] = [92.0, 92.0]
    o.loc[40, "volume"] = 500000.0
    o.loc[41, "close"] = 92.0
    o.loc[42, "open"] = 97.5
    o.loc[42, ["close", "volume"]] = [99.0, 500000.0]
    res = pe.detect_pead_at("X", o, dates[40], dates[45])
    check("a candidate was found", res is not None)
    if res:
        check("anchored on the results session",
              str(res["earnings_date"]) == str(dates[40].date()),
              f"got {res['earnings_date']}")
        check("records the results-day gap, which is negative",
              res["earnings_gap_pct"] < 0,
              f"{res['earnings_gap_pct']:.1f}% -> main() drops it as not long-side; "
              f"old code anchored the +6% rebound and would have bought")


def test_aggregation() -> None:
    section("aggregation: independent ideas, not repeated ones")
    ag = _load("ag", os.path.join(SCRIPTS, "aggregate_signals.py"))

    def sig(sym, strat, score=50.0, zone="buy", variant=None):
        return {"symbol": sym, "strategy": strat, "zone_type": zone,
                "score": score, "entry": 100.0, "stop": 90.0, "reason": "r",
                "variant": variant}

    rows = [sig("MOMO", f"momentum_{lb}", 95, variant=lb)
            for lb in ("1m", "2m", "3m", "6m", "12m")]
    rows += [sig("DIVERSE", s, 95, "add")
             for s in ("darvas", "pead", "volume_breakout")]
    u = ag.compute_unified(pd.DataFrame(rows))
    m = u[u["symbol"] == "MOMO"].iloc[0]
    d = u[u["symbol"] == "DIVERSE"].iloc[0]
    check("momentum's 5 lookbacks are 1 family", m["n_families"] == 1,
          f"n_strategies={m['n_strategies']} -> n_families=1")
    check("which lookbacks fired is still recorded",
          m["momentum_profile"] == "persistent")
    # Derived from the STRATEGY NAME, not a `variant` column. The column only
    # exists once the new momentum strategy is deployed; relying on it left
    # momentum_profile blank on every row against the currently-deployed output.
    no_col = pd.DataFrame([{"symbol": "A", "strategy": f"momentum_{lb}",
                            "zone_type": "buy", "score": 90.0, "entry": 100.0,
                            "stop": 90.0, "reason": "r"}
                           for lb in ("1m", "2m", "3m", "6m", "12m")])
    u2 = ag.compute_unified(no_col).iloc[0]
    check("profile works with NO variant column present",
          u2["momentum_profile"] == "persistent" and u2["momentum_variants"] == 5,
          f"{u2['momentum_profile']!r}, {u2['momentum_variants']} variants")
    check("a state-only name marks no event", m["n_event_families"] == 0)
    check("3 different ideas count as 3", d["n_families"] == 3)
    check("the diverse name now sorts first", list(u["symbol"])[0] == "DIVERSE",
          "old rule sorted MOMO first on n_strategies=5")

    for variants, want in [({"1m"}, "fresh"), ({"12m"}, "stale"),
                           ({"3m", "6m", "12m"}, "broad"),
                           ({"1m", "2m", "3m", "6m", "12m"}, "persistent")]:
        check(f"momentum profile {sorted(variants)} -> {want}",
              ag.momentum_profile(variants) == want)

    # Breadth must not lower the score. Filler signals give each strategy a real
    # percentile spread, without which every score_norm is 100 and the old
    # mean's defect is invisible.
    rows = [sig(f"F{s}{i}", s, float(i))
            for s in ("darvas", "pead", "minervini", "pullback")
            for i in range(20)]
    rows += [sig("SOLO", "darvas", 100.0)]
    rows += [sig("BROAD", s, 99.0)
             for s in ("darvas", "pead", "minervini", "pullback")]
    u = ag.compute_unified(pd.DataFrame(rows))
    solo = u[u["symbol"] == "SOLO"].iloc[0]
    broad = u[u["symbol"] == "BROAD"].iloc[0]
    check("the OLD mean penalised breadth",
          broad["composite_score"] < solo["composite_score"],
          f"{broad['composite_score']:.1f} < {solo['composite_score']:.1f}")
    check("conviction_v2 rewards it",
          broad["conviction_v2"] > solo["conviction_v2"],
          f"{broad['conviction_v2']:.1f} > {solo['conviction_v2']:.1f}")

    sg = pd.DataFrame([sig("LIQ", "darvas"), sig("ILLIQ", "darvas"),
                       sig("NOCAP", "darvas")])
    feat = pd.DataFrame({"symbol": ["LIQ", "ILLIQ"],
                         "avg_turnover_20d_cr": [5.0, 0.2]})
    out = ag.apply_liquidity_gate(sg, feat, 1.0)
    check("illiquid names dropped", "ILLIQ" not in set(out["symbol"]))
    check("unknown turnover dropped, not silently passed",
          "NOCAP" not in set(out["symbol"]))
    # strategy_ipo_base emits avg_turnover_20d_cr itself. Merging the gate's copy
    # under the same name collided into _x/_y and raised KeyError on the very
    # next line — a crash only visible once the new IPO strategy met the new
    # aggregator, which is to say only in a real integration run.
    own = pd.DataFrame([{**{"strategy": "ipo_base", "zone_type": "buy",
                            "score": 50.0, "entry": 100.0, "stop": 90.0,
                            "reason": "r"},
                         "symbol": s_, "avg_turnover_20d_cr": v}
                        for s_, v in (("LIQ", 5.0), ("ILLIQ", 0.2))])
    g = ag.apply_liquidity_gate(own, feat, 1.0)
    check("a strategy carrying the turnover column does not collide",
          set(g["symbol"]) == {"LIQ"} and "avg_turnover_20d_cr" in g.columns)
    check("no merge suffixes leak into the frame",
          not any(c.endswith(("_x", "_y")) for c in g.columns))

    mr = [sig("X", f"ma_respect_{v}")
          for v in ("20ema_30d", "20ema_60d", "50ema_60d")]
    check("ma_respect collapse OFF by default (not signed off)",
          ag.compute_unified(pd.DataFrame(mr)).iloc[0]["n_families"] == 3)
    check("--collapse-ma-respect makes it one vote",
          ag.compute_unified(pd.DataFrame(mr),
                             collapse_ma_respect=True).iloc[0]["n_families"] == 1)


def test_market_stance() -> None:
    section("market stance: direction and agreement, not one blended number")
    ms = _load("ms", os.path.join(SCRIPTS, "market_state.py"))

    def info(gap=5.0, breadth=70, hl=1.0, vix=12, fii=2000, adv=60):
        return (dict(nifty50_close=100 * (1 + gap / 100), nifty50_sma200=100.0),
                dict(pct_above_50sma=breadth),
                dict(highs_minus_lows_pct_univ=hl),
                dict(india_vix=vix), dict(fii_5d_net_cr=fii),
                dict(pct_advancing=adv))

    st, nb, nbe, _ = ms.stance_from(ms.component_directions(*info()))
    check("six bullish -> AGGRESSIVE", st == "AGGRESSIVE", f"{nb} bull / {nbe} bear")
    st, _, _, _ = ms.stance_from(ms.component_directions(
        *info(gap=-5, breadth=30, hl=-1.0, vix=25, fii=-2000, adv=40)))
    check("six bearish -> DEFENSIVE", st == "DEFENSIVE")
    # The case a single blended number cannot express: half strongly bullish,
    # half strongly bearish averages to a mid reading that looks like calm.
    st, nb, nbe, ag = ms.stance_from(ms.component_directions(
        *info(gap=5, breadth=70, hl=1.0, vix=25, fii=-2000, adv=40)))
    check("three-all -> NEUTRAL, stated as genuinely mixed", st == "NEUTRAL",
          f"{nb} bull / {nbe} bear, agreement {ag}")
    st, _, _, _ = ms.stance_from(ms.component_directions(
        *info(gap=5, breadth=70, hl=1.0, vix=17, fii=0, adv=50)))
    check("a moderate lean -> CONSTRUCTIVE", st == "CONSTRUCTIVE")
    d = ms.component_directions(None, dict(pct_above_50sma=70), {}, None, None,
                                dict(ad_score_basis="x", ad_proxy="y"))
    check("unreadable components abstain rather than vote neutral", len(d) == 1)

    h = pd.DataFrame({"date": ["2026-08-25", "2026-08-26", "2026-08-27", "2026-08-28"],
                      "stance": ["CAUTIOUS", "AGGRESSIVE", "AGGRESSIVE", "AGGRESSIVE"]})
    check("stance run length counted, today included",
          ms.stance_history(h, "AGGRESSIVE", "2026-09-01") == 4)
    check("a flip resets the run", ms.stance_history(h, "DEFENSIVE", "2026-09-01") == 1)
    check("every stance carries a playbook",
          all(k in ms.STANCE_PLAYBOOK for k in
              ("AGGRESSIVE", "CONSTRUCTIVE", "NEUTRAL", "CAUTIOUS", "DEFENSIVE")))


def test_signal_tracker() -> None:
    section("outcome tracker: R-multiples, exit-vs-hold, reliability")
    st = _load("st", os.path.join(SCRIPTS, "signal_tracker.py"))

    def bars(closes, highs=None, lows=None):
        n = len(closes)
        return pd.DataFrame({"date": pd.bdate_range("2026-06-01", periods=n),
                             "high": highs or [c * 1.01 for c in closes],
                             "low": lows or [c * 0.99 for c in closes],
                             "close": closes})

    # entry 100, stop 90 -> 1R = 10, a 2R target sits at 120
    r = st.evaluate_signal(bars([100, 105, 112, 121, 118]), 100, 90)
    check("target hit -> +2R", r["status"] == "target_hit" and r["r_multiple"] == 2.0)
    r = st.evaluate_signal(bars([100, 96, 91, 88, 95]), 100, 90)
    check("stop hit -> -1R", r["status"] == "stopped" and r["r_multiple"] == -1.0)

    same_bar = pd.DataFrame({"date": pd.bdate_range("2026-06-01", periods=2),
                             "high": [101, 125], "low": [99, 85],
                             "close": [100, 120]})
    check("stop and target on one bar -> the pessimistic read",
          st.evaluate_signal(same_bar, 100, 90)["status"] == "stopped",
          "daily bars cannot say which came first; assuming the good one "
          "would flatter every result")

    r = st.evaluate_signal(bars([100, 102, 104]), 100, 90)
    check("still running -> open, not counted as closed",
          r["status"] == "open" and not r["closed"])
    r = st.evaluate_signal(bars([100] * 70), 100, 90, max_hold_days=63)
    check("goes stale -> expired at the hold limit",
          r["status"] == "expired" and r["days_held"] == 63)

    # Stopped out, then the stock ran: the EXIT destroyed the trade, not the pick.
    r = st.evaluate_signal(bars([100, 95, 89] + [100 + 3 * i for i in range(70)]),
                           100, 90)
    check("exit_vs_hold catches an exit that cut a big winner",
          r["exit_vs_hold_r"] < -5,
          f"realised {r['r_multiple']}R vs {r['hold_r']}R holding")

    r = st.evaluate_signal(bars([100, 108, 104, 118, 112]), 100, 90, target_r=99)
    check("MFE/MAE captured (this is what sets the target)",
          r["mfe_r"] > 1.7 and r["mae_r"] < 0,
          f"MFE {r['mfe_r']}R, MAE {r['mae_r']}R")
    check("stop >= entry flagged invalid, not silently dropped",
          st.evaluate_signal(bars([100, 101]), 100, 105)["status"] == "invalid")

    small = pd.DataFrame({"conviction_at_signal": range(10),
                          "r_multiple": range(10)})
    check("reliability declines to report below n=20",
          pd.isna(st.reliability(small, "conviction_at_signal")["decile_lift"]))
    perfect = pd.DataFrame({"conviction_at_signal": np.arange(200),
                            "r_multiple": np.arange(200) * 0.01})
    o = st.reliability(perfect, "conviction_at_signal")
    check("a perfectly predictive score -> IC 1.0", abs(o["rank_ic"] - 1.0) < 1e-6)
    rng = np.random.default_rng(1)
    noise = pd.DataFrame({"conviction_at_signal": rng.normal(size=400),
                          "r_multiple": rng.normal(size=400)})
    o = st.reliability(noise, "conviction_at_signal")
    check("a meaningless score -> ~zero lift and ~zero IC",
          abs(o["decile_lift"]) < 1.0 and abs(o["rank_ic"]) < 0.15,
          f"lift {o['decile_lift']}R, IC {o['rank_ic']}")
    backwards = pd.DataFrame({"conviction_at_signal": np.arange(200),
                              "r_multiple": -np.arange(200) * 0.01})
    check("a BACKWARDS score is caught",
          st.reliability(backwards, "conviction_at_signal")["decile_lift"] < 0)
    check("no scipy needed (it is not in requirements.txt)",
          "scipy" not in sys.modules)


def test_ranking_terms() -> None:
    section("four ranking terms: computed, not yet blended")
    ag = _load("ag2", os.path.join(SCRIPTS, "aggregate_signals.py"))
    sig = pd.DataFrame({
        "symbol": ["TIGHT", "WIDE", "MID"], "strategy": "darvas",
        "zone_type": "buy", "score": [50.0, 50.0, 50.0],
        "date": pd.Timestamp("2026-09-01"),
        "entry": [100.0, 100.0, 100.0], "stop": [96.0, 86.0, 92.0],
        "reason": "r"})
    feat = pd.DataFrame({"symbol": ["TIGHT", "WIDE", "MID"],
                         "rs_vs_nifty500_6m_pct": [30.0, -5.0, 10.0],
                         "atr_expansion_ratio": [2.0, 1.0, 1.5]})
    opens = pd.DataFrame({"symbol": ["TIGHT", "WIDE", "MID"],
                          "first_date": ["2026-09-01", "2026-06-01", "2026-08-15"]})
    out = ag.add_ranking_terms(sig, feat, opens).set_index("symbol")
    check("risk_pct from entry/stop", abs(out.loc["TIGHT", "risk_pct"] - 4.0) < 1e-6)
    check("a tighter stop ranks higher",
          out.loc["TIGHT", "term_risk"] > out.loc["WIDE", "term_risk"],
          "4% stop vs 14% — the first can be sized 3x larger for the same risk")
    check("stronger vs the index ranks higher",
          out.loc["TIGHT", "term_rs"] > out.loc["WIDE", "term_rs"])
    check("fresher ranks higher (inverts the extension bias)",
          out.loc["TIGHT", "term_stage"] > out.loc["WIDE", "term_stage"])
    check("more range expansion ranks higher",
          out.loc["TIGHT", "term_confirm"] > out.loc["WIDE", "term_confirm"])
    check("all terms are 0-100",
          all(out[c].between(0, 100).all()
              for c in ("term_risk", "term_rs", "term_stage", "term_confirm")))
    bare = ag.add_ranking_terms(sig, None, None)
    check("no features -> terms NaN but risk still computed, no crash",
          bare["term_rs"].isna().all() and bare["term_risk"].notna().all())
    check("comparison sheet produced",
          len(ag.terms_report(out.reset_index(), n=2)) >= 4)


def test_tracker_replay() -> None:
    section("tracker replay: rebuild history from the dated snapshots")
    st = _load("st2", os.path.join(SCRIPTS, "signal_tracker.py"))
    ag = _load("ag3", os.path.join(SCRIPTS, "aggregate_signals.py"))

    def snap(rows):
        return pd.DataFrame(rows, columns=["symbol", "zone_type", "strategies",
                                           "entry_median", "stop_median",
                                           "composite_score"])

    snaps = [
        ("2026-06-01", snap([
            ["AAA", "buy", "momentum_1m, momentum_3m, momentum_6m", 100.0, 90.0, 80.0],
            ["BBB", "hold", "darvas", 50.0, 45.0, 70.0]])),
        ("2026-06-02", snap([
            ["AAA", "buy", "momentum_1m, darvas", 110.0, 99.0, 85.0],
            ["CCC", "add", "pead", 200.0, 190.0, 90.0]])),
    ]
    out = st.replay_open_signals(snaps, ag.family_of)
    momo = out[(out["symbol"] == "AAA") & (out["family"] == "momentum")]
    check("momentum's lookbacks collapse to one family in history too",
          len(momo) == 1, "otherwise replayed history would disagree with today")
    check("first sighting is the call — entry frozen at 100, not 110",
          float(momo["entry_at_signal"].iloc[0]) == 100.0)
    check("a second family on the same name dates from ITS first appearance",
          out[(out["symbol"] == "AAA") &
              (out["family"] == "darvas")]["first_date"].iloc[0] == "2026-06-02")
    check("'hold' zones are not replayed as calls", "BBB" not in set(out["symbol"]))
    check("new names on later days are captured", "CCC" in set(out["symbol"]))
    check("rows with no entry/stop are skipped, not defaulted",
          len(st.replay_open_signals(
              [("2026-06-01", snap([["X", "buy", "darvas", None, None, 1.0]]))],
              ag.family_of)) == 0)


def test_ipo_base() -> None:
    section("IPO first-base detection")
    m = _load("ipo", os.path.join(SCRIPTS, "strategy_ipo_base.py"))
    dv = _load("dv", os.path.join(SCRIPTS, "strategy_darvas.py"))

    def series(highs, lows, closes, vols=None):
        n = len(highs)
        return pd.DataFrame({"date": pd.bdate_range("2026-01-01", periods=n),
                             "open": closes, "high": highs, "low": lows,
                             "close": closes,
                             "volume": vols or [100000.0] * n})

    n = 60
    hi, lo, cl = [120.0] * n, [84.0] * n, [100.0] * n
    b = m.find_ipo_base(series(hi, lo, cl))
    check("finds a 30%-deep base", b is not None and
          abs(b["base_depth_pct"] - 30.0) < 0.1)
    # The invariant that matters: base_high is resistance built BEFORE today, so
    # a breakout today is representable. (base_days is now measured from the
    # PEAK, not the window length, so it is no longer n-1.)
    today_tops = series(hi + [200.0], lo + [150.0], cl + [190.0])
    b2 = m.find_ipo_base(today_tops)
    check("base_high excludes today's bar, so a breakout is possible",
          b2 is not None and b2["base_high"] < 200.0,
          f"base_high {b2['base_high']:.0f} < today's high 200")
    check("base_days measures the structure, not the scan window",
          b["base_days"] < m.BASE_MAX_DAYS, f"{b['base_days']} < {m.BASE_MAX_DAYS}")
    # A stock still making new lows is falling, not basing.
    fall = list(np.linspace(150, 90, 60))
    check("still making new lows -> not a base",
          m.find_ipo_base(series([x * 1.01 for x in fall],
                                 [x * 0.99 for x in fall], fall)) is None)
    check("a 2% drift is not a base",
          m.find_ipo_base(series([101.0] * n, [99.0] * n, [100.0] * n)) is None)
    check("a 75% collapse is not a base",
          m.find_ipo_base(series([200.0] * n, [50.0] * n, [100.0] * n)) is None)

    feat = pd.Series({"date": "2026-09-02", "symbol": "NEWCO",
                      "avg_turnover_20d_cr": 5.0})
    s_add = m.ipo_signal("NEWCO", series(hi + [126.0], lo + [118.0], cl + [125.0],
                                         [100000.0] * n + [220000.0]), feat, 180)
    check("break above the base on 2x volume -> add",
          s_add and s_add["zone_type"] == "add")
    check("stop is the base low, not an ATR guess", s_add["stop"] == 84.0)
    s_buy = m.ipo_signal("NEWCO", series(hi + [112.0], lo + [105.0], cl + [110.0]),
                         feat, 180)
    check("coiling just under the pivot -> buy",
          s_buy and s_buy["zone_type"] == "buy")
    # "Anywhere inside the base" fired on 68% of the live cohort — a description
    # of the cohort, not a selection from it.
    mid = m.ipo_signal("NEWCO", series(hi + [95.0], lo + [86.0], cl + [90.0]),
                       feat, 180)
    check("meandering 25% under the pivot -> no signal", mid is None,
          f"buy requires within {m.BUY_MAX_BELOW_HIGH_PCT:g}% of the pivot")
    check("below the base low -> no signal",
          m.ipo_signal("NEWCO", series(hi + [85.0], lo + [70.0], cl + [80.0]),
                       feat, 180) is None)

    # The reason this cannot just be darvas with different numbers.
    check("a 30%-deep IPO base is rejected by darvas' 10% tolerance",
          30.0 > dv.BOX_MAX_RANGE_PCT,
          f"darvas allows {dv.BOX_MAX_RANGE_PCT}%, IPO bases run "
          f"{m.BASE_MIN_DEPTH_PCT}-{m.BASE_MAX_DEPTH_PCT}%")


def main() -> int:
    cf = _load("cf", os.path.join(SCRIPTS, "compute_features.py"))
    psc = _load("psc", os.path.join(SCRIPTS, "pipeline_skip_check.py"))
    print("Phase 1 engine — offline checks")
    test_session_date(psc)
    test_bar_date_naming()
    if os.path.exists(GURU_TECH):
        gtm = _load("gtm", GURU_TECH)
        test_guru_fidelity(cf, gtm)
    else:
        print(f"\n  SKIP guru fidelity — {GURU_TECH} not present")
    test_history_gating(cf)
    test_rs_alignment(cf)
    test_strategy_scoring()
    test_momentum_index_relative()
    test_pead_anchor()
    test_aggregation()
    test_market_stance()
    test_signal_tracker()
    test_ranking_terms()
    test_tracker_replay()
    test_ipo_base()
    print()
    if _FAILS:
        print(f"{len(_FAILS)} FAILED:")
        for f in _FAILS:
            print(f"  - {f}")
    else:
        print("all checks passed")
    return len(_FAILS)


if __name__ == "__main__":
    sys.exit(main())
