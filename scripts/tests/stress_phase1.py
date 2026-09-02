r"""
stress_phase1.py — adversarial inputs for the Phase 1 engine.

test_phase1_engine.py checks that the code does the RIGHT thing on well-formed
data. This checks it does not BLOW UP on malformed data, which is what a live
pipeline actually serves it: empty frames after an upstream failure, all-NaN
columns when a feed dies, single rows on a stock's first day, missing columns
after a schema change, zero and negative prices from a bad tick.

Six defects in this work were found by running against real data rather than
synthetic frames. This file is the attempt to close that gap: every case here is
one the pipeline can genuinely produce.

Nothing here touches Drive or the network.

Run:
    python scripts/tests/stress_phase1.py
Exit code is the number of failures.
"""
from __future__ import annotations

import importlib.util
import os
import sys
import warnings

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = os.path.dirname(HERE)
if SCRIPTS not in sys.path:
    sys.path.insert(0, SCRIPTS)

_FAILS: list[str] = []


def _load(name, fn):
    spec = importlib.util.spec_from_file_location(name, os.path.join(SCRIPTS, fn))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def section(t):
    print(f"\n=== {t} ===")


def survives(label: str, fn, *a, **kw):
    """The assertion is simply: it must not raise. Returning None, NaN or an
    empty frame is a fine answer to nonsense input; a traceback is not."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", FutureWarning)
            fn(*a, **kw)
        print(f"  ok   {label}")
    except FutureWarning as e:
        _FAILS.append(f"{label} (FutureWarning)")
        print(f"  FAIL {label}  -> deprecation: {str(e)[:90]}")
    except Exception as e:
        _FAILS.append(label)
        print(f"  FAIL {label}  -> {type(e).__name__}: {str(e)[:110]}")


def ohlcv(n=60, close=100.0, nan=False, zero=False, neg=False):
    d = pd.DataFrame({
        "date": pd.bdate_range("2026-01-01", periods=n),
        "open": [close] * n, "high": [close * 1.01] * n,
        "low": [close * 0.99] * n, "close": [close] * n,
        "volume": [100000.0] * n})
    if nan:
        d[["open", "high", "low", "close", "volume"]] = np.nan
    if zero:
        d[["open", "high", "low", "close"]] = 0.0
    if neg:
        d[["open", "high", "low", "close"]] = -abs(close)
    return d


def main() -> int:
    cf = _load("cf", "compute_features.py")
    ag = _load("ag", "aggregate_signals.py")
    st = _load("st", "signal_tracker.py")
    sc = _load("sc", "strategy_common.py")
    ipo = _load("ipo", "strategy_ipo_base.py")
    ms = _load("ms", "market_state.py")
    bph = _load("bph", "build_price_highs.py")

    print("Phase 1 — stress / malformed input")

    section("compute_features_one: degenerate price frames")
    survives("empty frame", cf.compute_features_one, "X", pd.DataFrame())
    survives("None", cf.compute_features_one, "X", None)
    survives("single bar", cf.compute_features_one, "X", ohlcv(1))
    survives("59 bars (just under the 60 floor)", cf.compute_features_one, "X", ohlcv(59))
    survives("all-NaN prices", cf.compute_features_one, "X", ohlcv(nan=True))
    survives("zero prices", cf.compute_features_one, "X", ohlcv(zero=True))
    survives("negative prices", cf.compute_features_one, "X", ohlcv(neg=True))
    survives("zero volume throughout", cf.compute_features_one, "X",
             ohlcv().assign(volume=0.0))
    survives("duplicate dates", cf.compute_features_one, "X",
             pd.concat([ohlcv(40), ohlcv(40)], ignore_index=True))
    survives("unsorted dates", cf.compute_features_one, "X",
             ohlcv().sample(frac=1, random_state=1))
    survives("one NaN mid-series", cf.compute_features_one, "X",
             ohlcv().assign(close=lambda d: d["close"].mask(d.index == 30)))

    section("add_relative_strength: index frames")
    lb = ["1m", "2m", "3m", "6m", "12m"]
    feat = pd.DataFrame({"symbol": ["A"], "date": [pd.Timestamp("2026-09-01")],
                         **{f"return_{x}_pct": [10.0] for x in lb}})
    survives("no index at all", cf.add_relative_strength, feat.copy(), None)
    survives("index too short", cf.add_relative_strength, feat.copy(),
             pd.DataFrame({"date": pd.bdate_range("2026-01-01", periods=10),
                           "close": 100.0}))
    survives("index with NaT dates", cf.add_relative_strength, feat.copy(),
             pd.DataFrame({"date": [pd.NaT] * 300, "close": [100.0] * 300}))
    survives("stock row with a NaT date", cf.add_relative_strength,
             feat.assign(date=[pd.NaT]),
             pd.DataFrame({"date": pd.bdate_range("2024-01-01", periods=400),
                           "close": np.linspace(100, 200, 400)}))
    survives("empty features frame", cf.add_relative_strength,
             feat.iloc[0:0].copy(),
             pd.DataFrame({"date": pd.bdate_range("2024-01-01", periods=400),
                           "close": np.linspace(100, 200, 400)}))

    section("aggregator: malformed signal frames")
    base = pd.DataFrame({"symbol": ["A"], "strategy": ["darvas"],
                         "zone_type": ["buy"], "score": [50.0],
                         "entry": [100.0], "stop": [90.0], "reason": ["r"]})
    survives("empty signals", ag.compute_unified,
             base.iloc[0:0].copy())
    survives("all-NaN scores", ag.compute_unified, base.assign(score=np.nan))
    survives("NaN zone_type", ag.compute_unified, base.assign(zone_type=None))
    survives("no entry/stop columns", ag.compute_unified,
             base.drop(columns=["entry", "stop"]))
    survives("stop above entry", ag.compute_unified, base.assign(stop=200.0))
    survives("duplicate identical rows", ag.compute_unified,
             pd.concat([base, base], ignore_index=True))
    survives("unknown strategy name", ag.compute_unified,
             base.assign(strategy="something_new_2027"))
    survives("liquidity gate, no features", ag.apply_liquidity_gate,
             base.copy(), None, 1.0)
    survives("liquidity gate, empty features", ag.apply_liquidity_gate,
             base.copy(), pd.DataFrame(), 1.0)
    survives("liquidity gate, all-NaN turnover", ag.apply_liquidity_gate,
             base.copy(),
             pd.DataFrame({"symbol": ["A"], "avg_turnover_20d_cr": [np.nan]}), 1.0)
    survives("ranking terms, nothing to join", ag.add_ranking_terms,
             base.copy(), None, None)
    survives("ranking terms, zero entry price", ag.add_ranking_terms,
             base.assign(entry=0.0), None, None)
    survives("terms report on an empty frame", ag.terms_report,
             base.iloc[0:0].copy())

    section("tracker: malformed calls")
    bars = pd.DataFrame({"date": pd.bdate_range("2026-06-01", periods=30),
                         "high": 110.0, "low": 90.0, "close": 100.0})
    survives("no bars", st.evaluate_signal, pd.DataFrame(), 100, 90)
    survives("one bar", st.evaluate_signal, bars.head(1), 100, 90)
    survives("entry is NaN", st.evaluate_signal, bars, np.nan, 90)
    survives("stop is None", st.evaluate_signal, bars, 100, None)
    survives("entry equals stop (zero risk)", st.evaluate_signal, bars, 100, 100)
    survives("entry is a string", st.evaluate_signal, bars, "abc", 90)
    survives("all-NaN bars", st.evaluate_signal,
             bars.assign(high=np.nan, low=np.nan, close=np.nan), 100, 90)
    survives("summarise an empty frame", st.summarise_by, pd.DataFrame(), "family")
    survives("reliability on an empty frame", st.reliability,
             pd.DataFrame({"conviction_at_signal": [], "r_multiple": []}),
             "conviction_at_signal")
    survives("reliability on a constant score", st.reliability,
             pd.DataFrame({"conviction_at_signal": [5.0] * 100,
                           "r_multiple": np.random.default_rng(0).normal(size=100)}),
             "conviction_at_signal")
    survives("replay with no snapshots", st.replay_open_signals, [], ag.family_of)
    survives("replay with an empty snapshot", st.replay_open_signals,
             [("2026-06-01", pd.DataFrame())], ag.family_of)
    survives("replay, snapshot missing columns", st.replay_open_signals,
             [("2026-06-01", pd.DataFrame({"symbol": ["A"]}))], ag.family_of)

    section("scoring helpers")
    survives("slack with None", sc.slack, None, 10, 5)
    survives("slack with a zero span", sc.slack, 20, 10, 0)
    survives("min_slack on an empty dict", sc.min_slack_score, {})
    survives("min_slack all-NaN", sc.min_slack_score, {"a": np.nan})
    survives("min_slack all-negative", sc.min_slack_score, {"a": -1.0})
    survives("base_quality zero max range", sc.base_quality_score,
             5, 0, 30, 5, 90, 1.5, breakout=True)
    survives("base_quality min==max days", sc.base_quality_score,
             5, 15, 30, 30, 30, 1.5, breakout=False)
    survives("pct_rank on an empty series", sc.pct_rank, pd.Series(dtype=float))
    survives("pct_rank all-NaN", sc.pct_rank, pd.Series([np.nan] * 5))

    section("ipo base detector")
    survives("empty", ipo.find_ipo_base, pd.DataFrame())
    survives("one bar", ipo.find_ipo_base, ohlcv(1))
    survives("flat line (no depth)", ipo.find_ipo_base, ohlcv())
    survives("all-NaN", ipo.find_ipo_base, ohlcv(nan=True))
    survives("zero prices", ipo.find_ipo_base, ohlcv(zero=True))
    feat_row = pd.Series({"date": "2026-09-02", "symbol": "X",
                          "avg_turnover_20d_cr": 5.0})
    survives("signal on a flat series", ipo.ipo_signal, "X", ohlcv(), feat_row, 180)
    survives("signal with NaN turnover", ipo.ipo_signal, "X", ohlcv(),
             pd.Series({"date": "d", "symbol": "X",
                        "avg_turnover_20d_cr": np.nan}), 180)

    section("market stance")
    survives("no components readable", ms.component_directions,
             None, None, None, None, None, None)
    survives("empty dicts", ms.component_directions, {}, {}, {}, {}, {}, {})
    survives("sma200 of zero", ms.component_directions,
             {"nifty50_close": 100.0, "nifty50_sma200": 0.0}, {}, {}, {}, {}, {})
    survives("None values inside", ms.component_directions,
             {"nifty50_close": None, "nifty50_sma200": None},
             {"pct_above_50sma": None}, {"highs_minus_lows_pct_univ": None},
             {"india_vix": None}, {"fii_5d_net_cr": None},
             {"pct_advancing": None})
    survives("stance from an empty dict", ms.stance_from, {})
    survives("stance history, no history", ms.stance_history,
             pd.DataFrame(), "NEUTRAL", "2026-09-02")
    survives("stance history, no stance column", ms.stance_history,
             pd.DataFrame({"date": ["2026-09-01"]}), "NEUTRAL", "2026-09-02")

    section("price highs")
    survives("highs from an empty frame", bph.highs_from_bars, pd.DataFrame())
    survives("highs from one bar", bph.highs_from_bars, ohlcv(1)[["date", "high"]])
    survives("highs with NaT dates", bph.highs_from_bars,
             ohlcv()[["date", "high"]].assign(date=pd.NaT))
    survives("update with no table", bph.update_from_features, None, pd.DataFrame())
    survives("update with an empty table", bph.update_from_features,
             pd.DataFrame(), pd.DataFrame())
    survives("update, features lack a high column", bph.update_from_features,
             pd.DataFrame({"symbol": ["A"], "high_all_time": [100.0]}),
             pd.DataFrame({"symbol": ["A"]}))

    print()
    if _FAILS:
        print(f"{len(_FAILS)} FAILED:")
        for f in _FAILS:
            print(f"  - {f}")
    else:
        print("all stress cases survived")
    return len(_FAILS)


if __name__ == "__main__":
    sys.exit(main())
