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
