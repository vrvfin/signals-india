r"""
strategy_common.py — scoring helpers shared by the Phase 1 strategies.

Imported by scripts/strategy_*.py. Pure functions on numbers and DataFrames: no
Drive, no network, no argparse, so it is safe to import from anywhere and cheap
to test offline (scripts/tests/test_phase1_engine.py).

WHY THIS EXISTS
---------------
Two defects ran through nearly every strategy's `score`:

  1. The score WAS the gate. momentum gated on `rs_rank >= 90` and then ranked on
     `rs_rank`, so every signal scored 90-100 and the ordering carried almost no
     information beyond "how far past the bar", which at the top of a fat-tailed
     distribution is close to noise.

  2. Rule COUNT dominated. minervini and canslim scored `passed * 10 + rs/10`, so
     the count set the tier (60/70/80) and the only continuous input could move a
     stock 10 points at most — a 7-of-8 could never outrank an 8-of-8. That is
     the same non-discriminating rule-count trap guru measured directly:
     stocks passing 10 vs 20 families both returned ~42% with a ~67% win rate.

`min_slack_score` replaces the rule count with DISTANCE TO FAILURE: of all the
conditions a stock currently satisfies, how close is the tightest one to breaking?
A stock one point from failing its RS rule is fragile even if it technically
passes everything; a stock with room on every condition is robust. That is
continuous, bounded 0-100, explainable ("binding rule: RS, 12% of its slack"),
and — unlike "% above the 200 SMA" — it does not quietly reward extension, which
is what made ma_respect and qullamaggie rank the most-stretched name top.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = ["slack", "min_slack_score", "pct_rank", "base_quality_score"]


def slack(value, threshold, full, *, higher_is_better: bool = True):
    """Normalised head-room on one condition, in [0, 1] once clipped.

    0.0 = sitting exactly on the threshold (about to fail)
    1.0 = `full` beyond it, or further

    `full` is the point of diminishing returns — the distance past the threshold
    at which more stops counting as meaningfully safer. It is a judgement, so it
    is always passed in explicitly at the call site rather than guessed here.

    Returns NaN when the input is missing, so a missing metric can never be
    silently scored as if it had passed comfortably.
    """
    if value is None or pd.isna(value) or full in (0, None) or pd.isna(full):
        return np.nan
    margin = (float(value) - float(threshold)) if higher_is_better \
        else (float(threshold) - float(value))
    return margin / float(full)


def min_slack_score(slacks: dict[str, float]) -> tuple[float, str]:
    """Score a stock by its TIGHTEST satisfied condition.

    Returns (score 0-100, name of the binding condition).

    Only conditions the stock actually passes (slack >= 0) are considered, so a
    partially-qualifying stock is still ranked on the strength of what it does
    satisfy instead of collapsing to zero. Callers keep pass/fail as a separate
    gate — this function ranks, it does not decide membership.

    NaN slacks are ignored rather than treated as zero: a metric we could not
    compute is unknown, not tight.
    """
    usable = {k: v for k, v in slacks.items()
              if v is not None and not pd.isna(v) and v >= 0}
    if not usable:
        return 0.0, ""
    binding = min(usable, key=lambda k: usable[k])
    return round(100.0 * float(min(1.0, usable[binding])), 2), binding


def pct_rank(s: pd.Series, *, ascending: bool = True) -> pd.Series:
    """Percentile rank 0-100 within the frame, NaNs preserved as NaN.

    Used to put the W3 ranking terms on one comparable scale WITHIN a strategy's
    own signals for the day. `ascending=False` ranks smaller values higher, which
    is what risk wants: a tighter stop is a better trade.
    """
    v = pd.to_numeric(s, errors="coerce")
    if not ascending:
        v = -v
    return (v.rank(pct=True) * 100).round(2)


def _clip01(x: float) -> float:
    return 0.0 if x < 0 else (1.0 if x > 1 else float(x))


def base_quality_score(range_pct, max_range_pct, days, min_days, max_days,
                       vol_ratio, *, breakout: bool) -> tuple[float, dict]:
    """Quality of a consolidation base and its trigger, 0-100.

    Replaces the single-number scores the base strategies were using:
    qullamaggie ranked on `return_3m_pct` and darvas on distance-to-52w-high, so
    BOTH put the most-extended name at the top — the worst risk/reward in the
    list — while the base geometry each script had already computed (range,
    duration, volume) was discarded.

    Three parts, each normalised to 0-1:
      tightness    a narrower base is a cleaner, lower-risk setup and gives a
                   tighter stop. Measured against the strategy's own max range.
      duration     a longer base is more accumulation. Measured between the
                   strategy's own min and max base length.
      confirmation on a breakout, volume EXPANSION; inside a base, volume DRY-UP.
                   Deliberately opposite tells for opposite situations.

    Weighted 0.4 / 0.3 / 0.3 — tightness leads because it is the part that also
    determines the stop distance, and therefore position size. These weights are
    a judgement and are exactly what the W9 outcome tracker is meant to settle;
    the component parts are returned so they can be scored separately later.
    """
    tight = (1.0 - _clip01(float(range_pct) / float(max_range_pct))
             if max_range_pct else np.nan)
    span = max(float(max_days) - float(min_days), 1.0)
    dur = _clip01((float(days) - float(min_days)) / span)
    if breakout:
        conf = _clip01((float(vol_ratio) - 1.0) / 2.0)      # 3x avg = full marks
    else:
        conf = _clip01((1.0 - float(vol_ratio)) / 0.5)      # 0.5x avg = full marks
    parts = {"tightness": tight, "duration": dur, "confirmation": conf}
    if pd.isna(tight):
        return 0.0, parts
    score = 100.0 * (0.4 * tight + 0.3 * dur + 0.3 * conf)
    return round(score, 2), parts
