r"""
bucket_health.py — Phase 1: remember which (key, model) buckets are PerDay-dead, so the
next backfill slot skips them instead of re-discovering it by burning a call each.

WHY: BucketPool state is per-run. Every 4-hour slot rebuilds the pool with all buckets
ALIVE, so a bucket that hit PerDay quota gets a real document, 429s, and is re-marked dead
— EVERY slot, on EVERY exhausted bucket (measured: gemini-2.0-flash burned 174 fails/day
for 0 ok). The probe (probe_models) only catches 503/404 model outages, not PerDay.

The data to fix this already exists: persist_gemini_usage() logs per-(key_idx, model)
`state` to gemini_usage.parquet. This module reads it back and returns the set of buckets
seen `dead_today` SINCE the last quota reset, which the pool then pre-marks DEAD_TODAY.

Quota reset: free-tier PerDay resets at midnight Pacific = 08:00 UTC (≈13:30 IST). CI
runners are UTC and persist_gemini_usage stamps ts via datetime.now()=UTC, so all
comparisons here are UTC. Pure functions (no Drive) → unit-testable offline.
"""
from __future__ import annotations

from datetime import datetime, timedelta

import pandas as pd

_RESET_HOUR_UTC = 8  # 08:00 UTC == midnight US-Pacific == free-tier PerDay reset


def last_quota_reset_utc(now_utc: datetime) -> datetime:
    """Most recent 08:00 UTC at or before `now_utc` (naive UTC in/out)."""
    today_reset = now_utc.replace(hour=_RESET_HOUR_UTC, minute=0, second=0, microsecond=0)
    if now_utc >= today_reset:
        return today_reset
    return today_reset - timedelta(days=1)


def dead_buckets_since_reset(usage_df: pd.DataFrame,
                             now_utc: datetime | None = None) -> set[tuple[int, str]]:
    """Return {(key_idx, model)} that were `dead_today` (PerDay-exhausted) in any usage
    row stamped at/after the last quota reset. Robust to empty/missing columns. Only
    `dead_today` is returned — `dead_run` (transient 503/RPM) should retry next run."""
    if usage_df is None or usage_df.empty:
        return set()
    for c in ("ts", "state", "key_idx", "model"):
        if c not in usage_df.columns:
            return set()
    now_utc = now_utc or datetime.utcnow()
    reset = last_quota_reset_utc(now_utc)

    df = usage_df.copy()
    # parse ts as UTC, drop tz so it compares to naive `reset`
    ts = pd.to_datetime(df["ts"], errors="coerce", utc=True)
    df = df.assign(_ts=ts.dt.tz_localize(None))
    df = df[(df["_ts"].notna()) & (df["_ts"] >= reset)]
    df = df[df["state"].astype(str).str.lower() == "dead_today"]
    out: set[tuple[int, str]] = set()
    for _, r in df.iterrows():
        try:
            out.add((int(r["key_idx"]), str(r["model"])))
        except (ValueError, TypeError):
            continue
    return out
