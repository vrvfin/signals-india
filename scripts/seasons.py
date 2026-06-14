"""
seasons.py — single source of truth for India earnings-season (peak) windows.

Pure standard library (no Drive/Gemini deps) so any lightweight workflow gate —
pipeline_skip_check.py (Phase 2) and backfill_skip_check.py (backfill) — can
import it and decide season without installing the pipeline requirements.

India quarterly results windows (FY = Apr–Mar). Companies must file within 45
days of quarter-end (60 for the annual Q4), and concalls cluster ~a week after
the filing rush — so each window runs from ~day 15 to ~day 60 after quarter-end
(day 75 for the annual Q4).
"""
from __future__ import annotations

from datetime import datetime

# (start_month, start_day, end_month, end_day) — all within one calendar year
# (none cross the Dec→Jan boundary).
PEAK_SEASONS = [
    (1, 15, 3, 1),     # Q3 FY results (Dec qtr):   15 Jan – 1 Mar
    (4, 15, 6, 14),    # Q4 FY / annual (Mar qtr):  15 Apr – 14 Jun  (60-day window)
    (7, 15, 8, 29),    # Q1 FY results (Jun qtr):   15 Jul – 29 Aug
    (10, 15, 11, 29),  # Q2 FY results (Sep qtr):   15 Oct – 29 Nov
]


def is_peak_season(ist_now: datetime) -> bool:
    """True if the given IST datetime falls within any India results season."""
    m, d = ist_now.month, ist_now.day
    for sm, sd, em, ed in PEAK_SEASONS:
        after_start = (m > sm) or (m == sm and d >= sd)
        before_end = (m < em) or (m == em and d <= ed)
        if after_start and before_end:
            return True
    return False
