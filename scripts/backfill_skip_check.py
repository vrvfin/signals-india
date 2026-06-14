"""
backfill_skip_check.py — seasonal/time gate for backfill_nightly.yml.

Backfill is LOW priority vs Phase 2. During earnings season it must stay out of
Phase 2's way; off-season it can run freely. This gate (pure stdlib, runs before
deps install) emits three values to $GITHUB_OUTPUT:

    skip=true|false       whether to skip the whole backfill job this slot
    deadline_min=<int>    minutes the concall-extract step may run (capped so the
                          shared _extract.lock is released before Phase 2 starts)
    run_extras=true|false whether to also run the long extra lock-holders this slot
                          (T8 AR extract, T9 classification) — false on the short
                          morning run so they don't overrun the 09:30 cutoff

Decision rules (IST):
  • Manual workflow_dispatch         → never skip, full pipeline, 200-min cap.
  • Peak season, Mon–Sat             → run ONLY in the night window [22:30, 09:30];
                                       daytime slots (11/15/19 IST) are skipped.
  • Peak season, Sunday              → run every slot (Phase 2 is light on Sundays).
  • Off-season (any day)             → run every slot.
  • A peak Mon–Sat MORNING run that can't fit a full cycle before 09:30 IST gets
    its extract deadline capped to 09:30 and skips the extra lock-holders; if less
    than 30 min remain to 09:30, the slot is skipped entirely.

Phase 2's first peak slot is 10:00 IST, so the 09:30 cutoff leaves a 30-min buffer.
"""
from __future__ import annotations

import os
import sys
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from seasons import is_peak_season

IST = timezone(timedelta(hours=5, minutes=30))
FULL_DEADLINE_MIN = 200          # normal catch-up extract cap
CUTOFF_HH, CUTOFF_MM = 9, 30     # 09:30 IST — backfill must release the lock by here
NIGHT_START = (22, 30)           # 22:30 IST
MIN_USEFUL_MIN = 30              # below this much runway to the cutoff → skip


def _mins_to_cutoff(ist_now: datetime) -> int:
    """Minutes from now to the next 09:30 IST."""
    cutoff = ist_now.replace(hour=CUTOFF_HH, minute=CUTOFF_MM, second=0, microsecond=0)
    if ist_now >= cutoff:
        cutoff += timedelta(days=1)
    return int((cutoff - ist_now).total_seconds() // 60)


def _in_night_window(ist_now: datetime) -> bool:
    """True if within [22:30, 09:30] IST (the backfill-owned overnight window)."""
    t = (ist_now.hour, ist_now.minute)
    return t >= NIGHT_START or (ist_now.hour, ist_now.minute) <= (CUTOFF_HH, CUTOFF_MM)


def decide(ist_now: datetime, event: str) -> dict:
    is_peak = is_peak_season(ist_now)
    is_sunday = ist_now.weekday() == 6
    manual = event == "workflow_dispatch"
    mins = _mins_to_cutoff(ist_now)

    # 1) Skip decision.
    if manual:
        skip = False
    elif is_peak and not is_sunday:
        skip = not _in_night_window(ist_now)          # peak Mon–Sat: night only
    else:
        skip = False                                  # peak Sunday / off-season: all slots

    # 2) Deadline + extras. Only the peak Mon–Sat MORNING approach to 09:30 is capped.
    deadline_min = FULL_DEADLINE_MIN
    run_extras = True
    morning_short = (not manual and is_peak and not is_sunday
                     and _in_night_window(ist_now)
                     and mins < FULL_DEADLINE_MIN + 10)   # can't fit a full cycle
    if morning_short:
        if mins < MIN_USEFUL_MIN:
            skip = True                               # too little runway — skip
        else:
            deadline_min = max(MIN_USEFUL_MIN, mins - 15)  # 15-min buffer for cleanup
            run_extras = False                        # concall-extract only this slot

    return {"skip": skip, "deadline_min": int(deadline_min),
            "run_extras": run_extras, "is_peak": is_peak,
            "is_sunday": is_sunday, "mins_to_cutoff": mins}


def _emit(d: dict) -> None:
    out = os.environ.get("GITHUB_OUTPUT")
    if out:
        with open(out, "a", encoding="utf-8") as fh:
            fh.write(f"skip={'true' if d['skip'] else 'false'}\n")
            fh.write(f"deadline_min={d['deadline_min']}\n")
            fh.write(f"run_extras={'true' if d['run_extras'] else 'false'}\n")


def main() -> None:
    event = os.environ.get("GITHUB_EVENT_NAME", "schedule")
    try:
        ist_now = datetime.now(timezone.utc).astimezone(IST)
        d = decide(ist_now, event)
        print(f"[backfill-skip] IST={ist_now:%Y-%m-%d %H:%M} ({ist_now:%a}) event={event}")
        print(f"  peak={d['is_peak']} sunday={d['is_sunday']} "
              f"mins_to_0930={d['mins_to_cutoff']}")
        print(f"  -> skip={d['skip']} deadline_min={d['deadline_min']} "
              f"run_extras={d['run_extras']}")
        _emit(d)
    except Exception as e:
        # Fail-safe: on any error, RUN the slot with a full deadline (never
        # silently skip the whole backfill, never pass an empty deadline).
        print(f"[backfill-skip] ERROR ({str(e)[:120]}) — fail-safe: run, 200 min.")
        _emit({"skip": False, "deadline_min": FULL_DEADLINE_MIN, "run_extras": True})


if __name__ == "__main__":
    main()
