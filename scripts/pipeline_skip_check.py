"""
pipeline_skip_check.py — skip guard for Phase 1 and Phase 2 workflows.

Called as the FIRST step of each workflow. Reads the latest pipeline
status from Drive and decides whether the run should proceed or be skipped.

┌─────────────────────────────────────────────────────────────────────────┐
│  PHASE 2 SEASONAL RUN STRATEGY                                          │
│                                                                         │
│  India Q-end results seasons (peak = more runs per day):                │
│    Q3 FY:  15 Jan – 1 Mar    (quarter ended Dec)                        │
│    Q4 FY:  15 Apr – 14 Jun   (quarter ended Mar; 60-day annual season)  │
│    Q1 FY:  15 Jul – 29 Aug   (quarter ended Jun)                        │
│    Q2 FY:  15 Oct – 29 Nov   (quarter ended Sep)                        │
│                                                                         │
│  Peak season:                                                           │
│    phase2.yml runs every 2h 10:00–22:00 IST weekdays (7/day),          │
│    lighter Sat, 3× Sun. skip threshold = 45 min (drain as fast as able) │
│                                                                         │
│  Off-season:                                                            │
│    Runs are suppressed to 3/day for ALL day types.                      │
│    Allowed windows: IST 09:00 ±75 min, 14:00 ±75 min, 20:00 ±75 min   │
│    Runs outside those windows are skipped immediately.                   │
│    Within a window, skip threshold = 90 min (one run per window max).   │
│                                                                         │
│  GitHub Actions scheduled jobs can be delayed 30–90 min.               │
│  The ±75 min window radius absorbs those delays reliably.               │
└─────────────────────────────────────────────────────────────────────────┘

Phase 1 skip logic:
  Skip only if the last run already processed the session we would process now
  AND it did so AFTER that session closed. A run before the close sees a partial
  intraday bar; counting it as "done" cancels the real post-close run. Falls back
  to the old same-IST-calendar-day rule when the report predates bar_date.

Exit behaviour:
  - Writes skip=true  to $GITHUB_OUTPUT → caller skips pipeline steps
  - Writes skip=false to $GITHUB_OUTPUT → caller proceeds normally
  - On any error (Drive unreachable, file missing) → skip=false (fail-safe)

Usage:
    python scripts/pipeline_skip_check.py --pipeline phase2
    python scripts/pipeline_skip_check.py --pipeline phase1
"""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
from datetime import datetime, timezone, timedelta, date
from pathlib import Path

from dotenv import load_dotenv
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build

# Single source of truth for the earnings-season windows (shared with the
# backfill gate). Imported by name so this file's internal calls keep working.
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from seasons import PEAK_SEASONS, is_peak_season as _is_peak_season  # noqa: E402

SCOPES = ["https://www.googleapis.com/auth/drive"]
IST_OFFSET = timedelta(hours=5, minutes=30)

# ── Seasonal configuration ────────────────────────────────────────────────────

# India quarterly results seasons now live in seasons.py (imported above):
#   15 Jan–1 Mar · 15 Apr–14 Jun · 15 Jul–29 Aug · 15 Oct–29 Nov
# (PEAK_SEASONS / _is_peak_season are imported at the top of this file.)

# Off-season: only allow runs within ±WINDOW_RADIUS_MIN of these IST hours
# First slot 8 → 9 (2026-06-10): backfill owns 23:00–08:30 IST overnight.
OFFSEASON_WINDOW_HOURS_IST = [9, 14, 20]   # 3 designated daily slots
OFFSEASON_WINDOW_RADIUS_MIN = 75           # ±75 min absorbs GitHub job delays

# Skip thresholds: if queue is empty AND last run was this recent → skip
PHASE2_PEAK_SKIP_MIN     = 45   # peak: process as fast as queue allows
PHASE2_OFFSEASON_SKIP_MIN = 90   # off-season: at most 1 run per window

# Phase 1: the IST hour from which a session's bars should be on Drive. The
# pipeline is triggered at 16:00 IST, after the 15:30 IST close.
PHASE1_SESSION_CUTOFF_IST_HOUR = 16


# ── helpers ───────────────────────────────────────────────────────────────────

# Console encoding. Several scripts here log the rupee sign, a delta or an em
# dash, and a Windows console is cp1252 — so a run could complete all its work
# and then die in a log line. It cost three separate crashes before being fixed
# in one place. Degrade the characters, never the run.
try:
    sys.stdout.reconfigure(errors="replace")
    sys.stderr.reconfigure(errors="replace")
except Exception:          # pragma: no cover - not every stream supports it
    pass

def log(msg: str) -> None:
    print(f"[skip-check] {msg}")


def write_output(skip: bool) -> None:
    val = "true" if skip else "false"
    github_output = os.environ.get("GITHUB_OUTPUT")
    if github_output:
        with open(github_output, "a") as fh:
            fh.write(f"skip={val}\n")
    log(f"skip={val}")


def proceed() -> None:
    write_output(False)
    sys.exit(0)


def skip_run(reason: str) -> None:
    log(f"SKIPPING — {reason}")
    write_output(True)
    sys.exit(0)


def get_drive():
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
    tk_json = os.environ.get("GDRIVE_OAUTH_TOKEN_JSON")
    if tk_json:
        creds = Credentials.from_authorized_user_info(json.loads(tk_json), SCOPES)
        if creds.expired and creds.refresh_token:
            creds.refresh(Request())
        return build("drive", "v3", credentials=creds, cache_discovery=False)
    cs_path = os.environ.get("GDRIVE_OAUTH_CLIENT_SECRET_PATH", "")
    tk_path = os.environ.get("GDRIVE_OAUTH_TOKEN_PATH", "")
    if not cs_path or not tk_path:
        raise RuntimeError("No Drive credentials found in environment")
    creds = Credentials.from_authorized_user_file(tk_path, SCOPES)
    if creds.expired and creds.refresh_token:
        creds.refresh(Request())
    return build("drive", "v3", credentials=creds, cache_discovery=False)


def find_sub(drive, parent_id: str, name: str) -> str | None:
    q = (f"name='{name}' and '{parent_id}' in parents "
         f"and mimeType='application/vnd.google-apps.folder' and trashed=false")
    files = drive.files().list(q=q, fields="files(id)").execute().get("files", [])
    return files[0]["id"] if files else None


def find_file(drive, folder_id: str, name: str) -> str | None:
    q = f"name='{name}' and '{folder_id}' in parents and trashed=false"
    files = drive.files().list(q=q, fields="files(id)").execute().get("files", [])
    return files[0]["id"] if files else None


def read_json(drive, file_id: str) -> dict:
    raw = drive.files().get_media(fileId=file_id).execute()
    return json.loads(raw)


def age_minutes(iso_ts: str) -> float:
    dt = datetime.fromisoformat(iso_ts.replace("Z", "+00:00"))
    return (datetime.now(timezone.utc) - dt).total_seconds() / 60


# ── seasonal helpers ──────────────────────────────────────────────────────────
# _is_peak_season is imported from seasons.py (single source of truth).


def _expected_bar_date(ist_now: datetime) -> date:
    """The most recent trading session whose bars should already exist.

    Before PHASE1_SESSION_CUTOFF_IST_HOUR today's bars are not published yet, so
    the expectation rolls back a day; weekends roll back to Friday. Exchange
    holidays are deliberately NOT modelled: on a holiday the expected date has no
    bar, the comparison below fails, and we run again. Re-running is idempotent,
    whereas skipping a real session loses a day of signals — so the fail-safe
    direction is to run."""
    d = ist_now.date()
    if ist_now.hour < PHASE1_SESSION_CUTOFF_IST_HOUR:
        d -= timedelta(days=1)
    while d.weekday() >= 5:                 # Sat=5, Sun=6 → back to Friday
        d -= timedelta(days=1)
    return d


def _in_offseason_window(ist_now: datetime) -> bool:
    """Return True if ist_now is within ±OFFSEASON_WINDOW_RADIUS_MIN of a
    designated off-season run slot (09:00, 14:00, 20:00 IST)."""
    current_min = ist_now.hour * 60 + ist_now.minute
    for target_hour in OFFSEASON_WINDOW_HOURS_IST:
        target_min = target_hour * 60
        if abs(current_min - target_min) <= OFFSEASON_WINDOW_RADIUS_MIN:
            return True
    return False


# ── phase checks ─────────────────────────────────────────────────────────────

def check_phase2(drive, folder_id: str) -> None:
    ist_now  = datetime.now(timezone.utc) + IST_OFFSET
    is_peak  = _is_peak_season(ist_now)
    ist_str  = ist_now.strftime("%d %b %H:%M IST")
    season_label = "PEAK" if is_peak else "off-season"

    log(f"Season check: {season_label} ({ist_str})")

    # Off-season gate: skip immediately if not in a designated run window
    if not is_peak and not _in_offseason_window(ist_now):
        return skip_run(
            f"off-season ({ist_str}): outside designated windows "
            f"(09:00 / 14:00 / 20:00 IST ±{OFFSEASON_WINDOW_RADIUS_MIN} min)"
        )

    # Choose skip threshold based on season
    skip_minutes = PHASE2_PEAK_SKIP_MIN if is_peak else PHASE2_OFFSEASON_SKIP_MIN
    log(f"Skip threshold: {skip_minutes} min ({'peak' if is_peak else 'in window'})")

    # Read last run report from Drive
    logs_id   = find_sub(drive, folder_id, "logs")
    health_id = find_sub(drive, logs_id,   "health") if logs_id else None
    if not health_id:
        log("No logs/health folder found — proceeding")
        return proceed()

    fid = find_file(drive, health_id, "phase2_latest.json")
    if not fid:
        log("phase2_latest.json not found — proceeding")
        return proceed()

    report  = read_json(drive, fid)
    run_at  = report.get("run_at")
    pending = report.get("queue_totals", {}).get("pending", 1)  # default 1 → proceed if unknown

    if not run_at:
        return proceed()

    age = age_minutes(run_at)
    log(f"Last run: {age:.1f} min ago, pending={pending}")

    if age < skip_minutes and pending == 0:
        return skip_run(
            f"queue empty, last run {age:.0f}m ago "
            f"(threshold {skip_minutes}m, {season_label})"
        )

    proceed()


def check_phase1(drive, folder_id: str) -> None:
    logs_id   = find_sub(drive, folder_id, "logs")
    health_id = find_sub(drive, logs_id,   "health") if logs_id else None
    if not health_id:
        return proceed()

    fid = find_file(drive, health_id, "latest.json")
    if not fid:
        return proceed()

    report = read_json(drive, fid)
    run_at = report.get("run_at")
    if not run_at:
        return proceed()

    age = age_minutes(run_at)
    log(f"Phase 1 last ran {age:.1f} min ago")

    if age >= 23 * 60:
        return proceed()

    ist_now = datetime.now(timezone.utc) + IST_OFFSET

    # Prefer the SESSION the last run actually processed over the wall-clock date
    # it happened to start on. A run that begins at 02:30 IST is processing the
    # PREVIOUS session, but its calendar date is today — which is how the real
    # 16:00 IST run came to be skipped as a duplicate on 2026-08-28.
    bar_date = report.get("bar_date")
    if bar_date:
        try:
            last_bar = date.fromisoformat(str(bar_date)[:10])
        except ValueError:
            last_bar = None
        if last_bar is not None:
            want = _expected_bar_date(ist_now)
            # A run that happened BEFORE the close saw a partial intraday bar and
            # stamped it with today's date. Treating that as "today's session is
            # done" cancels the real post-close run — which is exactly what
            # happened on 2026-09-03: a 12:36 IST test run claimed the date and
            # the 16:00 IST trigger stood down 134 minutes later.
            ran_at_ist = (datetime.fromisoformat(run_at.replace("Z", "+00:00"))
                          + IST_OFFSET)
            complete = ran_at_ist.hour >= PHASE1_SESSION_CUTOFF_IST_HOUR
            if last_bar >= want and complete:
                return skip_run(f"session {last_bar} already processed after the "
                                f"close (expected {want}) — {age:.0f}m ago")
            if last_bar >= want and not complete:
                log(f"last run processed {last_bar} at "
                    f"{ran_at_ist:%H:%M} IST, BEFORE the "
                    f"{PHASE1_SESSION_CUTOFF_IST_HOUR}:00 close — that bar is "
                    f"partial, so this session is not done. Proceeding.")
                return proceed()
            log(f"last run processed session {last_bar}, expected {want} "
                f"— proceeding")
            return proceed()

    # Fallback for reports written before bar_date existed: the original
    # same-IST-calendar-day rule.
    dt_last   = datetime.fromisoformat(run_at.replace("Z", "+00:00"))
    today_ist = ist_now.date()
    last_ist  = (dt_last + IST_OFFSET).date()

    if last_ist == today_ist:
        return skip_run(f"Phase 1 already ran today (IST) — {age:.0f}m ago "
                        f"(no bar_date in report)")

    proceed()


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pipeline", choices=["phase1", "phase2"], required=True)
    # Accepted and ignored: this script writes nothing to Drive, it only decides
    # whether the caller should proceed. Declared so that dry-running the whole
    # pipeline with one flag does not fall over here with exit 2.
    parser.add_argument("--dry-run", action="store_true",
                        help="accepted for consistency; this script never writes")
    args = parser.parse_args()

    try:
        drive = get_drive()
        folder_id = os.environ["GDRIVE_FOLDER_ID"]
    except Exception as e:
        log(f"Drive unavailable ({str(e)[:100]}) — proceeding to be safe")
        return proceed()

    try:
        if args.pipeline == "phase2":
            check_phase2(drive, folder_id)
        else:
            check_phase1(drive, folder_id)
    except Exception as e:
        log(f"Check failed ({str(e)[:100]}) — proceeding to be safe")
        proceed()


if __name__ == "__main__":
    main()
