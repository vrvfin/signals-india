"""
pipeline_skip_check.py — skip guard for Phase 1 and Phase 2 workflows.

Called as the FIRST step of each workflow. Reads the latest pipeline
status from Drive and decides whether the run should proceed or be skipped.

┌─────────────────────────────────────────────────────────────────────────┐
│  PHASE 2 SEASONAL RUN STRATEGY                                          │
│                                                                         │
│  India Q-end results seasons (peak = more runs per day):                │
│    Q3 FY:  17 Jan – 28 Feb   (quarter ended Dec)                        │
│    Q4 FY:  17 Apr – 30 May   (quarter ended Mar; 60-day season)         │
│    Q1 FY:  17 Jul – 30 Aug   (quarter ended Jun)                        │
│    Q2 FY:  17 Oct – 30 Nov   (quarter ended Sep)                        │
│                                                                         │
│  Peak season:                                                           │
│    phase2.yml runs 6–7×/day weekdays, 5–6× Sat, 3× Sun.               │
│    skip threshold = 45 min (process as fast as the queue allows)        │
│                                                                         │
│  Off-season:                                                            │
│    Runs are suppressed to 3/day for ALL day types.                      │
│    Allowed windows: IST 08:00 ±75 min, 14:00 ±75 min, 20:00 ±75 min   │
│    Runs outside those windows are skipped immediately.                   │
│    Within a window, skip threshold = 90 min (one run per window max).   │
│                                                                         │
│  GitHub Actions scheduled jobs can be delayed 30–90 min.               │
│  The ±75 min window radius absorbs those delays reliably.               │
└─────────────────────────────────────────────────────────────────────────┘

Phase 1 skip logic (unchanged):
  Skip if last run was today (same calendar date, IST).

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
from datetime import datetime, timezone, timedelta
from pathlib import Path

from dotenv import load_dotenv
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build

SCOPES = ["https://www.googleapis.com/auth/drive"]
IST_OFFSET = timedelta(hours=5, minutes=30)

# ── Seasonal configuration ────────────────────────────────────────────────────

# India quarterly results seasons: (start_month, start_day, end_month, end_day)
# All within a single calendar year (none cross Dec→Jan boundary).
PEAK_SEASONS = [
    (1, 17, 2, 28),    # Q3 FY: 17 Jan – 28 Feb  (dec quarter)
    (4, 17, 5, 30),    # Q4 FY: 17 Apr – 30 May  (mar quarter; 60-day annual)
    (7, 17, 8, 30),    # Q1 FY: 17 Jul – 30 Aug  (jun quarter)
    (10, 17, 11, 30),  # Q2 FY: 17 Oct – 30 Nov  (sep quarter)
]

# Off-season: only allow runs within ±WINDOW_RADIUS_MIN of these IST hours
OFFSEASON_WINDOW_HOURS_IST = [8, 14, 20]   # 3 designated daily slots
OFFSEASON_WINDOW_RADIUS_MIN = 75           # ±75 min absorbs GitHub job delays

# Skip thresholds: if queue is empty AND last run was this recent → skip
PHASE2_PEAK_SKIP_MIN     = 45   # peak: process as fast as queue allows
PHASE2_OFFSEASON_SKIP_MIN = 90   # off-season: at most 1 run per window


# ── helpers ───────────────────────────────────────────────────────────────────

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

def _is_peak_season(ist_now: datetime) -> bool:
    """Return True if ist_now falls within any India results season."""
    m, d = ist_now.month, ist_now.day
    for sm, sd, em, ed in PEAK_SEASONS:
        after_start = (m > sm) or (m == sm and d >= sd)
        before_end  = (m < em) or (m == em and d <= ed)
        if after_start and before_end:
            return True
    return False


def _in_offseason_window(ist_now: datetime) -> bool:
    """Return True if ist_now is within ±OFFSEASON_WINDOW_RADIUS_MIN of a
    designated off-season run slot (08:00, 14:00, 20:00 IST)."""
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
            f"(08:00 / 14:00 / 20:00 IST ±{OFFSEASON_WINDOW_RADIUS_MIN} min)"
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

    dt_last   = datetime.fromisoformat(run_at.replace("Z", "+00:00"))
    today_ist = (datetime.now(timezone.utc) + IST_OFFSET).date()
    last_ist  = (dt_last + IST_OFFSET).date()

    if last_ist == today_ist:
        return skip_run(f"Phase 1 already ran today (IST) — {age:.0f}m ago")

    proceed()


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pipeline", choices=["phase1", "phase2"], required=True)
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
