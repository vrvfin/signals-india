"""
pipeline_skip_check.py — skip guard for self-hosted runner.

Called as the FIRST step of each workflow. Reads the latest pipeline
status from Drive and decides whether the run should proceed or be
skipped because work was already done recently.

Phase 2 skip logic:
  Skip if ALL of:
    - phase2_latest.json exists on Drive
    - last run was < PHASE2_SKIP_MINUTES ago
    - queue pending count is 0
  Reason: when the machine was off and multiple cron slots queued up,
  the first run processes everything; the rest find an empty queue and
  can exit immediately without wasting local compute.

Phase 1 skip logic:
  Skip if ALL of:
    - logs/health/latest.json exists on Drive
    - last run was today (same calendar date, IST)
  Reason: Phase 1 is a daily pipeline; if it already ran today, a
  second queued run (from a manual trigger or duplicate cron) is
  redundant — data is already fresh.

Exit behaviour:
  - Writes skip=true  to $GITHUB_OUTPUT → caller skips pipeline steps
  - Writes skip=false to $GITHUB_OUTPUT → caller proceeds normally
  - On any error (Drive unreachable, file missing, etc.) → skip=false
    (fail safe: always proceed if we cannot confirm freshness)

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

# Phase 2: skip if queue is empty AND last run was this recent
PHASE2_SKIP_MINUTES = 45

# Phase 1: skip if already ran today (same date in IST = UTC+5:30)
IST_OFFSET = timedelta(hours=5, minutes=30)


# ── helpers ──────────────────────────────────────────────────────────────────

def log(msg: str) -> None:
    print(f"[skip-check] {msg}")


def write_output(skip: bool) -> None:
    """Write skip=true/false to $GITHUB_OUTPUT (no-op if not in Actions)."""
    val = "true" if skip else "false"
    github_output = os.environ.get("GITHUB_OUTPUT")
    if github_output:
        with open(github_output, "a") as fh:
            fh.write(f"skip={val}\n")
    log(f"skip={val}")


def proceed() -> None:
    write_output(False)
    sys.exit(0)


def skip(reason: str) -> None:
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


# ── checks ───────────────────────────────────────────────────────────────────

def check_phase2(drive, folder_id: str) -> None:
    logs_id   = find_sub(drive, folder_id, "logs")
    health_id = find_sub(drive, logs_id,   "health") if logs_id else None
    if not health_id:
        log("No logs/health folder found — proceeding")
        return proceed()

    fid = find_file(drive, health_id, "phase2_latest.json")
    if not fid:
        log("phase2_latest.json not found — proceeding")
        return proceed()

    report = read_json(drive, fid)
    run_at  = report.get("run_at")
    totals  = report.get("queue_totals", {})
    pending = totals.get("pending", 1)   # default 1 → proceed if unknown

    if not run_at:
        return proceed()

    age = age_minutes(run_at)
    log(f"Phase 2 last ran {age:.1f} min ago, pending={pending}")

    if age < PHASE2_SKIP_MINUTES and pending == 0:
        return skip(f"queue empty, last run {age:.0f}m ago (threshold {PHASE2_SKIP_MINUTES}m)")

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

    # Skip only if it ran today (IST) — i.e. < 23h ago AND same IST date
    if age >= 23 * 60:
        return proceed()

    dt_last = datetime.fromisoformat(run_at.replace("Z", "+00:00"))
    today_ist = (datetime.now(timezone.utc) + IST_OFFSET).date()
    last_ist  = (dt_last + IST_OFFSET).date()

    if last_ist == today_ist:
        return skip(f"Phase 1 already ran today (IST) — {age:.0f}m ago")

    proceed()


# ── main ─────────────────────────────────────────────────────────────────────

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
