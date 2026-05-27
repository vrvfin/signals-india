"""
write_phase2_status.py — Phase 2 pipeline status writer.

Runs as the LAST step of phase2.yml (after all extractors).
Reads processing_queue.parquet from Drive and writes a summary to
logs/health/phase2_latest.json so app.py can show Phase 2 health
alongside the Phase 1 health report.

Output schema (logs/health/phase2_latest.json):
{
  "run_at": "<ISO timestamp>",
  "queue_totals": {"pending": N, "done": N, "error": N, "total": N},
  "by_doc_type": {
    "<type>": {"pending": N, "done": N, "error": N}
  }
}
"""

from __future__ import annotations

import io
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseUpload

SCOPES = ["https://www.googleapis.com/auth/drive"]


def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")


# ── Drive auth (identical pattern to other scripts) ───────────────────────────

def get_drive():
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
    tk_json = os.environ.get("GDRIVE_OAUTH_TOKEN_JSON")
    if tk_json:
        creds = Credentials.from_authorized_user_info(json.loads(tk_json), SCOPES)
        if creds.expired and creds.refresh_token:
            creds.refresh(Request())
        return build("drive", "v3", credentials=creds, cache_discovery=False)
    tk_path = Path(os.environ["GDRIVE_OAUTH_TOKEN_PATH"])
    creds = Credentials.from_authorized_user_file(str(tk_path), SCOPES)
    if creds.expired and creds.refresh_token:
        creds.refresh(Request())
    return build("drive", "v3", credentials=creds, cache_discovery=False)


# ── Drive helpers ─────────────────────────────────────────────────────────────

def find_sub(drive, parent_id, name):
    q = (f"name='{name}' and '{parent_id}' in parents "
         f"and mimeType='application/vnd.google-apps.folder' and trashed=false")
    f = drive.files().list(q=q, fields="files(id)").execute().get("files", [])
    return f[0]["id"] if f else None


def get_or_create_sub(drive, parent_id, name):
    fid = find_sub(drive, parent_id, name)
    if fid:
        return fid
    meta = {"name": name, "parents": [parent_id],
            "mimeType": "application/vnd.google-apps.folder"}
    return drive.files().create(body=meta, fields="id").execute()["id"]


def find_file(drive, folder_id, name):
    q = f"name='{name}' and '{folder_id}' in parents and trashed=false"
    f = drive.files().list(q=q, fields="files(id)").execute().get("files", [])
    return f[0]["id"] if f else None


def download_bytes(drive, file_id):
    from googleapiclient.http import MediaIoBaseDownload
    fh = io.BytesIO()
    dl = MediaIoBaseDownload(fh, drive.files().get_media(fileId=file_id))
    done = False
    while not done:
        _, done = dl.next_chunk()
    fh.seek(0)
    return fh.read()


def upload_json(drive, folder_id, filename, obj):
    body = json.dumps(obj, indent=2, default=str).encode()
    media = MediaIoBaseUpload(io.BytesIO(body), mimetype="application/json",
                              resumable=False)
    existing = find_file(drive, folder_id, filename)
    if existing:
        drive.files().update(fileId=existing, media_body=media).execute()
    else:
        drive.files().create(
            body={"name": filename, "parents": [folder_id]},
            media_body=media, fields="id").execute()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    log("Phase 2 status writer — starting")
    folder_id = os.environ.get("GDRIVE_FOLDER_ID")
    if not folder_id:
        load_dotenv(Path(__file__).resolve().parent.parent / ".env")
        folder_id = os.environ["GDRIVE_FOLDER_ID"]

    try:
        drive = get_drive()
    except Exception as e:
        log(f"Drive auth failed: {e}")
        sys.exit(0)          # non-critical — don't fail the workflow

    # ── Read queue ────────────────────────────────────────────────────────────
    status = {
        "run_at": datetime.now(timezone.utc).isoformat(),
        "queue_totals": {"pending": 0, "done": 0, "error": 0, "total": 0},
        "by_doc_type": {},
    }

    try:
        company_repo_id = find_sub(drive, folder_id, "company_repo")
        index_id = find_sub(drive, company_repo_id, "_index") if company_repo_id else None
        queue_fid = find_file(drive, index_id, "processing_queue.parquet") if index_id else None

        if queue_fid:
            raw = download_bytes(drive, queue_fid)
            queue = pd.read_parquet(io.BytesIO(raw))
            totals = queue["status"].value_counts().to_dict()
            status["queue_totals"] = {
                "pending": int(totals.get("pending", 0)),
                "done":    int(totals.get("done",    0)),
                "error":   int(totals.get("error",   0)),
                "total":   len(queue),
            }
            for dt, grp in queue.groupby("doc_type"):
                cnts = grp["status"].value_counts().to_dict()
                status["by_doc_type"][str(dt)] = {
                    "pending": int(cnts.get("pending", 0)),
                    "done":    int(cnts.get("done",    0)),
                    "error":   int(cnts.get("error",   0)),
                }
            log(f"  Queue: {status['queue_totals']}")
        else:
            log("  Queue file not found — writing empty status")
    except Exception as e:
        log(f"  Queue read error: {str(e)[:200]}")
        status["read_error"] = str(e)[:200]

    # ── Write to Drive: logs/health/phase2_latest.json ────────────────────────
    try:
        logs_id   = get_or_create_sub(drive, folder_id, "logs")
        health_id = get_or_create_sub(drive, logs_id,   "health")
        upload_json(drive, health_id, "phase2_latest.json", status)
        log("  Wrote logs/health/phase2_latest.json")
    except Exception as e:
        log(f"  Write failed: {str(e)[:200]}")

    log("Done.")


if __name__ == "__main__":
    main()
