"""
Storage hygiene — DRIVE-WIDE hard PDF retention.

USER RULE (2026-06-12): every PDF anywhere under the project Drive root dies
after RETAIN_DAYS (2), REGARDLESS of which process stored it and regardless of
queue status. PDFs are transient inputs — the lasting value lives in the
markdown summaries, company pages and parquet indexes, which are never touched
(only files that are PDFs by mime type or .pdf extension are deleted).

The whole tree under GDRIVE_FOLDER_ID is walked (BFS), not just
company_repo/<ISIN>/documents/. Queue rows whose PDF was deleted are marked
status="expired" (under the shared _extract.lock) so they never become
zombies; a re-fetch queues a fresh copy if ever needed.

Usage:
    python scripts/cleanup_company_docs.py
    python scripts/cleanup_company_docs.py --retain-days 14
    python scripts/cleanup_company_docs.py --dry-run     # list, delete nothing
"""

from __future__ import annotations

import argparse
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

from dotenv import load_dotenv
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

SCOPES = ["https://www.googleapis.com/auth/drive"]
RETAIN_DAYS = 2


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")


# ---------- Drive helpers ----------

def get_drive():
    import json
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
    cs_path = Path(os.environ["GDRIVE_OAUTH_CLIENT_SECRET_PATH"])
    cred_data = json.loads(cs_path.read_text())

    # Service account key — used in GitHub Actions (no browser flow needed)
    if cred_data.get("type") == "service_account":
        from google.oauth2 import service_account
        creds = service_account.Credentials.from_service_account_file(
            str(cs_path), scopes=SCOPES
        )
        return build("drive", "v3", credentials=creds, cache_discovery=False)

    # Saved OAuth token (Credentials.to_json() format) — has refresh_token directly
    if "refresh_token" in cred_data:
        creds = Credentials.from_authorized_user_file(str(cs_path), SCOPES)
        if not creds.valid:
            creds.refresh(Request())
            cs_path.write_text(creds.to_json())
        return build("drive", "v3", credentials=creds, cache_discovery=False)

    # Standard OAuth installed-app flow (proper client_secrets.json)
    token_path = Path(os.environ["GDRIVE_OAUTH_TOKEN_PATH"])
    creds = None
    if token_path.exists():
        creds = Credentials.from_authorized_user_file(str(token_path), SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            try:
                creds.refresh(Request())
            except Exception:
                creds = None
        if not creds or not creds.valid:
            flow = InstalledAppFlow.from_client_secrets_file(str(cs_path), SCOPES)
            creds = flow.run_local_server(port=0)
        token_path.parent.mkdir(parents=True, exist_ok=True)
        token_path.write_text(creds.to_json())
    return build("drive", "v3", credentials=creds, cache_discovery=False)


def find_subfolder(drive, parent_id, name):
    q = (f"name='{name}' and '{parent_id}' in parents "
         f"and mimeType='application/vnd.google-apps.folder' and trashed=false")
    found = drive.files().list(q=q, fields="files(id)").execute().get("files", [])
    return found[0]["id"] if found else None


def list_children(drive, parent_id, only_folders=False):
    """All non-trashed children of a folder (paginated)."""
    out, page_token = [], None
    mime = " and mimeType='application/vnd.google-apps.folder'" if only_folders else ""
    while True:
        resp = drive.files().list(
            q=f"'{parent_id}' in parents and trashed=false{mime}",
            fields="nextPageToken, files(id,name,mimeType,modifiedTime)",
            pageSize=1000, pageToken=page_token,
        ).execute()
        out.extend(resp.get("files", []))
        page_token = resp.get("nextPageToken")
        if not page_token:
            break
    return out


def mark_expired_rows(drive, repo_id, deleted_ids: set[str]) -> None:
    """Pending queue rows whose PDF was just deleted -> status='expired'.
    Shared-parquet write — taken under the same _extract.lock the extractors
    use (CLAUDE.md global rule #4). Fail-soft: a missed marking only means the
    extractor logs that row as Skipped (download fail) until the next pass."""
    if not deleted_ids:
        return
    index_id = find_subfolder(drive, repo_id, "_index")
    if not index_id:
        return
    try:
        import io
        import sys
        import pandas as pd
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from _extractor_base import (acquire_lock, release_lock, find_file,
                                     download_bytes, save_parquet)
        if not acquire_lock(drive, index_id, "_extract.lock", "cleanup"):
            log("  queue busy (_extract.lock) — expired-marking deferred to next run.")
            return
        try:
            # FULL frame read (load_parquet would subset columns and the save
            # below would then truncate the queue schema).
            qfid = find_file(drive, index_id, "processing_queue.parquet")
            q = (pd.read_parquet(io.BytesIO(download_bytes(drive, qfid)))
                 if qfid else pd.DataFrame())
            if q.empty or "drive_file_id" not in q.columns:
                return
            hit = (q["status"].astype(str).eq("pending")
                   & q["drive_file_id"].astype(str).isin(deleted_ids))
            if hit.any():
                q.loc[hit, "status"] = "expired"
                save_parquet(drive, index_id, "processing_queue.parquet", q)
                log(f"  queue: {int(hit.sum())} pending row(s) marked expired "
                    f"(PDF removed by retention).")
        finally:
            release_lock(drive, index_id, "_extract.lock")
    except Exception as e:
        log(f"  expired-marking failed ({str(e)[:100]}) — rows stay pending "
            f"(extractor will log them as Skipped).")


def hours_old(modified_time_iso: str) -> float:
    dt = datetime.fromisoformat(modified_time_iso.replace("Z", "+00:00"))
    return (datetime.now(timezone.utc) - dt).total_seconds() / 3600


# ---------- Main ----------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--retain-days", type=int, default=RETAIN_DAYS)
    parser.add_argument("--dry-run", action="store_true",
                        help="List what would be deleted; delete nothing.")
    args = parser.parse_args()

    print("Phase 2 / Stage A — Storage hygiene")
    print("-" * 56)
    cutoff_h = args.retain_days * 24
    log(f"Raw PDFs: delete older than {args.retain_days}d"
        f"{'  (DRY RUN)' if args.dry_run else ''}")
    log("Daily digests (_daily/*.md): kept forever — not touched.")

    drive = get_drive()
    folder_id = os.environ["GDRIVE_FOLDER_ID"]

    repo_id = find_subfolder(drive, folder_id, "company_repo")
    if not repo_id:
        print("company_repo/ does not exist yet — nothing to clean.")
        return

    # HARD RETENTION, DRIVE-WIDE (user 2026-06-12): walk the ENTIRE project
    # tree; age alone decides; only PDFs are candidates. Rows whose PDF dies
    # are marked 'expired' afterwards.
    scanned = deleted = kept = errors = folders = 0
    deleted_ids: set[str] = set()
    to_visit = [(folder_id, "")]            # (drive folder id, display path)
    while to_visit:
        fid, path = to_visit.pop()
        folders += 1
        for f in list_children(drive, fid):
            if f.get("mimeType") == "application/vnd.google-apps.folder":
                to_visit.append((f["id"], f"{path}{f['name']}/"))
                continue
            is_pdf = (f.get("mimeType") == "application/pdf"
                      or str(f.get("name", "")).lower().endswith(".pdf"))
            if not is_pdf:
                continue
            scanned += 1
            try:
                age_h = hours_old(f["modifiedTime"])
            except Exception:
                continue
            if age_h <= cutoff_h:
                kept += 1
                continue
            if args.dry_run:
                log(f"  would delete: {path}{f['name']} ({age_h/24:.0f}d old)")
                deleted += 1
                continue
            try:
                # permanent delete (frees quota immediately)
                drive.files().delete(fileId=f["id"]).execute()
                deleted += 1
                deleted_ids.add(f["id"])
            except Exception as e:
                errors += 1
                log(f"  ERROR deleting {path}{f['name']}: {str(e)[:100]}")
    log(f"Walked {folders} folders (entire Drive tree under project root).")

    if not args.dry_run:
        mark_expired_rows(drive, repo_id, deleted_ids)

    print("-" * 56)
    print(f"Raw docs scanned : {scanned}")
    print(f"Kept (<= {args.retain_days}d) : {kept}")
    print(f"{'Would delete' if args.dry_run else 'Deleted (trashed)'} : {deleted}")
    if errors:
        print(f"Errors           : {errors}")


if __name__ == "__main__":
    main()
