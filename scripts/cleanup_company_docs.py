"""
Phase 2 / Stage A — Storage hygiene.

Deletes raw document PDFs older than RETAIN_DAYS (default 10) from every
company_repo/<ISIN>/documents/ folder. Once a document has been summarised the
raw PDF is no longer needed — the company page and the structured indexes hold
the lasting value.

NEVER touches: _daily/ digest .md files, company_page.md/.docx,
               deep_report.*, summaries, _index/*.

Daily digest files (_daily/*.md) are persisted forever — only raw PDFs are
transient.

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

    company_folders = [f for f in list_children(drive, repo_id, only_folders=True)
                       if f["name"] not in ("_index", "_daily")]
    log(f"Scanning {len(company_folders)} company folders...")

    scanned = deleted = kept = errors = 0
    for cf in company_folders:
        docs_id = find_subfolder(drive, cf["id"], "documents")
        if not docs_id:
            continue
        for f in list_children(drive, docs_id):
            if f.get("mimeType") == "application/vnd.google-apps.folder":
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
                log(f"  would delete: {cf['name']}/documents/{f['name']} "
                    f"({age_h/24:.0f}d old)")
                deleted += 1
                continue
            try:
                # move to trash (parmanetly, frees quota)
                drive.files().delete(fileId=f["id"]).execute()
                deleted += 1
            except Exception as e:
                errors += 1
                log(f"  ERROR deleting {cf['name']}/{f['name']}: {str(e)[:100]}")

    print("-" * 56)
    print(f"Raw docs scanned : {scanned}")
    print(f"Kept (<= {args.retain_days}d) : {kept}")
    print(f"{'Would delete' if args.dry_run else 'Deleted (trashed)'} : {deleted}")
    if errors:
        print(f"Errors           : {errors}")


if __name__ == "__main__":
    main()
