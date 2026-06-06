"""
upload_pf_to_drive.py — Upload a Holdings Statement Excel to Drive pf_tracking/ folder.

Usage:
    python scripts/upload_pf_to_drive.py               # auto-find latest in D:\Downloads
    python scripts/upload_pf_to_drive.py "C:\path\to\file.xlsx"
"""
from __future__ import annotations

import io
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseUpload

ROOT   = Path(__file__).resolve().parent.parent
SCOPES = ["https://www.googleapis.com/auth/drive"]
XLSX_MIME = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
DOWNLOAD_DIR = Path(r"C:\Users\vaido\Downloads")

load_dotenv(ROOT / ".env")


# ── Drive auth (same pattern as fetch_pf_tracking.py) ─────────────────────────

def get_drive():
    tk_json = os.environ.get("GDRIVE_OAUTH_TOKEN_JSON")
    cs_json = os.environ.get("GDRIVE_OAUTH_CLIENT_SECRET_JSON")
    if tk_json and cs_json:
        creds = Credentials.from_authorized_user_info(json.loads(tk_json), SCOPES)
        if creds.expired and creds.refresh_token:
            creds.refresh(Request())
        return build("drive", "v3", credentials=creds, cache_discovery=False)
    tk_path = Path(os.environ["GDRIVE_OAUTH_TOKEN_PATH"])
    creds = None
    if tk_path.exists():
        creds = Credentials.from_authorized_user_file(str(tk_path), SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            from google_auth_oauthlib.flow import InstalledAppFlow
            cs_path = Path(os.environ["GDRIVE_OAUTH_CLIENT_SECRET_PATH"])
            flow = InstalledAppFlow.from_client_secrets_file(str(cs_path), SCOPES)
            creds = flow.run_local_server(port=0)
        tk_path.write_text(creds.to_json())
    return build("drive", "v3", credentials=creds, cache_discovery=False)


# ── Drive helpers ──────────────────────────────────────────────────────────────

def _find_subfolder(drive, parent_id: str, name: str):
    q = (f"name='{name}' and '{parent_id}' in parents "
         f"and mimeType='application/vnd.google-apps.folder' and trashed=false")
    r = drive.files().list(q=q, fields="files(id)").execute().get("files", [])
    return r[0]["id"] if r else None


def _get_or_create_folder(drive, parent_id: str, name: str) -> str:
    existing = _find_subfolder(drive, parent_id, name)
    if existing:
        return existing
    meta = {"name": name, "parents": [parent_id],
            "mimeType": "application/vnd.google-apps.folder"}
    return drive.files().create(body=meta, fields="id").execute()["id"]


def _list_folder_files(drive, folder_id: str) -> list[dict]:
    r = drive.files().list(
        q=f"'{folder_id}' in parents and trashed=false",
        fields="files(id,name,modifiedTime)",
        orderBy="modifiedTime desc",
        pageSize=100,
    ).execute()
    return r.get("files", [])


def _upload_file(drive, folder_id: str, filename: str, data: bytes) -> str:
    media = MediaIoBaseUpload(io.BytesIO(data), mimetype=XLSX_MIME, resumable=False)
    meta = {"name": filename, "parents": [folder_id]}
    f = drive.files().create(body=meta, media_body=media, fields="id").execute()
    return f["id"]


# ── Main ───────────────────────────────────────────────────────────────────────

def find_latest_holdings(folder: Path) -> Path | None:
    candidates = sorted(folder.glob("Holdings Statement*.xls"),
                        key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


def main():
    # ── 1. Resolve local file ──────────────────────────────────────────────────
    if len(sys.argv) > 1:
        local_path = Path(sys.argv[1].strip('"'))
        if not local_path.exists():
            print(f"ERROR: File not found: {local_path}")
            sys.exit(1)
        print(f"File : {local_path.name}")
    else:
        local_path = find_latest_holdings(DOWNLOAD_DIR)
        if not local_path:
            print(f"ERROR: No 'Holdings Statement*.xlsx' found in {DOWNLOAD_DIR}")
            sys.exit(1)
        print(f"Found: {local_path.name}")
        ans = input("Upload this file? [y/N] ").strip().lower()
        if ans != "y":
            print("Aborted.")
            sys.exit(0)

    # ── 2. Connect to Drive ────────────────────────────────────────────────────
    print("\nConnecting to Drive…")
    drive = get_drive()
    folder_id = os.environ["GDRIVE_FOLDER_ID"]
    pf_folder_id = _get_or_create_folder(drive, folder_id, "pf_tracking")

    # ── 3. List existing files ─────────────────────────────────────────────────
    existing = _list_folder_files(drive, pf_folder_id)
    if existing:
        print(f"\nFiles already in Drive pf_tracking/ ({len(existing)} total):")
        for i, f in enumerate(existing, 1):
            print(f"  {i}. {f['name']}")
        if len(existing) >= 1:
            ans = input("\nDelete old versions? Enter numbers (e.g. 1,2) or press Enter to skip: ").strip()
            if ans:
                to_delete = []
                for part in ans.split(","):
                    part = part.strip()
                    if part.isdigit():
                        idx = int(part) - 1
                        if 0 <= idx < len(existing):
                            to_delete.append(existing[idx])
                if to_delete:
                    for f in to_delete:
                        drive.files().delete(fileId=f["id"]).execute()
                        print(f"  Deleted: {f['name']}")

    # ── 4. Upload ──────────────────────────────────────────────────────────────
    print(f"\nUploading {local_path.name}…")
    data = local_path.read_bytes()
    file_id = _upload_file(drive, pf_folder_id, local_path.name, data)
    print(f"Done. Drive file ID: {file_id}")
    print(f"Uploaded to: pf_tracking/{local_path.name}")


if __name__ == "__main__":
    main()
