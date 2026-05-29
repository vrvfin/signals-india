"""
fetch_company_intel.py — Download a company's intelligence page (Table A,
GF1-GF4, summaries) from Drive, fix table formatting for Obsidian, and open.

Usage:
    python scripts/fetch_company_intel.py --symbol RELIANCE
    python scripts/fetch_company_intel.py --symbol INE002A01018
    python scripts/fetch_company_intel.py --symbol RELIANCE --no-open

One-time setup: set OUTPUT_DIR below to a folder inside your Obsidian vault.
"""

from __future__ import annotations

import argparse
import io
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

# ── CONFIG ────────────────────────────────────────────────────────────────────
OUTPUT_DIR = Path(r"D:\EMA_Screener\Reports\signals-india\company_intel")
# ─────────────────────────────────────────────────────────────────────────────

SCOPES = ["https://www.googleapis.com/auth/drive"]


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")


# ---------- Drive auth (same pattern as all pipeline scripts) ----------

def get_drive():
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
    tk_json = os.environ.get("GDRIVE_OAUTH_TOKEN_JSON")
    cs_json = os.environ.get("GDRIVE_OAUTH_CLIENT_SECRET_JSON")
    if tk_json and cs_json:
        import json
        creds = Credentials.from_authorized_user_info(json.loads(tk_json), SCOPES)
        if creds.expired and creds.refresh_token:
            creds.refresh(Request())
        return build("drive", "v3", credentials=creds, cache_discovery=False)
    cs_path = Path(os.environ["GDRIVE_OAUTH_CLIENT_SECRET_PATH"])
    tk_path = Path(os.environ["GDRIVE_OAUTH_TOKEN_PATH"])
    creds = None
    if tk_path.exists():
        creds = Credentials.from_authorized_user_file(str(tk_path), SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            from google_auth_oauthlib.flow import InstalledAppFlow
            flow = InstalledAppFlow.from_client_secrets_file(str(cs_path), SCOPES)
            creds = flow.run_local_server(port=0)
        tk_path.write_text(creds.to_json())
    return build("drive", "v3", credentials=creds, cache_discovery=False)


# ---------- Drive helpers ----------

def find_subfolder(drive, parent_id: str, name: str) -> str | None:
    q = (f"name='{name}' and '{parent_id}' in parents "
         f"and mimeType='application/vnd.google-apps.folder' and trashed=false")
    files = drive.files().list(q=q, fields="files(id)").execute().get("files", [])
    return files[0]["id"] if files else None


def find_file(drive, folder_id: str, filename: str) -> str | None:
    q = f"name='{filename}' and '{folder_id}' in parents and trashed=false"
    files = drive.files().list(q=q, fields="files(id)").execute().get("files", [])
    return files[0]["id"] if files else None


def download_file(drive, file_id: str) -> bytes:
    request = drive.files().get_media(fileId=file_id)
    fh = io.BytesIO()
    dl = MediaIoBaseDownload(fh, request)
    done = False
    while not done:
        _, done = dl.next_chunk()
    return fh.getvalue()


from _md_utils import fix_markdown_for_obsidian   # split-row fixer + whitespace + fence unwrap


# ---------- Open ----------

def open_file(path: Path) -> None:
    """Open file in Obsidian using the obsidian:// URI scheme.
    File MUST be inside an Obsidian vault — set OUTPUT_DIR to your vault folder.
    """
    import urllib.parse
    log(f"Opening in Obsidian: {path}")
    uri = "obsidian://open?path=" + urllib.parse.quote(str(path).replace("\\", "/"), safe=":/")
    try:
        subprocess.run(["cmd", "/c", "start", "", uri], shell=False)
    except Exception as e:
        log(f"Could not open ({e}) — path: {path}")


# ---------- Main ----------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--symbol", required=True,
                        help="NSE symbol or ISIN (e.g. RELIANCE or INE002A01018)")
    parser.add_argument("--no-open", action="store_true",
                        help="Download and fix only, don't open")
    args = parser.parse_args()

    key = args.symbol.strip().upper()
    log(f"Fetching company intel for: {key}")

    drive = get_drive()
    folder_id = os.environ["GDRIVE_FOLDER_ID"]

    repo_id = find_subfolder(drive, folder_id, "company_repo")
    if not repo_id:
        log("ERROR: company_repo folder not found on Drive.")
        sys.exit(1)

    # Try direct key folder first (ISIN or symbol)
    comp_id = find_subfolder(drive, repo_id, key)

    # If not found, search by symbol via universe — try common variations
    if not comp_id:
        log(f"  Folder '{key}' not found — searching universe for ISIN match…")
        # list all subfolders and look for matching symbol metadata is hard
        # without universe parquet locally; exit with helpful message
        log(f"  Not found. Try the ISIN instead of symbol, or check company_repo folder name on Drive.")
        log(f"  Hint: folder names in company_repo are typically the ISIN (e.g. INE002A01018)")
        sys.exit(1)

    file_id = find_file(drive, comp_id, "company_page.md")
    if not file_id:
        log(f"  company_page.md not found for {key}.")
        log("  This appears after at least one concall has been processed for this company.")
        sys.exit(1)

    log("Downloading company_page.md…")
    raw = download_file(drive, file_id)
    text = raw.decode("utf-8", errors="replace")
    log(f"  Downloaded ({len(raw)//1024:.0f} KB)")

    fixed = fix_markdown_for_obsidian(text)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / f"{key}_company_page.md"
    out_path.write_text(fixed, encoding="utf-8")
    log(f"Saved (Obsidian-fixed): {out_path}")

    if not args.no_open:
        open_file(out_path)

    print(f"\nFile: {out_path}")


if __name__ == "__main__":
    main()
