"""
fetch_quarterly_guidance.py — Download quarterly guidance .md files from Drive
and open in Obsidian. Each file contains ALL companies' Table A + GF summaries
for that quarter.

Usage (interactive):
    python scripts/fetch_quarterly_guidance.py

Usage (non-interactive):
    python scripts/fetch_quarterly_guidance.py --quarter Q4FY26
    python scripts/fetch_quarterly_guidance.py --all
"""

from __future__ import annotations

import argparse
import io
import os
import subprocess
import sys
import time
import urllib.parse
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

from _md_utils import fix_markdown_for_obsidian

OUTPUT_DIR = Path(r"D:\EMA_Screener\Reports\signals-india\quarterly_guidance")
SCOPES = ["https://www.googleapis.com/auth/drive"]


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")


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


def find_subfolder(drive, parent_id: str, name: str) -> str | None:
    q = (f"name='{name}' and '{parent_id}' in parents "
         f"and mimeType='application/vnd.google-apps.folder' and trashed=false")
    files = drive.files().list(q=q, fields="files(id)").execute().get("files", [])
    return files[0]["id"] if files else None


def list_quarterly_files(drive, folder_id: str) -> list[dict]:
    files = drive.files().list(
        q=f"'{folder_id}' in parents and trashed=false",
        fields="files(id, name, modifiedTime)",
        orderBy="name desc",
        pageSize=200,
    ).execute().get("files", [])
    return [f for f in files if f["name"].endswith(".md")]


def download_bytes(drive, file_id: str) -> bytes:
    request = drive.files().get_media(fileId=file_id)
    fh = io.BytesIO()
    dl = MediaIoBaseDownload(fh, request)
    done = False
    while not done:
        _, done = dl.next_chunk()
    return fh.getvalue()


def open_in_obsidian(path: Path) -> None:
    uri = "obsidian://open?path=" + urllib.parse.quote(
        str(path).replace("\\", "/"), safe=":/"
    )
    subprocess.run(["cmd", "/c", "start", "", uri], shell=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--quarter", help="Quarter to download e.g. Q4FY26")
    parser.add_argument("--all",     action="store_true", help="Download all quarters")
    parser.add_argument("--no-open", action="store_true", help="Download only, don't open")
    args = parser.parse_args()

    drive     = get_drive()
    folder_id = os.environ["GDRIVE_FOLDER_ID"]

    repo_id = find_subfolder(drive, folder_id, "company_repo")
    if not repo_id:
        log("ERROR: company_repo not found.")
        sys.exit(1)
    qrt_id = find_subfolder(drive, repo_id, "_quarterly")
    if not qrt_id:
        log("No _quarterly folder on Drive yet — run Phase 2 during results season.")
        sys.exit(0)

    all_files = list_quarterly_files(drive, qrt_id)
    if not all_files:
        log("No quarterly guidance files found yet.")
        sys.exit(0)

    print(f"\n  Available quarterly guidance files ({len(all_files)}):")
    for i, f in enumerate(all_files, 1):
        mod = f["modifiedTime"][:10]
        print(f"    [{i:2d}]  {f['name']:<35}  (updated {mod})")

    # Pick targets
    if args.all:
        targets = all_files
    elif args.quarter:
        needle = args.quarter.upper()
        targets = [f for f in all_files if needle in f["name"].upper()]
        if not targets:
            log(f"No file found matching '{args.quarter}'.")
            sys.exit(1)
    else:
        print()
        raw = input("  Enter number(s) to download (e.g. 1  or  1,2,3  or  all): ").strip().lower()
        if raw == "all":
            targets = all_files
        else:
            chosen = []
            for part in raw.replace(" ", "").split(","):
                if part.isdigit():
                    idx = int(part) - 1
                    if 0 <= idx < len(all_files):
                        chosen.append(all_files[idx])
            targets = chosen
        if not targets:
            log("Nothing selected.")
            sys.exit(0)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    downloaded: list[Path] = []

    for f in targets:
        out_path = OUTPUT_DIR / f["name"]
        if out_path.exists():
            log(f"Cached — re-fixing: {f['name']}")
            raw_text = out_path.read_text(encoding="utf-8", errors="replace")
        else:
            log(f"Downloading {f['name']}…")
            raw_bytes = download_bytes(drive, f["id"])
            raw_text  = raw_bytes.decode("utf-8", errors="replace")

        fixed = fix_markdown_for_obsidian(raw_text)
        out_path.write_text(fixed, encoding="utf-8")
        downloaded.append(out_path)
        log(f"  Saved: {out_path}")

    print(f"\n  Downloaded {len(downloaded)} file(s) to:\n    {OUTPUT_DIR}\n")

    if not args.no_open:
        log(f"Opening in Obsidian…")
        for i, p in enumerate(downloaded):
            open_in_obsidian(p)
            if i < len(downloaded) - 1:
                time.sleep(0.6)


if __name__ == "__main__":
    main()
