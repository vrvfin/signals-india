"""
fetch_concalls_range.py — Download all daily concall digests between two dates,
fix Obsidian table formatting, and open each one in Obsidian.

Usage:
    python scripts/fetch_concalls_range.py --from 20may2026 --to 29may2026
    python scripts/fetch_concalls_range.py --from 20may2026          # from date to today
    python scripts/fetch_concalls_range.py --last 7                  # last 7 days

Date formats accepted: 29may2026 / 29-may-2026 / 2026-05-29 / 29/05/2026
"""

from __future__ import annotations

import argparse
import io
import os
import subprocess
import sys
import time
import urllib.parse
from datetime import datetime, timedelta
from pathlib import Path

from dotenv import load_dotenv
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

from _md_utils import fix_markdown_for_obsidian

# ── CONFIG ────────────────────────────────────────────────────────────────────
OUTPUT_DIR = Path(r"D:\EMA_Screener\Reports\signals-india\concalls")
# ─────────────────────────────────────────────────────────────────────────────

SCOPES = ["https://www.googleapis.com/auth/drive"]


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")


# ---------- Drive auth ----------

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


def list_daily_files(drive, daily_folder_id: str) -> list[dict]:
    files = drive.files().list(
        q=f"'{daily_folder_id}' in parents and trashed=false",
        fields="files(id, name, modifiedTime)",
        orderBy="modifiedTime desc",
        pageSize=500,
    ).execute().get("files", [])
    return [f for f in files if f["name"].endswith(".md")]


def download_file(drive, file_id: str) -> bytes:
    request = drive.files().get_media(fileId=file_id)
    fh = io.BytesIO()
    dl = MediaIoBaseDownload(fh, request)
    done = False
    while not done:
        _, done = dl.next_chunk()
    return fh.getvalue()


# ---------- Date parsing ----------

_MONTH_MAP = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}


def parse_date(s: str) -> datetime | None:
    """Parse flexible date strings into datetime. Returns None on failure."""
    s = s.strip().lower().replace("-", "").replace("/", "").replace(" ", "")
    # Try YYYYMMDD
    if len(s) == 8 and s.isdigit():
        try:
            return datetime.strptime(s, "%Y%m%d")
        except ValueError:
            pass
    # Try DDmonYYYY e.g. 29may2026
    for mon, num in _MONTH_MAP.items():
        if mon in s:
            digits = s.replace(mon, "")
            if len(digits) == 6:  # DDYYYY
                day, year = digits[:2], digits[2:]
            elif len(digits) == 8:  # DDMMYYYY fallback
                day, year = digits[:2], digits[4:]
            else:
                continue
            try:
                return datetime(int(year), num, int(day))
            except ValueError:
                continue
    return None


def filename_to_date(name: str) -> datetime | None:
    """Extract date from concall_DD_MMMYYYY.md filename."""
    stem = name.replace(".md", "").replace("concall_", "")
    return parse_date(stem)


# ---------- Open in Obsidian ----------

def open_in_obsidian(path: Path) -> None:
    uri = "obsidian://open?path=" + urllib.parse.quote(
        str(path).replace("\\", "/"), safe=":/"
    )
    subprocess.run(["cmd", "/c", "start", "", uri], shell=False)


# ---------- Main ----------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--from", dest="date_from",
                        help="Start date e.g. 20may2026 or 2026-05-20")
    parser.add_argument("--to",   dest="date_to",
                        help="End date (inclusive). Defaults to today.")
    parser.add_argument("--last", type=int,
                        help="Last N days (alternative to --from/--to)")
    parser.add_argument("--no-open", action="store_true",
                        help="Download only, do not open in Obsidian")
    args = parser.parse_args()

    today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)

    # Resolve date range
    if args.last:
        date_from = today - timedelta(days=args.last - 1)
        date_to   = today
    elif args.date_from:
        date_from = parse_date(args.date_from)
        if not date_from:
            log(f"ERROR: Could not parse --from date: {args.date_from}")
            sys.exit(1)
        date_to = parse_date(args.date_to) if args.date_to else today
        if not date_to:
            log(f"ERROR: Could not parse --to date: {args.date_to}")
            sys.exit(1)
    else:
        parser.print_help()
        sys.exit(0)

    log(f"Date range: {date_from.strftime('%d %b %Y')} → {date_to.strftime('%d %b %Y')}")

    log("Connecting to Drive…")
    drive = get_drive()
    folder_id = os.environ["GDRIVE_FOLDER_ID"]

    repo_id = find_subfolder(drive, folder_id, "company_repo")
    if not repo_id:
        log("ERROR: company_repo not found on Drive.")
        sys.exit(1)
    daily_id = find_subfolder(drive, repo_id, "_daily")
    if not daily_id:
        log("ERROR: company_repo/_daily not found on Drive.")
        sys.exit(1)

    log("Fetching file list…")
    all_files = list_daily_files(drive, daily_id)

    # Filter by date range
    targets = []
    for f in all_files:
        fdate = filename_to_date(f["name"])
        if fdate and date_from <= fdate <= date_to:
            targets.append((fdate, f))

    targets.sort(key=lambda x: x[0])   # oldest first

    if not targets:
        log(f"No files found in range {date_from.date()} – {date_to.date()}.")
        log("Available files (last 10):")
        for f in all_files[:10]:
            print(f"  {f['name']}")
        sys.exit(0)

    log(f"Found {len(targets)} file(s) in range:")
    for fdate, f in targets:
        print(f"  {f['name']}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    downloaded: list[Path] = []

    for fdate, f in targets:
        out_path = OUTPUT_DIR / f["name"]
        if out_path.exists():
            log(f"Cached — re-fixing: {f['name']}")
            raw_text = out_path.read_text(encoding="utf-8", errors="replace")
        else:
            log(f"Downloading: {f['name']} …")
            raw_bytes = download_file(drive, f["id"])
            raw_text  = raw_bytes.decode("utf-8", errors="replace")

        fixed = fix_markdown_for_obsidian(raw_text)
        out_path.write_text(fixed, encoding="utf-8")
        downloaded.append(out_path)
        log(f"  Saved: {out_path.name}")

    print(f"\n{'='*50}")
    print(f"Downloaded {len(downloaded)} file(s) to:\n  {OUTPUT_DIR}")
    print('='*50)

    if not args.no_open:
        log(f"Opening {len(downloaded)} file(s) in Obsidian…")
        for i, path in enumerate(downloaded):
            open_in_obsidian(path)
            if i < len(downloaded) - 1:
                time.sleep(0.8)   # small gap so Obsidian can handle each URI


if __name__ == "__main__":
    main()
