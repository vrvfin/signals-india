"""
fetch_latest_concall.py — Download the latest daily concall digest from Drive
and open it in Obsidian (or any registered .md handler).

Usage:
    python scripts/fetch_latest_concall.py               # latest day
    python scripts/fetch_latest_concall.py --date 29may2026   # specific date
    python scripts/fetch_latest_concall.py --list        # show last 10 available
    python scripts/fetch_latest_concall.py --all-today   # all files from today

One-time setup:
    1. Set OUTPUT_DIR below to a folder inside your Obsidian vault.
    2. Run once — file downloads and Obsidian opens it automatically.

Requirements: pip install python-dotenv google-auth google-auth-oauthlib
              google-api-python-client  (all already in scripts/requirements.txt)
"""

from __future__ import annotations

import argparse
import io
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

# ── CONFIG ────────────────────────────────────────────────────────────────────
# Set this to a folder INSIDE your Obsidian vault so the file is indexed.
# Example: r"C:\Users\vaido\Documents\ObsidianVault\signals-india\concalls"
OUTPUT_DIR = Path(r"C:\Users\vaido\Downloads\concalls")
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
        creds = Credentials.from_authorized_user_info(
            json.loads(tk_json), SCOPES)
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


def list_daily_files(drive, daily_folder_id: str, limit: int = 30) -> list[dict]:
    """Return list of .md files in _daily/, newest first."""
    files = drive.files().list(
        q=f"'{daily_folder_id}' in parents and trashed=false",
        fields="files(id, name, modifiedTime)",
        orderBy="modifiedTime desc",
        pageSize=limit,
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


# ---------- Obsidian markdown fixer ----------

def _fix_markdown_for_obsidian(text: str) -> str:
    """Fix common rendering failures in Obsidian for LLM-generated markdown.

    Problems fixed:
    1. Leading whitespace before | rows → Obsidian treats them as code blocks.
    2. No blank line before a table that follows normal text → table not rendered.
    3. Windows \\r\\n or bare \\r line endings → strip to \\n.
    4. Tables wrapped inside triple-backtick fences → unwrap them.
    """
    # Normalise line endings
    text = text.replace('\r\n', '\n').replace('\r', '\n')

    # Unwrap tables that Gemini accidentally put inside code fences.
    # Pattern: ```\n<table lines>\n```
    def _unwrap_table_fence(m):
        inner = m.group(1)
        if any(l.lstrip().startswith('|') for l in inner.splitlines()):
            return inner          # remove the fence; keep the table
        return m.group(0)        # leave other code blocks alone
    text = re.sub(r'```[^\n]*\n(.*?)```', _unwrap_table_fence, text,
                  flags=re.DOTALL)

    lines = text.split('\n')
    out: list[str] = []
    in_fence = False

    for i, line in enumerate(lines):
        # Track code fences so we don't mangle real code blocks
        if re.match(r'^```', line):
            in_fence = not in_fence
            out.append(line)
            continue

        if not in_fence:
            stripped = line.lstrip()
            if stripped.startswith('|'):
                line = stripped   # kill all leading whitespace
                # Insert blank line before this table row if previous
                # non-blank line is NOT also a table row
                if out:
                    last_nonempty = next(
                        (l for l in reversed(out) if l.strip()), None
                    )
                    if (last_nonempty
                            and not last_nonempty.lstrip().startswith('|')
                            and out[-1].strip()):   # no blank already
                        out.append('')

        out.append(line)

    return '\n'.join(out)


# ---------- Open in Obsidian ----------

def open_file(path: Path) -> None:
    """Open a file with the system default handler (Obsidian if .md is registered)."""
    log(f"Opening: {path}")
    try:
        os.startfile(str(path))          # Windows — opens with default .md app
    except AttributeError:
        subprocess.run(["open", str(path)])   # macOS fallback
    except Exception as e:
        log(f"Could not auto-open: {e}")
        log(f"Open manually: {path}")


# ---------- Main ----------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--date",   help="Specific file date, e.g. 29may2026")
    parser.add_argument("--list",   action="store_true", help="List last 10 available files")
    parser.add_argument("--no-open", action="store_true", help="Download only, don't open")
    args = parser.parse_args()

    log("Connecting to Drive…")
    drive = get_drive()
    folder_id = os.environ["GDRIVE_FOLDER_ID"]

    repo_id = find_subfolder(drive, folder_id, "company_repo")
    if not repo_id:
        log("ERROR: company_repo folder not found on Drive.")
        sys.exit(1)

    daily_id = find_subfolder(drive, repo_id, "_daily")
    if not daily_id:
        log("ERROR: company_repo/_daily folder not found on Drive.")
        sys.exit(1)

    files = list_daily_files(drive, daily_id)
    if not files:
        log("No .md files found in _daily/.")
        sys.exit(1)

    # ── --list mode ──
    if args.list:
        log(f"Last {min(10, len(files))} concall digest files:")
        for i, f in enumerate(files[:10]):
            age = (datetime.now(timezone.utc) -
                   datetime.fromisoformat(f["modifiedTime"].replace("Z", "+00:00"))
                   ).total_seconds() / 3600
            print(f"  [{i+1:2d}]  {f['name']}  ({age:.0f}h ago)")
        return

    # ── pick target file ──
    if args.date:
        # match by date fragment anywhere in filename, e.g. "29may2026"
        needle = args.date.lower().replace("-", "").replace("_", "")
        matches = [f for f in files
                   if needle in f["name"].lower().replace("-", "").replace("_", "")]
        if not matches:
            log(f"No file found matching date '{args.date}'.")
            log("Available files:")
            for f in files[:5]:
                print(f"  {f['name']}")
            sys.exit(1)
        target = matches[0]
    else:
        target = files[0]   # newest

    log(f"Selected: {target['name']}")

    # ── download + fix ──
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / target["name"]

    if out_path.exists():
        log(f"Cached locally — re-applying Obsidian table fix: {out_path}")
        raw_text = out_path.read_text(encoding="utf-8", errors="replace")
    else:
        log("Downloading…")
        raw_bytes = download_file(drive, target["id"])
        raw_text   = raw_bytes.decode("utf-8", errors="replace")
        log(f"Downloaded ({len(raw_bytes)//1024:.0f} KB)")

    fixed = _fix_markdown_for_obsidian(raw_text)
    out_path.write_text(fixed, encoding="utf-8")
    log(f"Saved (Obsidian-fixed): {out_path}")

    # ── open ──
    if not args.no_open:
        open_file(out_path)

    print(f"\nFile path (copy for md_viewer.py):\n  {out_path}")


if __name__ == "__main__":
    main()
