"""
Quarterly Concall Gap Filler — Part B: Intelligent Queue Merger

Loads the gaps registry (from find_quarterly_gaps.py) and intelligently merges
new gaps into processing_queue.parquet. Respects:
  - Gemini quota headroom (keeps pending queue ≤ 75 for safe processing)
  - Queue history (dedup: won't re-add done/error items)
  - 2-day PDF storage constraint

Safety: Loads ENTIRE queue (all statuses) for dedup — prevents re-adding done items.

Usage:
    python scripts/merge_quarterly_gaps_to_queue.py              # auto-detect quarter, smart merge
    python scripts/merge_quarterly_gaps_to_queue.py --dry-run    # no queue update
    python scripts/merge_quarterly_gaps_to_queue.py --force      # skip queue depth check
"""

from __future__ import annotations

import argparse
import hashlib
import io
import os
import sys
from datetime import date, datetime
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload, MediaIoBaseUpload

SCOPES = ["https://www.googleapis.com/auth/drive"]

# Quarter detection: full 3-month windows per India FY
QUARTER_MAP = {
    (4, 5, 6):    ("Q4FY", (4, 1), (6, 30)),
    (7, 8, 9):    ("Q1FY", (7, 1), (9, 30)),
    (10, 11, 12): ("Q2FY", (10, 1), (12, 31)),
    (1, 2, 3):    ("Q3FY", (1, 1), (3, 31)),
}

# Queue schema
QUEUE_COLS = [
    "doc_id", "key", "isin", "symbol", "company_name", "doc_type",
    "title", "description", "announcement_date", "pdf_url",
    "drive_file_id", "status", "discovered_at", "processed_at"
]


def get_current_quarter_spec() -> dict | None:
    """Auto-detect current quarter. Returns {label, start, end} or None."""
    today = date.today()
    month = today.month

    for months, (label, (s_m, s_d), (e_m, e_d)) in QUARTER_MAP.items():
        if month in months:
            if month >= 4:
                start = date(today.year, s_m, s_d)
                end = date(today.year, e_m, e_d)
            else:
                start = date(today.year - 1, s_m, s_d)
                end = date(today.year, e_m, e_d)
            return {"label": label, "start": start, "end": end}

    return None


def get_drive_service():
    """Authenticate and return Google Drive service."""
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")

    creds = None
    token_file = Path(".gdrive_token.json")
    if token_file.exists():
        creds = Credentials.from_authorized_user_file(str(token_file), SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            raise RuntimeError("GDRIVE auth not set up")

    return build("drive", "v3", credentials=creds)


def find_file_by_name(drive, parent_id: str, filename: str) -> str | None:
    """Find a file by name in a Drive folder."""
    try:
        query = f"name='{filename}' and '{parent_id}' in parents and trashed=false"
        results = drive.files().list(q=query, spaces="drive", fields="files(id)", pageSize=1).execute()
        files = results.get("files", [])
        return files[0]["id"] if files else None
    except Exception as e:
        print(f"  [ERROR] find_file_by_name: {e}")
        return None


def load_parquet_from_drive(drive, folder_id: str, filename: str) -> pd.DataFrame:
    """Load a parquet file from Drive."""
    try:
        fid = find_file_by_name(drive, folder_id, filename)
        if not fid:
            print(f"  [WARN] {filename} not found on Drive")
            return pd.DataFrame()

        request = drive.files().get_media(fileId=fid)
        fh = io.BytesIO()
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        while not done:
            _, done = downloader.next_chunk()
        fh.seek(0)
        return pd.read_parquet(fh)
    except Exception as e:
        print(f"  [ERROR] load_parquet_from_drive: {e}")
        return pd.DataFrame()


def save_parquet_to_drive(drive, folder_id: str, df: pd.DataFrame, filename: str):
    """Save a parquet file to Drive (create or update)."""
    try:
        buf = io.BytesIO()
        df.to_parquet(buf, index=False)
        buf.seek(0)

        existing_id = find_file_by_name(drive, folder_id, filename)
        media = MediaIoBaseUpload(buf, mimetype='application/octet-stream', resumable=True)

        if existing_id:
            drive.files().update(fileId=existing_id, media_body=media).execute()
            print(f"  [OK] Updated {filename}")
        else:
            file_metadata = {'name': filename, 'parents': [folder_id]}
            drive.files().create(body=file_metadata, media_body=media).execute()
            print(f"  [OK] Created {filename}")

    except Exception as e:
        print(f"  [ERROR] save_parquet_to_drive: {e}")


def generate_doc_id(url: str) -> str:
    """Generate stable doc_id from URL."""
    # Try to extract Screener Pname parameter
    if 'Pname=' in url:
        match = url.split('Pname=')[1].split('&')[0]
        return match.replace('.pdf', '')

    # Otherwise, hash the URL
    return hashlib.md5(url.encode()).hexdigest()[:20]


def merge_gaps_to_queue(gaps_df: pd.DataFrame, queue_df: pd.DataFrame, dry_run: bool = False) -> tuple[int, int]:
    """
    Merge new gaps into queue.
    Returns: (num_pushed, num_remaining)
    """
    if gaps_df.empty:
        print("  [INFO] No gaps to merge")
        return 0, 0

    # Build dedup key set from queue: (isin, announcement_date)
    # CRITICAL: Load ALL queue rows (all statuses) to prevent re-adding done/error items
    queue_concalls = queue_df[queue_df.get('doc_type') == 'concall'] if not queue_df.empty else pd.DataFrame()

    if queue_concalls.empty:
        queue_keys = set()
    else:
        queue_keys = set(zip(
            queue_concalls.get('isin', pd.Series()).astype(str),
            queue_concalls.get('announcement_date', pd.Series()).astype(str).str[:10]
        ))

    # Find new gaps not in queue
    new_gaps = gaps_df[~gaps_df.apply(
        lambda r: (str(r.get('isin', '')), str(r.get('announcement_date', ''))[:10]) in queue_keys,
        axis=1
    )]

    print(f"  [DEBUG] Queue keys: {len(queue_keys)}")
    print(f"  [DEBUG] New gaps: {len(new_gaps)}")

    if new_gaps.empty:
        print("  [INFO] All gaps already in queue")
        return 0, 0

    # Check Gemini quota headroom
    pending_concalls = len(queue_df[queue_df.get('status') == 'pending']) if not queue_df.empty else 0
    print(f"  [DEBUG] Current pending: {pending_concalls}")

    # Smart quota safeguard
    if pending_concalls >= 150:
        print(f"  [WARN] Queue backlogged ({pending_concalls} >= 150 pending)")
        print(f"        Gaps dormant; will retry next weekend")
        return 0, len(new_gaps)

    # Decide batch size
    if pending_concalls < 20:
        available = 75 - pending_concalls
        to_push = min(available, len(new_gaps))
    else:
        print(f"  [INFO] Queue at {pending_concalls}; skipping push this cycle")
        return 0, len(new_gaps)

    print(f"  [DEBUG] Available slots: {available}, will push: {to_push}")

    if to_push <= 0:
        print(f"  [INFO] No available slots")
        return 0, len(new_gaps)

    # Build queue rows
    push_gaps = new_gaps.iloc[:to_push]
    new_rows = []

    for _, gap in push_gaps.iterrows():
        doc_id = generate_doc_id(gap['concall_url'])
        row = {
            'doc_id': doc_id,
            'key': gap['isin'],
            'isin': gap['isin'],
            'symbol': gap['symbol'],
            'company_name': gap['company_name'],
            'doc_type': 'concall',
            'title': f"Concall — {gap['company_name']}",
            'description': "Analyst concall (backfilled, quarterly gap filler)",
            'announcement_date': gap['announcement_date'],
            'pdf_url': gap['concall_url'],
            'drive_file_id': None,
            'status': 'pending',
            'discovered_at': datetime.now(),
            'processed_at': None,
        }
        new_rows.append(row)

    # Append to queue
    new_rows_df = pd.DataFrame(new_rows)
    updated_queue = pd.concat([queue_df, new_rows_df], ignore_index=True)

    # Ensure all columns exist
    for col in QUEUE_COLS:
        if col not in updated_queue.columns:
            updated_queue[col] = None

    if not dry_run:
        return to_push, len(new_gaps) - to_push, updated_queue
    else:
        print(f"  [DEBUG] DRY-RUN: would add {to_push} rows")
        return to_push, len(new_gaps) - to_push, None


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--dry-run', action='store_true', help='Don\'t update queue')
    parser.add_argument('--force', action='store_true', help='Skip queue depth check')
    args = parser.parse_args()

    # Get quarter spec
    quarter_spec = get_current_quarter_spec()
    if not quarter_spec:
        print("[INFO] No active quarter detected; exiting")
        return

    print(f"\n[DEBUG] Quarter: {quarter_spec['label']}")
    print(f"[DEBUG] Window: {quarter_spec['start']} – {quarter_spec['end']}")

    # Setup
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
    gdrive_folder_id = os.environ.get('GDRIVE_FOLDER_ID', '')

    if not gdrive_folder_id:
        print("[ERROR] GDRIVE_FOLDER_ID not set")
        return

    try:
        drive = get_drive_service()
    except Exception as e:
        print(f"[ERROR] Drive auth failed: {e}")
        return

    # Load gaps & queue
    print(f"\n[...] Loading gaps registry...")
    gaps_filename = f"concall_gaps_{quarter_spec['label']}.parquet"
    gaps_df = load_parquet_from_drive(drive, gdrive_folder_id, gaps_filename)

    if gaps_df.empty:
        print(f"[INFO] No gaps found in {gaps_filename}; exiting")
        return

    print(f"[OK] Loaded {len(gaps_df)} gaps")

    print(f"[...] Loading processing_queue.parquet...")
    queue_df = load_parquet_from_drive(drive, gdrive_folder_id, "processing_queue.parquet")
    print(f"[OK] Loaded {len(queue_df)} queue rows")

    # Filter PDF only (audio already skipped in find_quarterly_gaps)
    pdf_gaps = gaps_df[gaps_df.get('content_type') == 'pdf']
    print(f"[DEBUG] PDF gaps: {len(pdf_gaps)}")

    # Merge
    print(f"\n[...] Merging gaps into queue...")
    result = merge_gaps_to_queue(pdf_gaps, queue_df, dry_run=args.dry_run)

    if len(result) == 3:
        to_push, remaining, updated_queue = result
    else:
        to_push, remaining = result
        updated_queue = None

    print(f"\n[SUMMARY]")
    print(f"  Pushed to queue: {to_push}")
    print(f"  Remaining unpushed: {remaining}")
    print(f"  New queue depth: {len(queue_df) + to_push}")

    # Save updated queue
    if updated_queue is not None and not args.dry_run:
        print(f"\n[...] Saving updated queue...")
        save_parquet_to_drive(drive, gdrive_folder_id, updated_queue, "processing_queue.parquet")

    print(f"\n[DONE] Merge complete")


if __name__ == '__main__':
    main()
