"""
Quarterly Concall Gap Filler — Part A: Discovery

Identifies companies with published quarterly concalls (via Screener) that are NOT YET
in the processing_queue. Runs every weekend, auto-detects current quarter, queries the
full 3-month window per quarter. Detects MP3 vs PDF (3-layer: extension, Content-Type, magic bytes).

Output: concall_gaps_{QUARTER_LABEL}.parquet with columns:
  isin, symbol, company_name, announcement_date, concall_url, content_type, source, discovered_at

Usage:
    python scripts/find_quarterly_gaps.py              # auto-detect quarter
    python scripts/find_quarterly_gaps.py --dry-run    # no Drive upload
    python scripts/find_quarterly_gaps.py --quarter Q4FY --start 2026-04-01 --end 2026-06-30
"""

from __future__ import annotations

import argparse
import hashlib
import io
import os
import re
import time
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload, MediaIoBaseUpload

SCOPES = ["https://www.googleapis.com/auth/drive"]

# Screener filter IDs
FILTERS = {
    "concall": "76106",
}

# Quarter detection: full 3-month windows per India FY
QUARTER_MAP = {
    (4, 5, 6):    ("Q4FY", (4, 1), (6, 30)),     # Apr 1 – Jun 30 (Q4: Mar quarter)
    (7, 8, 9):    ("Q1FY", (7, 1), (9, 30)),     # Jul 1 – Sep 30 (Q1: Jun quarter)
    (10, 11, 12): ("Q2FY", (10, 1), (12, 31)),   # Oct 1 – Dec 31 (Q2: Sep quarter)
    (1, 2, 3):    ("Q3FY", (1, 1), (3, 31)),     # Jan 1 – Mar 31 (Q3: Dec quarter)
}


def get_current_quarter_spec() -> dict | None:
    """Auto-detect current quarter from today's date. Returns {label, start, end} or None."""
    today = date.today()
    month = today.month

    for months, (label, (s_m, s_d), (e_m, e_d)) in QUARTER_MAP.items():
        if month in months:
            # Compute start/end dates, handling year boundary
            if month >= 4:
                start = date(today.year, s_m, s_d)
                end = date(today.year, e_m, e_d)
            else:
                # Jan–Mar: year boundary
                start = date(today.year - 1, s_m, s_d)
                end = date(today.year, e_m, e_d)
            return {"label": label, "start": start, "end": end}

    return None


def detect_content_type(url: str, session: requests.Session | None = None) -> str:
    """3-layer detection: extension → HTTP Content-Type → PDF magic bytes. Returns 'pdf' | 'audio'."""
    # Layer 1: URL extension
    url_lower = url.lower()
    if any(url_lower.endswith(ext) for ext in ('.mp3', '.wav', '.m4a', '.aac')):
        return 'audio'
    if url_lower.endswith('.pdf'):
        return 'pdf'

    # Layer 2: HTTP Content-Type header
    if session:
        try:
            r = session.head(url, timeout=5, allow_redirects=True)
            ct = r.headers.get('Content-Type', '').lower()
            if 'pdf' in ct:
                return 'pdf'
            if 'audio' in ct or 'mpeg' in ct:
                return 'audio'
        except Exception:
            pass

    # Layer 3: Download first bytes, check magic
    try:
        if not session:
            session = requests.Session()
        r = session.get(url, timeout=5, stream=True)
        header = r.content[:20] if r.content else b''
        if header.startswith(b'%PDF'):
            return 'pdf'
        if header.startswith(b'RIFF') or header.startswith(b'ID3'):
            return 'audio'
    except Exception:
        pass

    # Default: assume PDF
    return 'pdf'


def get_drive_service():
    """Authenticate and return Google Drive service."""
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
    token_path = Path(os.environ.get("GDRIVE_OAUTH_TOKEN_PATH", ".env"))

    # Simplified auth: expect token to be pre-set
    creds = None
    token_file = Path(".gdrive_token.json")
    if token_file.exists():
        creds = Credentials.from_authorized_user_file(str(token_file), SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            raise RuntimeError("GDRIVE auth not set up. Run auth_helper first.")

    return build("drive", "v3", credentials=creds)


def find_file_by_name(drive, parent_id: str, filename: str) -> str | None:
    """Find a file by name in a Drive folder. Returns file ID or None."""
    try:
        query = f"name='{filename}' and '{parent_id}' in parents and trashed=false"
        results = drive.files().list(q=query, spaces="drive", fields="files(id)", pageSize=1).execute()
        files = results.get("files", [])
        return files[0]["id"] if files else None
    except Exception as e:
        print(f"  [ERROR] find_file_by_name: {e}")
        return None


def load_queue_from_drive(drive, index_folder_id: str) -> pd.DataFrame:
    """Load processing_queue.parquet from Drive."""
    try:
        fid = find_file_by_name(drive, index_folder_id, "processing_queue.parquet")
        if not fid:
            print("  [WARN] processing_queue.parquet not found on Drive")
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
        print(f"  [ERROR] load_queue_from_drive: {e}")
        return pd.DataFrame()


def query_screener_concalls(session: requests.Session, filter_id: str, start_date: date, end_date: date) -> list[dict]:
    """Query Screener's announcement filter for concalls in date range. Returns list of {symbol, date, url}."""
    results = []
    page = 1
    max_pages = 10  # Safety limit

    while page <= max_pages:
        try:
            url = f"https://www.screener.in/announcements/user-filters/{filter_id}/?page={page}"
            r = session.get(url, timeout=10)
            r.raise_for_status()
            soup = BeautifulSoup(r.content, 'html.parser')

            # Find announcement rows
            rows = soup.find_all('tr')
            if not rows:
                print(f"  [DEBUG] Page {page}: no rows found, stopping")
                break

            page_found_any = False
            for row in rows:
                cols = row.find_all('td')
                if len(cols) < 3:
                    continue

                # Extract symbol, announcement date, and link
                symbol_cell = cols[0].get_text(strip=True)
                date_cell = cols[1].get_text(strip=True)
                title_cell = cols[2].get_text(strip=True) if len(cols) > 2 else ""

                # Parse date
                try:
                    ann_date = datetime.strptime(date_cell, "%d %b %Y").date()
                    if not (start_date <= ann_date <= end_date):
                        continue
                except ValueError:
                    continue

                # Find PDF link
                link_tag = cols[2].find('a') if len(cols) > 2 else None
                if not link_tag or not link_tag.get('href'):
                    continue

                pdf_url = link_tag['href']
                if not pdf_url.startswith('http'):
                    pdf_url = "https://www.screener.in" + pdf_url

                # Detect MP3 vs PDF
                content_type = detect_content_type(pdf_url, session)

                results.append({
                    'symbol': symbol_cell.split()[0] if symbol_cell else "",
                    'announcement_date': ann_date,
                    'concall_url': pdf_url,
                    'content_type': content_type,
                    'title': title_cell,
                    'source': 'screener',
                })
                page_found_any = True

            if not page_found_any:
                break

            page += 1
            time.sleep(1)  # Be polite

        except Exception as e:
            print(f"  [ERROR] Page {page}: {e}")
            break

    return results


def build_gaps_registry(screener_results: list[dict], queue_df: pd.DataFrame, company_universe: pd.DataFrame) -> pd.DataFrame:
    """Build gaps registry by comparing screener results against queue."""
    gaps = []

    # Build dedup key set from queue: (symbol, announcement_date) or (isin, announcement_date)
    if queue_df.empty:
        queue_keys = set()
    else:
        queue_concalls = queue_df[queue_df.get('doc_type') == 'concall']
        queue_keys = set(zip(
            queue_concalls.get('symbol', pd.Series()).astype(str),
            queue_concalls.get('announcement_date', pd.Series()).astype(str).str[:10]
        ))

    for item in screener_results:
        symbol = item['symbol']
        ann_date = item['announcement_date']
        dedup_key = (symbol, str(ann_date))

        # Skip if already in queue
        if dedup_key in queue_keys:
            continue

        # Resolve ISIN from company_universe
        isin = ""
        if not company_universe.empty:
            matches = company_universe[company_universe.get('symbol', pd.Series()) == symbol]
            if not matches.empty:
                isin = matches.iloc[0].get('isin', '')

        company_name = ""
        if not company_universe.empty:
            matches = company_universe[company_universe.get('symbol', pd.Series()) == symbol]
            if not matches.empty:
                company_name = matches.iloc[0].get('company_name', '')

        gaps.append({
            'isin': isin,
            'symbol': symbol,
            'company_name': company_name,
            'announcement_date': ann_date,
            'concall_url': item['concall_url'],
            'content_type': item['content_type'],
            'source': item['source'],
            'discovered_at': datetime.now(),
        })

    return pd.DataFrame(gaps)


def save_gaps_to_drive(drive, index_folder_id: str, gaps_df: pd.DataFrame, quarter_label: str):
    """Save gaps registry to Drive."""
    if gaps_df.empty:
        print(f"  [INFO] No gaps found; skipping Drive upload")
        return

    try:
        filename = f"concall_gaps_{quarter_label}.parquet"

        # Serialize to bytes
        buf = io.BytesIO()
        gaps_df.to_parquet(buf, index=False)
        buf.seek(0)

        # Check if file exists
        existing_id = find_file_by_name(drive, index_folder_id, filename)

        media = MediaIoBaseUpload(buf, mimetype='application/octet-stream', resumable=True)

        if existing_id:
            # Update existing
            drive.files().update(fileId=existing_id, media_body=media).execute()
            print(f"  [OK] Updated {filename} on Drive")
        else:
            # Create new
            file_metadata = {'name': filename, 'parents': [index_folder_id]}
            drive.files().create(body=file_metadata, media_body=media).execute()
            print(f"  [OK] Created {filename} on Drive")

    except Exception as e:
        print(f"  [ERROR] save_gaps_to_drive: {e}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--quarter', type=str, default=None, help='Quarter label (Q4FY, Q1FY, etc); auto-detect if None')
    parser.add_argument('--start', type=str, default=None, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, default=None, help='End date (YYYY-MM-DD)')
    parser.add_argument('--dry-run', action='store_true', help='Don\'t upload to Drive')
    args = parser.parse_args()

    # Get quarter spec
    if args.quarter and args.start and args.end:
        quarter_spec = {
            'label': args.quarter,
            'start': datetime.strptime(args.start, '%Y-%m-%d').date(),
            'end': datetime.strptime(args.end, '%Y-%m-%d').date(),
        }
    else:
        quarter_spec = get_current_quarter_spec()

    if not quarter_spec:
        print("[INFO] No active quarter detected; exiting")
        return

    print(f"\n[DEBUG] Quarter: {quarter_spec['label']}")
    print(f"[DEBUG] Window: {quarter_spec['start']} – {quarter_spec['end']}")

    # Setup
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")

    # Screener session
    cookie = os.environ.get('SCREENER_SESSION_COOKIE', '').strip()
    if not cookie:
        print("[ERROR] SCREENER_SESSION_COOKIE not set in .env")
        return

    session = requests.Session()
    session.cookies.set('sessionid', cookie, domain='.screener.in')
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
    })

    # Drive setup
    drive = None
    if not args.dry_run:
        try:
            drive = get_drive_service()
            gdrive_folder_id = os.environ.get('GDRIVE_FOLDER_ID', '')
            if not gdrive_folder_id:
                print("[WARN] GDRIVE_FOLDER_ID not set; skipping Drive upload")
                drive = None
        except Exception as e:
            print(f"[WARN] Drive auth failed: {e}; continuing without Drive")
            drive = None

    # Query Screener
    print(f"\n[...] Querying Screener for concalls ({quarter_spec['start']} – {quarter_spec['end']})...")
    screener_results = query_screener_concalls(
        session,
        FILTERS['concall'],
        quarter_spec['start'],
        quarter_spec['end']
    )
    print(f"[OK] Found {len(screener_results)} concalls on Screener")

    # Load queue & company universe
    queue_df = pd.DataFrame()
    company_universe = pd.DataFrame()

    if drive:
        print(f"[...] Loading processing_queue.parquet from Drive...")
        queue_df = load_queue_from_drive(drive, gdrive_folder_id)
        print(f"[OK] Loaded {len(queue_df)} rows from queue")

    # Build gaps
    print(f"\n[...] Building gaps registry...")
    gaps_df = build_gaps_registry(screener_results, queue_df, company_universe)
    print(f"[OK] Found {len(gaps_df)} gaps (new, not yet in queue)")

    # Filter audio
    if not gaps_df.empty:
        pdf_gaps = gaps_df[gaps_df['content_type'] == 'pdf']
        audio_count = len(gaps_df) - len(pdf_gaps)
        if audio_count > 0:
            print(f"[INFO] Skipped {audio_count} audio files")
    else:
        pdf_gaps = gaps_df

    # Save to Drive
    if drive and not args.dry_run:
        print(f"\n[...] Saving gaps registry to Drive...")
        save_gaps_to_drive(drive, gdrive_folder_id, pdf_gaps, quarter_spec['label'])
    else:
        # Save locally for testing
        local_file = f"concall_gaps_{quarter_spec['label']}.parquet"
        pdf_gaps.to_parquet(local_file, index=False)
        print(f"[OK] Saved gaps locally to {local_file}")

    print(f"\n[DONE] Gap discovery complete")


if __name__ == '__main__':
    main()
