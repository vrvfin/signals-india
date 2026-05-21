"""
Stage cleanup — Fetch NIFTY MIDCAP 100 + NIFTY SMALLCAP 100 via NSE directly.

yfinance tickers ^CNXMIDCAP / ^CNXSC are broken. NSE's own indicesHistory API
works fine. This script backfills (and incrementally updates) those two indices
into data/indices/, matching the schema produced by ingest_indices_macro.py.

Usage:
    python scripts/fix_indices_nse.py             # 10-year backfill or incremental
    python scripts/fix_indices_nse.py --years 5
"""

from __future__ import annotations

import argparse
import io
import os
import time
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload, MediaIoBaseUpload

SCOPES = ["https://www.googleapis.com/auth/drive"]

# Our filename -> NSE's official index name
INDICES = {
    "NIFTY_MIDCAP_100":   "NIFTY MIDCAP 100",
    "NIFTY_SMALLCAP_100": "NIFTY SMALLCAP 100",
}

NSE_HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                   "AppleWebKit/537.36 (KHTML, like Gecko) "
                   "Chrome/120.0 Safari/537.36"),
    "Accept": "application/json, text/javascript, */*; q=0.01",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.nseindia.com/reports-indices-historical-index-data",
}


def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")


# ---------- Drive helpers ----------

def get_drive():
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
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


def get_or_create_subfolder(drive, parent_id, name):
    q = (f"name='{name}' and '{parent_id}' in parents "
         f"and mimeType='application/vnd.google-apps.folder' and trashed=false")
    found = drive.files().list(q=q, fields="files(id)").execute().get("files", [])
    if found:
        return found[0]["id"]
    meta = {"name": name, "parents": [parent_id],
            "mimeType": "application/vnd.google-apps.folder"}
    return drive.files().create(body=meta, fields="id").execute()["id"]


def find_file(drive, folder_id, name):
    q = f"name='{name}' and '{folder_id}' in parents and trashed=false"
    found = drive.files().list(q=q, fields="files(id)").execute().get("files", [])
    return found[0]["id"] if found else None


def download_parquet(drive, file_id):
    request = drive.files().get_media(fileId=file_id)
    fh = io.BytesIO()
    d = MediaIoBaseDownload(fh, request)
    done = False
    while not done:
        _, done = d.next_chunk()
    fh.seek(0)
    return pd.read_parquet(fh)


def upload_parquet(drive, folder_id, filename, df, existing_id=None):
    buf = io.BytesIO()
    df.to_parquet(buf, index=False)
    buf.seek(0)
    media = MediaIoBaseUpload(buf, mimetype="application/octet-stream", resumable=False)
    if existing_id:
        drive.files().update(fileId=existing_id, media_body=media).execute()
        return existing_id
    meta = {"name": filename, "parents": [folder_id]}
    return drive.files().create(body=meta, media_body=media, fields="id").execute()["id"]


# ---------- NSE index history ----------

def nse_session() -> requests.Session:
    s = requests.Session()
    s.headers.update(NSE_HEADERS)
    try:
        s.get("https://www.nseindia.com/", timeout=10)
        time.sleep(0.5)
        s.get("https://www.nseindia.com/reports-indices-historical-index-data",
              timeout=10)
        time.sleep(0.5)
    except Exception:
        pass
    return s


def fetch_chunk(session, index_name, from_d, to_d):
    """NSE indicesHistory API. Returns list of record dicts."""
    url = "https://www.nseindia.com/api/historical/indicesHistory"
    params = {
        "indexType": index_name,
        "from": from_d.strftime("%d-%m-%Y"),
        "to": to_d.strftime("%d-%m-%Y"),
    }
    for attempt in range(3):
        try:
            r = session.get(url, params=params, timeout=30)
            if r.status_code != 200:
                time.sleep(2)
                continue
            data = r.json()
            return data.get("data", {}).get("indexCloseOnlineRecords", [])
        except Exception:
            time.sleep(2)
    return []


def parse_records(records) -> pd.DataFrame:
    """NSE record fields -> our (date, open, high, low, close, volume) schema."""
    rows = []
    for rec in records:
        ts = (rec.get("EOD_TIMESTAMP") or rec.get("TIMESTAMP")
              or rec.get("HistoricalDate"))
        d = pd.to_datetime(ts, errors="coerce", dayfirst=True)
        rows.append({
            "date": d,
            "open": rec.get("EOD_OPEN_INDEX_VAL"),
            "high": rec.get("EOD_HIGH_INDEX_VAL"),
            "low": rec.get("EOD_LOW_INDEX_VAL"),
            "close": rec.get("EOD_CLOSE_INDEX_VAL"),
            "volume": 0,   # indices have no volume; downstream code only uses close
        })
    df = pd.DataFrame(rows).dropna(subset=["date", "close"])
    if df.empty:
        return df
    for c in ["open", "high", "low", "close"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    return df.dropna(subset=["close"]).sort_values("date").reset_index(drop=True)


def fetch_full_history(session, index_name, start: date, end: date) -> pd.DataFrame:
    """Fetch year-by-year (NSE caps each call at ~1 year)."""
    frames = []
    cur = start
    while cur < end:
        chunk_end = min(cur + timedelta(days=360), end)
        recs = fetch_chunk(session, index_name, cur, chunk_end)
        if recs:
            frames.append(parse_records(recs))
        log(f"    {index_name}: {cur} -> {chunk_end}  ({len(recs)} records)")
        time.sleep(1.0)
        cur = chunk_end + timedelta(days=1)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    return out.drop_duplicates(subset=["date"]).sort_values("date").reset_index(drop=True)


# ---------- Main ----------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--years", type=int, default=10)
    args = parser.parse_args()

    print("Cleanup — NSE Midcap/Smallcap index fetch")
    print("-" * 50)

    today = date.today()
    backfill_start = today - timedelta(days=365 * args.years)

    drive = get_drive()
    folder_id = os.environ["GDRIVE_FOLDER_ID"]
    data_id = get_or_create_subfolder(drive, folder_id, "data")
    indices_id = get_or_create_subfolder(drive, data_id, "indices")

    session = nse_session()

    for our_name, nse_name in INDICES.items():
        filename = f"{our_name}.parquet"
        existing_id = find_file(drive, indices_id, filename)

        # Incremental: if a non-trivial parquet exists, only fetch recent days
        start = backfill_start
        existing_df = None
        if existing_id:
            try:
                existing_df = download_parquet(drive, existing_id)
                if len(existing_df) > 5:
                    last = pd.to_datetime(existing_df["date"]).max().date()
                    start = last + timedelta(days=1)
            except Exception:
                existing_df = None

        if start > today:
            log(f"{our_name}: already up to date.")
            continue

        log(f"{our_name} ('{nse_name}'): fetching {start} -> {today}")
        new_df = fetch_full_history(session, nse_name, start, today)
        if new_df.empty:
            log(f"{our_name}: NSE returned no data — skipping (will retry next run).")
            continue

        if existing_df is not None and len(existing_df) > 5:
            merged = pd.concat([existing_df, new_df], ignore_index=True)
            merged = (merged.drop_duplicates(subset=["date"], keep="last")
                      .sort_values("date").reset_index(drop=True))
        else:
            merged = new_df

        upload_parquet(drive, indices_id, filename, merged, existing_id)
        log(f"{our_name}: wrote {len(merged)} rows -> data/indices/{filename}")

    print("-" * 50)
    print("Done.")


if __name__ == "__main__":
    main()
