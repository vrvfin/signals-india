"""
Stage 2e — Daily FII / DII cash-market flows.

Pulls today's FII (Foreign Institutional Investors) and DII (Domestic Institutional
Investors) net cash-market buy/sell from NSE's public daily report and appends to a
running CSV in Drive at `data/macro/FII_DII.csv`.

NSE doesn't provide a clean historical endpoint, so this script starts tracking from
the first day it's run. Historical backfill (if needed later) is a separate patch.

Usage:
    python scripts/ingest_fii_dii.py
"""

from __future__ import annotations

import io
import os
import sys
import time
from datetime import datetime
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
NSE_FIIDII_URL = "https://www.nseindia.com/api/fiidiiTradeReact"


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")


# ---------- Drive helpers ----------

def get_drive_service():
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
    cs_path = Path(os.environ["GDRIVE_OAUTH_CLIENT_SECRET_PATH"])
    token_path = Path(os.environ["GDRIVE_OAUTH_TOKEN_PATH"])
    creds = None
    if token_path.exists():
        creds = Credentials.from_authorized_user_file(str(token_path), SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(str(cs_path), SCOPES)
            creds = flow.run_local_server(port=0)
        token_path.parent.mkdir(parents=True, exist_ok=True)
        token_path.write_text(creds.to_json())
    return build("drive", "v3", credentials=creds, cache_discovery=False)


def get_or_create_subfolder(drive, parent_id: str, name: str) -> str:
    q = (f"name='{name}' and '{parent_id}' in parents "
         f"and mimeType='application/vnd.google-apps.folder' and trashed=false")
    found = drive.files().list(q=q, fields="files(id,name)").execute().get("files", [])
    if found:
        return found[0]["id"]
    meta = {"name": name, "parents": [parent_id],
            "mimeType": "application/vnd.google-apps.folder"}
    return drive.files().create(body=meta, fields="id").execute()["id"]


def find_file(drive, folder_id: str, name: str) -> str | None:
    q = f"name='{name}' and '{folder_id}' in parents and trashed=false"
    found = drive.files().list(q=q, fields="files(id)").execute().get("files", [])
    return found[0]["id"] if found else None


def download_csv(drive, file_id: str) -> pd.DataFrame:
    request = drive.files().get_media(fileId=file_id)
    fh = io.BytesIO()
    downloader = MediaIoBaseDownload(fh, request)
    done = False
    while not done:
        _, done = downloader.next_chunk()
    fh.seek(0)
    return pd.read_csv(fh)


def upload_csv(drive, folder_id: str, filename: str, df: pd.DataFrame,
               existing_id: str | None) -> str:
    csv_bytes = df.to_csv(index=False).encode()
    media = MediaIoBaseUpload(io.BytesIO(csv_bytes), mimetype="text/csv", resumable=False)
    if existing_id:
        drive.files().update(fileId=existing_id, media_body=media).execute()
        return existing_id
    meta = {"name": filename, "parents": [folder_id]}
    return drive.files().create(body=meta, media_body=media, fields="id").execute()["id"]


# ---------- NSE fetch ----------

def nse_session() -> requests.Session:
    """A session with cookies + headers that pass NSE's anti-bot checks."""
    s = requests.Session()
    s.headers.update({
        "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                       "AppleWebKit/537.36 (KHTML, like Gecko) "
                       "Chrome/120.0 Safari/537.36"),
        "Accept": "application/json, text/javascript, */*; q=0.01",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://www.nseindia.com/",
    })
    # Warm-up: hit home page to receive cookies, then hit the report page.
    s.get("https://www.nseindia.com/", timeout=10)
    time.sleep(0.5)
    s.get("https://www.nseindia.com/reports/fii-dii", timeout=10)
    time.sleep(0.5)
    return s


def fetch_fii_dii() -> pd.DataFrame:
    """Returns a DataFrame with columns: date, category, buy, sell, net."""
    s = nse_session()
    r = s.get(NSE_FIIDII_URL, timeout=30)
    r.raise_for_status()
    raw = r.json()
    if not raw:
        return pd.DataFrame()
    df = pd.DataFrame(raw)
    rename = {
        "category": "category",
        "buyValue": "buy",
        "sellValue": "sell",
        "netValue": "net",
        "date": "date",
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
    for col in ["buy", "sell", "net"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df["date"] = pd.to_datetime(df["date"], format="%d-%b-%Y", errors="coerce")
    df["date"] = df["date"].fillna(pd.to_datetime(df["date"], errors="coerce"))
    keep = [c for c in ["date", "category", "buy", "sell", "net"] if c in df.columns]
    return df[keep].dropna(subset=["date"]).reset_index(drop=True)


# ---------- Main ----------

def main() -> None:
    print("Stage 2e — FII/DII pull")
    print("-" * 50)

    log("Fetching from NSE...")
    new_df = fetch_fii_dii()
    if new_df.empty:
        print("ERROR: NSE returned no FII/DII data.")
        sys.exit(1)

    print(f"\nNew rows fetched: {len(new_df)}")
    print(new_df.to_string(index=False))

    drive = get_drive_service()
    folder_id = os.environ["GDRIVE_FOLDER_ID"]
    data_id = get_or_create_subfolder(drive, folder_id, "data")
    macro_id = get_or_create_subfolder(drive, data_id, "macro")

    filename = "FII_DII.csv"
    existing_id = find_file(drive, macro_id, filename)
    if existing_id:
        existing = download_csv(drive, existing_id)
        existing["date"] = pd.to_datetime(existing["date"])
        merged = pd.concat([existing, new_df], ignore_index=True)
        merged = merged.drop_duplicates(subset=["date", "category"], keep="last")
        merged = merged.sort_values(["date", "category"]).reset_index(drop=True)
        log(f"Existing rows: {len(existing)} → after merge: {len(merged)}")
    else:
        merged = new_df.sort_values(["date", "category"]).reset_index(drop=True)
        log(f"Creating new {filename} with {len(merged)} rows")

    upload_csv(drive, macro_id, filename, merged, existing_id)
    log(f"Uploaded data/macro/{filename}")
    print("-" * 50)


if __name__ == "__main__":
    main()
