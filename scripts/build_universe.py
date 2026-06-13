"""
Stage 2a — Build the master NSE equity universe.

Output (in your Google Drive `signals-india` folder):
    universe/master_list.csv             — current universe (overwritten each run)
    universe/history/master_list_YYYY-MM-DD.csv — daily snapshot

Run from project root, inside the `signals-india` conda env:
    python scripts/build_universe.py

BSE coverage is intentionally not in this script — added in 2a.2 once NSE flow is verified.
"""

from __future__ import annotations

import io
import os
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaInMemoryUpload

SCOPES = ["https://www.googleapis.com/auth/drive"]
NSE_EQUITY_LIST_URL = "https://archives.nseindia.com/content/equities/EQUITY_L.csv"
KEEP_SERIES = {"EQ", "BE", "BZ"}  # mainboard equity series; we tag BZ/BE for awareness


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")


def get_drive_service():
    """Authenticate and return Drive API service, reusing cached OAuth token."""
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


def fetch_nse_equity_list() -> pd.DataFrame:
    """Download NSE EQUITY_L.csv and normalize column names."""
    log(f"Fetching NSE equity list from {NSE_EQUITY_LIST_URL}")
    headers = {
        "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                       "AppleWebKit/537.36 (KHTML, like Gecko) "
                       "Chrome/120.0 Safari/537.36"),
        "Accept": "text/csv,*/*",
    }
    r = requests.get(NSE_EQUITY_LIST_URL, headers=headers, timeout=30)
    r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text))
    # NSE columns can have stray spaces — normalize.
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    rename = {
        "name_of_company": "name",
        "isin_number": "isin",
        "date_of_listing": "listing_date",
    }
    df = df.rename(columns=rename)
    df["symbol"] = df["symbol"].astype(str).str.strip()
    df["series"] = df["series"].astype(str).str.strip()
    df["exchange"] = "NSE"
    before = len(df)
    df = df[df["series"].isin(KEEP_SERIES)].copy()
    log(f"Series filter: {before} → {len(df)} symbols "
        f"(kept {sorted(KEEP_SERIES)})")
    cols = ["symbol", "exchange", "name", "isin", "series", "listing_date"]
    return df[cols].reset_index(drop=True)


def get_or_create_subfolder(drive, parent_id: str, name: str) -> str:
    """Find a child folder by name, or create it."""
    q = (f"name='{name}' and '{parent_id}' in parents "
         f"and mimeType='application/vnd.google-apps.folder' and trashed=false")
    found = drive.files().list(q=q, fields="files(id,name)").execute().get("files", [])
    if found:
        return found[0]["id"]
    meta = {"name": name, "parents": [parent_id],
            "mimeType": "application/vnd.google-apps.folder"}
    return drive.files().create(body=meta, fields="id").execute()["id"]


def upload_csv(drive, df: pd.DataFrame, folder_id: str, filename: str) -> str:
    """Upload CSV to Drive folder. Overwrites if filename already exists."""
    csv_bytes = df.to_csv(index=False).encode()
    media = MediaInMemoryUpload(csv_bytes, mimetype="text/csv")
    q = f"name='{filename}' and '{folder_id}' in parents and trashed=false"
    existing = drive.files().list(q=q, fields="files(id)").execute().get("files", [])
    if existing:
        drive.files().update(fileId=existing[0]["id"], media_body=media).execute()
        return existing[0]["id"]
    meta = {"name": filename, "parents": [folder_id]}
    return drive.files().create(body=meta, media_body=media, fields="id").execute()["id"]


def main() -> None:
    print("Stage 2a — Build universe (NSE)")
    print("-" * 50)
    df = fetch_nse_equity_list()
    log(f"NSE symbols ready: {len(df)}")
    print("\nSeries breakdown:")
    print(df["series"].value_counts().to_string())
    print("\nFirst 5 rows:")
    print(df.head().to_string(index=False))
    print()

    drive = get_drive_service()
    folder_id = os.environ["GDRIVE_FOLDER_ID"]
    universe_id = get_or_create_subfolder(drive, folder_id, "universe")
    history_id = get_or_create_subfolder(drive, universe_id, "history")

    today = datetime.now().strftime("%Y-%m-%d")
    upload_csv(drive, df, universe_id, "master_list.csv")
    log("Uploaded universe/master_list.csv")
    upload_csv(drive, df, history_id, f"master_list_{today}.csv")
    log(f"Uploaded universe/history/master_list_{today}.csv")

    print("-" * 50)
    print(f"Done. {len(df)} NSE symbols in master_list.csv.")


if __name__ == "__main__":
    main()