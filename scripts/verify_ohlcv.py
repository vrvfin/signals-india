"""
Verify a stored OHLCV Parquet from Drive — used as the Stage 2c sign-off check.

Usage:
    python scripts/verify_ohlcv.py             # checks RELIANCE, TCS, INFY, HDFCBANK
    python scripts/verify_ohlcv.py SBIN        # any single symbol
"""

from __future__ import annotations

import io
import os
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

SCOPES = ["https://www.googleapis.com/auth/drive"]
DEFAULT_SYMBOLS = ["RELIANCE", "TCS", "INFY", "HDFCBANK"]


def get_drive_service():
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
    token_path = Path(os.environ["GDRIVE_OAUTH_TOKEN_PATH"])
    creds = Credentials.from_authorized_user_file(str(token_path), SCOPES)
    if creds.expired and creds.refresh_token:
        creds.refresh(Request())
        token_path.write_text(creds.to_json())
    return build("drive", "v3", credentials=creds, cache_discovery=False)


def find_subfolder(drive, parent_id: str, name: str) -> str | None:
    q = (f"name='{name}' and '{parent_id}' in parents "
         f"and mimeType='application/vnd.google-apps.folder' and trashed=false")
    found = drive.files().list(q=q, fields="files(id)").execute().get("files", [])
    return found[0]["id"] if found else None


def find_file(drive, folder_id: str, name: str) -> str | None:
    q = f"name='{name}' and '{folder_id}' in parents and trashed=false"
    found = drive.files().list(q=q, fields="files(id)").execute().get("files", [])
    return found[0]["id"] if found else None


def download_parquet(drive, file_id: str) -> pd.DataFrame:
    request = drive.files().get_media(fileId=file_id)
    fh = io.BytesIO()
    downloader = MediaIoBaseDownload(fh, request)
    done = False
    while not done:
        _, done = downloader.next_chunk()
    fh.seek(0)
    return pd.read_parquet(fh)


def check_symbol(drive, ohlcv_folder_id: str, symbol: str) -> None:
    print(f"\n=== {symbol} ===")
    file_id = find_file(drive, ohlcv_folder_id, f"{symbol}.parquet")
    if not file_id:
        print(f"  NOT FOUND in Drive.")
        return
    df = download_parquet(drive, file_id)
    if len(df) == 0:
        print(f"  Parquet is empty.")
        return
    df["date"] = pd.to_datetime(df["date"])
    n = len(df)
    d_min, d_max = df["date"].min().date(), df["date"].max().date()
    years = (d_max - d_min).days / 365.25
    expected_rows = int(years * 250)
    print(f"  Rows         : {n}")
    print(f"  Date range   : {d_min} → {d_max}  ({years:.1f} yrs)")
    print(f"  Last close   : ₹{df['close'].iloc[-1]:,.2f}")
    print(f"  All-time high: ₹{df['high'].max():,.2f}")
    print(f"  All-time low : ₹{df['low'].min():,.2f}")
    print(f"  Avg daily vol: {df['volume'].mean():,.0f}")
    coverage = n / expected_rows if expected_rows else 0
    flag = "OK" if 0.90 <= coverage <= 1.05 else "CHECK"
    print(f"  Row coverage : {coverage:.0%} of expected ({expected_rows}) [{flag}]")
    print(f"  Last 3 rows  :")
    print(df.tail(3)[["date", "open", "high", "low", "close", "volume"]].to_string(index=False))


def main() -> None:
    symbols = [sys.argv[1].upper()] if len(sys.argv) > 1 else DEFAULT_SYMBOLS
    drive = get_drive_service()
    folder_id = os.environ["GDRIVE_FOLDER_ID"]
    data_id = find_subfolder(drive, folder_id, "data")
    if not data_id:
        print("data/ folder not found in Drive.")
        sys.exit(1)
    ohlcv_id = find_subfolder(drive, data_id, "ohlcv")
    if not ohlcv_id:
        print("data/ohlcv/ folder not found in Drive.")
        sys.exit(1)
    for sym in symbols:
        check_symbol(drive, ohlcv_id, sym)


if __name__ == "__main__":
    main()
