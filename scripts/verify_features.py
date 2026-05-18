"""
Verify features/latest.parquet — Stage 3 sign-off check.

Usage:
    python scripts/verify_features.py             # checks RELIANCE, TCS, INFY, HDFCBANK
    python scripts/verify_features.py SBIN ITC    # any list of symbols
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


def get_drive():
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
    token_path = Path(os.environ["GDRIVE_OAUTH_TOKEN_PATH"])
    creds = Credentials.from_authorized_user_file(str(token_path), SCOPES)
    if creds.expired and creds.refresh_token:
        creds.refresh(Request())
        token_path.write_text(creds.to_json())
    return build("drive", "v3", credentials=creds, cache_discovery=False)


def find_subfolder(drive, parent_id, name):
    q = (f"name='{name}' and '{parent_id}' in parents "
         f"and mimeType='application/vnd.google-apps.folder' and trashed=false")
    found = drive.files().list(q=q, fields="files(id)").execute().get("files", [])
    return found[0]["id"] if found else None


def find_file(drive, folder_id, name):
    q = f"name='{name}' and '{folder_id}' in parents and trashed=false"
    found = drive.files().list(q=q, fields="files(id)").execute().get("files", [])
    return found[0]["id"] if found else None


def download_parquet(drive, file_id):
    request = drive.files().get_media(fileId=file_id)
    fh = io.BytesIO()
    downloader = MediaIoBaseDownload(fh, request)
    done = False
    while not done:
        _, done = downloader.next_chunk()
    fh.seek(0)
    return pd.read_parquet(fh)


SHOW_COLS = [
    "symbol", "close", "ema_20", "sma_200", "atr_14", "adr_pct_20",
    "high_52w", "dist_from_52w_high_pct",
    "return_1m_pct", "return_3m_pct", "return_6m_pct", "return_12m_pct",
    "rs_rank_3m", "rs_rank_12m",
    "vol_today_ratio", "above_200sma",
    "days_above_ema_10", "days_above_ema_20", "days_above_ema_50",
]


def main() -> None:
    symbols = [s.upper() for s in sys.argv[1:]] or DEFAULT_SYMBOLS

    drive = get_drive()
    folder_id = os.environ["GDRIVE_FOLDER_ID"]
    features_id = find_subfolder(drive, folder_id, "features")
    latest_id = find_file(drive, features_id, "latest.parquet")
    if not latest_id:
        print("features/latest.parquet not found.")
        sys.exit(1)
    df = download_parquet(drive, latest_id)
    print(f"features/latest.parquet — {len(df)} rows, {len(df.columns)} columns\n")

    # Universe stats
    print("Universe sanity:")
    print(f"  Median 3m return     : {df['return_3m_pct'].median():.2f}%")
    print(f"  % above 200 SMA      : {df['above_200sma'].mean()*100:.1f}%")
    print(f"  Median ADR%(20)      : {df['adr_pct_20'].median():.2f}%")
    print(f"  Median days_above_20ema: {df['days_above_ema_20'].median():.0f}")
    print()

    # Top movers (3-month)
    top3m = df.nlargest(5, "return_3m_pct")[["symbol", "return_3m_pct", "rs_rank_3m", "dist_from_52w_high_pct"]]
    print("Top 5 by 3-month return:")
    print(top3m.to_string(index=False))
    print()

    # Spot-check symbols
    show = [c for c in SHOW_COLS if c in df.columns]
    for sym in symbols:
        row = df[df["symbol"] == sym]
        if row.empty:
            print(f"=== {sym}: NOT FOUND ===\n")
            continue
        print(f"=== {sym} ===")
        # Print as key:value, one per line for readability
        for col in show:
            val = row[col].iloc[0]
            if isinstance(val, (int, float)):
                if "ratio" in col or "rank" in col or "pct" in col or "adr" in col:
                    print(f"  {col:<30} {val:>10.2f}")
                else:
                    print(f"  {col:<30} {val:>10.2f}" if isinstance(val, float)
                          else f"  {col:<30} {val:>10}")
            else:
                print(f"  {col:<30} {val}")
        print()


if __name__ == "__main__":
    main()
