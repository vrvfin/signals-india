"""
Stage 2c — Bulk OHLCV ingestion for the NSE universe.

Reads universe from Drive (`universe/master_list.csv`), fetches adjusted OHLCV per
symbol via yfinance, and writes per-symbol Parquet to Drive at `data/ohlcv/<SYMBOL>.parquet`.

Incremental: if a Parquet already exists for a symbol, only days after its latest
date are fetched and appended. First run does a full backfill.

Usage:
    python scripts/ingest_ohlcv.py                # default: pilot of 50 symbols, 10yr backfill
    python scripts/ingest_ohlcv.py --all          # full universe
    python scripts/ingest_ohlcv.py --limit 200    # custom symbol count
    python scripts/ingest_ohlcv.py --years 5      # change backfill window
"""

from __future__ import annotations

import argparse
import io
import os
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
import yfinance as yf
from dotenv import load_dotenv
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload, MediaIoBaseUpload

SCOPES = ["https://www.googleapis.com/auth/drive"]


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


def list_files_in_folder(drive, folder_id: str) -> dict[str, str]:
    """Return {filename: file_id} for all files in folder."""
    out: dict[str, str] = {}
    page_token = None
    while True:
        resp = drive.files().list(
            q=f"'{folder_id}' in parents and trashed=false",
            fields="nextPageToken, files(id,name)",
            pageSize=1000, pageToken=page_token,
        ).execute()
        for f in resp.get("files", []):
            out[f["name"]] = f["id"]
        page_token = resp.get("nextPageToken")
        if not page_token:
            break
    return out


def download_csv(drive, file_id: str) -> pd.DataFrame:
    request = drive.files().get_media(fileId=file_id)
    fh = io.BytesIO()
    downloader = MediaIoBaseDownload(fh, request)
    done = False
    while not done:
        _, done = downloader.next_chunk()
    fh.seek(0)
    return pd.read_csv(fh)


def download_parquet(drive, file_id: str) -> pd.DataFrame:
    request = drive.files().get_media(fileId=file_id)
    fh = io.BytesIO()
    downloader = MediaIoBaseDownload(fh, request)
    done = False
    while not done:
        _, done = downloader.next_chunk()
    fh.seek(0)
    return pd.read_parquet(fh)


def upload_parquet(drive, folder_id: str, filename: str, df: pd.DataFrame,
                   existing_id: str | None) -> str:
    buf = io.BytesIO()
    df.to_parquet(buf, index=False)
    buf.seek(0)
    media = MediaIoBaseUpload(buf, mimetype="application/octet-stream", resumable=False)
    if existing_id:
        drive.files().update(fileId=existing_id, media_body=media).execute()
        return existing_id
    meta = {"name": filename, "parents": [folder_id]}
    return drive.files().create(body=meta, media_body=media, fields="id").execute()["id"]


# ---------- Fetch ----------

def fetch_ohlcv(symbol: str, start: date, end: date) -> pd.DataFrame:
    """Fetch OHLCV from yfinance, normalize column names."""
    df = yf.Ticker(f"{symbol}.NS").history(
        start=start.isoformat(), end=(end + timedelta(days=1)).isoformat(),
        auto_adjust=True, raise_errors=False,
    )
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.reset_index()
    df.columns = [c.lower().replace(" ", "_") for c in df.columns]
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None).dt.normalize()
    keep = ["date", "open", "high", "low", "close", "volume"]
    return df[keep].sort_values("date").reset_index(drop=True)


def process_symbol(drive, ohlcv_folder_id: str, symbol: str,
                   existing_files: dict[str, str], backfill_start: date,
                   today: date) -> dict:
    """Fetch + merge + upload one symbol. Returns status dict."""
    filename = f"{symbol}.parquet"
    existing_id = existing_files.get(filename)
    existing_df = None
    fetch_start = backfill_start

    if existing_id:
        try:
            existing_df = download_parquet(drive, existing_id)
            if len(existing_df) > 0:
                last_date = pd.to_datetime(existing_df["date"]).max().date()
                fetch_start = last_date + timedelta(days=1)
        except Exception as e:
            return {"symbol": symbol, "status": "read_error", "detail": str(e)[:120]}

    if fetch_start > today:
        return {"symbol": symbol, "status": "up_to_date", "rows_added": 0,
                "total_rows": len(existing_df) if existing_df is not None else 0}

    new_df = fetch_ohlcv(symbol, fetch_start, today)
    if new_df.empty:
        if existing_df is not None and len(existing_df) > 0:
            return {"symbol": symbol, "status": "no_new_data", "rows_added": 0,
                    "total_rows": len(existing_df)}
        return {"symbol": symbol, "status": "no_data", "rows_added": 0, "total_rows": 0}

    if existing_df is not None and len(existing_df) > 0:
        merged = pd.concat([existing_df, new_df], ignore_index=True)
        merged = merged.drop_duplicates(subset=["date"], keep="last")
        merged = merged.sort_values("date").reset_index(drop=True)
    else:
        merged = new_df

    upload_parquet(drive, ohlcv_folder_id, filename, merged, existing_id)
    return {"symbol": symbol, "status": "ok", "rows_added": len(new_df),
            "total_rows": len(merged)}


# ---------- Main ----------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot", action="store_true",
                        help="Pilot mode: only process the first --limit symbols (default: process full universe)")
    parser.add_argument("--limit", type=int, default=50,
                        help="Symbol cap when --pilot is set (default 50)")
    parser.add_argument("--years", type=int, default=10, help="Backfill years (default 10)")
    parser.add_argument("--sleep", type=float, default=0.4,
                        help="Seconds between symbols to be polite to Yahoo (default 0.4)")
    args = parser.parse_args()

    today = date.today()
    backfill_start = today - timedelta(days=365 * args.years)

    print("Stage 2c — Bulk OHLCV ingest")
    print("-" * 50)
    log(f"Backfill window: {backfill_start} → {today} ({args.years} years)")

    drive = get_drive_service()
    folder_id = os.environ["GDRIVE_FOLDER_ID"]

    # Load universe
    universe_folder_id = get_or_create_subfolder(drive, folder_id, "universe")
    universe_files = list_files_in_folder(drive, universe_folder_id)
    if "master_list.csv" not in universe_files:
        print("ERROR: universe/master_list.csv not found. Run build_universe.py first.")
        sys.exit(1)
    universe_df = download_csv(drive, universe_files["master_list.csv"])
    symbols = universe_df["symbol"].astype(str).tolist()
    if args.pilot:
        symbols = symbols[:args.limit]
        log(f"PILOT MODE: limited to first {args.limit} symbols.")
    log(f"Symbols to process: {len(symbols)}")

    # Set up data/ohlcv folder
    data_folder_id = get_or_create_subfolder(drive, folder_id, "data")
    ohlcv_folder_id = get_or_create_subfolder(drive, data_folder_id, "ohlcv")
    existing_files = list_files_in_folder(drive, ohlcv_folder_id)
    log(f"Existing OHLCV parquets in Drive: {len(existing_files)}")

    # Process
    results: list[dict] = []
    t_start = time.time()
    for i, sym in enumerate(symbols, 1):
        try:
            r = process_symbol(drive, ohlcv_folder_id, sym,
                               existing_files, backfill_start, today)
        except Exception as e:
            r = {"symbol": sym, "status": "error", "detail": str(e)[:120]}
        results.append(r)
        elapsed = time.time() - t_start
        rate = i / elapsed if elapsed else 0
        eta_s = (len(symbols) - i) / rate if rate else 0
        log(f"[{i}/{len(symbols)}] {sym:<14} {r['status']:<14} "
            f"{r.get('rows_added', 0):>5} new | "
            f"rate {rate:.1f}/s | ETA {eta_s/60:.1f}m")
        time.sleep(args.sleep)

    # Summary
    summary = pd.DataFrame(results)
    print()
    print("-" * 50)
    print("Status counts:")
    print(summary["status"].value_counts().to_string())
    total_rows_added = summary.get("rows_added", pd.Series(dtype=int)).sum()
    print(f"\nTotal rows added: {total_rows_added}")
    print(f"Elapsed: {(time.time()-t_start)/60:.1f} min")

    # Save run log to Drive
    logs_folder_id = get_or_create_subfolder(drive, folder_id, "logs")
    runs_folder_id = get_or_create_subfolder(drive, logs_folder_id, "ingest_ohlcv")
    log_name = f"ingest_{today.isoformat()}_{datetime.now().strftime('%H%M')}.csv"
    log_bytes = summary.to_csv(index=False).encode()
    media = MediaIoBaseUpload(io.BytesIO(log_bytes), mimetype="text/csv", resumable=False)
    drive.files().create(
        body={"name": log_name, "parents": [runs_folder_id]},
        media_body=media, fields="id",
    ).execute()
    log(f"Run log: logs/ingest_ohlcv/{log_name}")


if __name__ == "__main__":
    main()
