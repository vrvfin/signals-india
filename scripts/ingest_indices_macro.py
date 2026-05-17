"""
Stage 2d — Ingest Indian indices + macro series via yfinance.

Writes:
    data/indices/<NAME>.parquet   — NIFTY_50, NIFTY_BANK, INDIA_VIX, sector indices, etc.
    data/macro/<NAME>.parquet     — USD/INR, crude, gold, US indices.

Incremental like ingest_ohlcv.py. Failed tickers are logged but don't abort the run.

Usage:
    python scripts/ingest_indices_macro.py             # 10-year backfill or incremental
    python scripts/ingest_indices_macro.py --years 5
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

INDIAN_INDICES = {
    "NIFTY_50":          "^NSEI",
    "NIFTY_BANK":        "^NSEBANK",
    "INDIA_VIX":         "^INDIAVIX",
    "NIFTY_500":         "^CRSLDX",
    "NIFTY_MIDCAP_100":  "^CNXMIDCAP",
    "NIFTY_SMALLCAP_100":"^CNXSC",
    "NIFTY_IT":          "^CNXIT",
    "NIFTY_AUTO":        "^CNXAUTO",
    "NIFTY_PHARMA":      "^CNXPHARMA",
    "NIFTY_FMCG":        "^CNXFMCG",
    "NIFTY_METAL":       "^CNXMETAL",
    "NIFTY_REALTY":      "^CNXREALTY",
    "NIFTY_ENERGY":      "^CNXENERGY",
    "NIFTY_INFRA":       "^CNXINFRA",
}

MACRO = {
    "USD_INR":           "INR=X",
    "BRENT_CRUDE":       "BZ=F",
    "WTI_CRUDE":         "CL=F",
    "GOLD":              "GC=F",
    "DOW_JONES":         "^DJI",
    "NASDAQ":            "^IXIC",
    "SP500":             "^GSPC",
}


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")


# ---------- Drive helpers (same as ingest_ohlcv) ----------

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


# ---------- Fetch + process ----------

def fetch_series(yf_ticker: str, start: date, end: date) -> pd.DataFrame:
    df = yf.Ticker(yf_ticker).history(
        start=start.isoformat(), end=(end + timedelta(days=1)).isoformat(),
        auto_adjust=True, raise_errors=False,
    )
    if df is None or df.empty:
        df = yf.download(yf_ticker, start=start.isoformat(),
                         end=(end + timedelta(days=1)).isoformat(),
                         auto_adjust=True, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.reset_index()
    df.columns = [c.lower().replace(" ", "_") for c in df.columns]
    date_col = "date" if "date" in df.columns else "datetime"
    df = df.rename(columns={date_col: "date"})
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None).dt.normalize()
    keep = [c for c in ["date", "open", "high", "low", "close", "volume"] if c in df.columns]
    return df[keep].sort_values("date").reset_index(drop=True)


def process_one(drive, folder_id: str, name: str, yf_ticker: str,
                existing_files: dict[str, str], backfill_start: date, today: date) -> dict:
    filename = f"{name}.parquet"
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
            return {"name": name, "ticker": yf_ticker, "status": "read_error",
                    "detail": str(e)[:120]}

    if fetch_start > today:
        return {"name": name, "ticker": yf_ticker, "status": "up_to_date",
                "rows_added": 0, "total_rows": len(existing_df) if existing_df is not None else 0}

    new_df = fetch_series(yf_ticker, fetch_start, today)
    if new_df.empty:
        if existing_df is not None and len(existing_df) > 0:
            return {"name": name, "ticker": yf_ticker, "status": "no_new_data",
                    "rows_added": 0, "total_rows": len(existing_df)}
        return {"name": name, "ticker": yf_ticker, "status": "no_data",
                "rows_added": 0, "total_rows": 0}

    if existing_df is not None and len(existing_df) > 0:
        merged = pd.concat([existing_df, new_df], ignore_index=True)
        merged = merged.drop_duplicates(subset=["date"], keep="last")
        merged = merged.sort_values("date").reset_index(drop=True)
    else:
        merged = new_df

    upload_parquet(drive, folder_id, filename, merged, existing_id)
    return {"name": name, "ticker": yf_ticker, "status": "ok",
            "rows_added": len(new_df), "total_rows": len(merged)}


def process_group(drive, parent_folder_id: str, subfolder_name: str,
                  tickers: dict[str, str], backfill_start: date, today: date) -> list[dict]:
    folder_id = get_or_create_subfolder(drive, parent_folder_id, subfolder_name)
    existing = list_files_in_folder(drive, folder_id)
    log(f"[{subfolder_name}] {len(tickers)} tickers to process | "
        f"{len(existing)} existing parquets")
    results = []
    for name, yf_ticker in tickers.items():
        try:
            r = process_one(drive, folder_id, name, yf_ticker, existing, backfill_start, today)
        except Exception as e:
            r = {"name": name, "ticker": yf_ticker, "status": "error",
                 "detail": str(e)[:120]}
        log(f"  {name:<22} ({yf_ticker:<12}) {r['status']:<14} "
            f"+{r.get('rows_added', 0)} rows")
        results.append(r)
        time.sleep(0.3)
    return results


# ---------- Main ----------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--years", type=int, default=10, help="Backfill years (default 10)")
    args = parser.parse_args()

    today = date.today()
    backfill_start = today - timedelta(days=365 * args.years)

    print("Stage 2d — Indices + Macro ingest")
    print("-" * 50)
    log(f"Backfill window: {backfill_start} → {today}")

    drive = get_drive_service()
    folder_id = os.environ["GDRIVE_FOLDER_ID"]
    data_id = get_or_create_subfolder(drive, folder_id, "data")

    t_start = time.time()
    idx_results = process_group(drive, data_id, "indices", INDIAN_INDICES,
                                backfill_start, today)
    mac_results = process_group(drive, data_id, "macro", MACRO,
                                backfill_start, today)

    all_results = pd.DataFrame(idx_results + mac_results)
    print()
    print("-" * 50)
    print("Status counts:")
    print(all_results["status"].value_counts().to_string())
    print(f"\nFailed / no-data tickers (review these — yfinance symbol may be wrong):")
    bad = all_results[~all_results["status"].isin(["ok", "up_to_date"])]
    if len(bad) > 0:
        print(bad[["name", "ticker", "status"]].to_string(index=False))
    else:
        print("  (none)")
    print(f"\nElapsed: {(time.time()-t_start)/60:.1f} min")


if __name__ == "__main__":
    main()
