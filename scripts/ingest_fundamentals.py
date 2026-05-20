"""
Stage 11a — Fundamentals ingestion from Screener.in.

For each symbol in the universe, fetches the screener page, extracts key
financial fields (P/E, market cap, RoCE, last 4 quarters of EPS/Sales/NetProfit,
growth rates, promoter holding), writes:

  fundamentals/per_symbol/<SYMBOL>.parquet   — full row per stock (long-lived cache)
  fundamentals/summary.parquet              — all stocks unioned, one row each

The summary file is what CANSLIM and PEAD strategies consume.

Cookie expiry is detected and announced — script halts with refresh instructions.

Usage:
    python scripts/ingest_fundamentals.py             # full universe (~40-50 min)
    python scripts/ingest_fundamentals.py --limit 20  # quick test
"""

from __future__ import annotations

import argparse
import io
import os
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload, MediaIoBaseUpload

from screener_client import ScreenerClient, CookieExpiredError

SCOPES = ["https://www.googleapis.com/auth/drive"]


def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")


# ---- Drive helpers (same pattern as our other scripts) ----

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


def list_files_in_folder(drive, folder_id):
    out, page_token = {}, None
    while True:
        resp = drive.files().list(
            q=f"'{folder_id}' in parents and trashed=false",
            fields="nextPageToken, files(id,name)", pageSize=1000, pageToken=page_token,
        ).execute()
        for f in resp.get("files", []):
            out[f["name"]] = f["id"]
        page_token = resp.get("nextPageToken")
        if not page_token:
            break
    return out


def find_file(drive, folder_id, name):
    q = f"name='{name}' and '{folder_id}' in parents and trashed=false"
    found = drive.files().list(q=q, fields="files(id)").execute().get("files", [])
    return found[0]["id"] if found else None


def download_csv(drive, file_id):
    request = drive.files().get_media(fileId=file_id)
    fh = io.BytesIO()
    d = MediaIoBaseDownload(fh, request)
    done = False
    while not done:
        _, done = d.next_chunk()
    fh.seek(0)
    return pd.read_csv(fh)


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


# ---- Main ----

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--sleep", type=float, default=1.0,
                        help="Seconds between screener requests (default 1.0)")
    parser.add_argument("--resume", action="store_true",
                        help="Skip symbols already in per_symbol/")
    args = parser.parse_args()

    print("Stage 11a — Fundamentals ingestion")
    print("-" * 50)

    client = ScreenerClient(rate_limit_sec=args.sleep)
    log("Screener client initialized")

    drive = get_drive()
    folder_id = os.environ["GDRIVE_FOLDER_ID"]

    universe_id = get_or_create_subfolder(drive, folder_id, "universe")
    master_id = find_file(drive, universe_id, "master_list.csv")
    universe = download_csv(drive, master_id)
    symbols = universe["symbol"].astype(str).tolist()
    if args.limit:
        symbols = symbols[:args.limit]

    fund_id = get_or_create_subfolder(drive, folder_id, "fundamentals")
    per_sym_id = get_or_create_subfolder(drive, fund_id, "per_symbol")
    existing = list_files_in_folder(drive, per_sym_id)

    if args.resume:
        symbols = [s for s in symbols if f"{s}.parquet" not in existing]
        log(f"Resume mode: {len(symbols)} symbols left after skipping {len(existing)} done.")
    log(f"Symbols to fetch: {len(symbols)}")

    rows = []
    t_start = time.time()
    fail_count = 0
    for i, sym in enumerate(symbols, 1):
        try:
            soup = client.fetch_company(sym)
        except CookieExpiredError:
            log("Stopping run — cookie expired. Refresh instructions printed above.")
            return
        except Exception as e:
            log(f"  {sym}: fetch error — {str(e)[:80]}")
            fail_count += 1
            continue

        if soup is None:
            fail_count += 1
            continue

        try:
            summary = client.extract_summary(sym, soup)
            summary["fetched_at"] = datetime.now().isoformat()
            rows.append(summary)
            # Per-symbol parquet (for cache + future detail panels)
            sym_df = pd.DataFrame([summary])
            upload_parquet(drive, per_sym_id, f"{sym}.parquet", sym_df,
                           existing.get(f"{sym}.parquet"))
        except Exception as e:
            log(f"  {sym}: parse error — {str(e)[:80]}")
            fail_count += 1

        if i % 25 == 0:
            elapsed = time.time() - t_start
            rate = i / elapsed
            eta = (len(symbols) - i) / rate / 60
            log(f"  [{i}/{len(symbols)}] ok={len(rows)} fail={fail_count} "
                f"rate {rate:.2f}/s | ETA {eta:.1f}m")

    if rows:
        summary_df = pd.DataFrame(rows)
        upload_parquet(drive, fund_id, "summary.parquet", summary_df,
                       find_file(drive, fund_id, "summary.parquet"))
        log(f"Wrote fundamentals/summary.parquet ({len(summary_df)} rows)")
    log(f"Done. ok={len(rows)} fail={fail_count} elapsed={(time.time()-t_start)/60:.1f}m")


if __name__ == "__main__":
    main()
