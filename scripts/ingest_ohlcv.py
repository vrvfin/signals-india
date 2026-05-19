"""
Stage 2c (v2) — Bulk OHLCV ingestion via yfinance BATCH download.

Why batched (yf.download) vs per-symbol (yf.Ticker.history):
  - Batched uses Yahoo's chart endpoint, which works fine from GitHub Actions /
    datacenter IPs.
  - Per-symbol uses Yahoo's older finance-quote endpoint, which gets rate-limited
    from datacenter IPs and fails with "Expecting value: line 1 column 1".
  - Batched is also 5-10× faster overall.

Usage:
    python scripts/ingest_ohlcv.py                       # default: incremental, period=1mo
    python scripts/ingest_ohlcv.py --backfill            # full history (period=10y) for missing files
    python scripts/ingest_ohlcv.py --period 6mo          # custom window (overrides incremental)
    python scripts/ingest_ohlcv.py --pilot --limit 50    # test on first 50 symbols
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
DEFAULT_BATCH_SIZE = 50
DEFAULT_INCREMENTAL_PERIOD = "1mo"
BACKFILL_PERIOD = "10y"


def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")


# ---------- Drive helpers ----------

def get_drive_service():
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
    found = drive.files().list(q=q, fields="files(id,name)").execute().get("files", [])
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
            fields="nextPageToken, files(id,name)",
            pageSize=1000, pageToken=page_token,
        ).execute()
        for f in resp.get("files", []):
            out[f["name"]] = f["id"]
        page_token = resp.get("nextPageToken")
        if not page_token:
            break
    return out


def download_csv(drive, file_id):
    request = drive.files().get_media(fileId=file_id)
    fh = io.BytesIO()
    d = MediaIoBaseDownload(fh, request)
    done = False
    while not done:
        _, done = d.next_chunk()
    fh.seek(0)
    return pd.read_csv(fh)


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


# ---------- Batched fetch ----------

def fetch_ohlcv_batch(symbols: list[str], period: str) -> dict[str, pd.DataFrame]:
    """Batched yf.download. Returns {symbol_without_suffix: normalized DataFrame}."""
    if not symbols:
        return {}
    suffixed = [f"{s}.NS" for s in symbols]
    tickers_str = " ".join(suffixed)

    try:
        df = yf.download(
            tickers=tickers_str,
            period=period,
            group_by="ticker",
            auto_adjust=True,
            progress=False,
            threads=True,
        )
    except Exception as e:
        log(f"  Batch fetch raised: {str(e)[:160]}")
        return {}
    if df is None or df.empty:
        return {}

    out: dict[str, pd.DataFrame] = {}

    def _normalize(frame: pd.DataFrame) -> pd.DataFrame:
        f = frame.reset_index()
        f.columns = [str(c).lower().replace(" ", "_") for c in f.columns]
        if "date" not in f.columns and "datetime" in f.columns:
            f = f.rename(columns={"datetime": "date"})
        f["date"] = pd.to_datetime(f["date"]).dt.tz_localize(None).dt.normalize()
        keep = [c for c in ["date", "open", "high", "low", "close", "volume"]
                if c in f.columns]
        f = f[keep].dropna(subset=["close"]).sort_values("date").reset_index(drop=True)
        return f

    if len(symbols) == 1:
        sym = symbols[0]
        out[sym] = _normalize(df)
    else:
        # Multi-index columns: level 0 = ticker (e.g. "RELIANCE.NS"), level 1 = field
        if not isinstance(df.columns, pd.MultiIndex):
            return out  # unexpected shape, skip
        present_tickers = set(df.columns.get_level_values(0))
        for sym in symbols:
            full = f"{sym}.NS"
            if full not in present_tickers:
                continue
            try:
                sub = df[full]
                if sub is None or sub.empty:
                    continue
                normalized = _normalize(sub)
                if not normalized.empty:
                    out[sym] = normalized
            except Exception:
                continue
    return out


def merge_and_upload(drive, ohlcv_folder_id: str, symbol: str,
                     new_df: pd.DataFrame, existing_files: dict[str, str]) -> dict:
    """Upsert new rows into the symbol's parquet on Drive."""
    filename = f"{symbol}.parquet"
    existing_id = existing_files.get(filename)
    existing_df = None
    if existing_id:
        try:
            existing_df = download_parquet(drive, existing_id)
        except Exception as e:
            return {"symbol": symbol, "status": "read_error", "detail": str(e)[:120]}

    if new_df.empty:
        return {"symbol": symbol, "status": "no_data",
                "rows_added": 0,
                "total_rows": len(existing_df) if existing_df is not None else 0}

    if existing_df is not None and len(existing_df) > 0:
        existing_df["date"] = pd.to_datetime(existing_df["date"])
        max_existing_date = existing_df["date"].max()
        truly_new = new_df[new_df["date"] > max_existing_date]
        merged = pd.concat([existing_df, truly_new], ignore_index=True)
        merged = (merged.drop_duplicates(subset=["date"], keep="last")
                  .sort_values("date").reset_index(drop=True))
        rows_added = len(truly_new)
    else:
        merged = new_df
        rows_added = len(new_df)

    if rows_added == 0:
        return {"symbol": symbol, "status": "up_to_date",
                "rows_added": 0, "total_rows": len(merged)}

    upload_parquet(drive, ohlcv_folder_id, filename, merged, existing_id)
    return {"symbol": symbol, "status": "ok",
            "rows_added": rows_added, "total_rows": len(merged)}


# ---------- Main ----------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--period", type=str, default=None,
                        help="yfinance period (default: 1mo incremental, 10y if --backfill)")
    parser.add_argument("--backfill", action="store_true",
                        help="Force period=10y for full history")
    args = parser.parse_args()

    period = args.period
    if period is None:
        period = BACKFILL_PERIOD if args.backfill else DEFAULT_INCREMENTAL_PERIOD

    print("Stage 2c (v2) — Batched OHLCV ingest")
    print("-" * 50)
    log(f"Mode: period={period}, batch_size={args.batch_size}")

    drive = get_drive_service()
    folder_id = os.environ["GDRIVE_FOLDER_ID"]

    universe_folder_id = get_or_create_subfolder(drive, folder_id, "universe")
    universe_files = list_files_in_folder(drive, universe_folder_id)
    if "master_list.csv" not in universe_files:
        print("ERROR: universe/master_list.csv missing. Run build_universe.py first.")
        sys.exit(1)
    universe_df = download_csv(drive, universe_files["master_list.csv"])
    symbols = universe_df["symbol"].astype(str).tolist()
    if args.pilot:
        symbols = symbols[:args.limit]
        log(f"PILOT MODE: {len(symbols)} symbols")
    log(f"Symbols to process: {len(symbols)}")

    data_folder_id = get_or_create_subfolder(drive, folder_id, "data")
    ohlcv_folder_id = get_or_create_subfolder(drive, data_folder_id, "ohlcv")
    existing_files = list_files_in_folder(drive, ohlcv_folder_id)
    log(f"Existing OHLCV parquets: {len(existing_files)}")

    # Chunk into batches
    batches = [symbols[i:i + args.batch_size]
               for i in range(0, len(symbols), args.batch_size)]
    log(f"Batches: {len(batches)} of size up to {args.batch_size}")

    results: list[dict] = []
    t_start = time.time()
    for b_idx, batch in enumerate(batches, 1):
        b_start = time.time()
        fetched = fetch_ohlcv_batch(batch, period=period)
        not_returned = [s for s in batch if s not in fetched]

        # Process each fetched symbol
        for sym in batch:
            if sym in fetched:
                try:
                    r = merge_and_upload(drive, ohlcv_folder_id, sym,
                                         fetched[sym], existing_files)
                except Exception as e:
                    r = {"symbol": sym, "status": "upload_error",
                         "detail": str(e)[:120]}
            else:
                r = {"symbol": sym, "status": "no_data_returned",
                     "rows_added": 0,
                     "total_rows": (len(download_parquet(drive, existing_files[f"{sym}.parquet"]))
                                    if f"{sym}.parquet" in existing_files else 0)
                                    if False else 0}  # don't bother re-downloading just to count
            results.append(r)

        elapsed = time.time() - t_start
        done = b_idx * args.batch_size
        rate = done / elapsed if elapsed else 0
        eta = (len(symbols) - done) / rate / 60 if rate else 0
        ok_so_far = sum(1 for x in results if x["status"] == "ok")
        log(f"  Batch {b_idx}/{len(batches)}  fetched={len(fetched)}/{len(batch)}  "
            f"ok_so_far={ok_so_far}  rate={rate:.1f}/s  ETA={eta:.1f}m  "
            f"(batch took {time.time()-b_start:.1f}s)")

    summary = pd.DataFrame(results)
    print()
    print("-" * 50)
    print("Status counts:")
    print(summary["status"].value_counts().to_string())
    total_rows = summary["rows_added"].fillna(0).sum()
    print(f"\nTotal rows added: {int(total_rows)}")
    print(f"Elapsed: {(time.time()-t_start)/60:.1f} min")

    # Run log to Drive
    logs_id = get_or_create_subfolder(drive, folder_id, "logs")
    runs_id = get_or_create_subfolder(drive, logs_id, "ingest_ohlcv")
    today = date.today().isoformat()
    log_name = f"ingest_{today}_{datetime.now().strftime('%H%M')}.csv"
    media = MediaIoBaseUpload(io.BytesIO(summary.to_csv(index=False).encode()),
                              mimetype="text/csv", resumable=False)
    drive.files().create(
        body={"name": log_name, "parents": [runs_id]},
        media_body=media, fields="id").execute()
    log(f"Run log: logs/ingest_ohlcv/{log_name}")


if __name__ == "__main__":
    main()
