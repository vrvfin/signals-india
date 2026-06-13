"""
Stage 8.5 — Enrich universe with market capitalization.

For each symbol in universe/master_list.csv, fetches `marketCap` from yfinance
and assigns a segment label. Writes universe/market_cap.csv.

Segments (based on the user's sub-5k cr focus):
    Largecap : > ₹20,000 cr
    Midcap   : ₹5,000 - 20,000 cr
    Smallcap : ₹500 - 5,000 cr   ← user's sweet spot
    Microcap : < ₹500 cr

Usage:
    python scripts/enrich_market_cap.py             # full universe
    python scripts/enrich_market_cap.py --limit 50  # quick test

Note: ~2s per symbol via yfinance. Full universe takes ~60-80 min.
Run weekly is plenty — market cap doesn't change daily.
"""

from __future__ import annotations

import argparse
import io
import os
import time
from datetime import datetime
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

# Thresholds in INR crores
SEGMENT_THRESHOLDS = [
    ("Largecap", 20_000),
    ("Midcap",   5_000),
    ("Smallcap", 500),
    ("Microcap", 0),
]


def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")


def get_drive():
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


def download_csv(drive, file_id):
    request = drive.files().get_media(fileId=file_id)
    fh = io.BytesIO()
    downloader = MediaIoBaseDownload(fh, request)
    done = False
    while not done:
        _, done = downloader.next_chunk()
    fh.seek(0)
    return pd.read_csv(fh)


def upload_csv(drive, folder_id, filename, df, existing_id=None):
    media = MediaIoBaseUpload(io.BytesIO(df.to_csv(index=False).encode()),
                              mimetype="text/csv", resumable=False)
    if existing_id:
        drive.files().update(fileId=existing_id, media_body=media).execute()
        return existing_id
    meta = {"name": filename, "parents": [folder_id]}
    return drive.files().create(body=meta, media_body=media, fields="id").execute()["id"]


def fetch_market_cap_inr_cr(symbol: str) -> float | None:
    """Returns market cap in INR crores, or None."""
    try:
        info = yf.Ticker(f"{symbol}.NS").info
        mc = info.get("marketCap")
        if mc is None or mc <= 0:
            return None
        return float(mc) / 1e7   # INR → crores
    except Exception:
        return None


def assign_segment(mcap_cr: float | None) -> str:
    if mcap_cr is None:
        return "Unknown"
    for label, threshold in SEGMENT_THRESHOLDS:
        if mcap_cr >= threshold:
            return label
    return "Microcap"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit number of symbols (for testing)")
    parser.add_argument("--sleep", type=float, default=0.3,
                        help="Seconds between yfinance calls (default 0.3)")
    args = parser.parse_args()

    print("Stage 8.5 — Market cap enrichment")
    print("-" * 50)

    drive = get_drive()
    folder_id = os.environ["GDRIVE_FOLDER_ID"]
    universe_id = get_or_create_subfolder(drive, folder_id, "universe")
    master_id = find_file(drive, universe_id, "master_list.csv")
    if not master_id:
        print("universe/master_list.csv not found.")
        return
    universe = download_csv(drive, master_id)
    # NSE-only (uses .NS suffix); BSE-only market cap is a later layer.
    if "exchange" in universe.columns:
        universe = universe[universe["exchange"].astype(str) == "NSE"]
    symbols = universe["symbol"].astype(str).tolist()
    if args.limit:
        symbols = symbols[:args.limit]
    log(f"Symbols to enrich: {len(symbols)}")

    # Resume from existing file if present
    existing_id = find_file(drive, universe_id, "market_cap.csv")
    if existing_id:
        existing = download_csv(drive, existing_id)
        already_done = set(existing["symbol"].astype(str))
        log(f"Resuming: {len(already_done)} symbols already enriched, "
            f"{len(symbols) - len(already_done)} to go.")
        existing_rows = existing.to_dict("records")
    else:
        already_done = set()
        existing_rows = []

    rows = list(existing_rows)
    t_start = time.time()
    todo = [s for s in symbols if s not in already_done]

    for i, sym in enumerate(todo, 1):
        mcap = fetch_market_cap_inr_cr(sym)
        seg = assign_segment(mcap)
        rows.append({"symbol": sym, "market_cap_cr": mcap, "mcap_segment": seg})
        time.sleep(args.sleep)
        if i % 25 == 0 or i == len(todo):
            elapsed = time.time() - t_start
            rate = i / elapsed if elapsed else 0
            eta = (len(todo) - i) / rate / 60 if rate else 0
            log(f"  [{i}/{len(todo)}] {sym:<14} mcap_cr={mcap}  seg={seg}  "
                f"| rate {rate:.1f}/s | ETA {eta:.1f}m")
            # Periodic save so a crash doesn't lose work
            df = pd.DataFrame(rows)
            upload_csv(drive, universe_id, "market_cap.csv", df,
                       existing_id or find_file(drive, universe_id, "market_cap.csv"))

    # Final save + summary
    df = pd.DataFrame(rows)
    upload_csv(drive, universe_id, "market_cap.csv", df,
               find_file(drive, universe_id, "market_cap.csv"))
    log(f"Saved universe/market_cap.csv ({len(df)} rows)")

    print()
    print("Segment distribution:")
    print(df["mcap_segment"].value_counts().to_string())
    print(f"\nElapsed: {(time.time()-t_start)/60:.1f} min")


if __name__ == "__main__":
    main()
