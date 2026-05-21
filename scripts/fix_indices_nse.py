"""
Stage cleanup — Fetch NIFTY MIDCAP 100 + NIFTY SMALLCAP 100 via NSE index bhavcopy.

Why this version (v2):
  - The old version used NSE's indicesHistory API (www.nseindia.com/api/...).
    That API host is aggressively anti-bot and frequently returns 0 records
    (blocked) — see the "(0 records)" runs.
  - NSE also publishes a DAILY INDEX BHAVCOPY on the archives host:
        https://archives.nseindia.com/content/indices/ind_close_all_DDMMYYYY.csv
    One CSV per trading day holds OHLC for EVERY NSE index. The archives host
    is the same one ingest_ohlcv_bhavcopy.py uses successfully — it works from
    a laptop and from GitHub Actions runners.

  yfinance tickers ^CNXMIDCAP / ^CNXSC are also broken, hence this script.
  Output schema matches ingest_indices_macro.py: (date, open, high, low, close, volume).

Behaviour:
  - If a parquet already exists for an index, fetch only days AFTER its last
    date (fast daily incremental — a handful of files).
  - If not, backfill --years worth of trading days (default 3).
  - Holidays/weekends 404 on the archives host and are skipped automatically.

Usage:
    python scripts/fix_indices_nse.py              # incremental, or 3y backfill
    python scripts/fix_indices_nse.py --years 5    # backfill window if no parquet
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

# Our parquet filename -> set of acceptable bhavcopy "Index Name" values
# (matched case-insensitively, so casing differences don't matter).
INDICES = {
    "NIFTY_MIDCAP_100":   {"nifty midcap 100"},
    "NIFTY_SMALLCAP_100": {"nifty smallcap 100"},
}

NSE_HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                   "AppleWebKit/537.36 (KHTML, like Gecko) "
                   "Chrome/120.0 Safari/537.36"),
    "Accept": "text/csv,*/*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.nseindia.com/",
}


def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")


# ---------- Drive helpers ----------

def get_drive():
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
    cs_json = os.environ.get("GDRIVE_OAUTH_CLIENT_SECRET_JSON")
    tk_json = os.environ.get("GDRIVE_OAUTH_TOKEN_JSON")
    if cs_json and tk_json:
        import json as _json
        creds = Credentials.from_authorized_user_info(_json.loads(tk_json), SCOPES)
        if creds.expired and creds.refresh_token:
            creds.refresh(Request())
        return build("drive", "v3", credentials=creds, cache_discovery=False)
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


# ---------- NSE index bhavcopy ----------

def fetch_index_bhavcopy(d: date, session: requests.Session) -> pd.DataFrame:
    """One CSV per trading day, all indices. Empty DF on holiday/weekend (404)."""
    dd = d.strftime("%d%m%Y")
    url = f"https://archives.nseindia.com/content/indices/ind_close_all_{dd}.csv"
    try:
        r = session.get(url, headers=NSE_HEADERS, timeout=30)
    except requests.RequestException as e:
        log(f"  {d}: network error — {str(e)[:120]}")
        return pd.DataFrame()
    if r.status_code == 404:
        return pd.DataFrame()
    if r.status_code != 200:
        log(f"  {d}: HTTP {r.status_code}")
        return pd.DataFrame()
    try:
        df = pd.read_csv(io.StringIO(r.text))
    except Exception as e:
        log(f"  {d}: parse error — {str(e)[:120]}")
        return pd.DataFrame()
    df.columns = [c.strip() for c in df.columns]
    if not any(c.lower() == "index name" for c in df.columns):
        return pd.DataFrame()  # not the file we expected (e.g. HTML error page)
    return df


def extract_rows(bhav: pd.DataFrame, target_date: date) -> dict[str, dict]:
    """From a day's index bhavcopy, pull one OHLC row per index we track."""
    cols = {c.lower(): c for c in bhav.columns}
    name_c = cols.get("index name")

    def find(sub):
        return next((c for c in bhav.columns if sub in c.lower()), None)

    open_c, high_c, low_c = find("open index"), find("high index"), find("low index")
    close_c = find("closing index") or find("close index")
    if not (name_c and open_c and high_c and low_c and close_c):
        return {}

    out: dict[str, dict] = {}
    norm = bhav.copy()
    norm["_name_lc"] = norm[name_c].astype(str).str.strip().str.lower()
    for our_name, accepted in INDICES.items():
        match = norm[norm["_name_lc"].isin(accepted)]
        if match.empty:
            continue
        row = match.iloc[0]
        out[our_name] = {
            "date": pd.Timestamp(target_date),
            "open": pd.to_numeric(row[open_c], errors="coerce"),
            "high": pd.to_numeric(row[high_c], errors="coerce"),
            "low": pd.to_numeric(row[low_c], errors="coerce"),
            "close": pd.to_numeric(row[close_c], errors="coerce"),
            "volume": 0,  # indices have no volume; downstream uses close only
        }
    return out


# ---------- Main ----------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--years", type=int, default=3,
                        help="Backfill window if an index has no parquet yet")
    args = parser.parse_args()

    print("Cleanup — NSE Midcap/Smallcap index fetch (bhavcopy)")
    print("-" * 50)

    today = date.today()
    backfill_start = today - timedelta(days=365 * args.years)

    drive = get_drive()
    folder_id = os.environ["GDRIVE_FOLDER_ID"]
    data_id = get_or_create_subfolder(drive, folder_id, "data")
    indices_id = get_or_create_subfolder(drive, data_id, "indices")

    # Load existing parquets; decide how far back we must fetch.
    existing: dict[str, pd.DataFrame] = {}
    existing_ids: dict[str, str] = {}
    earliest_needed = today
    for our_name in INDICES:
        fname = f"{our_name}.parquet"
        fid = find_file(drive, indices_id, fname)
        start = backfill_start
        if fid:
            existing_ids[our_name] = fid
            try:
                edf = download_parquet(drive, fid)
                if len(edf) > 5:
                    existing[our_name] = edf
                    last = pd.to_datetime(edf["date"]).max().date()
                    start = last + timedelta(days=1)
            except Exception:
                pass
        earliest_needed = min(earliest_needed, start)
        log(f"{our_name}: will fetch from {start}")

    if earliest_needed > today:
        print("-" * 50)
        print("All indices already up to date. Nothing to do.")
        return

    # Build the list of trading-day candidates (weekdays) newest-first.
    candidates = []
    d = today
    while d >= earliest_needed:
        if d.weekday() < 5:  # Mon-Fri
            candidates.append(d)
        d -= timedelta(days=1)
    log(f"Scanning {len(candidates)} candidate trading days "
        f"({earliest_needed} -> {today})")

    session = requests.Session()
    try:
        session.get("https://www.nseindia.com/", headers=NSE_HEADERS, timeout=10)
    except Exception:
        pass

    # Fetch each day's bhavcopy once; collect rows per index.
    collected: dict[str, list] = {k: [] for k in INDICES}
    fetched_days = 0
    for cd in candidates:
        bhav = fetch_index_bhavcopy(cd, session)
        if bhav.empty:
            continue
        rows = extract_rows(bhav, cd)
        if rows:
            fetched_days += 1
            for our_name, row in rows.items():
                collected[our_name].append(row)
        time.sleep(0.15)  # be polite to the archives host
        if fetched_days and fetched_days % 100 == 0:
            log(f"  ...{fetched_days} trading days fetched")

    log(f"Fetched {fetched_days} trading days of data")

    # Merge per index and upload.
    for our_name in INDICES:
        new_rows = collected[our_name]
        if not new_rows:
            log(f"{our_name}: no new rows (already current or NSE returned nothing).")
            continue
        new_df = (pd.DataFrame(new_rows)
                  .dropna(subset=["close"])
                  .sort_values("date").reset_index(drop=True))
        new_df["date"] = pd.to_datetime(new_df["date"]).dt.normalize()

        if our_name in existing:
            merged = pd.concat([existing[our_name], new_df], ignore_index=True)
            merged["date"] = pd.to_datetime(merged["date"]).dt.normalize()
            merged = (merged.drop_duplicates(subset=["date"], keep="last")
                      .sort_values("date").reset_index(drop=True))
        else:
            merged = new_df

        merged = merged[["date", "open", "high", "low", "close", "volume"]]
        upload_parquet(drive, indices_id, f"{our_name}.parquet", merged,
                       existing_ids.get(our_name))
        log(f"{our_name}: wrote {len(merged)} rows "
            f"(+{len(new_df)} new) -> data/indices/{our_name}.parquet")

    print("-" * 50)
    print("Done.")


if __name__ == "__main__":
    main()
