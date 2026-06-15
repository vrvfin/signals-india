"""
Stage 11c — PEAD (Post-Earnings Announcement Drift) strategy.

Detects stocks that gapped up on earnings day on heavy volume, and are still in
the typical 30-60 day drift window. Outputs BUY for fresh post-earnings movers
and HOLD for the same names continuing to drift higher.

Rules:
  - Identify earnings day = first trading day within ±2 days of latest_quarter_label.
  - Earnings day gap (open vs prior close) ≥ 5%.
  - Earnings day volume ≥ 2× 20-day average volume.
  - Days since earnings between 1 and 60 (in the drift window).
  - Stock still up from earnings-day close.

Zones:
  add   — earnings was within last 5 trading days (fresh signal)
  buy   — earnings was 6-30 days ago, stock still above earnings-day close
  hold  — earnings was 31-60 days ago, stock still above earnings-day close

Score = earnings-day gap × volume multiple.

Outputs:
  signals/per_strategy/pead/<date>.csv
  signals/per_strategy/pead/latest.csv
"""

from __future__ import annotations

import argparse
import io
import os
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload, MediaIoBaseUpload

SCOPES = ["https://www.googleapis.com/auth/drive"]

MIN_GAP_PCT = 5.0
MIN_VOL_MULT = 2.0


def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")


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


# Per-thread Drive client (googleapiclient http transport isn't thread-safe).
_tl = threading.local()


def _thread_drive():
    d = getattr(_tl, "drive", None)
    if d is None:
        d = get_drive()
        _tl.drive = d
    return d


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


def download_parquet(drive, file_id):
    request = drive.files().get_media(fileId=file_id)
    fh = io.BytesIO()
    d = MediaIoBaseDownload(fh, request)
    done = False
    while not done:
        _, done = d.next_chunk()
    fh.seek(0)
    return pd.read_parquet(fh)


def upload_csv(drive, folder_id, filename, df, existing_id=None):
    media = MediaIoBaseUpload(io.BytesIO(df.to_csv(index=False).encode()),
                              mimetype="text/csv", resumable=False)
    if existing_id:
        drive.files().update(fileId=existing_id, media_body=media).execute()
        return existing_id
    meta = {"name": filename, "parents": [folder_id]}
    return drive.files().create(body=meta, media_body=media, fields="id").execute()["id"]


_MONTHS = {"Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4, "May": 5, "Jun": 6,
           "Jul": 7, "Aug": 8, "Sep": 9, "Oct": 10, "Nov": 11, "Dec": 12}


def parse_quarter_label(label) -> pd.Timestamp | None:
    """Screener labels look like 'Mar 2026', 'Jun 2025'. Map to end-of-month date."""
    if not isinstance(label, str):
        return None
    m = re.match(r"^\s*([A-Za-z]{3})\s+(\d{4})\s*$", label)
    if not m:
        return None
    mon, yr = m.group(1)[:3].title(), int(m.group(2))
    if mon not in _MONTHS:
        return None
    # End-of-quarter month. Most results are announced 30-60 days later.
    return pd.Timestamp(year=yr, month=_MONTHS[mon], day=28)


def detect_pead(symbol, ohlcv, qtr_end_dt, today_dt):
    """Find the earnings day (first volume spike + gap after qtr_end_dt) within +60 days."""
    df = ohlcv.sort_values("date").reset_index(drop=True).copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df[df["date"] >= qtr_end_dt]
    if df.empty or len(df) < 2:
        return None
    df["prev_close"] = df["close"].shift()
    df["gap_pct"] = (df["open"] / df["prev_close"] - 1) * 100
    df["vol_20avg"] = df["volume"].rolling(20, min_periods=5).mean()
    df["vol_mult"] = df["volume"] / df["vol_20avg"]

    qualifying = df[
        (df["gap_pct"].abs() >= MIN_GAP_PCT) &
        (df["vol_mult"] >= MIN_VOL_MULT) &
        df["gap_pct"].notna() & df["vol_mult"].notna()
    ]
    if qualifying.empty:
        return None
    er = qualifying.iloc[0]
    er_date = pd.to_datetime(er["date"])
    days_since = (today_dt - er_date).days
    if days_since < 1 or days_since > 60:
        return None
    today_close = float(df["close"].iloc[-1])
    er_close = float(er["close"])
    drift_pct = (today_close / er_close - 1) * 100
    if er["gap_pct"] > 0 and today_close < er_close:
        return None  # gave back the gain — skip
    return {
        "earnings_date": er_date.date(),
        "earnings_gap_pct": float(er["gap_pct"]),
        "earnings_vol_mult": float(er["vol_mult"]),
        "earnings_close": er_close,
        "today_close": today_close,
        "days_since_er": days_since,
        "drift_pct": drift_pct,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8,
                    help="Parallel OHLCV-download workers (default 8). Reads are "
                         "independent per symbol, results collected in the main "
                         "thread — data-safe; bounded for Drive API limits. Use 1 "
                         "for the serial path.")
    args = ap.parse_args()

    print("Stage 11c — PEAD signals")
    print("-" * 50)
    drive = get_drive()
    folder_id = os.environ["GDRIVE_FOLDER_ID"]

    fund_id = get_or_create_subfolder(drive, folder_id, "fundamentals")
    summary_fid = find_file(drive, fund_id, "summary.parquet")
    if not summary_fid:
        print("fundamentals/summary.parquet missing. Run `ingest_fundamentals.py` first.")
        return
    fund = download_parquet(drive, summary_fid)
    log(f"Fundamentals loaded: {len(fund)}")

    data_id = get_or_create_subfolder(drive, folder_id, "data")
    ohlcv_id = get_or_create_subfolder(drive, data_id, "ohlcv")
    ohlcv_files = list_files_in_folder(drive, ohlcv_id)
    log(f"OHLCV parquets available: {len(ohlcv_files)}")

    today_dt = pd.Timestamp(datetime.now().date())

    # Cheap pre-filter (no Drive I/O): only names with a quarter-end in the last
    # 90 days AND an OHLCV parquet on Drive. The download (the cost) happens only
    # for these, parallelized below.
    work = []
    for _, r in fund.iterrows():
        sym = r["symbol"]
        qtr_end = parse_quarter_label(r.get("latest_quarter_label"))
        if qtr_end is None or (today_dt - qtr_end).days > 90:
            continue
        fid = ohlcv_files.get(f"{sym}.parquet")
        if fid:
            work.append((sym, qtr_end, fid))

    workers = max(1, args.workers)
    log(f"PEAD candidates (recent quarter + OHLCV): {len(work)} | workers: {workers}")

    def _scan(item):
        sym, qtr_end, fid = item
        d = drive if workers == 1 else _thread_drive()
        try:
            ohlcv = download_parquet(d, fid)
            return sym, qtr_end, detect_pead(sym, ohlcv, qtr_end, today_dt)
        except Exception:
            return sym, qtr_end, None

    if workers == 1:
        scanned = [_scan(it) for it in work]
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            scanned = [f.result() for f in as_completed([pool.submit(_scan, it) for it in work])]

    rows = []
    for sym, qtr_end, pead in scanned:
        if pead is None:
            continue

        days = pead["days_since_er"]
        if days <= 5:
            zone = "add"
        elif days <= 30:
            zone = "buy"
        else:
            zone = "hold"

        if pead["earnings_gap_pct"] < 0:
            continue  # only long-side PEAD for v1

        score = pead["earnings_gap_pct"] * pead["earnings_vol_mult"]
        rows.append({
            "symbol": sym,
            "date": today_dt.strftime("%Y-%m-%d"),
            "strategy": "pead",
            "zone_type": zone,
            "score": round(score, 2),
            "entry": round(pead["today_close"], 2),
            "stop": round(pead["earnings_close"] * 0.95, 2),
            "earnings_date": pead["earnings_date"],
            "earnings_gap_pct": round(pead["earnings_gap_pct"], 2),
            "earnings_vol_mult": round(pead["earnings_vol_mult"], 2),
            "days_since_er": days,
            "drift_since_er_pct": round(pead["drift_pct"], 2),
            "reason": (f"Earnings {pead['earnings_date']}: "
                       f"+{pead['earnings_gap_pct']:.1f}% gap on "
                       f"{pead['earnings_vol_mult']:.1f}× vol; "
                       f"{days}d into drift, {pead['drift_pct']:+.1f}% since"),
        })

    sig_df = pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)
    log(f"PEAD signals: {len(sig_df)}")

    signals_id = get_or_create_subfolder(drive, folder_id, "signals")
    per_strat_id = get_or_create_subfolder(drive, signals_id, "per_strategy")
    pd_id = get_or_create_subfolder(drive, per_strat_id, "pead")
    today_str = today_dt.strftime("%Y-%m-%d")
    upload_csv(drive, pd_id, f"{today_str}.csv", sig_df,
               find_file(drive, pd_id, f"{today_str}.csv"))
    upload_csv(drive, pd_id, "latest.csv", sig_df,
               find_file(drive, pd_id, "latest.csv"))
    log("Saved signals/per_strategy/pead/")

    if not sig_df.empty:
        print("\nTop 10:")
        show = ["symbol", "zone_type", "earnings_date", "earnings_gap_pct",
                "earnings_vol_mult", "days_since_er", "drift_since_er_pct"]
        show = [c for c in show if c in sig_df.columns]
        print(sig_df.head(10)[show].to_string(index=False))


if __name__ == "__main__":
    main()
