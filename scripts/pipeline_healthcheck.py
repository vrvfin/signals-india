"""
Pipeline health-check — the truth gate.

Runs as the LAST step of the daily workflow. Verifies that the pipeline's outputs
are actually fresh and complete. Writes a health report to Drive and EXITS NON-ZERO
if anything critical is stale — which turns the GitHub Actions run RED and triggers
the failed-workflow email. So you can never silently get stale output.

Checks:
  CRITICAL (exit 1 if any fail):
    - Drive reachable (implicit — script can't run otherwise)
    - features/latest.parquet modified within last 24h AND has >= 1500 rows
    - OHLCV fresh: 5 sample blue-chips have a bar within the last 4 calendar days
  WARNING (logged, does not fail the run):
    - Each strategy's latest.csv modified within last 24h
    - data/market_state/latest.parquet date is today

Outputs:
    logs/health/health_<date>.json    — machine-readable
    logs/health/latest.json           — always-current (dashboard can read this)

Usage:
    python scripts/pipeline_healthcheck.py
"""

from __future__ import annotations

import io
import json
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload, MediaIoBaseUpload

SCOPES = ["https://www.googleapis.com/auth/drive"]
SAMPLE_SYMBOLS = ["RELIANCE", "TCS", "INFY", "HDFCBANK", "SBIN"]
# Features row gate: dynamic — 68% of the current master_list (so it scales with
# the universe instead of silently rotting like the old fixed 1500 did when the
# universe doubled), with a hard floor. Live calibration 2026-07-18: features
# = 4,086 rows vs master_list 5,616 (73%); 68% trips on a real regression while
# clearing holiday-thinned days.
MIN_FEATURE_ROWS_FLOOR = 3500
MIN_FEATURE_ROWS_PCT = 0.68
OHLCV_MAX_STALE_DAYS = 6   # 4 calendar days too tight for India's long holiday weekends
FRESH_WINDOW_HOURS = 24


def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")


# ---------- Drive helpers ----------

def get_drive():
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
    cs_json = os.environ.get("GDRIVE_OAUTH_CLIENT_SECRET_JSON")
    tk_json = os.environ.get("GDRIVE_OAUTH_TOKEN_JSON")
    if cs_json and tk_json:
        creds = Credentials.from_authorized_user_info(json.loads(tk_json), SCOPES)
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


def find_subfolder(drive, parent_id, name):
    q = (f"name='{name}' and '{parent_id}' in parents "
         f"and mimeType='application/vnd.google-apps.folder' and trashed=false")
    found = drive.files().list(q=q, fields="files(id)").execute().get("files", [])
    return found[0]["id"] if found else None


def list_subfolders(drive, parent_id):
    q = (f"'{parent_id}' in parents and "
         f"mimeType='application/vnd.google-apps.folder' and trashed=false")
    return drive.files().list(q=q, fields="files(id,name)").execute().get("files", [])


def get_file_meta(drive, folder_id, name):
    """Return (id, modifiedTime) for a file, or (None, None)."""
    q = f"name='{name}' and '{folder_id}' in parents and trashed=false"
    found = drive.files().list(
        q=q, fields="files(id,name,modifiedTime)").execute().get("files", [])
    if not found:
        return None, None
    return found[0]["id"], found[0]["modifiedTime"]


def download_parquet(drive, file_id):
    request = drive.files().get_media(fileId=file_id)
    fh = io.BytesIO()
    d = MediaIoBaseDownload(fh, request)
    done = False
    while not done:
        _, done = d.next_chunk()
    fh.seek(0)
    return pd.read_parquet(fh)


def upload_json(drive, folder_id, filename, obj, existing_id=None):
    body = json.dumps(obj, indent=2, default=str).encode()
    media = MediaIoBaseUpload(io.BytesIO(body), mimetype="application/json",
                              resumable=False)
    if existing_id:
        drive.files().update(fileId=existing_id, media_body=media).execute()
        return existing_id
    meta = {"name": filename, "parents": [folder_id]}
    return drive.files().create(body=meta, media_body=media, fields="id").execute()["id"]


# ---------- Checks ----------

def hours_since(modified_time_iso: str) -> float:
    dt = datetime.fromisoformat(modified_time_iso.replace("Z", "+00:00"))
    return (datetime.now(timezone.utc) - dt).total_seconds() / 3600


def main():
    print("Pipeline Health-Check")
    print("=" * 56)

    report = {
        "run_at": datetime.now(timezone.utc).isoformat(),
        "checks": [],
        "critical_failures": 0,
        "warnings": 0,
    }

    def record(name, level, ok, detail):
        report["checks"].append(
            {"check": name, "level": level, "status": "PASS" if ok else "FAIL",
             "detail": detail})
        mark = "PASS" if ok else "FAIL"
        print(f"  [{mark}] ({level:<8}) {name}: {detail}")
        if not ok:
            if level == "CRITICAL":
                report["critical_failures"] += 1
            else:
                report["warnings"] += 1

    try:
        drive = get_drive()
    except Exception as e:
        print(f"  [FAIL] (CRITICAL) Drive auth: {str(e)[:160]}")
        print("\n>>> OAuth/Drive auth failed. See runbook in chat for token refresh.")
        sys.exit(1)

    folder_id = os.environ["GDRIVE_FOLDER_ID"]

    # Dynamic features-row threshold from the current universe size (68%,
    # floored). Falls back to the floor if master_list is unreadable.
    min_feature_rows = MIN_FEATURE_ROWS_FLOOR
    try:
        uni_id = find_subfolder(drive, folder_id, "universe")
        if uni_id:
            mid, _ = get_file_meta(drive, uni_id, "master_list.csv")
            if mid:
                req = drive.files().get_media(fileId=mid)
                fh = io.BytesIO()
                dl = MediaIoBaseDownload(fh, req)
                done = False
                while not done:
                    _, done = dl.next_chunk()
                fh.seek(0)
                n_uni = len(pd.read_csv(fh))
                min_feature_rows = max(MIN_FEATURE_ROWS_FLOOR,
                                       int(n_uni * MIN_FEATURE_ROWS_PCT))
                log(f"features row gate: >={min_feature_rows} "
                    f"(68% of universe {n_uni}, floor {MIN_FEATURE_ROWS_FLOOR})")
    except Exception as e:
        log(f"universe count unavailable ({str(e)[:60]}) — "
            f"using floor {MIN_FEATURE_ROWS_FLOOR}")

    # --- features/latest.parquet ---
    feat_id = find_subfolder(drive, folder_id, "features")
    if not feat_id:
        record("features folder", "CRITICAL", False, "features/ folder missing")
    else:
        fid, mtime = get_file_meta(drive, feat_id, "latest.parquet")
        if not fid:
            record("features/latest.parquet", "CRITICAL", False, "file missing")
        else:
            age_h = hours_since(mtime)
            fresh = age_h <= FRESH_WINDOW_HOURS
            try:
                fdf = download_parquet(drive, fid)
                nrows = len(fdf)
            except Exception as e:
                nrows = -1
            ok = fresh and nrows >= min_feature_rows
            record("features/latest.parquet", "CRITICAL", ok,
                   f"age {age_h:.1f}h, {nrows} rows "
                   f"(need <{FRESH_WINDOW_HOURS}h & >={min_feature_rows} rows)")

            # OHLCV freshness via feature 'date' column on sample symbols
            if nrows > 0 and "date" in fdf.columns:
                fdf["date"] = pd.to_datetime(fdf["date"], errors="coerce")
                sample = fdf[fdf["symbol"].isin(SAMPLE_SYMBOLS)]
                if sample.empty:
                    record("OHLCV freshness", "CRITICAL", False,
                           "no sample symbols found in features")
                else:
                    latest_bar = sample["date"].max()
                    # Normalize to date-only so timezone-aware vs naive never raises TypeError.
                    # pd.Timestamp.date() strips any tz; datetime.now(utc).date() is UTC date.
                    try:
                        latest_date = pd.Timestamp(latest_bar).date()
                    except Exception:
                        latest_date = pd.to_datetime(latest_bar).dt.date.max()
                    stale_days = (datetime.now(timezone.utc).date() - latest_date).days
                    ok = stale_days <= OHLCV_MAX_STALE_DAYS
                    record("OHLCV freshness", "CRITICAL", ok,
                           f"latest bar {latest_date} "
                           f"({stale_days}d old, max {OHLCV_MAX_STALE_DAYS}d)")

    # --- per-strategy signal freshness (CRITICAL: a stale strategy would
    # otherwise ship yesterday's signals as today's — the aggregator now skips
    # them, and this gate turns the run RED so it cannot go unnoticed).
    # Strategies write latest.csv even when they have zero signals, so a
    # legitimately quiet day stays green. ---
    signals_id = find_subfolder(drive, folder_id, "signals")
    if signals_id:
        per_strat_id = find_subfolder(drive, signals_id, "per_strategy")
        if per_strat_id:
            for sub in list_subfolders(drive, per_strat_id):
                _, mtime = get_file_meta(drive, sub["id"], "latest.csv")
                if not mtime:
                    record(f"strategy:{sub['name']}", "CRITICAL", False,
                           "latest.csv missing")
                else:
                    age_h = hours_since(mtime)
                    record(f"strategy:{sub['name']}", "CRITICAL",
                           age_h <= FRESH_WINDOW_HOURS,
                           f"latest.csv age {age_h:.1f}h")

    # --- market_state freshness (WARNING) ---
    data_id = find_subfolder(drive, folder_id, "data")
    if data_id:
        ms_id = find_subfolder(drive, data_id, "market_state")
        if ms_id:
            _, mtime = get_file_meta(drive, ms_id, "latest.parquet")
            if mtime:
                age_h = hours_since(mtime)
                record("market_state/latest.parquet", "WARNING",
                       age_h <= FRESH_WINDOW_HOURS, f"age {age_h:.1f}h")
            else:
                record("market_state/latest.parquet", "WARNING", False, "missing")
    # --- indices freshness (WARNING) ---
    if data_id:
        idx_id = find_subfolder(drive, data_id, "indices")
        if idx_id:
            for idx_name in ["NIFTY_MIDCAP_100", "NIFTY_SMALLCAP_100"]:
                _, mtime = get_file_meta(drive, idx_id, f"{idx_name}.parquet")
                if not mtime:
                    record(f"index:{idx_name}", "WARNING", False,
                           f"{idx_name}.parquet missing")
                else:
                    age_h = hours_since(mtime)
                    record(f"index:{idx_name}", "WARNING",
                           age_h <= FRESH_WINDOW_HOURS,
                           f"{idx_name}.parquet age {age_h:.1f}h")
                    
    # --- Write report to Drive ---
    report["overall"] = ("FAIL" if report["critical_failures"] > 0
                         else ("DEGRADED" if report["warnings"] > 0 else "HEALTHY"))
    logs_id = get_or_create_subfolder(drive, folder_id, "logs")
    health_id = get_or_create_subfolder(drive, logs_id, "health")
    today = datetime.now().strftime("%Y-%m-%d")
    fname = f"health_{today}.json"
    ex_id, _ = get_file_meta(drive, health_id, fname)
    upload_json(drive, health_id, fname, report, ex_id)
    latest_ex_id, _ = get_file_meta(drive, health_id, "latest.json")
    upload_json(drive, health_id, "latest.json", report, latest_ex_id)

    print("=" * 56)
    print(f"  OVERALL: {report['overall']}  "
          f"(critical={report['critical_failures']}, warnings={report['warnings']})")
    print(f"  Report: logs/health/{fname}")
    print("=" * 56)

    if report["critical_failures"] > 0:
        print("\n>>> CRITICAL failures above. Workflow will be marked RED.")
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
