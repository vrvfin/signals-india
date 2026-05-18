"""
Stage 5 — MA-respect strategy family.

A stock "respects" an EMA if its daily close has stayed above that EMA for at least
K consecutive trading days, without a single violating close. Each (EMA, K) pair is
its own strategy.

Per DESIGN §1.5:
  buy       — stock has held above EMA_N for ≥ K consecutive days (current state)
  add       — pullback to the EMA without violating it, then close > prior high
              (deferred to v2 — needs intraday-style pattern detection)
  stop_loss — close > 1 ATR below the EMA (first violation)
  exit      — close below the next higher EMA (e.g., 50 EMA for a 20 EMA strategy)

v1 ships `buy` and `stop_loss` zones. `add` deferred.

Instances (configurable later via config.yaml):
  ma_respect_20ema_30d
  ma_respect_20ema_60d
  ma_respect_50ema_60d

Outputs:
  signals/per_strategy/ma_respect_<NAME>/<date>.csv
  signals/per_strategy/ma_respect_<NAME>/latest.csv
  signals/per_strategy/ma_respect_combined_latest.csv
"""

from __future__ import annotations

import io
import os
from datetime import datetime
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload, MediaIoBaseUpload

SCOPES = ["https://www.googleapis.com/auth/drive"]

INSTANCES = [
    {"name": "ma_respect_20ema_30d", "ema": 20, "days": 30},
    {"name": "ma_respect_20ema_60d", "ema": 20, "days": 60},
    {"name": "ma_respect_50ema_60d", "ema": 50, "days": 60},
]


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")


# ---------- Drive helpers ----------

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


def download_parquet(drive, file_id):
    request = drive.files().get_media(fileId=file_id)
    fh = io.BytesIO()
    downloader = MediaIoBaseDownload(fh, request)
    done = False
    while not done:
        _, done = downloader.next_chunk()
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


# ---------- Strategy ----------

def ma_respect_signals(features: pd.DataFrame, ema_n: int, days_n: int,
                       strategy_name: str) -> pd.DataFrame:
    """A stock qualifies if days_above_ema_<N> >= days_n."""
    days_col = f"days_above_ema_{ema_n}"
    ema_col = f"ema_{ema_n}"
    if days_col not in features.columns or ema_col not in features.columns:
        return pd.DataFrame()

    df = features.copy()
    df = df[df[days_col].fillna(0) >= days_n].copy()
    if df.empty:
        return df

    df["strategy"] = strategy_name
    df["zone_type"] = "buy"   # current state: holding above the EMA
    df["score"] = df[days_col]  # longer streak = higher conviction
    df["entry"] = df["close"].round(2)
    # Stop-loss: 1 ATR below the EMA (first-violation guard)
    df["stop"] = (df[ema_col] - df["atr_14"]).round(2)
    df["reason"] = df.apply(
        lambda r: f"Held above {ema_n} EMA for {int(r[days_col])} days "
                  f"(>= {days_n}d threshold), ADR {r['adr_pct_20']:.1f}%",
        axis=1)

    keep = ["symbol", "date", "strategy", "zone_type", "score",
            "entry", "stop", days_col, ema_col, "close",
            "adr_pct_20", "dist_from_52w_high_pct", "reason"]
    out = df[keep].copy()
    out = out.rename(columns={days_col: "days_above_ema", ema_col: "ema"})
    return out.sort_values("score", ascending=False).reset_index(drop=True)


# ---------- Main ----------

def main() -> None:
    print("Stage 5 — MA-respect signals")
    print("-" * 50)

    drive = get_drive()
    folder_id = os.environ["GDRIVE_FOLDER_ID"]

    # Load features
    features_id = get_or_create_subfolder(drive, folder_id, "features")
    latest_id = find_file(drive, features_id, "latest.parquet")
    if not latest_id:
        print("features/latest.parquet missing — run compute_features.py first.")
        return
    features = download_parquet(drive, latest_id)
    log(f"Features loaded: {len(features)} symbols")

    signals_id = get_or_create_subfolder(drive, folder_id, "signals")
    per_strategy_id = get_or_create_subfolder(drive, signals_id, "per_strategy")
    today_str = datetime.now().strftime("%Y-%m-%d")

    combined_rows = []
    print()
    print(f"{'Strategy':<26} {'BUY':>5}  Top 3 by streak length")
    print("-" * 80)
    for inst in INSTANCES:
        sig = ma_respect_signals(features, inst["ema"], inst["days"], inst["name"])
        if sig.empty:
            print(f"{inst['name']:<26}     0  (no signals)")
            continue

        top3 = sig.nlargest(3, "score")[["symbol", "days_above_ema"]].values.tolist()
        top3_str = ", ".join([f"{s}({int(d)}d)" for s, d in top3])
        print(f"{inst['name']:<26} {len(sig):>5}  {top3_str}")

        sub_id = get_or_create_subfolder(drive, per_strategy_id, inst["name"])
        upload_csv(drive, sub_id, f"{today_str}.csv", sig,
                   find_file(drive, sub_id, f"{today_str}.csv"))
        upload_csv(drive, sub_id, "latest.csv", sig,
                   find_file(drive, sub_id, "latest.csv"))

        combined_rows.append(sig)

    if combined_rows:
        combined = pd.concat(combined_rows, ignore_index=True)
        upload_csv(drive, per_strategy_id, "ma_respect_combined_latest.csv", combined,
                   find_file(drive, per_strategy_id, "ma_respect_combined_latest.csv"))
        log(f"Wrote ma_respect_combined_latest.csv ({len(combined)} rows total)")

    print("-" * 80)
    print("Saved to signals/per_strategy/ma_respect_*/")


if __name__ == "__main__":
    main()
