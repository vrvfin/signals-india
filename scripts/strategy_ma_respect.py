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
import sys
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

# Written on zero-signal days so latest.csv stays fresh (healthcheck treats a
# stale latest.csv as CRITICAL; the aggregator skips empty files cleanly).
_EMPTY_SIG_COLS = ["symbol", "date", "strategy", "zone_type", "score",
                   "entry", "stop", "reason"]


# Console encoding. Several scripts here log the rupee sign, a delta or an em
# dash, and a Windows console is cp1252 — so a run could complete all its work
# and then die in a log line. It cost three separate crashes before being fixed
# in one place. Degrade the characters, never the run.
try:
    sys.stdout.reconfigure(errors="replace")
    sys.stderr.reconfigure(errors="replace")
except Exception:          # pragma: no cover - not every stream supports it
    pass

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


# ── --dry-run guard ──────────────────────────────────────────────────────────
# This script writes signal files straight to Drive. Its scoring logic changed,
# and house rule 6 says a dry-run must be confirmed before any live run — so the
# upload is wrapped rather than the call sites edited. Set by _parse_dry_run().
DRY_RUN = False
_live_upload_csv = upload_csv


def upload_csv(drive, folder_id, filename, df, existing_id=None):   # noqa: F811
    if DRY_RUN:
        log(f"[DRY RUN] would write {filename} ({len(df)} rows)")
        return existing_id
    return _live_upload_csv(drive, folder_id, filename, df, existing_id)


def _parse_dry_run() -> bool:
    """--dry-run only; kept deliberately separate from any argparse the script
    already has, so adding it cannot disturb existing flags."""
    import argparse as _ap
    p = _ap.ArgumentParser(add_help=False)
    p.add_argument("--dry-run", action="store_true",
                   help="compute and report, write nothing to Drive")
    known, _ = p.parse_known_args()
    return known.dry_run


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
    df["entry"] = df["close"].round(2)
    # Stop-loss: 1 ATR below the EMA (first-violation guard)
    df["stop"] = (df[ema_col] - df["atr_14"]).round(2)

    # Was `df[days_col]` — longest streak wins. That ranked the MOST EXTENDED
    # name top: a stock 200 days above its 20 EMA is further into its move and
    # more likely to revert than one at 35 days, and it is also the one whose
    # stop sits furthest below the price. Two changes:
    #   risk      how far the stop is, as a fraction of price. Nearer is better,
    #             because it is what lets you size the position.
    #   streak    credit for CLEARING the bar, capped at twice the threshold.
    #             Meeting it comfortably counts; running on forever does not.
    _risk = ((df["close"] - df["stop"]) / df["close"]).clip(lower=0)
    _risk_score = (1.0 - (_risk / 0.15).clip(upper=1.0))     # 15% away = no marks
    _streak_score = ((df[days_col] - days_n) / max(days_n, 1)).clip(lower=0, upper=1)
    df["score"] = (100 * (0.6 * _risk_score + 0.4 * _streak_score)).round(2)
    df["risk_pct"] = (_risk * 100).round(2)
    df["reason"] = df.apply(
        lambda r: f"Held above {ema_n} EMA for {int(r[days_col])} days "
                  f"(>= {days_n}d threshold), ADR {r['adr_pct_20']:.1f}%",
        axis=1)

    keep = ["symbol", "date", "strategy", "zone_type", "score",
            "entry", "stop", "risk_pct", days_col, ema_col, "close",
            "adr_pct_20", "dist_from_52w_high_pct", "reason"]
    out = df[keep].copy()
    out = out.rename(columns={days_col: "days_above_ema", ema_col: "ema"})
    return out.sort_values("score", ascending=False).reset_index(drop=True)


# ---------- Main ----------

def main() -> None:
    global DRY_RUN
    DRY_RUN = _parse_dry_run()
    if DRY_RUN:
        log("DRY RUN — no Drive writes will be made")
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
            sub_id = get_or_create_subfolder(drive, per_strategy_id, inst["name"])
            upload_csv(drive, sub_id, "latest.csv",
                       pd.DataFrame(columns=_EMPTY_SIG_COLS),
                       find_file(drive, sub_id, "latest.csv"))
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
