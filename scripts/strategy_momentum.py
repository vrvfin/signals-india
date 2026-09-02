"""
Stage 4 — Momentum strategy family (1M / 2M / 3M / 6M / 12M).

Reads features/latest.parquet. For each lookback, produces zone signals per the
rules in DESIGN.md §1.5:

  buy   — top decile of RS rank AND price > 200 SMA
  add   — buy AND within 5% of 52-week high (high-conviction continuation)
  hold  — top quintile (80-90 percentile) AND price > 200 SMA

Outputs (per timeframe):
  signals/per_strategy/momentum_<label>/<date>.csv
  signals/per_strategy/momentum_<label>/latest.csv

Plus a combined `signals/per_strategy/momentum_combined_latest.csv` listing every
signal from every timeframe.

Usage:
    python scripts/strategy_momentum.py
"""

from __future__ import annotations

import io
import os
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

_SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)
from strategy_common import pct_rank
from dotenv import load_dotenv
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload, MediaIoBaseUpload

SCOPES = ["https://www.googleapis.com/auth/drive"]
LOOKBACKS = ["1m", "2m", "3m", "6m", "12m"]

# Written on zero-signal days so latest.csv stays fresh (healthcheck treats a
# stale latest.csv as CRITICAL; the aggregator skips empty files cleanly).
_EMPTY_SIG_COLS = ["symbol", "date", "strategy", "zone_type", "score",
                   "entry", "stop", "reason"]


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

def momentum_signals(features: pd.DataFrame, lookback: str,
                     top_pct: float = 10.0, near_high_pct: float = 5.0,
                     hold_pct: float = 20.0) -> pd.DataFrame:
    """
    Generate momentum signals for one timeframe.

      buy  : rs_rank >= (100 - top_pct)  AND above_200sma
      add  : buy  AND  dist_from_52w_high_pct >= -near_high_pct
      hold : rs_rank in [(100 - hold_pct), (100 - top_pct))  AND above_200sma
    """
    rs_col = f"rs_rank_{lookback}"
    ret_col = f"return_{lookback}_pct"
    exc_col = f"rs_vs_nifty500_{lookback}_pct"
    if rs_col not in features.columns:
        return pd.DataFrame()

    df = features.copy()
    df = df[df["above_200sma"] == True]  # trend filter

    # Gate on strength RELATIVE TO THE INDEX, not raw return percentile.
    # guru/backtest/family_lift.parquet, 3M: raw momentum (TECH_MOM) is +38.2pp
    # on a 55.6% win rate with a +3.4% median; index-relative
    # (TECH_RELSTRENGTH) is +38.0pp on a 72.6% win rate with a +14.8% median.
    # Nearly the same lift, a far better hit rate and four times the median.
    # rs_vs_nifty500 is an excess return in pp, not a percentile, so it is
    # ranked cross-sectionally here to keep the "top decile" semantics the
    # thresholds below are written in.
    if exc_col in df.columns and df[exc_col].notna().any():
        df["rs_basis"] = pct_rank(df[exc_col])
        rs_basis_name = "index-relative"
    else:
        # compute_features has not shipped the excess columns yet, or the index
        # is unavailable — fall back to the raw percentile rather than emitting
        # nothing, and say so in the log.
        df["rs_basis"] = df[rs_col]
        rs_basis_name = "raw-return (FALLBACK: no index-relative column)"
    df = df[df["rs_basis"].notna()]
    log(f"  {lookback}: gating on {rs_basis_name}")

    buy_thresh = 100 - top_pct          # 90 by default
    hold_thresh = 100 - hold_pct        # 80 by default

    is_buy = df["rs_basis"] >= buy_thresh
    is_hold = (df["rs_basis"] >= hold_thresh) & (df["rs_basis"] < buy_thresh)
    is_add = is_buy & (df["dist_from_52w_high_pct"] >= -near_high_pct)

    df["zone_type"] = None
    df.loc[is_hold, "zone_type"] = "hold"
    df.loc[is_buy, "zone_type"] = "buy"
    df.loc[is_add, "zone_type"] = "add"

    df = df[df["zone_type"].notna()].copy()
    if df.empty:
        return pd.DataFrame()

    df["strategy"] = f"momentum_{lookback}"
    # The variant is kept as its own column so aggregate_signals can collapse the
    # five lookbacks into ONE family vote while still knowing which of them
    # fired — a name in 1m only is a fresh mover, a name in all five is an
    # established leader, and they are not the same trade.
    df["variant"] = lookback
    # Score on the index-relative percentile, not the raw one that also gates.
    df["score"] = df["rs_basis"].round(2)
    df["entry"] = df["close"].round(2)
    df["stop"] = (df["close"] - 2 * df["atr_14"]).round(2)
    df["reason"] = df.apply(
        lambda r: f"RS rank {r[rs_col]:.0f} ({lookback}), "
                  f"return {r[ret_col]:.1f}%, "
                  f"{r['dist_from_52w_high_pct']:.1f}% from 52w high",
        axis=1)

    cols = ["symbol", "date", "strategy", "variant", "zone_type", "score",
            "entry", "stop", rs_col, ret_col, "dist_from_52w_high_pct",
            "adr_pct_20", "reason"]
    if exc_col in df.columns:
        df["rs_excess_pct"] = df[exc_col].round(2)
        cols.insert(cols.index("adr_pct_20"), "rs_excess_pct")
    out = df[cols].copy()
    out = out.rename(columns={rs_col: "rs_rank", ret_col: "return_pct"})
    return out.sort_values("score", ascending=False).reset_index(drop=True)


# ---------- Main ----------

def main() -> None:
    print("Stage 4 — Momentum signals")
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
    log(f"Features loaded: {len(features)} symbols, {len(features.columns)} columns")

    # Run each lookback
    signals_id = get_or_create_subfolder(drive, folder_id, "signals")
    per_strategy_id = get_or_create_subfolder(drive, signals_id, "per_strategy")
    today_str = datetime.now().strftime("%Y-%m-%d")

    combined_rows = []
    print()
    print(f"{'Strategy':<16} {'BUY':>5} {'ADD':>5} {'HOLD':>5}  Top 3 by RS rank")
    print("-" * 72)
    for lb in LOOKBACKS:
        sig = momentum_signals(features, lb)
        if sig.empty:
            print(f"momentum_{lb:<8}      0     0     0  (no signals)")
            sub_id = get_or_create_subfolder(drive, per_strategy_id, f"momentum_{lb}")
            upload_csv(drive, sub_id, "latest.csv",
                       pd.DataFrame(columns=_EMPTY_SIG_COLS),
                       find_file(drive, sub_id, "latest.csv"))
            continue

        n_buy = (sig["zone_type"] == "buy").sum()
        n_add = (sig["zone_type"] == "add").sum()
        n_hold = (sig["zone_type"] == "hold").sum()
        top3 = sig.nlargest(3, "score")["symbol"].tolist()
        print(f"momentum_{lb:<8} {n_buy:>5} {n_add:>5} {n_hold:>5}  "
              f"{', '.join(top3)}")

        # Save to Drive
        sub_id = get_or_create_subfolder(drive, per_strategy_id, f"momentum_{lb}")
        upload_csv(drive, sub_id, f"{today_str}.csv", sig,
                   find_file(drive, sub_id, f"{today_str}.csv"))
        upload_csv(drive, sub_id, "latest.csv", sig,
                   find_file(drive, sub_id, "latest.csv"))

        combined_rows.append(sig)

    # Combined latest
    if combined_rows:
        combined = pd.concat(combined_rows, ignore_index=True)
        upload_csv(drive, per_strategy_id, "momentum_combined_latest.csv", combined,
                   find_file(drive, per_strategy_id, "momentum_combined_latest.csv"))
        log(f"Wrote momentum_combined_latest.csv ({len(combined)} rows total)")

    print("-" * 72)
    print(f"Saved to signals/per_strategy/momentum_*/")


if __name__ == "__main__":
    main()
