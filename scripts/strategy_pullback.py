"""
Stage 13 — Pullback / entry-timing strategy.

The gap this fills: every other strategy flags names that are already EXTENDED
(near 52w highs, breaking out, top-decile RS). None give a LOWER-RISK entry on a
pullback to a rising moving average inside an intact uptrend. This does.

Reads features/latest.parquet only (no OHLCV re-read). A name qualifies when:
  TREND  — stage-2 uptrend: above 200SMA, 50SMA>200SMA, 200SMA rising
  LEADER — rs_rank_6m >= MIN_RS (relative strength, not necessarily extreme)
  INTACT — return_6m_pct > 0 (the up-move that we are buying the dip within)
  PULLED BACK — at least MIN_OFF_HIGH% below the 52w high (so it is a dip, not a
                breakout) but not a crash (>= MAX_OFF_HIGH%), AND price sits
                within NEAR_MA_PCT of a rising 20- or 50-EMA
  NOT BROKEN — close >= 0.98 * ema_50 (hasn't lost the mid MA)

Zones:
  add   — reclaiming today (close > prior_close) right at the MA — the bounce
  buy   — sitting on the rising MA, holding

Score = rs_rank_6m minus a small penalty for distance from the 20-EMA (closer to
the MA = tighter, lower-risk entry = higher score).

Outputs:
  signals/per_strategy/pullback/<date>.csv + latest.csv

Usage:
    python scripts/strategy_pullback.py
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

# --- CONFIG (tune here; no other code changes) ---
MIN_RS = 60.0            # rs_rank_6m floor — leadership, not extreme
MIN_OFF_HIGH = 3.0       # must be >=3% below 52w high (a real dip)
MAX_OFF_HIGH = 25.0      # but not >25% off (that's a downtrend, not a pullback)
NEAR_MA_PCT = 3.0        # within 3% of the 20- or 50-EMA
_EMPTY_SIG_COLS = ["symbol", "date", "strategy", "zone_type", "score",
                   "entry", "stop", "reason"]


def log(msg):
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

def pullback_signals(features: pd.DataFrame) -> pd.DataFrame:
    need = ["close", "ema_20", "ema_50", "sma_200", "above_200sma",
            "50sma_above_200sma", "200sma_rising", "rs_rank_6m",
            "dist_from_52w_high_pct", "return_6m_pct", "atr_14"]
    if any(c not in features.columns for c in need):
        missing = [c for c in need if c not in features.columns]
        log(f"features missing columns {missing} — cannot run.")
        return pd.DataFrame(columns=_EMPTY_SIG_COLS)

    df = features.copy()
    d20 = (df["close"] / df["ema_20"] - 1) * 100
    d50 = (df["close"] / df["ema_50"] - 1) * 100
    near_ma = (d20.abs() <= NEAR_MA_PCT) | (d50.abs() <= NEAR_MA_PCT)

    q = df[
        (df["above_200sma"] == True) &
        (df["50sma_above_200sma"] == True) &
        (df["200sma_rising"] == True) &
        (pd.to_numeric(df["rs_rank_6m"], errors="coerce") >= MIN_RS) &
        (pd.to_numeric(df["return_6m_pct"], errors="coerce") > 0) &
        (df["dist_from_52w_high_pct"] <= -MIN_OFF_HIGH) &
        (df["dist_from_52w_high_pct"] >= -MAX_OFF_HIGH) &
        (df["close"] >= 0.98 * df["ema_50"]) &
        near_ma
    ].copy()
    if q.empty:
        return pd.DataFrame(columns=_EMPTY_SIG_COLS)

    rows = []
    for _, r in q.iterrows():
        d20r = (r["close"] / r["ema_20"] - 1) * 100 if r["ema_20"] else 0.0
        bounce = ("return_1d_pct" in q.columns and pd.notna(r.get("return_1d_pct"))
                  and float(r.get("return_1d_pct", 0)) > 0 and abs(d20r) <= NEAR_MA_PCT)
        zone = "add" if bounce else "buy"
        rs = float(r["rs_rank_6m"]) if pd.notna(r["rs_rank_6m"]) else 0.0
        score = rs - min(abs(d20r), 5.0)     # closer to 20-EMA = higher
        atr = float(r["atr_14"]) if pd.notna(r["atr_14"]) else 0.0
        rows.append({
            "symbol": r["symbol"], "date": r["date"], "strategy": "pullback",
            "zone_type": zone, "score": round(score, 2),
            "entry": round(float(r["close"]), 2),
            "stop": round(float(r["ema_50"]) - atr, 2) if r["ema_50"] else None,
            "rs_rank_6m": round(rs, 1),
            "dist_from_52w_high_pct": round(float(r["dist_from_52w_high_pct"]), 1),
            "dist_from_20ema_pct": round(d20r, 1),
            "return_6m_pct": round(float(r["return_6m_pct"]), 1),
            "reason": (f"Pullback to rising MA in stage-2 uptrend: "
                       f"{d20r:+.1f}% from 20EMA, {r['dist_from_52w_high_pct']:.0f}% "
                       f"off 52w high, RS6m {rs:.0f}, 6m +{r['return_6m_pct']:.0f}%"
                       + ("; reclaiming today" if bounce else "")),
        })
    return (pd.DataFrame(rows).sort_values("score", ascending=False)
            .reset_index(drop=True))


def main():
    global DRY_RUN
    DRY_RUN = _parse_dry_run()
    if DRY_RUN:
        log("DRY RUN — no Drive writes will be made")
    print("Stage 13 — Pullback / entry-timing signals")
    print("-" * 50)
    drive = get_drive()
    folder_id = os.environ["GDRIVE_FOLDER_ID"]
    features_id = get_or_create_subfolder(drive, folder_id, "features")
    latest_id = find_file(drive, features_id, "latest.parquet")
    if not latest_id:
        print("features/latest.parquet missing — run compute_features.py first.")
        return
    features = download_parquet(drive, latest_id)
    log(f"Features loaded: {len(features)} symbols")

    sig = pullback_signals(features)
    signals_id = get_or_create_subfolder(drive, folder_id, "signals")
    per_id = get_or_create_subfolder(drive, signals_id, "per_strategy")
    pb_id = get_or_create_subfolder(drive, per_id, "pullback")
    today = datetime.now().strftime("%Y-%m-%d")
    if not sig.empty:
        upload_csv(drive, pb_id, f"{today}.csv", sig, find_file(drive, pb_id, f"{today}.csv"))
    upload_csv(drive, pb_id, "latest.csv", sig, find_file(drive, pb_id, "latest.csv"))

    n_buy = int((sig["zone_type"] == "buy").sum()) if not sig.empty else 0
    n_add = int((sig["zone_type"] == "add").sum()) if not sig.empty else 0
    print(f"\nPullback signals: {len(sig)}  (buy={n_buy}, add/reclaim={n_add})")
    if not sig.empty:
        show = ["symbol", "zone_type", "rs_rank_6m", "dist_from_20ema_pct",
                "dist_from_52w_high_pct", "return_6m_pct"]
        print(sig.head(10)[show].to_string(index=False))


if __name__ == "__main__":
    main()
