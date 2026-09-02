"""
Stage 12 — Volume strategy family (volume_breakout + volume_vcp).

Reads features/latest.parquet. Produces TWO independent signal sets from one
file (same pattern as strategy_momentum.py producing 1m..12m):

  volume_breakout — a volume SURGE confirming accumulation.
      The stock trades on far more volume than usual while in an uptrend and
      not far from its 52-week high. A big-volume up-move is the footprint of
      institutional buying.

  volume_vcp — a volume DRY-UP inside a tight base near the highs.
      The opposite tell: volume contracts and the daily range tightens while
      the stock holds near its highs — the classic pre-breakout coil
      (Volatility Contraction Pattern). The actual breakout is then caught by
      volume_breakout on a later day.

Outputs:
  signals/per_strategy/volume_breakout/<date>.csv  + latest.csv
  signals/per_strategy/volume_vcp/<date>.csv       + latest.csv
  signals/per_strategy/volume_combined_latest.csv

Usage:
    python scripts/strategy_volume.py

To add or tune thresholds, edit the CONFIG block below — no other code changes.
"""

from __future__ import annotations

import io
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload, MediaIoBaseUpload

SCOPES = ["https://www.googleapis.com/auth/drive"]

# ============================================================
# TUNABLE CONFIG — edit these numbers, nothing else needs to change.
# ============================================================
MIN_PRICE = 10.0            # drop stocks priced below this (penny-stock filter)

# --- volume_breakout: a volume surge confirming accumulation ---
BREAKOUT_VOL_MULT      = 2.5    # 'buy'  : today's volume >= this x its average
BREAKOUT_VOL_WATCH     = 1.5    # 'hold' : volume elevated but below a full spike
BREAKOUT_NEAR_HIGH_PCT = 25.0   # stock must be within this % of its 52-week high
ADD_NEAR_HIGH_PCT      = 5.0    # 'add'  : a buy that is ALSO within 5% of the high

# --- volume_vcp: volume dry-up in a tight base near highs (pre-breakout) ---
VCP_VOL_MAX        = 0.50   # volume drying up : today's vol <= this x its average
VCP_MAX_ADR        = 3    # tight range      : ADR%(20) must be <= this
VCP_NEAR_HIGH_PCT  = 10.0   # base near highs  : within this % of the 52-week high
VCP_TIGHT_ADR      = 2.0    # 'add' : an even tighter range
VCP_TIGHT_HIGH_PCT = 6.0    # 'add' : an even closer-to-high base
# ============================================================


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


# ---------- Drive helpers (identical to the other strategy files) ----------

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


# ---------- Shared helpers ----------

REQUIRED_COLS = ["symbol", "date", "close", "atr_14", "vol_today_ratio",
                 "above_200sma", "days_above_ema_50",
                 "dist_from_52w_high_pct", "adr_pct_20"]

OUT_COLS = ["symbol", "date", "strategy", "zone_type", "score", "entry", "stop",
            "vol_today_ratio", "atr_expansion_ratio", "dist_from_52w_high_pct",
            "adr_pct_20", "reason"]


def _check_columns(features: pd.DataFrame, label: str) -> bool:
    missing = [c for c in REQUIRED_COLS if c not in features.columns]
    if missing:
        log(f"{label}: features missing columns {missing} — skipping.")
        return False
    return True


def in_uptrend(df: pd.DataFrame) -> pd.Series:
    """Uptrend = currently trading above BOTH the 50-EMA and the 200-SMA.

    `days_above_ema_50` counts consecutive days above the 50-EMA and is 0 while
    below it, so `>= 1` simply means 'currently above the 50-EMA'. `above_200sma`
    is the long-term trend gate. Together they keep signals to constructive names
    and reject downtrending stocks that merely had a one-off volume spike.
    """
    return (df["above_200sma"] == True) & (df["days_above_ema_50"] >= 1)


# ---------- Strategy 1: volume_breakout ----------

def volume_breakout_signals(features: pd.DataFrame) -> pd.DataFrame:
    """
      buy  : volume >= BREAKOUT_VOL_MULT x avg, in uptrend, within NEAR_HIGH% of high
      add  : buy AND within ADD_NEAR_HIGH_PCT of the 52-week high
      hold : volume between WATCH and MULT (elevated, not a full spike), in uptrend
    """
    if not _check_columns(features, "volume_breakout"):
        return pd.DataFrame()

    df = features.copy()
    df = df[df["close"] >= MIN_PRICE]                       # penny filter
    df = df[df["vol_today_ratio"].notna()]
    df = df[in_uptrend(df)]                                 # trend filter
    df = df[df["dist_from_52w_high_pct"] >= -BREAKOUT_NEAR_HIGH_PCT]
    if df.empty:
        return pd.DataFrame()

    vr = df["vol_today_ratio"]
    is_buy = vr >= BREAKOUT_VOL_MULT
    is_hold = (vr >= BREAKOUT_VOL_WATCH) & (vr < BREAKOUT_VOL_MULT)
    is_add = is_buy & (df["dist_from_52w_high_pct"] >= -ADD_NEAR_HIGH_PCT)

    df["zone_type"] = None
    df.loc[is_hold, "zone_type"] = "hold"
    df.loc[is_buy, "zone_type"] = "buy"
    df.loc[is_add, "zone_type"] = "add"
    df = df[df["zone_type"].notna()].copy()
    if df.empty:
        return pd.DataFrame()

    df["strategy"] = "volume_breakout"
    # Rank on RANGE expansion, not the size of the volume spike.
    # guru/backtest/family_lift.parquet: TECH_VOLSURGE is +5.7pp at 3M, while
    # TECH_ATR (ATR expansion) is +22.1pp. Volume alone cannot tell accumulation
    # from a news dump; range expanding with it is the confirming tell. Volume
    # stays the GATE (>= BREAKOUT_VOL_MULT); this only changes the ordering.
    # df.get() returns None when the column is absent, and pd.to_numeric(None)
    # yields a SCALAR nan — which has no .notna(). Build an explicit Series so the
    # "features have not shipped atr_expansion_ratio yet" path stays a fallback
    # rather than an AttributeError.
    _atr_exp = (pd.to_numeric(df["atr_expansion_ratio"], errors="coerce")
                if "atr_expansion_ratio" in df.columns
                else pd.Series(np.nan, index=df.index, dtype="float64"))
    _vol = (df["vol_today_ratio"] / BREAKOUT_VOL_MULT * 60).clip(upper=100)
    if _atr_exp.notna().any():
        # 1.0 = no expansion, 2.0x = full marks.
        _rng = ((_atr_exp - 1.0) / 1.0 * 100).clip(lower=0, upper=100)
        df["score"] = (0.7 * _rng.fillna(_vol) + 0.3 * _vol).round(1)
        df["atr_expansion_ratio"] = _atr_exp.round(3)
    else:
        # compute_features has not shipped atr_expansion_ratio yet — fall back to
        # the old volume-only score rather than emitting nothing.
        df["score"] = _vol.round(1)
    df["entry"] = df["close"].round(2)
    df["stop"] = (df["close"] - 2 * df["atr_14"]).round(2)
    df["reason"] = df.apply(
        lambda r: f"Volume {r['vol_today_ratio']:.1f}x avg, "
                  f"{r['dist_from_52w_high_pct']:.1f}% from 52w high, "
                  f"ADR {r['adr_pct_20']:.1f}%",
        axis=1)
    # Select only what exists. atr_expansion_ratio arrives with the new
    # compute_features; until that ships, asking for it raises KeyError and
    # takes the strategy down rather than degrading to the volume-only score
    # the fallback above was written to provide.
    return (df[[c for c in OUT_COLS if c in df.columns]]
            .sort_values("score", ascending=False).reset_index(drop=True))


# ---------- Strategy 2: volume_vcp ----------

def volume_vcp_signals(features: pd.DataFrame) -> pd.DataFrame:
    """
      buy  : volume <= VCP_VOL_MAX x avg (dry-up) AND ADR%(20) <= VCP_MAX_ADR
             (tight range) AND within VCP_NEAR_HIGH_PCT of the 52-week high,
             in uptrend.
      add  : buy AND an even tighter range / even closer to the high.
    """
    if not _check_columns(features, "volume_vcp"):
        return pd.DataFrame()

    df = features.copy()
    df = df[df["close"] >= MIN_PRICE]                       # penny filter
    df = df[df["vol_today_ratio"].notna()]
    df = df[in_uptrend(df)]                                 # trend filter
    df = df[df["vol_today_ratio"] <= VCP_VOL_MAX]           # volume dry-up
    df = df[df["adr_pct_20"] <= VCP_MAX_ADR]                # tight range
    df = df[df["dist_from_52w_high_pct"] >= -VCP_NEAR_HIGH_PCT]  # near highs
    if df.empty:
        return pd.DataFrame()

    is_add = ((df["adr_pct_20"] <= VCP_TIGHT_ADR)
              & (df["dist_from_52w_high_pct"] >= -VCP_TIGHT_HIGH_PCT))
    df["zone_type"] = "buy"
    df.loc[is_add, "zone_type"] = "add"

    df["strategy"] = "volume_vcp"
    # Score 0-100: tighter range and closer to the high score higher.
    df["score"] = (100 - df["adr_pct_20"] * 10
                   - df["dist_from_52w_high_pct"].abs()).clip(lower=0).round(1)
    df["entry"] = df["close"].round(2)
    df["stop"] = (df["close"] - 2 * df["atr_14"]).round(2)
    df["reason"] = df.apply(
        lambda r: f"VCP setup: volume {r['vol_today_ratio']:.2f}x avg (dry-up), "
                  f"tight ADR {r['adr_pct_20']:.1f}%, "
                  f"{r['dist_from_52w_high_pct']:.1f}% from 52w high — "
                  f"buy on the breakout",
        axis=1)
    # Select only what exists. atr_expansion_ratio arrives with the new
    # compute_features; until that ships, asking for it raises KeyError and
    # takes the strategy down rather than degrading to the volume-only score
    # the fallback above was written to provide.
    return (df[[c for c in OUT_COLS if c in df.columns]]
            .sort_values("score", ascending=False).reset_index(drop=True))


# Each entry: (per_strategy folder name, signal function)
STRATEGIES = [
    ("volume_breakout", volume_breakout_signals),
    ("volume_vcp",      volume_vcp_signals),
]

# Written on zero-signal days so latest.csv stays fresh (healthcheck treats a
# stale latest.csv as CRITICAL; the aggregator skips empty files cleanly).
_EMPTY_SIG_COLS = ["symbol", "date", "strategy", "zone_type", "score",
                   "entry", "stop", "reason"]


# ---------- Main ----------

def main() -> None:
    global DRY_RUN
    DRY_RUN = _parse_dry_run()
    if DRY_RUN:
        log("DRY RUN — no Drive writes will be made")
    print("Stage 12 — Volume signals")
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

    signals_id = get_or_create_subfolder(drive, folder_id, "signals")
    per_strategy_id = get_or_create_subfolder(drive, signals_id, "per_strategy")
    today_str = datetime.now().strftime("%Y-%m-%d")

    combined_rows = []
    print()
    print(f"{'Strategy':<18} {'BUY':>5} {'ADD':>5} {'HOLD':>5}  Top 3 by score")
    print("-" * 72)
    for name, fn in STRATEGIES:
        sig = fn(features)
        if sig.empty:
            print(f"{name:<18}     0     0     0  (no signals)")
            sub_id = get_or_create_subfolder(drive, per_strategy_id, name)
            upload_csv(drive, sub_id, "latest.csv",
                       pd.DataFrame(columns=_EMPTY_SIG_COLS),
                       find_file(drive, sub_id, "latest.csv"))
            continue

        n_buy = int((sig["zone_type"] == "buy").sum())
        n_add = int((sig["zone_type"] == "add").sum())
        n_hold = int((sig["zone_type"] == "hold").sum())
        top3 = sig.nlargest(3, "score")["symbol"].tolist()
        print(f"{name:<18} {n_buy:>5} {n_add:>5} {n_hold:>5}  {', '.join(top3)}")

        sub_id = get_or_create_subfolder(drive, per_strategy_id, name)
        upload_csv(drive, sub_id, f"{today_str}.csv", sig,
                   find_file(drive, sub_id, f"{today_str}.csv"))
        upload_csv(drive, sub_id, "latest.csv", sig,
                   find_file(drive, sub_id, "latest.csv"))
        combined_rows.append(sig)

    if combined_rows:
        combined = pd.concat(combined_rows, ignore_index=True)
        upload_csv(drive, per_strategy_id, "volume_combined_latest.csv", combined,
                   find_file(drive, per_strategy_id, "volume_combined_latest.csv"))
        log(f"Wrote volume_combined_latest.csv ({len(combined)} rows total)")

    print("-" * 72)
    print("Saved to signals/per_strategy/volume_breakout/ and volume_vcp/")


if __name__ == "__main__":
    main()
