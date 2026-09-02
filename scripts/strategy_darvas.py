"""
Stage 6c — Darvas Box strategy.

The core Darvas insight: stocks at or near new 52-week highs that form a tight
"box" (sideways range bounded by a top and bottom over N days) and then break
out above the box top on volume.

Differs from Qullamaggie:
  - Darvas requires the stock to be at/near 52w high (Qullamaggie just needs a
    recent extended move; the stock might be a few months past the high).
  - Pure box logic — no consolidation-pattern flexibility.

Pre-filter (from features):
  dist_from_52w_high_pct >= -5   (within 5% of 52w high)
  above_200sma == True
  adr_pct_20 >= 2                (some movement — not glacial)

Per-candidate (downloads recent OHLCV):
  Box = last `box_days` bars whose high-low range is ≤ box_max_range_pct of mean.
  Box top    = max(high) over those bars
  Box bottom = min(low) over those bars

Zones:
  buy  — close inside the box (box_bottom..box_top)
  add  — today's close > box_top AND today's vol ≥ 1.5× 20d avg (breakout)
  stop — box_bottom − 0.5 × ATR

Outputs:
  signals/per_strategy/darvas/<date>.csv
  signals/per_strategy/darvas/latest.csv
"""

from __future__ import annotations

import io
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

_SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)
from strategy_common import base_quality_score
from dotenv import load_dotenv
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload, MediaIoBaseUpload

SCOPES = ["https://www.googleapis.com/auth/drive"]

# Written on zero-signal days so latest.csv stays fresh (healthcheck now treats a
# stale strategy latest.csv as CRITICAL; the aggregator skips empty files).
_EMPTY_SIG_COLS = ["symbol", "date", "strategy", "zone_type", "score",
                   "entry", "stop", "reason"]


def _write_empty_latest(drive, folder_id, strat_name):
    signals_id = get_or_create_subfolder(drive, folder_id, "signals")
    per_strategy_id = get_or_create_subfolder(drive, signals_id, "per_strategy")
    sub_id = get_or_create_subfolder(drive, per_strategy_id, strat_name)
    upload_csv(drive, sub_id, "latest.csv",
               pd.DataFrame(columns=_EMPTY_SIG_COLS),
               find_file(drive, sub_id, "latest.csv"))

# Strategy params
MAX_DIST_FROM_52W_HIGH_PCT = -5   # within 5% of 52w high
MIN_ADR_PCT = 2
BOX_MIN_DAYS = 5
BOX_MAX_DAYS = 20
BOX_MAX_RANGE_PCT = 10            # tighter than Qullamaggie (10% vs 15%)
BREAKOUT_VOLUME_MULTIPLIER = 1.5


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


def list_files_in_folder(drive, folder_id):
    out = {}
    page_token = None
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


# ---------- Box detection ----------

def find_darvas_box(ohlcv: pd.DataFrame) -> dict | None:
    """Find the longest recent window where range/mean ≤ BOX_MAX_RANGE_PCT.
    Returns dict with box info, or None.

    The box is measured EXCLUDING the current (last) bar, so box_top is the
    resistance BUILT BEFORE today. Otherwise box_top = highs[-n:].max() would
    include today's own high, making `close > box_top` impossible and the
    breakout ("add") zone permanently dead (audit #32)."""
    if len(ohlcv) < BOX_MIN_DAYS + 1:   # +1 box bars beyond today's bar
        return None
    highs = ohlcv["high"].astype(float).values
    lows = ohlcv["low"].astype(float).values
    closes = ohlcv["close"].astype(float).values

    # Each window ENDS at the prior bar ([-(n+1):-1]) — today's bar is the
    # breakout candidate, not part of the box.
    for n in range(min(BOX_MAX_DAYS, len(ohlcv) - 1), BOX_MIN_DAYS - 1, -1):
        h = highs[-(n + 1):-1].max()
        l = lows[-(n + 1):-1].min()
        avg = closes[-(n + 1):-1].mean()
        if avg <= 0:
            continue
        range_pct = (h - l) / avg * 100
        if range_pct <= BOX_MAX_RANGE_PCT:
            return {
                "box_days": n,
                "box_top": h,
                "box_bottom": l,
                "range_pct": range_pct,
            }
    return None


def darvas_signal(symbol: str, ohlcv: pd.DataFrame,
                  feat: pd.Series) -> dict | None:
    box = find_darvas_box(ohlcv)
    if not box:
        return None

    last = ohlcv.iloc[-1]
    close = float(last["close"])
    today_vol = float(last["volume"])
    vol_20d_avg = float(ohlcv["volume"].tail(20).mean())
    vol_ratio = today_vol / vol_20d_avg if vol_20d_avg else 0.0

    top = box["box_top"]
    bot = box["box_bottom"]

    if close > top and vol_ratio >= BREAKOUT_VOLUME_MULTIPLIER:
        zone = "add"
        reason = (f"Breakout > box top ₹{top:.2f} on vol {vol_ratio:.1f}× avg; "
                  f"{box['box_days']}d box ({box['range_pct']:.1f}% range), "
                  f"at 52w high ({feat['dist_from_52w_high_pct']:.1f}%)")
    elif bot <= close <= top:
        zone = "buy"
        reason = (f"Inside {box['box_days']}d Darvas box "
                  f"₹{bot:.2f}-{top:.2f} ({box['range_pct']:.1f}% range); "
                  f"{feat['dist_from_52w_high_pct']:.1f}% from 52w high")
    else:
        return None

    stop = bot - 0.5 * float(feat["atr_14"])
    _qual, _parts = base_quality_score(
        box["range_pct"], BOX_MAX_RANGE_PCT, box["box_days"], BOX_MIN_DAYS,
        BOX_MAX_DAYS, vol_ratio, breakout=(zone == "add"))
    return {
        "symbol": symbol,
        "date": feat["date"],
        "strategy": "darvas",
        "zone_type": zone,
        # Was -dist_from_52w_high_pct alone. Proximity to the high is already a
        # hard PRE-FILTER (within 5%), so re-ranking on it separated almost
        # nothing; box tightness, duration and breakout volume — all computed
        # above and previously discarded — are what actually differ.
        "score": _qual,
        "box_tightness": round(_parts["tightness"], 3),
        "box_duration": round(_parts["duration"], 3),
        "box_confirmation": round(_parts["confirmation"], 3),
        "entry": round(close, 2),
        "stop": round(stop, 2),
        "box_top": round(top, 2),
        "box_bottom": round(bot, 2),
        "box_days": box["box_days"],
        "range_pct": round(box["range_pct"], 2),
        "vol_today_ratio": round(vol_ratio, 2),
        "dist_from_52w_high_pct": round(float(feat["dist_from_52w_high_pct"]), 2),
        "adr_pct": round(float(feat["adr_pct_20"]), 2),
        "reason": reason,
    }


# ---------- Main ----------

def main() -> None:
    global DRY_RUN
    DRY_RUN = _parse_dry_run()
    if DRY_RUN:
        log("DRY RUN — no Drive writes will be made")
    print("Stage 6c — Darvas Box signals")
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

    candidates = features[
        (features["dist_from_52w_high_pct"] >= MAX_DIST_FROM_52W_HIGH_PCT) &
        (features["above_200sma"] == True) &
        (features["adr_pct_20"] >= MIN_ADR_PCT)
    ].copy()
    log(f"Pre-filter candidates (within 5% of 52w high, above 200SMA, "
        f"ADR>={MIN_ADR_PCT}%): {len(candidates)}")

    if candidates.empty:
        print("No candidates after pre-filter.")
        _write_empty_latest(drive, folder_id, "darvas")
        return

    data_id = get_or_create_subfolder(drive, folder_id, "data")
    ohlcv_id = get_or_create_subfolder(drive, data_id, "ohlcv")
    ohlcv_files = list_files_in_folder(drive, ohlcv_id)

    signals = []
    t_start = time.time()
    for i, (_, feat) in enumerate(candidates.iterrows(), 1):
        sym = feat["symbol"]
        fname = f"{sym}.parquet"
        if fname not in ohlcv_files:
            continue
        try:
            ohlcv = download_parquet(drive, ohlcv_files[fname])
            ohlcv = ohlcv.sort_values("date").tail(60).reset_index(drop=True)
            sig = darvas_signal(sym, ohlcv, feat)
            if sig:
                signals.append(sig)
        except Exception:
            pass
        if i % 25 == 0:
            elapsed = time.time() - t_start
            rate = i / elapsed
            eta = (len(candidates) - i) / rate
            log(f"  [{i}/{len(candidates)}] {len(signals)} setups | "
                f"rate {rate:.1f}/s | ETA {eta:.0f}s")

    if not signals:
        print("\nNo Darvas setups today.")
        _write_empty_latest(drive, folder_id, "darvas")
        return

    sig_df = pd.DataFrame(signals).sort_values("score", ascending=False).reset_index(drop=True)

    signals_id = get_or_create_subfolder(drive, folder_id, "signals")
    per_strategy_id = get_or_create_subfolder(drive, signals_id, "per_strategy")
    dv_id = get_or_create_subfolder(drive, per_strategy_id, "darvas")
    today_str = datetime.now().strftime("%Y-%m-%d")
    upload_csv(drive, dv_id, f"{today_str}.csv", sig_df,
               find_file(drive, dv_id, f"{today_str}.csv"))
    upload_csv(drive, dv_id, "latest.csv", sig_df,
               find_file(drive, dv_id, "latest.csv"))

    n_buy = (sig_df["zone_type"] == "buy").sum()
    n_add = (sig_df["zone_type"] == "add").sum()
    print()
    print(f"BUY (inside box)         : {n_buy}")
    print(f"ADD (breaking out today) : {n_add}")
    print("\nTop 10 closest to 52w high:")
    show = ["symbol", "zone_type", "dist_from_52w_high_pct", "box_days",
            "range_pct", "entry", "stop", "vol_today_ratio"]
    print(sig_df.head(10)[show].to_string(index=False))
    print(f"\nElapsed: {(time.time()-t_start)/60:.1f} min")


if __name__ == "__main__":
    main()
