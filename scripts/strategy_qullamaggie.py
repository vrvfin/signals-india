"""
Stage 6 — Qullamaggie setup detector.

Per DESIGN §1.5 + §9:
  Pre-conditions (from features):
    return_3m_pct >= 30      (the stock has had a recent extended move)
    adr_pct_20    >= 4       (enough daily range to make trading worthwhile)
    above_200sma  == True    (trend filter)

  Consolidation detection (downloads recent OHLCV per candidate):
    Find the longest contiguous window of `min_days..max_days` recent bars whose
    high-low range is ≤ consolidation_max_range_pct of the average close.

  Zones:
    buy  : today's close is inside the consolidation range (pivot_low..pivot_high)
    add  : today's close > pivot_high AND today's volume ≥ 1.5× 20-day avg (breakout)
    stop : pivot_low − 0.5 × ATR(14)
    sell : not emitted yet — Stage 9 portfolio tracking owns it

Output:
  signals/per_strategy/qullamaggie/<date>.csv
  signals/per_strategy/qullamaggie/latest.csv
"""

from __future__ import annotations

import io
import os
import time
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

# Strategy params
MIN_RETURN_3M_PCT = 30
MIN_ADR_PCT = 4
CONSOLIDATION_MIN_DAYS = 5
CONSOLIDATION_MAX_DAYS = 15
CONSOLIDATION_MAX_RANGE_PCT = 15
BREAKOUT_VOLUME_MULTIPLIER = 1.5


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


# ---------- Consolidation detection ----------

def find_consolidation(ohlcv: pd.DataFrame) -> dict | None:
    """Find the longest recent consolidation window meeting our tightness criterion.
    Returns dict with pivot info, or None."""
    if len(ohlcv) < CONSOLIDATION_MIN_DAYS:
        return None
    highs = ohlcv["high"].astype(float).values
    lows = ohlcv["low"].astype(float).values
    closes = ohlcv["close"].astype(float).values

    # Try longest first so we surface the widest stable base.
    for n in range(min(CONSOLIDATION_MAX_DAYS, len(ohlcv)),
                   CONSOLIDATION_MIN_DAYS - 1, -1):
        h = highs[-n:].max()
        l = lows[-n:].min()
        avg = closes[-n:].mean()
        if avg <= 0:
            continue
        range_pct = (h - l) / avg * 100
        if range_pct <= CONSOLIDATION_MAX_RANGE_PCT:
            return {
                "consolidation_days": n,
                "pivot_high": h,
                "pivot_low": l,
                "range_pct": range_pct,
            }
    return None


def qullamaggie_signal(symbol: str, ohlcv: pd.DataFrame,
                      feat: pd.Series) -> dict | None:
    """Apply Qullamaggie rules to one stock. Returns a signal dict or None."""
    consol = find_consolidation(ohlcv)
    if not consol:
        return None

    last = ohlcv.iloc[-1]
    close = float(last["close"])
    today_vol = float(last["volume"])
    vol_20d_avg = float(ohlcv["volume"].tail(20).mean())
    vol_today_ratio = today_vol / vol_20d_avg if vol_20d_avg else 0.0

    vol_5d_avg = float(ohlcv["volume"].tail(5).mean())
    vol_dryup = vol_5d_avg < 0.8 * vol_20d_avg if vol_20d_avg else False

    pivot_h = consol["pivot_high"]
    pivot_l = consol["pivot_low"]

    # Zone determination
    if close > pivot_h and vol_today_ratio >= BREAKOUT_VOLUME_MULTIPLIER:
        zone = "add"
        reason = (f"Breakout > pivot ₹{pivot_h:.2f} on vol {vol_today_ratio:.1f}× avg; "
                  f"{consol['consolidation_days']}d base ({consol['range_pct']:.1f}% range)")
    elif pivot_l <= close <= pivot_h:
        zone = "buy"
        reason = (f"Inside {consol['consolidation_days']}d consolidation "
                  f"₹{pivot_l:.2f}-{pivot_h:.2f} ({consol['range_pct']:.1f}% range); "
                  f"3m run {feat['return_3m_pct']:.0f}%, ADR {feat['adr_pct_20']:.1f}%, "
                  f"vol dryup={vol_dryup}")
    else:
        return None  # already extended above pivot or below; not actionable

    stop = pivot_l - 0.5 * float(feat["atr_14"])
    return {
        "symbol": symbol,
        "date": feat["date"],
        "strategy": "qullamaggie",
        "zone_type": zone,
        "score": float(feat["return_3m_pct"]),  # bigger run = higher conviction
        "entry": round(close, 2),
        "stop": round(stop, 2),
        "pivot_high": round(pivot_h, 2),
        "pivot_low": round(pivot_l, 2),
        "consolidation_days": consol["consolidation_days"],
        "range_pct": round(consol["range_pct"], 2),
        "vol_today_ratio": round(vol_today_ratio, 2),
        "vol_dryup": vol_dryup,
        "adr_pct": round(float(feat["adr_pct_20"]), 2),
        "return_3m_pct": round(float(feat["return_3m_pct"]), 2),
        "reason": reason,
    }


# ---------- Main ----------

def main() -> None:
    print("Stage 6 — Qullamaggie signals")
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

    # Pre-filter candidates
    candidates = features[
        (features["return_3m_pct"] >= MIN_RETURN_3M_PCT) &
        (features["adr_pct_20"] >= MIN_ADR_PCT) &
        (features["above_200sma"] == True)
    ].copy()
    log(f"Pre-filter candidates (3m>={MIN_RETURN_3M_PCT}%, ADR>={MIN_ADR_PCT}%, "
        f"above 200SMA): {len(candidates)}")

    if candidates.empty:
        print("No candidates after pre-filter.")
        return

    # OHLCV folder
    data_id = get_or_create_subfolder(drive, folder_id, "data")
    ohlcv_id = get_or_create_subfolder(drive, data_id, "ohlcv")
    ohlcv_files = list_files_in_folder(drive, ohlcv_id)

    # Process candidates
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
            sig = qullamaggie_signal(sym, ohlcv, feat)
            if sig:
                signals.append(sig)
        except Exception as e:
            log(f"  {sym}: error — {str(e)[:80]}")
        if i % 25 == 0:
            elapsed = time.time() - t_start
            rate = i / elapsed
            eta = (len(candidates) - i) / rate
            log(f"  [{i}/{len(candidates)}] {len(signals)} setups found | "
                f"rate {rate:.1f}/s | ETA {eta:.0f}s")

    if not signals:
        print("\nNo Qullamaggie setups in current market.")
        return

    sig_df = pd.DataFrame(signals)
    sig_df = sig_df.sort_values("score", ascending=False).reset_index(drop=True)

    # Save
    signals_id = get_or_create_subfolder(drive, folder_id, "signals")
    per_strategy_id = get_or_create_subfolder(drive, signals_id, "per_strategy")
    qm_id = get_or_create_subfolder(drive, per_strategy_id, "qullamaggie")
    today_str = datetime.now().strftime("%Y-%m-%d")
    upload_csv(drive, qm_id, f"{today_str}.csv", sig_df,
               find_file(drive, qm_id, f"{today_str}.csv"))
    upload_csv(drive, qm_id, "latest.csv", sig_df,
               find_file(drive, qm_id, "latest.csv"))

    # Summary
    print()
    print("-" * 50)
    n_buy = (sig_df["zone_type"] == "buy").sum()
    n_add = (sig_df["zone_type"] == "add").sum()
    print(f"BUY (in consolidation)    : {n_buy}")
    print(f"ADD (breaking out today)  : {n_add}")
    print(f"\nTop 10 by 3m run:")
    show = ["symbol", "zone_type", "return_3m_pct", "adr_pct",
            "consolidation_days", "range_pct", "entry", "stop"]
    print(sig_df.head(10)[show].to_string(index=False))
    print(f"\nElapsed: {(time.time()-t_start)/60:.1f} min")


if __name__ == "__main__":
    main()
