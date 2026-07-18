"""
Stage 7 — Market State + Health Score.

Computes the 6 inputs from DESIGN §10, blends them into a 0-100 Market Health
Score, and produces a sector-rotation table. Writes to Drive each run and
appends to a running history file so we can trend the score over time.

Components and weights (sum = 100):
  1.  Nifty 50 vs 200 SMA           (weight 25, binary: 100 if above, else 0)
  2.  % of universe above 50 SMA    (weight 20, the percentage itself)
  3.  New 52w highs minus new lows  (weight 15, normalized -50..+50 → 0..100)
  4.  India VIX level (inverted)    (weight 15, 30→0 score, 10→100 score)
  5.  FII 5-day net flow            (weight 10, +/- 10k cr → 0..100)
  6.  Advance/decline ratio         (weight 15, 1m-return-direction proxy
                                     if return_1d_pct not present)

Outputs:
  data/market_state/latest.parquet         — today's full snapshot row
  data/market_state/history.csv            — one row per run, appended
  data/market_state/sector_rotation_latest.csv

Usage:
    python scripts/market_state.py
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

WEIGHTS = {
    "nifty50_trend":  25,
    "breadth_50sma": 20,
    "highs_lows":    15,
    "vix":           15,
    "fii":           10,
    "ad_ratio":      15,
}

SECTORS = ["NIFTY_IT", "NIFTY_AUTO", "NIFTY_PHARMA", "NIFTY_FMCG",
           "NIFTY_METAL", "NIFTY_REALTY", "NIFTY_ENERGY", "NIFTY_INFRA",
           "NIFTY_BANK"]


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


def download_csv(drive, file_id):
    request = drive.files().get_media(fileId=file_id)
    fh = io.BytesIO()
    downloader = MediaIoBaseDownload(fh, request)
    done = False
    while not done:
        _, done = downloader.next_chunk()
    fh.seek(0)
    return pd.read_csv(fh)


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


def upload_csv(drive, folder_id, filename, df, existing_id=None):
    media = MediaIoBaseUpload(io.BytesIO(df.to_csv(index=False).encode()),
                              mimetype="text/csv", resumable=False)
    if existing_id:
        drive.files().update(fileId=existing_id, media_body=media).execute()
        return existing_id
    meta = {"name": filename, "parents": [folder_id]}
    return drive.files().create(body=meta, media_body=media, fields="id").execute()["id"]


# ---------- Component scorers (each returns 0-100) ----------

def score_nifty50(idx_files, drive):
    """Component 1: binary — Nifty 50 above 200 SMA = 100, else 0."""
    fid = idx_files.get("NIFTY_50.parquet")
    if not fid:
        return None, None
    df = download_parquet(drive, fid).sort_values("date")
    sma200 = df["close"].rolling(200).mean().iloc[-1]
    close = df["close"].iloc[-1]
    above = bool(close > sma200) if pd.notna(sma200) else False
    return (100 if above else 0), {
        "nifty50_close": round(float(close), 2),
        "nifty50_sma200": round(float(sma200), 2) if pd.notna(sma200) else None,
        "nifty50_above_200sma": above,
    }


def score_breadth(features):
    """Component 2: % of universe above 50 SMA, mapped 0-100 directly."""
    pct = features["above_50sma"].mean() * 100
    return float(pct), {"pct_above_50sma": round(float(pct), 1)}


def score_highs_lows(features):
    """Component 3: new 52w highs minus new lows as a % OF THE UNIVERSE.
    The old fixed ±50-count band was calibrated for a ~2.4k universe; at ~5.5k
    names the raw diff saturated the band almost daily, pinning this component
    at 0/100. Percent-based scaling survives universe growth: ±2% of the
    universe maps to the full 0..100 range (2% of names at fresh 52w highs
    with none at lows is an emphatically strong breadth day)."""
    highs = (features["dist_from_52w_high_pct"] >= -0.5).sum()  # within 0.5% counts
    lows = (features["dist_from_52w_low_pct"] <= 0.5).sum()
    diff = int(highs) - int(lows)
    n = max(1, len(features))
    diff_pct = diff / n * 100
    score = max(0, min(100, 50 + diff_pct * 25))   # ±2% → full range
    return float(score), {
        "new_52w_highs": int(highs),
        "new_52w_lows": int(lows),
        "highs_minus_lows": diff,
        "highs_minus_lows_pct_univ": round(diff_pct, 2),
    }


def score_vix(idx_files, drive):
    """Component 4: lower VIX = higher score. Linear: 30→0, 10→100."""
    fid = idx_files.get("INDIA_VIX.parquet")
    if not fid:
        return None, None
    df = download_parquet(drive, fid).sort_values("date")
    vix = float(df["close"].iloc[-1])
    vix_prev = float(df["close"].iloc[-6]) if len(df) >= 6 else vix
    score = max(0, min(100, (30 - vix) / 20 * 100))
    return score, {
        "india_vix": round(vix, 2),
        "india_vix_5d_change": round(vix - vix_prev, 2),
    }


def score_fii(macro_files, drive):
    """Component 5: 5-day FII net flow. +10k cr → 100, -10k cr → 0."""
    fid = macro_files.get("FII_DII.csv")
    if not fid:
        return None, None
    df = download_csv(drive, fid)
    df["date"] = pd.to_datetime(df["date"])
    fii = df[df["category"].str.contains("FII", na=False)].sort_values("date")
    if fii.empty:
        return None, None
    last_5d_net = float(fii.tail(5)["net"].sum())
    # +10000 cr over 5 days → 100, -10000 → 0
    score = max(0, min(100, (last_5d_net / 10000 + 1) * 50))
    return score, {
        "fii_5d_net_cr": round(last_5d_net, 2),
        "fii_5d_rows": int(len(fii.tail(5))),
    }


def score_ad_ratio(features):
    """Component 6: A/D ratio. Uses return_1d_pct if present, else 1m proxy."""
    if "return_1d_pct" in features.columns:
        col = "return_1d_pct"
        proxy = False
    else:
        col = "return_1m_pct"
        proxy = True
    up = (features[col] > 0).sum()
    down = (features[col] < 0).sum()
    total = up + down
    if total == 0:
        return 50.0, {"ad_score_basis": col, "ad_proxy": proxy}
    pct_up = up / total * 100
    return float(pct_up), {
        "advances": int(up),
        "declines": int(down),
        "pct_advancing": round(pct_up, 1),
        "ad_score_basis": col,
        "ad_proxy": proxy,
    }


# ---------- Sector rotation ----------

def sector_rotation(idx_files, drive):
    """Each sector's 1M and 3M return vs Nifty 500."""
    n500_fid = idx_files.get("NIFTY_500.parquet")
    if not n500_fid:
        return pd.DataFrame()
    n500 = download_parquet(drive, n500_fid).sort_values("date")
    rows = []
    for sec in SECTORS:
        fid = idx_files.get(f"{sec}.parquet")
        if not fid:
            continue
        df = download_parquet(drive, fid).sort_values("date")
        if len(df) < 65 or len(n500) < 65:
            continue
        ret_1m_sec = (df["close"].iloc[-1] / df["close"].iloc[-22] - 1) * 100
        ret_3m_sec = (df["close"].iloc[-1] / df["close"].iloc[-64] - 1) * 100
        ret_1m_idx = (n500["close"].iloc[-1] / n500["close"].iloc[-22] - 1) * 100
        ret_3m_idx = (n500["close"].iloc[-1] / n500["close"].iloc[-64] - 1) * 100
        rows.append({
            "sector": sec,
            "return_1m_pct": round(ret_1m_sec, 2),
            "vs_nifty500_1m_pct": round(ret_1m_sec - ret_1m_idx, 2),
            "return_3m_pct": round(ret_3m_sec, 2),
            "vs_nifty500_3m_pct": round(ret_3m_sec - ret_3m_idx, 2),
            "close": round(float(df["close"].iloc[-1]), 2),
        })
    return (pd.DataFrame(rows)
            .sort_values("vs_nifty500_3m_pct", ascending=False)
            .reset_index(drop=True))


# ---------- Main ----------

def main():
    print("Stage 7 — Market State + Health Score")
    print("-" * 50)

    drive = get_drive()
    folder_id = os.environ["GDRIVE_FOLDER_ID"]
    data_id = get_or_create_subfolder(drive, folder_id, "data")
    indices_id = get_or_create_subfolder(drive, data_id, "indices")
    macro_id = get_or_create_subfolder(drive, data_id, "macro")
    idx_files = list_files_in_folder(drive, indices_id)
    macro_files = list_files_in_folder(drive, macro_id)
    log(f"Indices found: {len(idx_files)} | Macro files: {len(macro_files)}")

    features_id = get_or_create_subfolder(drive, folder_id, "features")
    latest_id = find_file(drive, features_id, "latest.parquet")
    features = download_parquet(drive, latest_id)
    log(f"Features loaded: {len(features)} symbols")

    # Compute components
    nifty_score, nifty_info = score_nifty50(idx_files, drive)
    breadth_score, breadth_info = score_breadth(features)
    hl_score, hl_info = score_highs_lows(features)
    vix_score, vix_info = score_vix(idx_files, drive)
    fii_score, fii_info = score_fii(macro_files, drive)
    ad_score, ad_info = score_ad_ratio(features)

    # Weighted total
    components = {
        "nifty50_trend":  (nifty_score, WEIGHTS["nifty50_trend"]),
        "breadth_50sma": (breadth_score, WEIGHTS["breadth_50sma"]),
        "highs_lows":    (hl_score, WEIGHTS["highs_lows"]),
        "vix":           (vix_score, WEIGHTS["vix"]),
        "fii":           (fii_score, WEIGHTS["fii"]),
        "ad_ratio":      (ad_score, WEIGHTS["ad_ratio"]),
    }
    weighted = 0
    used_weight = 0
    for name, (s, w) in components.items():
        if s is not None:
            weighted += s * w
            used_weight += w
    health = weighted / used_weight if used_weight else 0

    # Build snapshot row
    today_str = datetime.now().strftime("%Y-%m-%d")
    row = {
        "date": today_str,
        "health_score": round(health, 1),
        "regime": "RISK_OFF" if health < 40 else ("NEUTRAL" if health < 60 else "RISK_ON"),
    }
    for name, (s, w) in components.items():
        row[f"{name}_score"] = round(s, 1) if s is not None else None
    row.update(nifty_info or {})
    row.update(breadth_info or {})
    row.update(hl_info or {})
    row.update(vix_info or {})
    row.update(fii_info or {})
    row.update(ad_info or {})

    snapshot_df = pd.DataFrame([row])

    # Save snapshot + sector rotation
    ms_id = get_or_create_subfolder(drive, data_id, "market_state")
    upload_parquet(drive, ms_id, "latest.parquet", snapshot_df,
                   find_file(drive, ms_id, "latest.parquet"))
    log("Wrote data/market_state/latest.parquet")

    # Append to history CSV
    hist_id = find_file(drive, ms_id, "history.csv")
    if hist_id:
        hist = download_csv(drive, hist_id)
        hist = hist[hist["date"] != today_str]   # replace today if rerun
        hist = pd.concat([hist, snapshot_df], ignore_index=True)
    else:
        hist = snapshot_df
    upload_csv(drive, ms_id, "history.csv", hist, hist_id)
    log(f"Appended to data/market_state/history.csv ({len(hist)} rows total)")

    # Sector rotation
    sec_df = sector_rotation(idx_files, drive)
    if not sec_df.empty:
        upload_csv(drive, ms_id, "sector_rotation_latest.csv", sec_df,
                   find_file(drive, ms_id, "sector_rotation_latest.csv"))
        log("Wrote data/market_state/sector_rotation_latest.csv")

    # Print summary
    print()
    print("=" * 50)
    print(f"  MARKET HEALTH SCORE : {row['health_score']:>5} / 100   [{row['regime']}]")
    print("=" * 50)
    print(f"  Nifty 50            : {nifty_info.get('nifty50_close')}  "
          f"(200 SMA {nifty_info.get('nifty50_sma200')})  "
          f"above={nifty_info.get('nifty50_above_200sma')}   score {row['nifty50_trend_score']}/100")
    print(f"  Breadth (% > 50SMA) : {breadth_info.get('pct_above_50sma')}%   "
          f"score {row['breadth_50sma_score']}/100")
    print(f"  New 52w highs/lows  : {hl_info.get('new_52w_highs')} highs vs "
          f"{hl_info.get('new_52w_lows')} lows  (diff {hl_info.get('highs_minus_lows')})   "
          f"score {row['highs_lows_score']}/100")
    if vix_info:
        print(f"  India VIX           : {vix_info.get('india_vix')}  "
              f"(5d Δ {vix_info.get('india_vix_5d_change')})   "
              f"score {row['vix_score']}/100")
    if fii_info:
        print(f"  FII 5d net (₹ cr)   : {fii_info.get('fii_5d_net_cr')}   "
              f"score {row['fii_score']}/100")
    print(f"  A/D ratio           : {ad_info.get('advances')} adv vs "
          f"{ad_info.get('declines')} dec   "
          f"({'1m proxy' if ad_info.get('ad_proxy') else 'daily'})   "
          f"score {row['ad_ratio_score']}/100")
    print()
    if not sec_df.empty:
        print("Sector rotation (ranked by 3M vs Nifty 500):")
        print(sec_df.to_string(index=False))


if __name__ == "__main__":
    main()
