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

# The summary prints a delta sign and a rupee sign; a Windows console is cp1252,
# so the run died AFTER completing all its work — and only while printing, which
# no unit test exercises. Degrade the characters instead of the run.
try:
    sys.stdout.reconfigure(errors="replace")
    sys.stderr.reconfigure(errors="replace")
except Exception:          # pragma: no cover - not all streams support it
    pass

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


# ── --dry-run guard ──────────────────────────────────────────────────────────
# market_state writes the snapshot, the history and the sector table. The stance
# columns are new, so house rule 6 applies: confirm a dry-run first.
DRY_RUN = False
_live_upload_csv = upload_csv
_live_upload_parquet = upload_parquet


def upload_csv(drive, folder_id, filename, df, existing_id=None):     # noqa: F811
    if DRY_RUN:
        log(f"[DRY RUN] would write {filename} ({len(df)} rows)")
        return existing_id
    return _live_upload_csv(drive, folder_id, filename, df, existing_id)


def upload_parquet(drive, folder_id, filename, df, existing_id=None):  # noqa: F811
    if DRY_RUN:
        log(f"[DRY RUN] would write {filename} ({len(df)} rows)")
        return existing_id
    return _live_upload_parquet(drive, folder_id, filename, df, existing_id)


def _parse_dry_run() -> bool:
    import argparse as _ap
    p = _ap.ArgumentParser(add_help=False)
    p.add_argument("--dry-run", action="store_true",
                   help="compute and report, write nothing to Drive")
    known, _ = p.parse_known_args()
    return known.dry_run


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

# ---------- Stance: direction, agreement, and what to DO about it ------------
#
# health_score blends six components into one number, which destroys direction:
# 45 could be "trend fine, breadth collapsing" or "trend broken, VIX calm", and
# nothing recorded how many components AGREED. It also had a cliff — the Nifty
# component was binary and worth 25 points, so a single day's cross of the 200
# SMA swung the score 25 points. And it changed nothing: the only consumer was
# CANSLIM's M >= 40 gate, sitting right beside a score that printed 44.8.
#
# health_score and `regime` are left EXACTLY as they were so CANSLIM and the
# HTML dashboard do not move. Everything below is additive.

STANCE_STRONG = 5      # components agreeing for the emphatic call
STANCE_CLEAR = 3       # ... for the moderate one


def component_directions(nifty_info, breadth_info, hl_info, vix_info,
                         fii_info, ad_info) -> dict:
    """+1 bullish / -1 bearish / 0 neutral for each component.

    Thresholds are stated on the component's own natural scale (a VIX level, a
    breadth percentage) rather than on its 0-100 score, so they can be argued
    with. The Nifty read is CONTINUOUS here — distance from the 200 SMA — even
    though health_score still uses the binary version; that removes the 25-point
    cliff from the stance without touching the published score."""
    d = {}
    if nifty_info and nifty_info.get("nifty50_sma200"):
        gap = (nifty_info["nifty50_close"] / nifty_info["nifty50_sma200"] - 1) * 100
        d["nifty50_trend"] = 1 if gap > 2 else (-1 if gap < -2 else 0)
    if breadth_info:
        pct = breadth_info.get("pct_above_50sma")
        if pct is not None:
            d["breadth_50sma"] = 1 if pct > 60 else (-1 if pct < 40 else 0)
    if hl_info:
        diff = hl_info.get("highs_minus_lows_pct_univ")
        if diff is not None:
            d["highs_lows"] = 1 if diff > 0.25 else (-1 if diff < -0.25 else 0)
    if vix_info:
        vix = vix_info.get("india_vix")
        if vix is not None:
            d["vix"] = 1 if vix < 15 else (-1 if vix > 20 else 0)
    if fii_info:
        net = fii_info.get("fii_5d_net_cr")     # Rs cr, net over 5 sessions
        if net is not None:
            d["fii"] = 1 if net > 1000 else (-1 if net < -1000 else 0)
    if ad_info:
        # pct_advancing is absent when score_ad_ratio falls back to its proxy
        # (the early-return branch returns only ad_score_basis/ad_proxy), in
        # which case this component simply abstains rather than guessing.
        pct_up = ad_info.get("pct_advancing")
        if pct_up is not None:
            d["ad_ratio"] = 1 if pct_up > 55 else (-1 if pct_up < 45 else 0)
    return d


def stance_from(directions: dict) -> tuple[str, int, int, int]:
    """(stance, n_bullish, n_bearish, agreement).

    Answers the question the blended score could not: am I meant to be
    aggressive or cautious, and do enough independent things agree to act on it?
    """
    n_bull = sum(1 for v in directions.values() if v > 0)
    n_bear = sum(1 for v in directions.values() if v < 0)
    agreement = max(n_bull, n_bear)
    if n_bull > n_bear:
        stance = ("AGGRESSIVE" if n_bull >= STANCE_STRONG
                  else "CONSTRUCTIVE" if n_bull >= STANCE_CLEAR else "NEUTRAL")
    elif n_bear > n_bull:
        stance = ("DEFENSIVE" if n_bear >= STANCE_STRONG
                  else "CAUTIOUS" if n_bear >= STANCE_CLEAR else "NEUTRAL")
    else:
        stance = "NEUTRAL"          # genuinely mixed — the honest answer
    return stance, n_bull, n_bear, agreement


# What each stance means for position-taking. Consumed by the aggregator's
# buy rules: regime-conditioned signals measured +31.6pp (COMBO_REGIME) in
# guru/backtest/family_lift.parquet, and nothing in Phase 1 used the regime for
# anything beyond a binary CANSLIM gate.
STANCE_PLAYBOOK = {
    "AGGRESSIVE":   "full size; SOFT BUY (state-only) acceptable",
    "CONSTRUCTIVE": "full size; HARD BUY plus the best SOFT BUYs",
    "NEUTRAL":      "HARD BUY only (needs an event firing today)",
    "CAUTIOUS":     "half size; HARD BUY only, and >=3 agreeing families",
    "DEFENSIVE":    "no new buys; manage existing positions only",
}


def stance_history(hist: pd.DataFrame, stance: str, today: str) -> int:
    """How many consecutive sessions this stance has held, today included.
    A one-day flip is noise; ten days is a regime."""
    if hist is None or not len(hist) or "stance" not in hist.columns:
        return 1
    prev = hist[hist["date"] != today].sort_values("date")
    run = 1
    for v in reversed(prev["stance"].tolist()):
        if str(v) == stance:
            run += 1
        else:
            break
    return run


def main():
    global DRY_RUN
    DRY_RUN = _parse_dry_run()
    print("Stage 7 — Market State + Health Score")
    if DRY_RUN:
        log("DRY RUN — no Drive writes will be made")
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

    # ---- STANCE: direction and agreement, not one blended number ----------
    dirs = component_directions(nifty_info, breadth_info, hl_info, vix_info,
                                fii_info, ad_info)
    stance, n_bull, n_bear, agreement = stance_from(dirs)
    row["stance"] = stance
    row["stance_playbook"] = STANCE_PLAYBOOK[stance]
    row["n_bullish"] = n_bull
    row["n_bearish"] = n_bear
    row["n_components"] = len(dirs)
    row["agreement"] = agreement
    for k, v in dirs.items():
        row[f"{k}_dir"] = v

    ms_id = get_or_create_subfolder(drive, data_id, "market_state")

    # History is read BEFORE the snapshot is written, so stance_days and the
    # score trend can go into the same row. Falling from 70 to 50 is a different
    # message from rising from 30 to 50, and the blended score alone said neither.
    hist_id = find_file(drive, ms_id, "history.csv")
    hist_prev = download_csv(drive, hist_id) if hist_id else pd.DataFrame()
    row["stance_days"] = stance_history(hist_prev, stance, today_str)
    if len(hist_prev) and "health_score" in hist_prev.columns:
        h = (hist_prev[hist_prev["date"] != today_str]
             .sort_values("date")["health_score"])
        hs = pd.to_numeric(h, errors="coerce").dropna()
        row["health_trend_5d"] = (round(float(health - hs.iloc[-5]), 1)
                                  if len(hs) >= 5 else None)
        row["health_trend_20d"] = (round(float(health - hs.iloc[-20]), 1)
                                   if len(hs) >= 20 else None)
    else:
        row["health_trend_5d"] = row["health_trend_20d"] = None

    snapshot_df = pd.DataFrame([row])

    # Save snapshot + sector rotation
    upload_parquet(drive, ms_id, "latest.parquet", snapshot_df,
                   find_file(drive, ms_id, "latest.parquet"))
    log("Wrote data/market_state/latest.parquet")

    # Append to history CSV
    if hist_id:
        hist = hist_prev[hist_prev["date"] != today_str]   # replace today if rerun
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
    _arrow = {1: "bullish", -1: "bearish", 0: "neutral"}
    print(f"  STANCE              : {row['stance']}  "
          f"({row['n_bullish']} bullish / {row['n_bearish']} bearish of "
          f"{row['n_components']} components, held {row['stance_days']}d)")
    print(f"  -> {row['stance_playbook']}")
    if row.get("health_trend_5d") is not None:
        print(f"  health 5d / 20d     : {row['health_trend_5d']:+.1f} / "
              f"{row['health_trend_20d'] if row['health_trend_20d'] is not None else 'n/a'}")
    for k, v in dirs.items():
        print(f"     {k:<16} {_arrow[v]}")
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
