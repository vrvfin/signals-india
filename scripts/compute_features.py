"""
Stage 3 — Feature / indicator engine.

For each symbol in the universe, reads OHLCV parquet from Drive, computes a standard
set of price/volume features, and writes one row per symbol to
`features/<date>.parquet`. Also keeps a rolling `features/latest.parquet` pointer for
the dashboard.

All numeric inputs the strategies need are computed here. Strategies do NOT recompute
indicators — they read this file.

Usage:
    python scripts/compute_features.py                  # process full universe
    python scripts/compute_features.py --limit 50 --no-upload   # safe debug slice
"""

from __future__ import annotations

import argparse
import io
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
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

# Indicator config
EMAS = [10, 20, 50, 100, 200]
SMAS = [50, 200]
RETURN_LOOKBACKS = {"1m": 21, "2m": 42, "3m": 63, "6m": 126, "12m": 252}


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")


# ---------- Drive helpers ----------

# Per-thread Drive client (googleapiclient http transport isn't thread-safe).
# The main thread builds the first service (refreshing the token once) before any
# worker starts, so workers only ever read a valid token.
_tl = threading.local()


def _thread_drive():
    d = getattr(_tl, "drive", None)
    if d is None:
        d = get_drive_service()
        _tl.drive = d
    return d


def get_drive_service():
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


def get_or_create_subfolder(drive, parent_id: str, name: str) -> str:
    q = (f"name='{name}' and '{parent_id}' in parents "
         f"and mimeType='application/vnd.google-apps.folder' and trashed=false")
    found = drive.files().list(q=q, fields="files(id,name)").execute().get("files", [])
    if found:
        return found[0]["id"]
    meta = {"name": name, "parents": [parent_id],
            "mimeType": "application/vnd.google-apps.folder"}
    return drive.files().create(body=meta, fields="id").execute()["id"]


def list_files_in_folder(drive, folder_id: str) -> dict[str, str]:
    out: dict[str, str] = {}
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


def find_file(drive, folder_id: str, name: str) -> str | None:
    q = f"name='{name}' and '{folder_id}' in parents and trashed=false"
    found = drive.files().list(q=q, fields="files(id)").execute().get("files", [])
    return found[0]["id"] if found else None


def download_csv(drive, file_id: str) -> pd.DataFrame:
    request = drive.files().get_media(fileId=file_id)
    fh = io.BytesIO()
    downloader = MediaIoBaseDownload(fh, request)
    done = False
    while not done:
        _, done = downloader.next_chunk()
    fh.seek(0)
    return pd.read_csv(fh)


def download_parquet(drive, file_id: str) -> pd.DataFrame:
    request = drive.files().get_media(fileId=file_id)
    fh = io.BytesIO()
    downloader = MediaIoBaseDownload(fh, request)
    done = False
    while not done:
        _, done = downloader.next_chunk()
    fh.seek(0)
    return pd.read_parquet(fh)


def upload_parquet(drive, folder_id: str, filename: str, df: pd.DataFrame,
                   existing_id: str | None) -> str:
    buf = io.BytesIO()
    df.to_parquet(buf, index=False)
    buf.seek(0)
    media = MediaIoBaseUpload(buf, mimetype="application/octet-stream", resumable=False)
    if existing_id:
        drive.files().update(fileId=existing_id, media_body=media).execute()
        return existing_id
    meta = {"name": filename, "parents": [folder_id]}
    return drive.files().create(body=meta, media_body=media, fields="id").execute()["id"]


# ---------- Indicator math ----------

def days_above_ma(close: pd.Series, ma: pd.Series) -> int:
    """Consecutive trading days the close has stayed above the MA, counting from the
    most recent bar. Returns 0 if today's close is below the MA."""
    above = (close > ma).values
    count = 0
    for v in above[::-1]:
        if v:
            count += 1
        else:
            break
    return count


def compute_features_one(symbol: str, df: pd.DataFrame) -> dict | None:
    """Compute a single-row feature dict for a symbol. Returns None if too short."""
    if df is None or len(df) < 60:
        return None
    df = df.sort_values("date").reset_index(drop=True)
    close = df["close"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    volume = df["volume"].astype(float)

    feat: dict = {
        "symbol": symbol,
        "date": df["date"].iloc[-1],
        "close": close.iloc[-1],
        "open": df["open"].iloc[-1],
        "high": high.iloc[-1],
        "low": low.iloc[-1],
        "volume": volume.iloc[-1],
    }

    # EMAs
    emas = {n: close.ewm(span=n, adjust=False).mean() for n in EMAS}
    for n, s in emas.items():
        feat[f"ema_{n}"] = s.iloc[-1]

    # SMAs
    smas = {n: close.rolling(n).mean() for n in SMAS}
    for n, s in smas.items():
        feat[f"sma_{n}"] = s.iloc[-1]

    # ATR(14) and ADR%(20)
    prev_close = close.shift()
    tr = pd.concat([(high - low), (high - prev_close).abs(),
                    (low - prev_close).abs()], axis=1).max(axis=1)
    feat["atr_14"] = tr.rolling(14).mean().iloc[-1]
    feat["adr_pct_20"] = (((high - low) / close) * 100).rolling(20).mean().iloc[-1]

    # 52-week high / low
    last_252 = df.tail(252)
    high_52w = last_252["high"].max()
    low_52w = last_252["low"].min()
    feat["high_52w"] = high_52w
    feat["low_52w"] = low_52w
    feat["dist_from_52w_high_pct"] = (close.iloc[-1] / high_52w - 1) * 100
    feat["dist_from_52w_low_pct"] = (close.iloc[-1] / low_52w - 1) * 100

    # Returns
    for label, n in RETURN_LOOKBACKS.items():
        if len(close) > n:
            feat[f"return_{label}_pct"] = (close.iloc[-1] / close.iloc[-1 - n] - 1) * 100
        else:
            feat[f"return_{label}_pct"] = np.nan

    # Volume features
    vol20 = volume.rolling(20).mean()
    feat["vol_20d_avg"] = vol20.iloc[-1]
    feat["vol_today_ratio"] = (volume.iloc[-1] / vol20.iloc[-1]) if vol20.iloc[-1] else np.nan
    last5_avg = volume.tail(5).mean()
    last20_avg = volume.tail(20).mean()
    feat["vol_dryup_flag"] = bool(last5_avg < 0.8 * last20_avg) if last20_avg else False
    # Avg daily ₹ turnover over 20d, in crores (shares x price). Liquidity proxy —
    # far more discriminating than raw share volume for filtering tradeability.
    turnover20 = (volume * close).rolling(20).mean()
    feat["avg_turnover_20d_cr"] = (turnover20.iloc[-1] / 1e7
                                   if pd.notna(turnover20.iloc[-1]) else np.nan)
    # 30-day avg ₹ turnover — the gallery's display liquidity floor (₹1cr/day).
    turnover30 = (volume * close).rolling(30).mean()
    feat["avg_turnover_30d_cr"] = (turnover30.iloc[-1] / 1e7
                                   if pd.notna(turnover30.iloc[-1]) else np.nan)

    # Trend flags
    feat["above_50sma"]  = bool(close.iloc[-1] > smas[50].iloc[-1]) if pd.notna(smas[50].iloc[-1]) else False
    feat["above_200sma"] = bool(close.iloc[-1] > smas[200].iloc[-1]) if pd.notna(smas[200].iloc[-1]) else False
    feat["50sma_above_200sma"] = bool(smas[50].iloc[-1] > smas[200].iloc[-1]) if pd.notna(smas[200].iloc[-1]) else False
    sma200_30d_ago = smas[200].iloc[-22] if len(smas[200]) > 22 else np.nan
    feat["200sma_rising"] = bool(smas[200].iloc[-1] > sma200_30d_ago) if pd.notna(sma200_30d_ago) else False

    # Consecutive days above EMA (for MA-respect)
    for n in [10, 20, 50]:
        feat[f"days_above_ema_{n}"] = days_above_ma(close, emas[n])

    # Gap + 1-day move
    if len(df) >= 2:
        prior_close = close.iloc[-2]
        feat["prior_close"] = prior_close
        feat["gap_pct"] = (df["open"].iloc[-1] / prior_close - 1) * 100
        feat["return_1d_pct"] = (close.iloc[-1] / prior_close - 1) * 100

    return feat


def add_relative_strength(feat_df: pd.DataFrame, nifty500_df: pd.DataFrame | None) -> pd.DataFrame:
    """Add cross-sectional RS rank columns (percentile rank within today's universe)
    and excess return vs Nifty 500 for each return lookback (if Nifty 500 available).

    Staleness guard: rank only rows within RS_RANK_MAX_STALE_DAYS calendar days
    of the frame's NEWEST bar (i.e. today's cohort plus the previous trading
    session, which can be up to ~4 calendar days back across a holiday weekend).
    A symbol whose feed lagged longer keeps NaN ranks — ranking a week-old return
    against today's universe skews every percentile. The window is measured off
    the frame's own max date (not the wall clock), so ranks stay internally
    consistent even on a day the whole feed is a little behind; absolute
    staleness is the health-check's job, not this function's."""
    RS_RANK_MAX_STALE_DAYS = 4
    dates = pd.to_datetime(feat_df["date"], errors="coerce")
    ref = dates.max()
    eligible = dates.notna() & ((ref - dates).dt.days <= RS_RANK_MAX_STALE_DAYS)
    n_stale = int((~eligible).sum())
    if n_stale:
        log(f"RS ranks: {n_stale} rows >{RS_RANK_MAX_STALE_DAYS}d behind the "
            f"newest bar ({pd.Timestamp(ref).date()}) excluded from ranking "
            f"(kept in file, ranks NaN)")
    # Cross-sectional rank (eligible rows only; others stay NaN)
    for label in RETURN_LOOKBACKS:
        col = f"return_{label}_pct"
        feat_df[f"rs_rank_{label}"] = (
            feat_df.loc[eligible, col].rank(pct=True) * 100)

    # Excess return vs Nifty 500
    if nifty500_df is not None and len(nifty500_df) > 252:
        n500 = nifty500_df.sort_values("date").reset_index(drop=True)
        n500_close = n500["close"].astype(float)
        for label, n in RETURN_LOOKBACKS.items():
            if len(n500_close) > n:
                n500_ret = (n500_close.iloc[-1] / n500_close.iloc[-1 - n] - 1) * 100
                feat_df[f"rs_vs_nifty500_{label}_pct"] = feat_df[f"return_{label}_pct"] - n500_ret

    return feat_df


# ---------- Main ----------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None,
                        help="Process only first N symbols (debug)")
    parser.add_argument("--no-upload", action="store_true",
                        help="compute and summarise but write NOTHING to Drive. "
                             "Required for any --limit run: uploading a limited "
                             "frame would overwrite features/latest.parquet with "
                             "a partial universe and break every downstream "
                             "strategy plus the health check.")
    parser.add_argument("--out", type=str, default=None,
                        help="with --no-upload, also save the frame locally for "
                             "inspection (parquet path)")
    parser.add_argument("--workers", type=int, default=8,
                        help="Parallel OHLCV-download workers (default 8). Reads are "
                             "independent per symbol and results are collected in the "
                             "main thread, so threading is data-safe; bounded for Drive "
                             "API limits. Use 1 for the original serial path.")
    args = parser.parse_args()

    print("Stage 3 — Feature engine")
    print("-" * 50)

    drive = get_drive_service()
    folder_id = os.environ["GDRIVE_FOLDER_ID"]

    # Load universe
    universe_folder_id = get_or_create_subfolder(drive, folder_id, "universe")
    uni_files = list_files_in_folder(drive, universe_folder_id)
    universe_df = download_csv(drive, uni_files["master_list.csv"])
    symbols = universe_df["symbol"].astype(str).tolist()
    if args.limit:
        symbols = symbols[:args.limit]
    log(f"Universe: {len(symbols)} symbols")

    # Load OHLCV file index
    data_id = get_or_create_subfolder(drive, folder_id, "data")
    ohlcv_id = get_or_create_subfolder(drive, data_id, "ohlcv")
    ohlcv_files = list_files_in_folder(drive, ohlcv_id)
    log(f"OHLCV parquets in Drive: {len(ohlcv_files)}")

    # Load Nifty 500 for RS-vs-index
    indices_id = get_or_create_subfolder(drive, data_id, "indices")
    indices_files = list_files_in_folder(drive, indices_id)
    nifty500_df = None
    if "NIFTY_500.parquet" in indices_files:
        nifty500_df = download_parquet(drive, indices_files["NIFTY_500.parquet"])
        log(f"Nifty 500 history loaded: {len(nifty500_df)} rows")
    else:
        log("WARNING: NIFTY_500.parquet not found — rs_vs_nifty500 columns will be skipped")

    # Compute per symbol
    workers = max(1, args.workers)
    present = [s for s in symbols if f"{s}.parquet" in ohlcv_files]
    missing: list[str] = [s for s in symbols if f"{s}.parquet" not in ohlcv_files]
    log(f"OHLCV present: {len(present)} | missing parquet: {len(missing)} | "
        f"workers: {workers}")

    def _one(sym: str):
        # Download (per-thread Drive client) + compute (pure pandas, thread-safe).
        d = drive if workers == 1 else _thread_drive()
        try:
            df = download_parquet(d, ohlcv_files[f"{sym}.parquet"])
            return sym, compute_features_one(sym, df), None
        except Exception as e:
            return sym, None, str(e)[:60]

    rows: list[dict] = []
    errors: list[tuple[str, str]] = []      # (symbol, error text)
    too_short: list[str] = []               # compute_features_one returned None
    t_start = time.time()
    done = 0

    def _collect(res):
        nonlocal done
        sym, r, err = res
        if err is not None:
            missing.append(f"{sym}({err})")
            errors.append((sym, err))
        elif r is None:
            # <60 bars — a real, countable outcome, not an invisible drop.
            too_short.append(sym)
        else:
            rows.append(r)
        done += 1
        if done % 200 == 0:
            elapsed = time.time() - t_start
            rate = done / elapsed if elapsed else 0
            eta = (len(present) - done) / rate / 60 if rate else 0
            log(f"  [{done}/{len(present)}] rate {rate:.1f}/s | ETA {eta:.1f}m")

    if workers == 1:
        for sym in present:
            _collect(_one(sym))
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for res in as_completed([pool.submit(_one, s) for s in present]):
                _collect(res.result())

    feat_df = pd.DataFrame(rows)
    log(f"Computed features for {len(feat_df)} symbols. Missing: {len(missing)}")
    # Attrition was previously invisible: 5,316 symbols with OHLCV produced 4,798
    # feature rows and the ~518 lost were only printed if fewer than 20. A
    # systematic feed regression looked identical to normal attrition.
    log(f"attrition: {len(present)} attempted -> {len(feat_df)} computed | "
        f"{len(too_short)} too short (<60 bars) | {len(errors)} errored")
    if errors:
        from collections import Counter
        kinds = Counter(e.split(":")[0].strip()[:60] for _, e in errors)
        log("  error breakdown (type -> count):")
        for kind, n in kinds.most_common(10):
            log(f"    {n:>5}  {kind}")
        log(f"  first 10 failing symbols: {[sym for sym, _ in errors[:10]]}")
    if too_short:
        log(f"  first 10 too-short symbols: {too_short[:10]}")

    # Penny-stock filter FIRST — sub-₹10 names must not shape the RS
    # percentiles they are about to be dropped from.
    before = len(feat_df)
    feat_df = feat_df[feat_df["close"] >= 10].reset_index(drop=True)
    log(f"Penny-stock filter: {before} -> {len(feat_df)} (dropped close < ₹10)")

    # Add RS rank + excess returns (ranks only current/previous trading date)
    feat_df = add_relative_strength(feat_df, nifty500_df)

    # Output. The dated snapshot is named for the SESSION it describes, not the
    # runner's UTC wall clock: a run that lands after 00:00 UTC would otherwise
    # label the previous session's bars with tomorrow's date, and
    # build_signal_membership.py derives tenure from these filenames.
    features_folder_id = get_or_create_subfolder(drive, folder_id, "features")
    bar_dates = pd.to_datetime(feat_df["date"], errors="coerce")
    bar_date = bar_dates.max()
    if pd.isna(bar_date):
        today_str = datetime.now().strftime("%Y-%m-%d")
        log(f"WARNING: no usable bar date in features — falling back to wall "
            f"clock {today_str}")
    else:
        today_str = pd.Timestamp(bar_date).strftime("%Y-%m-%d")
        log(f"dated snapshot named for bar date {today_str}")
    filename = f"{today_str}.parquet"
    if args.no_upload:
        log(f"--no-upload: NOT writing features/{filename} or "
            f"features/latest.parquet ({len(feat_df)} rows computed)")
        if args.out:
            feat_df.to_parquet(args.out, index=False)
            log(f"saved local copy -> {args.out}")
    else:
        if args.limit:
            # A limited frame is a partial universe. Publishing it as
            # latest.parquet silently starves every strategy and trips the
            # health check's row gate.
            raise SystemExit(
                "REFUSING to upload a --limit run: features/latest.parquet would "
                "be overwritten with a partial universe. Re-run with --no-upload "
                "(add --out <path> to inspect the frame), or drop --limit.")
        upload_parquet(drive, features_folder_id, filename, feat_df,
                       find_file(drive, features_folder_id, filename))
        log(f"Wrote features/{filename}")
        upload_parquet(drive, features_folder_id, "latest.parquet", feat_df,
                       find_file(drive, features_folder_id, "latest.parquet"))
        log(f"Wrote features/latest.parquet")

    # Summary
    print("-" * 50)
    print(f"Rows         : {len(feat_df)}")
    print(f"Columns      : {len(feat_df.columns)}")
    print(f"Missing      : {len(missing)}")
    if len(missing) > 0 and len(missing) < 20:
        print(f"  → {missing}")
    print(f"Elapsed      : {(time.time()-t_start)/60:.1f} min")
    print("\nSample (first 3 rows, key columns):")
    show = [c for c in ["symbol", "close", "return_3m_pct", "rs_rank_3m",
                        "adr_pct_20", "days_above_ema_20", "dist_from_52w_high_pct"]
            if c in feat_df.columns]
    print(feat_df.head(3)[show].to_string(index=False))


if __name__ == "__main__":
    main()
