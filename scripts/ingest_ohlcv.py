"""
Stage 2c (v2) — Bulk OHLCV ingestion via yfinance BATCH download.

Why batched (yf.download) vs per-symbol (yf.Ticker.history):
  - Batched uses Yahoo's chart endpoint, which works fine from GitHub Actions /
    datacenter IPs.
  - Per-symbol uses Yahoo's older finance-quote endpoint, which gets rate-limited
    from datacenter IPs and fails with "Expecting value: line 1 column 1".
  - Batched is also 5-10× faster overall.

Usage:
    python scripts/ingest_ohlcv.py                       # default: incremental, period=1mo
    python scripts/ingest_ohlcv.py --backfill            # full history (period=10y) for missing files
    python scripts/ingest_ohlcv.py --period 6mo          # custom window (overrides incremental)
    python scripts/ingest_ohlcv.py --pilot --limit 50    # test on first 50 symbols
"""

from __future__ import annotations

import argparse
import io
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
import yfinance as yf
from dotenv import load_dotenv
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload, MediaIoBaseUpload

SCOPES = ["https://www.googleapis.com/auth/drive"]
DEFAULT_BATCH_SIZE = 25
DEFAULT_INCREMENTAL_PERIOD = "3mo"
BACKFILL_PERIOD = "10y"

# #6 — Yahoo `<sym>.NS` TICKER COLLISIONS: these short SME/Emerge symbols resolve
# on Yahoo to a DIFFERENT (mainboard) company, so pulling them would append the
# wrong company's prices. Verified 2026-07-22 (stored vs Yahoo close mismatch,
# different listed name). Skip them entirely — they have no correct Yahoo feed
# and belong in the accepted no-price gap. NOT FOCUS (that one matches Yahoo).
COLLISION_SKIP = {"KEL", "KALYANI", "MAL", "SEL", "ZEAL", "GSTL"}


def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")


# ---------- Drive helpers ----------

def get_drive_service():
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
    cs_path = Path(os.environ["GDRIVE_OAUTH_CLIENT_SECRET_PATH"])
    tk_path = Path(os.environ["GDRIVE_OAUTH_TOKEN_PATH"])
    creds = None
    if tk_path.exists():
        creds = Credentials.from_authorized_user_file(str(tk_path), SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(str(cs_path), SCOPES)
            creds = flow.run_local_server(port=0)
        tk_path.write_text(creds.to_json())
    return build("drive", "v3", credentials=creds, cache_discovery=False)


# Per-thread Drive client. googleapiclient's underlying http transport is NOT
# thread-safe, so each worker thread builds (and caches) its own service from the
# same on-disk creds. The main thread builds the first service (refreshing the
# token file once) before any worker starts, so workers only ever READ a valid token.
_tl = threading.local()


def _thread_drive():
    d = getattr(_tl, "drive", None)
    if d is None:
        d = get_drive_service()
        _tl.drive = d
    return d


def get_or_create_subfolder(drive, parent_id, name):
    q = (f"name='{name}' and '{parent_id}' in parents "
         f"and mimeType='application/vnd.google-apps.folder' and trashed=false")
    found = drive.files().list(q=q, fields="files(id,name)").execute().get("files", [])
    if found:
        return found[0]["id"]
    meta = {"name": name, "parents": [parent_id],
            "mimeType": "application/vnd.google-apps.folder"}
    return drive.files().create(body=meta, fields="id").execute()["id"]


def list_files_in_folder(drive, folder_id):
    out, page_token = {}, None
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


def download_csv(drive, file_id):
    request = drive.files().get_media(fileId=file_id)
    fh = io.BytesIO()
    d = MediaIoBaseDownload(fh, request)
    done = False
    while not done:
        _, done = d.next_chunk()
    fh.seek(0)
    return pd.read_csv(fh)


def download_parquet(drive, file_id):
    request = drive.files().get_media(fileId=file_id)
    fh = io.BytesIO()
    d = MediaIoBaseDownload(fh, request)
    done = False
    while not done:
        _, done = d.next_chunk()
    fh.seek(0)
    return pd.read_parquet(fh)


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


# ---------- Batched fetch ----------
def fetch_ohlcv_batch(symbols: list[str], period: str) -> dict[str, pd.DataFrame]:
    """Minimal yf.download call relying entirely on yfinance's native curl_cffi wrapper."""
    if not symbols:
        return {}
    suffixed = [f"{s}.NS" for s in symbols]
    try:
        # Let yfinance natively handle cookie extraction and browser fingerprinting
        df = yf.download(suffixed, period=period, group_by="ticker", progress=False)
    except Exception as e:
        log(f"  Batch fetch raised: {str(e)[:160]}")
        return {}
    if df is None or df.empty:
        return {}

    out: dict[str, pd.DataFrame] = {}

    def _normalize(frame: pd.DataFrame) -> pd.DataFrame:
        f = frame.reset_index().dropna(how="all")
        f.columns = [str(c).lower().replace(" ", "_") for c in f.columns]
        if "date" not in f.columns and "datetime" in f.columns:
            f = f.rename(columns={"datetime": "date"})
        if "date" not in f.columns:
            return pd.DataFrame()
        f["date"] = pd.to_datetime(f["date"]).dt.tz_localize(None).dt.normalize()
        keep = [c for c in ["date", "open", "high", "low", "close", "volume"] if c in f.columns]
        return (f[keep].dropna(subset=["close"]).sort_values("date").reset_index(drop=True))

    if isinstance(df.columns, pd.MultiIndex):
        present = set(df.columns.get_level_values(0))
        for sym in symbols:
            full = f"{sym}.NS"
            if full not in present:
                continue
            try:
                sub = df[full].dropna(how="all")
                if not sub.empty:
                    norm = _normalize(sub)
                    if not norm.empty:
                        out[sym] = norm
            except Exception:
                continue
    elif len(symbols) == 1:
        norm = _normalize(df)
        if not norm.empty:
            out[symbols[0]] = norm
    return out

# ---------- Split/bonus scale repair (shared with repair_split_history.py) ----
# Yahoo restates the WHOLE history on a split/bonus ex-date, but we only ever
# append — so stored history keeps the old price scale and a fake cliff appears
# at the ex-date (e.g. VMARCIND 6:1 -> fake -83%). Before appending, compare the
# stored tail with the fresh fetch on overlapping dates; if the scale drifted,
# rescale our own stored bars in place. We never overwrite from Yahoo history
# (Yahoo can lag restating SME names); the junction check is the arbiter.

SCALE_TOL = 0.02        # >2% overlap drift = stale scale (dividend drift ~0.5%)
JUNCTION_TOL = 0.25     # residual jump allowed at a repaired boundary
MIN_STALE_RUN = 3       # scale changes persist; 1-2 day deviations are glitches
MAX_SEGMENTS = 10       # more distinct ratio steps than this = not a scale issue


def detect_drift(stored: pd.DataFrame, fresh: pd.DataFrame):
    """Returns (stale, boundary, segments). boundary = first date of the
    trailing in-sync run; segments = [(seg_start, factor), ...] oldest first."""
    m = stored.merge(fresh[["date", "close"]], on="date",
                     how="inner", suffixes=("", "_fresh"))
    if len(m) < 5:
        return False, None, []
    m = m.sort_values("date").reset_index(drop=True)
    m["ratio"] = m["close"] / m["close_fresh"]
    m["stale"] = (m["ratio"] - 1.0).abs() > SCALE_TOL
    # A real scale change PERSISTS; an isolated deviant bar is a data glitch
    # (e.g. Yahoo's bad 2026-05-21 session flagged 193 false positives).
    # Ignore stale runs shorter than MIN_STALE_RUN consecutive sessions.
    grp = (m["stale"] != m["stale"].shift()).cumsum()
    for _, g in m.groupby(grp):
        if g["stale"].iloc[0] and len(g) < MIN_STALE_RUN:
            m.loc[g.index, "stale"] = False
    if not m["stale"].any():
        return False, None, []
    i = len(m) - 1
    while i >= 0 and not m.loc[i, "stale"]:
        i -= 1
    boundary = m.loc[i + 1, "date"] if i + 1 < len(m) else None
    if boundary is None:
        return True, None, []
    stale_pre = m[(m["date"] < boundary) & m["stale"]]
    if stale_pre.empty:
        return False, None, []
    seg = (stale_pre["ratio"].round(2).diff().abs() > 0.01).cumsum()
    segments = [(g["date"].min(), float(g["ratio"].median()))
                for _, g in stale_pre.groupby(seg)]
    # A real restatement is a handful of flat steps. Dozens of wandering
    # ratios = stored data disagrees bar-by-bar (bad seed / feed mismatch) —
    # not a scale problem; refuse so it surfaces for manual attention.
    if len(segments) > MAX_SEGMENTS:
        return True, None, []
    return True, boundary, segments


def rescale_in_place(stored: pd.DataFrame, boundary, segments):
    """Divide OHLC of bars before `boundary` by their segment factor (volume
    multiplied). Bars older than the first observed segment use its factor."""
    out = stored.copy()
    starts = [s for s, _ in segments] + [boundary]
    for k, (_, factor) in enumerate(segments):
        lo = starts[k] if k > 0 else None
        hi = starts[k + 1]
        mask = out["date"] < hi
        if lo is not None:
            mask &= out["date"] >= lo
        for c in ("open", "high", "low", "close"):
            if c in out.columns:
                out.loc[mask, c] = out.loc[mask, c] / factor
        if "volume" in out.columns:
            out.loc[mask, "volume"] = (out.loc[mask, "volume"] * factor).round()
    return out


def junction_ok(df: pd.DataFrame, dates) -> bool:
    """No residual jump > JUNCTION_TOL at any repaired boundary."""
    s = df.sort_values("date").reset_index(drop=True)
    for d in dates:
        idx = s.index[s["date"] >= d]
        if len(idx) == 0 or idx[0] == 0:
            continue
        a, b = s.loc[idx[0] - 1, "close"], s.loc[idx[0], "close"]
        if a > 0 and abs(b / a - 1.0) > JUNCTION_TOL:
            return False
    return True


def merge_and_upload(drive, ohlcv_folder_id: str, symbol: str,
                     new_df: pd.DataFrame, existing_files: dict[str, str]) -> dict:
    """Upsert new rows into the symbol's parquet on Drive."""
    filename = f"{symbol}.parquet"
    existing_id = existing_files.get(filename)
    existing_df = None
    if existing_id:
        try:
            existing_df = download_parquet(drive, existing_id)
        except Exception as e:
            return {"symbol": symbol, "status": "read_error", "detail": str(e)[:120]}

    if new_df.empty:
        return {"symbol": symbol, "status": "no_data",
                "rows_added": 0,
                "total_rows": len(existing_df) if existing_df is not None else 0}

    rescaled = False
    if existing_df is not None and len(existing_df) > 0:
        existing_df["date"] = pd.to_datetime(existing_df["date"])
        # Split/bonus guard: if stored history is on a stale price scale
        # (Yahoo restated retroactively; we only append), rescale it in place
        # BEFORE appending — otherwise the ex-date becomes a fake cliff.
        stale, boundary, segments = detect_drift(existing_df, new_df)
        if stale and boundary is not None and segments:
            fixed = rescale_in_place(existing_df, boundary, segments)
            check_dates = [d for d, _ in segments] + [boundary]
            if junction_ok(fixed, check_dates):
                existing_df = fixed
                rescaled = True
            # else: leave stored data untouched (Yahoo-side inconsistency);
            # the append below is still safe — newest bars are on the live scale.
        max_existing_date = existing_df["date"].max()
        truly_new = new_df[new_df["date"] > max_existing_date]
        merged = pd.concat([existing_df, truly_new], ignore_index=True)
        merged = (merged.drop_duplicates(subset=["date"], keep="last")
                  .sort_values("date").reset_index(drop=True))
        rows_added = len(truly_new)
    else:
        merged = new_df
        rows_added = len(new_df)

    if rows_added == 0 and not rescaled:
        return {"symbol": symbol, "status": "up_to_date",
                "rows_added": 0, "total_rows": len(merged)}

    upload_parquet(drive, ohlcv_folder_id, filename, merged, existing_id)
    return {"symbol": symbol,
            "status": "ok_rescaled" if rescaled else "ok",
            "rows_added": rows_added, "total_rows": len(merged)}


# ---------- Main ----------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--period", type=str, default=None,
                        help="yfinance period (default: 1mo incremental, 10y if --backfill)")
    parser.add_argument("--backfill", action="store_true",
                        help="Force period=10y for full history")
    parser.add_argument("--workers", type=int, default=8,
                        help="Parallel Drive merge/upload workers (default 8). "
                             "Each iteration touches its OWN <sym>.parquet — no "
                             "shared-file writes — so threading is data-safe; "
                             "bounded to respect Drive API rate limits. Use 1 for "
                             "the original serial path.")
    args = parser.parse_args()

    period = args.period
    if period is None:
        period = BACKFILL_PERIOD if args.backfill else DEFAULT_INCREMENTAL_PERIOD

    print("Stage 2c (v2) — Batched OHLCV ingest")
    print("-" * 50)
    log(f"Mode: period={period}, batch_size={args.batch_size}")

    drive = get_drive_service()
    folder_id = os.environ["GDRIVE_FOLDER_ID"]

    universe_folder_id = get_or_create_subfolder(drive, folder_id, "universe")
    universe_files = list_files_in_folder(drive, universe_folder_id)
    if "master_list.csv" not in universe_files:
        print("ERROR: universe/master_list.csv missing. Run build_universe.py first.")
        sys.exit(1)
    universe_df = download_csv(drive, universe_files["master_list.csv"])
    # NSE-only here (uses the .NS Yahoo suffix). BSE-only rows in the unified
    # master_list are fetched by fetch_bse_only_ohlcv.py (.BO) — skip them so
    # we don't waste calls on <NSE-suffix> tickers that don't exist.
    if "exchange" in universe_df.columns:
        before = len(universe_df)
        universe_df = universe_df[universe_df["exchange"].astype(str) == "NSE"]
        if before != len(universe_df):
            log(f"NSE filter: {len(universe_df)}/{before} rows (BSE-only handled "
                f"by fetch_bse_only_ohlcv.py)")
    symbols = universe_df["symbol"].astype(str).tolist()
    # #6 — drop Yahoo ticker-collision symbols (would import the wrong company).
    _before_col = len(symbols)
    symbols = [s for s in symbols if s.upper() not in COLLISION_SKIP]
    if len(symbols) != _before_col:
        log(f"Collision skip: dropped {_before_col - len(symbols)} symbol(s) "
            f"{sorted(COLLISION_SKIP)} — no correct Yahoo feed")
    if args.pilot:
        symbols = symbols[:args.limit]
        log(f"PILOT MODE: {len(symbols)} symbols")
    log(f"Symbols to process: {len(symbols)}")

    data_folder_id = get_or_create_subfolder(drive, folder_id, "data")
    ohlcv_folder_id = get_or_create_subfolder(drive, data_folder_id, "ohlcv")
    existing_files = list_files_in_folder(drive, ohlcv_folder_id)
    log(f"Existing OHLCV parquets: {len(existing_files)}")

    # Chunk into batches
    batches = [symbols[i:i + args.batch_size]
               for i in range(0, len(symbols), args.batch_size)]
    log(f"Batches: {len(batches)} of size up to {args.batch_size}")

    workers = max(1, args.workers)
    log(f"Drive merge/upload workers: {workers}"
        + ("  (serial)" if workers == 1 else "  (parallel, per-thread Drive client)"))

    def _process(sym: str) -> dict:
        # Each worker uses its OWN Drive client (thread-local); serial path reuses
        # the main one. existing_files is read-only here (.get) so sharing it is safe.
        d = drive if workers == 1 else _thread_drive()
        try:
            return merge_and_upload(d, ohlcv_folder_id, sym, fetched[sym], existing_files)
        except Exception as e:
            return {"symbol": sym, "status": "upload_error", "detail": str(e)[:120]}

    results: list[dict] = []
    t_start = time.time()
    pool = None if workers == 1 else ThreadPoolExecutor(max_workers=workers)
    for b_idx, batch in enumerate(batches, 1):
        b_start = time.time()
        fetched = fetch_ohlcv_batch(batch, period=period)
        to_process = [s for s in batch if s in fetched]

        # Parallelize the per-symbol download→merge→upload (the Drive-I/O cost).
        # yfinance fetch above stays serial (it's already batched and cheap).
        if pool is None:
            for sym in to_process:
                results.append(_process(sym))
        else:
            futs = {pool.submit(_process, s): s for s in to_process}
            for f in as_completed(futs):
                results.append(f.result())
        # Names Yahoo didn't return at all
        for sym in batch:
            if sym not in fetched:
                results.append({"symbol": sym, "status": "no_data_returned",
                                "rows_added": 0, "total_rows": 0})

        elapsed = time.time() - t_start
        done = b_idx * args.batch_size
        rate = done / elapsed if elapsed else 0
        eta = (len(symbols) - done) / rate / 60 if rate else 0
        ok_so_far = sum(1 for x in results if x["status"] == "ok")
        log(f"  Batch {b_idx}/{len(batches)}  fetched={len(fetched)}/{len(batch)}  "
            f"ok_so_far={ok_so_far}  rate={rate:.1f}/s  ETA={eta:.1f}m  "
            f"(batch took {time.time()-b_start:.1f}s)")
        
        # Keep a small safety pause between bulk downloads
        if b_idx < len(batches):
            time.sleep(1)

    if pool is not None:
        pool.shutdown(wait=True)

    summary = pd.DataFrame(results)
    print()
    print("-" * 50)
    print("Status counts:")
    print(summary["status"].value_counts().to_string())
    total_rows = summary["rows_added"].fillna(0).sum()
    print(f"\nTotal rows added: {int(total_rows)}")
    print(f"Elapsed: {(time.time()-t_start)/60:.1f} min")

    # Refresh Google Drive client connection to prevent stale socket/SSLEOFError
    log("Refreshing Drive connection for run logs...")
    drive = get_drive_service()

    # Run log to Drive
    logs_id = get_or_create_subfolder(drive, folder_id, "logs")
    runs_id = get_or_create_subfolder(drive, logs_id, "ingest_ohlcv")
    today = date.today().isoformat()
    log_name = f"ingest_{today}_{datetime.now().strftime('%H%M')}.csv"
    media = MediaIoBaseUpload(io.BytesIO(summary.to_csv(index=False).encode()),
                              mimetype="text/csv", resumable=False)
    drive.files().create(
        body={"name": log_name, "parents": [runs_id]},
        media_body=media, fields="id").execute()
    log(f"Run log: logs/ingest_ohlcv/{log_name}")


if __name__ == "__main__":
    main()
