r"""
fetch_bse_only_ohlcv.py — STANDALONE OHLCV fetch for BSE-ONLY listed names
(no NSE listing). Plug-and-play: writes to a SEPARATE folder so the live
NSE pipeline (data/ohlcv/) is never touched until you review coverage.

Design (user spec 2026-06-13):
  - Universe: company_universe.csv -> rows with bse_code AND no nse_symbol.
    (Dual-listed names already fetch via NSE in the live pipeline — skipped.)
  - Yahoo ticker = "<bse_code>.BO". Storage key = bse_symbol, else "BSE<code>".
  - Schema IDENTICAL to data/ohlcv/<SYM>.parquet: date,open,high,low,close,volume
    so a later --promote drops straight into the live pipeline.
  - TEST MODE (default): writes only data/ohlcv_bse/<KEY>.parquet +
    data/ohlcv_bse/_coverage.csv. Never writes data/ohlcv/. If the .BO source
    returns NOTHING for everyone, logs and exits 0 (non-fatal).
  - --promote: copy validated parquets (rows >= --min-rows) into data/ohlcv/
    so compute_features + the 9 signals pick them up automatically. Run this
    ONLY after reviewing the coverage CSV.

Usage:
    python scripts/fetch_bse_only_ohlcv.py --limit 300        # pilot, test mode
    python scripts/fetch_bse_only_ohlcv.py                    # all BSE-only, test mode
    python scripts/fetch_bse_only_ohlcv.py --promote --min-rows 120
"""
from __future__ import annotations

import argparse
import io
import os
import sys
import time
from datetime import datetime

_SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

import pandas as pd
import yfinance as yf
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(_SCRIPTS_DIR), ".env"))

from _extractor_base import (get_drive, get_or_create_subfolder, find_file,
                             download_bytes, upload_bytes, log)

OHLCV_LIVE = "data/ohlcv"          # the live NSE folder — only --promote writes here
OHLCV_BSE = "data/ohlcv_bse"       # test folder — safe
COVERAGE_NAME = "_coverage.csv"
COV_COLS = ["isin", "bse_code", "bse_symbol", "name", "yf_ticker", "storage_key",
            "status", "rows", "first_date", "last_date", "fetched_at"]


def _folder(drive, parts):
    fid = os.environ["GDRIVE_FOLDER_ID"]
    for p in parts.split("/"):
        fid = get_or_create_subfolder(drive, fid, p)
    return fid


def _bse_only(drive) -> pd.DataFrame:
    idx = _folder(drive, "company_repo/_index")
    fid = find_file(drive, idx, "company_universe.csv")
    if not fid:
        return pd.DataFrame()
    uni = pd.read_csv(io.BytesIO(download_bytes(drive, fid))).fillna("")
    nse = uni["nse_symbol"].astype(str).str.strip()
    code = uni["bse_code"].astype(str).str.strip()
    keep = uni[(nse.isin(["", "nan"])) & (~code.isin(["", "nan"]))].copy()
    keep["bse_code"] = keep["bse_code"].astype(str).str.replace(r"\.0$", "",
                                                                regex=True)
    return keep.reset_index(drop=True)


def _storage_key(r) -> str:
    sym = str(r.get("bse_symbol", "")).strip()
    if sym and sym.lower() != "nan":
        return sym.upper()
    return f"BSE{str(r['bse_code']).strip()}"


def _normalize(frame: pd.DataFrame) -> pd.DataFrame:
    f = frame.reset_index().dropna(how="all")
    f.columns = [str(c).lower().replace(" ", "_") for c in f.columns]
    if "date" not in f.columns and "datetime" in f.columns:
        f = f.rename(columns={"datetime": "date"})
    if "date" not in f.columns:
        return pd.DataFrame()
    f["date"] = pd.to_datetime(f["date"]).dt.tz_localize(None).dt.normalize()
    keep = [c for c in ["date", "open", "high", "low", "close", "volume"]
            if c in f.columns]
    return f[keep].dropna(subset=["close"]).sort_values("date").reset_index(drop=True)


def _fetch_batch(tickers: list[str], period: str) -> dict[str, pd.DataFrame]:
    """ticker(<code>.BO) -> normalized frame. Mirrors ingest_ohlcv batch logic."""
    if not tickers:
        return {}
    try:
        df = yf.download(tickers, period=period, group_by="ticker", progress=False)
    except Exception as e:
        log(f"  batch raised: {str(e)[:120]}")
        return {}
    if df is None or df.empty:
        return {}
    out = {}
    if isinstance(df.columns, pd.MultiIndex):
        present = set(df.columns.get_level_values(0))
        for tk in tickers:
            if tk not in present:
                continue
            try:
                sub = df[tk].dropna(how="all")
                if not sub.empty:
                    n = _normalize(sub)
                    if not n.empty:
                        out[tk] = n
            except Exception:
                continue
    elif len(tickers) == 1:
        n = _normalize(df)
        if not n.empty:
            out[tickers[0]] = n
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="Pilot: first N names.")
    ap.add_argument("--batch", type=int, default=40)
    ap.add_argument("--period", type=str, default="1y")
    ap.add_argument("--promote", action="store_true",
                    help="Copy validated parquets into the LIVE data/ohlcv/ "
                         "(run only after reviewing coverage).")
    ap.add_argument("--min-rows", type=int, default=120,
                    help="Min rows to count as usable / to promote.")
    args = ap.parse_args()

    drive = get_drive()
    bse = _bse_only(drive)
    if bse.empty:
        log("No BSE-only names in universe — nothing to do. (test mode exit)")
        return
    if args.limit:
        bse = bse.head(args.limit)
    log(f"BSE-only names to fetch: {len(bse)}  (period={args.period}, "
        f"mode={'PROMOTE' if args.promote else 'TEST'})")

    bse_fid = _folder(drive, OHLCV_BSE)
    live_fid = _folder(drive, OHLCV_LIVE) if args.promote else None

    # ticker -> row mapping
    rows = bse.to_dict("records")
    tk_to_row = {f"{r['bse_code']}.BO": r for r in rows}
    tickers = list(tk_to_row.keys())

    cov, ok, nodata, promoted = [], 0, 0, 0
    for i in range(0, len(tickers), args.batch):
        chunk = tickers[i:i + args.batch]
        fetched = _fetch_batch(chunk, args.period)
        for tk in chunk:
            r = tk_to_row[tk]
            key = _storage_key(r)
            frame = fetched.get(tk)
            base = {"isin": r.get("isin", ""), "bse_code": r["bse_code"],
                    "bse_symbol": r.get("bse_symbol", ""), "name": r.get("name", ""),
                    "yf_ticker": tk, "storage_key": key,
                    "fetched_at": datetime.now().isoformat(timespec="seconds")}
            if frame is None or frame.empty:
                cov.append({**base, "status": "no_data", "rows": 0,
                            "first_date": "", "last_date": ""})
                nodata += 1
                continue
            data = frame.to_parquet(index=False)
            upload_bytes(drive, bse_fid, f"{key}.parquet", data,
                         "application/octet-stream",
                         existing_id=find_file(drive, bse_fid, f"{key}.parquet"))
            if args.promote and len(frame) >= args.min_rows:
                upload_bytes(drive, live_fid, f"{key}.parquet", data,
                             "application/octet-stream",
                             existing_id=find_file(drive, live_fid, f"{key}.parquet"))
                promoted += 1
            cov.append({**base,
                        "status": "ok" if len(frame) >= args.min_rows else "thin",
                        "rows": len(frame),
                        "first_date": str(frame["date"].min())[:10],
                        "last_date": str(frame["date"].max())[:10]})
            ok += 1
        log(f"  {min(i + args.batch, len(tickers))}/{len(tickers)} "
            f"(ok={ok} no_data={nodata})")
        time.sleep(0.5)

    cov_df = pd.DataFrame(cov, columns=COV_COLS)
    upload_bytes(drive, bse_fid, COVERAGE_NAME,
                 cov_df.to_csv(index=False).encode("utf-8"), "text/csv",
                 existing_id=find_file(drive, bse_fid, COVERAGE_NAME))

    usable = int((cov_df["status"] == "ok").sum())
    thin = int((cov_df["status"] == "thin").sum())
    log("-" * 56)
    log(f"BSE-only OHLCV: {len(cov_df)} attempted | usable(>= {args.min_rows}r)="
        f"{usable} | thin={thin} | no_data={nodata}"
        + (f" | PROMOTED to live={promoted}" if args.promote else ""))
    log(f"coverage -> {OHLCV_BSE}/{COVERAGE_NAME}"
        + ("" if args.promote else "  (TEST mode — live data/ohlcv untouched)"))
    if ok == 0:
        log("WARNING: .BO returned NO data for any name — check yfinance/BSE "
            "availability before relying on this source.")


if __name__ == "__main__":
    main()
