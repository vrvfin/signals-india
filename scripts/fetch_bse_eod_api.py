r"""
fetch_bse_eod_api.py — DAILY incremental EOD bar for BSE-only names via the BSE
API (api.bseindia.com), replacing the stale Yahoo `.BO` daily feed.

Why: Yahoo `.BO` leaves BSE-only names ~3-5 days stale (only ~2 of 2,635 current).
The BSE API returns the latest completed session for ~97% of names (tested). The
bulk Bhavcopy ZIP is Akamai-blocked, but the per-scrip API works with the repo's
existing header pattern (same as sync_pf._fetch_bse_isin_map).

Scope: appends ONE completed-session bar per scrip into the LIVE data/ohlcv/<KEY>.
Full 2y history stays seeded by fetch_bse_only_ohlcv.py --backfill (weekly). Run
AFTER market close so the API's "Ason" is today's completed bar.

Endpoints (proven):
  getScripHeaderData -> Header{Open, High, Low, LTP, PrevClose, Ason}
  StockTrading       -> TTQ (lakh) -> volume = TTQ * 1e5

Usage:
    python scripts/fetch_bse_eod_api.py --dry-run --limit 50   # coverage check, no writes
    python scripts/fetch_bse_eod_api.py --dry-run              # full-universe dry-run
    python scripts/fetch_bse_eod_api.py --workers 8            # live daily incremental
"""
from __future__ import annotations

import argparse
import io
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta

_SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(_SCRIPTS_DIR), ".env"))

from _extractor_base import (get_drive, get_or_create_subfolder, find_file,
                             download_bytes, upload_bytes, log)

OHLCV_LIVE = "data/ohlcv"
OHLCV_BSE = "data/ohlcv_bse"
COVERAGE_NAME = "_eod_api_coverage.csv"

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/124.0 Safari/537.36")
API_HDR = {"User-Agent": UA, "Accept": "application/json, text/plain, */*",
           "Referer": "https://www.bseindia.com/", "Origin": "https://www.bseindia.com"}
HDR_URL = ("https://api.bseindia.com/BseIndiaAPI/api/getScripHeaderData/w"
           "?Debtflag=&scripcode={code}&seriesid=")
TRD_URL = "https://api.bseindia.com/BseIndiaAPI/api/StockTrading/w?flag=&scripcode={code}"

# Per-thread requests.Session (Session is not safe to share across threads).
_tl = threading.local()


def _session() -> requests.Session:
    s = getattr(_tl, "s", None)
    if s is None:
        s = requests.Session()
        s.headers.update(API_HDR)
        _tl.s = s
    return s


def _folder(drive, parts: str) -> str:
    fid = os.environ["GDRIVE_FOLDER_ID"]
    for p in parts.split("/"):
        fid = get_or_create_subfolder(drive, fid, p)
    return fid


def _list_folder(drive, folder_id: str) -> dict:
    out, tok = {}, None
    while True:
        resp = drive.files().list(
            q=f"'{folder_id}' in parents and trashed=false",
            fields="nextPageToken, files(id, name)", pageSize=1000,
            pageToken=tok).execute()
        for f in resp.get("files", []):
            out[f["name"]] = f["id"]
        tok = resp.get("nextPageToken")
        if not tok:
            break
    return out


def _bse_only(drive) -> pd.DataFrame:
    idx = _folder(drive, "company_repo/_index")
    fid = find_file(drive, idx, "company_universe.csv")
    if not fid:
        return pd.DataFrame()
    uni = pd.read_csv(io.BytesIO(download_bytes(drive, fid))).fillna("")
    nse = uni["nse_symbol"].astype(str).str.strip()
    code = uni["bse_code"].astype(str).str.strip()
    keep = uni[(nse.isin(["", "nan"])) & (~code.isin(["", "nan"]))].copy()
    keep["bse_code"] = (keep["bse_code"].astype(str).str.strip()
                        .str.replace(r"\.0$", "", regex=True))
    return keep.reset_index(drop=True)


def _storage_key(r) -> str:
    sym = str(r.get("bse_symbol", "")).strip()
    if sym and sym.lower() != "nan":
        return sym.upper()
    return f"BSE{str(r['bse_code']).strip()}"


def _num(v):
    """Parse a BSE numeric string ('1,310.19') to float, or None."""
    if v is None:
        return None
    s = str(v).replace(",", "").strip()
    if s in ("", "-", "0", "0.00"):
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _parse_ason(s) -> date | None:
    m = re.match(r"(\d{1,2})\s+(\w{3})\s+(\d{2})", str(s or ""))
    if not m:
        return None
    try:
        return datetime.strptime(f"{m.group(1)} {m.group(2)} 20{m.group(3)}",
                                 "%d %b %Y").date()
    except ValueError:
        return None


def _get_json(url: str, retries: int = 1):
    s = _session()
    for attempt in range(retries + 1):
        try:
            r = s.get(url, timeout=25)
            if r.status_code == 200 and r.content[:1] in (b"{", b"["):
                return r.json()
        except Exception:
            pass
        if attempt < retries:
            time.sleep(0.4)
    return None


def fetch_bar(row, max_stale_days: int) -> dict:
    """Fetch one completed-session bar for a BSE scrip. Returns a result dict with
    status in {ok, no_volume, stale, no_data, error}."""
    code = str(row["bse_code"]).strip()
    key = _storage_key(row)
    base = {"key": key, "bse_code": code, "name": row.get("name", "")}
    j = _get_json(HDR_URL.format(code=code))
    if not j or not isinstance(j, dict):
        return {**base, "status": "no_data"}
    hd = j.get("Header") or {}
    ason = _parse_ason(hd.get("Ason"))
    o, h, l, c = (_num(hd.get("Open")), _num(hd.get("High")),
                  _num(hd.get("Low")), _num(hd.get("LTP")))
    if ason is None or None in (o, h, l, c):
        return {**base, "status": "no_data"}
    if (date.today() - ason).days > max_stale_days:
        return {**base, "status": "stale", "ason": ason}
    # Volume (second call). Missing volume is tolerated (bar still useful).
    vol = None
    tj = _get_json(TRD_URL.format(code=code))
    if tj and isinstance(tj, dict):
        ttq = _num(tj.get("TTQ"))
        if ttq is not None:
            vol = int(round(ttq * 1e5))   # TTQ is in lakh
    bar = {"date": pd.Timestamp(ason), "open": o, "high": h, "low": l,
           "close": c, "volume": vol if vol is not None else 0}
    return {**base, "status": "ok" if vol is not None else "no_volume",
            "ason": ason, "bar": bar}


def _merge_append(drive, folder_id, key, bar: dict, existing_id):
    """Append one bar to <key>.parquet on Drive (dedup on date). Creates if absent."""
    new = pd.DataFrame([bar])
    new["date"] = pd.to_datetime(new["date"])
    if existing_id:
        try:
            old = pd.read_parquet(io.BytesIO(download_bytes(drive, existing_id)))
            old["date"] = pd.to_datetime(old["date"])
            merged = (pd.concat([old, new], ignore_index=True)
                      .drop_duplicates(subset=["date"], keep="last")
                      .sort_values("date").reset_index(drop=True))
        except Exception:
            merged = new
    else:
        merged = new
    upload_bytes(drive, folder_id, f"{key}.parquet",
                 merged.to_parquet(index=False), "application/octet-stream",
                 existing_id=existing_id)
    return len(merged)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="Pilot: first N names.")
    ap.add_argument("--workers", type=int, default=8,
                    help="Parallel fetch workers (default 8; thread-local session).")
    ap.add_argument("--max-stale-days", type=int, default=7,
                    help="Skip a scrip whose latest session is older than this "
                         "(suspended/delisted return ancient dates).")
    ap.add_argument("--dry-run", action="store_true",
                    help="Fetch + report coverage (incl. volume); NO Drive writes.")
    args = ap.parse_args()

    drive = get_drive()
    bse = _bse_only(drive)
    if bse.empty:
        log("No BSE-only names in universe — nothing to do.")
        return
    if args.limit:
        bse = bse.head(args.limit)
    workers = max(1, args.workers)
    log(f"BSE-only EOD via API: {len(bse)} names | workers={workers} | "
        f"mode={'DRY-RUN' if args.dry_run else 'LIVE'}")

    live_fid = _folder(drive, OHLCV_LIVE)
    live_index = {} if args.dry_run else _list_folder(drive, live_fid)
    rows = bse.to_dict("records")

    results: list[dict] = []
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = [pool.submit(fetch_bar, r, args.max_stale_days) for r in rows]
        for f in as_completed(futs):
            results.append(f.result())

    # Append the good bars (skip dry-run).
    appended = 0
    if not args.dry_run:
        for res in results:
            if res["status"] in ("ok", "no_volume"):
                try:
                    _merge_append(drive, live_fid, res["key"], res["bar"],
                                  live_index.get(f"{res['key']}.parquet"))
                    appended += 1
                except Exception as e:
                    res["status"] = "error"
                    res["err"] = str(e)[:80]

    # Coverage summary.
    cov = pd.DataFrame([{k: v for k, v in r.items() if k != "bar"} for r in results])
    n = len(cov)
    sc = cov["status"].value_counts().to_dict()
    ok = sc.get("ok", 0) + sc.get("no_volume", 0)
    with_vol = sc.get("ok", 0)
    today_n = int((cov.get("ason") == date.today()).sum()) if "ason" in cov else 0
    log("-" * 60)
    log(f"BSE EOD API: {n} names | valid bar={ok} ({100*ok/n:.1f}%) | "
        f"with volume={with_vol} ({100*with_vol/n:.1f}%) | "
        f"dated today={today_n} ({100*today_n/n:.1f}%)")
    log(f"  status: {sc}")
    if not args.dry_run:
        log(f"  appended bars to live data/ohlcv/: {appended}")
        try:
            upload_bytes(drive, _folder(drive, OHLCV_BSE), COVERAGE_NAME,
                         cov.to_csv(index=False).encode("utf-8"), "text/csv",
                         existing_id=find_file(drive, _folder(drive, OHLCV_BSE),
                                               COVERAGE_NAME))
        except Exception as e:
            log(f"  coverage CSV upload failed: {str(e)[:80]} (bars already written)")
    else:
        log("  DRY-RUN — no Drive writes.")
        for r in results[:6]:
            if r["status"] in ("ok", "no_volume"):
                b = r["bar"]
                log(f"    {r['key']:<12} {str(b['date'])[:10]} O={b['open']} "
                    f"H={b['high']} L={b['low']} C={b['close']} V={b['volume']}")
    log(f"elapsed {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
