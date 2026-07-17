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

# Per-thread requests.Session AND Drive client (neither is safe to share across
# threads). The main thread builds the first Drive client before workers start so
# workers only ever read a valid token.
_tl = threading.local()


def _session() -> requests.Session:
    s = getattr(_tl, "s", None)
    if s is None:
        s = requests.Session()
        s.headers.update(API_HDR)
        _tl.s = s
    return s


def _thread_drive():
    d = getattr(_tl, "drive", None)
    if d is None:
        d = get_drive()
        _tl.drive = d
    return d


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


# ---------- Split/bonus scale guard (BSE side) ---------------------------------
# BSE serves RAW exchange prices and never restates history, so on a bonus/split
# ex-date the appended bar is on the new scale while stored history keeps the old
# one — a fake cliff (FREDUN 2:1 bonus 2026-07-16 -> fake -65%). Guard: when the
# new bar jumps > JUMP_TOL vs the stored last close, confirm against BSE's
# CorporateAction API; only an OFFICIAL Bonus/Sub-division record with a matching
# ex-date triggers a rescale of stored history by the OFFICIAL factor (never a
# price-implied one). Mirrors the NSE guard in ingest_ohlcv.py.

JUMP_TOL = 0.20        # |1-day move| that triggers a corp-action lookup
EXDATE_SLACK_DAYS = 5  # ex-date may differ from the jump bar by a few sessions
JUNCTION_TOL = 0.25    # residual jump allowed after rescale (same as NSE side)

CORP_URL = "https://api.bseindia.com/BseIndiaAPI/api/CorporateAction/w?scripcode={code}"


def corp_action_factor(code: str, around: date) -> tuple[float, date] | None:
    """Official split/bonus factor for a scrip with ex-date within
    EXDATE_SLACK_DAYS of `around`. Returns (factor, ex_date) or None.
    Bonus 'issue X:Y' -> (X+Y)/Y.  Sub-division 'from Rs A to Rs B' -> A/B."""
    j = _get_json(CORP_URL.format(code=str(code).strip()))
    if not j or not isinstance(j, dict):
        return None
    for row in (j.get("Table1") or []):
        xtype = str(row.get("XTYPE", "")).lower()
        val = str(row.get("VALUE", ""))
        exd = None
        m = re.match(r"(\d{1,2})\s+(\w{3})\s+(\d{4})", str(row.get("BCRD_FROM", "")))
        if m:
            try:
                exd = datetime.strptime(m.group(0), "%d %b %Y").date()
            except ValueError:
                pass
        if exd is None or abs((exd - around).days) > EXDATE_SLACK_DAYS:
            continue
        if "bonus" in xtype:
            m = re.search(r"(\d+)\s*:\s*(\d+)", val)
            if m:
                x, y = int(m.group(1)), int(m.group(2))
                if y > 0:
                    return (x + y) / y, exd
        elif "sub" in xtype or "split" in xtype:
            m = re.search(r"(\d+(?:\.\d+)?)\D+(\d+(?:\.\d+)?)", val)
            if m:
                a, b = float(m.group(1)), float(m.group(2))
                if b > 0 and a > b:
                    return a / b, exd
    return None


def rescale_history(df: pd.DataFrame, ex_date, factor: float) -> pd.DataFrame:
    """Divide OHLC of bars BEFORE ex_date by factor (volume multiplied)."""
    out = df.copy()
    mask = out["date"] < pd.Timestamp(ex_date)
    for c in ("open", "high", "low", "close"):
        if c in out.columns:
            out.loc[mask, c] = out.loc[mask, c] / factor
    if "volume" in out.columns:
        out.loc[mask, "volume"] = (out.loc[mask, "volume"] * factor).round()
    return out


def junction_ok(df: pd.DataFrame, ex_date) -> bool:
    s = df.sort_values("date").reset_index(drop=True)
    idx = s.index[s["date"] >= pd.Timestamp(ex_date)]
    if len(idx) == 0 or idx[0] == 0:
        return True
    a, b = s.loc[idx[0] - 1, "close"], s.loc[idx[0], "close"]
    return not (a > 0 and abs(b / a - 1.0) > JUNCTION_TOL)


def _merge_append(drive, folder_id, key, bar: dict, existing_id, bse_code: str = ""):
    """Append one bar to <key>.parquet on Drive (dedup on date). Creates if absent."""
    new = pd.DataFrame([bar])
    new["date"] = pd.to_datetime(new["date"])
    if existing_id:
        try:
            old = pd.read_parquet(io.BytesIO(download_bytes(drive, existing_id)))
            old["date"] = pd.to_datetime(old["date"])
            # Split/bonus guard: big jump vs stored last close -> confirm an
            # official corp action and rescale stored history BEFORE appending.
            prior = old[old["date"] < new["date"].iloc[0]].sort_values("date")
            if bse_code and len(prior):
                last = float(prior["close"].iloc[-1])
                if last > 0 and abs(bar["close"] / last - 1.0) > JUMP_TOL:
                    hit = corp_action_factor(bse_code, bar["date"].date())
                    if hit:
                        factor, exd = hit
                        fixed = rescale_history(old, exd, factor)
                        probe = (pd.concat([fixed, new], ignore_index=True)
                                 .drop_duplicates(subset=["date"], keep="last")
                                 .sort_values("date"))
                        if junction_ok(probe, exd):
                            old = fixed
                            log(f"    {key}: rescaled history /{factor:g} "
                                f"(official {exd} corp action)")
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

    # Worker: fetch the bar AND (live) append it — both parallelized. Each name
    # writes its OWN <key>.parquet via a thread-local Drive client, so there are
    # no shared-file writes; the per-name download+upload is the real cost and is
    # what must run in parallel (the dry-run skips it, hence it looked fast).
    def _process(row) -> dict:
        res = fetch_bar(row, args.max_stale_days)
        if not args.dry_run and res["status"] in ("ok", "no_volume"):
            try:
                _merge_append(_thread_drive(), live_fid, res["key"], res["bar"],
                              live_index.get(f"{res['key']}.parquet"),
                              bse_code=res.get("bse_code", ""))
                res["appended"] = True
            except Exception as e:
                res["status"] = "error"
                res["err"] = str(e)[:80]
        return res

    results: list[dict] = []
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = [pool.submit(_process, r) for r in rows]
        for f in as_completed(futs):
            results.append(f.result())
    appended = sum(1 for r in results if r.get("appended"))

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
