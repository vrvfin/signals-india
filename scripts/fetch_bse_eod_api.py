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
import math
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

from bse_http import bse_session
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


def _session():
    # Browser-fingerprint session (bse_http): since ~2026-09-24 BSE 403s
    # python-requests on every api.bseindia.com call.
    s = getattr(_tl, "s", None)
    if s is None:
        s = bse_session(API_HDR)
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
    # A BSE-only name whose storage key equals an NSE symbol would write into
    # that NSE company's price file: FOCUS, GSTL, KALYANI, MAL, SEL and ZEAL held
    # another company's prices for months (measured 2026-10-02). Skip them until
    # they get a key of their own.
    nse_keys = set(nse.str.upper()) - {"", "NAN"}
    if len(keep):
        clash = keep.apply(lambda r: _storage_key(r) in nse_keys, axis=1)
        if clash.any():
            log(f"  skipped {int(clash.sum())} BSE-only names whose key equals an "
                f"NSE symbol: {sorted(keep[clash].apply(_storage_key, axis=1))}")
            keep = keep[~clash]
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
    # Stock SPLITS are not in Table1: BSE lists them in Table2 as
    # {purpose_code 'SS', Ex_date '25 Sep 2026', purpose 'Stock  Split From
    # Rs.10/- to Rs.5/-'} (NAPL, REMAGNET, verified 2026-10-04). Reading only
    # Table1 meant no split was ever confirmed and history kept a fake cliff.
    for row in (j.get("Table2") or []):
        purpose = " ".join(str(row.get("purpose", "")).lower().split())
        try:
            exd = datetime.strptime(str(row.get("Ex_date", "")).strip(), "%d %b %Y").date()
        except ValueError:
            continue
        if abs((exd - around).days) > EXDATE_SLACK_DAYS:
            continue
        if "split" in purpose or "sub-division" in purpose or "sub division" in purpose:
            fv = [float(x) for x in re.findall(r"rs\.?\s*(\d+(?:\.\d+)?)", purpose)]
            if len(fv) >= 2 and fv[1] > 0 and fv[0] > fv[1]:
                return fv[0] / fv[1], exd
        elif "bonus" in purpose:
            m = re.search(r"(\d+)\s*:\s*(\d+)", purpose)
            if m and int(m.group(2)) > 0:
                return (int(m.group(1)) + int(m.group(2))) / int(m.group(2)), exd
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


# ---------- Official bhavcopy (default source from 2026-10) --------------------
# Since ~2026-09-24 BSE blocks bursts: ~5,000 per-scrip API calls a day got this
# pipeline 403'd (and once got a whole IP blocked). BSE's official daily file —
# ONE request, every BSE equity's OHLC + volume — replaces them. It is published
# ~11:00 UTC (Last-Modified 30 Sep 10:59, 1 Oct 11:04), a little after Phase 1's
# BSE step, so the step waits for it (--wait-min) and only then falls back to
# Yahoo `<bse_symbol>.BO`. A later official file always REPLACES a Yahoo bar for
# the same date; Yahoo bars only ever fill dates the official file has not.

BHAV_URL = ("https://www.bseindia.com/download/BhavCopy/Equity/"
            "BhavCopy_BSE_CM_0_0_0_{d}_F_0000.CSV")
WAIT_POLL_SEC = 120


def _today_ist() -> date:
    return (datetime.utcnow() + timedelta(hours=5, minutes=30)).date()


def fetch_bhavcopy(day: date):
    """(frame, status). status: ok | none (BSE serves its web page: weekend,
    holiday, or not published yet) | blocked (403, network error)."""
    try:
        r = _session().get(BHAV_URL.format(d=day.strftime("%Y%m%d")), timeout=40)
    except Exception:
        return None, "blocked"
    if r.status_code != 200 or b"Access Denied" in r.content[:400]:
        return None, "blocked"
    if not r.content.startswith(b"TradDt"):
        return None, "none"
    df = pd.read_csv(io.BytesIO(r.content), dtype=str)
    return df[df["FinInstrmTp"].astype(str).str.strip() == "STK"], "ok"


def bhav_bars(df: pd.DataFrame, code_to_key: dict) -> dict:
    """key -> bar dict, for the BSE-only names present in one bhavcopy."""
    out = {}
    for r in df.to_dict("records"):
        key = code_to_key.get(str(r.get("FinInstrmId", "")).strip())
        if not key:
            continue
        o, h, l, c = (_num(r.get(k)) for k in ("OpnPric", "HghPric", "LwPric", "ClsPric"))
        if not c or c <= 0:
            continue
        out[key] = {"date": pd.Timestamp(r["TradDt"]).normalize(), "open": o or c,
                    "high": h or c, "low": l or c, "close": c,
                    "volume": float(_num(r.get("TtlTradgVol")) or 0)}
    return out


def yahoo_bars(rows: list, days: set) -> dict:
    """Fallback: key -> [bars] for `days`, from Yahoo `<bse_symbol>.BO` raw
    (unadjusted) prices. Yahoo no longer answers the numeric `<code>.BO` form."""
    import yfinance as yf
    tk = {}
    for r in rows:
        sym = str(r.get("bse_symbol", "")).strip()
        if sym and sym.lower() != "nan":
            tk[f"{sym.upper()}.BO"] = _storage_key(r)
    out, names = {}, list(tk)
    for i in range(0, len(names), 40):
        batch = names[i:i + 40]
        try:
            df = yf.download(batch, period="1mo", group_by="ticker", progress=False,
                             auto_adjust=False, threads=True)
        except Exception as e:
            log(f"  yahoo batch failed: {str(e)[:80]}")
            continue
        if df is None or df.empty:
            continue
        for t in batch:
            try:
                sub = df[t] if isinstance(df.columns, pd.MultiIndex) else df
            except KeyError:
                continue
            sub = sub.dropna(subset=["Close"])
            for d_, r in sub.iterrows():
                d0 = pd.Timestamp(d_).tz_localize(None).normalize()
                if d0.date() in days and r["Close"] > 0:
                    out.setdefault(tk[t], []).append(
                        {"date": d0, "open": float(r["Open"]), "high": float(r["High"]),
                         "low": float(r["Low"]), "close": float(r["Close"]),
                         "volume": float(r.get("Volume", 0) or 0)})
        time.sleep(0.5)
    return out


def _apply_split(merged: pd.DataFrame, fresh_dates: set, exd, factor: float):
    """Put a confirmed split/bonus on ONE scale without ever dividing twice.

    The daily run re-takes the last 14 days from BSE's raw files. For a split
    inside that window the re-taken bars dated before the ex-date are on the OLD
    scale again, while the older stored history may already have been rescaled
    by an earlier run. Rescaling everything again divided that older history a
    second time (NAPL/METSL /4, REMAGNET /100 on 2026-10-07).

    Two candidates: (a) adjust only the re-taken bars before the ex-date (right
    when history was already adjusted), (b) rescale everything before the
    ex-date (right the first time the split is seen). Keep the one whose join
    between untouched history and the first re-taken bar is smoother — the wrong
    one leaves a jump the size of the split ratio there."""
    exd = pd.Timestamp(exd)
    full = rescale_history(merged, exd, factor)
    first = min(fresh_dates)
    if not (merged["date"] < first).any():
        return full, "rescaled"                    # nothing older to protect
    part = merged.copy()
    mask = part["date"].isin(fresh_dates) & (part["date"] < exd)
    for c in ("open", "high", "low", "close"):
        if c in part.columns:
            part.loc[mask, c] = part.loc[mask, c] / factor
    if "volume" in part.columns:
        part.loc[mask, "volume"] = (part.loc[mask, "volume"] * factor).round()

    def join_move(df):
        s = df.sort_values("date").reset_index(drop=True)
        i = int(s.index[s["date"] >= first][0])
        a, b = float(s.loc[i - 1, "close"]), float(s.loc[i, "close"])
        return abs(math.log(b / a)) if a > 0 and b > 0 else float("inf")

    if join_move(part) <= join_move(full):
        return part, "split already applied, re-taken bars adjusted"
    return full, "rescaled"


def _merge_bars(drive, folder_id, key, official: list, yahoo: list,
                existing_id, bse_code: str = "", write: bool = True) -> dict:
    """Official bars REPLACE stored bars of the same date; Yahoo bars only fill
    dates not stored and not official. Then the existing split/bonus guard runs
    on every new bar: a > JUMP_TOL move vs the previous close is rescaled ONLY on
    an official BSE corporate-action record (never price-implied); otherwise the
    history is left alone and the jump is reported."""
    old = pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume"])
    if existing_id:
        old = pd.read_parquet(io.BytesIO(download_bytes(drive, existing_id)))
        old["date"] = pd.to_datetime(old["date"]).dt.normalize()
    off = pd.DataFrame(official)
    off_dates = set(off["date"]) if len(off) else set()
    yh = pd.DataFrame(yahoo)
    if len(yh):
        yh = yh[~yh["date"].isin(set(old["date"]) | off_dates)]
    replaced = int(old["date"].isin(off_dates).sum())
    new_dates = sorted(off_dates | (set(yh["date"]) if len(yh) else set()))
    parts = [x for x in (old[~old["date"].isin(off_dates)], off, yh) if len(x)]
    if not parts:
        return {"key": key, "added": 0, "replaced": 0, "events": [], "new_dates": []}
    merged = pd.concat(parts, ignore_index=True)
    merged = (merged.drop_duplicates(subset=["date"], keep="last")
              .sort_values("date").reset_index(drop=True))
    events = []
    for d0 in new_dates:
        prior = merged[merged["date"] < d0]
        if not len(prior):
            continue
        last = float(prior["close"].iloc[-1])
        now = float(merged.loc[merged["date"] == d0, "close"].iloc[0])
        if last > 0 and abs(now / last - 1.0) > JUMP_TOL:
            hit = corp_action_factor(bse_code, d0.date()) if bse_code else None
            if hit:
                factor, exd = hit
                fixed, how = _apply_split(merged, set(new_dates), exd, factor)
                if junction_ok(fixed, exd):
                    merged = fixed
                    events.append(f"{how} /{factor:g} (official {exd})")
                    continue
            events.append(f"jump {100*(now/last-1):+.0f}% on {d0.date()} - no official "
                          f"record, history untouched")
    if write and new_dates:
        upload_bytes(drive, folder_id, f"{key}.parquet", merged.to_parquet(index=False),
                     "application/octet-stream", existing_id=existing_id)
    return {"key": key, "added": len(new_dates) - replaced, "replaced": replaced,
            "events": events, "new_dates": [str(d.date()) for d in new_dates]}


def main_bhavcopy(args, drive, bse: pd.DataFrame) -> None:
    rows = bse.to_dict("records")
    code_to_key = {str(r["bse_code"]).strip(): _storage_key(r) for r in rows}
    key_to_code = {v: k for k, v in code_to_key.items()}
    today = _today_ist()
    if args.from_date:
        start = date.fromisoformat(args.from_date)
        days = [start + timedelta(n) for n in range((today - start).days + 1)]
    else:
        # today + a catch-up window: any session missed in the last two weeks
        # (BSE refused, Yahoo used, run skipped) is re-taken from the official file
        days = [today - timedelta(n) for n in range(max(1, args.catchup_days))]
    days = sorted(d for d in days if d.weekday() < 5)
    log(f"BSE-only EOD via official bhavcopy: {len(rows)} names | sessions "
        f"{days[0]}..{days[-1]} | mode={'DRY-RUN' if args.dry_run else 'LIVE'}")

    official, yahoo_days, per_day = {}, set(), []
    for d in days:
        df, st = fetch_bhavcopy(d)
        if st == "none" and d == today and args.wait_min > 0:
            deadline = time.time() + args.wait_min * 60
            while st == "none" and time.time() < deadline:
                log(f"  {d}: official file not published yet - waiting")
                time.sleep(WAIT_POLL_SEC)
                df, st = fetch_bhavcopy(d)
        if st == "ok":
            bars = bhav_bars(df, code_to_key)
            for k, b in bars.items():
                official.setdefault(k, []).append(b)
            per_day.append(f"{d} official {len(bars)}")
        elif st == "blocked" or (st == "none" and d == today):
            yahoo_days.add(d)
            per_day.append(f"{d} {st} -> Yahoo")
        else:
            per_day.append(f"{d} no file (weekend/holiday)")
        time.sleep(1)
    log("  sessions: " + " | ".join(per_day))

    yahoo = {}
    if yahoo_days and not args.no_yahoo:
        yahoo = yahoo_bars(rows, yahoo_days)
        log(f"  Yahoo fallback for {sorted(str(d) for d in yahoo_days)}: "
            f"{sum(len(v) for v in yahoo.values())} bars, {len(yahoo)} names")

    keys = sorted(set(official) | set(yahoo))
    if args.limit:
        keys = keys[:args.limit]
    live_fid = _folder(drive, OHLCV_LIVE)
    live_index = _list_folder(drive, live_fid)

    def _one(k):
        try:
            return _merge_bars(_thread_drive(), live_fid, k, official.get(k, []),
                               yahoo.get(k, []), live_index.get(f"{k}.parquet"),
                               bse_code=key_to_code.get(k, ""), write=not args.dry_run)
        except Exception as e:
            return {"key": k, "added": 0, "replaced": 0, "events": [f"error {str(e)[:80]}"],
                    "new_dates": []}

    t0, res = time.time(), []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        for f in as_completed([pool.submit(_one, k) for k in keys]):
            res.append(f.result())
    added = sum(r["added"] for r in res)
    replaced = sum(r["replaced"] for r in res)
    ev = [(r["key"], e) for r in res for e in r["events"]]
    log("-" * 60)
    log(f"bhavcopy: {len(res)} names | bars added {added} | official replaced {replaced} "
        f"| split/bonus events {len(ev)} | "
        f"{'DRY-RUN - no Drive writes' if args.dry_run else 'written'} | {time.time()-t0:.0f}s")
    for k, e in sorted(ev)[:25]:
        log(f"    {k:<12} {e}")
    if len(ev) > 25:
        log(f"    ... {len(ev) - 25} more")
    for r in res[:4]:
        log(f"    sample {r['key']:<12} new dates {r['new_dates']}")


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
    ap.add_argument("--source", choices=["bhavcopy", "api"], default="bhavcopy",
                    help="bhavcopy (default): BSE's official daily file, one request; "
                         "api: the old per-scrip API (~5,000 calls, now 403-prone).")
    ap.add_argument("--from", dest="from_date", default="",
                    help="bhavcopy: fill every session from this date (YYYY-MM-DD).")
    ap.add_argument("--wait-min", type=float, default=30,
                    help="bhavcopy: minutes to wait for today's file before Yahoo.")
    ap.add_argument("--no-yahoo", action="store_true",
                    help="bhavcopy: never fall back to Yahoo.")
    ap.add_argument("--catchup-days", type=int, default=14,
                    help="bhavcopy: calendar days back to re-take from official files.")
    args = ap.parse_args()

    drive = get_drive()
    bse = _bse_only(drive)
    if bse.empty:
        log("No BSE-only names in universe — nothing to do.")
        return
    if args.source == "bhavcopy":
        main_bhavcopy(args, drive, bse)
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
