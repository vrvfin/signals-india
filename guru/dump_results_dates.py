r"""
P2d — dump_results_dates.py  (Project Guru, STANDALONE, RESUMABLE)

Historical RESULTS-announcement dates for the whole market from BSE's
announcement API — the base-date source for fundamental rules (spec §5).

Depth probed live 2026-07-04:
  * strCat='Result' works from 2012-01 onwards (clean category filter).
  * Pre-2012 (2008..2011): category filter returns nothing; we fetch ALL
    categories (strCat='-1') and keep rows whose subject matches result
    keywords. Noisier but recovers ~4 extra years.
  * Before ~2008 the API returns nothing -> those quarters will use the
    documented estimated-base-date fallback (quarter-end + median lag).

RESUMABLE per month-window: guru/data/_dump_status/results_windows_ledger.parquet
('done'/'empty' windows never re-fetched; retrigger continues from pending/error).
Rows append to guru/data/results_dates_hist.parquet (dedup on NEWSID).

Usage
-----
    python guru/dump_results_dates.py --dry-run
    python guru/dump_results_dates.py --limit 3      # pilot: 3 windows
    python guru/dump_results_dates.py                # full (resumes)
    python guru/dump_results_dates.py --status
"""
from __future__ import annotations

import argparse
import os
import re
import time
from datetime import date, datetime

import pandas as pd
import requests

GURU_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(GURU_DIR, "data")
STATUS_DIR = os.path.join(DATA_DIR, "_dump_status")
LEDGER_PATH = os.path.join(STATUS_DIR, "results_windows_ledger.parquet")
OUT_PATH = os.path.join(DATA_DIR, "results_dates_hist.parquet")

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/124.0 Safari/537.36")
HDR = {"User-Agent": UA, "Accept": "application/json, text/plain, */*",
       "Referer": "https://www.bseindia.com/", "Origin": "https://www.bseindia.com"}
API = "https://api.bseindia.com/BseIndiaAPI/api/AnnSubCategoryGetData/w"

START_CLEAN = date(2012, 1, 1)     # strCat='Result' reliable from here
START_RAW = date(2008, 1, 1)       # all-category + keyword filter era
RESULT_RE = re.compile(
    r"financial result|un-?audited.*result|audited.*result|results for the "
    r"(quarter|year)|statement of (standalone|consolidated).*result", re.I)

KEEP_FIELDS = ["NEWSID", "SCRIP_CD", "SLONGNAME", "NEWS_DT", "NEWSSUB",
               "CATEGORYNAME", "SUBCATNAME", "ATTACHMENTNAME"]


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def month_windows() -> list[tuple[str, date, date, str]]:
    """(window_id, from, to, mode) month windows from START_RAW to today."""
    out = []
    d = START_RAW
    today = date.today()
    while d <= today:
        nxt = date(d.year + (d.month == 12), (d.month % 12) + 1, 1)
        end = min(nxt - pd.Timedelta(days=1).to_pytimedelta(), today)
        mode = "clean" if d >= START_CLEAN else "raw"
        out.append((d.strftime("%Y-%m"), d, end, mode))
        d = nxt
    return out


def load_or_init_ledger() -> pd.DataFrame:
    wins = pd.DataFrame(month_windows(),
                        columns=["window", "from_d", "to_d", "mode"])
    wins["from_d"] = pd.to_datetime(wins["from_d"])
    wins["to_d"] = pd.to_datetime(wins["to_d"])
    if os.path.exists(LEDGER_PATH):
        led = pd.read_parquet(LEDGER_PATH)
        new = wins[~wins["window"].isin(led["window"])].copy()
        if not new.empty:
            new["status"] = "pending"
            new["rows"] = 0
            new["error"] = ""
            new["updated_at"] = ""
            led = pd.concat([led, new], ignore_index=True)
            log(f"ledger: +{len(new)} new windows")
        return led
    wins["status"] = "pending"
    wins["rows"] = 0
    wins["error"] = ""
    wins["updated_at"] = ""
    log(f"ledger: initialized with {len(wins)} month windows "
        f"({wins['window'].iloc[0]} .. {wins['window'].iloc[-1]})")
    return wins


def flush(led: pd.DataFrame) -> None:
    os.makedirs(STATUS_DIR, exist_ok=True)
    led.to_parquet(LEDGER_PATH, index=False)


def _fetch_pages(s: requests.Session, cat: str, frm: date, to: date,
                 max_pages: int = 200) -> list[dict]:
    out = []
    for pageno in range(1, max_pages + 1):
        params = {"pageno": pageno, "strCat": cat, "subcategory": "-1",
                  "strPrevDate": frm.strftime("%Y%m%d"),
                  "strToDate": to.strftime("%Y%m%d"),
                  "strSearch": "P", "strScrip": "", "strType": "C"}
        r = s.get(API, params=params, timeout=40)
        rows = r.json().get("Table", []) if r.status_code == 200 else []
        if not rows:
            break
        out += rows
        if len(rows) < 50:
            break
        time.sleep(0.25)
    return out


def fetch_window(s: requests.Session, frm: date, to: date, mode: str) -> list[dict]:
    if mode == "clean":
        out = _fetch_pages(s, "Result", frm, to)
    else:
        # strCat='-1' silently returns 0 for ANY multi-day range (verified live
        # 2026-07-04) — the raw era must be walked one day at a time.
        out = []
        d = frm
        while d <= to:
            out += _fetch_pages(s, "-1", d, d)
            d += pd.Timedelta(days=1).to_pytimedelta()
        out = [r for r in out
               if RESULT_RE.search(str(r.get("NEWSSUB", "")) +
                                   " " + str(r.get("HEADLINE", "")))]
    return [{k: r.get(k) for k in KEEP_FIELDS} for r in out]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="max windows this run")
    ap.add_argument("--retry-errors", action="store_true")
    ap.add_argument("--status", action="store_true")
    args = ap.parse_args()

    led = load_or_init_ledger()
    if args.status:
        print(f"windows: {led['status'].value_counts().to_dict()} | "
              f"announcement rows: {int(led['rows'].sum()):,}")
        return

    todo_mask = led["status"].eq("pending")
    if args.retry_errors:
        todo_mask |= led["status"].eq("error")
    # newest first: recent windows are the most immediately useful
    todo = led[todo_mask].sort_values("window", ascending=False)
    if args.limit:
        todo = todo.head(args.limit)
    log(f"windows to fetch: {len(todo)} "
        f"(ledger: {led['status'].value_counts().to_dict()})")

    if args.dry_run:
        log("DRY RUN — planned windows (first 8):")
        for _, r in todo.head(8).iterrows():
            print(f"   {r['window']}  mode={r['mode']}")
        return

    s = requests.Session()
    s.headers.update(HDR)
    existing = pd.read_parquet(OUT_PATH) if os.path.exists(OUT_PATH) else None
    n_rows_run = 0
    for i, (idx, w) in enumerate(todo.iterrows(), 1):
        led.at[idx, "updated_at"] = datetime.utcnow().isoformat(timespec="seconds")
        try:
            rows = fetch_window(s, w["from_d"].date(), w["to_d"].date(), w["mode"])
            if rows:
                df = pd.DataFrame(rows)
                df["window"] = w["window"]
                df["mode"] = w["mode"]
                existing = (pd.concat([existing, df], ignore_index=True)
                            if existing is not None else df)
                existing = existing.drop_duplicates(subset=["NEWSID"], keep="first")
                existing.to_parquet(OUT_PATH, index=False)
                led.at[idx, "status"] = "done"
                led.at[idx, "rows"] = len(df)
                n_rows_run += len(df)
            else:
                led.at[idx, "status"] = "empty"
                led.at[idx, "rows"] = 0
            led.at[idx, "error"] = ""
        except Exception as e:
            led.at[idx, "status"] = "error"
            led.at[idx, "error"] = str(e)[:200]
        flush(led)
        if i % 5 == 0:
            log(f"progress {i}/{len(todo)} windows | +{n_rows_run:,} rows this run")
        time.sleep(0.5)

    flush(led)
    total = len(existing) if existing is not None else 0
    log(f"RUN COMPLETE. windows: {led['status'].value_counts().to_dict()} | "
        f"total unique announcements: {total:,} -> {OUT_PATH}")


if __name__ == "__main__":
    main()
