r"""
P2a — dump_ohlcv_hist.py  (Project Guru, STANDALONE, RESUMABLE)

One-time ~20-year daily OHLCV dump for the ENTIRE historical universe (incl.
delisted), driven by guru/data/universe_hist.parquet (guru_key + yf_ticker).

RESUMABLE BY DESIGN (user requirement 2026-07-04): progress is tracked per
security in guru/data/_dump_status/ohlcv_ledger.parquet. Every retrigger loads
the ledger and processes ONLY rows whose status is 'pending' or 'error' —
'done' and 'empty' rows are never re-fetched. Ctrl-C / crash / quota-block are
all safe: the ledger is flushed to disk every FLUSH_EVERY batches.

Ledger schema (one row per guru_key):
    guru_key, yf_ticker, status (pending|done|empty|error), rows, first_date,
    last_date, error, attempts, updated_at

Price files: guru/data/ohlcv_hist/<guru_key>.parquet
    columns: date, open, high, low, close, adj_basis(bool), volume
    (auto_adjust=True -> close IS the adjusted/total-return close; raw close is
    refetchable later if ever needed. adj_basis marks the adjustment basis.)

Usage
-----
    python guru/dump_ohlcv_hist.py --dry-run          # plan only: counts, no fetch
    python guru/dump_ohlcv_hist.py --limit 50         # pilot batch
    python guru/dump_ohlcv_hist.py                    # full run (resumes automatically)
    python guru/dump_ohlcv_hist.py --retry-errors     # re-attempt rows stuck on 'error'
    python guru/dump_ohlcv_hist.py --status           # print ledger summary and exit
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime

import pandas as pd

GURU_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(GURU_DIR, "data")
OHLCV_DIR = os.path.join(DATA_DIR, "ohlcv_hist")
STATUS_DIR = os.path.join(DATA_DIR, "_dump_status")
LEDGER_PATH = os.path.join(STATUS_DIR, "ohlcv_ledger.parquet")

PERIOD = "25y"          # ask for max sensible window; Yahoo returns what it has
BATCH = 25              # yfinance batch size (Phase-1 proven)
FLUSH_EVERY = 4         # flush ledger every N batches (=100 tickers)
PAUSE_S = 1.0           # pause between batches (be polite to Yahoo)

LEDGER_COLS = ["guru_key", "yf_ticker", "status", "rows", "first_date",
               "last_date", "error", "attempts", "updated_at"]


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def now() -> str:
    return datetime.utcnow().isoformat(timespec="seconds")


def load_or_init_ledger(uni: pd.DataFrame) -> pd.DataFrame:
    """Ledger is the resume point. New universe rows are appended as pending;
    existing statuses are NEVER reset here."""
    fetchable = uni[uni["yf_ticker"].notna()][["guru_key", "yf_ticker"]].copy()
    if os.path.exists(LEDGER_PATH):
        led = pd.read_parquet(LEDGER_PATH)
        new = fetchable[~fetchable["guru_key"].isin(led["guru_key"])].copy()
        if not new.empty:
            new["status"] = "pending"
            new["rows"] = 0
            new["first_date"] = pd.NaT
            new["last_date"] = pd.NaT
            new["error"] = ""
            new["attempts"] = 0
            new["updated_at"] = now()
            led = pd.concat([led, new[LEDGER_COLS]], ignore_index=True)
            log(f"ledger: +{len(new)} new universe rows appended as pending")
        return led
    led = fetchable.copy()
    led["status"] = "pending"
    led["rows"] = 0
    led["first_date"] = pd.NaT
    led["last_date"] = pd.NaT
    led["error"] = ""
    led["attempts"] = 0
    led["updated_at"] = now()
    log(f"ledger: initialized fresh with {len(led)} rows")
    return led[LEDGER_COLS]


def flush_ledger(led: pd.DataFrame) -> None:
    os.makedirs(STATUS_DIR, exist_ok=True)
    led.to_parquet(LEDGER_PATH, index=False)


def summary(led: pd.DataFrame) -> str:
    counts = led["status"].value_counts().to_dict()
    done_rows = int(led.loc[led["status"] == "done", "rows"].sum())
    return (f"ledger: {counts} | total daily bars stored: {done_rows:,} | "
            f"path: {LEDGER_PATH}")


def fetch_batch(tickers: list[str]) -> dict[str, pd.DataFrame]:
    """Batched yfinance download (Phase-1 pattern); returns per-ticker frames."""
    import yfinance as yf
    raw = yf.download(tickers, period=PERIOD, interval="1d", group_by="ticker",
                      auto_adjust=True, threads=True, progress=False)
    out = {}
    if raw is None or raw.empty:
        return out
    if len(tickers) == 1:
        frames = {tickers[0]: raw}
    else:
        frames = {t: raw[t] for t in tickers if t in raw.columns.get_level_values(0)}
    for t, df in frames.items():
        df = df.dropna(subset=["Close"])
        if df.empty:
            continue
        df = df.reset_index()
        df.columns = [str(c).lower() for c in df.columns]
        df = df.rename(columns={"index": "date"})
        df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None).dt.normalize()
        df = df[["date", "open", "high", "low", "close", "volume"]].copy()
        df["adj_basis"] = True
        out[t] = df.sort_values("date").reset_index(drop=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="plan only, no fetch/writes")
    ap.add_argument("--limit", type=int, default=0, help="process at most N securities")
    ap.add_argument("--retry-errors", action="store_true",
                    help="also re-attempt rows with status=error")
    ap.add_argument("--status", action="store_true", help="print ledger summary, exit")
    args = ap.parse_args()

    uni = pd.read_parquet(os.path.join(DATA_DIR, "universe_hist.parquet"))
    led = load_or_init_ledger(uni)

    if args.status:
        print(summary(led))
        return

    todo_mask = led["status"].eq("pending")
    if args.retry_errors:
        todo_mask |= led["status"].eq("error")
    todo = led[todo_mask]
    if args.limit:
        todo = todo.head(args.limit)
    log(f"{summary(led)}")
    log(f"to fetch this run: {len(todo)} securities "
        f"({'incl. error retries' if args.retry_errors else 'pending only'})")

    if args.dry_run:
        log("DRY RUN — no fetching, no writes. First 10 planned:")
        for _, r in todo.head(10).iterrows():
            print(f"   {r['guru_key']}  <-  {r['yf_ticker']}")
        return

    os.makedirs(OHLCV_DIR, exist_ok=True)
    key_by_ticker = dict(zip(todo["yf_ticker"], todo["guru_key"]))
    tickers = todo["yf_ticker"].tolist()
    n_done = n_empty = n_err = 0

    for bi in range(0, len(tickers), BATCH):
        batch = tickers[bi:bi + BATCH]
        try:
            frames = fetch_batch(batch)
        except Exception as e:
            frames = {}
            log(f"batch {bi//BATCH}: download error {str(e)[:80]}")
        for t in batch:
            key = key_by_ticker[t]
            idx = led.index[led["guru_key"] == key][0]
            led.at[idx, "attempts"] = int(led.at[idx, "attempts"]) + 1
            led.at[idx, "updated_at"] = now()
            df = frames.get(t)
            try:
                if df is not None and len(df) > 0:
                    df.to_parquet(os.path.join(OHLCV_DIR, f"{key}.parquet"),
                                  index=False)
                    led.at[idx, "status"] = "done"
                    led.at[idx, "rows"] = len(df)
                    led.at[idx, "first_date"] = df["date"].iloc[0]
                    led.at[idx, "last_date"] = df["date"].iloc[-1]
                    led.at[idx, "error"] = ""
                    n_done += 1
                else:
                    led.at[idx, "status"] = "empty"
                    led.at[idx, "error"] = "no data returned"
                    n_empty += 1
            except Exception as e:
                led.at[idx, "status"] = "error"
                led.at[idx, "error"] = str(e)[:200]
                n_err += 1
        if (bi // BATCH) % FLUSH_EVERY == 0:
            flush_ledger(led)
            log(f"progress {bi + len(batch)}/{len(tickers)} "
                f"(done={n_done} empty={n_empty} err={n_err}) — ledger flushed")
        time.sleep(PAUSE_S)

    flush_ledger(led)
    log(f"RUN COMPLETE: done={n_done} empty={n_empty} err={n_err}")
    log(summary(led))


if __name__ == "__main__":
    main()
