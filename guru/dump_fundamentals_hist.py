r"""
P2b — dump_fundamentals_hist.py  (Project Guru, STANDALONE, RESUMABLE)

One-time historical fundamentals dump via Screener.in for the guru universe.
Reuses scripts/screener_client.py (cookie auth + parsers) READ-ONLY; writes only
under guru/data/ — no shared tables touched.

Per company we store LONG-format rows (statement, line_item, period, value) for:
    quarterly_pl, annual_pl, balance_sheet, cash_flow   (Screener depth ~10-12y)
    shareholding  <- quarterly promoter/FII/DII/public % — ALSO the data source
                     for the ownership rule families (task #9 gets its history here)

RESUMABLE: guru/data/_dump_status/fundamentals_ledger.parquet — retrigger only
processes status pending/error ('done'/'empty' never re-fetched). Flushed every
FLUSH_EVERY companies. Cookie expiry stops the run cleanly; rerun resumes.

Fetch order: securities WITH OHLCV history first (they're backtestable), then rest.
Token convention (same as Phase-1 ingest_fundamentals): NSE names by nse_symbol,
BSE-only names by bse_code.

Usage
-----
    python guru/dump_fundamentals_hist.py --dry-run       # plan only
    python guru/dump_fundamentals_hist.py --limit 20      # pilot
    python guru/dump_fundamentals_hist.py                 # full (resumes)
    python guru/dump_fundamentals_hist.py --status        # ledger summary
    python guru/dump_fundamentals_hist.py --retry-errors  # re-attempt errors
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime

import pandas as pd

GURU_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(GURU_DIR, "data")
FUND_DIR = os.path.join(DATA_DIR, "fundamentals_hist")
STATUS_DIR = os.path.join(DATA_DIR, "_dump_status")
LEDGER_PATH = os.path.join(STATUS_DIR, "fundamentals_ledger.parquet")

SCRIPTS_DIR = os.path.join(os.path.dirname(GURU_DIR), "scripts")
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

FLUSH_EVERY = 25

LEDGER_COLS = ["guru_key", "token", "status", "n_rows", "n_quarters", "n_years",
               "has_shareholding", "error", "attempts", "updated_at"]


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def now() -> str:
    return datetime.utcnow().isoformat(timespec="seconds")


def resolve_token(row: pd.Series) -> str | None:
    """Screener token: NSE symbol when NSE-listed, else bse_code."""
    nse = row.get("nse_symbol")
    if isinstance(nse, str) and nse.strip():
        return nse.strip()
    code = row.get("bse_code")
    if code is not None and str(code).strip() not in ("", "nan", "None"):
        return str(code).strip()
    return None


def load_or_init_ledger(uni: pd.DataFrame) -> pd.DataFrame:
    uni = uni.copy()
    uni["token"] = uni.apply(resolve_token, axis=1)
    fetchable = uni[uni["token"].notna()][["guru_key", "token"]]
    if os.path.exists(LEDGER_PATH):
        led = pd.read_parquet(LEDGER_PATH)
        new = fetchable[~fetchable["guru_key"].isin(led["guru_key"])].copy()
        if not new.empty:
            for col, val in [("status", "pending"), ("n_rows", 0), ("n_quarters", 0),
                             ("n_years", 0), ("has_shareholding", False),
                             ("error", ""), ("attempts", 0), ("updated_at", now())]:
                new[col] = val
            led = pd.concat([led, new[LEDGER_COLS]], ignore_index=True)
            log(f"ledger: +{len(new)} new rows appended as pending")
        return led
    led = fetchable.copy()
    for col, val in [("status", "pending"), ("n_rows", 0), ("n_quarters", 0),
                     ("n_years", 0), ("has_shareholding", False),
                     ("error", ""), ("attempts", 0), ("updated_at", now())]:
        led[col] = val
    log(f"ledger: initialized fresh with {len(led)} rows")
    return led[LEDGER_COLS]


def flush_ledger(led: pd.DataFrame) -> None:
    os.makedirs(STATUS_DIR, exist_ok=True)
    led.to_parquet(LEDGER_PATH, index=False)


def summary(led: pd.DataFrame) -> str:
    return (f"ledger: {led['status'].value_counts().to_dict()} | "
            f"statement rows stored: {int(led['n_rows'].sum()):,} | "
            f"with shareholding: {int(led['has_shareholding'].sum())}")


def parse_shareholding(client, soup) -> list[dict]:
    """Screener #shareholding section: quarterly Promoters/FIIs/DIIs/Public %."""
    tbl = client.parse_table_section(soup, "shareholding")
    out = []
    for line_item, values in tbl["rows"].items():
        for period, value in zip(tbl["headers"], values):
            out.append({"statement": "shareholding", "line_item": line_item,
                        "period": period, "value": value})
    return out


def fetch_one(client, token: str) -> pd.DataFrame | None:
    soup = client.fetch_company(token)
    if soup is None:
        return None
    rows = client.extract_statements(token, soup)
    rows = [{k: v for k, v in r.items() if k != "symbol"} for r in rows]
    rows.extend(parse_shareholding(client, soup))
    if not rows:
        return None
    df = pd.DataFrame(rows)
    df["fetched_at"] = now()
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--retry-errors", action="store_true")
    ap.add_argument("--status", action="store_true")
    args = ap.parse_args()

    uni = pd.read_parquet(os.path.join(DATA_DIR, "universe_hist.parquet"))
    led = load_or_init_ledger(uni)

    if args.status:
        print(summary(led))
        return

    # priority: OHLCV-done securities first (they're the backtestable ones)
    ohlcv_ledger = os.path.join(STATUS_DIR, "ohlcv_ledger.parquet")
    priced = set()
    if os.path.exists(ohlcv_ledger):
        ol = pd.read_parquet(ohlcv_ledger)
        priced = set(ol.loc[ol["status"] == "done", "guru_key"])
    led["_prio"] = (~led["guru_key"].isin(priced)).astype(int)

    todo_mask = led["status"].eq("pending")
    if args.retry_errors:
        todo_mask |= led["status"].eq("error")
    todo = led[todo_mask].sort_values(["_prio", "guru_key"])
    if args.limit:
        todo = todo.head(args.limit)
    log(summary(led))
    log(f"to fetch this run: {len(todo)} (priced-first={len(todo[todo._prio == 0])})")

    if args.dry_run:
        log("DRY RUN — no fetching, no writes. First 10 planned:")
        for _, r in todo.head(10).iterrows():
            print(f"   {r['guru_key']}  <-  token {r['token']}")
        return

    from screener_client import ScreenerClient, CookieExpiredError
    client = ScreenerClient()
    os.makedirs(FUND_DIR, exist_ok=True)

    n_done = n_empty = n_err = 0
    for i, (idx, r) in enumerate(todo.iterrows(), 1):
        led.at[idx, "attempts"] = int(led.at[idx, "attempts"]) + 1
        led.at[idx, "updated_at"] = now()
        try:
            df = fetch_one(client, r["token"])
            if df is not None and len(df) > 0:
                df.to_parquet(os.path.join(FUND_DIR, f"{r['guru_key']}.parquet"),
                              index=False)
                led.at[idx, "status"] = "done"
                led.at[idx, "n_rows"] = len(df)
                led.at[idx, "n_quarters"] = df[df.statement == "quarterly_pl"][
                    "period"].nunique()
                led.at[idx, "n_years"] = df[df.statement == "annual_pl"][
                    "period"].nunique()
                led.at[idx, "has_shareholding"] = bool(
                    (df.statement == "shareholding").any())
                led.at[idx, "error"] = ""
                n_done += 1
            else:
                led.at[idx, "status"] = "empty"
                led.at[idx, "error"] = "no page / no data"
                n_empty += 1
        except CookieExpiredError:
            flush_ledger(led)
            log("cookie expired — ledger flushed, rerun after refreshing cookie "
                "(resumes automatically).")
            return
        except Exception as e:
            led.at[idx, "status"] = "error"
            led.at[idx, "error"] = str(e)[:200]
            n_err += 1
        if i % FLUSH_EVERY == 0:
            flush_ledger(led)
            log(f"progress {i}/{len(todo)} (done={n_done} empty={n_empty} err={n_err})")

    flush_ledger(led)
    log(f"RUN COMPLETE: done={n_done} empty={n_empty} err={n_err}")
    log(summary(led))


if __name__ == "__main__":
    main()
