"""
build_share_count.py — WEEKLY share count for every company in the universe.

WHY
---
Market cap went stale two ways (measured 2026-10-02):
  - fundamentals/summary.parquet market_cap_cr is a Screener snapshot taken when
    the company is scraped. Since the 2026-08-07 gap-scan change that is roughly
    once a quarter (4,454 of 5,713 rows were 31-60 days old; LAURUSLABS 99,255 cr
    from 2026-08-03 vs 1,07,198 cr live).
  - universe/market_cap.csv (enrich_market_cap.py, Yahoo) skips every symbol it
    already holds, so an existing name is never refreshed (LAURUSLABS 73,238 cr).

A share count changes rarely (bonus, split, QIP), a price changes daily. So this
script stores the share count once a week and update_market_cap.py multiplies it
by the day's close at the end of Phase 1.

SOURCE
------
Screener company page (public, no login needed): shares = Market Cap / Current
Price, both read from the same page at the same moment. Screener id = the NSE
symbol for NSE rows, the BSE code (from yf_ticker "<code>.BO") for BSE rows.
A company with no Screener page keeps its previous row; with no previous row it
gets none, and update_market_cap.py leaves its existing market cap alone.

OUTPUT (NEW side file — no existing script reads it)
------
universe/share_count.csv:
  symbol, isin, exchange, screener_id, shares_cr, price_at_fetch,
  mcap_cr_at_fetch, source, fetched_at (UTC)

~1.4 s per company (consolidated + standalone variants at the client's 1 s rate
limit) -> ~135 min for the full universe. Checkpoints every 200 names; names
fetched within --max-age-days are skipped, so a re-run after a timeout resumes.

Usage:
    python scripts/build_share_count.py --dry-run --limit 20
    python scripts/build_share_count.py --symbols LAURUSLABS,TCS --dry-run
    python scripts/build_share_count.py              # weekly, full universe
"""
from __future__ import annotations

import argparse
import io
import os
import sys
import time
from datetime import datetime, timezone

_SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

import pandas as pd
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(_SCRIPTS_DIR), ".env"))

from _extractor_base import (get_drive, get_or_create_subfolder, find_file,
                             download_bytes, upload_bytes, log)
from screener_client import ScreenerClient, CookieExpiredError

OUT_NAME = "share_count.csv"
COLS = ["symbol", "isin", "exchange", "screener_id", "shares_cr",
        "price_at_fetch", "mcap_cr_at_fetch", "source", "fetched_at"]
CHECKPOINT_EVERY = 200


def _screener_id(row) -> str:
    if str(row.get("exchange", "")) == "BSE":
        return str(row.get("yf_ticker", "")).replace(".BO", "").strip()
    return str(row.get("symbol", "")).strip()


def _read_csv(drive, folder_id, name) -> pd.DataFrame:
    fid = find_file(drive, folder_id, name)
    if not fid:
        return pd.DataFrame()
    return pd.read_csv(io.BytesIO(download_bytes(drive, fid)), dtype=str)


def _save(drive, folder_id, rows: dict) -> None:
    df = pd.DataFrame(list(rows.values()), columns=COLS)
    upload_bytes(drive, folder_id, OUT_NAME, df.to_csv(index=False).encode(),
                 "text/csv", existing_id=find_file(drive, folder_id, OUT_NAME))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true",
                    help="Fetch and report; write nothing to Drive.")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--symbols", default="",
                    help="Comma-separated master_list symbols to fetch only.")
    ap.add_argument("--max-age-days", type=float, default=6.0,
                    help="Skip names fetched more recently than this (resume).")
    args = ap.parse_args()

    drive = get_drive()
    uni_id = get_or_create_subfolder(drive, os.environ["GDRIVE_FOLDER_ID"], "universe")
    ml = _read_csv(drive, uni_id, "master_list.csv")
    if ml.empty:
        log("universe/master_list.csv missing — nothing to do.")
        return 1
    ml = ml.fillna("")
    prev = _read_csv(drive, uni_id, OUT_NAME)
    rows = {} if prev.empty else {r["symbol"]: {c: r.get(c, "") for c in COLS}
                                  for r in prev.fillna("").to_dict("records")}
    log(f"master_list: {len(ml):,} | existing {OUT_NAME}: {len(rows):,} rows")

    todo = ml
    if args.symbols:
        want = {s.strip().upper() for s in args.symbols.split(",") if s.strip()}
        todo = ml[ml["symbol"].str.upper().isin(want)]
    else:
        now = pd.Timestamp.now(tz="UTC")
        def fresh(sym):
            t = pd.to_datetime(rows.get(sym, {}).get("fetched_at", ""),
                               errors="coerce", utc=True)
            return pd.notna(t) and (now - t).total_seconds() < args.max_age_days * 86400
        todo = ml[~ml["symbol"].map(fresh)]
    if args.limit:
        todo = todo.head(args.limit)
    log(f"to fetch: {len(todo):,}{' [DRY-RUN]' if args.dry_run else ''}")

    client = ScreenerClient()
    ok = nopage = noratio = 0
    t0 = time.time()
    for i, r in enumerate(todo.to_dict("records"), 1):
        sid = _screener_id(r)
        try:
            soup = client.fetch_company(sid) if sid else None
        except CookieExpiredError as e:
            log(f"Screener refused ({e}) — stopping; rows fetched so far are kept.")
            break
        except Exception as e:                       # network blip: skip, keep old row
            log(f"  {r['symbol']}: fetch error {str(e)[:80]}")
            soup = None
        if soup is None:
            nopage += 1
        else:
            top = client.parse_top_ratios(soup)
            mc, px = top.get("Market Cap"), top.get("Current Price")
            if mc and px and mc > 0 and px > 0:
                rows[r["symbol"]] = {
                    "symbol": r["symbol"], "isin": r.get("isin", ""),
                    "exchange": r.get("exchange", ""), "screener_id": sid,
                    "shares_cr": round(mc / px, 6), "price_at_fetch": px,
                    "mcap_cr_at_fetch": mc, "source": "screener",
                    "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                }
                ok += 1
            else:
                noratio += 1
        if i % 25 == 0 or i == len(todo):
            log(f"  {i}/{len(todo)} ok={ok} no_page={nopage} no_ratio={noratio} "
                f"({(time.time() - t0) / i:.2f}s/name)")
        if not args.dry_run and i % CHECKPOINT_EVERY == 0:
            _save(drive, uni_id, rows)

    log("-" * 56)
    log(f"fetched ok={ok} | no Screener page={nopage} | page without mcap/price={noratio} "
        f"| rows in file={len(rows):,}")
    sample = [rows[s] for s in ("LAURUSLABS", "TCS") if s in rows]
    for s in sample:
        log(f"  {s['symbol']}: {float(s['shares_cr']):.3f} cr shares "
            f"(mcap {s['mcap_cr_at_fetch']} / price {s['price_at_fetch']})")
    if args.dry_run:
        local = os.path.join(os.environ.get("TEMP", "."), OUT_NAME)
        pd.DataFrame(list(rows.values()), columns=COLS).to_csv(local, index=False)
        log(f"[DRY-RUN] nothing written to Drive; table saved locally: {local}")
        return 0
    _save(drive, uni_id, rows)
    log(f"wrote universe/{OUT_NAME} ({len(rows):,} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
