r"""
backfill_results_3stmt.py — Phase 3 / T2 producer (NO Gemini).

Scrapes Screener's structured tables for the three statements and writes the raw
line items to company_repo/_index/financials_3stmt.parquet (frozen schema).

Screener reality (drives the schema):
  - QUARTERLY exists only for the P&L  (#quarters, ~12 quarters)
  - BALANCE-SHEET and CASH-FLOW are ANNUAL only (#balance-sheet / #cash-flow)
  - Consolidated preferred; falls back to standalone, recorded in `basis`
  - Receivables/Inventory are NOT raw lines (only Debtor/Inventory Days) and
    Capex/FCF are NOT raw lines (CFO/CFI/CFF are) — those live in T2-derived.

Reuse: ScreenerClient (cookie session + parse_table_section), _extractor_base
(Drive + column-safe parquet I/O), run_backfill.build_company_order (priority).

Usage:
    python scripts/backfill_results_3stmt.py --dry-run
    python scripts/backfill_results_3stmt.py --symbols TCS,INFY
    python scripts/backfill_results_3stmt.py --max-companies 50 --start 0
"""
from __future__ import annotations

import argparse
import atexit
import io
import os
import sys
import time
from datetime import datetime
from pathlib import Path

_SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

import pandas as pd
from bs4 import BeautifulSoup
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(_SCRIPTS_DIR), ".env"))

from screener_client import ScreenerClient, BASE_URL, CookieExpiredError
from _extractor_base import (get_drive, get_or_create_subfolder, find_file,
                             download_bytes, load_parquet, save_parquet,
                             acquire_lock, release_lock, log)
from earnings_calendar import CAL_PARQUET, recent_reporter_symbols
from run_backfill import build_company_order   # canonical priority order

LOCK_NAME = "_fin3stmt.lock"   # mutual-exclusion: backfill vs daily incremental

# ---- Frozen schema (see phase3-task-briefs) ----
FIN3_COLS = ["isin", "symbol", "statement", "line_item", "period", "period_type",
             "value", "basis", "qoq_pct", "yoy_pct", "scraped_at"]
PARQUET_NAME = "financials_3stmt.parquet"

QUARTERS_KEEP = 12     # ≥12 quarters of P&L (Sales/EBITDA/PAT/EPS) for charts/PEAD
ANNUAL_KEEP   = 7      # 7 FYs of annual 3-statement for deep-dive history
_CHECKPOINT_N = 500    # flush to Drive every N companies to survive SSL timeouts

# Canonical line_item -> list of candidate Screener row labels (first match wins).
# 'sum' entries are summed across several Screener rows.
INCOME_MAP = {
    "Sales":            ["Sales", "Revenue", "Sales +", "Revenue +"],
    "Operating Profit": ["Operating Profit"],
    "OPM %":            ["OPM %"],
    "Net Profit":       ["Net Profit", "Net Profit +"],
    "EPS":              ["EPS in Rs", "EPS", "EPS in ₹"],
    "Tax %":            ["Tax %"],
    "Interest":         ["Interest"],          # for interest_coverage (T2-derived)
}
# Screener "Ratios" section (#ratios) — efficiency days + ROCE over years (annual).
# Stored with statement="ratios"; passed through to financials_derived by build_derived.
RATIOS_MAP = {
    "receivable_days":  ["Debtor Days"],
    "inventory_days":   ["Inventory Days"],
    "payable_days":     ["Days Payable"],
    "wc_days":          ["Working Capital Days"],
    "roce_pct":         ["ROCE %"],
}
BALANCE_MAP = {
    "Net Worth":        {"sum": ["Equity Capital", "Reserves"]},
    "Borrowings":       ["Borrowings", "Borrowings +"],
    "Total Assets":     ["Total Assets"],
    "Net Block":        {"sum": ["Fixed Assets", "CWIP"]},
    "Investments":      ["Investments"],
    "Other Liabilities":["Other Liabilities", "Other Liabilities +"],
    "Other Assets":     ["Other Assets", "Other Assets +"],
}
CASHFLOW_MAP = {
    "CFO":           ["Cash from Operating Activity", "Cash from Operating Activity +"],
    "CFI":           ["Cash from Investing Activity", "Cash from Investing Activity +"],
    "CFF":           ["Cash from Financing Activity", "Cash from Financing Activity +"],
    "Net Cash Flow": ["Net Cash Flow"],
    "FCF":           ["Free Cash Flow"],   # Screener exposes this directly
}


def _pct(a, b):
    if a is None or b in (None, 0):
        return None
    try:
        return round((a - b) / abs(b) * 100.0, 2)
    except Exception:
        return None


def _row_series(rows: dict, candidates) -> list | None:
    """Return the values list for the first matching Screener row label."""
    for c in candidates:
        if c in rows:
            return rows[c]
    return None


def _sum_series(rows: dict, labels) -> list | None:
    """Element-wise sum of several Screener rows (None treated as 0; all-None -> None)."""
    series = [rows[l] for l in labels if l in rows]
    if not series:
        return None
    n = max(len(s) for s in series)
    out = []
    for i in range(n):
        vals = [s[i] for s in series if i < len(s) and s[i] is not None]
        out.append(sum(vals) if vals else None)
    return out


def _resolve_series(rows: dict, mapping_entry):
    if isinstance(mapping_entry, dict) and "sum" in mapping_entry:
        return _sum_series(rows, mapping_entry["sum"])
    return _row_series(rows, mapping_entry)


def _emit(statement, line_item, headers, values, period_type, isin, symbol,
          basis, now, keep) -> list[dict]:
    """Build frozen-schema rows for one line_item, with QoQ/YoY computed on the
    FULL series then tail-sliced to `keep` periods."""
    if not values:
        return []
    m = min(len(headers), len(values))
    headers, values = headers[:m], values[:m]
    out = []
    for i in range(m):
        qoq = _pct(values[i], values[i - 1]) if (period_type == "quarterly" and i >= 1) else None
        if period_type == "quarterly":
            yoy = _pct(values[i], values[i - 4]) if i >= 4 else None
        else:
            yoy = _pct(values[i], values[i - 1]) if i >= 1 else None
        out.append({
            "isin": isin, "symbol": symbol, "statement": statement,
            "line_item": line_item, "period": str(headers[i]),
            "period_type": period_type, "value": values[i], "basis": basis,
            "qoq_pct": qoq, "yoy_pct": yoy, "scraped_at": now,
        })
    return out[-keep:]


def fetch_with_basis(client: ScreenerClient, symbol: str):
    """Return (soup, basis) — consolidated preferred, standalone fallback."""
    for variant, basis in (("consolidated/", "consolidated"), ("", "standalone")):
        url = f"{BASE_URL}/company/{symbol}/{variant}"
        client._wait()
        try:
            r = client.session.get(url, timeout=30, allow_redirects=True)
        except Exception:
            continue
        if r.status_code == 404:
            continue
        client._check_auth(r, symbol)
        if r.status_code == 200 and 'id="profit-loss"' in r.text:
            return BeautifulSoup(r.text, "lxml"), basis
    return None, None


def scrape_company(client: ScreenerClient, isin: str, symbol: str) -> list[dict]:
    soup, basis = fetch_with_basis(client, symbol)
    if soup is None:
        log(f"    no Screener page for {symbol}")
        return []
    now = datetime.now().isoformat(timespec="seconds")
    rows: list[dict] = []

    # Income — quarterly (#quarters); keep series to compute TTM
    q = client.parse_table_section(soup, "quarters")
    q_series: dict[str, list] = {}
    for li, cand in INCOME_MAP.items():
        s = _resolve_series(q["rows"], cand)
        if s:
            q_series[li] = s
            rows += _emit("income", li, q["headers"], s,
                          "quarterly", isin, symbol, basis, now, QUARTERS_KEEP)

    # TTM (trailing-4-quarters) — COMPUTED, since this Screener view exposes no TTM
    # column. Flow metrics summed; OPM% recomputed. Auto-refreshes as quarters update.
    def _last4(li):
        s = q_series.get(li)
        if not s:
            return None
        vals = [v for v in s[-4:] if v is not None]
        return round(sum(vals), 2) if len(vals) == 4 else None
    ttm_sales, ttm_op = _last4("Sales"), _last4("Operating Profit")
    ttm_vals = {
        "Sales": ttm_sales, "Operating Profit": ttm_op,
        "Net Profit": _last4("Net Profit"), "EPS": _last4("EPS"),
        "OPM %": (round(ttm_op / ttm_sales * 100, 2)
                  if (ttm_op is not None and ttm_sales) else None),
    }
    for li, v in ttm_vals.items():
        if v is not None:
            rows.append({"isin": isin, "symbol": symbol, "statement": "income",
                         "line_item": li, "period": "TTM", "period_type": "ttm",
                         "value": v, "basis": basis, "qoq_pct": None,
                         "yoy_pct": None, "scraped_at": now})

    # Income — annual (#profit-loss)
    pl = client.parse_table_section(soup, "profit-loss")
    for li, cand in INCOME_MAP.items():
        rows += _emit("income", li, pl["headers"], _resolve_series(pl["rows"], cand),
                      "annual", isin, symbol, basis, now, ANNUAL_KEEP)

    # Balance sheet — annual only
    bs = client.parse_table_section(soup, "balance-sheet")
    for li, cand in BALANCE_MAP.items():
        rows += _emit("balance", li, bs["headers"], _resolve_series(bs["rows"], cand),
                      "annual", isin, symbol, basis, now, ANNUAL_KEEP)

    # Cash flow — annual only
    cf = client.parse_table_section(soup, "cash-flow")
    for li, cand in CASHFLOW_MAP.items():
        rows += _emit("cashflow", li, cf["headers"], _resolve_series(cf["rows"], cand),
                      "annual", isin, symbol, basis, now, ANNUAL_KEEP)

    # Ratios — annual efficiency days + ROCE (Screener computes these; pass through)
    rt = client.parse_table_section(soup, "ratios")
    for li, cand in RATIOS_MAP.items():
        rows += _emit("ratios", li, rt["headers"], _resolve_series(rt["rows"], cand),
                      "annual", isin, symbol, basis, now, ANNUAL_KEEP)

    log(f"    {symbol}: {len(rows)} rows (basis={basis})")
    return rows


def _read_parquet(drive, folder_id, name):
    """Drive parquet -> DataFrame, or None when absent/unreadable."""
    fid = find_file(drive, folder_id, name)
    if not fid:
        return None
    try:
        return pd.read_parquet(io.BytesIO(download_bytes(drive, fid)))
    except Exception as e:
        log(f"  --incremental: could not read {name} ({str(e)[:60]})")
        return None


def incremental_companies(drive, root_id, index_id, order, days: int) -> list[dict]:
    """NEW reporters within `days`, as {symbol,isin,name} entries from `order`.

    Trigger = results.parquet (Screener's /results/latest/ feed) UNION
    results_calendar.parquet (NSE+BSE board-meeting calendar), resolved by the
    shared earnings_calendar.recent_reporter_symbols so this and
    ingest_fundamentals cannot drift apart. The feed alone misses whatever
    scrolled out of its fixed 25-item window between two scrapes.

    An empty result is valid: nothing newly declared.
    """
    uni_id = get_or_create_subfolder(drive, root_id, "universe")
    uni_fid = find_file(drive, uni_id, "master_list.csv")
    if not uni_fid:
        log("  --incremental: universe/master_list.csv not found — skipping.")
        return []
    universe = pd.read_csv(io.BytesIO(download_bytes(drive, uni_fid)))
    res = _read_parquet(drive, index_id, "results.parquet")
    cal = _read_parquet(drive, index_id, CAL_PARQUET)
    if res is None and cal is None:
        log(f"  --incremental: no results.parquet and no {CAL_PARQUET} — "
            "nothing to refresh.")
        return []
    want, stats = recent_reporter_symbols(universe, days,
                                          results_df=res, calendar_df=cal)
    by_sym = {str(c.get("symbol", "")).upper(): c for c in order}
    out = [by_sym[s.upper()] for s in sorted(want) if s.upper() in by_sym]
    log(f"  --incremental: {len(out)} reporter(s) to refresh within {days}d "
        f"(feed {stats['feed_symbols']} of {stats['feed_tokens']}, "
        f"calendar {stats['cal_symbols']} of {stats['cal_tokens']}, "
        f"{stats['unresolved_n']} token(s) unresolved)")
    return out


def _checkpoint_save(drive, index_id: str,
                     pending_rows: list, checkpoint_isins: set) -> None:
    """Upsert pending_rows for checkpoint_isins into the Drive parquet.

    Clears both lists in-place on success so the next batch starts fresh.
    Only replaces rows for ISINs in *this* checkpoint — earlier checkpoints
    are preserved in the parquet exactly as saved.
    """
    if not pending_rows:
        return
    new_df = pd.DataFrame(pending_rows, columns=FIN3_COLS)
    existing = load_parquet(drive, index_id, PARQUET_NAME, FIN3_COLS)
    if not existing.empty and checkpoint_isins:
        existing = existing[~existing["isin"].astype(str).isin(checkpoint_isins)]
    merged = pd.concat([existing, new_df], ignore_index=True)
    save_parquet(drive, index_id, PARQUET_NAME, merged)
    log(f"  [checkpoint] flushed {len(pending_rows)} rows "
        f"({len(checkpoint_isins)} co), parquet total: {len(merged)}")
    pending_rows.clear()
    checkpoint_isins.clear()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--symbols", default="", help="Comma list of NSE symbols (skips priority list).")
    ap.add_argument("--incremental", action="store_true",
                    help="Refresh only today's reporters (from results.parquet) — daily/PEAD.")
    ap.add_argument("--recent-days", type=int, default=2,
                    help="With --incremental: results refreshed within N days (default 2).")
    ap.add_argument("--max-companies", type=int, default=0, help="Cap companies this run (0=all).")
    ap.add_argument("--start", type=int, default=0, help="Skip first N companies (resume).")
    ap.add_argument("--sleep", type=float, default=1.0, help="Seconds between companies.")
    ap.add_argument("--no-lock", action="store_true", help="Skip the Drive lock (testing only).")
    ap.add_argument("--dry-run", action="store_true", help="Scrape + print sample; no Drive write.")
    args = ap.parse_args()

    print("Phase 3 / T2 — 3-statement backfill (Screener, no Gemini)")
    print("-" * 60)

    drive = get_drive()
    root_id = os.environ["GDRIVE_FOLDER_ID"]
    repo_id = get_or_create_subfolder(drive, root_id, "company_repo")
    index_id = get_or_create_subfolder(drive, repo_id, "_index")

    # Mutual-exclusion lock so the rolling backfill and the daily incremental
    # refresh can't clobber financials_3stmt concurrently (T2.4).
    if not args.no_lock and not args.dry_run:
        if not acquire_lock(drive, index_id, LOCK_NAME, "backfill_results_3stmt",
                            max_age_min=240):
            sys.exit(0)
        atexit.register(release_lock, drive, index_id, LOCK_NAME)

    # Company list
    explicit = [s.strip() for s in args.symbols.split(",") if s.strip()]
    if args.incremental:
        order = build_company_order(drive, root_id)
        companies = incremental_companies(drive, root_id, index_id, order,
                                          args.recent_days)
    elif explicit:
        order = build_company_order(drive, root_id)  # resolve ISIN via universe
        by_sym = {c["symbol"].upper(): c for c in order}
        companies = [by_sym.get(s.upper(), {"symbol": s.upper(), "isin": "", "name": s.upper()})
                     for s in explicit]
    else:
        companies = build_company_order(drive, root_id)

    companies = companies[args.start:]
    if args.max_companies:
        companies = companies[: args.max_companies]
    if not companies:
        sys.exit("No companies resolved — check universe/master_list.csv.")

    try:
        client = ScreenerClient()
    except Exception as e:
        sys.exit(f"Screener client error: {e}")

    log(f"Companies this run: {len(companies)} (start={args.start})")
    checkpoint_rows: list[dict] = []   # rows since last checkpoint (cleared on flush)
    checkpoint_isins: set[str] = set() # ISINs since last checkpoint (cleared on flush)
    done_isins: set[str] = set()       # all processed ISINs (for summary + dry-run)
    dry_all_rows: list[dict] = []      # dry-run accumulator (not used in live mode)
    for i, co in enumerate(companies, 1):
        sym, isin = co["symbol"], co.get("isin", "")
        log(f"[{i}/{len(companies)}] {sym}  ISIN={isin or '?'}")
        try:
            rows = scrape_company(client, isin, sym)
        except CookieExpiredError:
            sys.exit("Screener cookie expired — refresh SCREENER_SESSION_COOKIE in .env.")
        except Exception as e:
            log(f"    error: {str(e)[:100]}")
            continue
        if args.dry_run:
            dry_all_rows += rows
        else:
            checkpoint_rows += rows
            if isin:
                checkpoint_isins.add(str(isin))
        if isin:
            done_isins.add(str(isin))
        if args.sleep and i < len(companies):
            time.sleep(args.sleep)

        # Periodic checkpoint — flush every _CHECKPOINT_N companies to avoid
        # losing all data if a large final Drive upload hits an SSL timeout.
        if not args.dry_run and i % _CHECKPOINT_N == 0 and checkpoint_rows:
            _checkpoint_save(drive, index_id, checkpoint_rows, checkpoint_isins)

    if args.dry_run:
        new_df = pd.DataFrame(dry_all_rows, columns=FIN3_COLS)
        print("-" * 60)
        print(f"DRY RUN — {len(new_df)} rows scraped (no Drive write). Sample:")
        print(new_df.head(20).to_string(index=False))
        if not new_df.empty:
            print("\nline_item coverage:")
            print(new_df.groupby(["statement", "period_type"])["line_item"].nunique().to_string())
        return

    # Flush any remaining rows not covered by the last periodic checkpoint.
    if checkpoint_rows:
        _checkpoint_save(drive, index_id, checkpoint_rows, checkpoint_isins)
    elif not done_isins:
        print("No rows scraped — nothing written.")
        return

    final = load_parquet(drive, index_id, PARQUET_NAME, FIN3_COLS)
    print("-" * 60)
    print(f"Companies scraped : {len(done_isins)}")
    print(f"Parquet total rows: {len(final)}")
    print(f"Output: company_repo/_index/{PARQUET_NAME}")


if __name__ == "__main__":
    main()
