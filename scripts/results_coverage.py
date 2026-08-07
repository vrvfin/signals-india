r"""
results_coverage.py — which companies are actually MISSING the current quarter.

The weekly sweep used to visit all ~5,600 companies blindly, one per second, for
~290 minutes. Most of that work is wasted: a company either already has the
current quarter stored, or has not reported it yet. This module computes the
real gap so the sweep fetches only that.

Three rules, in order (see build_gap_list):
  1. summary.latest_quarter_label == expected period  -> SKIP, already have it
  2. board meeting is in the FUTURE                   -> SKIP, not due yet
  3. no calendar entry and still inside the SEBI      -> SKIP, not due yet
     filing deadline
  everything else                                     -> FETCH

Rule 3 is what stops the whole market being flagged mid-season: the SEBI LODR
deadline is 45 days after quarter end (60 for Q4/annual), so on 07-Aug-2026
hundreds of companies legitimately have no Q1 FY27 numbers and are not late.

Inputs, all already on Drive — nothing new is scraped:
  fundamentals/summary.parquet                  -> latest_quarter_label per symbol
  company_repo/_index/results_calendar.parquet  -> board-meeting dates
  universe/master_list.csv                      -> the company list

Deliberately self-contained: screener_scraper.current_season_key() does the same
date->quarter mapping, but that module is UNTRACKED and therefore absent in CI
(commit 46f33b7 fixed a crash caused by importing it), so the mapping is
reimplemented here rather than imported.

Usage:
    python scripts/results_coverage.py                 # report the gap, no write
    python scripts/results_coverage.py --as-of 2026-11-05
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from datetime import date, datetime, timedelta

_SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

import pandas as pd

from earnings_calendar import _universe_maps

# Screener labels a quarter by its ENDING month: Q1 FY27 (Apr-Jun 2026) is
# "Jun 2026". Q4 is the only one whose ending month falls in the FY's own end
# year — hence the +1.
_Q_END = {"Q1": ("Jun", 6, 30, 0), "Q2": ("Sep", 9, 30, 0),
          "Q3": ("Dec", 12, 31, 0), "Q4": ("Mar", 3, 31, 1)}

# SEBI LODR: unaudited quarterly results within 45 days of quarter end;
# the Q4 / annual audited results get 60.
_DEADLINE_DAYS = {"Q1": 45, "Q2": 45, "Q3": 45, "Q4": 60}

# Instruments that never declare earnings: ETFs, index trackers, fund units.
# They stay in master_list (they are tradable and OHLCV/features/gallery read
# that file), but chasing quarterly results for them is pointless — past the
# filing deadline every one of them would otherwise look "overdue" forever.
_NO_EARNINGS = re.compile(
    r"\bETF\b|\bBeES\b|\bINDEX\s+FUND\b|MUTUAL\s+FUND|SEGREGATED\s+PORTFOLIO|"
    r"\bIDCW\b|(?:DIRECT|REGULAR)\s+PLAN|\bLONG[-\s]?SHORT\s+FUND\b",
    re.IGNORECASE)


def current_season(today: date | None = None) -> tuple[str, int]:
    """(quarter, fy_end_year) for the quarter whose results are being announced.

    Mirrors screener_scraper.current_season_key():
      Apr-Jun -> Q4 of the FY that just ended in March
      Jul-Sep -> Q1 of the FY that ends next March
      Oct-Dec -> Q2 of that same FY
      Jan-Mar -> Q3 of the FY ending this March
    """
    t = today or date.today()
    m, y = t.month, t.year
    if m in (4, 5, 6):
        return "Q4", y
    if m in (7, 8, 9):
        return "Q1", y + 1
    if m in (10, 11, 12):
        return "Q2", y + 1
    return "Q3", y


def expected_period_label(today: date | None = None) -> str:
    """Screener's period string for the quarter being announced — e.g. 'Jun 2026'."""
    q, fy_end = current_season(today)
    mon, _, _, bump = _Q_END[q]
    return f"{mon} {fy_end - 1 + bump}"


def quarter_end_date(today: date | None = None) -> date:
    """Calendar date the announced quarter ended — e.g. 2026-06-30."""
    q, fy_end = current_season(today)
    _, mm, dd, bump = _Q_END[q]
    return date(fy_end - 1 + bump, mm, dd)


def filing_deadline(today: date | None = None) -> date:
    """Last day a company may file the announced quarter without being late."""
    q, _ = current_season(today)
    return quarter_end_date(today) + timedelta(days=_DEADLINE_DAYS[q])


def build_gap_list(universe, summary=None, calendar_df=None,
                   today: date | None = None,
                   recheck_days: int = 7) -> tuple[list[str], dict]:
    """Symbols whose current-quarter numbers are due-or-late and NOT yet stored.

    recheck_days bounds the long tail. Once the filing deadline passes, every
    company WITHOUT a calendar entry counts as overdue — but many are dormant
    shells that will never report, so without a brake they would be re-fetched
    on every run forever. Those are re-checked at most once every recheck_days.

    Companies WITH a past board meeting are exempt from that brake and retried
    every run: they demonstrably reported, so Screener will carry the numbers
    within a day or two and we want them the moment it does.

    Returns (symbols, stats). stats is diagnostic only, never control flow.
    """
    t = today or date.today()
    expected = expected_period_label(t)
    deadline = filing_deadline(t)

    # Drop the instruments that never report earnings before anything else, so
    # they can neither enter the fetch list nor inflate the skip counters.
    uni = universe
    n_no_earnings = 0
    if "name" in uni.columns:
        mask = uni["name"].astype(str).str.contains(_NO_EARNINGS)
        n_no_earnings = int(mask.sum())
        uni = uni[~mask]

    all_syms = [str(s).strip() for s in uni["symbol"].astype(str)
                if str(s).strip()]

    # 1. already stored
    have = set()
    if summary is not None and not summary.empty \
            and "latest_quarter_label" in summary.columns:
        hit = summary[summary["latest_quarter_label"].astype(str).str.strip()
                      == expected]
        have = set(hit["symbol"].astype(str).str.strip())

    # 2/3. when is each company due? Calendar rows are keyed by NSE symbol or
    # BSE code, so resolve them onto master_list symbols the same way the
    # refresh trigger does.
    by_isin, by_sym, by_bse = _universe_maps(universe)
    due_on: dict[str, date] = {}
    if calendar_df is not None and not calendar_df.empty \
            and "symbol" in calendar_df.columns:
        md = pd.to_datetime(calendar_df["meeting_date"], errors="coerce")
        for tok, m in zip(calendar_df["symbol"].astype(str).str.strip(), md):
            if pd.isna(m):
                continue
            sym = by_isin.get(tok) or by_sym.get(tok.upper()) or by_bse.get(tok)
            if not sym:
                continue
            d = m.date()
            # keep the LATEST scheduled meeting for the company
            if sym not in due_on or d > due_on[sym]:
                due_on[sym] = d

    # When we last looked at each company, for the recheck brake below.
    last_seen: dict[str, date] = {}
    if summary is not None and not summary.empty and "fetched_at" in summary.columns:
        fa = pd.to_datetime(summary["fetched_at"], errors="coerce")
        for sym, f in zip(summary["symbol"].astype(str).str.strip(), fa):
            if not pd.isna(f):
                last_seen[sym] = f.date()

    fetch, skip_have, skip_future, skip_early, skip_recent = [], 0, 0, 0, 0
    for sym in all_syms:
        if sym in have:
            skip_have += 1
            continue
        meeting = due_on.get(sym)
        if meeting is not None:
            if meeting > t:
                skip_future += 1          # scheduled, hasn't happened yet
                continue
            # reported already — retry every run until the numbers show up
        elif t <= deadline:
            skip_early += 1               # no entry, still inside the deadline
            continue
        else:
            # Overdue with no calendar entry: likely dormant. Back off.
            seen = last_seen.get(sym)
            if seen is not None and (t - seen).days < recheck_days:
                skip_recent += 1
                continue
        fetch.append(sym)

    stats = {
        "expected": expected,
        "quarter_end": quarter_end_date(t).isoformat(),
        "deadline": deadline.isoformat(),
        "universe": len(all_syms),
        "skip_no_earnings": n_no_earnings,
        "skip_already_have": skip_have,
        "skip_not_yet_due": skip_future,
        "skip_inside_deadline": skip_early,
        "skip_rechecked_recently": skip_recent,
        "fetch": len(fetch),
        "calendar_resolved": len(due_on),
    }
    return fetch, stats


def log_stats(stats: dict, log=print) -> None:
    """One consistent block wherever the gap is computed."""
    log(f"  expected quarter        : {stats['expected']} "
        f"(ended {stats['quarter_end']}, filing deadline {stats['deadline']})")
    log(f"  universe (earnings-bearing): {stats['universe']} "
        f"(excluded {stats['skip_no_earnings']} ETF/index/fund units)")
    log(f"  skip — already stored   : {stats['skip_already_have']}")
    log(f"  skip — meeting in future: {stats['skip_not_yet_due']}")
    log(f"  skip — inside deadline  : {stats['skip_inside_deadline']}")
    log(f"  skip — checked recently : {stats['skip_rechecked_recently']}")
    log(f"  TO FETCH                : {stats['fetch']}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--as-of", default="", help="Evaluate as if today were YYYY-MM-DD.")
    ap.add_argument("--show", type=int, default=25, help="List this many gap symbols.")
    args = ap.parse_args()

    today = (datetime.strptime(args.as_of, "%Y-%m-%d").date()
             if args.as_of else date.today())

    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(_SCRIPTS_DIR), ".env"))
    import ingest_fundamentals as IF
    from earnings_calendar import CAL_PARQUET

    drive = IF.get_drive()
    root = os.environ["GDRIVE_FOLDER_ID"]
    fund = IF.get_or_create_subfolder(drive, root, "fundamentals")
    repo = IF.get_or_create_subfolder(drive, root, "company_repo")
    idx = IF.get_or_create_subfolder(drive, repo, "_index")

    universe = IF.download_csv(
        drive, IF.find_file(drive, IF.get_or_create_subfolder(drive, root, "universe"),
                            "master_list.csv"))
    sfid = IF.find_file(drive, fund, "summary.parquet")
    cfid = IF.find_file(drive, idx, CAL_PARQUET)
    summary = IF.download_parquet(drive, sfid) if sfid else None
    calendar = IF.download_parquet(drive, cfid) if cfid else None
    if calendar is None:
        print(f"  WARNING: {CAL_PARQUET} missing — every company past the "
              "deadline will be treated as a gap.")

    gap, stats = build_gap_list(universe, summary, calendar, today)
    print(f"Results coverage as of {today}")
    print("-" * 58)
    log_stats(stats)
    if gap:
        print(f"\n  first {min(args.show, len(gap))} gap symbols:")
        for s in gap[:args.show]:
            print(f"    {s}")


if __name__ == "__main__":
    main()
