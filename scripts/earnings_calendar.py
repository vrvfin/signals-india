r"""
earnings_calendar.py — Phase 3 / T2.5 — forthcoming results calendar.

Primary source: NSE event-calendar API (official; works from residential IPs,
may 403 from CI datacenter IPs — degrades gracefully). Filters to "Financial
Results" board meetings within the next N days.

BSE source: api.bseindia.com Corpforthresults (the Forth_Results.aspx page's
own XHR; needs a bseindia.com Referer). Merged with NSE, deduped on
(symbol, date) — dual-listed names keep the NSE row.

Also the TRIGGER source for the daily financial refresh. Screener's
/results/latest/ feed is a fixed 25-item window that fully turns over inside an
hour at peak season, so the reporters it scrolls past never get their statements
re-scraped until the Monday full sweep. This calendar is complete by
construction (SEBI mandates prior board-meeting intimation), so `--persist`
stores it and `recent_reporter_symbols()` unions it with the feed.

BOTH APIs ARE STRICTLY FORTHCOMING — measured 2026-08-06: NSE's earliest row was
tomorrow, BSE's was today, nothing before. So the backward window the refresh
needs cannot be fetched; it has to be ACCUMULATED. `--persist` runs daily with a
wide --days-ahead, storing meetings before they happen (2,644 rows were visible
across the next 9 days), and results_calendar.parquet then answers "who reported
in the last N days" from its own history. A meeting that was never persisted
before it happened is unrecoverable from this source.

Usage:
    python scripts/earnings_calendar.py --days-ahead 1            # tomorrow
    python scripts/earnings_calendar.py --days-ahead 3 --dry-run
    python scripts/earnings_calendar.py --days-ahead 1 --email    # mail the list
    python scripts/earnings_calendar.py --days-ahead 14 --persist # feed the refresh
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

_SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

import pandas as pd
import requests

CAL_PARQUET = "results_calendar.parquet"
CAL_COLS = ["symbol", "company", "meeting_date", "purpose", "source",
            "first_seen_at"]

UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
      "Accept": "application/json, text/plain, */*",
      "Accept-Language": "en-US,en;q=0.9"}


def _parse_date(s: str):
    for fmt in ("%d-%b-%Y", "%Y-%m-%d", "%d-%m-%Y", "%d %b %Y"):
        try:
            return datetime.strptime(str(s).strip()[:11], fmt).date()
        except Exception:
            continue
    return None


def nse_results_calendar(start: date, end: date) -> list[dict]:
    """NSE 'Financial Results' board meetings with date in [start, end]."""
    try:
        s = requests.Session(); s.headers.update(UA)
        s.get("https://www.nseindia.com", timeout=12)               # bootstrap cookies
        r = s.get("https://www.nseindia.com/api/event-calendar", timeout=20)
        d = r.json()
        rows = d if isinstance(d, list) else d.get("data", [])
    except Exception as e:
        print(f"  NSE calendar fetch failed ({type(e).__name__}) — skipping.")
        return []
    out = []
    for x in rows:
        if "result" not in str(x.get("purpose", "")).lower():
            continue
        dt = _parse_date(x.get("date"))
        if dt and start <= dt <= end:
            out.append({"date": dt.isoformat(), "symbol": str(x.get("symbol", "")).strip(),
                        "company": str(x.get("company", "")).strip(),
                        "purpose": str(x.get("purpose", "")).strip(), "source": "NSE"})
    return out


BSE_FORTH_RESULTS = ("https://api.bseindia.com/BseIndiaAPI/api/"
                     "Corpforthresults/w?scripcode=")


def bse_results_calendar(start: date, end: date) -> list[dict]:
    """BSE forthcoming results board meetings with date in [start, end]."""
    try:
        r = requests.get(BSE_FORTH_RESULTS, timeout=20,
                         headers={**UA, "Referer": "https://www.bseindia.com/"})
        rows = r.json()
        if not isinstance(rows, list):
            rows = []
    except Exception as e:
        print(f"  BSE calendar fetch failed ({type(e).__name__}) — skipping.")
        return []
    out = []
    for x in rows:
        dt = _parse_date(x.get("meeting_date"))
        if dt and start <= dt <= end:
            sym = str(x.get("short_name", "")).strip() or str(x.get("scrip_Code", ""))
            out.append({"date": dt.isoformat(), "symbol": sym,
                        "company": str(x.get("Long_Name", "")).strip(),
                        "purpose": "Financial Results", "source": "BSE"})
    return out


def get_results_calendar(days_ahead: int = 1, days_back: int = 0) -> list[dict]:
    """Companies with a Financial-Results board meeting in the window.

    days_back=0 (default) keeps the original forward-only window
    (today, today+days_ahead] used by the "results tomorrow" email. days_back>0
    widens the FILTER to include today and earlier — but note both upstream APIs
    only publish forthcoming meetings, so in practice it just adds today's rows.
    Past meetings come from the persisted parquet, not from here.
    """
    today = date.today()
    start = (today - timedelta(days=days_back) if days_back > 0
             else today + timedelta(days=1))
    end = today + timedelta(days=days_ahead)
    nse = nse_results_calendar(start, end)
    bse = bse_results_calendar(start, end)
    print(f"  calendar {start}..{end}: NSE {len(nse)}, BSE {len(bse)}")
    seen, merged = set(), []
    for ev in nse + bse:
        key = (ev["symbol"], ev["date"])
        if key not in seen:
            seen.add(key); merged.append(ev)
    merged.sort(key=lambda e: (e["date"], e["symbol"]))
    return merged


def persist_calendar(events: list[dict], dry_run: bool = False) -> int:
    """Upsert `events` into company_repo/_index/results_calendar.parquet.

    Keyed on (symbol, meeting_date); first_seen_at is preserved across runs so a
    consumer can tell when a meeting was FIRST announced. An empty fetch (both
    APIs down / 403 from a datacenter IP) NEVER overwrites a good parquet.
    """
    if not events:
        print(f"  persist: 0 events — leaving {CAL_PARQUET} untouched.")
        return 0
    from _extractor_base import (get_drive, get_or_create_subfolder,
                                 load_parquet, save_parquet)
    now = datetime.now().isoformat(timespec="seconds")
    fresh = pd.DataFrame([{"symbol": e["symbol"], "company": e["company"],
                           "meeting_date": e["date"], "purpose": e["purpose"],
                           "source": e["source"], "first_seen_at": now}
                          for e in events])
    drive = get_drive()
    repo_id = get_or_create_subfolder(drive, os.environ["GDRIVE_FOLDER_ID"],
                                      "company_repo")
    index_id = get_or_create_subfolder(drive, repo_id, "_index")
    old = load_parquet(drive, index_id, CAL_PARQUET, CAL_COLS)
    if not old.empty:
        prev = {(r.symbol, r.meeting_date): r.first_seen_at
                for r in old.itertuples()}
        fresh["first_seen_at"] = [prev.get(k, now) for k in
                                  zip(fresh["symbol"], fresh["meeting_date"])]
        combined = pd.concat([old, fresh], ignore_index=True)
    else:
        combined = fresh
    combined = (combined.drop_duplicates(subset=["symbol", "meeting_date"],
                                         keep="last")
                .reset_index(drop=True))[CAL_COLS]
    n_new = len(combined) - len(old)
    if dry_run:
        print(f"  persist DRY-RUN: {len(fresh)} scraped, {n_new} new -> "
              f"{len(combined)} rows (no write)")
        return n_new
    save_parquet(drive, index_id, CAL_PARQUET, combined)
    print(f"  persist: {len(fresh)} scraped, {n_new} new -> {CAL_PARQUET} "
          f"now {len(combined)} rows")
    return n_new


# ---------------------------------------------------------------------------
#  Shared trigger for the daily financial refresh.
#  Imported by BOTH ingest_fundamentals.py (--recent-results-days) and
#  backfill_results_3stmt.py (--incremental) so the two cannot drift apart.
# ---------------------------------------------------------------------------

def _universe_maps(universe) -> tuple[dict, dict, dict]:
    """master_list.csv -> (isin->symbol, SYMBOL->symbol, bse_code->symbol).

    bse_code is yf_ticker minus the '.BO' suffix — the same convention
    ingest_fundamentals._token() uses to resolve BSE-only names on Screener.
    """
    by_isin, by_sym, by_bse = {}, {}, {}
    for r in universe.itertuples():
        sym = str(getattr(r, "symbol", "") or "").strip()
        if not sym:
            continue
        isin = str(getattr(r, "isin", "") or "").strip()
        if isin:
            by_isin.setdefault(isin, sym)
        by_sym.setdefault(sym.upper(), sym)
        yt = str(getattr(r, "yf_ticker", "") or "").strip()
        if yt.endswith(".BO"):
            by_bse.setdefault(yt[:-3], sym)
    return by_isin, by_sym, by_bse


def recent_reporter_symbols(universe, days: int, results_df=None,
                            calendar_df=None) -> tuple[set, dict]:
    """master_list `symbol`s whose results landed in the last `days`.

    UNION of two independent triggers:
      - results.parquet          — Screener's /results/latest/ feed. Only ever
                                   complete for what sat in its 25-item window
                                   at the moment we scraped.
      - results_calendar.parquet — NSE+BSE board-meeting calendar. Complete by
                                   construction, so it catches the reporters
                                   that window scrolled past.

    A feed row resolves by isin first, then by its slug (the NSE symbol, or a
    raw BSE code). The slug fallback matters: a third of recent feed rows carry
    a blank isin because load_slug_isin_map could not resolve them, and an
    isin-only join silently dropped every one of them.

    Returns (symbols, stats) — stats is for logging, never for control flow.
    """
    by_isin, by_sym, by_bse = _universe_maps(universe)
    cutoff = datetime.now() - timedelta(days=days)

    def _resolve(tok: str):
        tok = str(tok or "").strip()
        if not tok:
            return None
        return by_isin.get(tok) or by_sym.get(tok.upper()) or by_bse.get(tok)

    def _window(df, cols, floor_day: bool = False, drop_future: bool = False):
        """Rows whose timestamp (first matching column) is inside the window.

        floor_day for date-only columns: meeting_date is midnight, so comparing
        it against a mid-afternoon cutoff would drop the whole oldest day.
        drop_future for the calendar: it is stored ahead of time and mostly
        holds meetings that have NOT happened yet — those have no numbers to
        scrape, and without this cap the whole forward book (~2,600 rows in
        results season) would flood the refresh.
        """
        edge = cutoff.replace(hour=0, minute=0, second=0, microsecond=0) \
            if floor_day else cutoff
        for c in cols:
            if c not in df.columns:
                continue
            ts = pd.to_datetime(df[c], errors="coerce")
            keep = ts >= edge
            if drop_future:
                keep &= ts <= pd.Timestamp.now().normalize()
            return df[keep]
        return df          # no timestamp column: treat everything as in-window

    feed_toks, cal_toks = set(), set()
    if results_df is not None and not results_df.empty:
        df = _window(results_df, ("first_seen_at", "scraped_at"))
        blank = pd.Series("", index=df.index)
        isin = df["isin"].astype(str).str.strip() if "isin" in df.columns else blank
        slug = df["slug"].astype(str).str.strip() if "slug" in df.columns else blank
        feed_toks = set(isin.where(isin != "", slug)) - {""}
    if calendar_df is not None and not calendar_df.empty:
        df = _window(calendar_df, ("meeting_date",), floor_day=True,
                     drop_future=True)
        if "symbol" in df.columns:
            cal_toks = set(df["symbol"].astype(str).str.strip()) - {""}

    feed = {s for s in (_resolve(t) for t in feed_toks) if s}
    cal = {s for s in (_resolve(t) for t in cal_toks) if s}
    unresolved = [t for t in (feed_toks | cal_toks) if not _resolve(t)]
    stats = {"feed_tokens": len(feed_toks), "feed_symbols": len(feed),
             "cal_tokens": len(cal_toks), "cal_symbols": len(cal),
             "unresolved_n": len(unresolved), "unresolved": unresolved[:10]}
    return feed | cal, stats


def _html_table(events: list[dict]) -> str:
    rows = "".join(
        f"<tr><td>{e['date']}</td><td><b>{e['symbol']}</b></td>"
        f"<td>{e['company']}</td><td>{e['source']}</td></tr>" for e in events)
    return (f"<p><b>{len(events)} company(ies)</b> reporting results:</p>"
            f"<table border=1 cellpadding=5 cellspacing=0>"
            f"<tr><th>Date</th><th>Symbol</th><th>Company</th><th>Src</th></tr>"
            f"{rows}</table>")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--days-ahead", type=int, default=1, help="Window size (default 1=tomorrow).")
    ap.add_argument("--days-back", type=int, default=0,
                    help="Also include meetings this many days in the PAST — "
                         "those are the ones whose numbers just landed (0=off).")
    ap.add_argument("--persist", action="store_true",
                    help=f"Upsert the window into {CAL_PARQUET} on Drive, which "
                         "is what triggers the daily financial refresh.")
    ap.add_argument("--email", action="store_true", help="Email the list (mailer.py).")
    ap.add_argument("--dry-run", action="store_true", help="Print only.")
    args = ap.parse_args()

    events = get_results_calendar(args.days_ahead, args.days_back)
    span = (f"-{args.days_back}..+{args.days_ahead}" if args.days_back
            else f"next {args.days_ahead}")
    print(f"Results calendar — {span} day(s): {len(events)} company(ies)")
    for e in events:
        print(f"  {e['date']}  {e['symbol']:<14} {e['company'][:40]}  [{e['source']}]")

    if args.persist:
        persist_calendar(events, dry_run=args.dry_run)

    if args.email and events and not args.dry_run:
        from mailer import send_email
        subj = f"📅 Results tomorrow: {len(events)} cos ({date.today()+timedelta(days=1)})"
        send_email(subj, _html_table(events))


if __name__ == "__main__":
    main()
