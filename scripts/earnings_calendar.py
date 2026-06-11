r"""
earnings_calendar.py — Phase 3 / T2.5 — forthcoming results calendar.

Primary source: NSE event-calendar API (official; works from residential IPs,
may 403 from CI datacenter IPs — degrades gracefully). Filters to "Financial
Results" board meetings within the next N days.

BSE source: api.bseindia.com Corpforthresults (the Forth_Results.aspx page's
own XHR; needs a bseindia.com Referer). Merged with NSE, deduped on
(symbol, date) — dual-listed names keep the NSE row.

Usage:
    python scripts/earnings_calendar.py --days-ahead 1            # tomorrow
    python scripts/earnings_calendar.py --days-ahead 3 --dry-run
    python scripts/earnings_calendar.py --days-ahead 1 --email    # mail the list
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

import requests

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


def get_results_calendar(days_ahead: int = 1) -> list[dict]:
    """Companies with a Financial-Results board meeting in (today, today+days_ahead]."""
    today = date.today()
    start, end = today + timedelta(days=1), today + timedelta(days=days_ahead)
    seen, merged = set(), []
    for ev in nse_results_calendar(start, end) + bse_results_calendar(start, end):
        key = (ev["symbol"], ev["date"])
        if key not in seen:
            seen.add(key); merged.append(ev)
    merged.sort(key=lambda e: (e["date"], e["symbol"]))
    return merged


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
    ap.add_argument("--email", action="store_true", help="Email the list (mailer.py).")
    ap.add_argument("--dry-run", action="store_true", help="Print only.")
    args = ap.parse_args()

    events = get_results_calendar(args.days_ahead)
    print(f"Results calendar — next {args.days_ahead} day(s): {len(events)} company(ies)")
    for e in events:
        print(f"  {e['date']}  {e['symbol']:<14} {e['company'][:40]}  [{e['source']}]")

    if args.email and events and not args.dry_run:
        from mailer import send_email
        subj = f"📅 Results tomorrow: {len(events)} cos ({date.today()+timedelta(days=1)})"
        send_email(subj, _html_table(events))


if __name__ == "__main__":
    main()
