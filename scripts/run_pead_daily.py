r"""
run_pead_daily.py — Phase 3 / T2.5 daily mailer (NO Gemini).

Sends the two daily emails (reuses mailer.py / the Gmail config):
  A) "Results tomorrow" — companies with a Financial-Results board meeting in the
     next --days-ahead days (earnings_calendar, NSE source).
  B) "Earnings vs guidance (today)" — companies whose results refreshed TODAY
     (results.parquet), with their PEAD verdict (pead_flags = guidance vs actual).

Intended to run AFTER the daily refresh + flag steps (see pead.yml):
    scrape_results_table.py  →  backfill_results_3stmt.py --incremental
    →  build_pead_flags.py   →  run_pead_daily.py

Usage:
    python scripts/run_pead_daily.py --dry-run
    python scripts/run_pead_daily.py --days-ahead 1
"""
from __future__ import annotations

import argparse
import io
import os
import sys
from datetime import date
from pathlib import Path

_SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

import pandas as pd
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(_SCRIPTS_DIR), ".env"))

from _extractor_base import (get_drive, get_or_create_subfolder, find_file,
                             download_bytes, load_parquet, log)
from build_pead_flags import PEAD_COLS
from earnings_calendar import get_results_calendar, _html_table
from mailer import send_email

_VERDICT_COLOR = {"BEAT": "#27ae60", "MISS": "#e74c3c", "INLINE": "#777", "NA": "#aaa"}


def _today_reporters(drive, index_id) -> pd.DataFrame:
    """Companies whose results.parquet rows were scraped today (= reported today)."""
    fid = find_file(drive, index_id, "results.parquet")
    if not fid:
        return pd.DataFrame()
    try:
        res = pd.read_parquet(io.BytesIO(download_bytes(drive, fid)))
    except Exception:
        return pd.DataFrame()
    if res.empty or "isin" not in res.columns or "scraped_at" not in res.columns:
        return pd.DataFrame()
    res = res.copy()
    res["_d"] = pd.to_datetime(res["scraped_at"], errors="coerce").dt.date
    today = res[res["_d"] == date.today()]
    cols = [c for c in ["isin", "symbol", "company_name", "latest_q"] if c in today.columns]
    return today[cols].drop_duplicates("isin") if cols else pd.DataFrame()


def _email_b_html(reporters: pd.DataFrame, pead: pd.DataFrame) -> str | None:
    if reporters.empty:
        return None
    rows_html = []
    for _, rep in reporters.iterrows():
        isin = str(rep.get("isin", ""))
        sym = str(rep.get("symbol", isin))
        comp = str(rep.get("company_name", ""))
        qtr = str(rep.get("latest_q", ""))
        f = pead[pead["isin"].astype(str) == isin] if not pead.empty else pd.DataFrame()
        if f.empty:
            verdict_cell = "<i>no guidance to compare</i>"
        else:
            bits = []
            for _, fr in f.iterrows():
                v = str(fr.get("verdict", "NA"))
                color = _VERDICT_COLOR.get(v, "#777")
                bits.append(f'{fr.get("metric","")}: '
                            f'<b style="color:{color}">{v}</b> '
                            f'({fr.get("delta_pct","")}%)')
            verdict_cell = "<br>".join(bits)
        rows_html.append(f"<tr><td><b>{sym}</b></td><td>{comp[:34]}</td>"
                         f"<td>{qtr}</td><td>{verdict_cell}</td></tr>")
    return (f"<p><b>{len(reporters)} company(ies)</b> reported today — guidance vs actual:</p>"
            f"<table border=1 cellpadding=5 cellspacing=0>"
            f"<tr><th>Symbol</th><th>Company</th><th>Qtr</th>"
            f"<th>Guidance vs Actual</th></tr>{''.join(rows_html)}</table>"
            f"<p style='font-size:11px;color:#999'>BEAT/MISS = actual beat/missed guidance "
            f"by &gt;2 (% for levels, pp for growth/margin).</p>")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--days-ahead", type=int, default=1, help="Calendar window (default 1).")
    ap.add_argument("--dry-run", action="store_true", help="Print emails; do not send.")
    args = ap.parse_args()

    drive = get_drive()
    root_id = os.environ["GDRIVE_FOLDER_ID"]
    repo_id = get_or_create_subfolder(drive, root_id, "company_repo")
    index_id = get_or_create_subfolder(drive, repo_id, "_index")

    # ---- Email B: today's earnings vs guidance ----
    reporters = _today_reporters(drive, index_id)
    pead = load_parquet(drive, index_id, "pead_flags.parquet", PEAD_COLS)
    html_b = _email_b_html(reporters, pead)

    # ---- Email A: tomorrow's announcers ----
    events = get_results_calendar(args.days_ahead)
    html_a = _html_table(events) if events else None

    if args.dry_run:
        print("=== EMAIL A (results tomorrow) ===")
        print(f"{len(events)} events" if events else "(none)")
        print("\n=== EMAIL B (today's earnings vs guidance) ===")
        print(html_b or "(no reporters today)")
        return

    sent = 0
    if html_b:
        sent += send_email(f"📊 Earnings today vs guidance — {date.today()}", html_b)
    if html_a:
        sent += send_email(f"📅 Results tomorrow ({len(events)} cos)", html_a)
    print(f"run_pead_daily: {sent} email(s) sent "
          f"(reporters today={0 if reporters is None else len(reporters)}, "
          f"calendar={len(events)}).")


if __name__ == "__main__":
    main()
