r"""
run_pead_daily.py — Phase 3 / T2.5 daily mailer (NO Gemini).

Sends the two daily emails (reuses mailer.py / the Gmail config):
  A) "Results tomorrow" — companies with a Financial-Results board meeting in the
     next --days-ahead days (earnings_calendar, NSE source).
  B) "Earnings vs guidance (last 24h)" — companies whose results FIRST appeared
     on Screener's latest-results feed since the previous scrape (first_seen_at
     in results.parquet), with actual Sales/PAT numbers and their PEAD verdict
     (pead_flags = guidance vs actual).

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
from mailer import send_email, load_mail_settings

_VERDICT_COLOR = {"BEAT": "#27ae60", "MISS": "#e74c3c", "INLINE": "#777", "NA": "#aaa"}


def _load_results(drive, index_id) -> pd.DataFrame:
    """Full results.parquet (scrape_results_table output) or empty frame."""
    fid = find_file(drive, index_id, "results.parquet")
    if not fid:
        return pd.DataFrame()
    try:
        res = pd.read_parquet(io.BytesIO(download_bytes(drive, fid)))
    except Exception:
        return pd.DataFrame()
    if res.empty or "isin" not in res.columns:
        return pd.DataFrame()
    return res


def _recent_reporters(res: pd.DataFrame, hours: int = 20) -> pd.DataFrame:
    """Companies whose (slug, latest_q) FIRST appeared on Screener's feed within
    `hours` = declared since the previous daily scrape. 20h sits safely between
    the same-run delay (minutes) and the shortest scrape-to-scrape gap (~23h),
    so nothing is missed or double-reported. Falls back to scraped_at for
    parquets written before first_seen_at existed."""
    if res.empty:
        return pd.DataFrame()
    ts_col = "first_seen_at" if "first_seen_at" in res.columns else "scraped_at"
    if ts_col not in res.columns:
        return pd.DataFrame()
    r = res.copy()
    r["_ts"] = pd.to_datetime(r[ts_col], errors="coerce")
    recent = r[r["_ts"] >= pd.Timestamp.now() - pd.Timedelta(hours=hours)]
    if recent.empty:
        return pd.DataFrame()
    return (recent.sort_values("_ts")
            .groupby("slug", as_index=False)
            .agg(isin=("isin", "first"), company_name=("company_name", "first"),
                 latest_q=("latest_q", "first"), result_dt=("_ts", "min")))


def _isin_symbol_map(drive, index_id) -> dict:
    """isin -> NSE symbol from company_universe.csv (email display only)."""
    fid = find_file(drive, index_id, "company_universe.csv")
    if not fid:
        return {}
    try:
        uni = pd.read_csv(io.BytesIO(download_bytes(drive, fid)))
    except Exception:
        return {}
    m = {}
    for _, r in uni.iterrows():
        isin = str(r.get("isin", "")).strip()
        sym = str(r.get("nse_symbol", "")).strip()
        if isin and sym and sym.lower() != "nan":
            m[isin] = sym
    return m


def _fmt_val(v) -> str:
    try:
        return f"{float(v):,.0f}"
    except (TypeError, ValueError):
        return "-"


def _actual_cell(comp_rows: pd.DataFrame, *terms: str) -> str:
    """'12,345 (+8%)' for the first metric row containing any term (YoY colored)."""
    for t in terms:
        hit = comp_rows[comp_rows["metric"].astype(str).str.lower()
                        .str.contains(t, na=False)]
        if not hit.empty:
            r0 = hit.iloc[0]
            cell = _fmt_val(r0.get("latest_val"))
            yoy = pd.to_numeric(r0.get("yoy_pct"), errors="coerce")
            if pd.notna(yoy):
                color = "#27ae60" if yoy >= 0 else "#e74c3c"
                cell += f' <span style="color:{color}">({yoy:+.0f}%)</span>'
            return cell
    return "-"


_SRC_LABEL = {"concall": "concall", "presentation": "investor presentation",
              "annual_report": "annual report"}


def _verdict_sentence(fr, v: str, color: str) -> str:
    """Self-explanatory guidance-vs-actual line (user 2026-07-12), e.g.
    'Revenue growth guided ~15% for FY26 (concall) → actual +21% YoY →
    BEAT by +6.2pp'. kind decides the units: growth/margin compare in
    percentage-points; level compares ₹Cr and the delta is %."""
    metric = str(fr.get("metric", "") or "").title()
    metric = {"Pat": "PAT", "Ebitda": "EBITDA", "Eps": "EPS",
              "Opm": "OPM"}.get(metric, metric)
    fy = str(fr.get("quarter", "") or "")
    src = _SRC_LABEL.get(str(fr.get("guidance_source") or ""), "concall")
    kind = str(fr.get("kind") or "")
    gv, av = fr.get("guided_value"), fr.get("actual_value")
    d = pd.to_numeric(fr.get("delta_pct"), errors="coerce")
    if kind == "growth":
        what = f"{metric} growth guided ~{_fmt_val(gv)}% for {fy} ({src})"
        act = f"actual {float(av):+,.1f}% YoY" if pd.notna(pd.to_numeric(av, errors='coerce')) else "actual n/a"
        by = f"by {d:+.1f}pp" if pd.notna(d) else ""
    elif kind == "margin":
        what = f"{metric} guided ~{_fmt_val(gv)}% for {fy} ({src})"
        act = f"actual {_fmt_val(av)}%"
        by = f"by {d:+.1f}pp" if pd.notna(d) else ""
    else:   # level (₹Cr); old rows without kind also land here
        what = f"{metric} guided ~₹{_fmt_val(gv)} cr for {fy} ({src})"
        act = f"actual ₹{_fmt_val(av)} cr"
        by = f"by {d:+.1f}%" if pd.notna(d) else ""
    return (f"{what} → {act} → <b style='color:{color}'>{v}</b> {by}")


def _email_b_html(reporters: pd.DataFrame, res: pd.DataFrame,
                  pead: pd.DataFrame, sym_map: dict) -> str | None:
    if reporters.empty:
        return None
    rows_html = []
    for _, rep in reporters.iterrows():
        slug = str(rep.get("slug", ""))
        isin = str(rep.get("isin", ""))
        comp = str(rep.get("company_name", ""))
        qtr = str(rep.get("latest_q", ""))
        sym = sym_map.get(isin) or slug or isin
        rdate = (rep["result_dt"].strftime("%d-%b")
                 if pd.notna(rep.get("result_dt")) else "")
        comp_rows = res[(res["slug"] == slug) & (res["latest_q"] == qtr)]
        sales_cell = _actual_cell(comp_rows, "sales", "revenue")
        pat_cell = _actual_cell(comp_rows, "net profit")
        f = (pead[pead["isin"].astype(str) == isin]
             if (isin and not pead.empty) else pd.DataFrame())
        if f.empty:
            verdict_cell = ("<i style='color:#999'>no quantified guidance on "
                            "record for this company</i>")
        else:
            bits = []
            for _, fr in f.iterrows():
                v = str(fr.get("verdict", "NA"))
                color = _VERDICT_COLOR.get(v, "#777")
                bits.append(_verdict_sentence(fr, v, color))
            verdict_cell = "<br>".join(bits)
        rows_html.append(
            f"<tr><td><b>{sym}</b></td><td>{comp[:34]}</td><td>{qtr}</td>"
            f"<td>{rdate}</td><td align=right>{sales_cell}</td>"
            f"<td align=right>{pat_cell}</td><td>{verdict_cell}</td></tr>")
    return (f"<p><b>{len(reporters)} company(ies)</b> declared results in the last 24h"
            f" — actuals + guidance check:</p>"
            f"<table border=1 cellpadding=5 cellspacing=0>"
            f"<tr><th>Symbol</th><th>Company</th><th>Qtr</th><th>Result date</th>"
            f"<th>Sales ₹Cr (YoY)</th><th>Net profit ₹Cr (YoY)</th>"
            f"<th>Guidance vs Actual</th></tr>{''.join(rows_html)}</table>"
            f"<p style='font-size:11px;color:#999'><b>How to read the guidance "
            f"column:</b> 'guided' = what management publicly committed to (in the "
            f"named concall / presentation / annual report) for that fiscal year; "
            f"'actual' = the reported number from Screener's financials. "
            f"<b>BEAT</b> = actual exceeded guidance by more than 2 "
            f"(percentage-points for growth/margin guidance, % for ₹Cr levels); "
            f"<b>MISS</b> = fell short by more than 2; <b>INLINE</b> = within ±2. "
            f"Result date = first seen on Screener's latest-results feed. "
            f"Sales = Revenue for banks/financials.</p>")


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

    # ---- Email B: last-24h reporters vs guidance ----
    res = _load_results(drive, index_id)
    reporters = _recent_reporters(res)
    pead = load_parquet(drive, index_id, "pead_flags.parquet", PEAD_COLS)
    sym_map = _isin_symbol_map(drive, index_id) if not reporters.empty else {}
    html_b = _email_b_html(reporters, res, pead, sym_map)

    # ---- Email A: tomorrow's announcers ----
    events = get_results_calendar(args.days_ahead)
    html_a = _html_table(events) if events else None

    if args.dry_run:
        print("=== EMAIL A (results tomorrow) ===")
        print(f"{len(events)} events" if events else "(none)")
        print("\n=== EMAIL B (today's earnings vs guidance) ===")
        print(html_b or "(no reporters today)")
        return

    toggles = load_mail_settings(drive, index_id)
    sent = 0
    if html_b:
        if toggles.get("pead_guidance", True):
            sent += send_email(f"📊 Results last 24h vs guidance — {date.today()}", html_b)
        else:
            print("pead_guidance mail toggled OFF — skipped.")
    if html_a:
        if toggles.get("pead_tomorrow", True):
            sent += send_email(f"📅 Results tomorrow ({len(events)} cos)", html_a)
        else:
            print("pead_tomorrow mail toggled OFF — skipped.")
    print(f"run_pead_daily: {sent} email(s) sent "
          f"(reporters today={0 if reporters is None else len(reporters)}, "
          f"calendar={len(events)}).")


if __name__ == "__main__":
    main()
