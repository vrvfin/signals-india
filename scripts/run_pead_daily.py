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
import re
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
                             download_bytes, load_parquet, log, load_portfolio_isins)
from build_pead_flags import PEAD_COLS
from earnings_calendar import get_results_calendar, _html_table
from guidance_digest_email import (GUIDANCE_COLS, HORIZON_LABEL, METRIC_LABEL,
                                   _to_pct)
from mailer import send_email, load_mail_settings, esc as _esc
# season_quarter, NOT screener_scraper.current_season_key — that module is gitignored
# and importing it dies in CI with ModuleNotFoundError (see commit 46f33b7).
from quarterly_table import season_quarter

_VERDICT_COLOR = {"BEAT": "#27ae60", "MISS": "#e74c3c", "INLINE": "#777", "NA": "#aaa"}
MAX_GUIDANCE_LINES = 4      # per company; the cell used to print ALL of them (max 34)
_FY_RE = re.compile(r"FY\s*'?(\d{2,4})", re.I)
_QTR_RE = re.compile(r"Q([1-4])\s*FY[\s'\-]*?(\d{2,4})", re.I)


def _norm_fy(s) -> int | None:
    """'FY2022' / 'FY22' / "FY'22" -> 22. Screener/Gemini emit both 2- and 4-digit
    forms for the SAME year, which previously duplicated every such guidance row."""
    m = _FY_RE.search(str(s or ""))
    return int(m.group(1)) % 100 if m else None


def _fy_of_quarter(q) -> int | None:
    """'Jun 2026' -> 27 (Indian FY: Apr-Mar). The FY the reported quarter sits in."""
    d = pd.to_datetime(q, format="%b %Y", errors="coerce")
    if pd.isna(d):
        return None
    return (d.year + 1 if d.month >= 4 else d.year) % 100


def _guid_qkey(q) -> int:
    """'Q3 FY25' -> sortable int, for ranking guidance by how recently it was said."""
    m = _QTR_RE.search(str(q or ""))
    return (int(m.group(2)) % 100) * 4 + int(m.group(1)) if m else -1


def _fmt_date(s) -> str:
    try:
        return pd.to_datetime(s).strftime("%d %b %Y")
    except Exception:
        return ""


def _relevant_guidance(f: pd.DataFrame, reported_fy: int | None) -> tuple[pd.DataFrame, int]:
    """Trim a company's guidance-vs-actual rows to what still matters.

    Was: print EVERY row (mean 6.7/co, max 34, horizons back to FY19). Now:
    drop horizons older than the just-reported FY, de-duplicate the 2-/4-digit
    FY spellings, rank by the most RECENT concall, cap the list.
    Returns (rows_to_show, n_hidden)."""
    if f.empty:
        return f, 0
    d = f.copy()
    d["_fy"] = d["quarter"].map(_norm_fy)
    d["_said"] = d["guid_quarter"].map(_guid_qkey) if "guid_quarter" in d.columns else -1
    if reported_fy is not None:
        keep = d[d["_fy"].notna() & (d["_fy"] >= reported_fy)]
        if keep.empty and d["_fy"].notna().any():
            keep = d[d["_fy"] == d["_fy"].max()]   # nothing current -> newest available
        d = keep if not keep.empty else d
    d = d.drop_duplicates(subset=["_fy", "metric", "guided_value"])
    d = d.sort_values(["_said", "_fy"], ascending=False)
    return d.head(MAX_GUIDANCE_LINES), max(0, len(d) - MAX_GUIDANCE_LINES)


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
    """'12,345 (+8%)' for the first metric row containing any term (YoY colored).

    Screener leaves yoy_pct blank on ~16% of rows (nearly always Net profit), so
    fall back to computing it from the latest/year-ago LEVELS — otherwise the
    headline PAT column showed a bare number with no growth (user 2026-07-18)."""
    for t in terms:
        hit = comp_rows[comp_rows["metric"].astype(str).str.lower()
                        .str.contains(t, na=False)]
        if not hit.empty:
            r0 = hit.iloc[0]
            cell = _fmt_val(r0.get("latest_val"))
            yoy = pd.to_numeric(r0.get("yoy_pct"), errors="coerce")
            if pd.isna(yoy):
                lv = pd.to_numeric(r0.get("latest_val"), errors="coerce")
                yv = pd.to_numeric(r0.get("yearago_val"), errors="coerce")
                if pd.notna(lv) and pd.notna(yv) and yv > 0:
                    yoy = (lv - yv) / yv * 100.0
            if pd.notna(yoy):
                color = "#27ae60" if yoy >= 0 else "#e74c3c"
                cell += f' <span style="color:{color}">({yoy:+.0f}%)</span>'
            return cell
    return "-"


_SRC_LABEL = {"concall": "concall", "presentation": "investor presentation",
              "annual_report": "annual report"}


def _verdict_sentence(fr, v: str, color: str, doc_by: dict | None = None) -> str:
    """Self-explanatory guidance-vs-actual line (user 2026-07-12), e.g.
    'Revenue growth guided ~15% for FY26 (concall) → actual +21% YoY →
    BEAT by +6.2pp'. kind decides the units: growth/margin compare in
    percentage-points; level compares ₹Cr and the delta is %."""
    metric = str(fr.get("metric", "") or "").title()
    metric = {"Pat": "PAT", "Ebitda": "EBITDA", "Eps": "EPS",
              "Opm": "OPM"}.get(metric, metric)
    fy = str(fr.get("quarter", "") or "")
    src = _SRC_LABEL.get(str(fr.get("guidance_source") or ""), "concall")
    # WHERE/WHEN it was said: source concall quarter + that filing's date
    said_q = str(fr.get("guid_quarter") or "").strip()
    doc = (doc_by or {}).get(str(fr.get("source_doc_id") or ""), {})
    said_dt = _fmt_date(doc.get("date"))
    prov = ", ".join(x for x in (said_q, said_dt) if x)
    src = f"{src} {prov}" if prov else src
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


def _pf_symbol_map(drive, root_id, index_id) -> dict:
    """NSE symbol -> (isin, name) for PF holdings only. The calendar keys on symbol;
    the portfolio keys on ISIN, so one of them has to be translated."""
    isins = load_portfolio_isins(drive, root_id)
    if not isins:
        return {}
    fid = find_file(drive, index_id, "company_universe.csv")
    if not fid:
        return {}
    try:
        uni = pd.read_csv(io.BytesIO(download_bytes(drive, fid))).fillna("")
    except Exception:
        return {}
    out = {}
    for _, r in uni.iterrows():
        isin = str(r.get("isin", "")).strip()
        if isin not in isins:
            continue
        sym = str(r.get("nse_symbol") or r.get("bse_symbol") or "").strip().upper()
        if sym and sym.lower() != "nan":
            out[sym] = (isin, str(r.get("name", "")).strip() or sym)
    return out


def _open_guidance(g: pd.DataFrame, isin: str, cur_fy: int | None
                   ) -> tuple[pd.DataFrame, int]:
    """A company's guidance that is still OPEN — the promises tomorrow's print can be
    read against. Mirrors _relevant_guidance (the after-the-fact side): drop horizons
    that have already passed, de-duplicate the 2-/4-digit FY spellings, rank by the most
    recent concall, cap the list. Returns (rows_to_show, n_hidden)."""
    if g is None or g.empty:
        return pd.DataFrame(), 0
    d = g[g["isin"].astype(str) == isin].copy()
    if d.empty:
        return d, 0
    d["_fy"] = d["horizon_fy"].map(_norm_fy)
    d["_said"] = d["quarter"].map(_guid_qkey)
    if cur_fy is not None:
        keep = d[d["_fy"].isna() | (d["_fy"] >= cur_fy)]
        # A horizon that is purely qualitative ("NEAR_TERM") has no FY to compare, so it
        # stays; one that names a year already past does not.
        d = keep if not keep.empty else d
    d = d.drop_duplicates(subset=["_fy", "metric", "value"])
    d = d.sort_values(["_said", "_fy"], ascending=False)
    return d.head(MAX_GUIDANCE_LINES), max(0, len(d) - MAX_GUIDANCE_LINES)


def _guidance_cell(rows: pd.DataFrame, hidden: int) -> str:
    """The promise lines for one company, most recently said first."""
    if rows.empty:
        return (f"<div style='color:#888;font-size:12.5px'>No guidance on record &mdash; "
                f"nothing to hold the print against.</div>")
    out = []
    for _, r in rows.iterrows():
        metric = str(r.get("metric") or "")
        horizon = str(r.get("horizon_fy") or "")
        pct = _to_pct(r.get("value"), r.get("cagr_pct"), metric, horizon)
        # _to_pct is the plausibility-capped path: re-parsing `value` here is what once
        # turned a capacity target of "178,000" into 178000%.
        raw = str(r.get("value") or "").strip()
        shown = f"<b>{pct:+.1f}%</b>" if pct is not None else f"<b>{_esc(raw, 60)}</b>"
        unit = str(r.get("unit") or "").strip()
        # The unit is often already spelled inside the value ('150,000 units/annum'
        # + unit 'UNITS', '1.0%' + unit '%'), so append it only when it adds something.
        if (pct is None and unit and unit.lower() != "nan"
                and unit.lower().rstrip("s") not in raw.lower()):
            shown += f" {_esc(unit, 12)}"
        label = METRIC_LABEL.get(metric, metric.title() or "&mdash;")
        hz = HORIZON_LABEL.get(horizon, horizon)
        said = str(r.get("quarter") or "")
        out.append(f"<li style='margin:2px 0'>{_esc(label, 22)} {shown}"
                   f"<span style='color:#888'> &middot; {_esc(hz, 16)}"
                   f"{f' &middot; said in {_esc(said, 12)}' if said else ''}</span></li>")
    tail = (f"<div style='color:#888;font-size:11.5px'>+{hidden} older line(s) not "
            f"shown.</div>" if hidden else "")
    return f"<ul style='margin:2px 0 0 16px;padding:0;font-size:13px'>{''.join(out)}</ul>{tail}"


def _pf_preview_html(events: list[dict], pf_map: dict, g: pd.DataFrame) -> tuple[str, int]:
    """The PF block that leads the 'results tomorrow' mail: for each holding on the
    calendar, what management has already promised. Returns (html, n_pf)."""
    hits = [e for e in events if str(e.get("symbol", "")).strip().upper() in pf_map]
    if not hits:
        return "", 0
    cur_fy = _norm_fy(season_quarter())
    blocks = []
    for e in sorted(hits, key=lambda x: (x.get("date", ""), x.get("symbol", ""))):
        sym = str(e["symbol"]).strip().upper()
        isin, name = pf_map[sym]
        rows, hidden = _open_guidance(g, isin, cur_fy)
        blocks.append(
            f"<div style='border:1px solid #e0e6ea;border-radius:6px;padding:10px 13px;"
            f"margin:0 0 9px;background:#fbfcfd'>"
            f"<div style='font-size:14px'><b>{_esc(sym, 18)}</b> "
            f"<span style='color:#666'>{_esc(name, 60)}</span></div>"
            f"<div style='color:#888;font-size:11.5px;margin:1px 0 6px'>"
            f"board meeting {_esc(e.get('date'), 12)} &middot; {_esc(e.get('source'), 8)}"
            f"</div>{_guidance_cell(rows, hidden)}</div>")
    return (f"<h3 style='margin:0 0 3px;font-size:15px'>&#128188; Your portfolio: "
            f"{len(hits)} holding{'' if len(hits) == 1 else 's'} reporting</h3>"
            f"<div style='color:#888;font-size:12px;margin:0 0 10px'>What management has "
            f"already promised &mdash; the open guidance tomorrow's numbers can be read "
            f"against. Guidance only; no forecast is implied.</div>"
            f"{''.join(blocks)}"
            f"<div style='border-top:1px solid #ddd;margin:14px 0 10px'></div>"), len(hits)


def _email_b_html(reporters: pd.DataFrame, res: pd.DataFrame,
                  pead: pd.DataFrame, sym_map: dict,
                  doc_by: dict | None = None) -> str | None:
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
            # only guidance for the reported FY onward, deduped, newest concall
            # first, capped — was printing every historical row (up to 34).
            f, n_hidden = _relevant_guidance(f, _fy_of_quarter(qtr))
            bits = []
            for _, fr in f.iterrows():
                v = str(fr.get("verdict", "NA"))
                color = _VERDICT_COLOR.get(v, "#777")
                bits.append(_verdict_sentence(fr, v, color, doc_by))
            if n_hidden:
                bits.append(f"<span style='color:#999;font-size:11px'>"
                            f"…{n_hidden} older guidance line(s) hidden</span>")
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
            f"named source, with the concall quarter + filing date it was said in) "
            f"for that fiscal year; "
            f"'actual' = the reported number from Screener's financials. "
            f"<b>BEAT</b> = actual exceeded guidance by more than 2 "
            f"(percentage-points for growth/margin guidance, % for ₹Cr levels); "
            f"<b>MISS</b> = fell short by more than 2; <b>INLINE</b> = within ±2. "
            f"Only guidance for the reported fiscal year onward is shown, newest "
            f"concall first (max {MAX_GUIDANCE_LINES}); older horizons are hidden. "
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
    # source document behind each guidance line (date + filing), via source_doc_id
    doc_by = {}
    if not reporters.empty:
        from _extractor_base import QUEUE_COLS
        q = load_parquet(drive, index_id, "processing_queue.parquet", QUEUE_COLS)
        if not q.empty:
            doc_by = {str(d): {"date": a} for d, a in
                      zip(q["doc_id"], q["announcement_date"])}
    html_b = _email_b_html(reporters, res, pead, sym_map, doc_by)

    # ---- Email A: tomorrow's announcers, PF holdings first ----
    events = get_results_calendar(args.days_ahead)
    # The PF block is the reason to open this mail, but it must never be the reason the
    # mail fails to arrive — the universe-wide calendar below stands on its own.
    pf_html, n_pf = "", 0
    if events:
        try:
            pf_map = _pf_symbol_map(drive, root_id, index_id)
            gtrack = load_parquet(drive, index_id, "guidance_tracker.parquet",
                                  GUIDANCE_COLS)
            pf_html, n_pf = _pf_preview_html(events, pf_map, gtrack)
            log(f"  PF preview: {n_pf} of {len(pf_map)} holding(s) on the calendar")
        except Exception as exc:                       # noqa: BLE001 — mail must still go
            log(f"  PF preview FAILED ({type(exc).__name__}: {exc}) — "
                f"sending the calendar without it")
    html_a = (pf_html + _html_table(events)) if events else None

    if args.dry_run:
        print("=== EMAIL A (results tomorrow) ===")
        print(f"{len(events)} events; {n_pf} PF holding(s)" if events else "(none)")
        print(pf_html or "(no PF holdings on the calendar)")
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
            # PF count leads the subject only when there is one — an unqualified
            # "0 PF holdings" every off-season day is noise, not information.
            subj = (f"📅 Results tomorrow — {n_pf} PF holding"
                    f"{'' if n_pf == 1 else 's'} ({len(events)} cos)" if n_pf
                    else f"📅 Results tomorrow ({len(events)} cos)")
            sent += send_email(subj, html_a)
        else:
            print("pead_tomorrow mail toggled OFF — skipped.")
    print(f"run_pead_daily: {sent} email(s) sent "
          f"(reporters today={0 if reporters is None else len(reporters)}, "
          f"calendar={len(events)}, pf={n_pf}).")


if __name__ == "__main__":
    main()
