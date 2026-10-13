r"""
pf_results_digest.py — daily earnings-season mail: PF companies that have reported
the CURRENT quarter, newest reporter first, with the 6-quarter revenue / profit /
EPS / margin table and YoY + QoQ.

The one hard rule: ONE quarter per season. A holding whose newest quarterly column
is still Q4FY26 is NOT reported under a Q1FY27 heading — it is listed as *awaiting*
until it actually files. Quarter labelling uses the Phase-1/2 results convention
(screener_scraper.current_season_key / build_gallery._qtr_label — the quarter that
just ENDED), NOT extract_concall._current_india_quarter (quarter in progress).

Cumulative, not incremental: every covered company keeps its full table in every
mail. Companies first covered today appear under "NEW TODAY", the rest under
"EARLIER THIS SEASON", both sorted newest-report-first.

Season lifecycle
  - season_quarter = current_season_key(), e.g. "Q1FY27" during Jul-Sep.
  - Mail goes out daily while any PF holding is still awaiting, plus one final mail
    on the day coverage completes. Then it goes quiet until the next season.
  - A season reset is automatic: when season_quarter flips, the ledger has no rows
    for the new key so coverage starts at zero.
  - PF membership is read LIVE every run, so adding a holding mid-season (that has
    already reported) re-opens coverage and the mail resumes.

Inputs (all already on Drive, no schema change):
  pf_tracking|portfolio holdings                -> PF ISINs (load_portfolio_isins)
  company_repo/_index/company_universe.csv      -> isin -> symbol / name
  fundamentals/statements/<SYM>.parquet         -> the quarterly numbers
  company_repo/_index/results.parquet           -> first_seen_at = declaration proxy
  company_repo/_index/processing_queue.parquet  -> results filing announcement_date

Output: company_repo/_index/pf_results_digest.parquet (coverage ledger) + the mail
(toggle 'pf_results').

Usage:
    python scripts/pf_results_digest.py --dry-run    # preview html only, no write/mail
    python scripts/pf_results_digest.py              # ledger + mail (season-gated)
    python scripts/pf_results_digest.py --force      # ignore the season-window gate
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

_SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

import pandas as pd
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(_SCRIPTS_DIR), ".env"))

from _extractor_base import (get_drive, get_or_create_subfolder, load_parquet,
                             save_parquet, log)
from build_gallery import _bulk_parquet, _folder
from daily_brief import load_pf
from mailer import send_email, load_mail_settings, esc
from seasons import in_results_mail_window
import quarterly_table as QT
import filing_results as FR
import results_day_guidance as RDG

LEDGER_NAME = "pf_results_digest.parquet"
# `data_source` added 2026-08-14 (ADDITIVE — old rows read NaN and are treated as
# "screener"): records whether a company was mailed off Screener's statements or off
# its own filing, so the filing-sourced version can be re-mailed once the full table
# lands. Adding a column is safe; readers get NaN for old rows.
LEDGER_COLS = ["season_quarter", "isin", "symbol", "name", "quarter_label",
               "reported_on", "date_source", "data_source",
               "first_covered_at", "last_mailed_at"]
MAIL_KEY = "pf_results"

# only the columns the date cascade needs (load_parquet slices to exactly these)
RESULTS_COLS = ["isin", "latest_q", "first_seen_at", "scraped_at"]
# doc_id / title / pdf_url are for the filing_results join: results_gemini carries no
# filing date of its own, only the timestamp Gemini ran at.
QUEUE_COLS = ["isin", "doc_type", "period", "announcement_date",
              "doc_id", "title", "pdf_url"]

# Gmail clips a message around 102 KB behind a "View entire message" link, which
# would cut the mail off mid-table. Past this the already-covered tail collapses
# to one line each (new reporters always keep their full table).
MAX_HTML_BYTES = 90_000

# date_source preference — a better source upgrades an already-stored date
SRC_RANK = {"screener": 0, "filing": 1, "detected": 2}
SRC_NOTE = {"screener": "", "filing": " (filing)", "detected": " (detected)"}


# ── report-date cascade ──────────────────────────────────────────────────────
def _results_dates(results: pd.DataFrame, season: str) -> dict[str, str]:
    """{isin: YYYY-MM-DD} from results.parquet first_seen_at — the declaration
    proxy strategy_pead.py and flag_growth_surge.py already rely on. Restricted to
    rows whose latest_q maps to THIS season, so a stale row can't date a company."""
    if results is None or results.empty or "latest_q" not in results.columns:
        return {}
    r = results.copy()
    r["_q"] = r["latest_q"].map(lambda v: QT.norm_q(QT.qtr_label(v)))
    r = r[r["_q"] == QT.norm_q(season)]
    if r.empty:
        return {}
    col = "first_seen_at" if "first_seen_at" in r.columns else "scraped_at"
    r["_d"] = r[col].astype(str).str.slice(0, 10)
    r = r[r["_d"].str.match(r"\d{4}-\d{2}-\d{2}", na=False)]
    if r.empty:
        return {}
    return r.groupby(r["isin"].astype(str).str.strip())["_d"].min().to_dict()


def _queue_dates(queue: pd.DataFrame, season: str) -> dict[str, str]:
    """{isin: YYYY-MM-DD} from the global processing queue — results filings for
    this period. Covers the ~150 reporters the 25-item Screener /results/latest/
    window misses each season.

    Delegates to quarterly_table.queue_report_dates. The previous local copy mapped
    announcement_date -> quarter via backfill_company_docs._fy_quarter_label, imported
    under a try/except that leaves it None whenever that import chain fails — which it
    always does on a CI runner, because the chain pulls ingest_company_docs and its PDF
    dependencies and those are not in scripts/requirements.txt. The failure was silent:
    no error, no warning, just every filing-dated reporter missing from the digest.
    Measured against the live queue: 0 dates in CI vs 42 locally.
    """
    return QT.queue_report_dates(queue, season)


def resolve_date(isin, screener_d, queue_d, prev_row, today):
    """(date, source) — best available 'populated on' date. Falls back to the day
    this digest first saw the company flip, so every covered company has a date."""
    if isin in screener_d:
        return screener_d[isin], "screener"
    if isin in queue_d:
        return queue_d[isin], "filing"
    if prev_row is not None and str(prev_row.get("reported_on") or "").strip():
        return str(prev_row["reported_on"])[:10], str(prev_row.get("date_source") or "detected")
    return today, "detected"


# ── coverage ─────────────────────────────────────────────────────────────────
def build_coverage(pf, stmts_map, results, queue, ledger, season, today, filings=None):
    """(covered_rows, awaiting_rows). `covered` = this holding has reported the season
    quarter, by EITHER of two routes, recorded per row as `data_source`:

      screener  the newest quarterly_pl column IS the season quarter (the full table)
      filing    Screener has not republished yet, but the company's own filing has
                been extracted into results_gemini for this quarter (one quarter, no
                history — rendered as its own card, never as a column in a Screener
                table)

    The no-mixing guard is unchanged: a holding whose statements still sit on an older
    quarter never gets a 6-quarter table under this quarter's heading. The filing route
    adds a company the mail would otherwise not mention at all — VMARCIND and OBSCP
    filed on 12 Aug and were invisible in this mail for days because Screener lagged.
    """
    filings = filings or {}
    prev = {}
    if ledger is not None and not ledger.empty:
        for _, r in ledger[ledger["season_quarter"].astype(str)
                           == season].iterrows():
            prev[str(r["isin"]).strip()] = r

    screener_d = _results_dates(results, season)
    queue_d = _queue_dates(queue, season)
    log(f"report dates available: screener={len(screener_d)} queue={len(queue_d)} "
        f"filing={len(filings)}")

    covered, awaiting = [], []
    for isin, sym, name in pf:
        stmts = stmts_map.get(sym)
        lbl = QT.latest_quarter_label(stmts)
        if lbl is None or QT.norm_q(lbl) != QT.norm_q(season):
            fil = filings.get(isin)
            if fil is not None:
                p = prev.get(isin)
                covered.append({
                    "season_quarter": season, "isin": isin, "symbol": sym, "name": name,
                    "quarter_label": fil["quarter"],
                    "reported_on": fil.get("filed_on") or queue_d.get(isin) or today,
                    "date_source": "filing", "data_source": "filing",
                    "first_covered_at": (str(p["first_covered_at"])[:10]
                                         if p is not None
                                         and str(p.get("first_covered_at") or "").strip()
                                         else today),
                    "last_mailed_at": (str(p.get("last_mailed_at") or "")
                                       if p is not None else ""),
                    "upgraded": False,
                    "stmts": None, "filing": fil,
                })
                continue
            awaiting.append({"isin": isin, "symbol": sym, "name": name,
                             "quarter_label": lbl or "",
                             "reason": "no quarterly data" if lbl is None else f"still {lbl}"})
            continue
        p = prev.get(isin)
        rep, src = resolve_date(isin, screener_d, queue_d, p, today)
        # a better source upgrades a date stored earlier as "detected"
        if p is not None and str(p.get("date_source") or "") in SRC_RANK:
            if SRC_RANK[src] > SRC_RANK[str(p["date_source"])]:
                rep, src = str(p["reported_on"])[:10], str(p["date_source"])
        # A holding mailed earlier off its own filing now has the full Screener table.
        # Treat that as new for exactly one mail so the 6-quarter history actually
        # reaches the reader instead of being suppressed by the already-mailed ledger.
        was_filing = (p is not None
                      and str(p.get("data_source") or "screener").strip() == "filing")
        covered.append({
            "season_quarter": season, "isin": isin, "symbol": sym, "name": name,
            "quarter_label": lbl, "reported_on": rep, "date_source": src,
            "data_source": "screener", "upgraded": was_filing,
            "first_covered_at": (today if was_filing else
                                 (str(p["first_covered_at"])[:10]
                                  if p is not None
                                  and str(p.get("first_covered_at") or "").strip()
                                  else today)),
            "last_mailed_at": (str(p.get("last_mailed_at") or "") if p is not None else ""),
            "stmts": stmts, "filing": None,
        })
    covered.sort(key=lambda r: (r["reported_on"], r["symbol"]), reverse=True)
    awaiting.sort(key=lambda r: r["symbol"])
    return covered, awaiting


# ── html ─────────────────────────────────────────────────────────────────────
_WRAP = "font-family:Arial,Helvetica,sans-serif;font-size:13px;color:#222"


def _fmt_day(d: str) -> str:
    try:
        return datetime.strptime(str(d)[:10], "%Y-%m-%d").strftime("%d %b")
    except Exception:
        return str(d)[:10]


def _filing_block(row) -> str:
    """One quarter, straight from the exchange filing, for a holding Screener has not
    republished yet.

    Deliberately NOT a 6-quarter table with the filing pasted into the last column.
    The filing may be consolidated where Screener is standalone, and its "Revenue"
    need not tie to Screener's "Sales" — putting the two bases in one row would hide
    that. One quarter, labelled, on its own.
    """
    f = row.get("filing") or {}
    sym, name = esc(row["symbol"], 24), esc(row["name"], 70)
    when = _fmt_day(row["reported_on"])

    def _line(label, val, dp, suffix, extra=""):
        if val is None:
            return ""
        return (f"<tr><td>{label}</td>"
                f"<td style='text-align:right;font-weight:700'>"
                f"{format(val, f',.{dp}f')}{suffix}</td>"
                f"<td style='color:#666'>{extra}</td></tr>")

    def _yoy(v):
        if v is None:
            return ""
        col = QT.UP if v >= 0 else QT.DOWN
        return f"<span style='color:{col};font-weight:700'>{v:+.1f}% YoY</span>"

    rows = (
        _line("Revenue", f.get("revenue_cr"), 2, " Cr", _yoy(f.get("revenue_yoy_pct")))
        + _line("EBITDA", f.get("ebitda_cr"), 2, " Cr",
                "" if f.get("ebitda_margin_pct") is None
                else f"{f['ebitda_margin_pct']:.2f}% margin")
        + _line("PAT", f.get("pat_cr"), 2, " Cr", _yoy(f.get("pat_yoy_pct")))
        + _line("EPS", f.get("eps"), 2, "", "")
    )
    # Margins survive even when the absolute line did not parse, so show them rather
    # than leaving the company with a single revenue figure.
    if f.get("pat_cr") is None and f.get("pat_margin_pct") is not None:
        rows += ("<tr><td>PAT margin</td>"
                 f"<td style='text-align:right;font-weight:700'>"
                 f"{f['pat_margin_pct']:.2f}%</td>"
                 "<td style='color:#666'>absolute PAT not readable in the filing</td></tr>")
    if not rows:
        return ""

    title = esc(str(f.get("title") or "")[:70], 70)
    return (
        "<div style='margin:0 0 14px 0;border-left:3px solid #b8860b;padding-left:9px'>"
        f"<div style='font-size:14px;font-weight:700;color:#111'>{name} "
        f"<span style='color:#888;font-weight:400'>· {sym}</span></div>"
        f"<div style='color:#b8860b;font-size:11px;margin:0 0 3px'>"
        f"{esc(row['quarter_label'], 12)} · from the exchange filing, {esc(when, 12)}"
        + (f" · {title}" if title else "") + "</div>"
        f"<table cellpadding='4' cellspacing='0' style='{QT._TBL}'>{rows}</table>"
        "<div style='color:#999;font-size:10.5px;margin-top:2px'>"
        "Screener has not republished this company yet, so there is no quarter history "
        "to compare against. The full table follows once it does.</div></div>"
    )


def _company_block(row) -> str:
    if row.get("data_source") == "filing":
        return _filing_block(row)
    table = QT.quarterly_table_html(row["stmts"])
    if not table:
        return ""
    sym, name = esc(row["symbol"], 24), esc(row["name"], 70)
    when = _fmt_day(row["reported_on"]) + SRC_NOTE.get(row["date_source"], "")
    return (
        "<div style='margin:0 0 14px 0'>"
        f"<div style='font-size:14px;font-weight:700;color:#111'>{name} "
        f"<span style='color:#888;font-weight:400'>· {sym}</span></div>"
        f"<div style='color:#888;font-size:11px;margin:0 0 3px'>"
        f"{esc(row['quarter_label'], 12)} · populated {esc(when, 24)}</div>"
        f"{table}</div>"
    )


def _compact_rows(rows) -> str:
    """One line per company — the fallback for already-covered names when the full
    -table mail would exceed Gmail's clip threshold. Same numbers, no history."""
    head = ("<tr style='color:#666'>"
            + "".join(f"<td style='{QT._HD};{a}'>{h}</td>" for h, a in
                      (("Company", "text-align:left"), ("Reported", ""),
                       ("Revenue", ""), ("Rev YoY", ""), ("Net Profit", ""),
                       ("PAT YoY", ""), ("NPM %", "")))
            + "</tr>")
    body = ""
    for r in rows:
        if r.get("data_source") == "filing":
            # No statements to headline off — take the filing's own figures, and mark
            # the row so a filing number is never read as a Screener one.
            f = r.get("filing") or {}
            def _fv(v, dp=0):
                return "—" if v is None else format(v, f",.{dp}f")
            def _fg(v):
                if v is None:
                    return "<td style='color:#bbb'>—</td>"
                col = QT.UP if v >= 0 else QT.DOWN
                return f"<td style='font-weight:700;color:{col}'>{v:+.0f}%</td>"
            body += (f"<tr><td style='text-align:left'><b>{esc(r['symbol'], 24)}</b>"
                     f"<span style='color:#b8860b'> (filing)</span></td>"
                     f"<td>{_fmt_day(r['reported_on'])}</td>"
                     f"<td>{_fv(f.get('revenue_cr'))}</td>{_fg(f.get('revenue_yoy_pct'))}"
                     f"<td>{_fv(f.get('pat_cr'))}</td>{_fg(f.get('pat_yoy_pct'))}"
                     f"<td>{_fv(f.get('pat_margin_pct'), 1)}</td></tr>")
            continue
        h = QT.headline(r["stmts"])
        if not h:
            continue
        def _v(d, dp=0):
            v = d.get("value")
            return "—" if v is None else format(v, f",.{dp}f")
        def _g(d):
            g = d.get("yoy")
            if g is None:
                return "<td style='color:#bbb'>—</td>"
            col = QT.UP if g >= 0 else QT.DOWN
            return f"<td style='font-weight:700;color:{col}'>{g:+.0f}%</td>"
        body += (f"<tr><td style='text-align:left'><b>{esc(r['symbol'], 24)}</b></td>"
                 f"<td>{_fmt_day(r['reported_on'])}</td>"
                 f"<td>{_v(h['revenue'])}</td>{_g(h['revenue'])}"
                 f"<td>{_v(h['pat'])}</td>{_g(h['pat'])}"
                 f"<td>{_v(h['npm'], 1)}</td></tr>")
    if not body:
        return ""
    return (f"<table cellpadding='4' cellspacing='0' style='{QT._TBL}'>{head}{body}</table>")

# ── PF scoreboard (user 2026-10-09) ──────────────────────────────────────────
# The holdings side by side: one row each, ranked by PAT YoY, under a strip of PF
# totals. AGGREGATES USE SCREENER ROWS ONLY — a filing-sourced row has no ladder and
# is not comparable line-for-line (the same no-mixing rule as the tables below); it is
# listed, marked, and kept out of every total. Totals are UNWEIGHTED (user choice):
# like-for-like sums over the companies that have both this quarter and a year ago.

def _yoy_pair(stmts, label: str):
    """(this quarter, same quarter last year) for one quarterly_rows line."""
    periods, data = QT.quarterly_rows(stmts, quarters=6)
    if len(periods) < 5:
        return None, None
    d = data.get(label, {})

    def _v(p):
        v = d.get(p)
        return None if v is None or pd.isna(v) else float(v)
    return _v(periods[-1]), _v(periods[-5])


def scoreboard_rows(covered, guidance: dict | None = None) -> list[dict]:
    """One dict per reported holding, ranked by PAT YoY (None last)."""
    guidance = guidance or {}
    out = []
    for r in covered:
        g = RDG.summary_label(guidance.get(r["isin"]) or {})
        row = {"symbol": r["symbol"], "isin": r["isin"], "reported_on": r["reported_on"],
               "filing": r.get("data_source") == "filing", "guidance": g,
               "rev_yoy": None, "rev_qoq": None, "pat_yoy": None, "pat_qoq": None,
               "opm_dpp": None, "rev": (None, None), "pat": (None, None)}
        if row["filing"]:
            f = r.get("filing") or {}
            row["rev_yoy"], row["pat_yoy"] = f.get("revenue_yoy_pct"), f.get("pat_yoy_pct")
        else:
            h = QT.headline(r.get("stmts"))
            if h:
                row["rev_yoy"], row["rev_qoq"] = h["revenue"]["yoy"], h["revenue"]["qoq"]
                row["pat_yoy"], row["pat_qoq"] = h["pat"]["yoy"], h["pat"]["qoq"]
                row["opm_dpp"] = h["opm"]["yoy"]
            row["rev"] = _yoy_pair(r.get("stmts"), "Revenue")
            row["pat"] = _yoy_pair(r.get("stmts"), "Net Profit")
        out.append(row)
    out.sort(key=lambda x: (x["pat_yoy"] is None, -(x["pat_yoy"] or 0), x["symbol"]))
    return out


def scoreboard_totals(rows: list[dict]) -> dict:
    """PF-level numbers off the Screener rows only (see the block comment)."""
    scr = [r for r in rows if not r["filing"]]

    def _g(pairs):
        cur, prev = sum(c for _, c, _ in pairs), sum(p for _, _, p in pairs)
        return (cur - prev) / abs(prev) * 100 if prev else None

    def _lfl(key):
        """(growth, n, dominant symbol, growth without it). One company can own a sum:
        RAYMOND's year-ago PAT of Rs 5,328 cr (a demerger one-off) turned a book whose
        median PAT grew +78% into "PF PAT -39%" (live dry-run 2026-10-11). When one
        holding is more than half of the gross change, it is named and the total is
        also shown without it."""
        pairs = [(r["symbol"], *r[key]) for r in scr if None not in r[key]]
        g = _g(pairs)
        gross = sum(abs(c - p) for _, c, p in pairs)
        top = max(pairs, key=lambda x: abs(x[1] - x[2]), default=None)
        if top and gross and abs(top[1] - top[2]) > 0.5 * gross and len(pairs) > 1:
            return g, len(pairs), top[0], _g([x for x in pairs if x is not top])
        return g, len(pairs), None, None

    def _med(key):
        v = [r[key] for r in scr if r[key] is not None]
        return (float(pd.Series(v).median()) if v else None), len(v)

    rev_g, rev_n, rev_dom, rev_ex = _lfl("rev")
    pat_g, pat_n, pat_dom, pat_ex = _lfl("pat")
    gl = [r["guidance"] for r in rows if r["guidance"]]
    return {
        "n_screener": len(scr), "n_filing": len(rows) - len(scr),
        "rev_lfl": rev_g, "rev_lfl_n": rev_n, "pat_lfl": pat_g, "pat_lfl_n": pat_n,
        "rev_dom": rev_dom, "rev_ex": rev_ex, "pat_dom": pat_dom, "pat_ex": pat_ex,
        "rev_med": _med("rev_yoy")[0], "pat_med": _med("pat_yoy")[0],
        "rev_up": sum(1 for r in scr if (r["rev_yoy"] or 0) > 0),
        "pat_up": sum(1 for r in scr if (r["pat_yoy"] or 0) > 0),
        "opm_up": sum(1 for r in scr if (r["opm_dpp"] or 0) > 0),
        "opm_n": sum(1 for r in scr if r["opm_dpp"] is not None),
        "rev_n": sum(1 for r in scr if r["rev_yoy"] is not None),
        "pat_n": sum(1 for r in scr if r["pat_yoy"] is not None),
        "g_ahead": gl.count("AHEAD"), "g_on": gl.count("ON_TRACK"),
        "g_mixed": gl.count("MIXED"), "g_behind": gl.count("BEHIND"),
    }


def _pct_td(v, pp: bool = False) -> str:
    if v is None:
        return "<td style='color:#bbb'>—</td>"
    col = QT.UP if v >= 0 else QT.DOWN
    return (f"<td style='font-weight:700;color:{col}'>{v:+.1f}"
            f"{' pp' if pp else '%'}</td>")


def _pct_txt(v) -> str:
    return "—" if v is None else f"{v:+.1f}%"


def _dom_txt(sym, ex) -> str:
    if not sym:
        return ""
    return (f" <span style='color:#b8860b'>— mostly {esc(sym, 20)}; "
            f"without it <b>{_pct_txt(ex)}</b></span>")


def build_scoreboard(covered, pf_n: int, guidance: dict | None = None) -> str:
    rows = scoreboard_rows(covered, guidance)
    if not rows:
        return ""
    t = scoreboard_totals(rows)
    gchips = {"AHEAD": "AHEAD", "ON_TRACK": "ON_TRACK", "BEHIND": "BEHIND",
              "MIXED": "BEHIND"}
    strip = (
        f"<div style='background:#f6f8fa;border:1px solid #e0e6ea;border-radius:6px;"
        f"padding:8px 11px;margin:0 0 8px;font-size:12px;line-height:1.7'>"
        f"<b>{len(rows)} of {pf_n}</b> reported &middot; "
        f"PF revenue <b>{_pct_txt(t['rev_lfl'])}</b> YoY "
        f"({t['rev_lfl_n']} cos){_dom_txt(t['rev_dom'], t['rev_ex'])} &middot; "
        f"PF PAT <b>{_pct_txt(t['pat_lfl'])}</b> YoY "
        f"({t['pat_lfl_n']} cos){_dom_txt(t['pat_dom'], t['pat_ex'])}<br>"
        f"Median revenue {_pct_txt(t['rev_med'])} &middot; median PAT "
        f"{_pct_txt(t['pat_med'])} &middot; revenue up {t['rev_up']}/{t['rev_n']} "
        f"&middot; PAT up {t['pat_up']}/{t['pat_n']} &middot; OPM expanding "
        f"{t['opm_up']}/{t['opm_n']}<br>"
        f"Guidance pace: {t['g_ahead']} ahead &middot; {t['g_on']} on track &middot; "
        f"{t['g_mixed']} mixed &middot; {t['g_behind']} behind</div>")
    head = ("<tr style='color:#666'>"
            + "".join(f"<td style='{QT._HD};{a}'>{h}</td>" for h, a in
                      (("#", ""), ("Company", "text-align:left"), ("PAT YoY", ""),
                       ("PAT QoQ", ""), ("Rev YoY", ""), ("Rev QoQ", ""),
                       ("OPM Δ YoY", ""), ("Guidance", ""), ("Reported", "")))
            + "</tr>")
    body = ""
    for i, r in enumerate(rows, 1):
        g = r["guidance"]
        gcell = (RDG.chip(gchips[g]).replace(gchips[g].replace("_", " "),
                                             g.replace("_", " "))
                 if g else "<span style='color:#bbb'>—</span>")
        body += (f"<tr><td style='color:#999'>{i}</td>"
                 f"<td style='text-align:left'><b>{esc(r['symbol'], 24)}</b>"
                 + ("<span style='color:#b8860b'> (filing)</span>" if r["filing"] else "")
                 + "</td>"
                 + _pct_td(r["pat_yoy"]) + _pct_td(r["pat_qoq"])
                 + _pct_td(r["rev_yoy"]) + _pct_td(r["rev_qoq"])
                 + _pct_td(r["opm_dpp"], pp=True)
                 + f"<td>{gcell}</td><td>{_fmt_day(r['reported_on'])}</td></tr>")
    note = ("<div style='color:#999;font-size:11px;margin:3px 0 0'>Ranked by PAT YoY. "
            "PF totals are unweighted like-for-like sums (this quarter vs the same "
            "quarter last year, companies with both). Totals, medians and counts use "
            "Screener rows only"
            + (f"; the {t['n_filing']} <b>(filing)</b> row"
               f"{'' if t['n_filing'] == 1 else 's'} are listed but not counted"
               if t["n_filing"] else "")
            + ". YoY/QoQ % shown only where the base quarter was positive. Guidance = "
            "pace of open revenue/EBITDA/PAT/margin promises including this quarter."
            "</div>")
    return (f"<h3 style='margin:12px 0 6px;font-size:13px;color:#34495e;"
            f"border-bottom:2px solid #34495e;padding-bottom:2px'>🏁 PF scoreboard</h3>"
            + strip
            + f"<table cellpadding='4' cellspacing='0' style='{QT._TBL}'>{head}{body}"
            f"</table>" + note)


def load_guidance_summaries(drive, index_id, covered, season) -> dict:
    """{isin: RDG.summary} for the covered holdings. Read-only; empty on failure."""
    from build_pead_flags import AR_G_COLS, GUIDANCE_COLS, MGMT_CRED_COLS, PPT_G_COLS
    commits = RDG.prepare_commitments(
        load_parquet(drive, index_id, "guidance_tracker.parquet", GUIDANCE_COLS),
        load_parquet(drive, index_id, "ppt_guidance.parquet", PPT_G_COLS),
        load_parquet(drive, index_id, "ar_guidance.parquet", AR_G_COLS),
        load_parquet(drive, index_id, "mgmt_credibility.parquet", MGMT_CRED_COLS),
        isins={r["isin"] for r in covered})
    out = {}
    for r in covered:
        ser = RDG.series_from_statements(r.get("stmts"), r["isin"])
        if r.get("filing"):
            RDG.add_filing_quarter(ser, r["isin"], r["filing"])
        out[r["isin"]] = RDG.summary(
            RDG.company_progress(commits.get(r["isin"], []), ser, season))
    return out


def build_html(covered, awaiting, season, today, pf_n, in_window,
               compact_old: bool = False, scoreboard: str = "") -> str:
    new = [r for r in covered if str(r["first_covered_at"])[:10] == today]
    old = [r for r in covered if str(r["first_covered_at"])[:10] != today]
    season_lbl = esc(f"{season[:2]} {season[2:]}", 12)   # Q1FY27 -> "Q1 FY27"
    done = len(covered) >= pf_n and pf_n > 0

    n_filing = sum(1 for r in covered if r.get("data_source") == "filing")
    head = (f"<h2 style='margin:0 0 2px'>📊 PF results — {season_lbl}</h2>"
            f"<div style='color:#888;font-size:12px;margin:0 0 8px'>"
            f"{len(covered)} of {pf_n} holdings reported · as of {today}"
            + (f" · {n_filing} from the filing, awaiting Screener" if n_filing else "")
            + ("" if in_window else " · season window closed") + "</div>")
    if done:
        head += ("<div style='background:#e8f5e9;border-left:3px solid #1a7a3a;"
                 "padding:6px 10px;margin:0 0 10px;font-size:12px'>"
                 f"✅ All {pf_n} PF holdings have reported {season_lbl} — "
                 "this is the final mail for this season.</div>")

    def _sec(title, rows, colour, blocks):
        if not blocks:
            return ""
        return (f"<h3 style='margin:14px 0 6px;font-size:13px;color:{colour};"
                f"border-bottom:2px solid {colour};padding-bottom:2px'>"
                f"{title} ({len(rows)})</h3>{blocks}")

    # new reporters ALWAYS keep the full table; only the already-covered tail
    # collapses, and only when the full mail would be clipped by Gmail
    old_blocks = (_compact_rows(old) if compact_old
                  else "".join(_company_block(r) for r in old))
    body = (scoreboard
            + _sec("🆕 NEW TODAY", new, "#1a7a3a",
                 "".join(_company_block(r) for r in new))
            + _sec("EARLIER THIS SEASON", old, "#34495e", old_blocks))
    if compact_old and old:
        body += ("<div style='color:#999;font-size:11px;margin:2px 0 0'>"
                 "Earlier reporters shown as one line each — the full-table mail "
                 "would exceed Gmail's size limit and be clipped.</div>")

    foot = ""
    if awaiting:
        names = ", ".join(esc(r["symbol"] or r["name"], 24) for r in awaiting[:40])
        more = f" +{len(awaiting) - 40} more" if len(awaiting) > 40 else ""
        foot = ("<hr style='border:0;border-top:1px solid #ddd;margin:12px 0 6px'>"
                f"<div style='color:#666;font-size:12px'><b>Awaiting {season_lbl}</b> "
                f"({len(awaiting)} of {pf_n}): {names}{more}</div>")
        if not in_window:
            foot += ("<div style='color:#999;font-size:11px;margin-top:4px'>"
                     f"Season closed — {len(awaiting)} never reported.</div>")
    foot += ("<div style='color:#aaa;font-size:11px;margin-top:8px'>"
             "Revenue / Net Profit in ₹ Cr, EPS in ₹, margins in %. "
             "YoY = same quarter last year; QoQ = previous quarter; "
             "margin deltas in percentage points. Source: Screener quarterly P&amp;L."
             + (" Blocks marked <b>from the exchange filing</b> were read from the "
                "company's own results filing because Screener has not republished it "
                "yet — one quarter, no history, and not comparable line-for-line with "
                "the Screener tables above." if n_filing else "")
             + "</div>")
    return f"<div style='{_WRAP}'>{head}{body}{foot}</div>"


# ── main ─────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true",
                    help="preview html only — no Drive write, no mail")
    ap.add_argument("--force", action="store_true",
                    help="ignore the season-window gate and the already-complete stop")
    args = ap.parse_args()

    drive = get_drive()
    root_id = os.environ["GDRIVE_FOLDER_ID"]
    repo_id = get_or_create_subfolder(drive, root_id, "company_repo")
    index_id = get_or_create_subfolder(drive, repo_id, "_index")

    season = QT.season_quarter()
    today = datetime.now().strftime("%Y-%m-%d")
    in_window = in_results_mail_window(datetime.now())
    log(f"season quarter={season} · peak-season={in_window} · today={today}")

    if not in_window and not args.force and not args.dry_run:
        log("outside the results-season window — no mail. (--force to override)")
        return

    pf = load_pf(drive, root_id, index_id)
    if not pf:
        log("no PF holdings resolved — nothing to do.")
        return
    syms = sorted({s for _, s, _ in pf if s})
    log(f"PF holdings: {len(pf)} ({len(syms)} symbols)")

    stmts_map = _bulk_parquet(drive, _folder(drive, "fundamentals/statements"), syms)
    log(f"statements downloaded: {sum(1 for v in stmts_map.values() if v is not None and not v.empty)}/{len(syms)}")

    # load_parquet returns df[cols] — the column list is mandatory, never []
    results = load_parquet(drive, index_id, "results.parquet", RESULTS_COLS)
    queue = load_parquet(drive, index_id, "processing_queue.parquet", QUEUE_COLS)
    ledger = load_parquet(drive, index_id, LEDGER_NAME, LEDGER_COLS)

    # The filing fallback: companies whose own results filing has been extracted but
    # whose Screener statements still sit on the previous quarter.
    # Reference revenue per holding = its most recent Screener quarter, used to catch
    # filings stated in lakh rather than crore (PRECWIRE filed 177,048 against a prior
    # quarter of 1,763 Cr — a unit mismatch, not a hundredfold jump).
    refs = {}
    for _isin, _sym, _n in pf:
        _st = stmts_map.get(_sym)
        if _st is not None and not _st.empty:
            _r = FR.reference_revenue(_st)
            if _r:
                refs[str(_isin).strip()] = _r
    filings = FR.load_filing_results(drive, index_id, season, queue=queue, log=log,
                                     refs=refs)

    covered, awaiting = build_coverage(pf, stmts_map, results, queue, ledger, season,
                                       today, filings=filings)
    new_today = [r for r in covered if str(r["first_covered_at"])[:10] == today]
    never_mailed = [r for r in covered if not str(r["last_mailed_at"] or "").strip()]
    upgraded = [r for r in covered if r.get("upgraded")]
    n_filing = sum(1 for r in covered if r.get("data_source") == "filing")
    log(f"covered={len(covered)} (filing-sourced={n_filing}) awaiting={len(awaiting)} "
        f"new_today={len(new_today)} never_mailed={len(never_mailed)} "
        f"upgraded={len(upgraded)}")
    if upgraded:
        log("  Screener caught up for: "
            + ", ".join(r["symbol"] for r in upgraded))

    if not covered:
        log(f"no PF holding has reported {season} yet — no mail.")
        return

    # The scoreboard is extra: a failure drops it, never the mail.
    try:
        guidance = load_guidance_summaries(drive, index_id, covered, season)
        log(f"guidance pace measured for {sum(1 for v in guidance.values() if v.get('measured'))}"
            f"/{len(covered)} covered holdings")
    except Exception as e:
        guidance = {}
        log(f"WARNING: guidance pace unavailable ({str(e)[:70]})")
    try:
        board = build_scoreboard(covered, len(pf), guidance)
    except Exception as e:
        board = ""
        log(f"WARNING: scoreboard skipped ({str(e)[:70]})")

    html = build_html(covered, awaiting, season, today, len(pf), in_window,
                      scoreboard=board)
    if len(html.encode("utf-8")) > MAX_HTML_BYTES:
        html = build_html(covered, awaiting, season, today, len(pf), in_window,
                          compact_old=True, scoreboard=board)
        log(f"full-table mail exceeded {MAX_HTML_BYTES // 1024} KB — "
            f"earlier reporters collapsed to one line each "
            f"({len(html.encode('utf-8')) / 1024:.0f} KB)")

    if args.dry_run:
        show = pd.DataFrame([{k: r[k] for k in
                              ("symbol", "quarter_label", "reported_on", "date_source",
                               "data_source", "first_covered_at")} for r in covered])
        print(show.to_string(index=False))
        if awaiting:
            print("\nAWAITING:")
            print(pd.DataFrame(awaiting)[["symbol", "quarter_label", "reason"]]
                  .to_string(index=False))
        prev = Path(__file__).resolve().parent.parent / "pf_results_digest_preview.html"
        prev.write_text(html, encoding="utf-8")
        print(f"\nDRY RUN — preview saved to {prev.name} "
              f"({len(html.encode('utf-8')) / 1024:.0f} KB); no Drive write, no mail.")
        return

    # stop rule: mail daily while incomplete; one final mail when coverage completes.
    # An upgrade (filing -> Screener) re-opens the mail even when everything is
    # complete and already mailed, or the fuller table would never be delivered.
    complete = len(covered) >= len(pf)
    if complete and not never_mailed and not upgraded and not args.force:
        log("season complete and already mailed — no mail. (--force to override)")
        return

    if not load_mail_settings(drive, index_id).get(MAIL_KEY, True):
        log(f"{MAIL_KEY} mail toggled OFF — skipped (ledger unchanged).")
        return

    season_lbl = f"{season[:2]} {season[2:]}"
    subject = (f"📊 PF results {season_lbl} — {len(covered)}/{len(pf)} reported"
               + (f" · {len(new_today)} new" if new_today else "")
               + (f" · {n_filing} from filings" if n_filing else "")
               + (" · complete" if complete else ""))
    sent = send_email(subject, html)
    # ascii-only in log lines — local console is cp1252, emoji crash print()
    log(f"Email {'sent' if sent else 'FAILED'}: "
        f"{subject.encode('ascii', 'ignore').decode().strip()}")

    # The ledger advances ONLY on a delivered mail. Stamping last_mailed_at after a
    # failed send would mark these companies as already reported to the user and
    # demote them out of "NEW TODAY" on the next run — they must stay new until a
    # mail actually goes out.
    if not sent:
        log("mail not sent — ledger NOT advanced; these stay new for the next run.")
        return

    out = ledger[ledger["season_quarter"].astype(str) != season] if not ledger.empty \
        else pd.DataFrame(columns=LEDGER_COLS)
    rows = [{k: r[k] for k in LEDGER_COLS if k != "last_mailed_at"} for r in covered]
    for r in rows:
        r["last_mailed_at"] = datetime.now().isoformat(timespec="seconds")
    out = pd.concat([out, pd.DataFrame(rows, columns=LEDGER_COLS)], ignore_index=True)
    save_parquet(drive, index_id, LEDGER_NAME, out)
    log(f"wrote _index/{LEDGER_NAME} ({len(out)} rows, {len(rows)} this season)")


def _self_test() -> int:
    """Offline: the scoreboard's ranking, totals and no-mixing rule."""
    fails = 0

    def check(name, ok):
        nonlocal fails
        print(("  ok   " if ok else "  FAIL ") + name)
        fails += 0 if ok else 1

    per = ["Jun 2025", "Sep 2025", "Dec 2025", "Mar 2026", "Jun 2026", "Sep 2026"]

    def st(rev, pat, opm):
        rows = []
        for line, vals in (("Sales +", rev), ("Net Profit +", pat), ("OPM %", opm)):
            rows += [{"statement": "quarterly_pl", "line_item": line, "period": p,
                      "value": v} for p, v in zip(per, vals)]
        return pd.DataFrame(rows)

    cov = [
        {"symbol": "AAA", "isin": "I1", "reported_on": "2026-10-08",
         "data_source": "screener",
         "stmts": st([0, 100, 0, 0, 0, 120], [0, 10, 0, 0, 0, 15], [0, 10, 0, 0, 0, 12])},
        {"symbol": "BBB", "isin": "I2", "reported_on": "2026-10-09",
         "data_source": "screener",
         "stmts": st([0, 300, 0, 0, 0, 300], [0, 30, 0, 0, 0, 27], [0, 20, 0, 0, 0, 19])},
        {"symbol": "FFF", "isin": "I3", "reported_on": "2026-10-09",
         "data_source": "filing", "stmts": None,
         "filing": {"revenue_yoy_pct": 90.0, "pat_yoy_pct": 400.0}},
    ]
    rows = scoreboard_rows(cov, {"I2": {"measured": 1, "ahead": 0, "on_track": 0,
                                        "behind": 1}})
    check("ranked by PAT YoY desc", [r["symbol"] for r in rows] == ["FFF", "AAA", "BBB"])
    t = scoreboard_totals(rows)
    check("filing row kept out of totals", t["n_filing"] == 1 and t["n_screener"] == 2)
    check("like-for-like revenue = (420-400)/400",
          t["rev_lfl"] is not None and abs(t["rev_lfl"] - 5.0) < 1e-9)
    check("like-for-like PAT = (42-40)/40",
          t["pat_lfl"] is not None and abs(t["pat_lfl"] - 5.0) < 1e-9)
    check("counts", t["rev_up"] == 1 and t["pat_up"] == 1 and t["opm_up"] == 1)
    check("guidance label carried", rows[2]["guidance"] == "BEHIND" and t["g_behind"] == 1)
    check("two holdings: the larger mover (+5 of 8 gross) is named",
          t["pat_dom"] == "AAA")
    # RAYMOND shape: one holding's year-ago one-off swamps the sum
    dom = [{"symbol": "BIG", "filing": False, "rev": (100, 100), "pat": (31, 5328),
            "rev_yoy": 0, "pat_yoy": None, "opm_dpp": None, "guidance": ""},
           {"symbol": "A", "filing": False, "rev": (120, 100), "pat": (15, 10),
            "rev_yoy": 20, "pat_yoy": 50, "opm_dpp": None, "guidance": ""},
           {"symbol": "B", "filing": False, "rev": (220, 200), "pat": (30, 20),
            "rev_yoy": 10, "pat_yoy": 50, "opm_dpp": None, "guidance": ""}]
    td = scoreboard_totals(dom)
    check("dominant holding named", td["pat_dom"] == "BIG")
    check("total without it = (45-30)/30",
          td["pat_ex"] is not None and abs(td["pat_ex"] - 50.0) < 1e-9)
    check("strip says so", "mostly BIG" in _dom_txt(td["pat_dom"], td["pat_ex"]))
    html = build_scoreboard(cov, 5, {})
    check("scoreboard renders with (filing) marker",
          "PF scoreboard" in html and "(filing)" in html and "3 of 5" in html)
    check("scoreboard is compact", len(html.encode("utf-8")) < 8000)
    check("no covered -> no scoreboard", build_scoreboard([], 5) == "")
    for c in cov:
        c.update(name=c["symbol"], quarter_label="Q2 FY27", date_source="screener",
                 first_covered_at="2026-10-09")
    full = build_html(cov, [], "Q2FY27", "2026-10-09", 5, True, scoreboard=html)
    check("scoreboard leads the mail", full.index("PF scoreboard") < full.index("NEW TODAY"))
    print(f"\n{'ALL PASS' if not fails else f'{fails} FAILED'}")
    return 1 if fails else 0


if __name__ == "__main__":
    if "--self-test" in sys.argv:
        raise SystemExit(_self_test())
    main()
