r"""
results_day_guidance.py — where THIS quarter's numbers leave management's open guidance.

WHY THIS EXISTS (user 2026-10-09, PF results-mail gap #2)
    On a results day the per-company teardown's Block D read guidance_vs_actual, which
    build_pead_flags fills from ANNUAL rows only — so on a Q1/Q2/Q3 results day it can
    say nothing about the quarter just reported. The mid-horizon measurement already
    exists (guidance_progress.compute_progress, driven by build_guidance_progress.build),
    but it runs weekly off financials_3stmt, so on the day it matters it is up to a week
    stale and misses the new quarter entirely.

    This module runs that SAME measurement for one company at mail time, off the
    company's own statements parquet — and, while Screener lags, with the season quarter
    filled from the company's own exchange filing.

WHAT IT REUSES (no copies of the maths)
    build_guidance_progress._commitments_from_tracker / _commitments_from_credibility
    build_guidance_progress.build      -> pace, status, segment-target and units guards
    guidance_progress                   -> quarter arithmetic, classify_kind

SCOPE
    P&L commitments only (revenue / EBITDA / PAT / margin). Order-book and capex targets
    need the announcement ledger and annual Net Block — those stay in the weekly
    guidance-progress mail. Nothing here writes to Drive.

Usage:
    python scripts/results_day_guidance.py --self-test
"""
from __future__ import annotations

import html as _html
import os
import re
import sys
from datetime import date, datetime

_SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

import pandas as pd

import build_guidance_progress as BGP  # noqa: E402
import guidance_progress as GP  # noqa: E402
import quarterly_table as QT  # noqa: E402

# financials_3stmt line name (what GP.METRIC_ACTUAL asks for) -> the aliases the
# per-symbol statements parquet may carry. Same aliases as quarterly_table.ROWS, plus
# the bank spelling of operating profit.
STMT_LINES = {
    "Sales":            ["Sales", "Revenue", "Sales +", "Revenue +"],
    "Operating Profit": ["Operating Profit", "Financing Profit"],
    "Net Profit":       ["Net Profit", "Net Profit +"],
    "OPM %":            ["OPM %", "Financing Margin %"],
}
# filing_results field -> statements line
FILING_LINES = {"revenue_cr": "Sales", "ebitda_cr": "Operating Profit",
                "pat_cr": "Net Profit", "ebitda_margin_pct": "OPM %"}
PL_KINDS = ("level", "growth", "margin")

STATUS_ORDER = ["AT_RISK", "BEHIND", "ON_TRACK", "AHEAD", "ACHIEVED"]
STATUS_CHIP = {
    "ACHIEVED": ("#1a7a3a", "#fff"),
    "AHEAD":    ("#d4edda", "#155724"),
    "ON_TRACK": ("#e2f0d9", "#2d5016"),
    "BEHIND":   ("#fff3cd", "#856404"),
    "AT_RISK":  ("#f8d7da", "#721c24"),
}
MAX_ROWS = 6


def _esc(s, n=200) -> str:
    return _html.escape(str(s if s is not None else "")[:n])


# --------------------------------------------------------------------------- #
#  inputs                                                                     #
# --------------------------------------------------------------------------- #

def _by_isin(df: pd.DataFrame, isins) -> pd.DataFrame:
    if df is None or getattr(df, "empty", True) or "isin" not in df.columns:
        return pd.DataFrame()
    if isins is None:
        return df
    return df[df["isin"].astype(str).isin(isins)]


def prepare_commitments(tracker, ppt, ar, cred, isins=None) -> dict[str, list[dict]]:
    """{isin: [commitment, ...]} for P&L promises, filtered to `isins` BEFORE the
    (per-row) value parse, so a 36k-row tracker costs a PF-sized parse."""
    isins = {str(i).strip() for i in isins} if isins is not None else None
    tracker, ppt = _by_isin(tracker, isins), _by_isin(ppt, isins)
    ar, cred = _by_isin(ar, isins), _by_isin(cred, isins)
    # presentation / AR use `horizon` / `fy_year`; normalise onto the tracker's shape
    # exactly as build_guidance_progress.main does
    if not ppt.empty and "horizon_fy" not in ppt.columns:
        ppt = ppt.rename(columns={"horizon": "horizon_fy"})
    if not ar.empty and "quarter" not in ar.columns:
        ar = ar.rename(columns={"fy_year": "quarter"})
    commits = []
    for df, src in ((tracker, "concall"), (ppt, "presentation"), (ar, "annual_report")):
        if not df.empty:
            commits += BGP._commitments_from_tracker(df, src)
    if not cred.empty:
        commits += BGP._commitments_from_credibility(cred)
    out: dict[str, list[dict]] = {}
    for c in commits:
        if GP.classify_kind(c["metric"], c["value_type"]) in PL_KINDS:
            out.setdefault(c["isin"], []).append(c)
    return out


def series_from_statements(stmts, isin: str) -> dict:
    """{(isin, line): {quarter_idx: value}} from the quarterly_pl slice — the same
    shape build_guidance_progress._financial_series makes from financials_3stmt."""
    out: dict = {}
    if stmts is None or getattr(stmts, "empty", True) or \
            not {"statement", "line_item", "period", "value"} <= set(stmts.columns):
        return out
    q = stmts[stmts["statement"].astype(str) == "quarterly_pl"]
    if q.empty:
        return out
    li = q["line_item"].astype(str).str.strip()
    for line, aliases in STMT_LINES.items():
        for a in aliases:
            sub = q[li == a]
            if sub.empty:
                continue
            ser = {}
            for p, v in zip(sub["period"].astype(str), sub["value"]):
                qi = GP.parse_q(QT.qtr_label(p))
                fv = BGP._f(v)
                if qi and fv is not None:
                    ser[GP.q_idx(*qi)] = fv
            if ser:
                out[(isin, line)] = ser
            break
    return out


def add_filing_quarter(series: dict, isin: str, filing: dict | None) -> bool:
    """Put the season quarter from the company's own filing into `series`, for the
    days Screener has not republished it. Never overwrites a quarter Screener has.

    Refuses a filing whose revenue the unit check suppressed (or could not run on):
    every figure in a filing shares one unit, so a revenue we could not trust makes
    the EBITDA and PAT beside it untrustworthy too.
    """
    if not filing:
        return False
    qi = GP.parse_q(filing.get("quarter"))
    if not qi or filing.get("revenue_cr") is None:
        return False
    if "suppressed" in str(filing.get("units_note") or ""):
        return False
    qi = GP.q_idx(*qi)
    added = False
    for field, line in FILING_LINES.items():
        v = BGP._f(filing.get(field))
        if v is None:
            continue
        ser = series.setdefault((isin, line), {})
        if qi not in ser:
            ser[qi] = v
            added = True
    return added


# --------------------------------------------------------------------------- #
#  the measurement                                                            #
# --------------------------------------------------------------------------- #

def company_progress(commits: list[dict], series: dict, season: str | None = None
                     ) -> pd.DataFrame:
    """guidance_progress rows (PROGRESS_COLS) for one company's open P&L promises,
    measured INCLUDING the quarter just reported. Empty frame when nothing is open."""
    if not commits:
        return pd.DataFrame(columns=BGP.PROGRESS_COLS)
    now_q = GP.q_idx(*GP.parse_q(QT.norm_q(season or QT.season_quarter())))
    cal_q = GP.q_from_date(date.today())
    now = datetime.utcnow().isoformat(timespec="seconds")
    # orders / netblock are empty on purpose: only P&L kinds reach here
    df = BGP.build(commits, series, {}, {}, now_q, cal_q, set(), {}, now)
    return results_day_filter(df, now_q)


# A margin promise is only comparable to the company's OPM when it IS the company's
# operating margin. "Aero margin 23-25%" or "Real Estate ROCE 25%" measured against
# consolidated OPM produced ten false AT_RISK rows for RAYMOND (live dry-run
# 2026-10-10). Precision over recall: an unqualified "margin" is dropped too.
_CO_MARGIN = re.compile(r"ebitda|operating|\bopm\b|consolidated|overall|company", re.I)
_NOT_CO = re.compile(r"\bro(c)?e\b|segment|division|business|subsidiar", re.I)
RATE_MAX_QUARTERS_AHEAD = 4     # a rate target ending > 1 year out is not judgeable now
# A parenthetical naming a business line marks a SEGMENT target: RAYMOND's
# "INR 3,500-4,000 cr (Real Estate)" read BEHIND against total company revenue. Words
# that only restate the metric, unit or basis are allowed inside the brackets.
_PAREN = re.compile(r"\(([^)]*)\)")
_PAREN_OK = re.compile(
    r"^[\s\d.,%~+\-–$₹]*(?:\b(?:inr|rs|usd|us\$|cr|crs?|crores?|mn|million|bn|billion|"
    r"lakh|approx|approximately|about|around|consolidated|standalone|company|overall|"
    r"total|ebitda|operating|opm|margin|revenue|sales|topline|pat|profit|net|cagr|yoy|"
    r"growth|y-?o-?y|fy\s*\d{2,4}|h[12]|q[1-4]|annual|annually|p\.?a\.?|organic|"
    r"guidance|target|level|lvl|abs)\b[\s\d.,%~+\-–$₹/&]*)*$", re.I)
# Vague horizons get a GUESSED window in guidance_progress (NEAR_TERM = 2 quarters).
# Fine for a weekly trend; not for a results-day verdict, where "100% of time
# elapsed, AT_RISK" on a promise that never named a date is simply unfair.
_VAGUE = set(GP._VAGUE_QUARTERS)


def _segment_text(text: str) -> bool:
    return any(not _PAREN_OK.match(m) for m in _PAREN.findall(str(text or "")))


def results_day_filter(df: pd.DataFrame, now_q: int) -> pd.DataFrame:
    """Keep only promises a results day can fairly judge (results-day mail only; the
    weekly guidance-progress table is untouched):
      1. margin rows must name the company-level margin (see _CO_MARGIN / _NOT_CO);
      2. growth/margin targets ending more than a year out are dropped - a rate is
         judged on level, not pace, so 'long run 25%' would read AT_RISK forever;
      3. one row per promise: the same target restated across concalls (7 wordings of
         RAYMOND's Aero margin) collapses to its newest telling;
      4. any row naming a business line in brackets is a segment target (_PAREN_OK);
      5. vague horizons (NEAR_TERM, MEDIUM_TERM, ...) are dropped - no stated date,
         so no fair pace verdict.
    """
    if df is None or df.empty:
        return df
    d = df.copy()
    txt = d["guided_text"].astype(str)
    bad_margin = (d["kind"] == "margin") & (~txt.str.contains(_CO_MARGIN)
                                           | txt.str.contains(_NOT_CO))
    end_q = d["horizon_end"].map(lambda x: GP.q_idx(*GP.parse_q(x))
                                 if GP.parse_q(x) else None)
    far = d["kind"].isin(["growth", "margin"]) & end_q.map(
        lambda e: e is None or e > now_q + RATE_MAX_QUARTERS_AHEAD)
    segment = txt.map(_segment_text)
    vague = d["target_period"].astype(str).str.strip().str.upper().isin(_VAGUE)
    d = d[~bad_margin & ~far & ~segment & ~vague]
    if d.empty:
        return d
    d["_gv"] = pd.to_numeric(d["guided_value"], errors="coerce").round(1)
    d["_qo"] = d["guid_quarter"].map(QT.q_order)
    d = (d.sort_values("_qo")
          .drop_duplicates(["metric", "kind", "horizon_end", "_gv"], keep="last")
          .drop(columns=["_gv", "_qo"]))
    return d.reset_index(drop=True)


def measured(df: pd.DataFrame) -> pd.DataFrame:
    """Rows with a real pace verdict, worst first (the reader's question is 'what is
    slipping?'), then by how far along."""
    if df is None or df.empty:
        return pd.DataFrame(columns=BGP.PROGRESS_COLS)
    m = df[df["status"].isin(STATUS_ORDER)].copy()
    m["_o"] = m["status"].map(STATUS_ORDER.index)
    return m.sort_values(["_o", "pct_of_target"]).drop(columns="_o")


def summary(df: pd.DataFrame) -> dict:
    """{'ahead': n, 'on_track': n, 'behind': n, 'measured': n, 'open': n}.
    ACHIEVED counts as ahead; AT_RISK as behind."""
    m = measured(df)
    s = m["status"].value_counts().to_dict() if not m.empty else {}
    return {"ahead": s.get("AHEAD", 0) + s.get("ACHIEVED", 0),
            "on_track": s.get("ON_TRACK", 0),
            "behind": s.get("BEHIND", 0) + s.get("AT_RISK", 0),
            "measured": len(m), "open": 0 if df is None else len(df)}


def summary_label(s: dict) -> str:
    """One-word guidance state for a league-table cell."""
    if not s or not s.get("measured"):
        return ""
    if s["behind"] and not (s["ahead"] or s["on_track"]):
        return "BEHIND"
    if s["behind"]:
        return "MIXED"
    if s["ahead"] and not s["on_track"]:
        return "AHEAD"
    return "ON_TRACK"


# --------------------------------------------------------------------------- #
#  html                                                                       #
# --------------------------------------------------------------------------- #

def chip(status: str) -> str:
    bg, fg = STATUS_CHIP.get(status, ("#eee", "#666"))
    return (f"<span style='background:{bg};color:{fg};padding:1px 6px;border-radius:3px;"
            f"font-size:11px;font-weight:700'>{_esc(status.replace('_', ' '), 12)}</span>")


def _so_far(r) -> str:
    atd = BGP._f(r.get("actual_to_date"))
    if atd is None:
        return "&mdash;"
    k = str(r.get("kind") or "")
    if k == "growth":
        return f"{atd:+.1f}% YoY"
    if k == "margin":
        return f"{atd:.1f}% latest qtr"
    n, t = r.get("periods_elapsed"), r.get("periods_total")
    return f"&#8377;{atd:,.0f} cr" + (f" &middot; {n} of {t} qtrs" if n and t else "")


def _progress(r) -> str:
    pct, tp = BGP._f(r.get("pct_of_target")), BGP._f(r.get("time_pct"))
    if pct is None:
        return "&mdash;"
    if str(r.get("kind") or "") == "level" and tp:
        return f"{pct:.0f}% of target &middot; {tp:.0f}% of time"
    return f"{pct:.0f}% of the guided figure"


def render_block(df: pd.DataFrame, filing_quarter: str | None = None,
                 tbl=None, heading=None, max_rows: int = MAX_ROWS) -> str:
    """The per-company block. `tbl` / `heading` are the caller's own table and heading
    helpers, so the block looks like the page it sits in."""
    m = measured(df)
    title = "D2 &middot; Where this quarter leaves the guidance"
    sub = ("Each open promise measured including the quarter just reported: how much "
           "has landed against how much of the horizon has passed.")
    head = heading(title, sub) if heading else f"<h3>{title}</h3><div>{sub}</div>"
    muted = "color:#8a97a0;font-size:12px"
    if m.empty:
        n_open = 0 if df is None else len(df)
        why = (f"{n_open} open promise{'' if n_open == 1 else 's'}, none measurable yet "
               f"(horizon not started, a segment figure, or no matching line)."
               if n_open else "No open revenue / EBITDA / PAT / margin guidance on record.")
        return head + f"<div style='{muted}'>{why}</div>"
    rows = []
    for _, r in m.head(max_rows).iterrows():
        rows.append([
            _esc(str(r.get("metric") or "").replace("_", " ").title(), 20),
            f"{_esc(r.get('guided_text'), 60)} "
            f"<span style='color:#8a97a0'>({_esc(r.get('target_period'), 12)})</span>",
            _so_far(r), _progress(r), chip(str(r.get("status")))])
    hdr = ["Metric", "Guided (horizon)", "Delivered so far", "Progress", "Pace"]
    table = tbl(hdr, rows) if tbl else (
        "<table>" + "".join("<tr>" + "".join(f"<td>{c}</td>" for c in rw) + "</tr>"
                            for rw in [hdr] + rows) + "</table>")
    s = summary(df)
    notes = [f"{s['ahead']} ahead &middot; {s['on_track']} on track &middot; "
             f"{s['behind']} behind, of {s['measured']} measurable."]
    if len(m) > max_rows:
        notes.append(f"+{len(m) - max_rows} more not shown (worst shown first).")
    if filing_quarter:
        notes.append(f"{_esc(filing_quarter, 12)} figures are the company's own exchange "
                     f"filing &mdash; Screener has not republished them yet.")
    notes.append("Order-book and capex targets are in the Monday guidance-progress mail.")
    return head + table + f"<div style='{muted}'>{' '.join(notes)}</div>"


# --------------------------------------------------------------------------- #
#  self-test — no Drive, no network                                           #
# --------------------------------------------------------------------------- #

def _self_test() -> int:
    fails = 0

    def check(name, ok):
        nonlocal fails
        print(("  ok   " if ok else "  FAIL ") + name)
        fails += 0 if ok else 1

    season = "Q2FY27"
    isin = "INE_T"

    def stmts(periods_vals: dict[str, dict]) -> pd.DataFrame:
        rows = []
        for line, pv in periods_vals.items():
            for p, v in pv.items():
                rows.append({"statement": "quarterly_pl", "line_item": line,
                             "period": p, "value": v})
        return pd.DataFrame(rows)

    # FY26 quarters (Jun25..Mar26) + FY27 Q1, Q2. Revenue 100/qtr in FY26, 130 in FY27.
    per = ["Jun 2025", "Sep 2025", "Dec 2025", "Mar 2026", "Jun 2026", "Sep 2026"]
    st = stmts({"Sales +": dict(zip(per, [100, 100, 100, 100, 130, 130])),
                "Net Profit +": dict(zip(per, [10, 10, 10, 10, 9, 9])),
                "OPM %": dict(zip(per, [15, 15, 15, 15, 18, 18]))})
    ser = series_from_statements(st, isin)
    check("aliases resolve to financials_3stmt line names",
          set(ser) == {(isin, "Sales"), (isin, "Net Profit"), (isin, "OPM %")})
    check("Sep 2026 maps to Q2 FY27",
          ser[(isin, "Sales")].get(GP.q_idx(27, 2)) == 130.0)

    tracker = pd.DataFrame([
        # revenue growth 20% FY27 -> realised +30% -> AHEAD
        {"isin": isin, "symbol": "T", "company_name": "T", "quarter": "Q4 FY26",
         "metric": "Revenue", "horizon_fy": "1Y", "value": "20%",
         "source_doc_id": "d1"},
        # PAT Rs 60 cr FY27 -> 18 of 60 at 50% time -> 30% of target -> behind pace
        {"isin": isin, "symbol": "T", "company_name": "T", "quarter": "Q4 FY26",
         "metric": "PAT", "horizon_fy": "FY27", "value": "Rs 60 crore",
         "source_doc_id": "d1"},
        # another company — must be filtered out before parsing
        {"isin": "OTHER", "symbol": "O", "company_name": "O", "quarter": "Q4 FY26",
         "metric": "Revenue", "horizon_fy": "FY27", "value": "Rs 500 crore",
         "source_doc_id": "d2"},
    ])
    commits = prepare_commitments(tracker, None, None, None, isins={isin})
    check("isin filter applied before parse", set(commits) == {isin})
    df = company_progress(commits.get(isin, []), ser, season)
    st_by = dict(zip(df["metric"], df["status"])) if not df.empty else {}
    check("growth promise measured AHEAD", st_by.get("revenue") == "AHEAD")
    check("PAT level promise BEHIND or AT_RISK",
          st_by.get("pat") in ("BEHIND", "AT_RISK"))
    s = summary(df)
    check("summary counts", s["ahead"] == 1 and s["behind"] == 1 and s["measured"] == 2)
    check("summary label MIXED", summary_label(s) == "MIXED")

    # Screener lagging: statements stop at Q1; the filing supplies Q2.
    st_lag = st[st["period"] != "Sep 2026"]
    ser_lag = series_from_statements(st_lag, isin)
    fil = {"quarter": "Q2 FY27", "revenue_cr": 130.0, "pat_cr": 9.0,
           "ebitda_cr": None, "ebitda_margin_pct": 18.0}
    check("filing quarter injected", add_filing_quarter(ser_lag, isin, fil))
    df_lag = company_progress(commits[isin], ser_lag, season)
    check("lagging + filing == full statements",
          dict(zip(df_lag["metric"], df_lag["pct_of_target"]))
          == dict(zip(df["metric"], df["pct_of_target"])))
    check("filing never overwrites Screener",
          not add_filing_quarter(ser, isin, {**fil, "revenue_cr": 999.0})
          and ser[(isin, "Sales")][GP.q_idx(27, 2)] == 130.0)
    check("suppressed unit check refuses the filing",
          not add_filing_quarter({}, isin, {**fil, "units_note": "implausible ... "
                                            "suppressed rather than shown"}))
    check("no revenue refuses the filing",
          not add_filing_quarter({}, isin, {**fil, "revenue_cr": None}))

    html = render_block(df, filing_quarter="Q2 FY27")
    check("block renders both rows", "AHEAD" in html and "Pat" in html)
    check("filing note present", "exchange filing" in html)
    check("block is small", len(html) < 6000)
    empty = render_block(company_progress([], {}, season))
    check("no guidance -> says so", "No open" in empty)
    check("summary_label empty", summary_label(summary(pd.DataFrame())) == "")

    # results_day_filter — the RAYMOND cases from the 2026-10-10 live dry-run
    now_q = GP.q_idx(27, 2)

    def prow(text, kind, end, gq="Q4 FY26", gv=25.0, metric="margin", tp="FY27"):
        return {"metric": metric, "kind": kind, "guided_text": text,
                "horizon_end": end, "guided_value": gv, "guid_quarter": gq,
                "status": "AT_RISK", "target_period": tp}
    f = results_day_filter(pd.DataFrame([
        prow("Aero margin 23% to 25% in the long run", "margin", "Q4 FY27"),
        prow("LVL: 25% (Real Estate ROCE)", "margin", "Q4 FY27"),
        prow("LVL: ~25% (EBITDA margin)", "margin", "Q4 FY27"),
        prow("LVL: ~25% (EBITDA margin)", "margin", "Q4 FY27", gq="Q1 FY27"),
        prow("EBITDA margin 25% long term", "margin", "Q4 FY30"),
        prow("Rs 500 crore", "level", "Q4 FY30", metric="revenue", gv=500.0),
    ]), now_q)
    check("segment margin dropped", not f["guided_text"].str.contains("Aero").any())
    check("ROCE dropped", not f["guided_text"].str.contains("ROCE").any())
    check("company EBITDA margin kept, deduped to newest",
          (f["metric"] == "margin").sum() == 1
          and f.loc[f["metric"] == "margin", "guid_quarter"].iloc[0] == "Q1 FY27")
    check("far-off rate target dropped, far-off level target kept",
          set(f["kind"]) == {"margin", "level"})
    f2 = results_day_filter(pd.DataFrame([
        prow("ABS: INR 3,500-4,000 cr (Real Estate)", "level", "Q4 FY27",
             metric="revenue", gv=3750.0),
        prow("2,200 crores", "level", "Q2 FY27", metric="ebitda", gv=2200.0,
             tp="NEAR_TERM"),
        prow("INR 500+ cr (FY27)", "level", "Q4 FY27", metric="ebitda", gv=500.0),
    ]), now_q)
    check("segment level target dropped, vague horizon dropped, FY target kept",
          list(f2["guided_text"]) == ["INR 500+ cr (FY27)"])

    print(f"\n{'ALL PASS' if not fails else f'{fails} FAILED'}")
    return 1 if fails else 0


if __name__ == "__main__":
    if "--self-test" in sys.argv:
        raise SystemExit(_self_test())
    print(__doc__)
