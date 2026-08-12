r"""
quarter_teardown.py — the quarterly teardown page: results + deck + early warnings.

Implements docs/quarterly_teardown_framework.md block for block:

  A  verdict strip            quarterly_pl                      arithmetic
  B  quality of the number    quarterly_pl                      arithmetic
  D  said vs delivered        guidance_vs_actual                already computed
  H1 divergence engine        financials_derived + balance_sheet arithmetic
  H2 red-flag register        redflag_register (7 sources)      no model
  H3 AR forensic score        ar_scorecard (parsed from md)     no model
  C/E/F/G deck                deck_metrics/diff/flags/questions  model, grounded

Six of the eight blocks never touch a model. The two that do are grounded at parse time
by deck_teardown.parse_teardown, so a figure that does not appear in its own evidence span
never reaches this file.

NO BARE NUMBERS. Every value is a provenance.Fact carrying source, field, period, grain
and origin. `--audit` writes the whole fact graph beside the page, so any figure can be
walked back to the parquet cell or the verbatim sentence it came from.

    python scripts/quarter_teardown.py --symbols APLAPOLLO --html
    python scripts/quarter_teardown.py --pf --mail --dry-run
    python scripts/quarter_teardown.py --symbols APLAPOLLO --html --audit
"""
from __future__ import annotations

import argparse
import io
import os
import sys
from datetime import date, datetime

_SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

import pandas as pd
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(_SCRIPTS_DIR), ".env"))

import quarterly_table as QT
import redflag_register as RR
import ar_scorecard as ARS
import deck_teardown as DT
from provenance import (Fact, MISSING, SCREENER, ANNUAL, QUARTERLY, write_audit, audit)

MAIL_KEY = "quarter_teardown"
MAX_HTML_BYTES = 90_000          # Gmail clips near 102 KB

# H1 thresholds. Each is the rule this repo already encodes, not a new opinion:
# cfo/leverage/coverage mirror build_fraud_risk.forensic_flags().
THRESHOLDS = {
    "cfo_pat_ratio":     (1.0, "below", "CFO below PAT"),
    "net_debt_ebitda":   (4.0, "above", "leverage above 4x"),
    "interest_coverage": (1.5, "below", "coverage below 1.5x"),
}
UP, DOWN, MUTED = "#1a7a3a", "#c0392b", "#8a97a0"
AMBER = "#b8860b"

# guidance_vs_actual.parquet carries TWO verdict vocabularies in ONE table:
#   source=pead_*   -> BEAT / INLINE / MISS / NA   (build_pead_flags._verdict, ±2 band)
#   source=gf_track -> DELIVERED / EXCEEDED / MISSED / PARTIAL / TOO_EARLY / NA
#                      (concall_prompt.txt §GF_TRACK, same ±2 band)
# MEASURED = the guidance was joined back to a reported outcome, so the row answers
# "was the promise kept?". Filtering on the pead spelling alone silently discarded
# every Gemini verdict for any company that also had one pead row.
MEASURED_VERDICTS = {"BEAT", "INLINE", "MISS",
                     "DELIVERED", "EXCEEDED", "MISSED", "PARTIAL"}
# Points behind the credibility score, straight from concall_prompt.txt §Mgmt Credibility
# Summary. Only the gf_track vocabulary has defined points — BEAT/INLINE/MISS carry none,
# and inventing a mapping for them would put a number on the page that no rule produced.
CRED_POINTS = {"EXCEEDED": 5, "DELIVERED": 4, "PARTIAL": 3, "MISSED": 1}
VERDICT_TONE = {"BEAT": UP, "EXCEEDED": UP, "DELIVERED": UP,
                "MISS": DOWN, "MISSED": DOWN,
                "INLINE": MUTED, "PARTIAL": AMBER}
# Credibility pattern vocabulary — concall_prompt.txt §Mgmt Credibility Summary.
CRED_PATTERNS = {"Calibrated", "Optimistic Bias", "Conservative Bias",
                 "Erratic", "Insufficient Data"}


# --------------------------------------------------------------------------- #
# Drive I/O
# --------------------------------------------------------------------------- #

def _folder(drive, parts):
    from _extractor_base import get_or_create_subfolder
    fid = os.environ["GDRIVE_FOLDER_ID"]
    for p in parts.split("/"):
        fid = get_or_create_subfolder(drive, fid, p)
    return fid


def _read(drive, fid, name) -> pd.DataFrame:
    from _extractor_base import find_file, download_bytes
    f = find_file(drive, fid, name)
    if not f:
        return pd.DataFrame()
    try:
        return pd.read_parquet(io.BytesIO(download_bytes(drive, f)))
    except Exception:
        return pd.DataFrame()


# --------------------------------------------------------------------------- #
# Fact builders
# --------------------------------------------------------------------------- #

def _q_series(stmts: pd.DataFrame, item: str) -> dict:
    """{period: float} for one quarterly_pl line item, junk rows removed."""
    if stmts is None or stmts.empty:
        return {}
    q = stmts[(stmts["statement"] == "quarterly_pl")
              & (stmts["line_item"].astype(str).str.strip() == item)]
    out = {}
    for _, r in q.iterrows():
        try:
            out[str(r["period"])] = float(r["value"])
        except (TypeError, ValueError):
            pass
    return out


def _bs_series(stmts: pd.DataFrame, item: str) -> dict:
    if stmts is None or stmts.empty:
        return {}
    b = stmts[(stmts["statement"] == "balance_sheet")
              & (stmts["line_item"].astype(str).str.strip() == item)]
    out = {}
    for _, r in b.iterrows():
        try:
            out[str(r["period"])] = float(r["value"])
        except (TypeError, ValueError):
            pass
    return out


def _periods(stmts: pd.DataFrame) -> list[str]:
    q = stmts[stmts["statement"] == "quarterly_pl"] if not stmts.empty else stmts
    if q is None or q.empty:
        return []
    ps = {str(p) for p in q["period"]}
    return sorted(ps, key=lambda p: QT.q_order(QT.qtr_label(p)))


def _fact(series: dict, period: str, item: str, src: str, isin: str,
          unit: str = "Cr") -> Fact:
    v = series.get(period)
    if v is None:
        return MISSING(item, src, f"no {item} row for {period}", period, QUARTERLY, isin)
    return Fact(v, src, item, period, QUARTERLY, SCREENER, unit, key=isin)


def _derived_map(derived: pd.DataFrame, isin: str, metric: str) -> dict:
    """{period: value} for one annual financials_derived metric."""
    if derived is None or derived.empty:
        return {}
    d = derived[(derived["isin"].astype(str) == isin)
                & (derived["metric"] == metric)
                & (derived["period_type"] == "annual")]
    out = {}
    for _, r in d.iterrows():
        try:
            out[str(r["period"])] = float(r["value"])
        except (TypeError, ValueError):
            pass
    return {k: v for k, v in out.items() if k.upper() != "TTM"}


def _ordered_annual(m: dict) -> list[tuple[str, float]]:
    def key(p):
        try:
            return pd.to_datetime(p).toordinal()
        except Exception:
            return 0
    return sorted(m.items(), key=lambda kv: key(kv[0]))


def build_facts(isin: str, symbol: str, name: str, stmts: pd.DataFrame,
                derived: pd.DataFrame, gva: pd.DataFrame, register: pd.DataFrame,
                arsc: pd.DataFrame, deck: dict) -> dict:
    """Every Fact the page can draw, keyed by block. Missing inputs yield MISSING
    Facts, never zeros — see provenance.MISSING."""
    src = f"fundamentals/statements/{symbol}.parquet"
    P = _periods(stmts)
    out: dict = {"isin": isin, "symbol": symbol, "name": name, "periods": P,
                 "register": register, "ar_scorecard": arsc, "deck": deck,
                 "gva": gva, "facts": []}
    if len(P) < 2:
        out["quarter"] = None
        return out

    cur, qoq = P[-1], P[-2]
    yoy = P[-5] if len(P) >= 5 else None
    out["period"], out["quarter"] = cur, QT.qtr_label(cur)
    out["qoq_period"], out["yoy_period"] = qoq, yoy

    S = {k: _q_series(stmts, k) for k in
         ("Sales", "Expenses", "Operating Profit", "OPM %", "Other Income", "Interest",
          "Depreciation", "Profit before tax", "Tax %", "Net Profit", "EPS in Rs")}

    def F(item, p, unit="Cr"):
        return _fact(S[item], p, item, src, isin, unit)

    # ---- Block A
    A = {}
    for label, item, unit in (("revenue", "Sales", "Cr"), ("pat", "Net Profit", "Cr"),
                              ("eps", "EPS in Rs", "Rs"), ("opm", "OPM %", "%")):
        c = F(item, cur, unit)
        A[label] = {"cur": c,
                    "yoy": _delta(c, F(item, yoy, unit) if yoy else None, unit),
                    "qoq": _delta(c, F(item, qoq, unit), unit)}
    npm_cur = _npm(S, cur, src, isin)
    A["npm"] = {"cur": npm_cur,
                "yoy": _delta(npm_cur, _npm(S, yoy, src, isin) if yoy else None, "%"),
                "qoq": _delta(npm_cur, _npm(S, qoq, src, isin), "%")}
    out["A"] = A

    # ---- Block B
    B = {}
    pbt_c, oi_c = F("Profit before tax", cur), F("Other Income", cur)
    B["oi_share"] = _share(oi_c, pbt_c, "other_income_pct_of_pbt")
    if yoy:
        B["oi_share_yoy"] = _share(F("Other Income", yoy), F("Profit before tax", yoy),
                                   "other_income_pct_of_pbt")
    tax_c = F("Tax %", cur, "%")
    prior4 = [S["Tax %"].get(p) for p in P[-5:-1] if S["Tax %"].get(p) is not None]
    if prior4:
        tax_mean = Fact.derive(sum(prior4) / len(prior4), "tax_rate_prior4_mean",
                               [Fact(v, src, "Tax %", p, QUARTERLY, SCREENER, "%",
                                     key=isin)
                                for p, v in zip(P[-5:-1], prior4)],
                               "mean of the previous four quarters", "%")
    else:
        tax_mean = MISSING("tax_rate_prior4_mean", src, "fewer than 4 prior quarters")
    B["tax_cur"], B["tax_mean"] = tax_c, tax_mean

    for label, item in (("interest", "Interest"), ("depreciation", "Depreciation")):
        B[label] = F(item, cur)
        B[label + "_qoq"] = _pct_change(F(item, cur), F(item, qoq), label + "_qoq_pct")

    # ---- Adjusted PAT, built as an explicit bridge FROM reported PAT.
    #
    # Anchored on the reported figure rather than recomputed from PBT, so the
    # adjustments tie out exactly and every line can be shown:
    #
    #   Reported PAT
    #     less   the YoY INCREASE in other income, after tax at the actual rate
    #     add    the amount over-taxed this quarter vs the prior-4Q average rate
    #   = Adjusted PAT
    #
    # Recomputing as PBT x (1 - normal rate) instead leaves an unexplained residual,
    # because Screener rounds PAT and PBT independently to whole crore.
    pat_c = F("Net Profit", cur)
    tax_c_f = B["tax_cur"]
    if yoy and pat_c.present and pbt_c.present and oi_c.present \
            and tax_mean.present and tax_c_f.present:
        oi_y = F("Other Income", yoy)
        if oi_y.present:
            t_cur, t_bar = tax_c_f.num, tax_mean.num
            oi_excl = oi_c.num - oi_y.num
            oi_effect = oi_excl * (1 - t_cur / 100.0)
            tax_norm = (pbt_c.num - oi_excl) * (t_cur - t_bar) / 100.0
            adjusted = pat_c.num - oi_effect + tax_norm

            B["oi_excluded"] = Fact.derive(
                round(oi_excl, 1), "other_income_excluded_pretax", [oi_c, oi_y],
                f"other income {oi_c.num:,.0f} this quarter less {oi_y.num:,.0f} a year "
                f"ago — only the INCREASE is excluded, not the whole line", "Cr")
            B["oi_effect"] = Fact.derive(
                round(oi_effect, 1), "other_income_effect_posttax",
                [B["oi_excluded"], tax_c_f],
                f"the excluded {oi_excl:,.0f} Cr after tax at the actual {t_cur:.1f}%",
                "Cr")
            B["tax_norm"] = Fact.derive(
                round(tax_norm, 1), "tax_normalisation_effect",
                [pbt_c, B["oi_excluded"], tax_c_f, tax_mean],
                f"tax at {t_cur:.1f}% vs the prior-4Q average {t_bar:.2f}%, applied to "
                f"{pbt_c.num - oi_excl:,.0f} Cr", "Cr")
            B["adjusted_pat"] = Fact.derive(
                round(adjusted, 1), "adjusted_pat",
                [pat_c, B["oi_effect"], B["tax_norm"]],
                "reported PAT, less the post-tax effect of the rise in other income, "
                "plus the amount over- or under-taxed against the prior-4Q rate", "Cr")
            B["adjusted_gap"] = Fact.derive(
                round((adjusted / pat_c.num - 1) * 100, 1) if pat_c.num else None,
                "adjusted_pat_gap_pct", [B["adjusted_pat"], pat_c],
                "adjusted PAT vs reported PAT", "%")
        else:
            B["adjusted_pat"] = MISSING("adjusted_pat", src, "no year-ago other income")
    else:
        B["adjusted_pat"] = MISSING("adjusted_pat", src, "insufficient history")
    B["pat"] = pat_c
    # Screener publishes one combined Expenses line — gross margin is not separable.
    B["gross_vs_opex"] = MISSING(
        "gross_margin_split", src,
        "Screener publishes a single combined 'Expenses' line — not computable")
    out["B"] = B

    # ---- Block H1
    H1, latest_fy = {}, None
    for metric, unit in (("cfo_pat_ratio", "x"), ("net_debt_ebitda", "x"),
                         ("interest_coverage", "x"), ("roce_pct", "%"),
                         ("roe_pct", "%"), ("rev_cagr_3y_pct", "%"),
                         ("receivable_days", "days"), ("inventory_days", "days"),
                         ("wc_days", "days"), ("ccc_days", "days")):
        m = _derived_map(derived, isin, metric)
        ser = _ordered_annual(m)
        if not ser:
            H1[metric] = MISSING(metric, "financials_derived.parquet",
                                 "not computed for this company", grain=ANNUAL, key=isin)
            H1[metric + "_series"] = []
            continue
        p, v = ser[-1]
        latest_fy = latest_fy or p
        H1[metric] = Fact(v, "financials_derived.parquet", metric, p, ANNUAL,
                          SCREENER, unit, key=isin)
        H1[metric + "_series"] = ser[-6:]
    for item in ("Borrowings", "CWIP", "Fixed Assets", "Total Assets"):
        ser = _ordered_annual(_bs_series(stmts, item))
        H1[item + "_series"] = ser[-5:]
        H1[item] = (Fact(ser[-1][1], src, item, ser[-1][0], ANNUAL, SCREENER, "Cr",
                         key=isin) if ser
                    else MISSING(item, src, "no balance-sheet row", grain=ANNUAL, key=isin))
    H1["_fy"] = latest_fy or ""
    H1["agency_leverage_trigger"] = _agency_leverage_trigger(register)
    out["H1"] = H1

    out["facts"] = _collect(out)
    return out


def _collect(out: dict) -> list[Fact]:
    facts = []
    for blk in ("A", "B", "H1"):
        for v in (out.get(blk) or {}).values():
            if isinstance(v, Fact):
                facts.append(v)
            elif isinstance(v, dict):
                facts.extend(x for x in v.values() if isinstance(x, Fact))
    return facts


def _npm(S, period, src, isin) -> Fact:
    if period is None:
        return MISSING("npm_pct", src, "no period")
    pat, sales = S["Net Profit"].get(period), S["Sales"].get(period)
    if pat is None or not sales:
        return MISSING("npm_pct", src, "missing PAT or Sales", period, QUARTERLY, isin)
    pf = Fact(pat, src, "Net Profit", period, QUARTERLY, SCREENER, "Cr", key=isin)
    sf = Fact(sales, src, "Sales", period, QUARTERLY, SCREENER, "Cr", key=isin)
    return Fact.derive(round(pat / sales * 100, 2), "npm_pct", [pf, sf],
                       "Net Profit / Sales x 100", "%")


def _delta(cur: Fact, base: Fact | None, unit: str) -> Fact:
    """Relative % for amounts; percentage POINTS for anything already a %."""
    if base is None:
        return MISSING("delta", "computed", "no comparison period")
    if cur.missing or base.missing or base.num in (None, 0):
        return MISSING("delta", "computed", "missing input")
    if unit == "%":
        return Fact.derive(round(cur.num - base.num, 2), f"{cur.field}_delta_pp",
                           [cur, base], "percentage-point change", "pp")
    return Fact.derive(round((cur.num / base.num - 1) * 100, 1), f"{cur.field}_delta_pct",
                       [cur, base], "relative % change", "%")


def _pct_change(cur: Fact, base: Fact, field_: str) -> Fact:
    if cur.missing or base.missing or base.num in (None, 0):
        return MISSING(field_, "computed", "missing input")
    return Fact.derive(round((cur.num / base.num - 1) * 100, 1), field_, [cur, base],
                       "relative % change", "%")


def _share(part: Fact, whole: Fact, field_: str) -> Fact:
    if part.missing or whole.missing or whole.num in (None, 0):
        return MISSING(field_, "computed", "missing input")
    return Fact.derive(round(part.num / whole.num * 100, 2), field_, [part, whole],
                       f"{part.field} / {whole.field} x 100", "%")


def _agency_leverage_trigger(register: pd.DataFrame) -> Fact:
    """The numeric debt threshold an agency named in its own downgrade trigger.

    Grounded by construction: the number is read out of the verbatim trigger text and
    the text travels with it as evidence.
    """
    import re
    if register is None or register.empty:
        return MISSING("agency_leverage_trigger", "rating_sensitivity", "no register")
    d = register[register["kind"] == "downgrade_trigger"]
    for _, r in d.iterrows():
        txt = str(r.get("detail") or "")
        m = re.search(r"(?:debt\s*/\s*OPBDITA|debt/OPBDITA|debt\s*to\s*OPBDITA|"
                      r"total\s+debt\s*/\s*\w+)[^0-9]{0,40}(\d+(?:\.\d+)?)\s*times", txt, re.I)
        if m:
            return Fact(float(m.group(1)), "rating_sensitivity.parquet",
                        "agency_leverage_trigger", str(r.get("period") or ""), ANNUAL,
                        "agency", "x", key=str(r.get("isin") or ""), evidence=txt[:300],
                        note=f"stated by {r.get('ref') or 'the agency'}")
    return MISSING("agency_leverage_trigger", "rating_sensitivity",
                   "no numeric leverage threshold in any downgrade trigger")


# --------------------------------------------------------------------------- #
# rendering helpers
# --------------------------------------------------------------------------- #

def esc(s, n=400) -> str:
    from mailer import esc as _e
    return _e(s, n)


def fmt(f: Fact, dp: int = 0, suffix: str = "") -> str:
    if f is None or f.missing:
        return "&mdash;"
    v = f.num
    if v is None:
        return esc(f.value, 60)
    return f"{v:,.{dp}f}{suffix}"


def signed(f: Fact, dp: int = 1) -> str:
    if f is None or f.missing or f.num is None:
        return "&mdash;"
    unit = "pp" if f.unit == "pp" else "%"
    return f"{f.num:+,.{dp}f}{unit}"


def tone(f: Fact, good_up: bool = True) -> str:
    if f is None or f.missing or f.num is None:
        return MUTED
    if abs(f.num) < 0.05:
        return MUTED
    return UP if ((f.num > 0) == good_up) else DOWN


def _spark(series, w=104, h=30, colour=UP) -> str:
    """Inline SVG sparkline — no JS, survives email clients that allow SVG and degrades
    to nothing in those that do not."""
    vals = [v for _p, v in series if v is not None]
    if len(vals) < 2:
        return ""
    lo, hi = min(vals), max(vals)
    rng = (hi - lo) or 1.0
    step = (w - 6) / (len(vals) - 1)
    pts = " ".join(f"{3 + i * step:.1f},{h - 4 - (v - lo) / rng * (h - 9):.1f}"
                   for i, v in enumerate(vals))
    last = pts.split()[-1]
    return (f"<svg width='{w}' height='{h}' viewBox='0 0 {w} {h}'>"
            f"<polyline points='{pts}' fill='none' stroke='{colour}' stroke-width='1.8' "
            f"stroke-linejoin='round'/>"
            f"<circle cx='{last.split(',')[0]}' cy='{last.split(',')[1]}' r='2.5' "
            f"fill='{colour}'/></svg>")


def _verdict(metric: str, f: Fact) -> tuple[str, str]:
    """(label, colour) against the repo's own encoded rule."""
    if f is None or f.missing or f.num is None:
        return "not covered", MUTED
    thr = THRESHOLDS.get(metric)
    if not thr:
        return "", MUTED
    limit, direction, _why = thr
    bad = f.num < limit if direction == "below" else f.num > limit
    return ("breach" if bad else "clear"), (DOWN if bad else UP)


# --------------------------------------------------------------------------- #
# blocks -> HTML (shared by both renderers; `rich` adds sparklines and CSS classes)
# --------------------------------------------------------------------------- #

_TD = "padding:6px 10px;border:1px solid #ddd;font-size:13px;font-family:Arial,sans-serif;"
_TH = _TD + "background:#34495e;color:#fff;text-align:left;"


def _tbl(headers, rows) -> str:
    h = "".join(f"<th style='{_TH}'>{c}</th>" for c in headers)
    b = "".join("<tr>" + "".join(f"<td style='{_TD}'>{c}</td>" for c in r) + "</tr>"
                for r in rows)
    return (f"<table style='border-collapse:collapse;margin:6px 0 14px'>"
            f"<tr>{h}</tr>{b}</table>")


def _h(title: str, sub: str = "") -> str:
    s = (f"<div style='color:#7a8791;font-size:12px;margin:0 0 8px'>{sub}</div>"
         if sub else "")
    return (f"<h3 style='margin:20px 0 4px;font-size:14px;color:#34495e;"
            f"border-bottom:2px solid #34495e;padding-bottom:3px'>{title}</h3>{s}")


def _verdicts(gva) -> pd.Series:
    """Upper-cased verdict column, or an empty Series when there is nothing to read."""
    if gva is None or getattr(gva, "empty", True) or "verdict" not in gva.columns:
        return pd.Series(dtype=str)
    return gva["verdict"].astype(str).str.strip().str.upper()


def _norm_period(s) -> str:
    """'Q1 FY27' / 'q1fy27' -> 'Q1FY27'. Periods arrive spelled both ways."""
    return "".join(str(s or "").split()).upper()


def verdict_chip(gva, period: str = "", compact: bool = False) -> str:
    """Block A's chip, measured against the company's OWN prior guidance.

    Scoped to `period` when given — Block A is "the quarter in one line", so a chip
    summing every verdict on record would answer a different question than the block
    it sits in. Block D keeps the full history.

    Framework §Block A: never `INLINE` when there is simply no guidance, and never
    collapse a mix of outcomes to a single word — a company that beat on revenue and
    missed on margin has not "beaten". Both verdict vocabularies count.
    """
    v = _verdicts(gva)
    if period and not v.empty and "period" in gva.columns:
        want = _norm_period(period)
        v = v[gva["period"].map(_norm_period) == want]
    measured = v[v.isin(MEASURED_VERDICTS)] if not v.empty else v
    if measured.empty:
        # In the by-day table most holdings made no measurable promise for the quarter,
        # and 30-odd full chips cost ~4 KB against an 8 KB clip headroom. A dash says
        # the same thing there; the spelled-out chip stays on the per-company page.
        if compact:
            return "&mdash;"
        return (f"<span style='background:#eef1f3;color:{MUTED};border-radius:3px;"
                f"padding:2px 8px;font-size:12px;font-weight:600'>NO GUIDANCE</span>")
    counts = measured.value_counts()
    if len(counts) == 1:
        label, colour = counts.index[0], VERDICT_TONE.get(counts.index[0], MUTED)
    else:
        label = " &middot; ".join(f"{n} {k}" for k, n in counts.items())
        colour = "#34495e"
    return (f"<span style='background:{colour};color:#fff;border-radius:3px;"
            f"padding:2px 8px;font-size:12px;font-weight:600'>{label}</span>")


def _cred_pattern(raw) -> str:
    """Match a stored pattern to the fixed vocabulary. Gemini returns the label with
    trailing punctuation and sometimes a parenthetical rationale ('Conservative Bias
    (management consistently guided lower...)'), so match on the leading label rather
    than demanding an exact string. Anything unrecognised is dropped, not guessed."""
    s = str(raw or "").strip()
    for p in CRED_PATTERNS:
        if s.upper().startswith(p.upper()):
            return p
    return ""


def cred_score(gva) -> tuple[float | None, dict, str]:
    """(score, verdict counts, pattern) computed from the rows this page actually shows.

    The score is the average of the points each kept promise earns — EXCEEDED 5,
    DELIVERED 4, PARTIAL 3, MISSED 1 — over the promises that have been measured.
    TOO_EARLY and NA are excluded, because a promise whose deadline has not arrived is
    not a broken one; including them would drag every score toward zero as a company
    guided FURTHER ahead, which is backwards.

    Computed here rather than read from the stored `cred_score` column on purpose. The
    stored value is Gemini's own average over ONE concall's historical context, so it
    cannot be checked against anything on the page. This one is the arithmetic of the
    verdicts in the table above it, so the reader can count the rows and verify it.
    """
    if gva is None or getattr(gva, "empty", True) or "source" not in gva.columns:
        return None, {}, ""
    gf = gva[gva["source"].astype(str) == "gf_track"].copy()
    if gf.empty:
        return None, {}, ""
    v = _verdicts(gf)
    scored = v[v.isin(CRED_POINTS)]
    pattern = ""
    if "cred_pattern" in gf.columns:
        order = gf.sort_values("as_of", ascending=False) if "as_of" in gf.columns else gf
        for raw in order["cred_pattern"]:
            pattern = _cred_pattern(raw)
            if pattern and pattern != "Insufficient Data":
                break
    if scored.empty:
        return None, {}, pattern
    counts = scored.value_counts().to_dict()
    total = sum(CRED_POINTS[k] * n for k, n in counts.items())
    return total / len(scored), counts, pattern


def _cred_line(gva) -> str:
    """Credibility score with its arithmetic shown, so the number can be checked.

    Omitted entirely when nothing has been measured — a missing score is not a zero.
    """
    score, counts, pattern = cred_score(gva)
    if score is None:
        return ""
    # e.g. "7 DELIVERED x4 + 2 EXCEEDED x5"
    terms = " + ".join(f"{n} {k}&#215;{CRED_POINTS[k]}"
                       for k, n in sorted(counts.items(),
                                          key=lambda kv: -CRED_POINTS[kv[0]]))
    n = sum(counts.values())
    total = sum(CRED_POINTS[k] * c for k, c in counts.items())
    pat = f" &middot; <b>{esc(pattern, 30)}</b>" if pattern else ""
    return (f"<div style='margin:2px 0 10px;font-size:13px'>Management credibility "
            f"<b>{score:.1f} / 5</b>{pat}"
            f"<div style='color:{MUTED};font-size:11.5px;margin-top:2px'>"
            f"{terms} = {total} &divide; {n} promise{'' if n == 1 else 's'} measured "
            f"= <b>{score:.1f}</b>. Scale: EXCEEDED 5 &middot; DELIVERED 4 &middot; "
            f"PARTIAL 3 &middot; MISSED 1. TOO_EARLY and NA are excluded &mdash; a "
            f"promise whose deadline has not arrived is not a broken one."
            f"</div></div>")


def block_a(d: dict) -> str:
    A = d["A"]
    rows = []
    for label, key, dp, suf in (("Revenue", "revenue", 0, " Cr"), ("Net profit", "pat", 0, " Cr"),
                                ("EPS", "eps", 2, ""), ("OPM", "opm", 1, "%"),
                                ("NPM", "npm", 2, "%")):
        g = A[key]
        rows.append([
            f"<b>{label}</b>", fmt(g["cur"], dp, suf),
            f"<span style='color:{tone(g['yoy'])}'>{signed(g['yoy'])}</span>",
            f"<span style='color:{tone(g['qoq'])}'>{signed(g['qoq'])}</span>"])
    # The chip is measured against the company's OWN prior guidance — there is no
    # consensus feed in this repo and implying one would be fiction.
    chip = (f"<div style='margin:2px 0 8px'>"
            f"{verdict_chip(d.get('gva'), d.get('quarter', ''))} "
            f"<span style='color:{MUTED};font-size:11.5px'>vs what the company itself "
            f"guided for {esc(d.get('quarter'), 12)} &mdash; not consensus.</span></div>")
    return (_h("A &middot; The quarter in one line",
               f"{d['quarter']} ({d['period']}) &middot; YoY vs {d['yoy_period'] or 'n/a'}"
               f" &middot; QoQ vs {d['qoq_period']}. Margins in percentage points.")
            + chip
            + _tbl(["Metric", d["quarter"], "YoY", "QoQ"], rows))


def block_b(d: dict) -> str:
    B = d["B"]
    out = [_h("B &middot; Is this profit real?",
              "Reported profit stripped of the two easiest levers &mdash; the year-on-year "
              "swing in other income, and drift in the tax rate. Pure arithmetic off the "
              "quarterly P&amp;L; no model in this path.")]

    if B["adjusted_pat"].present:
        gap = B.get("adjusted_gap")
        oi_x, oi_e, tax_n = B["oi_excluded"], B["oi_effect"], B["tax_norm"]
        adj = B["adjusted_pat"]
        # The full bridge, line by line, so the number can be checked by hand.
        bridge = [
            ["Reported PAT", "", f"<b>{fmt(B['pat'], 1, '')}</b>",
             "as filed"],
            ["Less: rise in other income",
             f"&minus;{fmt(oi_x, 1, '')} pre-tax",
             f"<span style='color:{DOWN}'>&minus;{fmt(oi_e, 1, '')}</span>",
             esc(oi_e.note, 130)],
            ["Add: over/under-tax vs normal",
             f"{signed(B['tax_cur'])} vs {fmt(B['tax_mean'], 2, '%')} avg",
             (f"<span style='color:{UP}'>+{fmt(tax_n, 1, '')}</span>" if (tax_n.num or 0) >= 0
              else f"<span style='color:{DOWN}'>{fmt(tax_n, 1, '')}</span>"),
             esc(tax_n.note, 130)],
            ["<b>Adjusted PAT</b>", "",
             f"<b>{fmt(adj, 1, '')}</b>",
             f"<b>{signed(gap)}</b> vs reported"],
        ]
        out.append(_tbl(["Bridge", "Basis", "&#8377; Cr", "How it was worked out"],
                        bridge))
        out.append(
            f"<div style='color:{MUTED};font-size:11.5px;margin:-8px 0 14px'>"
            f"Only the <i>increase</i> in other income is removed, not the whole line — "
            f"a business is allowed its usual treasury income. The tax line adds back "
            f"what was over-taxed this quarter relative to its own recent average "
            f"(or subtracts what was under-taxed). Both adjustments are applied to "
            f"reported PAT, so the bridge ties out exactly.</div>")
    else:
        out.append(f"<div style='color:{MUTED};font-size:12.5px;margin:6px 0 12px'>"
                   f"Adjusted PAT not computable &mdash; "
                   f"{esc(B['adjusted_pat'].note, 90)}.</div>")

    rows = []
    ois, oiy = B.get("oi_share"), B.get("oi_share_yoy")
    if ois and ois.present:
        move = (f"{ois.num - oiy.num:+.2f} pp" if oiy and oiy.present else "&mdash;")
        flag = ("watch" if oiy and oiy.present and (ois.num - oiy.num) > 5 else "ok")
        rows.append(["Other income &divide; PBT", f"{ois.num:.2f}%",
                     f"{oiy.num:.2f}% (YoY)" if oiy and oiy.present else "&mdash;",
                     move, flag])
    tc, tm = B.get("tax_cur"), B.get("tax_mean")
    if tc and tc.present and tm and tm.present:
        rows.append(["Tax rate", f"{tc.num:.1f}%", f"{tm.num:.2f}% (prior 4Q)",
                     f"{tc.num - tm.num:+.2f} pp",
                     "headwind" if tc.num > tm.num else "tailwind"])
    for lbl, key in (("Interest", "interest"), ("Depreciation", "depreciation")):
        f_, ch = B.get(key), B.get(key + "_qoq")
        if f_ and f_.present:
            rows.append([lbl, fmt(f_, 0, " Cr"), "QoQ", signed(ch),
                         "step" if ch.present and ch.num and abs(ch.num) > 25 else "stable"])
    rows.append(["Gross margin vs opex split", "&mdash;", "&mdash;", "&mdash;",
                 "not computable"])
    out.append(_tbl(["Check", d["quarter"], "Comparison", "Move", "Read"], rows))
    out.append(f"<div style='color:{MUTED};font-size:11.5px'>"
               f"{esc(B['gross_vs_opex'].note, 140)}.</div>")
    return "".join(out)


def block_ladder(d: dict, stmts: pd.DataFrame) -> str:
    tbl = QT.quarterly_table_html(stmts, quarters=6)
    if not tbl:
        return ""
    return (_h("B &middot; Six quarters", "Revenue and profit in &#8377; Cr, EPS in &#8377;, "
               "margins in %. YoY = same quarter last year; QoQ = previous quarter.")
            + tbl)


def block_h1(d: dict, rich: bool) -> str:
    H1 = d["H1"]
    out = [_h("H1 &middot; Divergence engine",
              f"Annual data, {esc(H1.get('_fy') or 'latest FY', 12)} &mdash; stamped as such "
              f"because Screener publishes no quarterly balance sheet or cash flow. "
              f"Thresholds are the rules this repo already encodes.")]

    rows = []
    for metric, label, dp, unit in (
            ("cfo_pat_ratio", "Cash conversion (CFO &divide; PAT)", 2, "x"),
            ("net_debt_ebitda", "Leverage (debt &divide; op. profit)", 2, "x"),
            ("interest_coverage", "Interest coverage", 1, "x"),
            ("roce_pct", "ROCE", 1, "%"), ("roe_pct", "ROE", 1, "%"),
            ("rev_cagr_3y_pct", "Revenue CAGR (3y)", 1, "%"),
            ("receivable_days", "Receivable days", 0, ""),
            ("inventory_days", "Inventory days", 0, ""),
            ("wc_days", "Working-capital days", 0, ""),
            ("ccc_days", "Cash conversion cycle", 0, "")):
        f_ = H1.get(metric)
        if f_ is None:
            continue
        if f_.missing:
            rows.append([label, "&mdash;", "&mdash;",
                         f"<span style='color:{MUTED}'>not covered</span>", ""])
            continue
        verdict, col = _verdict(metric, f_)
        ser = H1.get(metric + "_series") or []
        spark = _spark(ser, colour=col if verdict else UP) if (rich and len(ser) > 1) else ""
        first = f"{ser[0][1]:,.{dp}f}" if ser else ""
        rows.append([label, f"<b>{f_.num:,.{dp}f}{unit}</b>",
                     f"{first} &rarr; {f_.num:,.{dp}f}" if ser else "&mdash;",
                     f"<span style='color:{col}'>{verdict or '&mdash;'}</span>", spark])
    out.append(_tbl(["Check", H1.get("_fy") or "latest", "Trend", "vs rule", ""], rows))

    trig = H1.get("agency_leverage_trigger")
    lev = H1.get("net_debt_ebitda")
    if trig is not None and trig.present and lev is not None and lev.present:
        head = trig.num - lev.num
        col = UP if head > 0 else DOWN
        out.append(
            f"<div style='border-left:3px solid {col};background:#f6f8f9;padding:10px 14px;"
            f"margin:8px 0 14px;font-size:13px'>"
            f"<b>Distance to the agency's own tripwire.</b> {esc(trig.note, 40)}: downgrade "
            f"above <b>{trig.num:g}x</b>. Measured leverage <b>{lev.num:g}x</b> &mdash; "
            f"<span style='color:{col}'>{abs(head):.2f}x of headroom</span>."
            f"<div style='color:{MUTED};font-size:11px;margin-top:5px'>"
            f"&ldquo;{esc(trig.evidence, 220)}&rdquo;</div></div>")

    bs = [(lbl, H1.get(lbl + "_series") or []) for lbl in
          ("Borrowings", "CWIP", "Fixed Assets", "Total Assets")]
    brows, span = [], None
    for lbl, ser in bs:
        if len(ser) < 2:
            continue
        # Header periods come from the first line that actually HAS a series — not from
        # bs[0], which is empty for companies with no Borrowings row.
        if span is None:
            span = (ser[0][0], ser[-1][0])
        chg = (ser[-1][1] / ser[0][1] - 1) * 100 if ser[0][1] else None
        brows.append([lbl, f"{ser[0][1]:,.0f}", f"{ser[-1][1]:,.0f}",
                      f"{chg:+.1f}%" if chg is not None else "&mdash;",
                      _spark(ser) if rich else ""])
    if brows and span:
        out.append("<div style='font-size:12px;color:#7a8791;margin:10px 0 3px'>"
                   "Balance sheet, annual</div>")
        out.append(_tbl(["Line", span[0], span[1], "Change", ""], brows))

    missing = [m for m in ("receivable_days", "inventory_days", "wc_days", "ccc_days")
               if H1.get(m) is not None and H1[m].missing]
    if missing:
        out.append(
            f"<div style='border:1px dashed #ccc;padding:10px 14px;font-size:12.5px;"
            f"color:#555;margin:6px 0'><b>Not yet available for this company:</b> "
            f"{esc(', '.join(missing), 200)}. These come from Screener's #ratios section; "
            f"re-run ingest_fundamentals for this symbol to populate them.</div>")
    return "".join(out)


def block_h2(d: dict) -> str:
    reg = d["register"]
    out = [_h("H2 &middot; The register",
              "Merged from seven tables that already exist. No model was called.")]
    if reg is None or reg.empty:
        return "".join(out) + (
            f"<div style='color:{MUTED};font-size:13px'>No source had a row for this "
            f"company. That means <b>not processed</b>, not clean.</div>")

    adv, fav = RR.split(reg)
    live_adv = adv[~adv["stale"]]
    hl = RR.headline_risks(reg, n=8)
    if not hl.empty:
        out.append(_tbl(["Period", "Severity", "Type", "What"],
                        [[esc(r["period"], 12), r["severity"],
                          f"<code>{esc(r['kind'], 34)}</code>",
                          esc(r["detail"] or r["label"], 200)]
                         for _, r in hl.iterrows()]))
    if not fav.empty:
        out.append("<div style='font-size:12px;color:#7a8791;margin:10px 0 3px'>"
                   "Favourable</div>")
        out.append(_tbl(["Period", "Type", "What"],
                        [[esc(r["period"], 12), f"<code>{esc(r['kind'], 26)}</code>",
                          esc(r["detail"] or r["label"], 200)]
                         for _, r in fav.head(6).iterrows()]))
    n_stale = int(reg["stale"].sum())
    out.append(f"<div style='color:{MUTED};font-size:11.5px;margin-top:6px'>"
               f"{len(reg)} rows &middot; {len(live_adv)} live adverse &middot; "
               f"{len(fav)} favourable &middot; {n_stale} superseded or aged. "
               f"Annual-report flags are superseded by a newer report, not by the calendar. "
               f"<b>Absence of a flag is not a clean opinion</b> &mdash; it can simply mean "
               f"that document was never processed.</div>")

    arsc = d.get("ar_scorecard")
    if arsc is not None and not arsc.empty:
        try:
            L = ARS.latest(arsc)
        except Exception:
            L = arsc
        if not L.empty:
            r0 = L.iloc[0]
            if r0["confidence"] == "none":
                out.append(
                    f"<div style='border-left:3px solid {MUTED};background:#f6f8f9;"
                    f"padding:9px 13px;margin:10px 0;font-size:12.5px'>"
                    f"<b>AR forensic score: not assessed.</b> A score of "
                    f"{r0['overall_score']} was generated for {esc(r0['fy_year'], 10)}, but "
                    f"every dimension was DATA_MISSING &mdash; that is the absence of a "
                    f"verdict, not a verdict.</div>")
            else:
                out.append(f"<div style='font-size:12px;color:#7a8791;margin:10px 0 3px'>"
                           f"AR forensic score &mdash; {esc(r0['fy_year'], 10)} &middot; "
                           f"<b>{r0['overall_score']}/10 {esc(r0['risk_label'], 12)}</b> "
                           f"(confidence {r0['confidence']}, "
                           f"{int(r0['n_data_missing'])}/{int(r0['n_dims'])} unevidenced)"
                           f"</div>")
                out.append(_tbl(["Dimension", "Weight", "Score", "Justification"],
                                [[esc(x["dimension"], 30), f"{x['weight_pct']:.0f}%",
                                  f"{x['score']:.1f}", esc(x["justification"], 150)]
                                 for _, x in L.iterrows()]))
    return "".join(out)


def block_deck(d: dict) -> str:
    deck = d.get("deck") or {}
    m, df_, fl, qs = (deck.get(k) for k in ("metrics", "diff", "flags", "questions"))
    out = [_h("C &middot; E &middot; F &middot; What the deck is doing",
              "Every row below quotes the deck verbatim; a figure that does not appear in "
              "its own quote is dropped before it reaches this page.")]
    any_rows = False

    if m is not None and not m.empty:
        any_rows = True
        out.append("<div style='font-size:12px;color:#7a8791;margin:8px 0 3px'>"
                   "C &middot; operating engine</div>")
        out.append(_tbl(["Category", "Metric", "Value", "Period", "Evidence"],
                        [[esc(r["category"], 16), esc(r["metric"], 40),
                          f"{r['value']:,.2f} {esc(r['unit'], 8)}"
                          if pd.notna(r["value"]) else "&mdash;",
                          esc(r["period"], 12),
                          f"<i>&ldquo;{esc(r['evidence'], 150)}&rdquo;</i>"]
                         for _, r in m.head(14).iterrows()]))
    if df_ is not None and not df_.empty:
        any_rows = True
        out.append("<div style='font-size:12px;color:#7a8791;margin:8px 0 3px'>"
                   "E &middot; changed since the last deck</div>")
        out.append(_tbl(["Change", "Item", "Was", "Now", "Severity"],
                        [[f"<code>{esc(r['change_type'], 22)}</code>", esc(r["item"], 60),
                          esc(r["prior_state"], 80), esc(r["current_state"], 80),
                          r["severity"]] for _, r in df_.head(10).iterrows()]))
    if fl is not None and not fl.empty:
        any_rows = True
        out.append("<div style='font-size:12px;color:#7a8791;margin:8px 0 3px'>"
                   "F &middot; framing</div>")
        out.append(_tbl(["Flag", "Slide", "Severity", "Evidence"],
                        [[f"<code>{esc(r['flag_type'], 26)}</code>", esc(r["slide_ref"], 16),
                          r["severity"], f"<i>&ldquo;{esc(r['evidence'], 170)}&rdquo;</i>"]
                         for _, r in fl.head(8).iterrows()]))
    if qs is not None and not qs.empty:
        any_rows = True
        out.append(_h("G &middot; Questions this quarter leaves open"))
        out.append("<ul style='font-size:13px;color:#333;margin:4px 0;padding-left:20px'>"
                   + "".join(f"<li>{esc(q, 240)}</li>" for q in qs["question"].head(5))
                   + "</ul>")

    if not any_rows:
        out.append(
            f"<div style='border:1px dashed #ccc;padding:11px 14px;font-size:12.5px;"
            f"color:#555'>No deck teardown stored for this quarter. The pass runs at "
            f"ingest only &mdash; source PDFs are deleted two days after processing, so a "
            f"deck processed before this shipped cannot be re-read. Run "
            f"<code>extract_presentation.py --teardown</code> and this fills in from the "
            f"next deck onward.</div>")
    return "".join(out)


def block_d(d: dict) -> str:
    gva = d.get("gva")
    out = [_h("D &middot; Said versus delivered")]
    if gva is None or gva.empty:
        return "".join(out) + (f"<div style='color:{MUTED};font-size:13px'>No guidance "
                               f"rows for this company.</div>")
    g = gva.copy()
    g["_v"] = _verdicts(g)
    g["_measured"] = g["_v"].isin(MEASURED_VERDICTS)
    # Sort, never filter. Both verdict vocabularies live in this one table, so filtering
    # on either spelling drops the other side's rows entirely; ordering shows the
    # answered promises first and keeps the open ones visible underneath.
    show = g.sort_values("_measured", ascending=False).head(10)
    n_measured = int(g["_measured"].sum())

    out.append(_cred_line(g))
    rows = []
    for _, r in show.iterrows():
        v = r["_v"]
        colour = VERDICT_TONE.get(v, MUTED)
        src = str(r.get("source") or "").replace("pead_", "").replace("_", " ") or "&mdash;"
        rows.append([
            esc(r.get("period"), 12), esc(r.get("metric"), 26),
            esc(r.get("guided"), 30), esc(r.get("actual"), 30) or "&mdash;",
            f"<span style='color:{colour};font-weight:600'>{esc(v, 12)}</span>",
            f"<span style='color:{MUTED};font-size:11.5px'>{esc(src, 20)}</span>"])
    out.append(_tbl(["Period", "Metric", "Guided", "Actual", "Verdict", "Source"], rows))

    n_open = len(g) - n_measured
    if n_measured == 0:
        out.append(f"<div style='color:{MUTED};font-size:12px'>"
                   f"{len(g)} guidance rows exist and none has been joined back to a "
                   f"reported outcome &mdash; every one resolves to TOO_EARLY or NA. Shown "
                   f"as unmeasured rather than filled with a guess.</div>")
    elif n_open:
        out.append(f"<div style='color:{MUTED};font-size:12px'>"
                   f"{n_measured} promise{'' if n_measured == 1 else 's'} measured against "
                   f"a reported outcome; {n_open} still open (TOO_EARLY or NA). "
                   f"<b>Source</b> is which document the promise came from &mdash; "
                   f"gf_track rows are read from the concall transcript.</div>")
    return "".join(out)


# --------------------------------------------------------------------------- #
# page assembly
# --------------------------------------------------------------------------- #

_WRAP = "font-family:Arial,Helvetica,sans-serif;font-size:13px;color:#222"


def render(d: dict, stmts: pd.DataFrame, rich: bool = True) -> str:
    if not d.get("quarter"):
        return ""
    head = (f"<h2 style='margin:0 0 2px'>&#128200; {esc(d['name'], 70)} "
            f"<span style='color:#888;font-weight:400'>&middot; {esc(d['symbol'], 20)}"
            f"</span></h2>"
            f"<div style='color:#888;font-size:12px;margin:0 0 10px'>Quarterly teardown "
            f"&middot; <b>{d['quarter']}</b> ({d['period']}) &middot; ISIN "
            f"{esc(d['isin'], 16)} &middot; generated "
            f"{datetime.now().strftime('%d %b %Y')}</div>")
    body = (block_a(d) + block_b(d) + block_ladder(d, stmts) + block_h1(d, rich)
            + block_h2(d) + block_deck(d) + block_d(d))
    foot = (
        f"<div style='margin-top:22px;padding-top:12px;border-top:1px solid #ddd;"
        f"font-size:11.5px;color:{MUTED};line-height:1.6'>"
        f"<b>Grain.</b> Blocks A, B and the ladder are quarterly. Every H1 rail and "
        f"balance-sheet row is annual &mdash; Screener publishes no quarterly balance sheet "
        f"or cash flow. Annual-report flags carry their own FY. Nothing here compares "
        f"across grains.<br>"
        f"<b>Blind spots.</b> Going-concern language is captured by no prompt in this repo. "
        f"Rating actions recognise only Upgrade / Downgrade / Reaffirmed &mdash; never "
        f"Withdrawn, Suspended or Rating Watch.<br>"
        f"Signals only &mdash; human-in-the-loop. Not investment advice.</div>")
    return f"<div style='{_WRAP}'>{head}{body}{foot}</div>"


def render_by_day(pages, reported: dict, quarter: str, pf_total: int) -> str:
    """Season catch-up: one mail, grouped by reporting day, newest day first.

    A full teardown per company would be ~35 KB each and clip Gmail after two, so each
    company is a compact row and the day is the unit of narrative. The per-company
    detail lives in the local HTML files.
    """
    from collections import defaultdict
    by_day = defaultdict(list)
    for sym, d, _html in pages:
        when, src = reported.get(d["isin"], ("", ""))
        by_day[(when or "")[:10]].append((sym, d, src))

    n_clean, n_dirty, flagged = 0, 0, 0
    for _sym, d, _html in pages:
        g = d["B"].get("adjusted_gap")
        if g is not None and g.present and g.num is not None:
            if abs(g.num) <= 3:
                n_clean += 1
            else:
                n_dirty += 1
        reg = d.get("register")
        if reg is not None and not reg.empty and not RR.headline_risks(reg, n=1).empty:
            flagged += 1

    head = (
        f"<h2 style='margin:0 0 2px'>&#128200; {quarter} teardown &mdash; season to date</h2>"
        f"<div style='color:#888;font-size:12px;margin:0 0 10px'>"
        f"<b>{len(pages)} of {pf_total}</b> PF holdings have reported &middot; grouped by "
        f"the day their numbers landed &middot; generated "
        f"{datetime.now().strftime('%d %b %Y')}</div>"
        f"<div style='background:#f6f8f9;border:1px solid #e0e6ea;border-radius:6px;"
        f"padding:11px 14px;margin:0 0 14px;font-size:13px'>"
        f"<b>{n_clean}</b> reported profit that is essentially all operating "
        f"(adjusted PAT within 3% of reported) &middot; <b>{n_dirty}</b> where other income or the "
        f"tax rate moved the number materially &middot; <b>{flagged}</b> carry at least "
        f"one live adverse flag.</div>")

    out = [head]
    for day in sorted(by_day, reverse=True):
        rows = by_day[day]
        try:
            pretty = datetime.strptime(day, "%Y-%m-%d").strftime("%d %b %Y (%a)")
        except ValueError:
            pretty = day or "date unresolved"
        srcs = {s for _sym, _d, s in rows}
        note = "Screener" if srcs == {"screener"} else (
            "exchange filing" if srcs == {"filing"} else "mixed sources")
        out.append(_h(f"{pretty} &mdash; {len(rows)} compan"
                      f"{'y' if len(rows) == 1 else 'ies'}",
                      f"reporting date from {note}"))
        out.append(compact_table("".join(render_compact(d) for _sym, d, _s in rows)))

    out.append(
        f"<div style='margin-top:18px;padding-top:12px;border-top:1px solid #ddd;"
        f"font-size:11.5px;color:{MUTED};line-height:1.6'>"
        f"<b>Adjusted PAT</b> = reported PAT, <i>less</i> the post-tax effect of the "
        f"year-on-year INCREASE in other income (the usual level is left alone), "
        f"<i>plus</i> whatever was over-taxed this quarter against the prior-4-quarter "
        f"average rate. Both adjustments are applied to the reported figure, so the "
        f"bridge ties out exactly &mdash; the full line-by-line working is in block B of "
        f"each company's page. A wide gap is not itself bad; it means the headline was "
        f"moved by something other than operations.<br>"
        f"<b>CFO/PAT</b> is annual (latest FY), not quarterly &mdash; Screener publishes no "
        f"quarterly cash flow. Below 1.0 for two years running is the anchor forensic flag."
        f"<br>Signals only &mdash; human-in-the-loop. Not investment advice.</div>")
    return f"<div style='{_WRAP}'>{''.join(out)}</div>"


def _page_shell(title: str, inner: str) -> str:
    """Standalone-file chrome for --html. The block markup itself stays inline-styled so
    the same functions serve the mail; this adds only page-level typography, width and a
    dark ground, which email clients would strip anyway."""
    return f"""<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{esc(title, 90)}</title>
<style>
  :root {{ --ground:#eef1f3; --card:#ffffff; --edge:#d6dde2; }}
  @media (prefers-color-scheme: dark) {{
    :root {{ --ground:#0f151a; --card:#161f26; --edge:#2a3841; }}
    body {{ color-scheme: dark; }}
    .card {{ filter: invert(1) hue-rotate(180deg); }}
  }}
  * {{ box-sizing: border-box; }}
  body {{ margin:0; padding:24px 16px 64px; background:var(--ground);
         font-family: system-ui,-apple-system,"Segoe UI",Roboto,sans-serif; }}
  .card {{ max-width:1040px; margin:0 auto; background:var(--card);
          border:1px solid var(--edge); border-radius:10px; padding:28px;
          box-shadow:0 1px 2px rgba(20,29,36,.06), 0 8px 28px rgba(20,29,36,.06); }}
  table {{ max-width:100%; }}
  .card > div {{ overflow-x:auto; }}
  h2, h3 {{ text-wrap: balance; }}
  td, th {{ font-variant-numeric: tabular-nums; }}
</style>
<body><div class="card">{inner}</div></body></html>"""


def render_compact(d: dict) -> str:
    """One row per company: headline numbers, reported vs adjusted PAT, the anchor
    divergence, and the single most severe live flag. Roughly 1.7 KB — used when the mail
    would otherwise be clipped, so no holding disappears from the digest.

    Shows adjusted PAT as a FIGURE beside reported, not just a percentage gap: a bare
    "gap" column asks the reader to trust arithmetic they cannot see. The full bridge
    (what was excluded, at what tax rate) is in block B of the per-company page.
    """
    if not d.get("quarter"):
        return ""
    A, B, H1 = d["A"], d["B"], d["H1"]
    reg = d.get("register")
    top = ""
    if reg is not None and not reg.empty:
        hl = RR.headline_risks(reg, n=1)
        if not hl.empty:
            r = hl.iloc[0]
            top = f"{esc(r['kind'], 30)}: {esc(r['detail'] or r['label'], 90)}"
    cfo = H1.get("cfo_pat_ratio")
    cells = [
        f"<b>{esc(d['symbol'], 18)}</b>",
        fmt(A["revenue"]["cur"], 0), f"<span style='color:{tone(A['revenue']['yoy'])}'>"
                                     f"{signed(A['revenue']['yoy'])}</span>",
        fmt(A["pat"]["cur"], 0), f"<span style='color:{tone(A['pat']['yoy'])}'>"
                                 f"{signed(A['pat']['yoy'])}</span>",
        fmt(B.get("adjusted_pat"), 0),
        f"<span style='color:{tone(B.get('adjusted_gap'))}'>"
        f"{signed(B.get('adjusted_gap'))}</span>",
        verdict_chip(d.get("gva"), d.get("quarter", ""), compact=True),
        f"{cfo.num:.2f}x" if cfo is not None and cfo.present else "&mdash;",
        top or "&mdash;",
    ]
    return ("<tr>" + "".join(f"<td style='{_TD}'>{c}</td>" for c in cells) + "</tr>")


def compact_table(rows_html: str) -> str:
    heads = ["Company", "Revenue", "YoY", "Reported PAT", "YoY", "Adjusted PAT",
             "vs reported", "Guidance", "CFO/PAT", "Top live flag"]
    h = "".join(f"<th style='{_TH}'>{c}</th>" for c in heads)
    return (f"<table style='border-collapse:collapse;margin:6px 0 14px'>"
            f"<tr>{h}</tr>{rows_html}</table>")


# --------------------------------------------------------------------------- #
# loading
# --------------------------------------------------------------------------- #

LEDGER_NAME = "quarter_teardown_mailed.parquet"
LEDGER_COLS = ["season_quarter", "isin", "symbol", "quarter_label", "reported_on",
               "date_source", "mailed_at"]


def recent_reporters(drive, idx, pf, season: str, hours: float, log) -> list[tuple]:
    """PF holdings whose results for THIS season landed inside the window.

    Reuses pf_results_digest's date cascade rather than inventing a second one:
      screener  results.parquet.first_seen_at   (true timestamp)
      filing    processing_queue announcement_date (date only -> whole-day resolution)

    This returns DATE candidates only. It does NOT prove the company's numbers have
    landed — a filing can be dated before the quarterly_pl column flips. The no-mixing
    guard lives in the render loop, which checks the statements themselves and skips
    anything still sitting on an older quarter. Both are needed: this function alone
    once selected two holdings whose statements were still a quarter behind.
    """
    import pf_results_digest as PRD
    from _extractor_base import load_queue

    results = _read(drive, idx, "results.parquet")
    try:
        queue = load_queue(drive, idx)
    except Exception:
        queue = pd.DataFrame()

    screener_d = PRD._results_dates(results, season)
    queue_d = QT.queue_report_dates(queue, season)
    log(f"  report dates: screener={len(screener_d)} filing={len(queue_d)}")
    if not queue_d and queue is not None and not queue.empty:
        log("  WARNING: zero filing dates from a non-empty queue — check the "
            "announcement_date column")

    # exact timestamps where the screener scrape has them
    exact: dict[str, str] = {}
    if results is not None and not results.empty and "first_seen_at" in results.columns:
        r = results.copy()
        r["_q"] = r["latest_q"].map(lambda v: QT.norm_q(QT.qtr_label(v)))
        r = r[r["_q"] == QT.norm_q(season)]
        if not r.empty:
            exact = (r.groupby(r["isin"].astype(str).str.strip())["first_seen_at"]
                     .min().astype(str).to_dict())

    now = datetime.now()
    # hours=None -> no floor: every holding that has reported this season so far.
    floor_ts = (now - pd.Timedelta(hours=hours)) if hours is not None \
        else pd.Timestamp.min + pd.Timedelta(days=1)
    floor_day = floor_ts.date()
    out = []
    for isin, sym, name in pf:
        when, src = None, ""
        if isin in exact:
            try:
                when, src = pd.to_datetime(exact[isin]), "screener"
            except Exception:
                when = None
        if when is None and isin in screener_d:
            when, src = pd.to_datetime(screener_d[isin]), "screener"
        if when is None and isin in queue_d:
            when, src = pd.to_datetime(queue_d[isin]), "filing"
        if when is None:
            continue
        # day-granular sources compare on the date; timestamped ones on the instant
        inside = (when >= floor_ts) if src == "screener" and len(str(exact.get(isin, ""))) > 10 \
            else (when.date() >= floor_day)
        if inside:
            out.append((isin, sym, name, str(when)[:19], src))
    return out


def _write_and_send(args, drive, idx, season, pages, reported, body, subject, log) -> None:
    """Write the preview, then send if this environment can. Shared by the day-grouped
    and per-company mail paths so the ledger/toggle/credential rules cannot diverge."""
    prev = os.path.join(args.out_dir, "quarter_teardown_preview.html")
    with open(prev, "w", encoding="utf-8") as fh:
        fh.write(_page_shell(subject, body))
    log(f"  preview: {prev} ({len(body.encode()):,} bytes)")
    if len(body.encode()) > MAX_HTML_BYTES:
        log(f"  WARNING: body exceeds {MAX_HTML_BYTES:,} B — Gmail may clip it")
    if args.dry_run:
        log("  --dry-run: not sending")
        return
    from mailer import send_email, load_mail_settings
    if not load_mail_settings(drive, idx).get(MAIL_KEY, True):
        log(f"  mail toggle '{MAIL_KEY}' is OFF — not sending")
        return
    if not (os.getenv("GMAIL_USER") and os.getenv("GMAIL_APP_PASSWORD")
            and os.getenv("NOTIFY_EMAIL")):
        log("  mail NOT sent — GMAIL_USER / GMAIL_APP_PASSWORD / NOTIFY_EMAIL are not "
            "set in this environment. The preview above is the exact body that would "
            "go out. Add them to .env, or run this from CI.")
        return
    ok = send_email(subject, body)
    log(f"  mail sent={ok}  ({subject.encode('ascii', 'ignore').decode()})")
    if ok and reported:
        _stamp_ledger(drive, idx, season, pages, reported, log)


def _stamp_ledger(drive, idx, season, pages, reported, log) -> None:
    """Record what was mailed, AFTER a confirmed send."""
    from _extractor_base import load_parquet, save_parquet
    now = datetime.now().isoformat(timespec="seconds")
    rows = []
    for _sym, d, _h in pages:
        when, src = reported.get(d["isin"], ("", ""))
        rows.append({"season_quarter": season, "isin": d["isin"], "symbol": d["symbol"],
                     "quarter_label": d["quarter"], "reported_on": when,
                     "date_source": src, "mailed_at": now})
    if not rows:
        return
    led = load_parquet(drive, idx, LEDGER_NAME, LEDGER_COLS)
    new = pd.DataFrame(rows, columns=LEDGER_COLS)
    out = pd.concat([led, new], ignore_index=True) if led is not None and not led.empty else new
    out = out.drop_duplicates(subset=["season_quarter", "isin"], keep="last")
    save_parquet(drive, idx, LEDGER_NAME, out)
    log(f"  ledger: +{len(rows)} row(s) -> _index/{LEDGER_NAME} ({len(out)} total)")


def load_company(drive, idx, stmt_fid, repo_id, isin, symbol, name, tables) -> tuple:
    stmts = _read(drive, stmt_fid, f"{symbol}.parquet")
    register = RR.build_register(isin, symbol, **tables["register"])

    arsc = pd.DataFrame()
    try:
        from _extractor_base import get_or_create_subfolder, find_file, download_bytes
        cdir = get_or_create_subfolder(drive, repo_id, isin)
        fid = find_file(drive, cdir, "company_page.md")
        if fid:
            md = download_bytes(drive, fid).decode("utf-8", "ignore")
            arsc = ARS.parse_company_page(md, isin, symbol, name)
    except Exception:
        pass

    def sub(df):
        if df is None or df.empty or "isin" not in df.columns:
            return pd.DataFrame()
        return df[df["isin"].astype(str) == isin]

    gva = sub(tables.get("gva"))
    deck = {k: sub(tables["deck"].get(k)) for k in
            ("metrics", "diff", "flags", "questions")}
    # only the quarter being reported
    return stmts, register, arsc, gva, deck


# --------------------------------------------------------------------------- #
# self-test — no Drive, no Gemini
# --------------------------------------------------------------------------- #

def _gva(*rows) -> pd.DataFrame:
    """Synthetic guidance_vs_actual frame. Columns mirror build_pead_flags.GVA_COLS."""
    cols = ["isin", "symbol", "period", "metric", "guided", "actual", "delta",
            "verdict", "source", "cred_score", "cred_pattern", "as_of"]
    base = {c: None for c in cols}
    return pd.DataFrame([{**base, **r} for r in rows], columns=cols)


def _self_test() -> int:
    ok = fail = 0

    def check(name, cond):
        nonlocal ok, fail
        if cond:
            ok += 1
        else:
            fail += 1
            print(f"  FAIL {name}")

    pead = {"period": "FY26", "metric": "revenue", "guided": "~15%", "actual": "+21%",
            "verdict": "BEAT", "source": "pead_concall"}
    gf = {"period": "Q1FY27", "metric": "ebitda margin", "guided": "18%",
          "actual": "17.4%", "verdict": "DELIVERED", "source": "gf_track",
          "cred_score": 3.6, "cred_pattern": "Optimistic Bias", "as_of": "2026-08-01"}
    early = {"period": "FY28", "metric": "capacity", "guided": "150k units",
             "verdict": "TOO_EARLY", "source": "gf_track"}

    # THE REGRESSION: a gf_track verdict must survive alongside a pead row.
    mixed = block_d({"gva": _gva(pead, gf)})
    check("mixed: pead verdict rendered", "BEAT" in mixed)
    check("mixed: gf_track verdict rendered", "DELIVERED" in mixed)
    check("mixed: source column rendered", "gf track" in mixed)
    check("mixed: credibility line rendered", "4.0 / 5" in mixed)
    check("mixed: credibility pattern rendered", "Optimistic Bias" in mixed)

    gf_only = block_d({"gva": _gva(gf)})
    check("gf_track only: verdict rendered", "DELIVERED" in gf_only)

    unmeasured = block_d({"gva": _gva(early)})
    check("unmeasured: row still shown", "capacity" in unmeasured)
    check("unmeasured: explained as open", "TOO_EARLY" in unmeasured)
    check("unmeasured: no credibility score invented", "/ 5" not in unmeasured)

    check("empty gva -> message, no raise",
          "No guidance" in block_d({"gva": pd.DataFrame()}))
    check("missing gva -> message, no raise", "No guidance" in block_d({}))

    check("chip: unmeasured only -> NO GUIDANCE", "NO GUIDANCE" in verdict_chip(_gva(early)))
    check("chip: empty frame -> NO GUIDANCE", "NO GUIDANCE" in verdict_chip(pd.DataFrame()))
    check("chip: none -> NO GUIDANCE", "NO GUIDANCE" in verdict_chip(None))
    check("chip: single verdict", ">BEAT<" in verdict_chip(_gva(pead)))
    mix_chip = verdict_chip(_gva(pead, {**gf, "verdict": "MISSED"}))
    check("chip: mixed shows counts, not one word",
          "1 BEAT" in mix_chip and "1 MISSED" in mix_chip)
    check("chip: unmeasured rows ignored in counts",
          "TOO_EARLY" not in verdict_chip(_gva(pead, early)))

    # Block A's chip is scoped to the quarter being reported — an all-history tally
    # would answer a different question than the block it sits in.
    scoped = _gva(pead, gf)                      # pead=FY26, gf=Q1FY27
    check("chip: scoped to the reported quarter",
          ">DELIVERED<" in verdict_chip(scoped, "Q1 FY27"))
    check("chip: other periods excluded from scope",
          "BEAT" not in verdict_chip(scoped, "Q1 FY27"))
    check("chip: spacing in the period label ignored",
          verdict_chip(scoped, "Q1FY27") == verdict_chip(scoped, "Q1 FY27"))
    check("chip: quarter with no guidance -> NO GUIDANCE",
          "NO GUIDANCE" in verdict_chip(scoped, "Q3 FY27"))
    check("chip: no period -> full history",
          "1 BEAT" in verdict_chip(scoped) and "1 DELIVERED" in verdict_chip(scoped))
    check("chip: compact renders a dash, not a chip",
          verdict_chip(scoped, "Q3 FY27", compact=True) == "&mdash;")
    check("chip: compact still shows a real verdict",
          ">DELIVERED<" in verdict_chip(scoped, "Q1 FY27", compact=True))

    # credibility must never be invented from pead rows — BEAT/INLINE/MISS have no
    # defined points, so scoring them would put a number on the page no rule produced.
    check("no cred line from pead-only", _cred_line(_gva(pead)) == "")
    check("pead verdicts score nothing", cred_score(_gva(pead))[0] is None)

    # THE ARITHMETIC: 7 DELIVERED x4 + 2 EXCEEDED x5 = 38 / 9 = 4.222
    seven_d = [{**gf, "metric": f"m{i}"} for i in range(7)]
    two_e = [{**gf, "metric": f"e{i}", "verdict": "EXCEEDED"} for i in range(2)]
    s, counts, _p = cred_score(_gva(*seven_d, *two_e))
    check("score is the weighted average", round(s, 3) == round(38 / 9, 3))
    check("counts are the tally used", counts == {"DELIVERED": 7, "EXCEEDED": 2})
    line = _cred_line(_gva(*seven_d, *two_e))
    check("arithmetic is shown, not just the result", "= 38 &divide; 9" in line)
    check("both terms shown", "7 DELIVERED" in line and "2 EXCEEDED" in line)
    check("scale legend shown", "EXCEEDED 5" in line and "MISSED 1" in line)
    check("rounded score matches the arithmetic", "4.2 / 5" in line)

    # every point value, one promise each: (5+4+3+1)/4 = 3.25
    one_each = [{**gf, "metric": k, "verdict": k} for k in CRED_POINTS]
    check("all four verdict points score", cred_score(_gva(*one_each))[0] == 3.25)

    # TOO_EARLY / NA never dilute — guiding further ahead must not lower the score
    check("TOO_EARLY excluded from the average",
          cred_score(_gva(*seven_d, *two_e, early, early))[0] == cred_score(
              _gva(*seven_d, *two_e))[0])
    na_score = {**gf, "verdict": "NA", "cred_score": "NA", "cred_pattern": "NA"}
    check("NA excluded from the average",
          cred_score(_gva(*seven_d, na_score))[0] == 4.0)
    check("nothing measured -> no line at all", _cred_line(_gva(early, na_score)) == "")
    check("nothing measured -> no raise",
          isinstance(block_d({"gva": _gva(early, na_score)}), str))
    # the stored cred_score column is no longer trusted — it is Gemini's own per-concall
    # average and cannot be checked against the table, so a bogus one must not leak through
    check("stored cred_score is ignored",
          "9.9 / 5" not in _cred_line(_gva({**gf, "cred_score": 9.9})))

    # pattern vocabulary arrives with trailing punctuation and parentheticals
    check("pattern with trailing period matched",
          _cred_pattern("Insufficient Data.") == "Insufficient Data")
    check("pattern with parenthetical matched",
          _cred_pattern("Conservative Bias (guided lower than actual)") == "Conservative Bias")
    check("unknown pattern dropped, not guessed", _cred_pattern("Sandbagging") == "")
    check("empty pattern dropped", _cred_pattern(None) == "")
    check("a real pattern outranks 'Insufficient Data'",
          cred_score(_gva({**gf, "cred_pattern": "Insufficient Data",
                           "as_of": "2026-08-11"},
                          {**gf, "cred_pattern": "Erratic",
                           "as_of": "2025-01-01"}))[2] == "Erratic")

    print(f"\nquarter_teardown self-test: {ok} passed, {fail} failed")
    return 1 if fail else 0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--symbols", help="Comma-separated symbols.")
    ap.add_argument("--pf", action="store_true", help="All PF holdings.")
    ap.add_argument("--daily", action="store_true",
                    help="Daily run: PF holdings that reported in the last 24h, mailed. "
                         "Equivalent to --since-hours 24 --pf --mail.")
    ap.add_argument("--since-hours", type=float, default=None,
                    help="Only companies whose results landed within this window. "
                         "Screener-sourced dates are exact; filing-sourced dates are "
                         "day-granular, so those round to whole days.")
    ap.add_argument("--season-all", action="store_true",
                    help="ONE-OFF catch-up: every PF holding that has reported this "
                         "season so far, in one mail grouped by reporting day. Implies "
                         "--pf --mail --by-day --force.")
    ap.add_argument("--by-day", action="store_true",
                    help="Group the mail by reporting date, newest day first, with a "
                         "compact row per company.")
    ap.add_argument("--force", action="store_true",
                    help="Ignore the mailed-already ledger and rebuild everything.")
    ap.add_argument("--html", action="store_true", help="Write a rich local page.")
    ap.add_argument("--mail", action="store_true", help="Build (and send) the email.")
    ap.add_argument("--audit", action="store_true",
                    help="Write the fact-provenance sidecar JSON beside the page.")
    ap.add_argument("--dry-run", action="store_true", help="Never send; write previews.")
    ap.add_argument("--out-dir", default=".")
    ap.add_argument("--self-test", action="store_true",
                    help="Offline render checks — no Drive, no Gemini, no mail.")
    args = ap.parse_args()
    if args.self_test:
        sys.exit(_self_test())
    if args.daily:
        args.pf = True
        args.mail = True
        if args.since_hours is None:
            args.since_hours = 24.0
    if args.season_all:
        args.pf = True
        args.mail = True
        args.by_day = True
        args.force = True
        args.since_hours = None
    if not (args.html or args.mail):
        args.html = True

    from _extractor_base import get_drive, log
    drive = get_drive()
    root = os.environ["GDRIVE_FOLDER_ID"]
    idx = _folder(drive, "company_repo/_index")
    repo_id = _folder(drive, "company_repo")
    stmt_fid = _folder(drive, "fundamentals/statements")

    targets = []
    if args.pf:
        from daily_brief import load_pf
        targets = load_pf(drive, root, idx)
    if args.symbols:
        want = {s.strip().upper() for s in args.symbols.split(",") if s.strip()}
        from daily_brief import load_pf
        allpf = load_pf(drive, root, idx)
        targets = [t for t in allpf if t[1].upper() in want] or targets
        missing = want - {t[1].upper() for t in targets}
        if missing:
            log(f"Not in PF, skipped: {', '.join(sorted(missing))}")
    season = QT.season_quarter()
    reported = {}
    pf_all = list(targets)
    windowed = args.since_hours is not None or args.season_all
    if windowed:
        pf_n = len(targets)
        recent = recent_reporters(drive, idx, targets, season, args.since_hours, log)
        reported = {r[0]: (r[3], r[4]) for r in recent}
        if args.season_all:
            # Season catch-up selects on the STATEMENTS (the authoritative "has
            # reported" signal), not on having a resolvable filing date — three
            # holdings sat at the season quarter with no date in either source and
            # would otherwise have been dropped from a catch-up that claims to be
            # complete. They keep an empty date and render under "date unresolved".
            log(f"  season catch-up: checking all {pf_n} holdings against {season}")
        else:
            targets = [(i, s, n) for i, s, n, _w, _src in recent]
            log(f"  {len(targets)} of {pf_n} PF holding(s) have a {season} report date "
                f"in the last {args.since_hours:g}h")

        if not args.force:
            from _extractor_base import load_parquet
            led = load_parquet(drive, idx, LEDGER_NAME, LEDGER_COLS)
            done = set()
            if led is not None and not led.empty:
                done = {str(r["isin"]).strip() for _, r in
                        led[led["season_quarter"].astype(str) == season].iterrows()}
            before = len(targets)
            targets = [t for t in targets if t[0] not in done]
            if before != len(targets):
                log(f"  {before - len(targets)} already mailed this season — skipped "
                    f"(use --force to rebuild)")

    if not targets:
        if windowed:
            log("Nothing new to report in the window — no mail sent.")
        else:
            log("No targets. Use --symbols <SYM>, --pf, or --daily.")
        return

    log(f"Loading shared tables for {len(targets)} company(ies)...")
    tables = {
        "register": dict(
            ar_red_flags=_read(drive, idx, "ar_red_flags.parquet"),
            rating_concerns=_read(drive, idx, "rating_concerns.parquet"),
            rating_sensitivity=_read(drive, idx, "rating_sensitivity.parquet"),
            ratings=_read(drive, idx, "ratings.parquet"),
            gf4=_read(drive, idx, "gf4_quality_flags.parquet"),
            announcements=_read(drive, idx, "announcement_ledger.parquet"),
            fraud_tracker=_read(drive, idx, "fraud_tracker.parquet")),
        "derived": _read(drive, idx, "financials_derived.parquet"),
        "gva": _read(drive, idx, "guidance_vs_actual.parquet"),
        "deck": {"metrics": _read(drive, idx, "deck_metrics.parquet"),
                 "diff": _read(drive, idx, "deck_diff.parquet"),
                 "flags": _read(drive, idx, "deck_flags.parquet"),
                 "questions": _read(drive, idx, "deck_questions.parquet")},
    }

    os.makedirs(args.out_dir, exist_ok=True)
    pages = []
    for isin, symbol, name in targets:
        stmts, register, arsc, gva, deck = load_company(
            drive, idx, stmt_fid, repo_id, isin, symbol, name, tables)
        if stmts is None or stmts.empty:
            log(f"  {symbol}: no statements parquet — skipped")
            continue
        # NO QUARTER MIXING. A filing can be dated before the numbers land, so the
        # statements are the authority on whether this company has actually reported
        # the season quarter. Rendering an older quarter under a Q1 FY27 heading is
        # the one error this whole framework exists to prevent.
        if windowed:
            lbl = QT.latest_quarter_label(stmts)
            if lbl is None or QT.norm_q(lbl) != QT.norm_q(season):
                log(f"  {symbol}: filing dated {season} but statements still at "
                    f"{lbl or 'n/a'} — skipped until the numbers land")
                continue
        d = build_facts(isin, symbol, name, stmts, tables["derived"], gva, register,
                        arsc, deck)
        if not d.get("quarter"):
            log(f"  {symbol}: fewer than two quarters of P&L — skipped")
            continue
        html = render(d, stmts, rich=args.html)
        pages.append((symbol, d, html))
        log(f"  {symbol}: {d['quarter']} · {len(d['facts'])} facts · "
            f"{len(register)} register rows · {len(html):,} bytes")

        if args.html:
            path = os.path.join(args.out_dir, f"quarter_teardown_{symbol}.html")
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(_page_shell(f"{symbol} — {d['quarter']} teardown", html))
            log(f"     wrote {path}")
        if args.audit:
            apath = os.path.join(args.out_dir, f"quarter_teardown_{symbol}_audit.json")
            a = write_audit(apath, d["facts"], isin, symbol, d["quarter"])
            log(f"     audit: {a['n_facts']} facts, {a['n_missing']} missing, "
                f"origins={a['by_origin']}, join_violations={len(a['join_violations'])}")

    if args.mail and pages:
        qlabel = pages[0][1]["quarter"]
        subject = (f"📊 Quarterly teardown — {qlabel} · "
                   f"{len(pages)} compan{'y' if len(pages) == 1 else 'ies'}")
        if args.by_day:
            subject = (f"📊 {qlabel} teardown — {len(pages)} PF holdings reported "
                       f"so far")
            body = render_by_day(pages, reported, qlabel, len(pf_all))
            log(f"  day-grouped body: {len(body.encode()):,} B")
            _write_and_send(args, drive, idx, season, pages, reported, body, subject, log)
            return
        body = "<hr>".join(h for _s, _d, h in pages)
        if len(body.encode()) > MAX_HTML_BYTES:
            # Degrade gracefully: keep full teardowns while they fit, compact the rest.
            # Dropping companies entirely would silently hide a holding that reported.
            full, size, n_full = [], 0, 0
            for _s, _d, h in pages:
                if size + len(h.encode()) < MAX_HTML_BYTES * 0.6:
                    full.append(h); size += len(h.encode()); n_full += 1
                else:
                    break
            compact = pages[n_full:]
            rest = "".join(render_compact(d) for _s, d, _h in compact)
            body = "<hr>".join(full) + (
                ("<hr>" + _h("Remaining holdings &mdash; compact",
                             "Full teardowns for these are in the local HTML files; the "
                             "mail is capped so Gmail does not clip it.")
                 + compact_table(rest)) if rest else "")
            log(f"  over budget — {n_full} full + {len(compact)} compact "
                f"({len(body.encode()):,} B)")
        prev = os.path.join(args.out_dir, "quarter_teardown_preview.html")
        with open(prev, "w", encoding="utf-8") as fh:
            fh.write(body)
        log(f"  preview: {prev} ({len(body.encode()):,} bytes)")
        if args.dry_run:
            log("  --dry-run: not sending")
        else:
            from mailer import send_email, load_mail_settings
            if not load_mail_settings(drive, idx).get(MAIL_KEY, True):
                log(f"  mail toggle '{MAIL_KEY}' is OFF — not sending")
            elif not (os.getenv("GMAIL_USER") and os.getenv("GMAIL_APP_PASSWORD")
                      and os.getenv("NOTIFY_EMAIL")):
                log("  mail NOT sent — GMAIL_USER / GMAIL_APP_PASSWORD / NOTIFY_EMAIL "
                    "are not set in this environment. The preview above is the exact "
                    "body that would go out. Add them to .env, or run this from CI.")
            else:
                ok = send_email(subject, body)
                log(f"  mail sent={ok}  ({subject.encode('ascii', 'ignore').decode()})")
                # Ledger is stamped ONLY after a confirmed send — a failed or toggled-off
                # mail must not mark the work as delivered.
                if ok and reported:
                    _stamp_ledger(drive, idx, season, pages, reported, log)


if __name__ == "__main__":
    main()
