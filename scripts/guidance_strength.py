r"""
guidance_strength.py — clean + score management guidance into an ANNUAL growth rate.

Pure functions, stdlib + pandas only. No Drive, no network, no Gemini — safe to
import from the audit, the watchlist builder, the mails and the app. Offline
unit-testable:  python scripts/guidance_strength.py --self-test

WHY THIS EXISTS
---------------
`build_gallery.guidance_scores()` is the incumbent scorer and stays untouched
(it drives the LIVE gallery_guidance.html). Measured against live data on
2026-08-21 it produced these top entries, every one an artefact:

    ANUP        4344%   "4344.44%" revenue over 3Y   -> a 3-year TOTAL read as annual
    KRISHNADEF  2189%   "INR5,378.256 million"       -> read Rs 5,378 cr, true Rs 538 cr
    SOLEX       1506%   "INR 26,000 million"         -> read Rs 26,000 cr, true Rs 2,600 cr
    EMBDL       1111%   "ABS: INR 15,323 Cr (GDV)"   -> a property pipeline, not revenue

Root causes, all in the incumbent path:
  * `_parse_cr` has no million/lakh/currency handling and never checks for "cr"
  * a range with no "cr" midpoints the first two numbers ("Line 3: $60-80 million" -> 31.5)
  * qualitative text yields a number ("Positive in Q1 FY27" -> 1.0)
  * `_horizon_years` hardcodes 2026 and floors an elapsed FY at 0.5y, which SQUARES
    the implied rate via (t/b)**(1/0.5)
  * absolutes are annualised, percentages are not -> two units in one ranking
  * "operating profit" matches "profit" and is scored against the NET profit base

This module fixes each one and rejects rather than guesses. No cap and no base
floor are applied here (user decision 2026-08-21) — implausible numbers are
settled by guidance_validate.py against the transcript, not by an arbitrary
ceiling.

PIPELINE
--------
  S1 normalise_amount  magnitude + currency, explicitly. Never assume crore.
  S2 classify_cell     growth / absolute / level / qualitative
  S3 reject_reason     segment splits, non-revenue concepts, aspirations
  S4 implied_cagr      -> one ANNUAL % for revenue or PAT

QUARTER CONVENTION — RESULTS convention throughout (the quarter that just ENDED),
matching quarterly_table.season_quarter(). Never extract_concall._current_india_quarter(),
which is one quarter ahead and expresses a guidance HORIZON, not a source quarter.
"""
from __future__ import annotations

import argparse
import re
import sys

import pandas as pd

import guidance_progress as GP
import guidance_value as GV

# Revenue OR profit only; margin is a LEVEL and is excluded by the metric set
# itself (user decision 2026-08-21).
GROWTH_METRICS = frozenset({"revenue", "pat"})

# Concepts that are NOT revenue but get parked in the revenue row. The extractor
# already LABELS every one of them -- "ABS: INR 15,323 Cr (GDV)",
# "ABS: INR 200-250 cr (Order Book)" -- the incumbent scorer just ignores the
# qualifier. Measured live: ~11 cells. Fixed here, so no prompt change and the
# P0 Phase-2 concall path is not touched.
NON_REVENUE_QUALIFIERS = (
    "gdv", "order book", "orderbook", "pipeline", "capex", "aum",
    "market cap", "marketcap", "booking", "enquiry", "bid book",
)

# Management hedges. A target nobody committed to is not guidance.
ASPIRATION_WORDS = (
    "aspiration", "aspirational", "aspire", "try to", "trying to", "hope",
    "hopeful", "wish", "dream", "ambition", "all-time high", "all time high",
)

_RE_ABS_PREFIX = re.compile(r"^\s*(ABS|LVL)\s*:\s*", re.I)
_RE_DERIVED = re.compile(r"^\s*\(?\s*derived\s*:?\s*", re.I)
# "derived: INR 40 cr -> INR 350 cr" / "(derived: 645 -> 3420 cr)" -- the cell
# shows its own workings, so the exact CAGR is computable instead of assumed.
_RE_CUR_TOK = r"(?:INR|Rs\.?|₹|USD|US\$|\$)?\s*"
_RE_WORKINGS = re.compile(
    _RE_CUR_TOK + r"([\d,]+(?:\.\d+)?)\s*(?:cr|crores?)?\s*(?:->|→|to)\s*"
    + _RE_CUR_TOK + r"([\d,]+(?:\.\d+)?)\s*(?:cr|crores?)", re.I)
# A "Label: value" pair. Two or more of them means the cell is a SEGMENT split,
# not a company-level number.
_RE_SEGMENT = re.compile(r"[A-Za-z][A-Za-z0-9 &/()-]{1,28}\s*:\s*(?=[^\s])")
# ABS: / LVL: / derived: declare the KIND of a cell and appear inline as well as
# at the start. They must never be counted as segment labels.
_RE_TYPE_TOKEN = re.compile(r"\b(?:ABS|LVL|derived)\s*:\s*", re.I)
# Text that says the percentage is ALREADY a per-year rate. "40-50% YoY growth"
# for FY28 means 40-50% EACH year -- annualising it again would understate it
# (measured: PRIMECAB 40-50% -> a wrong 20.4%/yr).
_RE_ANNUAL_WORD = re.compile(
    r"\bcagr\b|\byoy\b|\by-o-y\b|\byear[- ]on[- ]year\b|\bper year\b"
    r"|\bper annum\b|\bp\.?a\.?\b|\bannual(?:ly|ised|ized)?\b", re.I)
_RE_GROWTH_WORD = re.compile(r"\bgrowth\b|\bcagr\b|\byoy\b|\by-o-y\b|\bgrow\b", re.I)
_RE_PCT = re.compile(r"%")

# "double-digit" / "triple-digit" describe the NUMBER OF DIGITS, not a multiple.
# Without this guard "triple-digit" parses as 3x = +200%/yr (measured live:
# MATRIMONY ranked at 200% off that word alone).
_RE_DIGIT_WORD = re.compile(r"\b(?:single|double|triple|quadruple)[-\s]*digit", re.I)
# A TOTAL multiple stated in the cell: "4x", "3x", "double revenues".
_RE_MULT_X = re.compile(r"(\d+(?:\.\d+)?)\s*x\b", re.I)
_MULT_WORDS = {"double": 2.0, "doubling": 2.0, "triple": 3.0, "tripling": 3.0,
               "quadruple": 4.0, "quadrupling": 4.0}
_RE_MULT_WORD = re.compile(r"\b(" + "|".join(_MULT_WORDS) + r")\b", re.I)
# A period the cell states for ITSELF, which overrides the column horizon:
# "3x in 5-7 years", "double in 5 years", "double revenues by 2030".
_RE_IN_YEARS = re.compile(
    r"\b(?:in|over|within)\s+(\d+)(?:\s*(?:[-–]|to)\s*(\d+))?\s*year", re.I)
_RE_BY_YEAR = re.compile(r"\bby\s+(?:FY)?(20\d{2}|\d{2})\b", re.I)

# Horizons whose stated percentage is ALREADY an annual rate.
ANNUAL_HORIZONS = frozenset({"NEXT_QTR", "1Y"})

# What a bare multi-year percentage MEANS when the cell does not say.
# Measured on the live table 2026-08-21: of 919 (isin, quarter, metric) groups
# carrying an ambiguous multi-year %, 660 (72%) repeat the SAME value across
# FY27/FY28/FY29 -- a company restating one per-year rate, not a growing
# cumulative total. Only 259 vary across horizons. So "annual" is the base rate,
# and a TOTAL reading is used only where there is positive evidence for it:
#   * the cell shows workings ("derived: 40 cr -> 350 cr")   -> exact
#   * the value CHANGES across horizons                       -> cumulative
# Defaulting the other way understated ~72% of multi-year cells (PARTH
# "20-30% growth" over FY29 collapsed to 7.7%/yr) and pushed real names off the
# list silently, where an overstatement surfaces at the top and is caught by
# guidance_validate against the transcript.
DEFAULT_PCT_BASIS = "annual"

# Floor for the absolute -> CAGR denominator. The base is a trailing 12 months
# and the target is a full fiscal year, so the two periods are at least one
# annual step apart even when only a few months of the target year remain.
# Without this, "Rs X by FY27" guided IN Q1FY27 divides by 0.75 and inflates.
MIN_ABSOLUTE_YEARS = 1.0

_NA = {"", "na", "n/a", "nan", "none", "-", "--"}


def _clean(raw) -> str:
    return str(raw or "").strip()


def _is_na(raw) -> bool:
    return _clean(raw).lower() in _NA


def _strip_prefix(text: str) -> str:
    """Drop a leading ABS:/LVL: type prefix -- it declares the KIND, and is not
    part of the value or a segment label."""
    return _RE_ABS_PREFIX.sub("", text).strip()


# --------------------------------------------------------------------------- #
#  S1 -- normalise the amount                                                   #
# --------------------------------------------------------------------------- #
def normalise_amount(raw, metric: str = "", horizon: str = "") -> dict | None:
    """Rs-crore value of an absolute target, or None when it is not an amount.

    Delegates to guidance_value.parse_guidance_value, which already handles
    magnitude (cr/lakh/mn/bn/k) and currency (USD -> INR at USDINR) correctly --
    the incumbent `_parse_cr` handles none of it.

    ONE deliberate override: a "Derived: <n> crores (based on <x>% margin ...)"
    cell. The typed parser reaches into the parenthetical and returns the margin
    (IRCON: 6.2 instead of 589.12), while the incumbent gets it right. So for
    Derived: cells the number BEFORE the parenthetical wins. The fix must not be
    one-sided -- both parsers err, in opposite directions.
    """
    text = _clean(raw)
    if _is_na(text):
        return None

    if _RE_DERIVED.match(text):
        head = text.split("(", 1)[0]
        parsed = GV.parse_guidance_value(head, metric=metric, horizon=horizon)
        if parsed.get("value_num_inr_cr") is not None:
            return {"inr_cr": float(parsed["value_num_inr_cr"]),
                    "value_type": parsed["value_type"],
                    "source": "derived_head"}

    parsed = GV.parse_guidance_value(text, metric=metric, horizon=horizon)
    cr = parsed.get("value_num_inr_cr")
    if cr is None:
        return None
    return {"inr_cr": float(cr), "value_type": parsed["value_type"],
            "source": "typed"}


# --------------------------------------------------------------------------- #
#  S2 -- classify the cell                                                      #
# --------------------------------------------------------------------------- #
def classify_cell(raw, metric: str = "", horizon: str = "") -> str:
    """'growth' | 'absolute' | 'level' | 'qualitative'.

    The one rule the typed parser lacks: a cell that says "40-45% YoY growth" IS
    growth, whatever its column header. guidance_value types a % outside
    {NEXT_QTR,1Y,2Y,3Y,3Y+} as a LEVEL ("FY-column level"), so 98 real percentage
    cells across 39 companies -- "32.99% CAGR", ">20% YoY growth" -- had cagr_pct
    nulled and were silently dropped by the incumbent scorer. Measured live
    2026-08-21.
    """
    raw_text = _clean(raw)
    if _is_na(raw_text):
        return "qualitative"

    # The extractor DECLARES the kind with an ABS:/LVL: prefix (concall_prompt
    # [MANDATORY CELL FORMAT]). Take it at its word rather than re-inferring.
    #
    # LVL: means a percentage that is a STATE, not a change -- "LVL: 50% (CDMO
    # share)", "LVL: >50% (Krystal revenue share)". Stripping the prefix and
    # re-typing left guidance_value seeing a bare % in a 1Y column, which it
    # correctly calls growth -- so a business-MIX share was ranking as 50%/yr
    # growth. Measured 2026-08-21: 204 of 224 LVL:-tagged revenue/PAT cells were
    # misclassified this way, 5 of them reaching the published watchlist.
    m_decl = _RE_ABS_PREFIX.match(raw_text)
    declared = m_decl.group(1).upper() if m_decl else ""
    if declared == "LVL":
        return "level"

    text = _strip_prefix(raw_text)
    has_pct = bool(_RE_PCT.search(text))
    says_growth = bool(_RE_GROWTH_WORD.search(text))

    # The override: explicit growth language + a percentage beats the header.
    if has_pct and says_growth:
        return "growth"

    # ABS: declares a LEVEL/amount. Honour it whenever a real amount parses;
    # when none does ("ABS: Double revenues by 2030") fall through so the
    # stated-multiple path can still read it.
    if declared == "ABS" and normalise_amount(raw_text, metric, horizon):
        return "absolute"

    parsed = GV.parse_guidance_value(text, metric=metric, horizon=horizon)
    vt = str(parsed.get("value_type") or "")
    if vt == "growth_pct":
        return "growth"
    if vt in ("absolute_inr", "absolute_usd"):
        return "absolute"
    if vt in ("margin_pct", "level_pct", "utilisation_pct", "capacity_pct"):
        # A bare % in an FY column with no growth word is genuinely ambiguous;
        # treat it as a level and let it drop rather than inflate the list.
        return "level"
    if vt == "multiple" and parsed.get("growth_pct") is not None:
        return "growth"
    return "qualitative"


# --------------------------------------------------------------------------- #
#  S3 -- reject what cannot be scored                                           #
# --------------------------------------------------------------------------- #
# Words that make a parenthetical a UNIT, a derivation or a note rather than a
# named business — these must not be read as a segment.
_RE_PAREN_SKIP = re.compile(
    r"^(?:derived|abs|lvl|or|page|plus|minus|yoy|y-o-y|cagr|approx|about|est"
    r"|annual|share|\d|>|<|~|\+|-|%)", re.I)
_RE_PAREN_TIME = re.compile(
    r"^(?:FY\s?\d{2,4}|Q[1-4]|H[12]|\d{4}|Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep"
    r"|Oct|Nov|Dec|India|Current|Total|Overall|Consolidated)\b", re.I)


def _paren_scope(text: str):
    """A proper noun inside a parenthetical that names part of the business."""
    for m in re.finditer(r"\(([^)]{2,40})\)", text):
        inner = re.sub(r"^derived\s*:\s*", "", m.group(1).strip(), flags=re.I)
        if _RE_PAREN_SKIP.match(inner):
            continue
        for w in re.findall(r"\b[A-Z][a-z]{3,}\b", inner):
            if not _RE_PAREN_TIME.match(w):
                return w
    return None


def reject_reason(raw, metric: str = "") -> str | None:
    """Why this cell must not be scored, or None when it is usable."""
    text = _clean(raw)
    if _is_na(text):
        return "empty"

    low = text.lower()

    # A concept that is not revenue, parked in the revenue row. The extractor
    # already names it in the cell -- we just have to read it.
    for q in NON_REVENUE_QUALIFIERS:
        if q in low:
            return "not_revenue:" + q

    for w in ASPIRATION_WORDS:
        if w in low:
            return "aspiration:" + w

    # Two or more "Label: value" pairs -> a segment split, not a company number.
    # "Electronics: 40% growth; Railway: 30-35% growth; CD: ..." is three
    # segments; scoring the max of them overstates the company.
    #
    # Strip EVERY type token first, not just a leading one: the extractor writes
    # them inline too, so "28% (derived: ABS: INR 367 cr -> ABS: INR 470 cr)"
    # counted three "labels" and was wrongly rejected -- it is a clean workings
    # row (RACLGEAR, SGIL, SAKSOFT among ~33 lost this way).
    body = _RE_TYPE_TOKEN.sub("", text)
    if len(_RE_SEGMENT.findall(body)) >= 2:
        return "segment_split"

    # A parenthetical naming a PRODUCT, DIVISION or GEOGRAPHY scopes the number
    # to part of the company: "100% (Oncology)", "100% (derived: Rajasthan QoQ)".
    # This is the half of segment detection the cell CAN answer; the other half
    # (Info Edge's bare "100%", scoped only in the quote) is handled at
    # validation time by guidance_validate.scope_is_segment.
    scoped = _paren_scope(text)
    if scoped:
        return "segment_scoped:" + scoped

    return None


# --------------------------------------------------------------------------- #
#  horizon -> TRUE forward distance in years                                    #
# --------------------------------------------------------------------------- #
def horizon_years(horizon_fy, guid_quarter) -> float | None:
    """Years from the END of `guid_quarter` to the END of the guided horizon.

    None when unresolvable OR already elapsed -- the row is then DROPPED, never
    defaulted to 1.0. Contrast build_gallery._horizon_years, which hardcodes
    `(2000+yr) - 2026` (so it silently drifts wrong from Apr-2027) and floors an
    elapsed FY at 0.5y -- and a 0.5y divisor SQUARES the implied growth.
    """
    win = GP.resolve_window(horizon_fy, guid_quarter)
    base = GP.parse_q(guid_quarter)
    if not win or not base:
        return None
    _start, end, _basis = win
    yrs = (end - GP.q_idx(*base)) / 4.0
    return yrs if yrs > 0 else None


def _workings(text: str):
    """(from, to) when the cell shows its own arithmetic, e.g.
    '775% (derived: INR 40 cr -> INR 350 cr)'. Exact beats assumed.

    Inline ABS:/LVL:/derived: tokens are stripped first -- the extractor writes
    '28% (derived: ABS: INR 367 cr -> ABS: INR 470 cr)', and the ABS: sitting
    between the arrow and the second number blocked the match, so the row fell
    through to the horizon rule and published 28%/yr instead of 15.2%/yr."""
    m = _RE_WORKINGS.search(_RE_TYPE_TOKEN.sub("", text))
    if not m:
        return None
    try:
        a = float(m.group(1).replace(",", ""))
        b = float(m.group(2).replace(",", ""))
    except (TypeError, ValueError):
        return None
    return (a, b) if a > 0 and b > 0 else None


# A target stated as ONE pot covering several years: "cumulative Rs.4000 Crores
# of revenue in FY2027 and FY2028 together". Treating it as a single-year target
# overstates badly -- ZENTEC read as 91%/yr on exactly this shape.
# RULE (user, 2026-08-22): split the pot EQUALLY across the years it covers, and
# measure growth to the FIRST of those years, because that is when the annual
# run-rate has to be reached.
_RE_CUMULATIVE = re.compile(
    r"\bcumulative(?:ly)?\b|\bcombined\b|\btogether\b|\bin aggregate\b"
    r"|\baggregate of\b|\bover the (?:next )?\w+ years? (?:combined|together)\b", re.I)
_RE_FY_TOKEN = re.compile(r"\bFY\s?(\d{2,4})\b", re.I)


def cumulative_span(text: str):
    """(n_years, earliest_fy) when the cell states one pot across several years.

    None when it is an ordinary single-period target.
    """
    if not _RE_CUMULATIVE.search(text):
        return None
    fys = sorted({int(m.group(1)) % 100 for m in _RE_FY_TOKEN.finditer(text)})
    if len(fys) >= 2:
        return (fys[-1] - fys[0] + 1, fys[0])
    m = re.search(r"\b(?:next\s+)?(\d+)\s*years?\b", text, re.I)
    if m:
        n = int(m.group(1))
        if 2 <= n <= 10:
            return (n, None)
    return None


def stated_multiple(text: str):
    """A TOTAL multiple the cell states outright ("4x", "double"), or None.

    "300% (Derived: 4x)" is 4x over the whole horizon -- publishing 300%/yr
    quadruples the company every year instead of once. "double-digit" is
    explicitly excluded: it counts digits, it is not a multiple.
    """
    if _RE_DIGIT_WORD.search(text):
        return None
    m = _RE_MULT_X.search(text)
    if m:
        try:
            v = float(m.group(1))
        except (TypeError, ValueError):
            return None
        return v if 1.0 < v <= 100.0 else None
    w = _RE_MULT_WORD.search(text)
    return _MULT_WORDS.get(w.group(1).lower()) if w else None


def stated_years(text: str, guid_quarter) -> float | None:
    """A period the cell names for itself, which beats the column horizon.

    "3x in 5-7 years" is 6 years, not the 3.75 the 3Y+ column implies; "double
    revenues by 2030" runs to the end of FY30.
    """
    m = _RE_IN_YEARS.search(text)
    if m:
        lo = float(m.group(1))
        hi = float(m.group(2)) if m.group(2) else lo
        yrs = (lo + hi) / 2.0
        return yrs if yrs > 0 else None
    m = _RE_BY_YEAR.search(text)
    if m:
        raw = m.group(1)
        fy = int(raw) % 100
        base = GP.parse_q(guid_quarter)
        if not base:
            return None
        # a calendar year "by 2030" lands in the FY ending Mar of that year
        yrs = (GP.q_idx(fy, 4) - GP.q_idx(*base)) / 4.0
        return yrs if yrs > 0 else None
    return None


def _annualise(total_pct: float, years: float) -> float:
    """A multi-year TOTAL -> the equivalent annual rate."""
    return ((1.0 + total_pct / 100.0) ** (1.0 / years) - 1.0) * 100.0


# --------------------------------------------------------------------------- #
#  S4 -- one annual growth rate                                                 #
# --------------------------------------------------------------------------- #
def infer_pct_basis(sub) -> str:
    """'annual' | 'total' for one (isin, quarter, metric) slice of guidance rows.

    The same percentage restated across FY27/FY28/FY29 is one per-year rate.
    A percentage that GROWS with the horizon is cumulative.
    """
    if sub is None or len(sub) < 2:
        return DEFAULT_PCT_BASIS
    vals = {str(v).strip() for v in sub.get("value", []) if str(v).strip()}
    return DEFAULT_PCT_BASIS if len(vals) <= 1 else "total"


def implied_cagr(value, metric, horizon_fy, guid_quarter,
                 base_cr=None, cagr_pct=None,
                 pct_basis=DEFAULT_PCT_BASIS) -> dict | None:
    """One guidance cell -> an ANNUAL growth %, or None when unusable.

    Returns {cagr_pct, kind, years, target_cr, base_cr, value_type, rule} on
    success, or {reject: <why>, raw} so the caller can LOG why a cell was
    dropped rather than have it vanish silently.

    BOTH paths are annualised (user decision 2026-08-21). The incumbent
    annualises absolutes but takes a percentage verbatim and labels it '%/yr',
    so BIRLANU's "775% (derived: INR 40 cr -> INR 350 cr)" over 3Y+ -- really
    ~106%/yr -- ranked as 775%/yr against properly-annualised peers.
    """
    text = _clean(value)
    why = reject_reason(text, metric)
    if why:
        return {"reject": why, "raw": text}

    met = GP.canon_metric(metric)
    if met not in GROWTH_METRICS:
        return {"reject": "metric_not_scored:" + str(met), "raw": text}

    years = horizon_years(horizon_fy, guid_quarter)
    if years is None:
        return {"reject": "horizon_unresolved:" + str(horizon_fy), "raw": text}

    kind = classify_cell(text, metric=metric, horizon=horizon_fy)
    hz = str(horizon_fy or "").strip().upper()

    if kind == "growth":
        # 0. the cell states a TOTAL multiple (and often its own period), which
        #    overrides both the stated % and the column horizon.
        mult = stated_multiple(text)
        if mult:
            yrs = stated_years(text, guid_quarter) or years
            return {"cagr_pct": (mult ** (1.0 / yrs) - 1.0) * 100.0,
                    "kind": "growth", "years": yrs, "target_cr": None,
                    "base_cr": None, "value_type": "growth_pct",
                    "rule": "stated_multiple", "raw": text}
        # 0b. a vague "double-digit / triple-digit" carries no usable number
        if _RE_DIGIT_WORD.search(text):
            return {"reject": "vague_digit_word", "raw": text}

        # 1. the cell shows its workings -> exact, no assumption
        w = _workings(text)
        if w:
            a, b = w
            return {"cagr_pct": ((b / a) ** (1.0 / years) - 1.0) * 100.0,
                    "kind": "growth", "years": years, "target_cr": b,
                    "base_cr": a, "value_type": "growth_pct",
                    "rule": "workings", "raw": text}

        parsed = GV.parse_guidance_value(text, metric=metric, horizon=horizon_fy)
        pct = parsed.get("growth_pct")
        if pct is None:
            pct = GV._pct_from_text(text)
        if pct is None and cagr_pct is not None and pd.notna(cagr_pct):
            pct = float(cagr_pct)
        if pct is None:
            return {"reject": "growth_pct_unparsed", "raw": text}
        pct = float(pct)

        # 2. the text says CAGR / YoY / p.a. -> already a per-year rate
        if _RE_ANNUAL_WORD.search(text):
            return {"cagr_pct": pct, "kind": "growth", "years": years,
                    "target_cr": None, "base_cr": None,
                    "value_type": "growth_pct", "rule": "stated_annual",
                    "raw": text}
        # 3. a near horizon -> already annual
        if hz in ANNUAL_HORIZONS:
            return {"cagr_pct": pct, "kind": "growth", "years": years,
                    "target_cr": None, "base_cr": None,
                    "value_type": "growth_pct", "rule": "annual_horizon",
                    "raw": text}
        # 4. multi-year, unlabelled -> basis decided by infer_pct_basis()
        if pct <= -100.0:
            return {"reject": "growth_pct_implausible", "raw": text}
        if pct_basis == "total":
            return {"cagr_pct": _annualise(pct, years), "kind": "growth",
                    "years": years, "target_cr": None, "base_cr": None,
                    "value_type": "growth_pct", "rule": "annualised_total",
                    "raw": text}
        return {"cagr_pct": pct, "kind": "growth", "years": years,
                "target_cr": None, "base_cr": None,
                "value_type": "growth_pct", "rule": "assumed_annual",
                "raw": text}

    if kind == "absolute":
        amt = normalise_amount(text, metric=metric, horizon=horizon_fy)
        if amt is None:
            return {"reject": "amount_unparsed", "raw": text}
        tgt = amt["inr_cr"]
        if GP.is_implausible_amount(tgt):
            return {"reject": "amount_implausible", "raw": text}
        if base_cr is None or not (base_cr > 0):
            return {"reject": "no_base", "raw": text}

        # A pot covering several years is split EQUALLY per year, and growth is
        # measured to the FIRST year of the span.
        rule = "absolute_to_cagr"
        span = cumulative_span(text)
        if span:
            n_years, first_fy = span
            tgt = tgt / float(n_years)
            rule = "cumulative_split_%dy" % n_years
            if first_fy is not None:
                # measure to the END of the span, exactly like every other
                # absolute row measures to the end of its stated horizon.
                # Measuring to the FIRST year instead annualises a one-year jump
                # over a 9-month window and explodes it (ZENTEC: 328%/yr).
                y = horizon_years("FY%02d" % (first_fy + n_years - 1),
                                  guid_quarter)
                if y is None:
                    return {"reject": "cumulative_horizon_elapsed", "raw": text}
                years = y

        # A target for the fiscal year we are ALREADY IN sits <1 year away, and
        # annualising over a fraction of a year inflates the rate for no real
        # reason -- the base (a trailing 12 months) and the target (a full FY)
        # are one annual period apart however few months remain on the clock.
        years = max(years, MIN_ABSOLUTE_YEARS)

        if tgt <= base_cr:
            return {"reject": "target_below_base", "raw": text}
        return {"cagr_pct": ((tgt / base_cr) ** (1.0 / years) - 1.0) * 100.0,
                "kind": "absolute", "years": years, "target_cr": tgt,
                "base_cr": base_cr, "value_type": amt["value_type"],
                "rule": rule, "raw": text}

    return {"reject": "kind_not_scored:" + kind, "raw": text}


def best_per_key(rows, base_rev, base_pat, key="isin", min_cagr=None):
    """({key: winner}, {key: [rejects]}, {key: [every scored cagr_pct]}).

    The third return is what lets a caller say "this company cleared the bar on
    THREE separate statements, not one" -- a single borderline cell is much
    weaker evidence than several consistent ones.

    Unlike build_gallery.guidance_scores, this returns the WINNING ROW -- the
    caller needs its metric, horizon, target and base for the watchlist columns
    and for validation, and re-deriving the argmax afterwards is fragile.

    Grouping is by ISIN, not symbol: a blank SME symbol would otherwise group
    every such company under "" and one symbol can span two ISINs.
    """
    best, rejects, scored = {}, {}, {}
    if rows is None or len(rows) == 0:
        return best, rejects, scored
    for k, sub in rows.groupby(rows[key].astype(str)):
        # basis is a property of the (quarter, metric) GROUP, not of one cell
        basis = {}
        for (q, m), gsub in sub.groupby([sub["quarter"].astype(str),
                                         sub["metric"].astype(str)]):
            basis[(q, m)] = infer_pct_basis(gsub)
        for _, r in sub.iterrows():
            met = GP.canon_metric(r.get("metric"))
            base = (base_rev if met == "revenue" else base_pat).get(k)
            out = implied_cagr(r.get("value"), r.get("metric"),
                               r.get("horizon_fy"), r.get("quarter"),
                               base_cr=base, cagr_pct=r.get("cagr_pct"),
                               pct_basis=basis.get(
                                   (str(r.get("quarter")), str(r.get("metric"))),
                                   DEFAULT_PCT_BASIS))
            if not out:
                continue
            common = {"metric": met, "horizon_fy": r.get("horizon_fy"),
                      "quarter": r.get("quarter"), "symbol": r.get("symbol"),
                      "company_name": r.get("company_name"),
                      "source_doc_id": r.get("source_doc_id")}
            if out.get("reject"):
                rejects.setdefault(k, []).append({**out, **common})
                continue
            out = {**out, **common}
            scored.setdefault(k, []).append(float(out["cagr_pct"]))
            if k not in best or out["cagr_pct"] > best[k]["cagr_pct"]:
                best[k] = out
    if min_cagr is not None:
        scored = {k: [v for v in vs if v >= min_cagr] for k, vs in scored.items()}
    return best, rejects, scored


# --------------------------------------------------------------------------- #
#  self-test — one fixture per defect confirmed on live data 2026-08-21          #
# --------------------------------------------------------------------------- #
def _self_test() -> int:  # noqa: C901
    fails = []

    def chk(label, got, want, tol=None):
        ok = (abs(got - want) <= tol) if (tol is not None and got is not None
                                          and isinstance(got, float)) else got == want
        if not ok:
            fails.append(f"{label}: got {got!r}, want {want!r}")

    # --- S1 magnitude + currency (the 10x / 88x live defects) ---------------
    chk("SOLEX million", normalise_amount("INR 26,000 million")["inr_cr"],
        2600.0, tol=1.0)
    chk("KRISHNADEF million",
        normalise_amount("INR5,378.256 million")["inr_cr"], 537.83, tol=0.5)
    chk("MOTHERSON usd bn",
        normalise_amount("USD 108 Billion")["inr_cr"], 950400.0, tol=10.0)
    # IRCON: the typed parser grabs 6.2 from the parenthetical; head wins.
    ircon = normalise_amount(
        "Derived: approx 589.12 crores (based on 6.1%-6.3% PAT margin on FY27 revenue)")
    chk("IRCON derived head", round(ircon["inr_cr"], 2), 589.12)

    # --- S3 rejects ---------------------------------------------------------
    chk("GDV rejected",
        (reject_reason("ABS: INR 15,323 Cr (GDV)") or "").startswith("not_revenue"), True)
    chk("order book rejected",
        (reject_reason("ABS: INR 200-250 cr (Order Book)") or "").startswith("not_revenue"),
        True)
    chk("aspiration rejected",
        (reject_reason("1000 crores (aspiration)") or "").startswith("aspiration"), True)
    chk("all-time-high rejected",
        (reject_reason("Will try to break all-time high records") or "")
        .startswith("aspiration"), True)
    chk("segment split rejected",
        reject_reason("Electronics: 40% growth; Railway: 30-35% growth; "
                      "CD: Industry volume 12-13% growth"), "segment_split")
    chk("plain absolute kept", reject_reason("Rs. 5,000.0 crore"), None)
    # a parenthetical naming a product / geography scopes the number
    chk("Oncology paren scoped",
        (reject_reason("100% (Oncology)") or "").startswith("segment_scoped"), True)
    chk("Rajasthan paren scoped",
        (reject_reason("100% (derived: Rajasthan QoQ)") or "")
        .startswith("segment_scoped"), True)
    chk("unit paren kept", reject_reason("60-70% (or ABS: INR 320-340 cr)"), None)
    chk("plus-minus paren kept", reject_reason("80% (plus-minus 5%)"), None)
    chk("YoY paren kept", reject_reason("50% (YoY)"), None)
    # inline ABS:/derived: tokens are TYPE markers, not segment labels
    chk("workings with inline ABS kept",
        reject_reason("28% (derived: ABS: INR 367 cr -> ABS: INR 470 cr)"), None)
    chk("workings scores",
        round(implied_cagr("28% (derived: ABS: INR 367 cr -> ABS: INR 470 cr)",
                           "revenue", "1Y", "Q1 FY27").get("cagr_pct"), 1), 15.2)
    chk("plain growth kept", reject_reason("15-20% growth"), None)

    # --- S2 the growth override (98 cells / 39 companies recovered) ---------
    chk("FY-col % growth -> growth",
        classify_cell("40-45% YoY growth", metric="revenue", horizon="FY28"), "growth")
    chk("FY-col CAGR -> growth",
        classify_cell("32.99% CAGR", metric="revenue", horizon="FY27"), "growth")
    chk("margin % stays level",
        classify_cell("23.5%", metric="margin", horizon="FY27"), "level")
    chk("qualitative stays qualitative",
        classify_cell("Positive in Q1 FY27", metric="pat", horizon="FY27"),
        "qualitative")
    # An LVL: prefix DECLARES a level -- a business-mix share must never rank as
    # growth just because it sits in a 1Y column.
    chk("LVL share is a level",
        classify_cell("LVL: 50% (CDMO share)", metric="revenue", horizon="1Y"),
        "level")
    chk("LVL bare pct is a level",
        classify_cell("LVL: 51%", metric="revenue", horizon="1Y"), "level")
    # Rejected twice over: the parenthetical names a brand AND the LVL: prefix
    # says it is a level. reject_reason runs first, so the scope reason wins --
    # either way the row never scores.
    chk("LVL never scores",
        bool(implied_cagr("LVL: >50% (Krystal revenue share)", "revenue", "1Y",
                          "Q1 FY27").get("reject")), True)
    chk("LVL with no brand still a level",
        implied_cagr("LVL: 51%", "revenue", "1Y", "Q1 FY27").get("reject"),
        "kind_not_scored:level")
    chk("ABS with an amount is absolute",
        classify_cell("ABS: INR 900 cr", metric="pat", horizon="1Y"), "absolute")
    # ABS: with no parseable amount must still reach the stated-multiple path
    chk("ABS double still growth",
        classify_cell("ABS: Double revenues by 2030", metric="revenue",
                      horizon="3Y+"), "growth")

    # --- horizon: true distance, elapsed -> None ----------------------------
    chk("FY27 from Q1FY26", horizon_years("FY27", "Q1 FY26"), 1.75)
    chk("FY27 from Q1FY27", horizon_years("FY27", "Q1 FY27"), 0.75)
    chk("elapsed FY26 from Q1FY27", horizon_years("FY26", "Q1 FY27"), None)
    chk("unresolvable", horizon_years("", "Q1 FY27"), None)

    # --- S4 annualisation (BIRLANU: 775% over 3Y+ is ~106%/yr) --------------
    b = implied_cagr("775% (derived: INR 40 cr -> INR 350 cr)", "revenue",
                     "3Y+", "Q2 FY18")
    # 40 -> 350 cr is 8.75x. '3Y+' guided in Q2FY18 resolves to the END of FY21,
    # i.e. 3.5 years out -- so ~85.8%/yr, NOT the stored 775%.
    chk("BIRLANU rule", b.get("rule"), "workings")
    chk("BIRLANU annualised", b.get("cagr_pct"), 85.8, tol=1.0)
    a = implied_cagr("4344.44%", "revenue", "3Y", "Q4 FY26", pct_basis="total")
    chk("ANUP annualised below raw", a.get("cagr_pct") < 4344.44, True)
    chk("ANUP rule", a.get("rule"), "annualised_total")
    # A bare multi-year % defaults to per-year (72% of live cases restate one
    # rate across horizons); only a VARYING series is read as cumulative.
    d = implied_cagr("20-30% growth", "revenue", "FY29", "Q4 FY26")
    chk("bare multi-year defaults annual", round(d.get("cagr_pct"), 1), 25.0)
    chk("bare multi-year rule", d.get("rule"), "assumed_annual")
    import pandas as _pd
    same = _pd.DataFrame({"value": ["20-30% growth", "20-30% growth"]})
    diff = _pd.DataFrame({"value": ["100% growth", "300% growth"]})
    chk("repeated -> annual", infer_pct_basis(same), "annual")
    chk("varying -> total", infer_pct_basis(diff), "total")

    # --- cells that state their own multiple / period -----------------------
    m = implied_cagr("300% (Derived: 4x)", "revenue", "3Y+", "Q1 FY27")
    chk("4x rule", m.get("rule"), "stated_multiple")
    chk("4x annualised", round(m.get("cagr_pct")) < 60, True)
    t = implied_cagr("200% (derived: 3x in 5-7 years)", "revenue", "3Y+", "Q1 FY27")
    chk("explicit period used", t.get("years"), 6.0)
    chk("3x over 6y", round(t.get("cagr_pct"), 1), 20.1)
    dbl = implied_cagr("100% (derived: double in 5 years)", "revenue", "3Y+",
                       "Q1 FY27")
    chk("double in 5y", round(dbl.get("cagr_pct"), 1), 14.9)
    by = implied_cagr("ABS: Double revenues by 2030", "revenue", "3Y+", "Q1 FY27")
    chk("by-2030 rule", by.get("rule"), "stated_multiple")
    chk("by-2030 years", by.get("years"), 3.75)
    # "triple-digit" counts digits -- it is NOT 3x
    td = implied_cagr("triple-digit", "pat", "NEXT_QTR", "Q1 FY27")
    chk("triple-digit rejected", td.get("reject"), "vague_digit_word")
    chk("double-digit not a multiple", stated_multiple("double-digit growth"), None)
    c = implied_cagr("21.75% CAGR", "revenue", "3Y", "Q4 FY26")
    chk("stated CAGR untouched", round(c.get("cagr_pct"), 2), 21.75)
    chk("stated CAGR rule", c.get("rule"), "stated_annual")
    # "40-50% YoY growth" means 40-50% EACH year -- annualising again is wrong.
    y = implied_cagr("40-50% YoY growth", "revenue", "FY28", "Q4 FY26")
    chk("YoY treated as annual", round(y.get("cagr_pct"), 1), 45.0)
    chk("YoY rule", y.get("rule"), "stated_annual")
    n = implied_cagr("15% growth YoY", "revenue", "1Y", "Q1 FY27")
    chk("1Y pct untouched", round(n.get("cagr_pct"), 2), 15.0)

    # --- S4 absolute path ----------------------------------------------------
    ab = implied_cagr("Rs. 5,000.0 crore", "revenue", "FY28", "Q1 FY27",
                      base_cr=2000.0)
    # 2000 -> 5000 cr is 2.5x. FY28 guided in Q1FY27 ends 1.75y out -> ~68.8%/yr.
    chk("absolute rule", ab.get("rule"), "absolute_to_cagr")
    chk("absolute cagr", ab.get("cagr_pct"), 68.8, tol=1.0)
    # --- cumulative multi-year pot, split equally per year (ZENTEC) ---------
    # sub-year horizons must not inflate: FY27 guided in Q1FY27 is 0.75y away
    # but base and target are still one annual period apart
    sub = implied_cagr("ABS: INR 2000 cr", "revenue", "FY27", "Q1 FY27",
                       base_cr=1000.0)
    chk("sub-year floored to 1y", sub.get("years"), 1.0)
    chk("sub-year gives the plain doubling", round(sub.get("cagr_pct")), 100)

    chk("cumulative span detected",
        cumulative_span("we have a target of cumulative Rs.4000 Crores of "
                        "revenue in FY2027 and FY2028 together"), (2, 27))
    chk("ordinary target is not cumulative",
        cumulative_span("So we expect INR7,500 crores in FY28."), None)
    z = implied_cagr("ABS: INR 4000 cr cumulative in FY2027 and FY2028 together",
                     "revenue", "2Y", "Q1 FY27", base_cr=1000.0)
    chk("cumulative rule fires", z.get("rule"), "cumulative_split_2y")
    chk("pot split equally per year", z.get("target_cr"), 2000.0)
    # measured to the FIRST year of the span (FY27 = 0.75y out from Q1FY27)
    chk("horizon is the END of the span", z.get("years"), 1.75)
    # NOTE: the split does NOT always lower the number. Halving the pot lowers
    # the target, but measuring to the FIRST year of the span also shortens the
    # horizon, and the horizon usually dominates. What the rule guarantees is
    # that the figure means "revenue IN one year", not "a pot spread over N".
    chk("cumulative target is per-year, not the pot",
        z["target_cr"] * 2 == 4000.0, True)

    chk("no base -> reject",
        implied_cagr("Rs. 5,000.0 crore", "revenue", "FY28", "Q1 FY27").get("reject"),
        "no_base")
    chk("target below base -> reject",
        implied_cagr("Rs. 100 crore", "revenue", "FY28", "Q1 FY27",
                     base_cr=2000.0).get("reject"), "target_below_base")

    # --- metric routing: ebitda must NEVER reach the PAT base ---------------
    chk("operating profit not scored",
        (implied_cagr("Rs. 900 crore", "operating profit", "1Y", "Q1 FY27",
                      base_cr=100.0).get("reject") or "")
        .startswith("metric_not_scored"), True)
    chk("margin not scored",
        (implied_cagr("23.5%", "margin", "1Y", "Q1 FY27").get("reject") or "")
        .startswith("metric_not_scored"), True)
    chk("canon pat scored", GP.canon_metric("pat") in GROWTH_METRICS, True)

    for f in fails:
        print("FAIL " + f)
    print(("self-test FAILED (%d)" % len(fails)) if fails else "self-test OK")
    return 1 if fails else 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--self-test", action="store_true",
                    help="run the offline fixtures; no Drive, no network")
    args = ap.parse_args()
    if args.self_test:
        return _self_test()
    ap.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
