r"""
guidance_value.py — ONE typed parser for management-guidance cells (user 2026-07-18).

WHY THIS EXISTS
    extract_concall wrote every "growth guidance" cell as `unit="%"` +
    `cagr_pct=float(raw)`, with no check on what the cell actually held. Two
    failures followed, both visible in the mails:

      (a) FABRICATED growth — an absolute target parked in a growth column became
          a percentage: capacity "178,000" -> 178000%, revenue "1775" (a Rs-cr
          target) -> 1775%. The digest mail takes the MAX per company and sorts by
          it, so these floated to the top and drove the "high-growth" count.
      (b) LOST guidance (the bigger loss, ~2.7k rows) — float() fails on any text,
          so "19% - 26%", "32.5% CAGR", "106.33% (Derived)" and every range /
          multiple silently became NULL.

    A magnitude band cannot fix either: it deletes (a) and recovers none of (b),
    and it still cannot tell a MARGIN level (16.5%) from GROWTH (16.5%) — they are
    the same number with different meaning.

WHAT IT DOES
    parse_guidance_value() classifies a raw cell into an explicit `value_type` from
    the cell's own text plus the column it came from, and returns the number in its
    natural unit. `growth_pct` is returned ONLY when the cell really is a growth %,
    so `cagr_pct` finally means one thing everywhere.

    growth_pct        "32.5% CAGR", "19%-26%" (midpoint)   -> pct,   unit "%"
    margin_pct        metric=margin                        -> level, unit "%"
    utilisation_pct   explicit utilisation/run-rate wording-> level, unit "%"
    capacity_pct      any capacity % (addition OR utilisation — the tables do not
                      distinguish them, so it never feeds growth)   unit "%"
    absolute_inr      "INR530 crores", "26,000 million"    -> Rs cr, unit "INR_cr"
    absolute_units    "115,000 MTPA", "60k sites"          -> qty,   unit as written
    multiple          "double", "3x (Tripling)"            -> 2.0/3.0, unit "x"
    qualitative       "Double-digit", "Flattish"           -> None
    ambiguous_absolute bare "178,000" in a growth column   -> qty,   unit ""
    unparsed          nothing numeric recoverable          -> None

    Pure functions, stdlib only — safe to import from the extractor, the sanitizer,
    the mails and the app. Unit-testable offline.

Usage:
    from guidance_value import parse_guidance_value
    r = parse_guidance_value("19% - 26% (CAGR)", metric="revenue", horizon="3Y")
    r["value_type"], r["value_num"], r["value_unit"], r["growth_pct"]
    # ('growth_pct', 22.5, '%', 22.5)
"""
from __future__ import annotations

import re

# Horizons emitted by the extractor's GROWTH columns (Table_1 "... growth
# guidance %" headers) vs FY-headed columns, which hold absolute LEVEL targets.
GROWTH_HORIZONS = frozenset({"NEXT_QTR", "1Y", "2Y", "3Y", "3Y+"})

# Metrics whose % is a LEVEL, not a growth rate.
LEVEL_METRICS = frozenset({"margin"})

# A bare number this large in a growth column is not a growth rate — it is an
# absolute target/count the model parked in the wrong column (verified: every
# bare value >300 in the live table is a Rs-cr target or a unit count). Only ever
# applied to BARE, unit-less numbers — never to a cell that states "%".
_BARE_ABSOLUTE_MIN = 300.0

_NUM = r"[-+]?\d[\d,]*(?:\.\d+)?"

_RE_NUM = re.compile(_NUM)
# the first bound may carry its own '%' ("19% - 26%"), so allow it before the dash
_RE_RANGE = re.compile(rf"({_NUM})\s*%?\s*(?:-|–|—|to)\s*({_NUM})", re.I)
# A range that is genuinely a PERCENT range — the trailing bound must carry the '%'
# ("19% - 26%", "15-20%"). Without this anchor a parenthetical like
# "6.5% (CAGR 2026-2030)" matched the YEARS and returned 2028%.
_RE_PCT_RANGE = re.compile(rf"({_NUM})\s*%?\s*(?:-|–|—|to)\s*({_NUM})\s*%", re.I)
# a single "<number>%" token
_RE_NUM_PCT = re.compile(rf"({_NUM})\s*%", re.I)
_RE_PCT_TOKEN = re.compile(r"%")
_RE_BARE = re.compile(rf"^\s*({_NUM})\s*$")
_RE_COMMA_GROUPED = re.compile(r"\d,\d")

# currency -> multiplier into Rs crore (1 cr = 10 million)
_CUR_SCALE = (
    (re.compile(r"\bcrores?\b|\bcr\b", re.I), 1.0),
    (re.compile(r"\blakhs?\b", re.I), 0.01),
    (re.compile(r"\bbillions?\b|\bbn\b", re.I), 100.0),
    (re.compile(r"\bmillions?\b|\bmn\b", re.I), 0.1),
    (re.compile(r"\bthousand\b|\bk\b", re.I), 0.0001),
)
_RE_CURRENCY = re.compile(r"₹|\brs\.?\b|\binr\b|\$|\busd\b|\bcrores?\b|\bcr\b|"
                          r"\blakhs?\b|\bbillions?\b|\bbn\b|\bmillions?\b|\bmn\b", re.I)
# physical / count units (captured so we can report what was written)
_RE_PHYS = re.compile(
    r"\b(mtpa|mmtpa|mt|tonnes?|tons?|kt|mw|gw|kwh|kl|klpd|sq\.?\s?ft|sqft|acres?|"
    r"units?|seats?|stores?|outlets?|rooms?|beds?|sites?|branches?|vehicles?|"
    r"pieces?|nos\.?|customers?|subscribers?)\b", re.I)
# "double" = 2x, but "double digit" means 10-99% — never a multiple, so exclude it
_RE_MULTIPLE = re.compile(
    r"(\d+(?:\.\d+)?)\s*x\b|\b(double|doubling|two-?fold|2x)\b(?![-\s]*digits?)|"
    r"\b(triple|tripling|three-?fold|3x)\b|\b(quadruple|four-?fold|4x)\b", re.I)
# utilisation / level phrasing on capacity & volume
_RE_UTILISATION = re.compile(
    r"utili[sz]ation|utili[sz]ed|occupanc|capacity\s+level|run[-\s]?rate|"
    r"exit\s+(?:rate|margin)|\blevel\b", re.I)
# purely qualitative growth language (no usable number)
_RE_QUALITATIVE = re.compile(
    r"double[-\s]?digit|single[-\s]?digit|teens|flattish|flat\b|modest|marginal|"
    r"significant|robust|strong|steady|healthy|muted|decent|sustained|stable|"
    r"improve|maintain|similar|line with|na\b", re.I)


def _to_float(s: str) -> float | None:
    try:
        return float(str(s).replace(",", "").replace("+", "").strip())
    except (TypeError, ValueError):
        return None


def _numbers(text: str) -> list[float]:
    out = []
    for m in _RE_NUM.finditer(text):
        v = _to_float(m.group(0))
        if v is not None:
            out.append(v)
    return out


def _pct_from_text(text: str) -> float | None:
    """The percentage stated in the text. Only called when a '%' token is present.

    Order matters: a %-anchored RANGE first ("19% - 26%" / "15-20%" -> midpoint),
    then the first "<number>%" token, and only then any bare number. Anchoring on
    the '%' is what stops a parenthetical year span ("6.5% (CAGR 2026-2030)") from
    being read as a 2026-2030 range."""
    rng = _RE_PCT_RANGE.search(text)
    if rng:
        a, b = _to_float(rng.group(1)), _to_float(rng.group(2))
        if a is not None and b is not None:
            return round((a + b) / 2, 4)
    one = _RE_NUM_PCT.search(text)
    if one:
        v = _to_float(one.group(1))
        if v is not None:
            return round(v, 4)
    nums = _numbers(text)
    return round(nums[0], 4) if nums else None


def _result(value_type: str, value_num: float | None, value_unit: str,
            growth_pct: float | None, raw: str, note: str = "") -> dict:
    return {"value_type": value_type, "value_num": value_num,
            "value_unit": value_unit, "growth_pct": growth_pct,
            "raw": raw, "note": note}


def parse_guidance_value(raw, metric: str = "", horizon: str = "") -> dict:
    """Classify one guidance cell.

    raw      : the cell text as extracted ("19% - 26% (CAGR)", "INR530 crores").
    metric   : canonical metric (revenue/ebitda/pat/margin/volume/capacity).
    horizon  : horizon_fy tag — a GROWTH_HORIZONS value means the cell came from a
               growth-% column; an "FY26"-style tag means an absolute LEVEL column.

    Returns dict(value_type, value_num, value_unit, growth_pct, raw, note).
    `growth_pct` is non-None ONLY for a genuine growth rate — that is the single
    value any grower/ranker should consume.
    """
    text = str(raw or "").strip()
    metric = str(metric or "").strip().lower()
    horizon = str(horizon or "").strip().upper()
    if not text or text.upper() in ("NA", "N/A", "NONE", "-", "--"):
        return _result("unparsed", None, "", None, text, "empty")

    from_growth_col = horizon in GROWTH_HORIZONS
    has_pct = bool(_RE_PCT_TOKEN.search(text))
    has_cur = bool(_RE_CURRENCY.search(text))
    phys = _RE_PHYS.search(text)

    # 1) explicit currency -> absolute rupee target (normalise to Rs crore).
    #    Checked before '%' so "INR978.75 crores (Derived)" is not read as a rate.
    if has_cur and not (has_pct and not _RE_CURRENCY.search(text.split("%")[0])):
        nums = _numbers(text)
        if nums:
            scale = 1.0
            for rx, mult in _CUR_SCALE:
                if rx.search(text):
                    scale = mult
                    break
            rng = _RE_RANGE.search(text)
            base = ((_to_float(rng.group(1)) + _to_float(rng.group(2))) / 2
                    if rng and _to_float(rng.group(1)) is not None
                    and _to_float(rng.group(2)) is not None else nums[0])
            return _result("absolute_inr", round(base * scale, 4), "INR_cr", None,
                           text, "rupee target")
        return _result("qualitative", None, "", None, text, "currency, no number")

    # 2) physical / count units -> absolute quantity
    if phys and not has_pct:
        nums = _numbers(text)
        return _result("absolute_units", round(nums[0], 4) if nums else None,
                       phys.group(1).upper(), None, text, "physical target")

    # 3) multiples ("double", "3x") -> a real growth rate: 2x == +100%
    mult = _RE_MULTIPLE.search(text)
    if mult and not has_pct:
        if mult.group(1):
            factor = _to_float(mult.group(1))
        elif mult.group(2):
            factor = 2.0
        elif mult.group(3):
            factor = 3.0
        else:
            factor = 4.0
        if factor and factor > 0:
            growth = round((factor - 1.0) * 100.0, 2) if from_growth_col else None
            return _result("multiple", factor, "x", growth, text,
                           f"{factor}x -> {growth}% growth" if growth else "")

    # 4) a stated percentage
    if has_pct:
        pct = _pct_from_text(text)
        if pct is None:
            return _result("qualitative", None, "", None, text, "% with no number")
        if metric in LEVEL_METRICS:
            return _result("margin_pct", pct, "%", None, text, "margin LEVEL")
        if _RE_UTILISATION.search(text):
            return _result("utilisation_pct", pct, "%", None, text, "level, not growth")
        # CAPACITY is irreducibly ambiguous from the number alone: the same "75%"
        # is capacity ADDITION (growth) for one company and capacity UTILISATION
        # (a level) for the next, and the tables carry no marker. Verified on live
        # data: 553 of 2,098 capacity rows sit in the classic 60-100% utilisation
        # band with no keyword. So capacity gets its own tag and never silently
        # feeds growth math; a consumer that wants it must ask for it. (The word
        # is in GF1's exact_statement if a caller needs to disambiguate.)
        if metric == "capacity":
            return _result("capacity_pct", pct, "%", None, text,
                           "capacity % — addition or utilisation, not disambiguated")
        if not from_growth_col:
            # FY-headed column: a % here is a level target (e.g. FY27 margin)
            return _result("margin_pct" if metric in LEVEL_METRICS else "utilisation_pct",
                           pct, "%", None, text, "FY-column level")
        return _result("growth_pct", pct, "%", pct, text, "")

    # 5) bare number — no unit at all: meaning comes from the column
    bare = _RE_BARE.match(text)
    if bare:
        val = _to_float(bare.group(1))
        if val is None:
            return _result("unparsed", None, "", None, text, "")
        if metric in LEVEL_METRICS:
            return _result("margin_pct", val, "%", None, text, "margin LEVEL")
        if not from_growth_col:
            return _result("ambiguous_absolute", val, "", None, text,
                           "FY-column bare number")
        # In a growth column a bare number is meant as a %, but comma-grouped or
        # very large values are absolute targets/counts parked in the wrong column.
        if _RE_COMMA_GROUPED.search(text) or abs(val) >= _BARE_ABSOLUTE_MIN:
            return _result("ambiguous_absolute", val, "", None, text,
                           "absolute in a growth column")
        return _result("growth_pct", val, "%", val, text, "bare % in growth column")

    # 6) numbers embedded in short prose ("Exceed 115,000 MTPA" handled above).
    #    Long narrative ("Ascent-K trial production Q3 FY27, commercial ...") is
    #    commentary, not a number — a stray year would otherwise read as a value.
    nums = _numbers(text)
    if nums and not _RE_QUALITATIVE.search(text) and len(text.split()) <= 4:
        return _result("ambiguous_absolute", round(nums[0], 4), "", None, text,
                       "number in prose")

    # 7) qualitative language only
    return _result("qualitative", None, "", None, text, "no usable number")


def is_growth(parsed: dict) -> bool:
    """True when the parsed cell carries a usable GROWTH rate."""
    return parsed.get("growth_pct") is not None


def describe(parsed: dict) -> str:
    """Short human label for mails/app, e.g. '+25% growth', 'Rs 530cr target',
    '22% margin', '115,000 MTPA'."""
    t, n, u = parsed.get("value_type"), parsed.get("value_num"), parsed.get("value_unit")
    if t == "growth_pct":
        return f"{n:+.1f}% growth".replace(".0%", "%")
    if t == "margin_pct":
        return f"{n:.1f}% margin"
    if t == "utilisation_pct":
        return f"{n:.1f}% level"
    if t == "capacity_pct":
        return f"{n:.1f}% capacity"
    if t == "absolute_inr":
        return f"Rs {n:,.0f}cr target"
    if t == "absolute_units":
        return f"{n:,.0f} {u}"
    if t == "multiple":
        return f"{n:g}x"
    if t == "ambiguous_absolute":
        return f"{n:,.0f} (unit unclear)"
    return str(parsed.get("raw", ""))[:40]
