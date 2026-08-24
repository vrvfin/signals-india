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
    level_pct         a % in an FY-headed column = a LEVEL target for that FY
    absolute_inr      "INR530 crores", "26,000 million"    -> Rs cr, unit "INR_cr"
    absolute_usd      "USD 10 mn", "$550k - $600k"         -> USD mn, unit "USD_mn"
                      (2026-08-15; also fills value_num_inr_cr via USDINR. Before
                       this, USD was silently stamped INR_cr at the rupee scale.)
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

# Metrics whose % is a LEVEL, not a growth rate. Beyond margin these are the BFSI
# and IT ratios (2026-07-29): a 3.5% NIM, a 2.1% GNPA, an 18% ROA or 82% headcount
# utilisation are STATES — reading them as growth would put bank spreads straight
# into the growth rankings, the same class of bug the typed parse removed.
LEVEL_METRICS = frozenset({"margin", "nim", "npa", "credit_cost", "returns",
                           "utilisation"})

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

# --- currency + magnitude (rewritten 2026-08-15) ----------------------------------
# THE BUG THIS FIXES: every absolute currency amount was stamped "INR_cr" whatever
# the currency was actually written in, and the magnitude word was only recognised
# in its SPACED form. Verified on live guidance_tracker order_book rows:
#     "ABS: USD 10 mn"              -> 1.0    INR_cr   (real: ~Rs 88 cr)
#     "ABS: $30,000"                -> 30000  INR_cr   (real: ~Rs 0.26 cr)
#     "ABS: $550k - $600k"          -> 550    INR_cr   (\bk\b cannot match "550k":
#                                                       no word boundary digit->k)
#     "ABS: USD 0.55-0.60 Mn/month" -> 0.055  INR_cr
# 10 of 15 sampled order_book rows are USD, so rupee order targets were unusable.
#
# An INR marker always WINS when both currencies appear ("USD 10mn (INR 88 cr)"),
# so a cell that was already rupees keeps its old classification exactly.
_RE_INR = re.compile(r"₹|\brs\.?\b|\binr\b|\brupees?\b", re.I)
_RE_USD = re.compile(r"\busd\b|\bus\s?\$|\$|\bdollars?\b", re.I)
# LaTeX math delimiters leak out of PDF text extraction ("$\ge$ 10%", "$\sim$5%").
# The '$' there is a delimiter, not a dollar sign — stripped before any currency
# test so the cell types as the growth rate it actually is.
_RE_LATEX = re.compile(r"\$\s*\\[a-zA-Z]+\s*\$")

# Approximate USD->INR. Guidance is a forward target with range/rounding noise far
# wider than FX drift, so a constant is honest here — a live rate would imply a
# precision the source text does not have. Bump it when it drifts materially.
USDINR = 88.0

# Magnitude words, in the SPACED form ("10 mn") and the number-ATTACHED form
# ("550k", "3.5M", "530cr") that the old \b-anchored patterns could never match.
# Order matters: bn before mn so "10bn" is not read as millions.
_MAG_PATTERNS = (
    (re.compile(r"\bcrores?\b|\bcr\b|\d\s*crs?\b", re.I), "cr"),
    (re.compile(r"\blakhs?\b|\d\s*lakhs?\b", re.I), "lakh"),
    (re.compile(r"\bbillions?\b|\bbn\b|\d\s*bn?\b", re.I), "bn"),
    (re.compile(r"\bmillions?\b|\bmn\b|\d\s*mn?\b", re.I), "mn"),
    (re.compile(r"\bthousand\b|\bk\b|\d\s*k\b", re.I), "k"),
)
# magnitude -> multiplier into the currency's NATURAL unit:
#   INR -> crore (1 cr = 10 million)      USD -> million
_SCALE_INR = {"cr": 1.0, "lakh": 0.01, "bn": 100.0, "mn": 0.1, "k": 0.0001,
              None: 1.0}          # bare INR in a guidance cell has always meant crore
_SCALE_USD = {"cr": 10.0, "lakh": 0.1, "bn": 1000.0, "mn": 1.0, "k": 0.001,
              None: 1e-6}         # bare "$30,000" is literally dollars


def _magnitude(text: str):
    """The magnitude word in `text`: 'cr'|'lakh'|'bn'|'mn'|'k', or None."""
    for rx, tag in _MAG_PATTERNS:
        if rx.search(text):
            return tag
    return None


def _currency(text: str) -> str:
    """'USD' only when a dollar marker is present and no rupee marker is."""
    if _RE_INR.search(text):
        return "INR"
    return "USD" if _RE_USD.search(text) else "INR"


def _money(text: str, base: float) -> tuple[str, float, str, float]:
    """(value_type, value_num, value_unit, value_num_inr_cr) for an absolute amount.

    `base` is the raw number already chosen from the text (single or range midpoint).
    """
    cur = _currency(text)
    mag = _magnitude(text)
    if cur == "USD":
        usd_mn = round(base * _SCALE_USD[mag], 4)
        # 1 USD mn * rate = that many INR mn = rate/10 INR cr
        return ("absolute_usd", usd_mn, "USD_mn", round(usd_mn * USDINR / 10.0, 4))
    inr_cr = round(base * _SCALE_INR[mag], 4)
    return ("absolute_inr", inr_cr, "INR_cr", inr_cr)
# HARD currency markers — these unambiguously mean money.
_RE_CURRENCY_HARD = re.compile(r"₹|\brs\.?\b|\binr\b|\$|\busd\b|\bcrores?\b|\bcr\b", re.I)
# Magnitude words. "lakh"/"million" are SCALE, not currency: "5.25 lakh units" and
# "3.5-4 lakh cases/month" are quantities, not rupees. They only imply money when a
# hard marker is present, so they must never by themselves force absolute_inr.
_RE_CURRENCY_SCALE = re.compile(r"\blakhs?\b|\bbillions?\b|\bbn\b|\bmillions?\b|\bmn\b",
                                re.I)
_RE_CURRENCY = re.compile(_RE_CURRENCY_HARD.pattern + "|" + _RE_CURRENCY_SCALE.pattern,
                          re.I)
# physical / count units (captured so we can report what was written)
_RE_PHYS = re.compile(
    r"\b(mtpa|mmtpa|mt|tonnes?|tons?|kt|tpd|mtpd|mw|gw|kwh|kl|klpd|litres?|liters?|"
    r"sq\.?\s?ft|sqft|sqm|msf|acres?|"
    r"units?|cases?|boxes|bottles?|pieces?|pcs|nos\.?|panels?|modules?|cells?|"
    r"seats?|stores?|outlets?|rooms?|beds?|keys|sites?|branches?|centres?|centers?|"
    r"vehicles?|trucks?|wagons?|coaches?|"
    r"customers?|subscribers?|members?)\b", re.I)
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
            growth_pct: float | None, raw: str, note: str = "",
            inr_cr: float | None = None) -> dict:
    """`value_num_inr_cr` (2026-08-15) is the amount in Rs crore whatever currency
    the cell was written in — None for anything that is not an absolute amount. It
    is an ADDITIONAL key: every caller reads specific keys, never **unpacks, so old
    consumers are unaffected."""
    return {"value_type": value_type, "value_num": value_num,
            "value_unit": value_unit, "growth_pct": growth_pct,
            "raw": raw, "note": note, "value_num_inr_cr": inr_cr}


def parse_guidance_value(raw, metric: str = "", horizon: str = "",
                         typed_prompt: bool = False) -> dict:
    """Classify one guidance cell.

    raw      : the cell text as extracted ("19% - 26% (CAGR)", "INR530 crores").
    metric   : canonical metric (revenue/ebitda/pat/margin/volume/capacity).
    horizon  : horizon_fy tag — a GROWTH_HORIZONS value means the cell came from a
               growth-% column; an "FY26"-style tag means an absolute LEVEL column.
    typed_prompt : True when the row was produced by the concall prompt that
               REQUIRES an "LVL:" prefix on any level percentage (2026-07-18).
               Then an unprefixed capacity % can be trusted as genuine capacity
               GROWTH — the model would have written LVL: if it were utilisation.
               Left False for legacy rows, where capacity % stays ambiguous.

    Returns dict(value_type, value_num, value_unit, growth_pct, raw, note).
    `growth_pct` is non-None ONLY for a genuine growth rate — that is the single
    value any grower/ranker should consume.
    """
    text = _RE_LATEX.sub(" ", str(raw or "")).strip()
    metric = str(metric or "").strip().lower()
    horizon = str(horizon or "").strip().upper()
    if not text or text.upper() in ("NA", "N/A", "NONE", "-", "--"):
        return _result("unparsed", None, "", None, text, "empty")

    # --- explicit prefixes (concall_prompt "[MANDATORY CELL FORMAT]") ------------
    # When the model declares the kind we take it at its word — no inference. These
    # are the ONLY fully reliable signals; everything below is best-effort typing of
    # legacy/untagged cells. Handled here so the parser is ready BEFORE the prompt
    # change ships (an untagged cell simply falls through to the old logic).
    m_pref = re.match(r"^\s*(ABS|LVL)\s*:\s*(.+)$", text, re.I)
    if m_pref:
        kind, body = m_pref.group(1).upper(), m_pref.group(2).strip()
        if kind == "ABS":
            # same rule as below: a physical unit beats a bare scale word, so
            # "ABS: 5.25 lakh units" is a QUANTITY, not rupees
            _body_phys = _RE_PHYS.search(body)
            if _RE_CURRENCY_HARD.search(body) or (
                    _RE_CURRENCY_SCALE.search(body) and not _body_phys):
                nums = _numbers(body)
                if nums:
                    # nums[0], NOT the range midpoint — this branch has always taken
                    # the first bound and range handling is not what the currency fix
                    # is for. Keeping it means a declared-ABS rupee cell is untouched.
                    vt, vn, vu, inr = _money(body, nums[0])
                    return _result(vt, vn, vu, None, text, "declared ABS",
                                   inr_cr=inr)
            phys_m = _RE_PHYS.search(body)
            nums = _numbers(body)
            return _result("absolute_units", round(nums[0], 4) if nums else None,
                           phys_m.group(1).upper() if phys_m else "", None, text,
                           "declared ABS")
        # LVL: a percentage that is a STATE, never a growth rate
        pct = _pct_from_text(body) if _RE_PCT_TOKEN.search(body) else None
        if pct is None:
            nums = _numbers(body)
            pct = nums[0] if nums else None
        if metric == "margin" or re.search(r"margin", body, re.I):
            vt = "margin_pct"
        elif metric in LEVEL_METRICS:
            vt = "level_pct"
        elif _RE_UTILISATION.search(body):
            vt = "utilisation_pct"
        elif metric == "capacity":
            vt = "capacity_pct"
        else:
            vt = "level_pct"
        return _result(vt, pct, "%", None, text, "declared LVL")

    from_growth_col = horizon in GROWTH_HORIZONS
    has_pct = bool(_RE_PCT_TOKEN.search(text))
    phys = _RE_PHYS.search(text)
    # Money only when a HARD marker is present. A bare scale word ("5.25 lakh
    # units", "3.5-4 lakh cases/month") is a quantity — and an explicit physical
    # unit always wins over a scale word, so those stay absolute_units.
    has_cur = bool(_RE_CURRENCY_HARD.search(text)) or (
        bool(_RE_CURRENCY_SCALE.search(text)) and not phys)

    # 1) explicit currency -> absolute rupee target (normalise to Rs crore).
    #    Checked before '%' so "INR978.75 crores (Derived)" is not read as a rate.
    if has_cur and not (has_pct and not _RE_CURRENCY.search(text.split("%")[0])):
        nums = _numbers(text)
        if nums:
            rng = _RE_RANGE.search(text)
            base = ((_to_float(rng.group(1)) + _to_float(rng.group(2))) / 2
                    if rng and _to_float(rng.group(1)) is not None
                    and _to_float(rng.group(2)) is not None else nums[0])
            vt, vn, vu, inr = _money(text, base)
            return _result(vt, vn, vu, None, text,
                           "rupee target" if vt == "absolute_inr" else "USD target",
                           inr_cr=inr)
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
            # margin keeps its own tag; NIM / NPA / ROA / utilisation are levels too
            return _result("margin_pct" if metric == "margin" else "level_pct",
                           pct, "%", None, text, f"{metric} LEVEL")
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
            # Under the typed prompt an unprefixed capacity % is a CHANGE (a level
            # would carry "LVL:"), e.g. an airline guiding ASK capacity +10-15%.
            if typed_prompt and from_growth_col:
                return _result("growth_pct", pct, "%", pct, text,
                               "capacity growth (typed prompt, no LVL: marker)")
            return _result("capacity_pct", pct, "%", None, text,
                           "capacity % — addition or utilisation, not disambiguated")
        if not from_growth_col:
            # FY-headed columns hold LEVEL targets for that fiscal year ("FY27:
            # EBITDA 23.5%"), not growth rates. Neutral tag — calling a revenue %
            # here "utilisation" would be nonsense.
            return _result("level_pct", pct, "%", None, text, "FY-column level")
        return _result("growth_pct", pct, "%", pct, text, "")

    # 5) bare number — no unit at all: meaning comes from the column
    bare = _RE_BARE.match(text)
    if bare:
        val = _to_float(bare.group(1))
        if val is None:
            return _result("unparsed", None, "", None, text, "")
        if metric in LEVEL_METRICS:
            return _result("margin_pct" if metric == "margin" else "level_pct",
                           val, "%", None, text, f"{metric} LEVEL")
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
    if t == "level_pct":
        return f"{n:.1f}% level"
    if t == "absolute_inr":
        return f"Rs {n:,.0f}cr target"
    if t == "absolute_usd":
        inr = parsed.get("value_num_inr_cr")
        return (f"$ {n:,.2f}mn target".replace(".00mn", "mn")
                + (f" (~Rs {inr:,.0f}cr)" if inr else ""))
    if t == "absolute_units":
        return f"{n:,.0f} {u}"
    if t == "multiple":
        return f"{n:g}x"
    if t == "ambiguous_absolute":
        return f"{n:,.0f} (unit unclear)"
    return str(parsed.get("raw", ""))[:40]


# --------------------------------------------------------------------------- #
#  self-test — offline, no Drive, no network                                    #
# --------------------------------------------------------------------------- #

# (raw, metric, horizon, expected value_type, expected value_num, expected unit)
# The USD block is the 2026-08-15 fix; every other case is a REGRESSION guard —
# these are the classifications the live table already relies on and they must not
# move. Rupee cases were verified against live guidance_tracker rows.
_CASES = [
    # --- the fix: USD is never silently INR ---------------------------------
    ("ABS: USD 10 mn",              "order_book", "1Y",  "absolute_usd", 10.0,   "USD_mn"),
    ("ABS: $30,000 (small)",        "order_book", "1Y",  "absolute_usd", 0.03,   "USD_mn"),
    ("ABS: $550k - $600k",          "order_book", "NEXT_QTR", "absolute_usd", 0.55, "USD_mn"),
    ("ABS: USD 0.55-0.60 Mn/month", "order_book", "NEXT_QTR", "absolute_usd", 0.55, "USD_mn"),
    ("ABS: $1.5 million",           "order_book", "1Y",  "absolute_usd", 1.5,    "USD_mn"),
    ("ABS: $3.5-4.0M",              "order_book", "1Y",  "absolute_usd", 3.5,    "USD_mn"),
    ("USD 25 billion",              "revenue",    "3Y",  "absolute_usd", 25000.0, "USD_mn"),
    ("$100,000,000",                "revenue",    "1Y",  "absolute_usd", 100.0,  "USD_mn"),
    # LaTeX delimiter, NOT dollars — must stay a growth rate
    (r"$\ge$ 10%",                  "revenue",    "1Y",  "growth_pct",   10.0,   "%"),
    # an INR marker WINS when both appear -> unchanged rupee classification
    ("ABS: USD 10mn (INR 88 cr)",   "order_book", "1Y",  "absolute_inr", 10.0,   "INR_cr"),
    # --- regression: rupee cells must classify exactly as before -------------
    ("ABS: INR 530 crores",         "revenue",    "1Y",  "absolute_inr", 530.0,  "INR_cr"),
    ("ABS: INR 5300 cr (to be executed)", "order_book", "1Y", "absolute_inr", 5300.0, "INR_cr"),
    ("ABS: INR 20000 cr",           "order_book", "3Y+", "absolute_inr", 20000.0, "INR_cr"),
    ("ABS: >INR 357.63 cr",         "order_book", "1Y",  "absolute_inr", 357.63, "INR_cr"),
    ("INR978.75 crores (Derived)",  "revenue",    "1Y",  "absolute_inr", 978.75, "INR_cr"),
    ("Rs 26,000 million",           "revenue",    "1Y",  "absolute_inr", 2600.0, "INR_cr"),
    ("INR 100 cr/year (STC)",       "order_book", "1Y",  "absolute_inr", 100.0,  "INR_cr"),
    # --- regression: non-currency typing is untouched ------------------------
    ("19% - 26% (CAGR)",            "revenue",    "3Y",  "growth_pct",   22.5,   "%"),
    ("32.5% CAGR",                  "revenue",    "1Y",  "growth_pct",   32.5,   "%"),
    ("6.5% (CAGR 2026-2030)",       "revenue",    "3Y",  "growth_pct",   6.5,    "%"),
    ("LVL: 23.5% (EBITDA margin)",  "margin",     "1Y",  "margin_pct",   23.5,   "%"),
    ("LVL: 75% (utilisation)",      "capacity",   "1Y",  "utilisation_pct", 75.0, "%"),
    ("ABS: 115,000 MTPA",           "capacity",   "1Y",  "absolute_units", 115000.0, "MTPA"),
    ("ABS: 5.25 lakh units",        "volume",     "1Y",  "absolute_units", 5.25, "UNITS"),
    ("ABS: 1,409+ trains",          "order_book", "1Y",  "absolute_units", 1409.0, ""),
    ("ABS: 37 GW",                  "order_book", "3Y+", "absolute_units", 37.0,  "GW"),
    ("Double-digit",                "revenue",    "1Y",  "qualitative",  None,   ""),
    ("3x (Tripling)",               "revenue",    "3Y",  "multiple",     3.0,    "x"),
    ("178,000",                     "capacity",   "1Y",  "ambiguous_absolute", 178000.0, ""),
    ("NA",                          "revenue",    "1Y",  "unparsed",     None,   ""),
]


def _self_test() -> int:
    fails = []
    for raw, metric, hz, want_t, want_n, want_u in _CASES:
        got = parse_guidance_value(raw, metric=metric, horizon=hz, typed_prompt=True)
        ok = (got["value_type"] == want_t
              and got["value_unit"] == want_u
              and (want_n is None) == (got["value_num"] is None)
              and (want_n is None or abs(got["value_num"] - want_n) < 0.01))
        if not ok:
            fails.append(f"  {raw!r}\n     want {want_t}/{want_n}/{want_u!r}\n"
                         f"     got  {got['value_type']}/{got['value_num']}/"
                         f"{got['value_unit']!r}")
    # USD -> INR conversion must be populated and sane
    u = parse_guidance_value("ABS: USD 10 mn", metric="order_book", horizon="1Y")
    if u.get("value_num_inr_cr") != round(10.0 * USDINR / 10.0, 4):
        fails.append(f"  USD->INR conversion wrong: {u.get('value_num_inr_cr')}")
    # a rupee amount reports itself in rupees
    r = parse_guidance_value("ABS: INR 530 crores", metric="revenue", horizon="1Y")
    if r.get("value_num_inr_cr") != 530.0:
        fails.append(f"  INR passthrough wrong: {r.get('value_num_inr_cr')}")
    # a non-amount carries no rupee figure at all
    g = parse_guidance_value("19% - 26%", metric="revenue", horizon="3Y")
    if g.get("value_num_inr_cr") is not None:
        fails.append(f"  growth row should have no inr_cr: {g.get('value_num_inr_cr')}")

    total = len(_CASES) + 3
    if fails:
        print(f"FAIL {len(fails)}/{total}")
        print("\n".join(fails))
        return 1
    print(f"OK  {total}/{total} guidance_value cases pass  (USDINR={USDINR})")
    return 0


if __name__ == "__main__":
    import sys
    if "--self-test" in sys.argv:
        raise SystemExit(_self_test())
    for _a in sys.argv[1:]:
        print(f"{_a!r}\n  {parse_guidance_value(_a)}")
