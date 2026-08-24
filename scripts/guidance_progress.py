r"""
guidance_progress.py — how far along is a promise, RIGHT NOW (user 2026-08-15).

WHY THIS EXISTS
    The repo already records what management guided (guidance_tracker.parquet,
    mgmt_credibility.parquet) and, eventually, a terminal BEAT/MISS verdict once the
    horizon closes (build_pead_flags.py). It records NOTHING in between:
    `compute_flags` compares ANNUAL rows only and skips a commitment outright while
    "horizon not yet reported" — so guidance_vs_actual.parquet carries 23,919
    TOO_EARLY rows whose `actual_delivered` is the literal string "NA", forever.

        POWERMECH  Order Book  "INR 12,000 crores inflow"  FY27  actual=NA  TOO_EARLY

    A commitment is therefore invisible until the day it is already decided, which is
    the day it stops being tradeable information.

WHAT IT DOES
    Measures a commitment CUMULATIVELY while it is still open: how much of the guided
    number has landed so far, how much of the horizon has elapsed, and whether the
    first is keeping up with the second.

        pct_of_target = actual_to_date / guided        (how far along)
        time_pct      = quarters_elapsed / quarters     (how far through)
        pace_ratio    = pct_of_target / time_pct        (>1 = running ahead)

    Pure functions: stdlib + pandas only, no Drive, no network, no env. The Drive I/O
    lives in build_guidance_progress.py — the same split as guidance_value.py (pure)
    vs build_pead_flags.py (I/O), so this stays unit-testable offline.

QUARTER CONVENTION
    Everything here is the RESULTS convention (the quarter that just ENDED), matching
    quarterly_table.season_quarter(). Never extract_concall._current_india_quarter(),
    which is one quarter ahead and exists only to express guidance horizons.

Usage:
    python scripts/guidance_progress.py --self-test
"""
from __future__ import annotations

import hashlib
import re

# --------------------------------------------------------------------------- #
#  quarter arithmetic                                                           #
# --------------------------------------------------------------------------- #
# A fiscal quarter is one integer: fy*4 + (q-1), fy being the 2-digit FY year. That
# makes "the four quarters of FY27" a contiguous range and "how many quarters between
# these two" a subtraction — no calendar maths anywhere downstream.

_RE_Q = re.compile(r"Q([1-4])\s*FY\s*(\d{2,4})", re.I)
_RE_FY = re.compile(r"^\s*FY\s*(\d{2,4})\s*$", re.I)
# Q1 -> quarter ending Jun of (FY-1); Q4 -> quarter ending Mar of FY. Inverse of
# quarterly_table._QMAP, which maps "Jun 2026" -> "Q1 FY27".
_Q_MONTH = {1: ("Jun", -1), 2: ("Sep", -1), 3: ("Dec", -1), 4: ("Mar", 0)}

# GF_TRACK / GF1 timeframe vocabulary (concall_prompt.txt Timeframe enum) -> how many
# quarters the promise spans, measured from the quarter AFTER the one it was given in.
# These are deliberately conservative: a promise with a vague horizon is given the
# SHORTER reading, so it shows up as behind pace early rather than never.
_VAGUE_QUARTERS = {
    "IMMEDIATE": 1, "NEXT_QTR": 1,
    "NEAR_TERM": 2,
    "MEDIUM_TERM": 4,
    "LONG_TERM": 12, "STRATEGIC": 12,
}


def parse_q(label) -> tuple[int, int] | None:
    """'Q1 FY27' / 'Q1FY2027' -> (27, 1). None when it is not a quarter label."""
    m = _RE_Q.search(str(label or ""))
    if not m:
        return None
    return (int(m.group(2)) % 100, int(m.group(1)))


def parse_fy(label) -> int | None:
    """'FY27' / 'FY2027' -> 27. None when it is not a bare FY label."""
    m = _RE_FY.match(str(label or ""))
    return int(m.group(1)) % 100 if m else None


def q_idx(fy: int, q: int) -> int:
    return fy * 4 + (q - 1)


def idx_q(idx: int) -> tuple[int, int]:
    return (idx // 4, idx % 4 + 1)


def q_label(idx: int) -> str:
    fy, q = idx_q(idx)
    return f"Q{q} FY{fy:02d}"


def q_period(idx: int) -> str:
    """Fiscal quarter -> the Screener period column that holds it.
    Q1 FY27 -> 'Jun 2026' · Q4 FY27 -> 'Mar 2027'."""
    fy, q = idx_q(idx)
    mon, bump = _Q_MONTH[q]
    return f"{mon} {2000 + fy + bump}"


def q_from_date(d) -> int:
    """Calendar date -> the fiscal quarter INDEX containing it.

    NOT the same job as quarterly_table.qtr_label, which maps a Screener period
    COLUMN and therefore only ever sees quarter-END months (Mar/Jun/Sep/Dec) —
    feeding it an arbitrary date returns the input unchanged, so a 14-Aug filing
    silently failed to resolve. Announcement dates land on any day of any month.
    """
    m, y = d.month, d.year
    if m <= 3:
        return q_idx(y % 100, 4)            # Jan-Mar = Q4 of the FY ending this Mar
    if m <= 6:
        return q_idx((y + 1) % 100, 1)
    if m <= 9:
        return q_idx((y + 1) % 100, 2)
    return q_idx((y + 1) % 100, 3)


def fy_window(fy: int) -> tuple[int, int]:
    """The four quarters of a fiscal year, inclusive."""
    return (q_idx(fy, 1), q_idx(fy, 4))


def window_quarters(start: int, end: int) -> list[int]:
    return list(range(start, end + 1))


def resolve_window(target_period, guid_quarter) -> tuple[int, int, str] | None:
    """(start_idx, end_idx, basis) for a commitment, or None if unresolvable.

    Handles every horizon spelling the two guidance tables actually contain:
      'FY27'                  -> that whole fiscal year          (basis 'fy')
      'Q2 FY27'               -> that single quarter             (basis 'quarter')
      '1Y'/'2Y'/'3Y'/'3Y+'    -> FY(base+n), base from guid_quarter  (basis 'relative')
      'NEXT_QTR'              -> the quarter after guid_quarter  (basis 'relative')
      IMMEDIATE/NEAR_TERM/... -> n quarters after guid_quarter   (basis 'vague')

    The relative FY reading matches build_pead_flags._resolve_horizon (`Q3 FY25` + 1Y
    -> FY26) so a commitment resolves to the same horizon in both places.
    """
    tp = str(target_period or "").strip().upper()
    if not tp:
        return None

    q = parse_q(tp)
    if q:
        i = q_idx(*q)
        return (i, i, "quarter")

    fy = parse_fy(tp)
    if fy is not None:
        s, e = fy_window(fy)
        return (s, e, "fy")

    base = parse_q(guid_quarter)
    if not base:
        return None                      # relative horizon with no anchor
    base_idx = q_idx(*base)

    m = re.fullmatch(r"([123])\s*Y\+?", tp)
    if m:
        s, e = fy_window(base[0] + int(m.group(1)))
        return (s, e, "relative")

    n = _VAGUE_QUARTERS.get(tp)
    if n:
        return (base_idx + 1, base_idx + n, "vague" if n > 1 else "relative")
    return None


# --------------------------------------------------------------------------- #
#  metric -> what an "actual" even is                                           #
# --------------------------------------------------------------------------- #
# (financials_3stmt line_item, kind). kind drives the arithmetic:
#   level  — a Rs-cr amount, ACCUMULATED across the elapsed horizon quarters
#   margin — a % level, NOT accumulated: the latest quarter in the window
#   orders — a Rs-cr amount accumulated from announcement order wins, not financials
#   None   — no actuals feed exists anywhere in the repo -> status NO_DATA
METRIC_ACTUAL: dict[str, tuple[str | None, str | None]] = {
    "revenue":     ("Sales", "level"),
    "ebitda":      ("Operating Profit", "level"),
    "pat":         ("Net Profit", "level"),
    "margin":      ("OPM %", "margin"),
    "nim":         ("OPM %", "margin"),      # banks report Financing Margin % here
    "order_book":  (None, "orders"),
    "capex":       (None, "capex"),
    # Guidance is captured for these but NOTHING in the repo stores a matching
    # actual — deck_metrics.parquet has never been written and Q_FIELD drops them.
    # Listed so the commitment stays visible as NO_DATA instead of vanishing.
    "volume":      (None, None),
    "capacity":    (None, None),
    "utilisation": (None, None),
    "loan_aum":    (None, None),
    "deposits":    (None, None),
    "npa":         (None, None),
    "credit_cost": (None, None),
    "returns":     (None, None),
}

# value_type (guidance_value.py) -> how to compare. Anything absent is not a
# measurable number (units, multiples, prose) and yields NO_DATA.
VALUE_TYPE_KIND = {
    "growth_pct": "growth",
    "absolute_inr": "level",
    "absolute_usd": "level",
    "margin_pct": "margin",
    "level_pct": "margin",
    "utilisation_pct": "margin",
    "capacity_pct": "margin",
}


# mgmt_credibility (GF_TRACK) writes the prompt's Title-Case metric names
# ("Order Book", "Working Capital"); guidance_tracker writes the extractor's
# canonical snake_case. Both must land on the same key. Deliberately a LOCAL copy
# rather than importing extract_concall._identify_metric — that module pulls in
# google.genai, which has no business being a dependency of a pure transform (the
# same reason app.py keeps its own copy of _current_india_quarter).
_METRIC_ALIASES = (
    ("order book", "order_book"), ("order intake", "order_book"),
    ("order inflow", "order_book"), ("tcv", "order_book"),
    ("working capital", "working_capital"),
    ("net interest margin", "nim"), ("nim", "nim"),
    ("net profit", "pat"), ("pat", "pat"),
    ("capex", "capex"), ("capacity", "capacity"),
    ("utilisation", "utilisation"), ("utilization", "utilisation"),
    ("revenue", "revenue"), ("sales", "revenue"),
    ("ebitda", "ebitda"), ("margin", "margin"),
    ("volume", "volume"), ("loan", "loan_aum"), ("aum", "loan_aum"),
    ("deposit", "deposits"), ("npa", "npa"), ("credit cost", "credit_cost"),
)


def canon_metric(raw) -> str:
    """'Order Book' / 'order_book' / 'Order Intake' -> 'order_book'.

    Order matters, longest-specific first: "net interest margin" must beat
    "margin", and "order book" must beat the bare metric words after it.
    """
    low = re.sub(r"[_\s]+", " ", str(raw or "")).strip().lower()
    if not low:
        return ""
    for token, name in _METRIC_ALIASES:
        if token in low:
            return name
    return low.replace(" ", "_")


def classify_kind(metric: str, value_type: str) -> str | None:
    """The arithmetic to use, or None when the commitment is not measurable.

    The METRIC wins over the value_type for order_book and capex: an order-inflow
    target is a Rs-cr amount whose actual comes from announcements, never from a P&L
    line, so it must not be routed to the `level` (financials) path.
    """
    metric = str(metric or "").strip().lower()
    _, mkind = METRIC_ACTUAL.get(metric, (None, None))
    if mkind in ("orders", "capex"):
        return mkind if value_type in ("absolute_inr", "absolute_usd") else None
    if mkind is None:
        return None
    vkind = VALUE_TYPE_KIND.get(str(value_type or "").strip())
    if vkind is None:
        return None
    # A margin metric guided as a growth rate is still growth; a revenue metric
    # guided as a % LEVEL in an FY column is not comparable to a Rs-cr line.
    if mkind == "margin" and vkind == "level":
        return None
    if mkind == "level" and vkind == "margin":
        return None
    return vkind


# --------------------------------------------------------------------------- #
#  progress                                                                     #
# --------------------------------------------------------------------------- #

# Two families, and conflating them is the easiest way to get this badly wrong.
#
#   CUMULATIVE — a TOTAL you build up to: Rs 12,000 cr of order inflow, Rs 1,000 cr
#   of revenue. Half the money by half the year is on track, so progress is judged
#   against ELAPSED TIME (pace_ratio).
#
#   RATE — a rate or a level you run AT: 21% growth, a 12.5% EBITDA margin. Nothing
#   accumulates; being 25% through the year does not mean you should have delivered
#   a quarter of a margin. Judged DIRECTLY against the target.
#
# Verified consequence of getting it wrong: POWERMECH running a 10.0% margin against
# guided 12.5% scored pace 80/25 = 3.2 and reported AHEAD — the exact opposite of
# the truth. A rate must never be divided by elapsed time.
CUMULATIVE_KINDS = frozenset({"level", "orders", "capex"})
RATE_KINDS = frozenset({"growth", "margin", "runrate"})

# A Rs-cr LEVEL target over a MULTI-YEAR horizon is an exit RUN-RATE, not a sum.
# "We want to be a Rs 500 cr revenue company by FY30" means annual revenue reaching
# 500, not 500 accumulated over three years. Summing 12 quarters against it put
# BAJAJCON at 399% and SAKSOFT at 333% "ACHIEVED" — arithmetic, not achievement.
# So a level target spanning 5+ quarters is compared to the TRAILING FOUR quarters
# and judged like a rate. An ORDER-INFLOW target over the same horizon really is
# cumulative ("Rs 30,000 cr of inflow over 3 years"), so orders/capex are untouched.
RUNRATE_MIN_QUARTERS = 5

# pace bands for CUMULATIVE targets. Wide ON_TRACK because quarterly revenue is
# lumpy and Indian H2 is seasonally heavier than H1 for most manufacturers — a
# company at 0.9x pace in Q2 is normal, not a warning.
AHEAD, ON_TRACK_LO, BEHIND_LO = 1.15, 0.85, 0.60
# direct bands for RATE targets, in % of the guided rate/level
RATE_AHEAD, RATE_ON_TRACK, RATE_BEHIND = 110.0, 95.0, 80.0


def status_band(pct_of_target, time_pct, kind: str | None,
                horizon_done: bool = False) -> str:
    """ACHIEVED | AHEAD | ON_TRACK | BEHIND | AT_RISK | NO_DATA."""
    if kind is None or pct_of_target is None:
        return "NO_DATA"

    if kind in RATE_KINDS:
        # ACHIEVED only once the horizon is actually over — a rate being met in Q1
        # is "running ahead", not "done".
        if horizon_done and pct_of_target >= 100.0:
            return "ACHIEVED"
        if pct_of_target >= RATE_AHEAD:
            return "AHEAD"
        if pct_of_target >= RATE_ON_TRACK:
            return "ON_TRACK"
        if pct_of_target >= RATE_BEHIND:
            return "BEHIND"
        return "AT_RISK"

    if pct_of_target >= 100.0:
        return "ACHIEVED"
    if not time_pct:
        return "NO_DATA"                 # horizon has not started — nothing to judge
    pace = pct_of_target / time_pct
    if pace >= AHEAD:
        return "AHEAD"
    if pace >= ON_TRACK_LO:
        return "ON_TRACK"
    if pace >= BEHIND_LO:
        return "BEHIND"
    return "AT_RISK"


# A company-level Rs-cr target for a full year cannot be a small fraction of what
# that same line already did last year. Below this ratio the guided number is a
# SEGMENT, PROJECT or SUBSIDIARY figure that the extractor filed under the parent
# metric. Verified: POWERMECH's "INR 150 crores (Tasra)" and "INR 350 crores (KBP)"
# are project revenues, and comparing them to total company revenue of Rs 1,624 cr
# produced "1082% of target — ACHIEVED", which is worse than useless.
# 0.5 is deliberately generous: it still admits a target implying a 50% decline.
SEGMENT_TARGET_MAX = 0.5


def is_segment_target(guided_value, prior_total) -> bool:
    """True when a Rs-cr LEVEL target is too small to be a company-level figure."""
    try:
        g, p = float(guided_value), float(prior_total)
    except (TypeError, ValueError):
        return False
    if g <= 0 or p <= 0:
        return False
    return g < SEGMENT_TARGET_MAX * p


# Larger than any Indian company's annual revenue or order book by a wide margin
# (Reliance turns over ~1,000,000 cr). A "target" above this is a UNITS error, not a
# business: guidance cells carry bare rupee amounts that the parser reads as crore,
# so ASTRAMICRO's "INR 1,600,000,000" (Rs 160 cr written in rupees) became a
# 1.6-billion-crore target and outranked every real commitment in the table.
MAX_PLAUSIBLE_CR = 2_000_000.0


def is_implausible_amount(guided_cr) -> bool:
    """True when a Rs-cr target is too large to be a real company target."""
    try:
        v = float(guided_cr)
    except (TypeError, ValueError):
        return False
    return v > MAX_PLAUSIBLE_CR


# The percentage family, for the fraction fix below.
PCT_VALUE_TYPES = frozenset({"growth_pct", "margin_pct", "level_pct",
                             "utilisation_pct", "capacity_pct"})


def normalise_rate(value_num, raw_text, value_type):
    """A rate/level written as a DECIMAL FRACTION -> percent.

    The extractor is inconsistent about this. CPPLUS carries both "14.5% (EBITDA)"
    and "LVL: 0.125 (EBITDA)" for the SAME company and metric — the second is 12.5%
    written as a fraction. Left alone it compares a 0.125 target against a 15.0
    actual and reports 12,000% of target.

    Only fires when the cell contains no '%' at all AND the number is strictly
    between 0 and 1, so a genuine "1" (meaning 1%) and every "0.5%" are untouched.
    No company guides a 0.125% EBITDA margin.
    """
    if value_type not in PCT_VALUE_TYPES or value_num is None:
        return value_num
    try:
        v = float(value_num)
    except (TypeError, ValueError):
        return value_num
    if "%" in str(raw_text or ""):
        return v
    return v * 100.0 if 0 < v < 1.0 else v


# A realised YoY beyond this is arithmetic off a near-zero base, not a business
# result, and comparing it to a single-digit guided rate is meaningless.
MAX_PLAUSIBLE_RATE_PCT = 500.0
# A guided-vs-actual ratio beyond this means the two numbers are not the same
# kind of thing, whatever the types claimed.
MAX_RATE_RATIO_PCT = 1000.0
# MARGIN is a LEVEL, and levels are bounded: a company guiding a 12% margin does
# not deliver 1% or 120%. So a margin pair that far apart is measuring two
# different things (GICRE showed guided 1.5% against an actual 14.0%). GROWTH gets
# the looser cap above, because guiding 8.75% and delivering 86% is a real result,
# not a broken pair — SYRMA and MTARTECH both did exactly that.
MAX_MARGIN_RATIO_PCT = 300.0
MIN_MARGIN_RATIO_PCT = 20.0
# A CUMULATIVE target beaten by more than 5x is a scale mismatch, not a beat — no
# company delivers five times its own full-year revenue target. This is the
# backstop for segment/project figures the prior-year check could not catch
# because the horizon predates the 12 quarters financials_3stmt retains
# (AARTIPHARM: a Rs 250 cr project target scored against Rs 5,425 cr of revenue).
MAX_LEVEL_RATIO_PCT = 500.0


def commit_id(isin, metric, target_period, guided_text) -> str:
    """Stable identity for one commitment, so week-over-week movement can be
    computed. Deliberately NOT keyed on the source concall: when a company repeats
    the same promise next quarter it must stay the SAME commitment, otherwise every
    repeat would read as brand new and delta_week would always be blank."""
    norm = re.sub(r"\s+", " ", str(guided_text or "")).strip().lower()
    key = f"{isin}|{str(metric).lower()}|{str(target_period).upper()}|{norm}"
    return hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]


def compute_progress(guided_value, kind, window, actuals, prior_actuals=None,
                     elapsed_idx=None):
    """The measurement itself. Pure — every input is a plain value or dict.

    guided_value  : the target, already in its natural unit (Rs cr / % / pp).
    kind          : 'level' | 'growth' | 'margin' | 'orders' | 'capex' | None
    window        : (start_idx, end_idx) from resolve_window.
    actuals       : {quarter_idx: value} for the metric. For 'orders' these are the
                    order wins booked in that quarter; for the rest, the P&L line.
    prior_actuals : {quarter_idx: value} one FY earlier — only used by 'growth'.
    elapsed_idx   : quarters to COUNT AS ELAPSED, when that is not the same thing as
                    "quarters we have a number for".

        This distinction is the whole difference between the two feeds. For a P&L
        metric, a missing quarter means the company has not reported yet — Screener
        lags the filing — so elapsed must follow the DATA or a company that simply
        has not reported reads as though it delivered nothing. For ORDERS the
        opposite holds: a quarter with no order-win announcement means zero booked,
        which is a real and reportable state, so elapsed must follow the CALENDAR.
        Deriving elapsed from the data there would make every company with no orders
        look like it had a horizon that never started.

    Returns dict(actual_to_date, periods_total, periods_elapsed, time_pct,
                 pct_of_target, pace_ratio, status).
    """
    start, end = window
    qs = window_quarters(start, end)
    total = len(qs)
    have = [i for i in qs if actuals.get(i) is not None]
    if elapsed_idx is None:
        elapsed_qs = have
    else:
        elapsed_qs = [i for i in qs if i in set(elapsed_idx)]
    elapsed = len(elapsed_qs)
    time_pct = round(elapsed / total * 100, 2) if total else 0.0

    blank = {"actual_to_date": None, "periods_total": total, "periods_elapsed": elapsed,
             "time_pct": time_pct, "pct_of_target": None, "pace_ratio": None,
             "status": "NO_DATA"}
    if kind is None:
        return blank

    try:
        guided = float(guided_value)
    except (TypeError, ValueError):
        return blank
    if guided == 0:
        return blank                     # no division by a zero target

    eff = kind          # how the number is JUDGED, which is not always its kind
    if kind in ("level", "orders", "capex"):
        # A Rs-cr target accumulates across the horizon. An elapsed quarter with no
        # order win contributes zero rather than being skipped.
        if not have and not (kind == "orders" and elapsed):
            return blank
        if kind == "level" and total >= RUNRATE_MIN_QUARTERS:
            eff = "runrate"
            atd = round(sum(float(actuals[i]) for i in sorted(have)[-4:]), 4)
        else:
            atd = round(sum(float(actuals[i]) for i in have), 4)
        pct = round(atd / guided * 100, 2)
    elif kind == "margin":
        # A % LEVEL is a state, not a running total — the latest quarter in the
        # window IS the answer. Accumulating it would be meaningless.
        if not have:
            return blank
        atd = round(float(actuals[max(have)]), 4)
        pct = round(atd / guided * 100, 2)
    elif kind == "growth":
        # Realised YoY over the elapsed part of the horizon vs the same quarters a
        # year earlier — the like-for-like comparison the guided rate describes.
        prior_actuals = prior_actuals or {}
        pairs = [i for i in have if prior_actuals.get(i - 4) not in (None, 0)]
        if not pairs:
            return blank
        cur = sum(float(actuals[i]) for i in pairs)
        prev = sum(float(prior_actuals[i - 4]) for i in pairs)
        if prev == 0:
            return blank
        atd = round((cur - prev) / abs(prev) * 100, 2)
        pct = round(atd / guided * 100, 2)
    else:
        return blank

    # Incoherent rate pairs: a realised YoY off a near-zero base (HFCL showed
    # +1,378% operating-profit YoY against a guided 3.5%) or a ratio so extreme the
    # two numbers cannot be the same kind of thing. Reported as NO_DATA rather than
    # topping the "AHEAD" list, which is where they were landing.
    if kind in RATE_KINDS and (abs(atd) > MAX_PLAUSIBLE_RATE_PCT
                               or abs(pct) > MAX_RATE_RATIO_PCT):
        return {**blank, "actual_to_date": atd}
    if kind == "margin" and not (MIN_MARGIN_RATIO_PCT <= pct <= MAX_MARGIN_RATIO_PCT):
        return {**blank, "actual_to_date": atd}
    if pct > MAX_LEVEL_RATIO_PCT and (eff in CUMULATIVE_KINDS or eff == "runrate"):
        return {**blank, "actual_to_date": atd}

    # pace is only meaningful for a cumulative target — reporting it for a rate
    # would invite exactly the divide-by-elapsed-time mistake the bands avoid
    pace = (round(pct / time_pct, 3)
            if (time_pct and eff in CUMULATIVE_KINDS) else None)
    return {"actual_to_date": atd, "periods_total": total, "periods_elapsed": elapsed,
            "time_pct": time_pct, "pct_of_target": pct, "pace_ratio": pace,
            "status": status_band(pct, time_pct, eff,
                                  horizon_done=elapsed >= total)}


# --------------------------------------------------------------------------- #
#  self-test — offline, no Drive, no network                                    #
# --------------------------------------------------------------------------- #

def _self_test() -> int:  # noqa: C901
    f = []
    n = [0]

    def eq(got, want, what):
        n[0] += 1
        if got != want:
            f.append(f"  {what}: want {want!r}, got {got!r}")

    # --- quarter arithmetic -------------------------------------------------
    eq(parse_q("Q1 FY27"), (27, 1), "parse_q spaced")
    eq(parse_q("Q1FY2027"), (27, 1), "parse_q 4-digit")
    eq(parse_q("FY27"), None, "parse_q rejects bare FY")
    eq(parse_fy("FY27"), 27, "parse_fy")
    eq(parse_fy("Q1 FY27"), None, "parse_fy rejects quarter")
    eq(q_label(q_idx(27, 1)), "Q1 FY27", "q_label round trip")
    # inverse of quarterly_table.qtr_label: 'Jun 2026' -> 'Q1 FY27'
    eq(q_period(q_idx(27, 1)), "Jun 2026", "Q1 FY27 -> Jun 2026")
    eq(q_period(q_idx(27, 2)), "Sep 2026", "Q2 FY27 -> Sep 2026")
    eq(q_period(q_idx(27, 3)), "Dec 2026", "Q3 FY27 -> Dec 2026")
    eq(q_period(q_idx(27, 4)), "Mar 2027", "Q4 FY27 -> Mar 2027")
    eq(q_idx(27, 2) - q_idx(27, 1), 1, "adjacent quarters differ by 1")
    eq(q_idx(28, 1) - q_idx(27, 4), 1, "FY boundary is contiguous")

    # --- arbitrary dates -> quarter (announcement dates are not month-ends) --
    from datetime import date as _d
    eq(q_from_date(_d(2026, 6, 20)), q_idx(27, 1), "20 Jun 2026 -> Q1 FY27")
    eq(q_from_date(_d(2026, 4, 1)), q_idx(27, 1), "FY starts 1 Apr")
    eq(q_from_date(_d(2026, 8, 14)), q_idx(27, 2), "14 Aug 2026 -> Q2 FY27")
    eq(q_from_date(_d(2026, 7, 1)), q_idx(27, 2), "1 Jul -> Q2")
    eq(q_from_date(_d(2026, 12, 31)), q_idx(27, 3), "31 Dec 2026 -> Q3 FY27")
    eq(q_from_date(_d(2027, 1, 2)), q_idx(27, 4), "Jan is Q4 of the SAME FY")
    eq(q_from_date(_d(2027, 3, 31)), q_idx(27, 4), "FY ends 31 Mar")
    eq(q_from_date(_d(2027, 4, 1)), q_idx(28, 1), "next day is the next FY")

    # --- horizon resolution -------------------------------------------------
    eq(resolve_window("FY27", "Q1 FY27"), (q_idx(27, 1), q_idx(27, 4), "fy"), "FY horizon")
    eq(resolve_window("Q2 FY27", "Q1 FY27"),
       (q_idx(27, 2), q_idx(27, 2), "quarter"), "single quarter")
    # matches build_pead_flags._resolve_horizon: Q3 FY25 + 1Y -> FY26
    eq(resolve_window("1Y", "Q3 FY25"), (q_idx(26, 1), q_idx(26, 4), "relative"), "1Y")
    eq(resolve_window("3Y", "Q1 FY25"), (q_idx(28, 1), q_idx(28, 4), "relative"), "3Y")
    eq(resolve_window("NEXT_QTR", "Q1 FY27"),
       (q_idx(27, 2), q_idx(27, 2), "relative"), "NEXT_QTR")
    eq(resolve_window("NEAR_TERM", "Q1 FY27"),
       (q_idx(27, 2), q_idx(27, 3), "vague"), "NEAR_TERM = 2 quarters")
    eq(resolve_window("LONG_TERM", "Q1 FY27")[1] - resolve_window("LONG_TERM", "Q1 FY27")[0],
       11, "LONG_TERM spans 12 quarters")
    eq(resolve_window("1Y", ""), None, "relative horizon with no anchor")
    eq(resolve_window("", "Q1 FY27"), None, "empty horizon")
    eq(resolve_window("SOMEDAY", "Q1 FY27"), None, "unknown horizon")

    # --- metric canonicalisation (both guidance tables must agree) ----------
    eq(canon_metric("Order Book"), "order_book", "GF_TRACK title case")
    eq(canon_metric("order_book"), "order_book", "already canonical")
    eq(canon_metric("Order Intake"), "order_book", "intake is order book")
    eq(canon_metric("Order Book-TCV"), "order_book", "IT segment spelling")
    eq(canon_metric("Net Interest Margin"), "nim", "NIM beats bare margin")
    eq(canon_metric("Margin"), "margin", "bare margin")
    eq(canon_metric("Working Capital"), "working_capital", "no-feed metric")
    eq(canon_metric("Capex"), "capex", "capex")
    eq(canon_metric(""), "", "empty")
    eq(canon_metric(None), "", "None")

    # --- kind classification ------------------------------------------------
    eq(classify_kind("revenue", "absolute_inr"), "level", "revenue Rs-cr target")
    eq(classify_kind("revenue", "growth_pct"), "growth", "revenue growth target")
    eq(classify_kind("margin", "margin_pct"), "margin", "margin level")
    eq(classify_kind("order_book", "absolute_inr"), "orders", "order target -> orders")
    eq(classify_kind("order_book", "absolute_usd"), "orders", "USD order target")
    eq(classify_kind("order_book", "growth_pct"), None, "order growth % not measurable")
    eq(classify_kind("volume", "absolute_units"), None, "volume has no actuals feed")
    eq(classify_kind("capacity", "capacity_pct"), None, "capacity has no actuals feed")
    eq(classify_kind("revenue", "qualitative"), None, "prose is not measurable")
    eq(classify_kind("revenue", "margin_pct"), None, "Rs-cr line vs % level")

    # --- progress: the POWERMECH case ---------------------------------------
    # guided Rs 12,000 cr of FY27 order inflow; two quarters booked 2,000 + 2,100
    w = resolve_window("FY27", "Q1 FY27")
    p = compute_progress(12000.0, "orders", w[:2],
                         {q_idx(27, 1): 2000.0, q_idx(27, 2): 2100.0})
    eq(p["actual_to_date"], 4100.0, "orders accumulate")
    eq(p["periods_total"], 4, "FY27 is 4 quarters")
    eq(p["periods_elapsed"], 2, "two quarters booked")
    eq(p["time_pct"], 50.0, "half the year elapsed")
    eq(p["pct_of_target"], 34.17, "4100/12000")
    eq(p["status"], "BEHIND", "0.68 pace sits in the BEHIND band, not AT_RISK")

    # a level target keeping pace
    p = compute_progress(1000.0, "level", w[:2],
                         {q_idx(27, 1): 260.0, q_idx(27, 2): 250.0})
    eq(p["pct_of_target"], 51.0, "510/1000")
    eq(p["status"], "ON_TRACK", "1.02 pace")

    # ACHIEVED wins over pace
    p = compute_progress(100.0, "level", w[:2], {q_idx(27, 1): 140.0})
    eq(p["status"], "ACHIEVED", "past the target")
    p = compute_progress(100.0, "level", w[:2], {q_idx(27, 1): 480.0})
    eq(p["status"], "ACHIEVED", "a genuine 4.8x overshoot is still a beat")
    # ...but 5x+ is a scale mismatch, not a beat (AARTIPHARM: Rs 250 cr project
    # target scored against Rs 5,425 cr of company revenue)
    p = compute_progress(250.0, "level", w[:2], {q_idx(27, 1): 5425.0})
    eq(p["status"], "NO_DATA", "21x is a scale mismatch, not an achievement")
    eq(p["actual_to_date"], 5425.0, "...the observed number is still reported")

    # margin is a LEVEL: the latest quarter, never a sum
    p = compute_progress(20.0, "margin", w[:2],
                         {q_idx(27, 1): 18.0, q_idx(27, 2): 19.0})
    eq(p["actual_to_date"], 19.0, "margin takes the latest quarter")
    eq(p["pct_of_target"], 95.0, "19/20")
    eq(p["status"], "ON_TRACK", "95% of a rate target is on track")
    eq(p["pace_ratio"], None, "a rate has no pace")

    # THE REGRESSION THAT MATTERS: a rate must never be divided by elapsed time.
    # POWERMECH ran a 10.0% margin against guided 12.5% one quarter into FY27.
    # Pace maths gave 80/25 = 3.2 -> AHEAD, the exact opposite of the truth.
    p = compute_progress(12.5, "margin", w[:2], {q_idx(27, 1): 10.0})
    eq(p["pct_of_target"], 80.0, "10.0 vs 12.5")
    eq(p["time_pct"], 25.0, "one quarter of four")
    eq(p["status"], "BEHIND", "BELOW a margin target is never AHEAD")

    # growth compares like-for-like against the same quarters a year earlier
    p = compute_progress(20.0, "growth", w[:2],
                         {q_idx(27, 1): 120.0, q_idx(27, 2): 130.0},
                         {q_idx(26, 1): 100.0, q_idx(26, 2): 110.0})
    eq(p["actual_to_date"], 19.05, "realised YoY over elapsed quarters")
    eq(p["status"], "ON_TRACK", "95% of the guided rate, judged directly")

    # a rate beaten mid-horizon is AHEAD, not ACHIEVED — the year is not over
    p = compute_progress(21.5, "growth", w[:2], {q_idx(27, 1): 125.6},
                         {q_idx(26, 1): 100.0})
    eq(p["status"], "AHEAD", "25.6% vs 21.5% guided, one quarter in")
    # ...but once every quarter is in, meeting it IS achieving it
    p = compute_progress(20.0, "margin", w[:2],
                         {q_idx(27, 1): 21.0, q_idx(27, 2): 21.0,
                          q_idx(27, 3): 21.0, q_idx(27, 4): 21.0})
    eq(p["status"], "ACHIEVED", "rate met with the horizon complete")

    # --- multi-year LEVEL targets are exit run-rates, not sums ---------------
    # BAJAJCON: "Rs 500 cr revenue" over a 3-year horizon means annual revenue
    # reaching 500, not 500 summed across 12 quarters.
    long_w = (q_idx(27, 1), q_idx(29, 4))          # 12 quarters
    acts = {q_idx(27, 1) + i: 125.0 for i in range(8)}   # 8 quarters at 125 = 500/yr
    p = compute_progress(500.0, "level", long_w, acts)
    eq(p["periods_total"], 12, "3-year horizon")
    eq(p["actual_to_date"], 500.0, "trailing FOUR quarters, not all eight")
    eq(p["pct_of_target"], 100.0, "run-rate exactly at target")
    eq(p["pace_ratio"], None, "a run-rate is judged like a rate, not on pace")
    # the same numbers summed cumulatively would have read 1000/500 = 200%
    eq(sum(acts.values()), 1000.0, "cumulative sum would have doubled it")
    # a single-FY level target still accumulates
    p = compute_progress(500.0, "level", w[:2],
                         {q_idx(27, 1): 125.0, q_idx(27, 2): 125.0})
    eq(p["actual_to_date"], 250.0, "an FY target still sums its quarters")
    eq(p["status"], "ON_TRACK", "half the money at half the year")
    # an ORDER target over 3 years really is cumulative
    p = compute_progress(30000.0, "orders", long_w,
                         {q_idx(27, 1) + i: 2500.0 for i in range(8)},
                         elapsed_idx={q_idx(27, 1) + i for i in range(8)})
    eq(p["actual_to_date"], 20000.0, "order inflow accumulates over the horizon")

    # --- degenerate inputs must not raise -----------------------------------
    eq(compute_progress(0.0, "level", w[:2], {q_idx(27, 1): 5.0})["status"],
       "NO_DATA", "zero target -> no division")
    eq(compute_progress(None, "level", w[:2], {})["status"], "NO_DATA", "None target")
    eq(compute_progress("n/a", "level", w[:2], {})["status"], "NO_DATA", "text target")
    eq(compute_progress(100.0, None, w[:2], {})["status"], "NO_DATA", "unmeasurable kind")
    eq(compute_progress(100.0, "level", w[:2], {})["status"], "NO_DATA", "no actuals yet")
    eq(compute_progress(100.0, "level", w[:2], {})["periods_elapsed"], 0, "zero elapsed")
    # an order commitment with nothing booked yet is 0%, NOT unknown — that is a
    # real, reportable state and the whole point of tracking mid-flight. Elapsed
    # comes from the CALENDAR here, so half a year with no wins reads as at risk.
    z = compute_progress(12000.0, "orders", w[:2], {},
                         elapsed_idx={q_idx(27, 1), q_idx(27, 2)})
    eq(z["actual_to_date"], 0.0, "no orders booked = zero, not null")
    eq(z["periods_elapsed"], 2, "calendar drives elapsed for orders")
    eq(z["status"], "AT_RISK", "nothing booked at half-time reads as at risk")
    # ...but before the horizon starts there is nothing to judge
    eq(compute_progress(12000.0, "orders", w[:2], {}, elapsed_idx=set())["status"],
       "NO_DATA", "horizon not started")
    # a P&L metric must NOT do that: a company that has not reported is unknown,
    # not a zero — Screener lags the filing by weeks
    eq(compute_progress(1000.0, "level", w[:2], {})["status"],
       "NO_DATA", "unreported quarter is not a zero")
    # elapsed can exceed reported: booked 2,000 in Q1, Q2 ended with no win
    y = compute_progress(12000.0, "orders", w[:2], {q_idx(27, 1): 2000.0},
                         elapsed_idx={q_idx(27, 1), q_idx(27, 2)})
    eq(y["periods_elapsed"], 2, "silent quarter still counts as elapsed")
    eq(y["pct_of_target"], 16.67, "2000/12000")
    eq(compute_progress(20.0, "growth", w[:2], {q_idx(27, 1): 120.0}, {})["status"],
       "NO_DATA", "growth with no prior year")
    eq(compute_progress(20.0, "growth", w[:2],
                        {q_idx(27, 1): 120.0}, {q_idx(26, 1): 0.0})["status"],
       "NO_DATA", "growth off a zero base")

    # --- segment/project targets filed under the parent metric ---------------
    # POWERMECH FY26 revenue totalled 6,062 cr; "INR 150 crores (Tasra)" is a
    # project, not a company target
    eq(is_segment_target(150.0, 6062.0), True, "project revenue vs company total")
    eq(is_segment_target(350.0, 6062.0), True, "second project target")
    eq(is_segment_target(7000.0, 6062.0), False, "a genuine growth target")
    eq(is_segment_target(6000.0, 6062.0), False, "a flat-ish target is genuine")
    eq(is_segment_target(3100.0, 6062.0), False, "even a big decline is genuine")
    eq(is_segment_target(150.0, None), False, "no prior year -> cannot judge")
    eq(is_segment_target(150.0, 0), False, "zero prior year -> cannot judge")
    eq(is_segment_target(None, 6062.0), False, "no target -> nothing to judge")

    # --- units errors dressed up as enormous targets --------------------------
    # ASTRAMICRO "INR 1,600,000,000" is Rs 160 cr written in rupees
    eq(is_implausible_amount(1_600_000_000.0), True, "rupees read as crore")
    eq(is_implausible_amount(55_000.0), False, "BEL's real 55,000 cr inflow target")
    eq(is_implausible_amount(1_000_000.0), False, "Reliance-scale is still plausible")
    eq(is_implausible_amount(None), False, "no value -> nothing to judge")
    eq(is_implausible_amount("n/a"), False, "text -> nothing to judge")

    # --- margins written as decimal fractions --------------------------------
    # CPPLUS carries both spellings for the same metric
    eq(normalise_rate(0.125, "LVL: 0.125 (EBITDA)", "margin_pct"), 12.5, "fraction")
    eq(normalise_rate(0.08, "LVL: 0.08 (PAT)", "margin_pct"), 8.0, "fraction PAT")
    eq(normalise_rate(14.5, "14.5% (EBITDA)", "margin_pct"), 14.5, "already percent")
    eq(normalise_rate(0.5, "0.5%", "margin_pct"), 0.5, "'%' present -> untouched")
    eq(normalise_rate(75.0, "LVL: 75% (utilisation)", "utilisation_pct"), 75.0,
       "utilisation level untouched")
    eq(normalise_rate(1.0, "LVL: 1", "margin_pct"), 1.0, "a bare 1 is 1%, not 100%")
    eq(normalise_rate(530.0, "INR 530 crores", "absolute_inr"), 530.0,
       "amounts are never rescaled")
    eq(normalise_rate(None, "x", "margin_pct"), None, "None passes through")

    # --- incoherent rate pairs ------------------------------------------------
    # HFCL: +1,378% operating-profit YoY off a near-zero base vs a guided 3.5%
    p = compute_progress(3.5, "growth", w[:2], {q_idx(27, 1): 1478.0},
                         {q_idx(26, 1): 100.0})
    eq(p["status"], "NO_DATA", "YoY off a near-zero base is not comparable")
    eq(p["actual_to_date"], 1378.0, "...but the realised number is still reported")
    # a large-but-real YoY against a matching target stays measurable
    p = compute_progress(40.0, "growth", w[:2], {q_idx(27, 1): 145.0},
                         {q_idx(26, 1): 100.0})
    eq(p["status"], "AHEAD", "45% vs guided 40% is a real beat")
    # SYRMA guided 8.75% EBITDA growth and delivered 86% — a real result, kept
    p = compute_progress(8.75, "growth", w[:2], {q_idx(27, 1): 186.21},
                         {q_idx(26, 1): 100.0})
    eq(p["status"], "AHEAD", "a big genuine growth beat survives")

    # MARGIN is bounded, so a pair that far apart is two different measures.
    # GICRE: guided 1.5% against an actual 14.0% margin.
    p = compute_progress(1.5, "margin", w[:2], {q_idx(27, 1): 14.0})
    eq(p["status"], "NO_DATA", "a 9x margin gap is not a margin comparison")
    p = compute_progress(60.0, "margin", w[:2], {q_idx(27, 1): 6.0})
    eq(p["status"], "NO_DATA", "...and neither is a 10x gap the other way")
    p = compute_progress(20.0, "margin", w[:2], {q_idx(27, 1): 14.0})
    eq(p["status"], "AT_RISK", "a real margin shortfall is still measured")

    # --- commit_id stability ------------------------------------------------
    a = commit_id("INE1", "order_book", "FY27", "INR 12,000 crores inflow")
    b = commit_id("INE1", "order_book", "FY27", "INR 12,000  crores   inflow ")
    c = commit_id("INE1", "order_book", "FY27", "INR 12,000 crores")
    eq(a, b, "whitespace-insensitive")
    eq(a != c, True, "different text -> different id")
    eq(len(a), 16, "id length")

    if f:
        print(f"FAIL {len(f)}/{n[0]} checks")
        print("\n".join(f))
        return 1
    print(f"OK  all {n[0]} guidance_progress checks pass")
    return 0


if __name__ == "__main__":
    import sys
    raise SystemExit(_self_test() if "--self-test" in sys.argv else _self_test())
