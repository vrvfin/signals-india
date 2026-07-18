r"""
gradation.py — shared 6-rule attractiveness grading + color tiers.

One source of truth for build_screener_grades.py (writes tiers into a parquet) and
app.py (renders the color heatmap). Each metric maps a raw value to a TIER; tiers
map to a fixed color. Higher-is-better metrics (growth, ROE, CFO) and lower-is-better
metrics (valuation) are handled per-metric. "green" = Good/Great/Exceptional.

Tiers (best -> worst): exceptional, great, good, ok, decent, poor, na
"""
from __future__ import annotations

GREEN_TIERS = ("good", "great", "exceptional")

TIER_COLOR = {
    "exceptional": "#1a7a3a",   # dark green
    "great":       "#4caf50",
    "good":        "#a5d6a7",   # light green
    "ok":          "#ffe0b2",   # light amber
    "decent":      "#ffb74d",   # amber
    "poor":        "#ef9a9a",   # red
    "na":          "#eeeeee",   # grey / no data
}
TIER_RANK = {"exceptional": 5, "great": 4, "good": 3, "ok": 2, "decent": 1,
             "poor": 0, "na": -1}

# Text colour per tier — the light-amber/green/red backgrounds read fine with
# near-black text, but "exceptional" (#1a7a3a, dark green) needs white or the
# cell is black-on-dark and unreadable in email clients.
TIER_TEXT = {"exceptional": "#ffffff"}     # all others default to near-black below


def cell_css(tier: str) -> str:
    """Return 'background:..;color:..;' for a graded cell, or '' when the tier
    has no colour (na/unknown -> caller's default styling). One source of truth
    so every mail/heatmap cell stays legible."""
    bg = TIER_COLOR.get(tier)
    if not bg:
        return ""
    return f"background:{bg};color:{TIER_TEXT.get(tier, '#1a1a1a')};"


def _num(v):
    try:
        f = float(v)
        return f if f == f else None      # drop NaN
    except (TypeError, ValueError):
        return None


def _bucket_high(v, thr):
    """Higher-is-better: thr = (exceptional, great, good, ok, decent) lower-bounds."""
    if v is None:
        return "na"
    e, g, gd, ok, dc = thr
    if v >= e: return "exceptional"
    if v >= g: return "great"
    if v >= gd: return "good"
    if v >= ok: return "ok"
    if v >= dc: return "decent"
    return "poor"


def _bucket_low(v, thr):
    """Lower-is-better: thr = (exceptional, great, good, ok, decent) upper-bounds."""
    if v is None or v <= 0:               # negative/zero PE/PB/EV is not meaningful
        return "na"
    e, g, gd, ok, dc = thr
    if v < e: return "exceptional"
    if v < g: return "great"
    if v < gd: return "good"
    if v < ok: return "ok"
    if v < dc: return "decent"
    return "poor"


# Growth %: YOY, QOQ, Guidance all share this
GROWTH_THR = (100, 50, 30, 15, 0)          # >=100 exc, 50 great, 30 good, 15 ok, 0 decent, <0 poor


def grade_growth(v):
    return _bucket_high(_num(v), GROWTH_THR)


def grade_roe(v):
    return _bucket_high(_num(v), (30, 20, 15, 10, 0))


def grade_cfo(cfo_pat_ratio, cfo_positive=None):
    """CFO quality from CFO/PAT ratio. cfo_positive overrides sign if known."""
    r = _num(cfo_pat_ratio)
    if cfo_positive is False:
        return "poor"
    if r is None:
        return "good" if cfo_positive else "na"   # positive but ratio unknown
    if r >= 1.2: return "exceptional"
    if r >= 0.8: return "great"
    if r > 0:    return "good"
    return "poor"                                   # CFO opposite sign to PAT / negative


def grade_valuation(value, kind):
    """kind in {'pe','pb','ev_ebitda'}; lower is better."""
    thr = {"pe": (5, 15, 25, 40, 60),
           "pb": (1.0, 1.5, 3.0, 4.0, 6.0),
           "ev_ebitda": (5, 8, 10, 14, 20)}.get(kind)
    return _bucket_low(_num(value), thr) if thr else "na"


# Which valuation metric applies, by sector/segment text
_FINANCE = ("bank", "financ", "nbfc", "insurance", "lending", "broking", "amc",
            "capital market")
_ASSET_HEAVY = ("metal", "utilit", "power", "infra", "cement", "realty", "real estate",
                "capital good", "oil", "gas", "shipping", "mining", "construction")


def valuation_kind(sector_text: str) -> str:
    s = (sector_text or "").lower()
    if any(k in s for k in _FINANCE):
        return "pb"
    if any(k in s for k in _ASSET_HEAVY):
        return "ev_ebitda"
    return "pe"


def green_count(tiers: list[str]) -> int:
    return sum(1 for t in tiers if t in GREEN_TIERS)
