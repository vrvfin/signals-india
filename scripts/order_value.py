r"""
order_value.py — pull the Rs-crore value out of an order-win filing (user 2026-08-15).

WHY THIS EXISTS
    `ingest_announcements.py` already tags a BSE filing `event_type='order_win'` and
    stores a 3-5 line summary, but it never captures the AMOUNT. So the repo knows a
    company won an order and not how big it was — which makes "management guided
    Rs 12,000 cr of FY27 order inflow, how much has actually landed?" unanswerable.
    Nothing else stores it either: extract_concall's Q_FIELD recognises an order-book
    actual and drops it, and deck_metrics.parquet has never been written.

    The source PDF is deleted after 2 days (retention rule), so the headline and the
    stored summary are the ONLY text that survives. This parses those.

WHAT IT DOES
    Finds every currency amount in the text and picks the one that is actually the
    order value, preferring amounts anchored to an order-value cue ("worth",
    "valued at", "aggregating to") over incidental figures (an existing order book,
    a capex number, a turnover comparison). Currency + magnitude handling is reused
    from guidance_value.py, so "USD 30-75 m" and "~Rs 1,437 cr" scale the same way
    here as they do for guidance — one currency implementation, not two.

    Returns None rather than guessing when no amount is present: an unparsed order
    counts as zero downstream, which biases progress DOWN, and a silent zero is much
    safer than a fabricated crore figure.

Usage:
    from order_value import parse_order_value
    parse_order_value("won an order worth Rs 1,008.90 Crores from JSW")
    # {'value_cr': 1008.9, 'currency': 'INR', 'matched': 'Rs 1,008.90 Crores', ...}

    python scripts/order_value.py --self-test
"""
from __future__ import annotations

import re

from guidance_value import _money, _numbers, _RE_RANGE, _to_float

# BSE filings and the LLM summary are full of typographic unicode. Verified present
# in live rows: U+2011 NON-BREAKING HYPHEN inside "USD‑30‑75‑m",
# "5‑year", "1,215‑MW", "~₹1,437‑cr", plus nbsp/thin spaces.
# Written as escapes, not literals, so the source stays diff-safe in any editor.
#
# The dash family needs DIGIT-AWARE handling, which is the whole subtlety here. The
# same U+2011 is a range separator between two digits ("30‑75") and a plain word
# joiner everywhere else ("1,437‑cr", "5‑year"). Mapping it all to "-"
# keeps the range but welds the magnitude onto the number, so "-cr" stops reading as
# crore; mapping it all to " " fixes the magnitude and destroys the range. So:
# between digits it becomes "-", everywhere else a space.
_SPACES = "    ​"
_DASHES = "‐‑‒–—―−"
_RE_SPACES = re.compile(f"[{_SPACES}]")
_RE_DASH_RANGE = re.compile(rf"(?<=\d)\s*[{_DASHES}]\s*(?=\d)")
_RE_DASH_ANY = re.compile(f"[{_DASHES}]")

_NUM = r"\d[\d,]*(?:\.\d+)?"
# A currency marker is REQUIRED. A bare "255 Crores" with no Rs/INR/$ nearby is far
# more often a turnover or a market size than the order being announced.
# "m"/"bn" are accepted bare here (but never in guidance_value) because a currency
# marker already anchors the amount: "USD 30-75 m" can only mean millions.
_RE_AMOUNT = re.compile(
    r"(?:₹|\bRs\.?|\bINR\b|\bUSD\b|\bUS\s?\$|\$)[\s~-]*"
    rf"{_NUM}"
    rf"(?:\s*(?:-|to)\s*(?:₹|\bRs\.?|\bINR\b|\bUSD\b|\$)?\s*{_NUM})?"
    r"[\s-]*(?:crores?|crs?\b|cr\b|lakhs?|millions?|mn\b|m\b|billions?|bn\b|b\b"
    r"|thousand|k\b)?",
    re.I)

# Phrases that introduce THE order value. A hit within _CUE_WINDOW characters before
# the amount marks it as the headline number rather than incidental context.
_RE_CUE = re.compile(
    r"\b(?:worth|valued\s+at|value\s+of|order\s+(?:of|for)|contract\s+(?:of|for)|"
    r"totall?ing|aggregating(?:\s+to)?|amounting\s+to|for\s+a\s+total|"
    r"letter\s+of\s+(?:award|intent)|\bloa\b|\bwork\s+order\b|bagged|secured|"
    r"awarded|received\s+an?\s+order)\b", re.I)
_CUE_WINDOW = 60

# Amounts that are explicitly NOT the new order — an existing book, a past figure,
# or a forward projection. Checked in the same window as the cue.
_RE_ANTI = re.compile(
    r"\b(?:order\s+book|orderbook|outstanding\s+order|unexecuted|backlog|"
    r"revenue\s+of|turnover\s+of|market\s+cap|previous\s+year|last\s+year|"
    r"compared\s+(?:to|with)|versus|\bvs\.?\b|capex|market\s+size)\b", re.I)


def _norm(text: str) -> str:
    s = str(text or "")
    if not s:
        return ""
    s = _RE_SPACES.sub(" ", s)
    s = _RE_DASH_RANGE.sub("-", s)      # "30<nbhyph>75" -> "30-75"  (a range)
    s = _RE_DASH_ANY.sub(" ", s)        # "1,437<nbhyph>cr" -> "1,437 cr"
    return re.sub(r"\s+", " ", s)


def parse_order_value(text) -> dict | None:
    """The order value in `text`, or None when there is no usable amount.

    Returns dict(value_cr, currency, value_native, unit, matched, cued).
    `value_cr` is always Rs crore whatever currency was written, so callers can sum
    across filings without thinking about it.
    """
    s = _norm(text)
    if not s:
        return None

    best = None
    for m in _RE_AMOUNT.finditer(s):
        frag = m.group(0)
        nums = _numbers(frag)
        if not nums:
            continue
        # a range ("USD 30-75 m") is one order, take its midpoint
        rng = _RE_RANGE.search(frag)
        if rng and _to_float(rng.group(1)) is not None and _to_float(rng.group(2)) is not None:
            base = (_to_float(rng.group(1)) + _to_float(rng.group(2))) / 2
        else:
            base = nums[0]

        vtype, native, unit, inr_cr = _money(_mag_hint(frag), base)
        if inr_cr is None or inr_cr <= 0:
            continue

        lo = max(0, m.start() - _CUE_WINDOW)
        before = s[lo:m.start()]
        if _RE_ANTI.search(before):
            continue                        # an existing book / a comparison, not this order
        cued = bool(_RE_CUE.search(before))
        # Prefer a cued amount; among equals prefer the LARGER, since a filing that
        # mentions two cued amounts is usually stating the total and a component.
        rank = (1 if cued else 0, inr_cr)
        if best is None or rank > best[0]:
            best = (rank, {
                "value_cr": round(inr_cr, 4),
                "currency": "USD" if vtype == "absolute_usd" else "INR",
                "value_native": native,
                "unit": unit,
                "matched": frag.strip(),
                "cued": cued,
            })
    return best[1] if best else None


_RE_BARE_M = re.compile(r"(\d)\s*m\b", re.I)
_RE_BARE_B = re.compile(r"(\d)\s*b\b", re.I)


def _mag_hint(frag: str) -> str:
    """Spell a bare trailing "m"/"b" out for guidance_value._magnitude.

    guidance_value deliberately does NOT treat a lone "m" as millions — there it
    would collide with metres, MW and MT. Inside a currency-anchored order amount
    the reading is unambiguous, so the expansion happens here and the currency
    scaling itself stays in one place.
    """
    f = _RE_BARE_B.sub(r"\1 bn", frag)
    return _RE_BARE_M.sub(r"\1 mn", f)


# --------------------------------------------------------------------------- #
#  self-test — offline, no Drive, no network                                    #
# --------------------------------------------------------------------------- #

# Every case below is real text from live announcement_ledger order_win summaries,
# including the exact U+2011 / nbsp characters the live rows carry.
_CASES = [
    ("This filing announces Power Mech Projects has won a significant order worth "
     "₹1,008.90 Crores from JSW Thermal Energy Limited for civil works.", 1008.90),
    ("The ₹66.11 crore contract for building 10 hybrid electric ferries is a "
     "key catalyst.", 66.11),
    ("order win for Goodluck India's subsidiary for Rs. 255 Crores for 155mm shells.", 255.0),
    ("declared the Lowest Bidder (L1) for a 5-year contract on NH-502A in Mizoram, "
     "valued at ₹70.18 Crores.", 70.18),
    ("announced receipt of a domestic order worth approximately INR 26.31 Crores for "
     "RAPH Rotor Assembly supply.", 26.31),
    # non-breaking hyphens + tilde, exactly as stored
    ("secured a 5‑year O&M contract for Vedanta Aluminium's 1,215‑MW captive "
     "power plant, valued at ~₹1,437‑cr.", 1437.0),
    ("announcing new contracts for its subsidiary, STEAG India, totaling "
     "₹5,100 Cr.", 5100.0),
    # USD converts to Rs cr; the range takes its midpoint (30-75 -> 52.5 mn -> 462 cr)
    ("award of a 'Large' international contract to expand the Donauinsel Water Works "
     "in Vienna. The order (USD‑30‑75‑m) adds significant revenue.", 462.0),
    ("bagged an export order of USD 12 million from a European customer.", 105.6),
    # no amount at all -> None, never a guess
    ("win of a 'Mega' SWRO desalination plant project in Kuwait, marking its entry "
     "into the country.", None),
    ("reporting the successful commissioning of its Kavach Version 4.0 safety system "
     "on a 207‑km North Central Railway stretch.", None),
    ("The order is expected to increase topline by over 20% next year.", None),
    # the cue wins over a bigger incidental figure
    ("Total order book stands at Rs 20,000 crore. The new order is worth "
     "Rs 850 crore.", 850.0),
    # an existing-book figure alone is rejected outright
    ("The company's outstanding order book of Rs 15,000 crore provides visibility.",
     None),
    ("Revenue of Rs 4,200 crore was reported for the year.", None),
]


def _self_test() -> int:
    fails, n = [], 0
    for text, want in _CASES:
        n += 1
        got = parse_order_value(text)
        val = got["value_cr"] if got else None
        ok = (want is None and val is None) or (
            want is not None and val is not None and abs(val - want) < 0.6)
        if not ok:
            fails.append(f"  want {want}, got {val}   <- {text[:70]!r}"
                         + (f"\n      matched {got['matched']!r}" if got else ""))
    # currency is reported, not just folded away
    n += 1
    u = parse_order_value("bagged an export order of USD 12 million")
    if not u or u["currency"] != "USD" or u["value_native"] != 12.0:
        fails.append(f"  USD not reported natively: {u}")
    n += 1
    i = parse_order_value("order worth Rs 500 crore")
    if not i or i["currency"] != "INR":
        fails.append(f"  INR not reported: {i}")
    n += 1
    if parse_order_value("") is not None or parse_order_value(None) is not None:
        fails.append("  empty input must be None")
    # normalisation must not corrupt a plain ASCII string
    n += 1
    if _norm("order worth Rs 500 crore") != "order worth Rs 500 crore":
        fails.append("  ASCII text must pass through _norm unchanged")

    if fails:
        print(f"FAIL {len(fails)}/{n}")
        print("\n".join(fails))
        return 1
    print(f"OK  all {n} order_value cases pass")
    return 0


if __name__ == "__main__":
    import sys
    if "--self-test" in sys.argv:
        raise SystemExit(_self_test())
    for a in sys.argv[1:]:
        print(f"{a!r}\n  {parse_order_value(a)}")
