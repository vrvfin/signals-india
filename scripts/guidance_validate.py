r"""
guidance_validate.py — check a parsed guidance number against what management
actually said, using GF1 as the transcript-of-record.

Pure functions, stdlib + pandas only. No Drive, no network, no Gemini.
Offline unit-testable:  python scripts/guidance_validate.py --self-test

WHY THIS EXISTS (user decision, 2026-08-21)
-------------------------------------------
No capping. The repo's own MAX_GUIDANCE_CAGR=200 would block a genuine
high-growth name just as readily as a misparse, and a fixed ceiling cannot tell
the two apart. So instead of asking "is this number too big?", we ask "did
management actually say it?" — and if the evidence agrees, the exact CAGR is
published however large.

Raw concall PDFs are deleted 2 days after processing (global retention rule 1),
so the durable record of what was said is gf1_guidance_statements.parquet:
173,865 verbatim statements across 1,647 companies, each with metric_type,
timeframe, quantifiable, numeric_value and range_val.

    VERDICT        MEANING                                        ACTION
    CONFIRMED      figure agrees with a verbatim statement (<=1%)  publish, uncapped
    CONSISTENT     within tolerance of the statement (<=10%)       publish, uncapped
    CONTRADICTED   statements exist and all disagree               reject
    NO_EVIDENCE    no statement for that (isin, quarter, metric)   hold for review

COVERAGE, measured live 2026-08-21 — join rate of Table_A revenue/PAT rows to a
GF1 statement on (isin, quarter): 78.0% overall, revenue 82.0%, PAT 57.7%.
So ~4 in 10 PAT candidates cannot be validated and land in NO_EVIDENCE. That is
a real limit of the evidence, not a bug: publish them with an "unverified" chip
only if you decide the coverage cost outweighs the false-positive risk.

TWO DATA-QUALITY TRAPS in GF1, both handled here:
  * `metric_type` is FREE TEXT, not a controlled vocabulary — real values include
    '10,000', '12% CAGR"' and ':3' alongside ARR / AUM / ARPU. Match by REGEX,
    never by equality.
  * `quantifiable` has 10 spellings (Yes/YES/TRUE/True/Quantifiable/No/NO/FALSE/
    NA/None). Normalise before use.
"""
from __future__ import annotations

import argparse
import re
import sys

import pandas as pd

import guidance_value as GV

CONFIRMED, CONSISTENT = "CONFIRMED", "CONSISTENT"
CONTRADICTED, NO_EVIDENCE = "CONTRADICTED", "NO_EVIDENCE"
SEGMENT_SCOPED, NEGATED = "SEGMENT_SCOPED", "NEGATED"

PUBLISHABLE = (CONFIRMED, CONSISTENT)

# --------------------------------------------------------------------------- #
#  scope: is the growth claim about the COMPANY, or only part of it?            #
# --------------------------------------------------------------------------- #
# Measured on the 31 published Q1FY27 rows (2026-08-22): 6 were segment-level and
# the Table_A CELL almost never says so -- Info Edge's cell is a bare "100%",
# and only the quote reveals "...more than double this year in Job Hai", which
# is one product, not the company. So this test HAS to run on the statement,
# which means it belongs at validation time, not in the cell cleaner.
_RE_SCOPE_NOUN = re.compile(
    r"\b(segments?|divisions?|verticals?|business unit|SBU|sub-?segment"
    r"|category|franchise|product line)\b", re.I)
# "our branded alco-bev business", "the consumer business"
_RE_OUR_BIZ = re.compile(
    r"\b(?:our|the|its)\s+[\w&/,'-]+(?:\s+[\w&/,'-]+){0,3}\s+business\b", re.I)
# "<subject> will see 125% to 150%" — a named part of the business as the subject
_RE_SUBJ_SEE = re.compile(
    r"^[\"']?\s*([a-z][\w &/-]{2,24})\s+will\s+(?:see|grow|deliver|reach)\b")
# "...double this year in Job Hai."  A trailing "in <Proper Noun>".
_RE_IN_PROPER = re.compile(
    r"\bin\s+([A-Z][\w&.-]*(?:\s+[A-Z][\w&.-]*){0,2})\s*[.,;\"']*\s*$")
# A DATE is not a business. Without this, "we expect INR7,500 crores in FY28"
# reads as scoped to a segment called FY28.
_RE_TIME_TOK = re.compile(
    r"^(?:FY\s?\d{2,4}|Q[1-4]|H[12]|\d{4}|Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep"
    r"|Oct|Nov|Dec|India|The|This|Coming|Next|Current)\b", re.I)
_RE_GROWTH_TOK = re.compile(
    r"\d+(?:\.\d+)?\s*%|\bdouble|\btriple|\bgrow\w*|\bincreas\w*|\bexpand\w*", re.I)
# ENTITY-wide markers that veto a scope flag. NOTE: "full year" / "for the year"
# are deliberately NOT here -- they scope TIME, not entity, and including them
# let "our branded alco-bev business ... for the full year" through.
_RE_COMPANY_WIDE = re.compile(
    r"\b(overall|consolidated|company[- ]?(?:wide|level)|total revenue"
    r"|our revenues?|top ?line|turnover|blended|at a company"
    r"|in total|as a whole|our business\b)\b", re.I)
# "driven by X", "attributed to X" introduce a CAUSE, not a scope. Held-out test
# (2026-08-22) caught two false positives that turn on exactly this:
#   ANURAS    "a 50%+ growth in FY26 largely attributed due to ... pharma segment"
#   SERVOTECH "In total ... double our business, driven by these two key verticals"
# Both are company-wide claims that merely NAME segments as contributors.
_RE_ATTRIBUTION = re.compile(
    r"\b(driven by|attributed (?:to|due)|due to|on account of|led by"
    r"|contributed by|thanks to|because of|backed by|aided by|supported by)\b", re.I)
# A scope noun only scopes the growth when it is the SUBJECT — i.e. it appears
# BEFORE the growth claim and close to it ("the electrolyte salt segment to grow
# by at least 200%"), not trailing after it as an explanation.
SCOPE_NOUN_LOOKAHEAD = 60
# How close the "in <Proper>" must sit to the growth claim to be scoping IT.
# Measured: Info Edge "double ... in Job Hai" = 11 chars; Fujiyama
# "...70% ... capacities we are ready with in Ratlam" = 88 chars, where Ratlam
# is a PLANT, not the scope of the 70%.
SCOPE_NEAR_CHARS = 45

# "not to the extent we are intending... Not by the amount of 50%" — the sentence
# says they will MISS the number the cell claims.
_RE_NEGATION = re.compile(
    r"\bnot\s+(?:by|to|at|able|going|expect|likely|achiev\w*)\b"
    r"|\bshort\s+of\b|\bfall\s+short\b|\bwon'?t\b|\bunlikely\b", re.I)


def scope_is_segment(stmt: str):
    """Why this statement is only about PART of the company, or None."""
    s = str(stmt or "").strip()
    if not s or _RE_COMPANY_WIDE.search(s):
        return None
    growth_at = [g.start() for g in _RE_GROWTH_TOK.finditer(s)]
    m = _RE_SCOPE_NOUN.search(s)
    if m:
        g0 = min(growth_at) if growth_at else len(s)
        before = m.start() < g0                      # the segment is the subject
        near = abs(m.start() - g0) <= SCOPE_NOUN_LOOKAHEAD
        # an attribution phrase sitting between the growth and the noun makes the
        # noun a CAUSE, not the scope
        span = s[min(m.start(), g0):max(m.start(), g0)]
        if (before or near) and not _RE_ATTRIBUTION.search(span):
            return "scope_noun"
    if _RE_OUR_BIZ.search(s):
        return "named_business"
    if _RE_SUBJ_SEE.search(s):
        return "segment_subject"
    m = _RE_IN_PROPER.search(s)
    if m and not _RE_TIME_TOK.match(m.group(1).strip()):
        ends = [g.end() for g in _RE_GROWTH_TOK.finditer(s)]
        if ends and (m.start() - max(ends)) <= SCOPE_NEAR_CHARS:
            return "scoped_to:" + m.group(1).strip(".,;\"' ")[:24]
    return None


def is_negated(stmt: str) -> bool:
    """Does the sentence say the number will NOT be met?"""
    return bool(_RE_NEGATION.search(str(stmt or "")))

# Agreement bands. Absolute targets get a wider band than percentages because a
# statement often rounds ("about Rs 500 crores" vs a 496.3 table cell).
TOL_CONFIRMED = 0.01        # 1% relative
TOL_CONSISTENT = 0.10       # 10% relative
TOL_PCT_POINTS = 2.0        # or within 2 percentage points, for growth rates

# metric -> regex over GF1's free-text metric_type.
METRIC_PATTERNS = {
    "revenue": re.compile(r"revenue|sales|topline|top\s*line|turnover|arr", re.I),
    "pat": re.compile(r"\bpat\b|profit|earnings|\bpbt\b|bottom\s*line|\beps\b", re.I),
}

_TRUE = {"yes", "true", "quantifiable", "y", "1"}
_FALSE = {"no", "false", "n", "0"}
_NA = {"", "na", "n/a", "nan", "none", "-", "--"}


def normalise_quantifiable(v):
    """GF1's 10 spellings -> True / False / None."""
    s = str(v or "").strip().lower()
    if s in _TRUE:
        return True
    if s in _FALSE:
        return False
    return None


def metric_matches(gf1_metric_type, metric) -> bool:
    """Does this GF1 statement talk about the metric we scored?

    metric_type is free text, so this is a regex test, never equality.
    """
    rx = METRIC_PATTERNS.get(str(metric or "").strip().lower())
    if rx is None:
        return False
    return bool(rx.search(str(gf1_metric_type or "")))


_RE_NUM_ONLY = re.compile(r"-?\d+(?:\.\d+)?")


def _to_float(v):
    """GF1 numeric_value is FREE TEXT: '100%', '2,600', 'Rs 950 cr', 'NA'.

    A bare float() cast returns None for '100%', which silently discarded the
    confirming evidence -- ONEPOINT's "double our revenues this year" carried
    numeric_value='100%' and fell through to NO_EVIDENCE.
    """
    s = str(v or "").strip()
    if not s or s.lower() in _NA:
        return None
    m = _RE_NUM_ONLY.search(s.replace(",", ""))
    if not m:
        return None
    try:
        f = float(m.group(0))
    except (TypeError, ValueError):
        return None
    return f if f == f else None          # drop NaN


def _statement_amounts_cr(stmt: str):
    """Every rupee-crore reading of a verbatim statement.

    Routed through guidance_value so "26,000 million" reads as 2,600 cr and
    "USD 108 billion" as 950,400 cr — the same normalisation the scorer used, so
    a unit bug on either side shows up as a disagreement instead of cancelling.
    """
    out = []
    parsed = GV.parse_guidance_value(stmt)
    cr = parsed.get("value_num_inr_cr")
    if cr is not None:
        out.append(float(cr))
    # also read each number in the sentence under the sentence's own unit
    for n in GV._numbers(stmt):
        sub = GV.parse_guidance_value(f"{n} {GV._magnitude(stmt) or 'cr'}")
        c = sub.get("value_num_inr_cr")
        if c is not None:
            out.append(float(c))
    return [c for c in out if c > 0]


# "double / triple our revenues" is a growth rate stated in words.
_MULT_WORDS = {"double": 100.0, "doubling": 100.0, "triple": 200.0,
               "tripling": 200.0, "quadruple": 300.0, "quadrupling": 300.0}
_RE_MULT_WORD = re.compile(r"\b(" + "|".join(_MULT_WORDS) + r")\b", re.I)
_RE_DIGIT_WORD = re.compile(r"\b(?:single|double|triple|quadruple)[-\s]*digit", re.I)


def _statement_pcts(stmt: str):
    """Every growth percentage the sentence states, in figures OR in words."""
    txt = str(stmt or "")
    out = [float(m.group(1)) for m in re.finditer(r"(\d+(?:\.\d+)?)\s*%", txt)]
    if not _RE_DIGIT_WORD.search(txt):        # "double-digit" is not a multiple
        for m in _RE_MULT_WORD.finditer(txt):
            out.append(_MULT_WORDS[m.group(1).lower()])
    return out


def _agree(a: float, b: float, is_pct: bool):
    """(verdict_or_None, relative_delta) for one pair of numbers."""
    if a is None or b is None or a <= 0 or b <= 0:
        return None, None
    rel = abs(a - b) / max(abs(b), 1e-9)
    if rel <= TOL_CONFIRMED or (is_pct and abs(a - b) <= 0.5):
        return CONFIRMED, rel
    if rel <= TOL_CONSISTENT or (is_pct and abs(a - b) <= TOL_PCT_POINTS):
        return CONSISTENT, rel
    return None, rel


def candidates(gf1_slice, metric):
    """The GF1 statements that could speak to this metric, quantifiable first."""
    if gf1_slice is None or len(gf1_slice) == 0:
        return []
    rows = []
    for _, r in gf1_slice.iterrows():
        if not metric_matches(r.get("metric_type"), metric):
            continue
        rows.append(r)
    rows.sort(key=lambda r: 0 if normalise_quantifiable(r.get("quantifiable")) else 1)
    return rows


def validate(row: dict, gf1_slice) -> dict:
    """Cross-check one scored row against the transcript.

    `row` is an implied_cagr() result plus `metric`. For an absolute target the
    comparison is on the Rs-crore figure; for a growth rate it is on the
    percentage as STATED (row['raw']), not the annualised output — management
    said the former, and comparing against a number we derived would be circular.
    """
    metric = str(row.get("metric") or "")
    cands = candidates(gf1_slice, metric)
    if not cands:
        return {"verdict": NO_EVIDENCE, "evidence_stmt": "",
                "evidence_num": None, "evidence_delta_pct": None,
                "rule": "no_statement_for_metric"}

    is_abs = row.get("kind") == "absolute"
    target = row.get("target_cr") if is_abs else None
    stated_pct = None
    if not is_abs:
        stated_pct = GV._pct_from_text(str(row.get("raw") or ""))
        if stated_pct is None:
            stated_pct = row.get("cagr_pct")

    best = (NO_EVIDENCE, None, "", None, "no_comparable_number")
    rank = {CONFIRMED: 3, CONSISTENT: 2, CONTRADICTED: 1, NO_EVIDENCE: 0}
    saw_number = False

    for r in cands:
        stmt = str(r.get("exact_statement") or "")
        nums = []
        if is_abs:
            nums += _statement_amounts_cr(stmt)
            nv = _to_float(r.get("numeric_value"))
            if nv is not None:
                nums.append(nv)
            probe, want = nums, target
        else:
            nums += _statement_pcts(stmt)
            nv = _to_float(r.get("numeric_value"))
            if nv is not None:
                nums.append(nv)
            probe, want = nums, stated_pct
        if want is None or not probe:
            continue
        saw_number = True
        for n in probe:
            verdict, rel = _agree(n, want, is_pct=not is_abs)
            if verdict and rank[verdict] > rank[best[0]]:
                best = (verdict, n, stmt[:400], rel, "matched_statement")

    if best[0] == NO_EVIDENCE and saw_number:
        # statements exist and carry numbers, none of which agree
        top = cands[0]
        return {"verdict": CONTRADICTED,
                "evidence_stmt": str(top.get("exact_statement") or "")[:400],
                "evidence_num": _to_float(top.get("numeric_value")),
                "evidence_delta_pct": None, "rule": "no_number_agreed"}

    verdict, num, stmt, rel, rule = best
    # The number may agree perfectly and still be the wrong CLAIM: scoped to one
    # segment, or stated as something the company will NOT hit. Both are checked
    # on the statement that produced the verdict, so the reason is traceable.
    if verdict in PUBLISHABLE:
        why = scope_is_segment(stmt)
        if why:
            verdict, rule = SEGMENT_SCOPED, why
        elif is_negated(stmt):
            verdict, rule = NEGATED, "statement_negates_the_number"
    return {"verdict": verdict, "evidence_stmt": stmt, "evidence_num": num,
            "evidence_delta_pct": (round(rel * 100.0, 2) if rel is not None else None),
            "rule": rule}


def _self_test() -> int:
    fails = []

    def chk(label, got, want):
        if got != want:
            fails.append(f"{label}: got {got!r}, want {want!r}")

    # --- GF1's ten spellings of quantifiable ---------------------------------
    for v in ("Yes", "YES", "TRUE", "True", "Quantifiable"):
        chk(f"quantifiable {v}", normalise_quantifiable(v), True)
    for v in ("No", "NO", "FALSE"):
        chk(f"quantifiable {v}", normalise_quantifiable(v), False)
    for v in ("NA", "None", ""):
        chk(f"quantifiable {v}", normalise_quantifiable(v), None)

    # --- metric_type is free text: regex, never equality ---------------------
    chk("Revenue matches", metric_matches("Revenue", "revenue"), True)
    chk("ARR matches revenue", metric_matches("ARR (Revenue)", "revenue"), True)
    chk("Margin not revenue", metric_matches("Margin", "revenue"), False)
    chk("junk metric_type", metric_matches("10,000", "revenue"), False)
    chk("PAT matches", metric_matches("PAT", "pat"), True)
    chk("Capacity not pat", metric_matches("Capacity", "pat"), False)

    gf1 = pd.DataFrame([
        {"metric_type": "Revenue", "quantifiable": "Yes", "numeric_value": "2600",
         "exact_statement": "we target revenue of INR 26,000 million by FY27"},
        {"metric_type": "Margin", "quantifiable": "Yes", "numeric_value": "23.5",
         "exact_statement": "EBITDA margin around 23.5%"},
    ])

    # absolute agrees with the transcript once units are normalised
    v = validate({"kind": "absolute", "target_cr": 2600.0, "metric": "revenue",
                  "raw": "INR 26,000 million"}, gf1)
    chk("SOLEX confirmed", v["verdict"], CONFIRMED)

    # the OLD 10x misparse must be caught, without any cap
    v = validate({"kind": "absolute", "target_cr": 26000.0, "metric": "revenue",
                  "raw": "INR 26,000 million"}, gf1)
    chk("10x misparse contradicted", v["verdict"], CONTRADICTED)

    # a metric with no statement at all
    v = validate({"kind": "absolute", "target_cr": 900.0, "metric": "pat",
                  "raw": "ABS: INR 900 cr"}, gf1)
    chk("no pat statement", v["verdict"], NO_EVIDENCE)

    # empty evidence
    v = validate({"kind": "absolute", "target_cr": 900.0, "metric": "revenue",
                  "raw": "x"}, pd.DataFrame())
    chk("empty gf1", v["verdict"], NO_EVIDENCE)

    # growth compared on the STATED pct, not the annualised output
    g = pd.DataFrame([{"metric_type": "Revenue", "quantifiable": "Yes",
                       "numeric_value": "45",
                       "exact_statement": "we expect 40-50% YoY growth"}])
    v = validate({"kind": "growth", "cagr_pct": 45.0, "metric": "revenue",
                  "raw": "40-50% YoY growth"}, g)
    chk("growth confirmed", v["verdict"], CONFIRMED)

    # GF1 numeric_value is free text -- '100%' must parse, and "double our
    # revenues" states 100% growth in words (the real ONEPOINT row).
    chk("numeric_value with %", _to_float("100%"), 100.0)
    chk("numeric_value with commas", _to_float("2,600"), 2600.0)
    chk("numeric_value NA", _to_float("NA"), None)
    chk("double in words", _statement_pcts("we should double our revenues"), [100.0])
    chk("double-digit is not 100", _statement_pcts("double-digit growth"), [])
    op = pd.DataFrame([{"metric_type": "Revenue", "quantifiable": "Yes",
                        "numeric_value": "100%",
                        "exact_statement": "We should be able to try and double "
                                           "our revenues this year"}])
    v = validate({"kind": "growth", "cagr_pct": 100.0, "metric": "revenue",
                  "raw": "100%"}, op)
    chk("ONEPOINT confirmed", v["verdict"], CONFIRMED)

    # --- segment scoping (all real Q1FY27 statements) -----------------------
    chk("Job Hai is scoped",
        bool(scope_is_segment("We would like to more than double this year in Job Hai.")),
        True)
    chk("named business is scoped",
        scope_is_segment("We expect our branded alco-bev business to grow "
                         "approximately 60% to 70% for the full year"),
        "named_business")
    chk("segment noun is scoped",
        scope_is_segment("on track of achieving more than around 50% under the "
                         "D&B segment on a Y-o-Y basis"), "scope_noun")
    chk("segment subject is scoped",
        scope_is_segment("consumer will see about 125% to 150% over FY26."),
        "segment_subject")
    # ...and the things that must NOT be dropped
    chk("plant location is not a scope",
        scope_is_segment("we would like to revise our guidance for the full year "
                         "to 70% considering the robust demand which is there and "
                         "the capacities that we are ready with in Ratlam."), None)
    chk("a date is not a segment",
        scope_is_segment("So we expect INR7,500 crores in FY28."), None)
    chk("company-wide vetoes",
        scope_is_segment("we should double our revenues this year"), None)
    chk("plain growth is not scoped",
        scope_is_segment("we want to grow at 50% CAGR for the next 2 to 3 years"),
        None)

    # --- HELD-OUT cases (Q1FY25..Q4FY26 -- quarters the rules were NOT tuned
    # on). Six real segment claims that must drop, and three company-wide claims
    # that merely NAME segments as contributors and must survive.
    for label, stmt in [
        ("electrolyte salt segment",
         "we expect the electrolyte salt segment to grow by at least 200% this year."),
        ("Conductor/Cables Segment",
         "Conductor/Cables Segment: a strong H2FY26 rebound is expected with "
         "revenue projected to double vs H1FY26"),
        ("aerospace business",
         "We have secured large orders having the potential to double our "
         "aerospace business over the next 30 months."),
        ("CDMO business",
         "we would create an INR 100-crore opportunity from the CDMO business."),
        ("core B2B business",
         "that is the aim to double our sales over the next 3 to 4 years in our "
         "core B2B business"),
    ]:
        chk("drops " + label, bool(scope_is_segment(stmt)), True)
    for label, stmt in [
        ("in-total + driven-by",
         "In total, we are expecting to double our business compared to last "
         "year, driven by these two key verticals."),
        ("attributed-due-to",
         "we expect a 50%+ growth in FY26 largely attributed due to recovery in "
         "the agro chemicals sales and growth in polymer and pharma segment."),
        ("plant location",
         "we would like to revise our guidance for the full year to 70% "
         "considering the capacities that we are ready with in Ratlam."),
    ]:
        chk("keeps " + label, scope_is_segment(stmt), None)

    # --- negation -----------------------------------------------------------
    chk("negation caught",
        is_negated("we will increase with Q1 but not to the extent we are "
                   "intending. Not by the amount of 50%"), True)
    chk("plain statement not negated",
        is_negated("we gave a guidance of about 60% to 70% growth"), False)

    seg = pd.DataFrame([{"metric_type": "Revenue", "quantifiable": "Yes",
                         "numeric_value": "100",
                         "exact_statement": "We would like to more than double "
                                            "this year in Job Hai."}])
    v = validate({"kind": "growth", "cagr_pct": 100.0, "metric": "revenue",
                  "raw": "100%"}, seg)
    chk("segment row does not publish", v["verdict"], SEGMENT_SCOPED)
    chk("segment row is not publishable", v["verdict"] in PUBLISHABLE, False)

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
