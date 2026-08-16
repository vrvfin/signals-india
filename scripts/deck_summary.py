#!/usr/bin/env python3
"""deck_summary — the investor deck read as a STANDALONE company summary.

WHY THIS EXISTS, given deck_teardown already reads the same PDF.

`deck_teardown` is forensic and deliberately narrow: seven operating categories,
numbers only, with revenue/EBITDA/PAT explicitly BANNED because those belong to the
audited filing. That is the right shape when the deck is a supplement to a concall and
a P&L. It is the wrong shape for the many companies — most small and mid caps — that
never hold a concall at all. For those, the deck is the ONLY narrative management
publishes all quarter, and a seven-category numbers table throws away almost all of it.

So this module reads the same document for a different purpose: everything a reader
needs to understand the company from the deck alone. Eighteen sections spanning what
the business does, how it performed, what it is building, who it sells to, what it
promises and what it admits. The concall layer stays exactly where it is and becomes
INCREMENTAL — what the call adds on top of the deck — rather than the only source of
narrative.

WHAT KEEPS IT HONEST. The same rule as the teardown, applied to prose as well as
numbers: every row must quote the deck. A numeric row is dropped unless its value
appears verbatim inside its own evidence string (`provenance.grounded`). A narrative
row is dropped unless it carries both a detail and a quote. This is the whole defence
against a model that would otherwise happily write "the company is well positioned for
growth" about any company on earth — filler cannot quote a slide, so filler is dropped.

WHAT IT DOES NOT DO. It does not diff against a prior deck and it does not flag chart
tricks; deck_teardown owns both and they need slide-level forensics, not description.
Running them as one call starved the forensic half — the caps here alone are three
times the teardown's entire budget. Two passes, two jobs.

ESG GETS ITS OWN SECTION on purpose. The teardown bans it outright because emissions
targets were crowding out capacity numbers. Banning it here would push the model to
file "Net Zero by 2045" under strategy or guidance, which is worse — it would corrupt
the sections that matter. Give it a labelled bucket and it stays in the bucket.
"""

from __future__ import annotations

import os
import sys

_SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

import pandas as pd

from provenance import grounded
# Shared with the teardown pass by design (rule 4: reuse, do not re-implement). A second
# copy of the JSON salvage or the nan-scrubber would drift from this one within a quarter.
from deck_teardown import _s, _f, _clamp, _first_json_object, _BASE, _TAIL

PROMPT_FILE = "deck_summary_prompt.txt"

DECK_SUMMARY_COLS = _BASE + ["section", "label", "value", "unit", "period",
                             "detail", "slide_ref", "evidence"] + _TAIL

# The eighteen sections, in reading order — a deck read top to bottom answers them in
# roughly this sequence, and the renderer follows the same order so the mail reads like
# a note rather than a database dump.
SECTIONS: tuple[str, ...] = (
    "business_overview",      # what the company does, segments, products, positioning
    "financials",             # revenue / EBITDA / PAT / margin AS THE DECK STATES THEM
    "balance_sheet",          # net debt, gearing, net worth, cash, working capital
    "segment_performance",    # how each segment did
    "geography",              # where revenue comes from, and where it is going
    "capacity_expansion",     # new lines, plants, debottlenecking, commissioning dates
    "capex",                  # what is being spent, when, on what, funded how
    "orderbook_pipeline",     # order book, backlog, book-to-bill, execution window
    "customers",              # key clients, wins, concentration
    "products_rd",            # launches, R&D, patents, approvals, certifications
    "industry_market",        # TAM, industry growth, market share, competition, macro
    "strategy",               # the stated pillars and priorities
    "guidance_outlook",       # forward targets carrying a horizon
    "risks",                  # what the deck itself admits
    "management_commentary",  # the MD/CEO message
    "capital_allocation",     # dividend, buyback, shareholding, pledge
    "subsidiary_ma",          # subsidiaries, JVs, acquisitions, divestments
    "esg",                    # bucketed so it stops contaminating strategy and guidance
)
_SECTION_SET = set(SECTIONS)

# Total is well below 18 x per-section: a real deck is lopsided (a capital-goods deck is
# mostly order book, a pharma deck mostly products), and the cap exists to stop a model
# padding thin sections to look thorough, not to force even coverage.
MAX_ROWS, MAX_PER_SECTION = 75, 6

# A flat cap is wrong for the sections that carry a grid rather than a list. Financials
# alone is revenue / EBITDA / PAT / two margins, each consolidated AND standalone, each
# with a growth rate — a six-row cap truncated it mid-table on the first live deck and
# silently lost standalone PAT. These sections get room to finish the table they start.
_SECTION_CAP = {"financials": 12, "segment_performance": 8, "balance_sheet": 8,
                "capacity_expansion": 8, "orderbook_pipeline": 8}


def _cap_for(section: str) -> int:
    return _SECTION_CAP.get(section, MAX_PER_SECTION)


# When one statement is filed under two sections, this decides which copy survives. It is
# NOT reading order and it is NOT SECTIONS order: a forward-looking sentence gets emitted
# under segment_performance long before guidance_outlook, and keeping the first copy
# emptied guidance_outlook entirely on the first live deck. The two sections a reader
# scans first — what was promised, and what was admitted — outrank the descriptive ones.
_HOME_PRIORITY = ("guidance_outlook", "risks", "financials", "orderbook_pipeline",
                  "capacity_expansion", "capex", "balance_sheet", "segment_performance",
                  "geography", "customers", "products_rd", "subsidiary_ma", "strategy",
                  "industry_market", "management_commentary", "business_overview",
                  "capital_allocation", "esg")
_HOME_RANK = {s: i for i, s in enumerate(_HOME_PRIORITY)}


# Sections a model reliably confuses, with the tell that resolves it. Not a general
# classifier — two specific, observed confusions, each fixed only when the row names the
# other section outright. On the first live deck the model filed "Segment Wise - EEI
# Share 77.7%" under `geography`, immediately below the genuine Asia/Europe split.
_REROUTE = (
    ("geography", "segment_performance", ("segment wise", "segment mix", "segment share",
                                          "by segment", "segment-wise")),
    ("segment_performance", "geography", ("geography wise", "geography-wise", "by geography",
                                          "geographic mix", "region wise")),
)


def _reroute(section: str, label: str, detail: str) -> str:
    blob = f"{label} {detail}".lower()
    for src, dst, tells in _REROUTE:
        if section == src and any(t in blob for t in tells):
            return dst
    return section

# A number is only meaningful with a period attached, but decks are careless about this,
# so an unlabelled figure keeps its row rather than being dropped — the renderer shows
# the gap instead of inventing a period.
#
# _MIN_EVIDENCE APPLIES TO NARRATIVE ROWS ONLY, and that distinction was paid for. A
# flat 12-character floor deleted 19 of 36 rows on the first live deck — including every
# financial line, because the model quotes a table cell as the bare figure it is:
#     {"label": "Consolidated PAT", "value": 194, "evidence": "194"}
# That is a truthful quote of a table. A narrative row has no arithmetic check, so length
# is the only proxy for substance there; a numeric row is checked properly by
# `grounded()`, which is a stronger test than any length rule. Applying the floor to both
# threw away the strongest rows in the table to discipline the weakest.
_MIN_DETAIL, _MIN_EVIDENCE = 12, 12
_MIN_EVIDENCE_NUMERIC = 1

# Slides that are in every deck and say nothing about any company: the safe-harbour
# disclaimer, the IR contact card, the registration boilerplate. The model mines them
# because they are text-dense and quotable — a contact name arrived filed under
# `capital_allocation` and a forward-looking-statements disclaimer under `risks` on the
# first live run. Matched across label and detail, since either may carry the tell.
_BOILERPLATE = (
    "safe harbor", "safe harbour", "forward-looking statement", "forward looking statement",
    "disclaimer", "investor relation", "for further information", "for any queries",
    "contact person", "contact us", "company secretary related", "cin:", "cin no",
    "registered office", "corporate office address", "thank you", "stock exchange code",
    "bse code", "nse symbol", "this presentation",
)

# Filler the model reaches for when a section is genuinely empty. These phrases say
# nothing about THIS company and would pass every other check, because the model can
# always find a slide to point at. Matched on the detail only: a deck may legitimately
# use the words in a quote, and dropping the quote would lose the real point around it.
_FILLER = (
    "well positioned", "well-positioned", "strong position", "market leader in its",
    "committed to excellence", "customer centric", "customer-centric",
    "focus on growth", "poised for growth", "bright future", "continues to focus",
    "leveraging synergies", "robust business model",
)


def _is_filler(detail: str) -> bool:
    d = detail.lower()
    return any(f in d for f in _FILLER)


def _is_boilerplate(label: str, detail: str) -> bool:
    blob = f"{label} {detail}".lower()
    return any(b in blob for b in _BOILERPLATE)


# Indian decks report in whichever scale flatters the slide: Rishabh prints Rs Mn, APL
# Apollo prints Rs Cr. Storing "1983" without carrying "Mn" through to the reader is a
# ten-fold error waiting to happen, so the unit is normalised for DISPLAY ONLY and the
# value is never touched. Rescaling to a house unit is deliberately not done here — the
# deck's own number must stay the deck's own number, or the evidence stops matching it.
_UNIT_MAP = {
    "rs. mn": "Rs mn", "rs mn": "Rs mn", "inr mn": "Rs mn", "mn": "Rs mn",
    "rs. million": "Rs mn", "million": "Rs mn", "rs. lakh": "Rs lakh", "lakh": "Rs lakh",
    "rs. cr": "Rs cr", "rs cr": "Rs cr", "inr cr": "Rs cr", "cr": "Rs cr",
    "crore": "Rs cr", "crores": "Rs cr", "rs. crore": "Rs cr",
    "bn": "Rs bn", "rs. bn": "Rs bn", "inr bn": "Rs bn", "billion": "Rs bn",
}


def display_unit(unit: str) -> str:
    """The unit as a reader should see it. Never rescales, never infers a missing one."""
    u = _s(unit, 16)
    return _UNIT_MAP.get(u.lower().strip(), u)


def parse_summary(payload, row, quarter: str, now_str: str) -> dict:
    """Model output -> deck_summary rows, with every ungrounded claim dropped.

    `payload` may be the raw response text or an already-parsed dict.

    Returns {"rows": [...], "dropped": {reason: count}}. `dropped` goes to the run log
    so a model that starts inventing shows up in the logs rather than in the data.

    The grounding rule differs by row kind, because the two kinds fail differently:
      numeric  — the value MUST appear inside its own evidence. A number the deck does
                 not print is a hallucination, full stop.
      narrative— there is nothing to arithmetic-check, so the defence is that it must
                 quote the deck AND must not be generic filler.
    """
    obj = payload if isinstance(payload, dict) else _first_json_object(_s(payload, 400_000))
    if not obj:
        return {"rows": [], "dropped": {"unparseable_response": 1}}

    base = {"isin": _s(row.get("isin"), 20), "symbol": _s(row.get("symbol"), 24),
            "company_name": _s(row.get("company_name"), 120), "quarter": quarter,
            "processed_at": now_str, "source_doc_id": _s(row.get("doc_id"), 80)}
    dropped: dict[str, int] = {}

    def drop(reason):
        dropped[reason] = dropped.get(reason, 0) + 1

    rows: list[dict] = []
    seen: set = set()
    per_section: dict[str, int] = {}

    # Accept either a flat list or the section-keyed object shape. Lite models drift
    # between the two across runs on the same prompt; rejecting one shape would silently
    # halve coverage on the days it drifts.
    raw: list = []
    items = obj.get("summary")
    if isinstance(items, list):
        raw = [o for o in items if isinstance(o, dict)]
    elif isinstance(items, dict):
        for sec, lst in items.items():
            if isinstance(lst, list):
                raw += [{**o, "section": o.get("section") or sec}
                        for o in lst if isinstance(o, dict)]

    for o in raw[: MAX_ROWS * 3]:
        sec = _clamp(o.get("section"), _SECTION_SET, "")
        if not sec:
            drop("bad_section"); continue

        label = _s(o.get("label"), 120)
        detail = _s(o.get("detail"), 400)
        ev = _s(o.get("evidence"), 300)
        val = _f(o.get("value"))

        if not label:
            drop("no_label"); continue
        if _is_boilerplate(label, detail):
            drop("boilerplate_slide"); continue

        if val is not None:
            # Numeric row: `grounded` is the real test. A bare-figure quote is a truthful
            # quote of a table cell, so no length floor here — see _MIN_EVIDENCE_NUMERIC.
            if len(ev) < _MIN_EVIDENCE_NUMERIC:
                drop("no_evidence"); continue
            if not grounded(val, ev, tolerance_pct=1.0):
                drop("value_not_in_evidence"); continue
        else:
            # Narrative row: nothing to arithmetic-check, so it has to actually say
            # something, quote the deck for it, and not be sayable about any company.
            if len(ev) < _MIN_EVIDENCE:
                drop("no_evidence"); continue
            if len(detail) < _MIN_DETAIL:
                drop("narrative_without_detail"); continue
            if _is_filler(detail):
                drop("generic_filler"); continue

        sec = _reroute(sec, label, detail)

        # Dedupe on meaning, not on formatting: the same fact restated on two slides is
        # one fact. Value participates so "capacity 1.2 MT" and "capacity 1.5 MT" both
        # survive — those are two periods, not a repeat.
        k = (sec, label.lower(), str(val), _s(o.get("period"), 20), detail[:60].lower())
        if k in seen:
            drop("duplicate"); continue
        seen.add(k)

        rows.append({**base, "section": sec, "label": label, "value": val,
                     "unit": _s(o.get("unit"), 16), "period": _s(o.get("period"), 20),
                     "detail": detail, "slide_ref": _s(o.get("slide_ref"), 60),
                     "evidence": ev})

    # ---- cross-section duplicates, resolved by home rather than by reading order.
    # Two passes are needed: the winning copy is frequently emitted AFTER the losing one,
    # so this cannot be decided while streaming through the response.
    best: dict = {}
    for i, r in enumerate(rows):
        d = r["detail"]
        k2 = (str(r["value"]), d[:80].lower()) if d else (str(r["value"]), r["label"].lower())
        cur = best.get(k2)
        if cur is None or _HOME_RANK.get(r["section"], 99) < _HOME_RANK.get(rows[cur]["section"], 99):
            if cur is not None:
                drop("cross_section_duplicate")
            best[k2] = i
        else:
            drop("cross_section_duplicate")
    rows = [rows[i] for i in sorted(best.values())]

    # ---- caps last, so a row is never dropped for a cap and then replaced by a copy of
    # itself that the dedupe would have removed anyway.
    kept: list[dict] = []
    for r in rows:
        sec = r["section"]
        if per_section.get(sec, 0) >= _cap_for(sec):
            drop("section_cap"); continue
        per_section[sec] = per_section.get(sec, 0) + 1
        kept.append(r)
    if len(kept) > MAX_ROWS:
        drop_n = len(kept) - MAX_ROWS
        for _ in range(drop_n):
            drop("total_cap")
        kept = kept[:MAX_ROWS]

    # Reading order, fixed by SECTIONS and stable within a section. The model emits in
    # slide order, which puts the company's vintage after the risk factors because that
    # is where the annexure sat — a reader wants the note, not the slide deck's shuffle.
    _rank = {s: i for i, s in enumerate(SECTIONS)}
    kept.sort(key=lambda r: _rank.get(r["section"], 99))

    return {"rows": kept, "dropped": dropped}


def section_counts(rows) -> dict:
    """{section: n} in SECTIONS order — for the run log and the coverage view."""
    out: dict[str, int] = {}
    for r in rows:
        s = r.get("section") if isinstance(r, dict) else None
        if s:
            out[s] = out.get(s, 0) + 1
    return {s: out[s] for s in SECTIONS if s in out}


def coverage(rows) -> tuple[int, int]:
    """(sections present, 18). How complete a standalone read this deck supports."""
    return len(section_counts(rows)), len(SECTIONS)


def run_summary(gemini, prompt: str, doc_bytes: bytes, row, quarter: str,
                now_str: str) -> dict:
    """One structured pass over the deck. Shared machinery only (rule 4).

    8192 output tokens against the teardown's 4096: sixty rows carrying a quote each
    cannot fit in the teardown's budget, and a truncated response loses the tail
    sections entirely rather than degrading evenly.
    """
    from _extractor_base import run_structured_over_doc
    resp = run_structured_over_doc(
        gemini, prompt, doc_bytes, max_output_tokens=8192,
        name=f"{row.get('symbol', 'DOC')}_PPT_summary")
    return parse_summary(resp, row, quarter, now_str)


# --------------------------------------------------------------------------- #
def _self_test() -> int:
    ok = fail = 0

    def check(name, cond):
        nonlocal ok, fail
        if cond:
            ok += 1
        else:
            fail += 1
            print(f"  FAIL {name}")

    row = {"isin": "INE702C01027", "symbol": "APLAPOLLO",
           "company_name": "APL Apollo", "doc_id": "doc1"}
    now = "2026-08-16T00:00:00"

    def P(obj):
        return parse_summary(obj, row, "Q1FY27", now)

    # --- shape and identity
    r = P({"summary": [{"section": "financials", "label": "Revenue",
                        "value": 5289, "unit": "cr", "period": "Q1FY27",
                        "evidence": "Revenue for Q1FY27 stood at Rs 5,289 cr"}]})
    check("numeric row kept", len(r["rows"]) == 1)
    check("isin carried", r["rows"][0]["isin"] == "INE702C01027")
    check("quarter carried", r["rows"][0]["quarter"] == "Q1FY27")
    check("source doc carried", r["rows"][0]["source_doc_id"] == "doc1")
    check("cols complete", set(r["rows"][0]) == set(DECK_SUMMARY_COLS))
    check("comma number grounds", r["rows"][0]["value"] == 5289.0)

    # --- THE point of the module: financials are allowed here, unlike the teardown
    from deck_teardown import _banned
    check("teardown still bans revenue", _banned("Revenue", "Revenue was 5,289 cr"))
    check("summary allows revenue",
          len(P({"summary": [{"section": "financials", "label": "EBITDA", "value": 412,
                              "unit": "cr", "period": "Q1FY27",
                              "evidence": "EBITDA of Rs 412 cr"}]})["rows"]) == 1)

    # --- grounding
    r = P({"summary": [{"section": "financials", "label": "PAT", "value": 999,
                        "unit": "cr", "evidence": "PAT for the quarter was Rs 250 cr"}]})
    check("ungrounded number dropped", not r["rows"])
    check("ungrounded reason logged", r["dropped"].get("value_not_in_evidence") == 1)

    r = P({"summary": [{"section": "strategy", "label": "Pillar",
                        "detail": "Move the mix toward value-added products",
                        "evidence": ""}]})
    check("narrative without quote dropped", not r["rows"])
    check("no-evidence reason logged", r["dropped"].get("no_evidence") == 1)

    r = P({"summary": [{"section": "strategy", "label": "Outlook",
                        "detail": "The company is well positioned for growth",
                        "evidence": "We believe we are well positioned for growth"}]})
    check("generic filler dropped", not r["rows"])
    check("filler reason logged", r["dropped"].get("generic_filler") == 1)

    r = P({"summary": [{"section": "risks", "label": "Input cost",
                        "detail": "x", "evidence": "Steel prices remain volatile"}]})
    check("empty narrative dropped", not r["rows"])

    # --- THE 19-of-36 regression: a table cell quoted as the bare figure is still a
    # truthful quote, and dropping it cost the whole financials section on live data.
    r = P({"summary": [{"section": "financials", "label": "Consolidated PAT",
                        "value": "194", "unit": "Rs. Mn", "period": "Q1FY27",
                        "detail": "Consolidated Profit After Tax for Q1FY27 was Rs. 194 Mn.",
                        "evidence": "194"}]})
    check("bare-figure quote kept for a numeric row", len(r["rows"]) == 1)
    check("string value coerced", r["rows"][0]["value"] == 194.0)
    r = P({"summary": [{"section": "strategy", "label": "Pillar",
                        "detail": "Move the mix toward value-added products",
                        "evidence": "mix"}]})
    check("bare quote still rejected for a narrative row", not r["rows"])
    r = P({"summary": [{"section": "financials", "label": "PAT", "value": 194,
                        "evidence": "250"}]})
    check("bare figure still checked against the value", not r["rows"])

    # --- unit is carried, never converted: Rs mn read as Rs cr is a 10x error
    r = P({"summary": [{"section": "financials", "label": "Revenue", "value": 1983,
                        "unit": "Rs. Mn", "period": "Q1FY27", "evidence": "1,983"}]})
    check("unit stored verbatim", r["rows"][0]["unit"] == "Rs. Mn")
    check("unit display normalised", display_unit("Rs. Mn") == "Rs mn")
    check("crore variants normalise", display_unit("crores") == display_unit("Cr") == "Rs cr")
    check("unknown unit passes through", display_unit("MT") == "MT")
    check("blank unit stays blank", display_unit("") == "" and display_unit(None) == "")
    check("value untouched by unit handling", r["rows"][0]["value"] == 1983.0)

    # --- boilerplate slides: present in every deck, informative in none
    r = P({"summary": [{"section": "capital_allocation", "label": "Contact Person - CS",
                        "detail": "For Company Secretary related queries, contact Mr Joglekar.",
                        "evidence": "Mr. Ajinkya Joglekar, Company Secretary"}]})
    check("IR contact card dropped", not r["rows"])
    check("boilerplate reason logged", r["dropped"].get("boilerplate_slide") == 1)
    r = P({"summary": [{"section": "risks", "label": "Safe Harbor Statement",
                        "detail": "The presentation contains forward-looking statements.",
                        "evidence": "This presentation contains forward-looking statements"}]})
    check("safe-harbour disclaimer dropped", not r["rows"])
    r = P({"summary": [{"section": "risks", "label": "Customer concentration",
                        "detail": "Top-5 customers are 46% of revenue, a stated dependency",
                        "evidence": "Top 5 customers contribute 46% of revenue"}]})
    check("a real risk is not caught by the boilerplate filter", len(r["rows"]) == 1)

    # A quote may contain filler words as long as the detail is substantive — dropping
    # it would lose the real point sitting next to the boilerplate.
    r = P({"summary": [{"section": "risks", "label": "Input cost",
                        "detail": "HRC price volatility can compress spreads in H2",
                        "evidence": "We are well positioned though HRC prices are volatile"}]})
    check("substantive detail survives filler quote", len(r["rows"]) == 1)

    # --- sections
    check("18 sections", len(SECTIONS) == 18 and len(_SECTION_SET) == 18)
    r = P({"summary": [{"section": "nonsense", "label": "X", "detail": "some detail here",
                        "evidence": "quoted from the deck"}]})
    check("unknown section dropped", not r["rows"] and r["dropped"].get("bad_section") == 1)
    r = P({"summary": [{"section": "Guidance Outlook", "label": "Target", "value": 20,
                        "unit": "%", "evidence": "targeting 20% volume growth"}]})
    check("section normalised from spaced title case",
          len(r["rows"]) == 1 and r["rows"][0]["section"] == "guidance_outlook")

    # --- ESG is bucketed, not banned (the teardown bans it; here it has a home)
    r = P({"summary": [{"section": "esg", "label": "Net zero", "detail": "Net zero by 2045",
                        "evidence": "We target net zero by 2045"}]})
    check("esg kept in its own bucket",
          len(r["rows"]) == 1 and r["rows"][0]["section"] == "esg")

    # --- caps
    many = [{"section": "risks", "label": f"R{i}", "detail": f"distinct risk number {i}",
             "evidence": f"risk {i} is disclosed on this slide"} for i in range(12)]
    r = P({"summary": many})
    check("per-section cap enforced", len(r["rows"]) == MAX_PER_SECTION)
    check("cap reason logged", r["dropped"].get("section_cap") == 12 - MAX_PER_SECTION)

    # A grid section must be allowed to finish its table — a flat 6 truncated financials
    # mid-row on live data and lost standalone PAT.
    fin = [{"section": "financials", "label": f"F{i}", "value": i, "unit": "cr",
            "evidence": f"line item value {i}"} for i in range(1, 13)]
    check("financials gets a wider cap", len(P({"summary": fin})["rows"]) == 12)
    check("cap lookup", _cap_for("financials") == 12 and _cap_for("risks") == 6)

    # --- section rerouting: the two confusions actually observed, and nothing else
    r = P({"summary": [{"section": "geography", "label": "Segment Wise - EEI Share",
                        "value": 77.7, "unit": "%", "evidence": "EEI 77.7% of business"}]})
    check("segment mix rerouted out of geography",
          r["rows"][0]["section"] == "segment_performance")
    r = P({"summary": [{"section": "segment_performance", "label": "Geography wise - Europe",
                        "value": 28, "unit": "%", "evidence": "Europe 28.0% of revenue"}]})
    check("geo mix rerouted out of segment", r["rows"][0]["section"] == "geography")
    r = P({"summary": [{"section": "geography", "label": "Asia share", "value": 62.5,
                        "unit": "%", "evidence": "Asia contributed 62.5% of revenue"}]})
    check("a genuine geography row is left alone", r["rows"][0]["section"] == "geography")
    check("reroute does not fire on other sections",
          P({"summary": [{"section": "risks", "label": "Segment wise exposure",
                          "detail": "One segment carries most of the volatility",
                          "evidence": "segment wise the risk is concentrated"}]}
            )["rows"][0]["section"] == "risks")

    # --- the same point filed twice under different sections is still one point
    dupe_pt = {"label": "Priorities for FY27",
               "detail": "Priorities for FY27 include profitable growth and better mix",
               "evidence": "Priorities for FY27: profitable growth, better product mix"}
    r = P({"summary": [{**dupe_pt, "section": "strategy"},
                       {**dupe_pt, "section": "capital_allocation"}]})
    check("cross-section duplicate dropped", len(r["rows"]) == 1)
    check("higher home wins", r["rows"][0]["section"] == "strategy")
    check("cross-section reason logged", r["dropped"].get("cross_section_duplicate") == 1)

    # THE regression this rule exists for: the winning copy is emitted SECOND, so a
    # first-wins rule emptied guidance_outlook on the first live deck.
    outlook = {"label": "HPDC outlook", "detail": "Management expects HPDC to remain "
                                                  "breakeven at Adj. EBITDA for full FY27",
               "period": "FY27", "evidence": "expect to remain break even in FY27"}
    r = P({"summary": [{**outlook, "section": "segment_performance"},
                       {**outlook, "section": "guidance_outlook"}]})
    check("guidance outranks segment even when emitted later",
          len(r["rows"]) == 1 and r["rows"][0]["section"] == "guidance_outlook")
    r = P({"summary": [{**outlook, "section": "guidance_outlook"},
                       {**outlook, "section": "segment_performance"}]})
    check("guidance still wins when emitted first",
          len(r["rows"]) == 1 and r["rows"][0]["section"] == "guidance_outlook")
    risk = {"label": "Cost", "detail": "HRC price volatility can compress spreads in H2",
            "evidence": "HRC prices remain volatile"}
    r = P({"summary": [{**risk, "section": "management_commentary"},
                       {**risk, "section": "risks"}]})
    check("risks outrank commentary", r["rows"][0]["section"] == "risks")
    check("home rank covers every section",
          set(_HOME_RANK) == _SECTION_SET and len(_HOME_RANK) == 18)
    r = P({"summary": [{"section": "capex", "label": "Capex", "value": 500, "unit": "cr",
                        "evidence": "capex of 500 cr"},
                       {"section": "capacity_expansion", "label": "New line", "value": 500,
                        "unit": "cr", "evidence": "the 500 cr line commissions in FY28"}]})
    check("same number, different point, both kept", len(r["rows"]) == 2)

    big = []
    for s in SECTIONS:
        big += [{"section": s, "label": f"{s}{i}", "detail": f"detail {s} number {i}",
                 "evidence": f"evidence for {s} item {i}"} for i in range(6)]
    r = P({"summary": big})
    check("total cap enforced", len(r["rows"]) == MAX_ROWS)
    check("coverage counts sections", coverage(r["rows"])[1] == 18)

    # --- dedupe
    dup = {"section": "capex", "label": "Capex", "value": 500, "unit": "cr",
           "period": "FY27", "evidence": "capex of Rs 500 cr planned in FY27"}
    r = P({"summary": [dup, dict(dup)]})
    check("duplicate dropped", len(r["rows"]) == 1 and r["dropped"].get("duplicate") == 1)
    r = P({"summary": [dup, {**dup, "value": 700,
                             "evidence": "and Rs 700 cr in FY28", "period": "FY28"}]})
    check("same label different period kept", len(r["rows"]) == 2)

    # --- input shapes
    r = P({"summary": {"risks": [{"label": "FX", "detail": "Export exposure to USD moves",
                                  "evidence": "60% of revenue is exported"}]}})
    check("section-keyed object shape accepted",
          len(r["rows"]) == 1 and r["rows"][0]["section"] == "risks")

    raw = ('```json\n{"summary": [{"section": "customers", "label": "Client win", '
           '"detail": "Added a large OEM in North America", '
           '"evidence": "Won a marquee OEM customer in North America"}]}\n```')
    check("fenced raw text parsed", len(P(raw)["rows"]) == 1)
    check("unparseable logged", P("not json at all")["dropped"].get("unparseable_response") == 1)
    check("empty payload safe", P({"summary": []})["rows"] == [])
    check("missing key safe", P({})["rows"] == [])

    # --- nan hygiene: these reach the mail, and "nan" rendered to the user is a bug
    r = P({"summary": [{"section": "geography", "label": "Exports", "value": 31,
                        "unit": "%", "period": "nan", "slide_ref": "None",
                        "detail": "NaN", "evidence": "Exports were 31% of revenue"}]})
    check("nan-ish strings scrubbed",
          r["rows"][0]["period"] == "" and r["rows"][0]["slide_ref"] == ""
          and r["rows"][0]["detail"] == "")

    # --- rows come back in reading order, not slide order. The live deck put the
    # company's founding year after the risk factors, splitting business_overview in two.
    r = P({"summary": [{"section": "risks", "label": "Risk", "detail": "Competition is rising",
                        "evidence": "competitive intensity has risen"},
                       {"section": "business_overview", "label": "Vintage", "value": 1982,
                        "evidence": "incorporated in 1982"},
                       {"section": "financials", "label": "PAT", "value": 194, "unit": "cr",
                        "evidence": "PAT of 194 cr"},
                       {"section": "business_overview", "label": "Segments",
                        "detail": "Two segments, EEI and HPDC",
                        "evidence": "operates in EEI and HPDC"}]})
    check("rows sorted into SECTIONS order",
          [x["section"] for x in r["rows"]]
          == ["business_overview", "business_overview", "financials", "risks"])
    check("order within a section preserved",
          [x["label"] for x in r["rows"][:2]] == ["Vintage", "Segments"])

    # --- section_counts is ordered by SECTIONS, so the mail order is deterministic
    c = section_counts([{"section": "risks"}, {"section": "financials"},
                        {"section": "risks"}])
    check("counts ordered by SECTIONS", list(c) == ["financials", "risks"])
    check("counts correct", c == {"financials": 1, "risks": 2})
    check("coverage pair", coverage([{"section": "risks"}]) == (1, 18))

    # --- a DataFrame built from the rows keeps the declared column order
    df = pd.DataFrame(P({"summary": [dup]})["rows"], columns=DECK_SUMMARY_COLS)
    check("dataframe round-trip", list(df.columns) == DECK_SUMMARY_COLS and len(df) == 1)

    print(f"deck_summary self-test: {ok} passed, {fail} failed")
    return 1 if fail else 0


if __name__ == "__main__":
    sys.exit(_self_test())
