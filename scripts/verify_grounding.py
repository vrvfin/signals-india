r"""
verify_grounding.py — Gates 1 and 2 of the anti-hallucination stack. NO LLM.

These run between generation (Layer B) and rendering (Layer D). They are mechanical, so
they cannot be talked out of a verdict the way a model can.

  GATE 1 — NUMERIC.  Every number a model writes must resolve to a fact in the fact pack.
    Presentation rounding is allowed (a fact of 81.7857 may be written "82%"), because
    the check compares at the PRECISION THE MODEL CHOSE TO DISPLAY. Writing more decimals
    is therefore a stricter claim, and a wrong one fails: 5.78% does not match 5.7189.

  GATE 2 — SPAN.  Every qualitative claim must carry an `evidence_span` that appears
    VERBATIM in the cited source document. Verified by string containment after
    whitespace/punctuation normalisation (PDF text has erratic spacing and curly quotes).
    A model cannot fabricate a quotation that is not literally in the document.

Neither gate judges whether a claim is *reasonable* — that is the auditor's job (Layer C).
These only establish that the number exists and the words were actually said.

Usage:
  python scripts/verify_grounding.py --selftest
  python scripts/verify_grounding.py --factpack fp.json --narrative nar.json \
                                     --sources srcs.json
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from dataclasses import dataclass, asdict
from pathlib import Path

# ------------------------------------------------------------------ tokens ----
# A number optionally preceded by a currency marker and followed by a unit/scale word.
#  The comma-grouped branch must REQUIRE at least one comma group. With `*` it matched
#  the first 1-3 digits of any long run, so "2026" was read as the number 202 and every
#  doc-id date produced a phantom finding. Full digit runs are matched by the second
#  branch.
_NUM_RE = re.compile(r"""
    (?P<sign>[+-])?
    (?P<num>\d{1,3}(?:,\d{2,3})+(?:\.\d+)?|\d+(?:\.\d+)?)
    \s*
    (?P<suffix>%|percent|bps|x|times|pp|percentage\s+points)?
""", re.VERBOSE | re.IGNORECASE)

# Numbers that are labels, not measurements. These are never checked against facts.
_FY_RE = re.compile(r"\bFY\s?\d{2,4}\b", re.IGNORECASE)
_QTR_RE = re.compile(r"\bQ[1-4]\s?FY?\s?\d{2,4}\b", re.IGNORECASE)
_DATE_RE = re.compile(r"\b(?:19|20)\d{2}\b")
_SECTION_RE = re.compile(r"\b(?:section|slide|page|note|part|table)\s*\.?\s*\d+", re.I)
_ORDINAL_RE = re.compile(r"\b\d+(?:st|nd|rd|th)\b", re.IGNORECASE)
_ID_RE = re.compile(r"\b[A-Z]{2}[A-Z0-9]{9}\d\b")          # ISIN
_LIST_RE = re.compile(r"^\s*\d+[.)]\s", re.MULTILINE)      # "1." list markers
# Source citations the model writes inline, e.g. "(concall_2026-06-03)" or a bare ISO
# date. `\b` does not fire between "_" and a digit, so a doc_id's year survived masking
# and every citation produced a phantom numeric finding.
_DOCID_RE = re.compile(r"\b[A-Za-z][A-Za-z_]*_\d{4}-\d{2}-\d{2}\b")
_ISODATE_RE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")
# Fact ids themselves ("lev.absorption_pct", "tbl.sensitivity_pe") — the trailing tokens
# can contain digits.
_FACTID_RE = re.compile(r"\b[a-z][a-z0-9]*\.[a-z0-9_.]+\b")

_SCALE = {"cr": 1.0, "crore": 1.0, "crores": 1.0, "lakh": 0.01, "lakhs": 0.01}


@dataclass
class Finding:
    gate: str
    severity: str          # "fail" | "warn"
    section: str
    detail: str
    excerpt: str = ""
    nearest: str = ""

    def line(self) -> str:
        n = f"  nearest: {self.nearest}" if self.nearest else ""
        return (f"[{self.gate}/{self.severity}] s{self.section}: {self.detail}\n"
                f"  ...{self.excerpt}...{n}")


# ------------------------------------------------------------ normalisation ---
def normalise(text: str) -> str:
    """Collapse the differences that PDF extraction introduces but meaning does not:
    unicode quotes/dashes/spaces -> ASCII, runs of whitespace -> one space."""
    t = unicodedata.normalize("NFKC", str(text))
    for a, b in (("‘", "'"), ("’", "'"), ("“", '"'), ("”", '"'),
                 ("–", "-"), ("—", "-"), ("−", "-"), (" ", " ")):
        t = t.replace(a, b)
    return re.sub(r"\s+", " ", t).strip()


def _mask_labels(text: str) -> str:
    """Blank out numbers that are identifiers rather than measurements, so Gate 1 does
    not demand a fact for 'FY26' or 'Q1 FY27'."""
    out = text
    # _DOCID_RE / _ISODATE_RE / _FACTID_RE must run BEFORE _DATE_RE so the whole citation
    # is blanked rather than leaving digit fragments behind.
    for rx in (_ID_RE, _DOCID_RE, _ISODATE_RE, _FACTID_RE, _QTR_RE, _FY_RE,
               _SECTION_RE, _ORDINAL_RE, _DATE_RE, _LIST_RE):
        out = rx.sub(lambda m: " " * len(m.group(0)), out)
    return out


def _decimals(literal: str) -> int:
    return len(literal.split(".")[1]) if "." in literal else 0


def _to_float(literal: str) -> float | None:
    try:
        return float(literal.replace(",", ""))
    except ValueError:
        return None


# ------------------------------------------------------------------ gate 1 ----
def _fact_values(factpack: dict) -> list[tuple[str, float, str]]:
    """(id, value, unit) for every numeric fact, plus every numeric table cell."""
    out = []
    for f in factpack.get("facts", []):
        v = f.get("value")
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            out.append((f["id"], float(v), str(f.get("unit", ""))))
    for t in factpack.get("tables", []):
        for i, row in enumerate(t.get("rows", [])):
            for k, v in row.items():
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    out.append((f"{t['id']}[{i}].{k}", float(v), ""))
    return out


def _label_numbers(factpack: dict) -> list[float]:
    """Numbers appearing in fact LABELS and table notes are licensed for use in prose.
    The fact "PBT move per 1% EBITDA move" makes "for every 1% change in EBITDA" a
    restatement of the fact's own definition, not an unsourced new figure."""
    vals: list[float] = []
    texts = [str(f.get("label", "")) for f in factpack.get("facts", [])]
    texts += [str(t.get("title", "")) + " " + str(t.get("note", ""))
              for t in factpack.get("tables", [])]
    for s in texts:
        for m in re.finditer(r"\d+(?:\.\d+)?", s):
            v = _to_float(m.group(0))
            if v is not None:
                vals.append(v)
    return vals


def verify_numbers(prose: str, factpack: dict, section: str = "?",
                   extra_allowed: list[float] | None = None) -> list[Finding]:
    """Gate 1. Every measurement in `prose` must match a fact at the precision written."""
    facts = _fact_values(factpack)
    allowed = list(extra_allowed or []) + _label_numbers(factpack)
    findings: list[Finding] = []
    masked = _mask_labels(prose)

    for m in _NUM_RE.finditer(masked):
        literal = m.group("num")
        val = _to_float(literal)
        if val is None:
            continue
        suffix = (m.group("suffix") or "").strip().lower()
        # Tiny bare integers are almost always counts in prose ("three brands",
        # "2 of the 4"). Requiring a fact for them produces noise, not safety.
        if not suffix and "." not in literal and "," not in literal and abs(val) < 10:
            continue
        if m.group("sign") == "-":
            val = -val

        dec = _decimals(literal)
        # Scale words let "1,051 Cr" match a fact stored in Rs Cr, and "21.96 lakh"
        # match one stored in lakh.
        tail = masked[m.end():m.end() + 12].lower()
        scale = next((s for w, s in _SCALE.items()
                      if re.match(rf"\s*{w}\b", tail)), 1.0)

        target = val * scale
        hit = None
        for fid, fv, _unit in facts:
            # Sign is compared on ABSOLUTE value. Prose legitimately carries direction in
            # words — "EBITDA lagged revenue by 2.63 percentage points" against a stored
            # -2.63 is correct English, and demanding the minus sign flagged good writing.
            # A genuinely inverted claim is a semantic error for the auditor (Gate 3),
            # not something this string check can adjudicate.
            for cand in (fv, fv * scale):
                if (round(cand, dec) == round(target, dec)
                        or round(abs(cand), dec) == round(abs(target), dec)):
                    hit = fid
                    break
            if hit:
                break
        if hit or any(round(abs(a), dec) == round(abs(target), dec) for a in allowed):
            continue

        # Report the closest fact, so a wrong number is obvious at a glance.
        near = min(facts, key=lambda t: abs(t[1] - target), default=None)
        findings.append(Finding(
            gate="G1-numeric", severity="fail", section=section,
            detail=f"number {m.group(0).strip()!r} does not resolve to any fact "
                   f"at {dec} dp",
            excerpt=normalise(prose[max(0, m.start() - 45):m.end() + 45]),
            nearest=(f"{near[0]} = {near[1]}" if near else "")))
    return findings


# ------------------------------------------------------------------ gate 2 ----
def verify_spans(claims: list[dict], sources: dict[str, str],
                 section: str = "?") -> list[Finding]:
    """Gate 2. Each claim needs source_ref{doc_id, evidence_span}, and that span must
    appear verbatim in sources[doc_id]."""
    findings: list[Finding] = []
    norm_sources = {k: normalise(v) for k, v in sources.items()}

    for c in claims:
        text = str(c.get("text", ""))[:110]
        ref = c.get("source_ref") or {}
        doc_id, span = ref.get("doc_id"), ref.get("evidence_span")
        if not doc_id or not span:
            findings.append(Finding("G2-span", "fail", section,
                                    "claim carries no source_ref{doc_id, evidence_span}",
                                    normalise(text)))
            continue
        if doc_id not in norm_sources:
            findings.append(Finding("G2-span", "fail", section,
                                    f"cites doc_id {doc_id!r}, which was not supplied "
                                    f"to the generator",
                                    normalise(text),
                                    f"available: {sorted(norm_sources)[:6]}"))
            continue
        nspan, ndoc = normalise(span), norm_sources[doc_id]
        if nspan in ndoc:
            continue
        # Case is the one difference that never changes what was said: quoting
        # "This is the year..." mid-sentence as "...this is the year..." is correct
        # practice, not fabrication. Match case-insensitively so the gate stays aimed
        # at invention, but surface the drift as a warning rather than hiding it.
        if nspan.lower() in ndoc.lower():
            findings.append(Finding("G2-span", "warn", section,
                                    f"evidence_span matches {doc_id} only after "
                                    f"case-folding — quote was re-cased",
                                    nspan[:160]))
            continue
        findings.append(Finding("G2-span", "fail", section,
                                f"evidence_span not found verbatim in {doc_id}",
                                nspan[:160]))
    return findings


# ----------------------------------------------------------------- driver -----
def verify_narrative(factpack: dict, narrative: dict,
                     sources: dict[str, str]) -> list[Finding]:
    """Run both gates over every section of a narrative document."""
    out: list[Finding] = []
    for sec in narrative.get("sections", []):
        sid = str(sec.get("id", sec.get("section", "?")))
        prose = " ".join(str(sec.get(k, "")) for k in
                         ("takeaway", "body", "text", "note"))
        out += verify_numbers(prose, factpack, sid)
        out += verify_spans(sec.get("claims", []) or [], sources, sid)
    return out


def summarise(findings: list[Finding]) -> dict:
    by_gate: dict[str, int] = {}
    for f in findings:
        by_gate[f.gate] = by_gate.get(f.gate, 0) + 1
    return {"total": len(findings), "by_gate": by_gate,
            "passed": not any(f.severity == "fail" for f in findings)}


# ---------------------------------------------------------------- selftest ----
def _selftest() -> int:
    fails: list[str] = []

    def expect(name, cond):
        print(f"  {'PASS' if cond else 'FAIL'}  {name}")
        if not cond:
            fails.append(name)

    factpack = {
        "facts": [
            {"id": "fin.revenue.FY26", "value": 4896.0, "unit": "Rs Cr"},
            {"id": "fin.pat.FY26", "value": 38.0, "unit": "Rs Cr"},
            {"id": "lev.absorption_pct", "value": 81.7857, "unit": "%"},
            {"id": "fin.margin.FY26", "value": 5.7189, "unit": "%"},
            {"id": "one.mcap_cr", "value": 2133.0, "unit": "Rs Cr"},
        ],
        "tables": [{"id": "tbl.sens", "rows": [{"revenue": 4896, "pe": 55.0}]}],
    }
    sources = {
        "concall_2026Q4": "We are now entering a more consolidation phase where the "
                          "emphasis is on optimizing our existing assets and sweating "
                          "them. This is the year where we want to get the profits back.",
        "ar_FY26": "The Company operates 140 outlets across 29 cities.",
    }

    print("\n[1] GATE 1 accepts numbers that resolve, at the precision written")
    ok = verify_numbers("Revenue was Rs 4,896 Cr and PAT Rs 38 Cr; the fixed block "
                        "absorbed 82% of EBITDA. Market cap Rs 2,133 Cr.", factpack, "18")
    for f in ok:
        print("   ", f.line())
    expect("clean prose produces no findings", not ok)

    print("\n[2] GATE 1 catches an ALTERED number (4,896 -> 4,869)")
    bad = verify_numbers("Revenue was Rs 4,869 Cr in FY26.", factpack, "18")
    for f in bad:
        print("   ", f.line())
    expect("altered number flagged", len(bad) == 1)
    expect("nearest fact reported", bool(bad and "fin.revenue.FY26" in bad[0].nearest))

    print("\n[3] GATE 1 is stricter when the model writes MORE decimals")
    strict = verify_numbers("The margin was 5.78%.", factpack, "18")
    loose = verify_numbers("The margin was 5.7%.", factpack, "18")
    for f in strict:
        print("   ", f.line())
    expect("5.78% rejected (fact is 5.7189)", len(strict) == 1)
    expect("5.7% accepted as presentation rounding", not loose)

    print("\n[4] GATE 1 ignores labels, not measurements")
    lab = verify_numbers("In FY26 and Q1 FY27, per slide 30 and section 22, the "
                         "ISIN INE559R01029 company grew. See page 207.",
                         factpack, "1")
    for f in lab:
        print("   ", f.line())
    expect("FY/quarter/slide/ISIN/page numbers not demanded as facts", not lab)

    print("\n[5] GATE 2 accepts a verbatim quote")
    good_claim = [{"text": "Management framed FY27 as consolidation.",
                   "source_ref": {"doc_id": "concall_2026Q4",
                                  "evidence_span": "this is the year where we want "
                                                   "to get the profits back"}}]
    g = verify_spans(good_claim, sources, "20")
    for f in g:
        print("   ", f.line())
    # Re-cased quotes pass (severity warn), because case never changes what was said.
    expect("re-cased span passes as a warning, not a failure",
           len(g) == 1 and g[0].severity == "warn")
    expect("a warning does not fail the run", summarise(g)["passed"] is True)

    exact_claim = [{"text": "Management framed FY27 as consolidation.",
                    "source_ref": {"doc_id": "concall_2026Q4",
                                   "evidence_span": "This is the year where we want "
                                                    "to get the profits back"}}]
    expect("exact span passes", not verify_spans(exact_claim, sources, "20"))

    print("\n[6] GATE 2 catches a FABRICATED quote")
    fake = [{"text": "Management guided to 12% margins.",
             "source_ref": {"doc_id": "concall_2026Q4",
                            "evidence_span": "we expect margins of 12% next year"}}]
    ff = verify_spans(fake, sources, "20")
    for f in ff:
        print("   ", f.line())
    expect("fabricated quote flagged", len(ff) == 1 and "verbatim" in ff[0].detail)

    print("\n[7] GATE 2 catches a missing ref and an unknown document")
    miss = verify_spans([{"text": "The business is durable."}], sources, "5")
    unknown = verify_spans([{"text": "x", "source_ref": {"doc_id": "ar_FY99",
                                                         "evidence_span": "anything"}}],
                           sources, "5")
    for f in miss + unknown:
        print("   ", f.line())
    expect("claim with no source_ref flagged", len(miss) == 1)
    expect("claim citing an unsupplied document flagged", len(unknown) == 1)

    print("\n[8] Normalisation tolerates PDF whitespace and curly quotes")
    messy = {"d": "We  are\nnow   entering a “more consolidation” phase"}
    nq = verify_spans([{"text": "x", "source_ref": {
        "doc_id": "d", "evidence_span": 'now entering a "more consolidation" phase'}}],
        messy, "1")
    expect("curly quotes + collapsed whitespace still match", not nq)

    print("\n[9] End-to-end over a narrative document")
    narrative = {"sections": [
        {"id": "18", "takeaway": "Revenue reached Rs 4,896 Cr.",
         "body": "PAT was Rs 38 Cr.", "claims": []},
        {"id": "20", "takeaway": "Margin was 5.78%.",     # wrong precision -> G1
         "body": "", "claims": fake},                     # fabricated quote  -> G2
    ]}
    res = verify_narrative(factpack, narrative, sources)
    for f in res:
        print("   ", f.line())
    s = summarise(res)
    print(f"   summary: {s}")
    expect("end-to-end finds exactly the two planted errors", s["total"] == 2)
    expect("one from each gate", s["by_gate"].get("G1-numeric") == 1
           and s["by_gate"].get("G2-span") == 1)
    expect("overall verdict is FAIL", s["passed"] is False)

    print("\n" + ("SELFTEST FAILED: " + "; ".join(fails) if fails else "SELFTEST PASSED"))
    return 1 if fails else 0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--factpack", help="factpack.json from narrative_factpack.py")
    ap.add_argument("--narrative", help="narrative.json from the generator")
    ap.add_argument("--sources", help="JSON {doc_id: full text} supplied to the generator")
    ap.add_argument("--json", action="store_true", help="emit findings as JSON")
    a = ap.parse_args()

    if a.selftest:
        return _selftest()
    if not (a.factpack and a.narrative):
        ap.error("--factpack and --narrative are required (or use --selftest)")

    fp = json.loads(Path(a.factpack).read_text(encoding="utf-8"))
    nar = json.loads(Path(a.narrative).read_text(encoding="utf-8"))
    src = json.loads(Path(a.sources).read_text(encoding="utf-8")) if a.sources else {}
    findings = verify_narrative(fp, nar, src)
    s = summarise(findings)
    if a.json:
        print(json.dumps({"summary": s, "findings": [asdict(f) for f in findings]},
                         indent=2))
    else:
        for f in findings:
            print(f.line())
        print(f"\n{s['total']} finding(s): {s['by_gate']} — "
              f"{'PASS' if s['passed'] else 'FAIL'}")
    return 0 if s["passed"] else 2


if __name__ == "__main__":
    sys.exit(main())
