r"""
provenance.py — every number on a teardown page carries where it came from.

The problem this solves: a rendered figure is indistinguishable from a fabricated one
once it is HTML. So nothing reaches the renderer as a bare float. It reaches as a Fact
that knows its source table, field, period, grain and origin — and the renderer is
physically unable to draw anything else.

Five failure modes, five defences:

  1. LLM invents a number      -> origin="llm" requires `evidence` (verbatim source text)
                                  AND grounded() must find the number inside it.
  2. Transcription drift       -> derived values are COMPUTED at render time from Facts
                                  (Fact.derive), never typed. audit() re-exports the whole
                                  chain so any figure can be traced to a parquet cell.
  3. Wrong join               -> every Fact carries `key` (the isin it was filtered on);
                                  audit_join() asserts one company per page.
  4. Grain mixing             -> grain is mandatory and Fact.derive REFUSES to combine
                                  quarterly with annual unless allow_mixed_grain=True.
  5. Absence read as clean     -> MISSING is a real Fact with value None; renderers show
                                  "not covered". There is no silent zero.

Pure stdlib + no Drive/network, so it is unit-testable offline.

    python scripts/provenance.py --self-test
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field, replace
from typing import Any, Iterable, Sequence

# origins
SCREENER = "screener"      # a cell read straight out of a Screener-sourced parquet
COMPUTED = "computed"      # arithmetic over other Facts
LLM = "llm"                # extracted from a document by a model — needs evidence
AGENCY = "agency"          # verbatim from a rating agency / auditor document

# grains
QUARTERLY = "quarterly"
ANNUAL = "annual"
EVENT = "event"            # a dated occurrence (rating action, announcement)
NONE_GRAIN = "none"

_GRAINS = {QUARTERLY, ANNUAL, EVENT, NONE_GRAIN}
_ORIGINS = {SCREENER, COMPUTED, LLM, AGENCY}


class ProvenanceError(ValueError):
    """Raised when a Fact would be constructed or combined unsafely."""


@dataclass(frozen=True)
class Fact:
    """One value plus everything needed to defend it.

    `value` may be None — that is a MISSING fact, and it renders as "not covered",
    never as zero. This distinction is load-bearing: a company with no annual report
    processed must never look like a company with a clean annual report.
    """
    value: Any
    source: str                       # e.g. "fundamentals/statements/APLAPOLLO.parquet"
    field: str                        # e.g. "Net Profit" / "cfo_pat_ratio"
    period: str = ""                  # "Jun 2026" | "FY26" | "2026-06-30"
    grain: str = NONE_GRAIN
    origin: str = SCREENER
    unit: str = ""
    key: str = ""                     # the isin this was filtered on
    evidence: str = ""                # verbatim source text (required for LLM/AGENCY)
    doc_id: str = ""                  # source_doc_id
    note: str = ""                    # how it was computed, in words
    inputs: tuple = field(default=(), repr=False)   # tuple[Fact, ...]

    def __post_init__(self):
        if self.origin not in _ORIGINS:
            raise ProvenanceError(f"unknown origin {self.origin!r}")
        if self.grain not in _GRAINS:
            raise ProvenanceError(f"unknown grain {self.grain!r}")
        if self.origin in (LLM, AGENCY) and self.present and not self.evidence.strip():
            raise ProvenanceError(
                f"{self.origin} fact {self.field!r} has no evidence — refusing to build it")

    # ---------- state ----------
    @property
    def present(self) -> bool:
        return self.value is not None

    @property
    def missing(self) -> bool:
        return self.value is None

    def __bool__(self) -> bool:          # so `if fact:` means "has a value"
        return self.present

    @property
    def num(self) -> float | None:
        """Value as float, or None. Never raises — a non-numeric Fact is simply missing
        for arithmetic purposes."""
        if self.value is None:
            return None
        try:
            return float(str(self.value).replace(",", "").replace("%", "").strip())
        except (TypeError, ValueError):
            return None

    # ---------- derivation ----------
    @classmethod
    def derive(cls, value, field_: str, inputs: Sequence["Fact"], note: str,
               unit: str = "", allow_mixed_grain: bool = False) -> "Fact":
        """A computed Fact. Inherits period/grain/key from its inputs and records them,
        so audit() can replay the arithmetic.

        Refuses to mix quarterly and annual inputs unless explicitly allowed — that is
        the single correctness rule this whole framework rests on.
        """
        inputs = tuple(inputs)
        grains = {f.grain for f in inputs if f.grain != NONE_GRAIN}
        if not allow_mixed_grain and len(grains) > 1:
            raise ProvenanceError(
                f"{field_}: refusing to mix grains {sorted(grains)} — pass "
                f"allow_mixed_grain=True and say so in the render if this is intended")
        keys = {f.key for f in inputs if f.key}
        if len(keys) > 1:
            raise ProvenanceError(f"{field_}: inputs span multiple companies {sorted(keys)}")
        periods = [f.period for f in inputs if f.period]
        return cls(
            value=value, source="computed", field=field_,
            period=periods[0] if periods else "",
            grain=(grains.pop() if len(grains) == 1 else NONE_GRAIN),
            origin=COMPUTED, unit=unit, key=(keys.pop() if keys else ""),
            note=note, inputs=inputs,
        )

    def with_key(self, key: str) -> "Fact":
        return replace(self, key=key)

    # ---------- audit ----------
    def trace(self, depth: int = 0) -> list[dict]:
        """This Fact and, recursively, everything it was computed from."""
        row = {
            "depth": depth, "field": self.field, "value": self.value, "unit": self.unit,
            "period": self.period, "grain": self.grain, "origin": self.origin,
            "source": self.source, "key": self.key, "note": self.note,
        }
        if self.evidence:
            row["evidence"] = self.evidence
        if self.doc_id:
            row["doc_id"] = self.doc_id
        out = [row]
        for f in self.inputs:
            out.extend(f.trace(depth + 1))
        return out


def MISSING(field_: str, source: str, why: str = "", period: str = "",
            grain: str = NONE_GRAIN, key: str = "") -> Fact:
    """An explicitly absent value. Renders as 'not covered' — never 0, never blank."""
    return Fact(value=None, source=source, field=field_, period=period, grain=grain,
                key=key, note=why or "no row in source")


# --------------------------------------------------------------------------- #
# Grounding — the defence against a model inventing a figure
# --------------------------------------------------------------------------- #

_NUM_RE = re.compile(r"-?\d[\d,]*\.?\d*")


def _norm_nums(text: str) -> set[str]:
    """Every number in `text`, normalised so 1,234.0 / 1234 / 1234.00 all match."""
    out = set()
    for m in _NUM_RE.findall(text or ""):
        try:
            f = float(m.replace(",", ""))
        except ValueError:
            continue
        out.add(f"{f:.4f}".rstrip("0").rstrip("."))
    return out


def grounded(value, evidence: str, tolerance_pct: float = 0.0) -> bool:
    """True when `value` actually occurs in `evidence`.

    This is how an LLM-extracted number earns the right to be displayed: the model must
    return the source span it read the number from, and the number must be findable in
    that span. A model that paraphrases a figure it invented will fail this.

    tolerance_pct > 0 allows rounding (a deck saying "8.1%" for a computed 8.13%).
    """
    if value is None:
        return False
    try:
        v = float(str(value).replace(",", "").replace("%", "").strip())
    except (TypeError, ValueError):
        # non-numeric claim: require the text to be non-trivial, nothing more to check
        return bool(str(value).strip()) and len(evidence.strip()) >= 10
    cands = _norm_nums(evidence)
    target = f"{v:.4f}".rstrip("0").rstrip(".")
    if target in cands:
        return True
    if tolerance_pct > 0:
        for c in cands:
            try:
                cv = float(c)
            except ValueError:
                continue
            if cv == 0:
                continue
            if abs(cv - v) / abs(cv) * 100 <= tolerance_pct:
                return True
    return False


def llm_fact(value, field_: str, evidence: str, doc_id: str, source: str,
             period: str = "", grain: str = NONE_GRAIN, unit: str = "", key: str = "",
             tolerance_pct: float = 1.0, origin: str = LLM) -> Fact:
    """Build an LLM-sourced Fact, or MISSING if it cannot be grounded in its evidence.

    Returning MISSING rather than raising is deliberate: one ungrounded row should drop
    out of the page, not kill the whole report.
    """
    if not evidence or not evidence.strip():
        return MISSING(field_, source, "model returned no evidence span", period, grain, key)
    if not grounded(value, evidence, tolerance_pct):
        return MISSING(field_, source,
                       f"value {value!r} not found in its own evidence — dropped",
                       period, grain, key)
    return Fact(value=value, source=source, field=field_, period=period, grain=grain,
                origin=origin, unit=unit, key=key, evidence=evidence.strip(), doc_id=doc_id)


def cross_check(llm: Fact, deterministic: Fact, tolerance_pct: float = 2.0) -> Fact:
    """Prefer the deterministic Fact whenever both exist and they disagree.

    A model reading revenue off a slide is never allowed to overrule the filing. When
    they agree we keep the deterministic one anyway — same number, better provenance.
    """
    if deterministic.missing:
        return llm
    if llm.missing:
        return deterministic
    a, b = llm.num, deterministic.num
    if a is None or b is None or b == 0:
        return deterministic
    if abs(a - b) / abs(b) * 100 > tolerance_pct:
        return replace(deterministic,
                       note=(deterministic.note + " | " if deterministic.note else "")
                            + f"model claimed {a:g}; filing says {b:g} — filing wins")
    return deterministic


# --------------------------------------------------------------------------- #
# Page-level audit
# --------------------------------------------------------------------------- #

def audit_join(facts: Iterable[Fact], expect_key: str) -> list[str]:
    """Every Fact on a page must belong to the company the page is about.

    Catches the join bug this repo is genuinely exposed to: ar_red_flags.symbol
    sometimes holds a BSE code, so a symbol-join can attribute another company's
    auditor flags to this one.
    """
    bad = []
    for f in facts:
        if f.key and f.key != expect_key:
            bad.append(f"{f.field} ({f.source}) carries key {f.key!r}, expected {expect_key!r}")
    return bad


def audit(facts: Iterable[Fact], key: str, symbol: str, quarter: str) -> dict:
    """The sidecar written beside every rendered page.

    With this file you can take any figure off the page and walk it back to the parquet
    cell it came from, or to the verbatim sentence a model read it in.
    """
    rows, seen = [], set()
    for f in facts:
        for r in f.trace():
            sig = (r["field"], str(r["value"]), r["period"], r["source"], r["depth"])
            if sig in seen:
                continue
            seen.add(sig)
            rows.append(r)
    by_origin: dict[str, int] = {}
    for r in rows:
        by_origin[r["origin"]] = by_origin.get(r["origin"], 0) + 1
    return {
        "isin": key, "symbol": symbol, "quarter": quarter,
        "n_facts": len(rows),
        "by_origin": by_origin,
        "n_missing": sum(1 for r in rows if r["value"] is None),
        "join_violations": audit_join(facts, key),
        "facts": rows,
    }


def write_audit(path: str, facts: Iterable[Fact], key: str, symbol: str,
                quarter: str) -> dict:
    a = audit(facts, key, symbol, quarter)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(a, fh, indent=1, default=str)
    return a


# --------------------------------------------------------------------------- #
def _self_test() -> int:
    ok, fail = 0, 0

    def check(name, cond):
        nonlocal ok, fail
        if cond:
            ok += 1
        else:
            fail += 1
            print(f"  FAIL {name}")

    sales = Fact(5607.0, "statements/APLAPOLLO.parquet", "Sales", "Jun 2026",
                 QUARTERLY, SCREENER, "Cr", key="INE702C01027")
    pat = Fact(263.0, "statements/APLAPOLLO.parquet", "Net Profit", "Jun 2026",
               QUARTERLY, SCREENER, "Cr", key="INE702C01027")
    cfo = Fact(2103.0, "statements/APLAPOLLO.parquet", "CFO", "Mar 2026",
               ANNUAL, SCREENER, "Cr", key="INE702C01027")

    npm = Fact.derive(pat.num / sales.num * 100, "npm_pct", [pat, sales],
                      "Net Profit / Sales x 100", "%")
    check("derive computes", abs(npm.num - 4.6906) < 1e-3)
    check("derive keeps grain", npm.grain == QUARTERLY)
    check("derive keeps key", npm.key == "INE702C01027")
    check("trace reaches leaves", len(npm.trace()) == 3)

    try:
        Fact.derive(1.0, "bad_mix", [pat, cfo], "quarterly / annual")
        check("grain mixing blocked", False)
    except ProvenanceError:
        check("grain mixing blocked", True)

    other = Fact(1.0, "x", "y", "Jun 2026", QUARTERLY, SCREENER, key="INE999")
    try:
        Fact.derive(1.0, "bad_join", [pat, other], "two companies")
        check("cross-company blocked", False)
    except ProvenanceError:
        check("cross-company blocked", True)

    m = MISSING("receivable_days", "financials_derived", "ratios section not scraped")
    check("MISSING is falsy", not m)
    check("MISSING is not zero", m.value is None and m.num is None)

    check("grounded finds plain", grounded(875000, "volume target of 875,000 tonnes"))
    check("grounded finds comma'd", grounded(4500, "spread above INR 4,500/tonne"))
    check("grounded rejects invented", not grounded(912000, "volume target of 875,000 tonnes"))
    check("grounded tolerance", grounded(8.13, "margin of 8.1%", tolerance_pct=1.0))

    good = llm_fact(875000, "volume_target", "volume target of 875,000 tonnes",
                    "doc1", "deck_metrics", key="INE702C01027")
    check("llm_fact keeps grounded", good.present and good.origin == LLM)
    bad = llm_fact(912000, "volume_target", "volume target of 875,000 tonnes",
                   "doc1", "deck_metrics", key="INE702C01027")
    check("llm_fact drops ungrounded", bad.missing)

    try:
        Fact(5.0, "deck", "utilisation", origin=LLM)
        check("llm without evidence blocked", False)
    except ProvenanceError:
        check("llm without evidence blocked", True)

    claim = llm_fact(5800, "revenue", "revenue of 5,800 crore in the quarter", "d",
                     "deck", period="Jun 2026", grain=QUARTERLY, key="INE702C01027")
    won = cross_check(claim, sales)
    check("filing overrules deck", won.num == 5607.0 and "filing says" in won.note)

    a = audit([npm, m, good], "INE702C01027", "APLAPOLLO", "Q1 FY27")
    check("audit counts missing", a["n_missing"] >= 1)
    check("audit finds no join violation", a["join_violations"] == [])
    stray = Fact(1.0, "ar_red_flags", "flag", key="INE_WRONG")
    check("audit catches bad join", len(audit_join([stray], "INE702C01027")) == 1)

    print(f"\nprovenance self-test: {ok} passed, {fail} failed")
    return 1 if fail else 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        sys.exit(_self_test())
    ap.print_help()
