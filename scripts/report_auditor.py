r"""
report_auditor.py — Layer C. An INDEPENDENT model re-reads a finished report against the
sources and names every claim it cannot support.

Independence is the whole point, so two rules are enforced structurally:
  1. The auditor sees the EVIDENCE — source documents plus the computed fact table — and
     the finished section text. It never sees the generator's prompt, its instructions or
     its reasoning, so it cannot inherit the generator's framing.

     The fact table is supplied deliberately, and this was a correction. Withholding it
     made every financial figure unverifiable: the first real run marked correct claims
     like "Revenue reached Rs 4,896 crore in FY26" as UNSUPPORTED, because that number
     comes from Screener statements rather than from any filing in the document bundle.
     A fact-checker denied the accounts cannot check the accounts. Independence means not
     sharing the generator's framing — not being refused the underlying data.
  2. It runs on a DIFFERENT MODEL FAMILY (Cerebras gpt-oss-120b) from the Gemini
     generator. A second Gemini call has correlated blind spots. If no alt provider is
     reachable the run falls back to a different Gemini model, sets
     `degraded_fallback=true`, and the renderer prints that on the report — a degraded
     audit must never look like a clean one.

Verdicts: VERIFIED · PARTIAL · UNSUPPORTED · CONTRADICTED · NOT_IN_SOURCE

Usage:
  python scripts/report_auditor.py --calibrate            # HARD GATE, see below
  python scripts/report_auditor.py --report r.md --sources s.json --out audit.json
  python scripts/report_auditor.py --narrative n.json --sources s.json --out audit.json

CALIBRATION IS A GATE, NOT A NICETY. `--calibrate` plants three errors (a wrong figure,
an invented management quote, a claim about an undisclosed metric) in a report whose
sources are known, and fails unless all three are caught. An auditor that flags nothing
manufactures false confidence and is worse than no auditor at all.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from gemini_pool import load_keys

# Independent adjudicators, tried in order. Cerebras handles long text on the free tier;
# Groq's ~8k tokens/min cap makes it viable only for short sections.
ALT_CHAIN = [("cerebras", "gpt-oss-120b"), ("groq", "openai/gpt-oss-120b")]
# Degraded fallback ONLY — same family as the generator, so correlated errors.
GEMINI_FALLBACK = ["gemini-2.5-flash", "gemini-flash-latest"]

#  TOTAL prompt budget, not per document. The first live run sent 3 documents at 60k
#  each (~45k tokens) and every Cerebras call returned HTTP 429. provider_router.py had
#  already measured this: `_ALT_MAX_CHARS = 80_000` — "eval showed ~80k chars (~20k tok)
#  is safe on Cerebras; bigger requests 429". Budgeting per-document instead of in total
#  was the bug.
MAX_TOTAL_SOURCE_CHARS = 70_000
MAX_SOURCE_CHARS = 45_000        # ceiling for any single document within that total
# On a 429 the payload is halved and retried once — a smaller audit beats no audit.
RETRY_SHRINK = 0.5
VERDICTS = ("VERIFIED", "PARTIAL", "UNSUPPORTED", "CONTRADICTED", "NOT_IN_SOURCE")
# Documents most likely to carry the evidence for a management claim, cheapest first.
# Annual reports are 400k chars and would consume the whole budget alone.
_DOC_PRIORITY = ("concall", "presentation", "results", "rating", "annual_report")

AUDIT_PROMPT = """You are an independent fact-checker. You did NOT write the text below \
and you have no stake in it being correct.

You are given EVIDENCE — a DATA TABLE of figures extracted from the company's filings, \
plus SOURCE DOCUMENTS (transcripts and reports) — and a SECTION of a finished \
equity-research report. Your only job is to decide, for each distinct factual claim in \
the section, whether the EVIDENCE supports it.

RULES
- Judge ONLY against the evidence provided. If neither the DATA TABLE nor the SOURCE \
DOCUMENTS mention something, the claim is UNSUPPORTED even if it sounds plausible or is \
common knowledge.
- Numeric claims are checked against the DATA TABLE. A figure matching a table row is \
VERIFIED even though no source document repeats it — the table IS a source. Rounding a \
table value for readability (81.7857 written as "82%") is VERIFIED, not PARTIAL.
- Claims about what management said, intends, or explains are checked against the SOURCE \
DOCUMENTS, not the table.
- A number is CONTRADICTED if the sources state a different value. Presentation rounding \
(4,896.2 written as "4,896") is NOT a contradiction.
- A quotation is CONTRADICTED or UNSUPPORTED unless those words appear in a source.
- Do NOT judge whether the analysis is wise, only whether it is supported.
- Do NOT invent claims that are not in the section. Do not audit headings or table \
formatting.
- Audit only FACTUAL ASSERTIONS ABOUT THE COMPANY. Skip disclaimers ("this is not a \
forecast"), methodology notes ("this framework shows how..."), statements about what the \
report itself does, and statements that a figure is not disclosed. These are not claims \
about the company and marking them UNSUPPORTED is noise that buries the real findings.
- Be conservative: if a claim is substantially supported but overstated in detail, use \
PARTIAL rather than UNSUPPORTED.

VERDICTS (use exactly one per claim)
  VERIFIED      the sources directly support it
  PARTIAL       broadly supported but the specifics are overstated or imprecise
  UNSUPPORTED   nothing in the sources speaks to it
  CONTRADICTED  the sources state otherwise
  NOT_IN_SOURCE it cites a document that was not provided to you

Return STRICT JSON, no prose, no markdown fence:
{"claims":[{"claim":"<the claim, quoted or closely paraphrased, max 200 chars>",
            "verdict":"<one of the five>",
            "reason":"<why, citing the source, max 300 chars>",
            "source_quote":"<verbatim supporting/contradicting text from a source, or \"\">"}]}

=== DATA TABLE — figures extracted from this company's filings and financial statements ===
{fact_table}

=== SOURCE DOCUMENTS ===
{sources}

=== REPORT SECTION {section_id}: {section_title} ===
{section_text}
"""


# ------------------------------------------------------------------- json -----
def _salvage_json(text: str) -> dict | None:
    """Lite models fence, prepend prose, or truncate. Recover what we can rather than
    discarding a whole section's audit over a stray backtick."""
    if not text:
        return None
    t = text.strip()
    t = re.sub(r"^```(?:json)?\s*", "", t)
    t = re.sub(r"\s*```$", "", t).strip()
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        pass
    i = t.find("{")
    if i < 0:
        return None
    # try progressively shorter prefixes ending at a closing brace
    for j in range(len(t), i, -1):
        if t[j - 1] != "}":
            continue
        try:
            return json.loads(t[i:j])
        except json.JSONDecodeError:
            continue
    # last resort: pull individual claim objects out of a truncated array
    claims = []
    for m in re.finditer(r'\{[^{}]*"verdict"\s*:\s*"[A-Z_]+"[^{}]*\}', t):
        try:
            claims.append(json.loads(m.group(0)))
        except json.JSONDecodeError:
            continue
    return {"claims": claims} if claims else None


# ------------------------------------------------------------------ pools -----
class Adjudicator:
    """Picks an independent model, falling back only when forced — and says so."""

    def __init__(self, prefer_alt: bool = True):
        self.model = None
        self.degraded = False
        self._alt = None
        self._gem = None
        if prefer_alt:
            from altllm_pool import AltPool
            for provider, model in ALT_CHAIN:
                try:
                    p = AltPool(provider)
                    if p.keys:
                        self._alt, self.model = (p, model), f"{provider}/{model}"
                        return
                except Exception:
                    continue
        keys = (load_keys(os.environ, prefix="FREE_POOL")
                or load_keys(os.environ, prefix="GEMINI_API_KEY"))
        if not keys:
            raise RuntimeError("no adjudicator keys: set CEREBRAS_API_KEY_n / "
                               "GROQ_API_KEY_n, or FREE_POOL_n for the degraded path")
        from _extractor_base import GeminiKeyPool
        self._gem = GeminiKeyPool(keys, GEMINI_FALLBACK)
        self.model = f"gemini/{GEMINI_FALLBACK[0]}"
        self.degraded = True

    def call(self, prompt: str, max_tokens: int = 4000) -> str:
        if self._alt:
            pool, model = self._alt
            return pool.call_text(prompt, model, max_output_tokens=max_tokens)
        return self._gem.call_text(prompt, "audit", max_output_tokens=max_tokens)


# ----------------------------------------------------------------- auditing ---
def _doc_rank(doc_id: str) -> int:
    for i, p in enumerate(_DOC_PRIORITY):
        if str(doc_id).startswith(p):
            return i
    return len(_DOC_PRIORITY)


def _fit_sources(sources: dict[str, str], budget: int) -> str:
    """Pack documents into a TOTAL character budget, most-relevant first. A document
    that does not fit is dropped and named, so a truncated evidence base is visible
    rather than silent."""
    if not sources:
        return "(no source documents were supplied)"
    parts, used, dropped = [], 0, []
    for k in sorted(sources, key=lambda d: (_doc_rank(d), d)):
        room = budget - used
        if room < 4_000:
            dropped.append(k)
            continue
        txt = sources[k][:min(MAX_SOURCE_CHARS, room)]
        parts.append(f"--- DOCUMENT {k} ---\n{txt}")
        used += len(txt)
    if dropped:
        parts.append("--- NOTE: these documents did not fit the context budget and were "
                     "NOT available to this audit: " + ", ".join(dropped) + " ---")
    return "\n\n".join(parts)


def fact_table_text(factpack: dict | None, section_id=None) -> str:
    """The computed figures, as an evidence table. Section-scoped when a section id is
    given, so the auditor sees what that section could legitimately have used."""
    if not factpack:
        return ("(no data table supplied — numeric claims cannot be verified and should "
                "be reported as UNSUPPORTED)")
    lines = ["label | value | unit | basis"]
    for f in factpack.get("facts", []):
        if section_id is not None and str(f.get("section")) != str(section_id):
            continue
        lines.append(f"{f.get('label')} | {f.get('value')} | {f.get('unit', '')} | "
                     f"{f.get('basis', '')}")
    for t in factpack.get("tables", []):
        if section_id is not None and str(t.get("section")) != str(section_id):
            continue
        lines.append(f"\nTABLE {t.get('title')} — columns {t.get('columns')}")
        for row in t.get("rows", [])[:20]:
            lines.append("  " + " | ".join(f"{k}={v}" for k, v in row.items()))
    return "\n".join(lines) if len(lines) > 1 else "(no figures for this section)"


def audit_section(adj: Adjudicator, section_id, section_title: str,
                  section_text: str, sources: dict[str, str],
                  factpack: dict | None = None) -> list[dict]:
    if not section_text.strip():
        return []
    budget = MAX_TOTAL_SOURCE_CHARS
    last_err = ""
    facts_blob = fact_table_text(factpack, section_id)
    for attempt in range(2):
        prompt = (AUDIT_PROMPT
                  .replace("{fact_table}", facts_blob)
                  .replace("{sources}", _fit_sources(sources, int(budget)))
                  .replace("{section_id}", str(section_id))
                  .replace("{section_title}", section_title)
                  .replace("{section_text}", section_text))
        try:
            raw = adj.call(prompt)
            break
        except Exception as e:
            last_err = str(e)[:200]
            # A rate-limit rejection is about payload size, so shrink and retry once.
            if attempt == 0 and ("429" in last_err or "rate" in last_err.lower()):
                budget = int(budget * RETRY_SHRINK)
                continue
            return [{"claim": f"(section {section_id} could not be audited)",
                     "verdict": "AUDIT_FAILED", "section": section_id,
                     "reason": f"adjudicator call failed: {last_err}",
                     "source_quote": ""}]
    data = _salvage_json(raw)
    if not data or "claims" not in data:
        return [{"claim": f"(section {section_id} audit returned unparseable output)",
                 "verdict": "AUDIT_FAILED", "section": section_id,
                 "reason": f"adjudicator response could not be parsed: {raw[:200]}",
                 "source_quote": ""}]
    out = []
    for c in data["claims"]:
        v = str(c.get("verdict", "")).upper().strip()
        out.append({"claim": str(c.get("claim", ""))[:300],
                    "verdict": v if v in VERDICTS else "UNSUPPORTED",
                    "reason": str(c.get("reason", ""))[:400],
                    "source_quote": str(c.get("source_quote", ""))[:400],
                    "section": section_id})
    return out


def split_markdown(md: str) -> list[tuple[str, str, str]]:
    """A markdown report -> [(id, title, body)] using '### <n>. <title>' headings, so an
    existing company_deepdive_*.md can be audited without being regenerated."""
    parts = re.split(r"^#{2,3}\s+(.+)$", md, flags=re.MULTILINE)
    out, i = [], 1
    while i + 1 < len(parts) + 1 and i < len(parts):
        head, body = parts[i].strip(), (parts[i + 1] if i + 1 < len(parts) else "")
        m = re.match(r"^(\d+(?:\.\d+)?)[.)]?\s+(.*)$", head)
        sid, title = (m.group(1), m.group(2)) if m else (head[:24], head)
        if body.strip():
            out.append((sid, title, body.strip()))
        i += 2
    return out


def audit_report(adj: Adjudicator, sections: list[tuple[str, str, str]],
                 sources: dict[str, str], limit: int = 0,
                 factpack: dict | None = None) -> dict:
    claims = []
    todo = sections[:limit] if limit else sections
    for sid, title, body in todo:
        claims += audit_section(adj, sid, title, body, sources, factpack)
    counts = {v.lower(): sum(1 for c in claims if c["verdict"] == v) for v in VERDICTS}
    failed = [c for c in claims if c["verdict"] == "AUDIT_FAILED"]
    counts["audit_failed"] = len(failed)
    counts["total"] = len(claims)
    # A run where every section errored previously reported "0 verified · 0 unsupported
    # · 0 contradicted", which reads like a clean audit. `ran` and `sections_failed`
    # make the difference between "nothing wrong was found" and "nothing was checked"
    # explicit for every consumer of this file.
    audited = {str(c["section"]) for c in claims if c["verdict"] != "AUDIT_FAILED"}
    counts["sections_audited"] = len(audited)
    counts["sections_failed"] = len({str(c["section"]) for c in failed})
    return {"model": adj.model, "degraded_fallback": adj.degraded,
            "ran": bool(audited),
            "failure_reason": (failed[0]["reason"] if failed and not audited else ""),
            "summary": counts, "claims": claims}


# -------------------------------------------------------------- calibration ---
# Real-shaped sources with known content. The report below contains exactly three
# planted errors; the auditor must catch all three to be considered fit.
_CAL_SOURCES = {
    "concall_Q4FY26": (
        "Sanjay Thakker: We are now entering a more consolidation phase where the "
        "emphasis is on optimizing our existing assets and sweating them. This is the "
        "year where we want to get the profits back. "
        "Analyst: What is the capex plan? "
        "CFO: We do not have a very large capex in mind for this year. Our historic "
        "average is around INR 50 crores. That could be taken as a ballpark. "
        "Analyst: And on margins? "
        "CFO: We are not giving a margin guidance for FY27 at this stage."
    ),
    "annual_report_FY26": (
        "Revenue from operations for the year ended 31 March 2026 was Rs 4,896.2 crore, "
        "against Rs 4,025.5 crore in the previous year. Profit after tax was Rs 38.1 "
        "crore. The Company operated 140 outlets across 29 cities as at 31 March 2026. "
        "Finance costs were Rs 79.8 crore. The Board recommended a dividend of Rs 1.50 "
        "per equity share."
    ),
}

_CAL_REPORT = [
    ("18", "The full financial record",
     # PLANT 1 — wrong figure: the sources say 4,896.2, this says 5,120.
     "Revenue from operations reached Rs 5,120 crore in FY26, up from Rs 4,025.5 crore "
     "the year before. Profit after tax was Rs 38.1 crore and finance costs were "
     "Rs 79.8 crore."),
    ("20", "Management's claim, tested",
     # PLANT 2 — invented quote: management explicitly declined to guide on margin.
     "Management framed FY27 as a year of consolidation, saying \"this is the year "
     "where we want to get the profits back\". The CFO added that the company "
     "\"expects EBITDA margins of at least 8% in FY27\" and guided capex to a "
     "historic average of about Rs 50 crore."),
    ("2", "The company on one page",
     # PLANT 3 — undisclosed metric: no source states market share.
     "The company operated 140 outlets across 29 cities. Its share of the Indian "
     "luxury car retail market stood at 21% in FY26."),
]

_CAL_EXPECT = {
    "18": ["revenue", "5,120", "5120"],
    "20": ["margin", "8%", "ebitda"],
    "2": ["share", "21%", "market"],
}


def calibrate(adj: Adjudicator, verbose: bool = True) -> int:
    print(f"adjudicator: {adj.model}"
          + ("  [DEGRADED — same family as the generator]" if adj.degraded else
             "  [independent family]"))
    print("planting 3 errors: a wrong figure (s18), an invented quote (s20), "
          "a claim about an undisclosed metric (s2)\n")
    res = audit_report(adj, _CAL_REPORT, _CAL_SOURCES)
    caught, missed = {}, []
    for sid, keys in _CAL_EXPECT.items():
        hits = [c for c in res["claims"]
                if str(c["section"]) == sid
                and c["verdict"] in ("UNSUPPORTED", "CONTRADICTED", "NOT_IN_SOURCE")
                and any(k in (c["claim"] + " " + c["reason"]).lower() for k in keys)]
        caught[sid] = hits
        if not hits:
            missed.append(sid)

    for sid, _, _ in _CAL_REPORT:
        print(f"--- section {sid} ---")
        for c in res["claims"]:
            if str(c["section"]) != sid:
                continue
            mark = "FLAG" if c["verdict"] in ("UNSUPPORTED", "CONTRADICTED",
                                              "NOT_IN_SOURCE") else "ok  "
            print(f"  [{mark}] {c['verdict']:<13} {c['claim'][:96]}")
            if verbose and mark == "FLAG":
                print(f"           reason: {c['reason'][:150]}")
    print(f"\nsummary: {res['summary']}")
    print(f"planted errors caught: {3 - len(missed)}/3")
    for sid in _CAL_EXPECT:
        print(f"  section {sid}: {'CAUGHT' if caught[sid] else 'MISSED'}")

    # A useful auditor must also not condemn the true statements around the plants.
    clean = [c for c in res["claims"]
             if c["verdict"] in ("UNSUPPORTED", "CONTRADICTED")
             and any(t in c["claim"].lower() for t in
                     ("140 outlets", "29 cities", "79.8", "38.1", "consolidation"))]
    if clean:
        print(f"\nWARNING: {len(clean)} well-supported claim(s) were flagged — "
              f"an auditor that over-flags is as unusable as one that under-flags:")
        for c in clean[:5]:
            print(f"  {c['verdict']}: {c['claim'][:110]}")

    ok = not missed and not clean
    print("\n" + ("CALIBRATION PASSED — auditor is fit for use"
                  if ok else
                  f"CALIBRATION FAILED — missed {missed or 'none'}, "
                  f"false positives {len(clean)}"))
    return 0 if ok else 1


# --------------------------------------------------------------------- main ---
def _safe(s: str) -> str:
    """Windows consoles default to cp1252, which cannot encode the rupee sign — printing
    an audited claim crashed the run after all the LLM spend. Degrade the glyph rather
    than lose the output."""
    enc = (sys.stdout.encoding or "utf-8")
    try:
        s.encode(enc)
        return s
    except UnicodeEncodeError:
        return s.encode(enc, errors="replace").decode(enc, errors="replace")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--calibrate", action="store_true",
                    help="plant 3 known errors and require the auditor to catch them")
    ap.add_argument("--report", help="markdown report to audit")
    ap.add_argument("--narrative", help="narrative.json to audit")
    ap.add_argument("--factpack", help="factpack.json — REQUIRED to verify numeric claims")
    ap.add_argument("--sources", help="JSON {doc_id: text}")
    ap.add_argument("--out", help="write audit.json here")
    ap.add_argument("--limit", type=int, default=0, help="audit only the first N sections")
    ap.add_argument("--force-gemini", action="store_true",
                    help="skip alt providers (produces a DEGRADED audit)")
    a = ap.parse_args()

    adj = Adjudicator(prefer_alt=not a.force_gemini)
    if a.calibrate:
        return calibrate(adj)

    if not (a.report or a.narrative):
        ap.error("give --report or --narrative (or --calibrate)")
    sources = json.loads(Path(a.sources).read_text(encoding="utf-8")) if a.sources else {}
    if a.report:
        sections = split_markdown(Path(a.report).read_text(encoding="utf-8"))
    else:
        nar = json.loads(Path(a.narrative).read_text(encoding="utf-8"))
        sections = [(str(s.get("id")), str(s.get("title", "")),
                     " ".join(str(s.get(k, "")) for k in ("takeaway", "body")))
                    for s in nar.get("sections", [])]
    fp = (json.loads(Path(a.factpack).read_text(encoding="utf-8"))
          if a.factpack else None)
    if fp is None:
        print("WARNING: no --factpack given. Numeric claims will be judged with no data "
              "table and will come back UNSUPPORTED.")
    print(f"auditing {len(sections)} section(s) with {adj.model}"
          + ("  [DEGRADED]" if adj.degraded else ""))
    res = audit_report(adj, sections, sources, a.limit, fp)
    print(json.dumps(res["summary"], indent=2))
    if not res.get("ran"):
        print("AUDIT DID NOT RUN — no section was checked. "
              + _safe(res.get("failure_reason", "")))
    for c in res["claims"]:
        if c["verdict"] not in ("VERIFIED", "PARTIAL"):
            print(_safe(f"  [{c['verdict']}] s{c['section']}: {c['claim'][:120]}"))
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=2), encoding="utf-8")
        print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
