r"""
narrative_generate.py — Layer B. Gemini writes the prose; it never produces a number.

For each section the model receives only that section's facts, tables, gaps and the
source documents. It returns strict JSON. Gates 1 and 2 (verify_grounding) run on the
result, and a section that fails is REGENERATED ONCE with the findings fed back as
constraints. Whatever survives the second attempt is kept and flagged — publishing an
annotated failure is honest; looping until the model deletes the hard claims is not.

Usage:
  python scripts/narrative_generate.py --factpack fp.json --sources s.json \
                                       --out narrative.json
  python scripts/narrative_generate.py --factpack fp.json --sections 18 19 --dry-run
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

import verify_grounding as VG
from gemini_pool import load_keys

GEN_MODELS = ["gemini-2.5-flash", "gemini-flash-latest", "gemini-2.5-pro"]
# Gemini 2.5 models are THINKING models: max_output_tokens covers reasoning tokens as
# well as the visible answer. At 3000 the budget was spent thinking and the JSON came
# back truncated mid-sentence (1,721 chars, unparseable) — which looked like a prompt
# problem but was purely a budget one. At 12000 the same call returns 5,057 chars and
# parses. Do not lower this without re-checking that sections still parse.
MAX_OUTPUT_TOKENS = 12000
INTER_CALL_SLEEP = 3.0
MAX_SOURCE_CHARS = 45_000

PROMPT_FILE = "narrative_report_prompt.txt"

# What each section is FOR. Given to the model so it writes the right thing, and kept
# here rather than in the prompt file so the two stay in step with the renderer.
SECTION_BRIEFS: dict[int, tuple[str, str]] = {
    1: ("History and strategic phases",
        "How the company got here, in dated milestones, and the two or three distinct "
        "strategic eras those milestones fall into."),
    2: ("The company on one page",
        "What this business is and its current scale. Orient a reader who knows nothing."),
    3: ("Legal entity structure",
        "The listed parent and its subsidiaries, and why the structure matters "
        "analytically — what it separates, what it exposes."),
    4: ("Management and governance",
        "Who runs it, their tenure and background, and any governance item on the record."),
    5: ("Scale history",
        "Whether profit kept up as the business grew. Use the CAGR pair and the gap "
        "between them; be explicit about which window is measured and why."),
    6: ("Segment economics",
        "Which part of the business earns the money, versus which part books the "
        "revenue. The asymmetry between revenue share and profit share is the point."),
    7: ("Accounting basis and credit ratings",
        "Which basis the headline figures are stated on, and what the credit-rating "
        "agencies concluded — an independent read on the balance sheet that does not "
        "come from the company."),
    8: ("The steadiest series",
        "The most stable disclosed series, the band it has held, and over which years. "
        "Do not claim stability across loss-making years."),
    9: ("The portfolio as disclosed",
        "The brands, products or plants the company discloses, with contribution."),
    10: ("A framework over the portfolio",
         "Group the portfolio by what actually drives its economics. State plainly "
         "that the grouping is the author's, not company terminology."),
    11: ("What moved this year",
         "The mix shift versus the prior year and what it implies."),
    12: ("Unit deep dives",
         "One paragraph per material unit: scale, trajectory, and its role in the whole."),
    16: ("Testing a management claim against independent data",
         "Take one specific, checkable management claim and test it against the "
         "independent series. State the verdict either way."),
    17: ("Independent data scorecard and its limits",
         "What the independent data says, its methodology, and explicitly what it does "
         "NOT tell you."),
    18: ("The full financial record",
         "Read the multi-year table: the trajectory, the inflection, and which basis "
         "each figure is on."),
    19: ("Operating leverage as arithmetic",
         "How much of EBITDA the fixed block absorbs and what that does to profit in "
         "both directions. This is division of disclosed figures, not a forecast."),
    20: ("Management's claim, tested across calls",
         "What management said across consecutive calls, verbatim, and what the "
         "disclosed record shows against it."),
    21: ("The peer set",
         "Who the comparable companies are and where this one sits. Name the limits of "
         "the comparison."),
    22: ("Implied multiples",
         "What the market capitalisation implies across the revenue and margin grid. "
         "A restatement under stated assumptions, never a target."),
    23: ("Structural and counterparty risks",
         "Risks arising from what the business does not control, each with the "
         "company's stated position and whether it is live or latent."),
    24: ("Financial, policy and execution risks",
         "Risks from the balance sheet, the policy environment and execution."),
    25: ("Policy backdrop",
         "The policy and regulatory changes bearing on this business."),
    26: ("Findings",
         "The numbered findings this report establishes, each pointing at the section "
         "that supports it. Introduce no new facts."),
    27: ("Recent exchange filings",
         "A dated digest of what the company has filed with BSE/NSE recently: orders, "
         "board actions, capacity, ratings. Say what was filed, not what it implies "
         "beyond the filing. Flag any filing that corroborates or contradicts a claim "
         "made elsewhere in this report."),
}


def _fact_table(facts: list[dict]) -> str:
    if not facts:
        return "(no facts available for this section)"
    lines = ["id | label | value | unit | basis"]
    for f in facts:
        lines.append(f"{f['id']} | {f['label']} | {f['value']} | "
                     f"{f.get('unit', '')} | {f.get('basis', '')}")
    return "\n".join(lines)


def _tables_blob(tables: list[dict]) -> str:
    if not tables:
        return "(none)"
    out = []
    for t in tables:
        out.append(f"{t['id']}: {t['title']} — columns {t['columns']}, "
                   f"{len(t['rows'])} rows. {t.get('note', '')}")
    return "\n".join(out)


def _sources_blob(sources: dict[str, str]) -> str:
    if not sources:
        return "(no source documents supplied — make no qualitative claims)"
    return "\n\n".join(f"--- DOCUMENT {k} ---\n{v[:MAX_SOURCE_CHARS]}"
                       for k, v in sources.items())


def build_prompt(tpl: str, company: dict, sec: int, facts, tables, gaps,
                 sources: dict[str, str]) -> str:
    title, brief = SECTION_BRIEFS.get(sec, (f"Section {sec}", ""))
    return (tpl
            .replace("{company}", f"{company['name']} ({company['symbol']} / "
                                  f"{company['isin']}) — an Indian listed company; "
                                  f"all figures in Indian rupees (Rs crore).")
            .replace("{section_id}", str(sec))
            .replace("{section_title}", title)
            .replace("{section_brief}", brief)
            .replace("{fact_table}", _fact_table(facts))
            .replace("{tables}", _tables_blob(tables))
            .replace("{gaps}", "\n".join(f"- {g}" for g in gaps) or "(none)")
            .replace("{sources}", _sources_blob(sources)))


def _parse(raw: str) -> dict | None:
    import re
    t = (raw or "").strip()
    t = re.sub(r"^```(?:json)?\s*", "", t)
    t = re.sub(r"\s*```$", "", t).strip()
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        pass
    i = t.find("{")
    for j in range(len(t), i, -1):
        if i < 0 or t[j - 1] != "}":
            continue
        try:
            return json.loads(t[i:j])
        except json.JSONDecodeError:
            continue
    return None


def generate_section(pool, tpl, pack, sec: int, sources: dict[str, str],
                     log=print) -> dict:
    company = pack["company"]
    facts = [f for f in pack["facts"] if f.get("section") == sec]
    tables = [t for t in pack["tables"] if t.get("section") == sec]
    gaps = ([g["reason"] for g in pack.get("coverage_gaps", []) if g["section"] == sec]
            + [u["reason"] for u in pack.get("uncovered_sections", [])
               if u["section"] == sec])
    if not (facts or tables):
        title, _ = SECTION_BRIEFS.get(sec, (f"Section {sec}", ""))
        return {"id": str(sec), "title": title, "takeaway": "", "body": "",
                "claims": [], "not_disclosed": gaps, "skipped": "no facts or tables"}

    prompt = build_prompt(tpl, company, sec, facts, tables, gaps, sources)
    subpack = {"facts": facts, "tables": tables}
    attempt, findings, out = 0, [], None
    # Keep the BEST attempt, not the last. A retry that returns unparseable output is
    # strictly worse than a first attempt that parsed with gate failures — overwriting
    # with the later garbage silently emptied real sections.
    best: tuple[int, dict, list] | None = None

    def _score(parsed, fnd) -> int:
        if parsed is None:
            return -1
        return 100 - len([f for f in fnd if f.severity == "fail"])

    while attempt < 2:
        attempt += 1
        try:
            raw = pool.call_text(prompt, f"narrative_s{sec}",
                                 max_output_tokens=MAX_OUTPUT_TOKENS)
        except Exception as e:
            log(f"    section {sec}: generation failed — {str(e)[:140]}")
            return {"id": str(sec), "title": SECTION_BRIEFS.get(sec, ("", ""))[0],
                    "takeaway": "", "body": "", "claims": [],
                    "not_disclosed": gaps, "error": str(e)[:200]}
        out = _parse(raw)
        if out is None:
            log(f"    section {sec}: unparseable JSON on attempt {attempt}")
            findings = [VG.Finding("parse", "fail", str(sec),
                                   "response was not valid JSON")]
            prompt += ("\n\nYour previous response was not valid JSON. Return ONLY the "
                       "JSON object, with no prose and no markdown fence.")
            continue

        prose = " ".join(str(out.get(k, "")) for k in ("takeaway", "body"))
        findings = (VG.verify_numbers(prose, subpack, str(sec))
                    + VG.verify_spans(out.get("claims") or [], sources, str(sec)))
        hard = [f for f in findings if f.severity == "fail"]
        if best is None or _score(out, findings) > _score(best[1], best[2]):
            best = (attempt, out, findings)
        if not hard:
            out["gate_findings"] = [f.__dict__ for f in findings]
            out["attempts"] = attempt
            return out
        log(f"    section {sec}: {len(hard)} gate failure(s) on attempt {attempt}")
        if attempt >= 2:
            break
        # Feed the failures back as constraints rather than a vague "try again".
        detail = "\n".join(f"- {f.detail} (near: {f.excerpt[:110]})" for f in hard)
        prompt += (f"\n\nYour previous answer FAILED these mechanical checks:\n{detail}\n"
                   f"Rewrite the section. Use ONLY numbers present in the FACT TABLE, at "
                   f"the precision shown, and make every evidence_span an exact "
                   f"substring of a SOURCE DOCUMENT. Drop any claim you cannot ground.")

    if best is not None:
        attempt_used, out, findings = best
    else:
        attempt_used = attempt
        out = {"id": str(sec), "title": SECTION_BRIEFS.get(sec, ("", ""))[0],
               "takeaway": "", "body": "", "claims": []}
    out["gate_findings"] = [f.__dict__ for f in findings]
    out["gate_failed"] = True
    out["attempts"] = attempt_used
    log(f"    section {sec}: KEPT WITH FLAGS (best of {attempt} attempts was #"
        f"{attempt_used}, {len([f for f in findings if f.severity == 'fail'])} "
        f"unresolved failure(s))")
    return out


def generate(pack: dict, sources: dict[str, str], sections=None, log=print) -> dict:
    keys = load_keys(os.environ, prefix="FREE_POOL") or \
        load_keys(os.environ, prefix="GEMINI_API_KEY")
    if not keys:
        raise RuntimeError("no Gemini keys — set FREE_POOL_n or GEMINI_API_KEY_n")
    from _extractor_base import GeminiKeyPool
    pool = GeminiKeyPool(keys, GEN_MODELS)
    tpl = (Path(_HERE) / PROMPT_FILE).read_text(encoding="utf-8")

    have = sorted({f["section"] for f in pack["facts"]}
                  | {t["section"] for t in pack["tables"]})
    todo = [s for s in have if not sections or s in sections]
    log(f"  generating {len(todo)} section(s): {todo}")
    out = []
    for i, sec in enumerate(todo):
        log(f"  [{i + 1}/{len(todo)}] section {sec}")
        out.append(generate_section(pool, tpl, pack, sec, sources, log))
        if i < len(todo) - 1:
            time.sleep(INTER_CALL_SLEEP)
    flagged = sum(1 for s in out if s.get("gate_failed"))
    return {"company": pack["company"], "sections": out,
            "generator_model": GEN_MODELS[0],
            "sections_with_unresolved_gate_failures": flagged}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--factpack", required=True)
    ap.add_argument("--sources", help="JSON {doc_id: text}")
    ap.add_argument("--sections", nargs="*", type=int, default=None)
    ap.add_argument("--out")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the prompt for the first section and exit; no LLM call")
    a = ap.parse_args()

    pack = json.loads(Path(a.factpack).read_text(encoding="utf-8"))
    sources = json.loads(Path(a.sources).read_text(encoding="utf-8")) if a.sources else {}

    if a.dry_run:
        tpl = (Path(_HERE) / PROMPT_FILE).read_text(encoding="utf-8")
        have = sorted({f["section"] for f in pack["facts"]})
        sec = (a.sections or have)[0]
        facts = [f for f in pack["facts"] if f["section"] == sec]
        tables = [t for t in pack["tables"] if t["section"] == sec]
        gaps = [g["reason"] for g in pack.get("coverage_gaps", [])
                if g["section"] == sec]
        p = build_prompt(tpl, pack["company"], sec, facts, tables, gaps, sources)
        print(p)
        print(f"\n--- DRY RUN: section {sec}, {len(facts)} facts, {len(tables)} tables, "
              f"{len(p):,} prompt chars. No LLM call made. ---")
        return 0

    res = generate(pack, sources, a.sections)
    print(json.dumps({"sections": len(res["sections"]),
                      "flagged": res["sections_with_unresolved_gate_failures"]}, indent=2))
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=2), encoding="utf-8")
        print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
