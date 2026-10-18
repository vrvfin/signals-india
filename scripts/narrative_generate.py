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

# FALLBACK only — the models come from the registry (model_registry.pick("WRITE"):
# discover -> probe -> quality exam), user 2026-10-03. gemini-2.5-pro was dropped: 404 on
# 2026-10-03. This list is used only when the registry is missing or stale.
GEN_MODELS = ["gemini-2.5-flash", "gemini-flash-latest"]
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
    return _write_and_gate(pool, prompt, subpack, sources, str(sec),
                           SECTION_BRIEFS.get(sec, ("", ""))[0], gaps, log)


def _write_and_gate(pool, prompt: str, subpack: dict, sources: dict[str, str],
                    uid: str, title: str, gaps: list, log=print,
                    label: str = "section") -> dict:
    """Write one unit (a story section, or a chapter of the merged report), run
    Gates 1 and 2 on it and regenerate ONCE with the failures fed back. Shared so
    the chapter writer gets exactly the grounding the section writer has."""
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
            raw = pool.call_text(prompt, f"narrative_s{uid}",
                                 max_output_tokens=MAX_OUTPUT_TOKENS)
        except Exception as e:
            log(f"    {label} {uid}: generation failed — {str(e)[:140]}")
            return {"id": uid, "title": title,
                    "takeaway": "", "body": "", "claims": [],
                    "not_disclosed": gaps, "error": str(e)[:200]}
        out = _parse(raw)
        if out is not None:
            # The model sometimes REWRITES the id it was given ("scale-history" for 5); the
            # renderer matches prose by id, so those sections printed no prose at all
            # (Morepen 2026-10-04: 5 and 19). The section number is ours, not the model's.
            out["id"] = uid
        if out is None:
            log(f"    {label} {uid}: unparseable JSON on attempt {attempt}")
            findings = [VG.Finding("parse", "fail", uid,
                                   "response was not valid JSON")]
            prompt += ("\n\nYour previous response was not valid JSON. Return ONLY the "
                       "JSON object, with no prose and no markdown fence.")
            continue

        prose = " ".join(str(out.get(k, "")) for k in ("takeaway", "body"))
        # Spans are checked FIRST so the ones that verify can license the numbers they
        # contain. A figure quoted from a filing is grounded even when it is not in the
        # computed fact table — but only if the surrounding span provably appears in
        # that filing, which is exactly what verify_spans establishes.
        claims = out.get("claims") or []
        span_findings, verified_spans = VG.verify_spans_detailed(
            claims, sources, uid)
        findings = (VG.verify_numbers(prose, subpack, uid,
                                      licensed_text=" ".join(verified_spans))
                    + span_findings)
        hard = [f for f in findings if f.severity == "fail"]
        if best is None or _score(out, findings) > _score(best[1], best[2]):
            best = (attempt, out, findings)
        if not hard:
            out["gate_findings"] = [f.__dict__ for f in findings]
            out["attempts"] = attempt
            return out
        log(f"    {label} {uid}: {len(hard)} gate failure(s) on attempt {attempt}")
        if attempt >= 2:
            break
        # Feed the failures back as constraints rather than a vague "try again".
        detail = "\n".join(f"- {f.detail} (near: {f.excerpt[:110]})" for f in hard)
        prompt += (f"\n\nYour previous answer FAILED these mechanical checks:\n{detail}\n"
                   f"Rewrite the {label}. Use ONLY numbers present in the FACT TABLE, at "
                   f"the precision shown, and make every evidence_span an exact "
                   f"substring of a SOURCE DOCUMENT. Drop any claim you cannot ground.")

    if best is not None:
        attempt_used, out, findings = best
    else:
        attempt_used = attempt
        out = {"id": uid, "title": title,
               "takeaway": "", "body": "", "claims": []}
    out["gate_findings"] = [f.__dict__ for f in findings]
    out["gate_failed"] = True
    out["attempts"] = attempt_used
    log(f"    {label} {uid}: KEPT WITH FLAGS (best of {attempt} attempts was #"
        f"{attempt_used}, {len([f for f in findings if f.severity == 'fail'])} "
        f"unresolved failure(s))")
    return out


def _writer_pool(log=print):
    """(pool, models) for the writer: FREE_POOL keys, models from the registry."""
    keys = load_keys(os.environ, prefix="FREE_POOL") or \
        load_keys(os.environ, prefix="GEMINI_API_KEY")
    if not keys:
        raise RuntimeError("no Gemini keys — set FREE_POOL_n or GEMINI_API_KEY_n")
    from _extractor_base import GeminiKeyPool
    from model_registry import pick, overload_keys
    models = pick("WRITE", GEN_MODELS, log=log)
    pool = GeminiKeyPool(keys, models)
    # Give-up rule scales with the keys (a third of them, not 3) — set on THIS pool only,
    # so the shared GeminiKeyPool / Phase-2 extractors are untouched. 2026-10-03: with 3,
    # two models were dropped after 3 busy replies each and 10 of 20 sections came back empty.
    pool._pool.model_overload_keys = overload_keys(len(keys))
    log(f"  writer pool: {len(keys)} key(s) · {len(models)} model(s) · a model is dropped "
        f"after busy replies on {pool._pool.model_overload_keys} keys")
    return pool, models


def generate(pack: dict, sources: dict[str, str], sections=None, log=print) -> dict:
    pool, models = _writer_pool(log)
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
    from model_registry import record_usage          # busy demotion sees these models
    record_usage(pool, "narrative_writer", "narrative", log=log)
    return {"company": pack["company"], "sections": out,
            "generator_model": models[0],
            "sections_with_unresolved_gate_failures": flagged}


# ---------------------------------------------------------- merged company report --
# User 2026-10-17: ONE report in one flow. A storyline pass plans the whole company
# (business dynamics + financial dynamics, one point per chapter); each chapter is then
# written CONTINUING the previous one, with the deep dive's text as analyst notes. The
# number gate and span check are the section writer's (_write_and_gate), unchanged.
STORY_PROMPT_FILE = "narrative_storyline_prompt.txt"
CHAPTER_PROMPT_FILE = "narrative_chapter_prompt.txt"
STORY_NOTES_CHARS = 9000


def _storyline(pool, pack: dict, chapters: list[dict], forensic_md: str, log=print) -> dict:
    """One call: the report's plan. Its numbers go through the same number gate (whole
    fact pack); a plan that still fails after one retry is kept and flagged."""
    import render_company_report as RCR
    dd = RCR.parse_forensic(forensic_md or "")
    notes = "\n\n".join(RCR._plain_notes(dd[k]) for k in ("onepager", "1.7", "0.5", "1.6", "0.6")
                        if dd.get(k))[:STORY_NOTES_CHARS]
    co = pack["company"]
    tpl = (Path(_HERE) / STORY_PROMPT_FILE).read_text(encoding="utf-8")
    prompt = (tpl
              .replace("{company}", f"{co['name']} ({co['symbol']} / {co['isin']}) — all "
                                    f"figures in Indian rupees (Rs crore).")
              .replace("{chapters}", "\n".join(f"{c['n']}. {c['title']} — {c['brief']}"
                                               for c in chapters))
              .replace("{fact_table}", _fact_table(pack["facts"]))
              .replace("{notes}", notes or "(no deep dive for this copy)"))
    best = None
    for attempt in (1, 2):
        try:
            raw = pool.call_text(prompt, "narrative_storyline",
                                 max_output_tokens=MAX_OUTPUT_TOKENS)
        except Exception as e:
            log(f"    storyline: generation failed — {str(e)[:140]}")
            break
        out = _parse(raw)
        if out is None:
            log(f"    storyline: unparseable JSON on attempt {attempt}")
            prompt += ("\n\nYour previous response was not valid JSON. Return ONLY the "
                       "JSON object, with no prose and no markdown fence.")
            continue
        prose = " ".join([str(out.get("one_page", ""))]
                         + [str(x) for k in ("business_dynamics", "financial_dynamics",
                                             "watch") for x in (out.get(k) or [])])
        hard = [f for f in VG.verify_numbers(prose, pack, "story") if f.severity == "fail"]
        if best is None or len(hard) < best[1]:
            best = (out, len(hard))
        if not hard:
            break
        log(f"    storyline: {len(hard)} gate failure(s) on attempt {attempt}")
        prompt += ("\n\nYour previous answer used numbers that are NOT in the FACT TABLE:\n"
                   + "\n".join(f"- {f.detail} (near: {f.excerpt[:110]})" for f in hard)
                   + "\nRewrite it with ONLY numbers from the FACT TABLE, or none.")
    if best is None:
        return {"one_page": "", "business_dynamics": [], "financial_dynamics": [],
                "watch": [], "chapters": [], "failed": True}
    out, n_hard = best
    out["gate_failed"] = bool(n_hard)
    return out


def generate_report(pack: dict, sources: dict[str, str], forensic_md: str = "",
                    log=print) -> dict:
    """The merged report's text: storyline + one gated chapter at a time. Chapters with
    nothing to write about are not called at all (render_company_report.chapter_inputs)."""
    import render_company_report as RCR
    pool, models = _writer_pool(log)
    chapters = RCR.chapter_inputs(pack, forensic_md)
    cpack = RCR.chapter_pack(pack)
    log(f"  writing {len(chapters)} chapter(s): {[c['n'] for c in chapters]}"
        + (" — with the deep dive as analyst notes" if forensic_md else " — no deep dive"))
    log("  [storyline]")
    story = _storyline(pool, pack, chapters, forensic_md, log)
    plan = {int(c.get("n", 0)): c for c in story.get("chapters") or []
            if str(c.get("n", "")).isdigit()}
    story_txt = "\n".join(
        [str(story.get("one_page", ""))]
        + [f"- {x}" for x in story.get("business_dynamics") or []]
        + [f"- {x}" for x in story.get("financial_dynamics") or []]) or "(no plan)"
    tpl = (Path(_HERE) / CHAPTER_PROMPT_FILE).read_text(encoding="utf-8")
    co = pack["company"]
    out, prev = [], "(this is the first chapter)"
    for i, ch in enumerate(chapters):
        p = plan.get(ch["n"], {})
        log(f"  [{i + 1}/{len(chapters)}] chapter {ch['n']}: {ch['title']}")
        prompt = (tpl
                  .replace("{company}", f"{co['name']} ({co['symbol']} / {co['isin']}) — "
                                        f"an Indian listed company; all figures in Indian "
                                        f"rupees (Rs crore).")
                  .replace("{storyline}", story_txt)
                  .replace("{chapter_n}", str(ch["n"]))
                  .replace("{chapter_title}", ch["title"])
                  .replace("{chapter_brief}", ch["brief"])
                  .replace("{chapter_point}", str(p.get("point") or "(use your judgement)"))
                  .replace("{chapter_hand_off}", str(p.get("hand_off") or "(none)"))
                  .replace("{previous}", prev)
                  .replace("{fact_table}", _fact_table(ch["facts"]) if ch["facts"]
                           else "(no computed figures for this chapter — write no numbers)")
                  .replace("{tables}", _tables_blob(ch["tables"]))
                  .replace("{notes}", ch["notes"] or "(none)")
                  .replace("{sources}", _sources_blob(sources)))
        subpack = {"facts": [f for f in cpack["facts"] if f.get("section") == ch["id"]],
                   "tables": [t for t in cpack["tables"] if t.get("section") == ch["id"]]}
        res = _write_and_gate(pool, prompt, subpack, sources, ch["id"], ch["title"],
                              ch["gaps"], log, label="chapter")
        res.update({"id": ch["id"], "chapter": ch["n"], "sections": ch["sections"]})
        out.append(res)
        paras = [x for x in str(res.get("body") or "").split("\n") if x.strip()]
        if paras:
            prev = paras[-1]
        if i < len(chapters) - 1:
            time.sleep(INTER_CALL_SLEEP)
    flagged = sum(1 for s in out if s.get("gate_failed"))
    from model_registry import record_usage          # busy demotion sees these models
    record_usage(pool, "narrative_writer", "narrative", log=log)
    return {"company": co, "layout": "chapters", "storyline": story, "sections": out,
            "generator_model": models[0],
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
