r"""
render_company_report.py — the ONE merged company report: story + deep dive in a single file.

User 2026-10-17: "deep dive + narrative ... merged and single file"; "actual source link goes to
appendix, a simple subscript number should suffice"; "I don't want to decide, at the time of
reading, that I need to skip this section because it has no information".

Layout (plan: .claude/plans/narrative-11-chapter-story.md):
  front page   the story on one page (deep dive one-pager + findings) and one audit line
  chapters     1 business  2 evolution  3 what is changing  4 financial picture  5 funding growth
               6 fraud scan  7 promoter quality  8 financial shenanigans  9 research, global
               learnings, management interaction  10 industry  11 valuation  + closing: risks
               and what could prove this wrong, questions for management
  appendix     A references (numbered) · B supporting tables · C data gaps · D audit ·
               E documents read

Rules, all mechanical (this module makes no model calls):
  * A chapter is printed only when it has prose or a key exhibit with rows; printed chapters
    are numbered 1..n without gaps.
  * The reading flow carries prose and the key exhibits only. Sources, fact tables, data gaps,
    audit verdicts and the remaining tables go to the appendix.
  * Citations are small <sup>n</sup> numbers; reference n in Appendix A names the document,
    its date and its link.
  * Model text is cleaned: code fences removed (one unclosed fence turned 16 of 22 pages of
    the Ather report into cut-off monospace, 2026-10-04), fact ids "(one.mcap_cr)" replaced
    by reference numbers, [WARNING]/[COMFORT]/[INFO] turned into words.

Usage (re-render saved layers, no model calls):
  python scripts/render_company_report.py --factpack fp.json --narrative nar.json \
         [--audit audit.json] [--forensic dd.md] [--links links.json] --out report.md [--pdf]
"""
from __future__ import annotations

import argparse
import json
import re
from collections import OrderedDict
from datetime import datetime
from pathlib import Path

from render_narrative_deck import _src_str, _fmt, DISCLAIMER

# (title, story sections, deep-dive keys). Deep-dive keys: "0.n" = Layer 0, "1.n" = Layer 1.
CHAPTERS: list[tuple[str, list[int], list[str]]] = [
    ("The business", [2, 9, 6, 10, 12], ["0.1", "0.2", "0.4", "0.5"]),
    ("How the business has evolved", [1, 5, 8, 3], []),
    ("What is changing", [11, 27], ["0.3", "0.6", "0.8"]),
    ("The financial picture — income statement, balance sheet, cash flow",
     [18, 19, 7], ["1.2", "1.3"]),
    ("Can it fund its growth?", [], ["1.8"]),
    ("Fraud scan", [], ["1.5", "1.6"]),
    ("Promoter quality", [4], []),
    ("Financial shenanigans check", [], ["1.4"]),
    ("Research, global learnings and management interaction", [14, 16, 17, 20], []),
    ("Industry scenario", [15, 21, 25], ["0.9"]),
    ("Valuation check", [22], []),
    ("Risks — and what could prove this reading wrong", [23, 24], ["0.7"]),
]
# What each chapter covers (the user's list, 2026-10-05). Given to the chapter writer.
CHAPTER_BRIEFS: list[str] = [
    "What the company does and how it makes money: products, customers, markets, plants, "
    "and what drives volume, price and mix.",
    "How the business has evolved over the years: its eras, how scale, mix and margins "
    "changed, and why.",
    "What is changing now: capacity and capex, new products or markets, mix shift, "
    "management's stated plans, and recent filings or news.",
    "What the income statement, balance sheet and cash flow look like over the years: "
    "growth, margins, returns, debt, working capital and cash from operations.",
    "Whether growth can be funded internally: tie the three statements together - does cash "
    "from operations cover capex, working capital and dividends, or does it need debt or "
    "new equity?",
    "Fraud scan: red flags from the fraud tracker, exchange surveillance, auditors and "
    "governance. If nothing is flagged, say so plainly.",
    "Promoter quality: holding and pledges, remuneration, related-party dealings, capital "
    "allocation, and whether management did what it said.",
    "Financial shenanigans check: cash from operations versus profit, receivable and "
    "inventory days, other income, capitalised costs, one-offs and accounting changes.",
    "What research, global learnings and management interactions add: broker views, lessons "
    "from global peers, and what management said on calls versus what followed.",
    "The industry scenario: demand, pricing, policy, competition and where this company "
    "stands among its peers.",
    "Valuation check: today's multiples against the company's own history and its peers, "
    "and what growth the current price assumes. No target price.",
    "Risks, and what could prove this reading wrong.",
]
# Tables printed INSIDE the flow (they are the chapter's argument); every other table goes
# to Appendix B so a chapter never turns into a stack of tables.
EXHIBITS = {"tbl.financials", "tbl.sensitivity_pe", "tbl.said_vs_delivered"}
# Deep-dive parts whose TABLES are shown as exhibits in the chaptered report: CFO vs PAT
# (1.4), the weighted forensic score (1.6), earnings quality (1.8). Labelled as model-
# extracted, because the deep dive's numbers are not mechanically checked.
DD_TABLES = {"1.4", "1.6", "1.8"}
# Deep-dive tokens that are signifiers, not sources.
_SIGNS = {"WARNING": "Risk:", "COMFORT": "Positive:", "INFO": "Note:"}
_NOT_SOURCES = set(_SIGNS) | {"DATA_MISSING", "DATA_NOT_DISCLOSED",
                              "NO_MATERIAL_STRUCTURAL_SHIFT_IDENTIFIED", "SYSTEM"}
_FENCE = re.compile(r"^\s*(```|~~~)")
_FACT_REF = re.compile(r"\s*\((?:[a-z][a-z_]*\.[A-Za-z0-9_.]+)(?:,\s*[a-z][a-z_]*\.[A-Za-z0-9_.]+)*\)")
_DOC_REF = re.compile(r"\s*\(((?:annual_report|concall|presentation|results|rating|"
                      r"announcement|drhp|rhp)_\d{4}-\d{2}-\d{2}[A-Za-z0-9_]*)\)")
_TAG = re.compile(r"\[([A-Z][^\[\]\n]{1,80})\]")


# ------------------------------------------------------------------ references --
class Refs:
    """Numbered references in order of first use. key -> (n, text)."""

    def __init__(self):
        self._d: "OrderedDict[str, tuple[int, str]]" = OrderedDict()

    def n(self, key: str, text: str) -> int:
        if key not in self._d:
            self._d[key] = (len(self._d) + 1, text)
        return self._d[key][0]

    def items(self):
        return [(n, t) for n, t in self._d.values()]


def _src_ref(src: dict) -> tuple[str, str]:
    """ONE reference per data FILE, not per line item / period: citing every fact's own
    source string gave Morepen 173 references. Calculations made by this report share one."""
    src = src or {}
    k, table = src.get("kind"), str(src.get("table") or "")
    if k == "computed":
        return ("src:computed", "Calculated in this report from the reported figures "
                                "(the method is stated under each exhibit)")
    if k == "statements":
        when = str(src.get("fetched_at") or "")[:10]
        return (f"src:{table}", f"Screener financial statements — {table}"
                                + (f" (retrieved {when})" if when else ""))
    if k == "parquet" and table:
        return (f"src:{table}", f"signals-india data table — {table}")
    s = _src_str(src)
    return (f"src:{s}", s)


def _sup(nums) -> str:
    nums = sorted(set(nums))
    return f"<sup>{','.join(str(x) for x in nums)}</sup>" if nums else ""


def _doc_text(doc_id: str, links: dict) -> str:
    """'Concall transcript · 2026-05-01 · title — url' for a document id."""
    m = re.match(r"([a-z_]+?)_(\d{4}-\d{2}-\d{2})", doc_id or "")
    kind, day = (m.group(1), m.group(2)) if m else (doc_id, "")
    label = {"annual_report": "Annual report", "concall": "Earnings call transcript",
             "presentation": "Investor presentation", "results": "Results filing",
             "rating": "Credit rating rationale", "announcement": "Exchange filing",
             "drhp": "Draft red herring prospectus", "rhp": "Red herring prospectus"
             }.get(kind, kind.replace("_", " ").capitalize())
    info = links.get(doc_id) or {}
    bits = [label, day or info.get("date", "")]
    if info.get("title"):
        bits.append(str(info["title"])[:110])
    out = " · ".join(b for b in bits if b)
    return out + (f" — {info['url']}" if info.get("url") else "")


# ------------------------------------------------------------------ cleaning --
def _clean(text: str) -> str:
    """Model text made safe for the page: no code fences, no stray markdown headings."""
    lines = [ln for ln in str(text or "").splitlines() if not _FENCE.match(ln)]
    return "\n".join(lines).strip()


def _no_codes(text: str) -> str:
    """Model text with fact ids "(one.mcap_cr)" and document ids removed (front page)."""
    return _DOC_REF.sub("", _FACT_REF.sub("", _clean(text))).strip()


def _paras(text: str) -> list[str]:
    return [p.strip() for p in re.split(r"\n\s*\n|\n", _clean(text)) if p.strip()]


def _words(s: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]{4,}", s.lower())}


def _attach(paras: list[str], claim_text: str) -> int:
    """Index of the paragraph that states this claim (word overlap), else the last one."""
    cw = _words(claim_text)
    best, score = len(paras) - 1, 0.0
    for i, p in enumerate(paras):
        pw = _words(p)
        if cw and pw:
            s = len(cw & pw) / len(cw)
            if s > score:
                best, score = i, s
    return best


# ------------------------------------------------------------------ deep dive --
def parse_forensic(md: str) -> dict[str, str]:
    """Split the deep dive markdown into its numbered parts. Keys: '0.1'..'0.9' (Layer 0),
    '1.1'..'1.9' (Layer 1), 'notes' (Layer 2), 'onepager' (Phase 4), 'docs' (Phase 5 +
    read confirmation). Headings vary slightly between runs, so they are matched by their
    leading number inside the layer, not by exact text."""
    out: dict[str, list[str]] = {}
    layer, key = None, None
    for ln in _clean(md).splitlines():
        h2 = re.match(r"^##\s+(.*)", ln)
        h3 = re.match(r"^###\s+(.*)", ln)
        if h2 and not h3:
            t = h2.group(1).lower()
            layer = ("0" if "layer 0" in t else "1" if "layer 1" in t else
                     "notes" if "layer 2" in t else "onepager" if "one-pager" in t or
                     "phase 4" in t else "docs" if ("phase 5" in t or "document" in t)
                     else layer)
            key = layer if layer in ("notes", "onepager", "docs") else None
            continue
        if h3 and layer in ("0", "1"):
            m = re.match(r"(\d+)(?:\.(\d+))?", h3.group(1).strip())
            if m:
                key = f"0.{m.group(2) or m.group(1)}" if layer == "0" else f"1.{m.group(1)}"
                continue
        if key:
            out.setdefault(key, []).append(ln)
    return {k: "\n".join(v).strip() for k, v in out.items() if "\n".join(v).strip()}


def _forensic_text(text: str, refs: Refs) -> str:
    """Deep-dive text for the flow: source tags -> reference numbers, signifier tokens ->
    words, DATA_MISSING-only lines dropped, sub-headings turned into bold lead-ins."""
    out = []
    for ln in text.splitlines():
        s = ln.strip()
        if not s or re.fullmatch(r"[*\-\s]*(DATA_MISSING|DATA_NOT_DISCLOSED)[.\s]*", s):
            continue
        if s in ("---", "***") or s.startswith("====="):
            continue
        h = re.match(r"^#{3,6}\s+(.*)", s)
        if h:
            out.append(f"**{h.group(1).strip()}**")
            continue
        nums: list[int] = []

        def repl(m):
            body = m.group(1).strip()
            if body in _SIGNS:
                return f"**{_SIGNS[body]}**"
            if body in _NOT_SOURCES:
                return "not disclosed" if body.startswith("DATA_") else ""
            for part in re.split(r",\s*(?=[A-Z])", body):
                nums.append(refs.n(f"dd:{part.strip()}", f"Deep dive source: {part.strip()}"))
            return ""
        s2 = _TAG.sub(repl, ln.rstrip())
        # bare tokens (table cells, mid-sentence) read as words, never as codes
        s2 = re.sub(r"\bDATA_(?:MISSING|NOT_DISCLOSED)\b", "not disclosed", s2)
        s2 = s2.replace("NO_MATERIAL_STRUCTURAL_SHIFT_IDENTIFIED",
                        "no material structural shift identified")
        s2 = re.sub(r"\s+([.,;:])", r"\1", s2).rstrip()
        out.append(s2 + _sup(nums) if nums else s2)
    # Markdown needs a blank line where a list or table starts after text (else the
    # bullets run into one paragraph) and after a bold lead-in.
    spaced: list[str] = []
    for ln in out:
        is_list = bool(re.match(r"\s*([*\-+]|\d+\.)\s", ln))
        is_row = ln.lstrip().startswith("|")
        prev = spaced[-1] if spaced else ""
        prev_list = bool(re.match(r"\s*([*\-+]|\d+\.)\s", prev))
        prev_row = prev.lstrip().startswith("|")
        if prev and ((is_list and not prev_list) or (is_row and not prev_row)
                     or (not is_list and not is_row and (prev_list or prev_row))
                     or re.fullmatch(r"\*\*[^*]+\*\*", prev.strip())):
            spaced.append("")
        spaced.append(ln)
    txt = "\n".join(spaced).strip()
    return re.sub(r"\n{3,}", "\n\n", txt)


# ------------------------------------------------------------------ chapters --
def _plain_notes(text: str) -> str:
    """Deep-dive text for a WRITER: tags and signifiers removed, numbers kept as context."""
    t = _clean(text)
    t = _TAG.sub(lambda m: (_SIGNS.get(m.group(1).strip(), "") if m.group(1).strip()
                            in _NOT_SOURCES else ""), t)
    t = re.sub(r"\bDATA_(?:MISSING|NOT_DISCLOSED)\b", "not disclosed", t)
    return re.sub(r"[ \t]+\n", "\n", t).strip()


def chapter_inputs(pack: dict, forensic_md: str = "", notes_chars: int = 6000) -> list[dict]:
    """Per chapter: its facts, tables, gaps and deep-dive notes. A chapter with no facts,
    no tables and no notes is left out HERE, before any model call (empty-chapter rule)."""
    dd = parse_forensic(forensic_md or "")
    out = []
    for n, ((title, secs, ddkeys), brief) in enumerate(zip(CHAPTERS, CHAPTER_BRIEFS), 1):
        facts = [f for f in pack.get("facts", []) if f.get("section") in secs]
        tables = [t for t in pack.get("tables", []) if t.get("section") in secs]
        notes = "\n\n".join(_plain_notes(dd[k]) for k in ddkeys if dd.get(k))[:notes_chars]
        gaps = ([g["reason"] for g in pack.get("coverage_gaps", []) if g["section"] in secs]
                + [u["reason"] for u in pack.get("uncovered_sections", [])
                   if u["section"] in secs])
        if not (facts or tables or notes.strip()):
            continue
        out.append({"n": n, "id": f"ch{n}", "title": title, "brief": brief,
                    "sections": secs, "facts": facts, "tables": tables,
                    "gaps": gaps, "notes": notes})
    return out


def chapter_pack(pack: dict) -> dict:
    """The fact pack with every fact / table re-keyed from its story section to its chapter
    id ('ch4'), so a section-scoped check (grounding gate, auditor) sees the whole chapter."""
    sec_to_ch = {s: f"ch{n}" for n, (_, secs, _) in enumerate(CHAPTERS, 1) for s in secs}
    p = dict(pack)
    p["facts"] = [dict(f, section=sec_to_ch.get(f.get("section"), f.get("section")))
                  for f in pack.get("facts", [])]
    p["tables"] = [dict(t, section=sec_to_ch.get(t.get("section"), t.get("section")))
                   for t in pack.get("tables", [])]
    return p


# ------------------------------------------------------------------ story --
def section_map(pack: dict, narrative: dict | None) -> dict[int, dict]:
    """section number -> narrative section. The model sometimes REWRITES the id it was given
    ("scale-history" for 5): those sections were silently left unprinted. Sections are
    generated in ascending order of the sections that have facts or tables, so a list whose
    ids are not all numeric is mapped back by that order."""
    secs = list((narrative or {}).get("sections") or [])
    if not secs or (narrative or {}).get("layout") == "chapters":
        return {}                     # chapters are keyed by chapter, never by position
    if all(str(s.get("id", "")).isdigit() for s in secs):
        return {int(s["id"]): s for s in secs}
    have = sorted({f["section"] for f in pack.get("facts", [])}
                  | {t["section"] for t in pack.get("tables", [])})
    if len(have) == len(secs):
        return dict(zip(have, secs))
    return {int(s["id"]): s for s in secs if str(s.get("id", "")).isdigit()}


def _story_block(num, sec: dict, pack: dict, links: dict, refs: Refs) -> list[str]:
    """Takeaway + body of one story section (num = its number) or one chapter (num = the
    list of story sections it covers), with its citations as reference numbers."""
    nums = set(num) if isinstance(num, (list, tuple, set)) else {num}
    paras = _paras(sec.get("body", ""))
    take = _clean(sec.get("takeaway", ""))
    if not paras and not take:
        return []
    facts = {f["id"]: f for f in pack.get("facts", [])}
    marks: dict[int, list[int]] = {}
    # 1. fact ids written into the prose -> the fact's source
    for i, p in enumerate(paras):
        found = re.findall(r"[a-z][a-z_]*\.[A-Za-z0-9_.]+", " ".join(_FACT_REF.findall(p)))
        for fid in found:
            if fid in facts:
                marks.setdefault(i, []).append(refs.n(*_src_ref(facts[fid].get("source"))))
        for did in _DOC_REF.findall(p):
            marks.setdefault(i, []).append(refs.n(f"doc:{did}", _doc_text(did, links)))
        paras[i] = _DOC_REF.sub("", _FACT_REF.sub("", p)).strip()
    # 2. each qualitative claim -> its document, on the paragraph that states it
    for c in sec.get("claims") or []:
        did = ((c.get("source_ref") or {}).get("doc_id") or "").strip()
        if did and paras:
            marks.setdefault(_attach(paras, c.get("text", "")), []).append(
                refs.n(f"doc:{did}", _doc_text(did, links)))
    # 3. the section's data sources -> once, on its first paragraph
    srcs = {_src_ref(f.get("source")) for f in pack.get("facts", []) if f.get("section") in nums}
    srcs |= {_src_ref(t.get("source")) for t in pack.get("tables", []) if t.get("section") in nums}
    if paras:
        marks.setdefault(0, []).extend(refs.n(k, t) for k, t in sorted(srcs) if t)
    out = [f"**{take}**"] if take else []
    out += [p + _sup(marks.get(i, [])) for i, p in enumerate(paras)]
    return out


# Column names a reader understands (the fact pack keeps short machine names).
COL_LABELS = {"revenue": "Revenue", "ebitda_incl_oi": "EBITDA incl. other income",
              "ebitda_op_only": "Operating EBITDA", "pbt": "PBT", "pat": "PAT",
              "dep": "Depreciation", "interest": "Interest", "eps": "EPS (Rs)",
              "cfo": "Cash from operations", "margin": "EBITDA margin %"}


def _table_md(t: dict) -> list[str]:
    cols = t["columns"]
    L = [f"**{t['title']}**", "",
         "| " + " | ".join(COL_LABELS.get(str(c), str(c)) for c in cols) + " |",
         "| " + " | ".join("---" for _ in cols) + " |"]
    for row in t["rows"]:
        L.append("| " + " | ".join(_fmt(row.get(c)).replace("|", "/") for c in cols) + " |")
    if t.get("note"):
        L += ["", f"*{t['note']}*"]
    return L


# ------------------------------------------------------------------ render --
def render(pack: dict, narrative: dict | None = None, audit: dict | None = None,
           forensic_md: str = "", links: dict | None = None) -> str:
    co, links = pack["company"], links or {}
    refs = Refs()
    smap = section_map(pack, narrative)
    dd = parse_forensic(forensic_md or (narrative or {}).get("forensic_report") or "")
    tables = {t["id"]: t for t in pack.get("tables", [])}
    L: list[str] = [f"# {co['name']} — company report", "",
                    f"**{co['symbol']} · {co['isin']}** · data current to "
                    f"{str(pack.get('as_of_utc', ''))[:10]}", ""]

    # ---- the story on one page -------------------------------------------
    front = []
    story = (narrative or {}).get("storyline") or {}
    if story.get("one_page"):
        # Written by the storyline pass (numbers gated against the whole fact pack).
        front.append(_no_codes(story["one_page"]))
        for key, label in (("business_dynamics", "Business dynamics"),
                           ("financial_dynamics", "Financial dynamics"),
                           ("watch", "What would change this reading")):
            items = [_no_codes(x) for x in story.get(key) or [] if _no_codes(x)]
            if items:
                front.append(f"**{label}**\n\n" + "\n".join(f"- {x}" for x in items))
        m = re.search(r"###\s*Forensic Risk View\s*\n(.*?)(?=\n###|\Z)",
                      dd.get("onepager", ""), re.S | re.I)
        if m and m.group(1).strip():
            front.append("**Forensic risk view (deep dive)**\n\n"
                         + _forensic_text(m.group(1), refs))
    elif dd.get("onepager"):
        front.append(_forensic_text(dd["onepager"], refs))
    elif dd.get("1.7"):
        front.append(_forensic_text(dd["1.7"], refs))
    if 26 in smap:
        front += _story_block(26, smap[26], pack, links, refs)
    if not front:
        front = [f"- {_clean(s.get('takeaway'))}" for _, s in sorted(smap.items())
                 if _clean(s.get("takeaway"))][:8]
    if front:
        L += ["## The story on one page", ""] + sum(([b, ""] for b in front), [])
    for n in pack.get("data_notes") or []:
        L += [f"*Data note: {n}*", ""]
    if audit:
        s = audit.get("summary") or {}
        L += [(f"*Independent audit: {s.get('verified', 0)} of {s.get('total', 0)} claims "
               f"verified — Appendix D.*" if audit.get("ran", True) else
               "*Independent audit did not run — no claim has been independently "
               "checked (Appendix D).*"), ""]

    # ---- chapters -----------------------------------------------------------
    used, num = set(), 0
    counters: list[str] = []
    if (narrative or {}).get("layout") == "chapters":
        # Written chapter by chapter (narrative_generate.generate_report): the deep dive
        # was the writer's analyst notes, so its text is NOT appended again — only its
        # forensic tables (numbers model-extracted, labelled as such).
        bych = {int(s.get("chapter", 0)): s for s in narrative.get("sections") or []}
        for n, (title, secs, ddkeys) in enumerate(CHAPTERS, 1):
            ch = bych.get(n) or {}
            body = _story_block(secs, ch, pack, links, refs) if ch else []
            body = [b for b in body if not re.fullmatch(r"\*\*\s*\*\*", b)]
            for s in secs:
                for tid, t in tables.items():
                    if t.get("section") == s and tid in EXHIBITS and t.get("rows"):
                        body.append("\n".join(_table_md(t)))
            for k in ddkeys:
                if k in DD_TABLES and dd.get(k):
                    rows = [ln for ln in dd[k].splitlines() if ln.lstrip().startswith("|")]
                    if len(rows) > 2:
                        body.append("*From the forensic deep dive (model-extracted figures, "
                                    "not mechanically checked):*\n\n"
                                    + _forensic_text("\n".join(rows), refs))
            v = _clean(ch.get("counter_argument"))
            if v:
                counters.append(v)
            if title.startswith("Risks"):
                body += [f"- {c}" for c in dict.fromkeys(counters)][:12]
            prose = [ln for b in body for ln in b.splitlines()
                     if ln.strip() and not ln.lstrip().startswith(("|", "*From the"))]
            if not prose:
                continue                           # nothing to say -> not printed at all
            num += 1
            L += [f"## {num}. {title}", ""] + sum(([b, ""] for b in body if b.strip()), [])
        smap = {}                                  # appendix: chapters carry not_disclosed
        for s in narrative.get("sections") or []:
            smap[f"ch{s.get('chapter')}"] = s
    for title, secs, ddkeys in ([] if (narrative or {}).get("layout") == "chapters"
                                else CHAPTERS):
        body: list[str] = []
        for s in secs:
            if s in smap:
                blk = _story_block(s, smap[s], pack, links, refs)
                body += blk
                used.add(s)
                for k in ("counter_argument", "falsifier"):
                    v = _clean(smap[s].get(k))
                    if v:
                        counters.append(v)
            for tid, t in tables.items():
                if t.get("section") == s and tid in EXHIBITS and t.get("rows"):
                    body.append("\n".join(_table_md(t)))     # ONE block, or it breaks
        for k in ddkeys:
            if dd.get(k):
                txt = _forensic_text(dd[k], refs)
                if txt:
                    body += [txt]
        if title.startswith("Risks"):
            body += [f"- {c}" for c in dict.fromkeys(counters)][:12]
        # Printed only when some LINE is prose (not a table row, not a bold label alone).
        # Checked per line: a deep-dive part that opens with a table is still prose.
        prose = [ln for b in body for ln in b.splitlines()
                 if ln.strip() and not ln.lstrip().startswith("|")
                 and not re.fullmatch(r"\*\*[^*]+\*\*", ln.strip())]
        if not prose:
            continue                               # nothing to say -> not printed at all
        num += 1
        L += [f"## {num}. {title}", ""] + sum(([b, ""] for b in body if b.strip()), [])
    if dd.get("notes"):
        q = _forensic_text(dd["notes"], refs)
        if q:
            num += 1
            L += [f"## {num}. Questions for management and analytical notes", "", q, ""]

    # ---- appendix ---------------------------------------------------------------
    L += ["---", "", "# Appendix", ""]
    L += ["## A. References", ""]
    L += [f"{n}. {t}" for n, t in refs.items()] or ["No citations."]
    L += [""]
    rest = [t for tid, t in tables.items() if tid not in EXHIBITS and t.get("rows")
            and tid != "tbl.documents_read"]
    if rest:
        L += ["## B. Supporting tables", ""]
        for t in rest:
            L += _table_md(t) + [""]
    gaps = [g["reason"] for g in pack.get("coverage_gaps", [])]
    gaps += [u["reason"] for u in pack.get("uncovered_sections", [])]
    for s in smap.values():
        gaps += [_clean(x) for x in (s.get("not_disclosed") or []) if _clean(x)]
    if gaps:
        L += ["## C. What the record does not disclose", ""]
        L += [f"- {g}" for g in dict.fromkeys(gaps)] + [""]
    if audit:
        s = audit.get("summary") or {}
        L += ["## D. Independent audit", "",
              f"Adjudicator `{audit.get('model', 'unknown')}`: {s.get('verified', 0)} of "
              f"{s.get('total', 0)} claims verified · {s.get('unsupported', 0)} unsupported · "
              f"{s.get('contradicted', 0)} contradicted."
              + (f" {s.get('sections_failed')} section(s) could not be audited."
                 if s.get("sections_failed") else ""), ""]
        for c in audit.get("claims") or []:
            v = str(c.get("verdict", "")).upper()
            if v in ("VERIFIED", "PARTIAL"):
                continue
            if "AUDIT_FAILED" in v or "unparseable" in str(c.get("claim", "")):
                L.append(f"- Section {c.get('section')}: could not be audited.")
                continue
            reason = " ".join(_clean(c.get("reason", "")).split())[:300]
            claim = " ".join(_clean(c.get("claim", "")).split())[:240]
            L.append(f"- **{v.replace('_', ' ').title()}** — “{claim}” — {reason}")
        L.append("")
    docs = tables.get("tbl.documents_read")
    if docs or dd.get("docs"):
        L += ["## E. Documents read", ""]
        if docs:
            L += _table_md(docs) + [""]
        if dd.get("docs"):
            L += [_forensic_text(dd["docs"], refs), ""]
    L += ["---", "", f"*{DISCLAIMER}*", "",
          f"*Generated {datetime.now():%Y-%m-%d %H:%M} · signals-india*"]
    return "\n".join(L).rstrip() + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--factpack", required=True)
    ap.add_argument("--narrative")
    ap.add_argument("--audit")
    ap.add_argument("--forensic", help="deep dive markdown (else narrative.forensic_report)")
    ap.add_argument("--links", help="JSON {doc_id: {title, date, url}}")
    ap.add_argument("--out", required=True)
    ap.add_argument("--pdf", action="store_true", help="also write <out>.pdf")
    a = ap.parse_args()
    rd = lambda p: json.loads(Path(p).read_text(encoding="utf-8")) if p else None
    pack = rd(a.factpack)
    md = render(pack, rd(a.narrative), rd(a.audit),
                Path(a.forensic).read_text(encoding="utf-8") if a.forensic else "",
                rd(a.links) or {})
    Path(a.out).write_text(md, encoding="utf-8")
    print(f"wrote {a.out} ({len(md):,} chars)")
    if a.pdf:
        import format_deepdive_pdf as FMT
        co = pack["company"]
        pdf = Path(a.out).with_suffix(".pdf")
        pdf.write_bytes(FMT.md_to_pdf(md, co["name"], co["symbol"], co["isin"],
                                      title="Company"))
        print(f"wrote {pdf}")


if __name__ == "__main__":
    main()
