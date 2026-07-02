r"""
research_snippet.py — the ONE shared extractor that turns a research_index
`summary_md` into a short, SUBSTANTIVE company-relevant snippet. Used by
daily_brief.py (mail), build_gallery.py (local galleries) and app.py (banners)
so all three surfaces tell the same story. Pure stdlib (re only).

Why it exists (user feedback 2026-07-02): the old per-line matcher returned the
company's SECTION HEADER ("--- LICHSGFIN (NA) ---", "### Navin Fluorine ...")
or a companies-list metadata row — a name with zero substance. The docs are
structured (research_doc_prompt): per-company sections with Financials /
Key Statements / Positives / Risks / Analyst View. So: find the company's
section and harvest the BODY; if the company has no section and no substantive
mention, return "" and let the caller DROP the item (no item beats noise).
"""
from __future__ import annotations
import re

# a company section header: "### Name (SYM)"  or  "--- Name (SYM) ---" / "[SYM]"
_HDR = re.compile(r"^(#{1,6}\s+|-{2,}\s*)")
# subsection labels inside a company section (not company headers)
_SUBSEC = re.compile(r"(?i)^\*\*(financials|key statements|guidance|positives|"
                     r"risks|red flags|analyst view|valuation|outlook)")
# metadata / boilerplate rows that must never be shown
_META = re.compile(r"(?i)^\|?\s*\**\s*(companies|source/author|document (type|date|header)|"
                   r"field\s*\||output section|sectors?|themes?|isins?)\b")
_TBL_SEP = re.compile(r"^\|?[\s:|-]+\|?$")          # | --- | :--- | separators
# table LABEL rows ("Target Price · Rating · Key Thesis", "Metric · FY25 · FY26")
_LABEL_ROW = re.compile(r"(?i)^(target price|metric|particulars|field|rating)\s*·")
_RATING = re.compile(r"(?i)\b(buy|sell|hold|reduce|accumulate|overweight|underweight|"
                     r"neutral|target price|tp\b|upgrade|downgrade)\b")
_NUMBERY = re.compile(r"\d")
_NA_ONLY = re.compile(r"(?i)^[\s*|•·-]*(na|n/a|nil|none|nan)?[\s*|•·.-]*$")


def _clean(line: str) -> str:
    """Markdown row/bullet -> readable fragment ('a | b |' -> 'a · b')."""
    s = line.strip().strip("|").strip()
    s = re.sub(r"^[*•·-]+\s*", "", s)
    s = re.sub(r"\*\*(.+?)\*\*", r"\1", s)
    s = re.sub(r"\s*\|\s*", " · ", s)
    return " ".join(s.split())


def _substantive(s: str) -> bool:
    """Worth showing: not empty/NA/separator/label-row, carries data or a view."""
    if not s or _NA_ONLY.match(s) or _LABEL_ROW.match(s) or len(s) < 12:
        return False
    return bool(_NUMBERY.search(s) or _RATING.search(s) or len(s) > 60)


def _key_re(name: str, symbol: str):
    """Word-bounded matcher for the company's symbol / leading name tokens.
    Boundaries stop 3-char keys like 'LIC' matching inside 'public'."""
    ks = []
    if symbol and len(str(symbol)) > 2:
        ks.append(re.escape(str(symbol)))
    tok = str(name or "").split()
    if tok:
        if len(tok) > 1 and len(tok[0]) < 5:          # "Sai Life", "CG Power"
            ks.append(re.escape(f"{tok[0]} {tok[1]}"))
        elif len(tok[0]) >= 4:
            ks.append(re.escape(tok[0]))
    if not ks:
        return None
    return re.compile(r"(?i)(?<![A-Za-z0-9])(" + "|".join(ks) + r")(?![A-Za-z0-9])")


def research_snippet(md, name, symbol="", maxlen=340, max_frags=3):
    """Substantive company snippet from a research summary_md, or "" (drop it)."""
    md = str(md or "")
    kre = _key_re(name, symbol)
    if not md or kre is None:
        return ""
    lines = md.splitlines()

    # ── 1. the company's own section: header line + harvest its body ─────────
    hdr_i = None
    for i, ln in enumerate(lines):
        s = ln.strip()
        if _HDR.match(s) and kre.search(s):
            hdr_i = i
            break
    frags = []
    if hdr_i is not None:
        for ln in lines[hdr_i + 1:]:
            s = ln.strip()
            if _HDR.match(s) and not _SUBSEC.match(s):     # next company / section
                break
            if _META.match(s) or _TBL_SEP.match(s) or _SUBSEC.match(s):
                continue
            c = _clean(s)
            if not _substantive(c):
                continue
            if _RATING.search(c) and frags:                # rating/TP rows lead
                frags.insert(0, c)
            else:
                frags.append(c)
            if len(frags) >= max_frags:
                break

    # ── 2. no section: substantive mention lines (skip headers/metadata) ─────
    if not frags:
        for ln in lines:
            s = ln.strip()
            if not kre.search(s):
                continue
            if _HDR.match(s) or _META.match(s) or _TBL_SEP.match(s):
                continue
            if s.count(",") >= 5 and not _RATING.search(s):   # symbol-list dump
                continue
            c = _clean(s)
            if _substantive(c):
                frags.append(c)
            if len(frags) >= max_frags:
                break

    # ── 3. single-company doc (key visible in the head): prose fallback ──────
    if not frags:
        m = re.search(r"(?i)^\|?\s*\**companies\**\s*\|(.+)$", md, re.M)
        n_cos = (m.group(1).count(",") + 1) if m else 99      # unknown -> assume multi
        head = " ".join(lines[:15])
        if n_cos <= 2 and kre.search(head):
            for ln in lines:
                s = ln.strip()
                if (len(s) > 55 and not s.startswith(("|", "#", "=", "-"))
                        and not _META.match(s)):
                    c = _clean(s)
                    if _substantive(c):
                        frags.append(c)
                        break

    if not frags:
        return ""                                          # nothing worth mailing
    out = " | ".join(frags[:max_frags])
    return (out[:maxlen] + "…") if len(out) > maxlen else out
