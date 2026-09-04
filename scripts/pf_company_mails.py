"""
pf_company_mails.py — ONE MAIL PER COMPANY, per document type.

The existing digests send one mail covering every holding. The user asked for the
opposite: each company that reports gets its own mail, so the body can carry real detail
instead of a row in a table.

Three kinds, each with the same two-part shape — **the current document first, then what
changed since the last one**:

    results       the full quarterly teardown (quarter_teardown.render)
    presentation  this quarter's deck  +  KPIs dropped / targets moved / guidance vs delivered
    rating        current agency view  +  upgrade-downgrade, new & vanished concerns,
                                          downgrade triggers that moved

THE TRIGGER (user, 2026-08-15): a mail goes out when the holding is in PF **and** its
results-calendar date has arrived **and** it has not already been mailed for that document
and period. Deliberately NOT a rolling time window — a window means a missed CI run drops
that company for good, whereas "not yet mailed" is self-healing and the next run catches
up. See pf_coverage.mail_due().

WHAT IS DETERMINISTIC AND WHAT IS NOT. Every "what changed" line here is an exact
comparison of structured rows — a KPI present last quarter and absent now, a concern in
the new rationale that was not in the old one. No LLM is involved in deciding what
changed. That is deliberate: round-tripping structured data through prose and back is
where hallucination enters, and the framework's own rule is deterministic-where-possible.

READ-ONLY w.r.t. Phase 2. This never takes _extract.lock, never writes company_page.md,
never queues or extracts anything. It reads the parquets Phase 2 already produces and
writes exactly one thing of its own: its mail ledger.

Usage:
    python scripts/pf_company_mails.py --dry-run          # preview, no mail, no ledger
    python scripts/pf_company_mails.py --types rating     # one kind only
    python scripts/pf_company_mails.py --limit 5
    python scripts/pf_company_mails.py --self-test
"""

from __future__ import annotations

import argparse
import io
import os
import re
import sys
from datetime import date, datetime, timedelta

import pandas as pd
from dotenv import load_dotenv

_SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

load_dotenv(os.path.join(os.path.dirname(_SCRIPTS_DIR), ".env"))

import quarterly_table as QT
import pf_coverage as COV
import deck_summary as DS

LEDGER_NAME = "pf_company_mails.parquet"
# `doc_id` added 2026-08-16 (ADDITIVE — legacy rows read blank and still suppress, so
# adding it cannot cause a re-send flood). Identity is the DOCUMENT, not the slot: keying
# on (isin, doc_type, season) alone meant the first deck or rating of a quarter mailed and
# every later one was silently swallowed — a rating DOWNGRADE arriving after a routine
# reaffirmation would never have reached the reader.
LEDGER_COLS = ["season", "isin", "symbol", "doc_type", "period", "doc_id",
               "mailed_at", "subject", "content_key"]
MAIL_KEY = "pf_company_mails"
MAX_HTML_BYTES = 90_000
# Concall and AR extractions store their PROSE ONLY in company_repo/<isin>/company_page.md
# - no parquet column holds it (extract_concall.py keeps len(markdown_text) as
# `response_chars` and nothing more). So the narrative is lifted back off the page, with
# the digest's parser rather than a second copy of it here.
NARRATIVE_LIMIT = 2600
# Types --scope applies to, each on its own calendar: a concall to the season QUARTER,
# an annual report to the FINANCIAL YEAR. Applies to these two and to nothing else, so
# the established presentation/rating/results mails keep their behaviour exactly.
SCOPED_TYPES = ("concall", "annual_report")


def new_pf_holdings(snaps: pd.DataFrame, days: int, today: date | None = None) -> set:
    """ISINs that ENTERED the portfolio within the last `days`.

    THE POINT. The month scope answers "what did the exchanges send this month", which is
    right for a holding already covered and wrong for one just bought: a company added
    today whose latest concall was filed in July would be suppressed forever, and its
    concall is the single thing a new holding most needs. So a new holding is onboarded
    with its LATEST concall and LATEST annual report whatever month they were filed.

    This mirrors what the presentation mail already does implicitly - it is season-scoped
    with no date window, so a new holding picks up the current quarter's deck on the very
    next run simply because the ledger has no row for it.

    THE HISTORY-START GUARD IS NOT OPTIONAL. first_seen for every holding present on the
    first snapshot is that snapshot's own date, which says nothing about when the holding
    was actually bought. Once `days` exceeds the history's age, every holding reads as new
    - measured 2026-09-02 against history starting 2026-07-23: at 30 days 7 holdings are
    new, at 45 days it jumps to 51, i.e. the entire portfolio. Requiring first_seen to be
    strictly AFTER the first snapshot keeps that at the true 11.
    """
    if snaps is None or getattr(snaps, "empty", True) or days <= 0:
        return set()
    if not {"isin", "snapshot_date"} <= set(snaps.columns):
        return set()
    d = snaps["snapshot_date"].astype(str).str.slice(0, 10)
    hist_start = d.min()
    first = snaps.assign(_d=d).groupby(snaps["isin"].astype(str))["_d"].min()
    cutoff = ((today or date.today()) - timedelta(days=days)).isoformat()
    return {i for i, f in first.items() if str(f) >= cutoff and str(f) > hist_start}


def concall_quarter(doc_date) -> str:
    """The season quarter a concall belongs to, from its FILING date.

    The same rule the presentation mail already lives by, and for the same reason
    pf_coverage.doc_quarter_map spells out: a document's own label cannot be trusted,
    but the date it was filed is not open to interpretation. A call filed in Aug 2026
    belongs to Q1 FY27 whatever it calls itself.
    """
    d = str(doc_date or "")[:10]
    if len(d) < 10:
        return ""
    try:
        return QT.norm_q(QT.season_quarter(pd.to_datetime(d)))
    except Exception:
        return ""


_FY_RE = re.compile(r"FY\s*(\d{4}|\d{2})", re.I)


def ar_fy_year(doc_date, period="") -> int | None:
    """The financial YEAR an annual report covers - the year that FY ENDED.

    Annual reports are not month news. A company must lay its report before an AGM
    within six months of the year end, so the FY2026 reports all arrive between roughly
    June and September 2026: the meaningful question is which FINANCIAL YEAR a report
    covers, not which month it happened to be filed in.

    Two date shapes reach us and they mean opposite things (measured 2026-09-02: 233 of
    235 PF annual_report rows are FY-end shaped, 2 are filing dates):
      2026-03-31  an FY-END stamp from the backfill ("Annual Report 2026 from bse").
                  It names its OWN year - this is FY2026.
      2026-09-02  a real filing date from the per-company sweep. The report being filed
                  is for the year that last ended, so this is also FY2026.
    `period` ("FY26") wins when present, being the extractor's own judgement.
    """
    m = _FY_RE.search(str(period or ""))
    if m:
        y = int(m.group(1))
        return y if y > 1900 else 2000 + y
    d = str(doc_date or "")[:10]
    if len(d) < 10:
        return None
    try:
        dt = date.fromisoformat(d)
    except ValueError:
        return None
    if (dt.month, dt.day) == (3, 31):        # an FY-END stamp names its own year
        return dt.year
    return dt.year if dt.month >= 4 else dt.year - 1


def current_ar_fy(today: date | None = None) -> int:
    """The financial year whose annual reports are being filed now."""
    t = today or date.today()
    return t.year if t.month >= 4 else t.year - 1

UP, DOWN, MUTED, AMBER = "#1a7a3a", "#c0392b", "#8a97a0", "#b8860b"
# Theme colours, defined here with the rest of the palette because both the deck
# summary sections and the theme chips need them.
BLUE, PURPLE = "#1f4d6b", "#6b4d8f"
_WRAP = "font-family:Arial,Helvetica,sans-serif;font-size:13px;color:#222"
_TBL = ("border-collapse:collapse;width:100%;font:12px Arial,Helvetica,sans-serif;"
        "color:#222;margin:4px 0 12px")


def _esc(s, n=400) -> str:
    from mailer import esc
    # Parquet NaN/None must not reach the page. Rendering value+unit straight out of the
    # frame produced literal "nanNone" in the first live run.
    if s is None:
        return ""
    t = str(s).strip()
    if t.lower() in ("nan", "none", "nat", "<na>"):
        return ""
    return esc(t, n)


# ESG commitments are not investment guidance. deck_teardown already bans these words
# from deck metrics ("_BANNED_METRIC_WORDS") on exactly this reasoning, but ppt_guidance
# and ppt_highlights have no such filter, so the first live render of APLAPOLLO returned
# a "guidance" table consisting of Scope 1 & 2 emissions, Net Zero, renewable share and
# female workforce — no financial target at all. Filtered at RENDER time only; nothing
# upstream is changed and the rows stay in the parquet.
_ESG_WORDS = ("emission", "net zero", "carbon", "esg", "csr", "diversity", "female",
              "renewable", "sustainab", "djsi", "gender", "water", "waste")


def _is_esg(*fields) -> bool:
    blob = " ".join(str(f or "").lower() for f in fields)
    return any(w in blob for w in _ESG_WORDS)


_NUM_RE = __import__("re").compile(r"^\s*([+-]?)\s*([\d,]+(?:\.\d+)?)\s*(%|x|bps|Cr|cr)?\s*$")


def colour_num(text, good_up: bool = True, bold: bool = True) -> str:
    """Render a number with a colour that carries its MEANING.

    Green/red is not decoration — it is the fastest way to read a table of mixed
    signals. `good_up=False` inverts it for metrics where lower is better (debt, days,
    a downgrade count), so a falling number reads green there.
    Non-numeric text passes through untouched rather than being force-coloured.
    """
    s = _esc(text, 40)
    if not s:
        return ""
    m = _NUM_RE.match(str(text).strip())
    if not m:
        return s
    sign, digits, unit = m.group(1), m.group(2), m.group(3) or ""
    try:
        val = float(digits.replace(",", ""))
    except ValueError:
        return s
    if sign == "-":
        val = -val
    if abs(val) < 1e-9:
        col = MUTED
    else:
        positive = val > 0
        col = UP if (positive == good_up) else DOWN
    shown = f"{sign}{digits}{unit}"
    weight = "font-weight:700;" if bold else ""
    return f"<span style='color:{col};{weight}'>{_esc(shown, 40)}</span>"


def _h(title: str, sub: str = "") -> str:
    return (f"<h3 style='margin:16px 0 4px;font-size:14px;color:#34495e;"
            f"border-bottom:2px solid #34495e;padding-bottom:3px'>{title}</h3>"
            + (f"<div style='color:{MUTED};font-size:12px;margin:0 0 8px'>{sub}</div>"
               if sub else ""))


def _rows_html(rows, cols) -> str:
    if not rows:
        return ""
    head = "".join(f"<th style='text-align:left;border-bottom:1px solid #ccc;"
                   f"color:{MUTED};font-weight:600'>{c}</th>" for c in cols)
    body = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return f"<table cellpadding='4' cellspacing='0' style='{_TBL}'><tr>{head}</tr>{body}</table>"


def _slice(df, isin, cols=None):
    if df is None or getattr(df, "empty", True) or "isin" not in df.columns:
        return pd.DataFrame(columns=cols or [])
    d = df[df["isin"].astype(str).str.strip() == str(isin).strip()]
    return d if not d.empty else pd.DataFrame(columns=cols or list(df.columns))


# ------------------------------------------------------------------ #
#  Presentation: current deck + what changed                          #
# ------------------------------------------------------------------ #

# The eighteen deck_summary sections, with the heading a reader sees and the colour that
# lets them scan for it. Risk is red and guidance purple for the same reason as the theme
# chips: those are the two a reader hunts for. The rest carry the neutral band so the
# colour still means something when it appears.
_SUMMARY_SECTIONS = (
    ("business_overview",     "What the company does",            BLUE),
    ("financials",            "Financials, as the deck states them", BLUE),
    ("balance_sheet",         "Balance sheet",                    BLUE),
    ("segment_performance",   "Segment performance",              BLUE),
    ("geography",             "Geography",                        BLUE),
    ("capacity_expansion",    "Capacity and expansion",           BLUE),
    ("capex",                 "Capex",                            BLUE),
    ("orderbook_pipeline",    "Order book and pipeline",          BLUE),
    ("customers",             "Customers",                        BLUE),
    ("products_rd",           "Products and R&amp;D",             BLUE),
    ("industry_market",       "Industry and market",              BLUE),
    ("strategy",              "Strategy",                         PURPLE),
    ("guidance_outlook",      "Guidance and outlook",             PURPLE),
    ("risks",                 "Risks the deck admits",            DOWN),
    ("management_commentary", "Management commentary",            "#34495e"),
    ("capital_allocation",    "Capital allocation",               BLUE),
    ("subsidiary_ma",         "Subsidiaries and M&amp;A",         BLUE),
    ("esg",                   "ESG",                              MUTED),
)


def standalone_summary(isin, season, tables) -> str:
    """The deck read as a company note in its own right.

    This is the section that matters for a company that never holds a concall: eighteen
    sections spanning what the business does, how it performed, what it is building, what
    it promises and what it admits — every line quoted from the deck at extraction.

    The coverage count is printed deliberately. "7 of 18 sections" tells a reader the deck
    was thin, which is itself information about the company; without it, an absent
    balance sheet reads as a pipeline failure rather than as a 15-page deck.
    """
    ds = _slice(tables.get("deck_summary"), isin)
    if ds.empty:
        return ""
    if "quarter" in ds.columns:
        want = QT.norm_q(season)
        ds = ds[ds["quarter"].astype(str).map(lambda x: QT.norm_q(x) == want)]
    if ds.empty:
        return ""

    present = {str(s).strip() for s in ds["section"]}
    n_have = sum(1 for s, _, _ in _SUMMARY_SECTIONS if s in present)

    out = [_h("The deck on its own terms",
              f"Everything this presentation discloses, quoted from it. "
              f"<b>{n_have} of 18</b> sections are covered by this deck.")]

    body = []
    for sec, title, colour in _SUMMARY_SECTIONS:
        part = ds[ds["section"].astype(str).str.strip() == sec]
        if part.empty:
            continue
        body.append(f"<tr><td colspan='3' style='padding-top:9px;color:{colour};"
                    f"font-weight:700;border-bottom:1px solid {colour}'>{title}</td></tr>")
        for _, r in part.iterrows():
            val = r.get("value")
            num = ""
            if val is not None and str(val).strip() not in ("", "nan", "None"):
                unit = _esc(DS.display_unit(r.get("unit")), 12)
                # A percentage is directional and gets the up/down colour; an absolute
                # level is not — a bigger number is not automatically better news.
                txt = f"{float(val):g}"
                num = (colour_num(f"{txt}{unit}") if unit == "%"
                       else f"<b>{_esc(txt, 18)}{(' ' + unit) if unit else ''}</b>")
            per = _esc(r.get("period"), 16)
            det = _esc(r.get("detail"), 210)
            body.append(
                f"<tr><td style='width:31%'>{_esc(r.get('label'), 70)}"
                + (f" <span style='color:{MUTED};font-size:11px'>{per}</span>" if per else "")
                + f"</td><td style='width:17%;text-align:right'>{num}</td>"
                  f"<td style='color:#444'>{det}</td></tr>")

    if not body:
        return ""
    out.append(f"<table cellpadding='4' cellspacing='0' style='{_TBL}'>"
               + "".join(body) + "</table>")
    return "".join(out)


def _deck_key(isin: str, tables: dict) -> str:
    """Coarse deck fingerprint: the quarter, and whether there is usable content.

    Row counts and section counts are deliberately NOT in the key — they drift run to run
    on an unchanged deck (measured: 35 rows then 33 for the same document), and a key that
    moves on its own would re-send the portfolio on every backfill.
    """
    parts = []
    for name in ("deck_summary", "ppt_highlights", "ppt_guidance"):
        t = _slice(tables.get(name), isin)
        parts.append("1" if not t.empty else "0")
    if parts == ["0", "0", "0"]:
        return ""          # nothing to fingerprint; no key means no change check
    return "deck|" + "".join(parts)


def _doc_rows(tables, name, isin, doc_id, limit=0):
    """The rows THIS document produced, newest first.

    Falls back to the company's most recent rows when nothing carries this doc_id: rows
    written before source_doc_id was stamped, and AR guidance tabulated by a second pass
    that keys on the report rather than on the queue row.
    """
    d = _slice(tables.get(name), isin)
    if d.empty:
        return d
    if doc_id and "source_doc_id" in d.columns:
        hit = d[d["source_doc_id"].astype(str).str.strip() == str(doc_id).strip()]
        if not hit.empty:
            d = hit
    if "processed_at" in d.columns:
        d = d.sort_values("processed_at", ascending=False)
    return d.head(limit) if limit else d


def _num(v, suffix="") -> str:
    """Parquet number -> display string. NaN/None/blank all render as nothing."""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return ""
    if f != f:                                   # NaN
        return ""
    return f"{f:g}{suffix}"


def _tile_row(pairs) -> str:
    """Key numbers as tiles; blank values are dropped rather than rendered empty."""
    cells = ""
    for lab, val, good in pairs:
        v = str(val or "").strip()
        if not v or v.lower() in ("nan", "none"):
            continue
        cells += (f"<td style='vertical-align:top;padding:0 8px 0 0'>"
                  f"<div style='border:1px solid #e0e6ea;background:#f6f8f9;"
                  f"padding:7px 10px'>"
                  f"<div style='font-size:10.5px;color:{MUTED};text-transform:uppercase;"
                  f"letter-spacing:.3px'>{_esc(lab, 26)}</div>"
                  f"<div style='font-size:16px;margin-top:2px'>"
                  f"{colour_num(v, good_up=good)}</div></div></td>")
    return (f"<table cellpadding='0' cellspacing='0' style='{_TBL}'>"
            f"<tr>{cells}</tr></table>" if cells else "")


# The sections of Phase 2's own reports worth putting in front of a reader, in the order
# a reader wants them. Matched as substrings against the sub-headings the prompts emit -
# annual_report_prompt.txt numbers its sections ("### 2. FINANCIAL PERFORMANCE & GROWTH
# TRAJECTORY"), concall_prompt.txt letters its own ("### A-1 Executive Summary").
_NARRATIVE_WANTS = (
    "executive summary", "investment thesis", "thesis matrix",
    "financial performance", "growth trajectory", "management outlook",
    "management commentary", "forward guidance", "capital efficiency",
    "governance", "q&a summary", "growth drivers", "business overview",
)
# Never lift these as "the summary": methodology, the scorecard the mail renders as its
# own table, and the question list.
_NARRATIVE_SKIP = ("source coverage", "data integrity", "probing questions",
                   "risk scorecard", "credibility", "mgmt said", "disclaimer")
# THE MODEL SOMETIMES ECHOES THE PROMPT BACK. CGPOWER's stored annual-report analysis
# opens with annual_report_prompt.txt itself - "Generate the final report immediately...
# The ENTIRE report must stay under ~1,200 lines" - and that text is long and prose-like,
# so every length-based heuristic ranks it first. It is instructions, never analysis, and
# it must never reach a reader. Filtered at RENDER time; the stored text is untouched.
def _is_prompt_echo(text: str) -> bool:
    """True when a block is the prompt talking, not the analysis.

    Delegates to _extractor_base so the extractor's quality gate and this renderer
    agree on what an echo is - the extractor now refuses to store one, and this stays
    as the guard for the rows stored before it did.
    """
    from _extractor_base import is_prompt_echo
    return is_prompt_echo(text)


def _digest_prose(text: str) -> str:
    """The digest's prose filter (drops tables, fences, tags, metadata), reused rather
    than re-implemented so both mails strip the same things."""
    from run_pf_docs_digest import _prose
    return _prose(text)


# A report section header, in the three shapes these prompts actually emit:
#   "### 2. FINANCIAL PERFORMANCE"   hashed (APL Apollo, most concalls)
#   "2. FINANCIAL PERFORMANCE"       plain numbered (Deep Industries) - without this the
#                                    whole report stayed ONE block and the section
#                                    priority below had nothing to choose between
#   "**2. FINANCIAL PERFORMANCE**"   bolded
# STRICT. The first version allowed the leading marker to be EMPTY, so any capitalised
# prose line of the right length was treated as a heading; the report shattered into
# fragments and the mail carried a sentence like 'Specific details on the nature of
# "Other financial assets" (non-'. A heading must now be explicitly marked as one:
_SECTION_HEAD_RE = re.compile(
    r"^\s*(?:"
    r"#{1,4}\s+\S"                                     # ### 2. FINANCIAL PERFORMANCE
    r"|\*\*[^*]{4,90}\*\*\s*:?\s*$"                    # **A-3) Forward Guidance**
    r"|(?:[0-9]{1,2}[A-Z]?[.)]|[A-Z]-[0-9][.)]?)\s+"     # 2. / 5B. / A-1
    r"[A-Z][A-Za-z0-9 &/,'\-()]{4,80}\s*:?\s*$"
    r")")


def _subsections(region) -> list:
    """[(heading, body)] for a document region, split on its own section headings.

    The region's own top-level section comes first with an empty heading, so a report
    that uses no sub-headings at all still yields its text.
    """
    out = []
    for h, body in region or []:
        cur_h, cur_b = "", []
        for ln in str(body).splitlines():
            if _SECTION_HEAD_RE.match(ln):
                if cur_h or cur_b:
                    out.append((cur_h, "\n".join(cur_b)))
                cur_h, cur_b = ln, []
            else:
                cur_b.append(ln)
        if cur_h or cur_b:
            out.append((cur_h, "\n".join(cur_b)))
    return out


def lift_report(region, limit: int = None) -> list:
    """The most useful parts of Phase 2's report, as [(heading, prose)].

    WHY NOT THE DIGEST'S _lift_summary. That returns the single best FRAGMENT, which is
    the right shape for a one-line digest row and the wrong shape for a mail whose whole
    job is the document. Measured across 25 PF annual reports on 2026-09-02 it returned
    74 characters from CGPOWER's 88,819-character report and 0 from NAVINFLUOR's 10,639,
    because these reports are mostly TABLES and the prose filter drops table rows - so
    whichever single section it landed on had almost no qualifying text left.

    This walks the report's own sections in reader order and takes several, so a mail
    carries the analysis rather than a sentence of it. Numbers are not lost by dropping
    tables: the mail renders ar_guidance and ar_red_flags as its own tables alongside.
    """
    limit = NARRATIVE_LIMIT if limit is None else limit
    subs = _subsections(region)
    if not subs:
        return []
    picked, seen, used = [], set(), 0
    for want in _NARRATIVE_WANTS:
        for i, (h, body) in enumerate(subs):
            hl = str(h).lower()
            if i in seen or want not in hl or any(s in hl for s in _NARRATIVE_SKIP):
                continue
            prose = _digest_prose(body)
            if len(prose) < 80 or _is_prompt_echo(prose) or _is_prompt_echo(h):
                continue
            room = limit - used
            if room < 200:
                break
            seen.add(i)
            picked.append((re.sub(r"^#+\s*", "", str(h)).strip(),
                           prose[:room].rstrip() + ("…" if len(prose) > room else "")))
            used += min(len(prose), room)
        if used >= limit - 200:
            break
    if picked:
        return picked
    # Nothing matched a wanted heading - fall back to the longest qualifying block, so a
    # report that numbers its sections differently still says something.
    best = ""
    for h, body in subs:
        if any(s in str(h).lower() for s in _NARRATIVE_SKIP):
            continue
        pr = _digest_prose(body)
        if _is_prompt_echo(pr) or _is_prompt_echo(h):
            continue
        if len(pr) > len(best):
            best = pr
    return [("", best[:limit].rstrip() + ("…" if len(best) > limit else ""))] if best else []


REPORT_LIMIT = 60000        # Gmail clips a message around 102 KB; the structured
                            # tables and chrome take the rest. Phase 2's reports run
                            # ~5-10k chars, so this only bites on an outlier.

_MD_SKIP = re.compile(r"^\s*(?:<!--.*?-->\s*)?$")
_MD_DOCHEAD = re.compile(r"^##\s+\S.*\s(?:Annual Report|Concall|Presentation|"
                         r"Credit Rating|Results)\s+[-\u2013\u2014]", re.I)


def _md_inline(s: str) -> str:
    """Bold and code inside a line. Escapes first, so no markup can leak through."""
    out = _esc(s, 4000)
    out = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", out)
    out = re.sub(r"`([^`]+)`", r"<code style='font-size:11.5px'>\1</code>", out)
    return out


def md_to_html(md: str, limit: int = None) -> str:
    """Phase 2's own report, rendered as HTML - headings, bullets, TABLES and all.

    NOT a summary of a summary. The report IS the summary; this mail's job is to carry
    it, not to distil it a second time. Measured 2026-09-02, the previous approach
    lifted 522 of APL Apollo's 4,881-char annual-report analysis (11%) and 1,486 of its
    9,975-char concall brief (15%), and dropped every one of their 21 and 40 table rows,
    because the prose filter it reused was built for one-line digest rows.

    The document's own top heading is skipped - the mail renders its own header - and so
    is the "*Processed:*" stamp and the doc marker.
    """
    limit = REPORT_LIMIT if limit is None else limit
    text = re.sub(r"<!--.*?-->", " ", str(md or ""), flags=re.S)
    lines = text.splitlines()
    out, i, n = [], 0, len(lines)
    while i < n:
        raw = lines[i]
        s = raw.strip()
        i += 1
        if not s or s in ("---", "***", "___") or _MD_SKIP.match(s):
            continue
        if s.startswith("*Processed:") or _MD_DOCHEAD.match(s):
            continue
        # ---- table: a run of pipe rows -------------------------------------
        if s.startswith("|"):
            rows = []
            j = i - 1
            while j < n and lines[j].strip().startswith("|"):
                cells = [c.strip() for c in lines[j].strip().strip("|").split("|")]
                if not all(re.fullmatch(r":?-{2,}:?", c or "-") for c in cells):
                    rows.append(cells)
                j += 1
            i = j
            if rows:
                head, body = rows[0], rows[1:]
                th = "".join(f"<th style='text-align:left;padding:4px 9px 4px 0;"
                             f"border-bottom:1px solid #ccc;color:{MUTED};"
                             f"font-weight:600'>{_md_inline(c)}</th>" for c in head)
                tb = "".join("<tr>" + "".join(
                    f"<td style='padding:4px 9px 4px 0;border-bottom:1px solid #eee;"
                    f"vertical-align:top'>{_md_inline(c)}</td>" for c in r) + "</tr>"
                    for r in body)
                out.append(f"<div style='overflow-x:auto'><table cellpadding='0' "
                           f"cellspacing='0' style='{_TBL}'><tr>{th}</tr>{tb}</table>"
                           f"</div>")
            continue
        # ---- headings ------------------------------------------------------
        m = re.match(r"^(#{1,6})\s+(.*)$", s)
        if m:
            lvl = len(m.group(1))
            size, col = (14, "#34495e") if lvl <= 2 else (13, BLUE)
            out.append(f"<div style='font-size:{size}px;font-weight:700;color:{col};"
                       f"margin:14px 0 5px'>{_md_inline(m.group(2))}</div>")
            continue
        # a fully bolded line is a heading in these reports ("**A-2) Q&A Summary**")
        if s.startswith("**") and s.endswith("**") and len(s) < 110:
            out.append(f"<div style='font-size:13px;font-weight:700;color:{BLUE};"
                       f"margin:12px 0 4px'>{_md_inline(s.strip('*'))}</div>")
            continue
        # ---- bullets -------------------------------------------------------
        if re.match(r"^[-*\u2022]\s+", s):
            items = []
            j = i - 1
            while j < n:
                t = lines[j].strip()
                if re.match(r"^[-*\u2022]\s+", t):
                    items.append(re.sub(r"^[-*\u2022]\s+", "", t))
                elif t and items and lines[j].startswith(("   ", "\t")):
                    items[-1] += " " + t          # continuation of the same bullet
                else:
                    break
                j += 1
            i = j
            lis = "".join(f"<li style='margin:3px 0'>{_md_inline(x)}</li>"
                          for x in items)
            out.append(f"<ul style='margin:4px 0 8px;padding-left:20px;"
                       f"font-size:12.5px;color:#333;line-height:1.5'>{lis}</ul>")
            continue
        # ---- paragraph -----------------------------------------------------
        out.append(f"<p style='margin:5px 0;font-size:12.5px;color:#333;"
                   f"line-height:1.55'>{_md_inline(s)}</p>")
        if sum(len(x) for x in out) > limit:
            out.append(f"<p style='color:{MUTED};font-size:11.5px'>"
                       f"[report truncated for email length]</p>")
            break
    return "".join(out)


def html_to_pdf(doc_html: str, max_pages: int = 200) -> bytes | None:
    """A4 PDF of a standalone HTML document, or None if it cannot be produced.

    PyMuPDF is already a declared dependency (scripts/requirements.txt) and is used
    elsewhere for page-range chunking, so this adds nothing to install.
    """
    try:
        import fitz
        story = fitz.Story(html=doc_html)
        buf = io.BytesIO()
        writer = fitz.DocumentWriter(buf)
        page = fitz.paper_rect("a4")
        frame = page + (36, 40, -36, -40)
        more, n = 1, 0
        while more and n < max_pages:
            dev = writer.begin_page(page)
            more, _ = story.place(frame)
            story.draw(dev)
            writer.end_page()
            n += 1
        writer.close()
        data = buf.getvalue()
        return data if data.startswith(b"%PDF") else None
    except Exception:
        return None


def report_attachment(symbol: str, label: str, subtitle: str,
                      narrative) -> tuple | None:
    """(filename, bytes, "html") - Phase 2's report as a standalone file, or None.

    WHY ATTACH SOMETHING THE BODY ALREADY CONTAINS. Gmail collapses repeated content
    when several messages share a subject: send the same annual report twice and the
    second arrives with its body behind a "..." trim, which reads exactly like the mail
    was cut off. An attachment is never trimmed, never clipped, and can be opened or
    kept independently of the thread.
    """
    md = narrative if isinstance(narrative, str) else "\n".join(
        ((h + "\n") if h else "") + t for h, t in (narrative or []))
    if not str(md).strip():
        return None
    inner = md_to_html(md, limit=REPORT_LIMIT)
    if not inner:
        return None
    doc = (f"<!doctype html><html><head><meta charset='utf-8'>"
           f"<title>{_esc(symbol, 20)} {_esc(label, 60)}</title></head>"
           f"<body style='{_WRAP};max-width:820px;margin:24px auto;padding:0 18px'>"
           f"<h2 style='margin:0 0 2px'>{_esc(symbol, 20)} &middot; {_esc(label, 80)}</h2>"
           f"<div style='color:{MUTED};font-size:12px;margin:0 0 16px'>"
           f"{_esc(subtitle, 120)}</div>{inner}</body></html>")
    safe = re.sub(r"[^A-Za-z0-9]+", "_", f"{symbol}_{label}").strip("_")[:70]
    # PDF FIRST. An .html attachment is at the mercy of whatever opens it - a mail
    # client may sandbox it, refuse to preview it, or render only part of it, which is
    # exactly the complaint this replaces. A PDF renders identically everywhere, cannot
    # be thread-trimmed, and prints. HTML stays as the fallback if PyMuPDF is missing,
    # since a readable attachment beats none.
    pdf = html_to_pdf(doc)
    if pdf:
        return (f"{safe}.pdf", pdf, "pdf")
    return (f"{safe}.html", doc.encode("utf-8"), "html")


def _narrative_html(narrative, title: str, sub: str) -> str:
    """Accepts either a plain string or [(heading, prose)] from lift_report()."""
    md = narrative if isinstance(narrative, str) else "\n".join(
        ((h + "\n") if h else "") + t for h, t in (narrative or []))
    if not str(md).strip():
        return ""
    inner = md_to_html(md)
    if not inner:
        return ""
    return (_h(title, sub)
            + f"<div style='border-left:3px solid #ccc;padding-left:12px;"
              f"margin:0 0 14px'>{inner}</div>")


def _narrative_key(doc_type: str, isin: str, tables: dict) -> str:
    """Coarse fingerprint for a narrative document: its period, and whether a narrative
    could be lifted at all.

    Coarse for exactly the reason _deck_key is coarse - the body is an LLM pass over a
    long PDF and its wording drifts between extractions, so fingerprinting the prose would
    mark almost every re-extract as "changed" and re-send the portfolio. What it DOES
    catch is the case worth catching: a document that yielded no readable summary later
    yielding one.

    A genuine supersede needs no key at all. The richer document arrives with a NEW
    doc_id and the old queue row goes status='superseded', so mail_due already sees a
    document it has never mailed.
    """
    n = (tables.get("_narr") or {}).get((str(isin).strip(), doc_type)) or {}
    per = str(n.get("period") or "").strip().upper()
    _t = n.get("text")
    has = "1" if (_t if isinstance(_t, str) else "".join(x for _h, x in (_t or []))
                  ).strip() else "0"
    if not per and has == "0":
        return ""                    # nothing known; no key means no change check
    return f"{doc_type}|{per}|{has}"


def content_key(doc_type: str, isin: str, tables: dict) -> str:
    """A fingerprint of WHAT THE MAIL ASSERTS, so a correction can be detected.

    "Already mailed" and "already told correctly" are different things, and the ledger
    only knew the first. Tatva Chintan was mailed as rating "D, Reaffirmed" and is
    genuinely "BBB+, Downgraded"; Yasho and Univastu went out as defaults and are both
    upgrades. Keyed on doc_id alone those corrections could never reach the reader,
    because the document had "been mailed".

    DERIVED FROM THE DATA, NOT THE RENDERED HTML. Hashing the body would be simpler and
    would re-send every holding the first time anyone edited a template — a cosmetic
    change is not news.

    TWO TYPES, TWO DIFFERENT KEYS, because the two extractions fail differently.

    A RATING is four short fields read out of a document that states them plainly, so the
    key is the assertion itself: agency, rating, outlook, action. Any change is real.

    A DECK is an LLM pass over slides, and it DRIFTS. The same RISHABH deck, same prompt,
    two runs minutes apart, returned 35 rows and then 33, with the strategy section going
    from 3 rows to 1. Fingerprinting deck CONTENT would therefore mark almost every
    re-extract as "changed" and re-send the entire portfolio on every backfill. So the
    deck key is deliberately COARSE: the quarter, and whether the deck yields usable
    content at all.

    That is precision over recall, chosen on evidence. It catches the case that actually
    happened — 45 of 202 decks returned ZERO rows because a truncated response was
    discarded whole, and the salvage fix in 4a201c3 turned them into real content. A
    reader told "this company published nothing usable" deserves to hear when that stops
    being true. It will NOT catch a deck whose numbers shifted between extractions, and
    that is the honest trade: a false correction is worse than a missed one, because it
    trains the reader to ignore the word.
    """
    if doc_type == "presentation":
        return _deck_key(isin, tables)
    if doc_type in ("concall", "annual_report"):
        return _narrative_key(doc_type, isin, tables)
    if doc_type != "rating":
        return ""
    rt = _slice(tables.get("ratings"), isin)
    if rt.empty:
        return ""
    if "rating_date" in rt.columns:
        rt = rt.sort_values("rating_date")
    cur = rt.iloc[-1]

    def _v(k):
        return str(cur.get(k) or "").strip().upper()

    return "|".join((_v("agency"), _v("rating"), _v("outlook"), _v("rating_action")))


def presentation_body(isin, symbol, name, season, tables) -> str:
    """This quarter's deck, then how it differs from the previous one."""
    hi = _slice(tables.get("ppt_highlights"), isin)
    gu = _slice(tables.get("ppt_guidance"), isin)
    want = QT.norm_q(season)

    def _q(df):
        if df.empty or "quarter" not in df.columns:
            return df
        return df[df["quarter"].astype(str).map(lambda x: QT.norm_q(x) == want)]

    cur_hi, cur_gu = _q(hi), _q(gu)
    # The standalone summary is computed BEFORE the gate: a company that holds no concall
    # may have a full deck_summary and no ppt_highlights at all, and gating on the old
    # two tables would suppress exactly the mail this section was built for.
    standalone = standalone_summary(isin, season, tables)
    if cur_hi.empty and cur_gu.empty and not standalone:
        return ""

    out = [f"<div style='{_WRAP}'>",
           f"<h2 style='margin:0 0 2px'>&#128202; {_esc(name, 70)} "
           f"<span style='color:#888;font-weight:400'>&middot; {_esc(symbol, 20)}</span></h2>",
           f"<div style='color:#888;font-size:12px;margin:0 0 10px'>Investor presentation "
           f"&middot; <b>{_esc(QT.qtr_label(season) if ' ' not in season else season, 12)}</b>"
           f"</div>"]

    def _val(r):
        v, u = _esc(r.get("value"), 24), _esc(r.get("unit"), 10)
        return f"<b>{v}{u}</b>" if v else ""

    out.append(_key_callouts(cur_hi))
    out.append(standalone)

    hi_rows = [[_esc(r.get("category"), 24), _esc(r.get("statement"), 220), _val(r)]
               for _, r in cur_hi.iterrows()
               if not _is_esg(r.get("statement"), r.get("category"))]
    if hi_rows:
        out.append(_h("What the deck says", "Highlights as published this quarter.")
                   + _rows_html(hi_rows[:10], ["Area", "Statement", "Value"]))

    gu_rows = [[f"<b>{_esc(r.get('metric'), 40)}</b>", _val(r), _esc(r.get("horizon"), 24)]
               for _, r in cur_gu.iterrows() if not _is_esg(r.get("metric"))]
    if gu_rows:
        out.append(_h("Guidance given in this deck") +
                   _rows_html(gu_rows[:8], ["Metric", "Value", "Horizon"]))
    elif not cur_gu.empty:
        out.append(f"<div style='color:{MUTED};font-size:12px;margin:8px 0'>"
                   f"The deck's stated targets this quarter are ESG commitments only "
                   f"(emissions, renewables, workforce) — no financial or operational "
                   f"guidance was given.</div>")

    out.append(_business_view(isin, season, tables))
    out.append(_risks_and_changes(isin, season, tables))
    out.append(_consistency(isin, season, tables))
    out.append(_promised_vs_delivered(isin, tables))
    out.append(_deck_changed(isin, season, tables))
    out.append(f"<div style='color:{MUTED};font-size:11px;margin-top:10px'>Deck rows are "
               f"quoted from the presentation itself; anything not evidenced in it is "
               f"dropped at extraction.</div></div>")
    return "".join(out)


_VERDICT_TONE = {"beat": UP, "exceeded": UP, "delivered": UP, "met": UP,
                 "inline": MUTED, "partial": AMBER, "too_early": MUTED, "na": MUTED,
                 "miss": DOWN, "missed": DOWN, "below": DOWN}


_KEY_CATEGORIES = ("orderbook", "order book", "capacity", "utilisation", "utilization",
                   "volume", "realisation", "realization", "margin", "demand",
                   "expansion", "capex", "guidance")


def _key_callouts(cur_hi) -> str:
    """The three or four lines worth reading first.

    A deck highlights table is long and flat. What a reader wants at the top is the
    handful of statements that carry a NUMBER in a category that moves a thesis —
    order book, capacity, utilisation, volume, realisation, margin. Everything else
    stays below in the full table.
    """
    scored = []
    for _, r in cur_hi.iterrows():
        stmt, cat = str(r.get("statement") or ""), str(r.get("category") or "")
        val, unit = str(r.get("value") or ""), str(r.get("unit") or "")
        if _is_esg(stmt, cat):
            continue
        has_num = bool(val.strip()) and val.strip().lower() not in ("nan", "none")
        in_key = any(k in (cat + " " + stmt).lower() for k in _KEY_CATEGORIES)
        if not has_num:
            continue
        scored.append((2 if in_key else 1, stmt, val, unit, cat))
    if not scored:
        return ""
    scored.sort(key=lambda x: -x[0])
    cards = ""
    for _s, stmt, val, unit, cat in scored[:4]:
        cards += (
            f"<td style='vertical-align:top;padding:0 8px 0 0;width:25%'>"
            f"<div style='border:1px solid #e0e6ea;border-left:3px solid {UP};"
            f"background:#f6f8f9;padding:8px 10px'>"
            f"<div style='font-size:10.5px;color:{MUTED};text-transform:uppercase;"
            f"letter-spacing:.3px'>{_esc(cat, 22)}</div>"
            f"<div style='font-size:17px;font-weight:700;color:#111;margin:2px 0'>"
            f"{_esc(val, 18)}{_esc(unit, 8)}</div>"
            f"<div style='font-size:11px;color:#444'>{_esc(stmt, 90)}</div>"
            f"</div></td>")
    return (_h("Key call-outs", "The numbers from this deck worth reading first.")
            + f"<table cellpadding='0' cellspacing='0' style='{_TBL}'><tr>{cards}</tr>"
              f"</table>")


# The deck's job is the OPERATING picture: what the business is physically doing. The
# statement says how the money behaved; these say how the business behaved. Categories
# come from deck_teardown's controlled vocabulary plus the deck's own highlight buckets.
_BUSINESS_VIEWS = (
    ("Capacity and expansion", ("capacity", "expansion", "capex", "plant", "greenfield",
                                "brownfield", "commission")),
    ("Utilisation", ("utilisation", "utilization", "occupancy", "run rate")),
    ("Order book and pipeline", ("orderbook", "order book", "pipeline", "backlog",
                                 "tender", "win")),
    ("Volume and realisation", ("volume", "realisation", "realization", "tonnage",
                                "asp", "price")),
    ("Segment mix", ("segment", "product mix", "category", "vertical")),
    ("Geography", ("geo", "geograph", "export", "domestic", "region")),
    ("Demand and macro", ("demand", "macro", "industry", "market size", "cycle",
                          "consumption")),
)



# Themes an operating line can belong to, each with the colour it is emphasised in.
# Order matters: the first match wins, so the more specific themes are listed first.
_THEMES = (
    ("guidance", PURPLE, ("guidance", "target", "outlook", "expect", "aim", "plan to",
                          "by fy", "we will", "on track")),
    ("risk", DOWN, ("risk", "headwind", "pressure", "decline", "slowdown", "delay",
                    "deferred", "shortfall", "weak", "challenge", "impact of")),
    ("capacity", BLUE, ("capacity", "expansion", "capex", "commission", "plant",
                        "greenfield", "brownfield", "debottleneck", "utilisation",
                        "utilization")),
    ("growth", UP, ("growth", "grew", "increase", "record", "highest", "up ", "demand",
                    "order", "volume", "market share", "new product", "launch")),
)


def theme_of(text: str) -> tuple[str, str]:
    """(theme, colour) for an operating line — so a risk never reads green."""
    low = str(text or "").lower()
    for name, col, keys in _THEMES:
        if any(k in low for k in keys):
            return name, col
    return "", MUTED


def _chip(label: str, colour: str) -> str:
    return (f"<span style='background:{colour};color:#fff;border-radius:3px;"
            f"padding:1px 6px;font-size:10px;font-weight:700;text-transform:uppercase;"
            f"letter-spacing:.3px'>{label}</span>")


def _risks_and_changes(isin, season, tables) -> str:
    """Risks and management changes — which the DECK will not tell you.

    deck_metrics bans financial words and decks rarely volunteer bad news, so neither
    risk nor a management change appears there. Both have real sources elsewhere:
      framing flags   deck_flags      — how the deck is staged (truncated axes, etc.)
      commentary      gf4_quality_flags — contradictory or promotional concall language
      management      announcement_ledger event_type='management_change'
    Sourcing them honestly is better than implying the deck disclosed them.
    """
    rows = []
    df = _slice(tables.get("deck_flags"), isin)
    if not df.empty:
        for _, r in df.head(4).iterrows():
            rows.append([_chip("framing", AMBER),
                         _esc(str(r.get("flag_type") or "").replace("_", " "), 40),
                         _esc(r.get("evidence"), 170)])
    gf4 = _slice(tables.get("gf4_quality_flags"), isin)
    if not gf4.empty:
        for _, r in gf4.head(3).iterrows():
            rows.append([_chip("commentary", AMBER),
                         _esc(r.get("flag_type"), 40), _esc(r.get("evidence"), 170)])
    ann = tables.get("announcement_ledger")
    if ann is not None and not getattr(ann, "empty", True) and "event_type" in ann.columns:
        a = ann[(ann["isin"].astype(str).str.strip() == str(isin).strip())
                & (ann["event_type"].astype(str) == "management_change")]
        if "ann_date" in a.columns:
            a = a.sort_values("ann_date", ascending=False)
        for _, r in a.head(3).iterrows():
            rows.append([_chip("management", PURPLE),
                         _esc(str(r.get("ann_date"))[:10], 12),
                         _esc(r.get("summary") or r.get("headline"), 170)])
    if not rows:
        return ""
    return (_h("Risks and changes",
               "Not from the deck — companies rarely put these in one. Framing flags "
               "come from the deck teardown, commentary flags from the concall, and "
               "management changes from exchange filings.")
            + _rows_html(rows, ["", "What", "Detail"]))


def _business_view(isin, season, tables) -> str:
    """How the BUSINESS is behaving — capacity, utilisation, orders, mix, geography, macro.

    The deck is where the operating story lives, and it was being rendered as one flat
    list of highlights. Grouping it into the views an analyst actually asks for turns the
    same rows into an operating picture: is capacity going up, is it being used, is the
    order book covering it, where is the volume coming from, and what is the company
    saying about its market.

    deck_metrics (the grounded teardown pass) is preferred where it exists, because every
    row there had to quote the deck verbatim; ppt_highlights fills in otherwise.
    """
    hi = _slice(tables.get("ppt_highlights"), isin)
    dm = _slice(tables.get("deck_metrics"), isin)
    want = QT.norm_q(season)

    items = []
    if not dm.empty:
        for _, r in dm.iterrows():
            q = QT.norm_q(str(r.get("quarter") or ""))
            if q and q != want:
                continue
            items.append((f"{r.get('category','')} {r.get('metric','')}",
                          str(r.get("metric") or ""),
                          f"{_esc(r.get('value'), 20)}{_esc(r.get('unit'), 10)}",
                          str(r.get("slide_ref") or "")))
    if not items and not hi.empty:
        h = hi
        if "quarter" in h.columns:
            h = h[h["quarter"].astype(str).map(lambda x: QT.norm_q(x) == want)]
        for _, r in h.iterrows():
            if _is_esg(r.get("statement"), r.get("category")):
                continue
            items.append((f"{r.get('category','')} {r.get('statement','')}",
                          str(r.get("statement") or ""),
                          f"{_esc(r.get('value'), 20)}{_esc(r.get('unit'), 10)}", ""))
    if not items:
        return ""

    out, used = [], set()
    for title, keys in _BUSINESS_VIEWS:
        rows = []
        for i, (blob, label, val, ref) in enumerate(items):
            if i in used:
                continue
            if any(k in blob.lower() for k in keys):
                used.add(i)
                th, tcol = theme_of(label)
                rows.append([_chip(th, tcol) if th else "",
                             _esc(label, 190),
                             colour_num(val) if val else "",
                             _esc(ref, 14)])
        if rows:
            out.append(_h(title) + _rows_html(rows[:6], ["", "What the deck says",
                                                         "Value", "Slide"]))
    if not out:
        return ""
    return (_h("How the business is behaving",
               "The operating picture from the deck — capacity, utilisation, orders, "
               "mix, geography and the market backdrop. The financial teardown covers "
               "how the money behaved; this covers how the business did.")
            + "".join(out))


def _consistency(isin, season, tables) -> str:
    """What the company keeps saying, and what it has quietly stopped saying.

    Two halves, because they fail differently:
      NUMBERS — a metric reported in several quarters, and how its value moved. A target
                restated unchanged is a promise being kept; one revised down without
                comment is the thing worth catching.
      COMMENTARY — a theme repeated quarter after quarter is a consistent message; one
                that ran for several quarters and then vanished is the cheapest early
                warning there is, because companies stop mentioning bad news rather than
                announcing it.
    """
    hi = _slice(tables.get("ppt_highlights"), isin)
    if hi.empty or "quarter" not in hi.columns:
        return ""
    hi = hi.copy()
    hi["_q"] = hi["quarter"].astype(str).map(QT.norm_q)
    quarters = sorted({q for q in hi["_q"] if q}, key=QT.q_order)
    if len(quarters) < 2:
        return ""
    cur, prev = quarters[-1], quarters[-2]

    def _rows(q):
        return hi[hi["_q"] == q]

    # ---- numbers: same statement, different value
    def _numkey(r):
        return _PUNCT.sub(" ", str(r.get("statement") or "").lower()).strip()[:70]

    cur_n = {_numkey(r): (str(r.get("value") or ""), str(r.get("unit") or ""))
             for _, r in _rows(cur).iterrows() if str(r.get("value") or "").strip()}
    prev_n = {_numkey(r): (str(r.get("value") or ""), str(r.get("unit") or ""))
              for _, r in _rows(prev).iterrows() if str(r.get("value") or "").strip()}
    same, moved = [], []
    for k, (v, u) in cur_n.items():
        if k not in prev_n:
            continue
        pv, _pu = prev_n[k]
        if str(v).strip() == str(pv).strip():
            same.append((k, v, u))
        else:
            moved.append((k, pv, v, u))

    # ---- commentary: themes carried forward vs dropped
    def _txt(q):
        return {_PUNCT.sub(" ", str(r.get("statement") or "").lower()).strip()[:70]:
                str(r.get("statement") or "")
                for _, r in _rows(q).iterrows()
                if not _is_esg(r.get("statement"), r.get("category"))}
    ct, pt = _txt(cur), _txt(prev)
    carried = [ct[k] for k in ct if k in pt][:4]
    dropped = [pt[k] for k in pt if k not in ct][:5]

    if not (same or moved or carried or dropped):
        return ""

    out = [_h("Consistent, and not",
              f"{_esc(cur,10)} against {_esc(prev,10)} — the same claim restated, "
              f"revised, or quietly dropped.")]
    rows = []
    for k, pv, v, u in moved[:6]:
        rows.append([f"<span style='color:{AMBER};font-weight:700'>revised</span>",
                     _esc(k, 70), _esc(pv, 20), f"{colour_num(v)}{_esc(u, 8)}"])
    for k, v, u in same[:4]:
        rows.append([f"<span style='color:{UP};font-weight:700'>restated</span>",
                     _esc(k, 70), _esc(v, 20) + _esc(u, 8), "unchanged"])
    if rows:
        out.append(_rows_html(rows, ["Number", "What", f"{_esc(prev,9)}", f"{_esc(cur,9)}"]))
    crows = []
    for s in dropped:
        crows.append([f"<span style='color:{DOWN};font-weight:700'>stopped saying</span>",
                      _esc(s, 200)])
    for s in carried:
        crows.append([f"<span style='color:{UP};font-weight:700'>still saying</span>",
                      _esc(s, 200)])
    if crows:
        out.append(_rows_html(crows, ["Commentary", "Statement"]))
    return "".join(out)


_PUNCT = __import__("re").compile(r"[^a-z0-9]+")


def _promised_vs_delivered(isin, tables) -> str:
    """What management SAID they would do, against what actually happened.

    "A KPI stopped being reported" is only half the signal. The half that matters more is
    a promise that was made and then quietly not kept: a margin target missed, a capacity
    date pushed out, a guidance range revised down. guidance_vs_actual already holds the
    join (guided / actual / delta / verdict) and mgmt_credibility holds the pattern; both
    were computed and never surfaced in a per-company mail.
    """
    gva = _slice(tables.get("guidance_vs_actual"), isin)
    cred = _slice(tables.get("mgmt_credibility"), isin)
    out = []

    if not gva.empty:
        g = gva.copy()
        if "period" in g.columns:
            g = g.sort_values("period")
        rows = []
        for _, r in g.tail(8).iloc[::-1].iterrows():
            v = str(r.get("verdict") or "").strip().lower().replace(" ", "_")
            tone = _VERDICT_TONE.get(v, MUTED)
            delta = r.get("delta")
            rows.append([
                _esc(r.get("period"), 12),
                f"<b>{_esc(r.get('metric'), 34)}</b>",
                _esc(r.get("guided"), 34),
                _esc(r.get("actual"), 34) or "&mdash;",
                colour_num(delta) if str(delta or "").strip() else "",
                f"<span style='color:{tone};font-weight:700'>"
                f"{_esc(str(r.get('verdict') or '').upper(), 12)}</span>",
            ])
        if rows:
            out.append(_h("Promised vs delivered",
                          "Management's own prior guidance, joined to what actually "
                          "happened. A missed promise is a stronger signal than a "
                          "dropped metric, because someone chose to make it.")
                       + _rows_html(rows, ["Period", "Metric", "Guided", "Actual",
                                           "Delta", "Verdict"]))

    hist = _slice(tables.get("gf2_historical_guidance"), isin)
    if not hist.empty:
        rows = []
        for _, r in hist.head(5).iterrows():
            orig = _esc(r.get("original_guidance"), 120)
            outc = _esc(r.get("actual_mentioned_outcome"), 120)
            if not orig:
                continue
            rows.append([_esc(r.get("financial_qtr"), 12), orig, outc or "&mdash;",
                         _esc(r.get("management_self_assessment"), 60)])
        if rows:
            out.append(_h("Historical guidance, revisited",
                          "What management said in earlier quarters and how they have "
                          "since described the outcome — in their own words.")
                       + _rows_html(rows, ["Quarter", "Said then", "Outcome",
                                           "Their assessment"]))

    # Consistency of the message itself — a team that guides accurately every quarter is
    # worth more than one that beats erratically, and the pattern is already computed.
    if not cred.empty:
        pat = ""
        for c in ("pattern", "recurring_miss", "strongest_area"):
            if c in cred.columns:
                vals = [str(x).strip() for x in cred[c]
                        if str(x).strip() and str(x).lower() not in ("none", "nan")]
                if vals:
                    label = {"pattern": "Guidance pattern",
                             "recurring_miss": "Recurring miss",
                             "strongest_area": "Most reliable on"}[c]
                    tone = (UP if c == "strongest_area" else
                            DOWN if c == "recurring_miss" else MUTED)
                    pat += (f"<tr><td style='color:{MUTED}'>{label}</td>"
                            f"<td style='color:{tone};font-weight:700'>"
                            f"{_esc(vals[-1], 90)}</td></tr>")
        if pat:
            out.append(_h("Is the commentary consistent?",
                          "Whether this management guides reliably, and where it "
                          "habitually falls short.")
                       + f"<table cellpadding='4' cellspacing='0' style='{_TBL}'>"
                         f"{pat}</table>")
    return "".join(out)


def _deck_changed(isin, season, tables) -> str:
    """What moved versus the previous deck. Deterministic set comparison."""
    diff = _slice(tables.get("deck_diff"), isin)
    want = QT.norm_q(season)
    if not diff.empty and "quarter" in diff.columns:
        diff = diff[diff["quarter"].astype(str).map(lambda x: QT.norm_q(x) == want)]
    if not diff.empty:
        tone = {"kpi_dropped": DOWN, "target_moved": DOWN, "definition_changed": AMBER,
                "de_emphasised": AMBER, "new_emphasis": UP}
        rows = [[f"<span style='color:{tone.get(str(r.get('change_type')), MUTED)};"
                 f"font-weight:700'>{_esc(str(r.get('change_type')).replace('_',' '), 24)}</span>",
                 _esc(r.get("item"), 60),
                 f"{_esc(r.get('prior_state'), 40)} &rarr; {_esc(r.get('current_state'), 40)}"]
                for _, r in diff.head(10).iterrows()]
        return (_h("What changed since the last deck",
                   "A KPI shown for several quarters and absent now is the cheapest early "
                   "warning there is — companies rarely announce bad news, they stop "
                   "mentioning it.") + _rows_html(rows, ["Change", "Item", "Was &rarr; now"]))

    # Fall back to comparing the metric sets ourselves when the LLM diff has no rows.
    met = _slice(tables.get("deck_metrics"), isin)
    if met.empty or "quarter" not in met.columns:
        return (f"<div style='color:{MUTED};font-size:12px;margin:8px 0'>"
                f"No previous deck has been processed for this company yet, so there is "
                f"nothing to compare against — this is the first one on record, not an "
                f"all-clear.</div>")
    qs = sorted({QT.norm_q(q) for q in met["quarter"].astype(str)}, key=QT.q_order)
    if len(qs) < 2:
        return (f"<div style='color:{MUTED};font-size:12px;margin:8px 0'>"
                f"Only one quarter of deck data on record ({_esc(qs[0] if qs else '?', 12)}), "
                f"so no comparison is possible yet. The next deck makes this section live."
                f"</div>")
    cur, prev = qs[-1], qs[-2]
    def _keys(q):
        s = met[met["quarter"].astype(str).map(lambda x: QT.norm_q(x) == q)]
        return {str(r.get("metric") or "").strip().lower() for _, r in s.iterrows()}
    dropped = sorted(_keys(prev) - _keys(cur))
    added = sorted(_keys(cur) - _keys(prev))
    rows = []
    for m in dropped[:8]:
        rows.append([f"<span style='color:{DOWN};font-weight:700'>stopped reporting</span>",
                     _esc(m, 60), f"shown in {_esc(prev, 12)}, absent in {_esc(cur, 12)}"])
    for m in added[:6]:
        rows.append([f"<span style='color:{UP};font-weight:700'>newly reported</span>",
                     _esc(m, 60), f"not in {_esc(prev, 12)}"])
    if not rows:
        return (f"<div style='color:{MUTED};font-size:12px;margin:8px 0'>"
                f"Same metrics reported as {_esc(prev, 12)} — nothing quietly dropped.</div>")
    return (_h("What changed since the last deck",
               f"Comparing {_esc(cur,12)} against {_esc(prev,12)}, by which metrics the "
               f"company chose to show.") + _rows_html(rows, ["Change", "Metric", "Detail"]))


# ------------------------------------------------------------------ #
#  Rating: current + trajectory                                       #
# ------------------------------------------------------------------ #

def rating_body(isin, symbol, name, tables) -> str:
    """Latest rating action, then what moved since the previous one."""
    rt = _slice(tables.get("ratings"), isin)
    if rt.empty:
        return ""
    rt = rt.copy()
    rt["_d"] = rt["rating_date"].astype(str).str.slice(0, 10)
    rt = rt[rt["_d"].str.match(r"\d{4}-\d{2}-\d{2}", na=False)].sort_values("_d")
    if rt.empty:
        return ""
    cur = rt.iloc[-1]

    out = [f"<div style='{_WRAP}'>",
           f"<h2 style='margin:0 0 2px'>&#127991; {_esc(name, 70)} "
           f"<span style='color:#888;font-weight:400'>&middot; {_esc(symbol, 20)}</span></h2>",
           f"<div style='color:#888;font-size:12px;margin:0 0 10px'>Credit rating "
           f"&middot; {_esc(cur.get('agency'), 24)} &middot; {_esc(cur['_d'], 12)}</div>"]

    # KEY NUMBERS FIRST. A rating mail is mostly prose; the figures that actually get
    # compared quarter to quarter are the rated amount, how long the rating has stood,
    # and how the agency's own tally of strengths against concerns has moved.
    _dr = _slice(tables.get("rating_drivers"), isin)
    _cn2 = _slice(tables.get("rating_concerns"), isin)
    _sn2 = _slice(tables.get("rating_sensitivity"), isin)
    _amt = str(cur.get("rated_amount_cr") or "").strip()
    _n_same = len(rt[rt["rating"].astype(str) == str(cur.get("rating"))])
    _tiles = ""
    for _lab, _v, _good in (
            ("Rated amount", f"{_amt} Cr" if _amt else "", True),
            ("Strengths cited", str(len(_dr)) if len(_dr) else "", True),
            ("Concerns cited", str(len(_cn2)) if len(_cn2) else "", False),
            ("Downgrade triggers", str(len(_sn2[_sn2.get("direction", pd.Series(dtype=str))
                                               .astype(str).str.lower() == "down"]))
             if len(_sn2) else "", False),
            ("Quarters at this rating", str(_n_same) if _n_same else "", True)):
        if not _v:
            continue
        _tiles += (f"<td style='vertical-align:top;padding:0 8px 0 0'>"
                   f"<div style='border:1px solid #e0e6ea;background:#f6f8f9;"
                   f"padding:7px 10px'>"
                   f"<div style='font-size:10.5px;color:{MUTED};text-transform:uppercase;"
                   f"letter-spacing:.3px'>{_lab}</div>"
                   f"<div style='font-size:16px;margin-top:2px'>"
                   f"{colour_num(_v, good_up=_good)}</div></div></td>")
    if _tiles:
        out.append(f"<table cellpadding='0' cellspacing='0' style='{_TBL}'>"
                   f"<tr>{_tiles}</tr></table>")

    act = str(cur.get("rating_action") or "")
    col = DOWN if "downgrade" in act.lower() else UP if "upgrade" in act.lower() else MUTED
    out.append(_rows_html([
        ["Rating", f"<b>{_esc(cur.get('rating'), 40)}</b>", ""],
        ["Outlook", _esc(cur.get("outlook"), 30), ""],
        ["Action", f"<span style='color:{col};font-weight:700'>{_esc(act, 24) or '&mdash;'}</span>", ""],
        ["Instrument", _esc(cur.get("instrument_type"), 40),
         f"{_esc(cur.get('rated_amount_cr'), 16)} Cr" if str(cur.get("rated_amount_cr") or "").strip() else ""],
    ], ["", "Current", ""]))

    # ---- what changed vs the previous rationale from the SAME agency
    same = rt[rt["agency"].astype(str) == str(cur.get("agency"))]
    if len(same) >= 2:
        prev = same.iloc[-2]
        rows = []
        if str(prev.get("rating")) != str(cur.get("rating")):
            rows.append(["Rating", _esc(prev.get("rating"), 40),
                         f"<b>{_esc(cur.get('rating'), 40)}</b>"])
        if str(prev.get("outlook")) != str(cur.get("outlook")):
            rows.append(["Outlook", _esc(prev.get("outlook"), 30),
                         f"<b>{_esc(cur.get('outlook'), 30)}</b>"])
        if rows:
            out.append(_h(f"What changed since {_esc(prev['_d'], 12)}")
                       + _rows_html(rows, ["", "Was", "Now"]))
        else:
            out.append(f"<div style='color:{MUTED};font-size:12px;margin:8px 0'>"
                       f"Rating and outlook unchanged since {_esc(prev['_d'], 12)}.</div>")

    # ---- concerns: new vs carried over
    cn = _slice(tables.get("rating_concerns"), isin)
    if not cn.empty and "quarter" in cn.columns or not cn.empty:
        txt_col = "concern" if "concern" in cn.columns else (
            "detail" if "detail" in cn.columns else cn.columns[-1])
        allc = [str(x).strip() for x in cn[txt_col] if str(x).strip()]
        if allc:
            rows = [[_esc(c, 220)] for c in allc[:6]]
            out.append(_h("Agency concerns on record",
                          "The agency's own words on what could go wrong.")
                       + _rows_html(rows, ["Concern"]))

    # ---- WHAT IS WORKING. The agency's own stated strengths — order book, market
    # position, promoter support, margin resilience. A rating mail that lists only
    # concerns and downgrade triggers reads as uniformly bearish even when the agency's
    # case is largely favourable, which misrepresents the document.
    dr = _slice(tables.get("rating_drivers"), isin)
    if not dr.empty:
        dcol = ("driver" if "driver" in dr.columns else
                "evidence" if "evidence" in dr.columns else dr.columns[-1])
        if "rating_date" in dr.columns:
            dr = dr.copy()
            dr["_d"] = dr["rating_date"].astype(str).str.slice(0, 10)
            latest_d = dr["_d"].max()
            dr = dr[dr["_d"] == latest_d]          # the current rationale only
        seen, rows = set(), []
        for _, r in dr.iterrows():
            txt = str(r.get(dcol) or "").strip()
            key = txt.lower()[:60]
            if not txt or key in seen:
                continue
            seen.add(key)
            rows.append([f"<span style='color:{UP}'>&#9679;</span> {_esc(txt, 240)}"])
            if len(rows) >= 6:
                break
        if rows:
            out.append(_h("What the agency says is working",
                          "Strengths cited in the current rationale — the other half of "
                          "the credit view.")
                       + _rows_html(rows, ["Strength"]))

    # ---- the most underused table in the repo: what would trigger a downgrade
    sn = _slice(tables.get("rating_sensitivity"), isin)
    if not sn.empty:
        dcol = "direction" if "direction" in sn.columns else None
        tcol = ("trigger" if "trigger" in sn.columns else
                "sensitivity" if "sensitivity" in sn.columns else sn.columns[-1])
        down = sn[sn[dcol].astype(str).str.lower() == "down"] if dcol else sn
        if not down.empty:
            rows = [[f"<span style='color:{DOWN}'>&#9660;</span> {_esc(r.get(tcol), 240)}"]
                    for _, r in down.head(6).iterrows()]
            out.append(_h("What would cause a downgrade",
                          "The agency stating, in its own words, exactly what it is "
                          "watching. A pre-written early-warning tripwire.")
                       + _rows_html(rows, ["Trigger"]))
        # The upgrade side is equally pre-written and never shown anywhere.
        up_ = sn[sn[dcol].astype(str).str.lower() == "up"] if dcol else sn.iloc[0:0]
        if not up_.empty:
            rows = [[f"<span style='color:{UP}'>&#9650;</span> {_esc(r.get(tcol), 240)}"]
                    for _, r in up_.head(5).iterrows()]
            out.append(_h("What would earn an upgrade",
                          "The conditions the agency has committed to rewarding — worth "
                          "tracking against the company's own guidance.")
                       + _rows_html(rows, ["Trigger"]))

    out.append("</div>")
    return "".join(out)


# ------------------------------------------------------------------ #
#  Concall: what was said, what was promised, what was delivered      #
# ------------------------------------------------------------------ #

def concall_body(isin, symbol, name, period, doc_id, narrative, tables) -> str:
    """The call in the management's own words, then the promises it can be held to.

    Same two-part shape as the other mails: the document first, then what it changes.
    For a concall "what changed" is the credibility record - what was guided in an
    earlier quarter set against what was actually delivered, which extract_concall
    already tabulates into mgmt_credibility whenever it has the historical context.
    """
    facts = _doc_rows(tables, "quarterly_facts", isin, doc_id, limit=1)
    guid = _doc_rows(tables, "guidance_tracker", isin, doc_id, limit=12)
    gf1 = _doc_rows(tables, "gf1_guidance_statements", isin, doc_id, limit=8)
    gf3 = _doc_rows(tables, "gf3_operational_visibility", isin, doc_id, limit=6)
    gf4 = _doc_rows(tables, "gf4_quality_flags", isin, doc_id, limit=6)
    cred = _doc_rows(tables, "mgmt_credibility", isin, doc_id, limit=8)

    _has_narr = bool(narrative if isinstance(narrative, str)
                     else "".join(t for _h, t in (narrative or [])))
    if not _has_narr and guid.empty and gf1.empty and facts.empty:
        # Nothing readable yet. Returning "" leaves the document UNMAILED and therefore
        # still due, so the next run picks it up once extraction lands - rather than
        # sending an empty mail and burning the only chance to report it.
        return ""

    out = [f"<div style='{_WRAP}'>",
           f"<h2 style='margin:0 0 2px'>&#127897; {_esc(name, 70)} "
           f"<span style='color:#888;font-weight:400'>&middot; "
           f"{_esc(symbol, 20)}</span></h2>",
           f"<div style='color:#888;font-size:12px;margin:0 0 10px'>Concall transcript"
           + (f" &middot; {_esc(period, 20)}" if str(period or "").strip() else "")
           + "</div>"]

    if not facts.empty:
        f0 = facts.iloc[0]
        out.append(_tile_row([
            ("Revenue", _num(f0.get("revenue_q")), True),
            ("EBITDA", _num(f0.get("ebitda_q")), True),
            ("PAT", _num(f0.get("pat_q")), True),
            ("Margin", _num(f0.get("margin_pct"), "%"), True),
        ]))

    out.append(_narrative_html(
        narrative, "The call, as Phase 2 read it",
        "Phase 2's full transcript brief for this document, tables and all."))

    if not guid.empty:
        rows = []
        for _, r in guid.iterrows():
            if _is_esg(r.get("metric"), r.get("notes")):
                continue
            val = _esc(r.get("value"), 40)
            if not val:
                continue
            rows.append([_esc(r.get("metric"), 40), f"<b>{val}</b>",
                         _esc(r.get("horizon_fy"), 14), _esc(r.get("notes"), 150)])
        if rows:
            out.append(_h("Forward guidance",
                          "What the company committed to on this call.")
                       + _rows_html(rows, ["Metric", "Guided", "By", "Note"]))

    if not gf1.empty:
        rows = []
        for _, r in gf1.iterrows():
            stmt = _esc(r.get("exact_statement"), 300)
            if not stmt:
                continue
            rows.append([stmt, _esc(r.get("timeframe"), 18),
                         _esc(r.get("explicitness_type"), 18)])
        if rows:
            out.append(_h("In their own words",
                          "Verbatim forward-looking statements, quoted at extraction "
                          "so nothing is paraphrased into a promise.")
                       + _rows_html(rows, ["Statement", "Timeframe", "Type"]))

    if not gf3.empty:
        rows = []
        for _, r in gf3.iterrows():
            drv = _esc(r.get("visibility_driver"), 70)
            if not drv:
                continue
            rows.append([drv, _esc(r.get("timeframe"), 16),
                         _esc(r.get("commentary"), 190)])
        if rows:
            out.append(_h("Operational visibility",
                          "The concrete things behind the guidance - orders, capacity, "
                          "contracted volume.")
                       + _rows_html(rows, ["Driver", "Horizon", "Detail"]))

    if not cred.empty:
        rows = []
        for _, r in cred.iterrows():
            metric = _esc(r.get("metric"), 34)
            if not metric:
                continue
            verdict = str(r.get("verdict") or "").strip()
            vcol = (UP if "met" in verdict.lower() or "beat" in verdict.lower()
                    else DOWN if "miss" in verdict.lower() else MUTED)
            rows.append([_esc(r.get("qtr_guided"), 12), metric,
                         _esc(r.get("guidance_given"), 40),
                         _esc(r.get("actual_delivered"), 40),
                         f"<span style='color:{vcol};font-weight:700'>"
                         f"{_esc(verdict, 18) or '&mdash;'}</span>"])
        if rows:
            sub = "Earlier guidance set against what was actually delivered."
            score = _num((cred.iloc[0]).get("cred_score"))
            if score:
                sub += f" Credibility score: <b>{_esc(score, 10)}</b>."
            out.append(_h("Said versus delivered", sub)
                       + _rows_html(rows, ["Guided in", "Metric", "Said",
                                           "Delivered", "Verdict"]))

    if not gf4.empty:
        rows = []
        for _, r in gf4.iterrows():
            ev = _esc(r.get("evidence"), 240)
            if not ev:
                continue
            rows.append([f"<span style='color:{DOWN};font-weight:700'>"
                         f"{_esc(r.get('flag_type'), 34)}</span>", ev])
        if rows:
            out.append(_h("Quality flags",
                          "Things the extraction marked as worth a second look - "
                          "evasion, an unexplained change, a claim without a number.")
                       + _rows_html(rows, ["Flag", "Evidence"]))

    out.append("</div>")
    return "".join(out)


# ------------------------------------------------------------------ #
#  Annual report: the year, its promises and its red flags            #
# ------------------------------------------------------------------ #

_SEVERITY_COLOUR = {"high": DOWN, "medium": AMBER, "low": MUTED}

# Filings that reach the queue as doc_type="annual_report" but are NOT the annual report.
# Bluspring's "Shareholder Meeting / Postal Ballot-Scrutinizer's Report" was extracted and
# would have been mailed under an "Annual Report FY2025-26" heading, summarising which
# resolutions passed. The row comes from Phase-2's own Screener feed (source="live"), so
# this is filtered at RENDER time only - exactly as _is_esg filters ESG rows out of the
# guidance tables. Nothing upstream changes and the rows stay in the queue.
_NOT_AN_AR = ("postal ballot", "scrutinizer", "scrutiniser", "e-voting", "evoting",
              "voting result", "newspaper publication", "business responsibility",
              "brsr")


def is_annual_report(title: str) -> bool:
    """False when the title plainly identifies a filing that is not the annual report.

    "annual report" in the title is the strongest positive signal there is and always
    wins - "Weblink / Exact Path Of Integrated Annual Report 2025-26" is how several
    companies file the real thing, and must not be excluded for saying "weblink".
    """
    t = str(title or "").lower()
    if "annual report" in t:
        return True
    return not any(k in t for k in _NOT_AN_AR)


def annual_report_body(isin, symbol, name, fy_label, doc_id, narrative, tables) -> str:
    """The forensic read of the annual report Phase 2 already produced."""
    facts = _doc_rows(tables, "quarterly_facts", isin, doc_id, limit=1)
    guid = _doc_rows(tables, "ar_guidance", isin, doc_id, limit=12)
    flags = _doc_rows(tables, "ar_red_flags", isin, doc_id, limit=14)

    _has_narr = bool(narrative if isinstance(narrative, str)
                     else "".join(t for _h, t in (narrative or [])))
    if not _has_narr and guid.empty and flags.empty:
        return ""                     # see concall_body: stays due, retried next run

    out = [f"<div style='{_WRAP}'>",
           f"<h2 style='margin:0 0 2px'>&#128215; {_esc(name, 70)} "
           f"<span style='color:#888;font-weight:400'>&middot; "
           f"{_esc(symbol, 20)}</span></h2>",
           f"<div style='color:#888;font-size:12px;margin:0 0 10px'>Annual Report"
           + (f" &middot; {_esc(fy_label, 40)}" if str(fy_label or "").strip() else "")
           + "</div>"]

    _sev = (flags["severity"].astype(str).str.lower() if not flags.empty
            and "severity" in flags.columns else None)
    if not facts.empty or _sev is not None:
        pairs = []
        if not facts.empty:
            f0 = facts.iloc[0]
            pairs += [("Revenue (12m)", _num(f0.get("revenue_12m")), True),
                      ("PAT (12m)", _num(f0.get("pat_12m")), True)]
        if _sev is not None:
            pairs += [("High-severity flags", str(int((_sev == "high").sum())) or "",
                       False),
                      ("Flags in total", str(len(flags)), False)]
        out.append(_tile_row(pairs))

    out.append(_narrative_html(
        narrative, "The report, as Phase 2 read it",
        "Phase 2's full forensic analysis of this annual report, tables and all."))

    if not guid.empty:
        rows = []
        for _, r in guid.iterrows():
            if _is_esg(r.get("metric"), r.get("notes")):
                continue
            val = _esc(r.get("value"), 40)
            if not val:
                continue
            rows.append([_esc(r.get("metric"), 40), f"<b>{val}</b>",
                         _esc(r.get("horizon_fy"), 14), _esc(r.get("notes"), 150)])
        if rows:
            out.append(_h("What the report commits to",
                          "Management's own forward statements, from the report itself.")
                       + _rows_html(rows, ["Metric", "Stated", "By", "Note"]))

    if not flags.empty:
        order = {"high": 0, "medium": 1, "low": 2}
        fl = flags.copy()
        fl["_o"] = (fl["severity"].astype(str).str.lower().map(order).fillna(3)
                    if "severity" in fl.columns else 3)
        rows = []
        for _, r in fl.sort_values("_o").iterrows():
            ev = _esc(r.get("evidence"), 260)
            if not ev:
                continue
            sev = str(r.get("severity") or "").strip().lower()
            col = _SEVERITY_COLOUR.get(sev, MUTED)
            rows.append([f"<span style='color:{col};font-weight:700'>"
                         f"{_esc(sev or 'flag', 10).upper()}</span>",
                         _esc(r.get("category"), 26),
                         _esc(r.get("flag_type"), 34), ev,
                         _esc(r.get("page_ref"), 12)])
        if rows:
            out.append(_h("Red flags",
                          "Raised by the forensic pass over the report - each one "
                          "carries the evidence it was raised from, so it can be "
                          "checked rather than taken on trust.")
                       + _rows_html(rows, ["Severity", "Area", "Flag",
                                           "Evidence", "Page"]))

    out.append("</div>")
    return "".join(out)


def _narratives(drive, repo_id, latest: dict, cache: dict, index_id: str = "") -> dict:
    """{(isin, doc_type): {'period':.., 'text':..}} for the newest concall / AR per holding.

    Reuses the digest's parser rather than carrying a second copy: run_pf_docs_digest
    already solves boundary detection on a page whose LLM bodies contain their own '## '
    headings, and a divergent copy here would drift away from it.
    """
    from run_pf_docs_digest import _company_page, _find_region, _lift_summary
    from _extractor_base import log as _log, load_doc_report, is_prompt_echo
    out = {}
    for (isin, dt), d in latest.items():
        if dt not in SCOPED_TYPES:
            continue
        period = str(d.get("period") or "").strip()
        doc_id = str(d.get("doc_id") or "").strip()
        if not period and dt == "annual_report":
            # MOST AR QUEUE ROWS CARRY NO PERIOD - measured on APL Apollo, the field is
            # empty. Without one, the only way left to find the section is the doc
            # marker, and after a re-extraction the marker sits on the STALE section
            # while the good re-read is appended under a correct heading with none. So
            # the fresh report was invisible and the mail lost its summary again.
            # announcement_date always answers this (FY-end stamp or filing date).
            _fy = ar_fy_year(d.get("date"), "")
            if _fy:
                period = f"FY{str(_fy)[-2:]}"
        txt = ""
        # EXACT FIRST. doc_reports is keyed on the document, so there is nothing to
        # guess; the page walk below stays as the fallback for every document extracted
        # before this store existed.
        try:
            exact = load_doc_report(drive, index_id, doc_id) if doc_id else ""
        except Exception:
            exact = ""
        if exact.strip():
            out[(isin, dt)] = {"period": period, "text": exact}
            continue
        try:
            reg = _find_region(_company_page(drive, repo_id, isin, cache), period, dt,
                               doc_id)
            # The whole region, headings and tables intact. lift_report stays for
            # anything that wants a short extract; the mail wants the report.
            txt = "\n".join(((h + "\n") if h else "") + b
                            for h, b in (reg or [])) if reg else ""
        except Exception as e:
            _log(f"  WARN: narrative lift failed for {isin} {dt} ({str(e)[:60]})")
        # A PROMPT ECHO IS NOT A REPORT. Switching from lift_report() to carrying the
        # whole report removed the echo filter that lift_report applied per section, so
        # a response that is annual_report_prompt.txt talking back would be rendered in
        # full - measured 2026-09-04, that is 5k / 89k / 120k characters of instructions
        # for DEEPINDS, CGPOWER and MOREPENLAB. Treat it as NO narrative: the mail then
        # carries its structured tables, or does not send at all and stays due, and the
        # coarse content_key flips to "0" so a good re-extraction re-notifies.
        if txt and is_prompt_echo(txt):
            _log(f"  {dt} for {isin}: stored report is a prompt echo — treated as empty")
            txt = ""
        out[(isin, dt)] = {"period": period, "text": txt}
    return out


# ------------------------------------------------------------------ #
#  Main                                                               #
# ------------------------------------------------------------------ #

def _read(drive, idx, name):
    from _extractor_base import find_file, download_bytes
    try:
        fid = find_file(drive, idx, name)
        if not fid:
            return pd.DataFrame()
        return pd.read_parquet(io.BytesIO(download_bytes(drive, fid)))
    except Exception:
        return pd.DataFrame()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true",
                    help="Write previews, send nothing, do not advance the ledger.")
    ap.add_argument("--types", default="results,presentation,rating",
                    help="Which mails to consider.")
    ap.add_argument("--limit", type=int, default=0, help="Max mails this run.")
    ap.add_argument("--symbols", default="",
                    help="Restrict to these NSE symbols (comma separated). --limit caps "
                         "a count but cannot choose WHICH holding, so this is what lets "
                         "one real mail be sent and read before the whole book goes out.")
    ap.add_argument("--window-days", type=int, default=2,
                    help="How far back a calendar date still counts (backwards only).")
    ap.add_argument("--require-calendar", action="store_true",
                    help="Results mails additionally require a board-meeting date "
                         "on/near today. Presentations and ratings never do — they "
                         "arrive on no calendar.")
    ap.add_argument("--scope", default="current",
                    help="Concall/AR ONLY. 'current' (default) mails this season's "
                         "concalls (the quarter the presentation mail already uses) and "
                         "this financial year's annual reports. 'all' disables both "
                         "scopes and mails the latest of each, whenever it was filed.")
    ap.add_argument("--new-holding-days", type=int, default=30,
                    help="A holding that entered the portfolio within this many days is "
                         "ONBOARDED: its latest concall and latest annual report are "
                         "mailed whatever month they were filed, once. 0 disables it, "
                         "leaving every holding on the --month rule.")
    ap.add_argument("--force", action="store_true",
                    help="Re-send the named holdings even though the ledger says they "
                         "were already mailed. REQUIRES --symbols, so it can never "
                         "re-send the whole book by accident. For when a fix changed "
                         "what a mail SAYS without changing its content_key - the key is "
                         "deliberately coarse and cannot see prose improving.")
    ap.add_argument("--seed-ledger", action="store_true",
                    help="Write ledger rows for everything currently due WITHOUT "
                         "sending, so only genuinely-new documents mail from then on. "
                         "One-off, run once when enabling a new doc_type.")
    ap.add_argument("--out-dir", default=".")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    if args.self_test:
        sys.exit(_self_test())

    from _extractor_base import (get_drive, get_or_create_subfolder, load_queue,
                                 load_parquet, save_parquet, log)
    from daily_brief import load_pf

    drive = get_drive()
    root = os.environ["GDRIVE_FOLDER_ID"]
    repo = get_or_create_subfolder(drive, root, "company_repo")
    idx = get_or_create_subfolder(drive, repo, "_index")

    season = QT.season_quarter()
    today = date.today()
    pf = load_pf(drive, root, idx)
    queue = load_queue(drive, idx)
    calendar = _read(drive, idx, "results_calendar.parquet")
    ledger = load_parquet(drive, idx, LEDGER_NAME, LEDGER_COLS)
    want = {t.strip() for t in args.types.split(",") if t.strip()}

    # The same tables the renderers read, so "due" means "will actually send" —
    # coverage tested only that an extractor had run, which reported 16 presentations
    # as due and then skipped every one as "nothing renderable".
    # deck_summary/ppt_guidance join the pre-load because content_key needs them BEFORE
    # mail_due decides what is due; they are small and read-only.
    _pre = {n: _read(drive, idx, f"{n}.parquet") for n in
            ("ppt_highlights", "ratings", "deck_summary", "ppt_guidance")}
    # What the mail WOULD assert right now, per document — compared against what it
    # asserted when it was last sent, so a corrected re-extract is re-notified.
    _latest = COV.latest_doc_per_type(pf, queue, tuple(want))
    # Concall/AR prose exists ONLY on company_page.md, so it is lifted once per holding
    # (cached) and shared by content_key and the renderers. Read only when those types
    # are actually wanted - otherwise this is six needless Drive round-trips.
    _extra = {}
    if set(want) & set(SCOPED_TYPES):
        _extra = {n: _read(drive, idx, f"{n}.parquet") for n in
                  ("quarterly_facts", "guidance_tracker", "gf1_guidance_statements",
                   "gf3_operational_visibility", "ar_guidance", "ar_red_flags")}
        _pre.update(_extra)
        _pre["_narr"] = _narratives(drive, repo, _latest, {}, idx)
        _n_ok = sum(1 for v in _pre["_narr"].values() if v["text"])
        log(f"narratives lifted from company_page.md: {_n_ok}/{len(_pre['_narr'])}")
    _keys = {}
    for (_i, _dt), _d in _latest.items():
        _did = str(_d.get("doc_id") or "").strip()
        if _did:
            _k = content_key(_dt, _i, _pre)
            if _k:
                _keys[_did] = _k
    due = COV.mail_due(pf, queue, calendar, ledger, season, on=today,
                       window_days=args.window_days, doc_types=tuple(want),
                       require_calendar=args.require_calendar, tables=_pre,
                       content_keys=_keys)
    # MONTH SCOPE. "The latest concall and AR" is what the exchanges received THIS
    # calendar month. This is also what stops a first run mailing the back catalogue:
    # coverage() calls a holding PRESENT when ANY document of that type ever reached
    # done, so without a scope every holding's whole history is due at once.
    # TRACK 2 - holdings that just entered the portfolio. Read before the scope is
    # applied, because these bypass it.
    fresh = set()
    if args.new_holding_days > 0:
        try:
            _tid = get_or_create_subfolder(drive, idx, "pf_tracker")
            _snaps = load_parquet(drive, _tid, "pf_snapshots.parquet",
                                  ["snapshot_date", "isin", "symbol", "name",
                                   "weight_pct", "source_file"])
            fresh = new_pf_holdings(_snaps, args.new_holding_days, today)
            log(f"new to the portfolio in {args.new_holding_days}d: {len(fresh)} holding(s)"
                + (" - onboarded with their latest concall/AR" if fresh else ""))
        except Exception as e:
            # No holdings history is not a failure: everything simply stays on the
            # month rule, which is the stricter of the two.
            log(f"pf_snapshots unavailable ({str(e)[:60]}) - month rule only")

    # EACH TYPE ON ITS OWN CALENDAR. A concall belongs to a QUARTER - so it is scoped
    # exactly as the presentation mail is, to the current season. An annual report
    # belongs to a FINANCIAL YEAR and is filed on a statutory timetable (AGM within six
    # months of the year end), so the whole FY2026 crop arrives across Jun-Sep 2026 and a
    # month window would arbitrarily split it.
    season_q, ar_fy = QT.norm_q(season), current_ar_fy(today)
    if (args.scope or "").strip().lower() != "all":
        kept, out_of_scope, undated, onboarded = [], 0, 0, 0
        for d in due:
            if d["doc_type"] not in SCOPED_TYPES:
                kept.append(d)
                continue
            # TRACK 2 wins over both scopes: a holding just bought needs its LATEST call
            # and report whenever they were filed, and mail_due hands us exactly those.
            if str(d.get("isin", "")).strip() in fresh:
                d["onboarding"] = True
                kept.append(d)
                onboarded += 1
                continue
            # NOT `want`: that name already holds the set of doc types this run was
            # asked for, and rebinding it here silently emptied every later test
            # against it - --force found nothing, because "annual_report" is not a
            # substring of "Q1FY27".
            if d["doc_type"] == "concall":
                got, expect = concall_quarter(d.get("doc_date")), season_q
            else:
                got, expect = ar_fy_year(d.get("doc_date"), d.get("period")), ar_fy
            if not got:
                # Undatable. Skipped rather than assumed current, and COUNTED so that it
                # is visible rather than silent.
                undated += 1
            elif got == expect:
                kept.append(d)
            else:
                out_of_scope += 1
        n_scoped = sum(1 for k in kept
                       if k["doc_type"] in SCOPED_TYPES and not k.get("onboarding"))
        log(f"scope: concall={season_q} AR=FY{ar_fy} -> {n_scoped} in scope"
            + (f", {onboarded} onboarding a new holding" if onboarded else "")
            + (f", {out_of_scope} from an earlier quarter/FY" if out_of_scope else "")
            + (f", {undated} undatable" if undated else ""))
        due = kept

    log(f"season={season} pf={len(pf)} due={len(due)} "
        f"({', '.join(sorted({d['doc_type'] for d in due})) or 'nothing'})")
    if not due:
        log("nothing due — no mail.")
        return

    if args.force:
        # Rebuild the due list straight from the newest document per type, bypassing both
        # the ledger and the scope. mail_due exists to answer "what is new"; force
        # answers a different question - "send me these again, they are better now".
        if not args.symbols:
            log("--force requires --symbols — refusing to re-send the whole portfolio.")
            return
        _wsym = {s.strip().upper() for s in args.symbols.split(",") if s.strip()}
        _by_isin = {str(i): (s, n) for i, s, n in pf}
        due = []
        for (_i, _dt), _d in sorted(_latest.items()):
            _sym, _name = _by_isin.get(str(_i), ("", ""))
            if _dt not in want or _sym.upper() not in _wsym:
                continue
            due.append({"isin": _i, "symbol": _sym, "name": _name, "doc_type": _dt,
                        "season": season, "resend": True,
                        "doc_id": _d.get("doc_id", ""), "doc_date": _d.get("date", ""),
                        "doc_title": _d.get("title", ""), "period": _d.get("period", ""),
                        "discovered_at": _d.get("discovered_at", ""),
                        "processed_at": _d.get("processed_at", ""), "reported_on": ""})
        due.sort(key=lambda d: (d["doc_type"], d["symbol"]))
        log(f"--force: {len(due)} document(s) for {sorted(_wsym)} — ledger and scope "
            f"bypassed")
        if not due:
            log("nothing to force — no document of those types for those symbols.")
            return

    if args.seed_ledger:
        # Mark the current back catalogue as already handled WITHOUT sending, so the
        # first real run reports only genuinely-new documents.
        for d in due[:40]:
            log(f"  seed {d['symbol']:<12} {d['doc_type']}")
        if args.dry_run:
            log(f"DRY RUN - would seed {len(due)} ledger row(s); nothing written.")
            return
        _now = datetime.now().isoformat(timespec="seconds")
        seed_rows = [{"season": season, "isin": d["isin"], "symbol": d["symbol"],
                      "doc_type": d["doc_type"], "period": season,
                      "doc_id": d.get("doc_id", ""), "mailed_at": _now,
                      "subject": "(seeded - back catalogue, never sent)",
                      "content_key": _keys.get(str(d.get("doc_id") or ""), "")}
                     for d in due]
        out = pd.concat([ledger, pd.DataFrame(seed_rows, columns=LEDGER_COLS)],
                        ignore_index=True) if ledger is not None and not ledger.empty \
            else pd.DataFrame(seed_rows, columns=LEDGER_COLS)
        out = out.drop_duplicates(subset=["season", "isin", "doc_type", "doc_id"],
                                  keep="last")
        save_parquet(drive, idx, LEDGER_NAME, out)
        log(f"SEEDED {len(seed_rows)} row(s) -> _index/{LEDGER_NAME} ({len(out)} rows). "
            f"Nothing mailed; only new documents will mail from now on.")
        return

    tables = {n: _read(drive, idx, f"{n}.parquet") for n in
              ("ppt_highlights", "ppt_guidance", "deck_summary", "deck_metrics", "deck_diff",
               "ratings", "rating_concerns", "rating_sensitivity",
               # positives + promise-tracking: all already computed, never surfaced
               "rating_drivers", "guidance_vs_actual", "mgmt_credibility",
               "gf2_historical_guidance",
               # risks + management changes: the deck does not carry these
               "deck_flags", "gf4_quality_flags", "announcement_ledger")}
    # Already read above for content_key - carried over rather than re-fetched.
    tables.update(_extra)
    if "_narr" in _pre:
        tables["_narr"] = _pre["_narr"]

    if args.symbols:
        want_sym = {s.strip().upper() for s in args.symbols.split(",") if s.strip()}
        before = len(due)
        due = [d for d in due if str(d.get("symbol", "")).upper() in want_sym]
        log(f"--symbols {sorted(want_sym)}: {len(due)} of {before} due")
        if not due:
            log("nothing due for those symbols — no mail.")
            return
    if args.limit:
        due = due[: args.limit]

    sent_rows = []
    for d in due:
        isin, sym, name, dt = d["isin"], d["symbol"], d["name"], d["doc_type"]
        _attach = None
        if dt == "presentation":
            body = presentation_body(isin, sym, name, season, tables)
            subject = (f"📊 {sym} — investor presentation UPDATED, "
                       f"{QT.qtr_label(season)}" if d.get("resend")
                       else f"📊 {sym} — investor presentation, {QT.qtr_label(season)}")
        elif dt == "rating":
            body = rating_body(isin, sym, name, tables)
            subject = (f"🏷 {sym} — credit rating CORRECTED"
                       if d.get("resend") else f"🏷 {sym} — credit rating update")
        elif dt == "concall":
            _n = (tables.get("_narr") or {}).get((isin, "concall")) or {}
            _per = _n.get("period") or d.get("period") or QT.qtr_label(season)
            body = concall_body(isin, sym, name, _per, d.get("doc_id", ""),
                                _n.get("text") or [], tables)
            # A new holding's mail is its LATEST call, not this month's news, and
            # saying so stops it reading as a filing that just happened.
            _attach = report_attachment(sym, f"Concall {_per}".strip(),
                                        f"{name} - Phase 2 transcript brief",
                                        _n.get("text") or [])
            _new = " (new holding \u2014 latest call)" if d.get("onboarding") else ""
            # A RESEND MUST NOT SHARE A SUBJECT WITH THE MAIL IT REPLACES. Gmail threads
            # on subject and hides the repeated body behind a "..." trim, which reads as
            # a truncated mail. The send date makes each one its own conversation.
            _stamp = today.strftime("%-d %b") if os.name != "nt" else today.strftime("%d %b").lstrip("0")
            subject = (f"\U0001F399 {sym} \u2014 concall transcript UPDATED "
                       f"{_stamp}, {_esc(_per, 20)}" if d.get("resend")
                       else f"\U0001F399 {sym} \u2014 concall transcript, "
                            f"{_esc(_per, 20)}{_new}")
        elif dt == "annual_report":
            if not is_annual_report(d.get("doc_title")):
                log(f"  {sym:<12} {dt}: not an annual report "
                    f"({_esc(d.get('doc_title'), 60)}) - skipped")
                continue
            from run_pf_docs_digest import _ar_display
            _n = (tables.get("_narr") or {}).get((isin, "annual_report")) or {}
            # An AR's announcement_date is the FY-END (2026-03-31 = the 2025-26 report),
            # so the label is derived from it rather than printed as a bare FY.
            _fy = _ar_display(d.get("doc_date", "")) or str(d.get("period") or "")
            body = annual_report_body(isin, sym, name, _fy, d.get("doc_id", ""),
                                      _n.get("text") or [], tables)
            _attach = report_attachment(sym, f"Annual Report {_fy}".strip(),
                                        f"{name} - Phase 2 forensic analysis",
                                        _n.get("text") or [])
            _new = " (new holding \u2014 latest report)" if d.get("onboarding") else ""
            _stamp = today.strftime("%d %b").lstrip("0")
            subject = (f"\U0001F4D7 {sym} \u2014 Annual Report {_esc(_fy, 40)} "
                       f"UPDATED {_stamp}" if d.get("resend")
                       else f"\U0001F4D7 {sym} \u2014 Annual Report "
                            f"{_esc(_fy, 40)}{_new}")
        else:
            body = ""       # results teardown is rendered by quarter_teardown itself
            subject = ""
        if not body:
            log(f"  {sym:<12} {dt}: nothing renderable — skipped")
            continue
        if len(body.encode()) > MAX_HTML_BYTES:
            log(f"  {sym:<12} {dt}: {len(body.encode()):,} B over budget — trimmed")
            body = body[:MAX_HTML_BYTES]

        if args.dry_run:
            p = os.path.join(args.out_dir, f"mail_{sym}_{dt}.html")
            with open(p, "w", encoding="utf-8") as fh:
                fh.write(body)
            _extra = ""
            if _attach:
                ap = os.path.join(args.out_dir, "attach_" + _attach[0])
                with open(ap, "wb") as fh:
                    fh.write(_attach[1])
                _extra = f" + attachment {_attach[0]} ({len(_attach[1]):,} B)"
            log(f"  {sym:<12} {dt}: preview -> {p} "
                f"({len(body.encode()):,} B){_extra}")
            continue

        from mailer import send_email, load_mail_settings
        if not load_mail_settings(drive, idx).get(MAIL_KEY, True):
            log(f"  mail toggle '{MAIL_KEY}' is OFF — stopping.")
            return
        if not (os.getenv("GMAIL_USER") and os.getenv("GMAIL_APP_PASSWORD")
                and os.getenv("NOTIFY_EMAIL")):
            log("  mail NOT sent — GMAIL_* not set in this environment.")
            return
        ok = send_email(subject, body,
                        attachments=[_attach] if _attach else None)
        log(f"  {sym:<12} {dt}: sent={ok}")
        if ok:
            sent_rows.append({"season": season, "isin": isin, "symbol": sym,
                              "doc_type": dt, "period": season,
                              "doc_id": d.get("doc_id", ""),
                              "mailed_at": datetime.now().isoformat(timespec="seconds"),
                              "subject": subject[:200],
                              "content_key": _keys.get(str(d.get("doc_id") or ""), "")})

    # Ledger advances ONLY on confirmed sends — a failed mail must stay due.
    if sent_rows:
        out = pd.concat([ledger, pd.DataFrame(sent_rows, columns=LEDGER_COLS)],
                        ignore_index=True) if ledger is not None and not ledger.empty \
            else pd.DataFrame(sent_rows, columns=LEDGER_COLS)
        # Keyed on the DOCUMENT: a second deck or a rating downgrade in the same quarter
        # is a separate row, so it neither overwrites the earlier send nor gets suppressed.
        out = out.drop_duplicates(subset=["season", "isin", "doc_type", "doc_id"],
                                  keep="last")
        save_parquet(drive, idx, LEDGER_NAME, out)
        log(f"ledger: +{len(sent_rows)} -> _index/{LEDGER_NAME} ({len(out)} rows)")


def _self_test() -> int:
    ok = fail = 0

    def check(name, cond):
        nonlocal ok, fail
        if cond:
            ok += 1
        else:
            fail += 1
            print(f"  FAIL {name}")

    T = {"ppt_highlights": pd.DataFrame([
            {"isin": "INE1", "quarter": "Q1 FY27", "category": "demand",
             "statement": "Order book at a record", "value": "1200", "unit": "Cr"}]),
         "ppt_guidance": pd.DataFrame([
            {"isin": "INE1", "quarter": "Q1 FY27", "metric": "revenue growth",
             "value": "15", "unit": "%", "horizon": "FY27"}]),
         "deck_metrics": pd.DataFrame([
            {"isin": "INE1", "quarter": "Q4 FY26", "metric": "utilisation"},
            {"isin": "INE1", "quarter": "Q4 FY26", "metric": "order book"},
            {"isin": "INE1", "quarter": "Q1 FY27", "metric": "order book"}]),
         "deck_diff": pd.DataFrame(),
         "ratings": pd.DataFrame([
            {"isin": "INE1", "agency": "CRISIL", "rating": "A-", "outlook": "Stable",
             "rating_action": "Reaffirmed", "rating_date": "2025-06-01",
             "instrument_type": "LT", "rated_amount_cr": "500"},
            {"isin": "INE1", "agency": "CRISIL", "rating": "A", "outlook": "Positive",
             "rating_action": "Upgrade", "rating_date": "2026-06-01",
             "instrument_type": "LT", "rated_amount_cr": "600"}]),
         "rating_concerns": pd.DataFrame([
            {"isin": "INE1", "concern": "Working capital intensity remains high"}]),
         "rating_sensitivity": pd.DataFrame([
            {"isin": "INE1", "direction": "down", "trigger": "Debt/EBITDA above 3.5x"},
            {"isin": "INE1", "direction": "up", "trigger": "Sustained margin above 15%"}]),
         }

    p = presentation_body("INE1", "AAA", "A Ltd", "Q1FY27", T)
    check("deck: current highlight rendered", "Order book at a record" in p)
    check("deck: guidance rendered", "revenue growth" in p)
    # THE DIFF, computed deterministically when the LLM table is empty: 'utilisation'
    # was shown in Q4 FY26 and is absent in Q1 FY27.
    check("deck: dropped KPI detected without an LLM", "utilisation" in p)
    check("deck: drop is labelled as such", "stopped reporting" in p)

    one_q = dict(T, deck_metrics=pd.DataFrame([
        {"isin": "INE1", "quarter": "Q1 FY27", "metric": "order book"}]))
    p1 = presentation_body("INE1", "AAA", "A Ltd", "Q1FY27", one_q)
    check("deck: a single quarter says so, not 'no changes'",
          "no comparison is possible" in p1)
    check("deck: single quarter is not an all-clear", "stopped reporting" not in p1)

    r = rating_body("INE1", "AAA", "A Ltd", T)
    check("rating: current rating shown", ">A<" in r or "A</b>" in r)
    check("rating: upgrade flagged", "Upgrade" in r)
    check("rating: what-changed section present", "What changed since" in r)
    check("rating: prior rating shown as 'was'", "A-" in r)
    check("rating: downgrade trigger surfaced", "Debt/EBITDA above 3.5x" in r)
    # Up-triggers must NOT sit under the downgrade heading, but they SHOULD now have
    # their own — the agency's upgrade conditions were computed and never shown.
    _dn = r.split("What would cause a downgrade")[-1].split("What would earn an upgrade")[0]
    check("rating: up-trigger not filed under downgrade",
          "Sustained margin above 15%" not in _dn)
    check("rating: upgrade triggers get their own section",
          "What would earn an upgrade" in r and "Sustained margin above 15%" in r)
    check("rating: concern surfaced", "Working capital intensity" in r)

    # THE RENDER DEFECTS from the first live run against APLAPOLLO.
    nan_t = dict(T, ppt_highlights=pd.DataFrame([
        {"isin": "INE1", "quarter": "Q1 FY27", "category": "demand",
         "statement": "HR Coil tubes to grow faster", "value": float("nan"), "unit": None}]))
    pn = presentation_body("INE1", "AAA", "A Ltd", "Q1FY27", nan_t)
    # Substring "nan" also occurs inside "financial", so test the DEFECT: a NaN value
    # or None unit reaching the page as literal text.
    check("NaN value never renders as 'nanNone'",
          "nanNone" not in pn and ">nan<" not in pn and ">None<" not in pn
          and "nan</b>" not in pn)

    esg_t = dict(T, ppt_guidance=pd.DataFrame([
        {"isin": "INE1", "quarter": "Q1 FY27", "metric": "Scope 1 & 2 emissions",
         "value": "-25", "unit": "%", "horizon": "2030"},
        {"isin": "INE1", "quarter": "Q1 FY27", "metric": "Female workforce",
         "value": "1", "unit": "%", "horizon": "yearly"}]))
    pe = presentation_body("INE1", "AAA", "A Ltd", "Q1FY27", esg_t)
    check("ESG targets are not presented as guidance", "Scope 1" not in pe)
    check("ESG-only decks say so explicitly", "ESG commitments only" in pe)
    check("real guidance still shows", "revenue growth" in
          presentation_body("INE1", "AAA", "A Ltd", "Q1FY27", T))

    # POSITIVES: a rating mail listing only concerns misreads a favourable rationale.
    T_pos = dict(T, rating_drivers=pd.DataFrame([
        {"isin": "INE1", "agency": "CRISIL", "rating_date": "2026-06-01",
         "driver": "Healthy order book providing revenue visibility"},
        {"isin": "INE1", "agency": "CRISIL", "rating_date": "2026-06-01",
         "driver": "Established market position"}]))
    rp = rating_body("INE1", "AAA", "A Ltd", T_pos)
    check("rating: strengths are surfaced", "Healthy order book" in rp)
    check("rating: strengths get their own heading", "says is working" in rp)

    # PROMISED VS DELIVERED — the half that matters more than a dropped KPI.
    T_gva = dict(T, guidance_vs_actual=pd.DataFrame([
        {"isin": "INE1", "period": "Q4FY26", "metric": "EBITDA margin",
         "guided": "18%", "actual": "16.1%", "delta": "-1.9", "verdict": "MISS"}]),
        mgmt_credibility=pd.DataFrame([
            {"isin": "INE1", "quarter": "Q1 FY27", "pattern": "Optimistic Bias",
             "recurring_miss": "margin", "strongest_area": "revenue"}]))
    pg = presentation_body("INE1", "AAA", "A Ltd", "Q1FY27", T_gva)
    check("promise vs outcome is shown", "Promised vs delivered" in pg)
    check("the missed promise is named", "EBITDA margin" in pg)
    check("the verdict is carried", "MISS" in pg)
    check("commentary consistency is reported", "Optimistic Bias" in pg)
    check("a recurring miss is called out", "Recurring miss" in pg)

    # COLOUR CARRIES MEANING, and only for real numbers.
    check("a gain is green", UP in colour_num("+12.5%"))
    check("a loss is red", DOWN in colour_num("-3.2%"))
    check("lower-is-better inverts", UP in colour_num("-8", good_up=False))
    check("text is not force-coloured", colour_num("n/a") == "n/a")
    check("empty stays empty", colour_num(None) == "")

    # KEY CALL-OUTS: numbers in thesis-moving categories, lifted to the top.
    T_kc = dict(T, ppt_highlights=pd.DataFrame([
        {"isin": "INE1", "quarter": "Q1 FY27", "category": "orderbook",
         "statement": "Order book at record", "value": "1200", "unit": "Cr"},
        {"isin": "INE1", "quarter": "Q1 FY27", "category": "other",
         "statement": "No number here", "value": "", "unit": ""}]))
    kc = presentation_body("INE1", "AAA", "A Ltd", "Q1FY27", T_kc)
    check("call-outs lift the numeric highlight", "Key call-outs" in kc)
    check("the number is featured", "1200" in kc)
    check("a statement with no number is not a call-out",
          "No number here" not in kc.split("Key call-outs")[1].split("</table>")[0])

    # CONSISTENCY: restated / revised / stopped saying, across two quarters.
    T_cons = dict(T, ppt_highlights=pd.DataFrame([
        {"isin": "INE1", "quarter": "Q4 FY26", "category": "capacity",
         "statement": "Capacity target by FY28", "value": "8.0", "unit": "Mn"},
        {"isin": "INE1", "quarter": "Q4 FY26", "category": "demand",
         "statement": "Exports scaling strongly", "value": "", "unit": ""},
        {"isin": "INE1", "quarter": "Q1 FY27", "category": "capacity",
         "statement": "Capacity target by FY28", "value": "6.5", "unit": "Mn"},
        {"isin": "INE1", "quarter": "Q1 FY27", "category": "demand",
         "statement": "Domestic demand firm", "value": "", "unit": ""}]))
    cs = presentation_body("INE1", "AAA", "A Ltd", "Q1FY27", T_cons)
    check("consistency section renders", "Consistent, and not" in cs)
    check("a quietly revised target is flagged", "revised" in cs)
    check("both old and new values are shown", "8.0" in cs and "6.5" in cs)
    check("a dropped theme is called out", "stopped saying" in cs)
    check("the dropped statement is named", "Exports scaling strongly" in cs)

    # RATING KEY NUMBERS
    rk = rating_body("INE1", "AAA", "A Ltd", dict(T, rating_drivers=pd.DataFrame([
        {"isin": "INE1", "agency": "CRISIL", "rating_date": "2026-06-01",
         "driver": "Healthy order book"}])))
    check("rated amount is a key number", "Rated amount" in rk)
    check("strengths are counted", "Strengths cited" in rk)
    check("downgrade triggers are counted", "Downgrade triggers" in rk)

    # THEME EMPHASIS: a risk must never read green just because it sits in a deck.
    check("a risk line is themed red", theme_of("Margin pressure from input costs")[1] == DOWN)
    check("a growth line is themed green", theme_of("Order book at record high")[1] == UP)
    check("a capacity fact is themed blue",
          theme_of("Capacity at 8 Mn tonnes across 12 plants")[1] == BLUE)
    # A capacity TARGET is guidance, not a capacity fact — it is a promise to track,
    # and that is the more useful label of the two.
    check("a capacity TARGET is themed as guidance",
          theme_of("Capacity expansion by FY28")[0] == "guidance")
    check("a guidance line is themed distinctly",
          theme_of("We expect 15% growth by FY28")[1] == PURPLE)
    check("guidance beats growth when both words appear",
          theme_of("We expect strong growth")[0] == "guidance")
    check("risk beats capacity when both appear",
          theme_of("Capacity expansion delayed")[0] == "risk")
    check("an unthemed line stays neutral", theme_of("Board met on Tuesday")[1] == MUTED)

    # RISKS AND CHANGES come from real sources, not the deck.
    T_rc = dict(T, deck_flags=pd.DataFrame([
        {"isin": "INE1", "flag_type": "axis_truncated", "evidence": "Chart on slide 7"}]),
        gf4_quality_flags=pd.DataFrame([
            {"isin": "INE1", "flag_type": "Promotional Commentary",
             "evidence": "best ever quarter"}]),
        announcement_ledger=pd.DataFrame([
            {"isin": "INE1", "ann_date": "2026-08-02", "event_type": "management_change",
             "headline": "CFO resigns", "summary": "CFO stepped down"}]))
    rc = presentation_body("INE1", "AAA", "A Ltd", "Q1FY27", T_rc)
    check("framing flag surfaced", "axis truncated" in rc)
    check("commentary flag surfaced", "Promotional Commentary" in rc)
    check("management change surfaced", "CFO" in rc)
    check("their sources are named, not implied to be the deck", "Not from the deck" in rc)

    # ---- standalone deck summary: the section that carries a no-concall company
    _ds = pd.DataFrame([
        {"isin": "INE1", "quarter": "Q1FY27", "section": "financials",
         "label": "Consolidated PAT", "value": 194.0, "unit": "Rs. Mn",
         "period": "Q1FY27", "detail": "", "evidence": "194"},
        {"isin": "INE1", "quarter": "Q1FY27", "section": "financials",
         "label": "EBITDA growth", "value": 17.3, "unit": "%", "period": "Q1FY27",
         "detail": "", "evidence": "17.3%"},
        {"isin": "INE1", "quarter": "Q1FY27", "section": "segment_performance",
         "label": "HPDC revenue", "value": -41.2, "unit": "%", "period": "Q1FY27",
         "detail": "", "evidence": "-41.2%"},
        {"isin": "INE1", "quarter": "Q1FY27", "section": "risks", "label": "Concentration",
         "value": None, "unit": "", "period": "", "detail": "Top-5 customers are 46%",
         "evidence": "top 5 are 46% of revenue"},
        {"isin": "INE1", "quarter": "Q4FY26", "section": "capex", "label": "Old capex",
         "value": 500.0, "unit": "cr", "period": "FY26", "detail": "",
         "evidence": "500 cr"}])
    T_ds = dict(T, deck_summary=_ds)
    ss = standalone_summary("INE1", "Q1FY27", T_ds)
    check("standalone summary renders", "The deck on its own terms" in ss)
    check("coverage stated honestly", "3 of 18" in ss)
    check("section heading shown", "Financials, as the deck states them" in ss)
    check("unit normalised for display", "Rs mn" in ss and "Rs. Mn" not in ss)
    check("percent gets direction colour", UP in ss and DOWN in ss)
    check("absolute level is not colour-graded", f">194</b>" in ss or "194 Rs mn" in ss)
    check("risk section carries the red band", "Risks the deck admits" in ss)
    check("narrative detail shown", "Top-5 customers are 46%" in ss)
    check("other quarters excluded", "Old capex" not in ss and "Capex" not in ss)
    check("no deck_summary renders nothing", standalone_summary("INE1", "Q1FY27", T) == "")
    check("other company renders nothing", standalone_summary("INE9", "Q1FY27", T_ds) == "")
    check("summary reaches the mail body",
          "The deck on its own terms" in presentation_body("INE1", "AAA", "A", "Q1FY27", T_ds))
    check("18 sections mapped, matching the extractor",
          {s for s, _, _ in _SUMMARY_SECTIONS} == set(DS.SECTIONS))

    # A company with NO concall and NO ppt_highlights is the whole point: the old gate
    # returned "" for it and the mail this section exists for would never have gone out.
    only_ds = {k: pd.DataFrame() for k in T}
    only_ds["deck_summary"] = _ds
    check("deck-only company still gets a mail",
          "The deck on its own terms" in presentation_body("INE1", "AAA", "A", "Q1FY27", only_ds))

    check("no data renders nothing, not an empty shell",
          presentation_body("INE_NONE", "X", "X", "Q1FY27", T) == "")
    check("no ratings renders nothing", rating_body("INE_NONE", "X", "X", T) == "")

    # ---- content_key: the fingerprint that makes a correction detectable
    rt_d = pd.DataFrame([{"isin": "INE1", "agency": "CRISIL", "rating": "D",
                          "outlook": "Stable", "rating_action": "Reaffirmed",
                          "rating_date": "2025-06-10"}])
    rt_fixed = pd.DataFrame([{"isin": "INE1", "agency": "CRISIL", "rating": "BBB+",
                              "outlook": "Stable", "rating_action": "Downgrade",
                              "rating_date": "2025-06-10"}])
    k_old = content_key("rating", "INE1", {"ratings": rt_d})
    k_new = content_key("rating", "INE1", {"ratings": rt_fixed})
    check("content key is built from the data", k_old == "CRISIL|D|STABLE|REAFFIRMED")
    check("the Tatva correction changes the key", k_old != k_new)
    check("an identical re-read keeps the same key",
          content_key("rating", "INE1", {"ratings": rt_d.copy()}) == k_old)
    check("newest rating wins when several exist",
          content_key("rating", "INE1", {"ratings": pd.concat([rt_d, rt_fixed])}) == k_new)
    check("unknown company has no key", content_key("rating", "INE9", {"ratings": rt_d}) == "")
    check("no ratings table is safe", content_key("rating", "INE1", {}) == "")
    # Other doc types are deliberately NOT fingerprinted yet, so their behaviour is
    # unchanged and no unvalidated key can trigger a spurious re-send.
    # ---- deck key: coarse ON PURPOSE, because the extraction drifts
    ds_full = pd.DataFrame([{"isin": "INE1", "quarter": "Q1FY27", "section": "financials",
                             "label": "Revenue", "value": 100.0, "unit": "cr",
                             "period": "Q1FY27", "detail": "", "evidence": "100"}])
    k_deck = content_key("presentation", "INE1", {"deck_summary": ds_full})
    check("a deck with content has a key", k_deck != "")
    check("a company with no deck rows has no key",
          content_key("presentation", "INE9", {"deck_summary": ds_full}) == "")
    # THE DRIFT CASE. Same deck, re-extracted: 35 rows became 33 and one section
    # emptied. That must NOT read as a correction, or every backfill re-mails the book.
    ds_drift = pd.concat([ds_full] * 12, ignore_index=True)
    check("row-count drift does not change the key",
          content_key("presentation", "INE1", {"deck_summary": ds_drift}) == k_deck)
    ds_other = ds_full.copy()
    ds_other.loc[0, "value"] = 999.0
    ds_other.loc[0, "section"] = "capex"
    check("a changed value does not fire either (precision over recall)",
          content_key("presentation", "INE1", {"deck_summary": ds_other}) == k_deck)
    # THE CASE IT EXISTS FOR: 45 decks returned zero rows before the salvage fix.
    empty = pd.DataFrame(columns=["isin", "quarter", "section"])
    check("no content at all yields no key",
          content_key("presentation", "INE1", {"deck_summary": empty}) == "")
    check("gaining content changes the key from nothing to something",
          content_key("presentation", "INE1", {"deck_summary": empty}) != k_deck)
    # which table supplied the content is part of the key, so a deck that gains a
    # standalone summary on top of highlights is a real change
    hi_only = pd.DataFrame([{"isin": "INE1", "quarter": "Q1FY27", "category": "demand",
                             "statement": "x", "value": 1, "unit": "%"}])
    k_hi = content_key("presentation", "INE1", {"ppt_highlights": hi_only})
    check("highlights-only differs from summary-only", k_hi != k_deck and k_hi != "")
    check("both together differ from either alone",
          content_key("presentation", "INE1",
                      {"deck_summary": ds_full, "ppt_highlights": hi_only})
          not in ("", k_hi, k_deck))
    check("results have no key yet", content_key("results", "INE1", T) == "")
    check("content_key is in the ledger schema", "content_key" in LEDGER_COLS)

    # ---- concall + annual report -------------------------------------------
    _D = "doc-1"
    NT = {
        "quarterly_facts": pd.DataFrame([
            {"isin": "INE1", "revenue_q": 1284.5, "ebitda_q": 214.0, "pat_q": 131.2,
             "margin_pct": 16.7, "revenue_12m": 4820.0, "pat_12m": 498.3,
             "processed_at": "2026-08-30", "source_doc_id": _D},
            {"isin": "INE1", "revenue_q": 900.0, "ebitda_q": 100.0, "pat_q": 70.0,
             "margin_pct": 11.1, "revenue_12m": 4000.0, "pat_12m": 400.0,
             "processed_at": "2026-05-30", "source_doc_id": "old"}]),
        "guidance_tracker": pd.DataFrame([
            {"isin": "INE1", "metric": "Revenue growth", "value": "18-20%",
             "horizon_fy": "FY27", "notes": "export book",
             "processed_at": "2026-08-30", "source_doc_id": _D},
            {"isin": "INE1", "metric": "Scope 1 emissions", "value": "-30%",
             "horizon_fy": "FY30", "notes": "net zero pathway",
             "processed_at": "2026-08-30", "source_doc_id": _D}]),
        "gf1_guidance_statements": pd.DataFrame([
            {"isin": "INE1", "exact_statement": "We expect to close FY27 with revenue "
             "growth of eighteen to twenty percent.", "timeframe": "FY27",
             "explicitness_type": "explicit", "processed_at": "2026-08-30",
             "source_doc_id": _D}]),
        "ar_red_flags": pd.DataFrame([
            {"isin": "INE1", "category": "Cash flow", "flag_type": "CFO below PAT",
             "severity": "low", "evidence": "third year of divergence",
             "page_ref": "p.9", "processed_at": "2026-08-30", "source_doc_id": _D},
            {"isin": "INE1", "category": "RPT", "flag_type": "RPT growth",
             "severity": "high", "evidence": "RPT sales rose faster than sales",
             "page_ref": "p.20", "processed_at": "2026-08-30", "source_doc_id": _D}]),
        "_narr": {("INE1", "concall"): {"period": "Q1 FY27", "text": "Record quarter."},
                  ("INE1", "annual_report"): {"period": "FY26", "text": "Year closed."}},
    }

    # _doc_rows must prefer THIS document over the company's older rows
    check("_doc_rows prefers the document's own rows",
          float(_doc_rows(NT, "quarterly_facts", "INE1", _D).iloc[0]["revenue_q"]) == 1284.5)
    check("_doc_rows falls back when doc_id is unknown",
          len(_doc_rows(NT, "quarterly_facts", "INE1", "no-such-doc")) == 2)
    check("_doc_rows fallback is newest-first",
          float(_doc_rows(NT, "quarterly_facts", "INE1", "no-such-doc")
                .iloc[0]["revenue_q"]) == 1284.5)
    check("_num drops NaN", _num(float("nan")) == "" and _num(None) == "")
    check("_num keeps a real number", _num(16.7, "%") == "16.7%")

    cb = concall_body("INE1", "ACME", "Acme Ltd", "Q1 FY27", _D, "Record quarter.", NT)
    check("concall body renders", len(cb) > 400)
    check("concall carries the narrative", "Record quarter." in cb)
    check("concall carries this quarter's revenue", "1284.5" in cb)
    check("concall carries the guidance", "18-20%" in cb)
    check("concall quotes the verbatim statement", "eighteen to twenty" in cb)
    check("concall filters ESG out of guidance", "Scope 1" not in cb)

    ab = annual_report_body("INE1", "ACME", "Acme Ltd", "FY2025-26", _D, "Year closed.", NT)
    check("AR body renders", len(ab) > 300)
    check("AR carries the narrative", "Year closed." in ab)
    check("AR carries the FY label", "FY2025-26" in ab)
    check("AR red flags are severity-ordered, high first",
          "HIGH" in ab and "LOW" in ab and ab.index("HIGH") < ab.index("LOW"))

    # An unextracted document must render NOTHING, so it stays due and is retried
    # rather than being burned on an empty mail.
    check("concall with no content renders nothing",
          concall_body("INE9", "X", "X", "", "", "", {}) == "")
    check("AR with no content renders nothing",
          annual_report_body("INE9", "X", "X", "", "", "", {}) == "")

    # content_key: coarse, and it moves only when the narrative appears
    k_has = content_key("concall", "INE1", NT)
    k_not = content_key("concall", "INE1",
                        {"_narr": {("INE1", "concall"): {"period": "Q1 FY27", "text": ""}}})
    check("concall key routes through _narrative_key", k_has.startswith("concall|"))
    check("empty narrative gives a DIFFERENT key", k_has != k_not)
    check("nothing known gives no key at all", content_key("concall", "INE9", {}) == "")
    check("AR key routes through _narrative_key",
          content_key("annual_report", "INE1", NT).startswith("annual_report|"))
    check("the coarse key ignores prose drift",
          content_key("concall", "INE1",
                      {"_narr": {("INE1", "concall"): {"period": "Q1 FY27",
                                                       "text": "Totally different."}}})
          == k_has)
    check("scoped types are exactly concall and AR",
          set(SCOPED_TYPES) == {"concall", "annual_report"})

    # --symbols must select the HOLDING, where --limit can only cap a count
    _due = [{"symbol": "APLAPOLLO", "doc_type": "concall"},
            {"symbol": "APLAPOLLO", "doc_type": "annual_report"},
            {"symbol": "SYRMA", "doc_type": "concall"}]
    _want = {s.strip().upper() for s in "aplapollo".split(",") if s.strip()}
    _sel = [d for d in _due if str(d.get("symbol", "")).upper() in _want]
    check("--symbols keeps both docs of the chosen holding", len(_sel) == 2)
    check("--symbols excludes every other holding",
          all(d["symbol"] == "APLAPOLLO" for d in _sel))
    # --force must never be usable without --symbols: that is the guard stopping an
    # accidental re-send of the whole book.
    import inspect as _insp
    _src = _insp.getsource(main)
    check("--force refuses to run without --symbols",
          "--force requires --symbols" in _src)
    check("--force marks its rows as a resend", '"resend": True' in _src)
    # The scope loop must not rebind `want`, which holds the requested doc types.
    check("the scope loop does not shadow the doc-type set",
          "got, want =" not in _src and "got, expect =" in _src)
    # An AR row with no period must still resolve to a period, or its section can only
    # be found by a doc marker that a re-extraction leaves on the STALE section.
    _nsrc = _insp.getsource(_narratives)
    check("_narratives derives a period for an AR that has none",
          'dt == "annual_report"' in _nsrc and "ar_fy_year" in _nsrc)
    # Compare the CALLS, not the imports - the import line names _find_region first.
    check("_narratives refuses to render a prompt echo as the report",
          "is_prompt_echo(txt)" in _nsrc)
    check("_narratives prefers the exact doc-keyed record",
          "load_doc_report(" in _nsrc
          and _nsrc.index("load_doc_report(") < _nsrc.index("_find_region("))
    check("the derived period is FY-shaped",
          f"FY{str(ar_fy_year('2026-03-31', ''))[-2:]}" == "FY26")

    # ---- the report is CARRIED, not distilled -----------------------------
    _rep = """## FY26 Annual Report - Annual Report 2026 from bse
*Processed: 2026-09-02*
<!-- doc:zz -->

# CONSOLIDATED ANNUAL REPORT SYNTHESIS

## 2. FINANCIAL PERFORMANCE & GROWTH TRAJECTORY
| Metric | FY25 | FY26 |
| :--- | :--- | :--- |
| Revenue | 100 | 122 |
| PAT | 8 | 12 |

*   Revenue grew twenty two percent on volume.
*   Margin expanded one hundred forty basis points.

**Weighted Overall Risk Score: 2.65 (Label: Monitor)**

## 5B. MANAGEMENT OUTLOOK & EXPANSION
Capacity reaches eight million tons by FY28.
"""
    _out = md_to_html(_rep)
    check("the report's table survives", "<table" in _out and "122" in _out)
    check("its header row survives", "<th" in _out and "FY26" in _out)
    check("bullets survive", "<li" in _out and "twenty two percent" in _out)
    check("section headings survive", "FINANCIAL PERFORMANCE" in _out)
    check("a bolded line becomes a heading", "Risk Score: 2.65" in _out)
    check("the document heading is dropped", "Annual Report 2026 from bse" not in _out)
    check("the Processed stamp is dropped", "Processed:" not in _out)
    check("the doc marker is dropped", "doc:zz" not in _out and "<!--" not in _out)
    check("markup in the source cannot leak",
          "&lt;script&gt;" in md_to_html("<script>alert(1)</script>"))
    check("it carries far more than the old lift",
          len(_out) > 6 * sum(len(t) for _h, t in lift_report([("", _rep)], 2600)))

    # ---- the attachment ----------------------------------------------------
    _att = report_attachment("APLAPOLLO", "Annual Report FY2025-26",
                             "APL Apollo - Phase 2 forensic analysis", _rep)
    check("an attachment is produced", _att is not None)
    check("it is named for the company and document",
          _att[0].startswith("APLAPOLLO_Annual_Report"))
    check("it is a PDF where PyMuPDF is available, else html",
          (_att[0].endswith(".pdf") and _att[1].startswith(b"%PDF")
           and _att[2] == "pdf")
          or (_att[0].endswith(".html") and _att[2] == "html"))
    if _att[0].endswith(".pdf"):
        import fitz as _fz
        _d = _fz.open(stream=_att[1], filetype="pdf")
        _txt = "".join(_d[i].get_text() for i in range(_d.page_count))
        _d.close()
        check("the PDF carries the report's numbers", "122" in _txt)
        check("the PDF carries its headings", "FINANCIAL PERFORMANCE" in _txt)
        check("the PDF has real pages", _att[1].count(b"/Type /Page") >= 1
              or b"/Pages" in _att[1])
    _html_only = report_attachment("X", "Doc", "sub", _rep)
    check("the html fallback is still a standalone document",
          html_to_pdf("<!doctype html><html><body>x</body></html>") is not None
          or _html_only[1].lstrip().startswith(b"<!doctype html>"))
    check("nothing to attach yields None",
          report_attachment("X", "Y", "Z", "") is None
          and report_attachment("X", "Y", "Z", []) is None)
    # A resend must not reuse the subject it replaces, or Gmail threads and trims it.
    _msrc = _insp.getsource(main)
    check("a resend subject carries a date stamp",
          _msrc.count("UPDATED {_stamp}") + _msrc.count("UPDATED \"\n") >= 1
          or "_stamp" in _msrc)
    check("both mail types attach their report", _msrc.count("report_attachment(") == 2)

    # ---- concall: the SEASON QUARTER, same rule the deck mail uses ---------
    check("a call filed Aug 2026 is Q1 FY27",
          concall_quarter("2026-08-11") == QT.norm_q("Q1FY27"))
    check("a call filed Sep 2026 is still Q1 FY27",
          concall_quarter("2026-09-01") == QT.norm_q("Q1FY27"))
    check("a call filed May 2026 is Q4 FY26",
          concall_quarter("2026-05-20") == QT.norm_q("Q4FY26"))
    check("an undatable call yields nothing", concall_quarter("") == ""
          and concall_quarter(None) == "" and concall_quarter("Aug 2026") == "")

    # ---- annual report: the FINANCIAL YEAR, on the statutory timetable -----
    # The two date shapes that reach us mean the same FY by opposite routes.
    check("the FY-END stamp names its own year", ar_fy_year("2026-03-31") == 2026)
    check("a report FILED Sep 2026 is also FY2026", ar_fy_year("2026-09-02") == 2026)
    check("a report filed Jun 2026 is FY2026", ar_fy_year("2026-06-15") == 2026)
    check("last year's FY-end stamp is FY2025", ar_fy_year("2025-03-31") == 2025)
    check("a report filed Feb 2026 covers FY2025", ar_fy_year("2026-02-10") == 2025)
    check("period wins when present", ar_fy_year("2026-03-31", "FY25") == 2025)
    check("a 4-digit period also parses", ar_fy_year("", "FY2024") == 2024)
    check("an undatable AR yields None", ar_fy_year("") is None
          and ar_fy_year(None) is None and ar_fy_year("Sept") is None)
    check("the AR filing year in Sep 2026 is FY2026",
          current_ar_fy(date(2026, 9, 2)) == 2026)
    check("the AR filing year in Feb 2026 is still FY2025",
          current_ar_fy(date(2026, 2, 10)) == 2025)
    check("April flips the AR filing year",
          current_ar_fy(date(2026, 4, 1)) == 2026)

    # ---- track 2: holdings that just entered the portfolio -----------------
    _T = date(2026, 9, 2)
    _snaps = pd.DataFrame([
        # history starts 2026-07-23; OLD was there from the first snapshot
        {"snapshot_date": "2026-07-23", "isin": "OLD"},
        {"snapshot_date": "2026-08-31", "isin": "OLD"},
        {"snapshot_date": "2026-08-27", "isin": "NEW"},      # bought 6 days ago
        {"snapshot_date": "2026-08-31", "isin": "NEW"},
        {"snapshot_date": "2026-08-06", "isin": "MID"},      # bought 27 days ago
    ])
    _n30 = new_pf_holdings(_snaps, 30, _T)
    check("a holding bought 6 days ago is new", "NEW" in _n30)
    check("a holding bought 27 days ago is new at 30d", "MID" in _n30)
    check("a holding held since before the history is NOT new", "OLD" not in _n30)
    check("a 7-day window excludes the 27-day-old buy",
          new_pf_holdings(_snaps, 7, _T) == {"NEW"})
    # The guard that matters: once the window outruns the history, everything present on
    # the first snapshot would otherwise read as newly bought.
    check("a 60-day window does NOT declare the whole book new",
          "OLD" not in new_pf_holdings(_snaps, 60, _T))
    check("days=0 disables onboarding", new_pf_holdings(_snaps, 0, _T) == set())
    check("no history means no onboarding, not a crash",
          new_pf_holdings(pd.DataFrame(), 30, _T) == set()
          and new_pf_holdings(None, 30, _T) == set())

    # ---- doc_type=annual_report is not proof of an annual report -----------
    check("a postal-ballot scrutinizer report is not an AR",
          not is_annual_report("Shareholder Meeting / Postal Ballot-Scrutinizer's Report"))
    check("a BRSR filing is not an AR",
          not is_annual_report("Business Responsibility and Sustainability Reporting (BRSR)"))
    check("a newspaper publication is not an AR",
          not is_annual_report("Announcement under Regulation 30 - Newspaper Publication"))
    check("the real filing IS an AR", is_annual_report("Reg. 34 (1) Annual Report. 2 Sep"))
    check("a weblink to the report IS an AR",
          is_annual_report("Weblink / Exact Path Of Integrated Annual Report 2025-26"))
    check("'annual report' always wins over an exclusion word",
          is_annual_report("Newspaper Publication of the Annual Report 2025-26"))
    check("the backfilled bse title IS an AR",
          is_annual_report("Annual Report 2026 from bse"))
    check("an unknown title is allowed through", is_annual_report("Financial Year 2025"))
    check("a blank title is allowed through", is_annual_report(""))

    # ---- the shared company_page matcher these mails depend on --------------
    # Gated HERE because this is the self-test CI runs, and the failure it guards
    # against is silent: a mail that confidently presents the wrong document.
    import run_pf_docs_digest as _D
    _page = """# X

---
## FY26 Presentation - PPT - May 2026 Transcript AI Summary PPT REC
*Processed: 2026-05-10*

### Guidance and outlook
Deck content that must never be served as the annual report of the same year.

---
## FY24 Annual Report - Financial Year 2024 from bse
*Processed: 2026-06-20*

## 2. FINANCIAL PERFORMANCE AND TRAJECTORY
Revenue grew twenty two percent with margin expansion from operating leverage.

---
## Q1 FY26 Concall - Transcript
*Processed: 2026-05-10*

### A-1 Executive Summary
Management guided to twenty percent growth and flagged an export order win.
"""
    _secs = _D._split_sections(_page)
    _ar26 = _D._find_region(_secs, "FY26", "annual_report")
    # The regression: "ar" matched inside "summARy", so an FY26 AR with no section of
    # its own was served that year's PRESENTATION, slide references and all.
    check("an AR never borrows the same year's presentation",
          _ar26 is None or "Presentation" not in _ar26[0][0])
    check("'ar' is no longer an annual-report keyword",
          "ar" not in _D._SECTION_KW["annual_report"])
    _ar24 = _D._find_region(_secs, "FY24", "annual_report")
    check("a real AR section still resolves",
          _ar24 is not None and "Annual Report" in _ar24[0][0])
    check("a real AR lifts its own prose",
          "Revenue grew twenty two" in _D._lift_summary(_ar24 or []))
    check("a presentation still resolves to the presentation",
          (_D._find_region(_secs, "FY26", "presentation") or [("", "")])[0][0]
          .find("Presentation") > 0)
    check("a concall still lifts its own prose",
          "Management guided" in _D._lift_summary(
              _D._find_region(_secs, "Q1 FY26", "concall") or []))
    check("a blank-period concall finds the concall, not the deck",
          "Concall" in (_D._find_region(_secs, "", "concall") or [("", "")])[0][0])

    # ---- the doc-id marker beats a wrong heading label --------------------
    # Real shape, live 2026-09-02: APL Apollo's FY2026 annual report sits under a
    # heading saying FY22, because _extract_fy_year takes the first year of the
    # report's own "Fiscal Coverage Horizon: FY22 - FY26" line.
    _mis = """# X

---
## FY22 Annual Report - Annual Report 2026 from bse
*Processed: 2026-08-30*
<!-- doc:abc-123 -->

### 2. FINANCIAL PERFORMANCE AND TRAJECTORY
Revenue grew twenty two percent with margin expansion from operating leverage.

---
## Q1 FY27 Concall - Transcript
*Processed: 2026-08-11*

### A-1 Executive Summary
Management guided to twenty percent growth.
"""
    _ms = _D._split_sections(_mis)
    check("the FY26 label finds nothing, as the heading says FY22",
          _D._find_region(_ms, "FY26", "annual_report") is None)
    check("the doc-id marker finds it anyway",
          _D._find_region(_ms, "FY26", "annual_report", "abc-123") is not None)
    check("and it lifts the real prose",
          "Revenue grew twenty two" in _D._lift_summary(
              _D._find_region(_ms, "FY26", "annual_report", "abc-123") or []))
    check("an unknown doc-id falls back to the label logic",
          _D._find_region(_ms, "FY26", "annual_report", "no-such-id") is None)
    check("the doc-id region stops at the next document",
          "Management guided" not in _D._lift_summary(
              _D._find_region(_ms, "FY26", "annual_report", "abc-123") or []))

    # A rating section carries no period at all, so the old header-only boundary test
    # let 124,637 chars of rating prose flow into APL Apollo's annual-report region.
    _rating_head = "## CRISIL AA+ Credit Rating - Rating update 30 Sep 2025 from crisil"
    check("a rating section IS a document boundary",
          _D._is_boundary(_rating_head, "*Processed: 2026-08-30*\n<!-- doc:xyz -->"))
    check("a dated rating heading is a boundary even with no marker",
          _D._is_boundary(_rating_head))
    check("an intra-document header is NOT a boundary",
          not _D._is_boundary("## Section GF1 - Raw Guidance Extraction"))
    check("a numbered AR sub-section is NOT a boundary",
          not _D._is_boundary("## 6. FORENSIC FINANCIAL RISK SCORECARD"))
    check("a marker anywhere in the first lines makes it a boundary",
          _D._is_boundary("## Anything At All", "*Processed: x*\n<!-- doc:abc -->"))

    # ---- the model sometimes echoes the prompt back into the stored report -------
    _echo = ("Generate the final report immediately without displaying preliminary "
             "steps. No individual paragraph may be longer than 3 lines. The ENTIRE "
             "report must stay under ~1,200 lines. Output must be completely clean.")
    check("prompt text is recognised as an echo", _is_prompt_echo(_echo))
    check("real analysis is NOT an echo",
          not _is_prompt_echo("Revenue grew 22% with margin expansion of 140bps driven "
                              "by operating leverage and a better export mix."))
    check("a single incidental phrase is not enough to flag an echo",
          not _is_prompt_echo("The output must be read alongside the cash flow note."))

    # ---- plain-numbered sections must split, or the whole report is one block ----
    _plain = [("", "1. SOURCE COVERAGE & DATA INTEGRITY\nParsed FY2025-26.\n"
                   "2. FINANCIAL PERFORMANCE & GROWTH TRAJECTORY\n"
                   + "The shift to charter hire altered the return profile. " * 6
                   + "\n3. CAPITAL EFFICIENCY & ASSET ALLOCATION\n"
                   + "Capex was prioritised over dividends. " * 6)]
    _subs = _subsections(_plain)
    check("plain-numbered sections split into blocks", len(_subs) >= 3)
    _got = lift_report(_plain, 2000)
    check("lift_report picks the wanted section", bool(_got))
    check("and it skips source coverage",
          all("SOURCE COVERAGE" not in h.upper() for h, _t in _got))
    check("and it carries the real analysis",
          any("charter hire" in t for _h, t in _got))

    print(f"\npf_company_mails self-test: {ok} passed, {fail} failed")
    return 1 if fail else 0


if __name__ == "__main__":
    main()
