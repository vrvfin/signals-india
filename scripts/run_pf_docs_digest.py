r"""
run_pf_docs_digest.py — PF "new documents" daily digest (user 2026-07-11; NO Gemini).

Purely COLLECTS what Phase 2 already extracted + summarised, filtered to the
PORTFOLIO, and mails it. One block per PF stock that had a NEW document arrive,
with the actual narrative summary pulled from that stock's company_page.md.

Sources (all already on Drive — nothing is fetched or re-summarised here):
  processing_queue.parquet  -> new concall / annual_report / presentation / results
                               / rating docs (status=done, arrival within window)
  company_page.md           -> the narrative summary section for each typed doc
                               (Executive Summary / Forward Guidance subsection)
  ratings.parquet           -> clean one-liner for a rating doc (agency/rating/action)
  announcement_ledger.parquet -> other BSE announcements + their LLM summary

Dedupe: pf_docs_mailed.parquet ledger (item_id = doc_id | newsid). A document is
mailed EXACTLY ONCE, ever — so the daily run only ever sends genuinely-new arrivals,
and re-running is safe. "Arrival" = announcement/ann date within --days (default 2),
which excludes ancient docs the backfill happens to (re)process today.

Usage:
    python scripts/run_pf_docs_digest.py --dry-run           # build preview, no write/mail
    python scripts/run_pf_docs_digest.py                     # daily (last 2 days, deduped)
    python scripts/run_pf_docs_digest.py --days 30           # one-time catch-up (AR season)
"""
from __future__ import annotations

import argparse
import io
import os
import re
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

_SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

import pandas as pd
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(_SCRIPTS_DIR), ".env"))

from _extractor_base import (get_drive, get_or_create_subfolder, find_file,
                             download_bytes, load_parquet, save_parquet,
                             load_portfolio_isins, log)
from mailer import send_email, load_mail_settings, esc

# narrative doc types drained from the global queue (rating comes from ratings.parquet)
TYPED = ("concall", "annual_report", "presentation", "results")
SUMMARY_LIMIT = 1600            # chars of narrative per doc in the mail
# per-(stock, doc_type) cap — the backfill bulk-loads historical ratings/ARs, so
# show only the most-recent few of each; the ledger then adds genuinely-new ones daily.
CAP = {"annual_report": 1, "concall": 2, "presentation": 1, "results": 2,
       "rating": 2, "announcement": 4}
PER_STOCK_CAP = 8              # hard ceiling per stock block (rest -> "+N more")
# ARs: announcement_date is the fiscal-year-END (FY2025 -> 2025-03-31), NOT the
# declaration date. A day-based cutoff wrongly drops recent FY2025 ARs whose FY-end
# is >1yr old. Key on FY-end YEAR instead: keep the last AR_FY_LOOKBACK+1 years
# (today.year - AR_FY_LOOKBACK), i.e. this + last AR season. cap=1 -> latest per co.
AR_FY_LOOKBACK = 1
MAILED_NAME = "pf_docs_mailed.parquet"
MAILED_COLS = ["item_id", "isin", "symbol", "doc_type", "arrival_date", "mailed_at",
               "summary_state"]
# A document whose narrative could not be lifted was still burned into the ledger, and
# item_id mails ONCE EVER - so the "(summary pending - the nightly backfill will extract
# this in a later pass)" line printed in the mail promised a retry the ledger made
# impossible. Pending rows now retry until the narrative appears, or until the FIRST
# sighting is this many days old, whichever comes first.
LEDGER_BURN_DAYS = 14

DOC_ICON = {"concall": "🎙", "annual_report": "📗", "presentation": "📊",
            "results": "📈", "rating": "🏷", "announcement": "📢"}
DOC_LABEL = {"concall": "Concall", "annual_report": "Annual Report",
             "presentation": "Presentation", "results": "Results",
             "rating": "Rating", "announcement": "Announcement"}
# doc_type -> keywords that appear in the company_page.md section header.
# "ar" WAS in the annual_report list and matched inside ordinary words, because _norm()
# strips spaces before the test: "summARy" contains it, so APLAPOLLO's
# "## FY26 Presentation - PPT - May 2026 Transcript AI Summary PPT REC" was served as
# that company's Annual Report, slide references and all. Every real AR heading is
# written as "<FY> Annual Report - <title>" by _extractor_base.append_company_page and
# extract_annual_report._replace_ar_section, so "annual" alone is sufficient AND safe.
_SECTION_KW = {"concall": ("concall",), "annual_report": ("annual",),
               "presentation": ("ppt", "presentation"), "results": ("result",)}
# Which keywords positively identify a section as belonging to a given document type.
# Used to keep one document's section from being served as another's.
_TYPE_KW = {"concall": ("concall", "transcript"),
            "annual_report": ("annualreport",),
            "presentation": ("ppt", "presentation"),
            "results": ("results",)}


def _other_type_heading(hn: str, doc_type: str) -> bool:
    """True when this normalised heading plainly belongs to a DIFFERENT document type.

    The period-only fallback below matches on the period alone, and a company page
    carries several documents per period - so without this guard an annual report with
    no section of its own silently borrows the quarter's presentation. Reporting nothing
    is correct there; reporting a deck as the annual report is not.
    """
    for t, kws in _TYPE_KW.items():
        if t == doc_type:
            continue
        if any(k in hn for k in kws):
            return True
    return False
# priority order of narrative subsections to lift from a company_page section.
# Covers concall headers (A-1 Executive Summary…) AND AR headers (numbered
# "2. FINANCIAL PERFORMANCE…", "7. INVESTMENT THESIS…").
_SUMMARY_HEADS = ("executive summary", "investment thesis", "forward guidance",
                  "financial performance", "growth trajectory", "q&a summary",
                  "management commentary", "growth drivers", "company overview",
                  "summary")
# boilerplate/methodology subsections to never lift as the "summary"
_SKIP_HEADS = ("source coverage", "data integrity", "probing questions",
               "forensic financial risk scorecard", "credibility", "mgmt said")

QUEUE_COLS = ["doc_id", "isin", "symbol", "company_name", "doc_type", "title",
              "announcement_date", "status", "discovered_at", "processed_at", "period"]
RATINGS_COLS = ["isin", "symbol", "company_name", "agency", "rating", "outlook",
                "rating_action", "instrument_type", "rated_amount_cr", "rating_date",
                "processed_at", "source_doc_id"]
ANN_COLS = ["newsid", "isin", "symbol", "ann_date", "category", "headline",
            "summary", "status", "processed_at", "materiality", "direction"]


def _settled_ids(mailed: pd.DataFrame) -> set:
    """item_ids that must never be reported again.

    SETTLED = the item was reported carrying a real narrative ("full"), or it predates
    summary_state entirely (legacy rows read back as None - treated as full, so the
    historical ledger keeps suppressing exactly what it suppressed before).

    A "pending" row is NOT settled: it was mailed with no liftable summary, so it is
    retried every run until the extractor produces the narrative - or until
    LEDGER_BURN_DAYS have passed since the FIRST sighting, so a permanently
    unextractable document cannot retry forever.
    """
    if mailed is None or mailed.empty:
        return set()
    ids = mailed["item_id"].astype(str)
    state = mailed["summary_state"].astype(str).str.strip().str.lower()
    settled = set(ids[state != "pending"])
    pend = mailed[state == "pending"]
    if not pend.empty:
        cutoff = (date.today() - timedelta(days=LEDGER_BURN_DAYS)).isoformat()
        first = pend.groupby(pend["item_id"].astype(str))["mailed_at"].min()
        settled |= {i for i, t in first.items() if str(t)[:10] <= cutoff}
    return settled


# ------------------------------------------------------------------ #
#  company_page.md narrative extraction                               #
# ------------------------------------------------------------------ #

def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(s or "").lower())


def _split_sections(md: str) -> list[tuple[str, str]]:
    """[(header_line, body)] for each top-level '## ' section (not ### / #)."""
    out, cur_h, cur_b = [], None, []
    for ln in md.splitlines():
        if re.match(r"^##\s", ln) and not re.match(r"^###", ln):
            if cur_h is not None:
                out.append((cur_h, "\n".join(cur_b)))
            cur_h, cur_b = ln, []
        elif cur_h is not None:
            cur_b.append(ln)
    if cur_h is not None:
        out.append((cur_h, "\n".join(cur_b)))
    return out


_BOUNDARY_KW = ("concall", "ppt", "presentation", "annualreport", "annual report",
                "rating", "results", "announcement")
_DOC_MARKER_RE = re.compile(r"<!--\s*doc:")
# A period OR a plain date. The period-only test missed rating sections outright:
# "## CRISIL AA+ Credit Rating - Rating update 30 Sep 2025 from crisil" carries no
# FY or quarter, so it was not read as a document boundary and 124,637 characters of
# RATING prose were absorbed into APL Apollo's annual-report region.
_PERIOD_RE = re.compile(r"q[1-4]\s*fy\s*\d{2}|\bfy\s*\d{2,4}"
                        r"|\d{1,2}\s+[a-z]{3}[a-z]*\.?\s+\d{4}|\d{4}-\d{2}-\d{2}", re.I)


def _is_boundary(header: str, body: str = "") -> bool:
    """A top-level header that STARTS a new document, as opposed to an intra-document
    header (Section 1), GF1, Mgmt Credibility…).

    THE DOC MARKER IS PROOF. append_company_page stamps "<!-- doc:<id> -->" directly
    under the heading of every section it writes, so a marker within the first few lines
    of a body means this heading begins a new document - whatever its label says. That
    matters because the labels are not dependable: extract_annual_report derives the FY
    from the report's own text and gets it wrong, and rating sections carry no period at
    all. The header heuristic stays as the fallback for older sections written before
    markers existed.
    """
    if body and _DOC_MARKER_RE.search(body[:400]):
        return True
    hn = header.lower()
    return bool(_PERIOD_RE.search(hn) and any(k in hn.replace(" ", "") or k in hn
                                              for k in _BOUNDARY_KW))


def _find_region(sections, period: str, doc_type: str, doc_id: str = "") -> str | None:
    """Concatenated body of a document's FULL block: its boundary section plus the
    following intra-doc sections, up to the next document boundary.

    doc_id, WHEN PRESENT, IS THE ONLY EXACT KEY. append_company_page embeds
    "<!-- doc:<id> -->" under the heading of every AR / rating / presentation section, and
    matching on it sidesteps the heading label entirely - which matters because the label
    is not reliable. extract_annual_report._extract_fy_year takes the FIRST year out of
    the report's own "Fiscal Coverage Horizon: FY22 - FY26" line, so APL Apollo's FY2026
    annual report sits on the page under "## FY22 Annual Report - Annual Report 2026 from
    bse". A period test for FY26 finds nothing there and the reader gets a mail with no
    summary, even though Phase 2 wrote a full forensic report. Concall sections carry no
    marker, so they still fall through to the period logic below.
    """
    pn = _norm(period)
    kws = _SECTION_KW.get(doc_type, (doc_type,))

    def _walk(i):
        region = [sections[i]]
        for h2, body2 in sections[i + 1:]:
            if _is_boundary(h2, body2):
                break
            region.append((h2, body2))
        return region

    # THE LAST MATCH WINS, NOT THE FIRST. A re-extraction can leave TWO sections for one
    # document: extract_annual_report._replace_ar_section looks the old section up by its
    # FY heading, and once _extract_fy_year started labelling correctly it no longer
    # matches the stale wrongly-labelled one, so it APPENDS. APL Apollo now carries both
    # "## FY22 Annual Report - Annual Report 2026 from bse" (the old truncated text, which
    # still holds the doc marker) and "## FY26 Annual Report - ..." (the good re-read).
    # Sections are appended chronologically, so the last is the freshest.
    # COLLECT EVERY IDENTIFICATION, THEN TAKE THE FRESHEST. Trying the doc marker first
    # and returning on it was wrong in exactly the case that matters: a re-extraction
    # leaves the STALE section holding the marker (append_company_page stamped it) while
    # the good re-read is appended under a correct heading with NO marker, because
    # _replace_ar_section drops it on the supersede path. APL Apollo carried both - the
    # marker match returned the old 764-char text and the fresh 20k-char report was
    # ignored. Sections are appended chronologically, so the highest index is newest.
    cands = []
    if doc_id:
        marker = f"<!-- doc:{str(doc_id).strip()} -->"
        cands += [i for i, (h, b) in enumerate(sections) if marker in b or marker in h]

    def _starts_here(hn: str) -> bool:
        if not (pn and pn in hn and any(_norm(k) in hn for k in kws)):
            return False
        if doc_type == "concall" and "ppt" in hn:   # don't grab the presentation
            return False
        return True

    cands += [i for i, (h, _b) in enumerate(sections) if _starts_here(_norm(h))]
    if cands:
        return _walk(max(cands))

    start = None
    if start is None:                                # fallback: period-only match
        for i, (h, _b) in enumerate(sections):
            hn = _norm(h)
            if pn and pn in hn and not _other_type_heading(hn, doc_type):
                start = i
                break
    if start is None and not pn:
        # PERIOD UNKNOWN. Concall sections carry no <!-- doc:... --> marker
        # (extract_concall.py writes the header without one), and `period` on the queue
        # row is frequently blank - which renders the heading as "##  Concall - Title"
        # and made both matches above impossible, so every such document reported no
        # summary at all. Fall back to the doc_type keyword alone and take the LAST
        # match, i.e. the most recent document of that type on the page.
        for i, (h, _b) in enumerate(sections):
            hn = _norm(h)
            if not any(_norm(k) in hn for k in kws):
                continue
            if _other_type_heading(hn, doc_type):          # not another doc's section
                continue
            start = i
    if start is None:
        return None
    region = [sections[start]]
    for h, body in sections[start + 1:]:
        if _is_boundary(h, body):
            break
        region.append((h, body))
    return region


_META_RE = re.compile(
    r"^(company name|sector\b|industry\b|annual report fiscal|reporting period|"
    r"call date|date of\b|active reporting|processed:|table_a\b)", re.I)


def _prose(text: str) -> str:
    """Human prose only: drop tables, code fences, headers, [TAG] lines, metadata."""
    # append_company_page embeds "<!-- doc:<id> -->" for idempotency. It sits on its own
    # line inside the section body, starts with "<" so no rule below caught it, and was
    # reaching the reader verbatim mid-sentence in the Morepen annual-report mail.
    text = re.sub(r"<!--.*?-->", " ", str(text or ""), flags=re.S)
    out = []
    for l in text.splitlines():
        raw = l.strip()
        s = raw.lstrip("*#>- ").strip()            # strip md emphasis/quote/bullet first
        if (not s or raw.startswith("|") or raw.startswith("```")
                or re.match(r"^#{1,6}\s", l) or re.match(r"^\[[A-Z ]+\]", s)
                or _META_RE.match(s) or not (set(s) - set("-:|*# "))):
            continue
        out.append(s)
    return " ".join(out).strip()


def _lift_summary(region: list, limit: int = SUMMARY_LIMIT) -> str:
    """Best narrative subsection across a doc's region. Handles AR (meaningful
    content in top-level '## 2. FINANCIAL PERFORMANCE…' sections) AND concall
    ('### A-1 Executive Summary' sub-headers), skipping boilerplate sections."""
    if not region:
        return ""
    # candidates = each region section header + its ###/#### sub-headers
    cands: list = []
    for h, body in region:
        cands.append((h, body))
        cur_h, cur_b = "", []
        for ln in body.splitlines():
            if re.match(r"^#{3,4}\s", ln):
                if cur_h or cur_b:
                    cands.append((cur_h, "\n".join(cur_b)))
                cur_h, cur_b = ln, []
            else:
                cur_b.append(ln)
        if cur_h or cur_b:
            cands.append((cur_h, "\n".join(cur_b)))

    def _pick(h, text):
        if any(sk in h.lower() for sk in _SKIP_HEADS):
            return None
        p = _prose(text)
        return (p[:limit].rstrip() + ("…" if len(p) > limit else "")) if len(p) > 60 else None

    for want in _SUMMARY_HEADS:                       # priority-ordered
        for h, text in cands:
            if want in h.lower():
                got = _pick(h, text)
                if got:
                    return got
    for h, text in cands:                             # fallback: first non-boilerplate prose
        got = _pick(h, text)
        if got:
            return got
    return ""


# ------------------------------------------------------------------ #
#  collection                                                         #
# ------------------------------------------------------------------ #

def _company_page(drive, repo_id: str, isin: str, cache: dict) -> list:
    if isin in cache:
        return cache[isin]
    sections = []
    q = (f"name='{isin}' and '{repo_id}' in parents "
         f"and mimeType='application/vnd.google-apps.folder' and trashed=false")
    found = drive.files().list(q=q, fields="files(id)").execute().get("files", [])
    if found:
        fid = find_file(drive, found[0]["id"], "company_page.md")
        if fid:
            try:
                md = download_bytes(drive, fid).decode("utf-8", "replace")
                sections = _split_sections(md)
            except Exception as e:
                log(f"  WARN: company_page.md read failed for {isin} ({str(e)[:50]})")
    cache[isin] = sections
    return sections


def _rating_line(r: pd.Series) -> str:
    def _v(x):
        s = str(r.get(x, "") or "").strip()
        return "" if s in ("", "DATA_MISSING", "nan") else s
    bits = [b for b in [_v("agency"), _v("rating"),
                        f"({_v('outlook')})" if _v("outlook") else "",
                        f"— {_v('rating_action')}" if _v("rating_action") else "",
                        f"₹{_v('rated_amount_cr')} cr" if _v("rated_amount_cr") else "",
                        _v("instrument_type")] if b]
    return " ".join(bits)


def _add(out, isin, sym, name, item):
    e = out.setdefault(isin, {"symbol": sym, "name": name, "items": []})
    if not e["symbol"]:
        e["symbol"] = sym
    if not e["name"]:
        e["name"] = name
    e["items"].append(item)


def _ar_display(announcement_date: str) -> str:
    """Unambiguous AR name (user 2026-07-12): announcement_date is the FY-END
    (2026-03-31 = the 2025-26 Annual Report) -> 'Annual Report FY2025-26
    (yr ended Mar 2026)'. Avoids the confusing bare 'FY25/FY26' labels."""
    yr = str(announcement_date or "")[:4]
    if not yr.isdigit():
        return ""
    y = int(yr)
    # the doc-type label ("📗 Annual Report") is rendered separately, so this is
    # just the fiscal-year qualifier
    return f"FY{y - 1}-{str(y)[2:]} (yr ended Mar {y})"


def _structured_fallback(dt: str, doc_id: str, isin: str, tables: dict) -> str:
    """When company_page.md has no clean narrative for a doc, fall back to the
    structured guidance rows the extractors DID tabulate (precise doc match
    first, else the company's latest rows)."""
    src = {"annual_report": "ar_g", "concall": "guid",
           "presentation": "ppt"}.get(dt)
    if not src:
        return ""
    df = tables.get(src)
    if df is None or df.empty:
        return ""
    hit = df[df["source_doc_id"].astype(str) == str(doc_id)]
    if hit.empty:
        hit = df[df["isin"].astype(str) == isin].tail(3)
    parts = []
    for _, x in hit.head(3).iterrows():
        v = str(x.get("value", "") or "").strip()
        if not v or v.lower() == "nan":
            continue
        hz = str(x.get("horizon_fy") or x.get("horizon") or "").strip()
        parts.append(f"{esc(x.get('metric', ''), 14)}: {esc(v, 24)}"
                     + (f" ({esc(hz, 8)})" if hz else ""))
    return ("guidance — " + " · ".join(parts)) if parts else ""


def collect(drive, repo_id, index_id, pf, since_date, mailed_ids, cache):
    """Return {isin: {'symbol','name','items':[(doc_type, header, summary, id, arr)]}}.
    Windows on the doc's TRUE date (rating_date / ann_date / recent arrival) and caps
    per (stock, type) so a bulk historical backfill can't flood the digest."""
    out: dict = {}
    ar_min_year = date.today().year - AR_FY_LOOKBACK   # e.g. 2026 -> keep FY-end >= 2025

    # structured-guidance fallbacks when company_page.md lacks a clean narrative
    tables = {
        "guid": load_parquet(drive, index_id, "guidance_tracker.parquet",
                             ["isin", "metric", "value", "horizon_fy",
                              "processed_at", "source_doc_id"]),
        "ar_g": load_parquet(drive, index_id, "ar_guidance.parquet",
                             ["isin", "metric", "value", "horizon_fy",
                              "processed_at", "source_doc_id"]),
        "ppt": load_parquet(drive, index_id, "ppt_guidance.parquet",
                            ["isin", "metric", "value", "horizon",
                             "processed_at", "source_doc_id"]),
    }

    # --- narrative docs from the global queue (concall/AR/presentation/results) ---
    q = load_parquet(drive, index_id, "processing_queue.parquet", QUEUE_COLS)
    if not q.empty:
        # ARRIVAL = recently DISCOVERED in our pipeline (not the filing's own date,
        # which for an AR is the FY-end months earlier) OR recently EXTRACTED. Eligibility
        # requires status=done, but discovery and extraction are different days: a doc
        # found Monday and extracted Thursday became `done` outside its own discovery
        # window, and keyed on discovered_at alone it was dropped silently and forever.
        _arrived = ((q["discovered_at"].astype(str) >= since_date)
                    | (q["processed_at"].astype(str) >= since_date))
        q = q[(q["status"].astype(str) == "done")
              & (q["isin"].astype(str).isin(pf))
              & (q["doc_type"].astype(str).isin(TYPED))
              & _arrived]
        q = q[~q["doc_id"].astype(str).isin(mailed_ids)]
        # ARs: keep only recent fiscal years (by FY-end YEAR, not a day-window — the
        # FY-end lags the declaration, so a day-cutoff wrongly dropped recent FY2025 ARs).
        ar_yr = pd.to_numeric(q["announcement_date"].astype(str).str[:4], errors="coerce")
        q = q[~((q["doc_type"].astype(str) == "annual_report") & (ar_yr < ar_min_year))]
        for (isin, dt), grp in q.groupby([q["isin"].astype(str), q["doc_type"].astype(str)]):
            grp = grp.sort_values("announcement_date", ascending=False).head(CAP.get(dt, 3))
            for _, r in grp.iterrows():
                period = str(r.get("period") or "").strip()
                doc_id = str(r["doc_id"])
                summary = _lift_summary(_find_region(
                    _company_page(drive, repo_id, isin, cache), period, dt, doc_id) or "")
                if not summary:   # narrative missing -> structured guidance rows
                    summary = _structured_fallback(dt, doc_id, isin, tables)
                # unambiguous doc name: ARs get FY2024-25-style labels
                label = (_ar_display(r.get("announcement_date"))
                         if dt == "annual_report" else period)
                header = " · ".join([x for x in [label, esc(r.get("title", ""), 70)] if x])
                _add(out, isin, str(r.get("symbol") or ""), str(r.get("company_name") or ""),
                     (dt, header, summary, doc_id,
                      str(r.get("announcement_date") or "")[:10]))

    # --- ratings from ratings.parquet, windowed on the actual rating_date ---
    rt = load_parquet(drive, index_id, "ratings.parquet", RATINGS_COLS)
    if not rt.empty:
        rt = rt[(rt["isin"].astype(str).isin(pf))
                & (rt["rating_date"].astype(str) >= since_date)]
        rt = rt[~rt["source_doc_id"].astype(str).isin(mailed_ids)]
        for isin, grp in rt.groupby(rt["isin"].astype(str)):
            grp = grp.sort_values("rating_date", ascending=False).head(CAP["rating"])
            for _, r in grp.iterrows():
                line = _rating_line(r)
                if not line:
                    continue
                _add(out, isin, str(r.get("symbol") or ""), str(r.get("company_name") or ""),
                     ("rating", "", line, str(r.get("source_doc_id") or ""),
                      str(r.get("rating_date") or "")[:10]))

    # --- announcements (own ledger, not the global queue) ---
    al = load_parquet(drive, index_id, "announcement_ledger.parquet", ANN_COLS)
    if not al.empty:
        al = al[(al["isin"].astype(str).isin(pf))
                & (al["ann_date"].astype(str) >= since_date)]
        al = al[~al["newsid"].astype(str).isin(mailed_ids)]
        for isin, grp in al.groupby(al["isin"].astype(str)):
            grp = grp.sort_values("ann_date", ascending=False).head(CAP["announcement"])
            for _, r in grp.iterrows():
                summ = str(r.get("summary") or "").strip()
                if summ.lower() in ("", "nan"):
                    continue
                head = " · ".join([x for x in [str(r.get("category") or "").strip(),
                                               esc(r.get("headline", ""), 80)] if x])
                _add(out, isin, str(r.get("symbol") or ""), str(r.get("symbol") or ""),
                     ("announcement", head, summ[:SUMMARY_LIMIT],
                      str(r["newsid"]), str(r.get("ann_date") or "")[:10]))
    return out


# ------------------------------------------------------------------ #
#  render                                                             #
# ------------------------------------------------------------------ #

def build_html(blocks: dict, since_date: str, days: float) -> str:
    order = {t: i for i, t in enumerate(("results", "concall", "presentation",
                                         "annual_report", "rating", "announcement"))}
    n_docs = sum(len(v["items"]) for v in blocks.values())
    parts = [
        f"<div style='max-width:720px;font-family:Arial,sans-serif'>"
        f"<p style='font-size:13px'><b>{n_docs} new document(s)</b> across "
        f"<b>{len(blocks)} PF stock(s)</b> since {since_date} (last {days:.0f}d). "
        f"Summaries are Phase-2 extractions from each filing.</p>"]
    for isin, v in sorted(blocks.items(), key=lambda kv: kv[1]["symbol"]):
        sym = esc(v["symbol"] or isin, 14)
        link = f"https://www.screener.in/company/{sym}/" if v["symbol"] else "#"
        items = sorted(v["items"], key=lambda it: (it[4] or "", -order.get(it[0], 9)),
                       reverse=True)
        extra = max(0, len(items) - PER_STOCK_CAP)
        items = items[:PER_STOCK_CAP]
        parts.append(f"<h3 style='margin:16px 0 4px'>"
                     f"<a href='{link}' style='color:#1a237e;text-decoration:none'>"
                     f"{sym}</a> · {esc(v['name'], 40)}</h3>")
        for dt, header, summary, _id, arr in items:
            icon = DOC_ICON.get(dt, "📄")
            body = (f"<div style='margin:2px 0 8px 6px;font-size:12.5px;color:#333;"
                    f"border-left:3px solid #ccc;padding-left:8px'>{esc(summary, SUMMARY_LIMIT)}</div>"
                    if summary else "<div style='margin:2px 0 8px 6px;font-size:12px;"
                    "color:#999'>(summary pending — the nightly backfill will extract "
                    "this document in a later pass)</div>")
            parts.append(
                f"<div style='margin:6px 0'><b>{icon} {DOC_LABEL.get(dt, dt)}</b> "
                f"<span style='color:#777;font-size:12px'>[{arr}] {header}</span></div>"
                + body)
        if extra:
            parts.append(f"<div style='margin:2px 0 8px 6px;font-size:12px;color:#999'>"
                         f"…+{extra} more document(s)</div>")
    parts.append("<p style='font-size:11px;color:#999'>One block per PF stock with a "
                 "new filing. Each document appears once (deduped). Toggle in the app "
                 "sidebar (📧 Email toggles).</p></div>")
    return "\n".join(parts)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--days", type=float, default=2.0,
                    help="Arrival window in days (default 2; use 30 for one-time catch-up).")
    ap.add_argument("--dry-run", action="store_true",
                    help="Build + save preview html; no ledger write, no mail.")
    args = ap.parse_args()

    print("PF new-documents digest — collect Phase-2 summaries for the portfolio")
    print("-" * 60)
    drive = get_drive()
    root_id = os.environ["GDRIVE_FOLDER_ID"]
    repo_id = get_or_create_subfolder(drive, root_id, "company_repo")
    index_id = get_or_create_subfolder(drive, repo_id, "_index")

    pf = load_portfolio_isins(drive, root_id) or set()
    if not pf:
        log("no portfolio file — nothing to do.")
        return

    mailed = load_parquet(drive, index_id, MAILED_NAME, MAILED_COLS)
    mailed_ids = _settled_ids(mailed)
    since_date = (date.today() - timedelta(days=int(args.days))).isoformat()
    log(f"PF ISINs: {len(pf)} · window since {since_date} · already-mailed: {len(mailed_ids)}")

    cache: dict = {}
    blocks = collect(drive, repo_id, index_id, pf, since_date, mailed_ids, cache)
    n_docs = sum(len(v["items"]) for v in blocks.values())
    log(f"new documents to report: {n_docs} across {len(blocks)} PF stock(s)")
    if blocks:
        by_type: dict = {}
        for v in blocks.values():
            for it in v["items"]:
                by_type[it[0]] = by_type.get(it[0], 0) + 1
        log("  by type: " + ", ".join(f"{k}={n}" for k, n in sorted(by_type.items())))

    if not blocks:
        log("no new PF documents in window — no mail.")
        if not args.dry_run:
            return

    html = build_html(blocks, since_date, args.days)
    if args.dry_run:
        prev = Path(__file__).resolve().parent.parent / "pf_docs_digest_preview.html"
        prev.write_text(html, encoding="utf-8")
        for isin, v in list(blocks.items())[:20]:
            log(f"  {v['symbol']:<12} {len(v['items'])} doc(s): "
                + ", ".join(f"{it[0]}" for it in v["items"][:6]))
        print(f"\nDRY RUN — preview saved to {prev.name}; no ledger write, no mail.")
        return

    # record every reported item so it never mails again
    _now = datetime.now().isoformat(timespec="seconds")
    new_rows = [{"item_id": _id, "isin": isin, "symbol": v["symbol"],
                 "doc_type": dt, "arrival_date": arr, "mailed_at": _now,
                 "summary_state": "full" if str(_s).strip() else "pending"}
                for isin, v in blocks.items()
                for (dt, _h, _s, _id, arr) in v["items"]]
    if new_rows:
        combined = pd.concat([mailed, pd.DataFrame(new_rows, columns=MAILED_COLS)],
                             ignore_index=True) if not mailed.empty \
            else pd.DataFrame(new_rows, columns=MAILED_COLS)
        save_parquet(drive, index_id, MAILED_NAME, combined)
        _pend = sum(1 for r in new_rows if r["summary_state"] == "pending")
        log(f"ledger updated: +{len(new_rows)} -> {len(combined)} rows"
            + (f" ({_pend} pending - will retry until extracted or "
               f"{LEDGER_BURN_DAYS}d old)" if _pend else ""))

    if not load_mail_settings(drive, index_id).get("pf_docs_digest", True):
        log("pf_docs_digest mail toggled OFF — skipped.")
        return
    subject = f"📂 PF new documents — {n_docs} across {len(blocks)} stocks — {date.today()}"
    sent = send_email(subject, html)
    log(f"Email {'sent' if sent else 'FAILED'}: "
        f"{subject.encode('ascii', 'ignore').decode().strip()}")


if __name__ == "__main__":
    main()
