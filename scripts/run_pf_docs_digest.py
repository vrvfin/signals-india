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
MAILED_COLS = ["item_id", "isin", "symbol", "doc_type", "arrival_date", "mailed_at"]

DOC_ICON = {"concall": "🎙", "annual_report": "📗", "presentation": "📊",
            "results": "📈", "rating": "🏷", "announcement": "📢"}
DOC_LABEL = {"concall": "Concall", "annual_report": "Annual Report",
             "presentation": "Presentation", "results": "Results",
             "rating": "Rating", "announcement": "Announcement"}
# doc_type -> keywords that appear in the company_page.md section header
_SECTION_KW = {"concall": ("concall",), "annual_report": ("annual", "ar"),
               "presentation": ("ppt", "presentation"), "results": ("result",)}
# priority order of narrative subsections to lift from a company_page section
_SUMMARY_HEADS = ("executive summary", "forward guidance", "q&a summary",
                  "management commentary", "growth drivers", "summary")

QUEUE_COLS = ["doc_id", "isin", "symbol", "company_name", "doc_type", "title",
              "announcement_date", "status", "discovered_at", "processed_at", "period"]
RATINGS_COLS = ["isin", "symbol", "company_name", "agency", "rating", "outlook",
                "rating_action", "instrument_type", "rated_amount_cr", "rating_date",
                "processed_at", "source_doc_id"]
ANN_COLS = ["newsid", "isin", "symbol", "ann_date", "category", "headline",
            "summary", "status", "processed_at", "materiality", "direction"]


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
                "rating", "results")


def _is_boundary(header: str) -> bool:
    """A top-level header that STARTS a new document (period + a doc keyword),
    as opposed to an intra-document header (Section 1), GF1, Mgmt Credibility…)."""
    hn = header.lower()
    has_period = re.search(r"q[1-4]\s*fy\s*\d{2}|\bfy\s*\d{2,4}", hn)
    return bool(has_period and any(k in hn.replace(" ", "") or k in hn
                                   for k in _BOUNDARY_KW))


def _find_region(sections, period: str, doc_type: str) -> str | None:
    """Concatenated body of a document's FULL block: its boundary section plus the
    following intra-doc sections, up to the next document boundary."""
    pn = _norm(period)
    kws = _SECTION_KW.get(doc_type, (doc_type,))

    def _starts_here(hn: str) -> bool:
        if not (pn and pn in hn and any(_norm(k) in hn for k in kws)):
            return False
        if doc_type == "concall" and "ppt" in hn:   # don't grab the presentation
            return False
        return True

    start = None
    for i, (h, _b) in enumerate(sections):
        if _starts_here(_norm(h)):
            start = i
            break
    if start is None:                                # fallback: period-only match
        for i, (h, _b) in enumerate(sections):
            if pn and pn in _norm(h):
                start = i
                break
    if start is None:
        return None
    parts = [sections[start][1]]
    for h, body in sections[start + 1:]:
        if _is_boundary(h):
            break
        parts.append(body)
    return "\n".join(parts)


def _lift_summary(body: str, limit: int = SUMMARY_LIMIT) -> str:
    """Pull the most summary-like subsection (Executive Summary etc.); else the
    first prose paragraph. Skips markdown tables (| ... |)."""
    if not body:
        return ""
    # subsections keyed by '### ' / '#### ' headers
    subs, cur_h, cur_b = [], "", []
    for ln in body.splitlines():
        if re.match(r"^#{3,4}\s", ln):
            if cur_h or cur_b:
                subs.append((cur_h, "\n".join(cur_b)))
            cur_h, cur_b = ln, []
        else:
            cur_b.append(ln)
    if cur_h or cur_b:
        subs.append((cur_h, "\n".join(cur_b)))

    def _prose(text: str) -> str:
        keep = [l for l in text.splitlines()
                if l.strip() and not l.lstrip().startswith("|")
                and not re.match(r"^#{1,6}\s", l) and set(l.strip()) - set("-:| ")]
        return " ".join(keep).strip()

    for want in _SUMMARY_HEADS:
        for h, text in subs:
            if want in h.lower():
                p = _prose(text)
                if len(p) > 40:
                    return p[:limit].rstrip() + ("…" if len(p) > limit else "")
    # fallback: first prose in the whole section
    p = _prose(body)
    return p[:limit].rstrip() + ("…" if len(p) > limit else "") if p else ""


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


def collect(drive, repo_id, index_id, pf, since_date, mailed_ids, cache):
    """Return {isin: {'symbol','name','items':[(doc_type, header, summary, id, arr)]}}.
    Windows on the doc's TRUE date (rating_date / ann_date / recent arrival) and caps
    per (stock, type) so a bulk historical backfill can't flood the digest."""
    out: dict = {}
    ar_min_year = date.today().year - AR_FY_LOOKBACK   # e.g. 2026 -> keep FY-end >= 2025

    # --- narrative docs from the global queue (concall/AR/presentation/results) ---
    q = load_parquet(drive, index_id, "processing_queue.parquet", QUEUE_COLS)
    if not q.empty:
        q = q[(q["status"].astype(str) == "done")
              & (q["isin"].astype(str).isin(pf))
              & (q["doc_type"].astype(str).isin(TYPED))
              # arrival = recently DISCOVERED in our pipeline (not the filing's own date,
              # which for an AR is the FY-end months earlier)
              & (q["discovered_at"].astype(str) >= since_date)]
        q = q[~q["doc_id"].astype(str).isin(mailed_ids)]
        # ARs: keep only recent fiscal years (by FY-end YEAR, not a day-window — the
        # FY-end lags the declaration, so a day-cutoff wrongly dropped recent FY2025 ARs).
        ar_yr = pd.to_numeric(q["announcement_date"].astype(str).str[:4], errors="coerce")
        q = q[~((q["doc_type"].astype(str) == "annual_report") & (ar_yr < ar_min_year))]
        for (isin, dt), grp in q.groupby([q["isin"].astype(str), q["doc_type"].astype(str)]):
            grp = grp.sort_values("announcement_date", ascending=False).head(CAP.get(dt, 3))
            for _, r in grp.iterrows():
                period = str(r.get("period") or "").strip()
                summary = _lift_summary(_find_region(
                    _company_page(drive, repo_id, isin, cache), period, dt) or "")
                header = " · ".join([x for x in [period, esc(r.get("title", ""), 70)] if x])
                _add(out, isin, str(r.get("symbol") or ""), str(r.get("company_name") or ""),
                     (dt, header, summary, str(r["doc_id"]),
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
                    "color:#999'>(summary not found in company_page.md)</div>")
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
    mailed_ids = set(mailed["item_id"].astype(str)) if not mailed.empty else set()
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
    new_rows = [{"item_id": _id, "isin": isin, "symbol": v["symbol"],
                 "doc_type": dt, "arrival_date": arr,
                 "mailed_at": datetime.now().isoformat(timespec="seconds")}
                for isin, v in blocks.items()
                for (dt, _h, _s, _id, arr) in v["items"]]
    if new_rows:
        combined = pd.concat([mailed, pd.DataFrame(new_rows, columns=MAILED_COLS)],
                             ignore_index=True) if not mailed.empty \
            else pd.DataFrame(new_rows, columns=MAILED_COLS)
        save_parquet(drive, index_id, MAILED_NAME, combined)
        log(f"ledger updated: +{len(new_rows)} -> {len(combined)} rows")

    if not load_mail_settings(drive, index_id).get("pf_docs_digest", True):
        log("pf_docs_digest mail toggled OFF — skipped.")
        return
    subject = f"📂 PF new documents — {n_docs} across {len(blocks)} stocks — {date.today()}"
    sent = send_email(subject, html)
    log(f"Email {'sent' if sent else 'FAILED'}: "
        f"{subject.encode('ascii', 'ignore').decode().strip()}")


if __name__ == "__main__":
    main()
