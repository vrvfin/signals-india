r"""
backfill_company_docs.py  —  on-demand FULL document history for one company.

The regular Phase-2 ingester (ingest_company_docs.py) only scans Screener's
cross-company announcement feeds within a 2-day lookback. For a deep dive we need
the company's *entire* history (annual reports going back years, all credit
ratings, all concalls). That history lives on the company's own Screener page in
the #documents section:

    div.documents.annual-reports   -> annual_report  (FY#### links to BSE/NSE PDFs)
    div.documents.credit-ratings   -> rating         (CRISIL/ICRA/CARE updates)
    div.documents.concalls         -> concall         (transcript / notes / ppt)
    div.documents (plain)          -> announcements   (skipped — recent-feed's job)

This script resolves the company, parses that section, downloads any NEW PDFs to
company_repo/<KEY>/documents/, and appends status=pending rows to
processing_queue.parquet — i.e. it feeds the SAME queue the existing doc-type
extractors (extract_concall/extract_results/...) already drain. Fully idempotent:
docs already in the queue are skipped.

Usage:
    python scripts/backfill_company_docs.py --token "venus remedies"
    python scripts/backfill_company_docs.py --symbol VENUSREM --isin INE411B01019
    python scripts/backfill_company_docs.py --token TCS --types annual_report,rating
    python scripts/backfill_company_docs.py --symbol VENUSREM --dry-run
    python scripts/backfill_company_docs.py --symbol VENUSREM --max 8
"""
from __future__ import annotations
import os, sys, re, time, hashlib, argparse, datetime as dt
from pathlib import Path

# Ensure scripts/ on sys.path whether run from repo root or scripts/.
_SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

import pandas as pd
from bs4 import BeautifulSoup
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(_SCRIPTS_DIR), ".env"))

# Reuse the exact ingest helpers so Drive layout + queue schema stay identical.
from ingest_company_docs import (
    get_drive, get_or_create_subfolder, find_file,
    upload_bytes, download_pdf, screener_session,
    load_queue, save_queue, load_symbol_isin_map, QUEUE_COLS, UA,
)


# ICRA serves the rationale as a JS-rendered HTML page whose real content is an
# embedded PDF (the bare page yields only website boilerplate). Rewrite the link
# /Rationale/ShowRationaleReport?Id=<N> -> the PDF endpoint /Rating/ShowRationalReportFilePdf/<N>
# (verified: returns application/pdf). Keeps the original link as the queue identity.
_ICRA_RATIONALE_RE = re.compile(r"icra\.in/.*ShowRationaleReport.*?[?&]Id=(\d+)", re.I)


def _resolve_doc_url(url: str) -> str:
    m = _ICRA_RATIONALE_RE.search(url or "")
    if m:
        return f"https://www.icra.in/Rating/ShowRationalReportFilePdf/{m.group(1)}"
    return url


def _is_drhp_link(title: str, href: str) -> bool:
    """A DRHP/RHP/prospectus link (often SEBI public-issues) surfaced under a company's
    AR subsection — handled by the DRHP pipeline, not the AR backfill."""
    t = (title or "").strip().lower()
    h = (href or "").lower()
    return (t in ("drhp", "rhp") or "prospectus" in t or "red herring" in t
            or "sebi.gov.in/filings/public-issues" in h or "/public-issues/" in h)


def fetch_document(session, url: str) -> tuple[bytes, str, str] | None:
    """Fetch a document URL. Returns (data, mime, ext) or None.

    Handles two shapes:
      • direct PDF (BSE/NSE annual reports)  -> ('%PDF...', application/pdf, .pdf)
      • HTML rating rationale (CRISIL/ICRA)  -> clean text bytes, text/plain, .txt
    ICRA rationale links are resolved to their embedded PDF first.
    """
    url = _resolve_doc_url(url)
    try:
        r = session.get(url, headers={"User-Agent": UA,
                                      "Referer": "https://www.bseindia.com/"},
                        timeout=60, allow_redirects=True)
    except Exception as e:
        log(f"    fetch error: {str(e)[:90]}")
        return None
    if r.status_code != 200 or not r.content:
        log(f"    HTTP {r.status_code}")
        return None
    head = r.content[:5]
    if head.startswith(b"%PDF"):
        return r.content, "application/pdf", ".pdf"
    # NSE annual reports are .zip archives wrapping the AR PDF — extract it.
    if head[:2] == b"PK":
        pdf = _pdf_from_zip(r.content)
        if pdf:
            return pdf, "application/pdf", ".pdf"
        log("    zip had no usable PDF — skipping")
        return None
    ctype = r.headers.get("content-type", "").lower()
    if "html" in ctype or r.content[:14].lower().startswith(b"<html") \
            or b"<html" in r.content[:200].lower():
        soup = BeautifulSoup(r.text, "lxml")
        for s in soup(["script", "style", "noscript"]):
            s.extract()
        text = " ".join(soup.get_text(" ", strip=True).split())
        if len(text) < 200:          # empty / interstitial page, not a real doc
            log(f"    HTML too short ({len(text)} chars) — skipping")
            return None
        return text.encode("utf-8"), "text/plain", ".txt"
    log(f"    unrecognised content ({r.content[:20]!r})")
    return None


def _pdf_from_zip(data: bytes) -> bytes | None:
    """Extract the annual-report PDF from an NSE archive zip. Prefer a file whose
    name looks like the AR (AR_/Annual), else the largest PDF in the archive."""
    import io as _io, zipfile
    try:
        zf = zipfile.ZipFile(_io.BytesIO(data))
    except Exception:
        return None
    pdfs = [n for n in zf.namelist() if n.lower().endswith(".pdf")]
    if not pdfs:
        return None
    pref = [n for n in pdfs if re.search(r"(?i)\bAR[_\s]|annual", n)]
    pick = (pref or sorted(pdfs, key=lambda n: zf.getinfo(n).file_size, reverse=True))[0]
    try:
        return zf.read(pick)
    except Exception:
        return None

# class on the subsection div  ->  queue doc_type
SUBSECTION_TYPES = {
    "annual-reports": "annual_report",
    "credit-ratings": "rating",
    "concalls":       "concall",
}

_SCREENER_CO = "https://www.screener.in/company/{symbol}/consolidated/"
_SCREENER_CO_STD = "https://www.screener.in/company/{symbol}/"


def log(msg: str) -> None:
    print(f"[{dt.datetime.now():%H:%M:%S}] {msg}")


# --------------------------------------------------------------------------- #
#  Parsing
# --------------------------------------------------------------------------- #
def _doc_id(pdf_url: str) -> str:
    """Stable id: BSE Pname param if present, else md5 of the URL (matches ingest)."""
    m = re.search(r"Pname=([^&]+?)(?:\.pdf)?$", pdf_url)
    return m.group(1) if m else hashlib.md5(pdf_url.encode()).hexdigest()[:20]


def _date_from_text(text: str, doc_type: str, run_date: dt.date) -> str:
    """Best-effort document date as ISO string."""
    t = text.strip()
    # Annual report: "Financial Year 2022 from bse" -> FY end 31 Mar of that year
    if doc_type == "annual_report":
        m = re.search(r"(19|20)\d{2}", t)
        if m:
            return f"{m.group(0)}-03-31"
    # Rating / concall: "Rating update 17 Dec 2020 from crisil"
    m = re.search(r"(\d{1,2}\s+[A-Za-z]{3,9}\s+(?:19|20)\d{2})", t)
    if m:
        d = pd.to_datetime(m.group(1), errors="coerce", dayfirst=True)
        if pd.notna(d):
            return d.date().isoformat()
    # bare year fallback
    m = re.search(r"(19|20)\d{2}", t)
    if m:
        return f"{m.group(0)}-03-31"
    return run_date.isoformat()


def _fy_quarter_label(iso_date: str) -> str:
    """Approx Indian-FY (Apr–Mar) quarter a concall/results/presentation announced
    on `iso_date` most likely REPORTS — i.e. the quarter that just ended. Coarse:
    used only for the coverage view + dedup grain. Concall supersede uses Gemini's
    parsed quarter, not this label."""
    d = pd.to_datetime(str(iso_date)[:10], errors="coerce")
    if pd.isna(d):
        return ""
    m, y = d.month, d.year
    if m in (4, 5, 6):       q, fy = 4, y       # reports Jan–Mar (FY ending Mar y)
    elif m in (7, 8, 9):     q, fy = 1, y + 1
    elif m in (10, 11, 12):  q, fy = 2, y + 1
    else:                    q, fy = 3, y       # Jan–Mar -> reports Oct–Dec, FY end Mar y
    return f"Q{q}FY{fy % 100:02d}"


def _period_for(doc_type: str, announcement_date: str) -> str:
    """Natural-grain period label for a document (T12 queue `period` column)."""
    if doc_type == "annual_report":
        m = re.search(r"(19|20)\d{2}", str(announcement_date))
        return f"FY{int(m.group(0)) % 100:02d}" if m else ""
    if doc_type == "rating":
        return str(announcement_date)[:10]      # ratings are dated events
    return _fy_quarter_label(announcement_date)  # concall / results / presentation


def _concall_date_from_div(text: str, run_date: dt.date) -> str:
    """Concall date on the company page lives in the <li>'s leading date <div>
    ('Apr 2026'), NOT in the <a> link text. Parse it to the 1st of that month."""
    d = pd.to_datetime(str(text).strip(), errors="coerce")
    if pd.notna(d):
        return d.date().replace(day=1).isoformat()
    return run_date.isoformat()


def parse_company_documents(html: str, run_date: dt.date,
                            want_types: set[str]) -> list[dict]:
    """Parse the #documents section into a list of document dicts."""
    soup = BeautifulSoup(html, "lxml")
    doc = soup.find(id="documents")
    if not doc:
        return []
    out: list[dict] = []
    for sub in doc.select("div.documents"):
        classes = sub.get("class", [])
        doc_type = next((SUBSECTION_TYPES[c] for c in classes if c in SUBSECTION_TYPES),
                        None)
        if not doc_type or doc_type not in want_types:
            continue

        if doc_type == "concall":
            # Each <li> is one call (one month); date is in the leading <div>, the
            # <a> links (Transcript/PPT/REC) carry no date. Iterate per <li> so each
            # link inherits the correct month — without this every concall fell back
            # to run_date and the date window/dedup collapsed (T12 fix).
            for li in sub.find_all("li"):
                date_div = li.find("div")
                ann = _concall_date_from_div(
                    date_div.get_text(" ", strip=True) if date_div else "", run_date)
                for a in li.select("a[href]"):
                    href = (a.get("href") or "").strip()
                    text = a.get_text(" ", strip=True)
                    if not href or text.lower() == "all" or "corp-announc" in href:
                        continue
                    out.append({
                        "doc_id":   _doc_id(href),
                        "doc_type": "concall",
                        "title":    text,
                        "announcement_date": ann,
                        "pdf_url":  href,
                        "is_zip":   href.lower().endswith(".zip"),
                    })
            continue

        for a in sub.select("a[href]"):
            href = (a.get("href") or "").strip()
            text = a.get_text(" ", strip=True)
            # skip the section's "All" listing link (not a document)
            if not href or text.lower() == "all" or "corp-announc" in href:
                continue
            is_zip = href.lower().endswith(".zip")   # NSE annual-report archive
            # DRHP/prospectus links (esp. for recently-IPO'd names) get surfaced under
            # the AR subsection but are NOT annual reports: SEBI public-issue PDFs that
            # SEBI blocks for bots → they only ever errored in the AR backfill. Tag them
            # 'drhp' so the AR pipeline ignores them; the DRHP pipeline seeds + processes
            # them via non-SEBI prospectus discovery (CLAUDE.md rule 7: DRHP own ledger).
            _dt = ("drhp" if (_is_drhp_link(text, href)) else doc_type)
            out.append({
                "doc_id":   _doc_id(href),
                "doc_type": _dt,
                "title":    text,
                "announcement_date": _date_from_text(text, doc_type, run_date),
                "pdf_url":  href,
                "is_zip":   is_zip,
            })
    # Annual reports: a year may appear as both a BSE PDF and an NSE .zip. Prefer
    # the direct PDF; keep the zip ONLY for years with no PDF (NSE-only companies).
    ar = [d for d in out if d["doc_type"] == "annual_report"]
    pdf_years = {d["announcement_date"] for d in ar if not d["is_zip"]}
    deduped = [d for d in out if not (d["doc_type"] == "annual_report"
                                      and d["is_zip"] and d["announcement_date"] in pdf_years)]
    return deduped


def fetch_company_page(session, symbol: str, bse_code: str = "") -> str:
    """Fetch the Screener company page. Screener accepts EITHER the NSE symbol or the
    BSE scrip code as the URL token — BSE-only companies (no NSE symbol) MUST use the
    bse_code (e.g. screener.in/company/539730/). Tries symbol first, then bse_code."""
    tokens = [t for t in (str(symbol or "").strip(), str(bse_code or "").strip())
              if t and t.lower() != "nan"]
    for tok in tokens:
        for url in (_SCREENER_CO.format(symbol=tok),
                    _SCREENER_CO_STD.format(symbol=tok)):
            try:
                r = session.get(url, timeout=30)
            except Exception as e:
                log(f"  fetch error {url}: {str(e)[:80]}")
                continue
            if r.status_code == 200 and 'id="documents"' in r.text:
                return r.text
            log(f"  HTTP {r.status_code} (or no #documents) for {url}")
    return ""


def _bse_annual_report_docs(bse_code: str, have_years: set[str]) -> list[dict]:
    """BSE-website AR search (api.bseindia.com AnnualReport_New): direct PDF links for
    every filed Annual Report, keyed by year. Used to FILL years the Screener page
    doesn't list (and as the source when Screener is unreachable). Best-effort."""
    code = str(bse_code or "").strip()
    if not code or code.lower() == "nan":
        return []
    out = []
    try:
        import requests as _rq
        r = _rq.get("https://api.bseindia.com/BseIndiaAPI/api/AnnualReport_New/w"
                    f"?scripcode={code}",
                    headers={"User-Agent": UA, "Referer": "https://www.bseindia.com/"},
                    timeout=25)
        if r.status_code != 200:
            return []
        data = r.json()
        items = data.get("Table", data) if isinstance(data, dict) else data
        for it in items or []:
            year = str(it.get("Year") or it.get("year") or "").strip()[:4]
            link = str(it.get("PDFDownload") or it.get("PDF_NAME") or
                       it.get("PDFDownloadNew") or "").strip()
            if not (year.isdigit() and link):
                continue
            ann = f"{year}-03-31"
            if ann in have_years:
                continue                      # Screener already covers this FY
            if not link.startswith("http"):
                link = "https://www.bseindia.com" + ("/" + link.lstrip("/"))
            out.append({"doc_id": _doc_id(link), "doc_type": "annual_report",
                        "title": f"BSE Annual Report {year}", "pdf_url": link,
                        "announcement_date": ann, "is_zip": False})
    except Exception as e:
        log(f"  BSE AR search failed: {str(e)[:70]}")
    if out:
        log(f"  BSE AR search: +{len(out)} year(s) not on Screener page")
    return out


# --------------------------------------------------------------------------- #
#  Resolution
# --------------------------------------------------------------------------- #
def resolve_company(token: str, symbol: str, isin: str,
                    drive, index_id) -> tuple[str, str]:
    """Return (symbol, isin). Prefer explicit flags; else resolve token via universe."""
    if symbol:
        if not isin:
            isin = load_symbol_isin_map(drive, index_id).get(symbol.upper(), "")
        return symbol.upper(), isin
    # resolve token (name / NSE / BSE / ISIN) using the deep-dive resolver
    import company_deep_report as cdr
    svc = cdr.drive_service(); root = os.environ["GDRIVE_FOLDER_ID"]
    universe = cdr._read_csv(svc, cdr.DRIVE["universe"], root)
    r_isin, r_symbol, r_name, _ = cdr.resolve_isin(token, universe)
    return r_symbol.upper(), r_isin


# --------------------------------------------------------------------------- #
#  Backfill core (importable)
# --------------------------------------------------------------------------- #
def backfill(symbol: str, isin: str = "", want_types: set[str] | None = None,
             max_docs: int = 0, dry_run: bool = False,
             drive=None, repo_id=None, index_id=None,
             since: str | None = None, bse_code: str = "") -> dict:
    """Fetch full doc history for one company, queue NEW docs. Returns counts.

    `since` (ISO date, T12): drop documents older than this date BEFORE the
    `max_docs` cap, so a deep request ("10 years") is not truncated to newest-N.
    `bse_code`: Screener URL fallback token for BSE-only companies (no NSE symbol),
    and enables the BSE-website Annual-Report search to fill years Screener misses.
    Defaults keep behaviour byte-for-byte identical for existing callers."""
    want_types = want_types or set(SUBSECTION_TYPES.values())
    run_date = dt.date.today()

    own_drive = drive is None
    if own_drive and not dry_run:
        drive = get_drive()
        folder_id = os.environ["GDRIVE_FOLDER_ID"]
        repo_id = get_or_create_subfolder(drive, folder_id, "company_repo")
        index_id = get_or_create_subfolder(drive, repo_id, "_index")

    session = screener_session()
    html = fetch_company_page(session, symbol, bse_code=bse_code)
    docs = parse_company_documents(html, run_date, want_types) if html else []
    if not html:
        log(f"  could not fetch Screener page for {symbol or bse_code}")
    # BSE-website AR search: fill FY years the Screener page doesn't list (or supply
    # ALL years when Screener is unreachable). Direct-source, same downstream pipeline.
    if "annual_report" in want_types:
        have = {d["announcement_date"] for d in docs if d["doc_type"] == "annual_report"}
        docs += _bse_annual_report_docs(bse_code, have)
    if not docs:
        return {"found": 0, "new": 0, "downloaded": 0}
    docs.sort(key=lambda d: d["announcement_date"], reverse=True)
    if since:
        _floor = str(since)[:10]
        docs = [d for d in docs if str(d["announcement_date"])[:10] >= _floor]
    if max_docs:
        docs = docs[:max_docs]
    log(f"  parsed {len(docs)} document(s) for {symbol} "
        f"({', '.join(sorted(want_types))}"
        f"{', since ' + str(since)[:10] if since else ''})")

    key = isin if isin else symbol
    counts = {"found": len(docs), "new": 0, "downloaded": 0,
              "dup": 0, "download_fail": 0, "drhp_docs": []}

    if dry_run:
        for d in docs:
            print(f"    [{d['doc_type']:<14}] {d['announcement_date']}  "
                  f"{d['title'][:50]}")
        return counts

    queue = load_queue(drive, index_id)
    # Retry previously failed downloads: drop their old rows so they re-attempt,
    # and DON'T treat their keys as known. Successful (pending/done) rows stay known.
    def _key(s): return s["doc_id"].astype(str) + "__" + s["announcement_date"].astype(str).str[:10]
    # CROSS-PATH CONCALL DEDUP: the live recent-feed (ingest_company_docs) and this
    # company-page path compute doc_id/date from DIFFERENT URLs, so the same concall
    # can get two different exact keys → it would be queued (and summarised) twice.
    # A concall is one event per company per month, so we also dedup concalls on a
    # coarse (key, YYYY-MM) regardless of doc_id, across ANY existing queue row
    # (live or backfill, pending or done).
    def _month_key(comp_key, ann): return f"{comp_key}__{str(ann)[:7]}"
    known_concall_month: set[str] = set()
    if not queue.empty:
        # Retry rows whose PDF never landed (download_failed) OR aged out before
        # extraction (expired): drop them so the doc re-downloads + re-queues.
        retry_mask = queue["status"].astype(str).isin(["download_failed", "expired"])
        queue = queue[~retry_mask].reset_index(drop=True)
        known = set(_key(queue)) if not queue.empty else set()
        if not queue.empty:
            cc = queue[queue["doc_type"].astype(str) == "concall"]
            for _, qr in cc.iterrows():
                known_concall_month.add(
                    _month_key(str(qr.get("key") or qr.get("isin") or qr.get("symbol") or ""),
                               qr.get("announcement_date")))
    else:
        known = set()

    new_rows = []
    for d in docs:
        # DRHP/RHP/prospectus links surface under a company's AR subsection but are
        # SEBI public-issue PDFs (SEBI blocks bots). They never enter the global queue
        # (Rule 7 exception) — collect them so run_backfill can seed the DRHP pipeline,
        # which resolves + summarises them via non-SEBI discovery.
        if d.get("doc_type") == "drhp":
            counts["drhp_docs"].append(
                {"title": d.get("title", ""), "url": d.get("pdf_url", ""),
                 "date": str(d.get("announcement_date", ""))[:10]})
            continue
        dedup_key = f"{d['doc_id']}__{str(d['announcement_date'])[:10]}"
        if dedup_key in known:
            counts["dup"] += 1
            continue
        # coarse cross-path guard for concalls (one per company per month)
        if d["doc_type"] == "concall":
            mk = _month_key(key, d["announcement_date"])
            if mk in known_concall_month:
                counts["dup"] += 1
                continue
        drive_file_id, status = "", "pending"
        try:
            fetched = fetch_document(session, d["pdf_url"])
            if fetched is None:
                counts["download_fail"] += 1
                status = "download_failed"
            else:
                data, mime, ext = fetched
                comp_id = get_or_create_subfolder(drive, repo_id, key)
                docs_id = get_or_create_subfolder(drive, comp_id, "documents")
                fname = f"{d['doc_type']}__{d['announcement_date']}__{d['doc_id']}{ext}"
                drive_file_id = upload_bytes(drive, docs_id, fname, data, mime)
                counts["downloaded"] += 1
                log(f"  + {d['doc_type']:<14} {d['announcement_date']}  "
                    f"{d['title'][:40]} ({ext})")
        except Exception as exc:
            log(f"  ! {d['doc_type']:<14} error ({str(exc)[:80]}) — skip")
            continue
        known.add(dedup_key)
        if d["doc_type"] == "concall":
            known_concall_month.add(_month_key(key, d["announcement_date"]))
        new_rows.append({
            "doc_id": d["doc_id"], "key": key, "isin": isin,
            "symbol": symbol, "company_name": symbol,
            "doc_type": d["doc_type"], "title": d["title"], "description": "",
            "announcement_date": str(d["announcement_date"]),
            "pdf_url": d["pdf_url"], "drive_file_id": drive_file_id,
            "status": status,
            "discovered_at": dt.datetime.now().isoformat(timespec="seconds"),
            "processed_at": "",
            "source": "backfill",      # enqueue origin -> backfill extractor only
            "period": _period_for(d["doc_type"], d["announcement_date"]),  # T12
            "content_sha256": "",      # T12 reserved (Stage C)
        })
        counts["new"] += 1
        time.sleep(0.4)             # polite to BSE/CRISIL

    if new_rows:
        queue = pd.concat([queue, pd.DataFrame(new_rows)], ignore_index=True)
        save_queue(drive, index_id, queue)

    return counts


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--token", default="",
                    help="Company name / NSE / BSE / ISIN (resolved via universe)")
    ap.add_argument("--symbol", default="", help="Screener/NSE symbol (skips resolution)")
    ap.add_argument("--isin", default="", help="ISIN (folder key); looked up if omitted")
    ap.add_argument("--types", default="",
                    help="Comma list of doc types to pull "
                         "(annual_report,rating,concall). Default: all.")
    ap.add_argument("--max", type=int, default=0, help="Cap newest N docs (0=all)")
    ap.add_argument("--dry-run", action="store_true",
                    help="List documents found; no download, no Drive, no queue.")
    args = ap.parse_args()

    if not args.token and not args.symbol:
        ap.error("provide --token or --symbol")

    want_types = (set(t.strip() for t in args.types.split(",") if t.strip())
                  if args.types else set(SUBSECTION_TYPES.values()))

    print("Backfill — full company document history")
    print("-" * 56)

    drive = repo_id = index_id = None
    if not args.dry_run:
        drive = get_drive()
        folder_id = os.environ["GDRIVE_FOLDER_ID"]
        repo_id = get_or_create_subfolder(drive, folder_id, "company_repo")
        index_id = get_or_create_subfolder(drive, repo_id, "_index")

    symbol, isin = resolve_company(args.token, args.symbol, args.isin, drive, index_id)
    if not symbol:
        sys.exit("Could not resolve a Screener symbol.")
    log(f"Company: {symbol}  ISIN={isin or '(unknown)'}")

    counts = backfill(symbol, isin, want_types, args.max, args.dry_run,
                      drive, repo_id, index_id)

    print("-" * 56)
    print(f"Documents found on Screener : {counts['found']}")
    if not args.dry_run:
        print(f"Already in queue (dup)      : {counts.get('dup', 0)}")
        print(f"New documents queued        : {counts['new']}")
        print(f"  PDFs downloaded           : {counts['downloaded']}")
        print(f"  download failures         : {counts.get('download_fail', 0)}")
    print("Next: run the doc-type extractors (or deep dive) to summarise.")


if __name__ == "__main__":
    main()
