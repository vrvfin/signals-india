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


def fetch_document(session, url: str) -> tuple[bytes, str, str] | None:
    """Fetch a document URL. Returns (data, mime, ext) or None.

    Handles two shapes:
      • direct PDF (BSE/NSE annual reports)  -> ('%PDF...', application/pdf, .pdf)
      • HTML rating rationale (CRISIL/ICRA)  -> clean text bytes, text/plain, .txt
    """
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
        for a in sub.select("a[href]"):
            href = (a.get("href") or "").strip()
            text = a.get_text(" ", strip=True)
            # skip the section's "All" listing link (not a document)
            if not href or text.lower() == "all" or "corp-announc" in href:
                continue
            is_zip = href.lower().endswith(".zip")   # NSE annual-report archive
            out.append({
                "doc_id":   _doc_id(href),
                "doc_type": doc_type,
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


def fetch_company_page(session, symbol: str) -> str:
    for url in (_SCREENER_CO.format(symbol=symbol),
                _SCREENER_CO_STD.format(symbol=symbol)):
        try:
            r = session.get(url, timeout=30)
        except Exception as e:
            log(f"  fetch error {url}: {str(e)[:80]}")
            continue
        if r.status_code == 200 and 'id="documents"' in r.text:
            return r.text
        log(f"  HTTP {r.status_code} (or no #documents) for {url}")
    return ""


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
             drive=None, repo_id=None, index_id=None) -> dict:
    """Fetch full doc history for one company, queue NEW docs. Returns counts."""
    want_types = want_types or set(SUBSECTION_TYPES.values())
    run_date = dt.date.today()

    own_drive = drive is None
    if own_drive and not dry_run:
        drive = get_drive()
        folder_id = os.environ["GDRIVE_FOLDER_ID"]
        repo_id = get_or_create_subfolder(drive, folder_id, "company_repo")
        index_id = get_or_create_subfolder(drive, repo_id, "_index")

    session = screener_session()
    html = fetch_company_page(session, symbol)
    if not html:
        log(f"  could not fetch Screener page for {symbol}")
        return {"found": 0, "new": 0, "downloaded": 0}

    docs = parse_company_documents(html, run_date, want_types)
    docs.sort(key=lambda d: d["announcement_date"], reverse=True)
    if max_docs:
        docs = docs[:max_docs]
    log(f"  parsed {len(docs)} document(s) for {symbol} "
        f"({', '.join(sorted(want_types))})")

    key = isin if isin else symbol
    counts = {"found": len(docs), "new": 0, "downloaded": 0,
              "dup": 0, "download_fail": 0}

    if dry_run:
        for d in docs:
            print(f"    [{d['doc_type']:<14}] {d['announcement_date']}  "
                  f"{d['title'][:50]}")
        return counts

    queue = load_queue(drive, index_id)
    # Retry previously failed downloads: drop their old rows so they re-attempt,
    # and DON'T treat their keys as known. Successful (pending/done) rows stay known.
    def _key(s): return s["doc_id"].astype(str) + "__" + s["announcement_date"].astype(str).str[:10]
    if not queue.empty:
        failed_mask = queue["status"].astype(str) == "download_failed"
        queue = queue[~failed_mask].reset_index(drop=True)
        known = set(_key(queue)) if not queue.empty else set()
    else:
        known = set()

    new_rows = []
    for d in docs:
        dedup_key = f"{d['doc_id']}__{str(d['announcement_date'])[:10]}"
        if dedup_key in known:
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
        new_rows.append({
            "doc_id": d["doc_id"], "key": key, "isin": isin,
            "symbol": symbol, "company_name": symbol,
            "doc_type": d["doc_type"], "title": d["title"], "description": "",
            "announcement_date": str(d["announcement_date"]),
            "pdf_url": d["pdf_url"], "drive_file_id": drive_file_id,
            "status": status,
            "discovered_at": dt.datetime.now().isoformat(timespec="seconds"),
            "processed_at": "",
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
