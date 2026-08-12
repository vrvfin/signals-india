r"""
pf_docs_sweep.py — guarantee that a PF holding's filings are never missed.

THE PROBLEM
-----------
`ingest_company_docs.py` reads Screener's SHARED announcement feeds. Its own comment
states the limit: "Page 1 is a hard 25-item rolling window; same-day completeness comes
from running the pipeline several times a day." On a peak results day hundreds of
companies file, the window turns over faster than the run cadence, and whether YOUR
holding is captured is a race.

Measured on 12 Aug 2026: V Marc filed a board outcome and an investor deck and was
captured; OBSC Perfection filed a board outcome and a press release the same day and was
captured NOWHERE — not the queue, not announcement_ledger, not results.parquet. Both are
NSE SME names with no bse_code, so it was not a symbol-resolution problem. OBSCP simply
lost the race.

THE FIX
-------
Poll PER COMPANY instead. Screener's company page carries a `#documents` section listing
that company's own filings, which cannot be crowded out by other companies. For OBSCP it
shows exactly what the feed missed:

    Press Release            12 Aug - Q1 FY27 revenue rose 84.7% to ...
    Outcome of Board Meeting 12 Aug - Board approved Q1 ...

Deterministic per holding, ~50 requests, no race.

SCOPE
-----
Detect-and-report by default: it compares each PF company's own document list against the
ONE global processing queue (rule 7) and reports what is missing. `--enqueue` additionally
appends the missing rows as `status=pending` for the normal extractors to drain — kept
opt-in because it writes to the live queue.

    python scripts/pf_docs_sweep.py --dry-run          # report only
    python scripts/pf_docs_sweep.py --symbols OBSCP    # one name
    python scripts/pf_docs_sweep.py --enqueue          # also queue what is missing
"""
from __future__ import annotations

import argparse
import hashlib
import os
import re
import sys
import time
from datetime import datetime, timedelta

_SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

import pandas as pd
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(_SCRIPTS_DIR), ".env"))

from _extractor_base import (get_drive, get_or_create_subfolder, load_queue, save_queue,
                             acquire_lock, release_lock, log)

_LOCK_NAME = "_extract.lock"

# Link text -> doc_type. Order matters: first match wins.
DOC_PATTERNS = [
    ("results", ("outcome of board meeting", "financial result", "press release",
                 "unaudited", "audited result", "quarterly result")),
    ("presentation", ("investor presentation", "ppt", "investor deck", "earnings deck")),
    # NOT "rec": too short, it matches inside unrelated words and tagged
    # "Change in Management" filings as concalls. Screener's bare "REC" link is
    # handled by the exact-token check in classify() instead.
    ("concall", ("transcript", "earnings call", "conference call", "concall")),
    ("annual_report", ("annual report", "annual_reports")),
    ("rating", ("rating update", "credit rating", "rating rationale")),
]
# Filings that are not company results/deck material — never queued.
IGNORE = ("newspaper publication", "shareholders meeting", "egm", "agm notice",
          "general updates", "trading window", "share transfer", "all")

_MONTHS = {m: i for i, m in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun",
     "jul", "aug", "sep", "oct", "nov", "dec"], 1)}
_DATE_RE = re.compile(r"\b(\d{1,2})\s+([A-Za-z]{3})\b")


def classify(text: str) -> str:
    low = (text or "").lower().strip()
    if low in ("rec", "ppt", "transcript"):      # Screener's bare link labels
        return {"rec": "concall", "ppt": "presentation",
                "transcript": "concall"}[low]
    if any(k in low for k in IGNORE) and not any(
            k in low for k in ("outcome of board meeting", "financial result")):
        return ""
    for doc_type, keys in DOC_PATTERNS:
        if any(k in low for k in keys):
            return doc_type
    return ""


def parse_date(text: str, today: datetime | None = None) -> str:
    """'Outcome of Board Meeting 12 Aug - ...' -> '2026-08-12'.

    Screener omits the year on recent items. A month ahead of the current one belongs
    to last year, so a January run does not date a December filing into the future.
    """
    today = today or datetime.now()
    m = _DATE_RE.search(text or "")
    if not m:
        return ""
    day, mon = int(m.group(1)), _MONTHS.get(m.group(2)[:3].lower())
    if not mon or not (1 <= day <= 31):
        return ""
    year = today.year if mon <= today.month else today.year - 1
    try:
        return datetime(year, mon, day).date().isoformat()
    except ValueError:
        return ""


def scrape_company_docs(client, symbol: str, today: datetime | None = None) -> list[dict]:
    """That company's own filings from its Screener #documents section."""
    soup = client.fetch_company(symbol)
    sec = soup.find(id="documents")
    if not sec:
        return []
    out, seen = [], set()
    for a in sec.find_all("a"):
        url = (a.get("href") or "").strip()
        text = a.get_text(" ", strip=True)
        if not url or not url.startswith("http") or url in seen:
            continue
        doc_type = classify(text)
        if not doc_type:
            continue
        seen.add(url)
        out.append({"doc_type": doc_type, "title": text[:200], "pdf_url": url,
                    "announcement_date": parse_date(text, today)})
    return out


def missing_vs_queue(docs: list[dict], queue: pd.DataFrame, isin: str,
                     since: str) -> list[dict]:
    """Docs this company has filed that the ONE global queue does not know about.

    Identity is the PDF url — the only stable key across both fetch paths. Rule 7:
    consult the shared queue, never a parallel one.
    """
    if not docs:
        return []
    known = set()
    if queue is not None and not queue.empty and "pdf_url" in queue.columns:
        sub = queue[queue["isin"].astype(str).str.strip() == isin]
        known = {str(u).strip() for u in sub["pdf_url"] if str(u).strip()}
    out = []
    for d in docs:
        if d["pdf_url"] in known:
            continue
        if since and d["announcement_date"] and d["announcement_date"] < since:
            continue
        if not d["announcement_date"]:
            continue
        out.append(d)
    return out


def _doc_id(url: str) -> str:
    return hashlib.md5(url.encode("utf-8")).hexdigest()[:16]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--symbols", default="", help="Comma-separated; default = all PF.")
    ap.add_argument("--days", type=int, default=10,
                    help="Only consider filings this recent (default 10).")
    ap.add_argument("--enqueue", action="store_true",
                    help="Append missing docs to processing_queue as status=pending.")
    ap.add_argument("--dry-run", action="store_true", help="Report only; never write.")
    ap.add_argument("--sleep", type=float, default=1.2)
    args = ap.parse_args()

    from screener_client import ScreenerClient
    drive = get_drive()
    root = os.environ["GDRIVE_FOLDER_ID"]
    idx = get_or_create_subfolder(
        drive, get_or_create_subfolder(drive, root, "company_repo"), "_index")

    from daily_brief import load_pf
    pf = load_pf(drive, root, idx)
    if args.symbols:
        want = {s.strip().upper() for s in args.symbols.split(",") if s.strip()}
        pf = [t for t in pf if t[1].upper() in want]
    pf = [t for t in pf if t[1] and not t[1].startswith("INE")]
    if not pf:
        log("No PF companies to sweep.")
        return

    since = (datetime.now() - timedelta(days=args.days)).date().isoformat()
    log(f"PF docs sweep — {len(pf)} holdings, filings since {since}")

    queue = load_queue(drive, idx)
    client = ScreenerClient()
    found, per_company = [], {}
    for n, (isin, sym, name) in enumerate(pf, 1):
        try:
            docs = scrape_company_docs(client, sym)
        except Exception as e:
            log(f"  ! {sym:<14} fetch failed ({str(e)[:60]})")
            continue
        miss = missing_vs_queue(docs, queue, isin, since)
        if miss:
            per_company[sym] = miss
            for d in miss:
                found.append({**d, "isin": isin, "symbol": sym, "company_name": name})
        if n % 10 == 0:
            log(f"    [{n}/{len(pf)}] missing so far: {len(found)}")
        time.sleep(args.sleep)

    if not found:
        log("Nothing missing — the shared feed caught every PF filing in the window.")
        return

    log(f"\nMISSING from the global queue: {len(found)} doc(s) "
        f"across {len(per_company)} holding(s)")
    for sym, docs in sorted(per_company.items()):
        for d in docs:
            # Titles carry the rupee sign; the local console is cp1252 and print()
            # raises on it. Repo convention: strip to ASCII for logs only.
            log(f"   {d['announcement_date']}  {sym:<14} {d['doc_type']:<13} "
                + d["title"][:62].encode("ascii", "ignore").decode())

    if args.dry_run or not args.enqueue:
        log("\nReport only. Re-run with --enqueue to add these to processing_queue.")
        return

    if not acquire_lock(drive, idx, _LOCK_NAME, "pf_sweep", max_age_min=360, wait_min=10):
        log("Could not take the extract lock — skipping the write this run.")
        return
    try:
        queue = load_queue(drive, idx)          # re-read under the lock
        known = set()
        if not queue.empty and "pdf_url" in queue.columns:
            known = {str(u).strip() for u in queue["pdf_url"] if str(u).strip()}
        now = datetime.now().isoformat(timespec="seconds")
        rows = [{
            "doc_id": _doc_id(d["pdf_url"]), "key": d["isin"], "isin": d["isin"],
            "symbol": d["symbol"], "company_name": d["company_name"],
            "doc_type": d["doc_type"], "title": d["title"], "description": "",
            "announcement_date": d["announcement_date"], "pdf_url": d["pdf_url"],
            "drive_file_id": "", "status": "pending", "discovered_at": now,
            "processed_at": "", "source": "pf_sweep",
        } for d in found if d["pdf_url"] not in known]
        if not rows:
            log("Another run queued these first — nothing to add.")
            return
        out = pd.concat([queue, pd.DataFrame(rows)], ignore_index=True) \
            if not queue.empty else pd.DataFrame(rows)
        save_queue(drive, idx, out)
        log(f"Queued {len(rows)} doc(s) as pending -> processing_queue.parquet "
            f"({len(out):,} rows total)")
    finally:
        release_lock(drive, idx, _LOCK_NAME)


if __name__ == "__main__":
    main()
