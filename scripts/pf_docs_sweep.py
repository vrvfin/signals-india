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


def hydrate(drive, idx, repo_id, args) -> None:
    """Download the PDF for queued rows that never got one, and fill drive_file_id.

    WHY THIS EXISTS. `--enqueue` writes rows with `drive_file_id=""` — it records the
    document, it does not fetch it. But every extractor skips exactly that case:

        extract_results.py:  drive_fid = str(row.get("drive_file_id") or "").strip()
                             if not drive_fid: log("  SKIP: no drive_file_id"); continue

    and `ingest_company_docs.py` only ever APPENDS newly-discovered feed rows — it never
    revisits an existing pending row to fill the id in. So a swept row could never be
    processed by anything, ever. Measured on Drive 2026-08-14: 30 rows stranded this way
    (24 results, 5 rating, 1 presentation) going back to 31 Jul, including both of
    OBSCP's Q1 FY27 filings — the very documents this sweep was written to rescue.

    Downloading is delegated to `ingest_company_docs.download_pdf`, which carries the
    BSE Akamai workaround (AnnPdfOpen.aspx serves an HTML block page for real PDFs;
    the same attachment is un-blocked under AttachLive/AttachHis). Re-implementing that
    here would mean re-learning it the hard way.
    """
    from _extractor_base import upload_bytes
    # Imported lazily: this pulls the Screener/BSE fetch stack, which the report-only
    # path has no need of.
    from ingest_company_docs import download_pdf, screener_session

    queue = load_queue(drive, idx)
    if queue.empty:
        log("Queue is empty — nothing to hydrate.")
        return
    pend = queue[(queue["status"].astype(str) == "pending")
                 & (queue["pdf_url"].astype(str).str.strip() != "")
                 & (queue["drive_file_id"].astype(str).str.strip() == "")]
    if args.symbols:
        want = {s.strip().upper() for s in args.symbols.split(",") if s.strip()}
        pend = pend[pend["symbol"].astype(str).str.upper().isin(want)]
    if pend.empty:
        log("No queued rows are missing their PDF — nothing to hydrate.")
        return

    log(f"Stranded rows (pending, have a url, no Drive file): {len(pend)}")
    for dt, n in pend["doc_type"].value_counts().items():
        log(f"   {str(dt):<14} {n}")
    idxs = list(pend.index)[: args.limit] if args.limit else list(pend.index)

    if args.dry_run:
        for i in idxs:
            r = queue.loc[i]
            log(f"   WOULD FETCH {str(r['announcement_date'])[:10]}  "
                f"{str(r['symbol']):<12} {str(r['doc_type']):<13} "
                + str(r["title"])[:52].encode("ascii", "ignore").decode())
        log(f"\nDRY RUN — {len(idxs)} document(s) would be downloaded. No writes.")
        return

    if not acquire_lock(drive, idx, _LOCK_NAME, "pf_sweep", max_age_min=360, wait_min=10):
        log("Could not take the extract lock — skipping the write this run.")
        return
    try:
        queue = load_queue(drive, idx)          # re-read under the lock
        session = screener_session()
        done = failed = 0
        for i in idxs:
            if i not in queue.index:
                continue
            r = queue.loc[i]
            if str(r.get("drive_file_id") or "").strip():
                continue                        # another run got there first
            sym = str(r["symbol"])
            try:
                pdf = download_pdf(session, str(r["pdf_url"]))
                if not pdf:
                    # Leave it pending: the url may work later, and marking it done
                    # would bury the document permanently.
                    log(f"  ! {sym:<12} download returned nothing — left pending")
                    failed += 1
                    continue
                key = str(r.get("key") or r.get("isin") or sym)
                comp_id = get_or_create_subfolder(drive, repo_id, key)
                docs_id = get_or_create_subfolder(drive, comp_id, "documents")
                fname = (f"{r['doc_type']}__{str(r['announcement_date'])[:10]}__"
                         f"{r['doc_id']}.pdf")
                fid = upload_bytes(drive, docs_id, fname, pdf, "application/pdf")
                queue.loc[i, "drive_file_id"] = fid
                done += 1
                log(f"  + {sym:<12} {str(r['doc_type']):<13} {len(pdf):,} bytes")
            except Exception as e:
                log(f"  ! {sym:<12} {str(e)[:70]} — left pending")
                failed += 1
            time.sleep(args.sleep)
            # Save as we go: a crash halfway must not throw away the uploads already
            # made, or the next run re-downloads them.
            if done and done % 5 == 0:
                save_queue(drive, idx, queue)
        if done:
            save_queue(drive, idx, queue)
        log(f"Hydrated {done} document(s); {failed} left pending. "
            f"The doc-type extractors can now drain them.")
    finally:
        release_lock(drive, idx, _LOCK_NAME)

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
_DATE_RE = re.compile(r"\b(\d{1,2})\s+([A-Za-z]{3})[a-z]*\.?(?:\s+(\d{4}))?\b")
# Screener's concall table dates a whole ROW ("Aug 2026  Transcript  PPT  REC") rather
# than each link, so month-year with no day has to parse too.
_MONYEAR_RE = re.compile(r"\b([A-Za-z]{3})[a-z]*\.?\s+(\d{4})\b")


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
    """Screener document label -> ISO date. Three shapes, in priority order:

        'Outcome of Board Meeting 12 Aug - ...'   -> 2026-08-12   (year inferred)
        'Rating update 7 Nov 2024 from icra'      -> 2024-11-07   (year EXPLICIT)
        'Aug 2026  Transcript  PPT  REC'          -> 2026-08-01   (concall row, no day)

    THE EXPLICIT YEAR MUST WIN. The old regex stopped at '7 Nov' and inferred the year
    from today, so every document older than ~12 months was dated into the last 12:
    '7 Nov 2024' became 2025-11-07 and '4 Sep 2024' became 2025-09-04. That is harmless
    for a 10-day sweep and fatal for a 6-quarter backfill, which would file historic
    decks under the wrong quarter and poison every deck-vs-deck comparison built on them.

    Only when no year is given is it inferred, and then a month ahead of the current one
    belongs to last year, so a January run does not date a December filing into the future.
    """
    today = today or datetime.now()
    m = _DATE_RE.search(text or "")
    if m:
        day, mon = int(m.group(1)), _MONTHS.get(m.group(2)[:3].lower())
        if mon and 1 <= day <= 31:
            if m.group(3):
                year = int(m.group(3))
            else:
                year = today.year if mon <= today.month else today.year - 1
            try:
                return datetime(year, mon, day).date().isoformat()
            except ValueError:
                return ""
    # Month-year only (the concall table's row label). Day is unknown; the 1st is used,
    # which is precise enough — every consumer maps this to a quarter, and no calendar
    # month straddles a quarter boundary.
    m = _MONYEAR_RE.search(text or "")
    if m:
        mon, year = _MONTHS.get(m.group(1)[:3].lower()), int(m.group(2))
        if mon:
            try:
                return datetime(year, mon, 1).date().isoformat()
            except ValueError:
                return ""
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
        when = parse_date(text, today)
        row_label = ""
        if not when:
            # Screener's concall table dates the ROW, not the link: the cell reads
            # "Aug 2026" and the links inside it are bare "Transcript" / "PPT" / "REC".
            # parse_date("PPT") is empty, and missing_vs_queue drops anything undated —
            # so EVERY historical investor deck in that table was being discarded
            # silently. That is the single largest reason presentation coverage sat at
            # 37% of holdings while APLAPOLLO showed 1 deck against 78 documents.
            parent = a.find_parent(["li", "tr", "div"])
            if parent is not None:
                row_label = parent.get_text(" ", strip=True)[:120]
                when = parse_date(row_label, today)
        title = text[:200]
        if row_label and len(text) < 20:
            # "PPT" alone is not a title anyone can read in a mail or a queue listing.
            title = f"{text} — {row_label}"[:200]
        out.append({"doc_type": doc_type, "title": title, "pdf_url": url,
                    "announcement_date": when})
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
    ap.add_argument("--hydrate", action="store_true",
                    help="Download the PDF for already-queued rows that have a url but "
                         "no Drive file, so the extractors can drain them. Does not "
                         "sweep Screener.")
    ap.add_argument("--limit", type=int, default=0,
                    help="With --hydrate: cap how many documents to fetch this run.")
    # Scope guard. Two reasons this matters, both learned the hard way:
    #  1. Phase-2 concall is P0 — a history backfill has no business adding rows to its
    #     queue as a side effect.
    #  2. Screener's bare "REC" link is an AUDIO RECORDING, and classify() maps it to
    #     concall. Queueing it sends an audio URL to a PDF extractor, which is exactly
    #     the "document queued as a type it can never satisfy" failure that accounts for
    #     the largest share of existing extraction errors.
    ap.add_argument("--types", default="",
                    help="Comma-separated doc_types to sweep (e.g. "
                         "presentation,rating,results). Blank = all types.")
    args = ap.parse_args()

    from screener_client import ScreenerClient
    drive = get_drive()
    root = os.environ["GDRIVE_FOLDER_ID"]
    repo_id = get_or_create_subfolder(drive, root, "company_repo")
    idx = get_or_create_subfolder(drive, repo_id, "_index")

    if args.hydrate:
        hydrate(drive, idx, repo_id, args)
        return

    from daily_brief import load_pf
    pf = load_pf(drive, root, idx)
    if args.symbols:
        want = {s.strip().upper() for s in args.symbols.split(",") if s.strip()}
        pf = [t for t in pf if t[1].upper() in want]
    pf = [t for t in pf if t[1] and not t[1].startswith("INE")]
    if not pf:
        log("No PF companies to sweep.")
        return

    want_types = {t.strip() for t in args.types.split(",") if t.strip()}
    since = (datetime.now() - timedelta(days=args.days)).date().isoformat()
    log(f"PF docs sweep — {len(pf)} holdings, filings since {since}"
        + (f", types={sorted(want_types)}" if want_types else ""))

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
        if want_types:
            miss = [d for d in miss if d["doc_type"] in want_types]
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
        log("Already-queued rows still needing their PDF: use --hydrate.")
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


def _self_test() -> int:
    ok = fail = 0

    def check(name, cond):
        nonlocal ok, fail
        if cond:
            ok += 1
        else:
            fail += 1
            print(f"  FAIL {name}")

    T = datetime(2026, 8, 15)

    # ---- the shape that always worked: day + month, year inferred
    check("day-month infers the current year",
          parse_date("Outcome of Board Meeting 12 Aug - x", T) == "2026-08-12")
    check("a month ahead of today belongs to last year",
          parse_date("Press Release 20 Dec - x", T) == "2025-12-20")

    # ---- THE REGRESSION: an explicit year must win over inference.
    # '7 Nov 2024' was becoming 2025-11-07, filing an 18-month-old document into the
    # last 12 and assigning it the wrong quarter.
    check("explicit year wins (Nov 2024)",
          parse_date("Rating update 7 Nov 2024 from icra", T) == "2024-11-07")
    check("explicit year wins (Sep 2024)",
          parse_date("Rating update 4 Sep 2024 from icra", T) == "2024-09-04")
    check("explicit year still right when it equals the inferred one",
          parse_date("Rating update 30 Sep 2025 from crisil", T) == "2025-09-30")
    check("no year given still infers",
          parse_date("Rating update 26 Feb from icra", T) == "2026-02-26")

    # ---- THE OTHER REGRESSION: the concall table dates the row, not the link.
    check("month-year row label parses", parse_date("Aug 2026 Transcript PPT REC", T)
          == "2026-08-01")
    check("older row label parses", parse_date("Jan 2026 Transcript PPT REC", T)
          == "2026-01-01")
    check("a bare link is still undated on its own", parse_date("PPT", T) == "")
    check("'Financial Year 2025 from bse' has no month, so no date",
          parse_date("Financial Year 2025 from bse", T) == "")

    # ---- classification of the bare labels the concall table uses
    check("bare PPT is a presentation", classify("PPT") == "presentation")
    check("bare Transcript is a concall", classify("Transcript") == "concall")

    # ---- the parent-row fallback, against real Screener markup
    from bs4 import BeautifulSoup
    html = """<div id='documents'><ul>
      <li>Aug 2026 <a href='http://x/t.pdf'>Transcript</a>
                   <a href='http://x/p.pdf'>PPT</a></li>
      <li>Jan 2026 <a href='http://x/p2.pdf'>PPT</a></li>
    </ul></div>"""
    sec = BeautifulSoup(html, "html.parser")

    class _C:
        def fetch_company(self, sym):
            return sec
    docs = scrape_company_docs(_C(), "TEST", today=T)
    ppts = [d for d in docs if d["doc_type"] == "presentation"]
    check("both historical decks recovered", len(ppts) == 2)
    check("deck dated from its row", any(d["announcement_date"] == "2026-08-01" for d in ppts))
    check("older deck dated from its row",
          any(d["announcement_date"] == "2026-01-01" for d in ppts))
    check("bare label gets a readable title",
          all("—" in d["title"] for d in ppts))
    check("no deck left undated", all(d["announcement_date"] for d in ppts))

    print(f"\npf_docs_sweep self-test: {ok} passed, {fail} failed")
    return 1 if fail else 0


if __name__ == "__main__":
    if "--self-test" in sys.argv:
        sys.exit(_self_test())
    main()
