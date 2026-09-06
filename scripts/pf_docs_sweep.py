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
                             acquire_lock, release_lock, log, mark_queue_error)

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
                    # NOT A PDF — but that does not make it useless. The extractors call
                    # _extractor_base.call_over_doc, which already auto-detects: bytes
                    # starting with %PDF go to call_pdf, anything else is decoded and
                    # sent to call_text. Rating agencies serve their rationales as HTML
                    # (measured: 22 stuck PF ratings — 17 Fitch, 3 CRISIL, 1 Brickwork,
                    # 1 ICRA), and discarding those bytes here is what stranded them:
                    # no drive_file_id, so every extractor skipped them forever.
                    #
                    # So fetch the raw document and store it as-is. Only genuinely empty
                    # or binary-junk responses are refused.
                    raw = _fetch_raw(session, str(r["pdf_url"]))
                    if raw:
                        pdf = raw
                        log(f"    not a PDF — stored {len(raw):,} bytes as text/HTML "
                            f"(call_over_doc handles it)")
                    else:
                        # Stays PENDING — the url may work later, and marking it done
                        # would bury the document permanently. Record WHY so a
                        # repeatedly unfetchable document is visible, not just untried.
                        mark_queue_error(queue, i, "no bytes returned (PDF or text)",
                                         status="pending")
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
                mark_queue_error(queue, i, str(e), status="pending")
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
# PHRASES, safe to test as substrings.
IGNORE = ("newspaper publication", "shareholders meeting", "agm notice",
          "general updates", "trading window", "share transfer")
# SHORT TOKENS, which must match as WHOLE WORDS. Tested as substrings they were
# catastrophic: "all" matched inside "c-all", so classify() rejected EVERY title
# containing the word call — "Concall transcript", "Earnings Call Transcript",
# "Conference Call Transcript" all returned "". Only Screener's bare "Transcript"
# label survived, via the exact-label branch above, which is why concall discovery
# worked from Screener and could never work from NSE or BSE, whose descriptions are
# full sentences. "egm" had the same flaw: it matches inside "s-egm-ent".
_IGNORE_WORD_RE = re.compile(r"\b(?:all|egm)\b", re.I)

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
    if (any(k in low for k in IGNORE) or _IGNORE_WORD_RE.search(low)) and not any(
            k in low for k in ("outcome of board meeting", "financial result")):
        return ""
    for doc_type, keys in DOC_PATTERNS:
        if any(k in low for k in keys):
            return doc_type
    return ""


# Screener's bare "REC" link is an AUDIO RECORDING, and classify() maps it to concall
# (that mapping is correct - it IS the concall's artefact). Queueing it sends an audio
# URL to a PDF extractor, which is the "document queued as a type it can never satisfy"
# failure that accounts for the largest share of existing extraction errors. classify()
# is left alone because other callers depend on it; the drop happens at the gate instead.
_AUDIO_EXT_RE = re.compile(r"\.(mp3|wav|m4a|aac|ogg|wma)(?:[?#]|$)", re.I)
_AUDIO_LABELS = {"rec", "recording", "audio", "audio recording", "concall recording"}
_AUDIO_TEXT_RE = re.compile(r"audio|recording", re.I)


def is_audio_link(title: str, url: str = "") -> bool:
    """True for a link that is a recording rather than a readable document.

    THE ORDER OF THESE FOUR TESTS IS LOAD-BEARING.

    1. The LEADING label, because scrape_company_docs enriches a bare label with its
       whole parent row: Screener's audio link becomes
       "REC — Aug 2026 Transcript AI Summary PPT REC". That string contains the word
       "transcript", so test 3 would wave it straight through if it ran first.
    2. The url's own extension, for a source that links the media file directly.
    3. "transcript" ANYWHERE is decisive proof of a readable document. Companies file
       "Earnings Call Transcript ... - Transcript of Earning Call", and an audio-word
       test alone would reject the very documents this sweep exists to find.
    4. Only then, an audio word. This is what catches recordings whose titles are full
       sentences rather than a label — measured live 2026-09-02:
         "Audio Recording Of Earning Conference Call Held On 17.08.2026"   (INDSWFTLAB)
         "Update Of Audio Recording For Earnings Conference Call For Q1"   (RISHABH)
         "Analyst / Investor Meet - Outcome 3 Aug - Audio recording of..." (YASHO)
       All three classify() as concall and all three are audio. Sending them to a PDF
       extractor is the largest single class of extraction error in this repo.
    """
    t = str(title or "").lower()
    label = t.split("—")[0].strip().strip("-").strip()
    if label in _AUDIO_LABELS:
        return True
    if _AUDIO_EXT_RE.search(str(url or "")):
        return True
    if "transcript" in t:
        return False
    return bool(_AUDIO_TEXT_RE.search(t))


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


_PUNCT_RE = re.compile(r"[^a-z0-9]+")


def _identity(d: dict) -> tuple:
    """Source-independent identity of a filing: (doc_type, date, normalised title).

    The SAME filing arrives from Screener and from NSE with DIFFERENT urls and different
    ids, so url identity — which is all the queue had — does not collapse them. Measured
    on VMARCIND: 'Outcome of Board Meeting' on 2026-05-11 and 2026-03-23 each appeared
    twice, once per source. Enqueuing both means downloading and Gemini-extracting the
    same document twice, at double the quota, and then rendering it twice in a mail.

    Title is normalised hard (case, punctuation, whitespace) because the two sources
    label the same filing slightly differently.
    """
    t = _PUNCT_RE.sub(" ", str(d.get("title", "")).lower()).strip()
    # ISIN leads the key (user, 2026-08-15). Symbols differ between sources — Screener,
    # NSE and BSE spell the same company differently, and SME names carry no bse_code at
    # all — but the ISIN is the one identifier every source agrees on. Keying on it makes
    # the identity safe to use globally, not just within one company's own document list.
    return (str(d.get("isin", "")).strip(),
            d.get("doc_type", ""), str(d.get("announcement_date", ""))[:10], t)


def quarter_of(date_str: str) -> str:
    """ISO date -> canonical season quarter ('2026-05-14' -> 'Q1FY27')."""
    import quarterly_table as _QT
    try:
        return _QT.norm_q(_QT.season_quarter(pd.to_datetime(str(date_str)[:10])))
    except Exception:
        return ""


def select_one_per_quarter(docs: list[dict], prefer_source: str = "screener") -> list[dict]:
    """Keep ONE document per (doc_type, quarter). Screener wins; NSE fills gaps.

    WHY THIS REPLACES CROSS-SOURCE TITLE MATCHING. Measured across 8 holdings: Screener
    offered 134 documents and NSE 124, with **zero** identity matches between them —
    the same filing is titled 'PPT — May 2026 Transcript AI Summary PPT REC' on Screener
    and 'Investor Presentation' on NSE, and dated 2026-05-01 (month precision, from the
    concall table) versus 2026-05-14 (the exact filing day). No title-and-date key can
    reconcile those, so merging both sources enqueued — and Gemini-extracted — every
    document twice.

    Quarter is the unit that both sources DO agree on, and it is what the mail actually
    needs: one deck per quarter to compare against the previous quarter. Screener's
    concall table is already one row per quarter, which is why it is the primary.

    A quarter with no Screener document falls through to whatever NSE has for it.
    """
    best: dict[tuple, dict] = {}
    for d in docs:
        q = quarter_of(d.get("announcement_date", ""))
        if not q:
            continue
        key = (d.get("doc_type", ""), q)
        cur = best.get(key)
        if cur is None:
            best[key] = {**d, "_quarter": q}
            continue
        # Screener beats NSE; within a source, the longer title carries more context
        # (Screener's "PPT — May 2026 …" over a bare "PPT").
        cur_pref = str(cur.get("source", prefer_source)) == prefer_source
        new_pref = str(d.get("source", prefer_source)) == prefer_source
        if (new_pref and not cur_pref) or (
                new_pref == cur_pref and len(str(d.get("title", ""))) >
                len(str(cur.get("title", "")))):
            best[key] = {**d, "_quarter": q}
    return list(best.values())


def missing_quarters(docs: list[dict], doc_types, quarters: list[str]) -> set:
    """(doc_type, quarter) pairs with no document — what NSE is asked to fill."""
    have = {(d.get("doc_type"), quarter_of(d.get("announcement_date", ""))) for d in docs}
    return {(t, q) for t in doc_types for q in quarters if (t, q) not in have}


def recent_quarters(n: int = 6) -> list[str]:
    """The last n season quarters, newest first."""
    import quarterly_table as _QT
    from datetime import datetime as _dt
    out, d = [], _dt.now()
    for _ in range(n):
        out.append(_QT.norm_q(_QT.season_quarter(d)))
        d = d - timedelta(days=92)
    seen = set()
    return [q for q in out if not (q in seen or seen.add(q))]


def dedupe_across_sources(docs: list[dict]) -> list[dict]:
    """Collapse the same filing seen from more than one source. First seen wins —
    callers add Screener first, which the user set as the primary/cleanest view."""
    out, seen = [], set()
    for d in docs:
        k = _identity(d)
        if k in seen:
            continue
        seen.add(k)
        out.append(d)
    return out


_NSE_ANN = "https://www.nseindia.com/api/corporate-announcements"


def nse_company_docs(session, symbol: str, board: str = "",
                     today: datetime | None = None) -> list[dict]:
    """That company's filings from NSE's per-symbol corporate-announcements API.

    THE INDEX MATTERS. NSE serves SME names from a different index than the mainboard,
    and querying the wrong one returns an empty list rather than an error:
        index=equities  mainboard
        index=sme       NSE Emerge
    Measured 2026-08-15 for OBSCP (NSE Emerge): equities -> 0 rows, sme -> 134 rows.

    WHY NSE AND NOT BSE FOR THESE. All four SME holdings in the portfolio — ANONDITA,
    OBSCP, VMARCIND, AIMTRON — have NO bse_code, so no BSE endpoint can reach them. They
    are also precisely the names the shared Screener feed loses. NSE is the only second
    source that covers them.

    Returns the same dict shape as scrape_company_docs, so both feed one dedupe path.
    """
    if not symbol:
        return []
    idx = "sme" if "sme" in str(board).lower() or "emerge" in str(board).lower() \
        else "equities"
    try:
        r = session.get(f"{_NSE_ANN}?index={idx}&symbol={symbol}", timeout=25)
        rows = r.json()
        if isinstance(rows, dict):
            rows = rows.get("data", [])
    except Exception as e:
        log(f"    NSE fetch failed for {symbol} ({str(e)[:50]})")
        return []

    out = []
    for x in rows or []:
        url = str(x.get("attchmntFile") or "").strip()
        desc = str(x.get("desc") or "").strip()
        if not url.startswith("http"):
            continue
        doc_type = classify(desc)
        if not doc_type:
            continue
        # 'an_dt' is "14-Aug-2026 13:50:24" — a real timestamp, unlike Screener's
        # year-less labels, so no inference is needed here.
        raw = str(x.get("an_dt") or "")[:11].strip()
        when = ""
        try:
            when = datetime.strptime(raw, "%d-%b-%Y").date().isoformat()
        except ValueError:
            when = parse_date(raw, today)
        out.append({"doc_type": doc_type, "title": (desc or url)[:200],
                    "pdf_url": url, "announcement_date": when})
    return out


def _fetch_raw(session, url: str) -> bytes | None:
    """Fetch a document that is NOT a PDF (an HTML rating rationale, typically).

    download_pdf() deliberately returns None for anything without a %PDF header, which
    is right for its own purpose but strands every agency that publishes HTML. The
    extractors are already equipped for this — call_over_doc decodes non-PDF bytes and
    uses the text path — so the bytes just have to reach Drive.

    Refuses empties and obvious binary junk; an HTML error page is still stored, because
    the extractor recording "this said 404" beats a row that looks untried forever.
    """
    try:
        r = session.get(url, timeout=45)
        if r.status_code != 200 or not r.content:
            return None
        body = r.content
        if len(body.strip()) < 200:          # too small to hold a rationale
            return None
        # Reject binary that is neither PDF nor decodable text (images, zips).
        try:
            body[:2048].decode("utf-8")
        except UnicodeDecodeError:
            return None
        return body
    except Exception:
        return None


_BSE_ATTACH = "https://www.bseindia.com/xml-data/corpfiling/AttachLive/"


def bse_company_docs(code: str, days: int) -> list[dict]:
    """That company's filings from BSE's per-scrip announcements API (strScrip=code).

    THE THIRD SOURCE, and each one reaches what the others cannot. Screener is primary
    and cleanest. NSE is the ONLY source that reaches the SME holdings, which have no
    bse_code at all. BSE reaches the mainboard filing at its origin, carries a real
    filing TIMESTAMP where Screener's concall table only dates the row, and covers
    BSE-listed names on the day they file rather than when Screener gets round to it.

    Reuses ingest_announcements.fetch_company_announcements - the same guaranteed
    per-company endpoint that pipeline already relies on - rather than a second raw BSE
    client. Note that pipeline then EXCLUDES annual reports and transcripts by design
    ("Phase 2 owns these"), which is precisely why the documents this sweep wants never
    reached the queue through it.

    Returns the same dict shape as scrape_company_docs, so all three sources feed one
    dedupe path.
    """
    code = str(code or "").strip()
    if not code or code.lower() in ("nan", "none", "0"):
        return []
    try:
        from ingest_announcements import fetch_company_announcements
        rows = fetch_company_announcements(code, lookback_days=max(1, int(days)))
    except Exception as e:
        log(f"    BSE fetch failed for {code} ({str(e)[:60]})")
        return []

    out = []
    for x in rows or []:
        att = str(x.get("ATTACHMENTNAME") or "").strip()
        if not att:
            continue
        # BSE flags the recording itself, so here the audio guard has a first-class
        # field instead of a label heuristic.
        if str(x.get("AUDIO_VIDEO_FILE") or "").strip():
            continue
        head = str(x.get("NEWSSUB") or x.get("HEADLINE") or "").strip()
        sub = str(x.get("SUBCATNAME") or "").strip()
        doc_type = classify(f"{head} {sub}")
        if not doc_type:
            continue
        when = str(x.get("NEWS_DT") or "")[:10]
        if not re.match(r"^\d{4}-\d{2}-\d{2}$", when):
            continue
        out.append({"doc_type": doc_type, "title": (head or sub)[:200],
                    "pdf_url": _BSE_ATTACH + att,
                    "announcement_date": when, "source": "bse"})
    return out


def _bse_map(drive, index_id) -> dict:
    """{isin: bse_code} from company_universe.csv - which scrip to ask BSE about."""
    import io
    from _extractor_base import find_file, download_bytes
    try:
        fid = find_file(drive, index_id, "company_universe.csv")
        if not fid:
            return {}
        u = pd.read_csv(io.BytesIO(download_bytes(drive, fid))).fillna("")
        if "bse_code" not in u.columns:
            log("  company_universe.csv has no bse_code column - BSE source disabled")
            return {}
        out = {}
        for _, r in u.iterrows():
            i, c = str(r.get("isin", "")).strip(), str(r.get("bse_code", "")).strip()
            if i and c and c.lower() not in ("nan", "none", "0"):
                out[i] = c.split(".")[0]        # read back from csv as 539730.0
        return out
    except Exception as e:
        log(f"  bse_code map unavailable ({str(e)[:60]}) - BSE source disabled")
        return {}


def _board_map(drive, index_id) -> dict:
    """{isin: board} from company_universe.csv — decides the NSE index to query."""
    import io
    from _extractor_base import find_file, download_bytes
    try:
        fid = find_file(drive, index_id, "company_universe.csv")
        if not fid:
            return {}
        u = pd.read_csv(io.BytesIO(download_bytes(drive, fid))).fillna("")
        return {str(r["isin"]).strip(): str(r.get("board", ""))
                for _, r in u.iterrows() if str(r.get("isin", "")).strip()}
    except Exception as e:
        log(f"  board map unavailable ({str(e)[:60]}) — NSE defaults to equities")
        return {}


def nse_session():
    """NSE rejects cold requests; hit the homepage first to pick up cookies.
    Same bootstrap earnings_calendar.nse_results_calendar already relies on."""
    import requests
    from earnings_calendar import UA
    s = requests.Session()
    s.headers.update(UA)
    try:
        s.get("https://www.nseindia.com", timeout=15)
    except Exception:
        pass
    return s


def missing_vs_queue(docs: list[dict], queue: pd.DataFrame, isin: str,
                     since: str, since_by_type: dict | None = None) -> list[dict]:
    """Docs this company has filed that the ONE global queue does not know about.

    Identity is the PDF url — the only stable key across both fetch paths. Rule 7:
    consult the shared queue, never a parallel one.

    `since_by_type` narrows the window for particular doc types. Concall needs it: its
    analysis is appended to concall_<TODAY>.md, a digest meaning "the calls processed
    today", so a ten- or thirty-day window tips weeks of backdated filings into one
    day's file. Types absent from the dict keep the wide `since`.
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
        _cut = (since_by_type or {}).get(str(d.get("doc_type") or ""), since)
        if _cut and d["announcement_date"] and d["announcement_date"] < _cut:
            continue
        if not d["announcement_date"]:
            continue
        out.append(d)
    return out


def _self_test_window() -> list:
    """[(name, ok)] — the concall window must be NARROWER than every other type's.

    concall_<TODAY>.md means "the calls processed today". A wide window tips weeks of
    backdated filings into one day's digest; that is the flood that looked like a
    backfill.
    """
    import pandas as _pd
    empty = _pd.DataFrame()
    docs = [
        {"pdf_url": "u1", "doc_type": "concall", "announcement_date": "2026-09-05"},
        {"pdf_url": "u2", "doc_type": "concall", "announcement_date": "2026-08-20"},
        {"pdf_url": "u3", "doc_type": "annual_report",
         "announcement_date": "2026-08-20"},
    ]
    got = missing_vs_queue(docs, empty, "X", "2026-08-15",
                           since_by_type={"concall": "2026-09-04"})
    urls = {d["pdf_url"] for d in got}
    return [
        ("a concall inside its own narrow window is kept", "u1" in urls),
        ("a BACKDATED concall is dropped even though the wide window allows it",
         "u2" not in urls),
        ("another type still uses the wide window", "u3" in urls),
        ("no per-type entry means the wide window applies",
         len(missing_vs_queue(docs, empty, "X", "2026-08-15")) == 3),
    ]


def _doc_id(url: str) -> str:
    return hashlib.md5(url.encode("utf-8")).hexdigest()[:16]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--symbols", default="", help="Comma-separated; default = all PF.")
    ap.add_argument("--concall-days", type=int, default=2,
                    help="Lookback for CONCALL only, in days (default 2). Concall "
                         "analysis is appended to concall_<TODAY>.md, a digest that "
                         "means 'the calls processed today', so a wide window would "
                         "tip a month of backdated filings into one day's file. Every "
                         "other type keeps --days.")
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
    ap.add_argument("--one-per-quarter", action="store_true",
                    help="Keep ONE document per (doc_type, quarter). Screener wins; "
                         "NSE only fills quarters Screener has nothing for.")
    ap.add_argument("--quarters", type=int, default=6,
                    help="How many recent quarters --one-per-quarter/--nse consider.")
    ap.add_argument("--bse", action="store_true",
                    help="Also query BSE's per-scrip announcements as a THIRD source, "
                         "after Screener and NSE, filling only the quarters neither "
                         "covered. Reaches a mainboard filing on the day it is made; "
                         "skipped for holdings with no bse_code (the SME names).")
    ap.add_argument("--nse", action="store_true",
                    help="Also query NSE's per-symbol announcements as a SECOND source "
                         "(index=sme for NSE Emerge names, equities otherwise). The only "
                         "source that reaches the SME holdings, which have no bse_code.")
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
    # CONCALL IS A DAILY FEED, NOT A MONTHLY ONE. Its analysis lands in
    # concall_<TODAY>.md, which means "the calls processed today". Sweeping it on the
    # same 10-30 day window as the other types tipped weeks of backdated filings into a
    # single day's digest - the flood that read like a backfill. Everything else keeps
    # the wide window, because a deck or a rating carries its own date in the mail and
    # does not share a dated digest file.
    concall_days = max(1, min(int(args.concall_days), int(args.days)))
    since_concall = (datetime.now()
                     - timedelta(days=concall_days)).date().isoformat()
    log(f"PF docs sweep — {len(pf)} holdings, filings since {since}"
        + (f" (concall since {since_concall})" if "concall" in want_types else "")
        + (f", types={sorted(want_types)}" if want_types else ""))

    queue = load_queue(drive, idx)
    client = ScreenerClient()
    boards = _board_map(drive, idx) if args.nse else {}
    bse_codes = _bse_map(drive, idx) if args.bse else {}
    nse = nse_session() if args.nse else None
    found, per_company = [], {}
    for n, (isin, sym, name) in enumerate(pf, 1):
        docs = []
        try:
            docs = scrape_company_docs(client, sym)          # PRIMARY: cleanest view
            for _d in docs:
                _d["source"] = "screener"
        except Exception as e:
            log(f"  ! {sym:<14} screener fetch failed ({str(e)[:60]})")

        if args.nse:
            # NSE FILLS GAPS ONLY (user, 2026-08-16). Screener is the primary and is
            # already one row per quarter; NSE is consulted only for the quarters
            # Screener has nothing for. Merging both wholesale enqueued every filing
            # twice — the two sources share no title or exact date, so nothing matched.
            types = want_types or {"presentation", "rating", "results"}
            gaps = missing_quarters(docs, types, recent_quarters(args.quarters))
            if gaps:
                try:
                    ndocs = nse_company_docs(nse, sym, boards.get(isin, ""))
                    for _d in ndocs:
                        _d["source"] = "nse"
                    fill = [d for d in ndocs
                            if (d.get("doc_type"),
                                quarter_of(d.get("announcement_date", ""))) in gaps]
                    if fill:
                        log(f"    {sym}: NSE fills {len(fill)} gap(s) "
                            f"{sorted({q for _t, q in gaps})[:4]}")
                    docs += fill
                except Exception as e:
                    log(f"  ! {sym:<14} nse fetch failed ({str(e)[:60]})")

        if args.bse:
            # BSE FILLS WHAT SCREENER AND NSE BOTH MISSED, on the same gap-only rule -
            # merging a third source wholesale would enqueue every filing a third time,
            # since none of the three share a title or an exact date.
            code = bse_codes.get(isin, "")
            types = want_types or {"presentation", "rating", "results"}
            gaps = missing_quarters(docs, types, recent_quarters(args.quarters))
            if code and gaps:
                try:
                    bdocs = bse_company_docs(code, args.days)
                    fill = [d for d in bdocs
                            if (d.get("doc_type"),
                                quarter_of(d.get("announcement_date", ""))) in gaps]
                    if fill:
                        log(f"    {sym}: BSE fills {len(fill)} gap(s) "
                            f"{sorted({q for _t, q in gaps})[:4]}")
                    docs += fill
                except Exception as e:
                    log(f"  ! {sym:<14} bse fetch failed ({str(e)[:60]})")

        if not docs:
            continue
        for _d in docs:
            _d.setdefault("isin", isin)      # identity keys on ISIN, not symbol
        docs = dedupe_across_sources(docs)
        if args.one_per_quarter:
            docs = select_one_per_quarter(docs)
        miss = missing_vs_queue(docs, queue, isin, since,
                                since_by_type={"concall": since_concall})
        if want_types:
            miss = [d for d in miss if d["doc_type"] in want_types]
        # One gate for BOTH sources (Screener and NSE): never let a recording through.
        _audio = [d for d in miss if is_audio_link(d.get("title"), d.get("pdf_url"))]
        if _audio:
            miss = [d for d in miss if d not in _audio]
            log(f"  - {sym:<14} dropped {len(_audio)} audio link(s) "
                f"(recordings are not extractable documents)")
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
            # SOURCE IS THE PIPELINE ORIGIN, NOT THE DISCOVERY CHANNEL. This sweep
            # fetches documents companies filed in the last few days - they ARE live
            # filings, so they carry the live tag and the live extractors drain them,
            # exactly as a Phase-2-ingested document does.
            #
            # It was briefly tagged "pf_sweep", which nothing downstream understood:
            # every consumer tests only for "backfill" (extract_concall:1661,
            # guidance_digest_email:242, run_growth_guidance_mail:89, requeue_orphans:72,
            # run_backfill:445), so a third value was live-by-accident rather than
            # live-by-design. One tag, meaning what it says.
            "processed_at": "", "source": "live",
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

    # ---- the audio guard: REC classifies as concall but must never be queued
    check("bare REC still classifies as concall", classify("REC") == "concall")

    # ---- "all" and "egm" are WORDS. As substrings they ate the whole concall feed.
    check("'Concall transcript' is a concall",
          classify("Concall transcript") == "concall")
    check("'Earnings Call Transcript' is a concall",
          classify("Earnings Call Transcript Q1 FY27") == "concall")
    check("the full NSE description is a concall",
          classify("Announcement under Regulation 30 (LODR)-Earnings Call Transcript")
          == "concall")
    check("'Conference Call Transcript' is a concall",
          classify("Conference Call Transcript") == "concall")
    # ...and the words they were meant to exclude still are
    check("a bare 'All' label is still ignored", classify("All") == "")
    check("an EGM notice is still ignored", classify("EGM Notice to shareholders") == "")
    check("a trading-window filing is still ignored",
          classify("Trading window closure") == "")
    check("a newspaper publication is still ignored",
          classify("Newspaper Publication of results") == "")
    # ...and the words that merely CONTAIN them are not swept in by the fix
    check("'Postal Ballot' is still not a document type",
          classify("Postal Ballot-Scrutinizer's Report") == "")
    check("'Allotment' is still not a document type",
          classify("Allotment of equity shares") == "")
    check("'Segment' does not trip the egm rule",
          classify("Segment wise Financial Results") == "results")
    check("bare REC is an audio link", is_audio_link("REC", "http://x/rec"))
    check("REC enriched with its parent row is still audio",
          is_audio_link("REC — Aug 2026 Transcript PPT REC", "http://x/r"))
    check("an .mp3 url is audio", is_audio_link("Transcript", "http://x/call.mp3"))
    check("an .mp3 url with a query is audio",
          is_audio_link("Transcript", "http://x/call.mp3?sig=1"))
    check("a real transcript is NOT audio",
          not is_audio_link("Transcript — Aug 2026", "http://x/t.pdf"))
    check("a real deck is NOT audio", not is_audio_link("PPT", "http://x/p.pdf"))
    check("an annual report is NOT audio",
          not is_audio_link("Financial Year 2025 from bse", "http://x/ar.pdf"))
    check("a title containing 'record' is NOT audio",
          not is_audio_link("Record Date for Dividend", "http://x/d.pdf"))

    # ---- audio titles that are SENTENCES, not the bare REC label (live 2026-09-02)
    check("'Audio Recording Of Earning Conference Call' is audio",
          is_audio_link("Audio Recording Of Earning Conference Call Held On 17.08.2026. "
                        "17 Aug - Audio recording of Q1", "http://x/a.pdf"))
    check("'Update Of Audio Recording For Earnings Conference Call' is audio",
          is_audio_link("Update Of Audio Recording For Earnings Conference Call For Q1 "
                        "- FY 2026-27 17 Aug", "http://x/a.pdf"))
    check("an Investor Meet whose payload is an audio recording is audio",
          is_audio_link("Announcement under Regulation 30 (LODR)-Analyst / Investor "
                        "Meet - Outcome 3 Aug - Audio recording of", "http://x/a.pdf"))
    # ...while the real transcripts these sit beside are kept
    check("'Earnings Call Transcript' is NOT audio",
          not is_audio_link("Announcement under Regulation 30 (LODR)-Earnings Call "
                            "Transcript 11 Aug - Please find enclosed", "http://x/t.pdf"))
    check("a transcript that MENTIONS the audio is still a transcript",
          not is_audio_link("Earnings Call Transcript 20 Aug - Transcript of Earning "
                            "Call, audio available on the website", "http://x/t.pdf"))
    # ...and the enriched REC label, which CONTAINS "transcript", is still audio
    check("the enriched REC label is still audio despite containing 'transcript'",
          is_audio_link("REC \u2014 Aug 2026 Transcript AI Summary PPT REC",
                        "http://x/r"))

    # ---- BSE, the third source
    import ingest_announcements as _IA
    _bse_rows = [
        {"ATTACHMENTNAME": "a1.pdf", "NEWSSUB": "Transcript of the earnings call",
         "SUBCATNAME": "", "NEWS_DT": "2026-09-02 15:04:00"},
        {"ATTACHMENTNAME": "a2.pdf", "NEWSSUB": "Reg. 34 (1) Annual Report",
         "SUBCATNAME": "", "NEWS_DT": "2026-09-01 11:00:00"},
        {"ATTACHMENTNAME": "a3.pdf", "NEWSSUB": "Audio recording of the earnings call",
         "SUBCATNAME": "", "NEWS_DT": "2026-09-02 16:00:00",
         "AUDIO_VIDEO_FILE": "call.mp3"},
        {"ATTACHMENTNAME": "", "NEWSSUB": "Transcript with no attachment",
         "SUBCATNAME": "", "NEWS_DT": "2026-09-02 16:00:00"},
        {"ATTACHMENTNAME": "a5.pdf", "NEWSSUB": "Trading window closure",
         "SUBCATNAME": "", "NEWS_DT": "2026-09-02 16:00:00"},
    ]
    _orig = _IA.fetch_company_announcements
    try:
        _IA.fetch_company_announcements = lambda code, lookback_days=30, **k: _bse_rows
        got = bse_company_docs("539730", 30)
    finally:
        _IA.fetch_company_announcements = _orig
    _types = sorted(d["doc_type"] for d in got)
    check("BSE yields the transcript and the annual report",
          _types == ["annual_report", "concall"])
    check("BSE drops the AUDIO_VIDEO_FILE row",
          not any("Audio recording" in d["title"] for d in got))
    check("BSE drops a row with no attachment", len(got) == 2)
    check("BSE builds the AttachLive url",
          all(d["pdf_url"].startswith(_BSE_ATTACH) for d in got))
    check("BSE keeps the real filing date",
          any(d["announcement_date"] == "2026-09-02" for d in got))
    check("BSE tags its source", all(d["source"] == "bse" for d in got))
    check("no bse_code yields nothing", bse_company_docs("", 30) == []
          and bse_company_docs("nan", 30) == [])

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

    # ---- cross-source dedupe: the same filing from Screener and from NSE
    same = [
        {"doc_type": "results", "title": "Outcome of Board Meeting",
         "announcement_date": "2026-05-11", "pdf_url": "https://screener/x.pdf"},
        {"doc_type": "results", "title": "Outcome of  Board   Meeting!",
         "announcement_date": "2026-05-11", "pdf_url": "https://nsearchives/y.pdf"},
        {"doc_type": "results", "title": "Outcome of Board Meeting",
         "announcement_date": "2026-03-23", "pdf_url": "https://screener/z.pdf"},
    ]
    dd = dedupe_across_sources(same)
    check("same filing from two sources collapses to one", len(dd) == 2)
    check("the PRIMARY (first, Screener) url is the one kept",
          dd[0]["pdf_url"].startswith("https://screener/"))
    check("a genuinely different date survives",
          {d["announcement_date"] for d in dd} == {"2026-05-11", "2026-03-23"})
    check("a different doc_type on the same date is not collapsed",
          len(dedupe_across_sources([
              {"doc_type": "results", "title": "T", "announcement_date": "2026-05-11"},
              {"doc_type": "presentation", "title": "T", "announcement_date": "2026-05-11"},
          ])) == 2)

    # THE CONCALL WINDOW: its digest file means "today", so its lookback must be
    # narrower than every other type's. See _self_test_window().
    for _name, _ok in _self_test_window():
        check(_name, _ok)

    print(f"\npf_docs_sweep self-test: {ok} passed, {fail} failed")
    return 1 if fail else 0


if __name__ == "__main__":
    if "--self-test" in sys.argv:
        sys.exit(_self_test())
    main()
