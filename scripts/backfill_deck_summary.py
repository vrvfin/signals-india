#!/usr/bin/env python3
"""backfill_deck_summary — the standalone 18-section read, over decks already filed.

WHY A SEPARATE SCRIPT RATHER THAN --deck-summary ON THE EXTRACTOR.

`extract_presentation.py --deck-summary` runs at INGEST, and it has to: retention rule 1
deletes the source PDF two days after processing. Measured on Drive 2026-08-16 — of the
25 most recent decks, 24 return HTTP 404 and only the one filed that morning still
downloads. So every deck older than two days is unreachable through Drive, and the
extractor's `drive_file_id` path cannot backfill anything.

But the queue keeps `pdf_url` forever. This script goes back to the SOURCE, downloads the
deck again, reads it, and stores only the extracted rows. Scope today: 212 decks across
45 of 51 PF holdings, spanning 2025Q2 to 2026Q3.

THREE THINGS IT DELIBERATELY DOES NOT DO.

  It never uploads a PDF. The bytes live in memory for one call and are discarded, so
  retention rule 1 is satisfied by construction rather than by a cleanup step that has
  to be remembered — there is nothing on Drive to clean up.

  It never mutates `processing_queue`. Those rows are already `done`; their status
  describes the Phase 2 extraction, not this pass. Re-writing them to `pending` to make
  the extractor pick them up would have corrupted the one global ledger (rule 7) and
  invited a genuine re-extract of documents that are already correctly processed.

  It never holds `_extract.lock` across the run. A 212-document pass would hold it for
  hours and starve live Phase 2 (rule 8). The lock is taken per FLUSH — a few seconds
  every `--flush-every` documents — and released immediately.

RESUMABLE BY CONSTRUCTION. Work is selected by "has no deck_summary row for this
source_doc_id", so a run killed by quota exhaustion, a timeout or Ctrl-C loses at most
one flush window, and the next run picks up exactly where it stopped. Free-tier quota
makes this a certainty rather than a precaution: at ~50 requests per project per day,
212 decks will take several runs.

Usage:
    python scripts/backfill_deck_summary.py --dry-run
    python scripts/backfill_deck_summary.py --limit 20
    python scripts/backfill_deck_summary.py --symbols APLAPOLLO,RISHABH
    python scripts/backfill_deck_summary.py --self-test
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime

_SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

import pandas as pd
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(_SCRIPTS_DIR), ".env"))

import deck_summary as DS
from _extractor_base import (get_drive, load_parquet, get_or_create_subfolder,
                             upsert_structured, acquire_lock, release_lock,
                             QUEUE_COLS, GeminiKeyPool, log, RateLimitExhausted)

_LOCK_NAME = "_extract.lock"
DOC_TYPE = "presentation"


def select_work(queue: pd.DataFrame, existing: pd.DataFrame, pf_isins: set,
                since: str, symbols: set | None = None) -> pd.DataFrame:
    """Decks that are in PF, inside the window, have a source url, and have no rows yet.

    `status` is deliberately NOT filtered. A queue row marked `error` failed the Phase 2
    presentation extraction — which says nothing about whether THIS pass can read the
    deck, since the two use different prompts and different output budgets. Excluding
    them would silently drop 12 PF decks whose only sin is that another pass failed.
    `superseded` is kept for the same reason: the deck was still published and still
    describes that quarter.
    """
    if queue is None or queue.empty:
        return pd.DataFrame(columns=QUEUE_COLS)
    q = queue[queue["doc_type"].astype(str) == DOC_TYPE].copy()
    q["_ad"] = pd.to_datetime(q["announcement_date"], errors="coerce")
    q = q[q["_ad"] >= pd.Timestamp(since)]
    q = q[q["isin"].astype(str).str.strip().isin(pf_isins)]
    url = q["pdf_url"].astype(str).str.strip()
    q = q[(url != "") & (url.str.lower() != "nan")]
    if symbols:
        q = q[q["symbol"].astype(str).str.upper().isin({s.upper() for s in symbols})]
    if existing is not None and not existing.empty and "source_doc_id" in existing.columns:
        done = {str(x) for x in existing["source_doc_id"]}
        q = q[~q["doc_id"].astype(str).isin(done)]
    # Newest first: if quota stops the run half way, the reader has the CURRENT quarter
    # rather than an arbitrary slice of history.
    return q.sort_values("_ad", ascending=False)


def quarter_of(announcement_date, fallback: str = "") -> str:
    """The season quarter a deck reports — the quarter that ENDED before it was filed.

    The deck's own self-declared label is not used, and that is on purpose: it is wrong
    often enough to matter (`pf_coverage` measured only 284 of 1,059 deck rows carrying a
    quarter-shaped label at all, the rest being FY26/FY25). The filing date is a fact.
    """
    ts = pd.to_datetime(announcement_date, errors="coerce")
    if pd.isna(ts):
        return fallback
    import quarterly_table as QT
    # season_quarter() is the project's one convention for "which quarter is being
    # reported on this date" — the same helper the mails and coverage use, so a deck and
    # the results it accompanies can never be filed under different quarters.
    return QT.season_quarter(ts.to_pydatetime())


def _flush(drive, idx, rows: list, note: str) -> int:
    """Write a batch under a briefly-held lock. Returns rows written."""
    if not rows:
        return 0
    if not acquire_lock(drive, idx, _LOCK_NAME, "deck_summary_backfill",
                        max_age_min=360, wait_min=10):
        log(f"  could not take the lock — {len(rows)} row(s) held for the next flush")
        return 0
    try:
        upsert_structured(drive, idx, "deck_summary.parquet", DS.DECK_SUMMARY_COLS, rows)
        log(f"  flushed {len(rows)} row(s) to Drive {note}")
        return len(rows)
    finally:
        release_lock(drive, idx, _LOCK_NAME)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true",
                    help="List the work and stop. No download, no Gemini, no writes.")
    ap.add_argument("--limit", type=int, default=0, help="Cap documents this run.")
    ap.add_argument("--symbols", type=str, default="", help="Comma list, e.g. APLAPOLLO,RISHABH")
    ap.add_argument("--since", type=str, default="2025-02-01",
                    help="Earliest announcement date (default ~6 quarters back).")
    ap.add_argument("--flush-every", type=int, default=5,
                    help="Write to Drive every N documents. Smaller = less lost to a "
                         "quota death, more Drive round-trips.")
    ap.add_argument("--key-prefix", type=str, default="FREE_POOL,BACKFILL_GEMINI_KEY",
                    help="Env prefix / comma list for the Gemini pool.")
    ap.add_argument("--deadline-min", type=int, default=0,
                    help="Stop cleanly after N minutes (0 = no deadline).")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    if args.self_test:
        sys.exit(_self_test())

    started = datetime.now()
    drive = get_drive()
    fid = os.environ["GDRIVE_FOLDER_ID"]
    repo = get_or_create_subfolder(drive, fid, "company_repo")
    idx = get_or_create_subfolder(drive, repo, "_index")

    from daily_brief import load_pf
    pf_isins = {str(t[0]).strip() for t in load_pf(drive, fid, idx)}
    queue = load_parquet(drive, idx, "processing_queue.parquet", QUEUE_COLS)
    existing = load_parquet(drive, idx, "deck_summary.parquet", DS.DECK_SUMMARY_COLS)
    symbols = {s.strip() for s in args.symbols.split(",") if s.strip()} or None

    work = select_work(queue, existing, pf_isins, args.since, symbols)
    log(f"PF holdings {len(pf_isins)} | decks needing a summary: {len(work)} "
        f"across {work['symbol'].nunique() if len(work) else 0} symbols "
        f"| already done: {existing['source_doc_id'].nunique() if len(existing) else 0}")

    if work.empty:
        log("Nothing to do.")
        return
    if args.limit:
        work = work.head(args.limit)

    if args.dry_run:
        for _, r in work.iterrows():
            log(f"   WOULD READ {str(r['announcement_date'])[:10]} {str(r['symbol']):<12} "
                + str(r["title"])[:56].encode("ascii", "ignore").decode())
        log(f"\nDRY RUN — {len(work)} deck(s) would be downloaded and read. No writes.")
        return

    from gemini_pool import load_keys_multi
    keys = load_keys_multi(os.environ, args.key_prefix)
    if not keys:
        log(f"ERROR: no keys under {args.key_prefix}")
        sys.exit(1)
    from extract_presentation import GEMINI_MODEL
    gemini = GeminiKeyPool(keys, GEMINI_MODEL)
    prompt = open(os.path.join(_SCRIPTS_DIR, DS.PROMPT_FILE), encoding="utf-8").read()
    from ingest_company_docs import download_pdf, screener_session
    session = screener_session()
    log(f"Loaded {len(keys)} key(s); reading {len(work)} deck(s)")

    pending: list = []
    n_ok = n_fail = n_rows = 0
    for n, (_, r) in enumerate(work.iterrows(), 1):
        sym, doc_id = str(r["symbol"]), str(r["doc_id"])
        if args.deadline_min and (datetime.now() - started).total_seconds() > args.deadline_min * 60:
            log(f"Deadline of {args.deadline_min} min reached — stopping cleanly.")
            break
        try:
            blob = download_pdf(session, str(r["pdf_url"]))
            if not blob or blob[:4] != b"%PDF":
                # A source that no longer serves the file, or serves an HTML block page.
                # Logged and skipped, never sent to Gemini as application/pdf (rule 8).
                log(f"[{n}/{len(work)}] {sym:<12} no usable PDF at source — skipped")
                n_fail += 1
                continue
            qtr = quarter_of(r["announcement_date"])
            out = DS.run_summary(gemini, prompt, blob,
                                 {"isin": r["isin"], "symbol": sym,
                                  "company_name": r.get("company_name", ""), "doc_id": doc_id},
                                 qtr, datetime.now().isoformat(timespec="seconds"))
            rows = out["rows"]
            have, tot = DS.coverage(rows)
            log(f"[{n}/{len(work)}] {sym:<12} {str(r['announcement_date'])[:10]} {qtr:<8} "
                f"{len(rows):>2} rows, {have}/{tot} sections"
                + (f" | dropped {out['dropped']}" if out["dropped"] else ""))
            if rows:
                pending += rows
                n_rows += len(rows)
            n_ok += 1
        except RateLimitExhausted:
            log("All keys rate-limited — flushing and stopping cleanly.")
            break
        except Exception as e:
            log(f"[{n}/{len(work)}] {sym:<12} FAILED: {str(e)[:110]}")
            n_fail += 1
        if len(pending) >= 1 and n % max(1, args.flush_every) == 0:
            if _flush(drive, idx, pending, f"(after {n} decks)"):
                pending = []

    _flush(drive, idx, pending, "(final)")
    log(f"\nDone: {n_ok} deck(s) read, {n_fail} failed, {n_rows} row(s) written. "
        f"Elapsed {(datetime.now() - started).total_seconds() / 60:.1f} min.")


# --------------------------------------------------------------------------- #
def _self_test() -> int:
    ok = fail = 0

    def check(name, cond):
        nonlocal ok, fail
        if cond:
            ok += 1
        else:
            fail += 1
            print(f"  FAIL {name}")

    def Q(**kw):
        base = {"doc_id": "d1", "isin": "INE1", "symbol": "AAA", "company_name": "A",
                "doc_type": "presentation", "title": "Investor presentation",
                "announcement_date": "2026-08-01", "pdf_url": "http://x/a.pdf",
                "drive_file_id": "", "status": "done", "key": "INE1", "description": "",
                "discovered_at": "", "processed_at": "", "attempts": 0,
                "last_error": "", "last_attempt_at": ""}
        return {**base, **kw}

    PF = {"INE1", "INE2"}
    empty = pd.DataFrame(columns=DS.DECK_SUMMARY_COLS)

    q = pd.DataFrame([Q()])
    check("basic selection", len(select_work(q, empty, PF, "2025-02-01")) == 1)
    check("non-PF excluded",
          select_work(pd.DataFrame([Q(isin="INE9")]), empty, PF, "2025-02-01").empty)
    check("other doc types excluded",
          select_work(pd.DataFrame([Q(doc_type="rating")]), empty, PF, "2025-02-01").empty)
    check("no url excluded",
          select_work(pd.DataFrame([Q(pdf_url="")]), empty, PF, "2025-02-01").empty)
    check("nan url excluded",
          select_work(pd.DataFrame([Q(pdf_url="nan")]), empty, PF, "2025-02-01").empty)
    check("outside window excluded",
          select_work(pd.DataFrame([Q(announcement_date="2024-01-01")]), empty, PF,
                      "2025-02-01").empty)
    check("undated row excluded",
          select_work(pd.DataFrame([Q(announcement_date="")]), empty, PF, "2025-02-01").empty)

    # error/superseded are KEPT — a failed Phase 2 extraction says nothing about this pass
    check("error rows kept",
          len(select_work(pd.DataFrame([Q(status="error")]), empty, PF, "2025-02-01")) == 1)
    check("superseded rows kept",
          len(select_work(pd.DataFrame([Q(status="superseded")]), empty, PF, "2025-02-01")) == 1)

    # resumability
    done = pd.DataFrame([{**{c: "" for c in DS.DECK_SUMMARY_COLS}, "source_doc_id": "d1"}])
    check("already-summarised doc skipped", select_work(q, done, PF, "2025-02-01").empty)
    check("a different doc is still selected",
          len(select_work(pd.DataFrame([Q(doc_id="d2")]), done, PF, "2025-02-01")) == 1)

    # newest first, so quota loss costs history not the current quarter
    two = pd.DataFrame([Q(doc_id="old", announcement_date="2025-06-01"),
                        Q(doc_id="new", announcement_date="2026-08-01")])
    check("newest first", list(select_work(two, empty, PF, "2025-02-01")["doc_id"])
          == ["new", "old"])

    # symbol filter
    mix = pd.DataFrame([Q(doc_id="a", symbol="AAA"), Q(doc_id="b", symbol="BBB", isin="INE2")])
    check("symbol filter", list(select_work(mix, empty, PF, "2025-02-01",
                                            {"bbb"})["doc_id"]) == ["b"])
    check("no symbol filter keeps both", len(select_work(mix, empty, PF, "2025-02-01")) == 2)

    check("empty queue safe", select_work(pd.DataFrame(), empty, PF, "2025-02-01").empty)
    check("none queue safe", select_work(None, empty, PF, "2025-02-01").empty)

    # quarter comes from the filing date, never the deck's own label
    check("Aug filing -> the quarter that just ended", quarter_of("2026-08-01") == "Q1FY27")
    check("May filing -> Jan-Mar quarter", quarter_of("2025-05-20") == "Q4FY25")
    check("Nov filing -> Jul-Sep quarter", quarter_of("2025-11-10") == "Q2FY26")
    check("Feb filing -> Oct-Dec quarter", quarter_of("2026-02-05") == "Q3FY26")
    check("undated falls back", quarter_of("", "Q1 FY27") == "Q1 FY27")
    check("garbage date falls back", quarter_of("not a date", "Q1 FY27") == "Q1 FY27")
    # The same convention the mails and coverage use, so a deck and the results it
    # accompanies can never land under different quarter headings.
    import quarterly_table as QT
    check("matches the project's season convention",
          quarter_of("2026-08-01") == QT.season_quarter(datetime(2026, 8, 1)))

    print(f"backfill_deck_summary self-test: {ok} passed, {fail} failed")
    return 1 if fail else 0


if __name__ == "__main__":
    main()
