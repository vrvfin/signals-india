#!/usr/bin/env python3
"""requeue_thin_ars — re-read annual reports whose stored analysis is a failed generation.

WHY. extract_annual_report used to store whatever the model returned and mark the row
`done`. `done` is terminal — nothing revisits it — so a single bad draw became permanent.
Measured over 51 PF FY2026 annual reports on 2026-09-02 the length distribution is
bimodal and the 1,000–2,500 char band is EMPTY:

      0-500       3      UNIVASTU (0 chars), INDOMIM (108)
    500-1000      5      WELCORP 544, SHILPAMED 642, PRECWIRE 649, RISHABH 725,
                         APLAPOLLO 764  <- ends mid-word: '"Other financial assets" (non-'
   1000-2500      0      <- nothing lives here
   2500-5000     13
  5000-20000     14
    20000-+      16

Eight of fifty-one (16%) hold a failed generation. CG Power's holds the PROMPT echoed
back. A code change cannot repair a stored row — the documents have to be read again.

WHAT THIS DOES. Flips `status` to `pending` for annual_report rows whose stored section on
company_page.md is below --min-chars, or is a prompt echo, so extract_annual_report picks
them up. Nothing else: no deletion, no re-fetch, no writes to any structured table.
`upsert_structured` dedupes on source_doc_id, so re-extraction REPLACES a document's rows
rather than duplicating them.

The gate now in extract_annual_report stops NEW rows going bad; this repairs the old ones.

SAFE TO STOP HALFWAY. A row left `pending` is extracted by the next nightly run anyway, so
an interrupted pass costs nothing and needs no cleanup.

COSTS QUOTA. Each requeued row is a fresh Gemini call on an annual report, the most
expensive document type here. Use --limit and let the nightly backfill spread the work.

Usage:
    python scripts/requeue_thin_ars.py --dry-run
    python scripts/requeue_thin_ars.py --limit 8
    python scripts/requeue_thin_ars.py --symbols APLAPOLLO,UNIVASTU
    python scripts/requeue_thin_ars.py --all-companies --dry-run
    python scripts/requeue_thin_ars.py --self-test
"""
from __future__ import annotations

import argparse
import os
import sys

_D = os.path.dirname(os.path.abspath(__file__))
if _D not in sys.path:
    sys.path.insert(0, _D)

import pandas as pd
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(_D), ".env"))

from _extractor_base import (get_drive, get_or_create_subfolder, load_parquet,
                             save_parquet, QUEUE_COLS, acquire_lock, release_lock,
                             is_prompt_echo, log)

MIN_CHARS = 2000        # matches extract_annual_report.MIN_REPORT_CHARS


def section_chars(sections, period: str, doc_id: str) -> tuple[int, str]:
    """(characters, text) of the stored analysis for one document.

    Uses the digest's region finder, which matches the "<!-- doc:<id> -->" marker
    append_company_page stamps under every annual-report heading. Matching the heading
    LABEL would not work: _extract_fy_year mislabels the year, so APL Apollo's FY2026
    report sits under "## FY22 Annual Report".
    """
    from run_pf_docs_digest import _find_region
    reg = _find_region(sections, period, "annual_report", doc_id)
    if not reg:
        return 0, ""
    text = "\n".join(b for _h, b in reg)
    return len(text), text


def is_thin(n_chars: int, text: str, min_chars: int = MIN_CHARS) -> bool:
    """A stored analysis that is a failed generation rather than a short report."""
    return n_chars < min_chars or is_prompt_echo(text)


def select(queue: pd.DataFrame, isins: set | None,
           symbols: set | None = None) -> pd.DataFrame:
    """Processed annual_report rows that are candidates for a re-read.

    `error` rows are deliberately EXCLUDED, exactly as in requeue_pf_ratings: this is a
    re-read of documents processed WRONGLY, not a retry of ones that failed outright.
    Those carry a recorded reason and requeue_error_docs already cycles them.
    """
    if queue is None or queue.empty:
        return pd.DataFrame(columns=QUEUE_COLS)
    q = queue[(queue["doc_type"].astype(str) == "annual_report")
              & (queue["status"].astype(str).isin(["done", "superseded"]))].copy()
    if isins:
        q = q[q["isin"].astype(str).str.strip().isin(isins)]
    if symbols:
        q = q[q["symbol"].astype(str).str.upper().isin({s.upper() for s in symbols})]
    # Newest first: this year's report matters more than one from 2017, and a run stopped
    # by quota should have spent it on the reports a reader is actually waiting for.
    return q.sort_values("announcement_date", ascending=False)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true",
                    help="Report what would be re-queued. No writes.")
    ap.add_argument("--limit", type=int, default=0,
                    help="Cap rows re-queued this pass (quota control).")
    ap.add_argument("--symbols", default="", help="Restrict to these NSE symbols.")
    ap.add_argument("--min-chars", type=int, default=MIN_CHARS,
                    help=f"Below this many stored characters the analysis is treated as "
                         f"a failed generation (default {MIN_CHARS}).")
    ap.add_argument("--all-companies", action="store_true",
                    help="Scan the whole repo, not just portfolio holdings.")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    if args.self_test:
        sys.exit(_self_test())

    drive = get_drive()
    fid = os.environ["GDRIVE_FOLDER_ID"]
    repo = get_or_create_subfolder(drive, fid, "company_repo")
    idx = get_or_create_subfolder(drive, repo, "_index")

    isins = None
    if not args.all_companies:
        from daily_brief import load_pf
        isins = {str(t[0]).strip() for t in load_pf(drive, fid, idx)}
    queue = load_parquet(drive, idx, "processing_queue.parquet", QUEUE_COLS)
    syms = {s.strip() for s in args.symbols.split(",") if s.strip()} or None

    cand = select(queue, isins, syms)
    if args.limit:
        cand = cand.head(max(args.limit * 6, args.limit))   # scan wider than we requeue
    log(f"annual_report rows to inspect: {len(cand)}")
    if cand.empty:
        log("Nothing to inspect.")
        return

    from run_pf_docs_digest import _company_page
    cache, thin = {}, []
    for _, r in cand.iterrows():
        n, text = section_chars(_company_page(drive, repo, str(r["isin"]), cache),
                                str(r.get("period") or ""), str(r["doc_id"]))
        if is_thin(n, text, args.min_chars):
            thin.append((str(r["symbol"]), str(r["doc_id"]), n,
                         "prompt echo" if is_prompt_echo(text) else f"{n} chars"))
    log(f"failed generations found: {len(thin)} of {len(cand)} inspected")
    if not thin:
        log("Nothing to re-queue.")
        return
    if args.limit:
        thin = thin[: args.limit]

    for sym, _did, _n, why in thin[:20]:
        log(f"   WOULD RE-READ {sym:<12} ({why})")
    if len(thin) > 20:
        log(f"   ... and {len(thin) - 20} more")
    if args.dry_run:
        log(f"DRY RUN — {len(thin)} row(s) would be set to pending. No writes.")
        return

    if not acquire_lock(drive, idx, "_extract.lock", "requeue_thin_ars",
                        max_age_min=360, wait_min=10):
        log("Could not take the lock — nothing changed.")
        sys.exit(1)
    try:
        queue = load_parquet(drive, idx, "processing_queue.parquet", QUEUE_COLS)
        ids = {d for _s, d, _n, _w in thin}
        hit = queue["doc_id"].astype(str).isin(ids)
        queue.loc[hit, "status"] = "pending"
        queue.loc[hit, "processed_at"] = ""
        save_parquet(drive, idx, "processing_queue.parquet", queue)
        log(f"{hit.sum()} annual_report row(s) set to pending — run "
            f"extract_annual_report to re-read them.")
    finally:
        release_lock(drive, idx, "_extract.lock")


def _self_test() -> int:
    ok = fail = 0

    def check(name, cond):
        nonlocal ok, fail
        if cond:
            ok += 1
        else:
            fail += 1
            print(f"  FAIL {name}")

    # ---- the thinness test, against the measured live distribution ----------
    check("0 chars is a failed generation", is_thin(0, ""))
    check("108 chars (INDOMIM) is a failed generation", is_thin(108, "x" * 108))
    check("764 chars (APLAPOLLO) is a failed generation", is_thin(764, "x" * 764))
    check("3,004 chars (the smallest real report) is KEPT",
          not is_thin(3004, "x" * 3004))
    check("a long report is kept", not is_thin(88819, "x" * 88819))
    check("the threshold sits inside the empty band",
          is_thin(999, "x" * 999) and not is_thin(2501, "x" * 2501))
    check("--min-chars is honoured", not is_thin(999, "x" * 999, min_chars=500))
    _echo = ("Generate the final report immediately without displaying preliminary "
             "steps. The ENTIRE report must stay under ~1,200 lines. Output must be "
             "completely clean." + "x" * 9000)
    check("a long PROMPT ECHO is still a failed generation", is_thin(len(_echo), _echo))

    # ---- selection ----------------------------------------------------------
    q = pd.DataFrame([
        {"doc_id": "a", "isin": "I1", "symbol": "AAA", "doc_type": "annual_report",
         "status": "done", "announcement_date": "2026-03-31"},
        {"doc_id": "b", "isin": "I2", "symbol": "BBB", "doc_type": "annual_report",
         "status": "error", "announcement_date": "2026-03-31"},
        {"doc_id": "c", "isin": "I1", "symbol": "AAA", "doc_type": "concall",
         "status": "done", "announcement_date": "2026-08-01"},
        {"doc_id": "d", "isin": "I3", "symbol": "CCC", "doc_type": "annual_report",
         "status": "superseded", "announcement_date": "2025-03-31"},
    ])
    sel = select(q, None)
    check("only annual reports are selected",
          set(sel["doc_id"]) == {"a", "d"})
    check("error rows are excluded, they have their own path",
          "b" not in set(sel["doc_id"]))
    check("newest first", list(sel["doc_id"]) == ["a", "d"])
    check("the portfolio filter applies", set(select(q, {"I1"})["doc_id"]) == {"a"})
    check("the symbol filter applies",
          set(select(q, None, {"ccc"})["doc_id"]) == {"d"})
    check("an empty queue is handled", select(pd.DataFrame(), None).empty)

    print(f"\nrequeue_thin_ars self-test: {ok} passed, {fail} failed")
    return 1 if fail else 0


if __name__ == "__main__":
    main()
