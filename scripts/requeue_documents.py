#!/usr/bin/env python3
"""requeue_documents.py — re-queue named documents, optionally under a different origin.

WHY THIS EXISTS
  Sometimes a specific document has to be read again: its stored analysis is a failed
  generation, or it was processed by the wrong pipeline. requeue_thin_ars.py does this
  for annual reports it judges thin; this does it for documents named EXPLICITLY, of any
  type, and can move them between origins.

  Moving the origin is the point. `source` decides which extractor drains a row:

      "backfill"  -> extract_concall.py --backfill, writing daily_backfill_<date>.md
      anything    -> the live run, writing concall_<date>.md
      else

  So re-queueing a concall as "backfill" sends its re-read down the backfill path and
  keeps its output out of the live daily digest — which is what you want when you are
  repairing history rather than reporting today.

TWO SAFETY RULES THIS FOLLOWS
  1. load_queue / save_queue, NEVER load_parquet(QUEUE_COLS) + save_parquet.
     load_parquet ends in `return df[cols]` — it SLICES — so writing the result back
     erases every column the caller's list does not name. That erased four columns from
     the live queue on 2026-09-05. load_queue returns the frame unsliced.
  2. drive_file_id is CLEARED, because retention deletes the raw PDF two days after it
     is processed. Without clearing it the extractor would look for a file that is gone.
     Re-hydrate with pf_docs_sweep.py --hydrate before extracting.

Usage:
    python scripts/requeue_documents.py --doc-ids a,b,c --dry-run
    python scripts/requeue_documents.py --doc-ids a,b,c --as-origin backfill --live
    python scripts/requeue_documents.py --symbols AVALON --doc-type concall --dry-run
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

from _extractor_base import (get_drive, get_or_create_subfolder, load_queue,  # noqa: E402
                             save_queue, acquire_lock, release_lock, log)


# ---------------------------------------------------------------------------- #
#  --retry-errors: the failures worth reading again                             #
# ---------------------------------------------------------------------------- #
# A DOCUMENT THAT FAILED ONCE USED TO SIT IN 'error' FOR EVER. Nothing re-queued it,
# so a model that echoed the prompt back, or a 503 on a busy afternoon, cost that
# document permanently. 13 PF annual reports were in exactly that state on 2026-09-13
# and had to be re-queued by hand.
#
# THE GENERATION FAILED, NOT THE DOCUMENT. Reading it again plausibly succeeds.
RETRYABLE = (
    "prompt echo", "echoed back",                 # the model answered with the prompt
    "thin report", "degenerate", "repeat",         # a failed generation
    "429", "503", "500", "timeout", "timed out", "deadline", "rate limit",
    "quota", "temporarily", "unavailable", "connection",
)
# THE DOCUMENT FAILED, and reading it again changes nothing until the source does.
# Listed EXPLICITLY rather than inferred as "not retryable", because the inference
# would also sweep in every row whose reason was never recorded — see retryable().
PERMANENT = (
    "not a pdf", "html", "no drive_file_id", "download returned nothing",
    "not an annual report", "needs_ocr", "audio", "encrypted", "0 bytes",
)


def retryable(reason: str) -> bool:
    """Is this failure worth spending a model call on again? A BLANK reason is NOT.

    2,326 historical failures carry no reason at all, and the PF book alone holds 97 of
    them. Re-queueing those would spend free-tier quota on documents that may be
    rights-issue pages, audio recordings or dead links: an UNKNOWN cause is not the
    same as a transient one, and treating it as one is how a retry loop becomes a
    quota fire. Silence means leave it alone.
    """
    s = str(reason or "").strip().lower()
    if not s or s in ("none", "nan", "nat", "<na>"):
        return False
    if any(p in s for p in PERMANENT):
        return False
    return any(p in s for p in RETRYABLE)


def _attempts(value) -> int:
    s = str(value).strip()
    if s in ("", "None", "nan", "<NA>", "NaT"):
        return 0
    try:
        return int(float(s))
    except (TypeError, ValueError):
        return 0


def select_retryable(queue: pd.DataFrame, max_attempts: int, cooldown_h: int,
                     isins: set | None, now=None) -> pd.DataFrame:
    """Error rows worth another read: retryable reason, under the cap, off cooldown.

    THREE GUARDS, EACH CLOSING A DIFFERENT LOOP:
      attempts  - mark_queue_error() increments it on every failure, so the cap is real
                  and a document that cannot be read stops costing quota after N tries.
      cooldown  - a row retried minutes ago must not be retried again in the next run;
                  the mail workflow fires five times a day.
      isins     - the PF book by default. The queue is global and holds thousands of
                  historical failures; draining all of them is a deliberate act.
    """
    now = now or pd.Timestamp.now()
    m = queue["status"].astype(str) == "error"
    if isins is not None:
        m &= queue["isin"].astype(str).isin(isins)
    m &= queue["last_error"].map(retryable)
    m &= queue["attempts"].map(_attempts) < int(max_attempts)
    if cooldown_h and "last_attempt_at" in queue.columns:
        last = pd.to_datetime(queue["last_attempt_at"], errors="coerce")
        m &= last.isna() | (last <= now - pd.Timedelta(hours=int(cooldown_h)))
    return queue[m]


def select(queue: pd.DataFrame, doc_ids: set, symbols: set,
           doc_type: str, since: str) -> pd.DataFrame:
    """Rows to re-queue. doc_ids wins; symbols+doc_type+since is the broader form."""
    if doc_ids:
        return queue[queue["doc_id"].astype(str).isin(doc_ids)]
    m = pd.Series(True, index=queue.index)
    if symbols:
        m &= queue["symbol"].astype(str).str.upper().isin({s.upper() for s in symbols})
    if doc_type:
        m &= queue["doc_type"].astype(str) == doc_type
    if since:
        m &= pd.to_datetime(queue["announcement_date"],
                            errors="coerce") >= pd.Timestamp(since)
    return queue[m]


def _self_test() -> int:
    passed, failed = 0, []

    def check(name, cond):
        nonlocal passed
        if cond:
            passed += 1
        else:
            failed.append(name)

    q = pd.DataFrame({
        "doc_id": ["a", "b", "c", "d"],
        "symbol": ["AVALON", "AVALON", "MOREPENLAB", "TCS"],
        "doc_type": ["concall", "annual_report", "concall", "concall"],
        "status": ["done", "done", "done", "pending"],
        "source": [None, None, None, "backfill"],
        "announcement_date": ["2026-08-11", "2026-03-31", "2026-08-01", "2026-01-01"],
        "drive_file_id": ["f1", "f2", "f3", "f4"],
    })
    check("doc_ids select exactly those rows",
          list(select(q, {"a", "c"}, set(), "", "")["doc_id"]) == ["a", "c"])
    check("symbol + type + since selects the right row",
          list(select(q, set(), {"avalon"}, "concall", "2026-06-01")["doc_id"]) == ["a"])
    check("a doc_type filter excludes other types",
          "b" not in list(select(q, set(), {"AVALON"}, "concall", "")["doc_id"]))
    check("no filters at all selects everything", len(select(q, set(), set(), "", "")) == 4)

    # the write itself
    sel = select(q, {"a", "c"}, set(), "", "")
    out = q.copy()
    out.loc[sel.index, "status"] = "pending"
    out.loc[sel.index, "source"] = "backfill"
    out.loc[sel.index, "drive_file_id"] = ""
    check("selected rows become pending", list(out.loc[sel.index, "status"]) ==
          ["pending", "pending"])
    check("selected rows take the new origin", list(out.loc[sel.index, "source"]) ==
          ["backfill", "backfill"])
    check("drive_file_id is cleared so the PDF is re-fetched",
          list(out.loc[sel.index, "drive_file_id"]) == ["", ""])
    check("untouched rows keep their status", out.loc[3, "status"] == "pending"
          and out.loc[1, "status"] == "done")
    check("untouched rows keep their file id", out.loc[1, "drive_file_id"] == "f2")
    check("no column is lost", list(out.columns) == list(q.columns))

    # ---- --retry-errors: which failures are worth another model call ---------
    check("a prompt echo is retried",
          retryable("prompt echoed back instead of a report"))
    check("a thin report is retried", retryable("thin report: 430 chars (min 2,000)"))
    check("a 503 is retried", retryable("503 Service Unavailable"))
    check("a rate limit is retried", retryable("429 rate limit exceeded"))
    # The guard that stops a retry loop becoming a quota fire.
    check("a BLANK reason is NOT retried", not retryable(""))
    check("the string None is not a reason", not retryable("None")
          and not retryable("nan") and not retryable(None))
    check("a non-PDF is settled, not retried", not retryable("HTML not PDF"))
    check("a missing download is settled",
          not retryable("download returned nothing"))
    check("an unrecognised reason is left alone",
          not retryable("something nobody has classified"))

    check("attempts parses the empty spellings", _attempts("") == 0
          and _attempts(None) == 0 and _attempts("nan") == 0)
    check("attempts parses a real count", _attempts("2") == 2 and _attempts(3.0) == 3)

    _now = pd.Timestamp("2026-09-14 12:00:00")
    _old = (_now - pd.Timedelta(hours=48)).isoformat()
    _recent = (_now - pd.Timedelta(hours=1)).isoformat()
    eq = pd.DataFrame({
        "doc_id": ["e1", "e2", "e3", "e4", "e5", "e6"],
        "isin": ["INE1", "INE1", "INE1", "INE9", "INE1", "INE1"],
        "status": ["error", "error", "error", "error", "done", "error"],
        "last_error": ["prompt echoed back instead of a report",
                       "prompt echoed back instead of a report",
                       "prompt echoed back instead of a report",
                       "prompt echoed back instead of a report",
                       "prompt echoed back instead of a report",
                       None],
        "attempts": [1, 3, 1, 1, 1, 1],
        "last_attempt_at": [_old, _old, _recent, _old, _old, _old],
    })
    got = set(select_retryable(eq, 3, 12, {"INE1"}, now=_now)["doc_id"])
    check("a retryable failure under the cap is selected", "e1" in got)
    check("the attempt CAP is honoured — 3 of 3 is not retried", "e2" not in got)
    check("the COOLDOWN is honoured — retried an hour ago, left alone", "e3" not in got)
    check("a holding outside the PF book is not touched", "e4" not in got)
    check("a row that is not in error is not touched", "e5" not in got)
    check("a failure with no recorded reason is not touched", "e6" not in got)
    check("exactly one row qualifies here", got == {"e1"})
    check("--all-isins reaches the other holding",
          "e4" in set(select_retryable(eq, 3, 12, None, now=_now)["doc_id"]))
    check("a row never attempted is not blocked by the cooldown",
          "n1" in set(select_retryable(pd.DataFrame({
              "doc_id": ["n1"], "isin": ["INE1"], "status": ["error"],
              "last_error": ["503 Service Unavailable"], "attempts": [None],
              "last_attempt_at": [None]}), 3, 12, {"INE1"}, now=_now)["doc_id"]))

    for name in failed:
        print(f"  FAIL  {name}")
    print(f"requeue_documents self-test: {passed} passed, {len(failed)} failed")
    return 1 if failed else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--doc-ids", default="", help="Comma-separated doc_id list.")
    ap.add_argument("--symbols", default="", help="Comma-separated NSE symbols.")
    ap.add_argument("--doc-type", default="", help="concall / annual_report / ...")
    ap.add_argument("--since", default="", help="Only rows announced on/after YYYY-MM-DD.")
    ap.add_argument("--as-origin", default="",
                    help="Set source to this ('backfill' routes the re-read down the "
                         "backfill path). Omit to leave source alone.")
    ap.add_argument("--retry-errors", action="store_true",
                    help="Re-queue every FAILED row whose reason is worth another read "
                         "(a prompt echo, a thin report, a 429/503), under the attempt "
                         "cap and off cooldown. The PF book unless --all-isins.")
    ap.add_argument("--max-attempts", type=int, default=3,
                    help="With --retry-errors: give up after this many failures. "
                         "mark_queue_error increments attempts on each one, so this "
                         "cap is what stops a permanently-broken document costing "
                         "quota for ever.")
    ap.add_argument("--cooldown-hours", type=int, default=12,
                    help="With --retry-errors: leave a row alone for this long after "
                         "its last attempt. The mail workflow runs five times a day; "
                         "without this, one bad document is retried five times a day.")
    ap.add_argument("--all-isins", action="store_true",
                    help="With --retry-errors: the WHOLE queue, not just the PF book. "
                         "Thousands of historical failures — deliberate act only.")
    ap.add_argument("--limit", type=int, default=50, help="Refuse to touch more than N.")
    ap.add_argument("--live", action="store_true", help="Actually write.")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()

    if a.self_test:
        return _self_test()

    doc_ids = {s.strip() for s in a.doc_ids.split(",") if s.strip()}
    symbols = {s.strip() for s in a.symbols.split(",") if s.strip()}
    if not a.retry_errors and not doc_ids and not symbols:
        log("Give --doc-ids or --symbols. Refusing to act on the whole queue.")
        return 1

    drive = get_drive()
    root = os.environ["GDRIVE_FOLDER_ID"]
    idx = get_or_create_subfolder(
        drive, get_or_create_subfolder(drive, root, "company_repo"), "_index")

    queue = load_queue(drive, idx)          # NEVER load_parquet(COLS) — see the docstring
    if a.retry_errors:
        isins = None
        if not a.all_isins:
            from daily_brief import load_pf
            _pf = load_pf(drive, root, idx)
            # load_pf yields (isin, symbol, name) triples, not bare ISINs.
            _rows = _pf if isinstance(_pf, list) else _pf["isin"].tolist()
            isins = {str(t[0] if isinstance(t, (tuple, list)) else t).strip()
                     for t in _rows}
            log(f"scope: the PF book ({len(isins)} holdings)")
        _err = int((queue["status"].astype(str) == "error").sum())
        sel = select_retryable(queue, a.max_attempts, a.cooldown_hours, isins)
        log(f"{_err:,} failed row(s) in the queue; {len(sel)} worth another read "
            f"(under {a.max_attempts} attempts, {a.cooldown_hours}h cooldown)")
        if len(sel) > a.limit:
            log(f"  taking the {a.limit} oldest — --limit bounds the quota spend")
            sel = sel.sort_values("last_attempt_at", na_position="first").head(a.limit)
    else:
        sel = select(queue, doc_ids, symbols, a.doc_type, a.since)
    log(f"matched {len(sel)} row(s)")
    for _, r in sel.iterrows():
        log(f"  {str(r['symbol']):12s} {str(r['doc_type']):14s} "
            f"{str(r['status']):11s} {str(r.get('announcement_date'))[:10]}  "
            f"source={str(r.get('source'))!r}  {str(r['doc_id'])[:22]}")
    if sel.empty:
        return 0
    if not a.retry_errors and len(sel) > a.limit:
        log(f"refusing: {len(sel)} rows exceeds --limit {a.limit}")
        return 1
    if not a.live:
        log("DRY RUN — nothing written. Re-run with --live to apply.")
        return 0

    if not acquire_lock(drive, idx, "_extract.lock", "requeue_documents"):
        log("could not take _extract.lock — nothing written.")
        return 1
    try:
        before_cols = list(queue.columns)
        queue.loc[sel.index, "status"] = "pending"
        queue.loc[sel.index, "drive_file_id"] = ""
        if a.as_origin:
            queue.loc[sel.index, "source"] = a.as_origin.strip().lower()
        assert list(queue.columns) == before_cols, "a column would be lost"
        save_queue(drive, idx, queue)
        log(f"{len(sel)} row(s) set to pending, drive_file_id cleared"
            + (f", source={a.as_origin}" if a.as_origin else "")
            + " — run pf_docs_sweep.py --hydrate, then the extractor for that origin.")
    finally:
        release_lock(drive, idx, "_extract.lock")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
