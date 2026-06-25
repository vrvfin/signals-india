r"""
extract_rating_parallel.py — Phase 3: PARALLEL alt-provider drain for credit ratings.

WHY THIS SHAPE (workable, not the fragile one):
- Gemini's grpc client + pyarrow are NOT thread-safe → the old `--workers 8` in-process
  thread attempt SEGFAULTED in CI. So this drain uses ALT providers only (Groq/Cerebras =
  plain HTTP = thread-safe, proven 3.8x in eval) for the slow LLM calls, and keeps EVERY
  Drive operation (download + all writes) on the MAIN thread (the Drive httplib2 client is
  also not thread-safe). Result: the LLM calls overlap (throughput) with zero shared-write
  race and zero Gemini-threading risk.
- It REUSES the live rating extractor's functions (parse/ tabulate/ upsert) — no logic
  duplication — and does NOT touch extract_rating.py, so the Gemini path is unchanged.
- It runs in its OWN CI slot (own _extract.lock hold), adding an independent extraction
  pass on independent Groq/Cerebras quota. It does not need to run concurrently with the
  Gemini lane, so there is no lock-model change and no Phase-2 risk.

Usage:
  python scripts/extract_rating_parallel.py --workers 4 --limit 20 --dry-run
  python scripts/extract_rating_parallel.py --workers 4 --max-age-hours 48 --deadline-min 30
"""
from __future__ import annotations

import argparse
import atexit
import os
import sys
import time
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(_HERE), ".env"))

from pathlib import Path
import extract_rating as er          # reuse parse/tabulate/upsert + constants (no duplication)
from _extractor_base import (
    RateLimitExhausted, get_drive, log, get_or_create_subfolder,
    load_queue, save_queue, download_bytes, append_company_page,
    upsert_structured, call_over_doc,
)
from provider_router import build_alt_pools, AltOnlyPool

_LOCK_NAME = er._LOCK_NAME
DOC_TYPE = er.DOC_TYPE


def _llm_work(altpool, prompt, struct_prompt, row, pdf_bytes):
    """Runs in a WORKER THREAD: alt LLM call + pure parse + structured alt call. No Drive."""
    name = f"{row.get('symbol', 'DOC')}_{str(row.get('doc_id', ''))[:12]}.pdf"
    md = call_over_doc(altpool, prompt, pdf_bytes, name=name,
                       max_output_tokens=er.RATING_MAX_OUTPUT_TOKENS)
    facts = er.parse_gemini_response(md, row)
    now = datetime.now().isoformat(timespec="seconds")
    dr = co = se = None
    try:
        dr, co, se = er.tabulate_rating(altpool, struct_prompt, pdf_bytes, row,
                                        facts.get("agency"), facts.get("rating_date"), now)
    except Exception:
        dr = co = se = None      # tabulation is best-effort; markdown + ratings still land
    return md, facts, dr, co, se


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4, help="parallel alt LLM calls (default 4)")
    ap.add_argument("--limit", type=int, default=0, help="max docs this run (0 = all pending)")
    ap.add_argument("--max-age-hours", type=float, default=0.0,
                    help="only docs discovered within N hours (quota guard)")
    ap.add_argument("--deadline-min", type=float, default=0.0,
                    help="release lock + exit before the CI job timeout")
    ap.add_argument("--dry-run", action="store_true", help="download + extract + print; NO writes")
    args = ap.parse_args()

    alt = build_alt_pools(os.environ)
    if not alt:
        print("ERROR: no Groq/Cerebras keys (GROQ_API_KEY / CEREBRAS_API_KEY) — nothing to do.")
        sys.exit(1)
    altpool = AltOnlyPool(alt)
    log(f"Alt-only parallel drain: providers={sorted(alt)}, workers={args.workers}")

    prompt = (Path(_HERE) / er.PROMPT_FILE).read_text(encoding="utf-8")
    sp = Path(_HERE) / er.STRUCT_PROMPT_FILE
    struct_prompt = sp.read_text(encoding="utf-8") if sp.exists() else ""

    drive = get_drive()
    folder_id = os.environ["GDRIVE_FOLDER_ID"]
    repo_id = get_or_create_subfolder(drive, folder_id, "company_repo")
    index_id = get_or_create_subfolder(drive, repo_id, "_index")

    if not args.dry_run:
        from _extractor_base import acquire_lock, release_lock
        if not acquire_lock(drive, index_id, _LOCK_NAME, DOC_TYPE,
                            max_age_min=er._LOCK_MAX_AGE_MIN, defer_to_phase2=True):
            log("  Lock unavailable (yielded to Phase 2) — exiting cleanly.")
            sys.exit(0)
        atexit.register(release_lock, drive, index_id, _LOCK_NAME)

    queue = load_queue(drive, index_id)
    pend = queue.index[(queue["status"] == "pending") & (queue["doc_type"] == DOC_TYPE)].tolist()
    if args.max_age_hours and "discovered_at" in queue.columns:
        cut = (datetime.now() - timedelta(hours=args.max_age_hours)).isoformat(timespec="seconds")
        pend = [i for i in pend if str(queue.loc[i, "discovered_at"]) >= cut]
    if args.limit:
        pend = pend[: args.limit]
    log(f"Queue: {len(queue)} rows, {len(pend)} pending {DOC_TYPE} to drain (parallel).")
    if not pend:
        return

    counts = {"processed": 0, "error": 0, "skipped": 0, "deferred": 0}
    t0 = time.time()
    stop = False
    # Batches of `workers`: main thread downloads (Drive not thread-safe), threads do the
    # LLM calls in parallel, main thread writes results sequentially (no write race).
    for b in range(0, len(pend), max(1, args.workers)):
        if stop:
            break
        if args.deadline_min and (time.time() - t0) / 60 >= args.deadline_min:
            log(f"  Deadline {args.deadline_min:.0f} min reached — exiting cleanly.")
            break
        batch = pend[b: b + max(1, args.workers)]

        jobs = []   # (queue_idx, row, pdf_bytes)
        for idx in batch:
            row = queue.loc[idx]
            fid = str(row.get("drive_file_id") or "").strip()
            if not fid:
                counts["skipped"] += 1
                continue
            try:
                pdf = download_bytes(drive, fid)        # MAIN thread (Drive)
            except Exception as e:
                log(f"  download failed {row.get('symbol')}: {str(e)[:70]} — skip")
                counts["skipped"] += 1
                continue
            jobs.append((idx, row, pdf))
        if not jobs:
            continue

        results = {}
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            fut = {ex.submit(_llm_work, altpool, prompt, struct_prompt, r, p): (i, r, p)
                   for (i, r, p) in jobs}
            for f in fut:
                results[fut[f][0]] = (fut[f], f)

        for idx, (meta, f) in results.items():
            _, row, pdf = meta
            sym = row.get("symbol", "?")
            try:
                md, facts, dr, co, se = f.result()
            except RateLimitExhausted:
                log(f"  {sym}: all alt providers exhausted — deferring rest.")
                counts["deferred"] += 1
                stop = True
                continue
            except Exception as e:
                log(f"  ERROR {sym}: {str(e)[:100]}")
                if not args.dry_run:
                    queue.loc[idx, "status"] = "error"
                    queue.loc[idx, "processed_at"] = datetime.now().isoformat(timespec="seconds")
                counts["error"] += 1
                continue

            if args.dry_run:
                print(f"  [dry] {sym:<12} {len(md):>5} chars  "
                      f"agency={facts.get('agency')} rating={facts.get('rating')} "
                      f"(drivers={len(dr or [])}, concerns={len(co or [])})")
                counts["processed"] += 1
                continue

            # ---- MAIN-thread writes (no race) ----
            key = str(row.get("key") or row.get("isin") or row.get("symbol") or "")
            if er.OUTPUT_COMPANY_MD:
                append_company_page(drive, repo_id, key=key, doc_type_label=er.DOC_TYPE_LABEL,
                                    content=md, doc_title=str(row.get("title", "")),
                                    quarter=f"{facts.get('agency')} {facts.get('rating')}")
            er.upsert_ratings(drive, index_id, facts)
            if dr is not None:
                upsert_structured(drive, index_id, "rating_drivers.parquet",
                                  er.RATING_DRIVERS_COLS, dr)
                upsert_structured(drive, index_id, "rating_concerns.parquet",
                                  er.RATING_CONCERNS_COLS, co)
                upsert_structured(drive, index_id, "rating_sensitivity.parquet",
                                  er.RATING_SENSITIVITY_COLS, se)
            queue.loc[idx, "status"] = "done"
            queue.loc[idx, "processed_at"] = datetime.now().isoformat(timespec="seconds")
            counts["processed"] += 1
            log(f"  done {sym} via alt (drivers={len(dr or [])})")

        if not args.dry_run:
            save_queue(drive, index_id, queue)          # one write per batch

    if not args.dry_run:
        from _extractor_base import persist_gemini_usage
        persist_gemini_usage(drive, index_id, altpool.summary(), DOC_TYPE, "backfill_alt")

    dt = time.time() - t0
    print("-" * 56)
    print(f"Processed (alt): {counts['processed']}   errors: {counts['error']}   "
          f"skipped: {counts['skipped']}   deferred: {counts['deferred']}")
    print(f"Elapsed: {dt:.1f}s  ({dt / max(1, counts['processed']):.1f}s/doc effective)")


if __name__ == "__main__":
    main()
