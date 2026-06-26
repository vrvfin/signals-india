r"""
extract_concall_parallel.py — Phase 3: PARALLEL alt-provider drain for BACKFILL concalls.

WHY: measured 2026-06-26 — the alt lane (Groq/Cerebras) was producing ZERO docs while
~147 FRESH concalls sat pending (Gemini's daily quota can't reach them). This drain chews
that backlog on INDEPENDENT alt quota, in parallel.

SAFETY / SCOPE:
- Does NOT touch extract_concall.py — the live Phase-2 concall path is untouched (it is
  off-limits). This is a separate, BACKFILL-ONLY catch-up drain.
- Threads ONLY the Groq/Cerebras HTTP calls (thread-safe); ALL Drive I/O on the main thread
  (httplib2 is not thread-safe). No shared-write race.
- Per-doc writes (no batching). company_page goes through _extractor_base.append_company_page
  with a doc_id dedup_marker → idempotent (same header format as the live concall: "## <Q>
  Concall — <title>"). The parquet upserts dedupe by source_doc_id. So a kill is safe.
- DUP/SUPERSEDE: this drain only FILLS quarters not already covered. If (isin, quarter) is
  already in quarterly_facts, it skips (the live Gemini extractor owns supersede/replace).
- Reuses concall's parse_gemini_response / parse_gf_sections / upsert_* verbatim → output
  matches the Gemini path. Skips the optional day-digest, quarterly-guidance page, and
  GF_TRACK history (catch-up drain; the core company_page + facts + GF parquets are written).

Usage:
  python scripts/extract_concall_parallel.py --workers 6 --limit 40 --dry-run
  python scripts/extract_concall_parallel.py --workers 6 --max-age-hours 48 --deadline-min 60
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
import extract_concall as ec          # reuse parse + upserts + constants (no duplication)
from _extractor_base import (
    RateLimitExhausted, get_drive, log, get_or_create_subfolder,
    load_queue, save_queue, download_bytes, load_parquet, append_company_page,
    call_over_doc, acquire_lock, release_lock,
)
from provider_router import build_alt_pools, AltOnlyPool

DOC_TYPE = "concall"
_LOCK_NAME = "_extract.lock"
_LOCK_MAX_AGE_MIN = 360
CONCALL_MAX_OUT = 8192               # concall summaries are long; give room


def _llm_work(altpool, prompt, row, pdf_bytes):
    """WORKER THREAD: alt LLM call + pure parses. No Drive."""
    name = f"{row.get('symbol', 'DOC')}_{str(row.get('doc_id', ''))[:12]}.pdf"
    md = call_over_doc(altpool, prompt, pdf_bytes, name=name, max_output_tokens=CONCALL_MAX_OUT)
    facts, guidance_rows = ec.parse_gemini_response(md, row)
    quarter = facts.get("quarter") or ""
    now_str = datetime.now().isoformat(timespec="seconds")
    gf1 = gf2 = gf3 = gf4 = []
    try:
        gf1, gf2, gf3, gf4 = ec.parse_gf_sections(md, row, quarter, now_str)
    except Exception:
        gf1 = gf2 = gf3 = gf4 = []
    return md, facts, guidance_rows, (gf1, gf2, gf3, gf4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-age-hours", type=float, default=48.0,
                    help="only concalls discovered within N hours (their PDF still exists)")
    ap.add_argument("--deadline-min", type=float, default=0.0)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    alt = build_alt_pools(os.environ)
    if not alt:
        print("ERROR: no Groq/Cerebras keys — nothing to do.")
        sys.exit(1)
    altpool = AltOnlyPool(alt)
    log(f"Concall alt drain: providers={sorted(alt)}, workers={args.workers}")

    prompt = (Path(_HERE) / "concall_prompt.txt").read_text(encoding="utf-8")
    drive = get_drive()
    folder_id = os.environ["GDRIVE_FOLDER_ID"]
    repo_id = get_or_create_subfolder(drive, folder_id, "company_repo")
    index_id = get_or_create_subfolder(drive, repo_id, "_index")

    if not args.dry_run:
        if not acquire_lock(drive, index_id, _LOCK_NAME, DOC_TYPE,
                            max_age_min=_LOCK_MAX_AGE_MIN, defer_to_phase2=True):
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
    log(f"Queue: {len(queue)} rows, {len(pend)} fresh pending {DOC_TYPE} to drain.")
    if not pend:
        return

    # Dup guard: skip quarters already covered (live Gemini owns supersede/replace).
    import pandas as pd
    qf = load_parquet(drive, index_id, "quarterly_facts.parquet", ec.QFACTS_COLS)
    covered = set()
    if not qf.empty and {"isin", "quarter"} <= set(qf.columns):
        covered = {(str(a), ec._norm_quarter(str(b)))
                   for a, b in zip(qf["isin"], qf["quarter"])}
    seen_run: set = set()

    counts = {"processed": 0, "error": 0, "skipped": 0, "dup": 0, "deferred": 0}
    t0 = time.time()
    stop = False
    for b in range(0, len(pend), max(1, args.workers)):
        if stop or (args.deadline_min and (time.time() - t0) / 60 >= args.deadline_min):
            if args.deadline_min:
                log(f"  Deadline {args.deadline_min:.0f} min — exiting cleanly.")
            break
        batch = pend[b: b + max(1, args.workers)]
        jobs = []
        for idx in batch:
            row = queue.loc[idx]
            fid = str(row.get("drive_file_id") or "").strip()
            if not fid:
                counts["skipped"] += 1
                continue
            try:
                pdf = download_bytes(drive, fid)        # MAIN thread (Drive)
            except Exception as e:
                log(f"  download failed {row.get('symbol')}: {str(e)[:60]} — skip")
                counts["skipped"] += 1
                continue
            jobs.append((idx, row, pdf))
        if not jobs:
            continue

        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(_llm_work, altpool, prompt, r, p): (i, r) for (i, r, p) in jobs}
            done = [(futs[f][0], futs[f][1], f) for f in futs]

        for idx, row, f in done:
            sym = row.get("symbol", "?")
            try:
                md, facts, guidance_rows, (gf1, gf2, gf3, gf4) = f.result()
            except RateLimitExhausted:
                log(f"  {sym}: alt providers exhausted — deferring rest.")
                counts["deferred"] += 1
                stop = True
                continue
            except Exception as e:
                log(f"  ERROR {sym}: {str(e)[:100]}")
                if not args.dry_run:
                    queue.loc[idx, "status"] = "error"
                    queue.loc[idx, "processed_at"] = datetime.now().isoformat(timespec="seconds")
                    save_queue(drive, index_id, queue)
                counts["error"] += 1
                continue

            isin_key = str(row.get("isin") or row.get("key") or "").strip()
            nq = ec._norm_quarter(str(facts.get("quarter") or "")) if facts.get("quarter") else ""
            qkey = (isin_key, nq)
            if nq and (qkey in covered or qkey in seen_run):
                if not args.dry_run:
                    queue.loc[idx, "status"] = "done"
                    queue.loc[idx, "processed_at"] = datetime.now().isoformat(timespec="seconds")
                    save_queue(drive, index_id, queue)
                counts["dup"] += 1
                continue
            if nq:
                seen_run.add(qkey)

            if args.dry_run:
                print(f"  [dry] {sym:<12} {len(md):>6} chars  q={facts.get('quarter')} "
                      f"guid={len(guidance_rows)} gf1={len(gf1)} gf2={len(gf2)}")
                counts["processed"] += 1
                continue

            # ---- MAIN-thread writes (idempotent) ----
            key = str(row.get("key") or row.get("isin") or row.get("symbol") or "")
            append_company_page(drive, repo_id, key=key, doc_type_label="Concall",
                                content=md, doc_title=str(row.get("title", "")),
                                quarter=str(facts.get("quarter") or ""),
                                dedup_marker=str(row.get("doc_id", "")))
            ec.upsert_facts(drive, index_id, facts)
            ec.upsert_guidance(drive, index_id, guidance_rows)
            ec.upsert_gf1(drive, index_id, gf1)
            ec.upsert_gf2(drive, index_id, gf2)
            ec.upsert_gf3(drive, index_id, gf3)
            ec.upsert_gf4(drive, index_id, gf4)
            queue.loc[idx, "status"] = "done"
            queue.loc[idx, "processed_at"] = datetime.now().isoformat(timespec="seconds")
            queue.loc[idx, "backfill_process_date"] = datetime.now().strftime("%Y-%m-%d")
            save_queue(drive, index_id, queue)
            counts["processed"] += 1
            log(f"  done {sym} via alt (q={facts.get('quarter')}, guid={len(guidance_rows)})")

    if not args.dry_run:
        from _extractor_base import persist_gemini_usage
        persist_gemini_usage(drive, index_id, altpool.summary(), DOC_TYPE, "backfill_alt")
    dt = time.time() - t0
    print("-" * 56)
    print(f"Concall alt drain — processed:{counts['processed']} dup:{counts['dup']} "
          f"err:{counts['error']} skip:{counts['skipped']} deferred:{counts['deferred']}")
    print(f"Elapsed {dt:.1f}s  ({dt/max(1,counts['processed']):.1f}s/doc)")


if __name__ == "__main__":
    main()
