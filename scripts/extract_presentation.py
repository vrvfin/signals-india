"""
Phase 2 / Stage D — Investor presentation extraction via Gemini.

Consumes pending presentation entries from the Drive processing queue, runs each
PDF through Gemini with presentation_prompt.txt (forensic narrative + operational
KPIs lens), appends the full analysis to company_page.md and the daily digest,
and marks queue rows done.

On Gemini 429 with all keys exhausted: stops cleanly (exit 0).

Usage:
    python scripts/extract_presentation.py
    python scripts/extract_presentation.py --limit 5
    python scripts/extract_presentation.py --dry-run
"""

from __future__ import annotations

import argparse
import atexit
import os
import re
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _extractor_base import (
    RateLimitExhausted, GeminiKeyPool, get_drive, load_api_keys, P1_MODELS,
    log, get_or_create_subfolder,
    load_queue, save_queue,
    load_parquet, save_parquet,
    download_bytes,
    extract_md_tables, clean_val, identify_metric,
    append_company_page, append_day_page,
    load_portfolio_isins,
    acquire_lock, release_lock,
    salvage_json_objects, clamp, sstr, fnum, upsert_structured,  # Stage 3 tabulation
    run_structured_over_doc,                                     # Stage 3b: from source
)

# T12: SAME lock the concall extractor uses → one global mutex on the shared queue /
# company_page.md / parquets (Phase-2 live vs backfill can never write concurrently).
_LOCK_NAME = "_extract.lock"
_LOCK_MAX_AGE_MIN = 360

# ---- Config ----
DOC_TYPE        = "presentation"
GEMINI_MODEL    = P1_MODELS          # lite chain, disjoint from concall (P0)
PROMPT_FILE     = "presentation_prompt.txt"
DOC_TYPE_LABEL  = "Presentation"

OUTPUT_COMPANY_MD   = True
OUTPUT_DAY_MD       = True
OUTPUT_COMPANY_DOCX = False  # [Stage C]
OUTPUT_DAY_DOCX     = False  # [Stage C]

# ---- Parquet schema (re-use quarterly_facts; presentations often have quarter data) ----
QFACTS_COLS = [
    "isin", "symbol", "company_name", "quarter", "fy_year",
    "revenue_q", "ebitda_q", "pat_q", "margin_pct", "volume_q", "capacity_q",
    "revenue_12m", "pat_12m", "processed_at", "source_doc_id",
]

# ---- Stage 3: structured tabulation (ADDITIVE — existing outputs unchanged) ----
STRUCT_PROMPT_FILE = "presentation_structured_prompt.txt"
STRUCT_INPUT_CHARS = 60000
# Bound the lite model's report (it rambles non-linearly regardless of input size),
# else the structured pass is starved. Mirrors AR. ENHANCEMENT: same markdown report +
# quarterly_facts, just not a runaway blob.
PPT_MAX_OUTPUT_TOKENS = 4096
PPT_GUIDANCE_COLS = [
    "isin", "symbol", "company_name", "quarter",
    "metric", "guidance_type", "horizon", "value", "unit", "notes",
    "processed_at", "source_doc_id",
]
PPT_HIGHLIGHTS_COLS = [
    "isin", "symbol", "company_name", "quarter",
    "category", "statement", "value", "unit",
    "processed_at", "source_doc_id",
]
_PPT_GTYPES = {"growth", "margin", "capacity", "orderbook", "capex", "other"}
_PPT_HCATS = {"demand", "capacity", "orderbook", "cost", "new_product",
              "margin", "expansion", "other"}


def parse_ppt_structured(text, row, quarter, now_str):
    """Salvage + dedupe the structured-pass JSON into (guidance_rows, highlight_rows)."""
    objs = salvage_json_objects(text)
    if not objs:
        return [], []
    base = {"isin": str(row.get("isin") or ""), "symbol": str(row.get("symbol") or ""),
            "company_name": str(row.get("company_name") or ""), "quarter": quarter,
            "processed_at": now_str, "source_doc_id": str(row.get("doc_id") or "")}
    g_rows, h_rows, seen_g, seen_h = [], [], set(), set()
    for o in objs:
        if "statement" in o:                                                   # highlight
            d = {**base, "category": clamp(o.get("category"), _PPT_HCATS, "other"),
                 "statement": sstr(o.get("statement")), "value": fnum(o.get("value")),
                 "unit": sstr(o.get("unit"))}
            k = (d["category"], (d["statement"] or "")[:120])
            if d["statement"] and k not in seen_h:
                seen_h.add(k); h_rows.append(d)
        elif "guidance_type" in o or "metric" in o:                            # guidance
            d = {**base, "metric": sstr(o.get("metric")),
                 "guidance_type": clamp(o.get("guidance_type"), _PPT_GTYPES, "other"),
                 "horizon": sstr(o.get("horizon")), "value": fnum(o.get("value")),
                 "unit": sstr(o.get("unit")), "notes": sstr(o.get("notes"))}
            k = (d["metric"], d["horizon"], str(d["value"]), d["guidance_type"])
            if d["metric"] and k not in seen_g:
                seen_g.add(k); g_rows.append(d)
    return g_rows[:12], h_rows[:12]


def tabulate_ppt(gemini, struct_prompt, doc_bytes, row, quarter, now_str):
    """Stage 3b: run the structured JSON pass DIRECTLY over the source deck (Gemini
    reads the slides) — not the lite model's rambling report. Best-effort."""
    resp = run_structured_over_doc(gemini, struct_prompt, doc_bytes,
                                   name=f"{row.get('symbol', 'DOC')}_PPT_struct")
    return parse_ppt_structured(resp, row, quarter, now_str)


# ------------------------------------------------------------------ #
#  Parser                                                              #
# ------------------------------------------------------------------ #

def parse_gemini_response(text: str, row: pd.Series) -> dict:
    """Extract any quarter/financial data visible in the presentation analysis.

    The presentation_prompt focuses on operational KPIs and narrative; structured
    financials may or may not be present. We do best-effort extraction and store
    whatever can be found — the full markdown is the primary output.
    """
    now_str = datetime.now().isoformat(timespec="seconds")
    isin    = str(row.get("isin") or "")
    symbol  = str(row.get("symbol") or "")
    company = str(row.get("company_name") or "")
    doc_id  = str(row.get("doc_id") or "")

    facts: dict = {c: None for c in QFACTS_COLS}
    facts.update({
        "isin": isin, "symbol": symbol, "company_name": company,
        "quarter": "", "fy_year": "",
        "processed_at": now_str, "source_doc_id": doc_id,
    })

    # Try to extract a quarter reference from the title or text
    title = str(row.get("title", ""))
    for src in (title, text[:1500]):
        m = re.search(r"(Q\d\s+FY\d{2,4}|FY\d{2,4})", src, re.IGNORECASE)
        if m:
            facts["quarter"] = m.group(1).strip()
            fy_m = re.search(r"FY(\d{2,4})", facts["quarter"], re.IGNORECASE)
            if fy_m:
                facts["fy_year"] = f"FY{fy_m.group(1)}"
            break

    tables = extract_md_tables(text)
    for t in tables:
        hdrs = t["headers"]
        # Look for a column with Q\d FY\d+ or FY\d+
        q_col = next((i for i, h in enumerate(hdrs)
                      if re.search(r"Q\d\s+FY\d{2,4}", h, re.IGNORECASE)), None)
        if q_col is None:
            q_col = next((i for i, h in enumerate(hdrs)
                          if re.search(r"FY\d{2,4}", h, re.IGNORECASE)), None)
        if q_col is None:
            continue

        if not facts["quarter"]:
            m = re.search(r"(Q\d\s+FY\d{2,4}|FY\d{2,4})", hdrs[q_col], re.IGNORECASE)
            if m:
                facts["quarter"] = m.group(1).strip()
                fy_m = re.search(r"FY(\d{2,4})", facts["quarter"], re.IGNORECASE)
                if fy_m:
                    facts["fy_year"] = f"FY{fy_m.group(1)}"

        for cells in t["rows"]:
            if not cells or q_col >= len(cells):
                continue
            metric = identify_metric(cells[0])
            val = clean_val(cells[q_col])
            if val == "NA":
                continue
            if metric == "revenue":
                facts["revenue_q"] = val
            elif metric == "ebitda":
                facts["ebitda_q"] = val
            elif metric == "pat":
                facts["pat_q"] = val
            elif metric == "margin":
                facts["margin_pct"] = val
            elif metric == "volume":
                facts["volume_q"] = val
            elif metric == "capacity":
                facts["capacity_q"] = val
        break

    return facts


def upsert_facts(drive, index_id: str, facts: dict) -> None:
    df = load_parquet(drive, index_id, "quarterly_facts.parquet", QFACTS_COLS)
    mask = (
        (df["isin"].astype(str) == str(facts["isin"])) &
        (df["quarter"].astype(str) == str(facts["quarter"])) &
        (df["source_doc_id"].astype(str) == str(facts["source_doc_id"]))
    )
    df = df[~mask]
    new_row = pd.DataFrame([{c: facts.get(c) for c in QFACTS_COLS}])
    df = pd.concat([df, new_row], ignore_index=True)
    save_parquet(drive, index_id, "quarterly_facts.parquet", df)


# ------------------------------------------------------------------ #
#  Main                                                                #
# ------------------------------------------------------------------ #

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 2 / Stage D — Presentation extraction via Gemini"
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    print(f"Phase 2 / Stage D — {DOC_TYPE_LABEL} extraction via Gemini")
    print("-" * 56)

    drive = get_drive()

    api_keys = load_api_keys()
    if not api_keys:
        print("ERROR: no GEMINI_API_KEY or GEMINI_API_KEY_* found in .env")
        sys.exit(1)
    log(f"Loaded {len(api_keys)} Gemini API key(s)")

    gemini = GeminiKeyPool(api_keys, GEMINI_MODEL)

    prompt_path = Path(__file__).resolve().parent / PROMPT_FILE
    if not prompt_path.exists():
        print(f"ERROR: prompt file not found: {prompt_path}")
        sys.exit(1)
    prompt = prompt_path.read_text(encoding="utf-8")
    _struct_path = Path(__file__).resolve().parent / STRUCT_PROMPT_FILE
    struct_prompt = _struct_path.read_text(encoding="utf-8") if _struct_path.exists() else ""

    folder_id = os.environ["GDRIVE_FOLDER_ID"]
    repo_id   = get_or_create_subfolder(drive, folder_id, "company_repo")
    index_id  = get_or_create_subfolder(drive, repo_id,   "_index")

    # T12 Phase-2 safety: serialize shared-file writes via the global _extract.lock.
    # On contention exit cleanly — the next run resumes (rows stay pending).
    if not args.dry_run:
        # Phase-2 priority: wait up to 15 min for the lock (backfill yields to us
        # within ~one document) instead of skipping the slot on contention.
        if not acquire_lock(drive, index_id, _LOCK_NAME, DOC_TYPE,
                            max_age_min=_LOCK_MAX_AGE_MIN, wait_min=15):
            log("  Lock unavailable after wait — exiting cleanly.")
            sys.exit(0)
        atexit.register(release_lock, drive, index_id, _LOCK_NAME)

    queue = load_queue(drive, index_id)
    pending_mask = (queue["status"] == "pending") & (queue["doc_type"] == DOC_TYPE)
    pending_idx = queue.index[pending_mask].tolist()
    log(f"Queue: {len(queue)} total rows, {len(pending_idx)} pending {DOC_TYPE}")

    # Portfolio filter: skip non-portfolio companies (rows stay pending for future runs)
    portfolio_isins = load_portfolio_isins(drive, folder_id)
    if portfolio_isins:
        before = len(pending_idx)
        pending_idx = [i for i in pending_idx
                       if str(queue.loc[i, "isin"]).strip() in portfolio_isins]
        log(f"  After portfolio filter: {len(pending_idx)}/{before} to process")

    if args.limit:
        pending_idx = pending_idx[: args.limit]

    counts = {"processed": 0, "error": 0, "skipped": 0}

    for queue_idx in pending_idx:
        row = queue.loc[queue_idx]
        label = f"{row.get('symbol', '?')!s:<14} {str(row.get('title', ''))[:55]}"
        log(f"Processing: {label}")

        drive_fid = str(row.get("drive_file_id") or "").strip()
        if not drive_fid:
            log("  SKIP: no drive_file_id")
            counts["skipped"] += 1
            continue

        try:
            pdf_bytes = download_bytes(drive, drive_fid)
            log(f"  PDF: {len(pdf_bytes):,} bytes")

            display_name = f"{row.get('symbol', 'DOC')}_{str(row.get('doc_id', ''))[:12]}.pdf"
            markdown_text = gemini.call(pdf_bytes, prompt, display_name,
                                        max_output_tokens=PPT_MAX_OUTPUT_TOKENS)
            log(f"  Gemini response: {len(markdown_text):,} chars")

            if args.dry_run:
                print(f"\n{'='*60}\nDRY RUN — {row.get('symbol')}\n"
                      f"{markdown_text[:800]}\n{'='*60}\n")
                counts["processed"] += 1
                continue

            facts = parse_gemini_response(markdown_text, row)
            log(f"  Parsed: quarter={facts['quarter'] or 'unknown'}")

            if OUTPUT_COMPANY_MD:
                append_company_page(
                    drive, repo_id,
                    key=str(row.get("key") or row.get("isin") or row.get("symbol") or ""),
                    doc_type_label=DOC_TYPE_LABEL,
                    content=markdown_text,
                    doc_title=str(row.get("title", "")),
                    quarter=facts["quarter"],
                )

            if OUTPUT_DAY_MD:
                append_day_page(
                    drive, repo_id,
                    doc_type=DOC_TYPE,
                    announcement_date=str(row.get("announcement_date", "")),
                    symbol=str(row.get("symbol", "")),
                    company_name=str(row.get("company_name", "")),
                    quarter=facts["quarter"],
                    content=markdown_text,
                )

            if facts["quarter"]:
                upsert_facts(drive, index_id, facts)

            # Stage 3 tabulation (ADDITIVE, best-effort): separate JSON-only pass →
            # ppt_guidance / ppt_highlights. Failure leaves markdown + quarterly_facts
            # untouched (no regression; Phase 2 outputs unchanged).
            try:
                g_rows, h_rows = tabulate_ppt(
                    gemini, struct_prompt, pdf_bytes, row, facts["quarter"],
                    datetime.now().isoformat(timespec="seconds"))
                upsert_structured(drive, index_id, "ppt_guidance.parquet",
                                  PPT_GUIDANCE_COLS, g_rows)
                upsert_structured(drive, index_id, "ppt_highlights.parquet",
                                  PPT_HIGHLIGHTS_COLS, h_rows)
                log(f"  Tabulated: guidance={len(g_rows)}, highlights={len(h_rows)}")
            except RateLimitExhausted:
                log("  Structured pass: keys exhausted — tabulation skipped.")
            except Exception as _e:
                log(f"  WARNING: presentation tabulation failed ({str(_e)[:90]}).")

            queue.loc[queue_idx, "status"] = "done"
            queue.loc[queue_idx, "processed_at"] = datetime.now().isoformat(timespec="seconds")
            save_queue(drive, index_id, queue)

            counts["processed"] += 1
            log(f"  Done: {row.get('symbol')}")

        except RateLimitExhausted:
            log("All Gemini keys rate-limited — stopping cleanly.")
            break

        except Exception as exc:
            log(f"  ERROR: {str(exc)[:120]}")
            queue.loc[queue_idx, "status"] = "error"
            queue.loc[queue_idx, "processed_at"] = datetime.now().isoformat(timespec="seconds")
            save_queue(drive, index_id, queue)
            counts["error"] += 1

    if not args.dry_run:
        from _extractor_base import persist_gemini_usage
        persist_gemini_usage(drive, index_id, gemini.summary(), DOC_TYPE,
                             "backfill" if getattr(args, "all_companies", False) else "phase2")
    print("-" * 56)
    print(f"Processed : {counts['processed']}")
    print(f"Errors    : {counts['error']}")
    print(f"Skipped   : {counts['skipped']}")
    if not args.dry_run:
        print("Output: company_repo/_index/quarterly_facts.parquet")
        print("Output: company_repo/<key>/company_page.md")
        print("Output: company_repo/_daily/presentation_DD_MMMYYYY.md")


if __name__ == "__main__":
    main()
