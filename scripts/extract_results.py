"""
Phase 2 / Stage B — Quarterly results extraction via Gemini.

Consumes pending results entries from the Drive processing queue, runs each PDF
through Gemini with results_prompt.txt, extracts structured quarterly financials
into _index/results_gemini.parquet, appends a markdown brief to the company's
company_page.md and to the daily digest, then marks queue rows done.

On Gemini 429 with all keys exhausted: stops cleanly (exit 0) so the next
scheduled run resumes from remaining pending rows.

Usage:
    python scripts/extract_results.py
    python scripts/extract_results.py --limit 5
    python scripts/extract_results.py --dry-run
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

# Ensure the scripts/ directory is on sys.path so _extractor_base can be found
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _extractor_base import (
    RateLimitExhausted, GeminiKeyPool, get_drive, load_api_keys, P1_MODELS,
    log, get_or_create_subfolder,
    load_queue, save_queue,
    load_parquet, save_parquet,
    download_bytes,
    extract_md_tables, clean_val, try_float, identify_metric,
    append_company_page, append_day_page,
    load_portfolio_isins,
)

# ---- Config ----
DOC_TYPE        = "results"
GEMINI_MODEL    = P1_MODELS          # lite chain, disjoint from concall (P0)
PROMPT_FILE     = "results_prompt.txt"
DOC_TYPE_LABEL  = "Results"

OUTPUT_COMPANY_MD   = True   # append to company_repo/<key>/company_page.md
OUTPUT_DAY_MD       = True   # append to company_repo/_daily/results_DD_MMMYYYY.md
OUTPUT_COMPANY_DOCX = False  # [Stage C]
OUTPUT_DAY_DOCX     = False  # [Stage C]

# ---- Parquet schema ----
# Separate from scrape_results_table.py's results.parquet (which has HTML scrape data).
# This parquet stores Gemini-extracted financials from the PDF filing.
RESULTS_GEMINI_COLS = [
    "isin", "symbol", "company_name", "quarter", "fy_year",
    "revenue_cr", "ebitda_cr", "pat_cr", "eps",
    "ebitda_margin_pct", "pat_margin_pct",
    "revenue_yoy_pct", "pat_yoy_pct",
    "processed_at", "source_doc_id",
]


# ------------------------------------------------------------------ #
#  Parser                                                              #
# ------------------------------------------------------------------ #

def parse_gemini_response(text: str, row: pd.Series) -> dict:
    """Parse Gemini response into a results facts dict."""
    now_str = datetime.now().isoformat(timespec="seconds")
    isin = str(row.get("isin") or "")
    symbol = str(row.get("symbol") or "")
    company_name = str(row.get("company_name") or "")
    source_doc_id = str(row.get("doc_id") or "")

    facts: dict = {c: None for c in RESULTS_GEMINI_COLS}
    facts.update({
        "isin": isin, "symbol": symbol, "company_name": company_name,
        "quarter": "", "fy_year": "",
        "processed_at": now_str, "source_doc_id": source_doc_id,
    })

    tables = extract_md_tables(text)
    if not tables:
        return facts

    t = tables[0]
    hdrs = t["headers"]

    # Identify the current-quarter column (Q\d FY\d+)
    q_col = next((i for i, h in enumerate(hdrs)
                  if re.search(r"Q\d\s+FY\d{2,4}", h, re.IGNORECASE)), None)
    if q_col is not None:
        m = re.search(r"(Q\d\s+FY\d{2,4})", hdrs[q_col], re.IGNORECASE)
        if m:
            facts["quarter"] = m.group(1).strip()
            fy_m = re.search(r"FY(\d{2,4})", facts["quarter"], re.IGNORECASE)
            if fy_m:
                facts["fy_year"] = f"FY{fy_m.group(1)}"

    # YoY% column (last column usually, or column containing "YoY" / "%")
    yoy_col = next((i for i, h in enumerate(hdrs)
                    if re.search(r"yoy|growth|%", h, re.IGNORECASE)), None)

    FIELD_MAP = {
        "revenue": ("revenue_cr", "revenue_yoy_pct"),
        "sales":   ("revenue_cr", "revenue_yoy_pct"),
        "ebitda":  ("ebitda_cr",  None),
        "pat":     ("pat_cr",     "pat_yoy_pct"),
        "eps":     ("eps",        None),
        "margin":  (None,         None),
    }
    MARGIN_MAP = {
        "ebitda": "ebitda_margin_pct",
        "pat":    "pat_margin_pct",
    }

    for cells in t["rows"]:
        if not cells:
            continue
        metric = identify_metric(cells[0])
        if not metric:
            continue

        # Value from current-quarter column
        if q_col is not None and q_col < len(cells):
            raw = clean_val(cells[q_col])
            if metric in ("revenue", "sales") and raw != "NA":
                facts["revenue_cr"] = raw
            elif metric == "ebitda" and raw != "NA":
                facts["ebitda_cr"] = raw
            elif metric == "pat" and raw != "NA":
                facts["pat_cr"] = raw
            elif metric == "eps" and raw != "NA":
                facts["eps"] = raw

        # Margin rows (labelled like "EBITDA Margin %" or "PAT Margin %")
        low_label = cells[0].lower()
        for pfx, col_name in MARGIN_MAP.items():
            if pfx in low_label and "margin" in low_label:
                if q_col is not None and q_col < len(cells):
                    facts[col_name] = clean_val(cells[q_col])

        # YoY% column
        if yoy_col is not None and yoy_col < len(cells):
            raw_yoy = clean_val(cells[yoy_col])
            if metric in ("revenue", "sales"):
                facts["revenue_yoy_pct"] = raw_yoy
            elif metric == "pat":
                facts["pat_yoy_pct"] = raw_yoy

    return facts


def upsert_results_gemini(drive, index_id: str, facts: dict) -> None:
    df = load_parquet(drive, index_id, "results_gemini.parquet", RESULTS_GEMINI_COLS)
    mask = (
        (df["isin"].astype(str) == str(facts["isin"])) &
        (df["quarter"].astype(str) == str(facts["quarter"])) &
        (df["source_doc_id"].astype(str) == str(facts["source_doc_id"]))
    )
    df = df[~mask]
    new_row = pd.DataFrame([{c: facts.get(c) for c in RESULTS_GEMINI_COLS}])
    df = pd.concat([df, new_row], ignore_index=True)
    save_parquet(drive, index_id, "results_gemini.parquet", df)


# ------------------------------------------------------------------ #
#  Main                                                                #
# ------------------------------------------------------------------ #

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 2 / Stage B — Results extraction via Gemini"
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    print(f"Phase 2 / Stage B — {DOC_TYPE_LABEL} extraction via Gemini")
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

    folder_id = os.environ["GDRIVE_FOLDER_ID"]
    repo_id   = get_or_create_subfolder(drive, folder_id, "company_repo")
    index_id  = get_or_create_subfolder(drive, repo_id,   "_index")

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
            markdown_text = gemini.call(pdf_bytes, prompt, display_name)
            log(f"  Gemini response: {len(markdown_text):,} chars")

            if args.dry_run:
                print(f"\n{'='*60}\nDRY RUN — {row.get('symbol')}\n"
                      f"{markdown_text[:800]}\n{'='*60}\n")
                counts["processed"] += 1
                continue

            facts = parse_gemini_response(markdown_text, row)
            log(f"  Parsed: quarter={facts['quarter'] or 'unknown'}, "
                f"revenue={facts['revenue_cr']}, pat={facts['pat_cr']}")

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

            upsert_results_gemini(drive, index_id, facts)

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

    print("-" * 56)
    print(f"Processed : {counts['processed']}")
    print(f"Errors    : {counts['error']}")
    print(f"Skipped   : {counts['skipped']}")
    if not args.dry_run:
        print("Output: company_repo/_index/results_gemini.parquet")
        print("Output: company_repo/<key>/company_page.md")
        print("Output: company_repo/_daily/results_DD_MMMYYYY.md")


if __name__ == "__main__":
    main()
