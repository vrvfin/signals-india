"""
Phase 2 / Stage D — Credit rating extraction via Gemini.

Consumes pending rating entries from the Drive processing queue, runs each PDF
through Gemini with rating_prompt.txt (solvency + credit forensic lens), appends
the full analysis to company_page.md and the daily digest, upserts extracted rating
metadata into _index/ratings.parquet, and marks queue rows done.

On Gemini 429 with all keys exhausted: stops cleanly (exit 0).

Usage:
    python scripts/extract_rating.py
    python scripts/extract_rating.py --limit 5
    python scripts/extract_rating.py --dry-run
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
    extract_md_tables, clean_val,
    append_company_page, append_day_page,
    load_portfolio_isins,
    acquire_lock, release_lock,
    salvage_json_objects, clamp, sstr, upsert_structured,   # Stage 3 tabulation
    run_structured_over_doc,                                # Stage 3b: extract from source
)

# T12: SAME lock the concall extractor uses → one global mutex on the shared queue /
# company_page.md / parquets (Phase-2 live vs backfill can never write concurrently).
_LOCK_NAME = "_extract.lock"
_LOCK_MAX_AGE_MIN = 360

# ---- Config ----
DOC_TYPE        = "rating"
GEMINI_MODEL    = P1_MODELS          # lite chain, disjoint from concall (P0)
PROMPT_FILE     = "rating_prompt.txt"
DOC_TYPE_LABEL  = "Credit Rating"

OUTPUT_COMPANY_MD   = True
OUTPUT_DAY_MD       = True
OUTPUT_COMPANY_DOCX = False  # [Stage C]
OUTPUT_DAY_DOCX     = False  # [Stage C]

# ---- Parquet schema ----
RATINGS_COLS = [
    "isin", "symbol", "company_name",
    "agency", "rating", "outlook", "rating_action",
    "instrument_type", "rated_amount_cr",
    "rating_date", "processed_at", "source_doc_id",
]

# ---- Stage 3: structured tabulation (ADDITIVE — ratings.parquet unchanged) ----
STRUCT_PROMPT_FILE = "rating_structured_prompt.txt"
STRUCT_INPUT_CHARS = 60000
# Bound the lite model's report (it rambles non-linearly — measured ~2M chars even on a
# 4KB rating doc — which starves the structured pass). Mirrors AR. ENHANCEMENT only:
# output is still the same markdown report + ratings.parquet, just not a runaway blob.
RATING_MAX_OUTPUT_TOKENS = 4096
_RATING_BASE = ["isin", "symbol", "company_name", "agency", "rating_date",
                "processed_at", "source_doc_id"]
RATING_DRIVERS_COLS    = _RATING_BASE[:5] + ["driver", "evidence"] + _RATING_BASE[5:]
RATING_CONCERNS_COLS   = _RATING_BASE[:5] + ["concern", "severity", "evidence"] + _RATING_BASE[5:]
RATING_SENSITIVITY_COLS = _RATING_BASE[:5] + ["direction", "trigger"] + _RATING_BASE[5:]
_SEV = {"low", "medium", "high"}
_DIR = {"up", "down"}


def parse_rating_structured(text, row, agency, rating_date, now_str):
    """Salvage + dedupe → (drivers, concerns, sensitivity) rows."""
    objs = salvage_json_objects(text)
    if not objs:
        return [], [], []
    base = {"isin": str(row.get("isin") or ""), "symbol": str(row.get("symbol") or ""),
            "company_name": str(row.get("company_name") or ""), "agency": agency or "",
            "rating_date": rating_date or "", "processed_at": now_str,
            "source_doc_id": str(row.get("doc_id") or "")}
    dr, co, se = [], [], []
    sd, sc, ss = set(), set(), set()
    for o in objs:
        if "concern" in o:
            d = {**base, "concern": sstr(o.get("concern")),
                 "severity": clamp(o.get("severity"), _SEV, "medium"),
                 "evidence": sstr(o.get("evidence"))}
            k = (d["concern"] or "")[:120]
            if d["concern"] and k not in sc:
                sc.add(k); co.append(d)
        elif "driver" in o:
            d = {**base, "driver": sstr(o.get("driver")), "evidence": sstr(o.get("evidence"))}
            k = (d["driver"] or "")[:120]
            if d["driver"] and k not in sd:
                sd.add(k); dr.append(d)
        elif "trigger" in o or "direction" in o:
            d = {**base, "direction": clamp(o.get("direction"), _DIR, "up"),
                 "trigger": sstr(o.get("trigger"))}
            k = (d["direction"], (d["trigger"] or "")[:120])
            if d["trigger"] and k not in ss:
                ss.add(k); se.append(d)
    return dr[:10], co[:10], se[:8]


def tabulate_rating(gemini, struct_prompt, doc_bytes, row, agency, rating_date, now_str):
    """Stage 3b: run the structured JSON pass DIRECTLY over the source rating doc
    (small/clean) — not the lite model's rambling report — so drivers/concerns/
    sensitivity actually populate. Best-effort."""
    resp = run_structured_over_doc(gemini, struct_prompt, doc_bytes,
                                   name=f"{row.get('symbol', 'DOC')}_RATING_struct")
    return parse_rating_structured(resp, row, agency, rating_date, now_str)

# Common rating agency name variants
_AGENCY_PATTERNS = [
    (r"crisil",   "CRISIL"),
    (r"icra",     "ICRA"),
    (r"care",     "CARE"),
    (r"india ratings|ind-ra", "India Ratings"),
    (r"brickwork", "Brickwork"),
    (r"acuite",   "Acuite"),
]


# ------------------------------------------------------------------ #
#  Parser                                                              #
# ------------------------------------------------------------------ #

def _detect_agency(text: str) -> str:
    low = text.lower()
    for pattern, name in _AGENCY_PATTERNS:
        if re.search(pattern, low):
            return name
    return "DATA_MISSING"


def _detect_rating(text: str) -> str:
    """Find the first SEBI/NIC rating symbol in text (e.g. AAA, AA+, BBB-)."""
    m = re.search(
        r"\b(AAA|AA\+|AA|AA-|A\+|A\b|A-|BBB\+|BBB|BBB-|BB\+|BB|BB-|B|C|D)"
        r"(?:\s*/\s*(Stable|Positive|Negative|Watch|CWN|CWP))?",
        text, re.IGNORECASE
    )
    if m:
        rating = m.group(1).upper()
        outlook = m.group(2).title() if m.group(2) else ""
        return f"{rating}/{outlook}" if outlook else rating
    return "DATA_MISSING"


def parse_gemini_response(text: str, row: pd.Series) -> dict:
    """Extract rating metadata from the forensic credit report.

    The rating_prompt produces a Section 1 table with Agency, Instrument,
    Rated Amount, Rating/Outlook, historical migrations. We read that table.
    """
    now_str = datetime.now().isoformat(timespec="seconds")
    isin    = str(row.get("isin") or "")
    symbol  = str(row.get("symbol") or "")
    company = str(row.get("company_name") or "")
    doc_id  = str(row.get("doc_id") or "")

    facts: dict = {c: None for c in RATINGS_COLS}
    facts.update({
        "isin": isin, "symbol": symbol, "company_name": company,
        "agency": _detect_agency(text),
        "rating": _detect_rating(text),
        "rating_date": str(row.get("announcement_date", ""))[:10] or None,
        "processed_at": now_str, "source_doc_id": doc_id,
    })

    # Try to parse the Section 1 chronology table for richer data
    tables = extract_md_tables(text)
    for t in tables:
        hdrs = [h.lower() for h in t["headers"]]
        # Identify columns
        agency_col    = next((i for i, h in enumerate(hdrs) if "agency" in h), None)
        rating_col    = next((i for i, h in enumerate(hdrs)
                              if "rating" in h or "outlook" in h), None)
        instrument_col = next((i for i, h in enumerate(hdrs)
                               if "instrument" in h or "facility" in h), None)
        amount_col    = next((i for i, h in enumerate(hdrs)
                              if "amount" in h or "size" in h or "limit" in h), None)

        if rating_col is None:
            continue

        if t["rows"]:
            # Use the first (most recent) row
            cells = t["rows"][0]
            if agency_col is not None and agency_col < len(cells):
                raw = clean_val(cells[agency_col])
                if raw != "NA":
                    facts["agency"] = raw

            if rating_col < len(cells):
                raw = clean_val(cells[rating_col])
                if raw != "NA":
                    # Separate "AA+/Stable" → rating + outlook
                    parts = raw.split("/")
                    facts["rating"]  = parts[0].strip()
                    if len(parts) > 1:
                        facts["outlook"] = parts[1].strip()
                    # Detect action from the text around the rating
                    low_raw = raw.lower()
                    if "upgrad" in low_raw or "upgrade" in low_raw:
                        facts["rating_action"] = "Upgrade"
                    elif "downgrad" in low_raw:
                        facts["rating_action"] = "Downgrade"
                    else:
                        facts["rating_action"] = "Reaffirmed"

            if instrument_col is not None and instrument_col < len(cells):
                raw = clean_val(cells[instrument_col])
                if raw != "NA":
                    facts["instrument_type"] = raw

            if amount_col is not None and amount_col < len(cells):
                raw = clean_val(cells[amount_col])
                if raw != "NA":
                    facts["rated_amount_cr"] = raw
        break

    # Detect outlook separately if not already set
    if not facts.get("outlook"):
        m = re.search(r"\b(Stable|Positive|Negative|Watch|CWN|CWP)\b", text, re.IGNORECASE)
        if m:
            facts["outlook"] = m.group(1).title()

    # Detect rating action from header text
    if not facts.get("rating_action"):
        low = text[:500].lower()
        if "upgrad" in low:
            facts["rating_action"] = "Upgrade"
        elif "downgrad" in low:
            facts["rating_action"] = "Downgrade"
        else:
            facts["rating_action"] = "Reaffirmed"

    return facts


def upsert_ratings(drive, index_id: str, facts: dict) -> None:
    df = load_parquet(drive, index_id, "ratings.parquet", RATINGS_COLS)
    mask = (
        (df["isin"].astype(str) == str(facts["isin"])) &
        (df["source_doc_id"].astype(str) == str(facts["source_doc_id"]))
    )
    df = df[~mask]
    new_row = pd.DataFrame([{c: facts.get(c) for c in RATINGS_COLS}])
    df = pd.concat([df, new_row], ignore_index=True)
    save_parquet(drive, index_id, "ratings.parquet", df)


# ------------------------------------------------------------------ #
#  Main                                                                #
# ------------------------------------------------------------------ #

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 2 / Stage D — Credit rating extraction via Gemini"
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
                                        max_output_tokens=RATING_MAX_OUTPUT_TOKENS)
            log(f"  Gemini response: {len(markdown_text):,} chars")

            if args.dry_run:
                print(f"\n{'='*60}\nDRY RUN — {row.get('symbol')}\n"
                      f"{markdown_text[:800]}\n{'='*60}\n")
                counts["processed"] += 1
                continue

            facts = parse_gemini_response(markdown_text, row)
            log(f"  Parsed: agency={facts['agency']}, rating={facts['rating']}, "
                f"outlook={facts['outlook']}, action={facts['rating_action']}")

            if OUTPUT_COMPANY_MD:
                append_company_page(
                    drive, repo_id,
                    key=str(row.get("key") or row.get("isin") or row.get("symbol") or ""),
                    doc_type_label=DOC_TYPE_LABEL,
                    content=markdown_text,
                    doc_title=str(row.get("title", "")),
                    quarter=f"{facts['agency']} {facts['rating']}",
                )

            if OUTPUT_DAY_MD:
                append_day_page(
                    drive, repo_id,
                    doc_type=DOC_TYPE,
                    announcement_date=str(row.get("announcement_date", "")),
                    symbol=str(row.get("symbol", "")),
                    company_name=str(row.get("company_name", "")),
                    quarter=f"{facts['agency']} {facts['rating']}",
                    content=markdown_text,
                )

            upsert_ratings(drive, index_id, facts)

            # Stage 3 tabulation (ADDITIVE, best-effort): separate JSON-only pass →
            # rating_drivers / rating_concerns / rating_sensitivity. ratings.parquet +
            # markdown untouched on failure (no regression; Phase 2 unchanged).
            try:
                dr, co, se = tabulate_rating(
                    gemini, struct_prompt, pdf_bytes, row,
                    facts.get("agency"), facts.get("rating_date"),
                    datetime.now().isoformat(timespec="seconds"))
                upsert_structured(drive, index_id, "rating_drivers.parquet",
                                  RATING_DRIVERS_COLS, dr)
                upsert_structured(drive, index_id, "rating_concerns.parquet",
                                  RATING_CONCERNS_COLS, co)
                upsert_structured(drive, index_id, "rating_sensitivity.parquet",
                                  RATING_SENSITIVITY_COLS, se)
                log(f"  Tabulated: drivers={len(dr)}, concerns={len(co)}, "
                    f"sensitivity={len(se)}")
            except RateLimitExhausted:
                log("  Structured pass: keys exhausted — tabulation skipped.")
            except Exception as _e:
                log(f"  WARNING: rating tabulation failed ({str(_e)[:90]}).")

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
        print("Output: company_repo/_index/ratings.parquet")
        print("Output: company_repo/<key>/company_page.md")
        print("Output: company_repo/_daily/rating_DD_MMMYYYY.md")


if __name__ == "__main__":
    main()
