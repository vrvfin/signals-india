"""
Phase 2 / Stage D — Annual report extraction via Gemini.

Consumes pending annual_report entries from the Drive processing queue, runs each
PDF through Gemini with annual_report_prompt.txt (forensic analysis lens), appends
the full analysis to company_page.md and the daily digest, upserts extracted annual
financials into quarterly_facts.parquet, and marks queue rows done.

Large PDFs (> MAP_REDUCE_THRESHOLD_MB): split into ~100-page chunks, summarise each
independently, then run a synthesis pass over the chunk summaries. Smaller PDFs are
processed inline (same as concall pipeline).

On Gemini 429 with all keys exhausted: stops cleanly (exit 0).

Usage:
    python scripts/extract_annual_report.py
    python scripts/extract_annual_report.py --limit 2
    python scripts/extract_annual_report.py --dry-run
"""

from __future__ import annotations

import argparse
import atexit
import io
import json
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
    extract_md_tables, clean_val, try_float, identify_metric,
    append_company_page, append_day_page,
    load_portfolio_isins,
    acquire_lock, release_lock, phase2_beacon_fresh,
)

# T12: SAME lock the concall extractor uses → one global mutex on the shared queue /
# company_page.md / parquets (Phase-2 live vs backfill can never write concurrently).
_LOCK_NAME = "_extract.lock"
_LOCK_MAX_AGE_MIN = 360

# ---- Config ----
DOC_TYPE        = "annual_report"
GEMINI_MODEL    = P1_MODELS          # lite chain, disjoint from concall (P0)
PROMPT_FILE     = "annual_report_prompt.txt"
STRUCT_PROMPT_FILE = "ar_structured_prompt.txt"   # JSON-only structured 2nd pass
STRUCT_INPUT_CHARS = 60000                        # cap report text fed to the 2nd call
DOC_TYPE_LABEL  = "Annual Report"

OUTPUT_COMPANY_MD   = True
OUTPUT_DAY_MD       = True
OUTPUT_COMPANY_DOCX = False  # [Stage C]
OUTPUT_DAY_DOCX     = False  # [Stage C]

# PDFs above this size are processed via map-reduce chunking.
# Gemini inline limit is ~20 MB total request; a 10 MB PDF → 13.3 MB base64
# which stays safely under that ceiling.
MAP_REDUCE_THRESHOLD_MB = 10

SYNTHESIS_PROMPT_PREFIX = """You are given multiple section-level summaries of a single Annual Report.
Synthesise them into one coherent final report following the same forensic structure.
Do not repeat headings. Merge tables where appropriate. Preserve all DATA_MISSING flags.

SECTION SUMMARIES:
"""

# ---- Parquet schema (annual totals stored as quarterly_facts rows with quarter=FY26 etc.) ----
QFACTS_COLS = [
    "isin", "symbol", "company_name", "quarter", "fy_year",
    "revenue_q", "ebitda_q", "pat_q", "margin_pct", "volume_q", "capacity_q",
    "revenue_12m", "pat_12m", "processed_at", "source_doc_id",
]

# ---- Structured tabulation schemas (from the prompt's machine-readable appendix) ----
# Mirror concall's guidance_tracker / gf4_quality_flags pattern at AR (FY) grain so
# downstream work (scorecard, fraud tracker, deep dive, screener) can query AR signals.
AR_GUIDANCE_COLS = [
    "isin", "symbol", "company_name", "fy_year",
    "metric", "guidance_type", "horizon_fy", "value", "unit", "cagr_pct", "notes",
    "processed_at", "source_doc_id",
]
AR_REDFLAG_COLS = [
    "isin", "symbol", "company_name", "fy_year",
    "category", "flag_type", "severity", "evidence", "page_ref",
    "processed_at", "source_doc_id",
]

_AR_GUIDANCE_TYPES = {"growth", "margin", "capacity", "orderbook", "capex", "other"}
_AR_FLAG_CATEGORIES = {"auditor", "notes_to_accounts", "accounting_policy", "cash_flow",
                       "balance_sheet", "governance", "related_party", "tax", "other"}
_AR_FLAG_TYPES = {
    "auditor_qualification", "emphasis_of_matter", "caro_adverse",
    "accounting_policy_change", "accounting_estimate_change", "revenue_recognition_change",
    "notes_to_accounts_deviation", "cfo_pat_divergence", "working_capital_stretch",
    "cwip_buildup", "related_party_transaction", "promoter_pledge", "kmp_churn",
    "contingent_liability", "tax_variance", "other",
}
_AR_SEVERITIES = {"low", "medium", "high"}


def _s(v):
    """Trimmed string, or None for empty/null (keeps parquet nulls clean)."""
    if v is None:
        return None
    s = str(v).strip()
    return s or None


def _num(v):
    """Coerce to float via the shared try_float, guarding None."""
    if v is None or v == "":
        return None
    try:
        return try_float(v)
    except Exception:
        return None


def _clamp(v, allowed: set, default: str) -> str:
    s = str(v or "").strip().lower()
    return s if s in allowed else default


_FY_RE = re.compile(r'"fy_year"\s*:\s*"([^"]{1,12})"')
_FLAT_OBJ_RE = re.compile(r"\{[^{}]*\}", re.DOTALL)


def _extract_json_block(text: str) -> dict:
    """Best-effort parse of one clean JSON object (bare or fenced). Returns {} on
    failure — parse_ar_structured does NOT rely on this; it falls back to per-object
    salvage. Kept for the happy path / fy_year lookup."""
    if not text:
        return {}
    t = text.strip()
    m = re.search(r"```(?:json)?\s*(.*?)```", t, re.DOTALL | re.IGNORECASE)
    if m:
        t = m.group(1).strip()
    first, last = t.find("{"), t.rfind("}")
    if first != -1 and last > first:
        try:
            obj = json.loads(t[first:last + 1])
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass
    return {}


def _salvage_objects(text: str) -> list[dict]:
    """Parse every FLAT {...} object in the text individually. Robust to a truncated or
    repetition-looped response: complete inner objects are recovered and the trailing
    truncated one is skipped. guidance / red_flag objects are flat (no nesting), so they
    are captured even when the enclosing JSON never closed (the live failure mode)."""
    out: list[dict] = []
    for m in _FLAT_OBJ_RE.findall(text or ""):
        try:
            o = json.loads(m)
            if isinstance(o, dict):
                out.append(o)
        except Exception:
            continue
    return out


def parse_ar_structured(text: str, row: pd.Series, fy_year: str,
                        now_str: str) -> tuple[list[dict], list[dict]]:
    """Parse the structured-extraction response into (guidance_rows, red_flag_rows).
    Salvage + DEDUPE based, so a truncated or looped model response still yields clean,
    de-duplicated rows (e.g. a 570x-repeated flag collapses to 1). Empty lists if
    nothing parseable — markdown report + quarterly_facts are unaffected."""
    objs = _salvage_objects(text)
    if not objs:
        return [], []
    m = _FY_RE.search(text or "")
    fy = (_s(_extract_json_block(text).get("fy_year"))
          or (m.group(1) if m else None) or fy_year or "")
    base = {
        "isin": str(row.get("isin") or ""),
        "symbol": str(row.get("symbol") or ""),
        "company_name": str(row.get("company_name") or ""),
        "fy_year": fy,
        "processed_at": now_str,
        "source_doc_id": str(row.get("doc_id") or ""),
    }

    g_rows, rf_rows = [], []
    seen_g, seen_rf = set(), set()
    for o in objs:
        if "flag_type" in o:                                   # red flag
            d = {**base,
                 "category": _clamp(o.get("category"), _AR_FLAG_CATEGORIES, "other"),
                 "flag_type": _clamp(o.get("flag_type"), _AR_FLAG_TYPES, "other"),
                 "severity": _clamp(o.get("severity"), _AR_SEVERITIES, "medium"),
                 "evidence": _s(o.get("evidence")),
                 "page_ref": _s(o.get("page_ref"))}
            k = (d["flag_type"], (d["evidence"] or "")[:120])
            if k not in seen_rf:
                seen_rf.add(k)
                rf_rows.append(d)
        elif ("guidance_type" in o) or ("metric" in o):        # guidance
            d = {**base,
                 "metric": _s(o.get("metric")),
                 "guidance_type": _clamp(o.get("guidance_type"), _AR_GUIDANCE_TYPES, "other"),
                 "horizon_fy": _s(o.get("horizon_fy")),
                 "value": _num(o.get("value")),
                 "unit": _s(o.get("unit")),
                 "cagr_pct": _num(o.get("cagr_pct")),
                 "notes": _s(o.get("notes"))}
            k = (d["metric"], d["horizon_fy"], str(d["value"]), d["guidance_type"])
            if k not in seen_g:
                seen_g.add(k)
                g_rows.append(d)
    return g_rows[:15], rf_rows[:25]            # hard caps (defence-in-depth)


def tabulate_ar(gemini, struct_prompt: str, report_text: str, row: pd.Series,
                fy_year: str, now_str: str) -> tuple[list[dict], list[dict]]:
    """Run the bounded JSON-only structured pass over the produced report and parse it.
    A SEPARATE call (not the report call) so the markdown report is never compromised;
    text-only (no PDF re-upload). Returns ([],[]) if disabled/failed."""
    if not struct_prompt or not report_text:
        return [], []
    struct_resp = gemini.call_text(
        struct_prompt + report_text[:STRUCT_INPUT_CHARS],
        f"{row.get('symbol', 'DOC')}_AR_struct")
    return parse_ar_structured(struct_resp, row, fy_year, now_str)


def _upsert_ar(drive, index_id: str, filename: str, cols: list[str],
               rows: list[dict]) -> None:
    """Delete existing rows for this source_doc_id, append new (idempotent re-extract).
    Mirrors concall's _upsert_gf."""
    if not rows:
        return
    df = load_parquet(drive, index_id, filename, cols)
    sdid = str(rows[0].get("source_doc_id", ""))
    if sdid and "source_doc_id" in df.columns:
        df = df[df["source_doc_id"].astype(str) != sdid]
    new_df = pd.DataFrame([{c: r.get(c) for c in cols} for r in rows])
    df = pd.concat([df, new_df], ignore_index=True)
    save_parquet(drive, index_id, filename, df)


# ------------------------------------------------------------------ #
#  Map-reduce chunked processing                                       #
# ------------------------------------------------------------------ #

def _split_pdf_chunks(pdf_bytes: bytes, chunk_mb: float = 4.0) -> list[bytes]:
    """Split a PDF into ~chunk_mb sized byte slices.

    Uses pypdf if available; falls back to naive byte-chunking otherwise.
    """
    try:
        from pypdf import PdfReader, PdfWriter  # type: ignore

        reader = PdfReader(io.BytesIO(pdf_bytes))
        n = len(reader.pages)
        pages_per_chunk = max(1, int((chunk_mb * 1024 * 1024) /
                                     (len(pdf_bytes) / max(n, 1))))
        chunks = []
        for start in range(0, n, pages_per_chunk):
            writer = PdfWriter()
            for p in reader.pages[start: start + pages_per_chunk]:
                writer.add_page(p)
            buf = io.BytesIO()
            writer.write(buf)
            chunks.append(buf.getvalue())
        return chunks
    except ImportError:
        # Naive fallback: split bytes evenly (Gemini may reject malformed PDFs)
        size = len(pdf_bytes)
        chunk_size = int(chunk_mb * 1024 * 1024)
        return [pdf_bytes[i: i + chunk_size]
                for i in range(0, size, chunk_size)]


def _process_with_map_reduce(gemini: GeminiKeyPool, pdf_bytes: bytes,
                              prompt: str, display_name: str) -> str:
    """Chunk large PDF, summarise each chunk, then synthesise."""
    log(f"  PDF > {MAP_REDUCE_THRESHOLD_MB}MB — using map-reduce chunking")
    chunks = _split_pdf_chunks(pdf_bytes)
    log(f"  Split into {len(chunks)} chunk(s)")

    chunk_summaries: list[str] = []
    for i, chunk in enumerate(chunks, 1):
        log(f"  Chunk {i}/{len(chunks)}: {len(chunk):,} bytes")
        summary = gemini.call(chunk, prompt, f"{display_name}_chunk{i}")
        chunk_summaries.append(f"=== CHUNK {i} ===\n{summary}")

    if len(chunk_summaries) == 1:
        return chunk_summaries[0]

    synthesis_prompt = SYNTHESIS_PROMPT_PREFIX + "\n\n".join(chunk_summaries)
    # Text-only call: all content is already in the prompt, no PDF needed
    return gemini.call_text(synthesis_prompt, f"{display_name}_synthesis")


# ------------------------------------------------------------------ #
#  Parser                                                              #
# ------------------------------------------------------------------ #

def _extract_fy_year(text: str, row: pd.Series) -> str:
    """Try to identify the fiscal year from title or text."""
    # Check title first
    title = str(row.get("title", ""))
    for src in (title, text[:2000]):
        m = re.search(r"FY\s*(\d{2,4})", src, re.IGNORECASE)
        if m:
            yr = m.group(1)
            return f"FY{yr}" if len(yr) <= 2 else f"FY{yr[-2:]}"
    return ""


def parse_gemini_response(text: str, row: pd.Series) -> dict:
    """Extract annual financial facts from the forensic report.

    The annual_report_prompt produces a Section 2 table with 5-year Revenue,
    EBITDA, PAT, CFO trends. We take the most recent FY row as the annual total.
    """
    now_str = datetime.now().isoformat(timespec="seconds")
    isin    = str(row.get("isin") or "")
    symbol  = str(row.get("symbol") or "")
    company = str(row.get("company_name") or "")
    doc_id  = str(row.get("doc_id") or "")

    fy_year = _extract_fy_year(text, row)
    facts: dict = {c: None for c in QFACTS_COLS}
    facts.update({
        "isin": isin, "symbol": symbol, "company_name": company,
        "quarter": fy_year,     # annual reports stored as "FY26" in the quarter column
        "fy_year": fy_year,
        "processed_at": now_str, "source_doc_id": doc_id,
    })

    tables = extract_md_tables(text)
    # Section 2 financial table: first table with Revenue / EBITDA / PAT rows
    for t in tables:
        hdrs = t["headers"]
        # Look for a table that has at least one FY\d+ column header
        _fy_re = re.compile(r"FY(\d{2,4})", re.IGNORECASE)
        fy_cols = {i: "FY" + _fy_re.search(h).group(1)
                   for i, h in enumerate(hdrs)
                   if _fy_re.search(h)}
        if not fy_cols:
            continue

        # Use the rightmost (most recent) FY column
        latest_col = max(fy_cols.keys())
        if not facts["quarter"]:
            facts["quarter"] = f"FY{fy_cols[latest_col]}"
            facts["fy_year"]  = facts["quarter"]

        for cells in t["rows"]:
            if not cells or latest_col >= len(cells):
                continue
            metric = identify_metric(cells[0])
            val = clean_val(cells[latest_col])
            if metric == "revenue" and val != "NA":
                facts["revenue_q"]  = val
                facts["revenue_12m"] = val
            elif metric == "ebitda" and val != "NA":
                facts["ebitda_q"] = val
            elif metric == "pat" and val != "NA":
                facts["pat_q"]  = val
                facts["pat_12m"] = val
            elif metric == "margin" and val != "NA":
                facts["margin_pct"] = val
        break  # only parse the first matching table

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
        description="Phase 2 / Stage D — Annual report extraction via Gemini"
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    # T8 flags (2026-06-12) — defaults leave Phase 2 live behaviour identical.
    parser.add_argument("--all-companies", action="store_true",
                        help="T8: skip the portfolio filter — process every "
                             "pending AR (use with --max-age-hours).")
    parser.add_argument("--key-prefix", type=str, default=None,
                        help="T8: load keys from this env prefix (e.g. "
                             "BACKFILL_GEMINI_KEY) instead of GEMINI_API_KEY.")
    parser.add_argument("--max-age-hours", type=float, default=None,
                        help="T8: only rows discovered within N hours (guards "
                             "against draining quota on stale legacy rows).")
    args = parser.parse_args()

    print(f"Phase 2 / Stage D — {DOC_TYPE_LABEL} extraction via Gemini")
    print("-" * 56)

    drive = get_drive()

    if args.key_prefix:
        from gemini_pool import load_keys
        api_keys = load_keys(os.environ, prefix=args.key_prefix)
        if not api_keys:
            print(f"ERROR: no {args.key_prefix}* keys found in env")
            sys.exit(1)
    else:
        api_keys = load_api_keys()
        if not api_keys:
            print("ERROR: no GEMINI_API_KEY or GEMINI_API_KEY_* found in .env")
            sys.exit(1)
    log(f"Loaded {len(api_keys)} Gemini API key(s)"
        + (f" [{args.key_prefix}]" if args.key_prefix else ""))

    gemini = GeminiKeyPool(api_keys, GEMINI_MODEL)

    prompt_path = Path(__file__).resolve().parent / PROMPT_FILE
    if not prompt_path.exists():
        print(f"ERROR: prompt file not found: {prompt_path}")
        sys.exit(1)
    prompt = prompt_path.read_text(encoding="utf-8")

    # Structured-extraction prompt (separate, bounded JSON-only pass). Optional — if
    # absent, tabulation is silently skipped and the markdown report still works.
    struct_path = Path(__file__).resolve().parent / STRUCT_PROMPT_FILE
    struct_prompt = struct_path.read_text(encoding="utf-8") if struct_path.exists() else ""
    if not struct_prompt:
        log(f"  NOTE: {STRUCT_PROMPT_FILE} not found — AR tabulation disabled this run.")

    folder_id = os.environ["GDRIVE_FOLDER_ID"]
    repo_id   = get_or_create_subfolder(drive, folder_id, "company_repo")
    index_id  = get_or_create_subfolder(drive, repo_id,   "_index")

    # T12 lock + priority: --all-companies is the backfill (T8) path → yield to
    # Phase 2; the PF path is Phase 2 itself → wait up to 15 min for the lock.
    _is_backfill = args.all_companies
    if not args.dry_run:
        if _is_backfill:
            got = acquire_lock(drive, index_id, _LOCK_NAME, DOC_TYPE,
                               max_age_min=_LOCK_MAX_AGE_MIN, defer_to_phase2=True)
        else:
            got = acquire_lock(drive, index_id, _LOCK_NAME, DOC_TYPE,
                               max_age_min=_LOCK_MAX_AGE_MIN, wait_min=15)
        if not got:
            log("  Lock unavailable (yielded to Phase 2 / timed out) — exiting cleanly.")
            sys.exit(0)
        atexit.register(release_lock, drive, index_id, _LOCK_NAME)

    queue = load_queue(drive, index_id)
    pending_mask = (queue["status"] == "pending") & (queue["doc_type"] == DOC_TYPE)
    pending_idx = queue.index[pending_mask].tolist()
    log(f"Queue: {len(queue)} total rows, {len(pending_idx)} pending {DOC_TYPE}")

    # T8: optional freshness window (discovered_at within N hours).
    if args.max_age_hours and "discovered_at" in queue.columns:
        from datetime import timedelta
        cutoff = (datetime.now() - timedelta(hours=args.max_age_hours)) \
            .isoformat(timespec="seconds")
        before = len(pending_idx)
        pending_idx = [i for i in pending_idx
                       if str(queue.loc[i, "discovered_at"]) >= cutoff]
        log(f"  After {args.max_age_hours:.0f}h freshness filter: "
            f"{len(pending_idx)}/{before} to process")

    # Portfolio filter: skip non-portfolio companies (rows stay pending for
    # future runs). T8 --all-companies bypasses (every fresh AR gets judged).
    if args.all_companies:
        log("  --all-companies: portfolio filter bypassed (T8)")
    else:
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
        # Priority yield: a backfill run steps aside the moment Phase 2 wants the
        # lock — finishes the current doc loop iteration boundary and exits cleanly.
        if _is_backfill and not args.dry_run and phase2_beacon_fresh(drive, index_id):
            log("  Phase 2 became active — yielding lock, exiting cleanly.")
            break
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

            size_mb = len(pdf_bytes) / (1024 * 1024)
            if size_mb > MAP_REDUCE_THRESHOLD_MB:
                markdown_text = _process_with_map_reduce(
                    gemini, pdf_bytes, prompt, display_name
                )
            else:
                markdown_text = gemini.call(pdf_bytes, prompt, display_name)

            log(f"  Gemini response: {len(markdown_text):,} chars")

            if args.dry_run:
                g_rows, rf_rows = [], []
                try:
                    g_rows, rf_rows = tabulate_ar(
                        gemini, struct_prompt, markdown_text, row,
                        _extract_fy_year(markdown_text, row), "")
                except Exception as e:
                    print(f"  structured pass failed: {str(e)[:90]}")
                print(f"\n{'='*60}\nDRY RUN — {row.get('symbol')}\n"
                      f"{markdown_text[:800]}\n"
                      f"-- structured pass: guidance={len(g_rows)} "
                      f"red_flags={len(rf_rows)} --\n{'='*60}\n")
                counts["processed"] += 1
                continue

            facts = parse_gemini_response(markdown_text, row)
            log(f"  Parsed: fy={facts['fy_year'] or 'unknown'}, "
                f"revenue={facts['revenue_q']}, pat={facts['pat_q']}")

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

            upsert_facts(drive, index_id, facts)

            # Structured tabulation: a SEPARATE bounded JSON-only pass over the report
            # (best-effort). Additive — on any failure the markdown + quarterly_facts
            # still persist (no regression). Same lock held.
            try:
                g_rows, rf_rows = tabulate_ar(
                    gemini, struct_prompt, markdown_text, row, facts["fy_year"],
                    datetime.now().isoformat(timespec="seconds"))
                _upsert_ar(drive, index_id, "ar_guidance.parquet", AR_GUIDANCE_COLS, g_rows)
                _upsert_ar(drive, index_id, "ar_red_flags.parquet", AR_REDFLAG_COLS, rf_rows)
                log(f"  Tabulated: guidance={len(g_rows)}, red_flags={len(rf_rows)}")
            except RateLimitExhausted:
                log("  Structured pass: keys exhausted — tabulation skipped this doc.")
            except Exception as _e:
                log(f"  WARNING: AR tabulation failed ({str(_e)[:100]}) — "
                    f"markdown + quarterly_facts still saved.")

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
        print("Output: company_repo/_index/quarterly_facts.parquet")
        print("Output: company_repo/_index/ar_guidance.parquet")
        print("Output: company_repo/_index/ar_red_flags.parquet")
        print("Output: company_repo/<key>/company_page.md")
        print("Output: company_repo/_daily/annual_report_DD_MMMYYYY.md")


if __name__ == "__main__":
    main()
