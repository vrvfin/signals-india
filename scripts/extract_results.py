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
import atexit
import os
import re
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

# Ensure the scripts/ directory is on sys.path so _extractor_base can be found
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _extractor_base import (
    load_table,
    RateLimitExhausted, GeminiKeyPool, get_drive, load_api_keys, P1_MODELS,
    log, get_or_create_subfolder,
    load_queue, save_queue,
    load_parquet, save_parquet,
    download_bytes,
    extract_md_tables, clean_val, try_float, identify_metric,
    append_company_page, append_day_page,
    load_portfolio_isins,
    acquire_lock, release_lock,
)

# T12: SAME lock the concall extractor uses → one global mutex on the shared queue /
# company_page.md / parquets, so a Phase-2 live run and a backfill run can never
# write them concurrently.
_LOCK_NAME = "_extract.lock"
_LOCK_MAX_AGE_MIN = 360

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

# ---- units (user-approved 2026-10-24) -------------------------------------------------
# results_gemini held revenue/PAT in the filing's OWN scale under a "(Cr)" label: of 65
# rows comparable with Screener, 10 were 10x / 100x off (E2E 1,568 vs 157, BETA 12,713 vs
# 126, PRECWIRE 177,048 vs 1,779) and TATVA's PAT was 214.5 against a real ~21.5. The
# prompt now states the scale ([Unit] ...), and the money rows are converted HERE, in the
# text, so the company page and the parsed facts both carry crore.
_UNIT_TO_CR = {"crore": 1.0, "crores": 1.0, "cr": 1.0, "lakh": 0.01, "lakhs": 0.01,
               "lac": 0.01, "million": 0.1, "mn": 0.1, "billion": 100.0, "bn": 100.0,
               "thousand": 0.00001, "thousands": 0.00001}
_UNIT_LINE = re.compile(r"(?im)^\s*\[unit\]\s*([A-Za-z]+)\s*$")
_MONEY_ROW = re.compile(r"(?i)^\s*\|?\s*(revenue|sales|total income|ebitda|ebidta|pat|net profit|"
                        r"profit after tax)\b")


def _fmt_cr(v: float) -> str:
    return f"{v:,.2f}".replace(",", "")


def normalize_result_units(text: str) -> tuple[str, str]:
    """(text with money rows in crore and labelled '(Cr)', unit found). Pure.

    Only the Revenue / EBITDA / PAT rows are scaled; EPS (Rs), margins and the YoY%
    column are left alone. No [Unit] line (old prompt, or the model omitted it) or
    'Unknown' -> text unchanged, so existing behaviour holds."""
    m = _UNIT_LINE.search(str(text or ""))
    unit = (m.group(1).lower() if m else "")
    scale = _UNIT_TO_CR.get(unit)
    if scale is None:
        return text, unit
    out = []
    for line in str(text).splitlines():
        if _MONEY_ROW.match(line) and line.count("|") >= 2 and "%" not in line.split("|")[0]:
            lead = line.lstrip().startswith("|")
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            label = re.sub(r"\s*\((cr|crore|lakhs?|mn|million|bn|billion)\)\s*$", "",
                           cells[0], flags=re.I).strip()
            new = [f"{label} (Cr)"]
            for j, c in enumerate(cells[1:], 1):
                # the last column is YoY% in the prompt's table: never scaled
                if j == len(cells) - 1 and len(cells) >= 4:
                    new.append(c)
                    continue
                v = try_float(clean_val(c)) if c else None
                new.append(_fmt_cr(v * scale) if v is not None and "%" not in c else c)
            line = ("| " if lead else "") + " | ".join(new) + (" |" if lead else "")
        out.append(line)
    return "\n".join(out), unit


def margin_fix(facts: dict) -> list[str]:
    """Fix an absolute figure that disagrees with its OWN stated margin by ~10x / ~100x
    (TATVA: revenue 217.32 right, PAT 214.51 against a 7.08% PAT margin = 15.4 real).
    Revenue is the anchor; only EBITDA / PAT are corrected. Mutates facts; returns notes.
    Pure apart from the mutation."""
    notes = []
    rev = try_float(clean_val(str(facts.get("revenue_cr") or "")))
    if not rev or rev <= 0:
        return notes
    for absk, pctk in (("pat_cr", "pat_margin_pct"), ("ebitda_cr", "ebitda_margin_pct")):
        v = try_float(clean_val(str(facts.get(absk) or "")))
        pct = try_float(clean_val(str(facts.get(pctk) or "")).replace("%", ""))
        if v is None or pct is None or abs(pct) < 0.5:
            continue
        implied = v / rev * 100.0
        ratio = implied / pct
        for f in (10.0, 100.0, 1000.0):
            if 0.8 <= ratio / f <= 1.25:
                facts[absk] = _fmt_cr(v / f)
                notes.append(f"{absk} {v:g} -> {v / f:g} (÷{f:g}; {pct:g}% margin on revenue {rev:g})")
                break
    return notes


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

        low_label = cells[0].lower()

        # A row labelled "PAT Margin %" identifies as `pat`, because METRIC_ALIASES is
        # first-match-wins with ("pat","pat") ahead of ("margin","margin") — so its
        # percentage was being written into pat_cr, clobbering the real PAT. Measured
        # 2026-08-14: only 25 of 123 stored rows had a numeric pat_cr. Margin rows are
        # captured by MARGIN_MAP below and must never write an absolute field.
        # The same first-match rule corrupted ebitda_cr from "EBITDA Margin %".
        is_margin_row = "margin" in low_label

        # Value from current-quarter column. Anything that is not a plain number is
        # dropped rather than stored: a repeated header row put the literal string
        # 'Revenue (Cr)' into revenue_cr on 49 of 123 rows, and a '%' in an absolute
        # field is always a mis-parse. Absent beats wrong.
        if q_col is not None and q_col < len(cells) and not is_margin_row:
            raw = clean_val(cells[q_col])
            numeric = raw != "NA" and "%" not in raw and try_float(raw) is not None
            if metric in ("revenue", "sales") and numeric:
                facts["revenue_cr"] = raw
            elif metric == "ebitda" and numeric:
                facts["ebitda_cr"] = raw
            elif metric == "pat" and numeric:
                facts["pat_cr"] = raw
            elif metric == "eps" and numeric:
                facts["eps"] = raw

        # Margin rows (labelled like "EBITDA Margin %" or "PAT Margin %")
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
    # load_table, NOT load_parquet: this frame is written straight back, and
    # load_parquet ends in `return df[cols]` - it SLICES - so any column this
    # module's list does not name would be DELETED for every other pipeline.
    df = load_table(drive, index_id, "results_gemini.parquet", RESULTS_GEMINI_COLS)
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
            markdown_text = gemini.call(pdf_bytes, prompt, display_name)
            log(f"  Gemini response: {len(markdown_text):,} chars")

            if args.dry_run:
                print(f"\n{'='*60}\nDRY RUN — {row.get('symbol')}\n"
                      f"{markdown_text[:800]}\n{'='*60}\n")
                counts["processed"] += 1
                continue

            markdown_text, _unit = normalize_result_units(markdown_text)
            facts = parse_gemini_response(markdown_text, row)
            for _n in margin_fix(facts):
                log(f"  units: {_n}")
            log(f"  Parsed: quarter={facts['quarter'] or 'unknown'}, unit={_unit or '?'}, "
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


def _self_test() -> int:
    ok = fail = 0

    def check(name, cond):
        nonlocal ok, fail
        ok, fail = (ok + 1, fail) if cond else (ok, fail + 1)
        print(f"  {'ok  ' if cond else 'FAIL'} {name}")

    row = pd.Series({"isin": "INE1", "symbol": "TATVA", "doc_id": "d1"})
    mn = ("[QuarterYear] Q2 FY27\n[Unit] Million\n\n"
          "Metric | Q2 FY27 | Q2 FY26 | YoY%\n"
          "Revenue | 2173.20 | 1670.55 | 30.09%\nEBITDA | 397.43 | 330.95 | 19.97%\n"
          "PAT | 153.87 | 104.46 | 47.30%\nEPS (Rs) | 6.57 | 4.47 | 47.00%\n"
          "EBITDA Margin % | 18.30% | 19.81% | -1.51%\nPAT Margin % | 7.08% | 6.25% | 0.83%\n")
    t, u = normalize_result_units(mn)
    f = parse_gemini_response(t, row)
    check("Million -> crore: revenue 2173.20 mn = 217.32 cr", u == "million"
          and try_float(f["revenue_cr"]) == 217.32)
    check("PAT scaled too, EPS and margins untouched",
          try_float(f["pat_cr"]) == 15.39 and try_float(f["eps"]) == 6.57
          and "18.30%" in t and "Revenue (Cr)" in t)
    check("YoY column never scaled", "30.09%" in t and "47.30%" in t)
    lakh = ("[Unit] Lakh\nMetric | Q1 FY27 | Q1 FY26 | YoY%\nRevenue | 12713.21 | 10000 | 27%\n")
    t2, _ = normalize_result_units(lakh)
    check("Lakh -> crore: BETA 12,713 lakh = 127.13 cr",
          try_float(parse_gemini_response(t2, row)["revenue_cr"]) == 127.13)
    old = "Metric | Q2 FY27 | Q2 FY26 | YoY%\nRevenue (Cr) | 217.32 | 123 | 76%\n"
    check("no [Unit] line -> text unchanged (old behaviour)", normalize_result_units(old)[0] == old)
    check("[Unit] Unknown -> unchanged", normalize_result_units("[Unit] Unknown\n" + old)[0]
          == "[Unit] Unknown\n" + old)
    facts = {"revenue_cr": "217.32", "pat_cr": "153.87", "pat_margin_pct": "7.08%",
             "ebitda_cr": "39.74", "ebitda_margin_pct": "18.30%"}
    notes = margin_fix(facts)
    check("margin check fixes a lone 10x PAT, leaves a consistent EBITDA",
          abs(try_float(facts["pat_cr"]) - 15.39) < 0.01 and facts["ebitda_cr"] == "39.74"
          and len(notes) == 1)
    odd = {"revenue_cr": "217.32", "pat_cr": "214.51", "pat_margin_pct": "7.08%"}
    check("not a clean 10x/100x (TATVA stored row, ~14x) -> left alone, never guessed",
          margin_fix(odd) == [] and odd["pat_cr"] == "214.51")
    good = {"revenue_cr": "100", "pat_cr": "7", "pat_margin_pct": "7%"}
    check("consistent figures are never touched", margin_fix(good) == [] and good["pat_cr"] == "7")
    print(f"\nextract_results self-test: {ok} passed, {fail} failed")
    return 1 if fail else 0


if __name__ == "__main__":
    if "--self-test" in sys.argv:
        raise SystemExit(_self_test())
    main()
