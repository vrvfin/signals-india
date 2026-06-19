r"""
backfill_pagecheck.py — T12 Stage 0 page-check ledger (the coverage DENOMINATOR).

`backfill_coverage.py` only records companies that HAVE documents, so it can never
say "we opened company X's Screener page for concalls and found 0". This ledger
fills that gap: every time `run_backfill.py` actually fetches+parses a company's
Screener page for a doc_type, it records ONE row here — INCLUDING `n_docs_found=0`.

That gives `coverage_report.py` a real denominator:
  page-checked / universe        → how far the cursor-free walk has swept
  of page-checked: has-docs / no-docs → the real document-bearing population

This is NOT a source of truth and never gates fetching (`_needs_fetch` keys off the
coverage view, not this). It is a regenerable, additive reporting ledger; if deleted
it simply repopulates as the walk re-checks pages. Keyed by (key, doc_type) — the
SAME `key = isin if isin else symbol` convention the queue + coverage use.

Reuses the Phase-2 Drive + parquet helpers verbatim (Rule 4 — no raw Drive calls).

Usage (also importable):
    python scripts/backfill_pagecheck.py            # print summary
"""
from __future__ import annotations

import os
import sys

# Ensure scripts/ on sys.path whether run from repo root or scripts/.
_SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

import pandas as pd
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(_SCRIPTS_DIR), ".env"))

from _extractor_base import load_parquet, save_parquet

PAGECHECK_FILE = "backfill_pagecheck.parquet"
PAGECHECK_COLS = ["key", "doc_type", "last_checked_at", "n_docs_found"]


def load_pagecheck(drive, index_id: str) -> pd.DataFrame:
    return load_parquet(drive, index_id, PAGECHECK_FILE, PAGECHECK_COLS)


def save_pagecheck(drive, index_id: str, df: pd.DataFrame) -> None:
    save_parquet(drive, index_id, PAGECHECK_FILE, df)


def upsert_pagecheck(existing: pd.DataFrame | None,
                     rows: list[dict]) -> pd.DataFrame:
    """Merge `rows` into `existing`, deduping by (key, doc_type) and keeping the row
    with the latest `last_checked_at`. Pure (no Drive) → unit-testable offline.

    `rows` are dicts with keys PAGECHECK_COLS. Returns a fresh DataFrame with exactly
    PAGECHECK_COLS, one row per (key, doc_type)."""
    frames = []
    if existing is not None and not existing.empty:
        frames.append(existing)
    if rows:
        frames.append(pd.DataFrame(rows))
    if not frames:
        return pd.DataFrame(columns=PAGECHECK_COLS)

    df = pd.concat(frames, ignore_index=True)
    for c in PAGECHECK_COLS:
        if c not in df.columns:
            df[c] = "" if c != "n_docs_found" else 0
    df["key"] = df["key"].astype(str).str.strip()
    df["doc_type"] = df["doc_type"].astype(str).str.strip()
    df["n_docs_found"] = pd.to_numeric(df["n_docs_found"], errors="coerce").fillna(0).astype(int)
    # Latest check wins: stable sort by timestamp, then keep the last per (key,doc_type).
    df["_ts"] = pd.to_datetime(df["last_checked_at"], errors="coerce")
    df = df.sort_values("_ts", na_position="first", kind="stable")
    df = df.drop_duplicates(subset=["key", "doc_type"], keep="last")
    return df[PAGECHECK_COLS].reset_index(drop=True)


def main() -> None:
    from ingest_company_docs import (
        get_drive, get_or_create_subfolder)
    drive = get_drive()
    root_id = os.environ["GDRIVE_FOLDER_ID"]
    repo_id = get_or_create_subfolder(drive, root_id, "company_repo")
    index_id = get_or_create_subfolder(drive, repo_id, "_index")

    pc = load_pagecheck(drive, index_id)
    print(f"Page-check rows: {len(pc)}")
    if pc.empty:
        return
    by_type = (pc.assign(has_docs=lambda d: pd.to_numeric(
                    d["n_docs_found"], errors="coerce").fillna(0) > 0)
                 .groupby("doc_type")
                 .agg(page_checked=("key", "nunique"),
                      has_docs=("has_docs", "sum"))
                 .reset_index())
    print("\nBy doc_type:")
    print(by_type.to_string(index=False))


if __name__ == "__main__":
    main()
