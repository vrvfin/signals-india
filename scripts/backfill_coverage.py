r"""
backfill_coverage.py — T12 derived coverage view over the ONE global queue.

This is NOT a source of truth. It is a fast, regenerable rollup of
`company_repo/_index/processing_queue.parquet`, keyed by (key, doc_type), that the
orchestrator (run_backfill.py) consults to decide — WITHOUT re-fetching a Screener
page — whether a company's documents for a requested time window are already
covered. It is rebuilt from the queue every run; if it is ever deleted it
regenerates exactly. (CLAUDE.md Rule 7: coverage is a derived view, never a
separate ledger.)

Each row records the WINDOW ACTUALLY PULLED (a date span), not a blanket "done"
flag — so a later, deeper request (e.g. "now fetch 10 years") correctly re-opens
the page for the older docs while the global queue dedup skips everything already
held.

Usage (also importable):
    python scripts/backfill_coverage.py            # rebuild + print summary
    python scripts/backfill_coverage.py --doc-type concall --top 30
"""
from __future__ import annotations

import argparse
import os
import sys

# Ensure scripts/ on sys.path whether run from repo root or scripts/.
_SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

import pandas as pd
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(_SCRIPTS_DIR), ".env"))

# Reuse Phase-2 Drive + parquet helpers verbatim (Rule 4 — no raw Drive calls).
from _extractor_base import load_parquet, save_parquet

COVERAGE_FILE = "backfill_coverage.parquet"
COVERAGE_COLS = [
    "key", "doc_type",
    "n_found", "n_done", "n_pending", "n_superseded", "n_expired",
    "covered_earliest_date", "covered_latest_date",
    "covered_earliest_period", "covered_latest_period",
    "last_fetched_at",
]

# Statuses that do NOT count as "covered": the doc was fetched but its PDF aged
# out (expired) or never downloaded (download_failed). Excluded from the covered
# date window so the company is re-fetched to recover them.
_NOT_COVERED = {"expired", "download_failed"}


def _row_key(r) -> str:
    """Queue rows are keyed by `key` (isin else symbol). Fall back defensively."""
    for c in ("key", "isin", "symbol"):
        v = str(r.get(c) or "").strip()
        if v and v.lower() not in ("nan", "none"):
            return v
    return ""


def build_coverage(queue: pd.DataFrame) -> pd.DataFrame:
    """Derive the (key, doc_type) coverage rollup from the global queue."""
    if queue is None or queue.empty:
        return pd.DataFrame(columns=COVERAGE_COLS)

    df = queue.copy()
    for c in ("key", "isin", "symbol", "doc_type", "status",
              "announcement_date", "discovered_at", "period"):
        if c not in df.columns:
            df[c] = ""
    df["_key"] = df.apply(_row_key, axis=1)
    df = df[(df["_key"] != "") & (df["doc_type"].astype(str) != "")]
    if df.empty:
        return pd.DataFrame(columns=COVERAGE_COLS)

    df["_ann"] = pd.to_datetime(
        df["announcement_date"].astype(str).str[:10], errors="coerce")
    status = df["status"].astype(str)

    rows: list[dict] = []
    for (key, doc_type), g in df.groupby(["_key", "doc_type"], sort=False):
        st = g["status"].astype(str)
        # Covered window = only rows that actually hold (or will hold) content;
        # expired / download_failed are excluded so they look "missing" and refetch.
        real = g[~st.isin(_NOT_COVERED)]
        g_dated = real[real["_ann"].notna()].sort_values("_ann")
        earliest = g_dated.iloc[0] if not g_dated.empty else None
        latest = g_dated.iloc[-1] if not g_dated.empty else None
        rows.append({
            "key": key,
            "doc_type": doc_type,
            "n_found": int(len(g)),
            "n_done": int((st == "done").sum()),
            "n_pending": int((st == "pending").sum()),
            "n_superseded": int((st == "superseded").sum()),
            "n_expired": int(st.isin(_NOT_COVERED).sum()),
            "covered_earliest_date": (earliest["_ann"].date().isoformat()
                                      if earliest is not None else ""),
            "covered_latest_date": (latest["_ann"].date().isoformat()
                                    if latest is not None else ""),
            "covered_earliest_period": (str(earliest["period"] or "")
                                        if earliest is not None else ""),
            "covered_latest_period": (str(latest["period"] or "")
                                      if latest is not None else ""),
            "last_fetched_at": str(
                pd.to_datetime(g["discovered_at"], errors="coerce").max() or ""),
        })
    return pd.DataFrame(rows, columns=COVERAGE_COLS)


def load_coverage(drive, index_id: str) -> pd.DataFrame:
    return load_parquet(drive, index_id, COVERAGE_FILE, COVERAGE_COLS)


def save_coverage(drive, index_id: str, df: pd.DataFrame) -> None:
    save_parquet(drive, index_id, COVERAGE_FILE, df)


def rebuild_and_save(drive, index_id: str, queue: pd.DataFrame) -> pd.DataFrame:
    """Rebuild coverage from the supplied queue and persist it. Returns the df."""
    cov = build_coverage(queue)
    save_coverage(drive, index_id, cov)
    return cov


def coverage_lookup(cov: pd.DataFrame) -> dict:
    """Index coverage as {(key, doc_type): row_dict} for O(1) orchestrator lookups."""
    if cov is None or cov.empty:
        return {}
    return {(str(r["key"]), str(r["doc_type"])): r.to_dict()
            for _, r in cov.iterrows()}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--doc-type", default="", help="Filter summary to one doc_type.")
    ap.add_argument("--top", type=int, default=20, help="Show N rows (by n_found).")
    args = ap.parse_args()

    # Reuse the ingest Drive helpers + queue loader (same path as run_backfill).
    from ingest_company_docs import (
        get_drive, get_or_create_subfolder, load_queue)
    drive = get_drive()
    root_id = os.environ["GDRIVE_FOLDER_ID"]
    repo_id = get_or_create_subfolder(drive, root_id, "company_repo")
    index_id = get_or_create_subfolder(drive, repo_id, "_index")

    queue = load_queue(drive, index_id)
    cov = rebuild_and_save(drive, index_id, queue)
    print(f"Coverage rows: {len(cov)}  (from {len(queue)} queue rows)")
    if cov.empty:
        return
    view = cov
    if args.doc_type:
        view = view[view["doc_type"] == args.doc_type]
    by_type = (cov.groupby("doc_type")
                  .agg(companies=("key", "nunique"),
                       found=("n_found", "sum"),
                       done=("n_done", "sum"),
                       pending=("n_pending", "sum"))
                  .reset_index())
    print("\nBy doc_type:")
    print(by_type.to_string(index=False))
    print(f"\nTop {args.top} by n_found"
          f"{' (' + args.doc_type + ')' if args.doc_type else ''}:")
    print(view.sort_values("n_found", ascending=False)
              .head(args.top).to_string(index=False))


if __name__ == "__main__":
    main()
