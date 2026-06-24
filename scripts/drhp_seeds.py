r"""
drhp_seeds.py — shared inbox for DRHP/RHP/prospectus links the company-page backfill
discovers under a company's AR subsection (CLAUDE.md Rule 7: DRHP keeps its OWN ledger,
never the global queue).

`backfill_company_docs.py` tags such links `doc_type='drhp'` and collects them instead
of queueing/extracting (SEBI public-issue PDFs are blocked for bots, so the AR pipeline
can only error on them). `run_backfill.py` writes those collected links here; the DRHP
pipeline (`ipo_drhp_watch.py`) drains this inbox, resolving + summarising each via the
non-SEBI prospectus discovery it already uses for Chittorgarh-listed IPOs.

This is an additive, regenerable inbox — NOT a source of truth. The authoritative DRHP
record stays `drhp_watch_ledger.parquet`. Keyed by a synthetic `seed_id` derived from
ISIN (falling back to symbol/name) so a company is seeded at most once.

Reuses the Phase-2 Drive + parquet helpers verbatim (Rule 4 — no raw Drive calls).
"""
from __future__ import annotations

import os
import sys
import datetime as dt

_SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

import pandas as pd

from _extractor_base import load_parquet, save_parquet

SEED_FILE = "drhp_seeds.parquet"
SEED_COLS = ["seed_id", "name", "isin", "symbol", "title", "url", "date",
             "seen_at", "status"]
# status lifecycle: new -> consumed (the DRHP pipeline picked it up; final state
# of the actual summarise lives in drhp_watch_ledger keyed by the same seed_id).


def _seed_id(isin: str = "", symbol: str = "", name: str = "") -> str:
    base = (str(isin or "").strip() or str(symbol or "").strip()
            or str(name or "").strip())
    return f"seed_{base}"


def load_seeds(drive, index_id: str) -> pd.DataFrame:
    return load_parquet(drive, index_id, SEED_FILE, SEED_COLS)


def save_seeds(drive, index_id: str, df: pd.DataFrame) -> None:
    save_parquet(drive, index_id, SEED_FILE, df)


def upsert_seeds(existing: pd.DataFrame | None, rows: list[dict]) -> pd.DataFrame:
    """Add `rows` (dicts with name/isin/symbol/title/url/date) as status='new', deduped
    by seed_id. NEVER resets an already-tracked seed (so a 'consumed' seed isn't
    re-opened just because the company page still lists the link). Pure (no Drive) →
    unit-testable offline."""
    now = dt.datetime.now().isoformat(timespec="seconds")
    new = [{
        "seed_id": _seed_id(r.get("isin"), r.get("symbol"), r.get("name")),
        "name": r.get("name", ""), "isin": r.get("isin", ""),
        "symbol": r.get("symbol", ""), "title": r.get("title", ""),
        "url": r.get("url", ""), "date": r.get("date", ""),
        "seen_at": now, "status": "new",
    } for r in (rows or [])]
    nf = pd.DataFrame(new, columns=SEED_COLS).drop_duplicates("seed_id", keep="last")
    if existing is None or existing.empty:
        return nf.reset_index(drop=True)
    known = set(existing["seed_id"].astype(str))
    nf = nf[~nf["seed_id"].isin(known)]
    return (pd.concat([existing, nf], ignore_index=True)
              .drop_duplicates("seed_id", keep="first").reset_index(drop=True))
