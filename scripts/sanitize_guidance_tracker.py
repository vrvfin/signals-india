r"""
sanitize_guidance_tracker.py — backfill the typed guidance parse over history.

extract_concall now types every guidance cell through guidance_value.py, but that
only applies to NEW extractions. The ~26k rows already on Drive still carry the old
`unit="%" + cagr_pct=float(raw)` treatment, which both

  * FABRICATED growth  — capacity "178,000" -> 178000%, revenue "1775" (a Rs-cr
    target) -> 1775%; the digest mail takes the MAX per company, so these ranked
    first and drove the "high-growth" count, and
  * DROPPED real guidance — float() fails on any text, so "19% - 26%",
    "32.5% CAGR" and "106.33% (Derived)" were stored as NULL.

This re-parses `value` for every row and fills value_type / value_num / value_unit,
rewriting `cagr_pct` so it is populated ONLY for a genuine growth rate. The raw
`value` text is never modified — nothing is lost, it is only correctly labelled.

Idempotent: re-running produces the same result (the parse is a pure function of
`value` + `metric` + `horizon_fy`).

Usage:
    python scripts/sanitize_guidance_tracker.py --dry-run   # report only, no write
    python scripts/sanitize_guidance_tracker.py             # write back to Drive
"""
from __future__ import annotations

import argparse
import os
import sys

_SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

import pandas as pd
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(_SCRIPTS_DIR), ".env"))

from _extractor_base import (get_drive, get_or_create_subfolder, load_parquet,
                             save_parquet, log)
from guidance_value import parse_guidance_value

TRACKER = "guidance_tracker.parquet"
GUIDANCE_COLS = [
    "isin", "symbol", "company_name", "quarter", "metric",
    "guidance_type", "horizon_fy", "value", "unit", "cagr_pct", "notes",
    "processed_at", "source_doc_id",
    "value_type", "value_num", "value_unit",
]


def sanitize(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Re-parse every row. Returns (new_df, stats). Pure — testable offline."""
    if df.empty:
        return df, {}
    out = df.copy()
    parsed = [parse_guidance_value(v, m, h) for v, m, h in
              zip(out["value"], out.get("metric", ""), out.get("horizon_fy", ""))]
    old = pd.to_numeric(out.get("cagr_pct"), errors="coerce")
    out["value_type"] = [p["value_type"] for p in parsed]
    out["value_num"] = [p["value_num"] for p in parsed]
    out["value_unit"] = [p["value_unit"] for p in parsed]
    out["cagr_pct"] = [p["growth_pct"] for p in parsed]
    # keep `unit` consistent with the typed parse (it was blanket "%" before)
    out["unit"] = [p["value_unit"] for p in parsed]
    new = pd.to_numeric(out["cagr_pct"], errors="coerce")
    stats = {
        "rows": len(out),
        "old_growth": int(old.notna().sum()),
        "new_growth": int(new.notna().sum()),
        "recovered": int((old.isna() & new.notna()).sum()),
        "removed": int((old.notna() & new.isna()).sum()),
        "types": out["value_type"].value_counts().to_dict(),
        "old_max": float(old.max()) if old.notna().any() else float("nan"),
        "new_max": float(new.max()) if new.notna().any() else float("nan"),
    }
    return out, stats


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true",
                    help="Report what would change; no Drive write.")
    args = ap.parse_args()

    drive = get_drive()
    root_id = os.environ["GDRIVE_FOLDER_ID"]
    repo_id = get_or_create_subfolder(drive, root_id, "company_repo")
    index_id = get_or_create_subfolder(drive, repo_id, "_index")

    df = load_parquet(drive, index_id, TRACKER, GUIDANCE_COLS)
    if df.empty:
        log("guidance_tracker empty — nothing to sanitize.")
        return
    log(f"loaded {len(df)} guidance rows")

    out, st = sanitize(df)
    log(f"growth% populated: {st['old_growth']} -> {st['new_growth']}")
    log(f"  recovered (was NULL, now a real growth%): {st['recovered']}")
    log(f"  removed   (was a fake %, now typed non-growth): {st['removed']}")
    log(f"  max growth%: {st['old_max']:,.0f} -> {st['new_max']:,.1f}")
    log("value_type distribution:")
    for k, v in sorted(st["types"].items(), key=lambda x: -x[1]):
        log(f"    {k:<20}{v:>7}")

    if args.dry_run:
        cols = ["symbol", "metric", "horizon_fy", "value", "value_type",
                "value_num", "cagr_pct"]
        print("\nsample re-typed rows:")
        print(out[out["value_type"].isin(["absolute_inr", "ambiguous_absolute",
                                          "capacity_pct", "margin_pct"])]
              [cols].head(15).to_string(index=False))
        print("\nDRY RUN — no Drive write.")
        return

    save_parquet(drive, index_id, TRACKER, out[GUIDANCE_COLS])
    log(f"wrote _index/{TRACKER} ({len(out)} rows, typed)")


if __name__ == "__main__":
    main()
