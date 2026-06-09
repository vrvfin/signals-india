r"""
build_pead_flags.py — Phase 3 / T2.5 producer (NO Gemini, NO scraping).

The free-tier "earnings surprise": compare what management GUIDED
(guidance_tracker.parquet, from concalls) against the ACTUAL reported number
(financials_3stmt.parquet, kept fresh daily by backfill_results_3stmt --incremental)
→ a rule-based BEAT / INLINE / MISS flag per company×metric×horizon.

Output company_repo/_index/pead_flags.parquet (frozen schema):
  isin, symbol, quarter, metric, guided_value, actual_value, delta_pct, verdict, as_of
  (`quarter` holds the compared horizon, e.g. "FY26"; verdict band ±2, like GF_TRACK.)

Pure transform over two parquets — feeds strategy_pead / a watchlist / the daily mail.

Usage:
    python scripts/build_pead_flags.py --dry-run
    python scripts/build_pead_flags.py --names TCS,INFY
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from datetime import datetime
from pathlib import Path

_SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

import pandas as pd
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(_SCRIPTS_DIR), ".env"))

from _extractor_base import (get_drive, get_or_create_subfolder,
                             load_parquet, save_parquet, log)

PEAD_COLS = ["isin", "symbol", "quarter", "metric", "guided_value",
             "actual_value", "delta_pct", "verdict", "as_of"]
OUT_NAME = "pead_flags.parquet"
GUIDANCE_NAME = "guidance_tracker.parquet"
FIN3_NAME = "financials_3stmt.parquet"
BAND = 2.0   # ±2 (% for abs, pp for growth/margin) — matches GF_TRACK

GUIDANCE_COLS = ["isin", "symbol", "company_name", "quarter", "metric",
                 "guidance_type", "horizon_fy", "value", "unit", "cagr_pct",
                 "notes", "processed_at", "source_doc_id"]
FIN3_COLS = ["isin", "symbol", "statement", "line_item", "period", "period_type",
             "value", "basis", "qoq_pct", "yoy_pct", "scraped_at"]

# guidance metric (lowercased contains) -> (financials line_item, kind)
#   kind: "abs" compares ₹Cr levels; "growth" compares YoY%; "margin" compares %level
METRIC_MAP = [
    ("margin",  ("OPM %", "margin")),
    ("opm",     ("OPM %", "margin")),
    ("revenue", ("Sales", "level")),
    ("sales",   ("Sales", "level")),
    ("ebitda",  ("Operating Profit", "level")),
    ("operating profit", ("Operating Profit", "level")),
    ("pat",     ("Net Profit", "level")),
    ("net profit", ("Net Profit", "level")),
    ("profit",  ("Net Profit", "level")),
    ("eps",     ("EPS", "level")),
]


def _map_metric(metric_raw: str):
    low = str(metric_raw).lower().strip()
    for token, mapped in METRIC_MAP:
        if token in low:
            return mapped
    return (None, None)


def _to_float(s):
    """Single number, or midpoint of a 'a-b' range; strip %/₹/Cr/commas."""
    if s is None:
        return None
    txt = str(s)
    nums = re.findall(r"-?\d+(?:\.\d+)?", txt.replace(",", ""))
    if not nums:
        return None
    vals = [float(n) for n in nums[:2]]
    return round(sum(vals) / len(vals), 4)


def _fy_to_period(horizon: str):
    """'FY26'/'FY2026' -> 'Mar 2026' (India FY ends March)."""
    m = re.search(r"(\d{2,4})", str(horizon))
    if not m:
        return None
    yy = m.group(1)
    year = int(yy) if len(yy) == 4 else 2000 + int(yy)
    return f"Mar {year}"


def _verdict(delta):
    if delta is None:
        return "NA"
    if delta > BAND:
        return "BEAT"
    if delta < -BAND:
        return "MISS"
    return "INLINE"


def compute_flags(g_df: pd.DataFrame, fin_df: pd.DataFrame, now: str) -> list[dict]:
    """Pure comparator — testable offline. One flag row per matchable guidance row."""
    if g_df.empty or fin_df.empty:
        return []
    ann = fin_df[(fin_df["statement"] == "income")
                 & (fin_df["period_type"] == "annual")].copy()
    out: list[dict] = []
    for _, g in g_df.iterrows():
        isin = str(g.get("isin") or "")
        symbol = str(g.get("symbol") or "")
        li, kind = _map_metric(g.get("metric"))
        if not li:
            continue
        period = _fy_to_period(g.get("horizon_fy"))
        gval = _to_float(g.get("value"))
        if not period or gval is None:
            continue
        unit = str(g.get("unit") or "").strip()
        # A %-unit guidance on a level metric = a GROWTH guidance.
        if kind == "level" and "%" in unit:
            kind = "growth"
        sub = ann[(ann["isin"].astype(str) == isin)
                  & (ann["line_item"] == li)
                  & (ann["period"].astype(str) == period)]
        if sub.empty:
            continue   # horizon not yet reported / no actual — skip (v1)
        row = sub.iloc[-1]
        if kind == "level":
            actual = row.get("value")
            delta = ((actual - gval) / abs(gval) * 100) if (actual is not None and gval) else None
        elif kind == "growth":
            actual = row.get("yoy_pct")           # actual YoY %
            delta = (actual - gval) if actual is not None else None   # pp gap
        else:                                     # margin
            actual = row.get("value")             # OPM % level
            delta = (actual - gval) if actual is not None else None   # pp gap
        out.append({
            "isin": isin, "symbol": symbol, "quarter": str(g.get("horizon_fy") or ""),
            "metric": str(g.get("metric") or ""),
            "guided_value": gval,
            "actual_value": round(float(actual), 4) if actual is not None else None,
            "delta_pct": round(float(delta), 2) if delta is not None else None,
            "verdict": _verdict(delta), "as_of": now,
        })
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--names", default="", help="Comma list of symbols to limit to.")
    ap.add_argument("--dry-run", action="store_true", help="Compute + print; no Drive write.")
    args = ap.parse_args()

    print("Phase 3 / T2.5 — PEAD surprise flags (guidance vs actual)")
    print("-" * 60)
    drive = get_drive()
    root_id = os.environ["GDRIVE_FOLDER_ID"]
    repo_id = get_or_create_subfolder(drive, root_id, "company_repo")
    index_id = get_or_create_subfolder(drive, repo_id, "_index")

    g_df = load_parquet(drive, index_id, GUIDANCE_NAME, GUIDANCE_COLS)
    fin_df = load_parquet(drive, index_id, FIN3_NAME, FIN3_COLS)
    if g_df.empty:
        sys.exit("guidance_tracker.parquet empty — need concall guidance first.")
    if fin_df.empty:
        sys.exit("financials_3stmt.parquet empty — run backfill_results_3stmt.py first.")

    names = {s.strip().upper() for s in args.names.split(",") if s.strip()}
    if names:
        g_df = g_df[g_df["symbol"].astype(str).str.upper().isin(names)]

    now = datetime.now().isoformat(timespec="seconds")
    flags = compute_flags(g_df, fin_df, now)
    out_df = pd.DataFrame(flags, columns=PEAD_COLS)

    if args.dry_run:
        print(f"DRY RUN — {len(out_df)} flag(s). Sample:")
        print(out_df.head(30).to_string(index=False))
        if not out_df.empty:
            print("\nverdict counts:\n" + out_df["verdict"].value_counts().to_string())
        return

    save_parquet(drive, index_id, OUT_NAME, out_df)
    print(f"Flags written: {len(out_df)}  →  company_repo/_index/{OUT_NAME}")
    if not out_df.empty:
        print(out_df["verdict"].value_counts().to_string())


if __name__ == "__main__":
    main()
