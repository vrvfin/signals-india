r"""
build_derived_metrics.py — Phase 3 / T2 producer (NO Gemini, NO scraping).

Reads company_repo/_index/financials_3stmt.parquet and computes company-wide
derived metrics into company_repo/_index/financials_derived.parquet (frozen
schema). Pure transform — depends only on the 3stmt parquet.

Derived (from income/balance/cashflow):
  npm_pct, rev_yoy_pct, rev_qoq_pct, pat_yoy_pct, pat_qoq_pct, opm_pct(passthrough),
  rev_cagr_3y_pct, fcf(=CFO+CFI), fcf_sales_pct, cfo_pat_ratio,
  net_debt_ebitda(=Borrowings/Operating Profit; gross-debt proxy),
  interest_coverage(=Operating Profit/Interest), roe_pct(=Net Profit/Net Worth)
Passed through from Screener "ratios": receivable_days, inventory_days, wc_days, roce_pct.

Usage:
    python scripts/build_derived_metrics.py
    python scripts/build_derived_metrics.py --dry-run
"""
from __future__ import annotations

import argparse
import os
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

FIN3_COLS = ["isin", "symbol", "statement", "line_item", "period", "period_type",
             "value", "basis", "qoq_pct", "yoy_pct", "scraped_at"]
DERIVED_COLS = ["isin", "symbol", "metric", "period", "period_type",
                "value", "unit", "scraped_at"]
SRC_NAME = "financials_3stmt.parquet"
OUT_NAME = "financials_derived.parquet"

RATIO_PASSTHROUGH = {  # statement="ratios" line_item -> (metric, unit)
    "receivable_days": ("receivable_days", "days"),
    "inventory_days":  ("inventory_days", "days"),
    "wc_days":         ("wc_days", "days"),
    "roce_pct":        ("roce_pct", "%"),
}


def _pct(a, b):
    if a is None or b in (None, 0):
        return None
    try:
        return round((a - b) / abs(b) * 100.0, 2)
    except Exception:
        return None


def _ratio(a, b, nd=2):
    if a is None or b in (None, 0):
        return None
    try:
        return round(a / b, nd)
    except Exception:
        return None


def _ordered(df_c, statement, line_item, ptype):
    """Ordered [(period, value)] for one series (preserves scrape/chronological order)."""
    sub = df_c[(df_c["statement"] == statement) &
               (df_c["line_item"] == line_item) &
               (df_c["period_type"] == ptype)]
    return list(zip(sub["period"].tolist(), sub["value"].tolist()))


def _dict(series):
    return {p: v for p, v in series}


def derive_company(df_c: pd.DataFrame, isin: str, symbol: str, now: str) -> list[dict]:
    out: list[dict] = []

    def emit(metric, period, ptype, value, unit):
        if value is None:
            return
        out.append({"isin": isin, "symbol": symbol, "metric": metric,
                    "period": str(period), "period_type": ptype,
                    "value": round(value, 4) if isinstance(value, float) else value,
                    "unit": unit, "scraped_at": now})

    # ---------- Quarterly (income) ----------
    sales_q = _ordered(df_c, "income", "Sales", "quarterly")
    np_q    = _ordered(df_c, "income", "Net Profit", "quarterly")
    opm_q   = _ordered(df_c, "income", "OPM %", "quarterly")
    npd     = _dict(np_q)
    for p, s in sales_q:                       # npm_pct quarterly
        emit("npm_pct", p, "quarterly", (npd.get(p) / s * 100) if s else None, "%")
    for i, (p, s) in enumerate(sales_q):       # revenue growth
        emit("rev_qoq_pct", p, "quarterly", _pct(s, sales_q[i-1][1]) if i >= 1 else None, "%")
        emit("rev_yoy_pct", p, "quarterly", _pct(s, sales_q[i-4][1]) if i >= 4 else None, "%")
    for i, (p, n) in enumerate(np_q):          # pat growth
        emit("pat_qoq_pct", p, "quarterly", _pct(n, np_q[i-1][1]) if i >= 1 else None, "%")
        emit("pat_yoy_pct", p, "quarterly", _pct(n, np_q[i-4][1]) if i >= 4 else None, "%")
    for p, v in opm_q:
        emit("opm_pct", p, "quarterly", v, "%")

    # ---------- Annual (income / balance / cashflow) ----------
    sales_a = _ordered(df_c, "income", "Sales", "annual")
    np_a = _dict(_ordered(df_c, "income", "Net Profit", "annual"))
    op_a = _dict(_ordered(df_c, "income", "Operating Profit", "annual"))
    int_a = _dict(_ordered(df_c, "income", "Interest", "annual"))
    nw_a = _dict(_ordered(df_c, "balance", "Net Worth", "annual"))
    borr_a = _dict(_ordered(df_c, "balance", "Borrowings", "annual"))
    cfo_a = _dict(_ordered(df_c, "cashflow", "CFO", "annual"))
    cfi_a = _dict(_ordered(df_c, "cashflow", "CFI", "annual"))

    for i, (p, s) in enumerate(sales_a):
        emit("npm_pct", p, "annual", (np_a.get(p) / s * 100) if s else None, "%")
        emit("fcf_sales_pct", p, "annual",
             ((cfo_a.get(p, 0) + cfi_a.get(p, 0)) / s * 100) if s else None, "%")
        if i >= 3 and sales_a[i-3][1] not in (None, 0) and s not in (None, 0):
            try:
                emit("rev_cagr_3y_pct", p, "annual",
                     ((s / sales_a[i-3][1]) ** (1/3) - 1) * 100, "%")
            except Exception:
                pass
    for p in set(cfo_a) | set(cfi_a):
        cfo, cfi = cfo_a.get(p), cfi_a.get(p)
        if cfo is not None or cfi is not None:
            emit("fcf", p, "annual", (cfo or 0) + (cfi or 0), "Cr")
    for p, cfo in cfo_a.items():
        emit("cfo_pat_ratio", p, "annual", _ratio(cfo, np_a.get(p)), "x")
    for p, borr in borr_a.items():
        emit("net_debt_ebitda", p, "annual", _ratio(borr, op_a.get(p)), "x")  # gross-debt proxy
    for p, op in op_a.items():
        emit("interest_coverage", p, "annual", _ratio(op, int_a.get(p)), "x")
    for p, nw in nw_a.items():
        emit("roe_pct", p, "annual", (np_a.get(p) / nw * 100) if nw else None, "%")

    # ---------- Passthrough from Screener "ratios" ----------
    for li, (metric, unit) in RATIO_PASSTHROUGH.items():
        for p, v in _ordered(df_c, "ratios", li, "annual"):
            emit(metric, p, "annual", v, unit)

    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true", help="Compute + print sample; no Drive write.")
    args = ap.parse_args()

    print("Phase 3 / T2 — derived company-wide metrics")
    print("-" * 60)

    drive = get_drive()
    root_id = os.environ["GDRIVE_FOLDER_ID"]
    repo_id = get_or_create_subfolder(drive, root_id, "company_repo")
    index_id = get_or_create_subfolder(drive, repo_id, "_index")

    src = load_parquet(drive, index_id, SRC_NAME, FIN3_COLS)
    if src.empty:
        sys.exit(f"{SRC_NAME} is empty — run backfill_results_3stmt.py first.")

    out_rows: list[dict] = []
    now = datetime.now().isoformat(timespec="seconds")
    for isin, df_c in src.groupby("isin", sort=False):
        sym = str(df_c["symbol"].iloc[0]) if not df_c.empty else ""
        out_rows += derive_company(df_c, str(isin), sym, now)

    out_df = pd.DataFrame(out_rows, columns=DERIVED_COLS)

    if args.dry_run:
        print(f"DRY RUN — {len(out_df)} derived rows (no Drive write). Sample:")
        print(out_df.head(25).to_string(index=False))
        if not out_df.empty:
            print("\nmetric coverage:")
            print(out_df["metric"].value_counts().to_string())
        return

    save_parquet(drive, index_id, OUT_NAME, out_df)
    print("-" * 60)
    print(f"Companies: {src['isin'].nunique()}  ·  derived rows: {len(out_df)}")
    print(f"Output: company_repo/_index/{OUT_NAME}")


if __name__ == "__main__":
    main()
