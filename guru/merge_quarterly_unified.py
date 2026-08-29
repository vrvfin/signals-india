r"""
P5b merge — merge_quarterly_unified.py  (Project Guru, RESUMABLE)

Unifies THREE quarterly sources into ONE per-company store:
  1. Screener quarterly     (2023+  , Rs crore, consolidated-first)   [cleanest]
  2. NSE XBRL era           (2018-24, ABSOLUTE rupees -> /1e7)        [tag-mapped]
  3. NSE HTML era           (2006-17, Rs LAKHS -> /100)               [label-mapped]

Mapping facts (verified 2026-07-07, TCS Sep-2018 vs published: exact):
  * XBRL: context OneD = current quarter; values absolute Rs; EPS in Rs as-is.
  * HTML: unit row 'Amount(Rs. in lakhs)' uniform across 150/150 sampled cos.
  * Both eras file Consolidated and Non-Consolidated separately ('consolidated').

Per (guru_key, period_end) pick ONE row: source precedence Screener > XBRL > HTML;
within a source prefer Consolidated over Standalone. Basis + source recorded.
announcement_date: NSE rows carry their own filing_date (true NSE timestamp);
Screener rows use the BSE-matched announcement date from the P4b metric store.

Output: guru/data/metrics/quarterly_unified/<guru_key>.parquet
  period_end, announcement_date, base_date_estimated, source, basis,
  sales_cr, other_income_cr, expenses_cr, depreciation_cr, finance_cost_cr,
  pbt_cr, tax_cr, pat_cr, eps, sales_label,
  sales_yoy_pct, pat_yoy_pct, eps_yoy_pct, sales_qoq_pct, pat_qoq_pct,
  net_margin_pct, margin_yoy_change_pct

Usage:
    python guru/merge_quarterly_unified.py --dry-run
    python guru/merge_quarterly_unified.py --limit 20
    python guru/merge_quarterly_unified.py            # full, resumes
    python guru/merge_quarterly_unified.py --status
"""
from __future__ import annotations

import argparse
import glob
import os
from datetime import datetime

import numpy as np
import pandas as pd

GURU_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(GURU_DIR, "data")
FACTS_DIR = os.path.join(DATA_DIR, "nse_results_facts")
XBRL_DIR = os.path.join(DATA_DIR, "nse_results_xbrl")
FUND_DIR = os.path.join(DATA_DIR, "fundamentals_hist")
PMETRIC_DIR = os.path.join(DATA_DIR, "metrics", "fundamental")
OUT_DIR = os.path.join(DATA_DIR, "metrics", "quarterly_unified")
STATUS_DIR = os.path.join(DATA_DIR, "_dump_status")
LEDGER_PATH = os.path.join(STATUS_DIR, "quarterly_unified_ledger.parquet")

# ---------- HTML-era label maps (two SEBI template generations) ----------
HTML_SALES = [
    "Total income from operations (net) ( a + b)",
    "Net Sales/Income from Operation",
    "(a) Net sales/income from operations (Net of excise duty)",
    "Total Income",                       # banks / fallback
]
HTML_PAT = [
    "Net Profit / (Loss) for the period",
    "Net Profit / (Loss) from ordinary activities after tax",
    "Net Profit / (Loss) after taxes, minority interest and share of profit / (loss) of associates",
    "Net Profit (+) / Loss (-) for the period",                      # bank template
    "Net Profit(+) / Loss(-) from Ordinary Activities after tax",    # bank template
    "Net Profit",
]
HTML_MAP_SIMPLE = {
    "other_income_cr": ["Other income", "Other Income"],
    "expenses_cr": ["Total expenses", "Total Expenditure"],
    "depreciation_cr": ["(e) Depreciation and amortisation expense", "Depreciation"],
    "finance_cost_cr": ["Finance costs", "Interest"],
    "pbt_cr": ["Profit / (Loss) from ordinary activities before tax",
               "Profit(+) / Loss(-) from Ordinary Activities before tax",  # bank
               "Total Profit Before Tax",                                   # bank
               "Profit before tax"],
    "tax_cr": ["Tax expense", "Tax Expense"],
    "eps": ["(a) Basic", "Basic EPS before Extraordinary items",
            "Basic EPS after Extraordinary items (in Rs.)",     # bank template
            "Basic EPS before Extraordinary items (in Rs.)",    # bank template
            "Basic EPS for the period"],
}

# ---------- XBRL tag map (context OneD, absolute Rs) ----------
XBRL_MAP = {
    "sales_cr": ["RevenueFromOperations", "Income"],   # Income = banks/total fallback
    "other_income_cr": ["OtherIncome"],
    "expenses_cr": ["Expenses"],
    "depreciation_cr": ["DepreciationDepletionAndAmortisationExpense"],
    "finance_cost_cr": ["FinanceCosts"],
    "pbt_cr": ["ProfitBeforeTax",
               "ProfitLossFromOrdinaryActivitiesBeforeTax",     # bank taxonomy
               "ProfitLossBeforeExceptionalItemsAndTax"],
    "tax_cr": ["TaxExpense"],
    "pat_cr": ["ProfitLossForPeriod",
               "ProfitLossForThePeriod",                        # bank taxonomy
               "ProfitLossAfterTaxesMinorityInterestAndShareOfProfitLossOfAssociates",
               "ProfitLossForPeriodFromContinuingOperations"],
}
XBRL_EPS = ["BasicEarningsLossPerShareFromContinuingOperations",
            "BasicEarningsLossPerShareFromContinuingAndDiscontinuedOperations",
            "BasicEarningsPerShareAfterExtraordinaryItems",     # bank taxonomy
            "BasicEarningsPerShareBeforeExtraordinaryItems"]

NUM_COLS = ["sales_cr", "other_income_cr", "expenses_cr", "depreciation_cr",
            "finance_cost_cr", "pbt_cr", "tax_cr", "pat_cr", "eps"]


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def _num(v) -> float:
    try:
        f = float(str(v).replace(",", "").strip())
        return f
    except (ValueError, TypeError):
        return np.nan


def _first(d: dict, labels: list[str]):
    for lb in labels:
        if lb in d and not pd.isna(d[lb]):
            return d[lb], lb
    return np.nan, None


def rows_from_html(guru_key: str) -> list[dict]:
    fp = os.path.join(FACTS_DIR, f"{guru_key}.parquet")
    # facts are keyed by SYMBOL filename — map handled by caller passing symbol
    if not os.path.exists(fp):
        return []
    d = pd.read_parquet(fp)
    out = []
    for det, sub in d.groupby("det_id"):
        vals = dict(zip(sub["line_item"], sub["value"]))
        num = {k: _num(v) for k, v in vals.items()}
        sales, slabel = _first(num, HTML_SALES)
        pat, _ = _first(num, HTML_PAT)
        row = {"period_end": pd.to_datetime(sub["period_to"].iloc[0],
                                            format="%d-%b-%Y", errors="coerce"),
               "filing_date": pd.to_datetime(sub["filing_date"].iloc[0],
                                             errors="coerce", dayfirst=True),
               "basis": ("Consolidated" if str(sub["consolidated"].iloc[0]).lower()
                         .startswith("c") else "Standalone"),
               "source": "nse_html", "sales_label": slabel,
               "sales_cr": sales / 100.0 if not pd.isna(sales) else np.nan,
               "pat_cr": pat / 100.0 if not pd.isna(pat) else np.nan}
        for col, labels in HTML_MAP_SIMPLE.items():
            v, _ = _first(num, labels)
            row[col] = v if col == "eps" else (v / 100.0 if not pd.isna(v) else np.nan)
        out.append(row)
    return out


def rows_from_xbrl(guru_key_symbol: str) -> list[dict]:
    fp = os.path.join(XBRL_DIR, f"{guru_key_symbol}.parquet")
    if not os.path.exists(fp):
        return []
    d = pd.read_parquet(fp)
    d = d[d["context"] == "OneD"]        # current quarter only (verified)
    out = []
    for det, sub in d.groupby("det_id"):
        vals = dict(zip(sub["tag"], sub["value"]))
        num = {k: _num(v) for k, v in vals.items()}
        row = {"period_end": pd.to_datetime(sub["period_to"].iloc[0],
                                            format="%d-%b-%Y", errors="coerce"),
               "filing_date": pd.to_datetime(sub["filing_date"].iloc[0],
                                             errors="coerce", dayfirst=True),
               "basis": ("Consolidated" if str(sub["consolidated"].iloc[0]).lower()
                         .startswith("c") else "Standalone"),
               "source": "nse_xbrl"}
        for col, tags in XBRL_MAP.items():
            v, lb = _first(num, tags)
            row[col] = v / 1e7 if not pd.isna(v) else np.nan
            if col == "sales_cr":
                row["sales_label"] = lb
        eps, _ = _first(num, XBRL_EPS)
        row["eps"] = eps
        out.append(row)
    return out


def rows_from_screener(guru_key: str) -> list[dict]:
    """Screener quarterly (already Rs cr) + announcement date from P4b metrics."""
    fp = os.path.join(FUND_DIR, f"{guru_key}.parquet")
    if not os.path.exists(fp):
        return []
    d = pd.read_parquet(fp)
    q = d[d["statement"] == "quarterly_pl"]
    if q.empty:
        return []
    w = q.pivot_table(index="period", columns="line_item", values="value",
                      aggfunc="first")
    # announcement dates from the P4b metric store (quarterly grain)
    ann = {}
    mp = os.path.join(PMETRIC_DIR, f"{guru_key}.parquet")
    if os.path.exists(mp):
        m = pd.read_parquet(mp, columns=["grain", "period", "announcement_date",
                                         "base_date_estimated"])
        m = m[m["grain"] == "quarterly"]
        ann = {r["period"]: (r["announcement_date"], r["base_date_estimated"])
               for _, r in m.iterrows()}
    out = []
    for period, row in w.iterrows():
        pe = pd.to_datetime(period, format="%b %Y", errors="coerce")
        if pd.isna(pe):
            continue
        pe = (pe + pd.offsets.MonthEnd(0)).normalize()
        a, est = ann.get(period, (None, True))
        out.append({"period_end": pe,
                    "filing_date": pd.to_datetime(a) if a is not None else pd.NaT,
                    "base_date_estimated": bool(est),
                    "basis": "ScreenerDefault",       # consolidated-first fallback
                    "source": "screener", "sales_label": "Sales",
                    "sales_cr": _num(row.get("Sales")),
                    "expenses_cr": _num(row.get("Expenses")),
                    "other_income_cr": _num(row.get("Other Income")),
                    "depreciation_cr": _num(row.get("Depreciation")),
                    "finance_cost_cr": _num(row.get("Interest")),
                    "pbt_cr": _num(row.get("Profit before tax")),
                    "pat_cr": _num(row.get("Net Profit")),
                    # Screener shows Tax % not amount; derive amount = PBT - PAT
                    "tax_cr": (_num(row.get("Profit before tax"))
                               - _num(row.get("Net Profit"))),
                    "eps": _num(row.get("EPS in Rs"))})
    return out


PRECEDENCE = {"screener": 0, "nse_xbrl": 1, "nse_html": 2}
BASIS_RANK = {"Consolidated": 0, "ScreenerDefault": 0, "Standalone": 1}


def unify(guru_key: str, symbol: str | None) -> pd.DataFrame:
    rows = rows_from_screener(guru_key)
    if symbol:
        rows += rows_from_xbrl(symbol)
        rows += rows_from_html(symbol)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df = df.dropna(subset=["period_end"])
    df = df[~(df[NUM_COLS].isna().all(axis=1))]
    if df.empty:
        return df
    df["_src_rank"] = df["source"].map(PRECEDENCE)
    df["_basis_rank"] = df["basis"].map(BASIS_RANK).fillna(1)
    df = (df.sort_values(["period_end", "_src_rank", "_basis_rank"])
            .drop_duplicates(subset=["period_end"], keep="first")
            .drop(columns=["_src_rank", "_basis_rank"])
            .sort_values("period_end").reset_index(drop=True))
    df["announcement_date"] = df["filing_date"]
    if "base_date_estimated" not in df.columns:
        df["base_date_estimated"] = df["announcement_date"].isna()
    df["base_date_estimated"] = df["base_date_estimated"].fillna(
        df["announcement_date"].isna())
    est = df["announcement_date"].isna()
    df.loc[est, "announcement_date"] = df.loc[est, "period_end"] + pd.Timedelta(days=45)
    # derived metrics — YoY = shift(4) is only valid on a CONTIGUOUS quarterly
    # grid; reindex to quarter-ends so gaps produce NaN, not wrong-quarter ratios
    df["_q"] = df["period_end"].dt.to_period("Q")
    df = df.drop_duplicates(subset=["_q"], keep="first")
    full = pd.period_range(df["_q"].min(), df["_q"].max(), freq="Q")
    df = df.set_index("_q").reindex(full)
    # growth ratios need a POSITIVE prior base: zero base -> inf (passes any
    # threshold — found via a shell-company chart 2026-07-09); negative base
    # flips the sign meaninglessly. Guard: prior must be > 0, else NaN.
    def _grow(base_s: pd.Series, n: int) -> pd.Series:
        prev = base_s.shift(n)
        out = (base_s / prev - 1.0) * 100
        out[~(prev > 0)] = np.nan
        return out
    for col, base in [("sales_yoy_pct", "sales_cr"), ("pat_yoy_pct", "pat_cr"),
                      ("eps_yoy_pct", "eps")]:
        df[col] = _grow(df[base], 4)
    for col, base in [("sales_qoq_pct", "sales_cr"), ("pat_qoq_pct", "pat_cr")]:
        df[col] = _grow(df[base], 1)
    # basis-mixing flags: a company can file Consolidated one quarter and only
    # Standalone another (verified: Infosys 2012) — a YoY/QoQ ratio across a
    # basis flip compares different accounting perimeters. Values are kept
    # (no-drop rule) but flagged so rules/scorecards can exclude mixed ratios.
    cons = df["basis"].isin(["Consolidated", "ScreenerDefault"])
    df["yoy_mixed_basis"] = (cons != cons.shift(4)) & df["sales_yoy_pct"].notna()
    df["qoq_mixed_basis"] = (cons != cons.shift(1)) & df["sales_qoq_pct"].notna()
    df["src_changed_vs_yoy"] = (df["source"] != df["source"].shift(4)) \
        & df["sales_yoy_pct"].notna()
    df["net_margin_pct"] = df["pat_cr"] / df["sales_cr"].replace(0, np.nan) * 100
    df["margin_yoy_change_pct"] = df["net_margin_pct"] - df["net_margin_pct"].shift(4)
    df = df[df["period_end"].notna()].reset_index(drop=True)
    df["guru_key"] = guru_key
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--retry-errors", action="store_true")
    ap.add_argument("--status", action="store_true")
    args = ap.parse_args()

    uni = pd.read_parquet(os.path.join(DATA_DIR, "universe_hist.parquet"),
                          columns=["guru_key", "nse_symbol"])
    have_fund = {os.path.basename(f)[:-8]
                 for f in glob.glob(os.path.join(FUND_DIR, "*.parquet"))}
    have_nse = {os.path.basename(f)[:-8]
                for f in glob.glob(os.path.join(FACTS_DIR, "*.parquet"))} | \
               {os.path.basename(f)[:-8]
                for f in glob.glob(os.path.join(XBRL_DIR, "*.parquet"))}
    uni["symbol"] = uni["nse_symbol"].astype(str)
    scope = uni[uni["guru_key"].isin(have_fund) | uni["symbol"].isin(have_nse)]
    scope = scope.drop_duplicates("guru_key")

    if os.path.exists(LEDGER_PATH):
        led = pd.read_parquet(LEDGER_PATH)
        new = scope[~scope["guru_key"].isin(led["guru_key"])]
        if not new.empty:
            led = pd.concat([led, pd.DataFrame(
                {"guru_key": new["guru_key"], "symbol": new["symbol"],
                 "status": "pending", "rows": 0, "error": ""})], ignore_index=True)
    else:
        led = pd.DataFrame({"guru_key": scope["guru_key"], "symbol": scope["symbol"],
                            "status": "pending", "rows": 0, "error": ""})
    if args.status:
        print(led["status"].value_counts().to_dict(),
              "| unified rows:", int(led["rows"].sum()))
        return

    todo_mask = led["status"].eq("pending")
    if args.retry_errors:
        todo_mask |= led["status"].eq("error")
    todo = led[todo_mask]
    if args.limit:
        todo = todo.head(args.limit)
    log(f"merge scope: {len(led)} companies | this run: {len(todo)}")
    if args.dry_run:
        [print("  ", r["guru_key"], r["symbol"]) for _, r in todo.head(10).iterrows()]
        return

    os.makedirs(OUT_DIR, exist_ok=True)
    n_ok = n_empty = n_err = 0
    for i, (li, r) in enumerate(todo.iterrows(), 1):
        try:
            sym = r["symbol"] if r["symbol"] not in ("", "nan", "None") else None
            df = unify(r["guru_key"], sym)
            if df.empty:
                led.at[li, "status"] = "empty"; n_empty += 1
            else:
                df.to_parquet(os.path.join(OUT_DIR, f"{r['guru_key']}.parquet"),
                              index=False)
                led.at[li, "status"] = "done"; led.at[li, "rows"] = len(df); n_ok += 1
            led.at[li, "error"] = ""
        except Exception as e:
            led.at[li, "status"] = "error"; led.at[li, "error"] = str(e)[:200]; n_err += 1
        if i % 300 == 0:
            os.makedirs(STATUS_DIR, exist_ok=True)
            led.to_parquet(LEDGER_PATH, index=False)
            log(f"  {i}/{len(todo)} (done={n_ok} empty={n_empty} err={n_err})")
    os.makedirs(STATUS_DIR, exist_ok=True)
    led.to_parquet(LEDGER_PATH, index=False)
    log(f"MERGE COMPLETE: done={n_ok} empty={n_empty} err={n_err}")


if __name__ == "__main__":
    main()
