r"""
P4b — compute_fundamental_metrics.py  (Project Guru, RESUMABLE)

Turns the long-format fundamentals dump into per-period metric rows keyed by
guru_key, with announcement dates (from results_dates_hist) and valuation joined
from the price dump. Output: guru/data/metrics/fundamental/<guru_key>.parquet
  one row per (grain, period): grain in {quarterly, annual}.

Metrics computed (NaN where source data absent — documented gaps: pledge_pct,
working_capital_days need data we don't have; those stay NaN and rules using
them simply won't trigger, per the no-drop rule):
  growth   sales_yoy_pct profit_yoy_pct eps_yoy_pct sales_qoq_pct profit_qoq_pct
           book_value_yoy_pct sales_cagr_pct_{3,5,10}y profit_cagr_pct_{3,5,10}y
  quality  net_margin_pct margin_yoy_change_pct roe_pct roce_pct debt_to_equity
           interest_coverage_ratio cfo_to_pat_pct_3y
  size/val market_cap_cr pe_ratio pe_percentile pb_ratio ps_ratio ev_ebitda
           peg_ratio earnings_yield_pct dividend_yield_pct dps_yoy_pct
           dividend_paid_years_{3,5,10}y net_profit_cr
  owner    promoter_holding_pct promoter_holding_change_pct fii_holding_pct
           fii_holding_change_pct retail_holding_change_{1,2}q_pct
  meta     period_end_date announcement_date base_date_estimated

Usage:
    python guru/compute_fundamental_metrics.py --dry-run
    python guru/compute_fundamental_metrics.py --limit 20
    python guru/compute_fundamental_metrics.py            # full, resumes
    python guru/compute_fundamental_metrics.py --status
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
FUND_DIR = os.path.join(DATA_DIR, "fundamentals_hist")
OHLCV_DIR = os.path.join(DATA_DIR, "ohlcv_hist")
OUT_DIR = os.path.join(DATA_DIR, "metrics", "fundamental")
STATUS_DIR = os.path.join(DATA_DIR, "_dump_status")
LEDGER_PATH = os.path.join(STATUS_DIR, "fund_metrics_ledger.parquet")

MAX_ANN_LAG_DAYS = 150   # results filing must fall within this after period-end

# Canonical output schema — EVERY file carries EVERY column (NaN where absent) so
# downstream readers never hit a missing column. Includes documented-gap metrics
# (pledge_pct, pledge_change_pct, working_capital_days) that stay NaN because the
# source data isn't captured — rules using them simply won't trigger (no-drop rule).
CANON_COLS = [
    "guru_key", "grain", "period", "period_end_date", "announcement_date",
    "base_date_estimated",
    "net_profit_cr", "sales_yoy_pct", "profit_yoy_pct", "eps_yoy_pct",
    "sales_qoq_pct", "profit_qoq_pct", "book_value_yoy_pct",
    "sales_cagr_pct_3y", "sales_cagr_pct_5y", "sales_cagr_pct_10y",
    "profit_cagr_pct_3y", "profit_cagr_pct_5y", "profit_cagr_pct_10y",
    "net_margin_pct", "margin_yoy_change_pct", "roe_pct", "roce_pct",
    "debt_to_equity", "interest_coverage_ratio", "cfo_to_pat_pct_3y",
    "market_cap_cr", "pe_ratio", "pe_percentile", "pb_ratio", "ps_ratio",
    "ev_ebitda", "peg_ratio", "earnings_yield_pct", "dividend_yield_pct",
    "dps_yoy_pct", "dividend_paid_years_3y", "dividend_paid_years_5y",
    "dividend_paid_years_10y",
    "promoter_holding_pct", "promoter_holding_change_pct", "fii_holding_pct",
    "fii_holding_change_pct", "retail_holding_change_1q_pct",
    "retail_holding_change_2q_pct",
    # documented gaps — always NaN until their data source is built (task #9 pledge)
    "pledge_pct", "pledge_change_pct", "working_capital_days",
]


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def parse_period(p: str) -> pd.Timestamp:
    try:
        return (pd.to_datetime(p, format="%b %Y") + pd.offsets.MonthEnd(0)).normalize()
    except Exception:
        return pd.NaT


def pivot_statement(df: pd.DataFrame, stmt: str) -> pd.DataFrame:
    """long -> wide: index=period (sorted by date), columns=line_item."""
    s = df[df["statement"] == stmt]
    if s.empty:
        return pd.DataFrame()
    w = s.pivot_table(index="period", columns="line_item", values="value",
                      aggfunc="first")
    w["_dt"] = [parse_period(p) for p in w.index]
    w = w.dropna(subset=["_dt"]).sort_values("_dt")
    return w


def pct_change_n(series: pd.Series, n: int) -> pd.Series:
    return (series / series.shift(n) - 1.0) * 100.0


def cagr(series: pd.Series, n: int) -> pd.Series:
    prev = series.shift(n)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = (series / prev) ** (1.0 / n) - 1.0
    out[(series <= 0) | (prev <= 0)] = np.nan
    return out * 100.0


def match_announcement(period_end: pd.Timestamp, ann_dates: list):
    """earliest results filing within [period_end, period_end+MAX_ANN_LAG]."""
    if pd.isna(period_end):
        return None, True
    lo, hi = period_end, period_end + pd.Timedelta(days=MAX_ANN_LAG_DAYS)
    cands = [d for d in ann_dates if lo <= d <= hi]
    if cands:
        return min(cands), False
    return (period_end + pd.Timedelta(days=45)).normalize(), True  # estimated fallback


def price_asof(px, when):
    if px is None or when is None or pd.isna(when):
        return np.nan
    sub = px[px["date"] <= when]
    return float(sub["close"].iloc[-1]) if len(sub) else np.nan


def estimate_shares(annual_w: pd.DataFrame) -> float:
    npf = annual_w.get("Net Profit"); eps = annual_w.get("EPS in Rs")
    if npf is None or eps is None:
        return np.nan
    both = pd.DataFrame({"n": npf, "e": eps}).dropna()
    both = both[both["e"] != 0]
    if both.empty:
        return np.nan
    return float((both["n"].iloc[-1] * 1e7) / both["e"].iloc[-1])


def _ann_and_price(index, dt_series, ann_dates, px):
    ann_dt, est, prices = [], [], []
    for p in index:
        a, e = match_announcement(dt_series.loc[p], ann_dates)
        ann_dt.append(a); est.append(e); prices.append(price_asof(px, a))
    return ann_dt, est, pd.Series(prices, index=index)


def compute_annual(w, bs, cf, shares, px, ann) -> pd.DataFrame:
    out = pd.DataFrame(index=w.index)
    out["period_end_date"] = w["_dt"]
    sales = w.get("Sales"); npf = w.get("Net Profit"); eps = w.get("EPS in Rs")
    opm = w.get("OPM %"); pbt = w.get("Profit before tax"); interest = w.get("Interest")
    payout = w.get("Dividend Payout %")
    out["net_profit_cr"] = npf
    out["sales_yoy_pct"] = pct_change_n(sales, 1) if sales is not None else np.nan
    out["profit_yoy_pct"] = pct_change_n(npf, 1) if npf is not None else np.nan
    out["eps_yoy_pct"] = pct_change_n(eps, 1) if eps is not None else np.nan
    out["net_margin_pct"] = (npf / sales * 100) if (sales is not None and npf is not None) else np.nan
    out["margin_yoy_change_pct"] = (opm - opm.shift(1)) if opm is not None else np.nan
    for n in (3, 5, 10):
        out[f"sales_cagr_pct_{n}y"] = cagr(sales, n) if sales is not None else np.nan
        out[f"profit_cagr_pct_{n}y"] = cagr(npf, n) if npf is not None else np.nan
    if not bs.empty:
        eq = bs.get("Equity Capital"); res = bs.get("Reserves"); bor = bs.get("Borrowings")
        if eq is not None and res is not None:
            nw = eq.reindex(w.index) + res.reindex(w.index)
            if npf is not None:
                out["roe_pct"] = npf / nw * 100
            out["book_value_yoy_pct"] = pct_change_n(nw, 1)
            if bor is not None:
                out["debt_to_equity"] = bor.reindex(w.index) / nw
                if pbt is not None and interest is not None:
                    out["roce_pct"] = (pbt + interest) / (nw + bor.reindex(w.index)) * 100
        if interest is not None and pbt is not None:
            out["interest_coverage_ratio"] = (pbt + interest) / interest.replace(0, np.nan)
    if not cf.empty and npf is not None:
        cfo = cf.get("Cash from Operating Activity")
        if cfo is not None:
            out["cfo_to_pat_pct_3y"] = (cfo.reindex(w.index).rolling(3).sum()
                                        / npf.rolling(3).sum() * 100)
    dps = None
    if payout is not None and eps is not None:
        dps = payout / 100.0 * eps
        out["dps_yoy_pct"] = pct_change_n(dps, 1)
        paid = (payout.fillna(0) > 0).astype(int)
        for n in (3, 5, 10):
            out[f"dividend_paid_years_{n}y"] = paid.rolling(n).sum()
    ann_dt, est, price = _ann_and_price(w.index, w["_dt"], ann, px)
    out["announcement_date"] = ann_dt; out["base_date_estimated"] = est
    if shares and shares > 0:
        out["market_cap_cr"] = price * shares / 1e7
        if sales is not None:
            out["ps_ratio"] = out["market_cap_cr"] / sales
        if not bs.empty and eq is not None and res is not None:
            bvps = (eq.reindex(w.index) + res.reindex(w.index)) * 1e7 / shares
            out["pb_ratio"] = price / bvps.replace(0, np.nan)
    if eps is not None:
        out["pe_ratio"] = price / eps.replace(0, np.nan)
        out["earnings_yield_pct"] = eps / price * 100
        out["pe_percentile"] = out["pe_ratio"].rank(pct=True) * 100
        out["peg_ratio"] = out["pe_ratio"] / out["profit_yoy_pct"].replace(0, np.nan)
    if dps is not None:
        out["dividend_yield_pct"] = dps / price * 100
    out["grain"] = "annual"; out["period"] = out.index
    return out


def compute_quarterly(w, shares, px, ann) -> pd.DataFrame:
    out = pd.DataFrame(index=w.index)
    out["period_end_date"] = w["_dt"]
    sales = w.get("Sales"); npf = w.get("Net Profit"); eps = w.get("EPS in Rs")
    opm = w.get("OPM %"); op = w.get("Operating Profit")
    out["net_profit_cr"] = npf
    out["sales_yoy_pct"] = pct_change_n(sales, 4) if sales is not None else np.nan
    out["profit_yoy_pct"] = pct_change_n(npf, 4) if npf is not None else np.nan
    out["eps_yoy_pct"] = pct_change_n(eps, 4) if eps is not None else np.nan
    out["sales_qoq_pct"] = pct_change_n(sales, 1) if sales is not None else np.nan
    out["profit_qoq_pct"] = pct_change_n(npf, 1) if npf is not None else np.nan
    out["net_margin_pct"] = (npf / sales * 100) if (sales is not None and npf is not None) else np.nan
    out["margin_yoy_change_pct"] = (opm - opm.shift(4)) if opm is not None else np.nan
    ttm_eps = eps.rolling(4).sum() if eps is not None else None
    ttm_sales = sales.rolling(4).sum() if sales is not None else None
    ttm_op = op.rolling(4).sum() if op is not None else None
    ann_dt, est, price = _ann_and_price(w.index, w["_dt"], ann, px)
    out["announcement_date"] = ann_dt; out["base_date_estimated"] = est
    if shares and shares > 0:
        out["market_cap_cr"] = price * shares / 1e7
        if ttm_sales is not None:
            out["ps_ratio"] = out["market_cap_cr"] / ttm_sales
        if ttm_op is not None:
            out["ev_ebitda"] = out["market_cap_cr"] / ttm_op.replace(0, np.nan)
    if ttm_eps is not None:
        out["pe_ratio"] = price / ttm_eps.replace(0, np.nan)
        out["earnings_yield_pct"] = ttm_eps / price * 100
        out["pe_percentile"] = out["pe_ratio"].rank(pct=True) * 100
    out["grain"] = "quarterly"; out["period"] = out.index
    return out


def compute_shareholding(df) -> pd.DataFrame:
    w = pivot_statement(df, "shareholding")
    if w.empty:
        return pd.DataFrame()
    out = pd.DataFrame(index=w.index)
    prom = w.get("Promoters"); fii = w.get("FIIs"); pub = w.get("Public")
    if prom is not None:
        out["promoter_holding_pct"] = prom
        out["promoter_holding_change_pct"] = prom - prom.shift(1)
    if fii is not None:
        out["fii_holding_pct"] = fii
        out["fii_holding_change_pct"] = fii - fii.shift(1)
    if pub is not None:
        out["retail_holding_change_1q_pct"] = pub - pub.shift(1)
        out["retail_holding_change_2q_pct"] = pub - pub.shift(2)
    return out


def process_key(guru_key, results_by_key) -> pd.DataFrame:
    df = pd.read_parquet(os.path.join(FUND_DIR, f"{guru_key}.parquet"))
    aw = pivot_statement(df, "annual_pl")
    qw = pivot_statement(df, "quarterly_pl")
    bs = pivot_statement(df, "balance_sheet")
    cf = pivot_statement(df, "cash_flow")
    shares = estimate_shares(aw) if not aw.empty else np.nan
    fp = os.path.join(OHLCV_DIR, f"{guru_key}.parquet")
    px = pd.read_parquet(fp, columns=["date", "close"]) if os.path.exists(fp) else None
    ann = sorted(results_by_key.get(guru_key, []))

    ann_m = compute_annual(aw, bs, cf, shares, px, ann) if not aw.empty else pd.DataFrame()
    qtr_m = compute_quarterly(qw, shares, px, ann) if not qw.empty else pd.DataFrame()
    sh = compute_shareholding(df)
    if not qtr_m.empty and not sh.empty:
        qtr_m = qtr_m.merge(sh, left_index=True, right_index=True, how="left")
    parts = [p for p in (ann_m, qtr_m) if not p.empty]
    if not parts:
        return pd.DataFrame()
    out = pd.concat(parts, ignore_index=True)
    out["guru_key"] = guru_key
    return out.reindex(columns=CANON_COLS)


def load_or_init_ledger(keys) -> pd.DataFrame:
    if os.path.exists(LEDGER_PATH):
        led = pd.read_parquet(LEDGER_PATH)
        new = [k for k in keys if k not in set(led["guru_key"])]
        if new:
            led = pd.concat([led, pd.DataFrame({"guru_key": new, "status": "pending",
                             "rows": 0, "error": ""})], ignore_index=True)
        return led
    return pd.DataFrame({"guru_key": keys, "status": "pending", "rows": 0, "error": ""})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--retry-errors", action="store_true")
    ap.add_argument("--status", action="store_true")
    args = ap.parse_args()

    keys = [os.path.basename(f)[:-8] for f in glob.glob(os.path.join(FUND_DIR, "*.parquet"))]
    led = load_or_init_ledger(keys)
    if args.status:
        print(f"{led['status'].value_counts().to_dict()} | metric rows: "
              f"{int(led['rows'].sum()):,}")
        return

    uni = pd.read_parquet(os.path.join(DATA_DIR, "universe_hist.parquet"))
    uni["bse_code"] = uni["bse_code"].astype(str)
    bmap = uni[uni.bse_code.str.match(r"^\d+$", na=False)].set_index("bse_code")["guru_key"]
    rd = pd.read_parquet(os.path.join(DATA_DIR, "results_dates_hist.parquet"),
                         columns=["SCRIP_CD", "NEWS_DT"])
    rd["SCRIP_CD"] = rd["SCRIP_CD"].astype(str)
    rd["guru_key"] = rd["SCRIP_CD"].map(bmap)
    # format='ISO8601': column mixes ms / non-ms ISO stamps; default inference
    # coerces all pre-2017 rows to NaT (silently drops 9y of announcement dates,
    # forcing the estimated-base-date fallback). See P5b bug note 2026-07-05.
    rd["NEWS_DT"] = pd.to_datetime(rd["NEWS_DT"], format="ISO8601",
                                   errors="coerce").dt.tz_localize(None).dt.normalize()
    rd = rd.dropna(subset=["guru_key", "NEWS_DT"])
    results_by_key = rd.groupby("guru_key")["NEWS_DT"].apply(list).to_dict()
    log(f"results-date index: {len(results_by_key)} companies")

    todo_mask = led["status"].eq("pending")
    if args.retry_errors:
        todo_mask |= led["status"].eq("error")
    todo = led[todo_mask]
    if args.limit:
        todo = todo.head(args.limit)
    log(f"fundamentals metric compute: {len(todo)} of {len(led)} keys")

    if args.dry_run:
        log("DRY RUN — first 10:"); [print("  ", k) for k in todo["guru_key"].head(10)]
        return

    os.makedirs(OUT_DIR, exist_ok=True)
    n_done = n_empty = n_err = 0
    for i, (idx, row) in enumerate(todo.iterrows(), 1):
        k = row["guru_key"]
        try:
            out = process_key(k, results_by_key)
            if out.empty:
                led.at[idx, "status"] = "empty"; n_empty += 1
            else:
                out.to_parquet(os.path.join(OUT_DIR, f"{k}.parquet"), index=False)
                led.at[idx, "status"] = "done"; led.at[idx, "rows"] = len(out); n_done += 1
            led.at[idx, "error"] = ""
        except Exception as e:
            led.at[idx, "status"] = "error"; led.at[idx, "error"] = str(e)[:200]; n_err += 1
        if i % 200 == 0:
            os.makedirs(STATUS_DIR, exist_ok=True)
            led.to_parquet(LEDGER_PATH, index=False)
            log(f"  {i}/{len(todo)} (done={n_done} empty={n_empty} err={n_err})")
    os.makedirs(STATUS_DIR, exist_ok=True)
    led.to_parquet(LEDGER_PATH, index=False)
    log(f"RUN COMPLETE: done={n_done} empty={n_empty} err={n_err}")


if __name__ == "__main__":
    main()
