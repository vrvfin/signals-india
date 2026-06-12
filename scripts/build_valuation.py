"""
Phase 3 — T4.1 Valuation.

Computes a per-company valuation "cheapness" score (0-100) and writes:

  company_repo/_index/valuation.parquet   — one row per company
  company_repo/_index/valuation.csv       — CSV snapshot (same rows)

Two cheapness signals are blended (whichever are available — graceful degradation):

  1. Cross-sectional  : P/E percentile *within the stock's market-cap segment*
                        (largecap / midcap / smallcap / microcap). Cheaper than
                        peers => higher score. Always available from the live
                        fundamentals summary.
  2. Own 3-yr history : today's P/E percentile vs the company's own ~3-yr P/E
                        distribution, where historical P/E = historical close
                        (data/ohlcv) / trailing-12m EPS (from T2's
                        financials_3stmt.parquet). DATA_MISSING until T2 lands.

A PEG-style proxy (P/E / 1-yr profit growth) is also percentile-ranked within
the segment and folded in as a third, lightly-weighted component.

Sources (all live in Phase 1/2 except T2):
  fundamentals/summary.parquet                  symbol, pe, market_cap_cr, profit_growth_1y, book_value
  universe/market_cap.csv                       symbol, mcap_segment   (fallback: bucketed from market_cap_cr)
  company_repo/_index/company_universe.csv      isin <-> nse_symbol map
  company_repo/_index/financials_3stmt.parquet  (T2) quarterly EPS  [optional]
  data/ohlcv/<SYMBOL>.parquet                   close history       [optional, for own-history]

Usage:
    python scripts/build_valuation.py                      # full universe (Drive)
    python scripts/build_valuation.py --names "TCS,INFY"   # ad-hoc subset
    python scripts/build_valuation.py --local --dry-run    # offline, read/write local mirror
"""

from __future__ import annotations

import argparse
import io
import json
import os
from datetime import datetime
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

# Shared Drive layer (CLAUDE.md global rule #4 — reuse, never raw API calls).
# build_valuation.py is in scripts/, so _extractor_base is importable directly.
from _extractor_base import (
    get_drive, get_or_create_subfolder, find_file, download_bytes, upload_bytes,
)

DATA_MISSING = "DATA_MISSING"

VALUATION_COLS = [
    "isin", "symbol", "company_name", "pe", "mcap_segment",
    "pe_pctile_segment", "pe_pctile_own3y", "peg_proxy",
    "valuation_score", "basis", "computed_at",
]

# Market-cap segment thresholds (₹ crore) — mirror enrich_market_cap.py buckets.
SEG_LARGE = 20000.0
SEG_MID = 5000.0
SEG_SMALL = 500.0

MIN_PEER_GROUP = 5   # below this a segment is too thin for a meaningful percentile


def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")


# ------------------------------------------------------------------ #
#  Storage abstraction: Drive (default) or local mirror (--local)     #
#  Local mode lets the whole pipeline be exercised offline with       #
#  synthetic fixtures dropped under <local-dir>/ in the Drive layout.  #
# ------------------------------------------------------------------ #

from _t4_store import Store




# ------------------------------------------------------------------ #
#  Helpers                                                             #
# ------------------------------------------------------------------ #

def segment_for_mcap(mcap_cr) -> str | None:
    try:
        m = float(mcap_cr)
    except (TypeError, ValueError):
        return None
    if pd.isna(m):
        return None
    if m >= SEG_LARGE:
        return "Largecap"
    if m >= SEG_MID:
        return "Midcap"
    if m >= SEG_SMALL:
        return "Smallcap"
    return "Microcap"


def cheapness_pctile(series: pd.Series) -> pd.Series:
    """Per-group: lower value (cheaper) -> higher 0-100 score.

    NaN/non-positive entries stay NaN. Groups thinner than MIN_PEER_GROUP get a
    neutral 50 (not enough peers to rank reliably)."""
    valid = series.where(series > 0)
    n = valid.notna().sum()
    if n < MIN_PEER_GROUP:
        return pd.Series([50.0 if pd.notna(v) else float("nan")
                          for v in valid], index=series.index)
    rank_pct = valid.rank(pct=True)          # cheapest -> smallest
    return (1.0 - rank_pct) * 100.0          # cheapest -> ~100


def own_history_pctile(store: Store, symbol: str, eps_3stmt: pd.DataFrame | None,
                       current_pe) -> float | None:
    """Today's P/E percentile vs the company's own ~3-yr history. Returns 0-100
    (cheaper-than-own-history => higher) or None if the data isn't present.

    Reads T2's financials_3stmt.parquet — frozen schema FIN3_COLS
    (statement/line_item/period/period_type/value). Quarterly EPS rows
    (period like 'Mar 2026') are rolled into a trailing-12m EPS series, then
    divided into the close on/before each period end to form a historical P/E."""
    if eps_3stmt is None or eps_3stmt.empty or current_pe in (None, "") or current_pe <= 0:
        return None
    try:
        eps = eps_3stmt[
            (eps_3stmt["symbol"].astype(str).str.upper() == symbol.upper())
            & (eps_3stmt["line_item"].astype(str).str.upper() == "EPS")
            & (eps_3stmt["period_type"].astype(str).str.lower() == "quarterly")
        ].copy()
        if eps.empty:
            return None
        # Screener quarter headers ('Mar 2026') -> month-end timestamp
        eps["pdate"] = (pd.to_datetime(eps["period"], format="%b %Y", errors="coerce")
                        + pd.offsets.MonthEnd(0))
        eps = eps.dropna(subset=["pdate"]).sort_values("pdate")
        eps["value"] = pd.to_numeric(eps["value"], errors="coerce")
        eps["ttm_eps"] = eps["value"].rolling(4).sum()
        eps = eps.dropna(subset=["ttm_eps"])
        eps = eps[eps["ttm_eps"] > 0]
        if len(eps) < 4:
            return None
        ohlcv = store.read_parquet(["data", "ohlcv", f"{symbol.upper()}.parquet"])
        if ohlcv is None or ohlcv.empty:
            return None
        ohlcv = ohlcv.copy()
        ohlcv["date"] = pd.to_datetime(ohlcv["date"], errors="coerce")
        ohlcv = ohlcv.dropna(subset=["date"]).sort_values("date")
        hist_pe = []
        for _, r in eps.iterrows():
            window = ohlcv[ohlcv["date"] <= r["pdate"]]
            if window.empty:
                continue
            close = float(window.iloc[-1]["close"])
            hist_pe.append(close / float(r["ttm_eps"]))
        hist_pe = [p for p in hist_pe if p > 0]
        if len(hist_pe) < 4:
            return None
        # cheaper-than-history fraction: share of history that was MORE expensive
        frac_more_expensive = sum(1 for p in hist_pe if p >= current_pe) / len(hist_pe)
        return round(frac_more_expensive * 100.0, 1)
    except Exception as e:
        log(f"  own-history P/E failed for {symbol}: {str(e)[:80]}")
        return None


def build_isin_map(store: Store) -> dict:
    """nse_symbol(upper) -> (isin, company_name)."""
    out = {}
    cu = store.read_csv(["company_repo", "_index", "company_universe.csv"])
    if cu is not None and not cu.empty:
        sym_col = "nse_symbol" if "nse_symbol" in cu.columns else "symbol"
        name_col = "name" if "name" in cu.columns else None
        for _, r in cu.iterrows():
            sym = str(r.get(sym_col, "")).strip().upper()
            if sym:
                out[sym] = (str(r.get("isin", "")).strip(),
                            str(r.get(name_col, "")).strip() if name_col else "")
    # fallback master_list
    ml = store.read_csv(["universe", "master_list.csv"])
    if ml is not None and not ml.empty and "symbol" in ml.columns:
        for _, r in ml.iterrows():
            sym = str(r.get("symbol", "")).strip().upper()
            if sym and sym not in out:
                out[sym] = (str(r.get("isin", "")).strip(),
                            str(r.get("name", "")).strip())
    return out


# ------------------------------------------------------------------ #
#  Main                                                                #
# ------------------------------------------------------------------ #

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--names", type=str, default=None,
                    help="Comma-separated symbols to restrict to (ad-hoc).")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--local", action="store_true",
                    help="Read/write a local mirror dir instead of Drive (offline).")
    ap.add_argument("--local-dir", type=str, default=None,
                    help="Local mirror root (default <repo>/.t4_local).")
    ap.add_argument("--dry-run", action="store_true",
                    help="Compute and log, but do not write outputs.")
    args = ap.parse_args()

    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
    local_dir = Path(args.local_dir) if args.local_dir else \
        Path(__file__).resolve().parent.parent / ".t4_local"
    store = Store(args.local, local_dir)
    log(f"build_valuation — mode={'LOCAL' if args.local else 'DRIVE'} "
        f"{'(dry-run)' if args.dry_run else ''}")

    summary = store.read_parquet(["fundamentals", "summary.parquet"])
    if summary is None or summary.empty:
        log("ERROR: fundamentals/summary.parquet not found or empty — cannot value.")
        return
    summary = summary.copy()
    summary["symbol"] = summary["symbol"].astype(str).str.upper()

    # market-cap segment (prefer explicit file; else bucket from summary mcap)
    seg_map = {}
    mc = store.read_csv(["universe", "market_cap.csv"])
    if mc is not None and not mc.empty and "mcap_segment" in mc.columns:
        for _, r in mc.iterrows():
            seg_map[str(r["symbol"]).strip().upper()] = str(r["mcap_segment"]).strip()
    summary["mcap_segment"] = summary.apply(
        lambda r: seg_map.get(r["symbol"]) or segment_for_mcap(r.get("market_cap_cr")),
        axis=1,
    )

    # restrict scope
    if args.names:
        wanted = {s.strip().upper() for s in args.names.split(",") if s.strip()}
        summary = summary[summary["symbol"].isin(wanted)]
    if args.limit:
        summary = summary.head(args.limit)
    if summary.empty:
        log("No rows after scope filter — nothing to do.")
        return

    summary["pe"] = pd.to_numeric(summary.get("pe"), errors="coerce")
    summary["profit_growth_1y"] = pd.to_numeric(summary.get("profit_growth_1y"), errors="coerce")
    summary["peg_proxy"] = summary.apply(
        lambda r: (r["pe"] / r["profit_growth_1y"])
        if pd.notna(r["pe"]) and pd.notna(r["profit_growth_1y"]) and r["profit_growth_1y"] > 0
        else float("nan"),
        axis=1,
    )

    # cross-sectional percentiles within each segment
    summary["pe_pctile_segment"] = (
        summary.groupby("mcap_segment", dropna=False)["pe"]
        .transform(cheapness_pctile)
    )
    summary["peg_pctile_segment"] = (
        summary.groupby("mcap_segment", dropna=False)["peg_proxy"]
        .transform(cheapness_pctile)
    )

    isin_map = build_isin_map(store)
    eps_3stmt = store.read_parquet(["company_repo", "_index", "financials_3stmt.parquet"])
    if eps_3stmt is None:
        log("financials_3stmt.parquet (T2) absent — own-history P/E = DATA_MISSING.")

    rows = []
    for _, r in summary.iterrows():
        sym = r["symbol"]
        isin, cname = isin_map.get(sym, ("", ""))
        own3y = own_history_pctile(store, sym, eps_3stmt, r["pe"])

        # blend whichever cheapness components are available
        comps, basis = [], []
        if pd.notna(r["pe_pctile_segment"]):
            comps.append(float(r["pe_pctile_segment"])); basis.append("segment")
        if pd.notna(r["peg_pctile_segment"]):
            comps.append(float(r["peg_pctile_segment"])); basis.append("peg")
        if own3y is not None:
            comps.append(float(own3y)); basis.append("own3y")
        val_score = round(sum(comps) / len(comps), 1) if comps else None

        rows.append({
            "isin": isin,
            "symbol": sym,
            "company_name": cname,
            "pe": None if pd.isna(r["pe"]) else round(float(r["pe"]), 2),
            "mcap_segment": r["mcap_segment"] or DATA_MISSING,
            "pe_pctile_segment": None if pd.isna(r["pe_pctile_segment"])
            else round(float(r["pe_pctile_segment"]), 1),
            # numeric cols stay numeric (None -> NaN); a mixed float/"DATA_MISSING"
            # column is unwritable by pyarrow. DATA_MISSING is re-applied in the CSV.
            "pe_pctile_own3y": own3y,
            "peg_proxy": None if pd.isna(r["peg_proxy"]) else round(float(r["peg_proxy"]), 2),
            "valuation_score": val_score,
            "basis": "+".join(basis) if basis else DATA_MISSING,
            "computed_at": datetime.now().isoformat(timespec="seconds"),
        })

    out = pd.DataFrame(rows, columns=VALUATION_COLS)
    scored = out[out["valuation_score"].notna()]
    log(f"Valued {len(out)} companies ({len(scored)} with a score). "
        f"own-history present: {out['pe_pctile_own3y'].notna().sum()}")

    if args.dry_run:
        log("DRY-RUN — not writing. Sample:")
        print(out.head(10).fillna(DATA_MISSING).to_string(index=False))
        return

    store.write_df(["company_repo", "_index", "valuation.parquet"], out)
    store.write_df(["company_repo", "_index", "valuation.csv"], out.fillna(DATA_MISSING))
    log("Wrote valuation.parquet + valuation.csv to _index/.")


if __name__ == "__main__":
    main()
