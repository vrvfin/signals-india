r"""
phase1_gap_analysis.py — READ-ONLY. Why isn't each dimension at 100% of the
ENTIRE universe (company_universe.csv = full Indian listed market), and what is
reachable by which data source.

NO writes. NO Gemini.

    python scripts/phase1_gap_analysis.py
"""
from __future__ import annotations

import io
import os
import sys

import pandas as pd

_SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

from write_phase2_status import get_drive, find_sub, find_file, download_bytes


def rp(drive, fid):
    return pd.read_parquet(io.BytesIO(download_bytes(drive, fid)))


def rc(drive, fid):
    return pd.read_csv(io.BytesIO(download_bytes(drive, fid)))


def clean(s):
    s = s.astype(str).str.strip()
    return s[~s.str.lower().isin(["", "nan", "none"])]


def main():
    drive = get_drive()
    root = os.environ["GDRIVE_FOLDER_ID"]
    repo = find_sub(drive, root, "company_repo")
    index_id = find_sub(drive, repo, "_index")
    fund_id = find_sub(drive, root, "fundamentals")
    uni_id = find_sub(drive, root, "universe")

    # ── ENTIRE universe ─────────────────────────────────────────────────────
    cu = rc(drive, find_file(drive, index_id, "company_universe.csv"))
    for c in ("isin", "nse_symbol", "bse_code", "board", "exchange"):
        if c not in cu.columns:
            cu[c] = ""
    cu = cu.fillna("")
    N = len(cu)
    has_nse = cu["nse_symbol"].astype(str).str.strip().replace({"nan": ""}) != ""
    has_bse = (cu["bse_code"].astype(str).str.strip().replace({"nan": ""}) != "")
    print("=" * 76)
    print(f"ENTIRE UNIVERSE  company_universe.csv = {N} ISINs")
    print("=" * 76)
    print(f"  has NSE symbol         : {has_nse.sum():>5} ({100*has_nse.mean():.1f}%)")
    print(f"  has BSE code           : {has_bse.sum():>5} ({100*has_bse.mean():.1f}%)")
    print(f"  NSE only (no bse)      : {(has_nse & ~has_bse).sum():>5}")
    print(f"  BSE only (no nse)      : {(~has_nse & has_bse).sum():>5}")
    print(f"  BOTH                   : {(has_nse & has_bse).sum():>5}")
    print(f"  NEITHER id             : {(~has_nse & ~has_bse).sum():>5}")
    print("\n  exchange col:", cu["exchange"].value_counts().to_dict())
    print("  board col   :", cu["board"].value_counts().to_dict())

    uni_isin = set(clean(cu["isin"]))
    nse_isin = set(clean(cu.loc[has_nse, "isin"]))
    bse_only_isin = set(clean(cu.loc[~has_nse & has_bse, "isin"]))
    screener_reachable = set(clean(cu.loc[has_nse | has_bse, "isin"]))  # symbol OR bse_code

    def cov(label, covered_isins, base=uni_isin):
        c = covered_isins & base
        print(f"  {label:<46} {len(c):>5} / {len(base)}  ({100*len(c)/len(base):4.1f}%)")
        return c

    # ── master_list (what Phase 1 actually iterates) ────────────────────────
    ml = rc(drive, find_file(drive, uni_id, "master_list.csv"))
    print("\n" + "=" * 76)
    print(f"master_list.csv (Phase-1 read path) = {len(ml)} rows  cols={list(ml.columns)}")
    print("=" * 76)
    ml_isin = set(clean(ml["isin"])) if "isin" in ml.columns else set()
    print(f"  master_list ISINs that ARE in entire universe : {len(ml_isin & uni_isin)}")
    print(f"  entire-universe ISINs MISSING from master_list: {len(uni_isin - ml_isin)}")

    # ── FUNDAMENTALS summary (mcap + latest results, Screener) ──────────────
    print("\n" + "=" * 76)
    print("COVERAGE vs ENTIRE UNIVERSE")
    print("=" * 76)
    summ = rp(drive, find_file(drive, fund_id, "summary.parquet"))
    print(f"\nfundamentals/summary.parquet rows={len(summ)} cols={list(summ.columns)[:14]}")
    # map summary.symbol -> isin via universe nse_symbol
    sym2isin = dict(zip(clean(cu["nse_symbol"]).str.upper(),
                        cu.loc[clean(cu["nse_symbol"]).index, "isin"]))
    if "symbol" in summ.columns:
        s_syms = summ["symbol"].astype(str).str.strip().str.upper()
        summ_isin = set(sym2isin.get(s) for s in s_syms) - {None}
    else:
        summ_isin = set(clean(summ["isin"])) if "isin" in summ.columns else set()
    fcov = cov("fundamentals/summary (any row)", summ_isin)
    # mcap present?
    mcols = [c for c in summ.columns if "market" in c.lower() or c.lower() in ("mcap", "market_cap")]
    print(f"    market-cap columns in summary: {mcols}")
    for mc in mcols:
        nn = pd.to_numeric(summ[mc], errors="coerce").notna().sum()
        print(f"      {mc}: non-null = {nn} / {len(summ)}")

    # market_cap.csv (yfinance)
    mcf_id = find_file(drive, uni_id, "market_cap.csv")
    if mcf_id:
        mcf = rc(drive, mcf_id)
        ok = pd.to_numeric(mcf.get("market_cap_cr"), errors="coerce")
        print(f"\nuniverse/market_cap.csv rows={len(mcf)}  "
              f"mcap non-null={ok.notna().sum()}  null/Unknown={ok.isna().sum()}")
        if "mcap_segment" in mcf.columns:
            print("  segment:", mcf["mcap_segment"].value_counts().to_dict())

    # ── results.parquet (Screener latest-results FEED, not per-company) ─────
    res = rp(drive, find_file(drive, index_id, "results.parquet"))
    rcov = cov("_index/results.parquet (scraped feed)", set(clean(res["isin"])) if "isin" in res else set())

    # ── derived layers ──────────────────────────────────────────────────────
    for fn in ["company_classification.parquet", "company_facts.parquet",
               "company_scorecard.parquet", "valuation.parquet",
               "investigative_fraud.parquet"]:
        fid = find_file(drive, index_id, fn)
        if fid:
            df = rp(drive, fid)
            isins = set(clean(df["isin"])) if "isin" in df.columns else set()
            cov(f"_index/{fn}", isins)

    # ── REACHABILITY summary ────────────────────────────────────────────────
    print("\n" + "=" * 76)
    print("REACHABILITY of the ENTIRE universe by data source")
    print("=" * 76)
    print(f"  NSE pipelines (OHLCV .NS / yfinance .NS / Screener by symbol):")
    print(f"      reachable now            : {len(nse_isin)} / {N} ({100*len(nse_isin)/N:.1f}%)")
    print(f"  Screener by symbol OR bse_code (fundamentals/results possible):")
    print(f"      reachable                : {len(screener_reachable)} / {N} ({100*len(screener_reachable)/N:.1f}%)")
    print(f"  BSE-only names needing .BO / bse_code path (NEW build):")
    print(f"      currently unreachable    : {len(bse_only_isin)} / {N} ({100*len(bse_only_isin)/N:.1f}%)")
    print(f"  NEITHER id (cannot be priced or scraped at all):")
    print(f"      hard-blocked             : {len(uni_isin - screener_reachable)}")

    # ── DOCUMENT COVERAGE (what IS captured, per doc_type) ──────────────────
    q = rp(drive, find_file(drive, index_id, "processing_queue.parquet"))
    qd = q.copy()
    qd["isin"] = qd["isin"].astype(str).str.strip()
    qd["doc_type"] = qd["doc_type"].astype(str)
    done = qd[qd["status"].astype(str) == "done"]
    print("\n" + "=" * 76)
    print("DOCUMENT COVERAGE — processing_queue.parquet (the ONE global ledger)")
    print("=" * 76)
    print(f"  queue rows total : {len(qd)}")
    print(f"  status totals    : {qd['status'].value_counts().to_dict()}")
    print(f"\n  per doc_type (DONE = extracted & in company_page):")
    for dt_ in ["annual_report", "concall", "rating", "results",
                "presentation", "announcement"]:
        sub = done[done["doc_type"] == dt_]
        cos = set(sub["isin"]) & uni_isin
        print(f"    {dt_:<14} done_docs={len(sub):>6}  companies={len(cos):>5} "
              f"({100 * len(cos) / N:4.1f}% of universe)")
    any_done = set(done["isin"]) & uni_isin
    print(f"\n  >= 1 done doc of ANY type : {len(any_done):>5} "
          f"({100 * len(any_done) / N:4.1f}% of universe)")
    print(f"  companies with ZERO docs  : {N - len(any_done):>5} "
          f"({100 * (N - len(any_done)) / N:4.1f}% — NOT captured at all)")
    # Annual-report history depth (distinct FY per company that has any AR)
    ar = done[done["doc_type"] == "annual_report"].copy()
    ar["fy"] = ar["announcement_date"].astype(str).str.extract(r"((?:19|20)\d{2})")
    depth = ar.dropna(subset=["fy"]).groupby("isin")["fy"].nunique()
    if len(depth):
        print(f"\n  AR history depth (yrs/company): mean={depth.mean():.1f} "
              f"median={depth.median():.0f} max={int(depth.max())} "
              f"(companies with >=1 AR = {len(depth)})")
        for band, lo, hi in [("1-2 yrs", 1, 2), ("3-5 yrs", 3, 5),
                             ("6-10 yrs", 6, 10), ("10+ yrs", 11, 999)]:
            n = int(((depth >= lo) & (depth <= hi)).sum())
            print(f"      {band:<9}: {n}")

    # ── QUEUE error breakdown ───────────────────────────────────────────────
    print("\n" + "=" * 76)
    print("QUEUE non-done breakdown (doc extraction)")
    print("=" * 76)
    print(f"  has error_msg column? {'error_msg' in q.columns or 'error' in q.columns}")
    for st in ("error", "expired", "download_failed", "pending"):
        sub = q[q["status"] == st]
        if len(sub):
            print(f"  {st:<16} {len(sub):>5}  by doc_type: {sub['doc_type'].value_counts().to_dict()}")


if __name__ == "__main__":
    main()
