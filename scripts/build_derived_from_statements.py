r"""
build_derived_from_statements.py — extend financials_derived to the FULL universe.

The forensic-fraud chain (build_fraud_risk -> financials_derived -> financials_3stmt)
only covered ~45 companies because financials_3stmt is a slow rolling Screener scrape.
But `fundamentals/statements/<SYM>.parquet` already holds full P&L / balance-sheet /
cash-flow history for the WHOLE universe (built by ingest_fundamentals). This script
adapts that into the exact shape build_derived_metrics.derive_company expects, reuses
its metric logic, and UNIONS the result into financials_derived — so build_fraud_risk,
build_scorecard and fraud_tracker all inherit full coverage with no other change.

Computable from statements/: cfo_pat_ratio, net_debt_ebitda (Borrowings/OP),
interest_coverage, roe_pct, npm/opm, revenue & PAT growth, fcf. The Screener "ratios"
section (receivable_days, roce_pct, wc_days) is NOT in statements/, so those passthrough
metrics stay sourced from the 45-row financials_3stmt path (kept on union) until the
statements capture is extended.

Usage:
  python scripts/build_derived_from_statements.py --dry-run        # compute + counts, no write
  python scripts/build_derived_from_statements.py --limit 50       # pilot
  python scripts/build_derived_from_statements.py                  # full universe
"""
from __future__ import annotations

import argparse
import io
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

_SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

import pandas as pd
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(_SCRIPTS_DIR), ".env"))

from _extractor_base import (get_drive, get_or_create_subfolder, find_file,
                             download_bytes, upload_bytes, log)
from build_derived_metrics import derive_company, DERIVED_COLS

# statements `statement` -> (derive_company `statement`, period_type)
STMT_MAP = {"quarterly_pl": ("income", "quarterly"),
            "annual_pl": ("income", "annual"),
            "balance_sheet": ("balance", "annual"),
            "cash_flow": ("cashflow", "annual")}
# statements `line_item` (Screener label) -> derive_company line_item
LINE_MAP = {"Sales": "Sales", "Net Profit": "Net Profit",
            "Operating Profit": "Operating Profit", "OPM %": "OPM %",
            "Interest": "Interest", "Borrowings": "Borrowings",
            "Cash from Operating Activity": "CFO",
            "Cash from Investing Activity": "CFI"}

_tl = threading.local()


def _tdrive():
    d = getattr(_tl, "d", None)
    if d is None:
        d = get_drive(); _tl.d = d
    return d


def _folder(drive, parts):
    fid = os.environ["GDRIVE_FOLDER_ID"]
    for p in parts.split("/"):
        fid = get_or_create_subfolder(drive, fid, p)
    return fid


def _num(v):
    try:
        f = float(str(v).replace(",", "").replace("%", "").strip())
        return f
    except Exception:
        return None


def _adapt(df_sym: pd.DataFrame) -> pd.DataFrame:
    """statements/<sym> rows -> df_c shaped for derive_company. Synthesises
    Net Worth = Equity Capital + Reserves per annual period."""
    rows, eq, res = [], {}, {}
    for _, r in df_sym.iterrows():
        sm = STMT_MAP.get(str(r["statement"]))
        if not sm:
            continue
        stmt, ptype = sm
        li_raw = str(r["line_item"]).strip()
        val = _num(r["value"])
        if val is None:
            continue
        period = str(r["period"])
        if stmt == "balance" and li_raw == "Equity Capital":
            eq[period] = val
        elif stmt == "balance" and li_raw == "Reserves":
            res[period] = val
        li = LINE_MAP.get(li_raw)
        if li:
            rows.append((stmt, li, ptype, period, val))
    for p in set(eq) | set(res):                    # Net Worth = Equity + Reserves
        rows.append(("balance", "Net Worth", "annual", p,
                     (eq.get(p) or 0) + (res.get(p) or 0)))
    return pd.DataFrame(rows, columns=["statement", "line_item", "period_type",
                                       "period", "value"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="Pilot: first N companies.")
    ap.add_argument("--workers", type=int, default=12, help="Parallel statement downloads.")
    ap.add_argument("--dry-run", action="store_true", help="Compute + counts; no Drive write.")
    args = ap.parse_args()

    drive = get_drive()
    idx = _folder(drive, "company_repo/_index")
    stmt_fid = _folder(drive, "fundamentals/statements")

    # symbol -> isin. statements/<SYM>.parquet were named by master_list `symbol`
    # (ingest_fundamentals), so resolve via master_list; fall back to company_universe.
    sym2isin = {}
    ml_fid = find_file(drive, _folder(drive, "universe"), "master_list.csv")
    if ml_fid:
        ml = pd.read_csv(io.BytesIO(download_bytes(drive, ml_fid))).fillna("")
        for _, r in ml.iterrows():
            s = str(r.get("symbol", "")).strip().upper()
            if s:
                sym2isin[s] = str(r.get("isin", "")).strip()
    cu = pd.read_csv(io.BytesIO(download_bytes(
        drive, find_file(drive, idx, "company_universe.csv")))).fillna("")
    cu_col = "nse_symbol" if "nse_symbol" in cu.columns else "symbol"
    for _, r in cu.iterrows():
        s = str(r.get(cu_col, "")).strip().upper()
        if s and s not in sym2isin:
            sym2isin[s] = str(r.get("isin", "")).strip()

    # list statement files
    files, tok = {}, None
    while True:
        resp = drive.files().list(
            q=f"'{stmt_fid}' in parents and trashed=false",
            fields="nextPageToken, files(id, name)", pageSize=1000, pageToken=tok).execute()
        for f in resp.get("files", []):
            if f["name"].endswith(".parquet"):
                files[f["name"][:-8].upper()] = f["id"]
        tok = resp.get("nextPageToken")
        if not tok:
            break
    syms = list(files)
    if args.limit:
        syms = syms[:args.limit]
    log(f"statement files: {len(files)} | processing {len(syms)} | workers {args.workers}")

    now = datetime.now().isoformat(timespec="seconds")

    def _one(sym):
        try:
            df = pd.read_parquet(io.BytesIO(download_bytes(_tdrive(), files[sym])))
            df_c = _adapt(df)
            if df_c.empty:
                return sym, []
            return sym, derive_company(df_c, sym2isin.get(sym, ""), sym, now)
        except Exception as e:
            return sym, f"ERR {str(e)[:50]}"

    out_rows, ok, fail = [], 0, 0
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        for i, fut in enumerate(as_completed([pool.submit(_one, s) for s in syms]), 1):
            sym, res = fut.result()
            if isinstance(res, str):
                fail += 1
            elif res:
                out_rows += res; ok += 1
            if i % 500 == 0:
                log(f"  [{i}/{len(syms)}] companies_with_metrics={ok} fail={fail} rows={len(out_rows)}")

    # Drop non-real values: derive_company's 3y-CAGR ((neg ratio)**(1/3)) can return
    # a complex number for turnaround companies, which breaks the parquet write.
    import numbers
    dropped = sum(1 for r in out_rows if not isinstance(r["value"], numbers.Real))
    out_rows = [r for r in out_rows if isinstance(r["value"], numbers.Real)]
    if dropped:
        log(f"  dropped {dropped} non-real (complex CAGR) values")
    new_df = pd.DataFrame(out_rows, columns=DERIVED_COLS)
    n_isin = new_df["isin"].astype(str).str.strip().replace({"": None}).nunique()
    log(f"derived: {len(new_df)} rows for {ok} companies ({n_isin} ISINs) | fail={fail}")
    log(f"  metric coverage: {new_df['metric'].value_counts().head(12).to_dict()}")

    if args.dry_run:
        log("DRY-RUN — no Drive write.")
        return

    # UNION with existing financials_derived: keep existing rows whose (isin, metric,
    # period) we did NOT recompute (preserves the 45-name ratio metrics from the
    # financials_3stmt path), then add ours.
    existing_fid = find_file(drive, idx, "financials_derived.parquet")
    if existing_fid:
        old = pd.read_parquet(io.BytesIO(download_bytes(drive, existing_fid)))
        key = ["isin", "metric", "period"]
        merged = (pd.concat([old, new_df], ignore_index=True)
                  .drop_duplicates(subset=key, keep="last").reset_index(drop=True))
    else:
        merged = new_df
    upload_bytes(drive, idx, "financials_derived.parquet",
                 merged.to_parquet(index=False), "application/octet-stream",
                 existing_id=existing_fid)
    log(f"financials_derived.parquet -> {len(merged)} rows "
        f"({merged['isin'].nunique()} ISINs). Run build_fraud_risk next.")


if __name__ == "__main__":
    main()
