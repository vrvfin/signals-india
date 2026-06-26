r"""
company_report.py — one company -> its company_page.md + a multi-tab Excel pulled from the
structured parquets, both saved locally and auto-opened. Driven by company_report.bat.

Resolves an NSE / BSE / Emerge / SME symbol (or ISIN) to the company, downloads its
company_page.md, and builds an .xlsx where each tab is that company's rows from one parquet
(facts, ratings, drivers/concerns, AR guidance/red-flags, GF1-4, doc inventory).

Usage:  python scripts/company_report.py TCS
        python scripts/company_report.py INE002A01018
"""
from __future__ import annotations
import os, sys, io, re, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))
import pandas as pd
from ingest_company_docs import get_drive, get_or_create_subfolder, find_file
from _extractor_base import download_bytes

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(_ROOT, "company_reports")

# (Excel tab name <=31 chars, parquet filename) — each filtered to this company.
TABS = [
    ("Doc Inventory",     "processing_queue.parquet"),
    ("Quarterly Facts",   "quarterly_facts.parquet"),
    ("Ratings",           "ratings.parquet"),
    ("Rating Drivers",    "rating_drivers.parquet"),
    ("Rating Concerns",   "rating_concerns.parquet"),
    ("Rating Sensitivity","rating_sensitivity.parquet"),
    ("AR Guidance",       "ar_guidance.parquet"),
    ("AR Red Flags",      "ar_red_flags.parquet"),
    ("GF1 Guidance",      "gf1_guidance_statements.parquet"),
    ("GF2 Hist Guidance", "gf2_historical_guidance.parquet"),
    ("GF3 Op Visibility", "gf3_operational_visibility.parquet"),
    ("GF4 Quality Flags", "gf4_quality_flags.parquet"),
    ("Mgmt Credibility",  "mgmt_credibility.parquet"),
]


def _load(drive, idx, name):
    fid = find_file(drive, idx, name)
    if not fid:
        return None
    try:
        return pd.read_parquet(io.BytesIO(download_bytes(drive, fid)))
    except Exception:
        return None


def _resolve(qdf, uni, token):
    """Resolve a token (NSE/BSE/Emerge/SME symbol, ISIN, or name) -> (key, isin, symbol, name)."""
    t = str(token).strip()
    tl = t.upper()
    # 1) the processing queue (what's actually been processed)
    for col in ("symbol", "isin", "key", "bse_code"):
        if col in qdf.columns:
            m = qdf[qdf[col].astype(str).str.upper() == tl]
            if len(m):
                r = m.iloc[0]
                key = str(r.get("key") or r.get("isin") or r.get("symbol") or t)
                return key, str(r.get("isin", "")), str(r.get("symbol", "")), str(r.get("company_name", ""))
    # 2) the universe (covers names + bse/nse not yet in the queue)
    if uni is not None and len(uni):
        U = uni.copy()
        U.columns = [c.lower() for c in U.columns]
        for col in ("nse_symbol", "symbol", "bse_code", "isin"):
            if col in U.columns:
                m = U[U[col].astype(str).str.upper() == tl]
                if len(m):
                    r = m.iloc[0]
                    isin = str(r.get("isin", "")); sym = str(r.get("nse_symbol") or r.get("symbol") or t)
                    return (isin or sym), isin, sym, str(r.get("name") or r.get("company_name") or "")
        if "name" in U.columns:
            m = U[U["name"].astype(str).str.upper().str.contains(tl, na=False)]
            if len(m):
                r = m.iloc[0]
                isin = str(r.get("isin", "")); sym = str(r.get("nse_symbol") or r.get("symbol") or t)
                return (isin or sym), isin, sym, str(r.get("name", ""))
    return t, t, t, ""


def _filter(df, isin, symbol):
    if df is None or df.empty:
        return None
    if "isin" in df.columns and isin:
        sub = df[df["isin"].astype(str) == isin]
        if len(sub):
            return sub
    if "symbol" in df.columns and symbol:
        return df[df["symbol"].astype(str) == symbol]
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("token", help="NSE / BSE / Emerge / SME symbol, or ISIN, or name")
    ap.add_argument("--no-open", action="store_true", help="don't auto-open the files")
    args = ap.parse_args()

    drive = get_drive(); root = os.environ["GDRIVE_FOLDER_ID"]
    repo = get_or_create_subfolder(drive, root, "company_repo")
    idx = get_or_create_subfolder(drive, repo, "_index")
    qdf = _load(drive, idx, "processing_queue.parquet")
    if qdf is None:
        print("ERROR: cannot read processing_queue.parquet"); sys.exit(1)
    uni = _load(drive, idx, "company_universe.parquet")
    if uni is None:  # universe is usually a CSV
        fid = find_file(drive, idx, "company_universe.csv")
        if fid:
            try: uni = pd.read_csv(io.BytesIO(download_bytes(drive, fid)))
            except Exception: uni = None

    key, isin, symbol, name = _resolve(qdf, uni, args.token)
    print(f"Resolved '{args.token}' -> key={key}  isin={isin}  symbol={symbol}  {name}")
    os.makedirs(OUT_DIR, exist_ok=True)
    safe = re.sub(r"[^A-Za-z0-9_-]+", "_", (symbol or key or args.token)).strip("_") or "company"

    # 1) company_page.md
    md_path = None
    comp = get_or_create_subfolder(drive, repo, key)
    fid = find_file(drive, comp, "company_page.md")
    if fid:
        md_path = os.path.join(OUT_DIR, f"{safe}_company_page.md")
        with open(md_path, "wb") as fh:
            fh.write(download_bytes(drive, fid))
        print(f"  company_page.md  -> {md_path}")
    else:
        print("  company_page.md  -> (none on Drive for this company)")

    # 2) Excel with one tab per parquet (this company's rows)
    xlsx_path = os.path.join(OUT_DIR, f"{safe}_report.xlsx")
    summary_rows = []
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as xl:
        # header/summary written last; collect content first
        wrote_any = False
        for tab, fn in TABS:
            sub = _filter(_load(drive, idx, fn), isin, symbol)
            n = 0 if sub is None else len(sub)
            summary_rows.append({"tab": tab, "source_parquet": fn, "rows": n})
            if sub is not None and n:
                sub.to_excel(xl, sheet_name=tab[:31], index=False)
                wrote_any = True
        summ = pd.DataFrame(
            [{"field": "token", "value": args.token},
             {"field": "key", "value": key}, {"field": "isin", "value": isin},
             {"field": "symbol", "value": symbol}, {"field": "name", "value": name}]
            + [{"field": f"rows: {r['tab']}", "value": r["rows"]} for r in summary_rows])
        summ.to_excel(xl, sheet_name="Summary", index=False)
        if not wrote_any:
            pd.DataFrame([{"note": "No structured rows found for this company yet."}]
                         ).to_excel(xl, sheet_name="No Data", index=False)
    print(f"  Excel report    -> {xlsx_path}")
    print("\n  Tab row counts:")
    for r in summary_rows:
        if r["rows"]:
            print(f"     {r['tab']:<22} {r['rows']:>4}")

    print(f"\nLOCATION: {OUT_DIR}")

    # 3) auto-open (Windows)
    if not args.no_open:
        for p in (md_path, xlsx_path):
            if p and os.path.exists(p):
                try:
                    os.startfile(p)  # type: ignore[attr-defined]
                except Exception as e:
                    print(f"  (could not auto-open {os.path.basename(p)}: {type(e).__name__})")


if __name__ == "__main__":
    main()
