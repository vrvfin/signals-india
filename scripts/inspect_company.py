r"""
inspect_company.py — local inspector: show a company's company_page.md + the structured
parquet rows written for it. For sanity-checking what the backfill actually produced.

Usage (via check_company.bat, or directly):
    python scripts/inspect_company.py TCS RELIANCE
    python scripts/inspect_company.py INE002A01018
    python scripts/inspect_company.py            # auto-picks 2 recently-processed names
"""
from __future__ import annotations
import os, sys, io, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))
import pandas as pd
from ingest_company_docs import get_drive, get_or_create_subfolder, find_file
from _extractor_base import download_bytes

# structured parquet -> column used to filter to one company (isin preferred; symbol fallback)
PARQUETS = [
    ("quarterly_facts.parquet",  "concall/AR facts"),
    ("ratings.parquet",          "credit ratings"),
    ("rating_drivers.parquet",   "rating drivers"),
    ("rating_concerns.parquet",  "rating concerns"),
    ("ar_guidance.parquet",      "AR guidance"),
    ("ar_red_flags.parquet",     "AR red flags"),
    ("gf1_guidance_statements.parquet", "concall GF1 guidance"),
]


def _load(drive, idx, name):
    fid = find_file(drive, idx, name)
    if not fid:
        return None
    try:
        return pd.read_parquet(io.BytesIO(download_bytes(drive, fid)))
    except Exception:
        return None


def _resolve(qdf, token):
    """Return (key, isin, symbol, name) for a token matching symbol/isin/key."""
    t = str(token).strip()
    for col in ("symbol", "isin", "key"):
        if col in qdf.columns:
            m = qdf[qdf[col].astype(str).str.upper() == t.upper()]
            if len(m):
                r = m.iloc[0]
                key = str(r.get("key") or r.get("isin") or r.get("symbol") or t)
                return key, str(r.get("isin","")), str(r.get("symbol","")), str(r.get("company_name",""))
    return t, t, t, ""


def inspect(drive, repo, idx, qdf, token):
    key, isin, symbol, name = _resolve(qdf, token)
    print("\n" + "=" * 72)
    print(f"COMPANY: {token}  ->  key={key}  isin={isin}  symbol={symbol}  {name}")
    print("=" * 72)

    # queue rows for this company
    qm = qdf[(qdf.get("isin","").astype(str) == isin) | (qdf.get("symbol","").astype(str) == symbol)]
    if len(qm):
        print(f"QUEUE: {len(qm)} rows  status={qm['status'].value_counts().to_dict()}  "
              f"types={qm['doc_type'].value_counts().to_dict()}")

    # company_page.md
    comp = get_or_create_subfolder(drive, repo, key)
    fid = find_file(drive, comp, "company_page.md")
    if fid:
        txt = download_bytes(drive, fid).decode("utf-8", "replace")
        secs = [ln for ln in txt.splitlines() if ln.startswith("## ")]
        print(f"\ncompany_page.md: {len(txt):,} chars, {len(secs)} sections")
        for s in secs[-6:]:
            print(f"   {s[:90]}")
        # snippet of the LAST section
        cut = txt.rfind("\n---\n## ")
        snippet = (txt[cut:cut+1100] if cut >= 0 else txt[-1100:]).strip()
        print("   --- latest section preview ---")
        print("   " + "\n   ".join(snippet.splitlines()[:22]))
    else:
        print("\ncompany_page.md: (none found)")

    # structured parquet rows
    print("\nSTRUCTURED PARQUET ROWS for this company:")
    for fn, label in PARQUETS:
        df = _load(drive, idx, fn)
        if df is None:
            print(f"   {label:<22} {fn}: (file absent)"); continue
        col = "isin" if "isin" in df.columns and isin else ("symbol" if "symbol" in df.columns else None)
        sub = df[df[col].astype(str) == (isin if col == "isin" else symbol)] if col else df.iloc[0:0]
        extra = ""
        if len(sub):
            for c in ("quarter", "fy_year", "agency", "rating", "driver", "category"):
                if c in sub.columns:
                    vals = [str(x) for x in sub[c].dropna().unique()[:5]]
                    extra = f"  e.g. {c}={vals}"; break
        print(f"   {label:<22} rows={len(sub):>4}{extra}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tokens", nargs="*", help="symbol / isin (1-2). Empty = auto-pick 2.")
    args = ap.parse_args()

    drive = get_drive(); root = os.environ["GDRIVE_FOLDER_ID"]
    repo = get_or_create_subfolder(drive, root, "company_repo")
    idx = get_or_create_subfolder(drive, repo, "_index")
    qdf = _load(drive, idx, "processing_queue.parquet")
    if qdf is None:
        print("ERROR: cannot read processing_queue.parquet"); sys.exit(1)

    tokens = args.tokens
    if not tokens:
        q = qdf.copy()
        q["pa"] = pd.to_datetime(q.get("processed_at"), errors="coerce")
        recent = q[q["status"] == "done"].sort_values("pa", ascending=False)
        tokens = [str(t) for t in recent["symbol"].dropna().unique()[:2]] or ["TCS"]
        print(f"(no tokens given — auto-picked recently processed: {tokens})")

    for tok in tokens[:3]:
        inspect(drive, repo, idx, qdf, tok)


if __name__ == "__main__":
    main()
