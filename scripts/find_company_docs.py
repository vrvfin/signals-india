"""
find_company_docs.py  —  search company_mentions.parquet and open matching PDFs

Usage:
  python scripts/find_company_docs.py "venus remedies"
  python scripts/find_company_docs.py INE680B01014
  python scripts/find_company_docs.py --doc-type single_company_note
  python scripts/find_company_docs.py "venus remedies" --open

Invoked by find_company_docs.bat
"""
from __future__ import annotations
import os, sys, re, argparse
from pathlib import Path
import pandas as pd

INTAKE_DIR    = Path(os.getenv("RESEARCH_INTAKE_DIR", r"D:\EMA_Screener\research_intake"))
MENTIONS_PATH = INTAKE_DIR / "_ledger" / "company_mentions.parquet"


def _slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower().strip()).strip("_")


def find_pdf(file_name: str, source: str) -> Path | None:
    for base in [INTAKE_DIR / "_processed" / source, INTAKE_DIR / source]:
        p = base / file_name
        if p.exists():
            return p
    return None


def load_mentions() -> pd.DataFrame:
    if not MENTIONS_PATH.exists():
        sys.exit(
            f"No mentions index at {MENTIONS_PATH}.\n"
            "Run daily_research_summary.py on at least one PDF first."
        )
    return pd.read_parquet(MENTIONS_PATH)


def search(df: pd.DataFrame, query: str, doc_type: str | None) -> pd.DataFrame:
    if doc_type:
        return df[df.doc_type == doc_type].copy()

    q = query.strip()
    # ISIN exact match
    if re.match(r"^INE[A-Z0-9]{9}$", q.upper()):
        return df[df.isin == q.upper()].copy()

    # Slug-based fuzzy name match — also match individual words (≥4 chars)
    slug = _slug(q)
    words = [w for w in slug.split("_") if len(w) >= 4]
    mask = df.company_name_slug.str.contains(slug, na=False)
    for w in words:
        mask = mask | df.company_name_slug.str.contains(w, na=False)
    return df[mask].copy()


def main():
    ap = argparse.ArgumentParser(
        description="Find research documents that mention a company.")
    ap.add_argument("query", nargs="?", default="",
                    help="Company name, ISIN (INE...), or keyword")
    ap.add_argument("--open", action="store_true",
                    help="Open each matching PDF with the default viewer")
    ap.add_argument("--doc-type",
                    help="Filter by doc_type instead of company (e.g. single_company_note)")
    args = ap.parse_args()

    if not args.query and not args.doc_type:
        ap.print_help()
        sys.exit(1)

    df = load_mentions()
    matches = search(df, args.query, args.doc_type)

    if matches.empty:
        print(f"No documents found for: {args.query or args.doc_type!r}")
        return

    # One row per document (a doc can mention the same company in multiple rows)
    cols = ["research_n", "file_name", "source", "doc_date", "doc_type", "company_name_raw"]
    docs = (matches[cols]
            .sort_values("doc_date")
            .drop_duplicates("file_name")
            .reset_index(drop=True))

    label = repr(args.query) if args.query else repr(args.doc_type)
    print(f"\nFound {len(docs)} document(s) mentioning {label}:\n")
    for i, r in docs.iterrows():
        path = find_pdf(r.file_name, r.source)
        loc = str(path) if path else "FILE NOT ON DISK"
        print(f"  [{r.research_n:04d}] {str(r.doc_date):<12}  {r.source}/{r.file_name}")
        print(f"         Type: {r.doc_type}  |  Mention: {r.company_name_raw}")
        print(f"         Path: {loc}")
        print()

    if args.open:
        opened = 0
        for _, r in docs.iterrows():
            path = find_pdf(r.file_name, r.source)
            if path:
                os.startfile(str(path))
                opened += 1
            else:
                print(f"  WARN: {r.file_name} not found on disk (moved or deleted)")
        print(f"Opened {opened}/{len(docs)} file(s).")


if __name__ == "__main__":
    main()
