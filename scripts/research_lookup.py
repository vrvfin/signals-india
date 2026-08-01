r"""
research_lookup.py — find broker research for a company, across every key it might be
filed under.

Workflow A ingests broker PDFs (Nuvama, Antique, Ventura, Motilal Oswal, Exencial ...)
into two tables: `research_index.parquet` (one row per DOCUMENT, with summary_md) and
`company_mentions.parquet` (one row per company MENTION inside a document). Initiating
coverage, result reviews and concall notes all land there.

WHY THIS IS NOT AN ISIN LOOKUP
Broker notes name companies the way the market does, not the way the registrar does.
Measured 2026-08-01:

    TCS      91 mentions by ISIN,  43 documents by name
    Ather    25 mentions by ISIN,  15 documents by name
    CPPLUS    0 mentions by ISIN — the registered name is "Aditya Infotech Limited"
              and the research says "CP Plus", which is the BRAND

An ISIN-keyed search returns nothing for that last case while the coverage plainly
exists. So this searches on every available key — ISIN, symbol, registered name, the
distinctive words of that name, and any alias the caller passes — and reports WHICH key
matched, so a loose hit can be told from an exact one.

Usage:
  python scripts/research_lookup.py --names TCS
  python scripts/research_lookup.py --names cpplus --alias "CP Plus"
  python scripts/research_lookup.py --names TCS Ather cpplus --summary
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import pandas as pd
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from narrative_factpack import Store, resolve, IDX

# Words that carry no identifying signal in an Indian company name. Dropping them stops
# "India"/"Limited" matching half the corpus.
_NOISE = {"limited", "ltd", "private", "pvt", "india", "indian", "the", "company",
          "corporation", "corp", "industries", "enterprises", "and", "of", "&",
          "holdings", "group", "co"}
# Broker note types worth surfacing separately — the rest are sector/macro context.
_COMPANY_DOC = ("single_company_note",)


# A name word is only usable as a search key if it is RARE in the corpus. "Energy" is a
# word in "Ather Energy Limited" and also in hundreds of other names — searching on it
# returned "Shyam Metalics & Energy Ltd." as Ather coverage (136 of 154 Ather mentions
# were this kind of false positive before gating). A word appearing in more than this
# share of mention rows is discarded as non-identifying.
_MAX_WORD_DF = 0.01


def _keywords(name: str) -> list[str]:
    """Distinctive words of a company name, longest first — 'Aditya Infotech Limited'
    -> ['infotech', 'aditya']."""
    words = [w for w in re.split(r"[^A-Za-z0-9]+", str(name).lower())
             if len(w) > 2 and w not in _NOISE]
    return sorted(set(words), key=len, reverse=True)


def _rare_keywords(name: str, cm: pd.DataFrame) -> list[str]:
    """Name words that actually identify THIS company, measured against how often each
    appears across all company mentions."""
    if cm.empty or "company_name_raw" not in cm.columns:
        return _keywords(name)[:2]
    blob = cm["company_name_raw"].astype(str).str.lower()
    n = max(1, len(blob))
    keep = []
    for w in _keywords(name):
        df = float(blob.str.contains(w, regex=False, na=False).sum()) / n
        if df <= _MAX_WORD_DF:
            keep.append((w, df))
    keep.sort(key=lambda x: x[1])          # rarest first
    return [w for w, _ in keep[:2]]


def _contains(series: pd.Series, needle: str) -> pd.Series:
    return series.astype(str).str.contains(needle, case=False, na=False, regex=False)


def lookup(store: Store, token: str, aliases: list[str] | None = None,
           max_docs: int = 40) -> dict:
    ri = store.parquet(IDX, "research_index.parquet")
    cm = store.parquet(IDX, "company_mentions.parquet")
    res = resolve(store, token)
    isin, symbol, name = res if res else (None, None, token)

    out = {"query": token, "isin": isin, "symbol": symbol, "name": name,
           "matched_by": {}, "documents": [], "mentions": 0,
           "unresolved": res is None}
    if ri.empty and cm.empty:
        out["error"] = "no research tables on Drive"
        return out

    keys: list[tuple[str, str]] = []
    if isin:
        keys.append(("isin", isin))
    if symbol:
        keys.append(("symbol", symbol))
    if name and name != token:
        keys.append(("name", name))
    keys.append(("query", token))
    for a in (aliases or []):
        keys.append(("alias", a))
    # Distinctive name words last: broadest, most likely to over-match, so they are
    # reported under their own label rather than silently inflating an exact hit — and
    # only words rare enough in the corpus to actually identify this company.
    #
    # When two rare words survive, they must BOTH appear: a single word from a
    # multi-word name is inherently ambiguous. "aditya" alone pulled in 39 mentions,
    # nearly all Aditya BIRLA group companies, for a query about Aditya INFOTECH.
    # Requiring "aditya" AND "infotech" together keeps the real hits and drops the rest.
    rare = _rare_keywords(name, cm)
    name_word_all = rare if len(rare) >= 2 else []
    if not name_word_all:
        for w in rare:
            keys.append(("name_word", w))

    doc_hits: dict[str, str] = {}          # research_n -> how it matched
    mention_rows = []

    # Conjunctive name-word pass: every rare word must be present in the same field.
    if name_word_all and not cm.empty and "company_name_raw" in cm.columns:
        blob = cm["company_name_raw"].astype(str).str.lower()
        m = pd.Series(True, index=cm.index)
        for w in name_word_all:
            m &= blob.str.contains(w, regex=False, na=False)
        for _, r in cm[m].iterrows():
            mention_rows.append({**r.to_dict(), "_matched_by": "name_words_all"})
            doc_hits.setdefault(str(r.get("research_n", "")), "name_words_all")
    if name_word_all and not ri.empty and "companies" in ri.columns:
        blob = ri["companies"].astype(str).str.lower()
        m = pd.Series(True, index=ri.index)
        for w in name_word_all:
            m &= blob.str.contains(w, regex=False, na=False)
        for _, r in ri[m].iterrows():
            doc_hits.setdefault(str(r.get("research_n", "")), "name_words_all")

    for how, needle in keys:
        if not needle:
            continue
        if not cm.empty:
            cols = ["isin", "company_name_raw", "company_name_slug"]
            m = pd.Series(False, index=cm.index)
            for c in cols:
                if c in cm.columns:
                    m |= _contains(cm[c], needle)
            hit = cm[m]
            for _, r in hit.iterrows():
                mention_rows.append({**r.to_dict(), "_matched_by": how})
                rn = str(r.get("research_n", ""))
                doc_hits.setdefault(rn, how)
        if not ri.empty:
            m = pd.Series(False, index=ri.index)
            for c in ("companies", "isins", "file_name"):
                if c in ri.columns:
                    m |= _contains(ri[c], needle)
            for _, r in ri[m].iterrows():
                doc_hits.setdefault(str(r.get("research_n", "")), how)

    if mention_rows:
        md = pd.DataFrame(mention_rows).drop_duplicates(
            subset=[c for c in ("research_n", "file_name") if c in mention_rows[0]])
        out["mentions"] = len(md)
        out["matched_by"] = md["_matched_by"].value_counts().to_dict()

    if not ri.empty and doc_hits:
        docs = ri[ri["research_n"].astype(str).isin(doc_hits)].copy()
        docs["matched_by"] = docs["research_n"].astype(str).map(doc_hits)
        if "doc_date" in docs.columns:
            docs = docs.sort_values("doc_date", ascending=False)
        for _, r in docs.head(max_docs).iterrows():
            out["documents"].append({
                "research_n": r.get("research_n"),
                "date": str(r.get("doc_date", ""))[:10],
                "source": str(r.get("source", "")),
                "doc_type": str(r.get("doc_type", "")).lower(),
                "title": str(r.get("file_name", ""))[:150],
                "fy_or_quarter": str(r.get("fy_or_quarter", "")),
                "matched_by": r.get("matched_by"),
                "summary_md": str(r.get("summary_md", "")),
            })
    by_type: dict[str, int] = {}
    for d in out["documents"]:
        by_type[d["doc_type"]] = by_type.get(d["doc_type"], 0) + 1
    out["by_doc_type"] = by_type
    out["by_source"] = {}
    for d in out["documents"]:
        out["by_source"][d["source"]] = out["by_source"].get(d["source"], 0) + 1
    out["company_specific"] = sum(
        n for t, n in by_type.items() if t in _COMPANY_DOC)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--names", nargs="+", required=True)
    ap.add_argument("--alias", nargs="*", default=[],
                    help="extra brand/trade names to search (e.g. 'CP Plus')")
    ap.add_argument("--summary", action="store_true",
                    help="print the first 400 chars of each document summary")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()

    store = Store()
    results = [lookup(store, t, a.alias) for t in a.names]
    if a.json:
        print(json.dumps(results, indent=2, default=str))
        return 0

    for r in results:
        print("\n" + "=" * 74)
        print(f"{r['query']}  ->  {r.get('name')} "
              f"({r.get('symbol')} / {r.get('isin')})"
              + ("   [NOT IN UNIVERSE]" if r.get("unresolved") else ""))
        print("=" * 74)
        if r.get("error"):
            print("  " + r["error"])
            continue
        print(f"  documents found : {len(r['documents'])}")
        print(f"  mentions        : {r['mentions']}")
        print(f"  company-specific notes : {r['company_specific']}")
        if r["matched_by"]:
            print(f"  matched by      : {r['matched_by']}")
        if r["by_doc_type"]:
            print(f"  by doc_type     : {r['by_doc_type']}")
        if r["by_source"]:
            top = sorted(r["by_source"].items(), key=lambda x: -x[1])[:5]
            print(f"  top sources     : {dict(top)}")
        for d in r["documents"][:8]:
            print(f"    [{d['date']}] {d['source'][:32]:<32} {d['doc_type'][:22]:<22} "
                  f"(via {d['matched_by']})")
            if a.summary and d["summary_md"]:
                s = re.sub(r"\s+", " ", d["summary_md"])[:400]
                print(f"        {s}...")
    return 0


if __name__ == "__main__":
    sys.exit(main())
