r"""
validate_peer_groups.py — is the peer taxonomy fit to compare companies against?

A peer set is only useful if the labels COLLAPSE. Gemini assigns peer_group per batch,
so the same industry can arrive under three near-identical names — observed live in
peer_aggregates on 2026-07-30:

    "API & Formulations"                n=10
    "API & Formulations Manufacturers"  n=2
    "API / Formulation Manufacturers"   n=1

That is one peer group of 13 fragmented into three, and it silently corrupts every
median computed from it: a "peer median P/E" over n=1 is that company's own P/E.

This script does NOT edit anything. It reports, so a merge can be decided deliberately
and the `locked` flag respected.

Checks
  COVERAGE    companies with no peer_group at all
  SINGLETON   groups of one — a median over n=1 is meaningless
  FRAGMENT    near-duplicate labels that should probably be one group
  MIXED       one peer_group spanning several macro_sectors — a mislabel
  THIN        groups below --min-n, where a median is unstable

Usage:
  python scripts/validate_peer_groups.py
  python scripts/validate_peer_groups.py --min-n 5 --json
"""
from __future__ import annotations

import argparse
import io
import json
import os
import re
import sys
from collections import defaultdict
from difflib import SequenceMatcher
from pathlib import Path

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import pandas as pd
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from _extractor_base import find_file, download_bytes, log
from narrative_factpack import Store, IDX

# Labels this similar are treated as candidates for the same group. 0.82 was chosen
# because it catches the API/Formulations family above while keeping genuinely distinct
# neighbours ("Housing Finance" vs "Vehicle Finance", 0.62) apart.
SIMILARITY = 0.82
STOPWORDS = {"ltd", "limited", "manufacturers", "manufacturing", "companies",
             "company", "and", "&", "/", "-", "the", "services", "products"}


def _norm(label: str) -> str:
    """Compare labels on their meaningful words only, so 'API & Formulations' and
    'API / Formulation Manufacturers' land on the same normalised key."""
    s = re.sub(r"[^a-z0-9 ]+", " ", str(label).lower())
    words = [w.rstrip("s") for w in s.split() if w and w not in STOPWORDS]
    return " ".join(sorted(words))


def load(store: Store) -> pd.DataFrame:
    fid = find_file(store.drive, store.folder(IDX), "company_classification.csv")
    if not fid:
        return pd.DataFrame()
    return pd.read_csv(io.BytesIO(download_bytes(store.drive, fid))).fillna("")


def validate(df: pd.DataFrame, min_n: int = 3) -> dict:
    out: dict = {"total": len(df), "findings": defaultdict(list)}
    if df.empty:
        return out
    pg = df["peer_group"].astype(str).str.strip()

    # COVERAGE ------------------------------------------------------------------
    missing = df[pg == ""]
    out["with_group"] = int((pg != "").sum())
    out["without_group"] = int(len(missing))
    if len(missing):
        by_board = (missing["symbol"].astype(str).head(8).tolist())
        out["findings"]["COVERAGE"].append({
            "detail": f"{len(missing):,} of {len(df):,} companies have no peer_group",
            "examples": by_board,
            "fix": "python scripts/build_classification.py --with-gemini --limit N"})

    grouped = df[pg != ""].copy()
    grouped["peer_group"] = pg[pg != ""]
    sizes = grouped.groupby("peer_group").size().sort_values()

    # SINGLETON / THIN ----------------------------------------------------------
    singles = sizes[sizes == 1]
    thin = sizes[(sizes > 1) & (sizes < min_n)]
    if len(singles):
        out["findings"]["SINGLETON"].append({
            "detail": f"{len(singles):,} group(s) contain exactly ONE company — any "
                      f"median over them is that company's own value",
            "examples": list(singles.index[:8])})
    if len(thin):
        out["findings"]["THIN"].append({
            "detail": f"{len(thin):,} group(s) have 2-{min_n - 1} members — medians "
                      f"are unstable",
            "examples": list(thin.index[:8])})

    # FRAGMENT ------------------------------------------------------------------
    labels = list(sizes.index)
    norm_map: dict[str, list[str]] = defaultdict(list)
    for lab in labels:
        norm_map[_norm(lab)].append(lab)
    frags = [v for v in norm_map.values() if len(v) > 1]
    # also catch near-misses that normalisation alone does not merge
    seen = {l for group in frags for l in group}
    for i, a in enumerate(labels):
        if a in seen:
            continue
        for b in labels[i + 1:]:
            if b in seen:
                continue
            if SequenceMatcher(None, _norm(a), _norm(b)).ratio() >= SIMILARITY:
                frags.append([a, b])
                seen.update((a, b))
                break
    for group in frags:
        members = {g: int(sizes[g]) for g in group}
        out["findings"]["FRAGMENT"].append({
            "detail": f"{len(group)} labels look like one group "
                      f"({sum(members.values())} companies split apart)",
            "labels": members,
            "fix": "pick one label, edit company_classification.csv, set locked=1"})

    # MIXED ---------------------------------------------------------------------
    if "macro_sector" in grouped.columns:
        for name, g in grouped.groupby("peer_group"):
            secs = {s for s in g["macro_sector"].astype(str).str.strip() if s}
            if len(secs) > 2 and len(g) >= min_n:
                out["findings"]["MIXED"].append({
                    "detail": f"peer_group '{name}' (n={len(g)}) spans "
                              f"{len(secs)} macro-sectors — likely a mislabel",
                    "sectors": sorted(secs)[:5]})

    out["groups"] = int(len(sizes))
    out["median_group_size"] = float(sizes.median()) if len(sizes) else 0.0
    out["findings"] = dict(out["findings"])
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--min-n", type=int, default=3,
                    help="below this a group's median is called unstable (default 3)")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()

    store = Store()
    df = load(store)
    if df.empty:
        print("company_classification.csv not found or empty.")
        return 1
    res = validate(df, a.min_n)

    if a.json:
        print(json.dumps(res, indent=2, default=str))
        return 0

    print(f"\ncompanies              : {res['total']:,}")
    print(f"  with a peer_group    : {res['with_group']:,}")
    print(f"  without              : {res['without_group']:,}")
    print(f"peer groups            : {res.get('groups', 0):,}")
    print(f"median group size      : {res.get('median_group_size', 0):.0f}")
    order = ["COVERAGE", "FRAGMENT", "MIXED", "SINGLETON", "THIN"]
    for kind in order:
        items = res["findings"].get(kind, [])
        if not items:
            continue
        print(f"\n[{kind}]  {len(items)} finding(s)")
        for it in items[:12]:
            print(f"  - {it['detail']}")
            if "labels" in it:
                for lab, n in it["labels"].items():
                    print(f"      {n:>4}x  {lab}")
            elif it.get("examples"):
                print(f"      e.g. {', '.join(map(str, it['examples'][:6]))}")
            if it.get("fix"):
                print(f"      fix: {it['fix']}")
        if len(items) > 12:
            print(f"  ... and {len(items) - 12} more")
    worst = sum(len(v) for v in res["findings"].values())
    print(f"\n{worst} finding(s) total. Nothing was modified — peer_group edits are "
          f"manual, and `locked=1` protects them from the next Gemini pass.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
