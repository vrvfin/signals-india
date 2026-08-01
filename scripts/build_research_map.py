r"""
build_research_map.py — one row per (document x company) and per (document x sector),
so a report can ask "what has been written about this company, and about its sector?"
and get an answer keyed on identifiers rather than on whatever name the broker typed.

WHY THIS EXISTS
Workflow A stores broker PDFs in research_index.parquet (one row per DOCUMENT) and
company_mentions.parquet (one row per MENTION). Neither is keyed to the universe:
mentions carry `company_name_raw` as printed, and `sectors` is almost always the
useless literal ["other_sector"] because the classifying prompt was never given a
vocabulary to choose from. So "find research for CPPLUS" fails — the notes say
"CP Plus" or "Aditya Infotech", never the ISIN.

This resolves every mention to the universe (isin / nse_symbol / bse_code /
screener/registered name), normalises 23 raw doc_type values into a small taxonomy,
and assigns a CONTROLLED sector label. Output:

  company_repo/_index/research_map.parquet
    research_n · doc_hash · doc_date · source · file_name
    scope            company | sector | macro | policy | other
    doc_kind         normalised type (note / results / concall / rating / AR / DRHP ...)
    isin · symbol · bse_code · company_name · matched_name_raw · match_key
    sector           one of SECTORS below, or "" when not determinable
    tags             pipe-joined
    summary_md · daily_md_ref

INCREMENTAL BY DEFAULT: documents already mapped are skipped, so this can run after
every research batch. --full rebuilds from scratch.

Usage:
  python scripts/build_research_map.py --dry-run
  python scripts/build_research_map.py                # incremental append
  python scripts/build_research_map.py --full         # one-time rebuild
"""
from __future__ import annotations

import argparse
import io
import os
import re
import sys
from datetime import datetime
from pathlib import Path

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import pandas as pd
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from _extractor_base import find_file, download_bytes, upload_bytes, log
from narrative_factpack import Store, IDX

MAP_FILE = "research_map.parquet"
MAP_COLS = ["research_n", "doc_hash", "doc_date", "source", "file_name", "scope",
            "doc_kind", "isin", "symbol", "bse_code", "company_name",
            "matched_name_raw", "match_key", "sector", "tags", "summary_md",
            "daily_md_ref", "mapped_at"]

# ---------------------------------------------------------------- SECTORS ----
# THE CONTROLLED VOCABULARY. Deliberately identical to the macro_sector values NSE
# publishes in its index lists, which company_classification.csv already uses — a
# research sector must be joinable to a company's sector, and inventing a second
# taxonomy would make that impossible. An LLM classifying a sector report MUST pick
# from this list; anything else is recorded as "" rather than a new label.
SECTORS = [
    "Automobile and Auto Components", "Capital Goods", "Chemicals", "Construction",
    "Construction Materials", "Consumer Durables", "Consumer Services", "Diversified",
    "Fast Moving Consumer Goods", "Financial Services", "Forest Materials",
    "Healthcare", "Information Technology", "Media Entertainment & Publication",
    "Metals & Mining", "Oil Gas & Consumable Fuels", "Power", "Realty", "Services",
    "Telecommunication", "Textiles", "Utilities",
]

# Keyword -> sector. Deterministic first pass so the common cases cost nothing; only
# genuinely ambiguous documents need an LLM.
SECTOR_HINTS: dict[str, str] = {
    "auto": "Automobile and Auto Components", "automobile": "Automobile and Auto Components",
    "vehicle": "Automobile and Auto Components", "two-wheeler": "Automobile and Auto Components",
    "tyre": "Automobile and Auto Components", "ev ": "Automobile and Auto Components",
    "capital goods": "Capital Goods", "engineering": "Capital Goods",
    "machinery": "Capital Goods", "defence": "Capital Goods",
    "chemical": "Chemicals", "specialty chem": "Chemicals", "agrochem": "Chemicals",
    "cement": "Construction Materials", "construction material": "Construction Materials",
    "infrastructure": "Construction", "epc": "Construction",
    "consumer durable": "Consumer Durables", "appliance": "Consumer Durables",
    "retail": "Consumer Services", "hotel": "Consumer Services",
    "quick commerce": "Consumer Services", "qsr": "Consumer Services",
    "fmcg": "Fast Moving Consumer Goods", "staples": "Fast Moving Consumer Goods",
    "bank": "Financial Services", "nbfc": "Financial Services",
    "insurance": "Financial Services", "financial": "Financial Services",
    "amc": "Financial Services", "lending": "Financial Services",
    "pharma": "Healthcare", "healthcare": "Healthcare", "hospital": "Healthcare",
    "diagnostic": "Healthcare", "api": "Healthcare", "cdmo": "Healthcare",
    "it services": "Information Technology", "software": "Information Technology",
    "technology": "Information Technology", "digital": "Information Technology",
    "media": "Media Entertainment & Publication", "entertainment": "Media Entertainment & Publication",
    "metal": "Metals & Mining", "steel": "Metals & Mining", "mining": "Metals & Mining",
    "aluminium": "Metals & Mining",
    "oil": "Oil Gas & Consumable Fuels", "gas": "Oil Gas & Consumable Fuels",
    "refining": "Oil Gas & Consumable Fuels", "coal": "Oil Gas & Consumable Fuels",
    "power": "Power", "renewable": "Power", "solar": "Power",
    "real estate": "Realty", "realty": "Realty", "housing": "Realty",
    "logistics": "Services", "shipping": "Services", "port": "Services",
    "telecom": "Telecommunication", "tower": "Telecommunication",
    "textile": "Textiles", "apparel": "Textiles", "garment": "Textiles",
    "utility": "Utilities", "water": "Utilities",
}

# 23 raw doc_type values collapse into these. `scope` decides which report section a
# document can feed; `doc_kind` preserves what it actually is.
DOC_KIND: dict[str, tuple[str, str]] = {          # raw -> (scope, doc_kind)
    "single_company_note": ("company", "analyst_note"),
    "analyst_note": ("company", "analyst_note"),
    "company_update": ("company", "company_update"),
    "single_company_ar": ("company", "annual_report"),
    "single_company_drhp": ("company", "drhp"),
    "single_company_rating": ("company", "rating"),
    "single_company_policy": ("company", "policy_impact"),
    "concall": ("company", "concall"),
    "results": ("company", "results"),
    "results_filing": ("company", "results"),
    "investor_presentation": ("company", "presentation"),
    "presentation": ("company", "presentation"),
    "corporate_announcement": ("company", "announcement"),
    "regulatory_filing": ("company", "regulatory_filing"),
    "multi_company_sector": ("sector", "sector_note"),
    "sector_report": ("sector", "sector_note"),
    "multi_company_seminar": ("sector", "seminar"),
    "macro_report": ("macro", "macro_note"),
    "gov_policy": ("policy", "govt_policy"),
    "govt_policy": ("policy", "govt_policy"),
    "govt_policy_other": ("policy", "govt_policy"),
    "government_and_regulatory_other": ("policy", "govt_policy"),
    "other": ("other", "other"),
}
# Documents whose scope is "company" but which mention many names are really sector
# notes; this many distinct companies flips the scope.
SECTOR_MENTION_THRESHOLD = 6


def _norm_type(raw: str) -> tuple[str, str]:
    return DOC_KIND.get(str(raw).strip().lower(), ("other", "other"))


def guess_sector(*texts: str) -> str:
    """Controlled-vocabulary sector from free text. Longest keyword wins, so
    'construction material' beats 'construction'. Returns '' rather than guessing.

    NEVER pass a file name, the `source` field, or raw summary_md here. Those carry the
    BROKER's name and its boilerplate, not the subject: matching on them labelled
    "JM Financial sees 31% UPSIDE in BlueStone" as Financial Services, and
    "SolarManufacturing_Thematic" likewise, because the publisher is a financial firm.
    Use it only on fields describing what the document is ABOUT.
    """
    blob = " ".join(str(t) for t in texts).lower()
    best, best_len = "", 0
    for kw, sec in SECTOR_HINTS.items():
        if kw in blob and len(kw) > best_len:
            best, best_len = sec, len(kw)
    return best


def sector_from_companies(isins: list[str], cls_by_isin: dict[str, str]) -> str:
    """The strongest available signal for a sector note: the modal macro_sector of the
    companies it actually names. A note covering eight Healthcare companies is a
    Healthcare note regardless of who published it or what the file is called.
    Requires a real majority, so a scattergun list yields "" rather than a coin toss.
    """
    secs = [cls_by_isin.get(i, "") for i in isins if i]
    secs = [s for s in secs if s]
    if not secs:
        return ""
    top, n = max(((s, secs.count(s)) for s in set(secs)), key=lambda x: x[1])
    return top if n >= max(2, len(secs) * 0.5) else ""


# ------------------------------------------------------------- resolution ----
class Resolver:
    """company_name_raw -> (isin, symbol, bse_code, registered name, match_key)."""

    def __init__(self, store: Store):
        fid = find_file(store.drive, store.folder(IDX), "company_universe.csv")
        self.uni = (pd.read_csv(io.BytesIO(download_bytes(store.drive, fid))).fillna("")
                    if fid else pd.DataFrame())
        self.by_isin, self.by_sym, self.by_name = {}, {}, {}
        for _, r in self.uni.iterrows():
            isin = str(r.get("isin", "")).strip().upper()
            rec = (isin, str(r.get("nse_symbol", "")).strip().upper()
                   or str(r.get("bse_symbol", "")).strip().upper(),
                   str(r.get("bse_code", "")).strip(),
                   str(r.get("name", "")).strip())
            if isin:
                self.by_isin[isin] = rec
            for s in (r.get("nse_symbol"), r.get("bse_symbol")):
                s = str(s or "").strip().upper()
                if s and s != "NAN":
                    self.by_sym[s] = rec
            nm = self._key(r.get("name"))
            if nm:
                self.by_name.setdefault(nm, rec)

    @staticmethod
    def _key(name) -> str:
        s = re.sub(r"[^a-z0-9 ]+", " ", str(name).lower())
        drop = {"limited", "ltd", "private", "pvt", "the", "company", "co", "india"}
        return " ".join(w for w in s.split() if w and w not in drop)

    def resolve(self, raw_name: str, isin_hint: str = "") -> tuple:
        h = str(isin_hint or "").strip().upper()
        if h and h in self.by_isin:
            return (*self.by_isin[h], "isin")
        raw = str(raw_name or "").strip()
        if not raw:
            return ("", "", "", "", "")
        up = raw.upper()
        if up in self.by_sym:
            return (*self.by_sym[up], "symbol")
        k = self._key(raw)
        if k in self.by_name:
            return (*self.by_name[k], "name")
        # containment both ways — "CP Plus" vs "Aditya Infotech", "TCS" vs "Tata
        # Consultancy Services". Only accept when the shorter key is >= 4 chars, or
        # two-letter fragments match half the universe.
        if len(k) >= 4:
            for nm, rec in self.by_name.items():
                if k in nm or (len(nm) >= 4 and nm in k):
                    return (*rec, "name_partial")
        return ("", "", "", "", "")


# ------------------------------------------------------------------ build ----
def build(store: Store, full: bool = False, limit: int = 0) -> pd.DataFrame:
    ri = store.parquet(IDX, "research_index.parquet")
    cm = store.parquet(IDX, "company_mentions.parquet")
    if ri.empty:
        log("research_index.parquet is empty — nothing to map.")
        return pd.DataFrame(columns=MAP_COLS)

    existing = pd.DataFrame(columns=MAP_COLS)
    if not full:
        fid = find_file(store.drive, store.folder(IDX), MAP_FILE)
        if fid:
            try:
                existing = pd.read_parquet(io.BytesIO(download_bytes(store.drive, fid)))
            except Exception as e:
                log(f"  could not read existing map ({str(e)[:70]}) — rebuilding")
    done = set(existing["research_n"].astype(str)) if not existing.empty else set()
    todo = ri[~ri["research_n"].astype(str).isin(done)]
    if limit:
        todo = todo.head(limit)
    log(f"documents: {len(ri):,} total · {len(done):,} already mapped · "
        f"{len(todo):,} to map")
    if todo.empty:
        return existing

    res = Resolver(store)
    # isin -> macro_sector, so a document's sector can be inferred from the companies
    # it names rather than from its file name.
    cls_by_isin: dict[str, str] = {}
    cfid = find_file(store.drive, store.folder(IDX), "company_classification.csv")
    if cfid:
        try:
            cls = pd.read_csv(io.BytesIO(download_bytes(store.drive, cfid))).fillna("")
            cls_by_isin = {str(r["isin"]).strip().upper():
                           str(r.get("macro_sector", "")).strip()
                           for _, r in cls.iterrows()}
        except Exception as e:
            log(f"  classification unreadable ({str(e)[:60]}) — sector inference weaker")

    stamp = datetime.now().isoformat(timespec="seconds")
    rows, unresolved = [], 0

    for _, d in todo.iterrows():
        rn = str(d.get("research_n", ""))
        scope, kind = _norm_type(d.get("doc_type"))
        base = {"research_n": rn, "doc_hash": str(d.get("doc_hash", "")),
                "doc_date": str(d.get("doc_date", ""))[:10],
                "source": str(d.get("source", "")),
                "file_name": str(d.get("file_name", ""))[:200],
                "doc_kind": kind,
                "summary_md": str(d.get("summary_md", ""))[:20000],
                "daily_md_ref": str(d.get("daily_md_ref", "")),
                "tags": "|".join(str(d.get(c, "")) for c in ("themes", "policies")
                                 if str(d.get(c, "")).strip() not in ("", "[]")),
                "mapped_at": stamp}

        mentions = (cm[cm["research_n"].astype(str) == rn]
                    if not cm.empty and "research_n" in cm.columns
                    else pd.DataFrame())
        # A "company" document naming many firms is really a sector note.
        if scope == "company" and len(mentions) >= SECTOR_MENTION_THRESHOLD:
            scope = "sector"

        # Resolve the mentioned companies first — they are the best sector evidence.
        resolved = []
        if not mentions.empty:
            for _, m in mentions.iterrows():
                resolved.append((m, res.resolve(m.get("company_name_raw"),
                                                m.get("isin"))))
        mention_isins = [r[1][0] for r in resolved if r[1][0]]

        # Priority: the companies named > the document's own sectors field > keywords
        # on the subject fields. File name and source are deliberately excluded.
        sector = (sector_from_companies(mention_isins, cls_by_isin)
                  or guess_sector(d.get("sectors"))
                  or guess_sector(d.get("companies")))

        if scope == "company" and resolved:
            for m, (isin, sym, bse, nm, how) in resolved:
                if not isin:
                    unresolved += 1
                rows.append({**base, "scope": "company",
                             "isin": isin, "symbol": sym, "bse_code": bse,
                             "company_name": nm,
                             "matched_name_raw": str(m.get("company_name_raw", ""))[:150],
                             "match_key": how, "sector": sector})
        else:
            # sector / macro / policy / other — one row per document, no company key
            rows.append({**base, "scope": scope, "isin": "", "symbol": "",
                         "bse_code": "", "company_name": "", "matched_name_raw": "",
                         "match_key": "", "sector": sector})

    new = pd.DataFrame(rows)[MAP_COLS] if rows else pd.DataFrame(columns=MAP_COLS)
    log(f"  {len(new):,} row(s) built; {unresolved:,} company mention(s) "
        f"unresolved to the universe")
    out = (pd.concat([existing, new], ignore_index=True)
           if not existing.empty else new)
    return out.drop_duplicates(subset=["research_n", "isin", "matched_name_raw",
                                       "scope"], keep="last")


def save(store: Store, df: pd.DataFrame) -> str:
    folder = store.folder(IDX)
    fid = find_file(store.drive, folder, MAP_FILE)
    buf = io.BytesIO()
    df.to_parquet(buf, index=False)
    upload_bytes(store.drive, folder, MAP_FILE, buf.getvalue(),
                 "application/octet-stream", existing_id=fid)
    return f"wrote {MAP_FILE}: {len(df):,} rows"


def report(df: pd.DataFrame):
    if df.empty:
        print("no rows")
        return
    print(f"\nrows                 : {len(df):,}")
    print(f"documents            : {df['research_n'].nunique():,}")
    print("\nby scope:")
    print(df["scope"].value_counts().to_string())
    print("\nby doc_kind:")
    print(df["doc_kind"].value_counts().head(10).to_string())
    comp = df[df["scope"] == "company"]
    if not comp.empty:
        res = (comp["isin"].astype(str).str.len() > 0).sum()
        print(f"\ncompany rows         : {len(comp):,}")
        print(f"  resolved to universe: {res:,} ({100.0 * res / len(comp):.0f}%)")
        print("  match_key:", comp[comp["isin"] != ""]["match_key"]
              .value_counts().to_dict())
        print(f"  distinct companies  : {comp[comp['isin'] != '']['isin'].nunique():,}")
    sec = df[df["sector"].astype(str).str.len() > 0]
    print(f"\nrows with a controlled sector: {len(sec):,} of {len(df):,}")
    if not sec.empty:
        print(sec["sector"].value_counts().head(10).to_string())


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--full", action="store_true",
                    help="rebuild every document instead of appending new ones")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    store = Store()
    df = build(store, full=a.full, limit=a.limit)
    report(df)
    if a.dry_run:
        print("\nDRY RUN — nothing written.")
        return 0
    print("\n" + save(store, df))
    return 0


if __name__ == "__main__":
    sys.exit(main())
