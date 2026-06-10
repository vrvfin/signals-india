"""
Phase 3 — T4.2 Fraud risk (FINANCIAL forensic score).

Deterministic, rule-based accounting-quality flags computed from T2's
`financials_derived.parquet`. Writes:

  company_repo/_index/fraud_risk.parquet   — one row per company
  company_repo/_index/fraud_risk.csv

`fraud_risk_score` is 0-100 where HIGHER = WORSE. The scorecard consumes it as
`score_fraud_risk = 100 - fraud_risk_score` (an inverse penalty factor).

Scope note (per user 2026-06-09): this is the *financial* fraud signal only.
A separate, investigation-based "fraud grade" for the whole universe (Google /
specialist-site search → letter grade) is a DISTINCT deliverable, proposed
separately — not built here.

Rule adaptations vs the original 5-rule spec (T2 schema constraints, flagged):
  R1 accrual      : CFO<PAT is ANNUAL-only in T2 -> cfo_pat_ratio<1 for >=2
                    consecutive latest annual years.  (quarterly CFO unavailable)
  R2 receivables  : no receivables balance line in T2 -> receivable_days latest
                    > 1.5x its prior trailing average.
  R3 other-income : NOT COMPUTABLE (no Other Income / EBIT line in T2) -> skipped.
  R4 leverage     : net_debt_ebitda > 4x  (note: T2's metric is a GROSS-debt proxy).
  R5 wc blowup    : wc_days latest > 1.5x its prior trailing average.
  bonus           : interest_coverage < 1.5 ; roce_pct declining over 3y.
TODO (coordinate with T2 owner): add Other Income + EBIT + quarterly cashflow to
backfill_results_3stmt.py to support full R1 (quarterly) and R3.

Optional enrichment (off by default, gated to forensically-flagged names only):
  --with-news    reliable-source news scan via social_signals (keyword match)
  --with-gemini  lite accounting-quality verdict via gemini_pool.BucketPool over
                 the dedicated FRAUD_API_KEY_<n> pool, falling back to
                 GEMINI_API_KEY then BACKFILL_GEMINI_KEY when exhausted.

Usage:
    python scripts/build_fraud_risk.py --local --dry-run         # offline
    python scripts/build_fraud_risk.py --names "TCS"             # real Drive
    python scripts/build_fraud_risk.py --with-news --with-gemini # full (flagged names)
"""

from __future__ import annotations

import argparse
import io
import os
from datetime import datetime
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

# Shared Drive layer (CLAUDE.md global rule #4 — reuse, never raw API calls).
from _extractor_base import (
    get_drive, get_or_create_subfolder, find_file, download_bytes, upload_bytes,
)

DATA_MISSING = "DATA_MISSING"

FRAUD_COLS = [
    "isin", "symbol", "company_name", "fraud_risk_score",
    "n_forensic_flags", "forensic_flags", "news_flags",
    "gemini_verdict", "confidence", "sources", "computed_at",
]

# Forensic rule weights (sum of fired-flag weights, capped at 100).
FLAG_WEIGHTS = {
    "cfo_below_pat_2y":   25,   # R1 (annual adaptation)
    "receivables_rising": 15,   # R2
    "high_leverage":      20,   # R4
    "wc_blowup":          15,   # R5
    "weak_interest_cover":15,   # bonus
    "roce_declining":     10,   # bonus
}
NEWS_PENALTY = 15               # added if reliable-source fraud news found
GEMINI_HIGH_PENALTY = 20        # added if Gemini judges accounting risk HIGH
GEMINI_MED_PENALTY = 10

FRAUD_NEWS_KEYWORDS = [
    "sebi probe", "sebi order", "forensic audit", "auditor resign",
    "auditor resignation", "fraud", "pledge", "pledged shares",
    "sfio", "enforcement directorate", "default", "insolvency",
]


def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")


# ------------------------------------------------------------------ #
#  Storage abstraction: Drive (default) or local mirror (--local)     #
#  Mirrors build_valuation.py — Drive mode delegates to the shared    #
#  _extractor_base helpers; local mode reads/writes a mirror dir.     #
# ------------------------------------------------------------------ #

class Store:
    def __init__(self, local: bool, local_dir: Path | None):
        self.local = local
        self.local_dir = local_dir
        self.drive = None
        if not local:
            self.drive = get_drive()
            self.root = os.environ["GDRIVE_FOLDER_ID"]

    def _folder(self, parts):
        fid = self.root
        for p in parts:
            fid = get_or_create_subfolder(self.drive, fid, p)
        return fid

    def read_parquet(self, path_parts):
        *folder, name = path_parts
        if self.local:
            fp = self.local_dir.joinpath(*path_parts)
            return pd.read_parquet(fp) if fp.exists() else None
        fid = find_file(self.drive, self._folder(folder), name)
        if not fid:
            return None
        return pd.read_parquet(io.BytesIO(download_bytes(self.drive, fid)))

    def read_csv(self, path_parts):
        *folder, name = path_parts
        if self.local:
            fp = self.local_dir.joinpath(*path_parts)
            return pd.read_csv(fp) if fp.exists() else None
        fid = find_file(self.drive, self._folder(folder), name)
        if not fid:
            return None
        return pd.read_csv(io.BytesIO(download_bytes(self.drive, fid)))

    def read_text(self, path_parts):
        *folder, name = path_parts
        if self.local:
            fp = self.local_dir.joinpath(*path_parts)
            return fp.read_text(encoding="utf-8") if fp.exists() else None
        fid = find_file(self.drive, self._folder(folder), name)
        if not fid:
            return None
        return download_bytes(self.drive, fid).decode("utf-8", errors="replace")

    def write_df(self, path_parts, df: pd.DataFrame):
        *folder, name = path_parts
        if self.local:
            fp = self.local_dir.joinpath(*path_parts)
            fp.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(fp, index=False) if name.endswith(".csv") else df.to_parquet(fp, index=False)
            return
        if name.endswith(".csv"):
            data = df.to_csv(index=False).encode("utf-8")
            mime = "text/csv"
        else:
            buf = io.BytesIO()
            df.to_parquet(buf, index=False)
            data = buf.getvalue()
            mime = "application/octet-stream"
        folder_id = self._folder(folder)
        existing = find_file(self.drive, folder_id, name)
        upload_bytes(self.drive, folder_id, name, data, mime, existing_id=existing)


# ------------------------------------------------------------------ #
#  Forensic rules (annual financials_derived)                         #
# ------------------------------------------------------------------ #

def _annual_series(dfm: pd.DataFrame, metric: str) -> list[float]:
    """Chronological list of annual values for one derived metric."""
    sub = dfm[(dfm["metric"] == metric) & (dfm["period_type"] == "annual")].copy()
    if sub.empty:
        return []
    sub["pdate"] = pd.to_datetime(sub["period"], format="%b %Y", errors="coerce")
    sub = sub.dropna(subset=["pdate"]).sort_values("pdate")
    sub["value"] = pd.to_numeric(sub["value"], errors="coerce")
    return [v for v in sub["value"].tolist() if pd.notna(v)]


def forensic_flags(dfm: pd.DataFrame) -> list[str]:
    """Return the list of fired forensic-flag names for one company's derived rows."""
    flags = []

    # R1 — accrual quality: CFO < PAT (cfo_pat_ratio < 1) for the 2 latest years
    cfo_pat = _annual_series(dfm, "cfo_pat_ratio")
    if len(cfo_pat) >= 2 and cfo_pat[-1] < 1 and cfo_pat[-2] < 1:
        flags.append("cfo_below_pat_2y")

    # R2 — receivable days rising sharply vs prior trailing average
    rd = _annual_series(dfm, "receivable_days")
    if len(rd) >= 2:
        prior = rd[:-1][-3:]                      # up to 3 prior years
        if prior and rd[-1] > 1.5 * (sum(prior) / len(prior)):
            flags.append("receivables_rising")

    # R4 — leverage: gross-debt/EBITDA proxy > 4x (latest)
    nde = _annual_series(dfm, "net_debt_ebitda")
    if nde and nde[-1] is not None and nde[-1] > 4:
        flags.append("high_leverage")

    # R5 — working-capital blowup vs prior trailing average
    wc = _annual_series(dfm, "wc_days")
    if len(wc) >= 2:
        prior = wc[:-1][-3:]
        if prior and wc[-1] > 1.5 * (sum(prior) / len(prior)):
            flags.append("wc_blowup")

    # bonus — weak interest coverage (latest < 1.5x)
    ic = _annual_series(dfm, "interest_coverage")
    if ic and ic[-1] is not None and ic[-1] < 1.5:
        flags.append("weak_interest_cover")

    # bonus — ROCE declining across the last 3 annual points
    roce = _annual_series(dfm, "roce_pct")
    if len(roce) >= 3 and roce[-1] < roce[-2] < roce[-3]:
        flags.append("roce_declining")

    return flags


def score_from_flags(flags: list[str]) -> int:
    return min(100, sum(FLAG_WEIGHTS.get(f, 0) for f in flags))


# ------------------------------------------------------------------ #
#  Optional enrichment (gated to flagged names)                       #
# ------------------------------------------------------------------ #

def scan_news(isin, symbol, company_name) -> tuple[str, str]:
    """Reliable-source fraud-keyword news scan. Returns (news_flags, sources)."""
    try:
        import social_signals
        res = social_signals.fetch_signals(
            [{"isin": isin, "symbol": symbol, "name": company_name}], hours_back=720)
        cs = res.get(isin) if isinstance(res, dict) else None
        if not cs or not getattr(cs, "news", None):
            return "", ""
        hits, srcs = set(), set()
        for n in cs.news:
            text = f"{getattr(n,'title','')} {getattr(n,'snippet','')}".lower()
            for kw in FRAUD_NEWS_KEYWORDS:
                if kw in text:
                    hits.add(kw)
                    srcs.add(getattr(n, "source", "") or "")
        return "; ".join(sorted(hits)), "; ".join(sorted(s for s in srcs if s))
    except Exception as e:
        log(f"  news scan failed for {symbol}: {str(e)[:80]}")
        return "", ""


def _build_gemini_pool():
    """One BucketPool over the dedicated FRAUD_API_KEY pool first, then
    GEMINI_API_KEY, then BACKFILL_GEMINI_KEY (rule-based cascade: when one
    pool's keys exhaust, BucketPool moves to the next). Lite models only.
    Returns None if no keys / library unavailable."""
    try:
        from gemini_pool import BucketPool, load_keys
        keys = load_keys(os.environ, prefix="FRAUD_API_KEY")
        for prefix in ("GEMINI_API_KEY", "BACKFILL_GEMINI_KEY"):
            keys += [k for k in load_keys(os.environ, prefix=prefix)
                     if k not in keys]
        if not keys:
            log("  --with-gemini: no FRAUD/GEMINI keys found — skipping Gemini pass.")
            return None
        return BucketPool(keys, ["gemini-2.5-flash-lite", "gemini-2.0-flash-lite"],
                          inter_call_s=6.0, logger=log)
    except Exception as e:
        log(f"  Gemini pool init failed: {str(e)[:80]}")
        return None


def gemini_verdict(pool, store: Store, isin, symbol, company_name,
                   flags: list[str]) -> tuple[str, str]:
    """Lite accounting-quality verdict from company_page.md. Returns (verdict, confidence)."""
    if pool is None:
        return "", ""
    page = store.read_text(["company_repo", isin, "company_page.md"]) or ""
    if not page.strip():
        return "", ""
    prompt = (
        "You are a forensic-accounting reviewer for an Indian listed company. "
        "Based ONLY on the company brief below, judge accounting/governance risk. "
        f"Rule-based flags already fired: {', '.join(flags) or 'none'}.\n"
        "Reply in exactly one line: 'RISK=<LOW|MEDIUM|HIGH> | <=20 word reason'.\n\n"
        f"--- COMPANY BRIEF ---\n{page[:12000]}"
    )
    try:
        text, _ = pool.call_text(prompt)
        line = text.strip().splitlines()[0][:200]
        verdict = ("HIGH" if "RISK=HIGH" in line.upper()
                   else "MEDIUM" if "RISK=MEDIUM" in line.upper()
                   else "LOW" if "RISK=LOW" in line.upper() else "")
        return verdict, line
    except Exception as e:
        log(f"  Gemini verdict failed for {symbol}: {str(e)[:80]}")
        return "", ""


# ------------------------------------------------------------------ #
#  Main                                                                #
# ------------------------------------------------------------------ #

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--names", type=str, default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--local", action="store_true")
    ap.add_argument("--local-dir", type=str, default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--with-news", action="store_true",
                    help="Reliable-source news scan for forensically-flagged names.")
    ap.add_argument("--with-gemini", action="store_true",
                    help="Lite Gemini accounting-quality verdict for flagged names.")
    args = ap.parse_args()

    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
    local_dir = Path(args.local_dir) if args.local_dir else \
        Path(__file__).resolve().parent.parent / ".t4_local"
    store = Store(args.local, local_dir)
    log(f"build_fraud_risk — mode={'LOCAL' if args.local else 'DRIVE'} "
        f"{'(dry-run)' if args.dry_run else ''}")

    derived = store.read_parquet(["company_repo", "_index", "financials_derived.parquet"])
    if derived is None or derived.empty:
        log("financials_derived.parquet (T2) absent/empty — every company is "
            "DATA_MISSING for fraud risk (scorecard will drop the factor). Writing nothing new.")
        return
    derived = derived.copy()
    derived["symbol"] = derived["symbol"].astype(str).str.upper()

    if args.names:
        wanted = {s.strip().upper() for s in args.names.split(",") if s.strip()}
        derived = derived[derived["symbol"].isin(wanted)]
    if derived.empty:
        log("No derived rows after scope filter — nothing to do.")
        return

    # name lookup
    name_map = {}
    cu = store.read_csv(["company_repo", "_index", "company_universe.csv"])
    if cu is not None and not cu.empty:
        sym_col = "nse_symbol" if "nse_symbol" in cu.columns else "symbol"
        for _, r in cu.iterrows():
            name_map[str(r.get(sym_col, "")).strip().upper()] = str(r.get("name", "")).strip()

    companies = list(derived.groupby(["isin", "symbol"], sort=False))
    if args.limit:
        companies = companies[:args.limit]

    pool = _build_gemini_pool() if (args.with_gemini and not args.dry_run) else None

    rows = []
    for (isin, symbol), dfm in companies:
        flags = forensic_flags(dfm)
        score = score_from_flags(flags)
        cname = name_map.get(symbol, "")
        news_flags, sources, gv, conf = "", "", "", ""

        if flags and args.with_news and not args.dry_run:
            news_flags, sources = scan_news(isin, symbol, cname)
            if news_flags:
                score = min(100, score + NEWS_PENALTY)

        if flags and args.with_gemini and not args.dry_run:
            gv, conf = gemini_verdict(pool, store, isin, symbol, cname, flags)
            if gv == "HIGH":
                score = min(100, score + GEMINI_HIGH_PENALTY)
            elif gv == "MEDIUM":
                score = min(100, score + GEMINI_MED_PENALTY)

        rows.append({
            "isin": isin, "symbol": symbol, "company_name": cname,
            "fraud_risk_score": score,
            "n_forensic_flags": len(flags),
            "forensic_flags": "; ".join(flags),
            "news_flags": news_flags,
            "gemini_verdict": gv,
            "confidence": conf,
            "sources": sources,
            "computed_at": datetime.now().isoformat(timespec="seconds"),
        })

    out = pd.DataFrame(rows, columns=FRAUD_COLS)
    flagged = out[out["n_forensic_flags"] > 0]
    log(f"Scored {len(out)} companies; {len(flagged)} have >=1 forensic flag "
        f"(max score {out['fraud_risk_score'].max() if not out.empty else 0}).")

    if args.dry_run:
        log("DRY-RUN — not writing. Flagged sample:")
        sample = (flagged if not flagged.empty else out).head(12)
        print(sample.to_string(index=False))
        return

    store.write_df(["company_repo", "_index", "fraud_risk.parquet"], out)
    store.write_df(["company_repo", "_index", "fraud_risk.csv"], out)
    log("Wrote fraud_risk.parquet + fraud_risk.csv to _index/.")


if __name__ == "__main__":
    main()
