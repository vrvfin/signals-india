"""
Phase 3 — T4.2 Company Scorecard (8-factor blend).

Combines the per-company signals into one auditable scorecard and writes:

  company_repo/_index/company_scorecard.parquet   — one row per company
  company_repo/_index/company_scorecard.csv

Eight sub-scores, each 0-100 (or DATA_MISSING when its source isn't available
yet). The composite is a weighted average over ONLY the factors present, with the
weights renormalised — so a company with partial data still gets a fair score and
`data_completeness_pct` tells you how complete it is.

  Factor              Weight  Source
  score_technical       18%   signals/aggregated/latest.csv (n_strategies, composite_score)
  score_fundamental     18%   financials_derived (rev/pat YoY+QoQ, OPM trend) — T2
  score_fin_health    13.5%   financials_derived annual (cfo_pat, coverage, leverage, roce, wc) — T2
  score_mgmt_cred     13.5%   mgmt_credibility.cred_score — T1
  score_valuation        9%   valuation.parquet (T4.1)
  score_guidance         9%   gf4_quality_flags net (+1/-1) — Phase 2
  score_fraud_risk       9%   100 - fraud_risk_score (T4.2 financial forensic)
  score_investigative   10%   (4 - investigative_grade) / 4 * 100 (T4.4 regulatory lists)

Option A wiring (locked 2026-06-10): the investigative grade is a normal weighted
factor — it highlights risk (red badge, low bar) but NEVER caps the composite.

Run AFTER build_valuation.py, build_fraud_risk.py and build_investigative_fraud.py
(reads their parquets if present; degrades gracefully if absent).

Usage:
    python scripts/build_scorecard.py --local --dry-run
    python scripts/build_scorecard.py --names "TCS"
"""

from __future__ import annotations

import argparse
import io
import os
from datetime import datetime
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from _extractor_base import (
    get_drive, get_or_create_subfolder, find_file, download_bytes, upload_bytes,
)

DATA_MISSING = "DATA_MISSING"

SCORECARD_COLS = [
    "isin", "symbol", "company_name", "mcap_segment",
    "score_technical", "score_fundamental", "score_fin_health", "score_mgmt_cred",
    "score_valuation", "score_guidance", "score_fraud_risk", "score_investigative",
    "composite_score", "data_completeness_pct", "computed_at",
]

# 8 factors; investigative gets a flat 10%, the original 7 shrink by x0.9. Sums to 1.0.
WEIGHTS = {
    "score_technical":     0.18,
    "score_fundamental":   0.18,
    "score_fin_health":    0.135,
    "score_mgmt_cred":     0.135,
    "score_valuation":     0.09,
    "score_guidance":      0.09,
    "score_fraud_risk":    0.09,
    "score_investigative": 0.10,
}

# Mirror app.py's GF4 quality keyword sets so the guidance factor matches the dashboard.
_GF4_POSITIVE = frozenset({
    "strong order book", "capacity backed", "high visibility",
    "order book backed", "confirmed orders", "take or pay",
    "long term contract", "pipeline visibility",
})
_GF4_NEGATIVE = frozenset({
    "weak visibility", "guidance ambiguous", "execution risk",
    "volume dependent", "macro dependent", "aspirational",
    "sector headwind", "demand uncertainty",
})


def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")


def _clip(x, lo=0.0, hi=100.0):
    return max(lo, min(hi, x))


# ------------------------------------------------------------------ #
#  Storage abstraction (mirrors build_valuation/build_fraud_risk)     #
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
#  Per-factor scorers (return 0-100 float, or None when no data)      #
# ------------------------------------------------------------------ #

def _latest_annual(dfm, metric):
    sub = dfm[(dfm["metric"] == metric) & (dfm["period_type"] == "annual")].copy()
    if sub.empty:
        return None, []
    sub["pdate"] = pd.to_datetime(sub["period"], format="%b %Y", errors="coerce")
    sub = sub.dropna(subset=["pdate"]).sort_values("pdate")
    vals = [v for v in pd.to_numeric(sub["value"], errors="coerce").tolist() if pd.notna(v)]
    return (vals[-1] if vals else None), vals


def _latest_quarterly(dfm, metric):
    sub = dfm[(dfm["metric"] == metric) & (dfm["period_type"] == "quarterly")].copy()
    if sub.empty:
        return None, []
    sub["pdate"] = pd.to_datetime(sub["period"], format="%b %Y", errors="coerce")
    sub = sub.dropna(subset=["pdate"]).sort_values("pdate")
    vals = [v for v in pd.to_numeric(sub["value"], errors="coerce").tolist() if pd.notna(v)]
    return (vals[-1] if vals else None), vals


def score_technical(sig_rows) -> float | None:
    if sig_rows is None or sig_rows.empty:
        return None
    ba = sig_rows[sig_rows["zone_type"].astype(str).str.lower().isin(["buy", "add"])]
    use = ba if not ba.empty else sig_rows
    n = pd.to_numeric(use.get("n_strategies"), errors="coerce").max()
    comp = pd.to_numeric(use.get("composite_score"), errors="coerce").max()
    parts = []
    if pd.notna(n):
        parts.append(min(n, 5) / 5 * 100)
    if pd.notna(comp):
        parts.append(_clip(float(comp)))
    return round(sum(parts) / len(parts), 1) if parts else None


def _growth_to_score(pct):
    # +33% -> 100, 0% -> 50, -33% -> 0
    return _clip(50 + float(pct) * 1.5) if pct is not None and pd.notna(pct) else None


def score_fundamental(dfm) -> float | None:
    if dfm is None or dfm.empty:
        return None
    comps = []
    for metric in ("rev_yoy_pct", "pat_yoy_pct", "rev_qoq_pct", "pat_qoq_pct"):
        latest, _ = _latest_quarterly(dfm, metric)
        s = _growth_to_score(latest)
        if s is not None:
            comps.append(s)
    # OPM trend: latest vs mean of up to 3 prior quarters
    opm_latest, opm_vals = _latest_quarterly(dfm, "opm_pct")
    if opm_latest is not None and len(opm_vals) >= 2:
        prior = opm_vals[:-1][-3:]
        delta = opm_latest - (sum(prior) / len(prior))
        comps.append(_clip(50 + delta * 5))     # +10pp -> 100
    return round(sum(comps) / len(comps), 1) if comps else None


def score_fin_health(dfm) -> float | None:
    if dfm is None or dfm.empty:
        return None
    comps = []
    cfo_pat, _ = _latest_annual(dfm, "cfo_pat_ratio")
    if cfo_pat is not None:
        comps.append(_clip(cfo_pat * 50))                 # 2x -> 100, 1x -> 50
    ic, _ = _latest_annual(dfm, "interest_coverage")
    if ic is not None:
        comps.append(_clip(ic * 10))                      # 10x -> 100
    nde, _ = _latest_annual(dfm, "net_debt_ebitda")
    if nde is not None:
        comps.append(_clip(100 - nde * 20))               # 0 -> 100, 5x -> 0
    roce, _ = _latest_annual(dfm, "roce_pct")
    if roce is not None:
        comps.append(_clip(roce * 2.5))                   # 40% -> 100, 20% -> 50
    wc, _ = _latest_annual(dfm, "wc_days")
    if wc is not None:
        comps.append(_clip(100 - wc))                     # 0d -> 100, 100d -> 0
    return round(sum(comps) / len(comps), 1) if comps else None


def score_mgmt_cred(mc_rows) -> float | None:
    if mc_rows is None or mc_rows.empty:
        return None
    rows = mc_rows.copy()
    if "processed_at" in rows.columns:
        rows = rows.sort_values("processed_at")
    cred = pd.to_numeric(rows["cred_score"], errors="coerce").dropna()
    if cred.empty:
        return None
    val = float(cred.iloc[-1])
    if val <= 10:           # 0-10 scale -> 0-100
        val *= 10
    return round(_clip(val), 1)


def score_guidance(gf4_rows) -> float | None:
    if gf4_rows is None or gf4_rows.empty or "flag_type" not in gf4_rows.columns:
        return None
    net = 0
    for flag in gf4_rows["flag_type"].astype(str).str.lower():
        if any(p in flag for p in _GF4_POSITIVE):
            net += 1
        elif any(ng in flag for ng in _GF4_NEGATIVE):
            net -= 1
    return round(_clip(50 + net * 15), 1)                 # +/-1 flag -> +/-15


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
    args = ap.parse_args()

    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
    local_dir = Path(args.local_dir) if args.local_dir else \
        Path(__file__).resolve().parent.parent / ".t4_local"
    store = Store(args.local, local_dir)
    log(f"build_scorecard — mode={'LOCAL' if args.local else 'DRIVE'} "
        f"{'(dry-run)' if args.dry_run else ''}")

    # ---- load all sources (any may be None) ----
    sig = store.read_csv(["signals", "aggregated", "latest.csv"])
    derived = store.read_parquet(["company_repo", "_index", "financials_derived.parquet"])
    mgmt = store.read_parquet(["company_repo", "_index", "mgmt_credibility.parquet"])
    val = store.read_parquet(["company_repo", "_index", "valuation.parquet"])
    gf4 = store.read_parquet(["company_repo", "_index", "gf4_quality_flags.parquet"])
    fraud = store.read_parquet(["company_repo", "_index", "fraud_risk.parquet"])
    inv = store.read_parquet(["company_repo", "_index", "investigative_fraud.parquet"])
    cu = store.read_csv(["company_repo", "_index", "company_universe.csv"])
    mc_csv = store.read_csv(["universe", "market_cap.csv"])

    present = [n for n, d in [("signals", sig), ("derived", derived), ("mgmt_cred", mgmt),
                              ("valuation", val), ("gf4", gf4), ("fraud", fraud),
                              ("investigative", inv)]
               if d is not None and not d.empty]
    log(f"sources present: {present or 'NONE'}")
    if not present:
        log("No factor sources available — nothing to score.")
        return

    def _upper(df, col="symbol"):
        if df is not None and not df.empty and col in df.columns:
            df = df.copy()
            df[col] = df[col].astype(str).str.upper()
        return df

    sig, derived, mgmt, val, gf4, fraud, inv = map(
        _upper, (sig, derived, mgmt, val, gf4, fraud, inv))

    # ---- universe = union of symbols across available sources ----
    # NOTE: `inv` (investigative_fraud) is deliberately EXCLUDED from the union.
    # It covers the whole universe, and a company whose ONLY factor is a clean
    # investigative grade would score composite=100 — misleading. Investigative
    # is an overlay: scored only for companies that have >=1 other signal.
    symbols = set()
    for d in (sig, derived, mgmt, val, gf4, fraud):
        if d is not None and not d.empty and "symbol" in d.columns:
            symbols |= set(d["symbol"].dropna().astype(str).str.upper())
    if args.names:
        wanted = {s.strip().upper() for s in args.names.split(",") if s.strip()}
        symbols &= wanted
    symbols = sorted(s for s in symbols if s and s != "NAN")
    if args.limit:
        symbols = symbols[:args.limit]
    if not symbols:
        log("No symbols after scope filter.")
        return

    # ---- lookups: isin, name, mcap_segment ----
    isin_map, name_map = {}, {}
    if cu is not None and not cu.empty:
        sc = "nse_symbol" if "nse_symbol" in cu.columns else "symbol"
        for _, r in cu.iterrows():
            s = str(r.get(sc, "")).strip().upper()
            if s:
                isin_map[s] = str(r.get("isin", "")).strip()
                name_map[s] = str(r.get("name", "")).strip()
    seg_map = {}
    if val is not None and not val.empty and "mcap_segment" in val.columns:
        for _, r in val.iterrows():
            seg_map[str(r["symbol"]).upper()] = r.get("mcap_segment")
    if mc_csv is not None and not mc_csv.empty and "mcap_segment" in mc_csv.columns:
        for _, r in mc_csv.iterrows():
            seg_map.setdefault(str(r["symbol"]).strip().upper(), r.get("mcap_segment"))

    def _coerce_num(v):
        try:
            return float(v)
        except (TypeError, ValueError):
            return None

    rows = []
    for sym in symbols:
        scores = {}
        scores["score_technical"] = score_technical(
            sig[sig["symbol"] == sym] if sig is not None else None)
        dcomp = derived[derived["symbol"] == sym] if derived is not None else None
        scores["score_fundamental"] = score_fundamental(dcomp)
        scores["score_fin_health"] = score_fin_health(dcomp)
        scores["score_mgmt_cred"] = score_mgmt_cred(
            mgmt[mgmt["symbol"] == sym] if mgmt is not None else None)
        # valuation passthrough
        sv = None
        if val is not None and not val.empty:
            vr = val[val["symbol"] == sym]
            if not vr.empty:
                sv = _coerce_num(vr.iloc[0].get("valuation_score"))
        scores["score_valuation"] = round(sv, 1) if sv is not None else None
        scores["score_guidance"] = score_guidance(
            gf4[gf4["symbol"] == sym] if gf4 is not None else None)
        # fraud penalty -> inverse
        sfr = None
        if fraud is not None and not fraud.empty:
            fr = fraud[fraud["symbol"] == sym]
            if not fr.empty:
                frs = _coerce_num(fr.iloc[0].get("fraud_risk_score"))
                if frs is not None:
                    sfr = _clip(100 - frs)
        scores["score_fraud_risk"] = round(sfr, 1) if sfr is not None else None
        # investigative grade (T4.4) -> (4 - grade) / 4 * 100; Option A, no cap
        siv = None
        if inv is not None and not inv.empty:
            ir = inv[inv["symbol"] == sym]
            if not ir.empty:
                grade = _coerce_num(ir.iloc[0].get("investigative_grade"))
                if grade is not None:
                    siv = _clip((4 - grade) / 4 * 100)
        scores["score_investigative"] = round(siv, 1) if siv is not None else None

        # weighted blend over available factors
        num = den = 0.0
        for k, w in WEIGHTS.items():
            if scores[k] is not None:
                num += w * scores[k]
                den += w
        composite = round(num / den, 1) if den > 0 else None
        n_avail = sum(1 for v in scores.values() if v is not None)

        row = {
            "isin": isin_map.get(sym, ""),
            "symbol": sym,
            "company_name": name_map.get(sym, ""),
            "mcap_segment": seg_map.get(sym) or DATA_MISSING,
            # Numeric columns stay numeric (None -> NaN) so the parquet keeps a
            # clean float dtype. A mixed float/"DATA_MISSING" column is unwritable
            # by pyarrow. Missing factors render as DATA_MISSING in the CSV + UI.
            "composite_score": composite,
            "data_completeness_pct": round(n_avail / len(WEIGHTS) * 100, 0),
            "computed_at": datetime.now().isoformat(timespec="seconds"),
        }
        for k in WEIGHTS:
            row[k] = scores[k]
        rows.append(row)

    out = pd.DataFrame(rows, columns=SCORECARD_COLS)
    scored = out[out["composite_score"].notna()]
    log(f"Scored {len(out)} companies ({len(scored)} with a composite). "
        f"Mean completeness: {out['data_completeness_pct'].mean():.0f}%")

    if args.dry_run:
        log("DRY-RUN — not writing. Sample (sorted by composite):")
        show = out.copy()
        show["_sort"] = pd.to_numeric(show["composite_score"], errors="coerce").fillna(-1)
        show = show.sort_values("_sort", ascending=False)
        cols = ["symbol", "composite_score", "data_completeness_pct", "score_technical",
                "score_fundamental", "score_fin_health", "score_mgmt_cred",
                "score_valuation", "score_guidance", "score_fraud_risk",
                "score_investigative"]
        print(show[cols].fillna(DATA_MISSING).head(12).to_string(index=False))
        return

    store.write_df(["company_repo", "_index", "company_scorecard.parquet"], out)
    store.write_df(["company_repo", "_index", "company_scorecard.csv"], out.fillna(DATA_MISSING))
    log("Wrote company_scorecard.parquet + company_scorecard.csv to _index/.")


if __name__ == "__main__":
    main()
