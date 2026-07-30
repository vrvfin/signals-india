r"""
narrative_factpack.py — Layer A of the narrative report. NO LLM, NO GENERATION.

Builds `factpack.json` for one company: every number the report may contain, each with
the source that produced it. The narrative model (Layer B) receives only these facts and
may not introduce a figure of its own — `verify_grounding.py` enforces that mechanically.

Fact shape:
  {id, label, value, unit, basis, section, source{...}, computed_from[...]}

`basis` is load-bearing and never omitted. "reported" and "screener_rounded" are different
claims about the same number, and the Landmark check showed the difference reaching 5-7%
in sensitivity cells (see narrative_compute --selftest [7b]/[7c]).

Sections covered here are the ones computable from data already on Drive:
  2  company on one page      18  full multi-year financials
  5  scale history            19  operating leverage + sensitivity
  8  stable series            21  peer set
                              22  sensitivity grid
                              24  financial/execution risk register

Sections 1, 3, 4, 9-17, 20, 23, 25, 26 need extraction that does not exist yet; they are
emitted as explicit coverage gaps so the report renders DATA_MISSING rather than inventing.

Usage:
  python scripts/narrative_factpack.py --names "Landmark Cars" --dry-run
  python scripts/narrative_factpack.py --isin INE559R01029 --out factpack.json
"""
from __future__ import annotations

import argparse
import io
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import pandas as pd
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from _extractor_base import (get_drive, get_or_create_subfolder, find_file,
                             download_bytes, log)
import narrative_compute as NC

IDX = "company_repo/_index"
FUND = "fundamentals"

# Sections that require extraction this repo does not yet perform. Listed so the pack
# reports its own blind spots instead of leaving them silently absent.
UNCOVERED = {
    1: "history timeline — needs AR/DRHP milestone extraction (N7)",
    3: "legal entity map — needs AR subsidiary-schedule extraction (N7)",
    4: "management bench — needs AR KMP/board extraction (N7)",
    9: "portfolio table — needs AR/presentation extraction (N7)",
    10: "tier framework — derived by Layer B from section 9",
    11: "mix shift — needs two years of section 9",
    12: "per-unit deep dives — needs AR segment extraction (N7)",
    16: "alt-data claim test — needs the sector alt-data registry (N9)",
    17: "alt-data scorecard — needs the sector alt-data registry (N9)",
    20: "verbatim quote spine — needs mgmt_quotes.parquet (N8)",
    23: "structural risk register — needs AR/concall risk extraction (N7)",
    25: "policy backdrop — needs the alt-data registry (N9)",
}


# ------------------------------------------------------------------ drive io --
class Store:
    """Thin cached reader over the Drive layout. One instance per run."""

    def __init__(self):
        self.drive = get_drive()
        self.root = os.environ["GDRIVE_FOLDER_ID"]
        self._folders: dict[str, str] = {}
        self._files: dict[tuple[str, str], pd.DataFrame] = {}

    def folder(self, path: str) -> str:
        if path not in self._folders:
            fid = self.root
            for part in path.split("/"):
                fid = get_or_create_subfolder(self.drive, fid, part)
            self._folders[path] = fid
        return self._folders[path]

    def parquet(self, path: str, name: str) -> pd.DataFrame:
        key = (path, name)
        if key not in self._files:
            try:
                fid = find_file(self.drive, self.folder(path), name)
                self._files[key] = (pd.DataFrame() if not fid else
                                    pd.read_parquet(io.BytesIO(
                                        download_bytes(self.drive, fid))))
            except Exception as e:
                log(f"  WARNING: {path}/{name} unreadable ({str(e)[:80]})")
                self._files[key] = pd.DataFrame()
        return self._files[key]

    def by_isin(self, name: str, isin: str, symbol: str = "") -> pd.DataFrame:
        df = self.parquet(IDX, name)
        if df.empty:
            return df
        if "isin" in df.columns:
            m = df[df["isin"].astype(str) == str(isin)]
            if not m.empty:
                return m
        if symbol and "symbol" in df.columns:
            return df[df["symbol"].astype(str).str.upper() == symbol.upper()]
        return df.iloc[0:0]


# -------------------------------------------------------------------- facts ---
class Pack:
    """Accumulates facts and coverage gaps for one company."""

    def __init__(self, isin: str, symbol: str, name: str):
        self.isin, self.symbol, self.name = isin, symbol, name
        self.facts: list[dict] = []
        self.gaps: list[dict] = []
        self.tables: list[dict] = []
        self.as_of = datetime.now(timezone.utc).isoformat(timespec="seconds")

    def add(self, fid, label, value, unit, basis, section, source,
            computed_from=None) -> str | None:
        """Record one fact. A None/NaN value is a GAP, never a fact — this is the
        single most important rule in the file."""
        if value is None or (isinstance(value, float) and value != value):
            self.gap(section, f"{label} unavailable ({fid})")
            return None
        self.facts.append({
            "id": fid, "label": label,
            "value": round(value, 4) if isinstance(value, float) else value,
            "unit": unit, "basis": basis, "section": section,
            "source": source, "computed_from": computed_from or [],
        })
        return fid

    def gap(self, section: int, reason: str):
        self.gaps.append({"section": section, "reason": reason})

    def table(self, tid, title, section, columns, rows, source, note=""):
        self.tables.append({"id": tid, "title": title, "section": section,
                            "columns": columns, "rows": rows, "source": source,
                            "note": note})

    def to_dict(self) -> dict:
        by_sec: dict[int, int] = {}
        for f in self.facts:
            by_sec[f["section"]] = by_sec.get(f["section"], 0) + 1
        return {
            "schema_version": 1,
            "company": {"isin": self.isin, "symbol": self.symbol, "name": self.name},
            "as_of_utc": self.as_of,
            "facts": self.facts,
            "tables": self.tables,
            "coverage_gaps": self.gaps,
            "uncovered_sections": [{"section": k, "reason": v}
                                   for k, v in sorted(UNCOVERED.items())],
            "fact_counts_by_section": dict(sorted(by_sec.items())),
        }


def _src_parquet(table: str, note: str = "") -> dict:
    return {"kind": "parquet", "table": table, "note": note}


def _src_stmt(symbol: str, line_items, period, fetched_at=None) -> dict:
    return {"kind": "statements",
            "table": f"fundamentals/statements/{symbol}.parquet",
            "line_items": list(line_items), "period": period,
            "fetched_at": fetched_at}


def _src_computed(note: str) -> dict:
    return {"kind": "computed", "note": note}


# ----------------------------------------------------------------- resolver ---
def resolve(store: Store, token: str) -> tuple[str, str, str] | None:
    """token (ISIN / symbol / name fragment) -> (isin, symbol, name)."""
    facts = store.parquet(IDX, "company_facts.parquet")
    if facts.empty:
        return None
    t = str(token).strip()
    for col in ("isin", "symbol"):
        m = facts[facts[col].astype(str).str.upper() == t.upper()]
        if not m.empty:
            r = m.iloc[0]
            return str(r["isin"]), str(r["symbol"]), str(r["name"])
    m = facts[facts["name"].astype(str).str.contains(t, case=False, na=False, regex=False)]
    if m.empty:
        return None
    if len(m) > 1:
        # Prefer an exact name match; otherwise the largest by market cap, and say so.
        exact = m[m["name"].astype(str).str.lower() == t.lower()]
        m = exact if not exact.empty else m.sort_values("mcap_cr", ascending=False)
        log(f"  NOTE: '{token}' matched {len(m)} companies; using "
            f"{m.iloc[0]['name']} ({m.iloc[0]['symbol']})")
    r = m.iloc[0]
    return str(r["isin"]), str(r["symbol"]), str(r["name"])


# ---------------------------------------------------------------- sections ----
def sec2_one_pager(pack: Pack, store: Store):
    """Section 2 — the company on one page."""
    f = store.by_isin("company_facts.parquet", pack.isin, pack.symbol)
    if f.empty:
        pack.gap(2, "no company_facts row")
        return
    r = f.iloc[0]
    src = _src_parquet(f"{IDX}/company_facts.parquet",
                       f"updated_at={r.get('updated_at')}")
    for col, label, unit in (
            ("mcap_cr", "Market capitalisation", "Rs Cr"),
            ("pe", "P/E (reported)", "x"),
            ("pb", "P/B", "x"),
            ("rev_ttm", "Revenue (TTM)", "Rs Cr"),
            ("pat_ttm", "PAT (TTM)", "Rs Cr"),
            ("eps_ttm", "EPS (TTM)", "Rs"),
            ("ret_12m_pct", "12-month price return", "%")):
        v = pd.to_numeric(r.get(col), errors="coerce")
        pack.add(f"one.{col}", label, None if pd.isna(v) else float(v),
                 unit, "reported", 2, src)

    s = store.parquet(FUND, "summary.parquet")
    s = s[s["symbol"].astype(str).str.upper() == pack.symbol.upper()] if not s.empty \
        else s
    if s.empty:
        pack.gap(2, "no fundamentals/summary row")
        return
    r2 = s.iloc[0]
    src2 = _src_parquet(f"{FUND}/summary.parquet", f"fetched_at={r2.get('fetched_at')}")
    for col, label, unit in (("roce_pct", "RoCE", "%"), ("roe_pct", "RoE", "%"),
                             ("debt_to_equity", "Debt / equity", "x"),
                             ("promoter_holding_pct", "Promoter holding", "%"),
                             ("dividend_yield_pct", "Dividend yield", "%")):
        v = pd.to_numeric(r2.get(col), errors="coerce")
        pack.add(f"one.{col}", label, None if pd.isna(v) else float(v),
                 unit, "reported", 2, src2)


def _statements(store: Store, symbol: str) -> pd.DataFrame:
    return store.parquet(f"{FUND}/statements", f"{symbol}.parquet")


def sec18_financials(pack: Pack, store: Store, st: pd.DataFrame):
    """Section 18 — the full multi-year table, EBITDA on BOTH bases."""
    if st.empty:
        pack.gap(18, f"no statements file for {pack.symbol}")
        return
    fetched = str(st["fetched_at"].iloc[0]) if "fetched_at" in st.columns else None
    series = {
        "revenue": NC.revenue_series(st),
        "ebitda_incl_oi": NC.ebitda_series(st, "incl_other_income"),
        "ebitda_op_only": NC.ebitda_series(st, "operating_profit"),
        "pbt": NC.pbt_series(st), "pat": NC.pat_series(st),
        "dep": NC.depreciation_series(st), "interest": NC.interest_series(st),
        "eps": NC.eps_series(st), "cfo": NC.cfo_series(st),
        "margin": NC.margin_series(st),
    }
    # Two EBITDA series coexist by design. Their labels MUST differ, or the report
    # shows two rows reading "EBITDA FY18" with different values and no way to tell
    # which is which.
    disambig = {"ebitda_incl_oi": "EBITDA (incl. other income)",
                "ebitda_op_only": "EBITDA (operating profit only)"}
    for key, s in series.items():
        name = disambig.get(key, s.name)
        for fy, v in s.as_map().items():
            pack.add(f"fin.{key}.{fy}", f"{name} {fy}", v, s.unit,
                     f"screener_rounded/{s.basis}", 18,
                     _src_stmt(pack.symbol, s.inputs, fy, fetched))

    fys = [NC.fy_label(p) for p in series["revenue"].periods]
    rows = []
    for fy in fys:
        rows.append({"FY": fy, **{k: series[k].as_map().get(fy) for k in series}})
    pack.table("tbl.financials", "Multi-year financials (Screener consolidated)", 18,
               ["FY"] + list(series), rows,
               _src_stmt(pack.symbol, ["annual_pl", "cash_flow"], "all", fetched),
               note="EBITDA shown on two bases: incl_other_income (how companies "
                    "usually quote it) and operating_profit (Screener's own line). "
                    "Figures are Screener's whole-crore rounding.")

    for key in ("revenue", "ebitda_incl_oi", "pat"):
        c = NC.cagr_pct(series[key])
        if c is None:
            pack.gap(18, f"{key} CAGR not computable (sign change or single point)")
            continue
        pack.add(f"fin.{key}.cagr", f"{series[key].name} CAGR "
                                    f"{c['from_fy']}-{c['to_fy']}",
                 c["cagr_pct"], "%", "computed", 18,
                 _src_computed(f"CAGR over {c['years']}y from {c['from']} to {c['to']}"),
                 [f"fin.{key}.{c['from_fy']}", f"fin.{key}.{c['to_fy']}"])


def sec5_scale_history(pack: Pack, store: Store, st: pd.DataFrame):
    """Section 5 — did revenue and profit scale together? Capacity is the missing leg."""
    if st.empty:
        pack.gap(5, "no statements file")
        return
    rev, eb = NC.revenue_series(st), NC.ebitda_series(st)
    rc, ec = NC.cagr_pct(rev), NC.cagr_pct(eb)          # windowed: last 5 FY points
    if rc and ec:
        pack.add("scale.revenue_cagr", f"Revenue CAGR ({rc['from_fy']}-{rc['to_fy']})",
                 rc["cagr_pct"], "%", "computed", 5,
                 _src_computed(f"{rc['window_note']}, {rc['from_fy']}-{rc['to_fy']}"))
        pack.add("scale.ebitda_cagr", f"EBITDA CAGR ({ec['from_fy']}-{ec['to_fy']})",
                 ec["cagr_pct"], "%", "computed", 5,
                 _src_computed(f"{ec['window_note']}, {ec['from_fy']}-{ec['to_fy']}"))
        # Signed so that NEGATIVE = profit grew slower than scale, which is the
        # condition worth noticing (capacity built ahead of the profit that funds it).
        pack.add("scale.gap_pp", "EBITDA CAGR minus revenue CAGR "
                                 "(negative = profit lagged scale)",
                 ec["cagr_pct"] - rc["cagr_pct"], "pp", "computed", 5,
                 _src_computed("profit growth less revenue growth over the same window"),
                 ["scale.ebitda_cagr", "scale.revenue_cagr"])
        for tag, c in (("revenue", rc), ("EBITDA", ec)):
            if c["base_warning"]:
                pack.gap(5, f"{tag} CAGR caveat — {c['base_warning']}")
        # The full-history figure is materially different often enough that hiding it
        # would itself be a distortion; both are reported, each labelled by window.
        for tag, label, s in (("revenue", "Revenue", rev), ("ebitda", "EBITDA", eb)):
            full = NC.cagr_pct(s, window=None)
            if full and abs(full["cagr_pct"] - (rc if tag == "revenue" else ec)
                            ["cagr_pct"]) > 3.0:
                pack.add(f"scale.{tag}_cagr_full",
                         f"{label} CAGR, full history "
                         f"({full['from_fy']}-{full['to_fy']})",
                         full["cagr_pct"], "%", "computed", 5,
                         _src_computed(f"{full['window_note']}"
                                       + (f" — CAVEAT: {full['base_warning']}"
                                          if full["base_warning"] else "")))
    else:
        pack.gap(5, "CAGR pair not computable")
    pack.gap(5, "physical capacity / outlet counts not extracted — needs N7")


def sec8_stable_series(pack: Pack, store: Store, st: pd.DataFrame):
    """Section 8 — which disclosed series is the steadiest? Deterministic, not a claim
    about segments (segment data needs N7)."""
    if st.empty:
        pack.gap(8, "no statements file")
        return
    marg = NC.margin_series(st)
    pairs = [(NC.fy_label(p), v) for p, v in zip(marg.periods, marg.values)
             if v is not None]
    if len(pairs) < 3:
        pack.gap(8, "margin series too short to characterise stability")
        return

    # A band measured across loss-making or COVID years is not evidence of stability —
    # quoting one under a "steadiest series" heading would assert the opposite of what
    # the data shows. Report the RECENT window as the headline and the full range
    # separately, each labelled with the years it covers.
    pat = NC.pat_series(st).as_map()
    recent = pairs[-5:]
    loss_years = [fy for fy, _ in recent if (pat.get(fy) or 0) < 0]
    window = [(fy, v) for fy, v in recent if (pat.get(fy) or 0) >= 0]
    if len(window) < 3:
        pack.gap(8, f"fewer than 3 profitable years in the recent window "
                    f"({', '.join(loss_years)} loss-making) — no stability claim "
                    f"is supportable")
        window = recent

    fys = [fy for fy, _ in window]
    vals = [v for _, v in window]
    lo, hi = min(vals), max(vals)
    span = f"{fys[0]}-{fys[-1]}"
    src = _src_computed(f"min/max of the disclosed EBITDA margin over {span}, "
                        f"profitable years only")
    pack.add("stable.margin_band_low", f"EBITDA margin — lowest year ({span})", lo,
             "%", "computed", 8, src)
    pack.add("stable.margin_band_high", f"EBITDA margin — highest year ({span})", hi,
             "%", "computed", 8, src)
    pack.add("stable.margin_band_bps", f"EBITDA margin band width ({span})",
             (hi - lo) * 100.0, "bps", "computed", 8,
             _src_computed(f"high minus low over {span}, in basis points"),
             ["stable.margin_band_low", "stable.margin_band_high"])
    pack.add("stable.years_in_band", "Years in the band", len(window), "count",
             "computed", 8, src)

    allv = [v for _, v in pairs]
    if (max(allv) - min(allv)) > (hi - lo) * 1.25:
        pack.add("stable.margin_band_bps_full",
                 f"EBITDA margin band width, full history "
                 f"({pairs[0][0]}-{pairs[-1][0]})",
                 (max(allv) - min(allv)) * 100.0, "bps", "computed", 8,
                 _src_computed("full-history range, which spans any downturn or "
                               "loss years — materially wider than the recent band"))


def sec19_operating_leverage(pack: Pack, store: Store, st: pd.DataFrame) -> dict | None:
    """Section 19 — the fixed block, its absorption of EBITDA, and the margin-only
    sensitivity. Returns the leverage dict so section 22 can reuse it."""
    if st.empty:
        pack.gap(19, "no statements file")
        return None
    lev = NC.operating_leverage(st)
    if lev is None:
        pack.gap(19, "operating leverage not computable (missing D&A/interest/PBT)")
        return None
    fy = lev["fy"]
    fetched = str(st["fetched_at"].iloc[0]) if "fetched_at" in st.columns else None
    src_calc = _src_computed(f"D&A + finance cost over EBITDA, {fy}, "
                             f"basis={lev['basis']}")
    # Raw figures cite the statements file they were read from; only the ratios
    # derived here cite "computed". Otherwise the section's source footer would name
    # no document at all.
    raw_inputs = {"ebitda": [NC.L_OP, NC.L_OTHER_INCOME], "depreciation": [NC.L_DEP],
                  "finance_cost": [NC.L_INTEREST], "pbt": [NC.L_PBT]}
    derived = ("fixed_block", "absorption_pct", "amplification_x")
    for k, label, unit in (("ebitda", "EBITDA", "Rs Cr"),
                           ("depreciation", "Depreciation & amortisation", "Rs Cr"),
                           ("finance_cost", "Finance cost", "Rs Cr"),
                           ("fixed_block", "Fixed block (D&A + finance)", "Rs Cr"),
                           ("pbt", "Profit before tax", "Rs Cr"),
                           ("absorption_pct", "Fixed block as % of EBITDA", "%"),
                           ("amplification_x", "PBT move per 1% EBITDA move", "x")):
        is_derived = k in derived
        pack.add(f"lev.{k}", f"{label} ({fy})", lev[k], unit,
                 "computed" if is_derived else "screener_rounded", 19,
                 src_calc if is_derived
                 else _src_stmt(pack.symbol, raw_inputs[k], fy, fetched),
                 [f"lev.{x}" for x in ("depreciation", "finance_cost")]
                 if k == "fixed_block" else [])

    if not lev["exceptionals_included"]:
        pack.gap(19, "exceptional items not available from Screener — fixed block may "
                     f"be understated (residual vs disclosed EBITDA-PBT gap: "
                     f"{lev['residual_vs_disclosed_gap']:+.1f} Rs Cr)")
    if not lev["reconciles"]:
        pack.gap(19, "WARNING: fixed block does NOT reconcile to the disclosed "
                     f"EBITDA-PBT gap (residual {lev['residual_vs_disclosed_gap']:+.1f} "
                     "Rs Cr) — an undisclosed item sits between them")

    shares = NC.shares_outstanding_cr(st)
    if shares:
        pack.add("lev.shares_cr", "Shares outstanding (derived)", shares, "Cr",
                 "computed", 19, _src_computed("PAT / EPS for the latest FY"),
                 [f"fin.pat.{fy}", f"fin.eps.{fy}"])
    tax = NC.effective_tax_pct(st, fy)
    if tax is not None:
        pack.add("lev.tax_pct", f"Effective tax rate ({fy})", tax, "%", "computed",
                 19, _src_computed("derived from the disclosed PBT/PAT pair, which is "
                                   "finer than Screener's rounded Tax % line"))

    marg = lev["ebitda"] / NC.revenue_series(st).as_map()[fy] * 100.0
    points = sorted({round(marg, 2), round(marg + 0.5, 2), round(marg + 1.0, 2),
                     round(max(v for v in NC.margin_series(st).values
                               if v is not None), 2)})
    rows = NC.leverage_sensitivity(st, points, shares_cr=shares)
    if rows:
        pack.table("tbl.leverage_sensitivity",
                   f"Margin-only sensitivity, revenue held at {fy} actual", 19,
                   ["margin_pct", "ebitda", "pbt", "pat", "eps", "vs_actual_pat_pct"],
                   rows, _src_computed("mechanical restatement of disclosed figures"),
                   note="NOT a forecast. Revenue, D&A and finance cost held at their "
                        f"{fy} disclosed values; only the EBITDA margin varies.")
    return lev


def sec22_sensitivity_grid(pack: Pack, store: Store, st: pd.DataFrame, lev: dict | None):
    """Section 22 — implied P/E and EPS across revenue x margin at today's mcap."""
    if lev is None or st.empty:
        pack.gap(22, "needs the section 19 leverage base")
        return
    fy = lev["fy"]
    f = store.by_isin("company_facts.parquet", pack.isin, pack.symbol)
    mcap = pd.to_numeric(f.iloc[0].get("mcap_cr"), errors="coerce") if not f.empty \
        else None
    if mcap is None or pd.isna(mcap):
        pack.gap(22, "no market capitalisation — grid not computable")
        return
    rev = NC.revenue_series(st).as_map()[fy]
    shares = NC.shares_outstanding_cr(st)
    tax = NC.effective_tax_pct(st, fy)
    if tax is None:
        pack.gap(22, "effective tax rate not derivable")
        return
    rev_levels = [round(rev * m) for m in (1.0, 1.10, 1.21, 1.31)]
    m_now = 100.0 * lev["ebitda"] / rev
    m_peak = max(v for v in NC.margin_series(st).values if v is not None)
    m_levels = sorted({round(m_now, 2), round(m_now + 0.5, 2), round(m_now + 1.0, 2),
                       round(m_peak, 2)})
    g = NC.sensitivity_grid(rev_levels, m_levels, lev["fixed_block"], float(tax),
                            float(mcap), shares)
    rows = [{"revenue": r, **{f"m{m}": (None if v is None else round(v, 1))
                              for m, v in zip(m_levels, pe_row)}}
            for r, pe_row in zip(rev_levels, g["implied_pe"])]
    pack.table("tbl.sensitivity_pe", f"Implied P/E at market cap Rs {mcap:,.0f} Cr", 22,
               ["revenue"] + [f"m{m}" for m in m_levels], rows,
               _src_computed(f"revenue x margin - fixed block {lev['fixed_block']:.1f}, "
                             f"taxed at {tax:.1f}%, over mcap {mcap:,.0f}"),
               note=g["note"])
    pack.add("grid.mcap_cr", "Market capitalisation used in the grid", float(mcap),
             "Rs Cr", "reported", 22,
             _src_parquet(f"{IDX}/company_facts.parquet"))


def sec21_peers(pack: Pack, store: Store):
    """Section 21 — peer group and where the company sits in it."""
    f = store.by_isin("company_facts.parquet", pack.isin, pack.symbol)
    if f.empty:
        pack.gap(21, "no company_facts row")
        return
    r = f.iloc[0]
    pg = str(r.get("peer_group") or "").strip()
    if not pg:
        pack.gap(21, "no peer_group assigned in classification")
        return
    pack.add("peer.group", "Peer group", pg, "label", "classification", 21,
             _src_parquet(f"{IDX}/company_facts.parquet"))
    pa = store.parquet(IDX, "peer_aggregates.parquet")
    m = pa[(pa["level"] == "peer_group") & (pa["group"] == pg)] if not pa.empty \
        else pd.DataFrame()
    if m.empty:
        pack.gap(21, f"no peer aggregates for group '{pg}'")
    else:
        a = m.iloc[0]
        src = _src_parquet(f"{IDX}/peer_aggregates.parquet", f"n={a.get('n')}")
        pack.add("peer.n", "Companies in peer group", int(a["n"]), "count",
                 "computed", 21, src)
        for col, label, unit in (("pe_median", "Peer median P/E", "x"),
                                 ("ret_12m_pct_median", "Peer median 12m return", "%"),
                                 ("rev_q_yoy_median", "Peer median revenue YoY", "%"),
                                 ("pat_q_yoy_median", "Peer median PAT YoY", "%")):
            v = pd.to_numeric(a.get(col), errors="coerce")
            pack.add(f"peer.{col}", label, None if pd.isna(v) else float(v),
                     unit, "peer_median", 21, src)
    val = store.by_isin("valuation.parquet", pack.isin, pack.symbol)
    if not val.empty:
        v = val.iloc[0]
        src = _src_parquet(f"{IDX}/valuation.parquet", f"basis={v.get('basis')}")
        for col, label, unit in (("pe_pctile_segment", "P/E percentile within segment", "%"),
                                 ("peg_proxy", "PEG proxy", "x"),
                                 ("valuation_score", "Valuation score", "/100")):
            x = pd.to_numeric(v.get(col), errors="coerce")
            pack.add(f"peer.{col}", label, None if pd.isna(x) else float(x),
                     unit, "computed", 21, src)
    pack.gap(21, "global/listed-overseas comparables are not in this system — the peer "
                 "set is Indian listed names only")


def sec24_risk_register(pack: Pack, store: Store):
    """Section 24 — the financial/execution risk register, from computed trackers."""
    ft = store.by_isin("fraud_tracker.parquet", pack.isin, pack.symbol)
    rows = []
    if not ft.empty:
        r = ft.iloc[0]
        src = _src_parquet(f"{IDX}/fraud_tracker.parquet", f"as_of={r.get('as_of')}")
        pack.add("risk.fraud_score", "Fraud tracker score",
                 float(pd.to_numeric(r.get("fraud_score"), errors="coerce")),
                 "/100", "computed", 24, src)
        pack.add("risk.fraud_band", "Fraud tracker band", str(r.get("band")),
                 "label", "computed", 24, src)
        flags = str(r.get("forensic_flags") or "").strip()
        if flags:
            for fl in [x.strip() for x in flags.split(";") if x.strip()]:
                rows.append({"risk": fl, "kind": "forensic accounting flag",
                             "source": "fraud_tracker.forensic_flags",
                             "status": "LIVE"})
    inv = store.by_isin("investigative_fraud.parquet", pack.isin, pack.symbol)
    if not inv.empty:
        r = inv.iloc[0]
        surv = [str(v) for v in (r.get("asm_level"), r.get("esm_level"))
                if str(v) not in ("none", "", "nan", "None")]
        gsm = pd.to_numeric(r.get("gsm_stage"), errors="coerce")
        if pd.notna(gsm) and int(gsm):
            surv.append(f"GSM-{int(gsm)}")
        if bool(r.get("t2t")):
            surv.append("T2T")
        for s in surv:
            rows.append({"risk": s, "kind": "exchange surveillance (price/volatility "
                                            "control, NOT fraud evidence)",
                         "source": "investigative_fraud", "status": "LIVE"})
        for col, lbl in (("sebi_actions", "SEBI order match"),
                         ("nfra_actions", "NFRA order")):
            n = pd.to_numeric(r.get(col), errors="coerce")
            if pd.notna(n) and int(n):
                rows.append({"risk": f"{int(n)} {lbl}(s)", "kind": "integrity signal",
                             "source": f"investigative_fraud.{col}", "status": "OPEN"})
    arf = store.by_isin("ar_red_flags.parquet", pack.isin, pack.symbol)
    if not arf.empty:
        for _, r in arf.sort_values("fy_year").tail(12).iterrows():
            # fy_year and page_ref already carry their own prefixes ("FY23", "Page 207")
            # in the stored data — do not add another.
            fy = str(r.get("fy_year") or "").strip()
            page = str(r.get("page_ref") or "").strip()
            where = " ".join(x for x in (fy, page) if x and x.lower() != "nan")
            rows.append({"risk": str(r.get("flag_type")),
                         "kind": f"annual report flag ({r.get('category')}, "
                                 f"severity {r.get('severity')})",
                         "source": f"ar_red_flags {where}".strip(),
                         "status": "LIVE"})
    if rows:
        pack.table("tbl.risk_register", "Financial, policy and execution risks", 24,
                   ["risk", "kind", "source", "status"], rows,
                   _src_parquet("fraud_tracker / investigative_fraud / ar_red_flags"))
    else:
        pack.gap(24, "no risk rows — trackers are clean or have no coverage")

    der = store.by_isin("financials_derived.parquet", pack.isin, pack.symbol)
    if not der.empty and "metric" in der.columns:
        latest = der.sort_values("period").groupby("metric").tail(1)
        for _, r in latest.iterrows():
            v = pd.to_numeric(r.get("value"), errors="coerce")
            pack.add(f"ratio.{r['metric']}", f"{r['metric']} ({r.get('period')})",
                     None if pd.isna(v) else float(v), str(r.get("unit") or ""),
                     "computed", 24,
                     _src_parquet(f"{IDX}/financials_derived.parquet",
                                  f"period={r.get('period')}"))


# --------------------------------------------------------------------- main ---
def build(store: Store, token: str) -> Pack | None:
    r = resolve(store, token)
    if r is None:
        log(f"  could not resolve '{token}' to a company")
        return None
    isin, symbol, name = r
    log(f"  {name} ({symbol} / {isin})")
    pack = Pack(isin, symbol, name)
    st = _statements(store, symbol)
    if st.empty:
        log(f"  WARNING: no fundamentals/statements/{symbol}.parquet — the financial "
            f"sections will be empty")
    sec2_one_pager(pack, store)
    sec18_financials(pack, store, st)
    sec5_scale_history(pack, store, st)
    sec8_stable_series(pack, store, st)
    lev = sec19_operating_leverage(pack, store, st)
    sec22_sensitivity_grid(pack, store, st, lev)
    sec21_peers(pack, store)
    sec24_risk_register(pack, store)
    return pack


def _report(pack: Pack):
    d = pack.to_dict()
    print(f"\n=== {pack.name} ({pack.symbol} / {pack.isin}) ===")
    print(f"  facts: {len(d['facts'])}   tables: {len(d['tables'])}   "
          f"gaps: {len(d['coverage_gaps'])}")
    print("  facts by section:")
    for sec, n in d["fact_counts_by_section"].items():
        print(f"    section {sec:>2}: {n:>3} facts")
    tb = {}
    for t in d["tables"]:
        tb.setdefault(t["section"], []).append(f"{t['id']} ({len(t['rows'])} rows)")
    for sec, items in sorted(tb.items()):
        print(f"    section {sec:>2}: tables {', '.join(items)}")
    if d["coverage_gaps"]:
        print("  coverage gaps (rendered as DATA_MISSING, never estimated):")
        for g in d["coverage_gaps"]:
            print(f"    [s{g['section']:>2}] {g['reason']}")
    print(f"  sections needing extraction that does not exist yet: "
          f"{[u['section'] for u in d['uncovered_sections']]}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--names", nargs="*", default=[],
                    help="company tokens (ISIN / symbol / name fragment)")
    ap.add_argument("--isin", nargs="*", default=[], help="alias for --names")
    ap.add_argument("--out", default="", help="write factpack JSON here (one company) "
                                              "or to <out>/<symbol>.json (several)")
    ap.add_argument("--dry-run", action="store_true",
                    help="build and report coverage, write nothing")
    a = ap.parse_args()

    tokens = list(a.names) + list(a.isin)
    if not tokens:
        ap.error("give at least one --names/--isin token")

    store = Store()
    packs = []
    for t in tokens:
        log(f"resolving '{t}'")
        p = build(store, t)
        if p:
            packs.append(p)
    if not packs:
        return 1
    for p in packs:
        _report(p)

    if a.dry_run:
        print("\nDRY RUN — nothing written.")
        return 0
    if not a.out:
        print("\nNo --out given; nothing written. Re-run with --out or --dry-run.")
        return 0
    out = Path(a.out)
    if len(packs) == 1 and out.suffix == ".json":
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(packs[0].to_dict(), indent=2), encoding="utf-8")
        print(f"\nwrote {out}")
    else:
        out.mkdir(parents=True, exist_ok=True)
        for p in packs:
            f = out / f"{p.symbol}.json"
            f.write_text(json.dumps(p.to_dict(), indent=2), encoding="utf-8")
            print(f"wrote {f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
