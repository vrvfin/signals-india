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
import re
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
# Sections with no data source at all. Everything else is populated from Drive; a
# section that HAS a source but no rows for this company reports a coverage gap instead,
# so "we cannot do this yet" stays distinct from "this company has none".
UNCOVERED = {
    10: "tier framework — derived by Layer B from section 9",
    16: "alt-data claim test — no automatable feed; Vahan connection-resets and "
        "analytics.parivahan returns 403 (probed 2026-07-30). Drop a CSV in the "
        "alt-data intake to populate this.",
    17: "alt-data scorecard — same as section 16",
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
                # A genuine absence (no fid) is a STABLE fact — cache the empty frame.
                # A read EXCEPTION is TRANSIENT (Drive rate-limit after heavy I/O), so
                # it must NOT be cached: caching it once poisoned every later resolve()
                # in the same run, which is how a mid-run --fetch-missing turned a
                # working TCS run into "could not resolve 'TCS'". Retry with a short
                # backoff, and on total failure return empty WITHOUT caching so the
                # next call can try again.
                if not fid:
                    self._files[key] = pd.DataFrame()
                else:
                    self._files[key] = pd.read_parquet(
                        io.BytesIO(download_bytes(self.drive, fid)))
            except Exception as e:
                import time as _t
                for attempt in range(2):
                    _t.sleep(1.5 * (attempt + 1))
                    try:
                        fid = find_file(self.drive, self.folder(path), name)
                        if fid:
                            return pd.read_parquet(io.BytesIO(
                                download_bytes(self.drive, fid)))
                    except Exception:
                        continue
                log(f"  WARNING: {path}/{name} unreadable after retries "
                    f"({str(e)[:70]}) — returning empty, NOT caching")
                return pd.DataFrame()
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


# ---------------------------------------------------- extracted structure ----
def _struct(store: Store, isin: str, symbol: str, kind: str) -> pd.DataFrame:
    df = store.by_isin("company_structure.parquet", isin, symbol)
    if df.empty or "kind" not in df.columns:
        return pd.DataFrame()
    return df[df["kind"].astype(str) == kind]


def _src_doc(row) -> dict:
    """Provenance for an extracted record: the document it came from, plus the verbatim
    span that survived the extractor's containment check."""
    return {"kind": "document", "doc_type": str(row.get("source_doc_id", "")).split("_")[0],
            "date": str(row.get("doc_date", "")), "title": str(row.get("source_doc_id", "")),
            "evidence_span": str(row.get("evidence_span", ""))[:300]}


# Indian filings print figures in Rs Million / Lakh / Crore interchangeably, while every
# computed figure in this pack is Rs Cr. Carrying both side by side without conversion
# invites a reader (or a model) to compare 46,886 against 4,896 as if they were the same
# scale. So: CONVERT to Rs Cr, keep the original, and mark every converted cell with *
# so the appendix can show exactly what was transformed.
_TO_CR = {"million": 0.1, "mn": 0.1, "rs million": 0.1, "rs mn": 0.1,
          "lakh": 0.01, "lakhs": 0.01, "rs lakh": 0.01,
          "billion": 100.0, "bn": 100.0,
          "crore": 1.0, "cr": 1.0, "rs cr": 1.0, "rs crore": 1.0}


def _to_crore(value: str, unit: str) -> tuple[float | None, float | None, str, bool]:
    """-> (converted_cr, original_value, original_unit, was_converted).

    Returns converted=None when the unit is not a currency scale (e.g. '%'), so the
    caller leaves the cell untouched rather than silently scaling a percentage.
    """
    u = str(unit or "").strip().lower()
    try:
        raw = float(str(value).replace(",", "").split()[0])
    except (ValueError, IndexError):
        return None, None, unit, False
    if u not in _TO_CR:
        return None, raw, unit, False
    factor = _TO_CR[u]
    return round(raw * factor, 2), raw, unit, factor != 1.0


def _pivot_struct(rows: pd.DataFrame) -> list[dict]:
    """kind rows (item/field/value) -> one dict per item with its fields as columns."""
    out: dict[str, dict] = {}
    for _, r in rows.iterrows():
        item = str(r.get("item", "")).strip()
        if not item:
            continue
        rec = out.setdefault(item, {"item": item})
        f = str(r.get("field", "")).strip() or "value"
        val = str(r.get("value", "")).strip()
        unit = str(r.get("unit", "")).strip()
        rec[f] = f"{val} {unit}".strip() if unit and unit != "%" else (
            f"{val}%" if unit == "%" else val)
        if str(r.get("period", "")).strip():
            rec["period"] = str(r.get("period")).strip()
        rec.setdefault("_src", str(r.get("source_doc_id", "")))
    return list(out.values())


def sec1_history(pack: Pack, store: Store):
    rows = _struct(store, pack.isin, pack.symbol, "milestone")
    if rows.empty:
        pack.gap(1, "no milestones extracted — run extract_structure.py, and check the "
                    "company has an annual_report in the queue")
        return
    recs = []
    for _, r in rows.iterrows():
        yr = str(r.get("period") or r.get("value") or "").strip()
        recs.append({"year": yr, "event": str(r.get("item", "")),
                     "source": str(r.get("source_doc_id", ""))})
    recs.sort(key=lambda x: (x["year"] or "9999"))
    pack.table("tbl.milestones", "Dated milestones, as disclosed", 1,
               ["year", "event", "source"], recs,
               _src_doc(rows.iloc[0]),
               note="Every row carries a verbatim span from the filing it was taken "
                    "from; records whose span could not be found were discarded at "
                    "extraction.")
    pack.add("hist.n_milestones", "Milestones on record", len(recs), "count",
             "extracted", 1, _src_doc(rows.iloc[0]))


def sec3_entity_map(pack: Pack, store: Store):
    rows = _struct(store, pack.isin, pack.symbol, "subsidiary")
    if rows.empty:
        pack.gap(3, "no subsidiaries extracted from the annual report")
        return
    recs = _pivot_struct(rows)
    pack.table("tbl.subsidiaries", "Subsidiaries and associates, as disclosed", 3,
               ["item", "ownership_pct", "activity"], recs, _src_doc(rows.iloc[0]))
    pack.add("entity.n_subsidiaries", "Entities disclosed", len(recs), "count",
             "extracted", 3, _src_doc(rows.iloc[0]))
    wholly = [r for r in recs if str(r.get("ownership_pct", "")).startswith("100")]
    if wholly:
        pack.add("entity.n_wholly_owned", "Wholly owned (100%)", len(wholly), "count",
                 "extracted", 3, _src_doc(rows.iloc[0]))


def sec4_management(pack: Pack, store: Store):
    rows = _struct(store, pack.isin, pack.symbol, "management")
    if rows.empty:
        pack.gap(4, "no management records extracted from the annual report")
        return
    recs = _pivot_struct(rows)
    pack.table("tbl.management", "Board and key management, as disclosed", 4,
               ["item", "role", "since", "background"], recs, _src_doc(rows.iloc[0]),
               note="Extracted from the annual report; roles and tenures are as the "
                    "filing states them.")
    pack.add("mgmt.n_people", "People on record", len(recs), "count", "extracted", 4,
             _src_doc(rows.iloc[0]))


def sec6_segment_economics(pack: Pack, store: Store):
    """The deck's central chart: revenue share against profit share. Where one segment
    books most of the revenue and another earns most of the profit, that asymmetry IS
    the business model."""
    rows = _struct(store, pack.isin, pack.symbol, "segment")
    if rows.empty:
        pack.gap(6, "no segment disclosures extracted — the AR may not report segments")
        return
    # Convert money columns to Rs Cr so they are comparable with every other figure in
    # the report; mark converted cells with * and record the original for the appendix.
    recs, conversions = [], []
    for _, r in rows.iterrows():
        item, field = str(r.get("item", "")).strip(), str(r.get("field", "")).strip()
        if not item:
            continue
        rec = next((x for x in recs if x["item"] == item and
                    x.get("period") == str(r.get("period", "")).strip()), None)
        if rec is None:
            rec = {"item": item, "period": str(r.get("period", "")).strip()}
            recs.append(rec)
        cr, raw, unit, converted = _to_crore(r.get("value"), r.get("unit"))
        if cr is not None:
            rec[field] = f"{cr:,.2f}*" if converted else f"{cr:,.2f}"
            if converted:
                conversions.append({"item": item, "field": field,
                                    "period": rec["period"],
                                    "as_printed": f"{raw:,.2f}",
                                    "printed_unit": unit,
                                    "converted_to": f"{cr:,.2f}",
                                    "converted_unit": "Rs Cr",
                                    "factor": f"x{_TO_CR[str(unit).strip().lower()]}"})
        else:
            u = str(r.get("unit", "")).strip()
            v = str(r.get("value", "")).strip()
            rec[field] = f"{v}{u}" if u == "%" else (f"{v} {u}".strip())

    pack.table("tbl.segments", "Segment economics, as disclosed (Rs Cr)", 6,
               ["item", "period", "revenue", "profit", "margin_pct"], recs,
               _src_doc(rows.iloc[0]),
               note="Money columns are shown in Rs Cr. A * marks a figure the filing "
                    "printed in a DIFFERENT unit (usually Rs Million) that has been "
                    "converted here — see the unit-conversion appendix for the "
                    "as-printed value, the unit and the factor applied.")
    if conversions:
        pack.table("tbl.unit_conversions",
                   "* Unit conversions applied to section 6", 28,
                   ["item", "field", "period", "as_printed", "printed_unit",
                    "converted_to", "converted_unit", "factor"], conversions,
                   _src_computed("scale conversion to Rs Cr; no other transformation"),
                   note="Every figure marked * in this report appears here with the "
                        "value and unit exactly as the filing printed it. Nothing "
                        "else about the number was changed.")
        pack.add("units.n_converted", "Figures converted to Rs Cr", len(conversions),
                 "count", "computed", 28,
                 _src_computed("count of * cells across the report"))

    # Revenue share vs profit share, computed only where both legs are present for the
    # same period and unit — a share computed across mixed units would be nonsense.
    def _n(v):
        try:
            return float(str(v).split()[0].replace(",", ""))
        except (ValueError, IndexError):
            return None

    per = {}
    for _, r in rows.iterrows():
        p = str(r.get("period", "")).strip()
        f = str(r.get("field", "")).strip()
        if f in ("revenue", "profit"):
            per.setdefault(p, {}).setdefault(f, {})[str(r.get("item"))] = (
                _n(r.get("value")), str(r.get("unit", "")))
    for p, legs in sorted(per.items()):
        rev, pro = legs.get("revenue", {}), legs.get("profit", {})
        if len(rev) < 2 or len(pro) < 2:
            continue
        units = {u for _, u in list(rev.values()) + list(pro.values())}
        if len(units) > 1:
            pack.gap(6, f"{p}: segment figures mix units {units} — share not computed")
            continue
        tr = sum(v for v, _ in rev.values() if v)
        tp = sum(v for v, _ in pro.values() if v)
        if not tr or not tp:
            continue
        share = []
        for name in sorted(set(rev) | set(pro)):
            rv = (rev.get(name) or (None, ""))[0]
            pv = (pro.get(name) or (None, ""))[0]
            share.append({"segment": name,
                          "revenue_share_pct": round(100.0 * rv / tr, 1) if rv else None,
                          "profit_share_pct": round(100.0 * pv / tp, 1) if pv else None})
        pack.table(f"tbl.segment_share_{p}",
                   f"Revenue share vs profit share, {p}", 6,
                   ["segment", "revenue_share_pct", "profit_share_pct"], share,
                   _src_computed(f"each segment over the disclosed {p} total"),
                   note="Computed from the disclosed segment figures above. Where a "
                        "segment's profit share exceeds its revenue share, it earns "
                        "more than it books.")


def sec9_portfolio(pack: Pack, store: Store):
    rows = _struct(store, pack.isin, pack.symbol, "portfolio_unit")
    if rows.empty:
        pack.gap(9, "no portfolio units extracted")
        return
    recs = _pivot_struct(rows)
    pack.table("tbl.portfolio", "Brands, products and units, as disclosed", 9,
               ["item", "contribution_pct", "count", "note"], recs,
               _src_doc(rows.iloc[0]))
    pack.add("port.n_units", "Units disclosed", len(recs), "count", "extracted", 9,
             _src_doc(rows.iloc[0]))


def sec12_unit_deepdives(pack: Pack, store: Store):
    seg = _struct(store, pack.isin, pack.symbol, "segment")
    if seg.empty:
        pack.gap(12, "per-unit detail needs segment disclosures, which were not found")
        return
    pack.add("unit.n_segments", "Segments with disclosed figures",
             int(seg["item"].nunique()), "count", "extracted", 12, _src_doc(seg.iloc[0]))


def sec23_structural_risks(pack: Pack, store: Store):
    """The deck's risk register: the company's OWN named risk, its OWN stated
    mitigation, side by side."""
    rows = _struct(store, pack.isin, pack.symbol, "risk")
    if rows.empty:
        pack.gap(23, "no company-named risks extracted from the annual report")
        return
    recs = _pivot_struct(rows)
    for r in recs:
        r["status"] = "STATED" if r.get("mitigation") else "NO STATED MITIGATION"
    pack.table("tbl.structural_risks",
               "Risks the company names, and its stated response", 23,
               ["item", "description", "mitigation", "status"], recs,
               _src_doc(rows.iloc[0]),
               note="Both the risk and the response are the company's own words, from "
                    "its annual report. A row marked NO STATED MITIGATION is one the "
                    "filing raises without answering.")
    pack.add("risk.n_named", "Risks the company names", len(recs), "count",
             "extracted", 23, _src_doc(rows.iloc[0]))
    unmitigated = [r for r in recs if not r.get("mitigation")]
    if unmitigated:
        pack.add("risk.n_unmitigated", "Named without a stated mitigation",
                 len(unmitigated), "count", "extracted", 23, _src_doc(rows.iloc[0]))


def sec20_quote_spine(pack: Pack, store: Store):
    """Management's own words across consecutive calls, set against the said-vs-delivered
    record. Every quote was string-matched to its transcript before storage."""
    q = store.by_isin("mgmt_quotes.parquet", pack.isin, pack.symbol)
    if q.empty:
        pack.gap(20, "no management quotes — run extract_mgmt_quotes.py; the section "
                     "wants 4+ concalls for a claim-across-calls spine")
    else:
        q = q.sort_values("call_date", ascending=False)
        recs = [{"quarter": str(r.get("quarter", "")), "date": str(r.get("call_date", ""))[:10],
                 "speaker": str(r.get("speaker", "")), "topic": str(r.get("topic", "")),
                 "quote": str(r.get("quote", "")), "commitment": str(r.get("commitment", ""))}
                for _, r in q.iterrows()]
        pack.table("tbl.quotes", "What management said, verbatim", 20,
                   ["quarter", "date", "speaker", "topic", "quote", "commitment"], recs,
                   {"kind": "document", "doc_type": "concall",
                    "title": ", ".join(sorted(q["source_doc_id"].astype(str).unique()))},
                   note="Every quote was verified to appear verbatim in its transcript "
                        "before being stored; any that did not was discarded.")
        pack.add("quote.n", "Quotes on record", len(recs), "count", "extracted", 20,
                 _src_parquet(f"{IDX}/mgmt_quotes.parquet"))
        nq = int(q["quarter"].nunique())
        pack.add("quote.n_quarters", "Quarters covered", nq, "count", "extracted", 20,
                 _src_parquet(f"{IDX}/mgmt_quotes.parquet"))
        if nq < 4:
            pack.gap(20, f"quotes span {nq} quarter(s); a said-vs-delivered spine across "
                         f"calls wants 4+. Run backfill_company_docs.py for more concalls.")
        commitments = [r for r in recs if r["commitment"]]
        if commitments:
            pack.add("quote.n_commitments", "Checkable commitments made",
                     len(commitments), "count", "extracted", 20,
                     _src_parquet(f"{IDX}/mgmt_quotes.parquet"))

    # said-vs-delivered, from the pipelines that already compute it
    pead = store.by_isin("pead_flags.parquet", pack.isin, pack.symbol)
    if not pead.empty:
        rows = pead.sort_values("as_of").tail(10)
        pack.table("tbl.said_vs_delivered", "Guided versus actual", 20,
                   ["quarter", "metric", "guided_value", "actual_value", "verdict"],
                   [{"quarter": str(r.get("quarter")), "metric": str(r.get("metric")),
                     "guided_value": r.get("guided_value"),
                     "actual_value": r.get("actual_value"),
                     "verdict": str(r.get("verdict"))} for _, r in rows.iterrows()],
                   _src_parquet(f"{IDX}/pead_flags.parquet"))
    else:
        pack.gap(20, "no guided-vs-actual rows (pead_flags) for this company")
    mc = store.by_isin("mgmt_credibility.parquet", pack.isin, pack.symbol)
    if not mc.empty and "cred_score" in mc.columns:
        r = mc.sort_values("quarter").iloc[-1]
        pack.add("quote.cred_score", f"Management credibility score ({r.get('quarter')})",
                 pd.to_numeric(r.get("cred_score"), errors="coerce"), "/100", "computed",
                 20, _src_parquet(f"{IDX}/mgmt_credibility.parquet",
                                  f"pattern={r.get('pattern')}"))


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


# ------------------------------------------------- exchange + rating feeds ---
def _bse_code(store: Store, isin: str) -> str:
    """bse_code from the universe — the key the BSE announcement API needs."""
    try:
        fid = find_file(store.drive, store.folder(IDX), "company_universe.csv")
        if not fid:
            return ""
        u = pd.read_csv(io.BytesIO(download_bytes(store.drive, fid)))
        m = u[u["isin"].astype(str) == str(isin)]
        if m.empty:
            return ""
        v = pd.to_numeric(m.iloc[0].get("bse_code"), errors="coerce")
        return "" if pd.isna(v) else str(int(v))
    except Exception:
        return ""


def sec27_exchange_filings(pack: Pack, store: Store, live: bool = True):
    """Section 27 — recent BSE/NSE filings. This is the live exchange feed: orders,
    board actions, ratings, capacity. Reuses the fetchers already in
    company_deep_report.py rather than writing new API calls (CLAUDE.md rule 4)."""
    if not live:
        pack.gap(27, "exchange feed skipped (--no-live)")
        return
    recs = []
    code = _bse_code(store, pack.isin)
    try:
        from company_deep_report import bse_announcements, nse_announcements
    except Exception as e:
        pack.gap(27, f"exchange fetchers unavailable: {str(e)[:80]}")
        return

    def _parse_feed(blob, exchange: str) -> list[dict]:
        """Both fetchers return a FORMATTED STRING of "- YYYY-MM-DD | subject [cat]"
        lines (or a "DATA_MISSING ..." sentinel), not structured rows. Parse rather
        than re-implement the API call — the fetchers already carry the BSE header
        dance and NSE's cookie bootstrap."""
        out = []
        if not blob or not isinstance(blob, str):
            return out
        if blob.startswith("DATA_MISSING") or blob.startswith("No announcements"):
            pack.gap(27, f"{exchange}: {blob[:120]}")
            return out
        for line in blob.splitlines():
            line = line.strip()
            if not line.startswith("- "):
                continue
            body = line[2:]
            date, _, rest = body.partition(" | ")
            cat = ""
            if rest.endswith("]") and " [" in rest:
                rest, _, cat = rest.rpartition(" [")
                cat = cat.rstrip("]")
            # BSE emits ISO (2026-05-30); NSE emits DD-MMM-YYYY (30-May-2026).
            # Truncating to 10 chars chopped NSE's year, and sorting two formats
            # together is meaningless — normalise to ISO and keep what fails as-is.
            raw = date.strip()
            # BSE is already ISO; only NSE's DD-MMM-YYYY needs dayfirst. Passing
            # dayfirst on an ISO string is ambiguous and warns on every row.
            iso = pd.to_datetime(
                raw, errors="coerce",
                dayfirst=not re.match(r"^\d{4}-\d{2}-\d{2}", raw))
            out.append({"exchange": exchange,
                        "date": raw if pd.isna(iso) else iso.strftime("%Y-%m-%d"),
                        "category": cat[:60], "headline": rest.strip()[:220]})
        return out

    if code:
        try:
            recs += _parse_feed(bse_announcements(code, limit=25), "BSE")
        except Exception as e:
            pack.gap(27, f"BSE announcement fetch failed: {str(e)[:90]}")
    else:
        pack.gap(27, "no bse_code in the universe for this company — BSE feed skipped")
    try:
        recs += _parse_feed(nse_announcements(pack.symbol, limit=15), "NSE")
    except Exception as e:
        pack.gap(27, f"NSE announcement fetch failed: {str(e)[:90]}")

    if not recs:
        pack.gap(27, "no exchange filings returned")
        return
    recs.sort(key=lambda r: r["date"], reverse=True)
    pack.table("tbl.filings", "Recent exchange filings", 27,
               ["date", "exchange", "category", "headline"], recs,
               {"kind": "api", "table": "BSE Direct / NSE corporate announcements",
                "note": f"fetched live {datetime.now(timezone.utc).date()}"},
               note="Filed events only, as published by the exchange. Headlines are "
                    "not interpreted here.")
    pack.add("filings.n_recent", "Filings retrieved", len(recs), "count", "reported",
             27, {"kind": "api", "table": "BSE Direct / NSE"})


def sec14_15_research(pack: Pack, store: Store):
    """Sections 14 and 15 — independent broker research, from research_map.parquet.

    Two passes, in the order the user asked for: first what has been written about THIS
    company, then what has been written about its SECTOR. The sector is taken from the
    company's own classification, so the sector view is the one its peers sit in.
    """
    rm = store.parquet(IDX, "research_map.parquet")
    if rm.empty:
        pack.gap(14, "research_map.parquet not built — run build_research_map.py "
                     "(it also runs at the end of run_daily_research.bat)")
        pack.gap(15, "research_map.parquet not built")
        return

    # ---- 14: company coverage -------------------------------------------------
    mine = rm[(rm["scope"] == "company") &
              (rm["isin"].astype(str) == str(pack.isin))]
    if mine.empty:
        pack.gap(14, "no broker research maps to this company. Coverage may exist "
                     "under a brand name — research_lookup.py --alias searches those.")
    else:
        mine = mine.sort_values("doc_date", ascending=False)
        rows = [{"date": str(r.get("doc_date", ""))[:10],
                 "source": str(r.get("source", ""))[:60],
                 "doc_kind": str(r.get("doc_kind", "")),
                 "named_as": str(r.get("matched_name_raw", ""))[:60],
                 "matched_by": str(r.get("match_key", ""))}
                for _, r in mine.head(25).iterrows()]
        pack.table("tbl.research_company", "Broker research naming this company", 14,
                   ["date", "source", "doc_kind", "named_as", "matched_by"], rows,
                   _src_parquet(f"{IDX}/research_map.parquet"),
                   note="`matched_by` shows which key linked the document to this "
                        "company — isin and symbol are exact; name_partial is a "
                        "looser brand/name match and is worth eyeballing.")
        pack.add("research.n_docs", "Research documents naming this company",
                 int(mine["research_n"].nunique()), "count", "extracted", 14,
                 _src_parquet(f"{IDX}/research_map.parquet"))
        notes = mine[mine["doc_kind"].isin(["analyst_note", "company_update"])]
        if not notes.empty:
            pack.add("research.n_company_notes", "Company-specific analyst notes",
                     int(notes["research_n"].nunique()), "count", "extracted", 14,
                     _src_parquet(f"{IDX}/research_map.parquet"))
        srcs = mine["source"].astype(str).replace("", pd.NA).dropna().unique()
        if len(srcs):
            pack.add("research.n_houses", "Distinct research houses", len(srcs),
                     "count", "extracted", 14,
                     _src_parquet(f"{IDX}/research_map.parquet",
                                  f"e.g. {', '.join(sorted(srcs)[:4])}"))

    # ---- 15: the sector this company sits in ----------------------------------
    facts = store.by_isin("company_facts.parquet", pack.isin, pack.symbol)
    sector = ""
    if not facts.empty:
        r0 = facts.iloc[0]
        sector = str(r0.get("macro_sector") or "").strip()
        if not sector:
            # macro_sector comes from the NSE index lists and covers only ~755
            # companies, so most names would get no sector view at all. Fall back to
            # the finer Gemini labels, mapped onto the SAME controlled vocabulary the
            # research map uses — otherwise the two sides cannot be joined.
            try:
                import build_research_map as BRM
                sector = BRM.guess_sector(r0.get("sector"), r0.get("subsector"),
                                          r0.get("peer_group"))
            except Exception:
                sector = ""
    if not sector:
        pack.gap(15, "company has no sector that maps onto the research vocabulary "
                     "(no macro_sector, and its sector/peer_group labels did not "
                     "resolve), so sector research cannot be selected")
        return
    pack.add("research.sector", "Sector used for the research view", sector, "label",
             "classification", 15, _src_parquet(f"{IDX}/company_facts.parquet"))
    sec = rm[(rm["scope"].isin(["sector", "macro", "policy"])) &
             (rm["sector"].astype(str) == sector)]
    if sec.empty:
        pack.gap(15, f"no sector/macro research mapped to '{sector}'. A document is "
                     f"only labelled with a sector when the companies it names agree "
                     f"on one — ambiguous notes are left unlabelled rather than guessed.")
        return
    sec = sec.sort_values("doc_date", ascending=False).drop_duplicates("research_n")
    rows = [{"date": str(r.get("doc_date", ""))[:10],
             "source": str(r.get("source", ""))[:60],
             "scope": str(r.get("scope", "")),
             "doc_kind": str(r.get("doc_kind", "")),
             "title": str(r.get("file_name", ""))[:80]}
            for _, r in sec.head(20).iterrows()]
    pack.table("tbl.research_sector", f"Research on {sector}", 15,
               ["date", "source", "scope", "doc_kind", "title"], rows,
               _src_parquet(f"{IDX}/research_map.parquet"),
               note="Sector, macro and policy notes covering this company's sector. "
                    "The sector label is inferred from the companies a document names, "
                    "not from its file name or publisher.")
    pack.add("research.n_sector_docs", f"Research documents on {sector}",
             int(sec["research_n"].nunique()), "count", "extracted", 15,
             _src_parquet(f"{IDX}/research_map.parquet"))


def sec7_ratings(pack: Pack, store: Store):
    """Credit-rating actions — the agencies' own view, which is independent of the
    company's framing."""
    r = store.by_isin("ratings.parquet", pack.isin, pack.symbol)
    if r.empty:
        pack.gap(7, "no credit-rating rows; CRISIL/ICRA/CARE rationales may still be in "
                    "the source bundle for the auditor even when not tabulated here")
        return
    cols = [c for c in ("agency", "rating", "outlook", "action", "instrument",
                        "rated_amount", "date", "fy_year") if c in r.columns]
    rows = r.sort_values(cols[-1] if cols else r.columns[0],
                         ascending=False).head(12)
    pack.table("tbl.ratings", "Credit-rating actions", 7, cols,
               [{c: str(x.get(c, "")) for c in cols} for _, x in rows.iterrows()],
               _src_parquet(f"{IDX}/ratings.parquet"))
    pack.add("rating.n", "Rating records", len(r), "count", "extracted", 7,
             _src_parquet(f"{IDX}/ratings.parquet"))


# --------------------------------------------------------------------- main ---
def build(store: Store, token: str, live: bool = True) -> Pack | None:
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
    # deterministic core (statements + computed tables)
    sec2_one_pager(pack, store)
    sec18_financials(pack, store, st)
    sec5_scale_history(pack, store, st)
    sec8_stable_series(pack, store, st)
    lev = sec19_operating_leverage(pack, store, st)
    sec22_sensitivity_grid(pack, store, st, lev)
    sec21_peers(pack, store)
    sec24_risk_register(pack, store)
    # extracted from filings (N7/N8)
    sec1_history(pack, store)
    sec3_entity_map(pack, store)
    sec4_management(pack, store)
    sec6_segment_economics(pack, store)
    sec7_ratings(pack, store)
    sec9_portfolio(pack, store)
    sec12_unit_deepdives(pack, store)
    sec14_15_research(pack, store)
    sec20_quote_spine(pack, store)
    sec23_structural_risks(pack, store)
    # live exchange feed
    sec27_exchange_filings(pack, store, live=live)
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
    ap.add_argument("--no-live", action="store_true",
                    help="skip the live BSE/NSE announcement fetch")
    a = ap.parse_args()

    tokens = list(a.names) + list(a.isin)
    if not tokens:
        ap.error("give at least one --names/--isin token")

    store = Store()
    packs = []
    for t in tokens:
        log(f"resolving '{t}'")
        p = build(store, t, live=not a.no_live)
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
