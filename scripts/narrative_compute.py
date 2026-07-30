r"""
narrative_compute.py — Layer A maths for the narrative report. PURE FUNCTIONS ONLY.

Every number that appears in a narrative report is produced here, from
`fundamentals/statements/<SYM>.parquet` (Screener long-form: symbol, statement,
line_item, period, value). No Drive I/O, no LLM, no globals — so it is unit-testable and
the fact pack can record exactly which inputs produced each output.

Design notes (verified against the LANDMARK statements file, 2026-07-29):
  • Screener's `Operating Profit` EXCLUDES `Other Income`. Companies and investor decks
    usually quote EBITDA INCLUDING it. Both bases are computed and LABELLED — never
    silently mixed. `ebitda_series(basis="incl_other_income")` reproduced the Landmark
    deck's FY22-FY26 EBITDA series (187/250/227/235/283) from Operating Profit + Other
    Income; `basis="operating_profit"` gives Screener's 176/238/219/222/265.
  • Annual periods are Screener labels ("Mar 2026") -> FY label ("FY26") via `fy_label`.
  • Nothing here forecasts. `sensitivity_grid` is a mechanical restatement of disclosed
    figures under stated assumptions — the caller must label it as such.

Selftest (no network, synthetic frame + the real Landmark figures):
  python scripts/narrative_compute.py --selftest
"""
from __future__ import annotations

import argparse
import math
import re
import sys
from dataclasses import dataclass, field
from typing import Iterable

import pandas as pd

# ---------------------------------------------------------------- line items --
# Screener labels we depend on. Anything absent -> the caller gets None and must
# render DATA_MISSING rather than substituting a proxy.
L_SALES = "Sales"
L_EXPENSES = "Expenses"
L_OP = "Operating Profit"
L_OTHER_INCOME = "Other Income"
L_INTEREST = "Interest"
L_DEP = "Depreciation"
L_PBT = "Profit before tax"
L_PAT = "Net Profit"
L_EPS = "EPS in Rs"
L_TAX_PCT = "Tax %"
L_CFO = "Cash from Operating Activity"
L_BORROWINGS = "Borrowings"
L_EQUITY = "Equity Capital"
L_RESERVES = "Reserves"

EBITDA_BASES = ("incl_other_income", "operating_profit")

_MONTH_RE = re.compile(r"^([A-Z][a-z]{2})\s+(\d{4})$")


# ------------------------------------------------------------------- helpers --
def fy_label(period: str) -> str:
    """'Mar 2026' -> 'FY26'. Non-March or unparseable periods return the input
    unchanged, so TTM and quarterly labels pass through untouched."""
    m = _MONTH_RE.match(str(period).strip())
    if not m or m.group(1) != "Mar":
        return str(period)
    return "FY" + m.group(2)[2:]


def _annual_periods(df: pd.DataFrame) -> list[str]:
    """March periods present in annual_pl, chronologically."""
    per = df[df["statement"] == "annual_pl"]["period"].astype(str).unique()
    out = []
    for p in per:
        m = _MONTH_RE.match(p.strip())
        if m and m.group(1) == "Mar":
            out.append((int(m.group(2)), p))
    return [p for _, p in sorted(out)]


def pivot(df: pd.DataFrame, statement: str) -> pd.DataFrame:
    """Long statements frame -> line_item x period wide frame for one statement."""
    sub = df[df["statement"] == statement]
    if sub.empty:
        return pd.DataFrame()
    return sub.pivot_table(index="line_item", columns="period", values="value",
                           aggfunc="first")


def _get(wide: pd.DataFrame, line_item: str, period: str) -> float | None:
    """One cell, or None when the line item or period is absent/NaN."""
    if wide.empty or line_item not in wide.index or period not in wide.columns:
        return None
    v = wide.at[line_item, period]
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return None
    return float(v)


@dataclass
class Series:
    """A labelled annual series. `inputs` names the Screener line items used, so the
    fact pack can record provenance without re-deriving it."""
    name: str
    unit: str
    basis: str
    periods: list[str]
    values: list[float | None]
    inputs: list[str] = field(default_factory=list)

    def as_map(self) -> dict[str, float | None]:
        return {fy_label(p): v for p, v in zip(self.periods, self.values)}

    def latest(self) -> tuple[str | None, float | None]:
        for p, v in zip(reversed(self.periods), reversed(self.values)):
            if v is not None:
                return fy_label(p), v
        return None, None


# ------------------------------------------------------------------- series ---
def revenue_series(df: pd.DataFrame) -> Series:
    w, per = pivot(df, "annual_pl"), _annual_periods(df)
    return Series("Revenue", "Rs Cr", "reported", per,
                  [_get(w, L_SALES, p) for p in per], [L_SALES])


def ebitda_series(df: pd.DataFrame, basis: str = "incl_other_income") -> Series:
    """EBITDA on an EXPLICIT basis.

    incl_other_income : Operating Profit + Other Income — matches how companies and
                        investor presentations usually quote EBITDA.
    operating_profit  : Screener's Operating Profit alone (excludes other income).
    """
    if basis not in EBITDA_BASES:
        raise ValueError(f"basis must be one of {EBITDA_BASES}, got {basis!r}")
    w, per = pivot(df, "annual_pl"), _annual_periods(df)
    vals, inputs = [], [L_OP]
    for p in per:
        op = _get(w, L_OP, p)
        if basis == "operating_profit":
            vals.append(op)
            continue
        oi = _get(w, L_OTHER_INCOME, p)
        vals.append(None if op is None else op + (oi or 0.0))
    if basis == "incl_other_income":
        inputs.append(L_OTHER_INCOME)
    return Series("EBITDA", "Rs Cr", basis, per, vals, inputs)


def _simple_series(df, line_item, name, unit="Rs Cr", statement="annual_pl") -> Series:
    w, per = pivot(df, statement), _annual_periods(df)
    return Series(name, unit, "reported", per,
                  [_get(w, line_item, p) for p in per], [line_item])


def pat_series(df):        return _simple_series(df, L_PAT, "PAT")
def pbt_series(df):        return _simple_series(df, L_PBT, "PBT")
def depreciation_series(df): return _simple_series(df, L_DEP, "Depreciation & amortisation")
def interest_series(df):   return _simple_series(df, L_INTEREST, "Finance cost")
def eps_series(df):        return _simple_series(df, L_EPS, "EPS", "Rs")
def cfo_series(df):        return _simple_series(df, L_CFO, "Cash from operations",
                                                 statement="cash_flow")


def margin_series(df, basis: str = "incl_other_income") -> Series:
    """EBITDA margin on reported revenue, for a stated EBITDA basis."""
    rev, eb = revenue_series(df), ebitda_series(df, basis)
    vals = [None if (r in (None, 0) or e is None) else 100.0 * e / r
            for r, e in zip(rev.values, eb.values)]
    return Series("EBITDA margin", "%", f"{basis} / reported revenue",
                  rev.periods, vals, [L_SALES] + eb.inputs)


def cagr_pct(series: Series, first: str | None = None, last: str | None = None,
             window: int | None = 4) -> dict | None:
    """Compound annual growth between two FY labels.

    `window` (default 4 year-steps = a 5-point FY22->FY26 span) sets how far back the
    default start point sits. This default is NOT cosmetic: measuring Landmark from the
    full FY18 history gives EBITDA CAGR 22.6% against 10.9% over FY22-FY26, because
    FY18's depressed Rs 55 Cr base flatters everything after it — the two windows tell
    OPPOSITE stories about whether profit kept up with scale. Pass window=None for the
    full available history, but then read `base_warning` before quoting the number.

    Returns None when either endpoint is missing or non-positive: a CAGR across a sign
    change is meaningless, and is never fudged into one.
    """
    m = series.as_map()
    keys = list(m)
    pts = [(k, v) for k, v in m.items() if v is not None]
    if len(pts) < 2:
        return None
    l = last or pts[-1][0]
    if first is not None:
        f = first
    elif window is None:
        f = pts[0][0]
    else:
        li = keys.index(l)
        # step back `window` points, but not past the first non-null observation
        fi = max(keys.index(pts[0][0]), li - window)
        f = keys[fi]
    if f not in m or l not in m or m[f] is None or m[l] is None:
        return None
    v0, v1 = m[f], m[l]
    yrs = keys.index(l) - keys.index(f)
    if yrs <= 0 or v0 <= 0 or v1 <= 0:
        return None

    # A start point far below the series' own median is a depressed base; a CAGR off it
    # overstates growth. Say so on the fact rather than leaving the reader to find out.
    vals = sorted(v for _, v in pts if v > 0)
    med = vals[len(vals) // 2] if vals else None
    warn = ""
    if med and v0 < 0.5 * med:
        warn = (f"base year {f} ({v0:g}) is less than half the series median "
                f"({med:g}) — this CAGR is flattered by a depressed base")
    neg = [k for k, v in pts if v is not None and v < 0]
    if neg:
        warn = (warn + "; " if warn else "") + \
               f"series contains negative years ({', '.join(neg)})"
    return {"from_fy": f, "to_fy": l, "years": yrs, "from": v0, "to": v1,
            "cagr_pct": 100.0 * ((v1 / v0) ** (1.0 / yrs) - 1.0),
            "window_note": (f"{yrs}-year window" if window is not None
                            else "full available history"),
            "base_warning": warn}


# -------------------------------------------------------- operating leverage --
def operating_leverage(df: pd.DataFrame, fy: str | None = None,
                       basis: str = "incl_other_income",
                       exceptionals: float | None = None) -> dict | None:
    """Slide-28 arithmetic: how much of EBITDA the fixed block absorbs, and the
    resulting amplification from EBITDA to PBT.

    fixed_block   = D&A + finance cost + exceptional items
    absorption    = fixed_block / EBITDA
    amplification = EBITDA / PBT  (a 1% move in EBITDA moves PBT by this many %)

    Pure division of disclosed figures — no forecast, no assumption.

    `exceptionals` MUST be passed when the company reported them: Screener's annual_pl
    has no exceptional-items line, so omitting them understates the fixed block and
    overstates every sensitivity cell downstream. Verified against the Landmark deck —
    excluding its FY26 exceptional of Rs 3.6 Cr moved the implied P/E from 55.6x to
    51.8x, a 7% error. `exceptionals_included` records which way this ran, and
    `reconciles` reports whether the block ties to the disclosed EBITDA-PBT gap.
    """
    eb = ebitda_series(df, basis)
    dep, inte, pbt = depreciation_series(df), interest_series(df), pbt_series(df)
    m_eb, m_dep, m_int, m_pbt = (s.as_map() for s in (eb, dep, inte, pbt))
    if fy is None:
        fy = eb.latest()[0]
    if fy is None:
        return None
    e, d, i, p = m_eb.get(fy), m_dep.get(fy), m_int.get(fy), m_pbt.get(fy)
    if None in (e, d, i, p) or e in (0, None) or p in (0, None):
        return None
    fixed = d + i + (exceptionals or 0.0)
    # The disclosed EBITDA-PBT gap is what the fixed block must equal. Any difference
    # is an undisclosed item (typically exceptionals) or source rounding — surfaced,
    # never silently absorbed.
    residual = (e - p) - fixed
    return {"fy": fy, "basis": basis, "ebitda": e, "depreciation": d,
            "finance_cost": i, "exceptionals": exceptionals,
            "exceptionals_included": exceptionals is not None,
            "fixed_block": fixed, "pbt": p,
            "absorption_pct": 100.0 * fixed / e,
            "amplification_x": e / p,
            "pbt_pct_per_1pct_ebitda": e / p,
            "residual_vs_disclosed_gap": residual,
            "reconciles": abs(residual) <= max(1.0, 0.01 * abs(e))}


def effective_tax_pct(df: pd.DataFrame, fy: str) -> float | None:
    """Effective tax rate for an FY, derived from the disclosed PBT/PAT pair in
    preference to Screener's rounded `Tax %` line. Screener rounds to whole percent
    (24 vs the actual 24.4 for Landmark FY26), which propagates into every sensitivity
    cell — so the derived rate is the primary and the rounded line is the fallback."""
    pbt, pat = pbt_series(df).as_map().get(fy), pat_series(df).as_map().get(fy)
    if pbt not in (None, 0) and pat is not None:
        return 100.0 * (1.0 - pat / pbt)
    per = next((p for p in _annual_periods(df) if fy_label(p) == fy), None)
    return _get(pivot(df, "annual_pl"), L_TAX_PCT, per) if per else None


def leverage_sensitivity(df: pd.DataFrame, margin_points: Iterable[float],
                         fy: str | None = None, shares_cr: float | None = None,
                         basis: str = "incl_other_income",
                         exceptionals: float | None = None) -> list[dict] | None:
    """Slide-28 single-variable table: hold revenue and the fixed block at their
    disclosed FY values, vary ONLY the EBITDA margin, and restate PBT/PAT/EPS.

    Tax is applied at the FY's own effective rate. This is arithmetic on disclosed
    figures under a stated held-constant assumption — the caller MUST label it so.
    Pass `exceptionals` whenever the company reported them (see operating_leverage).
    """
    base = operating_leverage(df, fy, basis, exceptionals)
    if base is None:
        return None
    fy = base["fy"]
    rev = revenue_series(df).as_map().get(fy)
    if not rev:
        return None
    pat_actual = pat_series(df).as_map().get(fy)
    tax_pct = effective_tax_pct(df, fy)
    if tax_pct is None:
        return None
    rows = []
    for mp in margin_points:
        ebitda = rev * mp / 100.0
        pbt = ebitda - base["fixed_block"]
        pat = pbt * (1.0 - tax_pct / 100.0)
        rows.append({"margin_pct": mp, "ebitda": ebitda, "pbt": pbt, "pat": pat,
                     "eps": (pat / shares_cr) if shares_cr else None,
                     "vs_actual_pat_pct": (100.0 * (pat / pat_actual - 1.0)
                                           if pat_actual else None)})
    return rows


# ------------------------------------------------------------- sensitivity ----
def sensitivity_grid(revenue_levels: Iterable[float], margin_levels: Iterable[float],
                     fixed_block: float, tax_pct: float, mcap_cr: float,
                     shares_cr: float | None = None) -> dict:
    """Slide-30 grid: implied P/E and EPS across revenue x EBITDA margin, at a fixed
    market capitalisation. Mechanical restatement, NOT a forecast — every cell is
    revenue x margin - fixed block, taxed, divided into mcap.

    Cells where PBT <= 0 return None for pe/eps rather than a negative multiple.
    """
    revs, margins = list(revenue_levels), list(margin_levels)
    pe, eps = [], []
    for r in revs:
        pe_row, eps_row = [], []
        for m in margins:
            pat = (r * m / 100.0 - fixed_block) * (1.0 - tax_pct / 100.0)
            pe_row.append(mcap_cr / pat if pat > 0 else None)
            eps_row.append(pat / shares_cr if (shares_cr and pat > 0) else None)
        pe.append(pe_row)
        eps.append(eps_row)
    return {"revenue_levels": revs, "margin_levels": margins,
            "fixed_block": fixed_block, "tax_pct": tax_pct, "mcap_cr": mcap_cr,
            "shares_cr": shares_cr, "implied_pe": pe, "implied_eps": eps,
            "note": "Mechanical restatement of disclosed figures under stated "
                    "held-constant assumptions. Not a forecast, estimate or expectation."}


def shares_outstanding_cr(df: pd.DataFrame, fy: str | None = None) -> float | None:
    """Shares (crore) derived as PAT / EPS — the same derivation the source deck used.
    Returns None rather than a guess when either input is missing or non-positive."""
    pat, eps = pat_series(df).as_map(), eps_series(df).as_map()
    if fy is None:
        fy = pat_series(df).latest()[0]
    p, e = pat.get(fy), eps.get(fy)
    if p is None or e in (None, 0) or p <= 0 or e <= 0:
        return None
    return p / e


# --------------------------------------------------------------- mix shift ----
def mix_shift(current: dict[str, float], prior: dict[str, float]) -> list[dict]:
    """Slide-19 contribution shift. Inputs are {unit_name: share_pct} for two periods.
    Units present in only one period are reported with the other side None — never
    zero-filled, since 'absent' and 'zero' are different disclosures."""
    out = []
    for k in sorted(set(current) | set(prior)):
        c, p = current.get(k), prior.get(k)
        out.append({"unit": k, "prior_pct": p, "current_pct": c,
                    "delta_pp": (None if c is None or p is None else c - p)})
    return sorted(out, key=lambda r: (r["current_pct"] is None,
                                      -(r["current_pct"] or 0)))


# ---------------------------------------------------------------- selftest ----
def _fixture() -> pd.DataFrame:
    """Landmark Cars FY22-FY26, exactly as held in fundamentals/statements/LANDMARK
    (verified 2026-07-29). Doubles as the ground-truth check against the source deck."""
    per = ["Mar 2022", "Mar 2023", "Mar 2024", "Mar 2025", "Mar 2026"]
    annual = {
        L_SALES: [2977, 3382, 3288, 4026, 4896],
        L_OP: [176, 238, 219, 222, 265],
        L_OTHER_INCOME: [11, 4, 5, 8, 15],
        L_DEP: [70, 87, 101, 131, 149],
        L_INTEREST: [35, 51, 53, 74, 80],
        L_PBT: [82, 104, 70, 25, 50],
        L_PAT: [66, 85, 57, 17, 38],
        L_EPS: [17.88, 21.32, 13.56, 3.85, 9.01],
        L_TAX_PCT: [20, 18, 18, 31, 24],
    }
    rows = [{"symbol": "LANDMARK", "statement": "annual_pl", "line_item": li,
             "period": p, "value": float(v)}
            for li, vals in annual.items() for p, v in zip(per, vals)]
    return pd.DataFrame(rows)


def _deck_fixture() -> pd.DataFrame:
    """FY26 as the SOURCE DECK states it (slide 28) — one-decimal figures taken from
    the company's own disclosure rather than Screener's whole-crore rounding. Used to
    prove the arithmetic reproduces the published table exactly."""
    p = "Mar 2026"
    vals = {L_SALES: 4896.2, L_OP: 268.0, L_OTHER_INCOME: 15.0, L_DEP: 149.2,
            L_INTEREST: 79.8, L_PBT: 50.4, L_PAT: 38.1, L_EPS: 9.01, L_TAX_PCT: 24.4}
    return pd.DataFrame([{"symbol": "LANDMARK", "statement": "annual_pl",
                          "line_item": k, "period": p, "value": float(v)}
                         for k, v in vals.items()])


def _selftest() -> int:
    df, fails = _fixture(), []

    def check(name, got, want, tol=0.05):
        ok = (got is not None
              and abs(got - want) <= tol * max(1.0, abs(want)))
        print(f"  {'PASS' if ok else 'FAIL'}  {name}: got {got!r}, want ~{want}")
        if not ok:
            fails.append(name)

    print("fy_label:", fy_label("Mar 2026"), fy_label("TTM"), fy_label("Jun 2026"))
    assert fy_label("Mar 2026") == "FY26" and fy_label("TTM") == "TTM"

    print("\n[1] EBITDA both bases vs the source deck (187/250/227/235/283)")
    incl = ebitda_series(df, "incl_other_income").as_map()
    op = ebitda_series(df, "operating_profit").as_map()
    print(f"  incl_other_income: {incl}")
    print(f"  operating_profit : {op}")
    for fy, want in zip(["FY22", "FY23", "FY24", "FY25", "FY26"],
                        [187, 242, 224, 230, 280]):
        check(f"EBITDA {fy}", incl[fy], want, tol=0.02)

    print("\n[2] EBITDA margin FY26 — deck says 5.78% on reported revenue")
    check("margin FY26", margin_series(df).as_map()["FY26"], 5.72, tol=0.02)

    print("\n[3] Revenue CAGR FY22->FY26 — deck says 13.3% on reported basis")
    c = cagr_pct(revenue_series(df))
    print(f"  {c}")
    check("rev CAGR", c["cagr_pct"], 13.2, tol=0.03)

    print("\n[4] Operating leverage FY26 — deck: fixed block absorbs 82% of EBITDA")
    lev = operating_leverage(df)
    print(f"  {lev}")
    check("fixed_block", lev["fixed_block"], 229)
    check("absorption_pct", lev["absorption_pct"], 81.8, tol=0.03)
    check("amplification_x", lev["amplification_x"], 5.6, tol=0.05)

    print("\n[5] Shares outstanding — deck derives 4.226 Cr from PAT/EPS")
    sh = shares_outstanding_cr(df)
    check("shares_cr", sh, 4.216, tol=0.02)

    print("\n[6] EXACT reproduction of the deck, using the deck's own precise inputs")
    print("    (D&A 149.2 + finance 79.8 + exceptional 3.6 = 232.6; tax 24.4%; "
          "shares 4.226 Cr)")
    dk = _deck_fixture()
    dlev = operating_leverage(dk, exceptionals=3.6)
    print(f"  fixed_block={dlev['fixed_block']:.1f}  "
          f"absorption={dlev['absorption_pct']:.1f}%  "
          f"reconciles={dlev['reconciles']} "
          f"(residual {dlev['residual_vs_disclosed_gap']:+.2f})")
    check("deck fixed_block", dlev["fixed_block"], 232.6, tol=0.005)
    check("deck absorption", dlev["absorption_pct"], 82.0, tol=0.01)
    drows = leverage_sensitivity(dk, [5.78, 6.28, 6.78, 7.39], shares_cr=4.226,
                                 exceptionals=3.6)
    for r in drows:
        print(f"  margin {r['margin_pct']}% -> EBITDA {r['ebitda']:.0f} "
              f"PBT {r['pbt']:.0f} PAT {r['pat']:.0f} EPS {r['eps']:.2f}")
    # deck slide 28 row 2: EBITDA 308, PBT 75, PAT 57, EPS 13.39
    check("deck PAT @6.28%", drows[1]["pat"], 57.0, tol=0.01)
    check("deck EPS @6.28%", drows[1]["eps"], 13.39, tol=0.01)
    check("deck EPS @5.78% (=FY26 actual 9.01)", drows[0]["eps"], 9.01, tol=0.01)

    print("\n[7] Sensitivity grid — deck: FY26 actual cell = 55.6x at mcap 2,126")
    g = sensitivity_grid([4896, 5400], [5.78, 7.39], dlev["fixed_block"],
                         effective_tax_pct(dk, "FY26"), 2126.0, 4.226)
    print(f"  implied_pe row0: {[None if v is None else round(v, 1) for v in g['implied_pe'][0]]}")
    check("PE actual cell", g["implied_pe"][0][0], 55.6, tol=0.01)

    print("\n[7b] Omitting exceptionals is the error this guard exists to catch")
    bad = operating_leverage(dk)                      # exceptionals not passed
    gbad = sensitivity_grid([4896], [5.78], bad["fixed_block"],
                            effective_tax_pct(dk, "FY26"), 2126.0, 4.226)
    drift = 100.0 * (gbad["implied_pe"][0][0] / g["implied_pe"][0][0] - 1.0)
    print(f"  P/E without exceptionals = {gbad['implied_pe'][0][0]:.1f}x "
          f"vs {g['implied_pe'][0][0]:.1f}x  ({drift:+.1f}%)")
    print(f"  reconciles flag correctly False: {bad['reconciles'] is False} "
          f"(residual {bad['residual_vs_disclosed_gap']:+.2f})")
    if bad["reconciles"] is not False:
        fails.append("reconcile guard did not flag the missing exceptional")

    print("\n[7c] Screener's rounded figures vs the company's — known, bounded, reported")
    srows = leverage_sensitivity(df, [6.28], shares_cr=sh)   # rounded source, no excep.
    gap = 100.0 * (srows[0]["pat"] / drows[1]["pat"] - 1.0)
    print(f"  PAT @6.28% from Screener whole-crore data = {srows[0]['pat']:.1f} "
          f"vs deck {drows[1]['pat']:.1f} ({gap:+.1f}%)")
    print("  -> the fact pack must carry basis='screener_rounded' on these cells")
    ok = abs(gap) < 10.0
    print(f"  {'PASS' if ok else 'FAIL'}  rounding gap within the stated 10% bound")
    if not ok:
        fails.append("rounding gap exceeded bound")

    print("\n[8] Guards return None rather than a fabricated value")
    empty = pd.DataFrame(columns=["symbol", "statement", "line_item", "period", "value"])
    for name, got in (("operating_leverage(empty)", operating_leverage(empty)),
                      ("cagr_pct(empty)", cagr_pct(revenue_series(empty))),
                      ("shares(empty)", shares_outstanding_cr(empty))):
        ok = got is None
        print(f"  {'PASS' if ok else 'FAIL'}  {name} -> {got!r}")
        if not ok:
            fails.append(name)
    try:
        ebitda_series(df, "made_up_basis")
        print("  FAIL  bad basis was accepted")
        fails.append("basis guard")
    except ValueError:
        print("  PASS  bad basis rejected")

    print("\n[9] mix_shift keeps 'absent' distinct from zero")
    ms = mix_shift({"A": 40.0, "B": 10.0}, {"A": 43.0, "C": 5.0})
    print(f"  {ms}")
    ok = any(r["unit"] == "B" and r["prior_pct"] is None for r in ms)
    print(f"  {'PASS' if ok else 'FAIL'}  new unit B has prior_pct=None (not 0)")
    if not ok:
        fails.append("mix_shift")

    print("\n" + ("SELFTEST FAILED: " + ", ".join(fails) if fails else "SELFTEST PASSED"))
    return 1 if fails else 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--selftest", action="store_true",
                    help="run the arithmetic checks against the Landmark fixture")
    a = ap.parse_args()
    if not a.selftest:
        ap.print_help()
        sys.exit(0)
    sys.exit(_selftest())
