r"""
narrative_dynamics.py — the merged company report's FINANCIAL DYNAMICS facts (P3).

User 2026-10-05 / 2026-10-17: chapter 4 must show what the income statement, balance sheet
and cash flow look like; chapter 5 whether growth is funded internally (tie the three
statements together); chapter 11 the current valuation AND bear / base / bull scenarios.
Every number here is COMPUTED IN CODE from the statements, prices and guidance tables; the
writer only explains them (the number gate rejects anything else).

  deeper_statements()   Screener live, the view with MORE annual years, when that is more than
                        the stored file has (Ather: stored FY26 only, Screener FY22-FY26). In
                        memory only — the database is NOT changed (user: highlight, don't fix).
  sec30_balance_cash()  10-year balance sheet + cash flow series            -> chapter 4
  sec31_cash_bridge()   5y / 10y: profit -> cash -> capex, dividends, debt   -> chapter 5
  sec33_pe_history()    year-end price / EPS for the last 5 years           -> chapter 11
  sec32_scenarios()     bear / base / bull, 3 years, from the company's own record -> ch 11

Scenario method (user "go", 2026-10-05): revenue growth bear = weakest 3-year stretch of the
last 10 years, base = 5-year growth, bull = management's latest explicit revenue-growth
guidance (guidance_tracker) else the best 3-year stretch; net margin = 5-year low / median /
high; exit P/E = own 5-year low / median / high. Implied market cap in 3 years and the yearly
return from today follow by arithmetic. Not a forecast, not a target.
"""
from __future__ import annotations

import math
import re
import os
import statistics
from datetime import datetime, timezone

import pandas as pd

import narrative_compute as NC

SEC_BSCF, SEC_CASH, SEC_SCN, SEC_PE = 30, 31, 32, 33
HORIZON_YEARS = 3


def _comp(note: str) -> dict:
    return {"kind": "computed", "note": note}


def _stmt_src(symbol: str, items, fetched=None) -> dict:
    return {"kind": "statements", "table": f"fundamentals/statements/{symbol}.parquet",
            "line_items": list(items), "period": "all", "fetched_at": fetched}


def _fys(st: pd.DataFrame, n: int | None = None) -> list[str]:
    per = NC._annual_periods(st)
    return per[-n:] if n else per


# ------------------------------------------------------------------ live pull --
def deeper_statements(symbol: str, st: pd.DataFrame, log=print) -> tuple[pd.DataFrame, str]:
    """(statements, note). Fetches BOTH Screener views (consolidated, standalone) and keeps
    the one with more annual years — the deep dive's rule — and uses it only when it has
    more years than the stored file. No threshold: more history wins, nothing else."""
    n_stored = len(_fys(st)) if not st.empty else 0
    try:
        from bs4 import BeautifulSoup
        from screener_client import ScreenerClient, BASE_URL
        c = ScreenerClient(cookie=os.environ.get("SCREENER_SESSION_COOKIE") or None)
    except Exception as e:
        log(f"  live statements skipped ({str(e)[:80]})")
        return st, ""
    best = None
    for variant, view in (("consolidated/", "consolidated"), ("", "standalone")):
        try:
            c._wait()
            r = c.session.get(f"{BASE_URL}/company/{symbol}/{variant}", timeout=30)
            if r.status_code != 200:
                continue
            df = pd.DataFrame(c.extract_statements(symbol, BeautifulSoup(r.text, "lxml")))
        except Exception as e:
            log(f"  live statements ({view}) failed: {str(e)[:80]}")
            continue
        if df.empty:
            continue
        n = len(_fys(df))
        if best is None or n > best[0]:
            best = (n, df, view)
    if best and best[0] > n_stored:
        n, df, view = best
        df["fetched_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        note = (f"Financial history pulled live from Screener ({view} view, {n} years); the "
                f"stored statements file has {n_stored} year(s).")
        log(f"  {note}")
        return df, note
    return st, ""


# ------------------------------------------------------------------ chapter 4 --
_BS = [("borrowings", "Borrowings", "balance_sheet", "Borrowings", "Rs Cr"),
       ("cwip", "Capital work in progress", "balance_sheet", "CWIP", "Rs Cr"),
       ("fixed_assets", "Fixed assets", "balance_sheet", "Fixed Assets", "Rs Cr"),
       ("cfo", "Cash from operations", "cash_flow", "Cash from Operating Activity", "Rs Cr"),
       ("fcf", "Free cash flow", "cash_flow", "Free Cash Flow", "Rs Cr"),
       ("roce", "Return on capital employed", "ratios", "ROCE %", "%"),
       ("debtor_days", "Debtor days", "ratios", "Debtor Days", "days"),
       ("inventory_days", "Inventory days", "ratios", "Inventory Days", "days"),
       ("payable_days", "Payable days", "ratios", "Days Payable", "days"),
       ("wc_days", "Working capital days", "ratios", "Working Capital Days", "days")]


def _net_worth(st, p):
    bs = NC.pivot(st, "balance_sheet")
    e, r = NC._get(bs, NC.L_EQUITY, p), NC._get(bs, NC.L_RESERVES, p)
    return None if e is None or r is None else e + r


def sec30_balance_cash(pack, st: pd.DataFrame, years: int = 10) -> None:
    if st.empty:
        pack.gap(SEC_BSCF, "no statements — balance sheet / cash flow not computable")
        return
    fys = _fys(st, years)
    wide = {s: NC.pivot(st, s) for s in ("balance_sheet", "cash_flow", "ratios")}
    fetched = str(st["fetched_at"].iloc[0]) if "fetched_at" in st.columns else None
    rows = []
    for p in fys:
        fy = NC.fy_label(p)
        row = {"FY": fy}
        nw = _net_worth(st, p)
        if nw is not None:
            pack.add(f"bs.net_worth.{fy}", f"Net worth (equity + reserves) {fy}", nw,
                     "Rs Cr", "reported", SEC_BSCF,
                     _stmt_src(pack.symbol, [NC.L_EQUITY, NC.L_RESERVES], fetched))
        row["net_worth"] = nw
        for key, label, stmt, item, unit in _BS:
            v = NC._get(wide[stmt], item, p)
            if v is not None:
                pack.add(f"bs.{key}.{fy}", f"{label} {fy}", v, unit, "reported", SEC_BSCF,
                         _stmt_src(pack.symbol, [item], fetched))
            row[key] = v
        cfo, fcf = row.get("cfo"), row.get("fcf")
        row["capex"] = (cfo - fcf) if (cfo is not None and fcf is not None) else None
        if row["capex"] is not None:
            pack.add(f"bs.capex.{fy}", f"Capex (cash from operations minus free cash flow) "
                     f"{fy}", row["capex"], "Rs Cr", "computed", SEC_BSCF,
                     _comp("capex = cash from operations - free cash flow (Screener)"),
                     [f"bs.cfo.{fy}", f"bs.fcf.{fy}"])
        if any(row.get(k) is not None for k in ("net_worth", "borrowings", "cfo")):
            rows.append(row)
    if rows:
        cols = ["FY", "net_worth", "borrowings", "cwip", "cfo", "capex", "fcf", "roce",
                "debtor_days", "inventory_days", "wc_days"]
        pack.table("tbl.balance_cash", "Balance sheet and cash flow", SEC_BSCF, cols,
                   [{c: r.get(c) for c in cols} for r in rows],
                   _stmt_src(pack.symbol, ["balance_sheet", "cash_flow", "ratios"], fetched),
                   note="Rs crore except ROCE (%) and days. Capex = cash from operations "
                        "minus free cash flow.")


# ------------------------------------------------------------------ chapter 5 --
def sec31_cash_bridge(pack, st: pd.DataFrame) -> None:
    """Profit -> cash -> what the cash paid for, summed over the last 5 and 10 years."""
    if st.empty:
        pack.gap(SEC_CASH, "no statements — cash bridge not computable")
        return
    pl, cf, bs = (NC.pivot(st, s) for s in ("annual_pl", "cash_flow", "balance_sheet"))
    all_fys = _fys(st)
    out_rows = []
    for n in (5, 10):
        fys = all_fys[-n:]
        if len(fys) < n:
            continue                        # never label a 6-year sum as 10 years
        span = f"{NC.fy_label(fys[0])}-{NC.fy_label(fys[-1])}"
        vals = {"pat": [], "cfo": [], "fcf": [], "div": []}
        for p in fys:
            pat = NC._get(pl, NC.L_PAT, p)
            cfo = NC._get(cf, NC.L_CFO, p)
            fcf = NC._get(cf, "Free Cash Flow", p)
            pay = NC._get(pl, "Dividend Payout %", p)
            if None in (pat, cfo, fcf):
                vals = None
                break
            vals["pat"].append(pat); vals["cfo"].append(cfo); vals["fcf"].append(fcf)
            vals["div"].append(max(pat, 0) * (pay or 0) / 100.0)
        if vals is None:
            pack.gap(SEC_CASH, f"{n}-year cash bridge: a year in {span} lacks profit, "
                               f"cash from operations or free cash flow")
            continue
        pat, cfo, fcf, div = (sum(vals[k]) for k in ("pat", "cfo", "fcf", "div"))
        capex = cfo - fcf
        b0, b1 = NC._get(bs, NC.L_BORROWINGS, fys[0]), NC._get(bs, NC.L_BORROWINGS, fys[-1])
        e0, e1 = NC._get(bs, NC.L_EQUITY, fys[0]), NC._get(bs, NC.L_EQUITY, fys[-1])
        tag, src = f"{n}y", _comp(f"sums over {span} of Screener annual figures")
        row = {"item": None}
        facts = [
            ("pat", f"Profit after tax, {span} total", pat, "Rs Cr"),
            ("cfo", f"Cash from operations, {span} total", cfo, "Rs Cr"),
            ("capex", f"Capex (cash from operations minus free cash flow), {span} total",
             capex, "Rs Cr"),
            ("fcf", f"Free cash flow, {span} total", fcf, "Rs Cr"),
            ("dividends", f"Dividends (payout % x profit), {span} total", div, "Rs Cr"),
            ("debt_change", f"Change in borrowings, {span}",
             None if None in (b0, b1) else b1 - b0, "Rs Cr"),
            ("equity_change", f"Change in equity share capital, {span}",
             None if None in (e0, e1) else e1 - e0, "Rs Cr"),
            ("cash_conversion", f"Cash from operations as % of profit, {span}",
             100.0 * cfo / pat if pat > 0 else None, "%"),
            ("self_funding", f"Cash from operations as % of capex + dividends, {span}",
             100.0 * cfo / (capex + div) if (capex + div) > 0 else None, "%"),
        ]
        for key, label, v, unit in facts:
            if v is not None:
                pack.add(f"cash.{key}.{tag}", label, float(v), unit, "computed", SEC_CASH, src)
        out_rows.append((n, span, {k: v for k, _, v, _ in facts}))
    if out_rows:
        labels = [("pat", "Profit after tax"), ("cfo", "Cash from operations"),
                  ("capex", "Capex"), ("fcf", "Free cash flow"), ("dividends", "Dividends"),
                  ("debt_change", "Change in borrowings"),
                  ("equity_change", "Change in equity share capital"),
                  ("cash_conversion", "Cash from operations / profit (%)"),
                  ("self_funding", "Cash from operations / (capex + dividends) (%)")]
        cols = ["item"] + [span for _, span, _ in out_rows]
        rows = [{"item": lab, **{span: (None if v.get(k) is None else round(v[k], 1))
                                 for _, span, v in out_rows}} for k, lab in labels]
        pack.table("tbl.cash_bridge", "Where the cash came from and where it went",
                   SEC_CASH, cols, rows,
                   _comp("sums of Screener annual figures; dividends = payout % x profit"),
                   note="Rs crore except the two percentages. Above 100% in the last row "
                        "means operations paid for capex and dividends without new money.")


# ------------------------------------------------------------------ chapter 11 --
def _fy_end_close(ohlcv: pd.DataFrame, period: str) -> float | None:
    """Last close on or before 31 March of the FY ('Mar 2026')."""
    if ohlcv is None or ohlcv.empty or "close" not in ohlcv.columns:
        return None
    try:
        yr = int(str(period).split()[-1])
    except ValueError:
        return None
    d = pd.to_datetime(ohlcv["date"], errors="coerce")
    sub = ohlcv[(d <= pd.Timestamp(yr, 3, 31)) & (d > pd.Timestamp(yr, 3, 31) - pd.Timedelta(days=10))]
    if sub.empty:
        return None
    v = pd.to_numeric(sub.sort_values("date")["close"], errors="coerce").dropna()
    return float(v.iloc[-1]) if len(v) else None


def sec33_pe_history(pack, store, st: pd.DataFrame, years: int = 5) -> dict | None:
    # Today's valuation belongs to the valuation chapter too (it lived only in chapter 1's
    # one-pager, so chapter 11 wrote "an elevated P/E" without the number — Morepen 18 Oct).
    try:
        cf = store.by_isin("company_facts.parquet", pack.isin, pack.symbol)
    except Exception:
        cf = pd.DataFrame()
    if not cf.empty:
        r = cf.iloc[0]
        when = str(r.get("mcap_as_of") or r.get("updated_at") or "")[:10]
        src = {"kind": "parquet", "table": "company_repo/_index/company_facts.parquet",
               "note": f"market cap as of {when}" if when else ""}
        for col, fid, label, unit in (("mcap_cr", "val.mcap_now", "Market cap today", "Rs Cr"),
                                      ("pe", "val.pe_now", "P/E today", "x"),
                                      ("pb", "val.pb_now", "Price to book value today", "x")):
            v = pd.to_numeric(r.get(col), errors="coerce")
            if pd.notna(v):
                pack.add(fid, label, float(v), unit, "reported", SEC_PE, src)
    if st.empty:
        return None
    try:
        ohlcv = store.parquet("data/ohlcv", f"{pack.symbol}.parquet")
    except Exception:
        ohlcv = pd.DataFrame()
    eps = dict(zip(NC.eps_series(st).periods, NC.eps_series(st).values))
    rows, pes = [], []
    for p in _fys(st, years):
        px, e = _fy_end_close(ohlcv, p), eps.get(p)
        fy = NC.fy_label(p)
        pe = px / e if (px and e and e > 0) else None      # no P/E in a loss year
        rows.append({"FY": fy, "fy_end_price": px, "eps": e,
                     "pe": None if pe is None else round(pe, 1)})
        if pe is not None:
            pes.append(pe)
            pack.add(f"val.pe.{fy}", f"P/E at {fy} year-end (price / EPS)", pe, "x",
                     "computed", SEC_PE, _comp(f"year-end close / EPS {fy}"))
    if not pes:
        pack.gap(SEC_PE, "no year-end price with a positive EPS — P/E history not computable")
        return None
    span = f"{rows[0]['FY']}-{rows[-1]['FY']}"
    res = {"low": min(pes), "median": statistics.median(pes), "high": max(pes), "span": span,
           "n": len(pes)}
    for k in ("low", "median", "high"):
        pack.add(f"val.pe_{k}", f"P/E {k} over {span} ({len(pes)} year(s) with profit)",
                 res[k], "x", "computed", SEC_PE, _comp(f"year-end P/E, {span}"))
    pack.table("tbl.pe_history", "P/E at each year-end", SEC_PE,
               ["FY", "fy_end_price", "eps", "pe"], rows,
               _comp("year-end close (price history) / EPS (Screener)"),
               note="Price from the daily price history (split-adjusted); EPS as Screener "
                    "reports it. A split or bonus year can distort that year's P/E.")
    return res


def _guided_growth(store, isin: str) -> tuple[float, str] | None:
    """Latest EXPLICIT revenue-growth guidance on record (guidance_tracker), as % a year."""
    try:
        g = store.by_isin("guidance_tracker.parquet", isin, "")
    except Exception:
        return None
    if g is None or g.empty:
        return None
    g = g[(g["metric"].astype(str) == "revenue") & (g["guidance_type"].astype(str) == "explicit")
          & (g["value_type"].astype(str) == "growth_pct")]
    g = g[pd.to_numeric(g["value_num"], errors="coerce").notna()]
    if g.empty:
        return None
    # Latest by FISCAL QUARTER, not by processing date: the backfill processes old calls
    # late, so sorting on processed_at picked Q4 FY18 guidance as "latest" for Morepen.
    def _qkey(q):
        m = re.match(r"Q([1-4])\s*FY\s*(\d{2,4})", str(q).strip(), re.I)
        if not m:
            return (-1, -1)
        y = int(m.group(2))
        return (y + 2000 if y < 100 else y, int(m.group(1)))
    g = g.assign(_k=g["quarter"].map(_qkey))
    g = g[g["_k"] != (-1, -1)].sort_values("_k")
    if g.empty:
        return None
    last = g[g["_k"] == g["_k"].iloc[-1]]
    pick = last[last["horizon_fy"].astype(str).isin(["3Y+", "3Y", "2Y"])]
    r = (pick if not pick.empty else last).iloc[-1]
    return float(r["value_num"]), f"management guidance, {r['quarter']} ({r['value']}, {r['horizon_fy']})"


def sec32_scenarios(pack, store, st: pd.DataFrame, pe: dict | None, mcap_now: float | None):
    if st.empty or not pe or not mcap_now:
        pack.gap(SEC_SCN, "scenarios need statements, a P/E history and today's market cap")
        return
    rev = [(p, v) for p, v in zip(NC.revenue_series(st).periods, NC.revenue_series(st).values)
           if v is not None]
    pat = dict(zip(NC.pat_series(st).periods, NC.pat_series(st).values))
    rev10 = rev[-10:]
    roll = []
    for i in range(3, len(rev10)):
        a, b = rev10[i - 3][1], rev10[i][1]
        if a > 0 and b > 0:
            roll.append(((b / a) ** (1 / 3) - 1) * 100)
    c5 = NC.cagr_pct(NC.revenue_series(st), window=5)
    margins = [100.0 * pat[p] / v for p, v in rev[-5:] if pat.get(p) is not None and v > 0]
    if not roll or not c5 or not margins:
        pack.gap(SEC_SCN, "scenarios need 6+ years of revenue and 5 years of profit")
        return
    guided = _guided_growth(store, pack.isin)
    g = {"bear": (min(roll), "weakest 3-year revenue growth in the last 10 years"),
         "base": (c5["cagr_pct"], f"revenue growth {c5['from_fy']}-{c5['to_fy']}"),
         "bull": guided or (max(roll), "best 3-year revenue growth in the last 10 years")}
    m = {"bear": (min(margins), "lowest net margin of the last 5 years"),
         "base": (statistics.median(margins), "median net margin of the last 5 years"),
         "bull": (max(margins), "highest net margin of the last 5 years")}
    x = {"bear": (pe["low"], f"lowest year-end P/E {pe['span']}"),
         "base": (pe["median"], f"median year-end P/E {pe['span']}"),
         "bull": (pe["high"], f"highest year-end P/E {pe['span']}")}
    rev0_p, rev0 = rev[-1]
    fy0 = NC.fy_label(rev0_p)
    rows, src = [], _comp(f"3-year arithmetic from {fy0} revenue on the stated assumptions")
    for case in ("bear", "base", "bull"):
        gr, mg, mult = g[case][0], m[case][0], x[case][0]
        rev3 = rev0 * (1 + gr / 100) ** HORIZON_YEARS
        pat3 = rev3 * mg / 100
        cap3 = pat3 * mult
        ret = ((cap3 / mcap_now) ** (1 / HORIZON_YEARS) - 1) * 100 if cap3 > 0 else None
        for key, label, v, unit in (
                ("growth", f"{case.title()} case: revenue growth a year", gr, "%"),
                ("margin", f"{case.title()} case: net margin", mg, "%"),
                ("pe", f"{case.title()} case: exit P/E", mult, "x"),
                ("revenue", f"{case.title()} case: revenue in {HORIZON_YEARS} years", rev3, "Rs Cr"),
                ("pat", f"{case.title()} case: profit in {HORIZON_YEARS} years", pat3, "Rs Cr"),
                ("mcap", f"{case.title()} case: implied market cap in {HORIZON_YEARS} years",
                 cap3, "Rs Cr"),
                ("return", f"{case.title()} case: implied return a year from today's market "
                           f"cap", ret, "%")):
            if v is not None and not (isinstance(v, float) and math.isnan(v)):
                pack.add(f"scn.{case}.{key}", label, float(v), unit, "computed", SEC_SCN, src)
        rows.append({"case": case.title(), "revenue_growth_pct": round(gr, 1),
                     "growth_from": g[case][1], "net_margin_pct": round(mg, 1),
                     "exit_pe": round(mult, 1), "implied_mcap_cr": round(cap3),
                     "return_pa_pct": None if ret is None else round(ret, 1)})
    pack.add("scn.mcap_now", "Market cap used for the scenarios", float(mcap_now), "Rs Cr",
             "reported", SEC_SCN, _comp("company_facts market cap"))
    pack.table("tbl.scenarios", f"Bear, base and bull — {HORIZON_YEARS} years from {fy0}",
               SEC_SCN, ["case", "revenue_growth_pct", "growth_from", "net_margin_pct",
                         "exit_pe", "implied_mcap_cr", "return_pa_pct"], rows,
               src, note="Arithmetic on assumptions taken from the company's own record — "
                         "not a forecast, not a target. Margins are net (profit / revenue); "
                         "P/E is the year-end range above.")


def build_dynamics(pack, store, st: pd.DataFrame, mcap_now: float | None) -> None:
    sec30_balance_cash(pack, st)
    sec31_cash_bridge(pack, st)
    pe = sec33_pe_history(pack, store, st)
    sec32_scenarios(pack, store, st, pe, mcap_now)
