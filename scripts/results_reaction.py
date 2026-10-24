r"""
results_reaction.py — results vs price reaction (open task O5, user rules 2026-10-18).

For every company that reported in the last few days (PF holdings first): was the
quarter STRONG, and how did the stock move against Nifty 1, 3 and 5 trading days after?

    STRONG RESULTS  = sales YoY > 30%  AND  PAT YoY > 30%  AND  EPS YoY > 30%
    STRONG MOVE     = stock beat Nifty by >= +5% (on any of day 1 / 3 / 5)

    OPPORTUNITY  strong results, but no strong move  -> the market has not reacted yet
    WARNING      strong move, but the results were NOT strong -> price ran ahead of numbers

No LLM. Inputs: results.parquet (Screener latest results, incl. EPS), Phase-1 OHLCV,
data/indices/NIFTY_50.parquet. Used by run_pead_daily.py (mail B gains a section).

    python scripts/results_reaction.py --self-test
"""
from __future__ import annotations

import html as _html
import sys

import numpy as np

import pandas as pd

STRONG_YOY = 30.0          # user 2026-10-18: > 30% YoY on sales, PAT AND EPS
STRONG_MOVE = 5.0          # user 2026-10-18: >= +5% vs Nifty
DAYS = (1, 3, 5)
LOOKBACK_DAYS = 8          # reporters this many calendar days back (room for day 5)


def _closes(df) -> pd.Series:
    if df is None or getattr(df, "empty", True) or "close" not in df.columns:
        return pd.Series(dtype=float)
    d = df[["date", "close"]].copy()
    d["date"] = pd.to_datetime(d["date"], errors="coerce")
    d["close"] = pd.to_numeric(d["close"], errors="coerce")
    d = d.dropna().drop_duplicates("date").sort_values("date")
    return d.set_index("date")["close"]


def yoy_of(comp_rows: pd.DataFrame, *terms: str) -> float | None:
    """YoY % of the first metric row matching any term; computed from the latest and
    year-ago LEVELS when Screener left yoy_pct blank (it does, mostly on Net profit).
    A non-positive year-ago base gives no YoY (a turnaround is not '+inf%'). Pure."""
    if comp_rows is None or comp_rows.empty:
        return None
    m = comp_rows["metric"].astype(str).str.lower()
    for t in terms:
        hit = comp_rows[m.str.contains(t, na=False, regex=False)]
        if hit.empty:
            continue
        r = hit.iloc[0]
        y = pd.to_numeric(r.get("yoy_pct"), errors="coerce")
        if pd.isna(y):
            lv = pd.to_numeric(r.get("latest_val"), errors="coerce")
            yv = pd.to_numeric(r.get("yearago_val"), errors="coerce")
            if pd.notna(lv) and pd.notna(yv) and yv > 0:
                y = (lv - yv) / yv * 100.0
        return None if pd.isna(y) else float(y)
    return None


def is_strong(sales, pat, eps) -> bool:
    """All three known and > 30%. A missing figure is NOT strong. Pure."""
    vals = (sales, pat, eps)
    return all(v is not None and v > STRONG_YOY for v in vals)


def reaction(stock: pd.Series, nifty: pd.Series, result_dt) -> dict:
    """{1: rel%, 3: rel%, 5: rel%} — stock return minus Nifty return from the last
    close ON OR BEFORE the result date to the Nth trading day after it. Days not yet
    traded are absent. Pure."""
    out = {}
    if stock is None or stock.empty or nifty is None or nifty.empty or result_dt is None:
        return out
    rd = pd.Timestamp(result_dt).normalize()
    base_s = stock[stock.index <= rd]
    base_n = nifty[nifty.index <= rd]
    if base_s.empty or base_n.empty:
        return out
    s0, n0 = float(base_s.iloc[-1]), float(base_n.iloc[-1])
    after = stock[stock.index > rd]
    for k in DAYS:
        if len(after) < k:
            break
        day = after.index[k - 1]
        n_at = nifty[nifty.index <= day]
        if n_at.empty or s0 <= 0 or n0 <= 0:
            break
        out[k] = (float(after.iloc[k - 1]) / s0 - 1) * 100 - (float(n_at.iloc[-1]) / n0 - 1) * 100
    return out


def flag(strong: bool, rel: dict) -> str:
    """'opportunity' | 'warning' | ''. Pure."""
    if not rel:
        return ""
    moved = max(rel.values()) >= STRONG_MOVE
    if strong and not moved:
        return "opportunity"
    if moved and not strong:
        return "warning"
    return ""


def _pct(v):
    return "–" if v is None else f"{v:+.0f}%"


def _rel_cell(v):
    if v is None:
        return "<span style='color:#999'>–</span>"
    col = "#1a7a3a" if v >= STRONG_MOVE else "#c0392b" if v <= -STRONG_MOVE else "#555"
    return f"<span style='color:{col}'>{v:+.1f}%</span>"


def section_html(rows: list[dict], chart_tags: dict | None = None) -> str:
    """rows: [{sym, name, qtr, result_dt, sales, pat, eps, strong, rel, flag, pf}].
    Flagged rows only (PF first), then a one-line count. Pure."""
    chart_tags = chart_tags or {}
    flagged = [r for r in rows if r["flag"]]
    flagged.sort(key=lambda r: (not r["pf"], r["flag"] != "opportunity", r["sym"]))
    n_strong = sum(1 for r in rows if r["strong"])
    head = (f"<h3 style='margin:16px 0 4px;font-size:15px'>&#128200; Results vs price "
            f"reaction (last {LOOKBACK_DAYS} days)</h3>"
            f"<div style='font-size:12px;color:#555;margin-bottom:6px'>Strong results = "
            f"sales, PAT <b>and</b> EPS all &gt; {STRONG_YOY:.0f}% YoY. Strong move = beat "
            f"Nifty by &ge; +{STRONG_MOVE:.0f}% on day 1, 3 or 5. "
            f"<b style='color:#1a7a3a'>Opportunity</b>: strong results, no strong move yet. "
            f"<b style='color:#b8860b'>Warning</b>: strong move without strong results. "
            f"{len(rows)} reporter(s) checked, {n_strong} with strong results.</div>")
    if not flagged:
        return head + "<p style='color:#999;font-size:12px'>No opportunity or warning flag.</p>"
    th = "".join(f"<th style='padding:3px 6px;text-align:left'>{h}</th>" for h in (
        "", "Stock", "Qtr", "Result", "Sales", "PAT", "EPS", "D1 vs Nifty", "D3", "D5"))
    trs, charts = [], []
    for r in flagged:
        tag = ("<b style='color:#1a7a3a'>OPPORTUNITY</b>" if r["flag"] == "opportunity"
               else "<b style='color:#b8860b'>WARNING</b>")
        pf = (" <span style='background:#1f4d6b;color:#fff;border-radius:3px;"
              "padding:0 4px;font-size:10px'>PF</span>" if r["pf"] else "")
        cells = [tag, f"<b>{_html.escape(r['sym'])}</b>{pf}<br><span style='color:#777;"
                 f"font-size:11px'>{_html.escape(str(r['name'])[:30])}</span>",
                 _html.escape(str(r["qtr"])), _html.escape(str(r["result_dt"])[:10]),
                 _pct(r["sales"]), _pct(r["pat"]), _pct(r["eps"]),
                 _rel_cell(r["rel"].get(1)), _rel_cell(r["rel"].get(3)), _rel_cell(r["rel"].get(5))]
        trs.append("<tr>" + "".join(f"<td style='padding:3px 6px;border-top:1px solid #eee;"
                                    f"font-size:12px'>{c}</td>" for c in cells) + "</tr>")
        if r["sym"] in chart_tags:
            charts.append(chart_tags[r["sym"]])
    return (head + "<table cellspacing='0' style='border-collapse:collapse'>"
            f"<tr style='background:#f4f6f8;font-size:12px'>{th}</tr>{''.join(trs)}</table>"
            + "".join(charts))


MAX_CHARTS = 4

# ---- O7 fresh winners (user 2026-10-24: "build the mail as a section") ----------------
# The SAME technical test the backtest ran (guru/fresh_winners_backtest.technical_flags),
# imported, not copied, so the live rule is the tested rule.
FW_WINDOW = 3              # trigger on one of the first 3 trading days from the result
FW_RECENT = 3              # listed while the result is within the last 3 trading days
FW_MIN_TURNOVER_CR = 1.0   # the backtest's liquidity floor (Guru daily screen's)
FW_CHARTS = 4
FW_RECORD = ("Backtest 2008-2026, 4,354 companies, 1,070 liquid trades: 3-month hit rate "
             "62.6%, median +6.7% (+4.1% vs Nifty); out of sample (2023+) 63.1% / +7.2%; "
             "worst 10% of trades about -20%. Strong results alone: 54.7% / +2.5%. "
             "Before costs. Entry = next day's open.")


def _fw_flags():
    import os as _os
    gdir = _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "guru")
    if gdir not in sys.path:
        sys.path.insert(0, gdir)
    import fresh_winners_backtest as FW
    return FW.technical_flags


def pat_base_positive(comp_rows: pd.DataFrame) -> bool:
    """Year-ago PAT > 0 (a loss turning into a profit is not 'growth' - backtest rule).
    Unknown year-ago value -> True only when Screener itself gave a YoY. Pure."""
    if comp_rows is None or comp_rows.empty:
        return False
    hit = comp_rows[comp_rows["metric"].astype(str).str.lower().str.contains("net prof", na=False)]
    if hit.empty:
        return False
    yv = pd.to_numeric(hit.iloc[0].get("yearago_val"), errors="coerce")
    if pd.notna(yv):
        return bool(yv > 0)
    return pd.notna(pd.to_numeric(hit.iloc[0].get("yoy_pct"), errors="coerce"))


def fresh_trigger(px: pd.DataFrame, result_dt, technical_flags) -> dict | None:
    """{'which', 'trigger_date', 'turnover_cr'} when the chart confirmed on one of the first
    FW_WINDOW trading days from the result AND the result is within the last FW_RECENT
    trading days of the price file; else None. Pure."""
    if px is None or getattr(px, "empty", True) or result_dt is None:
        return None
    f = technical_flags(px[["date", "open", "high", "low", "close", "volume"]].copy())
    dates = pd.to_datetime(f["date"]).values
    i0 = int(np.searchsorted(dates, np.datetime64(pd.Timestamp(result_dt).normalize())))
    if i0 >= len(f) or i0 < len(f) - FW_RECENT:
        return None
    for k in range(i0, min(i0 + FW_WINDOW, len(f))):
        a, b = bool(f["A"].iat[k]), bool(f["B"].iat[k])
        if a or b:
            return {"which": "52-week high + base breakout" if a and b else
                    "new 52-week high" if a else "60-day base breakout on 2x volume",
                    "trigger_date": str(pd.Timestamp(dates[k]).date()),
                    "turnover_cr": float(f["turnover_cr"].iat[k])
                    if pd.notna(f["turnover_cr"].iat[k]) else None}
    return None


def fresh_html(winners: list[dict], chart_tags: dict | None = None) -> str:
    """The 'Fresh winners' section; '' when there are none. Pure."""
    if not winners:
        return ""
    chart_tags = chart_tags or {}
    winners = sorted(winners, key=lambda r: (not r["pf"], r["sym"]))
    th = "".join(f"<th style='padding:3px 6px;text-align:left'>{h}</th>" for h in (
        "Stock", "Result", "Sales", "PAT", "EPS", "Chart signal", "On"))
    trs, charts = [], []
    for r in winners:
        pf = (" <span style='background:#1f4d6b;color:#fff;border-radius:3px;"
              "padding:0 4px;font-size:10px'>PF</span>" if r["pf"] else "")
        cells = [f"<b>{_html.escape(r['sym'])}</b>{pf}<br><span style='color:#777;"
                 f"font-size:11px'>{_html.escape(str(r['name'])[:30])}</span>",
                 _html.escape(str(r["result_dt"])[:10]), _pct(r["sales"]), _pct(r["pat"]),
                 _pct(r["eps"]), _html.escape(r["fresh"]["which"]),
                 _html.escape(r["fresh"]["trigger_date"])]
        trs.append("<tr>" + "".join(f"<td style='padding:3px 6px;border-top:1px solid #eee;"
                                    f"font-size:12px'>{c}</td>" for c in cells) + "</tr>")
        if r["sym"] in chart_tags:
            charts.append(chart_tags[r["sym"]])
    return (f"<h3 style='margin:16px 0 4px;font-size:15px'>&#127942; Fresh winners "
            f"({len(winners)})</h3>"
            f"<div style='font-size:12px;color:#555;margin-bottom:6px'>Strong results (sales, "
            f"PAT and EPS all &gt; {STRONG_YOY:.0f}% YoY, profitable a year ago) AND, within "
            f"{FW_WINDOW} trading days of the result, a new 52-week high or a close above the "
            f"60-day high on &ge; 2x volume. Liquid names only (&ge; Rs "
            f"{FW_MIN_TURNOVER_CR:g} cr/day). <i>{_html.escape(FW_RECORD)}</i></div>"
            "<table cellspacing='0' style='border-collapse:collapse'>"
            f"<tr style='background:#f4f6f8;font-size:12px'>{th}</tr>{''.join(trs)}</table>"
            + "".join(charts))


def build(drive, root_id: str, res: pd.DataFrame, reporters: pd.DataFrame,
          sym_map: dict, pf_isins: set, log=print) -> tuple[str, list]:
    """(section html, inline images) for `reporters` (slug/isin/latest_q/result_dt rows).
    Read-only: Phase-1 OHLCV + Nifty from Drive. Any failure -> ('', [])."""
    import io
    from _extractor_base import find_file, download_bytes, get_or_create_subfolder
    try:
        data = get_or_create_subfolder(drive, root_id, "data")
        ohlcv_id = get_or_create_subfolder(drive, data, "ohlcv")
        idx_id = get_or_create_subfolder(drive, data, "indices")
        nf = find_file(drive, idx_id, "NIFTY_50.parquet")
        nifty = _closes(pd.read_parquet(io.BytesIO(download_bytes(drive, nf)))) if nf else pd.Series(dtype=float)
    except Exception as e:                                   # noqa: BLE001
        log(f"  reaction: Nifty/ohlcv unavailable ({str(e)[:60]}) — section skipped")
        return "", []
    if nifty.empty:
        log("  reaction: no Nifty series — section skipped")
        return "", []
    rows, frames = [], {}
    try:
        _tf = _fw_flags()
    except Exception as e:                                   # noqa: BLE001
        _tf = None
        log(f"  fresh winners: backtest rule unavailable ({str(e)[:60]})")
    for _, rep in reporters.iterrows():
        isin = str(rep.get("isin", ""))
        sym = sym_map.get(isin) or str(rep.get("slug", "")) or isin
        comp = res[(res["slug"] == rep.get("slug")) & (res["latest_q"] == rep.get("latest_q"))]
        sales, pat, eps = (yoy_of(comp, "sales", "revenue"), yoy_of(comp, "net prof"),
                           yoy_of(comp, "eps"))
        strong = is_strong(sales, pat, eps)
        px = None
        try:
            fid = find_file(drive, ohlcv_id, f"{sym}.parquet")
            px = pd.read_parquet(io.BytesIO(download_bytes(drive, fid))) if fid else None
        except Exception:                                    # noqa: BLE001
            px = None
        rel = reaction(_closes(px), nifty, rep.get("result_dt"))
        fresh = None
        if strong and _tf is not None and pat_base_positive(comp):
            try:
                fresh = fresh_trigger(px, rep.get("result_dt"), _tf)
            except Exception:                                # noqa: BLE001
                fresh = None
            if fresh and (fresh["turnover_cr"] or 0) < FW_MIN_TURNOVER_CR:
                fresh = None
        rows.append({"sym": sym, "name": rep.get("company_name", ""), "qtr": rep.get("latest_q", ""),
                     "result_dt": rep.get("result_dt"), "sales": sales, "pat": pat, "eps": eps,
                     "strong": strong, "rel": rel, "flag": flag(strong, rel),
                     "pf": isin in pf_isins, "fresh": fresh})
        if px is not None:
            frames[sym] = (px, rep.get("result_dt"))
    images, tags = [], {}
    flagged = sorted((r for r in rows if r["flag"]), key=lambda r: (not r["pf"], r["sym"]))
    try:
        import mail_chart
        for r in flagged[:MAX_CHARTS]:
            px, rd = frames.get(r["sym"], (None, None))
            if px is None:
                continue
            png = mail_chart.render_png(px, f"{r['sym']} — {r['flag']}", days=120,
                                        markers=[(str(pd.Timestamp(rd).date()), "result", "#8e44ad")])
            if png:
                cid = f"react_{r['sym']}"
                images.append((cid, png))
                tags[r["sym"]] = mail_chart.img_tag(cid, r["sym"], width=560)
    except Exception as e:                                   # noqa: BLE001
        log(f"  reaction: charts skipped ({str(e)[:60]})")
    winners = [r for r in rows if r.get("fresh")]
    ftags = {}
    try:
        import mail_chart
        for r in sorted(winners, key=lambda r: (not r["pf"], r["sym"]))[:FW_CHARTS]:
            px, rd = frames.get(r["sym"], (None, None))
            if px is None:
                continue
            png = mail_chart.render_png(px, f"{r['sym']} - fresh winner", days=250,
                                        markers=[(str(pd.Timestamp(rd).date()), "result", "#8e44ad")])
            if png:
                cid = f"fresh_{r['sym']}"
                images.append((cid, png))
                ftags[r["sym"]] = mail_chart.img_tag(cid, r["sym"], width=560)
    except Exception as e:                                   # noqa: BLE001
        log(f"  fresh winners: charts skipped ({str(e)[:60]})")
    n_flag = sum(1 for r in rows if r["flag"])
    log(f"  reaction: {len(rows)} reporter(s), {sum(r['strong'] for r in rows)} strong, "
        f"{n_flag} flagged, {len(winners)} fresh winner(s), {len(images)} chart(s)")
    return fresh_html(winners, ftags) + section_html(rows, tags), images


def _self_test() -> int:
    ok = fail = 0

    def check(name, cond):
        nonlocal ok, fail
        ok, fail = (ok + 1, fail) if cond else (ok, fail + 1)
        print(f"  {'ok  ' if cond else 'FAIL'} {name}")

    rows = pd.DataFrame([
        {"metric": "Sales", "yoy_pct": 46.3, "latest_val": 94, "yearago_val": 64},
        {"metric": "Net Profit", "yoy_pct": None, "latest_val": 21.6, "yearago_val": 13.8},
        {"metric": "EPS in Rs", "yoy_pct": 56.0, "latest_val": 4.7, "yearago_val": 3.0}])
    s, p, e = yoy_of(rows, "sales", "revenue"), yoy_of(rows, "net prof"), yoy_of(rows, "eps")
    check("YoY read; PAT computed from levels when blank", round(p, 1) == 56.5 and s == 46.3)
    check("strong = all three > 30%", is_strong(s, p, e))
    check("one missing figure is NOT strong", not is_strong(45, 50, None))
    check("31/29/40 is NOT strong (all three must clear)", not is_strong(31, 29, 40))
    check("negative year-ago base -> no YoY",
          yoy_of(pd.DataFrame([{"metric": "Net Profit", "yoy_pct": None,
                                "latest_val": 5, "yearago_val": -2}]), "net prof") is None)
    d = pd.bdate_range("2026-10-12", periods=8)
    stock = pd.Series([100, 100, 100, 103, 106, 108, 109, 110], index=d, dtype=float)
    nifty = pd.Series([100] * 8, index=d, dtype=float)
    rel = reaction(stock, nifty, d[2])             # result on day index 2 (close 100)
    check("day 1/3/5 relative to Nifty", [round(rel[k]) for k in (1, 3, 5)] == [3, 8, 10])
    check("days not traded yet are absent", 5 not in reaction(stock, nifty, d[5]))
    check("strong results, no strong move -> opportunity", flag(True, {1: 1.0, 3: 4.9}) == "opportunity")
    check("strong move without strong results -> warning", flag(False, {1: 6.0}) == "warning")
    check("strong + strong move -> no flag", flag(True, {3: 7.0}) == "")
    check("no trading days yet -> no flag", flag(True, {}) == "")
    html = section_html([{"sym": "RPEL", "name": "Raghav", "qtr": "Sep 2026",
                          "result_dt": "2026-10-16", "sales": 46.3, "pat": 56.5, "eps": 56,
                          "strong": True, "rel": {1: 1.2}, "flag": "opportunity", "pf": True}])
    check("section shows the flag, PF chip and D1", all(x in html for x in ("OPPORTUNITY", "PF", "+1.2%")))
    # ---- O7 fresh winners: the live check uses the backtest's own technical test ----
    tf = _fw_flags()
    n = 300
    d = pd.bdate_range("2025-08-01", periods=n)
    cl = np.full(n, 100.0)
    cl[-2:] = 130.0                                        # new high on the last 2 days
    px = pd.DataFrame({"date": d, "open": cl, "high": cl, "low": cl, "close": cl,
                       "volume": np.full(n, 1e6)})
    ft = fresh_trigger(px, d[-2], tf)
    check("fresh: new 52-week high within 3 days of a recent result", ft is not None
          and "52-week" in ft["which"] and ft["trigger_date"] == str(d[-2].date()))
    check("fresh: result older than 3 trading days -> not listed",
          fresh_trigger(px, d[-10], tf) is None)
    flat = px.assign(close=100.0, open=100.0)
    check("fresh: no chart signal -> none", fresh_trigger(flat, d[-2], tf) is None)
    np_rows = pd.DataFrame([{"metric": "Net Profit", "yoy_pct": None, "latest_val": 5,
                             "yearago_val": -2}])
    check("fresh: loss a year ago is not growth", not pat_base_positive(np_rows))
    fh = fresh_html([{"sym": "NETWEB", "name": "Netweb", "result_dt": "2026-10-24",
                      "sales": 336, "pat": 335, "eps": 314, "pf": False,
                      "fresh": {"which": "new 52-week high", "trigger_date": "2026-10-26"}}])
    check("fresh section shows signal, numbers and the backtest record",
          all(x in fh for x in ("Fresh winners (1)", "new 52-week high", "62.6%", "+336%")))
    check("no winners -> no section", fresh_html([]) == "")
    print(f"\nresults_reaction self-test: {ok} passed, {fail} failed")
    return 1 if fail else 0


if __name__ == "__main__":
    if "--self-test" in sys.argv:
        raise SystemExit(_self_test())
