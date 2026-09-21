r"""
build_market_state_html.py — a market-state DASHBOARD (not per-stock), rendered
locally like build_gallery.py. Shows the current market health PLUS daily trend
charts of every indicator we already store, across daily/short/mid/long horizons.

Inputs (all already produced by the daily pipeline — READ ONLY):
  data/market_state/latest.parquet          today's health snapshot + components
  data/market_state/history.csv             daily history (health, breadth, VIX,
                                             FII, A/D, 52w H/L, component scores)
  data/market_state/sector_rotation_latest.csv
  data/indices/<INDEX>.parquet              daily OHLC (Nifty50/500/midcap/small/
                                             9 sectors/VIX) — for return dashboard
                                             + Nifty-vs-200SMA trend
  data/macro/FII_DII.csv                    daily FII vs DII net flows

Output:  market_state.html  (opens in browser unless --no-open)

Usage:
    python scripts/build_market_state_html.py            # build + open
    python scripts/build_market_state_html.py --no-open
    python scripts/build_market_state_html.py --dry-run  # counts only, no file
"""
from __future__ import annotations

import argparse
import io
import json
import os
import sys
import webbrowser
from datetime import datetime

import numpy as np
import pandas as pd
from dotenv import load_dotenv

_SD = os.path.dirname(os.path.abspath(__file__))
if _SD not in sys.path:
    sys.path.insert(0, _SD)
load_dotenv(os.path.join(os.path.dirname(_SD), ".env"))

from _extractor_base import (get_drive, get_or_create_subfolder, find_file,
                             download_bytes, log)

# Index → label + horizon bucket, for the returns dashboard.
# BROAD-MARKET indices only. The 9 sector indices are NOT repeated here — they
# get their own table below with the vs-Nifty500 comparison (they used to appear
# in both, which read as duplication).
INDICES = [
    ("NIFTY_50", "Nifty 50"), ("NIFTY_500", "Nifty 500"),
    ("NIFTY_MIDCAP_100", "Midcap 100"), ("NIFTY_SMALLCAP_100", "Smallcap 100"),
    ("INDIA_VIX", "India VIX"),
]
RET_WINDOWS = [("1D", 1), ("1W", 5), ("1M", 21), ("3M", 63), ("6M", 126), ("12M", 252)]


def _folder(drive, parts):
    fid = os.environ["GDRIVE_FOLDER_ID"]
    for p in parts.split("/"):
        fid = get_or_create_subfolder(drive, fid, p)
    return fid


def _read_csv(drive, folder, name):
    fid = find_file(drive, folder, name)
    return pd.read_csv(io.BytesIO(download_bytes(drive, fid))) if fid else pd.DataFrame()


def _read_parquet(drive, folder, name):
    fid = find_file(drive, folder, name)
    return pd.read_parquet(io.BytesIO(download_bytes(drive, fid))) if fid else pd.DataFrame()


def _series(df, val_col, date_col="date", tail=None):
    """[{time, value}] for lightweight-charts from a date+value frame."""
    if df.empty or val_col not in df.columns or date_col not in df.columns:
        return []
    d = df[[date_col, val_col]].copy()
    d[date_col] = pd.to_datetime(d[date_col], errors="coerce")
    d[val_col] = pd.to_numeric(d[val_col], errors="coerce")
    d = d.dropna().sort_values(date_col)
    if tail:
        d = d.tail(tail)
    return [{"time": t.strftime("%Y-%m-%d"), "value": round(float(v), 2)}
            for t, v in zip(d[date_col], d[val_col])]


def _ret(close: pd.Series, n: int):
    if len(close) > n and close.iloc[-1 - n] > 0:
        return (close.iloc[-1] / close.iloc[-1 - n] - 1) * 100
    return None


def _ret_cell(v):
    if v is None:
        return '<td style="text-align:right;color:#bbb">—</td>'
    col = "#1a7a3a" if v >= 0 else "#c0392b"
    return (f'<td style="text-align:right;font-weight:700;color:{col}">'
            f'{v:+.1f}%</td>')


def _tile(label, score, extra=""):
    v = None if score is None or pd.isna(score) else float(score)
    col = "#c0392b" if (v is None or v < 40) else ("#a66300" if v < 60 else "#1a7a3a")
    disp = "—" if v is None else f"{v:.0f}"
    return (f'<div style="flex:1;min-width:120px;background:#fff;border:1px solid #e3e7ee;'
            f'border-radius:8px;padding:8px 10px;text-align:center">'
            f'<div style="font-size:11px;color:#666">{label}</div>'
            f'<div style="font-size:22px;font-weight:800;color:{col}">{disp}</div>'
            f'<div style="font-size:10px;color:#999">{extra}</div></div>')


# ---------------------------------------------------------------- breadth ----
# guru/data/metrics/market_breadth.parquet — one row per (trading day, universe)
# back to 2001, built by guru/build_market_breadth.py. LOCAL and gitignored, so
# every function below fails soft: no file, no section, page renders as before.
BREADTH_PATH = os.path.join(os.path.dirname(_SD), "guru", "data", "metrics",
                            "market_breadth.parquet")
BREADTH_UNIS = [("liquid_nse", "Liquid NSE (>Rs1cr/day)"),
                ("top500_nse", "Top 500 NSE by turnover"),
                ("nse", "All NSE-listed"), ("all", "Everything (incl. BSE)")]
BREADTH_DEFAULT = "liquid_nse"
STALE_SESSIONS = 3
BREADTH_NOTES = {
    "liquid_nse": "NSE securities whose 20-day average turnover was at least "
                  "Rs1cr on that day. Membership moves with the market, which "
                  "is the point. NSE-only deliberately: the price store gets BSE "
                  "volume from yfinance, which reports none for a third of BSE "
                  "bars — so a turnover filter over the whole store was "
                  "silently selecting NSE names anyway and calling them liquid.",
    "top500_nse": "That day's 500 most-traded NSE securities — a turnover-rank "
                  "PROXY for large-cap, not index membership. No historical "
                  "Nifty 500 constituent list exists locally, and today's "
                  "market-cap segments cannot be applied backwards without "
                  "labelling a stock by what it later became.",
    "nse": "Every security carrying an NSE symbol (~2,365), including ones "
           "since delisted. No liquidity filter.",
    "all": "Every security with a bar that day, illiquid microcaps and BSE-only "
           "names included. Widest coverage and the noisiest movers counts — a "
           "thinly traded counter clears 5% on one trade. NOTE: the up-share-of-"
           "turnover chart is indicative only here, because yfinance reports no "
           "volume for many BSE names.",
}

_C = {"ma10": "#e67e22", "ma20": "#d35400", "ma50": "#16a085",
      "ma100": "#2980b9", "ma200": "#8e44ad", "up": "#1a7a3a", "dn": "#c0392b",
      "neutral": "#1a3d6e", "amber": "#a66300"}


def _load_breadth():
    if not os.path.exists(BREADTH_PATH):
        return None
    try:
        d = pd.read_parquet(BREADTH_PATH)
    except Exception as e:
        log(f"market_breadth.parquet unreadable ({e}) — breadth section skipped")
        return None
    if d.empty or "universe" not in d.columns:
        return None
    d["date"] = pd.to_datetime(d["date"])
    return d.sort_values("date")


def _vals(s, nd=2, as_int=False, neg=False):
    """Series -> compact JSON list. NaN becomes null so the chart breaks the line
    instead of drawing a zero, which would read as a real reading of zero."""
    a = pd.to_numeric(s, errors="coerce").to_numpy(dtype="float64")
    if neg:
        a = -a
    ok = np.isfinite(a)
    if as_int:
        return [int(round(v)) if k else None for v, k in zip(a, ok)]
    return [round(float(v), nd) if k else None for v, k in zip(a, ok)]


def _pctile(series, value):
    """Where today sits in its own history, 0-100. This is the whole point of
    keeping 25 years: 46% above the 50 DMA means nothing until you know it is
    the 38th percentile of every day since 2001."""
    a = pd.to_numeric(series, errors="coerce").dropna().to_numpy()
    if not len(a) or value is None or not np.isfinite(value):
        return None
    return float((a <= value).mean() * 100)


def _ord(n):
    n = int(round(n))
    if 10 <= n % 100 <= 20:
        return f"{n}th"
    return f"{n}{ {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th') }"


def _ctx_strip(g):
    """'Today in context' — each headline number beside its own 25-year
    percentile, so the reader can tell a normal day from an extreme one."""
    last = g.iloc[-1]
    items = [
        ("% above 50 DMA", "pct_above_50dma", "{:.0f}%", False),
        ("% above 200 DMA", "pct_above_200dma", "{:.0f}%", False),
        ("% >20% below own 52w high", "pct_dd20", "{:.0f}%", True),
        ("Median dist. from 52w high", "median_dd_from_52w_high", "{:.0f}%", False),
        ("New 52w highs − lows", "net_hl", "{:+.0f}", False),
        ("Advancing", "pct_advancing", "{:.0f}%", False),
        ("Up-share of turnover", "pct_turnover_up", "{:.0f}%", False),
        ("McClellan oscillator", "mcclellan_osc", "{:+.0f}", False),
    ]
    out = []
    for label, col, fmt, invert in items:
        if col not in g.columns:
            continue
        v = last.get(col)
        v = None if v is None or pd.isna(v) else float(v)
        p = _pctile(g[col], v)
        disp = "—" if v is None else fmt.format(v)
        if p is None:
            pc, ptxt = "#bbb", "—"
        else:
            rank = 100 - p if invert else p
            pc = "#1a7a3a" if rank >= 65 else ("#c0392b" if rank <= 35 else "#a66300")
            ptxt = f"{_ord(p)} pct"
        out.append(
            f'<div style="flex:1;min-width:132px;background:#fff;border:1px solid #e3e7ee;'
            f'border-radius:8px;padding:7px 9px">'
            f'<div style="font-size:10px;color:#666;line-height:1.25">{label}</div>'
            f'<div style="font-size:19px;font-weight:800;color:{pc}">{disp}</div>'
            f'<div style="font-size:10px;color:#999">{ptxt} since {g["date"].iloc[0].year}</div>'
            f'</div>')
    return "".join(out)


def _nifty_for_dates(dates, drive, idx_folder):
    """Nifty 50 close aligned to the breadth date index. Drive's copy is
    CI-fresh but shallow-ish; guru/data/macro_hist goes back to 2007 and is
    stale (refresh_daily.py does not touch macro_hist). Union them and prefer
    Drive, so the line is both deep and current without a new fetch."""
    frames = []
    loc = os.path.join(os.path.dirname(_SD), "guru", "data", "macro_hist",
                       "NIFTY_50.parquet")
    if os.path.exists(loc):
        try:
            frames.append(pd.read_parquet(loc, columns=["date", "close"]))
        except Exception:
            pass
    try:
        dr = _read_parquet(drive, idx_folder, "NIFTY_50.parquet")
        if not dr.empty and {"date", "close"} <= set(dr.columns):
            frames.append(dr[["date", "close"]])
    except Exception:
        pass
    if not frames:
        return None
    n = pd.concat(frames, ignore_index=True)
    n["date"] = pd.to_datetime(n["date"], errors="coerce")
    n = (n.dropna(subset=["date"]).drop_duplicates("date", keep="last")
           .sort_values("date").set_index("date")["close"])
    return pd.to_numeric(n, errors="coerce").reindex(dates).ffill()


# Chart id -> (heading, one-line explanation of what it tells you)
BREADTH_CHARTS = [
    ("bd_ma", "% of stocks above their moving average",
     "The core breadth picture. All five lines high = a broad advance; price "
     "holding while the 10/20 roll over = the rally narrowing."),
    ("bd_mv3", "Stocks moving more than 3% in a day (up above, down below)", ""),
    ("bd_mv5", "Stocks moving more than 5% in a day", ""),
    ("bd_mv10", "Stocks moving more than 10% in a day",
     "Clusters of big down-days mark capitulation; clusters of big up-days "
     "mark both bottoms and blow-off tops."),
    ("bd_hl", "New 52-week highs vs new 52-week lows",
     "Made an actual new 52-week extreme that day, on intraday high/low."),
    ("bd_ad", "Advance/decline line (normalised), with its 200-day mean",
     "Cumulative net advancers as a share of the universe — normalised because "
     "the universe grows from a few hundred names in 2001 to ~4,500 today. It "
     "drifts downward even through bull markets (more names fall than rise on "
     "a typical day), so read the SLOPE, not the level: above its 200-day mean "
     "and rising is broadening participation."),
    ("bd_updown", "Up-share of daily turnover",
     "Share of the day's traded value in advancing stocks. Below 50 for a "
     "stretch is distribution, whatever the index is doing."),
    ("bd_dd", "How beaten-up the average stock is",
     "Share of names more than 20% below their own 52-week high, and the "
     "median distance from that high. Answers 'the index is fine, so why "
     "does my book hurt?'"),
    ("bd_eqw", "Average stock vs Nifty 50 (ratio, rebased)",
     "Rising = the typical stock is beating the index. Falling = the index is "
     "being carried by a few heavyweights."),
    ("bd_mcc", "McClellan oscillator and summation index",
     "Ratio-adjusted, so it stays comparable as the universe grows."),
    ("bd_thrust", "Zweig breadth thrust",
     "10-day average of advancers / (advancers + decliners). A move from below "
     "0.40 to above 0.615 inside 10 sessions has marked major bottoms."),
]


def _breadth_payload(bd, nifty_by_uni):
    """{chart_id: {universe: [series...]}} plus a shared date index."""
    charts = {cid: {} for cid, _, _ in BREADTH_CHARTS}
    ctx = {}
    for uni, _ in BREADTH_UNIS:
        g = bd[bd["universe"].eq(uni)].sort_values("date")
        if g.empty:
            continue
        ctx[uni] = _ctx_strip(g)

        charts["bd_ma"][uni] = [
            {"n": f"{w} DMA", "c": _C[f"ma{w}"], "t": "line",
             "v": _vals(g[f"pct_above_{w}dma"], 1)}
            for w in (10, 20, 50, 100, 200) if f"pct_above_{w}dma" in g.columns]

        for cid, t in (("bd_mv3", 3), ("bd_mv5", 5), ("bd_mv10", 10)):
            charts[cid][uni] = [
                {"n": f"up >{t}%", "c": _C["up"], "t": "hist",
                 "v": _vals(g[f"up_{t}"], as_int=True)},
                {"n": f"down >{t}%", "c": _C["dn"], "t": "hist",
                 "v": _vals(g[f"dn_{t}"], as_int=True, neg=True)},
            ]

        charts["bd_hl"][uni] = [
            {"n": "new highs", "c": _C["up"], "t": "line",
             "v": _vals(g["new_52w_high"], as_int=True)},
            {"n": "new lows", "c": _C["dn"], "t": "line",
             "v": _vals(g["new_52w_low"], as_int=True)},
        ]
        # The A/D line drifts DOWN across the whole sample even through big bull
        # markets, because in a broad universe more names fall than rise on a
        # typical day (returns are positively skewed). The level therefore says
        # little; the slope says everything. Plotting its own 200-day mean
        # alongside makes that readable: above the mean = breadth improving.
        ad = pd.to_numeric(g["ad_line_norm"], errors="coerce")
        charts["bd_ad"][uni] = [
            {"n": "A/D line", "c": _C["neutral"], "t": "line",
             "v": _vals(ad, 1)},
            {"n": "200d mean", "c": _C["ma10"], "t": "line",
             "v": _vals(ad.rolling(200, min_periods=50).mean(), 1)},
        ]
        charts["bd_updown"][uni] = [
            {"n": "up-share %", "c": _C["neutral"], "t": "line",
             "v": _vals(g["pct_turnover_up"], 1)}]
        charts["bd_dd"][uni] = [
            {"n": "% >20% below high", "c": _C["dn"], "t": "line",
             "v": _vals(g["pct_dd20"], 1)},
            {"n": "median dist. from high", "c": _C["amber"], "t": "line",
             "v": _vals(g["median_dd_from_52w_high"], 1)},
        ]
        charts["bd_mcc"][uni] = [
            {"n": "oscillator", "c": _C["neutral"], "t": "line",
             "v": _vals(g["mcclellan_osc"], 1)},
            {"n": "summation", "c": _C["ma200"], "t": "line",
             "v": _vals(g["mcclellan_summation"], 0)},
        ]
        charts["bd_thrust"][uni] = [
            {"n": "thrust", "c": _C["neutral"], "t": "line",
             "v": _vals(g["zweig_thrust"], 3)}]

        # Average stock vs Nifty, as a RATIO. The equal-weight index is
        # rebalanced daily on noisy prices, so its ABSOLUTE level overstates an
        # investable return badly; the ratio against the index is the honest
        # form and is also the more useful read.
        nif = nifty_by_uni.get(uni)
        eq = pd.to_numeric(g["eqw_index"], errors="coerce").to_numpy("float64")
        if nif is not None:
            nv = nif.to_numpy("float64")
            both = np.isfinite(eq) & np.isfinite(nv) & (nv > 0) & (eq > 0)
            ratio = np.full(len(eq), np.nan)
            if both.any():
                i0 = int(np.argmax(both))
                ratio[both] = ((eq[both] / eq[i0]) / (nv[both] / nv[i0])) * 100.0
            charts["bd_eqw"][uni] = [
                {"n": "avg stock / Nifty", "c": _C["neutral"], "t": "line",
                 "v": _vals(pd.Series(ratio), 1)}]
        else:
            charts["bd_eqw"][uni] = []
    return charts, ctx


def _breadth_block(bd, drive, idx_folder):
    """Returns (section_html, payload_dict). ("", {}) when the table is absent,
    so the page is byte-identical to before in that case."""
    if bd is None or bd.empty:
        return "", {}
    dates = pd.DatetimeIndex(sorted(bd["date"].unique()))
    nifty = _nifty_for_dates(dates, drive, idx_folder)
    nifty_by_uni = {}
    for uni, _ in BREADTH_UNIS:
        g = bd[bd["universe"].eq(uni)].sort_values("date")
        nifty_by_uni[uni] = (None if nifty is None
                             else nifty.reindex(pd.DatetimeIndex(g["date"])))
    charts, ctx = _breadth_payload(bd, nifty_by_uni)

    last = dates[-1]
    age = int(np.busday_count(last.date(), datetime.now().date()))
    warn = ""
    if age > STALE_SESSIONS:
        warn = (f'<div style="margin:6px 0;padding:7px 10px;background:#fff4e5;'
                f'border-left:4px solid #e67e22;border-radius:6px;font-size:12px">'
                f'<b>Breadth data is {age} sessions behind.</b> The local price '
                f'store ends {last.date()}. Refresh it with '
                f'<code>python guru/refresh_daily.py --prices</code>, then '
                f'<code>python guru/build_market_breadth.py</code>.</div>')

    btns = "".join(
        f'<button class="ub" data-u="{u}" onclick="setUni(\'{u}\')">{lbl}</button>'
        for u, lbl in BREADTH_UNIS)
    zbtns = "".join(
        f'<button class="zb" data-z="{b}" onclick="setZoom({b})">{lbl}</button>'
        for lbl, b in (("1y", 252), ("3y", 756), ("5y", 1260), ("10y", 2520), ("max", 0)))
    cards = "".join(
        f'<div class="card"><h3>{human}</h3>'
        + (f'<div class="note">{note}</div>' if note else "")
        + f'<div class="chart bchart" id="{cid}"></div></div>'
        for cid, human, note in BREADTH_CHARTS)

    return (
        f'<h2>Market breadth — {dates[0].date()} to {last.date()}</h2>'
        f'{warn}'
        f'<div style="font-size:11px;color:#888;margin:0 6px 6px">'
        f'{len(dates):,} trading days from the local 25-year price store, which '
        f'includes delisted securities — so past bear markets are not flattered '
        f'by survivorship. Percentages use each measure\'s own denominator '
        f'(a 3-year-old stock has no 200 DMA and is excluded from that line).'
        f'</div>'
        f'<div class="btnrow"><span class="btnlbl">Universe</span>{btns}</div>'
        f'<div class="btnrow"><span class="btnlbl">Window</span>{zbtns}</div>'
        f'<div style="font-size:10px;color:#999;margin:2px 6px 8px" id="unote"></div>'
        f'<div class="tiles" id="bctx"></div>'
        f'<div class="charts">{cards}</div>',
        {"dates": [d.strftime("%Y-%m-%d") for d in dates],
         "charts": charts, "ctx": ctx, "notes": BREADTH_NOTES,
         "default": BREADTH_DEFAULT},
    )


_TPL = """<!doctype html><html><head><meta charset="utf-8">
<title>Market State __DATE__</title>
<script src="https://unpkg.com/lightweight-charts@4.1.3/dist/lightweight-charts.standalone.production.js"></script>
<style>
 body{font-family:-apple-system,Segoe UI,Roboto,Arial,sans-serif;background:#f4f6f9;margin:0;padding:14px;color:#222}
 h1{font-size:20px;margin:4px 6px}
 h2{font-size:14px;margin:18px 6px 6px;color:#1a3d6e}
 .wrap{max-width:1180px;margin:0 auto}
 .tiles{display:flex;gap:8px;flex-wrap:wrap;margin:6px 0}
 .charts{display:grid;grid-template-columns:1fr 1fr;gap:14px}
 .card{background:#fff;border:1px solid #e3e7ee;border-radius:8px;padding:8px 10px;box-shadow:0 1px 3px rgba(0,0,0,.05)}
 .card h3{font-size:12px;margin:2px 0 6px;color:#555;font-weight:600}
 .chart{height:200px}
 table{border-collapse:collapse;width:100%;font-size:12px}
 th,td{padding:3px 6px;border-bottom:1px solid #eee}
 th{text-align:right;color:#666;font-weight:600}
 th:first-child,td:first-child{text-align:left}
 .btnrow{margin:4px 6px;display:flex;flex-wrap:wrap;gap:6px;align-items:center}
 .btnlbl{font-size:11px;color:#666;min-width:56px}
 .btnrow button{font:inherit;font-size:11px;padding:4px 10px;border:1px solid #d5dbe5;background:#fff;color:#333;border-radius:14px;cursor:pointer}
 .btnrow button:hover{border-color:#1a3d6e}
 .btnrow button.on{background:#1a3d6e;border-color:#1a3d6e;color:#fff;font-weight:600}
 .note{font-size:10px;color:#888;margin:0 0 5px;line-height:1.35}
 .bchart{height:230px}
 @media(max-width:820px){.charts{grid-template-columns:1fr}}
</style></head><body><div class="wrap">
<h1>__HEADLINE__</h1>
<div style="font-size:11px;color:#888;margin:-2px 6px 8px">__SUB__</div>
<div class="tiles">__TILES__</div>
<h2>Trends (daily)</h2>
<div class="charts">__CHARTS__</div>
__BREADTH__
<h2>Index dashboard — returns by horizon</h2>
<div class="card"><table>__IDXTABLE__</table></div>
<h2>Sector rotation (vs Nifty 500)</h2>
<div class="card"><table>__SECTABLE__</table></div>
<h2>Breadth by market-cap segment</h2>
<div class="card"><table>__SEGBREADTH__</table></div>
<h2>Breadth by sector</h2>
<div class="card"><table>__SECBREADTH__</table></div>
</div>
<script>
const S=__PAYLOAD__;
function line(id,color,area){
 const el=document.getElementById(id); if(!el)return;
 const ch=LightweightCharts.createChart(el,{height:200,layout:{textColor:'#333',background:{color:'#fff'}},
   rightPriceScale:{borderColor:'#eee'},timeScale:{borderColor:'#eee'},
   grid:{horzLines:{color:'#f3f3f3'},vertLines:{color:'#fafafa'}}});
 (S[id]||[]).forEach(function(s){
   let ser = s.type==='hist'? ch.addHistogramSeries({color:s.color||color})
           : ch.addLineSeries({color:s.color||color,lineWidth:2,priceLineVisible:false});
   ser.setData(s.data||[]);
 });
 ch.timeScale().fitContent();
}
__DRAW__

/* ---- breadth history. Separate from line() above on purpose: these series
   ship as compact parallel arrays against one shared date index (~4x smaller
   than {time,value} objects over 25 years x 4 universes), and they are rebuilt
   on every universe switch. The seven charts above are untouched. ---- */
const BP=__BPAYLOAD__;
let _bch=[], _buni=(BP&&BP.default)||'liquid', _bwin=756;
function _bmk(cid,spec){
 const el=document.getElementById(cid); if(!el) return null;
 const ch=LightweightCharts.createChart(el,{height:230,
   layout:{textColor:'#333',background:{color:'#fff'}},
   rightPriceScale:{borderColor:'#eee'},timeScale:{borderColor:'#eee'},
   grid:{horzLines:{color:'#f3f3f3'},vertLines:{color:'#fafafa'}}});
 const D=BP.dates;
 (spec||[]).forEach(function(s){
   const d=[],v=s.v||[];
   for(let i=0;i<v.length;i++){ if(v[i]===null||v[i]===undefined) continue;
     d.push({time:D[i],value:v[i]}); }
   const ser = s.t==='hist'
     ? ch.addHistogramSeries({color:s.c,title:s.n,priceLineVisible:false})
     : ch.addLineSeries({color:s.c,lineWidth:2,title:s.n,priceLineVisible:false});
   ser.setData(d);
 });
 return ch;
}
function bwin(){
 const n=BP.dates.length;
 _bch.forEach(function(ch){ try{
   if(_bwin>0 && _bwin<n) ch.timeScale().setVisibleLogicalRange({from:n-_bwin,to:n-1});
   else ch.timeScale().fitContent();
 }catch(e){} });
}
function bdraw(){
 _bch.forEach(function(c){ try{c.remove();}catch(e){} }); _bch=[];
 Object.keys(BP.charts||{}).forEach(function(cid){
   const ch=_bmk(cid,(BP.charts[cid]||{})[_buni]); if(ch)_bch.push(ch); });
 const cx=document.getElementById('bctx'); if(cx) cx.innerHTML=(BP.ctx||{})[_buni]||'';
 const un=document.getElementById('unote'); if(un) un.innerHTML=(BP.notes||{})[_buni]||'';
 bwin();
}
function _bmark(cls,attr,val){
 document.querySelectorAll('.'+cls).forEach(function(b){
   b.classList.toggle('on', b.getAttribute(attr)===String(val)); });
}
function setUni(u){ _buni=u; _bmark('ub','data-u',u); bdraw(); }
function setZoom(b){ _bwin=b; _bmark('zb','data-z',b); bwin(); }
if(BP && BP.dates && BP.dates.length){
 _bmark('ub','data-u',_buni); _bmark('zb','data-z',_bwin); bdraw();
}
</script></body></html>"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-open", action="store_true")
    ap.add_argument("--out", default="market_state.html")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    drive = get_drive()
    ms = _folder(drive, "data/market_state")
    idx = _folder(drive, "data/indices")
    mac = _folder(drive, "data/macro")

    latest = _read_parquet(drive, ms, "latest.parquet")
    hist = _read_csv(drive, ms, "history.csv")
    sect = _read_csv(drive, ms, "sector_rotation_latest.csv")
    fd = _read_csv(drive, mac, "FII_DII.csv")
    if latest.empty:
        log("market_state/latest.parquet missing — run market_state.py first.")
        return
    row = latest.iloc[0].to_dict()
    log(f"health={row.get('health_score')} regime={row.get('regime')} "
        f"| history rows={len(hist)} | sectors={len(sect)}")

    # ---- headline + component tiles ----
    regime = str(row.get("regime", "?"))
    rcol = {"RISK_ON": "#1a7a3a", "NEUTRAL": "#a66300", "RISK_OFF": "#c0392b"}.get(regime, "#555")
    headline = (f'Market Health <b style="color:{rcol}">{row.get("health_score","?")}</b>/100 '
                f'&nbsp;·&nbsp; <b style="color:{rcol}">{regime}</b>')
    sub = (f'Nifty50 {row.get("nifty50_close","?")} (200SMA {row.get("nifty50_sma200","?")}, '
           f'{"above" if row.get("nifty50_above_200sma") else "below"}) · '
           f'VIX {row.get("india_vix","?")} · FII 5d ₹{row.get("fii_5d_net_cr","?")}cr · '
           f'{row.get("new_52w_highs","?")} new highs / {row.get("new_52w_lows","?")} lows · '
           f'as of {row.get("date","?")}')
    tiles = "".join([
        _tile("Nifty vs 200SMA", row.get("nifty50_trend_score"), "trend"),
        _tile("Breadth >50SMA", row.get("breadth_50sma_score"),
              f'{row.get("pct_above_50sma","?")}% above'),
        _tile("Highs−Lows", row.get("highs_lows_score"),
              f'{row.get("new_52w_highs","?")}H / {row.get("new_52w_lows","?")}L'),
        _tile("VIX", row.get("vix_score"), f'{row.get("india_vix","?")}'),
        _tile("FII flow", row.get("fii_score"), f'₹{row.get("fii_5d_net_cr","?")}cr 5d'),
        _tile("Adv/Decl", row.get("ad_ratio_score"),
              f'{row.get("advances","?")}/{row.get("declines","?")}'),
    ])

    # ---- trend charts (daily) ----
    payload, chart_ids = {}, []

    def add_chart(cid, human, series):
        chart_ids.append((cid, human))
        payload[cid] = series

    n50 = _read_parquet(drive, idx, "NIFTY_50.parquet")
    if not n50.empty:
        n50 = n50.sort_values("date")
        n50["sma200"] = pd.to_numeric(n50["close"], errors="coerce").rolling(200).mean()
        add_chart("nifty", "Nifty 50 vs 200-SMA (1y)", [
            {"type": "line", "color": "#1a3d6e", "data": _series(n50, "close", tail=252)},
            {"type": "line", "color": "#e67e22", "data": _series(n50, "sma200", tail=252)},
        ])
    add_chart("health", "Market Health Score", [
        {"type": "line", "color": "#8e44ad", "data": _series(hist, "health_score")}])
    add_chart("breadth", "% above 50-SMA", [
        {"type": "line", "color": "#16a085", "data": _series(hist, "pct_above_50sma")}])
    vix = _read_parquet(drive, idx, "INDIA_VIX.parquet")
    add_chart("vix", "India VIX (6m)", [
        {"type": "line", "color": "#c0392b", "data": _series(vix, "close", tail=126)}])
    # highs vs lows (two lines)
    add_chart("hl", "New 52w Highs vs Lows", [
        {"type": "line", "color": "#1a7a3a", "data": _series(hist, "new_52w_highs")},
        {"type": "line", "color": "#c0392b", "data": _series(hist, "new_52w_lows")}])
    # FII vs DII daily net (histograms)
    if not fd.empty and {"category", "net", "date"} <= set(fd.columns):
        fd["date"] = pd.to_datetime(fd["date"], errors="coerce")
        fii = fd[fd["category"].astype(str).str.contains("FII", na=False)].sort_values("date").tail(40)
        dii = fd[fd["category"].astype(str).str.contains("DII", na=False)].sort_values("date").tail(40)

        def _hist(d):
            return [{"time": t.strftime("%Y-%m-%d"),
                     "value": round(float(v), 1),
                     "color": "#1a7a3a" if v >= 0 else "#c0392b"}
                    for t, v in zip(d["date"], pd.to_numeric(d["net"], errors="coerce"))
                    if pd.notna(v)]
        add_chart("fii", "FII net (₹cr, 40d)", [{"type": "hist", "data": _hist(fii)}])
        add_chart("dii", "DII net (₹cr, 40d)", [{"type": "hist", "data": _hist(dii)}])

    charts_html = "".join(
        f'<div class="card"><h3>{human}</h3><div class="chart" id="{cid}"></div></div>'
        for cid, human in chart_ids)
    draw = "\n".join(f'line("{cid}","#1a3d6e");' for cid, _ in chart_ids)

    # ---- index returns dashboard ----
    hdr = "<tr><th>Index</th>" + "".join(f"<th>{lbl}</th>" for lbl, _ in RET_WINDOWS) + "</tr>"
    body = []
    for key, lbl in INDICES:
        df = _read_parquet(drive, idx, f"{key}.parquet")
        if df.empty:
            continue
        c = pd.to_numeric(df.sort_values("date")["close"], errors="coerce").dropna()
        cells = "".join(_ret_cell(_ret(c, n)) for _, n in RET_WINDOWS)
        body.append(f"<tr><td>{lbl}</td>{cells}</tr>")
    idxtable = hdr + "".join(body)

    # ---- sector rotation table ----
    if not sect.empty:
        sect = sect.sort_values("vs_nifty500_3m_pct", ascending=False)
        sh = ("<tr><td>Sector</td><th>1M</th><th>vs500 1M</th>"
              "<th>3M</th><th>vs500 3M</th></tr>")
        sb = "".join(
            f'<tr><td>{r["sector"].replace("NIFTY_","")}</td>'
            f'{_ret_cell(r.get("return_1m_pct"))}{_ret_cell(r.get("vs_nifty500_1m_pct"))}'
            f'{_ret_cell(r.get("return_3m_pct"))}{_ret_cell(r.get("vs_nifty500_3m_pct"))}</tr>'
            for _, r in sect.iterrows())
        sectable = sh + sb
    else:
        sectable = "<tr><td>no sector data</td></tr>"

    # ---- breadth by mcap-segment and by sector (#29 extension) ----
    # Join today's features (above_50/200SMA) to the classification (segment,
    # sector), both keyed by symbol — data we already store.
    feats = _read_parquet(drive, _folder(drive, "features"), "latest.parquet")
    cls = _read_parquet(drive, _folder(drive, "company_repo/_index"),
                        "company_classification.parquet")
    segbreadth = secbreadth = "<tr><td>no breadth data</td></tr>"
    if not feats.empty and {"symbol", "above_50sma", "above_200sma"} <= set(feats.columns):
        f = feats[["symbol", "above_50sma", "above_200sma"]].copy()
        f["symbol"] = f["symbol"].astype(str).str.upper()
        f["above_50sma"] = f["above_50sma"].astype(bool)
        f["above_200sma"] = f["above_200sma"].astype(bool)
        if not cls.empty and "symbol" in cls.columns:
            cmap = cls.copy()
            cmap["symbol"] = cmap["symbol"].astype(str).str.upper()
            f = f.merge(cmap[["symbol", "segment", "sector"]].drop_duplicates("symbol"),
                        on="symbol", how="left")
        else:
            f["segment"] = f["sector"] = None

        def _breadth(col, lbl, min_n=5, order=None):
            d = f.dropna(subset=[col])
            d = d[d[col].astype(str).str.lower() != "unknown"]
            if d.empty:
                return f"<tr><td>no {lbl.lower()} data</td></tr>"
            rows = []
            for name, g in d.groupby(col):
                if len(g) < min_n:
                    continue
                rows.append((str(name), len(g), 100 * g["above_50sma"].mean(),
                             100 * g["above_200sma"].mean()))
            if order:
                rows.sort(key=lambda x: order.index(x[0]) if x[0] in order else 99)
            else:
                rows.sort(key=lambda x: -x[3])          # by % >200SMA

            def pc(p):
                col_ = "#1a7a3a" if p >= 50 else ("#a66300" if p >= 35 else "#c0392b")
                return f'<td style="text-align:right;font-weight:700;color:{col_}">{p:.0f}%</td>'
            hdr = (f"<tr><td>{lbl}</td><th>Names</th>"
                   f"<th>% &gt;50SMA</th><th>% &gt;200SMA</th></tr>")
            return hdr + "".join(
                f"<tr><td>{nm}</td><td style='text-align:right'>{c}</td>"
                f"{pc(p50)}{pc(p200)}</tr>" for nm, c, p50, p200 in rows)

        segbreadth = _breadth("segment", "Segment",
                              order=["Largecap", "Midcap", "Smallcap", "Microcap"])
        secbreadth = _breadth("sector", "Sector")

    # ---- breadth history (local guru store; absent -> section omitted) ----
    bd = _load_breadth()
    breadth_html, bpayload = _breadth_block(bd, drive, idx)
    if bd is None:
        log("breadth: market_breadth.parquet not found — section omitted "
            "(build it with guru/build_market_breadth.py)")
    else:
        log(f"breadth: {bd['date'].nunique():,} days x "
            f"{bd['universe'].nunique()} universes, "
            f"{bd['date'].min().date()} -> {bd['date'].max().date()}")

    if args.dry_run:
        log(f"DRY-RUN — {len(chart_ids)} trend charts, "
            f"{len(BREADTH_CHARTS) if breadth_html else 0} breadth charts, "
            f"{len(body)} indices, {len(sect)} sectors; no file written.")
        return

    html = (_TPL.replace("__DATE__", str(row.get("date", "")))
                .replace("__HEADLINE__", headline).replace("__SUB__", sub)
                .replace("__TILES__", tiles).replace("__CHARTS__", charts_html)
                .replace("__IDXTABLE__", idxtable).replace("__SECTABLE__", sectable)
                .replace("__SEGBREADTH__", segbreadth).replace("__SECBREADTH__", secbreadth)
                .replace("__PAYLOAD__", json.dumps(payload, separators=(",", ":")))
                .replace("__DRAW__", draw)
                .replace("__BREADTH__", breadth_html)
                .replace("__BPAYLOAD__", json.dumps(bpayload, separators=(",", ":"))))
    out = os.path.join(os.path.dirname(_SD), args.out)
    with open(out, "w", encoding="utf-8") as f:
        f.write(html)
    log(f"wrote {out}  ({len(html)/1e6:.1f} MB, {len(chart_ids)} charts)")
    if not args.no_open:
        webbrowser.open("file://" + os.path.abspath(out))


if __name__ == "__main__":
    main()
