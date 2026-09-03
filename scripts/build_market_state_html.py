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


# Stance colours run bullish -> bearish. Deliberately a different scale from the
# 0-100 tiles below: a stance is a DIRECTION plus how many components agree,
# which is exactly what a single blended score cannot express.
_STANCE_COL = {"AGGRESSIVE": "#0d7a35", "CONSTRUCTIVE": "#1a7a3a",
               "NEUTRAL": "#a66300", "CAUTIOUS": "#c86a00",
               "DEFENSIVE": "#c0392b"}
_DIR_WORD = {1: ("bullish", "#1a7a3a"), 0: ("neutral", "#888"),
             -1: ("bearish", "#c0392b")}


def _stance_block(row: dict) -> str:
    """Direction, agreement and what to DO — none of which the health score says.

    Returns "" when market_state.py has not yet published the stance columns, so
    an older snapshot renders exactly as it did before."""
    stance = row.get("stance")
    if not stance or (isinstance(stance, float) and pd.isna(stance)):
        return ""
    col = _STANCE_COL.get(str(stance), "#555")
    nb, nbe = row.get("n_bullish"), row.get("n_bearish")
    ncomp, days = row.get("n_components"), row.get("stance_days")
    play = row.get("stance_playbook", "")

    def _fmt(v, suffix=""):
        return "—" if v is None or pd.isna(v) else f"{float(v):+.1f}{suffix}"

    t5, t20 = _fmt(row.get("health_trend_5d")), _fmt(row.get("health_trend_20d"))

    # per-component direction, by name — the part a blended number destroys
    rows_html = ""
    for key, label in (("nifty50_trend", "Nifty vs 200SMA"),
                       ("breadth_50sma", "Breadth >50SMA"),
                       ("highs_lows", "New highs − lows"),
                       ("vix", "India VIX"),
                       ("fii", "FII 5-day flow"),
                       ("ad_ratio", "Advance/decline")):
        d = row.get(f"{key}_dir")
        if d is None or pd.isna(d):
            word, c = "not read", "#bbb"
        else:
            word, c = _DIR_WORD.get(int(d), ("neutral", "#888"))
        rows_html += (f'<tr><td>{label}</td>'
                      f'<td style="text-align:right;color:{c};font-weight:600">'
                      f'{word}</td></tr>')

    return (
        f'<div class="card" style="margin:8px 6px;border-left:5px solid {col}">'
        f'<div style="display:flex;flex-wrap:wrap;gap:18px;align-items:baseline">'
        f'<div><span style="font-size:11px;color:#666">STANCE</span><br>'
        f'<b style="font-size:24px;color:{col}">{stance}</b></div>'
        f'<div><span style="font-size:11px;color:#666">agreement</span><br>'
        f'<b style="font-size:16px">{nb} bullish / {nbe} bearish</b>'
        f'<span style="font-size:11px;color:#888"> of {ncomp}</span></div>'
        f'<div><span style="font-size:11px;color:#666">held</span><br>'
        f'<b style="font-size:16px">{days}d</b></div>'
        f'<div><span style="font-size:11px;color:#666">health 5d / 20d</span><br>'
        f'<b style="font-size:16px">{t5} / {t20}</b></div>'
        f'</div>'
        f'<div style="margin-top:8px;padding:6px 10px;background:#f7f9fc;'
        f'border-radius:6px;font-size:13px"><b>What this means:</b> {play}</div>'
        f'<table style="margin-top:8px;max-width:420px">{rows_html}</table>'
        f'<div style="font-size:10px;color:#999;margin-top:4px">'
        f'A stance is a direction plus how many independent components agree. '
        f'The 0-100 score below blends them into one number, which cannot say '
        f'whether 50 means "calm" or "three strongly opposed readings".</div>'
        f'</div>')


def _tile(label, score, extra=""):
    v = None if score is None or pd.isna(score) else float(score)
    col = "#c0392b" if (v is None or v < 40) else ("#a66300" if v < 60 else "#1a7a3a")
    disp = "—" if v is None else f"{v:.0f}"
    return (f'<div style="flex:1;min-width:120px;background:#fff;border:1px solid #e3e7ee;'
            f'border-radius:8px;padding:8px 10px;text-align:center">'
            f'<div style="font-size:11px;color:#666">{label}</div>'
            f'<div style="font-size:22px;font-weight:800;color:{col}">{disp}</div>'
            f'<div style="font-size:10px;color:#999">{extra}</div></div>')


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
 @media(max-width:820px){.charts{grid-template-columns:1fr}}
</style></head><body><div class="wrap">
<h1>__HEADLINE__</h1>
<div style="font-size:11px;color:#888;margin:-2px 6px 8px">__SUB__</div>
__STANCE__
<div class="tiles">__TILES__</div>
<h2>Trends (daily)</h2>
<div class="charts">__CHARTS__</div>
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
        f"stance={row.get('stance', 'ABSENT')} "
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

    if args.dry_run:
        log(f"DRY-RUN — {len(chart_ids)} trend charts, "
            f"{len(body)} indices, {len(sect)} sectors; no file written.")
        return

    html = (_TPL.replace("__DATE__", str(row.get("date", "")))
                .replace("__STANCE__", _stance_block(row))
                .replace("__HEADLINE__", headline).replace("__SUB__", sub)
                .replace("__TILES__", tiles).replace("__CHARTS__", charts_html)
                .replace("__IDXTABLE__", idxtable).replace("__SECTABLE__", sectable)
                .replace("__SEGBREADTH__", segbreadth).replace("__SECBREADTH__", secbreadth)
                .replace("__PAYLOAD__", json.dumps(payload, separators=(",", ":")))
                .replace("__DRAW__", draw))
    out = os.path.join(os.path.dirname(_SD), args.out)
    with open(out, "w", encoding="utf-8") as f:
        f.write(html)
    log(f"wrote {out}  ({len(html)/1e6:.1f} MB, {len(chart_ids)} charts)")
    if not args.no_open:
        webbrowser.open("file://" + os.path.abspath(out))


if __name__ == "__main__":
    main()
