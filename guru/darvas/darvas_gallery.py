r"""
DARVAS BOX GALLERY — darvas_gallery.py  (Project Guru, runs after Phase 1)

Draws the Darvas box staircase on a one-year candlestick chart for the strongest
trends in the market: the TOP 10% of the liquid universe by 12-month return.

  universe   market cap >= 100 cr (Phase 1 market_cap_current.csv, via sync_from_drive --light)
  liquidity  20d avg traded value >= 1 cr/day          (same gate as daily_screen / daily_scan)
  trend      12m return in the top decile of the above (daily_screen.tech_metrics, unchanged)
  boxes      detector.find_boxes (classic Darvas, 3-day confirmation)

No other filter or cap: every top-decile stock is shown, sorted by 12m return.

Reads Phase 1's Drive prices only (via guru/daily_screen.sync_prices, imported
unchanged). Writes ONE file nobody else reads:
  Drive company_repo/_daily/darvas_gallery.html   (overwritten daily)
  local guru/darvas/darvas_gallery.html

Usage:
    python guru/darvas/darvas_gallery.py --dry-run          # local cache, local html only
    python guru/darvas/darvas_gallery.py --sync             # refresh prices, build, upload
Plan: .claude/plans/darvas-gallery.md
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
GURU = os.path.dirname(HERE)
ROOT = os.path.dirname(GURU)
DATA = os.path.join(GURU, "data")
LIVE = os.path.join(DATA, "live_ohlcv")
OUT_NAME = "darvas_gallery.html"
for p in (HERE, GURU, os.path.join(ROOT, "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)

from pathlib import Path                   # noqa: E402
from dotenv import load_dotenv             # noqa: E402
load_dotenv(Path(ROOT) / ".env")           # Drive credential paths (CI writes this .env)

from detector import find_boxes, summarize  # noqa: E402
from daily_screen import tech_metrics       # noqa: E402  guru/daily_screen.py, unchanged
from build_gallery import _ohlc_arrays      # noqa: E402  scripts/build_gallery.py, unchanged

TOP_PCTILE = 0.90          # top 10% by 12m return (user 2026-10-18)
CHART_BARS = 260           # ~1 trading year on the chart


def log(m):
    print(time.strftime("%H:%M:%S"), m, flush=True)


def scan(symbols: set, min_turnover: float) -> pd.DataFrame:
    rows = []
    for f in sorted(glob.glob(os.path.join(LIVE, "*.parquet"))):
        sym = os.path.basename(f)[:-8]
        if sym not in symbols:
            continue
        try:
            df = pd.read_parquet(f)
        except Exception:                                   # noqa: BLE001
            continue
        df = df[df["close"] > 0].sort_values("date").reset_index(drop=True)
        turnover = float((df["close"] * df["volume"]).tail(20).mean() / 1e7)  # = daily_screen
        if not turnover >= min_turnover:
            continue
        m = tech_metrics(df)
        r12 = m.get("price_return_12m_pct", np.nan)
        if not r12 == r12:                                   # < 1 year of history
            continue
        rows.append({"symbol": sym, "ret_12m_pct": r12, "turnover_20d_cr": turnover,
                     "close": m["close"],
                     "last_bar": pd.Timestamp(m["last_date"]).date().isoformat()})
    return pd.DataFrame(rows)


_TPL = r"""<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Darvas boxes __DATE__</title>
<script src="https://unpkg.com/lightweight-charts@4.1.3/dist/lightweight-charts.standalone.production.js"></script>
<style>
 body{font-family:-apple-system,Segoe UI,Roboto,Arial,sans-serif;background:#f4f6f9;margin:0;padding:12px}
 h1{font-size:17px;margin:6px 8px 2px}
 .sub{font-size:12px;color:#555;margin:0 8px 10px}
 .bar{margin:0 8px 12px;font-size:12px}
 .bar button{border:1px solid #c9d2df;background:#fff;border-radius:14px;padding:3px 10px;margin:2px;cursor:pointer}
 .bar button.on{background:#1a3d6e;color:#fff;border-color:#1a3d6e}
 .grid{display:grid;grid-template-columns:1fr;gap:16px;max-width:1100px;margin:0 auto}
 .card{background:#fff;border:1px solid #e3e7ee;border-radius:8px;padding:8px 10px;box-shadow:0 1px 3px rgba(0,0,0,.05)}
 .hd{font-size:15px;font-weight:800;color:#1a3d6e;border-top:3px solid #1a3d6e;padding-top:4px}
 .hd .nm{font-size:12px;font-weight:500;color:#666}
 .bd{display:flex;flex-wrap:wrap;gap:5px;margin:5px 0 0;font-size:11.5px}
 .bd span{background:#eef2f7;border-radius:4px;padding:2px 7px}
 .bd .up{background:#e8f5e9;color:#1b5e20;font-weight:700}
 .bd .hot{background:#fff3a0;color:#7a4a00;font-weight:700}
 .bd .bad{background:#fdecea;color:#96231a}
 .wrap{position:relative;height:420px;margin-top:6px}
 .chart{position:absolute;inset:0}
 .ov{position:absolute;inset:0;pointer-events:none;overflow:hidden;z-index:5}
 .box{position:absolute;border:1.5px solid #5b5fc7;background:rgba(91,95,199,.10)}
 .box.active{border-color:#e65100;background:rgba(230,81,0,.10)}
 .box.broken{border-color:#c62828;background:rgba(198,40,40,.07);border-style:dashed}
 .box.forming{border-color:#e65100;border-style:dotted;background:none;border-bottom:none}
 .box b{position:absolute;top:-15px;left:2px;font-size:10px;color:#5b5fc7}
</style></head><body>
<h1>Darvas boxes &middot; top 10% by 12-month return</h1>
<div class="sub">Prices as of __DATE__ &middot; __N__ stocks (12m return &ge; __CUT__%) &middot;
mcap &ge; 100 cr, turnover &ge; 1 cr/day &middot; boxes: 3-day Darvas confirmation &middot;
<span style="color:#e65100">orange = current box</span>, red dashed = broken</div>
<div class="bar" id="flt"></div>
<div class="grid" id="grid"></div>
<script>
const D=__DATA__;
const grid=document.getElementById('grid');
D.forEach((s,i)=>{
  const st=s.state, cls=(st==='breakout today'||st==='at box top'||st.startsWith('above'))?'hot':(st==='box broken'?'bad':'');
  const c=document.createElement('div');c.className='card';c.dataset.state=st;c.dataset.stair=s.staircase;
  c.innerHTML=`<div class="hd">${i+1}. ${s.symbol} <span class="nm">${s.name||''}</span></div>
  <div class="bd"><span class="up">12m ${s.ret_12m_pct>=0?'+':''}${s.ret_12m_pct.toFixed(0)}%</span>
  <span>&#8377;${s.close.toLocaleString('en-IN')}</span>
  ${s.mcap?`<span>mcap ${Math.round(s.mcap).toLocaleString('en-IN')} cr</span>`:''}
  <span class="${cls}">${st}</span>
  <span>${s.staircase} ascending box${s.staircase==1?'':'es'} (${s.n_boxes} total)</span>
  ${s.resistance!=null?`<span>resistance &#8377;${s.resistance}</span>`:''}
  ${s.support!=null?`<span>support &#8377;${s.support}</span>`:''}
  ${s.stale?`<span class="bad">last bar ${s.last_bar}</span>`:''}
  <a href="https://www.tradingview.com/chart/?symbol=NSE:${s.symbol}" target="_blank" style="font-size:11px">TV&#8599;</a></div>
  <div class="wrap"><div class="chart" id="c${i}"></div><div class="ov" id="o${i}"></div></div>`;
  grid.appendChild(c);
});
function mk(i){
  const s=D[i], el=document.getElementById('c'+i);
  if(el.dataset.done)return; el.dataset.done=1;
  const ch=LightweightCharts.createChart(el,{autoSize:true,layout:{background:{color:'#fff'},textColor:'#333'},
    grid:{vertLines:{color:'#f0f0f0'},horzLines:{color:'#f0f0f0'}},rightPriceScale:{scaleMargins:{top:.08,bottom:.22}},
    timeScale:{borderColor:'#ddd'}});
  const cs=ch.addCandlestickSeries({upColor:'#26a69a',downColor:'#ef5350',borderVisible:false,wickUpColor:'#26a69a',wickDownColor:'#ef5350'});
  cs.setData(s.r.map(a=>({time:a[0],open:a[1],high:a[2],low:a[3],close:a[4]})));
  const ln=e=>s.r.map((a,j)=>e[j]==null?null:{time:a[0],value:e[j]}).filter(Boolean);
  ch.addLineSeries({color:'#2962ff',lineWidth:1,priceLineVisible:false,lastValueVisible:false}).setData(ln(s.e20));
  ch.addLineSeries({color:'#ff9800',lineWidth:1,priceLineVisible:false,lastValueVisible:false}).setData(ln(s.e50));
  const vs=ch.addHistogramSeries({priceFormat:{type:'volume'},priceScaleId:'v'});
  ch.priceScale('v').applyOptions({scaleMargins:{top:.82,bottom:0}});
  vs.setData(s.r.map(a=>({time:a[0],value:a[5],color:a[4]>=a[1]?'#26a69a':'#ef5350'})));
  cs.setMarkers(s.boxes.filter(b=>b.breakout_date).map(b=>({time:b.breakout_date,position:'belowBar',color:'#1b5e20',shape:'arrowUp',text:'BO'})));
  const ov=document.getElementById('o'+i), ts=ch.timeScale();
  function x(t){const v=ts.timeToCoordinate(t);return v==null?null:v;}
  function draw(){
    ov.innerHTML='';
    s.boxes.forEach((b,k)=>{
      const x1=x(b.start), x2=x(b.end); if(x1==null||x2==null)return;
      const yt=cs.priceToCoordinate(b.top), yb=b.bottom==null?yt+1:cs.priceToCoordinate(b.bottom);
      if(yt==null||yb==null)return;
      const d=document.createElement('div');d.className='box '+b.status;
      d.style.left=x1+'px';d.style.width=Math.max(2,x2-x1)+'px';d.style.top=yt+'px';d.style.height=Math.max(1,yb-yt)+'px';
      d.innerHTML='<b>'+(k+1)+'</b>';ov.appendChild(d);
    });
  }
  ts.fitContent();
  ts.subscribeVisibleLogicalRangeChange(()=>requestAnimationFrame(draw));
  new ResizeObserver(()=>requestAnimationFrame(draw)).observe(el);
  setTimeout(draw,50);
}
const io=new IntersectionObserver(es=>{es.forEach(e=>{if(e.isIntersecting)mk(+e.target.id.slice(1));});},{rootMargin:'400px'});
document.querySelectorAll('.chart').forEach(e=>io.observe(e));
const F=[['all',()=>true],['at top / breakout',c=>/at box top|breakout|above/.test(c.dataset.state)],
 ['3+ ascending boxes',c=>+c.dataset.stair>=3],['inside box',c=>c.dataset.state==='inside box']];
const fb=document.getElementById('flt');
F.forEach(([n,fn],k)=>{const b=document.createElement('button');b.textContent=n;if(!k)b.className='on';
 b.onclick=()=>{fb.querySelectorAll('button').forEach(x=>x.className='');b.className='on';
  document.querySelectorAll('.card').forEach(c=>c.style.display=fn(c)?'':'none');};fb.appendChild(b);});
</script></body></html>"""


def build_html(top: pd.DataFrame, as_of: str, cut: float) -> str:
    data = []
    for r in top.itertuples(index=False):
        df = pd.read_parquet(os.path.join(LIVE, f"{r.symbol}.parquet"))
        df = df[df["close"] > 0].sort_values("date")
        candles, vols, e20, e50 = _ohlc_arrays(df, CHART_BARS)
        # compact rows [date, o, h, l, c, vol] + aligned EMA values; the page expands
        # them back into lightweight-charts objects (object form was ~8 MB for 142 stocks)
        e20m = {x["time"]: x["value"] for x in e20}
        e50m = {x["time"]: x["value"] for x in e50}
        rows = [[c["time"], round(c["open"], 2), round(c["high"], 2), round(c["low"], 2),
                 round(c["close"], 2), int(v["value"])] for c, v in zip(candles, vols)]
        candles, vols = rows, None
        e20 = [e20m.get(x[0]) for x in rows]
        e50 = [e50m.get(x[0]) for x in rows]
        for b in r.boxes:
            for k in ("top", "bottom"):
                if b[k] is not None:
                    b[k] = round(b[k], 2)
        data.append({"symbol": r.symbol, "name": r.name, "ret_12m_pct": round(r.ret_12m_pct, 1),
                     "close": round(r.close, 2), "mcap": None if pd.isna(r.market_cap_cr) else r.market_cap_cr,
                     "state": r.state, "staircase": int(r.staircase), "n_boxes": int(r.n_boxes),
                     "resistance": None if pd.isna(r.resistance) else r.resistance,
                     "support": None if pd.isna(r.support) else r.support,
                     "last_bar": r.last_bar, "stale": bool(r.stale),
                     "boxes": r.boxes, "r": candles, "e20": e20, "e50": e50})
    return (_TPL.replace("__DATE__", as_of).replace("__N__", str(len(top)))
            .replace("__CUT__", f"{cut:.0f}")
            .replace("__DATA__", json.dumps(data, separators=(",", ":"), default=str)))


def publish(html: str) -> None:
    from _extractor_base import get_drive, get_or_create_subfolder, find_file, upload_bytes
    d = get_drive()
    root = os.environ["GDRIVE_FOLDER_ID"]
    day = get_or_create_subfolder(d, get_or_create_subfolder(d, root, "company_repo"), "_daily")
    upload_bytes(d, day, OUT_NAME, html.encode("utf-8"), "text/html",
                 existing_id=find_file(d, day, OUT_NAME))
    log(f"published Drive company_repo/_daily/{OUT_NAME}")


def main():
    ap = argparse.ArgumentParser(description="Darvas box gallery for the top 12m-return decile")
    ap.add_argument("--sync", action="store_true", help="refresh Phase 1 prices from Drive first")
    ap.add_argument("--min-mcap", type=float, default=100.0, help="Rs crore")
    ap.add_argument("--min-turnover", type=float, default=1.0, help="Rs crore/day, 20d avg")
    ap.add_argument("--no-publish", action="store_true", help="write the local html only")
    ap.add_argument("--dry-run", action="store_true",
                    help="no Drive at all: local price cache, local html only")
    a = ap.parse_args()

    mc = pd.read_csv(os.path.join(DATA, "market_cap_current.csv"))
    mc["market_cap_cr"] = pd.to_numeric(mc["market_cap_cr"], errors="coerce")
    uni = mc[mc["market_cap_cr"] >= a.min_mcap]
    symbols = set(uni["symbol"].astype(str))
    mcap = dict(zip(uni["symbol"].astype(str), uni["market_cap_cr"].round(0)))
    log(f"universe: {len(mc):,} -> {len(symbols):,} with mcap >= {a.min_mcap:g} cr")

    if a.sync and not a.dry_run:
        from daily_screen import sync_prices
        sync_prices(symbols)

    sym2name = {}
    up = os.path.join(DATA, "universe_hist.parquet")
    if os.path.exists(up):                     # cosmetic; CI has no universe_hist
        u = pd.read_parquet(up, columns=["nse_symbol", "name"])
        u = u[u["nse_symbol"].notna()].drop_duplicates("nse_symbol")
        sym2name = dict(zip(u["nse_symbol"].astype(str), u["name"]))

    t0 = time.time()
    H = scan(symbols, a.min_turnover)
    if H.empty:
        log("no stock passed the gates"); return
    as_of = pd.Series(H["last_bar"]).mode().iloc[0]
    cut = float(H["ret_12m_pct"].quantile(TOP_PCTILE))
    top = H[H["ret_12m_pct"] >= cut].sort_values("ret_12m_pct", ascending=False).copy()
    log(f"liquid with 1y history: {len(H):,}; top-decile cutoff 12m >= {cut:.1f}% -> {len(top)} stocks")

    recs = []
    for r in top.itertuples(index=False):
        df = pd.read_parquet(os.path.join(LIVE, f"{r.symbol}.parquet"))
        df = df[df["close"] > 0].sort_values("date")
        boxes = find_boxes(df)
        recs.append({**summarize(df, boxes), "boxes": boxes})
    top = pd.concat([top.reset_index(drop=True), pd.DataFrame(recs)], axis=1)
    top["name"] = top["symbol"].map(sym2name).fillna("")
    top["market_cap_cr"] = top["symbol"].map(mcap)
    top["stale"] = top["last_bar"] < as_of
    log(f"scanned in {time.time() - t0:.0f}s, prices as of {as_of}")
    print(top["state"].value_counts().to_string())
    print(top["staircase"].value_counts().sort_index().rename("ascending boxes").to_string())
    print(top[["symbol", "ret_12m_pct", "state", "staircase", "n_boxes", "resistance",
               "support"]].head(20).to_string(index=False))

    html = build_html(top, as_of, cut)
    out = os.path.join(HERE, OUT_NAME)
    with open(out, "w", encoding="utf-8") as fh:
        fh.write(html)
    log(f"wrote {out} ({len(html.encode('utf-8')) / 1e6:.2f} MB)")
    if a.dry_run or a.no_publish:
        log("DRY RUN / no-publish: nothing sent to Drive"); return
    publish(html)


if __name__ == "__main__":
    main()
