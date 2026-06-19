r"""
build_gallery.py — LOCAL one-shot chart gallery (no Streamlit, no memory cap).

Reproduces the app's Graphs "Quick Scan" gallery as ONE self-contained .html:
every ranked name as a candlestick+volume chart drawn CLIENT-SIDE (TradingView
lightweight-charts via CDN), with the same mcap / 6-rule grades / 6-quarter
table / guidance / LLM-summary cards. Writes the file locally and opens it in
your default browser. Because rendering happens in the browser, there is no
1 GB Streamlit limit to hit.

Data is pulled from the same Drive tables the app uses:
  signals/per_strategy/<strat>/latest.csv      -> ranking (n_strategies, score)
  features/latest.parquet                       -> turnover floor
  data/ohlcv/<sym>.parquet                      -> candles + volume
  fundamentals/summary.parquet                  -> market cap (screener)
  fundamentals/statements/<sym>.parquet         -> 6-quarter table
  company_repo/_index/screener_grades.parquet   -> 6-rule grade strip + name
  company_repo/_index/guidance_tracker.parquet  -> guidance (1Y + all horizons)
  company_repo/_index/gf1_guidance_statements.parquet
  company_repo/_index/announcement_ledger.parquet -> LLM summary
  universe/master_list.csv                      -> NSE/BSE exchange split

Usage:
  python scripts/build_gallery.py                 # all ranked names, open browser
  python scripts/build_gallery.py --max 200       # cap count
  python scripts/build_gallery.py --min-strats 3 --turnover 2
  python scripts/build_gallery.py --dry-run       # counts only, no html/open
"""
from __future__ import annotations

import argparse
import io
import json
import os
import sys
import webbrowser
from datetime import datetime

_SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

import pandas as pd
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(_SCRIPTS_DIR), ".env"))

from _extractor_base import (get_drive, get_or_create_subfolder, find_file,
                             download_bytes, log)
import gradation as G

_EMPTY = pd.DataFrame()


# ─────────────────────────── Drive loaders (plain) ──────────────────────────
def _folder(drive, parts):
    fid = os.environ["GDRIVE_FOLDER_ID"]
    for p in parts.split("/"):
        fid = get_or_create_subfolder(drive, fid, p)
    return fid


def _read_parquet(drive, folder, name):
    fid = find_file(drive, folder, name)
    return pd.read_parquet(io.BytesIO(download_bytes(drive, fid))) if fid else _EMPTY


def _read_csv(drive, folder, name):
    fid = find_file(drive, folder, name)
    return pd.read_csv(io.BytesIO(download_bytes(drive, fid))) if fid else _EMPTY


def _list_folder(drive, folder_id):
    out, tok = {}, None
    while True:
        resp = drive.files().list(
            q=f"'{folder_id}' in parents and trashed=false",
            fields="nextPageToken, files(id,name)", pageSize=1000,
            pageToken=tok).execute()
        for f in resp.get("files", []):
            out[f["name"]] = f["id"]
        tok = resp.get("nextPageToken")
        if not tok:
            break
    return out


def _load_signals(drive):
    sig_id = _folder(drive, "signals/per_strategy")
    subs = drive.files().list(
        q=f"'{sig_id}' in parents and mimeType='application/vnd.google-apps.folder' "
          "and trashed=false", fields="files(id,name)").execute().get("files", [])
    frames = []
    for s in subs:
        files = _list_folder(drive, s["id"])
        fid = files.get("latest.csv")
        if fid:
            df = pd.read_csv(io.BytesIO(download_bytes(drive, fid)))
            df["strategy_group"] = s["name"]
            frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else _EMPTY


def _bulk_parquet(drive, folder_id, symbols):
    """{symbol: DataFrame} — list folder once, download only needed files."""
    all_files = _list_folder(drive, folder_id)
    out = {}
    for sym in symbols:
        fid = all_files.get(f"{sym}.parquet")
        if not fid:
            out[sym] = _EMPTY
            continue
        try:
            out[sym] = pd.read_parquet(io.BytesIO(download_bytes(drive, fid)))
        except Exception:
            out[sym] = _EMPTY
    return out


# ─────────────────────────── card helpers (ported) ──────────────────────────
_QMAP = {"mar": ("Q4", 0), "jun": ("Q1", 1), "sep": ("Q2", 1), "dec": ("Q3", 1)}
_HORIZON_ORDER = ["NEXT_QTR", "1Y", "2Y", "3Y", "3Y+"]
_HLABEL = {"NEXT_QTR": "Nxt-Q", "1Y": "Yr", "2Y": "2Y", "3Y": "3Y", "3Y+": "3Y+"}
_GROWTH_METRICS = ("revenue", "sales", "pat", "profit", "volume", "earnings")
_BARE_PCT_GROWTH = ("revenue", "sales", "volume")
_CUR_TOK = ("inr", "₹", "crore", " cr", "cr.", "cr ", "gigawatt", " gw", " mw",
            " ton", "capex", " bn", " mn")
_KIND_TAG = {"growth": ("growth", "#1a7a3a"), "margin": ("margin", "#8e6e00"),
             "absolute": ("abs", "#5a4bb3"), "qual": ("note", "#777")}
import re


def _qtr_label(period):
    p = str(period).strip()
    toks = p.replace("-", " ").split()
    if len(toks) >= 2:
        mon = toks[0][:3].lower()
        yr = "".join(c for c in toks[-1] if c.isdigit())
        if mon in _QMAP and yr:
            q, bump = _QMAP[mon]
            try:
                fy = (int(yr[-2:]) if len(yr) <= 2 else int(yr) % 100) + bump
                return f"{q} FY{fy % 100:02d}"
            except ValueError:
                pass
    return p


def _g_kind(metric, value):
    m = str(metric).lower(); v = str(value).lower().strip()
    if v in ("", "na", "nan", "n/a", "-"):
        return "qual"
    if "growth" in v or "yoy" in v or "cagr" in v:
        return "growth"
    if "margin" in v or "of revenue" in v or "of sales" in v or "bps" in v:
        return "margin"
    if any(c in v for c in _CUR_TOK):
        return "absolute"
    if "%" in v or re.fullmatch(r"[\d.\-– ]+", v):
        return "growth" if any(k in m for k in _BARE_PCT_GROWTH) else "margin"
    return "qual"


def _q_order(qs):
    m = re.match(r"\s*Q([1-4])\s*FY\s*0*(\d+)", str(qs), re.I)
    return int(m.group(2)) * 100 + int(m.group(1)) if m else -1


class Cards:
    """Holds the loaded lookups and builds each card's HTML + chart data."""

    def __init__(self, grades, statements_map, guidance, gf1, ann):
        self.grades_by = self._group(grades)
        self.stmts = statements_map
        self.guid_by = {k: g for k, g in guidance.groupby(guidance["symbol"].astype(str))} \
            if not guidance.empty and "symbol" in guidance.columns else {}
        self.gf1_by = self._group(gf1)
        self.ann_by = self._group(ann)
        self.name_by = {}
        if not grades.empty and {"symbol", "company_name"} <= set(grades.columns):
            for _, r in grades.iterrows():
                self.name_by[str(r["symbol"]).upper()] = str(r.get("company_name", "") or "")

    @staticmethod
    def _group(df):
        if df is None or df.empty or "symbol" not in df.columns:
            return {}
        return {k: g for k, g in df.groupby(df["symbol"].astype(str).str.upper())}

    # mcap chip
    def mcap(self, sym, mcap_map):
        mc = mcap_map.get(sym.upper())
        if mc is None or pd.isna(mc):
            return ""
        seg = ("Largecap" if mc >= 20000 else "Midcap" if mc >= 5000
               else "Smallcap" if mc >= 500 else "Microcap")
        return (f'<span style="background:#eceff1;color:#333;padding:1px 7px;'
                f'border-radius:6px;font-size:12px;font-weight:600;margin-right:4px">'
                f'₹{mc:,.0f} Cr · {seg}</span>')

    def grades_strip(self, sym):
        g = self.grades_by.get(sym.upper(), _EMPTY)
        if g is None or g.empty:
            return ""
        r = g.iloc[-1]
        tiles = []
        tile = lambda tier, txt: (
            f'<span style="background:{G.TIER_COLOR.get(tier, "#eee")};color:#111;'
            f'padding:1px 7px;border-radius:6px;font-size:12px;font-weight:600;'
            f'margin-right:3px;display:inline-block">{txt}</span>')
        specs = [("yoy_tier", "PAT YoY", "yoy", "{:.0f}%"),
                 ("qoq_tier", "PAT QoQ", "qoq", "{:.0f}%")]
        for tcol, lbl, vcol, vfmt in specs:
            v = r.get(vcol)
            try:
                vs = vfmt.format(float(v)) if pd.notna(v) else "—"
            except (TypeError, ValueError):
                vs = "—"
            tiles.append(tile(str(r.get(tcol, "na")), f"{lbl} {vs}"))
        gm, gv, _ = self.guid_1y_headline(sym)
        if gv is None:
            tiles.append(tile("na", "Guid 1Y —"))
        else:
            tiles.append(tile(G.grade_growth(gv),
                              f"Guid·{(gm.split()[0][:4] if gm else 'Guid')} 1Y +{gv:.0f}%"))
        for tcol, lbl, vcol, vfmt in [("val_tier", "Val", "val_value", "PE {:.0f}"),
                                      ("cfo_tier", "CFO", "cfo_ratio", "{:.1f}x"),
                                      ("roe_tier", "ROE", "roe", "{:.0f}%")]:
            v = r.get(vcol)
            try:
                vs = vfmt.format(float(v)) if pd.notna(v) else "—"
            except (TypeError, ValueError):
                vs = "—"
            tiles.append(tile(str(r.get(tcol, "na")), f"{lbl} {vs}"))
        return "".join(tiles)

    def quarterly(self, sym):
        sdf = self.stmts.get(sym, _EMPTY)
        if sdf is None or sdf.empty or "statement" not in sdf.columns:
            return ""
        q = sdf[sdf["statement"] == "quarterly_pl"]
        if q.empty:
            return ""
        ser = lambda it: list(zip(q[q["line_item"] == it]["period"].astype(str),
                                  pd.to_numeric(q[q["line_item"] == it]["value"], errors="coerce")))
        rowdefs = [("Sales", "Sales"), ("Profit", "Net Profit"), ("EPS", "EPS in Rs")]
        periods = None
        for _, it in rowdefs:
            s = ser(it)
            if s:
                periods = [p for p, _ in s][-6:]
                break
        if not periods:
            return ""

        def pct(cur, prev):
            if cur is None or prev is None or pd.isna(cur) or pd.isna(prev):
                return "<td style='font-size:11px;text-align:right;color:#bbb'>—</td>"
            up = cur > prev
            col = "#1a7a3a" if up else "#c0392b"
            txt = f"{(cur / prev - 1) * 100:+.0f}%" if prev > 0 else ("▲" if up else "▼") + " n/m"
            return (f"<td style='font-size:11px;text-align:right;font-weight:700;"
                    f"color:{col}'>{txt}</td>")

        th = ("<tr><td style='font-size:11px'></td>"
              + "".join(f"<td style='font-size:11px;text-align:right;color:#666;"
                        f"font-weight:600'>{_qtr_label(p)}</td>" for p in periods)
              + "<td style='font-size:11px;text-align:right;color:#666;font-weight:600;"
                "border-left:1px solid #ccc'>YoY</td>"
              + "<td style='font-size:11px;text-align:right;color:#666;font-weight:600'>QoQ</td></tr>")
        body = ""
        for lbl, it in rowdefs:
            vals = ser(it); d = dict(vals)
            cells = "".join(
                "<td style='font-size:11px;text-align:right;font-weight:600;color:#111'>"
                + ("—" if d.get(p) is None or pd.isna(d.get(p)) else format(d.get(p), ",.0f"))
                + "</td>" for p in periods)
            seq = [v for _, v in vals]
            cur = seq[-1] if seq else None
            yoy = pct(cur, seq[-5] if len(seq) >= 5 else None).replace(
                "text-align:right;", "text-align:right;border-left:1px solid #ccc;")
            qoq = pct(cur, seq[-2] if len(seq) >= 2 else None)
            body += (f"<tr><td style='font-size:11px;color:#333'><b>{lbl}</b></td>"
                     f"{cells}{yoy}{qoq}</tr>")
        return f"<table style='border-collapse:collapse;width:100%;margin:2px 0 4px 0'>{th}{body}</table>"

    # guidance ------------------------------------------------------------
    def _guid_latest(self, sym):
        g = self.guid_by.get(sym)
        if g is None or g.empty or "metric" not in g.columns:
            return "", None
        g = g.copy()
        g["_qo"] = g["quarter"].map(_q_order) if "quarter" in g.columns else -1
        if g["_qo"].max() >= 0:
            g = g[g["_qo"] == g["_qo"].max()]
        qlab = str(g["quarter"].iloc[0]) if "quarter" in g.columns and len(g) else ""
        return qlab, g.assign(_c=pd.to_numeric(g.get("cagr_pct"), errors="coerce"))

    def guid_1y(self, sym):
        qlab, g = self._guid_latest(sym)
        if g is None:
            return "", []
        kinds = [_g_kind(m, v) for m, v in zip(g["metric"], g.get("value", ""))]
        g = g.assign(_k=kinds)
        g1 = g[g["horizon_fy"].astype(str).str.upper().eq("1Y")
               & g["guidance_type"].astype(str).str.lower().str.contains("explicit")
               & g["_k"].eq("growth")].dropna(subset=["_c"])
        out = [(m, float(s["_c"].median())) for m, s in g1.groupby(g1["metric"].astype(str).str.title())]
        return qlab, sorted(out, key=lambda x: -x[1])

    def guid_1y_headline(self, sym):
        qlab, mets = self.guid_1y(sym)
        if not mets:
            return None, None, qlab
        for m, v in mets:
            if any(k in m.lower() for k in _GROWTH_METRICS):
                return m, v, qlab
        return mets[0][0], mets[0][1], qlab

    def growth_blob(self, sym):
        g = self.grades_by.get(sym.upper(), _EMPTY)
        if g is None or g.empty:
            return ""
        r = g.iloc[-1]
        sdf = self.stmts.get(sym, _EMPTY)
        qlab = ""
        if sdf is not None and not sdf.empty and "statement" in sdf.columns:
            qd = sdf[sdf["statement"] == "quarterly_pl"]
            if not qd.empty and "period" in qd.columns:
                per = list(qd["period"].astype(str))
                qlab = _qtr_label(per[-1]) if per else ""
        hot = []
        for key, lbl in (("yoy", "PAT YoY"), ("qoq", "PAT QoQ")):
            v = pd.to_numeric(r.get(key), errors="coerce")
            if pd.notna(v) and v > 30:
                hot.append(f"{lbl} +{v:.0f}%")
        for m, c in self.guid_1y(sym)[1]:
            if c > 30 and any(k in m.lower() for k in _GROWTH_METRICS):
                hot.append(f"Guid·{m} +{c:.0f}% (1Y growth)")
        if not hot:
            return ""
        qtag = f'<span style="opacity:.85">{qlab} · </span>' if qlab else ""
        return (f'<div style="background:#1a7a3a;color:#fff;padding:4px 10px;'
                f'border-radius:6px;font-size:13.5px;font-weight:700;margin:3px 0">'
                f'🚀 {qtag}{" · ".join(hot)}</div>')

    def guidance_panel(self, sym):
        lines = []
        qlab, g = self._guid_latest(sym)
        src_q = qlab
        if g is not None and not g.empty:
            for m, sub in g.groupby(g["metric"].astype(str).str.title()):
                if not m or m.lower() == "nan":
                    continue
                ks = [_g_kind(m, vv) for vv in sub.get("value", [])]
                kind = max(set(ks), key=ks.count) if ks else "qual"
                klab, kcol = _KIND_TAG.get(kind, ("", "#777"))
                bits, seen = [], set()
                for h in _HORIZON_ORDER:
                    hs = sub[sub["horizon_fy"].astype(str).str.upper() == h]
                    for vv in hs.get("value", []):
                        t = str(vv).strip()
                        if not t or t.lower() in ("na", "nan", "n/a", "-") or t in seen:
                            continue
                        seen.add(t)
                        disp = t + "%" if kind in ("growth", "margin") and re.fullmatch(r"[\d.\-– ]+", t) else t
                        disp = disp[:34] + ("…" if len(disp) > 34 else "")
                        tag = _HLABEL.get(h, "")
                        bits.append((f'<span style="color:#888">{tag}</span> ' if tag else "")
                                    + f'<b style="color:#0d2f5c">{disp}</b>')
                        break
                if bits:
                    chip = (f'<span style="background:{kcol};color:#fff;border-radius:4px;'
                            f'padding:0 5px;font-size:10px;margin-left:4px">{klab}</span>')
                    lines.append(f'<b style="color:#1565c0">{m}</b>{chip} ' + " · ".join(bits[:4]))
                if len(lines) >= 7:
                    break
        g1 = self.gf1_by.get(sym.upper(), _EMPTY)
        if g1 is not None and not g1.empty and "exact_statement" in g1.columns:
            g2 = g1.sort_values("processed_at") if "processed_at" in g1.columns else g1
            shown = 0
            for _, gr in g2.iloc[::-1].iterrows():
                stmt = str(gr.get("exact_statement", "") or "").strip()
                if not stmt or stmt.lower() == "nan":
                    continue
                mt = str(gr.get("metric_type", "") or "").strip()
                tf = str(gr.get("timeframe", "") or "").strip()
                tag = " · ".join(t for t in (mt, tf) if t and t.lower() != "nan")
                lines.append((f'<i style="color:#1565c0">[{tag}]</i> ' if tag else "") + stmt[:200])
                shown += 1
                if shown >= 3:
                    break
        if not lines:
            return ""
        src = (f'<span style="background:#1565c0;color:#fff;border-radius:4px;'
               f'padding:0 6px;font-size:10px;margin-left:6px">concall {src_q}</span>'
               if src_q else "")
        return (f'<div style="background:#eef6ff;border-left:3px solid #1565c0;'
                f'padding:6px 10px;font-size:12.5px;color:#222;margin:3px 0;line-height:1.55">'
                f'📋 <b style="font-size:13px">Guidance / outlook</b>{src}<br>'
                + "<br>".join(lines[:7]) + "</div>")

    def llm_summary(self, sym):
        a = self.ann_by.get(sym.upper(), _EMPTY)
        if a is None or a.empty or "summary" not in a.columns:
            return ""
        a2 = a.sort_values("ann_date") if "ann_date" in a.columns else a
        row = a2.iloc[-1]
        s = str(row.get("summary", "") or "").strip()
        if not s or s.lower() == "nan":
            return ""
        adate = str(row.get("ann_date", ""))[:10]
        return (f'<div style="background:#fffde7;border-left:3px solid #f9a825;'
                f'padding:6px 10px;font-size:12.5px;color:#222;margin:3px 0;line-height:1.55">'
                f'🧠 <b style="color:#8a6d00">{adate} · {row.get("category","")}:</b> {s[:400]}</div>')


def _ohlc_arrays(odf, days):
    if odf is None or odf.empty:
        return [], []
    d = odf.sort_values("date").tail(days)
    candles, vols = [], []
    for _, r in d.iterrows():
        try:
            o, h, l, c = float(r["open"]), float(r["high"]), float(r["low"]), float(r["close"])
        except (TypeError, ValueError):
            continue
        t = str(r["date"])[:10]
        candles.append({"time": t, "open": o, "high": h, "low": l, "close": c})
        v = pd.to_numeric(r.get("volume"), errors="coerce")
        vols.append({"time": t, "value": float(v) if pd.notna(v) else 0,
                     "color": "#26a69a" if c >= o else "#ef5350"})
    return candles, vols


_TPL = """<!doctype html><html><head><meta charset="utf-8">
<title>Signals gallery __DATE__</title>
<script src="https://unpkg.com/lightweight-charts@4.1.3/dist/lightweight-charts.standalone.production.js"></script>
<style>
 body{font-family:-apple-system,Segoe UI,Roboto,Arial,sans-serif;background:#f4f6f9;margin:0;padding:12px}
 h1{font-size:16px;margin:6px 8px}
 .grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px}
 @media(max-width:1200px){.grid{grid-template-columns:repeat(2,minmax(0,1fr))}}
 @media(max-width:760px){.grid{grid-template-columns:1fr}}
 .card{background:#fff;border:1px solid #e3e7ee;border-radius:8px;padding:8px 10px;box-shadow:0 1px 3px rgba(0,0,0,.05)}
 .hd{font-size:15px;font-weight:800;color:#1a3d6e;border-top:3px solid #1a3d6e;padding-top:4px;margin-bottom:3px}
 .hd .nm{font-size:12px;font-weight:500;color:#666}
 .row{margin:2px 0}
 .chart{height:300px;margin-top:6px}
 table{border-collapse:collapse;width:100%}
</style></head><body>
<h1>📊 Signals gallery — __N__ charts — __DATE__ (rendered in your browser)</h1>
<div class="grid">__CARDS__</div>
<script>
const D=__PAYLOAD__;
function mk(id){
 const el=document.getElementById('ch'+id); if(!el||el.dataset.done)return; el.dataset.done=1;
 const ch=LightweightCharts.createChart(el,{height:300,layout:{textColor:'#333',background:{color:'#fff'}},
   rightPriceScale:{borderColor:'#eee'},timeScale:{borderColor:'#eee'},grid:{horzLines:{color:'#f2f2f2'},vertLines:{color:'#f7f7f7'}}});
 const cs=ch.addCandlestickSeries(); cs.setData((D[id]||{}).c||[]);
 const vs=ch.addHistogramSeries({priceFormat:{type:'volume'},priceScaleId:''});
 vs.priceScale().applyOptions({scaleMargins:{top:0.82,bottom:0}}); vs.setData((D[id]||{}).v||[]);
 ch.timeScale().fitContent();
}
const io=new IntersectionObserver((es)=>{es.forEach(e=>{if(e.isIntersecting)mk(e.target.id.slice(2));});},{rootMargin:'300px'});
document.querySelectorAll('.chart').forEach(el=>io.observe(el));
</script></body></html>"""


def build_html(ranked, omap, cards: Cards, mcap_map, days):
    card_html, data = [], {}
    for j, (_, rr) in enumerate(ranked.iterrows()):
        s = rr["symbol"]
        nmj = cards.name_by.get(s.upper(), "")
        meta = "".join(x for x in [
            f'<div class="hd">{j + 1}. <b>{s}</b>'
            + (f' <span class="nm">{nmj}</span>' if nmj else "") + "</div>",
            f'<div class="row">{cards.mcap(s, mcap_map)}{cards.grades_strip(s)}</div>',
            cards.quarterly(s), cards.growth_blob(s), cards.guidance_panel(s),
            cards.llm_summary(s),
        ] if x)
        c, v = _ohlc_arrays(omap.get(s, _EMPTY), days)
        data[str(j)] = {"c": c, "v": v}
        card_html.append(f'<div class="card">{meta}<div class="chart" id="ch{j}"></div></div>')
    return (_TPL.replace("__PAYLOAD__", json.dumps(data, separators=(",", ":")))
                .replace("__CARDS__", "".join(card_html))
                .replace("__N__", str(len(card_html)))
                .replace("__DATE__", datetime.now().strftime("%d %b %Y %H:%M")))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-strats", type=int, default=2)
    ap.add_argument("--zones", default="buy,add", help="comma list; '' = all")
    ap.add_argument("--timeframe-days", type=int, default=252)
    ap.add_argument("--turnover", type=float, default=1.0, help="min ₹cr/day, 0=off")
    ap.add_argument("--max", type=int, default=0, help="cap chart count, 0=all")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(_SCRIPTS_DIR),
                                                  "gallery.html"))
    ap.add_argument("--no-open", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    drive = get_drive()
    log("loading signals…")
    sig = _load_signals(drive)
    if sig.empty:
        log("No signals found."); return
    if args.zones.strip():
        sig = sig[sig["zone_type"].isin([z.strip() for z in args.zones.split(",")])]
    conv = (sig.groupby("symbol")["strategy_group"].nunique()
            .reset_index(name="n_strategies"))
    conv = conv[conv["n_strategies"] >= args.min_strats]
    best = sig.groupby("symbol")["score"].max().reset_index(name="best_score")
    conv = conv.merge(best, on="symbol", how="left")
    log(f"  {len(conv)} names with >={args.min_strats} strategies")

    if args.turnover > 0:
        feats = _read_parquet(drive, _folder(drive, "features"), "latest.parquet")
        if not feats.empty and "symbol" in feats.columns:
            if "avg_turnover_20d_cr" in feats.columns:
                turn = pd.to_numeric(feats["avg_turnover_20d_cr"], errors="coerce")
            elif {"vol_20d_avg", "close"} <= set(feats.columns):
                turn = (pd.to_numeric(feats["vol_20d_avg"], errors="coerce")
                        * pd.to_numeric(feats["close"], errors="coerce")) / 1e7
            else:
                turn = pd.Series(float("nan"), index=feats.index)
            tmap = dict(zip(feats["symbol"].astype(str), turn))
            conv = conv[conv["symbol"].astype(str).map(tmap).fillna(-1.0) >= args.turnover]
            log(f"  {len(conv)} pass Rs{args.turnover:.0f}cr turnover floor")
    if conv.empty:
        log("Nothing to render."); return

    # exchange split + rank (NSE first, then BSE-only), like the app
    uni = _read_csv(drive, _folder(drive, "universe"), "master_list.csv")
    exch = dict(zip(uni["symbol"].astype(str), uni["exchange"].astype(str))) \
        if not uni.empty and {"symbol", "exchange"} <= set(uni.columns) else {}
    conv["_exch"] = conv["symbol"].astype(str).map(exch).fillna("NSE")
    conv = conv.sort_values(["n_strategies", "best_score"], ascending=[False, False])
    nse = conv[conv["_exch"] != "BSE"]
    bse = conv[conv["_exch"] == "BSE"]
    ranked = pd.concat([nse, bse], ignore_index=True)
    if args.max > 0:
        ranked = ranked.head(args.max)
    syms = ranked["symbol"].tolist()
    log(f"  ranked {len(syms)} charts (NSE {len(nse)} + BSE {len(bse)})")

    if args.dry_run:
        log("DRY-RUN — top 10: " + ", ".join(syms[:10]))
        return

    idx = _folder(drive, "company_repo/_index")
    fund = _folder(drive, "fundamentals")
    log("loading cards (grades/guidance/announcements)…")
    grades = _read_parquet(drive, idx, "screener_grades.parquet")
    guidance = _read_parquet(drive, idx, "guidance_tracker.parquet")
    gf1 = _read_parquet(drive, idx, "gf1_guidance_statements.parquet")
    ann = _read_parquet(drive, idx, "announcement_ledger.parquet")
    summ = _read_parquet(drive, fund, "summary.parquet")
    mcap_map = {}
    if not summ.empty and {"symbol", "market_cap_cr"} <= set(summ.columns):
        for _, r in summ.iterrows():
            v = pd.to_numeric(r.get("market_cap_cr"), errors="coerce")
            if pd.notna(v) and v > 0:
                mcap_map[str(r["symbol"]).upper()] = v

    log(f"downloading OHLCV for {len(syms)} names…")
    omap = _bulk_parquet(drive, _folder(drive, "data/ohlcv"), syms)
    log("downloading statements…")
    stmts = _bulk_parquet(drive, _folder(drive, "fundamentals/statements"), syms)

    cards = Cards(grades, stmts, guidance, gf1, ann)
    log("assembling HTML…")
    html = build_html(ranked, omap, cards, mcap_map, args.timeframe_days)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(html)
    log(f"wrote {args.out}  ({len(html) / 1e6:.1f} MB, {len(syms)} charts)")
    if not args.no_open:
        webbrowser.open("file://" + os.path.abspath(args.out))
        log("opened in default browser.")


if __name__ == "__main__":
    main()
