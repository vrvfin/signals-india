"""
fetch_phase1_report.py — Generate a local HTML report from Phase 1 outputs.

Two sections:
  1. Conviction Signals  — stocks flagged by ≥ MIN_STRATEGIES strategies
                           in buy/add zones, sorted by conviction
  2. My Portfolio        — holdings overlaid with RS rank, signals, features

Opens in your default browser. Print to PDF via Ctrl+P → Save as PDF.

Usage:
    python scripts/fetch_phase1_report.py                        # tables only
    python scripts/fetch_phase1_report.py --with-charts          # + Plotly charts (top 10)
    python scripts/fetch_phase1_report.py --with-charts --max-charts 20
    python scripts/fetch_phase1_report.py --min-strats 3         # stricter filter
    python scripts/fetch_phase1_report.py --no-portfolio         # signals only
    python scripts/fetch_phase1_report.py --no-open              # save only
"""

from __future__ import annotations

import argparse
import io
import os
import subprocess
import sys
import tempfile
import webbrowser
from datetime import datetime
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dotenv import load_dotenv
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

OUTPUT_DIR = Path(r"D:\EMA_Screener\Reports\signals-india\phase1_reports")
SCOPES     = ["https://www.googleapis.com/auth/drive"]

ZONE_COLORS = {
    "buy":       "#1a9e3f",
    "add":       "#0e7a6e",
    "hold":      "#6c757d",
    "stop_loss": "#c0392b",
    "exit":      "#e67e22",
    "sell":      "#922b21",
}


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")


# ── Drive auth ────────────────────────────────────────────────────────────────

def get_drive():
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
    tk_json = os.environ.get("GDRIVE_OAUTH_TOKEN_JSON")
    cs_json = os.environ.get("GDRIVE_OAUTH_CLIENT_SECRET_JSON")
    if tk_json and cs_json:
        import json
        creds = Credentials.from_authorized_user_info(json.loads(tk_json), SCOPES)
        if creds.expired and creds.refresh_token:
            creds.refresh(Request())
        return build("drive", "v3", credentials=creds, cache_discovery=False)
    tk_path = Path(os.environ["GDRIVE_OAUTH_TOKEN_PATH"])
    creds   = None
    if tk_path.exists():
        creds = Credentials.from_authorized_user_file(str(tk_path), SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            from google_auth_oauthlib.flow import InstalledAppFlow
            cs_path = Path(os.environ["GDRIVE_OAUTH_CLIENT_SECRET_PATH"])
            flow    = InstalledAppFlow.from_client_secrets_file(str(cs_path), SCOPES)
            creds   = flow.run_local_server(port=0)
        tk_path.write_text(creds.to_json())
    return build("drive", "v3", credentials=creds, cache_discovery=False)


def _find_subfolder(drive, parent_id: str, name: str) -> str | None:
    q = (f"name='{name}' and '{parent_id}' in parents "
         f"and mimeType='application/vnd.google-apps.folder' and trashed=false")
    r = drive.files().list(q=q, fields="files(id)").execute().get("files", [])
    return r[0]["id"] if r else None


def _list_folder(drive, folder_id: str) -> dict[str, str]:
    out, token = {}, None
    while True:
        r = drive.files().list(
            q=f"'{folder_id}' in parents and trashed=false",
            fields="nextPageToken, files(id,name)",
            pageSize=1000, pageToken=token,
        ).execute()
        for f in r.get("files", []):
            out[f["name"]] = f["id"]
        token = r.get("nextPageToken")
        if not token:
            break
    return out


def _dl(drive, file_id: str) -> bytes:
    req = drive.files().get_media(fileId=file_id)
    fh  = io.BytesIO()
    dl  = MediaIoBaseDownload(fh, req)
    done = False
    while not done:
        _, done = dl.next_chunk()
    return fh.getvalue()


# ── Data loaders ──────────────────────────────────────────────────────────────

def load_all_signals(drive, folder_id: str) -> pd.DataFrame:
    log("Loading strategy signals…")
    signals_id  = _find_subfolder(drive, folder_id, "signals")
    if not signals_id:
        return pd.DataFrame()
    per_strat_id = _find_subfolder(drive, signals_id, "per_strategy")
    if not per_strat_id:
        return pd.DataFrame()
    subs = drive.files().list(
        q=(f"'{per_strat_id}' in parents "
           f"and mimeType='application/vnd.google-apps.folder' and trashed=false"),
        fields="files(id,name)",
    ).execute().get("files", [])
    frames = []
    for s in subs:
        files  = _list_folder(drive, s["id"])
        fid    = files.get("latest.csv")
        if fid:
            df = pd.read_csv(io.BytesIO(_dl(drive, fid)))
            df["strategy_group"] = s["name"]
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    result = pd.concat(frames, ignore_index=True)
    log(f"  {len(result)} signal rows across {result['strategy_group'].nunique()} strategies")
    return result


def load_features(drive, folder_id: str) -> pd.DataFrame:
    log("Loading features…")
    feat_id = _find_subfolder(drive, folder_id, "features")
    if not feat_id:
        return pd.DataFrame()
    files = _list_folder(drive, feat_id)
    fid   = files.get("latest.parquet")
    if not fid:
        return pd.DataFrame()
    df = pd.read_parquet(io.BytesIO(_dl(drive, fid)))
    log(f"  {len(df)} feature rows")
    return df


def load_portfolio(drive, folder_id: str) -> pd.DataFrame:
    log("Loading portfolio…")
    pf_id = _find_subfolder(drive, folder_id, "portfolio")
    if not pf_id:
        log("  No portfolio/ folder on Drive — skipping portfolio section")
        return pd.DataFrame()
    files = drive.files().list(
        q=f"'{pf_id}' in parents and trashed=false",
        fields="files(id, name, modifiedTime)",
        orderBy="modifiedTime desc",
    ).execute().get("files", [])
    target = next((f for f in files
                   if f["name"].lower().endswith((".xls", ".xlsx", ".csv"))), None)
    if not target:
        log("  No .xls/.xlsx/.csv in portfolio/ — skipping")
        return pd.DataFrame()
    log(f"  Using: {target['name']}")
    raw  = _dl(drive, target["id"])
    fn   = target["name"].lower()
    eng  = "xlrd" if fn.endswith(".xls") else ("openpyxl" if fn.endswith(".xlsx") else "csv")
    if eng == "csv":
        df_raw = pd.read_csv(io.BytesIO(raw), header=None)
    else:
        df_raw = pd.read_excel(io.BytesIO(raw), engine=eng, header=None)
    hrow = None
    for i, row in df_raw.iterrows():
        if any(str(v).strip().upper() == "ISIN" for v in row.dropna()):
            hrow = i; break
    if hrow is None:
        log("  Could not find ISIN header — skipping portfolio")
        return pd.DataFrame()
    df = (pd.read_csv(io.BytesIO(raw), header=hrow) if eng == "csv"
          else pd.read_excel(io.BytesIO(raw), engine=eng, header=hrow))
    rename = {
        "ISIN": "isin", "Company name": "screener_name",
        "Stock/ETF Name": "screener_name",
        "Last price": "screener_last_price", "Market Price": "screener_last_price",
        "Rating": "screener_rating", "1D Change (%)": "one_d_change_pct",
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
    df = df.dropna(subset=["isin"]).copy()
    df["isin"] = df["isin"].astype(str).str.strip()
    return df


def load_universe(drive, folder_id: str) -> pd.DataFrame:
    uni_id = _find_subfolder(drive, folder_id, "universe")
    if not uni_id:
        return pd.DataFrame()
    files = _list_folder(drive, uni_id)
    fid   = files.get("master_list.csv")
    if not fid:
        return pd.DataFrame()
    return pd.read_csv(io.BytesIO(_dl(drive, fid)))


# ── OHLCV + chart helpers ─────────────────────────────────────────────────────

def load_ohlcv(drive, folder_id: str, symbol: str) -> pd.DataFrame:
    """Download OHLCV parquet for one symbol. Returns empty DF if missing."""
    data_id = _find_subfolder(drive, folder_id, "data")
    if not data_id:
        return pd.DataFrame()
    ohlcv_id = _find_subfolder(drive, data_id, "ohlcv")
    if not ohlcv_id:
        return pd.DataFrame()
    files = _list_folder(drive, ohlcv_id)
    fid   = files.get(f"{symbol}.parquet")
    if not fid:
        return pd.DataFrame()
    return pd.read_parquet(io.BytesIO(_dl(drive, fid)))


def build_chart_html(symbol: str, ohlcv: pd.DataFrame,
                     sym_signals: pd.DataFrame,
                     first_chart: bool = False,
                     days: int = 180) -> str:
    """Return Plotly chart as an HTML div string (no <html> wrapper)."""
    df = ohlcv.sort_values("date").tail(days).reset_index(drop=True)
    if df.empty:
        return f"<p style='color:#aaa'>No OHLCV data for {symbol}</p>"

    df["ema_20"]  = df["close"].ewm(span=20).mean()
    df["ema_50"]  = df["close"].ewm(span=50).mean()
    df["sma_200"] = df["close"].rolling(200).mean()

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        row_heights=[0.75, 0.25], vertical_spacing=0.03,
    )

    # Candlestick
    fig.add_trace(go.Candlestick(
        x=df["date"], open=df["open"], high=df["high"],
        low=df["low"], close=df["close"],
        name="Price", showlegend=False,
        increasing_line_color="#1a9e3f", decreasing_line_color="#c0392b",
    ), row=1, col=1)

    fig.add_trace(go.Scatter(x=df["date"], y=df["ema_20"],  name="20 EMA",
                             line=dict(color="#2980b9", width=1.2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df["date"], y=df["ema_50"],  name="50 EMA",
                             line=dict(color="#8e44ad", width=1.2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df["date"], y=df["sma_200"], name="200 SMA",
                             line=dict(color="#e67e22", width=1.2, dash="dash")),
                  row=1, col=1)

    # Signal lines
    if not sym_signals.empty:
        for _, sig in sym_signals.iterrows():
            zt    = sig.get("zone_type", "")
            entry = sig.get("entry")
            stop  = sig.get("stop")
            col   = ZONE_COLORS.get(zt, "#666")
            strat = sig.get("strategy", sig.get("strategy_group", ""))
            if pd.notna(entry):
                fig.add_hline(y=entry, line=dict(color=col, width=1),
                              annotation_text=f"{strat}:{zt}",
                              annotation_position="right", row=1, col=1)
            if pd.notna(stop):
                fig.add_hline(y=stop,
                              line=dict(color=ZONE_COLORS["stop_loss"], width=1, dash="dot"),
                              annotation_text="stop",
                              annotation_position="right", row=1, col=1)

    # Volume
    colors = ["#1a9e3f" if c >= o else "#c0392b"
              for c, o in zip(df["close"], df["open"])]
    fig.add_trace(go.Bar(x=df["date"], y=df["volume"], name="Vol",
                         marker_color=colors, showlegend=False), row=2, col=1)

    fig.update_layout(
        height=420, xaxis_rangeslider_visible=False,
        margin=dict(l=10, r=80, t=10, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    font=dict(size=11)),
        plot_bgcolor="#fafbfc", paper_bgcolor="#fff",
    )
    fig.update_yaxes(title_text="Price", row=1, col=1, tickfont=dict(size=10))
    fig.update_yaxes(title_text="Vol",   row=2, col=1, tickfont=dict(size=10))

    # First chart loads Plotly.js from CDN; subsequent charts reuse it
    include_js = "cdn" if first_chart else False
    return fig.to_html(full_html=False, include_plotlyjs=include_js,
                       config={"displayModeBar": False})


# ── HTML generation ───────────────────────────────────────────────────────────

_CSS = """
* { box-sizing: border-box; margin: 0; padding: 0; }
body {
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Arial, sans-serif;
  font-size: 13px; color: #1a1a2e; background: #f5f6fa; padding: 24px 32px;
}
h1 { font-size: 22px; margin-bottom: 4px; color: #1a1a2e; }
h2 { font-size: 16px; margin: 28px 0 10px; color: #2c3e50;
     border-bottom: 2px solid #3498db; padding-bottom: 5px; }
.meta { color: #666; font-size: 12px; margin-bottom: 20px; }
table {
  border-collapse: collapse; width: 100%; margin-bottom: 24px;
  background: #fff; border-radius: 6px; overflow: hidden;
  box-shadow: 0 1px 4px rgba(0,0,0,.08);
}
th { background: #2c3e50; color: #fff; padding: 8px 10px;
     text-align: left; font-size: 12px; font-weight: 600; white-space: nowrap; }
td { padding: 7px 10px; border-bottom: 1px solid #eef0f4; vertical-align: top; }
tr:last-child td { border-bottom: none; }
tr:nth-child(even) td { background: #f8f9fc; }
tr:hover td { background: #eaf4fb; }
.sym { font-weight: 700; color: #2c3e50; font-size: 13px; }
.num { text-align: right; font-variant-numeric: tabular-nums; }
.green { color: #1a9e3f; font-weight: 600; }
.red   { color: #c0392b; font-weight: 600; }
.badge {
  display: inline-block; padding: 2px 8px; border-radius: 10px;
  font-size: 11px; font-weight: 600; color: #fff; margin: 1px 2px; white-space: nowrap;
}
.na { color: #aaa; }

/* ── Collapsible chart blocks ── */
details.chart-block {
  background: #fff; border: 1px solid #e0e4ec; border-radius: 6px;
  margin: 8px 0; overflow: hidden;
  box-shadow: 0 1px 3px rgba(0,0,0,.06);
}
details.chart-block summary {
  cursor: pointer; padding: 10px 14px; font-weight: 600;
  font-size: 13px; color: #2c3e50; user-select: none;
  display: flex; align-items: center; gap: 10px;
  background: #f8f9fc; border-bottom: 1px solid transparent;
  list-style: none;
}
details.chart-block summary::-webkit-details-marker { display: none; }
details.chart-block summary::before {
  content: "▶"; font-size: 10px; color: #3498db;
  transition: transform .15s; display: inline-block;
}
details.chart-block[open] summary::before { transform: rotate(90deg); }
details.chart-block[open] summary { border-bottom-color: #e0e4ec; }
details.chart-block .chart-inner { padding: 8px 4px 4px; }
.chart-meta { font-size: 11px; color: #666; font-weight: 400; margin-left: 4px; }

@media print {
  body { background: #fff; padding: 12px; }
  table { box-shadow: none; }
  details.chart-block { border: none; box-shadow: none; }
  details.chart-block summary { background: #f0f0f0; }
}
"""

_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Signals India — Phase 1 Report {date}</title>
<style>{css}</style>
</head>
<body>
<h1>Signals India — Phase 1 Daily Report</h1>
<p class="meta">Generated {datetime} &nbsp;·&nbsp; {n_signals} total signals &nbsp;·&nbsp;
Print to PDF: Ctrl+P → Save as PDF (Landscape). Click ▶ to expand any chart.</p>

{conviction_section}

{portfolio_section}

<script>
/* Re-render Plotly chart when a details block is opened.
   Charts inside hidden elements don't render until visible. */
document.addEventListener('toggle', function(e) {{
  if (e.target.tagName === 'DETAILS' && e.target.open) {{
    var gd = e.target.querySelector('.plotly-graph-div');
    if (gd && window.Plotly) Plotly.relayout(gd, {{autosize: true}});
  }}
}}, true);
</script>
</body>
</html>
"""


def _badge(zone: str) -> str:
    col = ZONE_COLORS.get(zone.lower(), "#666")
    return f'<span class="badge" style="background:{col}">{zone.upper()}</span>'


def _pct(val, good_positive: bool = True) -> str:
    if pd.isna(val):
        return '<span class="na">—</span>'
    cls = ("green" if (val >= 0) == good_positive else "red")
    return f'<span class="{cls}">{val:+.1f}%</span>'


def _num(val, fmt=".0f") -> str:
    if pd.isna(val):
        return '<span class="na">—</span>'
    return f"{val:{fmt}}"


def build_conviction_section(signals: pd.DataFrame,
                              features: pd.DataFrame,
                              min_strats: int,
                              drive=None,
                              folder_id: str = "",
                              max_charts: int = 0) -> str:
    if signals.empty:
        return "<h2>Conviction Signals</h2><p>No signals data found.</p>"

    buy_add = signals[signals["zone_type"].isin(["buy", "add"])].copy()
    if buy_add.empty:
        return "<h2>Conviction Signals</h2><p>No buy/add signals today.</p>"

    # Count strategies per symbol
    strat_count = (buy_add.groupby("symbol")["strategy_group"]
                   .nunique().reset_index(name="n_strategies"))
    strat_count = strat_count[strat_count["n_strategies"] >= min_strats]

    # Best score per symbol
    best_score = buy_add.groupby("symbol")["score"].max().reset_index(name="best_score")

    # Zone summary per symbol
    zones = (buy_add.groupby("symbol")["zone_type"]
             .apply(lambda x: sorted(set(x))).reset_index(name="zones"))

    # Strategy names per symbol
    strats = (buy_add.groupby("symbol")["strategy_group"]
              .apply(lambda x: sorted(set(x))).reset_index(name="strategies"))

    conv = (strat_count
            .merge(best_score, on="symbol")
            .merge(zones, on="symbol")
            .merge(strats, on="symbol")
            .sort_values(["n_strategies", "best_score"], ascending=[False, False])
            .reset_index(drop=True))

    # Overlay features
    if not features.empty:
        feat_cols = [c for c in ["symbol", "close", "return_3m_pct", "rs_rank_3m",
                                  "dist_from_52w_high_pct", "above_200sma", "adr_pct_20"]
                     if c in features.columns]
        conv = conv.merge(features[feat_cols], on="symbol", how="left")

    total_buy_add = len(strat_count[strat_count["n_strategies"] >= 1])
    title = (f"<h2>Conviction Signals &nbsp;<small style='font-weight:normal;color:#666;'>"
             f"≥{min_strats} strategies · {len(conv)} stocks "
             f"(of {total_buy_add} with any buy/add signal)</small></h2>")

    rows = []
    for _, r in conv.iterrows():
        zones_html  = " ".join(_badge(z) for z in r["zones"])
        strats_html = ", ".join(f"<code>{s}</code>" for s in r["strategies"])

        close_str  = f"₹{r['close']:.2f}"   if "close"  in r.index and pd.notna(r.get("close"))  else "—"
        ret_str    = _pct(r.get("return_3m_pct"))
        rs_str     = _num(r.get("rs_rank_3m"), ".0f")
        dist_str   = _pct(r.get("dist_from_52w_high_pct"), good_positive=False)
        above_str  = ("✅" if r.get("above_200sma") else "❌") if "above_200sma" in r.index else "—"

        rows.append(
            f"<tr>"
            f"<td class='sym'>{r['symbol']}</td>"
            f"<td class='num'>{r['n_strategies']}</td>"
            f"<td>{zones_html}</td>"
            f"<td>{strats_html}</td>"
            f"<td class='num'>{r['best_score']:.1f}</td>"
            f"<td class='num'>{close_str}</td>"
            f"<td class='num'>{ret_str}</td>"
            f"<td class='num'>{rs_str}</td>"
            f"<td class='num'>{dist_str}</td>"
            f"<td class='num'>{above_str}</td>"
            f"</tr>"
        )

    table = (
        "<table><thead><tr>"
        "<th>Symbol</th><th>#Strats</th><th>Zones</th><th>Strategies</th>"
        "<th>Score</th><th>Close</th><th>3M Ret</th><th>RS Rank</th>"
        "<th>vs 52w High</th><th>Above 200</th>"
        "</tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table>"
    )

    # ── Embed charts as collapsible <details> blocks ─────────────────────────
    charts_html = ""
    if max_charts > 0 and drive and folder_id:
        n = min(max_charts, len(conv))
        log(f"  Downloading OHLCV + building charts for {n} stocks…")
        chart_syms = conv["symbol"].tolist()[:n]
        first = True
        for i, sym in enumerate(chart_syms, 1):
            log(f"    [{i}/{n}] {sym}")
            ohlcv    = load_ohlcv(drive, folder_id, sym)
            sym_sigs = (signals[signals["symbol"] == sym]
                        if not signals.empty else pd.DataFrame())
            chart_div  = build_chart_html(sym, ohlcv, sym_sigs, first_chart=first)
            row        = conv[conv["symbol"] == sym].iloc[0]
            strats_str = ", ".join(row["strategies"])
            zones_str  = " / ".join(row["zones"])
            n_str      = int(row["n_strategies"])
            charts_html += (
                f"<details class='chart-block'>"
                f"<summary>{sym}"
                f"<span class='chart-meta'>"
                f"· {n_str} strategies · {zones_str} · {strats_str}"
                f"</span></summary>"
                f"<div class='chart-inner'>{chart_div}</div>"
                f"</details>\n"
            )
            first = False

    return title + table + charts_html


def build_portfolio_section(portfolio: pd.DataFrame,
                             signals: pd.DataFrame,
                             features: pd.DataFrame,
                             universe: pd.DataFrame,
                             drive=None,
                             folder_id: str = "",
                             with_charts: bool = False) -> str:
    if portfolio.empty:
        return ""

    # Resolve ISIN → symbol
    if not universe.empty and "isin" in universe.columns and "symbol" in universe.columns:
        pf = portfolio.merge(universe[["isin", "symbol", "name"]], on="isin", how="left")
    else:
        pf = portfolio.copy()
        if "symbol" not in pf.columns:
            pf["symbol"] = pf.get("screener_name", "")

    pf = pf.dropna(subset=["symbol"]).copy()
    if pf.empty:
        return "<h2>Portfolio</h2><p>No holdings resolved to NSE symbols.</p>"

    # Overlay features
    if not features.empty:
        feat_cols = [c for c in ["symbol", "close", "return_1m_pct", "return_3m_pct",
                                  "rs_rank_3m", "dist_from_52w_high_pct",
                                  "above_200sma", "adr_pct_20"]
                     if c in features.columns]
        pf = pf.merge(features[feat_cols], on="symbol", how="left")

    # Overlay signals
    if not signals.empty:
        buy_add = signals[signals["zone_type"].isin(["buy", "add"])]
        sig_sum = (signals.groupby("symbol")
                   .agg(n_strategies=("strategy_group", "nunique"),
                        zones=("zone_type", lambda x: "/".join(sorted(set(x.dropna())))))
                   .reset_index())
        pf = pf.merge(sig_sum, on="symbol", how="left")
    else:
        pf["n_strategies"] = 0
        pf["zones"] = ""

    pf["n_strategies"] = pf["n_strategies"].fillna(0).astype(int)
    pf = pf.sort_values("return_3m_pct", ascending=False, na_position="last")

    n_above = int(pf["above_200sma"].fillna(False).sum()) if "above_200sma" in pf.columns else "—"
    n_flagged = int((pf["n_strategies"] > 0).sum())

    title = (f"<h2>Portfolio &nbsp;<small style='font-weight:normal;color:#666;'>"
             f"{len(pf)} holdings &nbsp;·&nbsp; "
             f"{n_above} above 200 SMA &nbsp;·&nbsp; "
             f"{n_flagged} flagged today</small></h2>")

    rows = []
    for _, r in pf.iterrows():
        name_str  = str(r.get("name", r.get("screener_name", "")))[:30]
        close_str = f"₹{r['screener_last_price']:.2f}" if pd.notna(r.get("screener_last_price")) else "—"
        ret1_str  = _pct(r.get("return_1m_pct"))
        ret3_str  = _pct(r.get("return_3m_pct"))
        rs_str    = _num(r.get("rs_rank_3m"), ".0f")
        dist_str  = _pct(r.get("dist_from_52w_high_pct"), good_positive=False)
        above_str = ("✅" if r.get("above_200sma") else "❌") if "above_200sma" in r.index else "—"
        n_str     = str(int(r["n_strategies"])) if r["n_strategies"] > 0 else "—"
        zones_str = " ".join(_badge(z) for z in str(r.get("zones", "")).split("/") if z)

        rows.append(
            f"<tr>"
            f"<td class='sym'>{r['symbol']}</td>"
            f"<td>{name_str}</td>"
            f"<td class='num'>{close_str}</td>"
            f"<td class='num'>{ret1_str}</td>"
            f"<td class='num'>{ret3_str}</td>"
            f"<td class='num'>{rs_str}</td>"
            f"<td class='num'>{dist_str}</td>"
            f"<td class='num'>{above_str}</td>"
            f"<td class='num'>{n_str}</td>"
            f"<td>{zones_str}</td>"
            f"</tr>"
        )

    table = (
        "<table><thead><tr>"
        "<th>Symbol</th><th>Name</th><th>Price</th>"
        "<th>1M Ret</th><th>3M Ret</th><th>RS Rank</th>"
        "<th>vs 52w High</th><th>200 SMA</th>"
        "<th>#Strats</th><th>Signal Zones</th>"
        "</tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table>"
    )

    # ── Portfolio charts ─────────────────────────────────────────────────────
    charts_html = ""
    if with_charts and drive and folder_id:
        syms = pf["symbol"].dropna().unique().tolist()
        log(f"  Building portfolio charts for {len(syms)} holdings…")
        first = True
        for i, sym in enumerate(syms, 1):
            log(f"    [{i}/{len(syms)}] {sym}")
            ohlcv    = load_ohlcv(drive, folder_id, sym)
            sym_sigs = (signals[signals["symbol"] == sym]
                        if not signals.empty else pd.DataFrame())
            chart_div = build_chart_html(sym, ohlcv, sym_sigs, first_chart=first)
            # Get holding info for summary line
            h_row     = pf[pf["symbol"] == sym].iloc[0]
            name_str  = str(h_row.get("name", h_row.get("screener_name", "")))[:35]
            ret3_str  = (f"{h_row['return_3m_pct']:+.1f}%"
                         if "return_3m_pct" in h_row.index and pd.notna(h_row.get("return_3m_pct"))
                         else "")
            zones_str = str(h_row.get("zones", "")) if h_row.get("zones") else "no signal"
            charts_html += (
                f"<details class='chart-block'>"
                f"<summary>{sym}"
                f"<span class='chart-meta'>"
                f"· {name_str} · 3M {ret3_str} · {zones_str}"
                f"</span></summary>"
                f"<div class='chart-inner'>{chart_div}</div>"
                f"</details>\n"
            )
            first = False

    return title + table + charts_html


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--min-strats",   type=int, default=2,
                        help="Minimum strategies for conviction section (default 2)")
    parser.add_argument("--with-charts",  action="store_true",
                        help="Embed interactive Plotly charts (downloads OHLCV per stock)")
    parser.add_argument("--max-charts",   type=int, default=10,
                        help="Max charts to generate when --with-charts (default 10)")
    parser.add_argument("--no-portfolio", action="store_true",
                        help="Skip portfolio section")
    parser.add_argument("--no-open",      action="store_true",
                        help="Save HTML only, don't open browser")
    args = parser.parse_args()

    drive     = get_drive()
    folder_id = os.environ["GDRIVE_FOLDER_ID"]

    signals  = load_all_signals(drive, folder_id)
    features = load_features(drive, folder_id)
    universe = load_universe(drive, folder_id)
    portfolio = pd.DataFrame() if args.no_portfolio else load_portfolio(drive, folder_id)

    log("Building HTML report…")
    now = datetime.now()

    max_charts = args.max_charts if args.with_charts else 0
    if args.with_charts:
        log(f"Charts enabled — will download OHLCV for up to {max_charts} stocks")
    conviction_html = build_conviction_section(
        signals, features, args.min_strats,
        drive=drive, folder_id=folder_id, max_charts=max_charts,
    )
    portfolio_html  = build_portfolio_section(
        portfolio, signals, features, universe,
        drive=drive, folder_id=folder_id,
        with_charts=args.with_charts,
    )

    html = _HTML.format(
        date      = now.strftime("%d-%b-%Y"),
        datetime  = now.strftime("%d %b %Y %H:%M IST"),
        n_signals = len(signals),
        css       = _CSS,
        conviction_section = conviction_html,
        portfolio_section  = portfolio_html,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fname    = f"phase1_{now.strftime('%Y%m%d_%H%M')}.html"
    out_path = OUTPUT_DIR / fname
    out_path.write_text(html, encoding="utf-8")
    log(f"Saved: {out_path}")

    if not args.no_open:
        webbrowser.open(f"file:///{out_path.as_posix()}")
        log("Opened in browser. Use Ctrl+P → Save as PDF to export.")

    print(f"\n  Report: {out_path}")


if __name__ == "__main__":
    main()
