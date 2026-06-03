"""
Stage 8 — Streamlit dashboard (v2 with enhanced breadth + macro strip).

Run locally:
    streamlit run app.py
"""

from __future__ import annotations

import io
import os
import re
from datetime import datetime
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
import json
import ssl
import socket
import time

st.set_page_config(page_title="Signals India", layout="wide",
                          initial_sidebar_state="expanded")

from dotenv import load_dotenv
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from plotly.subplots import make_subplots

SCOPES = ["https://www.googleapis.com/auth/drive"]

ZONE_COLORS = {
    "buy": "#2ecc71", "add": "#16a085", "hold": "#7f8c8d",
    "stop_loss": "#e74c3c", "exit": "#e67e22", "sell": "#c0392b",
}


TIMEFRAME_DAYS = {"3M": 63, "6M": 126, "1Y": 252, "2Y": 504}
STRATEGY_DOCS = {
    "momentum_1m": {"title": "Momentum (1 month)",
        "intent": "Captures stocks with sharp short-term outperformance.",
        "rules": ["Above 200 SMA", "BUY: RS rank ≥ 90 (top decile) over 21 days",
                  "ADD: BUY + within 5% of 52w high", "HOLD: RS rank 80-90"],
        "best_for": "Catching emerging leaders early",
        "caveat": "Higher whipsaw risk"},
    "momentum_2m": {"title": "Momentum (2 months)",
        "intent": "Bridge between short and intermediate momentum.",
        "rules": ["Above 200 SMA", "BUY: RS rank ≥ 90 over 42 days",
                  "ADD: BUY + within 5% of 52w high", "HOLD: RS rank 80-90"],
        "best_for": "Confirming 1M strength has staying power",
        "caveat": "Less responsive than 1M"},
    "momentum_3m": {"title": "Momentum (3 months)",
        "intent": "Classic 'leadership' timeframe; Qullamaggie's primary lookback.",
        "rules": ["Above 200 SMA", "BUY: RS rank ≥ 90 over 63 days",
                  "ADD: BUY + within 5% of 52w high", "HOLD: RS rank 80-90"],
        "best_for": "True trend leaders",
        "caveat": "Best paired with consolidation patterns"},
    "momentum_6m": {"title": "Momentum (6 months)",
        "intent": "Intermediate-term strength (Jegadeesh-Titman classic).",
        "rules": ["Above 200 SMA", "BUY: RS rank ≥ 90 over 126 days",
                  "ADD: BUY + within 5% of 52w high", "HOLD: RS rank 80-90"],
        "best_for": "Persistent winners across a full quarter cycle",
        "caveat": "Slow to react to regime changes"},
    "momentum_12m": {"title": "Momentum (12 months)",
        "intent": "Long-term trend persistence.",
        "rules": ["Above 200 SMA", "BUY: RS rank ≥ 90 over 252 days",
                  "ADD: BUY + within 5% of 52w high", "HOLD: RS rank 80-90"],
        "best_for": "Multi-quarter holds; institutional accumulation candidates",
        "caveat": "Slowest signal; positions can be extended"},
    "ma_respect_20ema_30d": {"title": "MA-Respect (20 EMA, 30 days)",
        "intent": "Stocks in a clean uptrend holding above 20 EMA for 30+ days.",
        "rules": ["BUY: close above 20 EMA ≥ 30 consecutive days", "Stop: 1 ATR below 20 EMA"],
        "best_for": "Uninterrupted trend stocks for pullback entries",
        "caveat": "Streak resets entirely on a single violation"},
    "ma_respect_20ema_60d": {"title": "MA-Respect (20 EMA, 60 days)",
        "intent": "Stricter — 60 consecutive days above 20 EMA.",
        "rules": ["BUY: close above 20 EMA ≥ 60 days", "Stop: 1 ATR below 20 EMA"],
        "best_for": "Strongest trend stocks",
        "caveat": "May fire on only 1-5 names in a bearish regime"},
    "ma_respect_50ema_60d": {"title": "MA-Respect (50 EMA, 60 days)",
        "intent": "Slower MA tolerates short-term noise while maintaining longer-term respect.",
        "rules": ["BUY: close above 50 EMA ≥ 60 days", "Stop: 1 ATR below 50 EMA"],
        "best_for": "Longer-term holders",
        "caveat": "Less precise for swing entries"},
    "qullamaggie": {"title": "Qullamaggie (Kristjan Kullamägi)",
        "intent": "Find stocks that already ran (≥30% in 3M), formed a tight consolidation, then break out on volume.",
        "rules": ["3M return ≥ 30%", "ADR(20) ≥ 4%", "Above 200 SMA",
                  "5-15 day consolidation, range ≤ 15%",
                  "BUY: inside consolidation", "ADD: close > pivot high on 1.5×+ volume",
                  "Stop: pivot low − 0.5×ATR"],
        "best_for": "Asymmetric swing trades with tight stops",
        "caveat": "Many setups never trigger — requires patience"},
    "minervini": {"title": "Mark Minervini SEPA / Trend Template",
        "intent": "Eight-point trend checklist — passing all 8 means textbook bullish structure.",
        "rules": ["Price > 150 SMA and 200 SMA", "150 SMA > 200 SMA",
                  "200 SMA rising ≥ 1 month", "50 SMA > 150 > 200 (stacked)",
                  "Price > 50 SMA", "Price ≥ 30% above 52w low",
                  "Within 25% of 52w high", "RS rank (6M) ≥ 70"],
        "best_for": "High-quality long entries with multiple confirmation",
        "caveat": "Doesn't include VCP entry trigger yet — trend filter only"},
    "darvas": {"title": "Darvas Box",
        "intent": "Stocks at/near 52w highs forming a tight box, then breaking out.",
        "rules": ["Within 5% of 52w high", "Above 200 SMA", "ADR ≥ 2%",
                  "5-20 day box, range ≤ 10%",
                  "BUY: close inside box", "ADD: close > box top on 1.5×+ volume",
                  "Stop: box bottom − 0.5×ATR"],
        "best_for": "Trend continuation at new highs",
        "caveat": "Few setups in bearish regimes"},
    "volume_breakout": {"title": "Volume Breakout",
        "intent": "A volume surge confirming institutional accumulation — the stock trades far above its average volume while in an uptrend near its highs.",
        "rules": ["Above 50 EMA and 200 SMA (uptrend)", "Within 25% of 52w high",
                  "BUY: volume ≥ 2.5× its average",
                  "ADD: BUY + within 5% of 52w high",
                  "HOLD: volume 1.5–2.5× average (elevated, not a full spike)"],
        "best_for": "Catching the day institutions step in",
        "caveat": "A single freak-volume day can fire it — confirm the move looks real"},
    "volume_vcp": {"title": "Volume VCP (dry-up)",
        "intent": "A volume dry-up inside a tight base near the highs — the classic pre-breakout coil (Volatility Contraction Pattern).",
        "rules": ["Above 50 EMA and 200 SMA (uptrend)", "Within 10% of 52w high",
                  "BUY: volume ≤ 0.5× average (dry-up) + tight range (ADR ≤ 3%)",
                  "ADD: an even tighter range / closer to the high"],
        "best_for": "Spotting setups before they break out",
        "caveat": "A setup, not a trigger — the breakout itself is caught by Volume Breakout"},
}


# ---------- Drive helpers ----------

def build_quick_chart(symbol, ohlcv, signals_for_stock, timeframe_days):
    """Lightweight line + volume chart for quick scan.
    Includes: price line, 20/50 EMA, volume bars, start/end date labels,
    signal zone lines, and 1M/3M/6M/1Y return annotations in the title."""
    full = ohlcv.sort_values("date").reset_index(drop=True)
    df   = full.tail(timeframe_days).reset_index(drop=True)
    df["ema_20"] = df["close"].ewm(span=20).mean()
    df["ema_50"] = df["close"].ewm(span=50).mean()

    # ── Return labels ─────────────────────────────────────────────────────────
    def _ret(days):
        if len(full) < days + 1:
            return None
        p0 = float(full["close"].iloc[-(days + 1)])
        p1 = float(full["close"].iloc[-1])
        return (p1 / p0 - 1) * 100 if p0 else None

    rets = {}
    for label, days in [("1M", 21), ("3M", 63), ("6M", 126), ("1Y", 252)]:
        v = _ret(days)
        if v is not None:
            rets[label] = v

    ret_str = "  ".join(
        f'<span style="color:{"#27ae60" if v >= 0 else "#e74c3c"}">'
        f'{label} {v:+.0f}%</span>'
        for label, v in rets.items()
    )

    last_close = float(df["close"].iloc[-1]) if not df.empty else 0
    title_html = f"<b>{symbol}</b>  ₹{last_close:,.0f}    {ret_str}"

    # ── Volume colour: green if close ≥ open, else red ────────────────────────
    vol_colors = [
        "#27ae60" if c >= o else "#e74c3c"
        for c, o in zip(df["close"], df["open"])
    ]

    # ── Date tick positions: first and last bar ───────────────────────────────
    dates = df["date"].tolist()
    tick_vals = [dates[0], dates[-1]]
    tick_text = [
        str(dates[0])[:10] if hasattr(dates[0], "__str__") else "",
        str(dates[-1])[:10] if hasattr(dates[-1], "__str__") else "",
    ]

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        row_heights=[0.72, 0.28], vertical_spacing=0.02,
    )

    # Price + EMAs
    fig.add_trace(go.Scatter(x=df["date"], y=df["close"],
                             line=dict(color="#2c3e50", width=1.5),
                             showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=df["date"], y=df["ema_20"],
                             line=dict(color="#2980b9", width=1),
                             showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=df["date"], y=df["ema_50"],
                             line=dict(color="#8e44ad", width=1),
                             showlegend=False), row=1, col=1)

    # Signal zone lines
    if not signals_for_stock.empty:
        for _, sig in signals_for_stock.iterrows():
            zt    = sig.get("zone_type")
            entry = sig.get("entry")
            if pd.notna(entry):
                fig.add_hline(y=entry,
                              line=dict(color=ZONE_COLORS.get(zt, "#666"), width=1),
                              row=1, col=1)

    # Volume bars
    fig.add_trace(go.Bar(x=df["date"], y=df["volume"],
                         marker_color=vol_colors, showlegend=False), row=2, col=1)

    fig.update_layout(
        height=300,
        title=dict(text=title_html, font=dict(size=11), x=0.02, y=0.97),
        margin=dict(l=4, r=4, t=32, b=4),
        plot_bgcolor="#fafafa",
        paper_bgcolor="white",
        bargap=0.1,
    )
    fig.update_xaxes(
        showgrid=False, zeroline=False,
        tickvals=tick_vals, ticktext=tick_text,
        tickfont=dict(size=8), tickangle=0,
        row=1, col=1,
    )
    fig.update_xaxes(
        showgrid=False, zeroline=False,
        tickvals=tick_vals, ticktext=tick_text,
        tickfont=dict(size=8), tickangle=0,
        row=2, col=1,
    )
    fig.update_yaxes(showgrid=False, zeroline=False, tickfont=dict(size=8),
                     row=1, col=1)
    fig.update_yaxes(showgrid=False, zeroline=False, showticklabels=False,
                     row=2, col=1)
    return fig


def build_stock_chart(symbol, ohlcv, signals_for_stock, timeframe_days, height=500):
    """Single-stock chart: candlestick + EMAs (with zone overlays) + volume."""
    df = ohlcv.sort_values("date").tail(timeframe_days).reset_index(drop=True)
    df["ema_20"] = df["close"].ewm(span=20).mean()
    df["ema_50"] = df["close"].ewm(span=50).mean()
    df["sma_200"] = df["close"].rolling(200).mean()

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.75, 0.25], vertical_spacing=0.03)
    fig.add_trace(go.Candlestick(x=df["date"], open=df["open"], high=df["high"],
                                 low=df["low"], close=df["close"], name="Price",
                                 showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=df["date"], y=df["ema_20"], name="20 EMA",
                             line=dict(color="#2980b9", width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df["date"], y=df["ema_50"], name="50 EMA",
                             line=dict(color="#8e44ad", width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df["date"], y=df["sma_200"], name="200 SMA",
                             line=dict(color="#e67e22", width=1, dash="dash")),
                  row=1, col=1)

    if not signals_for_stock.empty:
        for _, sig in signals_for_stock.iterrows():
            zt = sig.get("zone_type")
            entry = sig.get("entry")
            stop = sig.get("stop")
            strat = sig.get("strategy", "")
            color = ZONE_COLORS.get(zt, "#666")
            if pd.notna(entry):
                fig.add_hline(y=entry, line=dict(color=color, width=1),
                              annotation_text=f"{strat}:{zt}",
                              annotation_position="right", row=1, col=1)
            if pd.notna(stop):
                fig.add_hline(y=stop, line=dict(color=ZONE_COLORS["stop_loss"],
                                                 width=1, dash="dot"),
                              annotation_text=f"{strat}:stop",
                              annotation_position="right", row=1, col=1)

    fig.add_trace(go.Bar(x=df["date"], y=df["volume"], name="Vol",
                         marker_color="#95a5a6", showlegend=False), row=2, col=1)

    fig.update_layout(height=height, xaxis_rangeslider_visible=False,
                      margin=dict(l=10, r=10, t=20, b=10),
                      legend=dict(orientation="h", yanchor="bottom", y=1.02))
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Vol", row=2, col=1)
    return fig

def get_drive():
    load_dotenv(Path(__file__).parent / ".env")

    # Cloud path: credentials passed as inline JSON via env vars (Streamlit Cloud secrets)
    cs_json = os.environ.get("GDRIVE_OAUTH_CLIENT_SECRET_JSON")
    tk_json = os.environ.get("GDRIVE_OAUTH_TOKEN_JSON")
    if cs_json and tk_json:
        import json
        creds = Credentials.from_authorized_user_info(json.loads(tk_json), SCOPES)
        if creds.expired and creds.refresh_token:
            creds.refresh(Request())
        return build("drive", "v3", credentials=creds, cache_discovery=False)

    # Local path: credentials at file paths
    cs_path = Path(os.environ["GDRIVE_OAUTH_CLIENT_SECRET_PATH"])
    token_path = Path(os.environ["GDRIVE_OAUTH_TOKEN_PATH"])
    creds = None
    if token_path.exists():
        creds = Credentials.from_authorized_user_file(str(token_path), SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(str(cs_path), SCOPES)
            creds = flow.run_local_server(port=0)
        token_path.write_text(creds.to_json())
    return build("drive", "v3", credentials=creds, cache_discovery=False)

# ============================================================
# Data freshness banner — renders logs/health/latest.json
# (written by pipeline_healthcheck.py as the last workflow step)
# ============================================================
import json
from datetime import datetime, timezone

def _health_find_sub(drive, parent_id, name):
    q = (f"name='{name}' and '{parent_id}' in parents and "
         f"mimeType='application/vnd.google-apps.folder' and trashed=false")
    f = drive.files().list(q=q, fields="files(id)").execute().get("files", [])
    return f[0]["id"] if f else None

@st.cache_data(ttl=300)
def load_health_report():
    """Latest Phase 1 pipeline health report dict, or None if not found."""
    try:
        drive = get_drive()
        folder_id = os.environ["GDRIVE_FOLDER_ID"]
        logs_id = _health_find_sub(drive, folder_id, "logs")
        health_id = _health_find_sub(drive, logs_id, "health") if logs_id else None
        if not health_id:
            return None
        q = f"name='latest.json' and '{health_id}' in parents and trashed=false"
        files = drive.files().list(q=q, fields="files(id)").execute().get("files", [])
        if not files:
            return None
        return json.loads(drive.files().get_media(fileId=files[0]["id"]).execute())
    except Exception as e:
        return {"_error": str(e)[:200]}


@st.cache_data(ttl=300)
def load_phase2_status():
    """Latest Phase 2 status dict (queue counts), or None if not found."""
    try:
        drive = get_drive()
        folder_id = os.environ["GDRIVE_FOLDER_ID"]
        logs_id = _health_find_sub(drive, folder_id, "logs")
        health_id = _health_find_sub(drive, logs_id, "health") if logs_id else None
        if not health_id:
            return None
        q = f"name='phase2_latest.json' and '{health_id}' in parents and trashed=false"
        files = drive.files().list(q=q, fields="files(id)").execute().get("files", [])
        if not files:
            return None
        return json.loads(drive.files().get_media(fileId=files[0]["id"]).execute())
    except Exception as e:
        return {"_error": str(e)[:200]}

def _age_str(run_at_iso: str | None) -> str:
    """Return human-readable age like '2h ago' or 'unknown'."""
    if not run_at_iso:
        return "unknown"
    try:
        dt = datetime.fromisoformat(run_at_iso.replace("Z", "+00:00"))
        h = (datetime.now(timezone.utc) - dt).total_seconds() / 3600
        if h < 1:
            return f"{int(h*60)}m ago"
        return f"{h:.0f}h ago"
    except Exception:
        return "unknown"


def render_health_sidebar():
    """Compact pipeline status in a sidebar expander — call once per page."""
    rep  = load_health_report()
    rep2 = load_phase2_status()

    # ── Phase 1 label ─────────────────────────────────────────────────────────
    if rep is None or "_error" in rep:
        p1_icon, p1_txt = "❓", "unknown"
    else:
        overall = rep.get("overall", "UNKNOWN")
        age_h = None
        try:
            dt = datetime.fromisoformat(rep["run_at"].replace("Z", "+00:00"))
            age_h = (datetime.now(timezone.utc) - dt).total_seconds() / 3600
        except Exception:
            pass
        if age_h is not None and age_h > 30:
            p1_icon, p1_txt = "🔴", f"stale ({age_h:.0f}h)"
        elif overall == "HEALTHY":
            p1_icon, p1_txt = "🟢", _age_str(rep.get("run_at"))
        elif overall == "DEGRADED":
            p1_icon, p1_txt = "🟡", f"degraded · {_age_str(rep.get('run_at'))}"
        elif overall == "FAIL":
            p1_icon, p1_txt = "🔴", f"FAILED · {_age_str(rep.get('run_at'))}"
        else:
            p1_icon, p1_txt = "❓", overall

    # ── Phase 2 label ─────────────────────────────────────────────────────────
    if rep2 is None or "_error" in rep2:
        p2_icon, p2_txt = "❓", "no status yet"
    else:
        tot  = rep2.get("queue_totals", {})
        pend = tot.get("pending", 0)
        err  = tot.get("error", 0)
        p2_icon = "🔴" if err > 0 else ("🟡" if pend > 0 else "🟢")
        p2_txt  = f"{pend} pending · {err} err · {_age_str(rep2.get('run_at'))}"

    with st.sidebar.expander(f"📡 Pipeline  {p1_icon}{p2_icon}", expanded=False):
        st.markdown(f"**Phase 1 (data):** {p1_icon} {p1_txt}")
        if rep and "_error" not in rep:
            fails = rep.get("critical_failures", 0)
            warns = rep.get("warnings", 0)
            if fails or warns:
                st.caption(f"  {fails} critical · {warns} warnings")
        st.markdown(f"**Phase 2 (intel):** {p2_icon} {p2_txt}")
        if rep2 and "_error" not in rep2:
            by_type = rep2.get("by_doc_type", {})
            if by_type:
                rows = []
                for dt, cnts in sorted(by_type.items()):
                    rows.append({"type": dt,
                                 "⏳": cnts.get("pending", 0),
                                 "✅": cnts.get("done", 0),
                                 "❌": cnts.get("error", 0)})
                st.dataframe(pd.DataFrame(rows), hide_index=True,
                             use_container_width=True)

def render_health_banner():
    rep  = load_health_report()
    rep2 = load_phase2_status()

    # ── Phase 1 banner ────────────────────────────────────────────────────────
    if rep is None:
        st.error("No Phase 1 health report on Drive — daily pipeline may never have run.")
    elif "_error" in rep:
        st.warning(f"Could not read Phase 1 health report: {rep['_error']}")
    else:
        run_at  = rep.get("run_at")
        age_h   = None
        age_txt = ""
        if run_at:
            try:
                dt    = datetime.fromisoformat(run_at.replace("Z", "+00:00"))
                age_h = (datetime.now(timezone.utc) - dt).total_seconds() / 3600
                age_txt = f" — {_age_str(run_at)}"
            except Exception:
                pass
        overall = rep.get("overall", "UNKNOWN")
        if age_h is not None and age_h > 30:
            st.error(f"⚠️ Phase 1 (data pipeline) has not run for {age_h:.0f}h — "
                     f"signals below may be STALE.")
        elif overall == "HEALTHY":
            st.success(f"✅ Phase 1 HEALTHY{age_txt}")
        elif overall == "DEGRADED":
            st.warning(f"⚠️ Phase 1 DEGRADED — {rep.get('warnings', 0)} warning(s){age_txt}")
        elif overall == "FAIL":
            st.error(f"🔴 Phase 1 FAILED — {rep.get('critical_failures', 0)} critical{age_txt}")
        else:
            st.info(f"Phase 1: {overall}{age_txt}")
        checks = rep.get("checks", [])
        if checks:
            with st.expander("Phase 1 — data freshness detail"):
                st.dataframe(pd.DataFrame(checks),
                             use_container_width=True, hide_index=True)

    # ── Phase 2 banner ────────────────────────────────────────────────────────
    if rep2 is None:
        st.info("Phase 2 (Company Intel) has not written a status yet — "
                "it will appear here after the next run.")
    elif "_error" in rep2:
        st.warning(f"Could not read Phase 2 status: {rep2['_error']}")
    else:
        tot  = rep2.get("queue_totals", {})
        pend = tot.get("pending", 0)
        done = tot.get("done",    0)
        err  = tot.get("error",   0)
        age  = _age_str(rep2.get("run_at"))
        if err > 0:
            st.warning(f"⚠️ Phase 2 (Company Intel) — {pend} pending · "
                       f"{done} done · **{err} errors** · {age}")
        elif pend > 0:
            st.info(f"🟡 Phase 2 — {pend} pending · {done} done · {age}")
        else:
            st.success(f"✅ Phase 2 — queue clear ({done} done) · {age}")
        by_type = rep2.get("by_doc_type", {})
        if by_type:
            with st.expander("Phase 2 — queue detail by document type"):
                rows = [{"doc_type": dt,
                         "⏳ pending": cnts.get("pending", 0),
                         "✅ done":    cnts.get("done",    0),
                         "❌ error":   cnts.get("error",   0)}
                        for dt, cnts in sorted(by_type.items())]
                st.dataframe(pd.DataFrame(rows), hide_index=True,
                             use_container_width=True)

def _list_folder(drive, folder_id):
    out = {}
    page_token = None
    while True:
        resp = drive.files().list(
            q=f"'{folder_id}' in parents and trashed=false",
            fields="nextPageToken, files(id,name)",
            pageSize=1000, pageToken=page_token,
        ).execute()
        for f in resp.get("files", []):
            out[f["name"]] = f["id"]
        page_token = resp.get("nextPageToken")
        if not page_token:
            break
    return out


def _find_subfolder(drive, parent_id, name):
    q = (f"name='{name}' and '{parent_id}' in parents "
         f"and mimeType='application/vnd.google-apps.folder' and trashed=false")
    found = drive.files().list(q=q, fields="files(id)").execute().get("files", [])
    return found[0]["id"] if found else None


def _download_bytes(drive, file_id):
    request = drive.files().get_media(fileId=file_id)
    fh = io.BytesIO()
    downloader = MediaIoBaseDownload(fh, request)
    done = False
    while not done:
        _, done = downloader.next_chunk()
    return fh.getvalue()


@st.cache_resource(show_spinner=False)
def drive_service():
    return get_drive()

_TRANSIENT_NET = (ssl.SSLError, socket.error, ConnectionError, OSError)

def _drive_call(fn, attempts=4):
    """Run a Drive operation. On ANY failure (stale TLS connection, httplib2
    teardown quirks, truncated download), drop the cached Drive connection and
    retry with a fresh one. Re-raises if every attempt fails."""
    for i in range(attempts):
        try:
            return fn()
        except Exception:
            if i == attempts - 1:
                raise
            try:
                drive_service.clear()      # force a fresh connection next call
            except Exception:
                pass
            time.sleep(1.5 * (i + 1))

@st.cache_data(ttl=300, show_spinner=False)
def load_csv(path_parts):
    def _do():
        drive = drive_service()
        parent = os.environ["GDRIVE_FOLDER_ID"]
        for part in path_parts[:-1]:
            parent = _find_subfolder(drive, parent, part)
            if not parent:
                return pd.DataFrame()
        files = _list_folder(drive, parent)
        fid = files.get(path_parts[-1])
        if not fid:
            return pd.DataFrame()
        return pd.read_csv(io.BytesIO(_download_bytes(drive, fid)))
    return _drive_call(_do)


@st.cache_data(ttl=1800, show_spinner=False)
def load_parquet(path_parts):
    def _do():
        drive = drive_service()
        parent = os.environ["GDRIVE_FOLDER_ID"]
        for part in path_parts[:-1]:
            parent = _find_subfolder(drive, parent, part)
            if not parent:
                return pd.DataFrame()
        files = _list_folder(drive, parent)
        fid = files.get(path_parts[-1])
        if not fid:
            return pd.DataFrame()
        return pd.read_parquet(io.BytesIO(_download_bytes(drive, fid)))
    return _drive_call(_do)


@st.cache_data(ttl=300, show_spinner=False)
def load_all_strategy_signals():
    def _do():
        drive = drive_service()
        folder_id = os.environ["GDRIVE_FOLDER_ID"]
        signals_id = _find_subfolder(drive, folder_id, "signals")
        if not signals_id:
            return pd.DataFrame()
        per_strat_id = _find_subfolder(drive, signals_id, "per_strategy")
        if not per_strat_id:
            return pd.DataFrame()
        subs = drive.files().list(
            q=f"'{per_strat_id}' in parents and mimeType='application/vnd.google-apps.folder' and trashed=false",
            fields="files(id,name)",
        ).execute().get("files", [])
        frames = []
        for s in subs:
            files = _list_folder(drive, s["id"])
            latest_id = files.get("latest.csv")
            if latest_id:
                df = pd.read_csv(io.BytesIO(_download_bytes(drive, latest_id)))
                df["strategy_group"] = s["name"]
                frames.append(df)
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return _drive_call(_do)


# ---------- Bulk OHLCV loader (traverses Drive path ONCE for N symbols) ----------

@st.cache_data(ttl=1800, show_spinner=False)
def load_ohlcv_bulk(symbols: tuple) -> dict:
    """Download OHLCV for multiple symbols in a single Drive session.
    Folder lookups happen once; only the needed parquets are downloaded.
    Returns {symbol: DataFrame}. Missing symbols map to empty DataFrame."""
    def _do():
        drive  = drive_service()
        root   = os.environ["GDRIVE_FOLDER_ID"]
        data_id = _find_subfolder(drive, root, "data")
        if not data_id:
            return {}
        ohlcv_id = _find_subfolder(drive, data_id, "ohlcv")
        if not ohlcv_id:
            return {}
        all_files = _list_folder(drive, ohlcv_id)   # one API call lists everything
        result = {}
        for sym in symbols:
            fid = all_files.get(f"{sym}.parquet")
            if not fid:
                result[sym] = pd.DataFrame()
                continue
            try:
                result[sym] = pd.read_parquet(io.BytesIO(_download_bytes(drive, fid)))
            except Exception:
                result[sym] = pd.DataFrame()
        return result
    return _drive_call(_do)


# ---------- Company Intel helpers ----------

@st.cache_data(ttl=300, show_spinner=False)
def load_daily_index() -> pd.DataFrame:
    """List all files in company_repo/_daily/ with parsed metadata."""
    def _do():
        drive = drive_service()
        folder_id = os.environ["GDRIVE_FOLDER_ID"]
        repo_id = _find_subfolder(drive, folder_id, "company_repo")
        if not repo_id:
            return pd.DataFrame()
        daily_id = _find_subfolder(drive, repo_id, "_daily")
        if not daily_id:
            return pd.DataFrame()
        files = drive.files().list(
            q=f"'{daily_id}' in parents and trashed=false",
            fields="files(id, name, modifiedTime)",
            orderBy="modifiedTime desc",
            pageSize=200,
        ).execute().get("files", [])
        rows = []
        for f in files:
            name = f["name"]
            # Filename pattern: {doc_type}_{dd}_{mon}{yyyy}.md
            # doc_type may contain underscores (annual_report), so match from the right
            m = re.match(r"^(.+)_(\d{2})_([a-z]{3})(\d{4})\.md$", name, re.IGNORECASE)
            if m:
                doc_type, day, mon, year = m.groups()
                try:
                    dt = datetime.strptime(f"{day} {mon} {year}", "%d %b %Y")
                except ValueError:
                    dt = None
            else:
                doc_type = name.replace(".md", "")
                dt = None
            rows.append({"file_id": f["id"], "filename": name,
                         "doc_type": doc_type.replace("_", " ").title(),
                         "doc_type_raw": doc_type,
                         "date": dt})
        df = pd.DataFrame(rows)
        if not df.empty:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            df = df.sort_values("date", ascending=False)
        return df
    try:
        return _drive_call(_do)
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=600, show_spinner=False)
def load_md_content(file_id: str) -> str:
    """Download and return markdown content of a Drive file."""
    def _do():
        return _download_bytes(drive_service(), file_id).decode("utf-8", errors="replace")
    try:
        return _drive_call(_do)
    except Exception as e:
        return f"*Could not load file: {e}*"


@st.cache_data(ttl=300, show_spinner=False)
def find_company_page(key: str) -> str | None:
    """Return company_page.md content for a given ISIN or symbol key, or None."""
    if not key:
        return None
    def _do():
        drive = drive_service()
        folder_id = os.environ["GDRIVE_FOLDER_ID"]
        repo_id = _find_subfolder(drive, folder_id, "company_repo")
        if not repo_id:
            return None
        comp_id = _find_subfolder(drive, repo_id, key)
        if not comp_id:
            return None
        q = f"name='company_page.md' and '{comp_id}' in parents and trashed=false"
        files = drive.files().list(q=q, fields="files(id)").execute().get("files", [])
        if not files:
            return None
        return _download_bytes(drive, files[0]["id"]).decode("utf-8", errors="replace")
    try:
        return _drive_call(_do)
    except Exception as e:
        return f"*Error: {e}*"


# ---------- OT8: User document library (Drive helpers) ----------

def _get_or_create_folder(drive, parent_id: str, name: str) -> str:
    """Return folder ID, creating it if it does not exist."""
    existing = _find_subfolder(drive, parent_id, name)
    if existing:
        return existing
    meta = {"name": name, "parents": [parent_id],
            "mimeType": "application/vnd.google-apps.folder"}
    return drive.files().create(body=meta, fields="id").execute()["id"]


def _upload_bytes_to_drive(drive, folder_id: str, filename: str,
                            content: bytes, mime_type: str = "application/octet-stream") -> str:
    """Upload raw bytes to a Drive folder. Returns file_id."""
    from googleapiclient.http import MediaIoBaseUpload
    media = MediaIoBaseUpload(io.BytesIO(content), mimetype=mime_type, resumable=False)
    meta  = {"name": filename, "parents": [folder_id]}
    return drive.files().create(body=meta, media_body=media, fields="id").execute()["id"]


def _read_manifest(drive, folder_id: str) -> list:
    """Read _manifest.json from a Drive folder. Returns [] if absent."""
    q = f"name='_manifest.json' and '{folder_id}' in parents and trashed=false"
    files = drive.files().list(q=q, fields="files(id)").execute().get("files", [])
    if not files:
        return []
    try:
        return json.loads(_download_bytes(drive, files[0]["id"]).decode("utf-8"))
    except Exception:
        return []


def _write_manifest(drive, folder_id: str, manifest: list) -> None:
    """Upsert _manifest.json in a Drive folder."""
    from googleapiclient.http import MediaIoBaseUpload
    content = json.dumps(manifest, indent=2, default=str).encode("utf-8")
    media   = MediaIoBaseUpload(io.BytesIO(content), mimetype="application/json",
                                resumable=False)
    q = f"name='_manifest.json' and '{folder_id}' in parents and trashed=false"
    existing = drive.files().list(q=q, fields="files(id)").execute().get("files", [])
    if existing:
        drive.files().update(fileId=existing[0]["id"], media_body=media).execute()
    else:
        drive.files().create(
            body={"name": "_manifest.json", "parents": [folder_id]},
            media_body=media, fields="id",
        ).execute()


def _user_docs_company_folder(drive, company_key: str) -> str:
    """Return (creating if needed) user_docs/<company_key>/ folder ID."""
    root_id      = os.environ["GDRIVE_FOLDER_ID"]
    user_docs_id = _get_or_create_folder(drive, root_id, "user_docs")
    return _get_or_create_folder(drive, user_docs_id, company_key.upper())


_MIME = {
    ".pdf":  "application/pdf",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".doc":  "application/msword",
    ".txt":  "text/plain",
    ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    ".csv":  "text/csv",
}

_DOC_TYPES = [
    "Annual Report",
    "Analyst / Sell-side Note",
    "Credit Rating Report",
    "Industry Report",
    "Concall Transcript",
    "Financial Statements",
    "Other",
]

_DOC_TYPE_SLUG = {
    "Annual Report": "annual_report",
    "Analyst / Sell-side Note": "analyst_note",
    "Credit Rating Report": "credit_rating",
    "Industry Report": "industry_report",
    "Concall Transcript": "transcript",
    "Financial Statements": "financials",
    "Other": "doc",
}


# ---------- Guidance / Results data loaders ----------

@st.cache_data(ttl=300, show_spinner=False)
def load_guidance_tracker() -> pd.DataFrame:
    return load_parquet(["company_repo", "_index", "guidance_tracker.parquet"])


@st.cache_data(ttl=300, show_spinner=False)
def load_gf1_statements() -> pd.DataFrame:
    return load_parquet(["company_repo", "_index", "gf1_guidance_statements.parquet"])


@st.cache_data(ttl=300, show_spinner=False)
def load_gf4_flags() -> pd.DataFrame:
    return load_parquet(["company_repo", "_index", "gf4_quality_flags.parquet"])


@st.cache_data(ttl=300, show_spinner=False)
def load_results_summary() -> pd.DataFrame:
    return load_parquet(["company_repo", "_index", "results.parquet"])


@st.cache_data(ttl=300, show_spinner=False)
def load_quarterly_index() -> list[dict]:
    """List quarterly guidance .md files from company_repo/_quarterly/."""
    def _do():
        drive = drive_service()
        folder_id = os.environ["GDRIVE_FOLDER_ID"]
        repo_id = _find_subfolder(drive, folder_id, "company_repo")
        if not repo_id:
            return []
        qrt_id = _find_subfolder(drive, repo_id, "_quarterly")
        if not qrt_id:
            return []
        files = drive.files().list(
            q=f"'{qrt_id}' in parents and trashed=false",
            fields="files(id, name, modifiedTime)",
            orderBy="modifiedTime desc",
            pageSize=200,
        ).execute().get("files", [])
        return [{"file_id": f["id"], "filename": f["name"],
                 "modified": f["modifiedTime"]} for f in files if f["name"].endswith(".md")]
    try:
        return _drive_call(_do)
    except Exception:
        return []


def page_company_intel():
    st.title("Company Intelligence")
    st.caption(
        "Browse AI-generated summaries from concall transcripts, results PDFs, "
        "ratings, presentations and annual reports."
    )

    tab_digests, tab_company = st.tabs(["📅 Daily Digests", "🏢 Company Page"])

    # ── Tab 1: Daily Digests ──────────────────────────────────────────
    with tab_digests:
        with st.spinner("Loading digest index…"):
            df = load_daily_index()

        if df.empty:
            st.info("No digest files yet. They appear after the pipeline runs.")
            return

        # ── Latest file per doc type (quick-access strip) ─────────────
        st.subheader("Latest available")
        latest = (df.dropna(subset=["date"])
                    .sort_values("date", ascending=False)
                    .groupby("doc_type").first().reset_index())
        cols = st.columns(len(latest)) if len(latest) <= 5 else st.columns(5)
        for i, (_, row) in enumerate(latest.iterrows()):
            col = cols[i % len(cols)]
            date_str = row["date"].strftime("%d %b") if pd.notna(row["date"]) else "?"
            if col.button(f"📄 {row['doc_type']}\n{date_str}", key=f"lat_{i}",
                          use_container_width=True):
                st.session_state["intel_selected_file"] = row["file_id"]
                st.session_state["intel_selected_name"] = row["filename"]

        st.markdown("---")

        # ── Browse by period ──────────────────────────────────────────
        st.subheader("Browse")
        c1, c2, c3 = st.columns(3)

        all_types = ["All"] + sorted(df["doc_type"].dropna().unique().tolist())
        sel_type = c1.selectbox("Document type", all_types)
        view     = c2.selectbox("Group by", ["Day", "Week", "Month", "Quarter"])

        dff = df if sel_type == "All" else df[df["doc_type"] == sel_type]
        dff = dff.dropna(subset=["date"]).copy()

        if dff.empty:
            st.info("No files for this filter.")
        else:
            if view == "Day":
                periods = sorted(dff["date"].dt.date.unique(), reverse=True)
                sel = c3.selectbox("Date", [str(p) for p in periods])
                matches = dff[dff["date"].dt.date.astype(str) == sel]
            elif view == "Week":
                dff["_period"] = dff["date"].dt.to_period("W").astype(str)
                periods = sorted(dff["_period"].unique(), reverse=True)
                sel = c3.selectbox("Week", periods)
                matches = dff[dff["_period"] == sel]
            elif view == "Month":
                dff["_period"] = dff["date"].dt.to_period("M").dt.strftime("%b %Y")
                periods = sorted(dff["_period"].unique(),
                                 key=lambda x: datetime.strptime(x, "%b %Y"), reverse=True)
                sel = c3.selectbox("Month", periods)
                matches = dff[dff["_period"] == sel]
            else:
                dff["_period"] = dff["date"].dt.to_period("Q").astype(str)
                periods = sorted(dff["_period"].unique(), reverse=True)
                sel = c3.selectbox("Quarter", periods)
                matches = dff[dff["_period"] == sel]

            st.caption(f"{len(matches)} file(s)")
            for _, row in matches.iterrows():
                with st.expander(f"📄 {row['filename']}", expanded=(len(matches) == 1)):
                    st.markdown(load_md_content(row["file_id"]))

        # ── Show pre-selected file (from quick-access buttons) ────────
        if "intel_selected_file" in st.session_state:
            st.markdown("---")
            st.subheader(f"📄 {st.session_state.get('intel_selected_name', '')}")
            st.markdown(load_md_content(st.session_state["intel_selected_file"]))

    # ── Tab 2: Company Page ───────────────────────────────────────────
    with tab_company:
        st.subheader("Company intelligence page")
        st.caption("Permanent per-company log of all processed documents.")
        key_input = st.text_input(
            "ISIN or NSE symbol",
            placeholder="e.g. INE002A01018  or  RELIANCE",
        ).strip().upper()
        if key_input:
            with st.spinner(f"Loading {key_input}…"):
                content = find_company_page(key_input)
            if content is None:
                st.warning(
                    f"No company page found for **{key_input}**. "
                    "It may not have been processed yet (company must be in your portfolio, "
                    "or have at least one concall processed)."
                )
            else:
                st.markdown(content)


# ---------- Guidance helpers ----------

_GF4_POSITIVE = frozenset({
    "strong order book", "capacity backed", "high visibility",
    "order book backed", "confirmed orders", "take or pay",
    "long term contract", "pipeline visibility",
})
_GF4_NEGATIVE = frozenset({
    "weak visibility", "guidance ambiguous", "execution risk",
    "volume dependent", "macro dependent", "aspirational",
    "sector headwind", "demand uncertainty",
})


def _gf4_quality_score(gf4_df: pd.DataFrame, symbol: str) -> int:
    """Return integer quality score from GF4 flags for a symbol."""
    rows = gf4_df[gf4_df["symbol"].astype(str) == symbol]
    if rows.empty:
        return 0
    score = 0
    for flag in rows["flag_type"].astype(str).str.lower():
        if any(p in flag for p in _GF4_POSITIVE):
            score += 1
        elif any(ng in flag for ng in _GF4_NEGATIVE):
            score -= 1
    return score


def _guidance_is_active(horizon_fy) -> bool:
    """Return True if the horizon FY is the current FY or a future FY."""
    today = datetime.now()
    fy_end_yr = (today.year + 1) if today.month >= 4 else today.year
    m = re.search(r'FY(\d{2,4})', str(horizon_fy), re.IGNORECASE)
    if not m:
        return True
    yr = int(m.group(1))
    if yr < 100:
        yr += 2000
    return yr >= fy_end_yr


def page_guidance():
    st.title("Management Guidance")
    st.caption(
        "Structured guidance extracted from concall transcripts. "
        "Updated as new concalls are processed by the Phase 2 pipeline."
    )

    tab_tracker, tab_watchlist, tab_momentum = st.tabs([
        "📋 Guidance Tracker",
        "👁 Active Watchlist",
        "🚀 Guidance × Momentum",
    ])

    # ── Tab 1: Guidance Tracker ───────────────────────────────────────
    with tab_tracker:
        with st.spinner("Loading guidance data…"):
            gt  = load_guidance_tracker()
            gf1 = load_gf1_statements()

        if gt.empty and gf1.empty:
            st.info(
                "No guidance data yet. It appears after concalls are processed "
                "by the Phase 2 pipeline."
            )
        else:
            # Filters
            c1, c2, c3, c4 = st.columns(4)
            companies = sorted(gt["symbol"].dropna().unique().tolist()) if not gt.empty else []
            sel_company = c1.selectbox("Company", ["All"] + companies, key="gt_company")

            metrics = sorted(gt["metric"].dropna().unique().tolist()) if not gt.empty else []
            sel_metric = c2.selectbox("Metric", ["All"] + metrics, key="gt_metric")

            horizons = sorted(gt["horizon_fy"].dropna().unique().tolist()) if not gt.empty else []
            sel_horizon = c3.selectbox("Horizon FY", ["All"] + [str(h) for h in horizons],
                                       key="gt_horizon")

            g_types = sorted(gt["guidance_type"].dropna().unique().tolist()) if not gt.empty else []
            sel_type = c4.selectbox("Type", ["All"] + g_types, key="gt_type")

            gf = gt.copy() if not gt.empty else pd.DataFrame()
            if not gf.empty:
                if sel_company != "All":
                    gf = gf[gf["symbol"] == sel_company]
                if sel_metric != "All":
                    gf = gf[gf["metric"] == sel_metric]
                if sel_horizon != "All":
                    gf = gf[gf["horizon_fy"].astype(str) == sel_horizon]
                if sel_type != "All":
                    gf = gf[gf["guidance_type"] == sel_type]

                st.caption(f"{len(gf)} guidance rows")
                show_cols = [c for c in [
                    "symbol", "company_name", "quarter", "metric",
                    "guidance_type", "horizon_fy", "value", "unit",
                    "cagr_pct", "notes",
                ] if c in gf.columns]
                st.dataframe(gf[show_cols], use_container_width=True,
                             hide_index=True, height=400)

            if not gf1.empty:
                with st.expander("📝 Raw forward-looking statements (GF1)"):
                    gf1_flt = gf1.copy()
                    if sel_company != "All" and "symbol" in gf1.columns:
                        gf1_flt = gf1_flt[gf1_flt["symbol"] == sel_company]
                    show_gf1 = [c for c in [
                        "symbol", "company_name", "quarter",
                        "exact_statement", "metric_type", "timeframe",
                        "explicitness_type", "quantifiable", "numeric_value",
                    ] if c in gf1_flt.columns]
                    st.dataframe(gf1_flt[show_gf1], use_container_width=True,
                                 hide_index=True, height=300)

    # ── Tab 2: Active Watchlist ───────────────────────────────────────
    with tab_watchlist:
        with st.spinner("Loading guidance and quality data…"):
            gt  = load_guidance_tracker()
            gf4 = load_gf4_flags()

        if gt.empty:
            st.info("No guidance data yet.")
        else:
            active_mask = gt["horizon_fy"].apply(_guidance_is_active)
            active = gt[active_mask].copy()

            if "guidance_type" in active.columns:
                explicit = active[
                    active["guidance_type"].astype(str).str.lower()
                    .isin(["explicit", "quantified", "numeric"])
                ]
                if explicit.empty:
                    explicit = active
            else:
                explicit = active

            explicit = explicit.copy()
            explicit["quality_score"] = explicit["symbol"].apply(
                lambda s: _gf4_quality_score(gf4, s) if not gf4.empty else 0
            )

            guidance_counts = (
                explicit.groupby("symbol")
                .agg(
                    n_guidance=("metric", "count"),
                    quality_score=("quality_score", "first"),
                    metrics=("metric", lambda x: ", ".join(sorted(set(x.dropna())))),
                )
                .reset_index()
                .sort_values(["quality_score", "n_guidance"], ascending=[False, False])
                .reset_index(drop=True)
            )

            st.subheader("Companies with active guidance")
            st.caption(
                f"{len(guidance_counts)} companies with explicit guidance for current/future FY.  "
                "Quality score: +1 per positive flag (strong order book, capacity backed…), "
                "−1 per negative flag (execution risk, macro dependent…)."
            )
            st.dataframe(
                guidance_counts[["symbol", "n_guidance", "quality_score", "metrics"]],
                use_container_width=True, hide_index=True, height=400,
            )

            syms = guidance_counts["symbol"].tolist()
            if syms:
                sel_sym = st.selectbox("Drill into company", syms, key="watchlist_drill")
                detail = explicit[explicit["symbol"] == sel_sym]
                detail_cols = [c for c in [
                    "quarter", "metric", "guidance_type",
                    "horizon_fy", "value", "unit", "cagr_pct", "notes",
                ] if c in detail.columns]
                st.dataframe(detail[detail_cols], use_container_width=True, hide_index=True)

    # ── Tab 3: Guidance × Momentum ────────────────────────────────────
    with tab_momentum:
        with st.spinner("Loading signals, guidance and quality data…"):
            signals = load_all_strategy_signals()
            gt      = load_guidance_tracker()
            gf4     = load_gf4_flags()

        if signals.empty:
            st.info("No strategy signals today.")
        else:
            buy_add  = signals[signals["zone_type"].isin(["buy", "add"])]
            momentum = (buy_add.groupby("symbol")["strategy_group"]
                        .nunique().reset_index(name="n_strategies"))

            if not gt.empty:
                active_mask = gt["horizon_fy"].apply(_guidance_is_active)
                guidance_agg = (gt[active_mask].groupby("symbol")["metric"]
                                .count().reset_index(name="n_guidance"))
            else:
                guidance_agg = pd.DataFrame(columns=["symbol", "n_guidance"])

            # Pre-compute quality scores only for symbols appearing in signals
            sig_syms = buy_add["symbol"].dropna().unique()
            quality_map = (
                {s: _gf4_quality_score(gf4, s) for s in sig_syms}
                if not gf4.empty else {}
            )

            combined = momentum.merge(guidance_agg, on="symbol", how="left")
            combined["n_guidance"] = combined["n_guidance"].fillna(0).astype(int)
            combined["quality"]    = combined["symbol"].map(quality_map).fillna(0).astype(int)
            combined["guidance_multiplier"] = combined["quality"].apply(
                lambda q: round(max(0.5, 1 + q * 0.3), 2)
            )
            combined["combined_score"] = (
                combined["n_strategies"] * combined["guidance_multiplier"]
                + combined["n_guidance"] * 0.2
            ).round(2)
            combined = combined.sort_values("combined_score", ascending=False).reset_index(drop=True)

            st.subheader("Guidance-backed momentum leaders")
            st.caption(
                "**Score** = n\\_strategies × max(0.5, 1 + quality × 0.3) + n\\_guidance × 0.2  \n"
                "Positive quality flags amplify the score; negative flags dampen it."
            )

            c1, c2 = st.columns(2)
            min_strats   = c1.number_input("Min strategies",     min_value=1, max_value=10,
                                            value=1, step=1, key="gxm_min_strats")
            min_guidance = c2.number_input("Min guidance items", min_value=0, max_value=20,
                                            value=0, step=1, key="gxm_min_guidance")

            view = combined[
                (combined["n_strategies"] >= int(min_strats)) &
                (combined["n_guidance"]   >= int(min_guidance))
            ]
            st.dataframe(
                view[["symbol", "n_strategies", "n_guidance",
                       "quality", "guidance_multiplier", "combined_score"]],
                use_container_width=True, hide_index=True, height=500,
            )


# ---------- Doc Viewer ----------

def _match_by_key(df: pd.DataFrame, key: str) -> pd.DataFrame:
    """Return rows matching key against 'symbol' or 'isin' column (whichever exists)."""
    if df.empty:
        return df
    sym = df["symbol"].astype(str) == key if "symbol" in df.columns else pd.Series(False, index=df.index)
    isn = df["isin"].astype(str) == key if "isin" in df.columns else pd.Series(False, index=df.index)
    return df[sym | isn]


def page_doc_viewer():
    st.title("📄 Doc Viewer")
    st.caption(
        "Read pipeline documents with properly formatted tables. "
        "All pipe-table columns render as real grids here — no raw `|` characters."
    )

    key_input = st.text_input(
        "ISIN or NSE Symbol",
        placeholder="e.g. RELIANCE or INE002A01018",
        key="dv_key",
    ).strip().upper()

    tab_page, tab_gf, tab_results, tab_quarterly = st.tabs([
        "🏢 Company Page",
        "📋 GF Tables (GF1 / GF4 / Guidance)",
        "📊 Results",
        "📅 Quarterly Guidance (.md)",
    ])

    # ── Tab 1: Company Page ───────────────────────────────────────────
    with tab_page:
        if not key_input:
            st.info("Enter a symbol or ISIN above.")
        else:
            with st.spinner(f"Loading company page for {key_input}…"):
                content = find_company_page(key_input)
            if content is None:
                st.warning(
                    f"No company page found for **{key_input}**.  "
                    "It appears after at least one concall has been processed for this company."
                )
            else:
                st.download_button(
                    "⬇ Download .md",
                    data=content.encode("utf-8"),
                    file_name=f"{key_input}_company_page.md",
                    mime="text/markdown",
                    key="dv_dl_page",
                )
                st.markdown(content)

    # ── Tab 2: GF Tables ──────────────────────────────────────────────
    with tab_gf:
        if not key_input:
            st.info("Enter a symbol or ISIN above.")
        else:
            with st.spinner("Loading guidance & GF data…"):
                gt  = load_guidance_tracker()
                gf1 = load_gf1_statements()
                gf4 = load_gf4_flags()

            sym_gt  = _match_by_key(gt,  key_input)
            sym_gf1 = _match_by_key(gf1, key_input)
            sym_gf4 = _match_by_key(gf4, key_input)

            if sym_gt.empty and sym_gf1.empty and sym_gf4.empty:
                st.info(
                    f"No structured GF data found for **{key_input}** yet.  "
                    "This appears after concalls are processed."
                )
            else:
                if not sym_gt.empty:
                    st.subheader("Table A — Structured Guidance")
                    st.caption(f"{len(sym_gt)} rows across "
                               f"{sym_gt['quarter'].nunique() if 'quarter' in sym_gt.columns else '?'} quarter(s)")
                    st.dataframe(sym_gt, use_container_width=True, hide_index=True)

                if not sym_gf1.empty:
                    st.subheader("GF1 — Forward-Looking Statements")
                    st.caption(f"{len(sym_gf1)} statements")
                    # Show key columns first; keep the rest in an expander
                    key_cols_gf1 = [c for c in [
                        "quarter", "exact_statement", "metric_type",
                        "timeframe", "explicitness_type", "quantifiable", "numeric_value",
                    ] if c in sym_gf1.columns]
                    st.dataframe(sym_gf1[key_cols_gf1], use_container_width=True,
                                 hide_index=True, height=350)
                    with st.expander("All GF1 columns"):
                        st.dataframe(sym_gf1, use_container_width=True, hide_index=True)

                if not sym_gf4.empty:
                    st.subheader("GF4 — Quality Flags")
                    st.caption(f"{len(sym_gf4)} flags")
                    st.dataframe(sym_gf4, use_container_width=True, hide_index=True)

    # ── Tab 3: Results ────────────────────────────────────────────────
    with tab_results:
        if not key_input:
            st.info("Enter a symbol or ISIN above.")
        else:
            with st.spinner("Loading results…"):
                res = load_results_summary()
            sym_res = _match_by_key(res, key_input)
            # Also try matching on symbol via universe join if empty
            if sym_res.empty and not res.empty:
                universe = load_csv(["universe", "master_list.csv"])
                if not universe.empty and "symbol" in universe.columns and "isin" in universe.columns:
                    isin_for_sym = universe[universe["symbol"] == key_input]["isin"].tolist()
                    if isin_for_sym and "isin" in res.columns:
                        sym_res = res[res["isin"].isin(isin_for_sym)]

            if sym_res.empty:
                st.info(f"No results data found for **{key_input}**.")
            else:
                st.subheader("Quarterly Results Summary")
                show_cols = [c for c in [
                    "metric", "latest_q", "latest_val", "prev_q", "prev_val",
                    "yearago_q", "yearago_val", "yoy_pct", "qoq_pct",
                ] if c in sym_res.columns]
                st.dataframe(sym_res[show_cols], use_container_width=True, hide_index=True)

    # ── Tab 4: Quarterly Guidance .md ─────────────────────────────────
    with tab_quarterly:
        with st.spinner("Loading quarterly guidance file list…"):
            q_files = load_quarterly_index()

        if not q_files:
            st.info(
                "No quarterly guidance files yet. They appear after the Phase 2 "
                "pipeline processes concalls during a results season."
            )
        else:
            filenames = [f["filename"] for f in q_files]
            sel_file = st.selectbox(
                "Select quarterly file",
                filenames,
                key="dv_q_file",
            )
            file_meta = next((f for f in q_files if f["filename"] == sel_file), None)
            if file_meta:
                st.caption(f"Last modified: {file_meta['modified'][:10]}")
                with st.spinner(f"Loading {sel_file}…"):
                    qcontent = load_md_content(file_meta["file_id"])
                st.download_button(
                    "⬇ Download .md",
                    data=qcontent.encode("utf-8"),
                    file_name=sel_file,
                    mime="text/markdown",
                    key="dv_dl_quarterly",
                )
                st.markdown(qcontent)


# ---------- Market Overview helpers ----------

def _apply_segment_filter(features, segment, mcap_df):
    if segment == "All" or mcap_df.empty or "mcap_segment" not in mcap_df.columns:
        return features
    merged = features.merge(mcap_df[["symbol", "mcap_segment"]], on="symbol", how="left")
    return merged[merged["mcap_segment"] == segment]


def _macro_strip():
    items = [
        ("USD/INR", "USD_INR.parquet"),
        ("Brent",   "BRENT_CRUDE.parquet"),
        ("Dow",     "DOW_JONES.parquet"),
        ("Nasdaq",  "NASDAQ.parquet"),
    ]
    cols = st.columns(len(items))
    for col, (label, fname) in zip(cols, items):
        df = load_parquet(["data", "macro", fname])
        if df.empty or len(df) < 6:
            col.metric(label, "—")
            continue
        df = df.sort_values("date")
        last = float(df["close"].iloc[-1])
        prev = float(df["close"].iloc[-6])
        chg = (last / prev - 1) * 100 if prev else 0
        col.metric(label, f"{last:,.2f}", f"{chg:+.2f}% (5d)")

def _fii_dii_panel():
    df = load_csv(["data", "macro", "FII_DII.csv"])
    if df.empty or "category" not in df.columns or "net" not in df.columns:
        return
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    cat = df["category"].astype(str).str.upper()
    df = df.assign(grp=pd.NA)
    df.loc[cat.str.contains("FII") | cat.str.contains("FPI"), "grp"] = "FII"
    df.loc[cat.str.contains("DII"), "grp"] = "DII"
    df = df.dropna(subset=["grp"])
    if df.empty:
        return
    pivot = df.groupby(["date", "grp"])["net"].sum().unstack("grp").sort_index()

    st.subheader("FII / DII net cash flows (₹ cr)")
    latest = pivot.iloc[-1]
    c1, c2, c3 = st.columns(3)
    fii, dii = latest.get("FII"), latest.get("DII")
    c1.metric("FII net (latest)", f"₹{fii:,.0f} cr" if pd.notna(fii) else "—")
    c2.metric("DII net (latest)", f"₹{dii:,.0f} cr" if pd.notna(dii) else "—")
    c3.caption(f"as of {pivot.index[-1].date()}")

    recent = pivot.tail(20)
    fig = go.Figure()
    if "FII" in recent.columns:
        fig.add_trace(go.Bar(x=recent.index, y=recent["FII"], name="FII",
                             marker_color="#2980b9"))
    if "DII" in recent.columns:
        fig.add_trace(go.Bar(x=recent.index, y=recent["DII"], name="DII",
                             marker_color="#e67e22"))
    fig.add_hline(y=0, line=dict(color="gray", dash="dot"))
    fig.update_layout(barmode="group", height=300,
                      margin=dict(l=10, r=10, t=10, b=10),
                      legend=dict(orientation="h", yanchor="bottom", y=1.02))
    st.plotly_chart(fig, use_container_width=True)

def _breadth_panel(features):
    if features.empty:
        st.warning("No features to compute breadth.")
        return
    n = len(features)
    st.caption(f"Breadth computed over **{n}** symbols")

    # Multi-MA breadth
    ema_above = {
        "Above 10 EMA":  (features["days_above_ema_10"] > 0).mean() * 100,
        "Above 20 EMA":  (features["days_above_ema_20"] > 0).mean() * 100,
        "Above 50 EMA":  (features["days_above_ema_50"] > 0).mean() * 100,
        "Above 50 SMA":  features["above_50sma"].mean() * 100,
        "Above 200 SMA": features["above_200sma"].mean() * 100,
    }
    breadth_df = pd.DataFrame({"MA": list(ema_above.keys()),
                               "pct": list(ema_above.values())})
    colors = ["#27ae60" if v >= 60 else "#f39c12" if v >= 40 else "#e74c3c"
              for v in breadth_df["pct"]]
    fig = go.Figure(go.Bar(x=breadth_df["MA"], y=breadth_df["pct"],
                           marker_color=colors,
                           text=[f"{v:.0f}%" for v in breadth_df["pct"]],
                           textposition="outside"))
    fig.update_layout(title="% of stocks above each moving average",
                      yaxis_range=[0, 100], yaxis_title="%",
                      height=320, margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(fig, use_container_width=True)

    col_dist, col_trend = st.columns(2)

    # 52w high distribution
    with col_dist:
        d = features["dist_from_52w_high_pct"]
        buckets = {
            "At 52w high (≥-1%)":  int((d >= -1).sum()),
            "Within 5%":           int(((d < -1) & (d >= -5)).sum()),
            "Within 10%":          int(((d < -5) & (d >= -10)).sum()),
            "Within 25%":          int(((d < -10) & (d >= -25)).sum()),
            ">25% off high":       int((d < -25).sum()),
        }
        bdf = pd.DataFrame({"bucket": list(buckets.keys()),
                            "count": list(buckets.values())})
        bdf["pct"] = bdf["count"] / n * 100
        fig = go.Figure(go.Bar(
            x=bdf["count"], y=bdf["bucket"], orientation="h",
            marker_color=["#27ae60", "#2ecc71", "#f1c40f", "#e67e22", "#c0392b"],
            text=[f"{c} ({p:.0f}%)" for c, p in zip(bdf["count"], bdf["pct"])],
            textposition="outside",
        ))
        fig.update_layout(title="Distance from 52-week high",
                          height=300, margin=dict(l=10, r=80, t=40, b=10),
                          yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig, use_container_width=True)

    # Trend stack
    with col_trend:
        st.markdown("**Trend stack**")
        pct_50_above_200 = features["50sma_above_200sma"].mean() * 100
        pct_200_rising = features["200sma_rising"].mean() * 100
        new_high_count = int((features["dist_from_52w_high_pct"] >= -0.5).sum())
        new_low_count = int((features["dist_from_52w_low_pct"] <= 0.5).sum())

        st.metric("Golden cross (50 SMA > 200 SMA)", f"{pct_50_above_200:.1f}%")
        st.metric("200 SMA rising (1 mo)", f"{pct_200_rising:.1f}%")
        st.metric("New 52w highs / lows today",
                  f"{new_high_count} / {new_low_count}",
                  f"{new_high_count - new_low_count:+d} net")
        median_ret_3m = features["return_3m_pct"].median()
        median_ret_12m = features["return_12m_pct"].median()
        st.metric("Median 3m return",  f"{median_ret_3m:+.1f}%")
        st.metric("Median 12m return", f"{median_ret_12m:+.1f}%")


# ---------- Pages ----------

def page_market_overview():
    st.title("Market Overview")

    render_health_banner()
    state = load_parquet(["data", "market_state", "latest.parquet"])
    if state.empty:
        st.error("market_state/latest.parquet not found. Run `python scripts/market_state.py`.")
        return
    row = state.iloc[0]

    # Macro strip
    _macro_strip()
    st.markdown("---")

    # Health header
    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        st.metric("Market Health Score", f"{row['health_score']:.1f} / 100")
        st.metric("Regime", row["regime"])
    with col2:
        st.metric("Nifty 50", f"{row.get('nifty50_close', 0):,.2f}",
                  f"vs 200SMA {row.get('nifty50_sma200', 0):,.0f}")
        st.metric("India VIX", f"{row.get('india_vix', 0):.2f}",
                  f"{row.get('india_vix_5d_change', 0):+.2f} (5d)")
    with col3:
        comp_cols = [c for c in state.columns if c.endswith("_score")
                     and c != "health_score"]
        comp_df = pd.DataFrame({
            "component": [c.replace("_score", "") for c in comp_cols],
            "score": [float(row[c]) for c in comp_cols],
        })
        fig = go.Figure(go.Bar(x=comp_df["component"], y=comp_df["score"],
                               marker_color="#3498db"))
        fig.update_layout(title="Health-score components (0-100)", height=260,
                          yaxis_range=[0, 100],
                          margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # Universe breadth with segment toggle
    st.subheader("Universe breadth")

    mcap_df = load_csv(["universe", "market_cap.csv"])
    has_mcap = (not mcap_df.empty) and ("mcap_segment" in mcap_df.columns)
    if has_mcap:
        segment_options = ["All", "Largecap", "Midcap", "Smallcap", "Microcap"]
        seg_help = None
    else:
        segment_options = ["All"]
        seg_help = ("To enable segment filtering, run "
                    "`python scripts/enrich_market_cap.py` once. ~60-80 min.")
    segment = st.radio("Market-cap segment", segment_options,
                       horizontal=True, help=seg_help)
    if not has_mcap:
        st.caption("Showing All universe — market cap data not yet loaded.")

    features = load_parquet(["features", "latest.parquet"])
    filtered = _apply_segment_filter(features, segment, mcap_df)
    _breadth_panel(filtered)

    st.markdown("---")

    # Nifty 50 chart
    st.subheader("Nifty 50 (2-year)")
    nifty = load_parquet(["data", "indices", "NIFTY_50.parquet"])
    if not nifty.empty:
        nifty = nifty.sort_values("date").tail(504)
        nifty["sma_200"] = nifty["close"].rolling(200).mean()
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=nifty["date"], y=nifty["close"],
                                 name="Nifty 50", line=dict(color="#2c3e50")))
        fig.add_trace(go.Scatter(x=nifty["date"], y=nifty["sma_200"],
                                 name="200 SMA", line=dict(color="#e67e22", dash="dash")))
        fig.update_layout(height=380, margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    _fii_dii_panel()
    
    # Sector rotation
    st.subheader("Sector rotation (vs Nifty 500)")
    sec = load_csv(["data", "market_state", "sector_rotation_latest.csv"])
    if not sec.empty:
        fig = go.Figure(go.Scatter(
            x=sec["vs_nifty500_1m_pct"], y=sec["vs_nifty500_3m_pct"],
            mode="markers+text", text=sec["sector"], textposition="top center",
            marker=dict(size=12, color=sec["vs_nifty500_3m_pct"],
                        colorscale="RdYlGn", showscale=False),
        ))
        fig.add_hline(y=0, line=dict(color="gray", dash="dot"))
        fig.add_vline(x=0, line=dict(color="gray", dash="dot"))
        fig.update_layout(xaxis_title="1M relative return vs Nifty 500 (%)",
                          yaxis_title="3M relative return vs Nifty 500 (%)",
                          height=480, margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(sec, use_container_width=True, hide_index=True)


def page_signals():
    st.title("Today's Signals")
    signals = load_all_strategy_signals()
    if signals.empty:
        st.error("No signals found. Run the strategy scripts first.")
        return

    st.caption(f"Loaded {len(signals)} signals across {signals['strategy_group'].nunique()} strategies")

    col1, col2, col3 = st.columns(3)
    with col1:
        strategies = st.multiselect("Strategies",
            sorted(signals["strategy_group"].unique()),
            default=sorted(signals["strategy_group"].unique()))
    with col2:
        zones = st.multiselect("Zones",
            sorted(signals["zone_type"].dropna().unique()),
            default=["buy", "add"])
    with col3:
        search = st.text_input("Symbol contains").upper().strip()

    flt = signals.copy()
    flt = flt[flt["strategy_group"].isin(strategies)]
    flt = flt[flt["zone_type"].isin(zones)]
    if search:
        flt = flt[flt["symbol"].str.contains(search, na=False)]

    flt = flt.sort_values("score", ascending=False)

    with st.expander("📈 Filter by results growth (optional)", expanded=False):
        results  = load_results_summary()
        universe = load_csv(["universe", "master_list.csv"])

        if results.empty:
            st.caption("No results.parquet yet — run scrape_results_table.py first.")
        else:
            c_r1, c_r2 = st.columns(2)
            min_rev_yoy = c_r1.number_input(
                "Min Revenue YoY %", min_value=-100.0, max_value=500.0,
                value=0.0, step=5.0, key="sig_min_rev_yoy",
            )
            min_pat_yoy = c_r2.number_input(
                "Min PAT/Profit YoY %", min_value=-100.0, max_value=500.0,
                value=0.0, step=5.0, key="sig_min_pat_yoy",
            )
            apply_growth = st.checkbox("Apply growth filter", value=False,
                                       key="sig_apply_growth")
            if apply_growth:
                # Map ISIN → symbol from universe so results join works
                if (not universe.empty
                        and "isin" in universe.columns
                        and "symbol" in universe.columns):
                    isin_sym    = universe[["isin", "symbol"]].drop_duplicates()
                    results_sym = results.merge(isin_sym, on="isin", how="left")
                else:
                    results_sym = results.copy()
                    if "symbol" not in results_sym.columns:
                        results_sym["symbol"] = None

                rev_rows = results_sym[
                    results_sym["metric"].astype(str).str.lower()
                    .str.contains("revenue|sales|income from operations", na=False)
                ]
                pat_rows = results_sym[
                    results_sym["metric"].astype(str).str.lower()
                    .str.contains("net profit|pat|profit after tax", na=False)
                ]
                rev_best = (rev_rows.groupby("symbol")["yoy_pct"].max()
                            .reset_index(name="rev_yoy_pct"))
                pat_best = (pat_rows.groupby("symbol")["yoy_pct"].max()
                            .reset_index(name="pat_yoy_pct"))

                growth_pass = pd.DataFrame({"symbol": flt["symbol"].unique()})
                growth_pass = growth_pass.merge(rev_best, on="symbol", how="left")
                growth_pass = growth_pass.merge(pat_best, on="symbol", how="left")
                rev_ok = (growth_pass["rev_yoy_pct"].isna() |
                          (growth_pass["rev_yoy_pct"] >= min_rev_yoy))
                pat_ok = (growth_pass["pat_yoy_pct"].isna() |
                          (growth_pass["pat_yoy_pct"] >= min_pat_yoy))
                pass_syms = growth_pass[rev_ok & pat_ok]["symbol"].tolist()

                before = len(flt)
                flt = flt[flt["symbol"].isin(pass_syms)]
                st.caption(
                    f"Growth filter applied — {len(flt)} of {before} signals pass "
                    f"(rev YoY ≥ {min_rev_yoy:.0f}%, PAT YoY ≥ {min_pat_yoy:.0f}%)"
                )

    st.caption(f"Showing {len(flt)} signals")
    show_cols = [c for c in ["symbol", "strategy", "zone_type", "score",
                              "entry", "stop", "reason"] if c in flt.columns]
    st.dataframe(flt[show_cols], use_container_width=True, hide_index=True, height=600)

    selected_sym = st.text_input("Open Stock Detail for symbol (paste from table)",
                                  "").upper().strip()
    if selected_sym:
        st.session_state["stock_detail_symbol"] = selected_sym
        st.info(f"Switch to **Stock Detail** in the sidebar to view {selected_sym}.")


def page_stock_detail():
    st.title("Stock Detail")
    default_sym = st.session_state.get("stock_detail_symbol", "RELIANCE")
    symbol = st.text_input("Symbol", default_sym).upper().strip()
    if not symbol:
        return

    ohlcv = load_parquet(["data", "ohlcv", f"{symbol}.parquet"])
    if ohlcv.empty:
        st.error(f"No OHLCV found for {symbol}.")
        return
    ohlcv = ohlcv.sort_values("date").tail(252).reset_index(drop=True)

    features = load_parquet(["features", "latest.parquet"])
    feat_row = features[features["symbol"] == symbol]
    if not feat_row.empty:
        r = feat_row.iloc[0]
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Close", f"₹{r['close']:.2f}")
        c2.metric("3M return", f"{r['return_3m_pct']:.1f}%")
        c3.metric("RS rank (3M)", f"{r['rs_rank_3m']:.0f}")
        c4.metric("Dist from 52w high", f"{r['dist_from_52w_high_pct']:.1f}%")
        c5.metric("ADR%(20)", f"{r['adr_pct_20']:.2f}%")

    signals = load_all_strategy_signals()
    sym_sigs = signals[signals["symbol"] == symbol] if not signals.empty else pd.DataFrame()

    ohlcv["ema_20"] = ohlcv["close"].ewm(span=20).mean()
    ohlcv["ema_50"] = ohlcv["close"].ewm(span=50).mean()
    ohlcv["sma_200"] = ohlcv["close"].rolling(200).mean()

    n500 = load_parquet(["data", "indices", "NIFTY_500.parquet"]).sort_values("date")
    rs_series = None
    if not n500.empty:
        merged = ohlcv.merge(n500[["date", "close"]].rename(columns={"close": "n500"}),
                             on="date", how="left")
        rs_series = (merged["close"] / merged["close"].iloc[0]) / \
                    (merged["n500"] / merged["n500"].iloc[0]) * 100

    fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                        row_heights=[0.6, 0.2, 0.2], vertical_spacing=0.03,
                        subplot_titles=("Price + MAs", "Volume", "RS vs Nifty 500"))
    fig.add_trace(go.Candlestick(x=ohlcv["date"], open=ohlcv["open"], high=ohlcv["high"],
                                 low=ohlcv["low"], close=ohlcv["close"], name="Price"),
                  row=1, col=1)
    fig.add_trace(go.Scatter(x=ohlcv["date"], y=ohlcv["ema_20"], name="20 EMA",
                             line=dict(color="#2980b9", width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=ohlcv["date"], y=ohlcv["ema_50"], name="50 EMA",
                             line=dict(color="#8e44ad", width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=ohlcv["date"], y=ohlcv["sma_200"], name="200 SMA",
                             line=dict(color="#e67e22", width=1, dash="dash")),
                  row=1, col=1)

    if not sym_sigs.empty:
        for _, sig in sym_sigs.iterrows():
            zt = sig.get("zone_type")
            entry = sig.get("entry")
            stop = sig.get("stop")
            color = ZONE_COLORS.get(zt, "#666")
            if pd.notna(entry):
                fig.add_hline(y=entry, line=dict(color=color, width=1),
                              annotation_text=f"{sig.get('strategy', '')} {zt}",
                              annotation_position="right", row=1, col=1)
            if pd.notna(stop):
                fig.add_hline(y=stop, line=dict(color=ZONE_COLORS["stop_loss"],
                                                width=1, dash="dot"),
                              annotation_text=f"{sig.get('strategy', '')} stop",
                              annotation_position="right", row=1, col=1)

    fig.add_trace(go.Bar(x=ohlcv["date"], y=ohlcv["volume"], name="Volume",
                         marker_color="#95a5a6"), row=2, col=1)
    if rs_series is not None:
        fig.add_trace(go.Scatter(x=ohlcv["date"], y=rs_series, name="RS vs N500",
                                 line=dict(color="#16a085")), row=3, col=1)
        fig.add_hline(y=100, line=dict(color="gray", dash="dot"), row=3, col=1)

    fig.update_layout(height=800, xaxis_rangeslider_visible=False,
                      margin=dict(l=10, r=10, t=40, b=10), showlegend=True)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader(f"Signals on {symbol} today")
    if sym_sigs.empty:
        st.info(f"No strategies flagged {symbol} today.")
    else:
        for _, sig in sym_sigs.iterrows():
            strat = sig.get("strategy", "")
            doc = STRATEGY_DOCS.get(strat, {})
            with st.expander(f"{doc.get('title', strat)} — **{sig.get('zone_type')}** "
                             f"(score {sig.get('score'):.1f})"):
                st.markdown(f"**Intent:** {doc.get('intent', '(no doc)')}")
                if doc.get("rules"):
                    st.markdown("**Rules used:**")
                    for r in doc["rules"]:
                        st.markdown(f"- {r}")
                st.markdown(f"**Why this stock fired now:** {sig.get('reason', '')}")
                cols_to_show = [c for c in sig.index
                                if c not in ["symbol", "date", "strategy",
                                              "zone_type", "reason", "strategy_group"]
                                and pd.notna(sig[c])]
                if cols_to_show:
                    st.markdown("**Signal values:**")
                    st.json({c: (float(sig[c]) if isinstance(sig[c], (int, float))
                                 else str(sig[c])) for c in cols_to_show})

    # ── Management Guidance section ──────────────────────────────────────
    st.markdown("---")
    st.subheader("Management Guidance")
    with st.spinner("Loading guidance…"):
        gt_sd  = load_guidance_tracker()
        gf1_sd = load_gf1_statements()
        gf4_sd = load_gf4_flags()

    sym_gt  = gt_sd[gt_sd["symbol"]  == symbol] if not gt_sd.empty  else pd.DataFrame()
    sym_gf1 = gf1_sd[gf1_sd["symbol"] == symbol] if not gf1_sd.empty else pd.DataFrame()

    if sym_gt.empty and sym_gf1.empty:
        st.info(f"No management guidance found for {symbol} yet.")
    else:
        quality = _gf4_quality_score(gf4_sd, symbol) if not gf4_sd.empty else 0
        q_icon  = "🟢" if quality > 0 else ("🔴" if quality < 0 else "⚪")
        st.caption(f"GF4 guidance quality score: {q_icon} {quality:+d}")

        if not sym_gt.empty:
            st.markdown("**Structured guidance (Table A)**")
            active_gt = sym_gt[sym_gt["horizon_fy"].apply(_guidance_is_active)]
            if not active_gt.empty:
                show_cols_gt = [c for c in [
                    "quarter", "metric", "guidance_type",
                    "horizon_fy", "value", "unit", "cagr_pct", "notes",
                ] if c in active_gt.columns]
                st.dataframe(active_gt[show_cols_gt], use_container_width=True,
                             hide_index=True)
            else:
                st.caption("No active (current/future FY) guidance on record.")

            with st.expander("All quarters (including past guidance)"):
                past_cols_gt = [c for c in [
                    "quarter", "metric", "guidance_type",
                    "horizon_fy", "value", "unit", "cagr_pct", "notes",
                ] if c in sym_gt.columns]
                st.dataframe(sym_gt[past_cols_gt], use_container_width=True,
                             hide_index=True)

        if not sym_gf1.empty:
            with st.expander(f"📝 Raw forward-looking statements ({len(sym_gf1)})"):
                show_gf1_sd = [c for c in [
                    "quarter", "exact_statement", "metric_type",
                    "timeframe", "explicitness_type", "quantifiable", "numeric_value",
                ] if c in sym_gf1.columns]
                st.dataframe(sym_gf1[show_gf1_sd], use_container_width=True,
                             hide_index=True)


def page_strategy_docs():
    st.title("Strategy Documentation")
    st.markdown("Each strategy describes its intent, exact rules, what it's best at, and its caveats.")
    for key, doc in STRATEGY_DOCS.items():
        with st.expander(doc["title"]):
            st.markdown(f"**Intent:** {doc['intent']}")
            st.markdown("**Rules:**")
            for r in doc["rules"]:
                st.markdown(f"- {r}")
            st.markdown(f"**Best for:** {doc['best_for']}")
            st.markdown(f"**Caveat:** {doc['caveat']}")

def page_graphs():
    st.title("Graphs — signal chart gallery")
    signals = load_all_strategy_signals()
    if signals.empty:
        st.error("No signals found. Run the strategy scripts first.")
        return

    strat_groups = sorted(signals["strategy_group"].unique())
    zone_opts    = sorted(signals["zone_type"].dropna().unique())

    # ── Filters row ───────────────────────────────────────────────────────────
    c1, c2 = st.columns([3, 2])
    with c1:
        chosen = st.multiselect("Strategies", strat_groups, default=strat_groups)
    with c2:
        zones = st.multiselect("Zones", zone_opts, default=["buy", "add"])

    c3, c4, c5, c6 = st.columns(4)
    with c3:
        min_strats = st.number_input("Min strategies", min_value=1,
                                     max_value=10, value=1, step=1)
    with c4:
        tf = st.selectbox("Timeframe", list(TIMEFRAME_DAYS.keys()), index=2)
    with c5:
        view_mode = st.radio("View mode", ["Quick Scan", "Detailed"],
                             horizontal=True,
                             help="Quick Scan: 3-column line charts, scroll all at once. "
                                  "Detailed: full candlestick with volume, paginated.")
    with c6:
        if view_mode == "Detailed":
            per_page = int(st.number_input("Charts/page", min_value=2,
                                           max_value=20, value=6, step=2))

    # ── Filter signals ────────────────────────────────────────────────────────
    sel = signals[signals["strategy_group"].isin(chosen)]
    if zones:
        sel = sel[sel["zone_type"].isin(zones)]
    if sel.empty:
        st.info("No signals match this filter.")
        return

    conv = (sel.groupby("symbol")["strategy_group"].nunique()
            .reset_index(name="n_strategies"))
    conv = conv[conv["n_strategies"] >= int(min_strats)]
    if conv.empty:
        st.info(f"No stock satisfies ≥ {int(min_strats)} strategies. Lower the threshold.")
        return

    best = sel.groupby("symbol")["score"].max().reset_index(name="best_score")
    conv = conv.merge(best, on="symbol", how="left")
    conv = conv.sort_values(["n_strategies", "best_score"],
                            ascending=[False, False]).reset_index(drop=True)

    total = len(conv)
    sym_list = tuple(conv["symbol"].tolist())
    st.caption(f"{total} stock(s) matching filters")

    # ─────────────────────────────────────────────────────────────────────────
    # QUICK SCAN MODE — load all OHLCV in one Drive session, render 3-col grid
    # ─────────────────────────────────────────────────────────────────────────
    if view_mode == "Quick Scan":
        st.info(
            f"Loading {total} charts. First load downloads from Drive (~1-3 min "
            f"for 300+ stocks). Subsequent visits use the 30-min cache — instant.",
            icon="ℹ️",
        )
        prog = st.progress(0, text="Loading OHLCV from Drive (one batch call)…")
        ohlcv_map = load_ohlcv_bulk(sym_list)
        prog.progress(100, text="Done — rendering charts…")

        tf_days = TIMEFRAME_DAYS[tf]
        cols3   = st.columns(3)

        for i, (_, crow) in enumerate(conv.iterrows()):
            sym      = crow["symbol"]
            sym_sigs = sel[sel["symbol"] == sym]
            ohlcv    = ohlcv_map.get(sym, pd.DataFrame())

            with cols3[i % 3]:
                zone_colors_html = " ".join(
                    f'<span style="background:{ZONE_COLORS.get(sg.get("zone_type",""),"#666")};'
                    f'color:white;padding:1px 5px;border-radius:8px;font-size:10px;">'
                    f'{sg.get("zone_type","")}</span>'
                    for _, sg in sym_sigs.drop_duplicates("zone_type").iterrows()
                )
                st.markdown(zone_colors_html, unsafe_allow_html=True)
                if ohlcv.empty:
                    st.caption(f"{sym} — no data")
                else:
                    fig = build_quick_chart(sym, ohlcv, sym_sigs, tf_days)
                    st.plotly_chart(fig, use_container_width=True,
                                    key=f"qs_{sym}", config={"displayModeBar": False})
        return

    # ─────────────────────────────────────────────────────────────────────────
    # DETAILED MODE — paginated candlestick + volume
    # ─────────────────────────────────────────────────────────────────────────
    n_pages = max(1, (total + per_page - 1) // per_page)
    if "graphs_page" not in st.session_state:
        st.session_state["graphs_page"] = 0
    st.session_state["graphs_page"] = min(st.session_state["graphs_page"], n_pages - 1)
    page = st.session_state["graphs_page"]

    def _nav(suffix):
        at_first = page <= 0
        at_last  = page >= n_pages - 1
        n1, n2, n3, n4, n5 = st.columns([1, 1, 2, 1, 1])
        with n1:
            if st.button("⏮ First", key=f"first_{suffix}",
                         disabled=at_first, use_container_width=True):
                st.session_state["graphs_page"] = 0; st.rerun()
        with n2:
            if st.button("◀ Prev", key=f"prev_{suffix}",
                         disabled=at_first, use_container_width=True):
                st.session_state["graphs_page"] = page - 1; st.rerun()
        with n3:
            if suffix == "top":
                jump = st.number_input(
                    f"Page (1–{n_pages})", min_value=1, max_value=n_pages,
                    value=page + 1, step=1, label_visibility="collapsed",
                    key=f"graphs_jump_{n_pages}_{per_page}",
                )
                st.caption(f"of {n_pages}  ·  {total} stock(s)  — type page & press Enter")
                if jump - 1 != page:
                    st.session_state["graphs_page"] = jump - 1; st.rerun()
            else:
                st.markdown(f"<div style='text-align:center;padding-top:8px;'>"
                            f"Page {page + 1} of {n_pages}</div>",
                            unsafe_allow_html=True)
        with n4:
            if st.button("Next ▶", key=f"next_{suffix}",
                         disabled=at_last, use_container_width=True):
                st.session_state["graphs_page"] = page + 1; st.rerun()
        with n5:
            if st.button("Last ⏭", key=f"last_{suffix}",
                         disabled=at_last, use_container_width=True):
                st.session_state["graphs_page"] = n_pages - 1; st.rerun()

    _nav("top")
    st.markdown("---")

    # Load only the symbols on this page in one bulk call
    page_syms = tuple(conv.iloc[page * per_page:(page + 1) * per_page]["symbol"].tolist())
    with st.spinner(f"Loading {len(page_syms)} charts from Drive…"):
        ohlcv_map = load_ohlcv_bulk(page_syms)

    for _, crow in conv.iloc[page * per_page:(page + 1) * per_page].iterrows():
        sym      = crow["symbol"]
        sym_sigs = sel[sel["symbol"] == sym]
        st.markdown(f"### {sym}  ·  {int(crow['n_strategies'])} strategies")
        chips = []
        for _, sg in sym_sigs.iterrows():
            z = sg.get("zone_type", "")
            s = sg.get("strategy", sg.get("strategy_group", ""))
            chips.append(f'<span style="background:{ZONE_COLORS.get(z,"#666")};color:white;'
                         f'padding:2px 8px;border-radius:10px;font-size:12px;'
                         f'margin-right:4px;">{s} · {z}</span>')
        st.markdown(" ".join(chips), unsafe_allow_html=True)
        ohlcv = ohlcv_map.get(sym, pd.DataFrame())
        if ohlcv.empty:
            st.caption("No OHLCV available for this symbol.")
        else:
            fig = build_stock_chart(sym, ohlcv, sym_sigs, TIMEFRAME_DAYS[tf], height=460)
            st.plotly_chart(fig, use_container_width=True, key=f"graphs_{sym}")
        st.markdown("---")

    _nav("bottom")






def _find_latest_portfolio_file(drive, portfolio_folder_id):
    """Return (file_id, filename) of the most-recently-modified .xls/.xlsx/.csv
    file in the portfolio folder. None, None if nothing matching."""
    files = drive.files().list(
        q=f"'{portfolio_folder_id}' in parents and trashed=false",
        fields="files(id, name, modifiedTime)",
        orderBy="modifiedTime desc",
    ).execute().get("files", [])
    for f in files:
        n = f["name"].lower()
        if n.endswith((".xls", ".xlsx", ".csv")):
            return f["id"], f["name"]
    return None, None


def _read_portfolio_table(raw_bytes: bytes, filename: str):
    """Read Screener.in's Investment export. Auto-detects the header row
    (Screener leaves ~13 blank rows above it). Returns a normalized DataFrame
    with our standard column names."""
    import io
    import pandas as pd
    fn = filename.lower()
    if fn.endswith(".csv"):
        df_raw = pd.read_csv(io.BytesIO(raw_bytes), header=None)
        engine = "csv"
    else:
        engine = "xlrd" if fn.endswith(".xls") else "openpyxl"
        df_raw = pd.read_excel(io.BytesIO(raw_bytes), engine=engine, header=None)

    # Find the header row: first row that contains "ISIN" cell
    header_row = None
    for i, row in df_raw.iterrows():
        if any(str(v).strip().upper() == "ISIN" for v in row.dropna()):
            header_row = i
            break
    if header_row is None:
        raise ValueError("Could not find an 'ISIN' header in the file")

    if engine == "csv":
        df = pd.read_csv(io.BytesIO(raw_bytes), header=header_row)
    else:
        df = pd.read_excel(io.BytesIO(raw_bytes), engine=engine, header=header_row)

    # Drop rows without ISIN (totals rows, footers)
    df = df.dropna(subset=["ISIN"]).copy()

    # Normalize column names → our convention.
    # Handles both Screener "My Investments" export and broker "Holdings Statement" format.
    rename = {
        "ISIN":             "isin",
        # company name — Screener vs broker
        "Company name":     "screener_name",
        "Stock/ETF Name":   "screener_name",
        # rating — Screener only
        "Rating":           "screener_rating",
        # price — Screener vs broker
        "Last price":       "screener_last_price",
        "Market Price":     "screener_last_price",
        "Last price date":  "screener_last_price_date",
        # intraday change — Screener only
        "1D Change (INR)":  "one_d_change_inr",
        "1D Change (%)":    "one_d_change_pct",
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
    df["isin"] = df["isin"].astype(str).str.strip()
    return df


def _resolve_isins(pf, universe):
    """Join portfolio to universe on ISIN to recover NSE symbol + name."""
    merged = pf.merge(universe[["symbol", "isin", "name"]], on="isin", how="left")
    return merged


def page_portfolio():
    import pandas as pd
    import streamlit as st

    st.title("My Portfolio")
    st.caption("Reads the latest Screener.in 'My Investments' export from your "
               "Drive `portfolio/` folder.")

    # Locate file
    drive = drive_service()
    folder_id = os.environ["GDRIVE_FOLDER_ID"]
    pf_folder_id = _find_subfolder(drive, folder_id, "portfolio")
    if not pf_folder_id:
        st.warning(
            "**Setup**: in your Drive `signals-india` folder, create a subfolder "
            "named **`portfolio`** and upload your Screener `my-investment-overview.xls` "
            "export. The page auto-picks the most recent file.")
        return
    file_id, fname = _find_latest_portfolio_file(drive, pf_folder_id)
    if not file_id:
        st.warning("No .xls / .xlsx / .csv found in the `portfolio/` folder. "
                   "Upload your Screener export there.")
        return

    st.caption(f"Using file: **{fname}**")

    # Read + normalize
    raw = _download_bytes(drive, file_id)
    try:
        pf = _read_portfolio_table(raw, fname)
    except Exception as e:
        st.error(f"Could not parse the file: {e}")
        st.info("If this is a Screener export, the columns should include ISIN, "
                "Company name, Rating, Last price, 1D Change. Re-download from "
                "Screener if needed.")
        return

    # Resolve ISINs → NSE symbol
    universe = load_csv(["universe", "master_list.csv"])
    if universe.empty:
        st.error("universe/master_list.csv missing. Run `build_universe.py` first.")
        return
    pf = _resolve_isins(pf, universe)

    unresolved = pf[pf["symbol"].isna()]
    if not unresolved.empty:
        with st.expander(f"{len(unresolved)} ISIN(s) not in our NSE universe", expanded=False):
            cols = [c for c in ["isin", "screener_name", "screener_rating",
                                 "screener_last_price"] if c in unresolved.columns]
            st.dataframe(unresolved[cols], use_container_width=True, hide_index=True)
            st.caption("Likely BSE-only listings, recently delisted, or pre-IPO. "
                       "They're skipped below.")
    pf = pf.dropna(subset=["symbol"]).copy()
    if pf.empty:
        st.warning("None of your ISINs resolved to NSE symbols.")
        return

    # Pull today's features for the resolved set
    features = load_parquet(["features", "latest.parquet"])
    feat_cols = ["symbol", "close", "return_1m_pct", "return_3m_pct", "return_12m_pct",
                 "rs_rank_3m", "rs_rank_12m",
                 "dist_from_52w_high_pct", "above_200sma", "adr_pct_20"]
    feat_cols = [c for c in feat_cols if c in features.columns]
    pf = pf.merge(features[feat_cols], on="symbol", how="left")

    # Today's signal state per holding (from combined signals)
    signals = load_all_strategy_signals()

    # Per-holding aggregated signal info
    if not signals.empty:
        signal_summary = (signals.groupby("symbol")
                          .agg(n_strategies=("strategy", "nunique"),
                               zone_types=("zone_type",
                                            lambda x: ",".join(sorted(set(x.dropna())))))
                          .reset_index())
        pf = pf.merge(signal_summary, on="symbol", how="left")
    else:
        pf["n_strategies"] = 0
        pf["zone_types"] = ""
    pf["n_strategies"] = pf["n_strategies"].fillna(0).astype(int)
    pf["zone_types"] = pf["zone_types"].fillna("")

    # --- Top summary metrics ---
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Holdings", f"{len(pf)}")

    # Average rating (count stars in the rating string)
    def _stars(r):
        s = str(r) if pd.notna(r) else ""
        return s.count("★") if "★" in s else s.count("*")
    pf["rating_stars"] = pf["screener_rating"].apply(_stars)
    avg_stars = pf.loc[pf["rating_stars"] > 0, "rating_stars"].mean()
    c2.metric("Avg ★ rating", f"{avg_stars:.1f}" if pd.notna(avg_stars) else "—")

    n_above_200 = int(pf["above_200sma"].fillna(False).sum())
    c3.metric("Above 200 SMA", f"{n_above_200} / {len(pf)}")

    n_with_signal = int((pf["n_strategies"] > 0).sum())
    c4.metric("Flagged today", f"{n_with_signal} / {len(pf)}")

    n_buy_or_add = int(pf["zone_types"].str.contains("buy|add", na=False).sum())
    c5.metric("Buy/Add today", f"{n_buy_or_add}")

    st.markdown("---")

    # --- Holdings table ---
    st.subheader("Holdings")

    sort_by = st.selectbox(
        "Sort by",
        ["3M return (high→low)", "RS rank 3M (high→low)",
         "★ rating (high→low)", "1D change % (high→low)",
         "Dist from 52w high (close→far)", "# strategies flagging (high→low)",
         "Symbol (A→Z)"],
        index=0,
    )
    sort_map = {
        "3M return (high→low)": ("return_3m_pct", False),
        "RS rank 3M (high→low)": ("rs_rank_3m", False),
        "★ rating (high→low)": ("rating_stars", False),
        "1D change % (high→low)": ("one_d_change_pct", False),
        "Dist from 52w high (close→far)": ("dist_from_52w_high_pct", False),
        "# strategies flagging (high→low)": ("n_strategies", False),
        "Symbol (A→Z)": ("symbol", True),
    }
    sort_col, asc = sort_map[sort_by]
    if sort_col in pf.columns:
        pf = pf.sort_values(sort_col, ascending=asc, na_position="last")

    show_cols = [c for c in [
        "symbol", "name", "screener_rating", "screener_last_price",
        "one_d_change_pct", "return_1m_pct", "return_3m_pct", "return_12m_pct",
        "rs_rank_3m", "dist_from_52w_high_pct", "above_200sma",
        "adr_pct_20", "n_strategies", "zone_types",
    ] if c in pf.columns]
    st.dataframe(pf[show_cols], use_container_width=True, hide_index=True, height=420)

    st.markdown("---")

    # --- Chart cards ---
    st.subheader("Charts (per holding)")
    pf_c1, pf_c2 = st.columns(2)
    tf = pf_c1.selectbox("Timeframe", list(TIMEFRAME_DAYS.keys()),
                         index=2, key="pf_tf")
    pf_per_page = pf_c2.number_input("Charts per page", min_value=2,
                                     max_value=20, value=5, step=1, key="pf_per_page")

    pf_total = len(pf)
    pf_per_page = int(pf_per_page)
    pf_n_pages = max(1, (pf_total + pf_per_page - 1) // pf_per_page)
    if "pf_page" not in st.session_state:
        st.session_state["pf_page"] = 0
    st.session_state["pf_page"] = min(st.session_state["pf_page"], pf_n_pages - 1)
    pf_page = st.session_state["pf_page"]

    pf_a, pf_b, pf_c, pf_d = st.columns([1, 1, 1, 1])
    if pf_a.button("◀ Prev", key="pf_prev", disabled=(pf_page <= 0)):
        st.session_state["pf_page"] = pf_page - 1
        st.rerun()
    pf_b.caption(f"Page {pf_page + 1} of {pf_n_pages}  ·  {pf_total} holdings")
    if pf_c.button("Next ▶", key="pf_next", disabled=(pf_page >= pf_n_pages - 1)):
        st.session_state["pf_page"] = pf_page + 1
        st.rerun()
    if pf_d.button("Reset", key="pf_reset"):
        st.session_state["pf_page"] = 0
        st.rerun()

    pf_start = pf_page * pf_per_page
    pf_slice = pf.iloc[pf_start:pf_start + pf_per_page]

    for _, h in pf_slice.iterrows():
        sym = h["symbol"]
        with st.container():
            c1, c2, c3, c4, c5 = st.columns([3, 1, 1, 1, 1])
            c1.markdown(f"### {sym} — {h.get('name', h.get('screener_name', ''))}")
            c2.metric("★", h.get("screener_rating", "—"))
            scr_price = h.get("screener_last_price")
            c3.metric("Last px", f"₹{scr_price:.2f}" if pd.notna(scr_price) else "—")
            chg = h.get("one_d_change_pct")
            try:
                chg_str = f"{float(chg):+.2f}%" if pd.notna(chg) else "—"
            except Exception:
                chg_str = "—"
            c4.metric("1D Δ", chg_str)
            ret_3m = h.get("return_3m_pct")
            c5.metric("3M ret", f"{ret_3m:+.1f}%" if pd.notna(ret_3m) else "—")

            sym_sigs = signals[signals["symbol"] == sym] if not signals.empty else pd.DataFrame()
            if not sym_sigs.empty:
                chips = []
                for _, sig in sym_sigs.iterrows():
                    z = sig.get("zone_type", "")
                    s = sig.get("strategy", "")
                    color = ZONE_COLORS.get(z, "#666")
                    chips.append(f'<span style="background:{color};color:white;'
                                 f'padding:2px 8px;border-radius:10px;'
                                 f'font-size:12px;margin-right:4px;">{s} · {z}</span>')
                st.markdown("**Today's signals:** " + " ".join(chips),
                            unsafe_allow_html=True)
            else:
                st.caption("No strategies flagged this stock today.")

            ohlcv = load_parquet(["data", "ohlcv", f"{sym}.parquet"])
            if not ohlcv.empty:
                fig = build_stock_chart(sym, ohlcv, sym_sigs,
                                        TIMEFRAME_DAYS[tf], height=480)
                st.plotly_chart(fig, use_container_width=True, key=f"pf_{sym}")
            else:
                st.caption("No OHLCV available for this symbol.")

            st.markdown("---")


# ---------- Main ----------

def page_doc_upload():
    st.title("📂 Document Library")
    st.caption(
        "Upload company documents to Drive. These become the input for "
        "AI Deep-Dive analysis. Supports Annual Reports, analyst notes, "
        "credit ratings, transcripts and more."
    )

    tab_upload, tab_browse = st.tabs(["⬆ Upload Documents", "📋 My Library"])

    # ── Tab 1: Upload ─────────────────────────────────────────────────────────
    with tab_upload:
        st.subheader("Upload new documents")

        col1, col2 = st.columns(2)
        company_key = col1.text_input(
            "Company ISIN or NSE Symbol",
            placeholder="e.g. RELIANCE or INE002A01018",
            key="du_company",
        ).strip().upper()
        company_name = col2.text_input(
            "Company name (for display)",
            placeholder="e.g. Reliance Industries",
            key="du_cname",
        ).strip()

        col3, col4 = st.columns(2)
        doc_type = col3.selectbox("Document type", _DOC_TYPES, key="du_dtype")
        label    = col4.text_input(
            "Year / label",
            placeholder="e.g. FY25  or  Q4FY26  or  May2026",
            key="du_label",
        ).strip()

        uploaded_files = st.file_uploader(
            "Select file(s) — PDF, DOCX, TXT, XLSX accepted",
            type=["pdf", "docx", "doc", "txt", "xlsx", "csv"],
            accept_multiple_files=True,
            key="du_files",
        )

        if uploaded_files and company_key:
            st.caption(f"{len(uploaded_files)} file(s) ready to upload for **{company_key}**")

        if st.button("⬆ Upload to Drive", type="primary",
                     disabled=not (uploaded_files and company_key)):
            drive = drive_service()
            comp_folder = _user_docs_company_folder(drive, company_key)
            manifest    = _read_manifest(drive, comp_folder)
            existing_names = {e["filename"] for e in manifest}

            progress = st.progress(0)
            results  = []

            for i, f in enumerate(uploaded_files):
                ext      = "." + f.name.rsplit(".", 1)[-1].lower()
                slug     = _DOC_TYPE_SLUG.get(doc_type, "doc")
                lbl_safe = label.replace(" ", "_").replace("/", "_") if label else "unlabelled"
                fname    = f"{slug}_{lbl_safe}{ext}"

                # Avoid name collision
                if fname in existing_names:
                    base = f"{slug}_{lbl_safe}"
                    for n in range(2, 100):
                        candidate = f"{base}_{n}{ext}"
                        if candidate not in existing_names:
                            fname = candidate
                            break

                mime    = _MIME.get(ext, "application/octet-stream")
                content = f.read()

                try:
                    fid = _upload_bytes_to_drive(drive, comp_folder, fname, content, mime)
                    entry = {
                        "file_id":       fid,
                        "filename":      fname,
                        "original_name": f.name,
                        "doc_type":      doc_type,
                        "label":         label,
                        "company_key":   company_key,
                        "company_name":  company_name,
                        "uploaded_at":   datetime.now().isoformat(),
                        "size_kb":       round(len(content) / 1024, 1),
                        "deep_dive_status": "pending",
                    }
                    manifest.append(entry)
                    existing_names.add(fname)
                    results.append(("✅", fname, f"{len(content)//1024:.0f} KB"))
                except Exception as e:
                    results.append(("❌", f.name, str(e)[:60]))

                progress.progress((i + 1) / len(uploaded_files))

            _write_manifest(drive, comp_folder, manifest)
            st.cache_data.clear()   # refresh library tab

            for icon, name, info in results:
                st.write(f"{icon} **{name}** — {info}")
            if all(r[0] == "✅" for r in results):
                st.success(f"Uploaded {len(results)} file(s) for **{company_key}**. "
                           "Ready for Deep-Dive analysis.")

    # ── Tab 2: Browse library ─────────────────────────────────────────────────
    with tab_browse:
        st.subheader("Uploaded document library")

        try:
            drive     = drive_service()
            folder_id = os.environ["GDRIVE_FOLDER_ID"]
            udocs_id  = _find_subfolder(drive, folder_id, "user_docs")

            if not udocs_id:
                st.info("No documents uploaded yet. Use the Upload tab to add your first document.")
                return

            # List all company folders
            q = (f"'{udocs_id}' in parents and "
                 f"mimeType='application/vnd.google-apps.folder' and trashed=false")
            companies = drive.files().list(q=q, fields="files(id,name)").execute().get("files", [])

            if not companies:
                st.info("No documents yet.")
                return

            all_docs: list[dict] = []
            for comp in sorted(companies, key=lambda x: x["name"]):
                manifest = _read_manifest(drive, comp["id"])
                for entry in manifest:
                    entry["_folder_id"]   = comp["id"]
                    entry["_company_key"] = comp["name"]
                    all_docs.append(entry)

            if not all_docs:
                st.info("Folders exist but no documents uploaded yet.")
                return

            # Summary metrics
            n_companies = len({d["_company_key"] for d in all_docs})
            n_pending   = sum(1 for d in all_docs
                              if d.get("deep_dive_status") == "pending")
            c1, c2, c3 = st.columns(3)
            c1.metric("Companies", n_companies)
            c2.metric("Total documents", len(all_docs))
            c3.metric("Pending deep-dive", n_pending)

            st.markdown("---")

            # Company filter
            comp_names = sorted({d["_company_key"] for d in all_docs})
            sel = st.selectbox("Filter by company", ["All"] + comp_names, key="dl_sel")
            filtered = all_docs if sel == "All" else [d for d in all_docs
                                                      if d["_company_key"] == sel]

            # Display table
            rows = []
            for d in filtered:
                rows.append({
                    "Company":     d.get("_company_key", ""),
                    "Name":        d.get("company_name", ""),
                    "File":        d.get("filename", ""),
                    "Type":        d.get("doc_type", ""),
                    "Label":       d.get("label", ""),
                    "Size (KB)":   d.get("size_kb", ""),
                    "Uploaded":    str(d.get("uploaded_at", ""))[:10],
                    "Deep-dive":   d.get("deep_dive_status", ""),
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True,
                         hide_index=True, height=400)

            # Per-company ready indicator
            if sel != "All":
                n_docs = len(filtered)
                n_ar   = sum(1 for d in filtered if "Annual Report" in d.get("doc_type", ""))
                st.caption(
                    f"**{sel}**: {n_docs} document(s), {n_ar} Annual Report(s). "
                    + ("✅ Ready for Deep-Dive." if n_ar >= 1 else
                       "⚠️ Upload at least 1 Annual Report for best Deep-Dive results.")
                )

        except Exception as e:
            st.error(f"Could not load library: {e}")


def _safe_render(page_fn):
    """Run a page function; show a recoverable error instead of crashing."""
    try:
        page_fn()
    except Exception as exc:
        st.error(f"⚠️ Page error: {str(exc)[:300]}")
        st.caption(
            "This is usually a temporary Drive connection issue. "
            "Refresh the page (F5) or wait a moment and try again."
        )
        if st.button("🔄 Clear cache & retry"):
            st.cache_data.clear()
            st.rerun()


def main():
    st.sidebar.title("Signals India")

    page = st.sidebar.radio("Page", [
        "Market Overview",
        "Today's Signals",
        "Company Intel",
        "Mgmt Guidance",
        "Doc Viewer",
        "Doc Library",
        "My Portfolio",
        "PF Tracking",
        "Graphs",
        "Stock Detail",
        "Strategy Docs",
    ])
    render_health_sidebar()
    st.sidebar.markdown("---")
    st.sidebar.caption(f"Loaded at {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    st.sidebar.caption("Data refreshes from Drive every 5 min (cache TTL)")

    if page == "Market Overview":
        _safe_render(page_market_overview)
    elif page == "Today's Signals":
        _safe_render(page_signals)
    elif page == "Company Intel":
        _safe_render(page_company_intel)
    elif page == "Mgmt Guidance":
        _safe_render(page_guidance)
    elif page == "Doc Viewer":
        _safe_render(page_doc_viewer)
    elif page == "Doc Library":
        _safe_render(page_doc_upload)
    elif page == "My Portfolio":
        _safe_render(page_portfolio)
    elif page == "Graphs":
        _safe_render(page_graphs)
    elif page == "Stock Detail":
        _safe_render(page_stock_detail)
    elif page == "Strategy Docs":
        _safe_render(page_strategy_docs)
    elif page == "PF Tracking":
        _safe_render(page_pf_tracking)



main()
