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

def build_quick_chart(symbol, ohlcv, signals_for_stock, timeframe_days,
                      normalize: bool = True):
    """Lightweight line + volume chart for quick scan.

    normalize=True  → all prices indexed to 0% at window start so every chart
                      uses the same scale and run-ups are directly comparable.
    normalize=False → absolute ₹ prices (default axis scale per stock).
    """
    full = ohlcv.sort_values("date").reset_index(drop=True)
    df   = full.tail(timeframe_days).reset_index(drop=True)
    df["ema_20"] = df["close"].ewm(span=20).mean()
    df["ema_50"] = df["close"].ewm(span=50).mean()

    # ── Return labels (always from full history, not window) ──────────────────
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

    ret_parts = "  ".join(
        f'<span style="color:{"#27ae60" if v >= 0 else "#e74c3c"}">'
        f'{label} {v:+.0f}%</span>'
        for label, v in rets.items()
    )

    last_close = float(df["close"].iloc[-1]) if not df.empty else 0
    # Returns on second line so long names never crowd them out
    title_html = (f"<b>{symbol}</b>  ₹{last_close:,.0f}"
                  f"<br><span style='font-size:10px'>{ret_parts}</span>")

    # ── Normalise to % from first close in window ─────────────────────────────
    base = float(df["close"].iloc[0]) if not df.empty and df["close"].iloc[0] != 0 else 1.0
    if normalize:
        y_close  = (df["close"]  / base - 1) * 100
        y_ema20  = (df["ema_20"] / base - 1) * 100
        y_ema50  = (df["ema_50"] / base - 1) * 100
        y_suffix = "%"
        ytick_fmt = ".0f"
    else:
        y_close  = df["close"]
        y_ema20  = df["ema_20"]
        y_ema50  = df["ema_50"]
        y_suffix = ""
        ytick_fmt = ",.0f"

    # ── Volume colour: green if close ≥ open, else red ───────────────────────
    vol_colors = [
        "#27ae60" if c >= o else "#e74c3c"
        for c, o in zip(df["close"], df["open"])
    ]

    # ── Date tick: start and end only ────────────────────────────────────────
    dates     = df["date"].tolist()
    tick_vals = [dates[0], dates[-1]]
    tick_text = [str(dates[0])[:10], str(dates[-1])[:10]]

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        row_heights=[0.72, 0.28], vertical_spacing=0.02,
    )

    # Price + EMAs
    fig.add_trace(go.Scatter(x=df["date"], y=y_close,
                             line=dict(color="#2c3e50", width=1.5),
                             showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=df["date"], y=y_ema20,
                             line=dict(color="#2980b9", width=1),
                             showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=df["date"], y=y_ema50,
                             line=dict(color="#8e44ad", width=1),
                             showlegend=False), row=1, col=1)

    # Zero / baseline reference line when normalised
    if normalize:
        fig.add_hline(y=0, line=dict(color="#bdc3c7", width=0.8, dash="dot"),
                      row=1, col=1)

    # Signal zone lines — labelled so the colour is self-explanatory
    if not signals_for_stock.empty:
        seen_zones: set = set()
        for _, sig in signals_for_stock.iterrows():
            zt    = sig.get("zone_type", "")
            entry = sig.get("entry")
            if not pd.notna(entry):
                continue
            y_val = (float(entry) / base - 1) * 100 if normalize else float(entry)
            color = ZONE_COLORS.get(zt, "#666")
            ann   = zt if zt not in seen_zones else ""   # label each zone type once
            seen_zones.add(zt)
            fig.add_hline(
                y=y_val,
                line=dict(color=color, width=1, dash="dash"),
                annotation_text=ann,
                annotation_position="right",
                annotation_font=dict(size=8, color=color),
                row=1, col=1,
            )

    # Volume bars
    fig.add_trace(go.Bar(x=df["date"], y=df["volume"],
                         marker_color=vol_colors, showlegend=False), row=2, col=1)

    fig.update_layout(
        height=480,
        title=dict(text=title_html, font=dict(size=13), x=0.02, y=0.97),
        margin=dict(l=4, r=55, t=52, b=4),   # r=55 leaves room for hline labels
        plot_bgcolor="#fafafa",
        paper_bgcolor="white",
        bargap=0.1,
    )
    xaxis_cfg = dict(showgrid=False, zeroline=False,
                     tickvals=tick_vals, ticktext=tick_text,
                     tickfont=dict(size=8), tickangle=0)
    fig.update_xaxes(**xaxis_cfg, row=1, col=1)
    fig.update_xaxes(**xaxis_cfg, row=2, col=1)
    fig.update_yaxes(showgrid=False, zeroline=False, tickfont=dict(size=8),
                     ticksuffix=y_suffix, tickformat=ytick_fmt, row=1, col=1)
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

@st.cache_data(ttl=1800, show_spinner=False)
def _resolve_path_folder(path_parts: tuple):
    """Resolve a chain of subfolder names to the final folder ID, cached so
    repeated loads from the same folder don't re-traverse Drive every call.
    Returns the folder ID, or None if any segment is missing. Empty path → root."""
    def _do():
        drive = drive_service()
        parent = os.environ["GDRIVE_FOLDER_ID"]
        for part in path_parts:
            parent = _find_subfolder(drive, parent, part)
            if not parent:
                return None
        return parent
    return _drive_call(_do)


@st.cache_data(ttl=300, show_spinner=False, max_entries=16)
def load_csv(path_parts):
    def _do():
        drive = drive_service()
        parent = _resolve_path_folder(tuple(path_parts[:-1]))
        if not parent:
            return pd.DataFrame()
        files = _list_folder(drive, parent)
        fid = files.get(path_parts[-1])
        if not fid:
            return pd.DataFrame()
        return pd.read_csv(io.BytesIO(_download_bytes(drive, fid)))
    return _drive_call(_do)


@st.cache_data(ttl=1800, show_spinner=False, max_entries=24)
def load_parquet(path_parts):
    def _do():
        drive = drive_service()
        parent = _resolve_path_folder(tuple(path_parts[:-1]))
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

@st.cache_data(ttl=1800, show_spinner=False, max_entries=3)
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


@st.cache_data(ttl=600, show_spinner=False, max_entries=20)
def load_md_content(file_id: str) -> str:
    """Download and return markdown content of a Drive file."""
    def _do():
        return _download_bytes(drive_service(), file_id).decode("utf-8", errors="replace")
    try:
        return _drive_call(_do)
    except Exception as e:
        return f"*Could not load file: {e}*"


@st.cache_data(ttl=300, show_spinner=False, max_entries=10)
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

# NOTE (2026-06-11 memory fix): these thin wrappers are deliberately UNcached.
# load_parquet() already caches the frame (ttl=1800); decorating the wrappers
# too stored a SECOND pickled copy of every big table for zero freshness gain
# (the inner cache governed anyway) — a Streamlit-Cloud OOM contributor.

def load_guidance_tracker() -> pd.DataFrame:
    return load_parquet(["company_repo", "_index", "guidance_tracker.parquet"])


def load_gf1_statements() -> pd.DataFrame:
    return load_parquet(["company_repo", "_index", "gf1_guidance_statements.parquet"])


def load_gf4_flags() -> pd.DataFrame:
    return load_parquet(["company_repo", "_index", "gf4_quality_flags.parquet"])


def load_results_summary() -> pd.DataFrame:
    return load_parquet(["company_repo", "_index", "results.parquet"])


def load_financials_3stmt() -> pd.DataFrame:
    """Phase 3 T2 — quarterly/annual/TTM 3-statement line items."""
    return load_parquet(["company_repo", "_index", "financials_3stmt.parquet"])


def load_scorecard() -> pd.DataFrame:
    """Phase 3 T4 — 8-factor company scorecard."""
    return load_parquet(["company_repo", "_index", "company_scorecard.parquet"])


def load_catalyst_index() -> pd.DataFrame:
    """Phase 3 T5 — catalyst notes index (why is it moving)."""
    return load_parquet(["company_repo", "_index", "catalyst_index.parquet"])


def _catalyst_text(row) -> str:
    """Best available note text for diffing: full note_text if present, else fall
    back to headline + what-to-track (older rows have no note_text)."""
    t = str(row.get("note_text", "") or "").strip()
    if t and t.lower() != "nan":
        return t
    parts = [str(row.get("headline", "") or "").strip()]
    wt = str(row.get("what_to_track", "") or "").strip()
    if wt and wt.lower() != "nan":
        parts.append("What to track: " + wt)
    return "\n".join(p for p in parts if p)


def _catalyst_diff_html(today_txt: str, prev_txt):
    """(html, changed) — today's note with new/changed sentences highlighted green
    vs the previous note. changed=False means the text is 100% identical."""
    import difflib
    import html as _h
    import re as _re
    esc = lambda s: _h.escape(s)
    split = lambda s: [x.strip() for x in _re.split(r"(?<=[.!?])\s+|\n+", s.strip()) if x.strip()]
    if not prev_txt:
        return esc(today_txt).replace("\n", "<br>"), True   # first note — no baseline
    a, b = split(prev_txt), split(today_txt)
    if a == b:
        return esc(today_txt).replace("\n", "<br>"), False  # 100% identical
    out = []
    for op, _i1, _i2, j1, j2 in difflib.SequenceMatcher(None, a, b).get_opcodes():
        if op == "delete":
            continue                                        # focus on today's text
        seg = " ".join(b[j1:j2])
        if op == "equal":
            out.append(esc(seg))
        else:                                               # insert / replace = new
            out.append(f'<mark style="background:#d4f8d4;padding:0 2px;'
                       f'border-radius:3px">{esc(seg)}</mark>')
    return " ".join(out), True


def _freshness_badge(as_of, label: str = "updated") -> str:
    """Colored age pill for a date: green <=2d, amber <=7d, red older, grey unknown."""
    import datetime as _dt
    s = str(as_of or "").strip()[:10]
    try:
        age = (_dt.date.today() - _dt.date.fromisoformat(s)).days
    except Exception:
        return ('<span style="background:#aaa;color:#fff;padding:1px 7px;'
                'border-radius:8px;font-size:11px">freshness unknown</span>')
    color = "#27ae60" if age <= 2 else ("#f39c12" if age <= 7 else "#e74c3c")
    when = "today" if age <= 0 else ("yesterday" if age == 1 else f"{age}d ago")
    return (f'<span style="background:{color};color:#fff;padding:1px 7px;'
            f'border-radius:8px;font-size:11px;font-weight:600">'
            f'{label} {s} · {when}</span>')


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


def _current_india_quarter() -> str:
    """Current India FY quarter tag, e.g. 'Q1FY27' (Phase 3 T3.1).
    Duplicated from extract_concall.py on purpose: importing that module here
    would pull Phase-2 Gemini deps (google.genai) that are NOT installed in the
    Streamlit Cloud env (root requirements.txt), crashing the dashboard.
    India FY: Apr-Jun=Q1, Jul-Sep=Q2, Oct-Dec=Q3, Jan-Mar=Q4."""
    m, y = datetime.now().month, datetime.now().year
    if m in (4, 5, 6):    return f"Q1FY{str(y + 1)[2:]}"
    if m in (7, 8, 9):    return f"Q2FY{str(y + 1)[2:]}"
    if m in (10, 11, 12): return f"Q3FY{str(y + 1)[2:]}"
    return f"Q4FY{str(y)[2:]}"  # Jan–Mar


def _norm_q(q) -> str:
    """Normalise a quarter tag for comparison: 'Q4 FY26' -> 'Q4FY26'."""
    return re.sub(r"\s+", "", str(q)).upper()


def _style_current_q(df: pd.DataFrame, cur_q: str):
    """Return a Styler highlighting rows whose `quarter` is the running quarter.
    Falls back to the plain DataFrame if there is no quarter column or styling
    is unavailable — never breaks the existing table render."""
    if df.empty or "quarter" not in df.columns:
        return df
    target = _norm_q(cur_q)

    def _hl(row):
        hit = _norm_q(row.get("quarter", "")) == target
        return ["background-color: #fff3cd" if hit else "" for _ in row]

    try:
        return df.style.apply(_hl, axis=1)
    except Exception:
        return df


def page_guidance():
    st.title("Management Guidance")
    st.caption(
        "Structured guidance extracted from concall transcripts. "
        "Updated as new concalls are processed by the Phase 2 pipeline."
    )

    cur_q = _current_india_quarter()
    st.caption(f"📆 Running quarter: **{cur_q}** — rows for this quarter are highlighted.")

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
                st.dataframe(_style_current_q(gf[show_cols], cur_q),
                             use_container_width=True, hide_index=True, height=400)

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
                st.dataframe(_style_current_q(detail[detail_cols], cur_q),
                             use_container_width=True, hide_index=True)

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


# ---------- T6: Market Trends (time-series views) ----------

def _fii_dii_pivot():
    """date x {FII, DII} net-flow frame from data/macro/FII_DII.csv (or empty)."""
    df = load_csv(["data", "macro", "FII_DII.csv"])
    if df.empty or "category" not in df.columns or "net" not in df.columns:
        return pd.DataFrame()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    cat = df["category"].astype(str).str.upper()
    df = df.assign(grp=pd.NA)
    df.loc[cat.str.contains("FII") | cat.str.contains("FPI"), "grp"] = "FII"
    df.loc[cat.str.contains("DII"), "grp"] = "DII"
    df = df.dropna(subset=["grp"])
    if df.empty:
        return pd.DataFrame()
    return df.groupby(["date", "grp"])["net"].sum().unstack("grp").sort_index()


_FLOW_COLORS = {"FII": "#2980b9", "DII": "#e67e22"}


def page_market_trends():
    st.title("📈 Market Trends")
    st.caption("Time-series of the nightly market-state snapshot "
               "(`data/market_state/history.csv`) + FII/DII flow trends.")

    hist = load_csv(["data", "market_state", "history.csv"])
    if hist.empty or "date" not in hist.columns:
        st.warning("`data/market_state/history.csv` not found yet — it grows one "
                   "row per market_state.py run.")
    else:
        hist["date"] = pd.to_datetime(hist["date"], errors="coerce")
        hist = hist.dropna(subset=["date"]).sort_values("date")
        window = st.radio("Window", ["30d", "90d", "180d", "All"], index=1,
                          horizontal=True)
        view = hist
        if window != "All":
            cutoff = hist["date"].max() - pd.Timedelta(days=int(window[:-1]))
            view = hist[hist["date"] >= cutoff]
        if view.empty:
            view = hist

        latest = view.iloc[-1]
        c1, c2, c3 = st.columns(3)
        hs = pd.to_numeric(latest.get("health_score"), errors="coerce")
        c1.metric("Health score", f"{hs:.0f}/100" if pd.notna(hs) else "—")
        c2.metric("Regime", str(latest.get("regime", "—")))
        c3.metric("History depth", f"{len(hist)} day(s)")

        fig = go.Figure()
        fig.add_hrect(y0=0, y1=40, fillcolor="#e74c3c", opacity=0.08, line_width=0)
        fig.add_hrect(y0=40, y1=60, fillcolor="#f39c12", opacity=0.08, line_width=0)
        fig.add_hrect(y0=60, y1=100, fillcolor="#27ae60", opacity=0.08, line_width=0)
        fig.add_trace(go.Scatter(
            x=view["date"], y=pd.to_numeric(view["health_score"], errors="coerce"),
            mode="lines+markers", name="Health", line=dict(color="#2c3e50", width=2)))
        fig.update_layout(height=320, yaxis=dict(range=[0, 100], title="Health score"),
                          margin=dict(l=10, r=10, t=10, b=10), showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Bands: <40 RISK_OFF · 40–60 NEUTRAL · >60 RISK_ON")

        comp_cols = [c for c in view.columns
                     if c.endswith("_score") and c != "health_score"]
        if comp_cols:
            st.subheader("Component scores")
            pick = st.multiselect("Components", comp_cols, default=comp_cols,
                                  format_func=lambda c: c[:-6].replace("_", " "))
            if pick:
                figc = go.Figure()
                for c in pick:
                    figc.add_trace(go.Scatter(
                        x=view["date"], y=pd.to_numeric(view[c], errors="coerce"),
                        mode="lines", name=c[:-6].replace("_", " ")))
                figc.update_layout(height=300, yaxis=dict(range=[0, 100]),
                                   margin=dict(l=10, r=10, t=10, b=10),
                                   legend=dict(orientation="h", yanchor="bottom",
                                               y=1.02))
                st.plotly_chart(figc, use_container_width=True)

        raw_cols = [c for c in view.columns
                    if c not in ("date", "regime", "health_score")
                    and not c.endswith("_score")
                    and pd.to_numeric(view[c], errors="coerce").notna().any()]
        if raw_cols:
            with st.expander("🔬 Raw snapshot metrics over time"):
                pick2 = st.multiselect("Metrics", raw_cols, default=raw_cols[:2])
                if pick2:
                    figr = go.Figure()
                    for c in pick2:
                        figr.add_trace(go.Scatter(
                            x=view["date"], y=pd.to_numeric(view[c], errors="coerce"),
                            mode="lines", name=c))
                    figr.update_layout(height=300,
                                       margin=dict(l=10, r=10, t=10, b=10),
                                       legend=dict(orientation="h",
                                                   yanchor="bottom", y=1.02))
                    st.plotly_chart(figr, use_container_width=True)

    st.markdown("---")
    pivot = _fii_dii_pivot()
    if pivot.empty:
        st.info("`data/macro/FII_DII.csv` not found — FII/DII trends unavailable.")
        return
    st.subheader("FII / DII flows — trend (₹ cr)")
    tab_m, tab_c = st.tabs(["Monthly net", "Cumulative (90d)"])
    with tab_m:
        monthly = pivot.resample("ME").sum().tail(12)
        figm = go.Figure()
        for grp in ("FII", "DII"):
            if grp in monthly.columns:
                figm.add_trace(go.Bar(x=monthly.index.strftime("%b %y"),
                                      y=monthly[grp], name=grp,
                                      marker_color=_FLOW_COLORS[grp]))
        figm.add_hline(y=0, line=dict(color="gray", dash="dot"))
        figm.update_layout(barmode="group", height=320,
                           margin=dict(l=10, r=10, t=10, b=10),
                           legend=dict(orientation="h", yanchor="bottom", y=1.02))
        st.plotly_chart(figm, use_container_width=True)
    with tab_c:
        recent = pivot[pivot.index >= pivot.index.max() - pd.Timedelta(days=90)]
        figcu = go.Figure()
        for grp in ("FII", "DII"):
            if grp in recent.columns:
                figcu.add_trace(go.Scatter(
                    x=recent.index, y=recent[grp].fillna(0).cumsum(),
                    mode="lines", name=f"{grp} cumulative",
                    line=dict(color=_FLOW_COLORS[grp], width=2)))
        figcu.add_hline(y=0, line=dict(color="gray", dash="dot"))
        figcu.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10),
                            legend=dict(orientation="h", yanchor="bottom", y=1.02))
        st.plotly_chart(figcu, use_container_width=True)
        st.caption("Running sum of daily net flows over the last 90 days.")


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
        def _fmt(val, fmt, fallback="—"):
            try:
                return format(float(val), fmt) if pd.notna(val) else fallback
            except (TypeError, ValueError):
                return fallback
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Close",              f"₹{_fmt(r.get('close'), ',.2f')}")
        c2.metric("3M return",          f"{_fmt(r.get('return_3m_pct'), '.1f')}%")
        c3.metric("RS rank (3M)",       f"{_fmt(r.get('rs_rank_3m'), '.0f')}")
        c4.metric("Dist from 52w high", f"{_fmt(r.get('dist_from_52w_high_pct'), '.1f')}%")
        c5.metric("ADR%(20)",           f"{_fmt(r.get('adr_pct_20'), '.2f')}%")

    # T4 — conviction scorecard badge
    _sc_badge = _scorecard_badge(symbol, load_scorecard())
    if _sc_badge:
        st.markdown(_sc_badge, unsafe_allow_html=True)

    # Last 6 quarters: Sales / Net Profit / EPS with latest-qtr YoY & QoQ + guidance
    _render_quarterly_table(symbol)

    # T5 — latest catalyst note headline
    _cat = load_catalyst_index()
    if not _cat.empty and "symbol" in _cat.columns:
        _crows = _cat[_cat["symbol"].astype(str).str.upper() == symbol]
        if not _crows.empty:
            _cr = _crows.sort_values("as_of").iloc[-1]
            st.caption(
                f"💡 **{_cr.get('headline', '')}** · "
                f"{_cr.get('catalyst_type', '')} · {_cr.get('as_of', '')}"
            )

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
        if not merged.empty:
            p0_close = float(merged["close"].iloc[0]) if pd.notna(merged["close"].iloc[0]) else 0.0
            p0_n500  = float(merged["n500"].iloc[0])  if pd.notna(merged["n500"].iloc[0])  else 0.0
            if p0_close and p0_n500:
                rs_series = (merged["close"] / p0_close) / (merged["n500"] / p0_n500) * 100

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
            _sc = sig.get('score')
            _sc_str = f"{_sc:.1f}" if pd.notna(_sc) else "—"
            with st.expander(f"{doc.get('title', strat)} — **{sig.get('zone_type')}** "
                             f"(score {_sc_str})"):
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

    # ── Quarterly Financials (last 12Q) — Phase 3 T2.3 ───────────────────
    st.markdown("---")
    st.subheader("Quarterly Financials (last 12Q)")
    fin_sd = load_financials_3stmt()
    if fin_sd.empty or "symbol" not in fin_sd.columns:
        st.info("Quarterly financials not available yet "
                "(run `scripts/backfill_results_3stmt.py`).")
    else:
        q = fin_sd[(fin_sd["symbol"] == symbol)
                   & (fin_sd["statement"] == "income")
                   & (fin_sd["period_type"] == "quarterly")]
        if q.empty:
            st.info(f"No quarterly financials for {symbol} yet.")
        else:
            def _q_series(li):
                s = q[q["line_item"] == li]
                return s["period"].tolist(), s["value"].tolist()
            periods, sales_v = _q_series("Sales")
            _, pat_v = _q_series("Net Profit")
            _, eps_v = _q_series("EPS")
            figq = go.Figure()
            figq.add_bar(x=periods, y=sales_v, name="Sales (₹Cr)", marker_color="#2980b9")
            figq.add_bar(x=periods, y=pat_v, name="PAT (₹Cr)", marker_color="#27ae60")
            if any(pd.notna(x) for x in eps_v):
                figq.add_trace(go.Scatter(
                    x=periods, y=eps_v, name="EPS (₹)", yaxis="y2",
                    mode="lines+markers", line=dict(color="#e67e22")))
                figq.update_layout(yaxis2=dict(overlaying="y", side="right",
                                               showgrid=False, title="EPS"))
            figq.update_layout(barmode="group", height=300,
                               margin=dict(l=10, r=10, t=10, b=10),
                               legend=dict(orientation="h", y=1.12))
            st.plotly_chart(figq, use_container_width=True)
            ttm = fin_sd[(fin_sd["symbol"] == symbol)
                         & (fin_sd["period_type"] == "ttm")]
            if not ttm.empty:
                bits = []
                for li in ["Sales", "Operating Profit", "Net Profit", "EPS"]:
                    r = ttm[ttm["line_item"] == li]
                    if not r.empty and pd.notna(r["value"].iloc[0]):
                        bits.append(f"{li} {float(r['value'].iloc[0]):,.0f}")
                if bits:
                    st.caption("TTM — " + "  ·  ".join(bits))

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
                                     max_value=10, value=2, step=1)
    with c4:
        tf = st.selectbox("Timeframe", list(TIMEFRAME_DAYS.keys()), index=2)
    with c5:
        view = st.radio(
            "View",
            ["NSE · 1–60", "NSE · 61–120", "NSE · 121–180", "NSE · 181–240",
             "NSE · 241–300", "NSE · 301–360",
             "BSE-only · 1–60", "BSE-only · 61–120", "Detailed"],
            help="NSE views = stocks listed on NSE (may also trade on BSE). "
                 "BSE-only = stocks not on NSE. Each view renders ONE slice of ≤60 "
                 "charts so Cloud memory stays safe (real tabs would render all at "
                 "once and crash). Detailed = paginated candlesticks over the full set.")
    view_mode = "Detailed" if view == "Detailed" else "Quick Scan"
    with c6:
        if view_mode == "Quick Scan":
            normalize = st.toggle("Normalise (% from start)",
                                  value=True,
                                  help="Index all charts to 0% at window start so run-ups "
                                       "are directly comparable across stocks.")
        else:
            normalize = False
            per_page = int(st.number_input("Charts/page", min_value=2,
                                           max_value=20, value=6, step=2))

    min_turnover_cr = st.number_input(
        "Min avg ₹ turnover (cr/day, 20d) — 0 = off", min_value=0.0,
        max_value=500.0, value=1.0, step=1.0,
        help="Liquidity floor: avg daily shares×price over 20 days, in ₹ crores. "
             "Drops illiquid names (~45% of signals are sub-₹1cr/day). "
             "Stocks with no volume data are dropped when the floor is on. 0 disables.")

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

    # ── Liquidity floor (avg ₹ turnover, 20d) ─────────────────────────────────
    # Prefers the avg_turnover_20d_cr feature (added in compute_features.py); until
    # Phase 1 re-runs and writes it, falls back to vol_20d_avg × close on the fly.
    if min_turnover_cr > 0:
        feats = load_parquet(["features", "latest.parquet"])
        if not feats.empty and "symbol" in feats.columns:
            if "avg_turnover_20d_cr" in feats.columns:
                turn = pd.to_numeric(feats["avg_turnover_20d_cr"], errors="coerce")
            elif {"vol_20d_avg", "close"} <= set(feats.columns):
                turn = (pd.to_numeric(feats["vol_20d_avg"], errors="coerce")
                        * pd.to_numeric(feats["close"], errors="coerce")) / 1e7
            else:
                turn = pd.Series(float("nan"), index=feats.index)
            tmap = dict(zip(feats["symbol"].astype(str), turn))
            before = len(conv)
            keep = conv["symbol"].astype(str).map(tmap).fillna(-1.0) >= min_turnover_cr
            conv = conv[keep]
            st.caption(f"Liquidity floor ≥ ₹{min_turnover_cr:.0f}cr/day turnover — "
                       f"{len(conv)} of {before} stocks pass "
                       f"(no-data names dropped).")
            if conv.empty:
                st.info("No stock passes the liquidity floor. Lower it.")
                return

    # ── Exchange split + bounded slice (memory-safe segments) ─────────────────
    # Each Quick-Scan view renders ONE slice so the OHLCV bulk-load and the Plotly
    # render stay bounded (hundreds of figures blow Cloud's ~1GB → native crash).
    # Slice membership uses a CHEAP pre-rank (n_strategies, then score) so we don't
    # have to load every symbol's OHLCV just to rank — n_strategies is the default
    # sort's primary key, so this is faithful; the chosen sort then reorders the
    # slice for display below.
    # view -> (which exchange set, slice start, slice end). Each slice is ≤60 —
    # 100-chart slices were still OOM-segfaulting Cloud, so dropped to 60.
    SEG = {
        "NSE · 1–60":        ("nse",   0,  60),
        "NSE · 61–120":      ("nse",  60, 120),
        "NSE · 121–180":     ("nse", 120, 180),
        "NSE · 181–240":     ("nse", 180, 240),
        "NSE · 241–300":     ("nse", 240, 300),
        "NSE · 301–360":     ("nse", 300, 360),
        "BSE-only · 1–60":   ("bse",   0,  60),
        "BSE-only · 61–120": ("bse",  60, 120),
    }
    if view_mode == "Quick Scan":
        uni = load_csv(["universe", "master_list.csv"])
        exch = {}
        if not uni.empty and {"symbol", "exchange"} <= set(uni.columns):
            exch = dict(zip(uni["symbol"].astype(str), uni["exchange"].astype(str)))
        conv["_exch"] = conv["symbol"].astype(str).map(exch).fillna("NSE")
        conv = conv.sort_values(["n_strategies", "best_score"],
                                ascending=[False, False]).reset_index(drop=True)
        nse = conv[conv["_exch"] != "BSE"].reset_index(drop=True)   # NSE-listed
        bse = conv[conv["_exch"] == "BSE"].reset_index(drop=True)   # BSE-only
        which, lo, hi = SEG[view]
        src = nse if which == "nse" else bse
        conv = src.iloc[lo:hi].reset_index(drop=True)
        label = "NSE-listed" if which == "nse" else "BSE-only"
        st.caption(f"{label} · rank {lo + 1}–{lo + len(conv)} of {len(src)}.")
        if conv.empty:
            st.info("No stocks in this segment for the current filters.")
            return

    # ── Sort ──────────────────────────────────────────────────────────────────
    SORT_OPTIONS = {
        "Strategies ↓  →  1M ret ↓  →  3M ret ↓  (default)": "default",
        "1M Return ↓": "ret_1m",
        "3M Return ↓": "ret_3m",
        "6M Return ↓": "ret_6m",
        "Score ↓": "score",
        "Symbol A→Z": "alpha",
    }
    sort_choice = st.selectbox("Sort by", list(SORT_OPTIONS.keys()),
                               index=0, key="qs_sort")
    sort_key = SORT_OPTIONS[sort_choice]

    # We need return data for sorting — load OHLCV once so we can compute it.
    # This is the same bulk call used for rendering, so it hits cache if already loaded.
    sym_list_unsorted = tuple(conv["symbol"].tolist())

    if sort_key in ("default", "ret_1m", "ret_3m", "ret_6m"):
        # Pre-load returns for sorting (uses the same cache as rendering)
        with st.spinner("Computing returns for sort…"):
            ohlcv_map_sort = load_ohlcv_bulk(sym_list_unsorted)

        def _period_ret(sym, days):
            df_s = ohlcv_map_sort.get(sym, pd.DataFrame())
            if df_s.empty or len(df_s) < days + 1:
                return -999.0
            df_s = df_s.sort_values("date")
            p0 = float(df_s["close"].iloc[-(days + 1)])
            p1 = float(df_s["close"].iloc[-1])
            return (p1 / p0 - 1) * 100 if p0 else -999.0

        conv["ret_1m"] = conv["symbol"].apply(lambda s: _period_ret(s, 21))
        conv["ret_3m"] = conv["symbol"].apply(lambda s: _period_ret(s, 63))
        conv["ret_6m"] = conv["symbol"].apply(lambda s: _period_ret(s, 126))

        if sort_key == "default":
            conv = conv.sort_values(
                ["n_strategies", "ret_1m", "ret_3m"],
                ascending=[False, False, False],
            ).reset_index(drop=True)
        elif sort_key == "ret_1m":
            conv = conv.sort_values("ret_1m", ascending=False).reset_index(drop=True)
        elif sort_key == "ret_3m":
            conv = conv.sort_values("ret_3m", ascending=False).reset_index(drop=True)
        elif sort_key == "ret_6m":
            conv = conv.sort_values("ret_6m", ascending=False).reset_index(drop=True)
    elif sort_key == "score":
        conv = conv.sort_values("best_score", ascending=False).reset_index(drop=True)
    else:  # alpha
        conv = conv.sort_values("symbol").reset_index(drop=True)

    total    = len(conv)
    sym_list = tuple(conv["symbol"].tolist())
    st.caption(f"{total} stock(s) matching filters")

    # ─────────────────────────────────────────────────────────────────────────
    # QUICK SCAN MODE — bulk load, 2-column grid
    # ─────────────────────────────────────────────────────────────────────────
    if view_mode == "Quick Scan":
        st.info(
            f"Loading {total} charts. First load downloads from Drive (~1-3 min "
            f"for 300+ stocks). Subsequent visits use the 30-min cache — instant.",
            icon="ℹ️",
        )
        # ohlcv_map_sort populated above when sort needed returns; load now if not
        _sort_loaded = sort_key in ("default", "ret_1m", "ret_3m", "ret_6m")
        if not _sort_loaded:
            prog = st.progress(0, text="Loading OHLCV from Drive (one batch call)…")
            ohlcv_map_sort = load_ohlcv_bulk(sym_list)
            prog.progress(100, text="Done — rendering charts…")
        else:
            st.caption("OHLCV already loaded for sort — rendering now.")

        tf_days = TIMEFRAME_DAYS[tf]

        # ── Pre-load fundamentals (already-cached parquets, no Drive calls) ──
        results_df   = load_results_summary()    # has yoy_pct, qoq_pct per metric row
        guidance_df  = load_guidance_tracker()   # structured guidance rows
        scorecard_df = load_scorecard()          # T4 conviction scores (empty until built)
        catalyst_df  = load_catalyst_index()     # T5 latest note + what-to-track

        # Pre-group each table by symbol ONCE. The render loop below visits every
        # matching stock; without this, each per-symbol lookup was a full-table
        # scan → O(stocks × rows). Dict lookups make it O(rows) total. Output is
        # identical — same rows, same order within each symbol's group.
        _EMPTY = pd.DataFrame()

        def _group_exact(df):
            if df is None or df.empty or "symbol" not in df.columns:
                return {}
            return {k: g for k, g in df.groupby("symbol", sort=False)}

        def _group_upper(df):
            if df is None or df.empty or "symbol" not in df.columns:
                return {}
            keys = df["symbol"].astype(str).str.upper()
            return {k: g for k, g in df.groupby(keys, sort=False)}

        sel_by_sym       = _group_exact(sel)
        results_by_sym   = _group_exact(results_df)
        guidance_by_sym  = _group_exact(guidance_df)
        catalyst_by_sym  = _group_upper(catalyst_df)
        scorecard_by_sym = _group_upper(scorecard_df)

        def _catalyst_line(sym: str) -> str:
            """💡 headline · 👁 what-to-track for the latest catalyst note."""
            try:
                rows = catalyst_by_sym.get(sym.upper())
                if rows is None or rows.empty:
                    return ""
                r = rows.sort_values("as_of").iloc[-1]
                head = str(r.get("headline", "")).strip()
                if not head:
                    return ""
                out = (f'💡 <b>{r.get("as_of", "")}</b> '
                       f'[{r.get("catalyst_type", "?")}] {head[:150]}')
                track = str(r.get("what_to_track", "") or "").strip()
                if track and track.lower() != "nan":
                    out += f'<br>👁 <i>track: {track[:220]}</i>'
                return out
            except Exception:
                return ""

        def _fund_line_inner(sym: str) -> str:
            parts = []

            # ── Results: latest Sales + Net Profit row for this symbol ────────
            sym_res = results_by_sym.get(sym)
            if sym_res is not None and "metric" in sym_res.columns:
                for metric_kw, label in [
                    ("sales|revenue|income from operations", "Sales"),
                    ("net profit|pat|profit after tax",      "PAT"),
                ]:
                    rows = sym_res[
                        sym_res["metric"].astype(str).str.lower()
                        .str.contains(metric_kw, na=False, regex=True)
                    ]
                    if rows.empty:
                        continue
                    r = rows.iloc[0]
                    qtr   = str(r.get("latest_q", "")).strip()
                    yoy   = r.get("yoy_pct")
                    qoq   = r.get("qoq_pct")
                    yoy_s = (f'<span style="color:{"#27ae60" if float(yoy)>=0 else "#e74c3c"}">'
                             f'{float(yoy):+.0f}% YoY</span>' if pd.notna(yoy) else "")
                    qoq_s = (f'<span style="color:{"#27ae60" if float(qoq)>=0 else "#e74c3c"}">'
                             f'{float(qoq):+.0f}% QoQ</span>' if pd.notna(qoq) else "")
                    sub = " / ".join(x for x in [yoy_s, qoq_s] if x)
                    if sub:
                        qtr_tag = f"<b>{qtr}</b> " if qtr else ""
                        parts.append(f"{qtr_tag}{label}: {sub}")

            # ── Guidance: active revenue/PAT/EPS rows ─────────────────────────
            sym_g = guidance_by_sym.get(sym)
            if (sym_g is not None
                    and "metric" in sym_g.columns
                    and "horizon_fy" in sym_g.columns):
                sym_g = sym_g[sym_g["horizon_fy"].apply(_guidance_is_active)]
                sym_g = sym_g[
                    sym_g["metric"].astype(str).str.lower()
                    .str.contains("revenue|sales|pat|profit|eps|margin", na=False, regex=True)
                ]
                g_bits = []
                for _, gr in sym_g.head(2).iterrows():
                    metric = str(gr.get("metric", "")).title()
                    val    = gr.get("value", "")
                    unit   = gr.get("unit", "")
                    fy     = gr.get("horizon_fy", "")
                    if val:
                        g_bits.append(f"{metric} {val}{' '+unit if unit else ''} {fy}".strip())
                if g_bits:
                    parts.append("📋 " + " | ".join(g_bits))

            return "  ·  ".join(parts) if parts else ""

        def _fund_line(sym: str) -> str:
            try:
                return _fund_line_inner(sym)
            except Exception:
                return ""

        # ── Strategy chip builder ─────────────────────────────────────────────
        def _strat_chips(sym_sigs_df) -> str:
            seen = set()
            chips = []
            for _, sg in sym_sigs_df.iterrows():
                strat = sg.get("strategy", sg.get("strategy_group", ""))
                zt    = sg.get("zone_type", "")
                key   = (strat, zt)
                if key in seen:
                    continue
                seen.add(key)
                color = ZONE_COLORS.get(zt, "#555")
                chips.append(
                    f'<span style="background:{color};color:white;'
                    f'padding:1px 7px;border-radius:8px;font-size:11px;'
                    f'margin-right:3px;">{strat} · {zt}</span>'
                )
            return " ".join(chips)

        # ── 6-rule grade strip + 6-quarter table (chart+table+grades+summary glance) ──
        import sys as _sysg
        _spg = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scripts")
        if _spg not in _sysg.path:
            _sysg.path.insert(0, _spg)
        from gradation import TIER_COLOR as _TC
        grades_by_sym = _group_upper(load_screener_grades())
        # (tier_col, label, value_col, value_format)
        _GTILES = [("yoy_tier", "YOY", "yoy", "{:.0f}%"),
                   ("qoq_tier", "QOQ", "qoq", "{:.0f}%"),
                   ("guidance_tier", "Guid", "guidance", "{:.0f}%"),
                   ("val_tier", "Val", "val_value", "PE {:.0f}"),
                   ("cfo_tier", "CFO", "cfo_ratio", "{:.1f}x"),
                   ("roe_tier", "ROE", "roe", "{:.0f}%")]

        def _grades_strip(sym: str) -> str:
            g = grades_by_sym.get(sym.upper(), _EMPTY)
            if g is None or g.empty:
                return ""
            r = g.iloc[-1]
            tiles = []
            for tcol, lbl, vcol, vfmt in _GTILES:
                tier = str(r.get(tcol, "na"))
                v = r.get(vcol)
                try:
                    vs = vfmt.format(float(v)) if v is not None and pd.notna(v) else "—"
                except (TypeError, ValueError):
                    vs = "—"
                tiles.append(
                    f'<span style="background:{_TC.get(tier, "#eee")};color:#111;'
                    f'padding:1px 6px;border-radius:6px;font-size:11px;margin-right:3px;'
                    f'display:inline-block">{lbl} {vs}</span>')
            return "".join(tiles)

        def _quarterly_html(sym: str) -> str:
            sdf = load_parquet(["fundamentals", "statements", f"{sym}.parquet"])
            if sdf.empty or "statement" not in sdf.columns:
                return ""
            q = sdf[sdf["statement"] == "quarterly_pl"]
            if q.empty:
                return ""
            def ser(item):
                sub = q[q["line_item"] == item]
                return list(zip(sub["period"].astype(str),
                                pd.to_numeric(sub["value"], errors="coerce")))
            rowdefs = [("Sales", "Sales"), ("Profit", "Net Profit"), ("EPS", "EPS in Rs")]
            periods = None
            for _, it in rowdefs:
                s = ser(it)
                if s:
                    periods = [p for p, _ in s][-6:]
                    break
            if not periods:
                return ""
            th = ("<tr><td style='font-size:10px'></td>"
                  + "".join(f"<td style='font-size:10px;text-align:right;color:#888'>{p}</td>"
                            for p in periods) + "</tr>")
            body = ""
            for lbl, it in rowdefs:
                d = dict(ser(it))
                cells = "".join(
                    "<td style='font-size:10px;text-align:right'>"
                    + ("—" if d.get(p) is None or pd.isna(d.get(p))
                       else format(d.get(p), ',.0f')) + "</td>"
                    for p in periods)
                body += f"<tr><td style='font-size:10px'><b>{lbl}</b></td>{cells}</tr>"
            return (f"<table style='border-collapse:collapse;width:100%;"
                    f"margin:2px 0 4px 0'>{th}{body}</table>")

        # mcap (market_cap.csv) · announcement LLM summary · GF1 guidance · >30% blob
        _mc_df = load_csv(["universe", "market_cap.csv"])
        mcap_by_sym = {}
        if not _mc_df.empty and "symbol" in _mc_df.columns:
            for _, r in _mc_df.iterrows():
                mcap_by_sym[str(r["symbol"]).upper()] = (
                    pd.to_numeric(r.get("market_cap_cr"), errors="coerce"),
                    str(r.get("mcap_segment", "") or ""))
        ann_by_sym = _group_upper(load_parquet(
            ["company_repo", "_index", "announcement_ledger.parquet"]))
        gf1_by_sym = _group_upper(load_parquet(
            ["company_repo", "_index", "gf1_guidance_statements.parquet"]))

        def _mcap_str(sym):
            mc, seg = mcap_by_sym.get(sym.upper(), (None, ""))
            if mc is None or pd.isna(mc):
                return ""
            txt = f"₹{mc:,.0f} Cr" + (f" · {seg}" if seg else "")
            return (f'<span style="background:#eceff1;color:#333;padding:1px 7px;'
                    f'border-radius:6px;font-size:11px;margin-right:4px">{txt}</span>')

        def _growth_blob(sym):
            g = grades_by_sym.get(sym.upper(), _EMPTY)
            if g is None or g.empty:
                return ""
            r = g.iloc[-1]
            hot = []
            for key, lbl in (("yoy", "YoY"), ("qoq", "QoQ"), ("guidance", "Guidance")):
                v = pd.to_numeric(r.get(key), errors="coerce")
                if pd.notna(v) and v > 30:
                    hot.append(f"{lbl} +{v:.0f}%")
            if not hot:
                return ""
            return (f'<div style="background:#1a7a3a;color:#fff;padding:3px 9px;'
                    f'border-radius:6px;font-size:12px;font-weight:600;margin:3px 0">'
                    f'🚀 {" · ".join(hot)}</div>')

        def _gf1_blob(sym):
            g = gf1_by_sym.get(sym.upper(), _EMPTY)
            if g is None or g.empty or "exact_statement" not in g.columns:
                return ""
            g2 = g.sort_values("processed_at") if "processed_at" in g.columns else g
            stmt = str(g2.iloc[-1].get("exact_statement", "") or "").strip()
            if not stmt or stmt.lower() == "nan":
                return ""
            return (f'<div style="background:#eef6ff;border-left:3px solid #1565c0;'
                    f'padding:3px 8px;font-size:11px;color:#333;margin:2px 0">'
                    f'📋 <b>Guidance:</b> {stmt[:240]}</div>')

        def _llm_summary(sym):
            a = ann_by_sym.get(sym.upper(), _EMPTY)
            if a is not None and not a.empty and "summary" in a.columns:
                a2 = a.sort_values("ann_date") if "ann_date" in a.columns else a
                row = a2.iloc[-1]
                s = str(row.get("summary", "") or "").strip()
                if s and s.lower() != "nan":
                    return (f'<div style="background:#fffde7;border-left:3px solid #f9a825;'
                            f'padding:4px 8px;font-size:11px;color:#333;margin:2px 0">'
                            f'🧠 <b>{str(row.get("ann_date",""))[:10]} {row.get("category","")}:</b> '
                            f'{s[:400]}</div>')
            return _catalyst_line(sym)   # fall back to catalyst note

        for i, (_, crow) in enumerate(conv.iterrows()):
            sym      = crow["symbol"]
            sym_sigs = sel_by_sym.get(sym, _EMPTY)
            ohlcv    = ohlcv_map_sort.get(sym, pd.DataFrame())

            with st.container():
                # Header — mcap + 6-rule grade strip (green/amber) + conviction badge
                line1_html = (_mcap_str(sym) + _grades_strip(sym) + " "
                              + _scorecard_badge(sym, scorecard_by_sym.get(sym.upper(), _EMPTY)))
                if line1_html.strip():
                    st.markdown(line1_html, unsafe_allow_html=True)

                # last-6-quarter table (Sales / Profit / EPS)
                qhtml = _quarterly_html(sym)
                if qhtml:
                    st.markdown(qhtml, unsafe_allow_html=True)

                # fundamentals growth (results YoY/QoQ + guidance)
                fund = _fund_line(sym)
                if fund:
                    st.markdown(
                        f'<div style="font-size:11px;color:#555;'
                        f'margin:1px 0 3px 0;line-height:1.4">{fund}</div>',
                        unsafe_allow_html=True,
                    )

                # Chart (bigger)
                if ohlcv.empty:
                    st.caption(f"{sym} — no OHLCV data")
                else:
                    fig = build_quick_chart(sym, ohlcv, sym_sigs, tf_days,
                                            normalize=normalize)
                    st.plotly_chart(fig, use_container_width=True,
                                    key=f"qs_{sym}", config={"displayModeBar": False})

                # ── Below the chart: >30% growth highlight + GF1 guidance + LLM summary ──
                for blob in (_growth_blob(sym), _gf1_blob(sym), _llm_summary(sym)):
                    if blob:
                        st.markdown(blob, unsafe_allow_html=True)
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

    # Pre-group this page's signal rows by symbol once (same pattern as Quick Scan).
    page_sel = sel[sel["symbol"].isin(page_syms)]
    sel_by_sym = ({k: g for k, g in page_sel.groupby("symbol", sort=False)}
                  if not page_sel.empty else {})

    for _, crow in conv.iloc[page * per_page:(page + 1) * per_page].iterrows():
        sym      = crow["symbol"]
        sym_sigs = sel_by_sym.get(sym, pd.DataFrame())
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
    # Screener-only columns are optional — a broker export may omit them. Default
    # so downstream (rating stars, display, sort) never KeyErrors.
    for _opt in ("screener_rating", "screener_name", "screener_last_price"):
        if _opt not in df.columns:
            df[_opt] = pd.NA
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


def _app_drive_download(drive, path_parts: list[str]) -> bytes | None:
    """Download a file from Drive by path_parts list using app's own Drive helpers."""
    try:
        parent = os.environ["GDRIVE_FOLDER_ID"]
        for part in path_parts[:-1]:
            parent = _find_subfolder(drive, parent, part)
            if not parent:
                return None
        files = _list_folder(drive, parent)
        fid = files.get(path_parts[-1])
        if not fid:
            return None
        return _download_bytes(drive, fid)
    except Exception:
        return None


def _app_drive_upload(drive, path_parts: list[str], content: bytes, mime: str) -> None:
    """Upload/overwrite a file on Drive by path_parts list."""
    from googleapiclient.http import MediaIoBaseUpload
    parent = os.environ["GDRIVE_FOLDER_ID"]
    for part in path_parts[:-1]:
        sub = _find_subfolder(drive, parent, part)
        if not sub:
            sub = drive.files().create(body={
                "name": part, "mimeType": "application/vnd.google-apps.folder",
                "parents": [parent]}, fields="id").execute()["id"]
        parent = sub
    files = _list_folder(drive, parent)
    media = MediaIoBaseUpload(io.BytesIO(content), mimetype=mime, resumable=False)
    if path_parts[-1] in files:
        drive.files().update(fileId=files[path_parts[-1]], media_body=media).execute()
    else:
        drive.files().create(
            body={"name": path_parts[-1], "parents": [parent]},
            media_body=media, fields="id").execute()


@st.cache_data(ttl=300)
def load_deep_dive_index():
    drive = drive_service()
    b = _app_drive_download(drive, ["company_repo", "_index", "deep_dive_index.parquet"])
    return pd.read_parquet(io.BytesIO(b)) if b else pd.DataFrame()


@st.cache_data(ttl=300)
def load_deep_dive_queue():
    drive = drive_service()
    b = _app_drive_download(drive, ["company_repo", "_index", "deep_dive_queue.parquet"])
    return pd.read_parquet(io.BytesIO(b)) if b else pd.DataFrame()


def page_deep_dive():
    st.title("🔬 Deep Dive")
    st.caption(
        "Forensic equity analysis: 4-phase deep dive covering financials, "
        "governance, risk scorecard, and PM one-pager verdict."
    )

    tab_queue, tab_reports = st.tabs(["📋 Queue", "📄 Reports"])

    # ── Tab 1: Queue ─────────────────────────────────────────────────────────
    with tab_queue:
        st.subheader("Request a Deep Dive")
        col1, col2 = st.columns([3, 1])
        with col1:
            company_input = st.text_input(
                "Company name, NSE/BSE symbol, or ISIN — comma-separate for multiple",
                placeholder="e.g. TCS, VENUSREM, INE467B01029  or just  VENUSREM")
        with col2:
            st.write("")
            st.write("")
            add_btn = st.button("Add to Queue", type="primary")

        if add_btn and company_input.strip():
            try:
                drive = drive_service()
                univ_b = _app_drive_download(drive, ["universe", "master_list.csv"])
                univ   = pd.read_csv(io.BytesIO(univ_b)) if univ_b else pd.DataFrame()
                import sys as _sys, os as _os
                _sdir = _os.path.join(_os.path.dirname(__file__), "scripts")
                if _sdir not in _sys.path: _sys.path.insert(0, _sdir)
                from company_deep_report import resolve_isin

                tokens  = [t.strip() for t in company_input.split(",") if t.strip()]
                queued, failed = [], []
                for token in tokens:
                    isin, symbol, name, _ = resolve_isin(token, univ)
                    if isin == token and symbol == token:
                        failed.append(token)
                    else:
                        queued.append(dict(token=isin, status="pending",
                                           added_at=datetime.now().isoformat(),
                                           _label=f"{name} ({symbol})"))

                if queued:
                    q_b = _app_drive_download(drive, ["company_repo", "_index", "deep_dive_queue.parquet"])
                    q   = pd.read_parquet(io.BytesIO(q_b)) if q_b else pd.DataFrame()
                    new_rows = pd.DataFrame([{k: v for k, v in r.items() if not k.startswith("_")}
                                             for r in queued])
                    q = pd.concat([q, new_rows], ignore_index=True)
                    buf = io.BytesIO(); q.to_parquet(buf, index=False)
                    _app_drive_upload(drive, ["company_repo", "_index", "deep_dive_queue.parquet"],
                                      buf.getvalue(), "application/octet-stream")
                    labels = ", ".join(r["_label"] for r in queued)
                    st.success(f"Queued {len(queued)} company/companies: **{labels}**. "
                               "CI runs at 08:00 IST — you'll receive an email when done.")
                    st.cache_data.clear()
                if failed:
                    st.warning(f"Could not resolve: {', '.join(failed)} — check spelling or use ISIN.")
            except Exception as e:
                st.error(f"Failed to queue: {e}")

        st.markdown("---")
        st.subheader("Current Queue")
        queue_df = load_deep_dive_queue()
        if queue_df.empty:
            st.caption("Queue is empty.")
        else:
            show_cols = [c for c in ["token", "status", "added_at", "done_at", "error"]
                         if c in queue_df.columns]
            st.dataframe(queue_df[show_cols].sort_values(
                "added_at", ascending=False).head(50),
                use_container_width=True)

    # ── Tab 2: Reports ────────────────────────────────────────────────────────
    with tab_reports:
        idx = load_deep_dive_index()
        if idx.empty:
            st.info("No deep-dive reports yet. Use the Queue tab to request one.")
            return

        # build display label
        def _label(row):
            name = str(row.get("name", row.get("isin", "?")))
            sym  = str(row.get("symbol", ""))
            upd  = str(row.get("last_update", ""))[:10]
            return f"{name} ({sym}) · {upd}"

        idx = idx.sort_values("last_update", ascending=False)
        labels = [_label(r) for _, r in idx.iterrows()]
        choice = st.selectbox("Select company report", labels)

        if choice:
            row = idx.iloc[labels.index(choice)]
            isin        = str(row.get("isin", ""))
            symbol      = str(row.get("symbol", ""))
            name        = str(row.get("name", isin))
            report_path = str(row.get("report_path", ""))
            last_update = str(row.get("last_update", ""))[:16]

            # coverage badge
            try:
                import json as _json
                cov = _json.loads(row.get("coverage", "{}"))
                ar  = cov.get("ar_years", [])
                nc  = cov.get("n_concall", 0)
                nr  = cov.get("n_research", 0)
                st.caption(
                    f"📅 Generated {last_update} · "
                    f"AR years: {ar or 'none'} · "
                    f"Concalls: ~{nc} · Research docs: ~{nr}")
            except Exception:
                st.caption(f"Generated {last_update}")

            if report_path:
                try:
                    drive = drive_service()
                    raw   = _app_drive_download(drive, report_path.strip("/").split("/"))
                    if raw:
                        md_text = raw.decode("utf-8")
                        st.markdown(md_text)

                        # download buttons
                        st.markdown("---")
                        dl1, dl2 = st.columns(2)
                        with dl1:
                            st.download_button(
                                "⬇ Download .md",
                                data=raw,
                                file_name=f"deepdive_{symbol}_{last_update[:10]}.md",
                                mime="text/markdown")
                        with dl2:
                            # docx on demand
                            try:
                                import sys, os as _os
                                _sdir = _os.path.join(_os.path.dirname(__file__), "scripts")
                                if _sdir not in sys.path: sys.path.insert(0, _sdir)
                                from format_deepdive_docx import md_to_docx
                                docx_bytes = md_to_docx(md_text, name, symbol, isin)
                                st.download_button(
                                    "⬇ Download .docx",
                                    data=docx_bytes,
                                    file_name=f"deepdive_{symbol}_{last_update[:10]}.docx",
                                    mime="application/vnd.openxmlformats-officedocument"
                                         ".wordprocessingml.document")
                            except Exception:
                                pass
                    else:
                        st.warning("Report file not found on Drive.")
                except Exception as e:
                    st.error(f"Could not load report: {e}")
            else:
                st.warning("No report path recorded in index.")


# ═══════════════════════════════════════════════════════════════════════════
# T4 — Scorecard helpers + page
# ═══════════════════════════════════════════════════════════════════════════

def _scorecard_badge(symbol: str, sc_df: pd.DataFrame) -> str:
    """Return a coloured composite-score HTML chip, or '' if data is absent."""
    if sc_df is None or sc_df.empty or "symbol" not in sc_df.columns:
        return ""
    row = sc_df[sc_df["symbol"].astype(str).str.upper() == symbol.upper()]
    if row.empty:
        return ""
    v = row.iloc[0].get("composite_score")
    if pd.isna(v):
        return ""
    score = float(v)
    color = (
        "#27ae60" if score >= 75 else
        "#f39c12" if score >= 55 else
        "#e67e22" if score >= 35 else
        "#e74c3c"
    )
    return (
        f'<span title="Conviction scorecard" style="background:{color};color:white;'
        f'padding:2px 8px;border-radius:8px;font-size:11px;'
        f'margin-left:4px;font-weight:600;">⭐ {score:.0f}</span>'
    )


_SCORE_FACTORS: list[tuple[str, str, str]] = [
    ("score_technical",     "Technical",     "📈"),
    ("score_fundamental",   "Fundamental",   "📊"),
    ("score_fin_health",    "Fin. Health",   "🏦"),
    ("score_mgmt_cred",     "Mgmt Cred.",    "🎯"),
    ("score_valuation",     "Valuation",     "💰"),
    ("score_guidance",      "Guidance",      "📋"),
    ("score_fraud_risk",    "Fraud Safety",  "🛡️"),
    ("score_investigative", "Investigative", "🔍"),
]


def _score_band_color(v) -> str:
    """Hex color for a 0–100 score, or grey for missing."""
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return "#aaaaaa"
    try:
        s = float(v)
    except (TypeError, ValueError):
        return "#aaaaaa"
    if s >= 75: return "#27ae60"
    if s >= 55: return "#f39c12"
    if s >= 35: return "#e67e22"
    return "#e74c3c"


_BAND_COLORS = {"RED": "#e74c3c", "ALERT": "#e67e22", "WATCH": "#f39c12"}
_TREND_ARROW = {"UP": "▲", "DOWN": "▼", "FLAT": "—", "NEW": "★"}


def _ask_assemble_ctx(isin, sym):
    """Per-company context from the app's cached loaders (no Gemini)."""
    parts = []
    page = find_company_page(isin) or find_company_page(sym) or ""
    if page:
        parts.append("## COMPANY PAGE\n" + page[-50_000:])
    for label, df, col in (
            ("SCORECARD", load_scorecard(), "symbol"),
            ("FRAUD TRACKER",
             load_parquet(["company_repo", "_index", "fraud_tracker.parquet"]),
             "symbol"),
            ("CATALYSTS", load_catalyst_index(), "symbol"),
            ("GUIDANCE", load_guidance_tracker(), "symbol")):
        try:
            if not df.empty and col in df.columns:
                rows = df[df[col].astype(str).str.upper() == sym].tail(8)
                if not rows.empty:
                    parts.append(f"## {label}\n" + rows.to_csv(index=False)[:6000])
        except Exception:
            pass
    return "\n\n".join(parts) or "DATA_MISSING (no coverage yet)"


def page_ask():
    st.title("💬 Ask")
    st.caption("Just type a question and **mention the company** — "
               "*\"how is anondita doing?\"*, *\"what did TCS guide and are they "
               "credible?\"*. Answers come from everything on Drive (concall/AR "
               "summaries, scorecard, fraud tracker, guidance, catalysts, "
               "community) and cite their source. Mention a different company "
               "any time to switch. (Local fallback that never goes down: "
               "`scripts\\ask.bat`.)")

    import sys as _sys
    _sp = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scripts")
    if _sp not in _sys.path:
        _sys.path.insert(0, _sp)

    uni = load_csv(["company_repo", "_index", "company_universe.csv"])
    if uni.empty:
        st.error("Universe not loadable.")
        return

    cur = st.session_state.get("ask_sym")
    if cur:
        st.markdown(f"**Talking about: {cur} · "
                    f"{st.session_state.get('ask_name', '')}**")
    for u, a in st.session_state.get("ask_hist", []):
        st.chat_message("user").write(u)
        st.chat_message("assistant").write(a)

    msg = st.chat_input("Ask about a company…")
    if not msg:
        return
    st.chat_message("user").write(msg)

    try:
        from ask_company import resolve, SYSTEM, answer, build_pool
    except ImportError as e:
        st.chat_message("assistant").write(
            f"Chat library not installed in this deployment ({e}). "
            "Use `scripts\\ask.bat` locally.")
        return

    # (re)lock the company whenever the message names a different one
    hit = resolve(None, None, msg, uni)
    if hit and hit[1] != cur:
        isin, sym, name = hit
        with st.spinner(f"Loading everything on Drive for {sym}…"):
            st.session_state["ask_ctx"] = _ask_assemble_ctx(isin, sym)
        st.session_state["ask_sym"] = sym
        st.session_state["ask_name"] = name
        st.session_state["ask_hist"] = []
        cur = sym
    if not cur:
        st.chat_message("assistant").write(
            "Which company? Mention its name or symbol — e.g. "
            "*\"how is Anondita doing?\"*")
        return

    # message was only the company name → context loaded, invite a question
    if hit and msg.strip().upper() in (cur, st.session_state["ask_name"].upper()):
        st.chat_message("assistant").write(
            f"Loaded everything on Drive for **{cur}** "
            f"({st.session_state['ask_name']}). What would you like to know?")
        return

    try:
        if "ask_pool" not in st.session_state:
            st.session_state["ask_pool"] = build_pool()
        base = SYSTEM.format(company=st.session_state["ask_name"], symbol=cur,
                             context=st.session_state["ask_ctx"])
        with st.spinner("Thinking…"):
            a = answer(st.session_state["ask_pool"], base,
                       st.session_state["ask_hist"], msg)
        st.chat_message("assistant").write(a)
        st.session_state["ask_hist"].append((msg, a))
    except SystemExit:
        st.chat_message("assistant").write(
            "No Gemini keys configured — add `BACKFILL_GEMINI_KEY_1` (or "
            "`DAILY_GEMINI_KEY_1` / `GEMINI_API_KEY`) to this app's Streamlit "
            "secrets, then reboot. Local alternative: `scripts\\ask.bat`.")
    except Exception as e:
        st.chat_message("assistant").write(f"Chat failed: {str(e)[:160]}")


def page_fraud_tracker():
    st.title("🕵️ Fraud Tracker")
    st.caption(
        "Standalone company-wise fraud signal (T7) — the **worst** of the two engines: "
        "exchange/regulator surveillance (investigative grade 0–4) and T2 forensic "
        "accounting rules (0–100). Bands: 🔴 RED ≥70 · 🟠 ALERT ≥45 · 🟡 WATCH ≥20. "
        "Refreshed nightly after the scorecard. **Read the why column** — ASM/ESM/GSM/"
        "T2T listings are price-volatility *control measures*, often speculative price "
        "action rather than fraud; only SEBI/NFRA orders, forensic flags and fraud-news "
        "are integrity signals."
    )
    snap = load_parquet(["company_repo", "_index", "fraud_tracker.parquet"])
    if snap.empty:
        st.info("fraud_tracker.parquet not found yet — it appears after the first "
                "nightly t4 run (or `python scripts/build_fraud_tracker.py`).")
        return

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Tracked", len(snap))
    for col, b in ((c2, "RED"), (c3, "ALERT"), (c4, "WATCH")):
        col.metric(b.title(), int((snap["band"] == b).sum()))

    f1, f2 = st.columns([1, 2])
    bands = f1.multiselect("Band", ["RED", "ALERT", "WATCH"],
                           default=["RED", "ALERT", "WATCH"])
    search = f2.text_input("Search symbol / company", "")
    view = snap[snap["band"].isin(bands)]
    if search.strip():
        q = search.strip().upper()
        view = view[view["symbol"].astype(str).str.upper().str.contains(q)
                    | view["company_name"].astype(str).str.upper().str.contains(q)]

    disp = view.copy()
    disp["trend"] = disp["trend"].map(lambda t: _TREND_ARROW.get(str(t), str(t)))
    disp["days_on_tracker"] = (
        pd.to_datetime(disp["as_of"], errors="coerce")
        - pd.to_datetime(disp["first_flagged_at"], errors="coerce")).dt.days
    cols = ["symbol", "company_name", "fraud_score", "band", "reason", "trend",
            "investigative_grade", "forensic_score", "days_on_tracker",
            "first_flagged_at"]
    cols = [c for c in cols if c in disp.columns]   # older parquet w/o reason
    styled = (disp[cols].style
              .map(lambda b: f"color:white;background-color:"
                             f"{_BAND_COLORS.get(b, '#777')}", subset=["band"])
              .format({"fraud_score": "{:.0f}", "forensic_score": "{:.0f}"}))
    st.dataframe(styled, use_container_width=True, hide_index=True,
                 height=min(38 * (len(disp) + 1), 600),
                 column_config={"reason": st.column_config.TextColumn(
                     "why (engine(points): causes)", width="large")})

    # ---- drill-down: score history + reasons ----
    st.markdown("---")
    st.subheader("🔬 Company drill-down")
    drill = st.text_input("Symbol", "", key="ft_drill").strip().upper()
    if not drill:
        return
    row = snap[snap["symbol"].astype(str).str.upper() == drill]
    if row.empty:
        st.warning(f"{drill} is not on the tracker (score < 20 or no data).")
        return
    r = row.iloc[0]
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Fraud score", f"{r['fraud_score']:.0f}/100", r["band"])
    m2.metric("Investigative grade", f"{int(r['investigative_grade'])}/4")
    m3.metric("Forensic score", f"{r['forensic_score']:.0f}/100")
    m4.metric("First flagged", str(r["first_flagged_at"]))
    if str(r.get("reason", "")).strip():
        st.markdown(f"**Why {r['band']}** (driver: {r.get('score_driver', '?')}): "
                    f"{r['reason']}")
    if str(r.get("grade_reason", "")).strip():
        st.markdown(f"**Surveillance / regulator:** {r['grade_reason']}")
    if str(r.get("forensic_flags", "")).strip():
        st.markdown(f"**Forensic flags:** {r['forensic_flags']}")

    hist = load_parquet(["company_repo", "_index", "fraud_tracker_history.parquet"])
    if not hist.empty:
        h = hist[hist["symbol"].astype(str).str.upper() == drill].copy()
        if not h.empty:
            h["as_of"] = pd.to_datetime(h["as_of"], errors="coerce")
            h = h.dropna(subset=["as_of"]).sort_values("as_of")
            fig = go.Figure()
            fig.add_hrect(y0=70, y1=100, fillcolor="#e74c3c", opacity=0.08, line_width=0)
            fig.add_hrect(y0=45, y1=70, fillcolor="#e67e22", opacity=0.08, line_width=0)
            fig.add_hrect(y0=20, y1=45, fillcolor="#f39c12", opacity=0.08, line_width=0)
            fig.add_trace(go.Scatter(x=h["as_of"], y=h["fraud_score"],
                                     mode="lines+markers", name="Fraud score",
                                     line=dict(color="#c0392b", width=2)))
            fig.update_layout(height=300, yaxis=dict(range=[0, 100], title="Fraud score"),
                              margin=dict(l=10, r=10, t=10, b=10), showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
            st.caption("Score history (a drop to 0 = cleared off the tracker that day).")


def page_scorecard():
    st.title("🏆 Company Scorecard")
    st.caption(
        "8-factor conviction blend — **Technical · Fundamental · Financial Health · "
        "Mgmt Credibility · Valuation · Guidance · Fraud Safety · Investigative**. "
        "Greyed bars = source parquet not yet populated (auto-lights when data lands)."
    )

    with st.expander("📐 How the maths works (with a worked example)"):
        st.markdown(
            """
Every factor is **0–100, higher = better** (risk factors are flipped so 100 = safe).
The composite is a weighted average over **only the factors that have data** — the
weights of missing factors are removed from the denominator, not treated as zero:

```
composite = Σ (weight × score)  /  Σ (weight of factors present)
```

| Factor | Weight | Formula (intuition) |
|---|---|---|
| 📈 Technical | 18% | (strategies firing ÷ 5 × 100 + signal score) ÷ 2 |
| 📊 Fundamental | 18% | avg of `50 + growth% × 1.5` for rev/PAT YoY+QoQ, + OPM trend |
| 🏦 Fin. Health | 13.5% | avg of CFO/PAT×50, int-cover×10, 100−debt/EBITDA×20, ROCE×2.5, 100−WC days |
| 🎯 Mgmt Cred. | 13.5% | latest said-vs-delivered credibility score (×10 if 0–10 scale) |
| 💰 Valuation | 9% | avg of P/E rank vs segment peers, PEG rank, own-3y P/E percentile (cheap = high) |
| 📋 Guidance | 9% | `50 + 15 × (positive − negative guidance flags)` |
| 🛡️ Fraud Safety | 9% | `100 −` forensic penalty (CFO<PAT, receivables, leverage, WC, coverage, ROCE) |
| 🔍 Investigative | 10% | `(4 − grade) ÷ 4 × 100` from NSE ASM/ESM/GSM/T2T + BSE lists (0=clean → 100) |

**Worked example — TCS (live run 10-Jun-2026), 5 of 8 factors present (62% complete):**

```
fundamental 66.7 × 0.18  = 12.01
fin health  82.3 × 0.135 = 11.11
valuation   92.2 × 0.09  =  8.30
fraud safe 100.0 × 0.09  =  9.00
investig.  100.0 × 0.10  = 10.00
                  ─────────────
sum = 50.41   available weight = 0.595
composite = 50.41 / 0.595 = 84.7
```

Technical / Mgmt-Cred / Guidance were missing for TCS, so their 45% combined weight
dropped out of the denominator. As the nightly backfill fills those sources the bars
light up and the composite re-blends automatically.

⚠️ **Read it honestly:** a high composite at low completeness (≤38%) rests on thin
evidence — use the *Min completeness* slider. Risk flags **highlight, never filter**:
a 90-composite company on a watchlist keeps its 90 but shows a low red Investigative bar.
            """
        )

    sc = load_scorecard()

    if sc.empty:
        st.info(
            "No scorecard data yet — `company_scorecard.parquet` hasn't been generated.\n\n"
            "Run `python scripts/build_scorecard.py` (or wait for the nightly CI job). "
            "This page auto-refreshes once Drive data is available.",
            icon="🔄",
        )
        return

    # ── Filters ───────────────────────────────────────────────────────────────
    f1, f2 = st.columns([2, 1])
    with f1:
        seg_opts = ["All"] + sorted(
            sc["mcap_segment"].dropna().unique().tolist()
        ) if "mcap_segment" in sc.columns else ["All"]
        seg_filter = st.selectbox("Segment", seg_opts, key="sc_seg")
    with f2:
        min_comp = st.slider("Min completeness %", 0, 100, 0, 10, key="sc_min_comp")

    df = sc.copy()
    if seg_filter != "All" and "mcap_segment" in df.columns:
        df = df[df["mcap_segment"] == seg_filter]
    if "data_completeness_pct" in df.columns:
        df = df[df["data_completeness_pct"].fillna(0) >= min_comp]

    if df.empty:
        st.warning("No companies match the selected filters.")
        return

    df = df.sort_values(
        "composite_score", ascending=False, na_position="last"
    ).reset_index(drop=True)

    # ── Ranked table ──────────────────────────────────────────────────────────
    st.subheader(f"Ranked — {len(df)} companies")

    tbl_cols = ["symbol"]
    if "company_name" in df.columns:
        tbl_cols.append("company_name")
    tbl_cols += ["composite_score", "data_completeness_pct"]
    tbl_cols += [c for c, _, _ in _SCORE_FACTORS if c in df.columns]
    tbl = df[[c for c in tbl_cols if c in df.columns]].copy()

    for col in ["composite_score"] + [c for c, _, _ in _SCORE_FACTORS]:
        if col in tbl.columns:
            tbl[col] = tbl[col].apply(
                lambda v: round(float(v), 1) if pd.notna(v) else None
            )
    if "data_completeness_pct" in tbl.columns:
        tbl["data_completeness_pct"] = tbl["data_completeness_pct"].apply(
            lambda v: round(float(v), 0) if pd.notna(v) else None
        )

    rename_map = {
        "symbol": "Symbol", "company_name": "Company",
        "composite_score": "Composite ▼", "data_completeness_pct": "Complete %",
    }
    rename_map.update({c: lbl for c, lbl, _ in _SCORE_FACTORS})
    tbl = tbl.rename(columns=rename_map)
    st.dataframe(tbl, use_container_width=True, height=420)

    # ── Per-company drill-down ────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("Factor Breakdown")

    default_sym = df["symbol"].iloc[0] if not df.empty else ""
    drill_sym = st.text_input(
        "Symbol", default_sym, key="sc_drill_sym",
        help="Type any symbol to see its full 7-factor breakdown.",
    ).upper().strip()

    if not drill_sym:
        return

    r_rows = sc[sc["symbol"].astype(str).str.upper() == drill_sym]
    if r_rows.empty:
        st.warning(
            f"No scorecard row for **{drill_sym}** — "
            "run `build_scorecard.py --names \"{drill_sym}\"` to add it."
        )
        return

    r = r_rows.iloc[0]
    comp_v = r.get("composite_score")
    comp_s = f"{float(comp_v):.1f}" if pd.notna(comp_v) else "N/A"
    cplt_v = r.get("data_completeness_pct")
    cplt_s = f"{float(cplt_v):.0f}%" if pd.notna(cplt_v) else "N/A"
    seg_s  = str(r.get("mcap_segment", "")) \
        if pd.notna(r.get("mcap_segment", float("nan"))) else ""

    badge_html = _scorecard_badge(drill_sym, sc)
    st.markdown(
        f"<h4>{drill_sym}&nbsp;{badge_html}&nbsp;&nbsp;"
        f"<span style='font-size:13px;color:#888;font-weight:normal'>"
        f"{seg_s}{' · ' if seg_s else ''}Completeness: {cplt_s}</span></h4>",
        unsafe_allow_html=True,
    )

    # 7-metric summary row
    factor_cols = st.columns(len(_SCORE_FACTORS))
    for fc, (col_key, label, icon) in zip(factor_cols, _SCORE_FACTORS):
        v = r.get(col_key)
        if pd.isna(v):
            fc.metric(f"{icon} {label}", "—", help="Source data not yet available")
        else:
            fc.metric(f"{icon} {label}", f"{float(v):.0f}")

    # Horizontal bar chart
    labels = [f"{icon} {lbl}" for _, lbl, icon in _SCORE_FACTORS]
    vals   = [r.get(c) for c, _, _ in _SCORE_FACTORS]
    colors = [_score_band_color(v) for v in vals]
    y_vals = [float(v) if pd.notna(v) else 0.0 for v in vals]
    text_  = [f"{float(v):.0f}" if pd.notna(v) else "N/A" for v in vals]

    fig_sc = go.Figure(go.Bar(
        x=y_vals,
        y=labels,
        orientation="h",
        marker_color=colors,
        text=text_,
        textposition="outside",
        cliponaxis=False,
    ))
    for i, v in enumerate(vals):
        if pd.isna(v):
            fig_sc.add_shape(
                type="rect",
                x0=0, x1=112,
                y0=i - 0.45, y1=i + 0.45,
                fillcolor="rgba(200,200,200,0.15)",
                line_width=0,
                layer="below",
            )
    fig_sc.add_vline(
        x=50, line_dash="dot", line_color="#aaa",
        annotation_text="50", annotation_position="top right",
    )
    fig_sc.update_layout(
        xaxis=dict(range=[0, 115], title="Score (0–100)"),
        yaxis=dict(autorange="reversed"),
        height=360,
        margin=dict(t=20, b=20, l=10, r=55),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig_sc, use_container_width=True, key=f"sc_bar_{drill_sym}")

    missing = [lbl for c, lbl, _ in _SCORE_FACTORS if pd.isna(r.get(c))]
    if missing:
        st.caption(
            "⚠️ Missing factors (source not yet populated): "
            + ", ".join(f"**{m}**" for m in missing)
            + ".  Weights auto-renormalize — composite reflects available data only."
        )

    # T5 — catalyst note, diffed vs the previous note (new = green) + freshness pill
    cat = load_catalyst_index()
    if not cat.empty and "symbol" in cat.columns:
        crows = cat[cat["symbol"].astype(str).str.upper() == drill_sym].sort_values("as_of")
        if not crows.empty:
            cur = crows.iloc[-1]
            prev = crows.iloc[-2] if len(crows) >= 2 else None
            prev_txt = _catalyst_text(prev) if prev is not None else None
            diff_html, changed = _catalyst_diff_html(_catalyst_text(cur), prev_txt)
            st.markdown(
                f"💡 **Catalyst** · {cur.get('catalyst_type', '')} &nbsp; "
                + _freshness_badge(cur.get("as_of", "")), unsafe_allow_html=True)
            if prev is not None and not changed:
                st.success(f"🟰 No change since {str(prev.get('as_of',''))[:10]} "
                           "— catalyst text identical.")
            elif prev is not None:
                st.caption("🟢 Highlighted = new / changed since the previous note.")
            st.markdown(diff_html, unsafe_allow_html=True)

    # Fraud / risk checks for the drilled symbol (T4/T7)
    _render_fraud_checks(drill_sym)


def _render_fraud_checks(sym: str) -> None:
    """Fraud-tracker band + investigative grade + forensic flags, freshness-stamped."""
    ft = load_parquet(["company_repo", "_index", "fraud_tracker.parquet"])
    inv = load_parquet(["company_repo", "_index", "investigative_fraud.parquet"])
    su = sym.upper()
    frow = (ft[ft["symbol"].astype(str).str.upper() == su]
            if not ft.empty and "symbol" in ft.columns else pd.DataFrame())
    irow = (inv[inv["symbol"].astype(str).str.upper() == su]
            if not inv.empty and "symbol" in inv.columns else pd.DataFrame())
    if frow.empty and irow.empty:
        st.markdown("🛡️ **Fraud checks:** no flags — untracked / clean "
                    "(below the WATCH threshold).")
        return
    parts = ["🛡️ **Fraud / forensic checks**"]
    if not frow.empty:
        r = frow.iloc[-1]
        band = str(r.get("band", "") or "—")
        color = _BAND_COLORS.get(band, "#7f8c8d")
        score = r.get("fraud_score")
        trend = _TREND_ARROW.get(str(r.get("trend", "")), "")
        parts.append(
            f"&nbsp; <span style='background:{color};color:#fff;padding:1px 7px;"
            f"border-radius:8px;font-size:11px;font-weight:600'>{band} "
            f"{'' if pd.isna(score) else int(score)}</span> {trend} &nbsp; "
            + _freshness_badge(str(r.get("last_changed_at", r.get("computed_at", "")))[:10],
                               "checked"))
        reason = str(r.get("reason", "") or "").strip()
        if reason and reason.lower() != "nan":
            parts.append(f"<br>**Why:** {reason}")
    if not irow.empty:
        r = irow.iloc[-1]
        grade = r.get("investigative_grade")
        greason = str(r.get("grade_reason", "") or "").strip()
        flags = [f"{k.upper()}={r.get(k)}" for k in ("asm_level", "esm_level", "gsm_stage")
                 if str(r.get(k, "")).strip() not in ("", "0", "none", "nan", "False")]
        sebi = r.get("sebi_actions"); nfra = r.get("nfra_actions")
        extra = []
        if sebi and str(sebi) not in ("0", "nan"): extra.append(f"SEBI×{sebi}")
        if nfra and str(nfra) not in ("0", "nan"): extra.append(f"NFRA×{nfra}")
        line = f"<br>**Investigative:** grade {'' if pd.isna(grade) else int(grade)}/4"
        if greason and greason.lower() not in ("nan", "clean"):
            line += f" — {greason}"
        if flags or extra:
            line += " · " + ", ".join(flags + extra)
        parts.append(line)
    st.markdown(" ".join(parts), unsafe_allow_html=True)


def _render_quarterly_table(symbol: str) -> None:
    """Last 6 quarters of Sales / Net Profit / EPS, with latest-quarter YoY & QoQ
    and a management-guidance row, growth cells color-graded."""
    import sys as _sys
    _sp = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scripts")
    if _sp not in _sys.path:
        _sys.path.insert(0, _sp)
    from gradation import grade_growth, TIER_COLOR

    stmt = load_parquet(["fundamentals", "statements", f"{symbol}.parquet"])
    if stmt.empty or "statement" not in stmt.columns:
        return
    q = stmt[stmt["statement"] == "quarterly_pl"]
    if q.empty:
        return

    def _series(item):
        sub = q[q["line_item"] == item]
        per = sub["period"].astype(str).tolist()
        val = pd.to_numeric(sub["value"], errors="coerce").tolist()
        return per, val

    rows_def = [("Sales", "Sales"), ("Net Profit", "Net Profit"), ("EPS", "EPS in Rs")]
    periods = None
    for _, item in rows_def:
        p, _v = _series(item)
        if p:
            periods = p[-6:]
            break
    if not periods:
        return

    st.markdown("##### 📊 Last 6 quarters")
    yoy_qoq = {}
    table = {"Quarter": periods}
    for label, item in rows_def:
        p, v = _series(item)
        d = dict(zip(p, v))
        table[label] = [d.get(per) for per in periods]
        full_v = v
        yoy = (_pct_change(full_v[-1], full_v[-5]) if len(full_v) >= 5 else None)
        qoq = (_pct_change(full_v[-1], full_v[-2]) if len(full_v) >= 2 else None)
        yoy_qoq[label] = (yoy, qoq)
    df = pd.DataFrame(table)
    st.dataframe(df, use_container_width=True, hide_index=True)

    # YoY / QoQ summary for the latest quarter, color-graded
    guid = load_guidance_tracker()
    gtxt = ""
    if not guid.empty and "symbol" in guid.columns:
        gr = guid[guid["symbol"].astype(str).str.upper() == symbol]
        gr = gr[pd.to_numeric(gr.get("cagr_pct"), errors="coerce").notna()]
        if not gr.empty:
            gv = pd.to_numeric(gr["cagr_pct"], errors="coerce").max()
            gtxt = (f"<span style='background:{TIER_COLOR[grade_growth(gv)]};"
                    f"padding:1px 7px;border-radius:8px'>Guidance ~{gv:.0f}%</span>")

    def _chip(v):
        if v is None:
            return "<span style='color:#999'>—</span>"
        return (f"<span style='background:{TIER_COLOR[grade_growth(v)]};"
                f"padding:1px 7px;border-radius:8px'>{v:+.0f}%</span>")

    cells = []
    for label in ("Sales", "Net Profit", "EPS"):
        yoy, qoq = yoy_qoq.get(label, (None, None))
        cells.append(f"<b>{label}</b> &nbsp;YoY {_chip(yoy)} &nbsp;QoQ {_chip(qoq)}")
    st.markdown("Latest quarter: &nbsp; " + " &nbsp;|&nbsp; ".join(cells)
                + (f" &nbsp;|&nbsp; {gtxt}" if gtxt else ""), unsafe_allow_html=True)


def _pct_change(a, b):
    try:
        a, b = float(a), float(b)
        if b == 0 or pd.isna(a) or pd.isna(b):
            return None
        return (a - b) / abs(b) * 100.0
    except (TypeError, ValueError):
        return None


@st.cache_data(ttl=300, show_spinner=False)
def load_screener_grades() -> pd.DataFrame:
    """6-rule attractiveness grades (build_screener_grades.py)."""
    return load_parquet(["company_repo", "_index", "screener_grades.parquet"])


def page_screener():
    import sys as _sys
    _sp = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scripts")
    if _sp not in _sys.path:
        _sys.path.insert(0, _sp)
    from gradation import GREEN_TIERS, TIER_COLOR

    st.title("🟢 Attractiveness Screener")
    st.caption("Six rules, color-graded. Green = Good/Great/Exceptional · amber = "
               "Ok/Decent · red = Poor. Sorted by how many rules are green.")
    df = load_screener_grades()
    if df.empty:
        st.info("`screener_grades.parquet` not found yet — runs nightly "
                "(`build_screener_grades.py`).")
        return

    metrics = [("yoy", "YOY"), ("qoq", "QOQ"), ("guidance", "Guidance"),
               ("val", "Valuation"), ("cfo", "CFO"), ("roe", "ROE")]
    cols = st.columns(len(metrics))
    for (key, lbl), c in zip(metrics, cols):
        green = int(df[f"{key}_tier"].isin(GREEN_TIERS).sum())
        c.metric(f"{lbl} 🟢", f"{green}", help=f"{lbl} Good-or-better")

    c1, c2 = st.columns([1, 2])
    min_g = c1.slider("Min green count", 0, 6, 3)
    qtext = c2.text_input("Filter symbol / name", "")
    view = df[df["green_count"] >= min_g]
    if qtext:
        m = (view["symbol"].astype(str).str.contains(qtext, case=False, na=False) |
             view["company_name"].astype(str).str.contains(qtext, case=False, na=False))
        view = view[m]
    view = view.sort_values("green_count", ascending=False).head(300).reset_index(drop=True)
    st.caption(f"Showing {len(view)} companies (green ≥ {min_g}).")

    disp = pd.DataFrame({
        "Symbol": view["symbol"], "Name": view["company_name"].astype(str).str[:26],
        "🟢#": view["green_count"],
        "YOY%": pd.to_numeric(view["yoy"], errors="coerce").round(0),
        "QOQ%": pd.to_numeric(view["qoq"], errors="coerce").round(0),
        "Guid%": pd.to_numeric(view["guidance"], errors="coerce").round(0),
        "Val(PE)": pd.to_numeric(view["val_value"], errors="coerce").round(1),
        "CFO(x)": pd.to_numeric(view["cfo_ratio"], errors="coerce").round(2),
        "ROE%": pd.to_numeric(view["roe"], errors="coerce").round(0),
    })
    tier_of = {"YOY%": "yoy_tier", "QOQ%": "qoq_tier", "Guid%": "guidance_tier",
               "Val(PE)": "val_tier", "CFO(x)": "cfo_tier", "ROE%": "roe_tier"}

    def _style(_):
        sty = pd.DataFrame("", index=disp.index, columns=disp.columns)
        for dcol, tcol in tier_of.items():
            colors = view[tcol].map(lambda t: f"background-color:{TIER_COLOR.get(t, '#eeeeee')}")
            sty[dcol] = colors.values
        return sty

    st.dataframe(disp.style.apply(_style, axis=None), use_container_width=True,
                 height=620, hide_index=True)
    st.caption("Valuation = PE (v1; PB for finance / EV-EBITDA for asset-heavy coming). "
               "Guidance shown only where management gave a number.")


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


# ---------- Email toggles (CI mailers read mail_settings.json from Drive) ----------

_MAIL_TOGGLES = [
    ("pead_guidance",   "📊 Results vs guidance (20:00)"),
    ("pead_tomorrow",   "📅 Results tomorrow (20:00)"),
    ("fraud_scan",      "🚨 Fraud scan findings (21:30)"),
    ("catalyst",        "💡 Catalyst notes (21:30)"),
    ("guidance_digest", "🎯 Guidance table + 🚀 flags (20:00)"),
    ("ops_digest",      "🩺 Ops digest (08:30)"),
    ("ar_focus",        "📚 AR focus/defocus digest"),
    ("pf_digest",       "💼 PF daily digest"),
]
_MAIL_SETTINGS_NAME = "mail_settings.json"


def _mail_settings_loc(drive):
    """(index_folder_id, existing_file_id_or_None) for company_repo/_index."""
    repo = _find_subfolder(drive, os.environ["GDRIVE_FOLDER_ID"], "company_repo")
    idx = _find_subfolder(drive, repo, "_index") if repo else None
    if not idx:
        return None, None
    q = f"name='{_MAIL_SETTINGS_NAME}' and '{idx}' in parents and trashed=false"
    files = drive.files().list(q=q, fields="files(id)").execute().get("files", [])
    return idx, (files[0]["id"] if files else None)


def render_mail_toggles_sidebar():
    with st.sidebar.expander("📧 Email toggles"):
        if "mail_settings" not in st.session_state:
            cur = {}
            try:
                drive = drive_service()
                _, fid = _mail_settings_loc(drive)
                if fid:
                    cur = json.loads(_download_bytes(drive, fid).decode("utf-8"))
            except Exception:
                pass
            st.session_state["mail_settings"] = {
                k: bool(cur.get(k, True)) for k, _ in _MAIL_TOGGLES}
        saved = st.session_state["mail_settings"]
        for key, label in _MAIL_TOGGLES:
            st.toggle(label, value=saved.get(key, True), key=f"mailtog_{key}")
        if st.button("💾 Save", key="mail_save", use_container_width=True):
            new = {k: bool(st.session_state.get(f"mailtog_{k}", True))
                   for k, _ in _MAIL_TOGGLES}
            try:
                from googleapiclient.http import MediaIoBaseUpload
                drive = drive_service()
                idx, fid = _mail_settings_loc(drive)
                if not idx:
                    st.error("Drive _index folder not found.")
                    return
                payload = json.dumps(
                    {**new, "updated_at": datetime.now().isoformat(timespec="seconds")},
                    indent=2).encode("utf-8")
                media = MediaIoBaseUpload(io.BytesIO(payload),
                                          mimetype="application/json", resumable=False)
                if fid:
                    drive.files().update(fileId=fid, media_body=media).execute()
                else:
                    drive.files().create(body={"name": _MAIL_SETTINGS_NAME,
                                               "parents": [idx]},
                                         media_body=media, fields="id").execute()
                st.session_state["mail_settings"] = new
                st.success("Saved — applies from the next scheduled run.")
            except Exception as e:
                st.error(f"Save failed: {str(e)[:120]}")


def render_review_flag_sidebar():
    """🚩 One-box channel to Claude: rows land in _index/review_flags.csv on
    Drive; Claude reads them at session start and the morning ops digest lists
    the open ones."""
    with st.sidebar.expander("🚩 Flag for review"):
        txt = st.text_area("What should Claude look at?", "",
                           key="review_flag_text",
                           placeholder="e.g. WOCKPHARMA growth mail looks wrong")
        if st.button("Submit flag", key="review_flag_btn",
                     use_container_width=True) and txt.strip():
            try:
                drive = drive_service()
                idx, _ = _mail_settings_loc(drive)   # same _index folder
                if not idx:
                    st.error("Drive _index folder not found.")
                    return
                q = f"name='review_flags.csv' and '{idx}' in parents and trashed=false"
                files = drive.files().list(q=q, fields="files(id)").execute() \
                    .get("files", [])
                fid = files[0]["id"] if files else None
                if fid:
                    old = pd.read_csv(io.BytesIO(_download_bytes(drive, fid)))
                else:
                    old = pd.DataFrame(columns=["ts", "flag", "status"])
                row = {"ts": datetime.now().strftime("%Y-%m-%d %H:%M"),
                       "flag": txt.strip()[:500], "status": "open"}
                out = pd.concat([old, pd.DataFrame([row])], ignore_index=True)
                from googleapiclient.http import MediaIoBaseUpload
                media = MediaIoBaseUpload(
                    io.BytesIO(out.to_csv(index=False).encode("utf-8")),
                    mimetype="text/csv", resumable=False)
                if fid:
                    drive.files().update(fileId=fid, media_body=media).execute()
                else:
                    drive.files().create(body={"name": "review_flags.csv",
                                               "parents": [idx]},
                                         media_body=media, fields="id").execute()
                st.success("Flagged — appears in the morning ops digest and "
                           "Claude's next session.")
            except Exception as e:
                st.error(f"Flag failed: {str(e)[:120]}")


def main():
    st.sidebar.title("Signals India")

    page = st.sidebar.radio("Page", [
        "Market Overview",
        "Market Trends",
        "Today's Signals",
        "Screener 🟢",
        "Company Intel",
        "Mgmt Guidance",
        "Doc Viewer",
        "Doc Library",
        "Deep Dive",
        "My Portfolio",
        "Graphs",
        "Scorecard",
        "Fraud Tracker",
        "Ask 💬",
        "Stock Detail",
        "Strategy Docs",
    ])
    render_health_sidebar()
    render_mail_toggles_sidebar()
    render_review_flag_sidebar()
    st.sidebar.markdown("---")
    st.sidebar.caption(f"Loaded at {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    st.sidebar.caption("Data refreshes from Drive every 5 min (cache TTL)")

    if page == "Market Overview":
        _safe_render(page_market_overview)
    elif page == "Market Trends":
        _safe_render(page_market_trends)
    elif page == "Today's Signals":
        _safe_render(page_signals)
    elif page == "Screener 🟢":
        _safe_render(page_screener)
    elif page == "Company Intel":
        _safe_render(page_company_intel)
    elif page == "Mgmt Guidance":
        _safe_render(page_guidance)
    elif page == "Doc Viewer":
        _safe_render(page_doc_viewer)
    elif page == "Doc Library":
        _safe_render(page_doc_upload)
    elif page == "Deep Dive":
        _safe_render(page_deep_dive)
    elif page == "My Portfolio":
        _safe_render(page_portfolio)
    elif page == "Graphs":
        _safe_render(page_graphs)
    elif page == "Scorecard":
        _safe_render(page_scorecard)
    elif page == "Fraud Tracker":
        _safe_render(page_fraud_tracker)
    elif page == "Ask 💬":
        _safe_render(page_ask)
    elif page == "Stock Detail":
        _safe_render(page_stock_detail)
    elif page == "Strategy Docs":
        _safe_render(page_strategy_docs)



main()
