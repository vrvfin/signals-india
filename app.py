"""
Stage 8 — Streamlit dashboard (v2 with enhanced breadth + macro strip).

Run locally:
    streamlit run app.py
"""

from __future__ import annotations

import io
import os
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
    """Latest pipeline health report dict, or None if not found."""
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

def render_health_sidebar():
    """Compact pipeline status line — call once in the sidebar, every page."""
    rep = load_health_report()
    if rep is None or "_error" in rep:
        st.sidebar.error("Pipeline status: unknown")
        return

    run_at = rep.get("run_at")
    age_h = None
    if run_at:
        try:
            dt = datetime.fromisoformat(run_at.replace("Z", "+00:00"))
            age_h = (datetime.now(timezone.utc) - dt).total_seconds() / 3600
        except Exception:
            pass

    overall = rep.get("overall", "UNKNOWN")
    if age_h is not None and age_h > 30:
        st.sidebar.error(f"Pipeline stale — last run {age_h:.0f}h ago")
    elif overall == "HEALTHY":
        st.sidebar.success("Data: HEALTHY")
    elif overall == "DEGRADED":
        st.sidebar.warning(f"Data: DEGRADED ({rep.get('warnings', 0)} warn)")
    elif overall == "FAIL":
        st.sidebar.error(f"Data: FAILED ({rep.get('critical_failures', 0)} critical)")
    else:
        st.sidebar.info(f"Data: {overall}")

def render_health_banner():
    rep = load_health_report()
    if rep is None:
        st.error("No pipeline health report found on Drive — "
                 "the daily workflow may have never run.")
        return
    if "_error" in rep:
        st.warning(f"Could not read health report: {rep['_error']}")
        return

    run_at = rep.get("run_at")
    age_h = None
    if run_at:
        try:
            dt = datetime.fromisoformat(run_at.replace("Z", "+00:00"))
            age_h = (datetime.now(timezone.utc) - dt).total_seconds() / 3600
        except Exception:
            pass

    overall = rep.get("overall", "UNKNOWN")
    age_txt = f" — checked {age_h:.0f}h ago" if age_h is not None else ""

    if age_h is not None and age_h > 30:
        st.error(f"Pipeline has not run for {age_h:.0f} hours "
                 f"(last check: {run_at}). It may not be running at all — "
                 f"data below is STALE.")
    elif overall == "HEALTHY":
        st.success(f"All data fresh — pipeline HEALTHY{age_txt}.")
    elif overall == "DEGRADED":
        st.warning(f"Pipeline DEGRADED — {rep.get('warnings', 0)} warning(s){age_txt}. "
                   f"Some strategies below may be stale (see detail).")
    elif overall == "FAIL":
        st.error(f"Pipeline FAILED — {rep.get('critical_failures', 0)} critical "
                 f"failure(s){age_txt}. Data below may be stale.")
    else:
        st.info(f"Pipeline status: {overall}{age_txt}")

    checks = rep.get("checks", [])
    if checks:
        with st.expander("Data freshness detail (per strategy & data pull)"):
            st.dataframe(pd.DataFrame(checks),
                         use_container_width=True, hide_index=True)

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


@st.cache_data(ttl=300, show_spinner=False)
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
    zone_opts = sorted(signals["zone_type"].dropna().unique())

    c1, c2 = st.columns([3, 2])
    with c1:
        chosen = st.multiselect("Strategies (add / remove)",
                                strat_groups, default=strat_groups)
    with c2:
        zones = st.multiselect("Zones", zone_opts, default=["buy", "add"])

    c3, c4, c5 = st.columns(3)
    with c3:
        min_strats = st.number_input("Min strategies a stock must satisfy",
                                     min_value=1, max_value=10, value=2, step=1)
    with c4:
        per_page = st.number_input("Charts per page", min_value=4,
                                   max_value=40, value=12, step=4)
    with c5:
        tf = st.selectbox("Timeframe", list(TIMEFRAME_DAYS.keys()), index=2)

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
        st.info(f"No stock satisfies ≥ {int(min_strats)} of the selected "
                f"strategies. Lower the threshold to see more.")
        return

    best = sel.groupby("symbol")["score"].max().reset_index(name="best_score")
    conv = conv.merge(best, on="symbol", how="left")
    conv = conv.sort_values(["n_strategies", "best_score"],
                            ascending=[False, False]).reset_index(drop=True)

    # --- Pagination ---
    total = len(conv)
    per_page = int(per_page)
    n_pages = max(1, (total + per_page - 1) // per_page)

    if "graphs_page" not in st.session_state:
        st.session_state["graphs_page"] = 0
    st.session_state["graphs_page"] = min(st.session_state["graphs_page"],
                                          n_pages - 1)
    page = st.session_state["graphs_page"]

    def _nav(suffix):
        n1, n2, n3 = st.columns([1, 2, 1])
        with n1:
            if st.button("◀ Prev", key=f"prev_{suffix}", disabled=(page <= 0),
                         use_container_width=True):
                st.session_state["graphs_page"] = page - 1
                st.rerun()
        with n2:
            st.markdown(f"<div style='text-align:center;'>Page {page + 1} of "
                        f"{n_pages} · {total} stock(s) "
                        f"(≥ {int(min_strats)} strategies)</div>",
                        unsafe_allow_html=True)
        with n3:
            if st.button("Next ▶", key=f"next_{suffix}",
                         disabled=(page >= n_pages - 1),
                         use_container_width=True):
                st.session_state["graphs_page"] = page + 1
                st.rerun()

    _nav("top")
    st.markdown("---")

    start = page * per_page
    for _, crow in conv.iloc[start:start + per_page].iterrows():
        sym = crow["symbol"]
        sym_sigs = sel[sel["symbol"] == sym]
        st.markdown(f"### {sym}  ·  {int(crow['n_strategies'])} strategies")
        chips = []
        for _, sg in sym_sigs.iterrows():
            z = sg.get("zone_type", "")
            s = sg.get("strategy", sg.get("strategy_group", ""))
            color = ZONE_COLORS.get(z, "#666")
            chips.append(f'<span style="background:{color};color:white;'
                         f'padding:2px 8px;border-radius:10px;font-size:12px;'
                         f'margin-right:4px;">{s} · {z}</span>')
        st.markdown(" ".join(chips), unsafe_allow_html=True)
        ohlcv = load_parquet(["data", "ohlcv", f"{sym}.parquet"])
        if ohlcv.empty:
            st.caption("No OHLCV available for this symbol.")
        else:
            fig = build_stock_chart(sym, ohlcv, sym_sigs,
                                    TIMEFRAME_DAYS[tf], height=460)
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

    # Normalize Screener column names → our convention
    rename = {
        "ISIN": "isin",
        "Company name": "screener_name",
        "Rating": "screener_rating",
        "Last price": "screener_last_price",
        "Last price date": "screener_last_price_date",
        "1D Change (INR)": "one_d_change_inr",
        "1D Change (%)": "one_d_change_pct",
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
    tf = st.selectbox("Timeframe", list(TIMEFRAME_DAYS.keys()),
                      index=2, key="pf_tf")

    for _, h in pf.iterrows():
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

def main():
    
    st.sidebar.title("Signals India")
    
    page = st.sidebar.radio("Page",
                        ["Market Overview", "Today's Signals", "Graphs",
                         "My Portfolio", "Stock Detail", "Strategy Docs"])
    render_health_sidebar()
    st.sidebar.markdown("---")
    st.sidebar.caption(f"Loaded at {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    st.sidebar.caption("Data refreshes from Drive every 5 min (cache TTL)")
    if page == "Market Overview":
        page_market_overview()
    elif page == "Today's Signals":
        page_signals()
    elif page == "Stock Detail":
        page_stock_detail()
    elif page == "My Portfolio":
        page_portfolio()
    elif page == "Strategy Docs":
        page_strategy_docs()
    elif page == "Graphs":
        page_graphs()



main()
