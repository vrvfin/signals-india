"""
fetch_phase1_report.py — Generate a local HTML report from Phase 1 outputs.

Two sections:
  1. Conviction Signals  — stocks flagged by ≥ MIN_STRATEGIES strategies
                           in buy/add zones, sorted by conviction
  2. My Portfolio        — holdings overlaid with RS rank, signals, features

Opens in your default browser. Print to PDF via Ctrl+P → Save as PDF.

Usage:
    python scripts/fetch_phase1_report.py                  # default (min 2 strategies)
    python scripts/fetch_phase1_report.py --min-strats 3   # stricter filter
    python scripts/fetch_phase1_report.py --no-portfolio   # signals only
    python scripts/fetch_phase1_report.py --no-open        # save only, don't open
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


# ── HTML generation ───────────────────────────────────────────────────────────

_CSS = """
* { box-sizing: border-box; margin: 0; padding: 0; }
body {
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Arial, sans-serif;
  font-size: 13px; color: #1a1a2e; background: #f5f6fa;
  padding: 24px 32px;
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
th {
  background: #2c3e50; color: #fff; padding: 8px 10px;
  text-align: left; font-size: 12px; font-weight: 600;
  white-space: nowrap;
}
td { padding: 7px 10px; border-bottom: 1px solid #eef0f4; vertical-align: top; }
tr:last-child td { border-bottom: none; }
tr:nth-child(even) td { background: #f8f9fc; }
tr:hover td { background: #eaf4fb; }
.sym { font-weight: 700; color: #2c3e50; font-size: 13px; }
.num { text-align: right; font-variant-numeric: tabular-nums; }
.green  { color: #1a9e3f; font-weight: 600; }
.red    { color: #c0392b; font-weight: 600; }
.badge {
  display: inline-block; padding: 2px 8px; border-radius: 10px;
  font-size: 11px; font-weight: 600; color: #fff; margin: 1px 2px;
  white-space: nowrap;
}
.stars { color: #f39c12; }
.na { color: #aaa; }
@media print {
  body { background: #fff; padding: 12px; }
  table { box-shadow: none; }
  h2 { break-before: avoid; }
  tr { break-inside: avoid; }
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
Print to PDF: Ctrl+P → Save as PDF (use Landscape for wide tables)</p>

{conviction_section}

{portfolio_section}
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
                              min_strats: int) -> str:
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
    return title + table


def build_portfolio_section(portfolio: pd.DataFrame,
                             signals: pd.DataFrame,
                             features: pd.DataFrame,
                             universe: pd.DataFrame) -> str:
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
    return title + table


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--min-strats",   type=int, default=2,
                        help="Minimum strategies for conviction section (default 2)")
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

    conviction_html = build_conviction_section(signals, features, args.min_strats)
    portfolio_html  = build_portfolio_section(portfolio, signals, features, universe)

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
