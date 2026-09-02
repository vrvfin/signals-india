"""
Stage 11b — CANSLIM strategy.

Rules:
  C  Current quarter EPS YoY growth >= 25%
  A  Annual EPS YoY growth >= 25%
  N  Stock within 25% of 52-week high (proxy for 'new high / new product')
  S  Promoter holding ≥ 40%  (proxy for 'supply' — low float / strong inside ownership)
  L  RS rank (6M) >= 80
  I  ROE >= 15%  (proxy for institutional appeal — efficient capital deployment)
  M  Market Health Score >= 40  (the 'M' = market direction filter)

Zones:
  buy   — passes all 7
  hold  — passes 5 or 6

Outputs:
  signals/per_strategy/canslim/<date>.csv
  signals/per_strategy/canslim/latest.csv

Usage:
    python scripts/strategy_canslim.py
"""

from __future__ import annotations

import io
import os
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

_SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)
from strategy_common import slack, min_slack_score
from dotenv import load_dotenv
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload, MediaIoBaseUpload

SCOPES = ["https://www.googleapis.com/auth/drive"]


# Console encoding. Several scripts here log the rupee sign, a delta or an em
# dash, and a Windows console is cp1252 — so a run could complete all its work
# and then die in a log line. It cost three separate crashes before being fixed
# in one place. Degrade the characters, never the run.
try:
    sys.stdout.reconfigure(errors="replace")
    sys.stderr.reconfigure(errors="replace")
except Exception:          # pragma: no cover - not every stream supports it
    pass

def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")


def get_drive():
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
    cs_path = Path(os.environ["GDRIVE_OAUTH_CLIENT_SECRET_PATH"])
    tk_path = Path(os.environ["GDRIVE_OAUTH_TOKEN_PATH"])
    creds = None
    if tk_path.exists():
        creds = Credentials.from_authorized_user_file(str(tk_path), SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(str(cs_path), SCOPES)
            creds = flow.run_local_server(port=0)
        tk_path.write_text(creds.to_json())
    return build("drive", "v3", credentials=creds, cache_discovery=False)


def get_or_create_subfolder(drive, parent_id, name):
    q = (f"name='{name}' and '{parent_id}' in parents "
         f"and mimeType='application/vnd.google-apps.folder' and trashed=false")
    found = drive.files().list(q=q, fields="files(id)").execute().get("files", [])
    if found:
        return found[0]["id"]
    meta = {"name": name, "parents": [parent_id],
            "mimeType": "application/vnd.google-apps.folder"}
    return drive.files().create(body=meta, fields="id").execute()["id"]


def find_file(drive, folder_id, name):
    q = f"name='{name}' and '{folder_id}' in parents and trashed=false"
    found = drive.files().list(q=q, fields="files(id)").execute().get("files", [])
    return found[0]["id"] if found else None


def download_parquet(drive, file_id):
    request = drive.files().get_media(fileId=file_id)
    fh = io.BytesIO()
    d = MediaIoBaseDownload(fh, request)
    done = False
    while not done:
        _, done = d.next_chunk()
    fh.seek(0)
    return pd.read_parquet(fh)


def upload_csv(drive, folder_id, filename, df, existing_id=None):
    media = MediaIoBaseUpload(io.BytesIO(df.to_csv(index=False).encode()),
                              mimetype="text/csv", resumable=False)
    if existing_id:
        drive.files().update(fileId=existing_id, media_body=media).execute()
        return existing_id
    meta = {"name": filename, "parents": [folder_id]}
    return drive.files().create(body=meta, media_body=media, fields="id").execute()["id"]


# ── --dry-run guard ──────────────────────────────────────────────────────────
# This script writes signal files straight to Drive. Its scoring logic changed,
# and house rule 6 says a dry-run must be confirmed before any live run — so the
# upload is wrapped rather than the call sites edited. Set by _parse_dry_run().
DRY_RUN = False
_live_upload_csv = upload_csv


def upload_csv(drive, folder_id, filename, df, existing_id=None):   # noqa: F811
    if DRY_RUN:
        log(f"[DRY RUN] would write {filename} ({len(df)} rows)")
        return existing_id
    return _live_upload_csv(drive, folder_id, filename, df, existing_id)


def _parse_dry_run() -> bool:
    """--dry-run only; kept deliberately separate from any argparse the script
    already has, so adding it cannot disturb existing flags."""
    import argparse as _ap
    p = _ap.ArgumentParser(add_help=False)
    p.add_argument("--dry-run", action="store_true",
                   help="compute and report, write nothing to Drive")
    known, _ = p.parse_known_args()
    return known.dry_run


# "M" is one number for the WHOLE market on a given day, so counting it as a
# seventh per-stock rule added zero cross-sectional information — it just shifted
# every stock's score by the same 10 points, and on a day it flipped, every
# CANSLIM name silently dropped a tier at once. It is a gate on the day, not a
# property of the stock.
M_MARKET_HEALTH_MIN = 40

# Rules that describe the STOCK. C and A are the CANSLIM thesis (current and
# annual earnings acceleration); the old 5-of-7 "hold" tier let a stock fail both
# and still signal, which is not CANSLIM in any meaningful sense.
STOCK_RULES = ["C_qtr_eps_growth_25pct", "A_ann_eps_growth_25pct",
               "N_within_25pct_52w_high", "S_promoter_holding_40pct",
               "L_rs_rank_6m_80", "I_roe_15pct"]
REQUIRED_RULES = ["C_qtr_eps_growth_25pct", "A_ann_eps_growth_25pct"]


def evaluate(row) -> tuple[int, dict]:
    """Returns (count of STOCK rules passed, all checks incl. M for reporting)."""
    checks = {
        "C_qtr_eps_growth_25pct": (pd.notna(row.get("q_eps_yoy_pct"))
                                    and row["q_eps_yoy_pct"] >= 25),
        "A_ann_eps_growth_25pct": (pd.notna(row.get("ann_eps_yoy_pct"))
                                    and row["ann_eps_yoy_pct"] >= 25),
        "N_within_25pct_52w_high": (pd.notna(row.get("dist_from_52w_high_pct"))
                                     and row["dist_from_52w_high_pct"] >= -25),
        "S_promoter_holding_40pct": (pd.notna(row.get("promoter_holding_pct"))
                                      and row["promoter_holding_pct"] >= 40),
        "L_rs_rank_6m_80": (pd.notna(row.get("rs_rank_6m"))
                             and row["rs_rank_6m"] >= 80),
        "I_roe_15pct": (pd.notna(row.get("roe_pct"))
                         and row["roe_pct"] >= 15),
        "M_market_health_40": (pd.notna(row.get("market_health_score"))
                                and row["market_health_score"] >= M_MARKET_HEALTH_MIN),
    }
    return sum(1 for k in STOCK_RULES if checks[k]), checks


def canslim_slacks(row) -> dict:
    """Head-room on each stock rule. `full` = where more stops meaning safer.

    NOTE on S: OWN_PROMHOLD measures -0.7pp at 3M and -7.0pp at 12M in
    guru/backtest/family_lift.parquet, while PROJECT_STATUS.md section 3 reports
    high promoter holding as a winner characteristic. Those are different
    measurements (timing lift vs winner profiling) and they disagree. The rule is
    left EXACTLY as it was pending that reconciliation — flagged, not changed."""
    return {
        "C": slack(row.get("q_eps_yoy_pct"), 25, 50),
        "A": slack(row.get("ann_eps_yoy_pct"), 25, 50),
        "N": slack(row.get("dist_from_52w_high_pct"), -25, 25),
        "S": slack(row.get("promoter_holding_pct"), 40, 35),
        "L": slack(row.get("rs_rank_6m"), 80, 20),
        "I": slack(row.get("roe_pct"), 15, 25),
    }


def main():
    global DRY_RUN
    DRY_RUN = _parse_dry_run()
    if DRY_RUN:
        log("DRY RUN — no Drive writes will be made")
    print("Stage 11b — CANSLIM signals")
    print("-" * 50)
    drive = get_drive()
    folder_id = os.environ["GDRIVE_FOLDER_ID"]

    # Load inputs: features + fundamentals + market state
    features_id = get_or_create_subfolder(drive, folder_id, "features")
    feat = download_parquet(drive, find_file(drive, features_id, "latest.parquet"))
    log(f"Features loaded: {len(feat)}")

    fund_id = get_or_create_subfolder(drive, folder_id, "fundamentals")
    summary_fid = find_file(drive, fund_id, "summary.parquet")
    if not summary_fid:
        print("fundamentals/summary.parquet missing. Run `ingest_fundamentals.py` first.")
        return
    fund = download_parquet(drive, summary_fid)
    log(f"Fundamentals loaded: {len(fund)}")

    data_id = get_or_create_subfolder(drive, folder_id, "data")
    ms_id = get_or_create_subfolder(drive, data_id, "market_state")
    ms_fid = find_file(drive, ms_id, "latest.parquet")
    market_health = None
    if ms_fid:
        ms = download_parquet(drive, ms_fid)
        if not ms.empty and "health_score" in ms.columns:
            market_health = float(ms["health_score"].iloc[0])
    log(f"Market Health Score: {market_health}")

    # Merge
    merged = feat.merge(fund, on="symbol", how="inner")
    merged["market_health_score"] = market_health
    log(f"Universe with fundamentals: {len(merged)}")

    # M is a gate on the DAY, not a rule on the stock. Below the threshold
    # CANSLIM emits nothing at all rather than quietly demoting every name.
    if market_health is None or market_health < M_MARKET_HEALTH_MIN:
        log(f"M gate: market health {market_health} < {M_MARKET_HEALTH_MIN} "
            f"— CANSLIM stands down today (0 signals)")
        merged = merged.iloc[0:0]

    # Evaluate
    rows = []
    for _, r in merged.iterrows():
        passed, checks = evaluate(r)
        # C and A are the thesis. Without earnings acceleration this is not a
        # CANSLIM setup, however many of the other rules happen to pass.
        if not all(checks[k] for k in REQUIRED_RULES):
            continue
        if passed < 5:
            continue
        zone = "buy" if passed == len(STOCK_RULES) else "hold"
        sl = canslim_slacks(r)
        margin, binding = min_slack_score(sl)
        failed = [k for k, v in checks.items() if not v]
        rows.append({
            "symbol": r["symbol"],
            "date": r.get("date"),
            "strategy": "canslim",
            "zone_type": zone,
            # Distance to failing the tightest satisfied rule, not boxes-ticked
            # times ten (which capped RS's influence at 10 points and made a
            # 5-of-6 unable to ever outrank a 6-of-6).
            "score": margin,
            "canslim_margin": margin,
            "binding_rule": binding,
            "entry": round(r["close"], 2) if pd.notna(r.get("close")) else None,
            "stop": (round(r["close"] - 2 * r["atr_14"], 2)
                     if pd.notna(r.get("atr_14")) else None),
            "rules_passed": passed,
            "rules_failed": ", ".join(failed),
            "q_eps_yoy_pct": r.get("q_eps_yoy_pct"),
            "ann_eps_yoy_pct": r.get("ann_eps_yoy_pct"),
            "promoter_holding_pct": r.get("promoter_holding_pct"),
            "roe_pct": r.get("roe_pct"),
            "pe": r.get("pe"),
            "rs_rank_6m": r.get("rs_rank_6m"),
            "reason": f"CANSLIM {passed}/7" + (
                f" — missing: {', '.join(failed)}" if failed else ""),
        })
    sig_df = pd.DataFrame(rows)
    if sig_df.empty:
        # Zero-signal day: still write a HEADER so latest.csv is parseable.
        # A column-less pd.DataFrame([]) writes a 1-byte file that crashes every
        # reader (build_gallery._load_signals, aggregate_signals) with EmptyDataError.
        sig_df = pd.DataFrame(columns=["symbol", "date", "strategy", "zone_type",
                                       "score", "entry", "stop", "reason"])
    else:
        sig_df = sig_df.sort_values("score", ascending=False).reset_index(drop=True)
    log(f"CANSLIM signals: {len(sig_df)} (BUY={(sig_df['zone_type']=='buy').sum()})")

    # Save
    signals_id = get_or_create_subfolder(drive, folder_id, "signals")
    per_strat_id = get_or_create_subfolder(drive, signals_id, "per_strategy")
    cs_id = get_or_create_subfolder(drive, per_strat_id, "canslim")
    today = datetime.now().strftime("%Y-%m-%d")
    upload_csv(drive, cs_id, f"{today}.csv", sig_df,
               find_file(drive, cs_id, f"{today}.csv"))
    upload_csv(drive, cs_id, "latest.csv", sig_df,
               find_file(drive, cs_id, "latest.csv"))
    log("Saved signals/per_strategy/canslim/")

    if not sig_df.empty:
        print("\nTop 10:")
        show = ["symbol", "zone_type", "rules_passed", "rs_rank_6m",
                "q_eps_yoy_pct", "ann_eps_yoy_pct", "pe"]
        show = [c for c in show if c in sig_df.columns]
        print(sig_df.head(10)[show].to_string(index=False))


if __name__ == "__main__":
    main()
