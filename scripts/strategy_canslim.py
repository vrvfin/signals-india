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
from datetime import datetime
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload, MediaIoBaseUpload

SCOPES = ["https://www.googleapis.com/auth/drive"]


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


def evaluate(row) -> tuple[int, dict]:
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
                                and row["market_health_score"] >= 40),
    }
    return sum(1 for v in checks.values() if v), checks


def main():
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

    # Evaluate
    rows = []
    for _, r in merged.iterrows():
        passed, checks = evaluate(r)
        if passed < 5:
            continue
        zone = "buy" if passed == 7 else "hold"
        failed = [k for k, v in checks.items() if not v]
        rows.append({
            "symbol": r["symbol"],
            "date": r.get("date"),
            "strategy": "canslim",
            "zone_type": zone,
            "score": passed * 10 + (r.get("rs_rank_6m", 0) / 10
                                     if pd.notna(r.get("rs_rank_6m")) else 0),
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
    sig_df = pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)
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
