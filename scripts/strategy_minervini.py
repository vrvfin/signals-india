"""
Stage 6b — Mark Minervini SEPA / Trend Template.

Eight-point trend filter (Mark Minervini, "Trade Like a Stock Market Wizard").
v1 uses features only — VCP-pattern breakout detection is deferred.

Trend Template (we use ema_100 as a stand-in for 150 SMA — exact 150 SMA can be
added to compute_features.py later if you want strict parity):

  1.  Price > sma_200 AND > ema_100 (our 150 SMA proxy)
  2.  ema_100 > sma_200
  3.  sma_200 trending up (200sma_rising flag)
  4.  sma_50 > ema_100 > sma_200 (stacked)
  5.  Price > sma_50
  6.  Price ≥ 30% above 52-week low (dist_from_52w_low_pct >= 30)
  7.  Price within 25% of 52-week high (dist_from_52w_high_pct >= -25)
  8.  RS rank (6-month) ≥ 70

Zones:
  buy  — passes all 8
  hold — passes 6 or 7 of 8 (developing setup; close to template)

Outputs:
  signals/per_strategy/minervini/<date>.csv
  signals/per_strategy/minervini/latest.csv
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

# Written on zero-signal days so latest.csv stays fresh (healthcheck treats a
# stale latest.csv as CRITICAL; the aggregator skips empty files cleanly).
_EMPTY_SIG_COLS = ["symbol", "date", "strategy", "zone_type", "score",
                   "entry", "stop", "reason"]


# Console encoding. Several scripts here log the rupee sign, a delta or an em
# dash, and a Windows console is cp1252 — so a run could complete all its work
# and then die in a log line. It cost three separate crashes before being fixed
# in one place. Degrade the characters, never the run.
try:
    sys.stdout.reconfigure(errors="replace")
    sys.stderr.reconfigure(errors="replace")
except Exception:          # pragma: no cover - not every stream supports it
    pass

def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")


def get_drive():
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
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
        token_path.parent.mkdir(parents=True, exist_ok=True)
        token_path.write_text(creds.to_json())
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
    downloader = MediaIoBaseDownload(fh, request)
    done = False
    while not done:
        _, done = downloader.next_chunk()
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


# ---------- Strategy ----------

def evaluate_trend_template(row: pd.Series) -> tuple[int, dict]:
    """Returns (passed_count, breakdown_dict)."""
    close = row["close"]
    sma50 = row["sma_50"]
    sma200 = row["sma_200"]
    ema100 = row["ema_100"]  # 150-SMA proxy
    dist_high = row["dist_from_52w_high_pct"]
    dist_low = row["dist_from_52w_low_pct"]
    rs_rank = row.get("rs_rank_6m")
    rising = row.get("200sma_rising", False)

    checks = {
        "1_price_above_sma200_and_ema100": (
            pd.notna(sma200) and pd.notna(ema100) and close > sma200 and close > ema100),
        "2_ema100_above_sma200": (
            pd.notna(sma200) and pd.notna(ema100) and ema100 > sma200),
        "3_sma200_rising": bool(rising),
        "4_50_above_100_above_200": (
            pd.notna(sma50) and pd.notna(ema100) and pd.notna(sma200)
            and sma50 > ema100 > sma200),
        "5_price_above_sma50": (pd.notna(sma50) and close > sma50),
        "6_30pct_above_52w_low": (pd.notna(dist_low) and dist_low >= 30),
        "7_within_25pct_of_52w_high": (pd.notna(dist_high) and dist_high >= -25),
        "8_rs_rank_6m_ge_70": (pd.notna(rs_rank) and rs_rank >= 70),
    }
    passed = sum(1 for v in checks.values() if v)
    return passed, checks


def template_slacks(row: pd.Series) -> dict:
    """How much head-room the stock has on each CONTINUOUS template condition.

    Rule 3 (200 SMA rising) is boolean and has no slack, so it stays a pure gate.
    The `full` values are the point past each threshold where more stops meaning
    materially safer — judgements, stated here rather than buried in a formula."""
    close = row["close"]
    return {
        # rule 8: RS rank 70 floor, rank 100 is full slack
        "rs":    slack(row.get("rs_rank_6m"), 70, 30),
        # rule 7: within 25% of the 52w high; AT the high is full slack
        "high":  slack(row.get("dist_from_52w_high_pct"), -25, 25),
        # rule 6: 30% above the 52w low; +60% is full slack
        "low":   slack(row.get("dist_from_52w_low_pct"), 30, 30),
        # rule 5: above the 50 SMA; +10% is full slack
        "ma50":  slack(close / row["sma_50"] - 1 if row.get("sma_50") else None,
                       0, 0.10),
        # rule 1: above the 200 SMA; +30% is full slack
        "ma200": slack(close / row["sma_200"] - 1 if row.get("sma_200") else None,
                       0, 0.30),
        # rule 4: 50 SMA above the 100 EMA; +5% is full slack
        "stack": slack(row["sma_50"] / row["ema_100"] - 1
                       if row.get("sma_50") and row.get("ema_100") else None,
                       0, 0.05),
        # rule 2: 100 EMA above the 200 SMA; +5% is full slack
        "e100":  slack(row["ema_100"] / row["sma_200"] - 1
                       if row.get("ema_100") and row.get("sma_200") else None,
                       0, 0.05),
    }


def minervini_signals(features: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in features.iterrows():
        passed, checks = evaluate_trend_template(r)
        # 8/8 ONLY. The old 6-or-7 "hold" tier was producing most of this
        # strategy's ~1,185 daily signals, and a 6-of-8 Minervini is not a
        # Minervini — the template is a conjunction, not a scorecard.
        if passed < 8:
            continue
        zone = "buy"
        rs_rank = r.get("rs_rank_6m", float("nan"))
        atr = r.get("atr_14", 0)
        rules_passed = [k for k, v in checks.items() if v]
        rules_failed = [k for k, v in checks.items() if not v]
        sl = template_slacks(r)
        margin, binding = min_slack_score(sl)
        rows.append({
            "symbol": r["symbol"],
            "date": r["date"],
            "strategy": "minervini",
            "zone_type": zone,
            # Distance to failing the template, not boxes-ticked-times-ten. Every
            # signal here is 8/8, so a rule count would now be a constant.
            "score": margin,
            "template_margin": margin,
            "binding_rule": binding,
            "entry": round(r["close"], 2),
            "stop": round(r["close"] - 2 * atr, 2) if atr else None,
            "rules_passed": passed,
            "rules_failed": ", ".join(rules_failed) if rules_failed else "",
            "rs_rank_6m": round(rs_rank, 1) if pd.notna(rs_rank) else None,
            "dist_from_52w_high_pct": round(r["dist_from_52w_high_pct"], 1),
            "dist_from_52w_low_pct": round(r["dist_from_52w_low_pct"], 1),
            "return_6m_pct": round(r["return_6m_pct"], 1) if pd.notna(r["return_6m_pct"]) else None,
            "reason": (f"Trend template 8/8; tightest condition '{binding}' "
                       f"at {margin:.0f}% of full slack"),
        })
    return pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)


def main() -> None:
    global DRY_RUN
    DRY_RUN = _parse_dry_run()
    if DRY_RUN:
        log("DRY RUN — no Drive writes will be made")
    print("Stage 6b — Minervini SEPA / Trend Template")
    print("-" * 50)

    drive = get_drive()
    folder_id = os.environ["GDRIVE_FOLDER_ID"]
    features_id = get_or_create_subfolder(drive, folder_id, "features")
    latest_id = find_file(drive, features_id, "latest.parquet")
    if not latest_id:
        print("features/latest.parquet missing — run compute_features.py first.")
        return
    features = download_parquet(drive, latest_id)
    log(f"Features loaded: {len(features)} symbols")

    sig_df = minervini_signals(features)
    if sig_df.empty:
        print("No stocks pass 6+ of Minervini's 8 trend-template rules.")
        signals_id = get_or_create_subfolder(drive, folder_id, "signals")
        per_strategy_id = get_or_create_subfolder(drive, signals_id, "per_strategy")
        mv_id = get_or_create_subfolder(drive, per_strategy_id, "minervini")
        upload_csv(drive, mv_id, "latest.csv",
                   pd.DataFrame(columns=_EMPTY_SIG_COLS),
                   find_file(drive, mv_id, "latest.csv"))
        return

    # Save
    signals_id = get_or_create_subfolder(drive, folder_id, "signals")
    per_strategy_id = get_or_create_subfolder(drive, signals_id, "per_strategy")
    mv_id = get_or_create_subfolder(drive, per_strategy_id, "minervini")
    today_str = datetime.now().strftime("%Y-%m-%d")
    upload_csv(drive, mv_id, f"{today_str}.csv", sig_df,
               find_file(drive, mv_id, f"{today_str}.csv"))
    upload_csv(drive, mv_id, "latest.csv", sig_df,
               find_file(drive, mv_id, "latest.csv"))

    n_buy = (sig_df["zone_type"] == "buy").sum()
    n_hold = (sig_df["zone_type"] == "hold").sum()
    print()
    print(f"BUY  (passes all 8)   : {n_buy}")
    print(f"HOLD (passes 6-7 of 8): {n_hold}")
    print("\nTop 10 (BUY first, then HOLD):")
    show = ["symbol", "zone_type", "rules_passed", "rs_rank_6m",
            "return_6m_pct", "dist_from_52w_high_pct", "rules_failed"]
    print(sig_df.head(10)[show].to_string(index=False))


if __name__ == "__main__":
    main()
