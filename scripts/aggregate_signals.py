"""
Stage 9 — Aggregator.

Combines all per-strategy `latest.csv` files into:

  signals/aggregated/latest.csv             — unified table, one row per
                                                (symbol, zone_type) with composite
                                                score and the strategies that agree
  signals/aggregated/conviction.csv         — Multi-Strategy Conviction:
                                                stocks flagged by ≥ 2 strategies
                                                with the same zone_type, sorted
                                                by number-of-strategies desc
  signals/aggregated/<date>.csv             — dated snapshot
  signals/aggregated/diff_vs_yesterday.csv  — NEW today / DROPPED today /
                                                MOVED rank — vs yesterday's file

Run after the strategy scripts:
    python scripts/strategy_momentum.py
    python scripts/strategy_ma_respect.py
    python scripts/strategy_qullamaggie.py
    python scripts/strategy_minervini.py
    python scripts/strategy_darvas.py
    python scripts/aggregate_signals.py
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


# ---------- Drive helpers ----------

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


def list_subfolders(drive, parent_id):
    q = (f"'{parent_id}' in parents and "
         f"mimeType='application/vnd.google-apps.folder' and trashed=false")
    return drive.files().list(q=q, fields="files(id,name)").execute().get("files", [])


def list_files_in_folder(drive, folder_id):
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


def download_csv(drive, file_id):
    request = drive.files().get_media(fileId=file_id)
    fh = io.BytesIO()
    downloader = MediaIoBaseDownload(fh, request)
    done = False
    while not done:
        _, done = downloader.next_chunk()
    fh.seek(0)
    return pd.read_csv(fh)


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


# ---------- Core aggregation ----------

def load_all_strategy_signals(drive, folder_id, ref_date=None):
    """Read every per_strategy/<NAME>/latest.csv and union into a single DataFrame.

    Ghost-signal guard: a strategy that crashed upstream leaves YESTERDAY's
    latest.csv behind, which would silently be aggregated as today's signals.
    Any file whose newest signal date is older than `ref_date` (the features
    bar date this run is built on) is skipped, loudly."""
    signals_id = get_or_create_subfolder(drive, folder_id, "signals")
    per_strategy_id = get_or_create_subfolder(drive, signals_id, "per_strategy")
    subfolders = list_subfolders(drive, per_strategy_id)
    log(f"Found {len(subfolders)} strategy folders")

    frames, skipped = [], []
    for sub in subfolders:
        files = list_files_in_folder(drive, sub["id"])
        latest_id = files.get("latest.csv")
        if not latest_id:
            continue
        try:
            df = download_csv(drive, latest_id)
        except pd.errors.EmptyDataError:
            log(f"  {sub['name']:<28}  SKIPPED — empty/corrupt latest.csv")
            continue
        if df.empty:
            log(f"  {sub['name']:<28}      0 signals (empty — ok)")
            continue
        if ref_date is not None and "date" in df.columns:
            sig_date = pd.to_datetime(df["date"], errors="coerce").max()
            if pd.notna(sig_date) and sig_date < ref_date:
                skipped.append(sub["name"])
                log(f"  {sub['name']:<28}  SKIPPED — stale "
                    f"(signals dated {sig_date.date()}, features at "
                    f"{pd.Timestamp(ref_date).date()})")
                continue
        df["strategy_group"] = sub["name"]
        if "strategy" not in df.columns:
            df["strategy"] = sub["name"]
        frames.append(df)
        log(f"  {sub['name']:<28}  {len(df):>5} signals")
    if skipped:
        log(f"STALE strategies excluded from today's aggregation: {skipped}")

    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True, sort=False)
    return combined


def compute_unified(signals: pd.DataFrame) -> pd.DataFrame:
    """One row per (symbol, zone_type) with composite score + agreeing strategies.

    Scores are percentile-normalized WITHIN each strategy first (0-100): raw
    score units differ wildly per strategy (RS rank 0-100, streak DAYS, raw 3m
    return %, distance-to-high...) so a raw mean is dominated by whichever
    strategy uses big numbers. After normalization, 90 means "top decile of
    that strategy's signals today" for every strategy alike."""
    keep_cols = ["symbol", "zone_type", "score", "entry", "stop", "strategy", "reason"]
    keep = [c for c in keep_cols if c in signals.columns]
    df = signals[keep].copy()
    df["score_norm"] = (df.groupby("strategy")["score"]
                          .rank(pct=True) * 100).round(1)

    grouped = df.groupby(["symbol", "zone_type"], dropna=False)
    rows = []
    for (sym, zone), g in grouped:
        rows.append({
            "symbol": sym,
            "zone_type": zone,
            # unique strategies, not rows — duplicate rows from one strategy
            # must not masquerade as multi-strategy agreement
            "n_strategies": int(g["strategy"].nunique()),
            "strategies": ", ".join(sorted(g["strategy"].unique())),
            "composite_score": float(g["score_norm"].mean()),
            "max_score": float(g["score_norm"].max()),
            "composite_score_raw": float(g["score"].mean()),
            "entry_median": float(g["entry"].median()) if "entry" in g.columns else None,
            "stop_median": float(g["stop"].median()) if "stop" in g.columns else None,
            "reasons": " || ".join(
                f"[{row['strategy']}] {row['reason']}" for _, row in g.iterrows()
                if "reason" in row and pd.notna(row.get("reason"))
            )[:1000],
        })
    out = pd.DataFrame(rows)
    return out.sort_values(["n_strategies", "composite_score"],
                           ascending=[False, False]).reset_index(drop=True)


def compute_conviction(unified: pd.DataFrame, min_strategies: int = 2) -> pd.DataFrame:
    """Stocks flagged by >= min_strategies with the same zone_type."""
    return unified[unified["n_strategies"] >= min_strategies].copy().reset_index(drop=True)


def compute_diff(today: pd.DataFrame, yday: pd.DataFrame) -> pd.DataFrame:
    """Compare today's unified vs yesterday's. Returns long-format diff."""
    if yday.empty:
        # First run — everything is NEW
        rows = [{"symbol": r["symbol"], "zone_type": r["zone_type"],
                 "change": "NEW", "today_score": r["composite_score"],
                 "yday_score": None,
                 "today_n": r["n_strategies"], "yday_n": None}
                for _, r in today.iterrows()]
        return pd.DataFrame(rows)

    today_keyed = today.set_index(["symbol", "zone_type"])
    yday_keyed = yday.set_index(["symbol", "zone_type"])

    today_keys = set(today_keyed.index)
    yday_keys = set(yday_keyed.index)
    new_keys = today_keys - yday_keys
    dropped_keys = yday_keys - today_keys
    common_keys = today_keys & yday_keys

    rows = []
    for key in new_keys:
        r = today_keyed.loc[key]
        rows.append({"symbol": key[0], "zone_type": key[1], "change": "NEW",
                     "today_score": float(r["composite_score"]),
                     "yday_score": None,
                     "today_n": int(r["n_strategies"]), "yday_n": None})
    for key in dropped_keys:
        r = yday_keyed.loc[key]
        rows.append({"symbol": key[0], "zone_type": key[1], "change": "DROPPED",
                     "today_score": None,
                     "yday_score": float(r["composite_score"]),
                     "today_n": None, "yday_n": int(r["n_strategies"])})
    for key in common_keys:
        rt, ry = today_keyed.loc[key], yday_keyed.loc[key]
        if int(rt["n_strategies"]) != int(ry["n_strategies"]):
            change = ("MORE_STRATEGIES" if rt["n_strategies"] > ry["n_strategies"]
                      else "FEWER_STRATEGIES")
            rows.append({"symbol": key[0], "zone_type": key[1], "change": change,
                         "today_score": float(rt["composite_score"]),
                         "yday_score": float(ry["composite_score"]),
                         "today_n": int(rt["n_strategies"]),
                         "yday_n": int(ry["n_strategies"])})

    df_diff = pd.DataFrame(rows)
    if df_diff.empty:
        return df_diff
    return df_diff.sort_values(
        ["change", "today_n"], ascending=[True, False]).reset_index(drop=True)

    


# ---------- Main ----------

def main():
    print("Stage 9 — Aggregator + Multi-Strategy Conviction")
    print("-" * 60)

    drive = get_drive()
    folder_id = os.environ["GDRIVE_FOLDER_ID"]

    # Reference date = the bar date this run's features were computed on.
    # Strategy files older than this are yesterday's leftovers (ghost signals).
    ref_date = None
    feat_id = get_or_create_subfolder(drive, folder_id, "features")
    latest_feat = find_file(drive, feat_id, "latest.parquet")
    if latest_feat:
        try:
            fdf = download_parquet(drive, latest_feat)
            ref_date = pd.to_datetime(fdf["date"], errors="coerce").max()
            log(f"Reference signal date (features): {ref_date.date()}")
        except Exception as e:
            log(f"WARNING: could not read features date ({str(e)[:60]}) — "
                f"stale-strategy guard disabled this run")

    # 1. Load all strategy signals
    signals = load_all_strategy_signals(drive, folder_id, ref_date=ref_date)
    if signals.empty:
        print("No signals found. Run strategy scripts first.")
        return
    log(f"Total per-strategy signals loaded: {len(signals)}")

    # 2. Unified view
    unified = compute_unified(signals)
    log(f"Unified rows (one per (symbol, zone)): {len(unified)}")

    # 3. Conviction list
    conviction = compute_conviction(unified, min_strategies=2)
    log(f"Multi-Strategy Conviction (>=2 strategies agree): {len(conviction)} rows")

    # 4. Diff vs yesterday
    agg_id = get_or_create_subfolder(drive, folder_id, "signals")
    agg_id = get_or_create_subfolder(drive, agg_id, "aggregated")

    yday_id = find_file(drive, agg_id, "latest.csv")
    yday_unified = download_csv(drive, yday_id) if yday_id else pd.DataFrame()
    diff = compute_diff(unified, yday_unified)
    log(f"Diff vs yesterday: {len(diff)} change rows")
    if not diff.empty and "change" in diff.columns:
        print()
        print("  Change breakdown:")
        print(diff["change"].value_counts().to_string())

    # 5. Write all outputs. The dated snapshot is named for the SESSION it
    # describes (ref_date = the features bar date), not the runner's UTC wall
    # clock. build_signal_membership.py derives per-symbol tenure from these
    # filenames, so a mislabelled snapshot silently corrupts "days on list".
    if ref_date is not None and pd.notna(ref_date):
        today_str = pd.Timestamp(ref_date).strftime("%Y-%m-%d")
        log(f"dated snapshot named for bar date {today_str}")
    else:
        today_str = datetime.now().strftime("%Y-%m-%d")
        log(f"WARNING: no features bar date — dated snapshot falls back to "
            f"wall clock {today_str}")
    upload_csv(drive, agg_id, "latest.csv", unified,
               find_file(drive, agg_id, "latest.csv"))
    upload_csv(drive, agg_id, f"{today_str}.csv", unified,
               find_file(drive, agg_id, f"{today_str}.csv"))
    upload_csv(drive, agg_id, "conviction.csv", conviction,
               find_file(drive, agg_id, "conviction.csv"))
    upload_csv(drive, agg_id, "diff_vs_yesterday.csv", diff,
               find_file(drive, agg_id, "diff_vs_yesterday.csv"))
    log("Wrote: latest.csv, dated snapshot, conviction.csv, diff_vs_yesterday.csv")

    # 6. Summary
    print()
    print("-" * 60)
    if not conviction.empty:
        print(f"Top 10 Multi-Strategy Conviction names:")
        show = ["symbol", "zone_type", "n_strategies", "composite_score", "strategies"]
        print(conviction.head(10)[show].to_string(index=False))


if __name__ == "__main__":
    main()
