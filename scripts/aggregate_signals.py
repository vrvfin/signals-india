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

import argparse
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


# ── families, and which of them mark an EVENT ────────────────────────────────
#
# `n_strategies` counted momentum FIVE times: momentum_1m/2m/3m/6m/12m are five
# folders, but they are five views of one idea and are ~90% the same names, so a
# stock in a 12-month uptrend collected five "agreeing strategies" for free. On
# 2026-08-28 the momentum family alone supplied 3,260 of 6,854 signals (48%).
# Since n_strategies is the primary sort key, the top of the conviction list was
# structurally momentum-biased.
#
# darvas and qullamaggie stay SEPARATE by explicit user decision (2026-09-01):
# darvas requires the stock within 5% of its 52w high, qullamaggie allows one
# further past its high, so they are not simply variants of each other.
# volume_breakout and volume_vcp are genuine opposites (expansion vs dry-up).

def family_of(strategy: str, collapse_ma_respect: bool = False) -> str:
    s = str(strategy)
    if s.startswith("momentum_"):
        return "momentum"
    # ma_respect_20ema_30d / _20ema_60d / _50ema_60d are three variants of one
    # idea, structurally identical to momentum's five — but collapsing them was
    # NOT signed off, so current behaviour is preserved unless asked for.
    if collapse_ma_respect and s.startswith("ma_respect_"):
        return "ma_respect"
    return s


def is_event(family: str, zone: str) -> bool:
    """Did something HAPPEN today, or is this a state that has been true for weeks?

    Six of the nine strategies describe a STATE — in an uptrend, above an EMA,
    high RS, passes a template. Those stay true for months and re-fire every
    single day, which is why the conviction list reached 1,762 of ~4,100
    tradeable names. Only a handful mark an event on the day, and only an event
    can time an entry.
    """
    z = str(zone)
    if family in ("darvas", "qullamaggie"):
        return z == "add"          # broke the box/pivot TODAY
    if family == "volume_breakout":
        return z in ("buy", "add")  # the volume spike IS today
    if family == "pead":
        return z == "add"          # within 5 sessions of the result
    return False


def momentum_profile(variants: set) -> str:
    """Which momentum lookbacks fired — a name unique to 1m is a fresh mover, a
    name in all five is an established leader, and they are not the same trade.
    Collapsing to one vote must not throw that away (user, 2026-09-01)."""
    v = {str(x) for x in variants if x and str(x) != "nan"}
    if not v:
        return ""
    if len(v) >= 5:
        return "persistent"
    if v <= {"1m", "2m"}:
        return "fresh"
    if v <= {"12m"}:
        return "stale"
    return "broad"


def compute_unified(signals: pd.DataFrame,
                    collapse_ma_respect: bool = False) -> pd.DataFrame:
    """One row per (symbol, zone_type) with composite score + agreeing families.

    Scores are percentile-normalized WITHIN each strategy first (0-100): raw
    score units differ wildly per strategy (RS rank 0-100, streak DAYS, raw 3m
    return %, distance-to-high...) so a raw mean is dominated by whichever
    strategy uses big numbers. After normalization, 90 means "top decile of
    that strategy's signals today" for every strategy alike."""
    keep_cols = ["symbol", "zone_type", "score", "entry", "stop", "strategy",
                 "reason", "variant"]
    keep = [c for c in keep_cols if c in signals.columns]
    df = signals[keep].copy()
    df["score_norm"] = (df.groupby("strategy")["score"]
                          .rank(pct=True) * 100).round(1)
    df["family"] = df["strategy"].map(
        lambda x: family_of(x, collapse_ma_respect))

    grouped = df.groupby(["symbol", "zone_type"], dropna=False)
    rows = []
    for (sym, zone), g in grouped:
        fams = sorted(g["family"].unique())
        norms = sorted(g.groupby("family")["score_norm"].max().tolist(),
                       reverse=True)
        # conviction_v2: the best family leads, every ADDITIONAL agreeing family
        # adds a little. The old composite_score was a MEAN, so one strategy at
        # rank 100 scored 100 while four at 90/85/80/75 scored 82.5 — breadth
        # LOWERED the score. n_strategies sorting first masked it, but any
        # consumer sorting on composite_score alone had it backwards.
        conv_v2 = round(norms[0] + sum(norms[1:]) / 100.0, 2) if norms else 0.0
        ev = [f for f in fams if is_event(f, zone)]
        rows.append({
            "symbol": sym,
            "zone_type": zone,
            # unique strategies, not rows — duplicate rows from one strategy
            # must not masquerade as multi-strategy agreement
            "n_strategies": int(g["strategy"].nunique()),
            # the number that should actually be trusted: independent IDEAS
            "n_families": int(g["family"].nunique()),
            "families": ", ".join(fams),
            "strategies": ", ".join(sorted(g["strategy"].unique())),
            "n_event_families": len(ev),
            "event_families": ", ".join(ev),
            "momentum_profile": momentum_profile(
                set(g.loc[g["family"] == "momentum", "variant"].dropna())
                if "variant" in g.columns else set()),
            "momentum_variants": int(
                g.loc[g["family"] == "momentum", "variant"].nunique()
                if "variant" in g.columns else 0),
            "conviction_v2": conv_v2,
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
    # Rank on independent ideas, then on the breadth-rewarding score. The old
    # keys (n_strategies, composite_score) are retained as columns so nothing
    # downstream breaks.
    return out.sort_values(["n_families", "conviction_v2"],
                           ascending=[False, False]).reset_index(drop=True)


def compute_conviction(unified: pd.DataFrame, min_families: int = 2) -> pd.DataFrame:
    """Stocks flagged by >= min_families INDEPENDENT ideas, same zone_type.

    Was min_strategies, which momentum's five lookbacks satisfied on their own."""
    return unified[unified["n_families"] >= min_families].copy().reset_index(drop=True)


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

def apply_liquidity_gate(signals: pd.DataFrame, feat: pd.DataFrame,
                         min_turnover_cr: float) -> pd.DataFrame:
    """Drop signals in names that cannot actually be traded.

    No strategy applied a liquidity filter — avg_turnover_20d_cr is computed in
    compute_features.py and read by NONE of them. Untradeable names therefore sat
    inside the per-strategy percentile ranks that score everyone else. Gating
    once here, before ranking, fixes both the list and the ranks."""
    if feat is None or "avg_turnover_20d_cr" not in getattr(feat, "columns", []):
        log("liquidity gate SKIPPED — features lack avg_turnover_20d_cr")
        return signals
    t = (feat[["symbol", "avg_turnover_20d_cr"]]
         .dropna(subset=["symbol"]).drop_duplicates("symbol"))
    t["symbol"] = t["symbol"].astype(str)
    before = len(signals)
    out = signals.copy()
    out["symbol"] = out["symbol"].astype(str)
    out = out.merge(t, on="symbol", how="left")
    keep = pd.to_numeric(out["avg_turnover_20d_cr"],
                         errors="coerce") >= min_turnover_cr
    n_nocap = int(out["avg_turnover_20d_cr"].isna().sum())
    out = out[keep].drop(columns=["avg_turnover_20d_cr"])
    log(f"liquidity gate >= Rs {min_turnover_cr}cr/day: {before:,} -> {len(out):,} "
        f"signals ({n_nocap:,} had no turnover figure and were dropped)")
    return out


def update_open_signals(drive, agg_id, unified: pd.DataFrame, bar_date: str):
    """Remember the entry and stop that were live the day a signal first fired.

    entry = today's close is recomputed every run, so the stop followed the price
    DOWN as well as up and the system could never say "this position is 1R
    offside". One row per (symbol, family), first_date/entry/stop frozen at first
    appearance; same carry-forward pattern as membership.parquet."""
    prev = pd.DataFrame(columns=["symbol", "family", "first_date", "entry_at_signal",
                                 "stop_at_signal", "last_seen", "times_seen",
                                 "zone_at_signal", "conviction_at_signal",
                                 "n_families_at_signal", "n_events_at_signal"])
    fid = find_file(drive, agg_id, "open_signals.csv")
    if fid:
        try:
            prev = download_csv(drive, fid)
        except Exception as e:
            log(f"open_signals unreadable ({str(e)[:60]}) — starting fresh")

    cur = unified.copy()
    cur = cur[cur["zone_type"].isin(["buy", "add"])]
    rows = []
    seen_before = {(str(r["symbol"]), str(r["family"])): r
                   for _, r in prev.iterrows()} if len(prev) else {}
    for _, r in cur.iterrows():
        for fam in [f.strip() for f in str(r.get("families", "")).split(",") if f.strip()]:
            key = (str(r["symbol"]), fam)
            old = seen_before.get(key)
            rows.append({
                "symbol": r["symbol"], "family": fam,
                "first_date": old["first_date"] if old is not None else bar_date,
                # frozen at first sighting — never recomputed
                "entry_at_signal": (old["entry_at_signal"] if old is not None
                                    else r.get("entry_median")),
                "stop_at_signal": (old["stop_at_signal"] if old is not None
                                   else r.get("stop_median")),
                "last_seen": bar_date,
                "times_seen": (int(old["times_seen"]) + 1) if old is not None else 1,
                # Frozen alongside the price: signal_tracker.py needs the score
                # AS IT WAS when the call was made to ask whether it predicted
                # anything. Reading today's score against a months-old outcome
                # would be measuring the wrong thing.
                "zone_at_signal": (old["zone_at_signal"] if old is not None
                                   else r.get("zone_type")),
                "conviction_at_signal": (old["conviction_at_signal"] if old is not None
                                         else r.get("conviction_v2")),
                "n_families_at_signal": (old["n_families_at_signal"] if old is not None
                                         else r.get("n_families")),
                "n_events_at_signal": (old["n_events_at_signal"] if old is not None
                                       else r.get("n_event_families")),
            })
    live = pd.DataFrame(rows)
    live_keys = {(str(r["symbol"]), str(r["family"])) for _, r in live.iterrows()}         if len(live) else set()
    # names that dropped off keep their history untouched
    gone = [r for _, r in prev.iterrows()
            if (str(r["symbol"]), str(r["family"])) not in live_keys] if len(prev) else []
    out = pd.concat([live, pd.DataFrame(gone)], ignore_index=True) if gone else live
    n_new = int((live["times_seen"] == 1).sum()) if len(live) else 0
    log(f"open_signals: {len(out):,} rows ({n_new:,} first appeared today, "
        f"{len(gone):,} carried forward after dropping off)")
    upload_csv(drive, agg_id, "open_signals.csv", out,
               find_file(drive, agg_id, "open_signals.csv"))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-turnover", type=float, default=1.0,
                    help="minimum avg 20d traded value, Rs cr/day (default 1)")
    ap.add_argument("--min-families", type=int, default=2,
                    help="conviction list: minimum INDEPENDENT families agreeing")
    ap.add_argument("--collapse-ma-respect", action="store_true",
                    help="treat ma_respect_* as one family, as momentum_* already "
                         "is. NOT the default: the collapse is not signed off.")
    ap.add_argument("--dry-run", action="store_true",
                    help="compute and report, write nothing to Drive")
    args = ap.parse_args()

    print("Stage 9 — Aggregator + Multi-Strategy Conviction")
    print("-" * 60)

    drive = get_drive()
    folder_id = os.environ["GDRIVE_FOLDER_ID"]

    # Reference date = the bar date this run's features were computed on.
    # Strategy files older than this are yesterday's leftovers (ghost signals).
    ref_date = None
    feat_df = None
    feat_id = get_or_create_subfolder(drive, folder_id, "features")
    latest_feat = find_file(drive, feat_id, "latest.parquet")
    if latest_feat:
        try:
            fdf = download_parquet(drive, latest_feat)
            feat_df = fdf                      # reused by the liquidity gate
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

    # 1b. Liquidity gate BEFORE ranking, so untradeable names do not sit inside
    # the percentiles that score everyone else.
    signals = apply_liquidity_gate(signals, feat_df, args.min_turnover)
    if signals.empty:
        print("No signals survived the liquidity gate.")
        return

    # 2. Unified view
    unified = compute_unified(signals, collapse_ma_respect=args.collapse_ma_respect)
    log(f"Unified rows (one per (symbol, zone)): {len(unified)}")

    # 3. Conviction list — INDEPENDENT families, not strategy count
    conviction = compute_conviction(unified, min_families=args.min_families)
    log(f"Conviction (>={args.min_families} independent families agree): "
        f"{len(conviction)} rows")
    if len(unified):
        n_old = int((unified["n_strategies"] >= args.min_families).sum())
        log(f"  for comparison, the old n_strategies rule would give {n_old} rows "
            f"({n_old - len(conviction):+d})")
        ev = int((conviction["n_event_families"] > 0).sum()) if len(conviction) else 0
        log(f"  of those, {ev} have an EVENT firing today "
            f"(the rest are states that have been true for a while)")

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
    if args.dry_run:
        log("DRY RUN — nothing written to Drive")
        print(conviction.head(15).to_string(index=False) if len(conviction) else "(none)")
        return

    upload_csv(drive, agg_id, "latest.csv", unified,
               find_file(drive, agg_id, "latest.csv"))
    upload_csv(drive, agg_id, f"{today_str}.csv", unified,
               find_file(drive, agg_id, f"{today_str}.csv"))
    upload_csv(drive, agg_id, "conviction.csv", conviction,
               find_file(drive, agg_id, "conviction.csv"))
    upload_csv(drive, agg_id, "diff_vs_yesterday.csv", diff,
               find_file(drive, agg_id, "diff_vs_yesterday.csv"))
    log("Wrote: latest.csv, dated snapshot, conviction.csv, diff_vs_yesterday.csv")

    # 5b. Remember the entry/stop that were live when each signal first fired.
    try:
        update_open_signals(drive, agg_id, conviction, today_str)
    except Exception as e:
        # Never fail the aggregation over the memory file; the lists are the
        # deliverable and this can be rebuilt from the dated snapshots.
        log(f"open_signals update SKIPPED ({type(e).__name__}: {str(e)[:80]})")

    # 6. Summary
    print()
    print("-" * 60)
    if not conviction.empty:
        print(f"Top 10 Multi-Strategy Conviction names:")
        show = ["symbol", "zone_type", "n_strategies", "composite_score", "strategies"]
        print(conviction.head(10)[show].to_string(index=False))


if __name__ == "__main__":
    main()
