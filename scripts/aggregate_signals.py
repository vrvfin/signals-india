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
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

_SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)
from strategy_common import pct_rank
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


def _variants(g: pd.DataFrame) -> set:
    """Which momentum lookbacks fired, for this (symbol, zone).

    Derived from the STRATEGY NAME (`momentum_6m` -> `6m`) rather than from a
    `variant` column. The column only exists once the new momentum strategy is
    deployed, so relying on it left momentum_profile blank on every row when run
    against the currently-deployed output — a headline feature silently dead.
    The name is always there, so this works before and after."""
    if "family" not in g.columns:
        return set()
    m = g.loc[g["family"] == "momentum", "strategy"].astype(str)
    out = {s_.split("momentum_", 1)[1] for s_ in m if s_.startswith("momentum_")}
    if "variant" in g.columns:                # prefer an explicit column if set
        out |= {str(v) for v in
                g.loc[g["family"] == "momentum", "variant"].dropna()
                if str(v) not in ("", "nan")}
    return out


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


def add_ranking_terms(signals: pd.DataFrame, feat: pd.DataFrame | None,
                      opens: pd.DataFrame | None) -> pd.DataFrame:
    """Four orthogonal ranking measures, percentile-ranked WITHIN each strategy.

    Computed here rather than inside each of the nine strategies: one
    implementation instead of nine, identical normalisation by construction, and
    the strategy files stay untouched. Every strategy already emits entry, stop
    and symbol, which is all three of the price-based terms need.

    The point of all four is to rank on what the GATE DID NOT ALREADY MEASURE.
    momentum gated on rs_rank and then ranked on rs_rank; minervini and canslim
    ranked on how many boxes were ticked. Neither ordering carried information.

      term_risk     (entry - stop) / entry, ranked ASCENDING. The dimension
                    missing everywhere except pullback. Two identical-looking
                    stocks with stops 4% and 14% away are not the same trade:
                    you can size the first three times larger for the same rupee
                    risk. Free to compute — the numbers are already there.
      term_rs       index-relative strength. +38.0pp lift at a 72.6% win rate in
                    guru/backtest/family_lift.parquet, computed daily and, until
                    now, read by nothing.
      term_stage    freshness. Ranked so a name that qualified TODAY beats one
                    that has been on the list for two months — inverting the
                    bias that had ma_respect and qullamaggie ranking the most
                    extended stock top.
      term_confirm  range expansion. TECH_ATR is +22.1pp; TECH_VOLSURGE, which
                    volume_breakout used to rank on, is +5.7pp.

    NOTE: these are written as OBSERVABLE COLUMNS only. Nothing consumes them
    yet — the blend weights are deliberately not chosen here. signal_tracker.py
    measures whether each term separates outcomes, and that is what should pick
    the weights (user sign-off pending, 2026-09-01).
    """
    df = signals.copy()
    df["symbol"] = df["symbol"].astype(str)

    # --- risk: from entry/stop, which every strategy already emits ----------
    e = pd.to_numeric(df.get("entry"), errors="coerce")
    st = pd.to_numeric(df.get("stop"), errors="coerce")
    risk_pct = ((e - st) / e * 100).where((e > 0) & (st < e))
    df["risk_pct"] = risk_pct.round(2)

    # --- rs + confirmation: joined from features ----------------------------
    for col, src in (("_rs", "rs_vs_nifty500_6m_pct"),
                     ("_confirm", "atr_expansion_ratio")):
        if feat is not None and src in getattr(feat, "columns", []):
            m = (feat[["symbol", src]].dropna(subset=["symbol"])
                 .drop_duplicates("symbol"))
            m["symbol"] = m["symbol"].astype(str)
            df = df.merge(m.rename(columns={src: col}), on="symbol", how="left")
        else:
            df[col] = np.nan

    # --- stage: how long this name has already been on the list -------------
    if opens is not None and len(opens) and "first_date" in opens.columns:
        o = opens.copy()
        o["symbol"] = o["symbol"].astype(str)
        age = (o.groupby("symbol")["first_date"]
                .min().rename("_first_seen").reset_index())
        df = df.merge(age, on="symbol", how="left")
        first = pd.to_datetime(df["_first_seen"], errors="coerce")
        today = pd.to_datetime(df.get("date"), errors="coerce")
        df["_stage_days"] = (today - first).dt.days
        # A name with no history is brand new today, which is the freshest case.
        df["_stage_days"] = df["_stage_days"].fillna(0)
        df = df.drop(columns=["_first_seen"])
    else:
        df["_stage_days"] = np.nan

    # Percentile WITHIN each strategy, so 90 means "top decile of that
    # strategy's signals today" for every strategy alike.
    g = df.groupby("strategy")
    df["term_risk"] = g["risk_pct"].transform(
        lambda s: pct_rank(s, ascending=False))       # tighter stop ranks higher
    df["term_rs"] = g["_rs"].transform(pct_rank)
    df["term_stage"] = g["_stage_days"].transform(
        lambda s: pct_rank(s, ascending=False))       # fresher ranks higher
    df["term_confirm"] = g["_confirm"].transform(pct_rank)
    return df.drop(columns=["_rs", "_confirm", "_stage_days"])


def terms_report(signals: pd.DataFrame, n: int = 20) -> pd.DataFrame:
    """Per strategy: the current top N by `score` beside the top N by each term.

    This is the sheet the blend weights should be chosen against — real names,
    not a formula argued in the abstract."""
    terms = ["score", "term_risk", "term_rs", "term_stage", "term_confirm"]
    rows = []
    for strat, g in signals.groupby("strategy"):
        for t in terms:
            if t not in g.columns or g[t].isna().all():
                continue
            top = g.nlargest(n, t)["symbol"].tolist()
            rows.append({"strategy": strat, "ranked_by": t, "n_signals": len(g),
                         "top_n": ", ".join(top[:n]),
                         "overlap_with_score_pct": (
                             round(100.0 * len(set(top) & set(
                                 g.nlargest(n, "score")["symbol"])) / max(len(top), 1), 1)
                             if "score" in g.columns else np.nan)})
    return pd.DataFrame(rows)


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
            "momentum_profile": momentum_profile(_variants(g)),
            "momentum_variants": len(_variants(g)),
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

def _safe_print(text: str) -> None:
    """Print without dying on a non-UTF-8 console.

    The strategies build their `reason` strings with the rupee sign, and a
    Windows console is cp1252, so printing the preview raised
    UnicodeEncodeError and took the whole run down with it — after all the work
    was done, and only in --dry-run, which is the mode the house rules require
    before every live run."""
    enc = (getattr(sys.stdout, "encoding", None) or "utf-8")
    try:
        print(text)
    except UnicodeEncodeError:
        print(text.encode(enc, errors="replace").decode(enc, errors="replace"))


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
    ap.add_argument("--terms-report", action="store_true",
                    help="also write signals/aggregated/ranking_terms_report.csv "
                         "— per strategy, the current top 20 by score beside the "
                         "top 20 by each ranking term. This is the sheet the "
                         "blend weights should be chosen against.")
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

    # 1c. Four orthogonal ranking measures, as OBSERVABLE COLUMNS. Nothing
    # consumes them yet — the blend is deliberately not chosen here.
    agg_id_early = get_or_create_subfolder(
        drive, get_or_create_subfolder(drive, folder_id, "signals"), "aggregated")
    opens_prev = None
    try:
        _f = find_file(drive, agg_id_early, "open_signals.csv")
        opens_prev = download_csv(drive, _f) if _f else None
    except Exception:
        pass
    signals = add_ranking_terms(signals, feat_df, opens_prev)
    _have = [c for c in ("term_risk", "term_rs", "term_stage", "term_confirm")
             if c in signals.columns and signals[c].notna().any()]
    log(f"ranking terms populated: {_have}")

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
        _safe_print(conviction.head(15).to_string(index=False)
                    if len(conviction) else "(none)")
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

    if args.terms_report:
        rep = terms_report(signals)
        upload_csv(drive, agg_id, "ranking_terms_report.csv", rep,
                   find_file(drive, agg_id, "ranking_terms_report.csv"))
        log(f"Wrote ranking_terms_report.csv ({len(rep)} rows) — inspect this "
            f"before choosing any blend weights")

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
        _safe_print(conviction.head(10)[show].to_string(index=False))


if __name__ == "__main__":
    main()
