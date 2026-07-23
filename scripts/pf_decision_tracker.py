"""
pf_decision_tracker.py — Portfolio decision & performance tracker (Workflow: PF).

Answers, in PLAIN, COUNTABLE, non-black-box terms, every daily run:
  1. How is my PF moving?           -> weighted PF index (rebased to 100).
  2. Movers most/least              -> rank holdings over 1/2/3/6/12-month windows.
  3. Are my BUY/SELL/HOLD calls right? -> forward return of each decision vs the
     MEDIAN & QUARTILES of my OWN BOOK over the same window (no composite score).
  4. Relook sold stocks             -> forward return since exit + latest results YoY.
  Plus a REVIEW-NOW block: recent buys sinking to the bottom quartile and recent
  sells that ran up — so wrong calls surface within weeks, not quarters.

Design (confirmed with user 2026-07-23):
  - CI-only (daily, after the price/feature refresh) + a local one-click trigger.
  - Reads the LATEST holdings file from Drive (pf_tracking/ or portfolio/, newest
    wins) — user uploads only on change. History is kept in our OWN append-only
    ledger, so sync_pf's "delete old file" prompt can never erase it.
  - WEIGHTAGE-ONLY export: BUY (0->+) and SELL (+->0) are clean; ADD/TRIM are
    inferred by removing price-drift from the weight change; no rupee P&L.
  - Correctness judged vs the user's OWN portfolio median/quartiles (not NIFTY).
  - Email digest only (no LLM, no ntfy, no Streamlit).

Reuses the shared Drive/mail helpers per CLAUDE.md rule 4 (no raw Drive calls,
no parallel queue). New Drive tables are additive; no existing schema is touched.

Usage:
    python scripts/pf_decision_tracker.py                    # live: write + email
    python scripts/pf_decision_tracker.py --dry-run          # read-only, no write/mail, writes an HTML preview locally
    python scripts/pf_decision_tracker.py --no-mail          # compute + write parquets, skip email
    python scripts/pf_decision_tracker.py --asof 2026-07-20  # recompute as of a past date (testing)
"""
from __future__ import annotations

import argparse
import io
import os
import sys
from datetime import datetime, date
from pathlib import Path

import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
load_dotenv(ROOT / ".env")

# Shared infra (CLAUDE.md rule 4 — reuse, never hand-roll Drive calls).
from _extractor_base import (
    log, get_drive, get_or_create_subfolder, find_file, download_bytes,
    load_parquet, save_parquet, find_latest_portfolio_file, isin_symbol_map,
)
from mailer import send_email, load_mail_settings, esc

# ── Constants ────────────────────────────────────────────────────────────────
MAIL_KEY = "pf_decision"                     # matches mailer.MAIL_KEYS
HORIZONS_M = [1, 2, 3, 6, 12]                # decision-scoring windows (months)
# Feature return columns already computed by compute_features.py (trailing, as-of-today).
FEAT_RET_COLS = {m: f"return_{lbl}_pct"
                 for m, lbl in {1: "1m", 2: "2m", 3: "3m", 6: "6m", 12: "12m"}.items()}
LOCAL_OUT = Path(r"D:\EMA_Screener\Reports\signals-india\pf_tracker")   # dry-run preview

# Output-ledger schemas (additive; stored in company_repo/_index/pf_tracker/).
SNAP_COLS  = ["snapshot_date", "isin", "symbol", "name", "weight_pct", "source_file"]
DEC_COLS   = ["event_date", "isin", "symbol", "name", "action",
              "weight_before", "weight_after"]
IDX_COLS   = ["date", "pf_index", "daily_ret_pct", "n_holdings", "n_priced"]
SCORE_COLS = ["asof", "event_date", "isin", "symbol", "action", "horizon_m",
              "fwd_ret_pct", "book_median_pct", "quartile", "peer_n",
              "verdict", "correct", "matured"]
MOVER_COLS = ["symbol", "isin", "name", "weight_pct",
              "return_1m_pct", "return_2m_pct", "return_3m_pct",
              "return_6m_pct", "return_12m_pct",
              "leader_windows", "laggard_windows", "tag"]

# ADD/TRIM band: a weight change counts as a real trade only if it exceeds the
# drift-implied weight by BOTH an absolute (pp) and a relative margin.
ADD_TRIM_ABS_PP  = 0.5
ADD_TRIM_REL     = 0.20


# ── Drive navigation ─────────────────────────────────────────────────────────

def _folder(drive, parent_id, *names):
    """Resolve (creating if needed) a nested subfolder path, return its id."""
    fid = parent_id
    for n in names:
        fid = get_or_create_subfolder(drive, fid, n)
    return fid


def _read_drive_parquet(drive, folder_id, filename, cols=None):
    """Read a parquet living directly in folder_id. Raw (all columns) unless cols
    given. Never raises — returns empty df on any miss (matches load_parquet spirit)."""
    try:
        fid = find_file(drive, folder_id, filename)
        if not fid:
            return pd.DataFrame(columns=cols or [])
        raw = download_bytes(drive, fid)
        df = (pd.read_csv(io.BytesIO(raw)) if filename.lower().endswith(".csv")
              else pd.read_parquet(io.BytesIO(raw)))
        return df[cols] if cols else df
    except Exception as e:
        log(f"  WARN: could not read {filename} ({str(e)[:80]})")
        return pd.DataFrame(columns=cols or [])


# ── Holdings parse (weightage auto-detect) ───────────────────────────────────

_WEIGHT_HINTS = ["weightage", "weight (%)", "weight%", "weight %", "% weight",
                 "portfolio weight", "weight", "% of portfolio", "allocation"]


def detect_weight_col(cols) -> str | None:
    """Find the '% weightage' column by known names, then fuzzy 'weight' match."""
    lowered = {str(c).strip().lower(): c for c in cols}
    for hint in _WEIGHT_HINTS:
        if hint in lowered:
            return lowered[hint]
    for lc, orig in lowered.items():
        if "weight" in lc:
            return orig
    return None


def parse_holdings(raw: bytes, filename: str) -> tuple[pd.DataFrame, str | None]:
    """Parse a Screener 'Holdings Statement' export -> df[isin, weight_pct, screener_name].
    Mirrors _extractor_base.load_portfolio_isins header-detection, but KEEPS the
    weightage column. Returns (df, detected_weight_col_name)."""
    fn = filename.lower()
    eng = "xlrd" if fn.endswith(".xls") else ("openpyxl" if fn.endswith(".xlsx") else "csv")
    read = (lambda h: pd.read_csv(io.BytesIO(raw), header=h)) if eng == "csv" \
        else (lambda h: pd.read_excel(io.BytesIO(raw), engine=eng, header=h))

    df_raw = read(None)
    hrow = None
    for i, row in df_raw.iterrows():
        if any(str(v).strip().upper() == "ISIN" for v in row.dropna()):
            hrow = i
            break
    if hrow is None:
        log("  ERROR: ISIN header not found in holdings file.")
        return pd.DataFrame(columns=["isin", "weight_pct", "screener_name"]), None

    df = read(hrow)
    wcol = detect_weight_col(df.columns)

    name_col = next((c for c in ("Company name", "Stock/ETF Name", "Name")
                     if c in df.columns), None)
    out = pd.DataFrame()
    out["isin"] = df["ISIN"].astype(str).str.strip().str.upper()
    out["screener_name"] = (df[name_col].astype(str).str.strip()
                            if name_col else out["isin"])
    if wcol:
        w = (df[wcol].astype(str).str.replace("%", "", regex=False)
             .str.replace(",", "", regex=False).str.strip())
        out["weight_pct"] = pd.to_numeric(w, errors="coerce")
    else:
        out["weight_pct"] = np.nan

    out = out[out["isin"].str.match(r"^IN[A-Z0-9]{10}$", na=False)].copy()
    out = out.dropna(subset=["isin"]).drop_duplicates("isin")

    # Equal-weight fallback when the export has no weightage column / all-blank.
    if out["weight_pct"].notna().sum() == 0 and len(out):
        out["weight_pct"] = 100.0 / len(out)
        log(f"  WARN: no weightage column detected — using equal weights (1/{len(out)}).")
    return out.reset_index(drop=True), wcol


# ── Symbol / price / results loaders ─────────────────────────────────────────

def resolve_symbols(drive, root_id, pf: pd.DataFrame) -> pd.DataFrame:
    """Attach NSE symbol + name via universe/master_list.csv (isin,symbol,name)."""
    uni_id = get_or_create_subfolder(drive, root_id, "universe")
    master = _read_drive_parquet(drive, uni_id, "master_list.csv")
    if master.empty:
        log("  WARN: universe/master_list.csv missing — symbols unresolved.")
        pf["symbol"] = pf["isin"]
        pf["name"] = pf["screener_name"]
        return pf
    i2s = isin_symbol_map(master)
    name_map = (dict(zip(master["isin"].astype(str), master["name"].astype(str)))
                if {"isin", "name"} <= set(master.columns) else {})
    pf["symbol"] = pf["isin"].map(i2s).fillna(pf["screener_name"]).fillna(pf["isin"])
    pf["name"] = pf["isin"].map(name_map).fillna(pf["screener_name"]).fillna(pf["symbol"])
    return pf


def load_close_series(drive, ohlcv_id, symbol: str, cache: dict) -> pd.Series | None:
    """Per-symbol daily close series (date-indexed, ascending). Cached across calls."""
    if symbol in cache:
        return cache[symbol]
    df = _read_drive_parquet(drive, ohlcv_id, f"{symbol}.parquet", cols=None)
    ser = None
    if not df.empty and {"date", "close"} <= set(df.columns):
        s = df[["date", "close"]].copy()
        s["date"] = pd.to_datetime(s["date"])
        ser = s.dropna().sort_values("date").set_index("date")["close"]
        ser = ser[~ser.index.duplicated(keep="last")]
    cache[symbol] = ser
    return ser


def fwd_return(ser: pd.Series | None, start, end) -> float | None:
    """% return between the last close <= start and the last close <= end."""
    if ser is None or ser.empty:
        return None
    p0, p1 = ser.asof(pd.Timestamp(start)), ser.asof(pd.Timestamp(end))
    if pd.isna(p0) or pd.isna(p1) or p0 == 0:
        return None
    return float((p1 / p0 - 1.0) * 100.0)


# ── Snapshot + decision ledgers ──────────────────────────────────────────────

def append_snapshot(drive, out_id, snaps: pd.DataFrame, pf: pd.DataFrame,
                    asof: date, source_file: str, dry: bool) -> tuple[pd.DataFrame, bool]:
    """Append today's holdings as a new snapshot IFF holdings/weights changed vs the
    last snapshot. Returns (snaps, changed)."""
    cur = {r.isin: round(float(r.weight_pct), 4) for r in pf.itertuples()}
    if not snaps.empty:
        last_date = snaps["snapshot_date"].max()
        last = snaps[snaps["snapshot_date"] == last_date]
        prev = {r.isin: round(float(r.weight_pct), 4) for r in last.itertuples()}
        if prev == cur:
            log(f"  Snapshot unchanged vs {last_date} — no new snapshot.")
            return snaps, False
        if str(last_date) == str(asof):
            # Same-day re-run with a changed file: replace today's rows, don't dup.
            snaps = snaps[snaps["snapshot_date"] != last_date].copy()

    rows = pf.assign(snapshot_date=str(asof), source_file=source_file)[
        ["snapshot_date", "isin", "symbol", "name", "weight_pct", "source_file"]]
    snaps = pd.concat([snaps, rows], ignore_index=True)
    if not dry:
        save_parquet(drive, out_id, "pf_snapshots.parquet", snaps)
    log(f"  Snapshot appended for {asof}: {len(pf)} holdings.")
    return snaps, True


def derive_decisions(snaps: pd.DataFrame, decs: pd.DataFrame, asof: date,
                     price_cache: dict, drive, ohlcv_id) -> pd.DataFrame:
    """Diff the two most-recent DISTINCT snapshots -> new BUY/SELL/ADD/TRIM rows.
    First-ever snapshot -> INIT rows (pre-existing holdings, not scored as buys)."""
    if snaps.empty:
        return decs
    dates = sorted(snaps["snapshot_date"].unique())
    cur_d = dates[-1]
    cur = snaps[snaps["snapshot_date"] == cur_d].set_index("isin")

    if len(dates) == 1:
        if not decs[decs["action"] == "INIT"].empty:
            return decs
        new = [dict(event_date=cur_d, isin=i, symbol=r.symbol, name=r.name,
                    action="INIT", weight_before=np.nan, weight_after=r.weight_pct)
               for i, r in cur.iterrows()]
        return pd.concat([decs, pd.DataFrame(new, columns=DEC_COLS)], ignore_index=True)

    prev_d = dates[-2]
    prev = snaps[snaps["snapshot_date"] == prev_d].set_index("isin")

    # Idempotence: never re-emit events already recorded for cur_d.
    if not decs[decs["event_date"] == cur_d].empty:
        return decs

    # Price move between the two snapshot dates (for drift-adjusted ADD/TRIM).
    def _move(isin, symbol):
        s = load_close_series(drive, ohlcv_id, symbol, price_cache)
        r = fwd_return(s, pd.Timestamp(prev_d), pd.Timestamp(cur_d))
        return 0.0 if r is None else r / 100.0

    # Drift-implied weights: what prev weights would become if the user did nothing.
    drift = {}
    denom = 0.0
    for i, r in prev.iterrows():
        g = 1.0 + (_move(i, r.symbol) if i in cur.index else 0.0)
        drift[i] = float(r.weight_pct) * g
        denom += drift[i]
    if denom > 0:
        drift = {i: v / denom * 100.0 for i, v in drift.items()}

    new = []
    for i, r in cur.iterrows():
        if i not in prev.index:
            new.append(dict(event_date=cur_d, isin=i, symbol=r.symbol, name=r.name,
                            action="BUY", weight_before=np.nan, weight_after=r.weight_pct))
        else:
            wb = float(prev.loc[i, "weight_pct"]); wa = float(r.weight_pct)
            w_exp = drift.get(i, wb)
            diff = wa - w_exp
            if abs(diff) > ADD_TRIM_ABS_PP and abs(diff) > ADD_TRIM_REL * max(w_exp, 1e-9):
                new.append(dict(event_date=cur_d, isin=i, symbol=r.symbol, name=r.name,
                                action="ADD" if diff > 0 else "TRIM",
                                weight_before=wb, weight_after=wa))
    for i, r in prev.iterrows():
        if i not in cur.index:
            new.append(dict(event_date=cur_d, isin=i, symbol=r.symbol, name=r.name,
                            action="SELL", weight_before=float(r.weight_pct),
                            weight_after=0.0))
    if new:
        decs = pd.concat([decs, pd.DataFrame(new, columns=DEC_COLS)], ignore_index=True)
        log(f"  {len(new)} new decision event(s) on {cur_d}: "
            + ", ".join(f"{d['action']} {d['symbol']}" for d in new[:8])
            + ("…" if len(new) > 8 else ""))
    else:
        log(f"  Holdings changed on {cur_d} but no BUY/SELL/ADD/TRIM crossed the band.")
    return decs


# ── PF index (fixed-weight, daily-rebalanced weighted-return) ────────────────

def compute_index(snaps: pd.DataFrame, price_cache: dict, drive, ohlcv_id,
                  asof: date) -> tuple[pd.DataFrame, list]:
    """Daily weighted-return index rebased to 100 at the first snapshot. Each day's
    return = Σ wᵢ·rᵢ over holdings priced that day (weights = active snapshot,
    renormalised over priced names). Returns (index_df, latest_contributors)."""
    if snaps.empty:
        return pd.DataFrame(columns=IDX_COLS), []
    syms = sorted(snaps["symbol"].unique())
    series = {s: load_close_series(drive, ohlcv_id, s, price_cache) for s in syms}
    series = {s: v for s, v in series.items() if v is not None and not v.empty}
    if not series:
        return pd.DataFrame(columns=IDX_COLS), []

    end = pd.Timestamp(asof)
    # Trailing view: extend back up to 12M before the first snapshot using the
    # earliest snapshot's weights (active_weights() falls back to it), so the index
    # is useful from day 1 and becomes fully real as snapshots accrue.
    start = min(pd.Timestamp(snaps["snapshot_date"].min()), end - relativedelta(months=12))
    close = pd.DataFrame(series).sort_index()
    close = close.loc[(close.index >= start) & (close.index <= end)]
    if len(close) < 2:
        return pd.DataFrame(columns=IDX_COLS), []
    ret = close.pct_change()

    snap_dates = sorted(pd.Timestamp(d) for d in snaps["snapshot_date"].unique())

    def active_weights(ts):
        d = max((sd for sd in snap_dates if sd <= ts), default=snap_dates[0])
        rows = snaps[snaps["snapshot_date"] == str(d.date())]
        return {r.symbol: float(r.weight_pct) for r in rows.itertuples()
                if pd.notna(r.weight_pct)}

    daily_ret, n_hold, n_priced = [], [], []
    for ts in close.index:
        w = active_weights(ts)
        row_ret = ret.loc[ts]
        num, den, npr = 0.0, 0.0, 0
        for sym, wt in w.items():
            r = row_ret.get(sym, np.nan)
            if pd.notna(r):
                num += wt * r; den += wt; npr += 1
        daily_ret.append(num / den if den > 0 else 0.0)
        n_hold.append(len(w)); n_priced.append(npr)

    idx = pd.DataFrame({"date": close.index, "daily_ret_pct": np.array(daily_ret) * 100.0,
                        "n_holdings": n_hold, "n_priced": n_priced})
    idx.iloc[0, idx.columns.get_loc("daily_ret_pct")] = 0.0     # first day is the base
    idx["pf_index"] = 100.0 * (1.0 + idx["daily_ret_pct"] / 100.0).cumprod()
    idx["date"] = idx["date"].dt.strftime("%Y-%m-%d")
    idx = idx[IDX_COLS]

    # Latest-day contributors, in percentage POINTS of today's PF return
    # (normalised by the day's active weight sum, so it's scale-independent).
    last_ts = close.index[-1]
    w = active_weights(last_ts); rr = ret.loc[last_ts]
    priced = {s: wt for s, wt in w.items() if pd.notna(rr.get(s, np.nan))}
    den = sum(priced.values()) or 1.0
    contrib = [(s, wt * float(rr[s]) / den * 100.0) for s, wt in priced.items()]
    contrib.sort(key=lambda x: x[1], reverse=True)
    return idx, contrib


def index_period_returns(idx: pd.DataFrame) -> dict:
    """1D/1W/1M/3M/6M/12M PF return read off the index level series."""
    if idx.empty:
        return {}
    s = idx.set_index(pd.to_datetime(idx["date"]))["pf_index"].sort_index()
    last = s.iloc[-1]; end = s.index[-1]
    out = {"level": float(last)}
    for lbl, off in {"1D": relativedelta(days=1), "1W": relativedelta(weeks=1),
                     "1M": relativedelta(months=1), "3M": relativedelta(months=3),
                     "6M": relativedelta(months=6), "12M": relativedelta(months=12)}.items():
        base = s.asof(end - off)
        out[lbl] = float((last / base - 1) * 100) if pd.notna(base) and base else None
    return out


# ── Movers (as-of-today trailing, from features) ─────────────────────────────

def compute_movers(drive, root_id, pf: pd.DataFrame) -> pd.DataFrame:
    feat_id = get_or_create_subfolder(drive, root_id, "features")
    feat = _read_drive_parquet(drive, feat_id, "latest.parquet")
    m = pf[["symbol", "isin", "name", "weight_pct"]].copy()
    ret_cols = list(FEAT_RET_COLS.values())
    if not feat.empty and "symbol" in feat.columns:
        keep = ["symbol"] + [c for c in ret_cols if c in feat.columns]
        m = m.merge(feat[keep].drop_duplicates("symbol"), on="symbol", how="left")
    for c in ret_cols:
        if c not in m.columns:
            m[c] = np.nan

    # Consistent leader/laggard = top/bottom quartile in ≥3 of the 5 windows.
    lead = pd.Series(0, index=m.index); lag = pd.Series(0, index=m.index)
    for c in ret_cols:
        col = m[c]
        if col.notna().sum() >= 4:
            hi, lo = col.quantile(0.75), col.quantile(0.25)
            lead += (col >= hi).astype(int)
            lag += (col <= lo).astype(int)
    m["leader_windows"] = lead; m["laggard_windows"] = lag
    m["tag"] = np.where(lead >= 3, "consistent leader",
                        np.where(lag >= 3, "consistent laggard", ""))
    return m[MOVER_COLS]


# ── Decision scorecard (vs own-book median/quartiles) ────────────────────────

def _peer_isins_at(snaps: pd.DataFrame, event_date: str) -> pd.DataFrame:
    cand = sorted(d for d in snaps["snapshot_date"].unique() if d <= event_date)
    d = cand[-1] if cand else snaps["snapshot_date"].min()
    return snaps[snaps["snapshot_date"] == d]


def _verdict(action: str, r, median, quartile) -> tuple[str, bool | None]:
    if r is None or median is None:
        return "n/a", None
    buyish = action in ("BUY", "ADD", "INIT", "HOLD")
    if buyish:
        correct = r >= median
        v = "great" if quartile == 1 else ("working" if correct
             else ("poor" if quartile == 4 else "lagging"))
    else:  # SELL / TRIM — a correct exit is one that did WORSE than the book
        correct = r <= median
        v = "great exit" if quartile == 4 else ("good exit" if correct
             else ("premature (regret)" if quartile == 1 else "early (regret)"))
    return v, bool(correct)


def score_decisions(decs: pd.DataFrame, snaps: pd.DataFrame, price_cache: dict,
                    drive, ohlcv_id, asof: date) -> pd.DataFrame:
    """For each event × horizon: forward return vs the book's median & quartiles over
    the same [event_date, event_date+H] window. Regenerated fully each run."""
    if decs.empty:
        return pd.DataFrame(columns=SCORE_COLS)
    rows = []
    for ev in decs.itertuples():
        peers = _peer_isins_at(snaps, ev.event_date)
        peer_syms = peers["symbol"].tolist()
        ev_start = pd.Timestamp(ev.event_date)
        for h in HORIZONS_M:
            ev_end_full = ev_start + relativedelta(months=h)
            ev_end = min(ev_end_full, pd.Timestamp(asof))
            matured = ev_end_full <= pd.Timestamp(asof)
            r = fwd_return(load_close_series(drive, ohlcv_id, ev.symbol, price_cache),
                           ev_start, ev_end)
            peer_rets = [pr for s in peer_syms
                         if (pr := fwd_return(load_close_series(drive, ohlcv_id, s, price_cache),
                                              ev_start, ev_end)) is not None]
            median = float(np.median(peer_rets)) if peer_rets else None
            quartile = peer_n = None
            if r is not None and len(peer_rets) >= 4:
                pct = float((np.array(peer_rets) <= r).mean())   # fraction of book at/below r
                quartile = 1 if pct >= 0.75 else 2 if pct >= 0.5 else 3 if pct >= 0.25 else 4
                peer_n = len(peer_rets)
            v, correct = _verdict(ev.action, r, median, quartile)
            rows.append(dict(asof=str(asof), event_date=ev.event_date, isin=ev.isin,
                             symbol=ev.symbol, action=ev.action, horizon_m=h,
                             fwd_ret_pct=r, book_median_pct=median, quartile=quartile,
                             peer_n=peer_n, verdict=v, correct=correct, matured=matured))
    return pd.DataFrame(rows, columns=SCORE_COLS)


# ── Sold-stock relook ────────────────────────────────────────────────────────

def sold_relook(decs: pd.DataFrame, price_cache: dict, drive, ohlcv_id,
                results: pd.DataFrame, asof: date, lookback_m: int = 12) -> pd.DataFrame:
    sells = decs[decs["action"] == "SELL"].copy()
    if sells.empty:
        return pd.DataFrame(columns=["event_date", "symbol", "isin", "name",
                                     "ret_since_sell_pct", "latest_yoy_pct", "verdict"])
    cutoff = pd.Timestamp(asof) - relativedelta(months=lookback_m)
    sells = sells[pd.to_datetime(sells["event_date"]) >= cutoff]
    yoy_map = {}
    if not results.empty and {"symbol", "yoy_pct"} <= set(results.columns):
        pcol = "period" if "period" in results.columns else None
        for sym, g in results.groupby("symbol"):
            if pcol:
                g = g.sort_values(pcol)
            yoy_map[sym] = pd.to_numeric(g["yoy_pct"], errors="coerce").dropna().iloc[-1] \
                if g["yoy_pct"].notna().any() else None
    rows = []
    for ev in sells.itertuples():
        r = fwd_return(load_close_series(drive, ohlcv_id, ev.symbol, price_cache),
                       pd.Timestamp(ev.event_date), pd.Timestamp(asof))
        yoy = yoy_map.get(ev.symbol)
        if r is None:
            v = "n/a"
        elif r <= 0 or (yoy is not None and yoy < 0):
            v = "validated exit"           # fell since sell and/or poor results
        elif r >= 15:
            v = "regret — ran up"
        else:
            v = "neutral"
        rows.append(dict(event_date=ev.event_date, symbol=ev.symbol, isin=ev.isin,
                         name=ev.name, ret_since_sell_pct=r, latest_yoy_pct=yoy, verdict=v))
    return pd.DataFrame(rows).sort_values("ret_since_sell_pct", na_position="last")


# ── HTML digest ──────────────────────────────────────────────────────────────

def _pct(v, signed=True):
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return "—"
    col = "#1a7a3c" if v >= 0 else "#c0392b"
    return f"<span style='color:{col}'>{v:+.1f}%</span>" if signed else f"{v:.1f}%"


def build_html(asof, wcol, pf, idx, periods, contrib, movers, score, relook, decs):
    H = []
    H.append(f"<h2 style='margin:0'>PF Decision Tracker — {asof}</h2>")
    H.append(f"<p style='color:#666;margin:2px 0 14px'>{len(pf)} holdings · "
             f"weightage col: <code>{esc(str(wcol))}</code> · judged vs your own book's median · "
             f"detailed tables in the attached Excel</p>")

    # REVIEW NOW — recent buys sinking / recent sells that ran up (short horizon).
    warn = []
    if not score.empty:
        recent = pd.to_datetime(score["event_date"]) >= (pd.Timestamp(asof) - relativedelta(months=3))
        sub = score[recent & score["horizon_m"].isin([1, 2])]
        for r in sub.itertuples():
            if r.action in ("BUY", "ADD") and r.quartile == 4:
                warn.append(f"⚠️ <b>{esc(r.symbol)}</b> buy is bottom-quartile at {r.horizon_m}M "
                            f"({_pct(r.fwd_ret_pct)} vs book {_pct(r.book_median_pct)})")
            if r.action in ("SELL", "TRIM") and r.quartile == 1:
                warn.append(f"↗️ <b>{esc(r.symbol)}</b> sold but ran up at {r.horizon_m}M "
                            f"({_pct(r.fwd_ret_pct)}) — reconsider")
    if warn:
        H.append("<div style='background:#fff7e6;border:1px solid #ffd591;border-radius:6px;"
                 "padding:10px 12px;margin:8px 0'><b>REVIEW NOW</b><ul style='margin:6px 0'>"
                 + "".join(f"<li>{w}</li>" for w in dict.fromkeys(warn)) + "</ul></div>")

    # 1) PF index
    if periods:
        cells = "".join(f"<td style='padding:4px 10px'>{k}<br><b>{_pct(periods.get(k))}</b></td>"
                        for k in ["1D", "1W", "1M", "3M", "6M", "12M"])
        H.append(f"<h3>1 · How the PF is moving</h3>"
                 f"<p>Index level <b>{periods.get('level', 100):.1f}</b> (base 100)</p>"
                 f"<table style='border-collapse:collapse'><tr>{cells}</tr></table>"
                 f"<p style='color:#888;font-size:12px'>Weighted-return index. Before tracking "
                 f"began it assumes your current weights (a constant-weight trailing view); it "
                 f"becomes fully real as daily snapshots accrue.</p>")
        if contrib:
            up = contrib[0]; dn = contrib[-1]
            H.append(f"<p style='color:#666'>Today's biggest lift: <b>{esc(up[0])}</b> "
                     f"({_pct(up[1])} of PF) · biggest drag: <b>{esc(dn[0])}</b> ({_pct(dn[1])})</p>")

    # 2) Movers
    def mover_table(df, cols, title):
        h = [f"<h3>{title}</h3><table style='border-collapse:collapse;font-size:13px'>"
             "<tr><th align='left'>Symbol</th>"
             + "".join(f"<th style='padding:2px 8px'>{c.replace('return_','').replace('_pct','').upper()}</th>"
                       for c in cols) + "</tr>"]
        for r in df.itertuples():
            tds = "".join(f"<td align='right' style='padding:2px 8px'>{_pct(getattr(r, c))}</td>"
                          for c in cols)
            tag = f" <span style='color:#888'>({r.tag})</span>" if getattr(r, "tag", "") else ""
            h.append(f"<tr><td>{esc(r.symbol)}{tag}</td>{tds}</tr>")
        h.append("</table>")
        return "".join(h)

    if not movers.empty:
        rc = list(FEAT_RET_COLS.values())
        # Rank only holdings that actually have a 3M return (SME/new listings with no
        # feature row must not masquerade as the 'bottom' movers).
        ranked = movers.dropna(subset=["return_3m_pct"]).sort_values(
            "return_3m_pct", ascending=False)
        H.append("<h3>2 · Movers (trailing, as of today)</h3>")
        H.append(mover_table(ranked.head(5), rc, "Top 5 by 3M"))
        H.append(mover_table(ranked.tail(5).iloc[::-1], rc, "Bottom 5 by 3M"))

    # 3) Decision correctness (matured hit-rates)
    if not score.empty:
        H.append("<h3>3 · Are my decisions correct? (vs own-book median)</h3>")
        body = []
        for act, label in [("BUY", "Buys correct"), ("SELL", "Sells correct"),
                           ("ADD", "Adds correct"), ("TRIM", "Trims correct")]:
            a = score[(score["action"] == act) & (score["matured"]) & score["correct"].notna()]
            if a.empty:
                continue
            tds = []
            for h in HORIZONS_M:
                hh = a[a["horizon_m"] == h]
                tds.append(f"<td align='center' style='padding:2px 10px'>"
                           f"{int(hh['correct'].sum())}/{len(hh)}</td>" if len(hh)
                           else "<td align='center'>—</td>")
            body.append(f"<tr><td>{label}</td>{''.join(tds)}</tr>")
        if body:
            H.append("<table style='border-collapse:collapse;font-size:13px'>"
                     "<tr><th align='left'>Decision</th>"
                     + "".join(f"<th style='padding:2px 10px'>{h}M</th>" for h in HORIZONS_M)
                     + "</tr>" + "".join(body) + "</table>"
                     "<p style='color:#888;font-size:12px'>“Correct”: a buy/add beat your book's "
                     "median that window; a sell/trim undershot it (good exit). Matured windows only.</p>")
        else:
            H.append("<p style='color:#888'>No matured buy/sell/add/trim decisions yet — this fills "
                     "in as you trade and each window (1/2/3/6/12M) elapses. Your existing book is "
                     "covered by the movers above.</p>")

    # 4) Sold-stock relook
    if not relook.empty:
        H.append("<h3>4 · Relook: stocks you sold</h3>"
                 "<table style='border-collapse:collapse;font-size:13px'>"
                 "<tr><th align='left'>Sold</th><th>On</th><th>Since sell</th>"
                 "<th>Latest YoY</th><th>Verdict</th></tr>")
        for r in relook.itertuples():
            H.append(f"<tr><td>{esc(r.symbol)}</td><td style='padding:2px 8px'>{esc(r.event_date)}</td>"
                     f"<td align='right'>{_pct(r.ret_since_sell_pct)}</td>"
                     f"<td align='right'>{_pct(r.latest_yoy_pct)}</td>"
                     f"<td style='padding:2px 8px'>{esc(r.verdict)}</td></tr>")
        H.append("</table>")

    H.append("<p style='color:#aaa;font-size:11px;margin-top:16px'>Signals only — human-in-the-loop. "
             "Decision dates are snapshot-observation dates. Weightage-only: no rupee P&L.</p>")
    return "".join(H)


def build_excel_bytes(pf, idx, movers, score, relook, decs) -> bytes:
    """Detailed tables as a multi-sheet .xlsx (the email is the summary; this is the
    drill-down). One sheet per view; header frozen + light auto-width."""
    holdings = pf[["isin", "symbol", "name", "weight_pct"]].copy()
    sheets = {
        "Holdings Today":     holdings,
        "Movers":             movers,
        "Decision Scorecard": score,
        "Sold Relook":        relook,
        "Decisions Log":      decs,
        "PF Index":           idx,
    }
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as xw:
        for name, df in sheets.items():
            sheet = name[:31]
            out = df if len(df.columns) else pd.DataFrame({"(none)": []})
            out.to_excel(xw, sheet_name=sheet, index=False)
            ws = xw.sheets[sheet]
            ws.freeze_panes = "A2"
            for col in ws.columns:
                width = max((len(str(c.value)) for c in col if c.value is not None), default=8)
                ws.column_dimensions[col[0].column_letter].width = min(max(width + 2, 10), 42)
    return buf.getvalue()


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true",
                    help="Read-only: no Drive writes, no email; writes an HTML preview locally.")
    ap.add_argument("--no-mail", action="store_true", help="Compute + write, but skip email.")
    ap.add_argument("--asof", default=None, help="Recompute as of YYYY-MM-DD (default: today).")
    args = ap.parse_args()

    asof = (datetime.strptime(args.asof, "%Y-%m-%d").date() if args.asof else date.today())
    dry = args.dry_run
    log("=" * 64)
    log(f"PF Decision Tracker — asof {asof}{'  [DRY-RUN]' if dry else ''}")
    log("=" * 64)

    drive = get_drive()
    root_id = os.environ["GDRIVE_FOLDER_ID"]
    index_id = _folder(drive, root_id, "company_repo", "_index")
    out_id = get_or_create_subfolder(drive, index_id, "pf_tracker")
    ohlcv_id = _folder(drive, root_id, "data", "ohlcv")

    # 1. Ingest latest holdings (newest across pf_tracking/ + portfolio/).
    target = find_latest_portfolio_file(drive, root_id)
    if not target:
        log("ERROR: no holdings file in pf_tracking/ or portfolio/. Nothing to do.")
        sys.exit(1)
    log(f"Holdings file: {target['name']} (from {target.get('_folder', '?')}/)")
    pf, wcol = parse_holdings(download_bytes(drive, target["id"]), target["name"])
    if pf.empty:
        log("ERROR: could not parse holdings. Exiting.")
        sys.exit(1)
    log(f"  {len(pf)} holdings · weightage column: {wcol!r}")
    pf = resolve_symbols(drive, root_id, pf)

    # 2. Load ledgers.
    snaps = load_parquet(drive, out_id, "pf_snapshots.parquet", SNAP_COLS)
    decs = load_parquet(drive, out_id, "pf_decisions.parquet", DEC_COLS)
    price_cache: dict = {}

    # 3. Snapshot capture (change-gated) + decision diff.
    snaps, changed = append_snapshot(drive, out_id, snaps, pf, asof, target["name"], dry)
    decs_before = len(decs)
    decs = derive_decisions(snaps, decs, asof, price_cache, drive, ohlcv_id)
    if not dry and len(decs) != decs_before:
        save_parquet(drive, out_id, "pf_decisions.parquet", decs)

    # 4. Compute everything.
    idx, contrib = compute_index(snaps, price_cache, drive, ohlcv_id, asof)
    periods = index_period_returns(idx)
    movers = compute_movers(drive, root_id, pf)
    score = score_decisions(decs, snaps, price_cache, drive, ohlcv_id, asof)
    results = _read_drive_parquet(drive, index_id, "results.parquet")
    relook = sold_relook(decs, price_cache, drive, ohlcv_id, results, asof)

    # 5. Persist derived tables (overwrite; idempotent).
    if not dry:
        save_parquet(drive, out_id, "pf_index.parquet", idx)
        save_parquet(drive, out_id, "pf_decision_scorecard.parquet", score)
        save_parquet(drive, out_id, "pf_movers.parquet", movers)
        log("  Wrote pf_index / pf_decision_scorecard / pf_movers.")

    # 6. Digest (HTML findings) + Excel workbook (detailed tables).
    html = build_html(asof, wcol, pf, idx, periods, contrib, movers, score, relook, decs)
    subject = f"PF Decision Tracker — {asof} ({len(pf)} holdings)"
    xlsx = build_excel_bytes(pf, idx, movers, score, relook, decs)
    xlsx_name = f"pf_tracker_{asof}.xlsx"
    XLSX_MIME = "vnd.openxmlformats-officedocument.spreadsheetml.sheet"

    if dry:
        LOCAL_OUT.mkdir(parents=True, exist_ok=True)
        (LOCAL_OUT / f"pf_tracker_preview_{asof}.html").write_text(html, encoding="utf-8")
        (LOCAL_OUT / xlsx_name).write_bytes(xlsx)
        log(f"[DRY-RUN] preview HTML + {xlsx_name} -> {LOCAL_OUT}")
        log(f"[DRY-RUN] index rows={len(idx)} decisions={len(decs)} "
            f"scored={len(score)} movers={len(movers)} relook={len(relook)}")
        return

    if args.no_mail:
        log("Skipping email (--no-mail).")
        return
    if not load_mail_settings(drive, index_id).get(MAIL_KEY, True):
        log(f"Mail '{MAIL_KEY}' toggled OFF — not sending.")
        return
    send_email(subject, html, attachments=[(xlsx_name, xlsx, XLSX_MIME)])
    log("Done.")


if __name__ == "__main__":
    main()
