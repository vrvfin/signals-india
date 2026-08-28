"""
pf_decision_tracker.py — Portfolio decision & performance tracker (Workflow: PF).

Answers, in PLAIN, COUNTABLE, non-black-box terms, every daily run:
  1. How is my PF moving?           -> weighted PF index (rebased to 100).
  2. Movers most/least              -> rank holdings over 1/2/3/6/12-month windows.
  3. Are my BUY/SELL/HOLD calls right? -> forward return of each decision vs the
     MEDIAN & QUARTILES of my OWN BOOK over the same window (no composite score).
  4. Relook sold stocks             -> "what if I hadn't sold?": return since exit vs
     what my OWN PF earned over the identical days, priced in portfolio points.
  5. Like-for-like vs the index     -> every stock, held or sold, measured from the day
     it entered the book, with NIFTY 500 / SMALLCAP 100 over the SAME days beside it.
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
    python scripts/pf_decision_tracker.py --no-write         # send a PREVIEW email, write nothing
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

HOLD_COLS  = ["rank", "symbol", "isin", "name", "weight_pct", "first_seen",
              "anchor_date", "days_held", "full_window", "ret_since_pct",
              "pace_wk_pct", "vs_median_pp", "vs_bench_pp",
              "n500_ret_pct", "vs_n500_pp", "sc100_ret_pct", "vs_sc100_pp",
              "contrib_pp",
              "contrib_share_pct", "contrib_rank", "left_on_table_pp",
              "return_1m_pct", "return_2m_pct", "return_3m_pct", "return_6m_pct",
              "return_12m_pct", "flat_5d", "flat_10d", "neg_windows",
              "below_med_windows", "below_target", "flags"]

# Closed positions. Every window is measured from dates this book actually records:
# spell_start (the day it entered) and sell_date (the day it left).
EXIT_COLS  = ["symbol", "isin", "name", "first_seen", "sell_date",
              "days_held", "days_since_sell", "weight_at_sell",
              "realised_pct", "since_sell_pct", "full_hold_pct",
              "pf_since_sell_pct", "opportunity_pp",
              "n500_since_sell_pct", "vs_n500_pp",
              "sc100_since_sell_pct", "vs_sc100_pp",
              "latest_yoy_pct", "verdict"]

SPELL_COLS = ["isin", "symbol", "name", "spell_start", "spell_end", "status",
              "weight_at_entry", "weight_at_exit"]

# A corporate action (face-value split/consolidation) changes a company's ISIN.
# Identity is keyed on ISIN, so without this the old code reads the swap as a SELL
# of the whole position followed by a BUY. Persisted so an alias is auditable.
ALIAS_COLS = ["old_isin", "new_isin", "symbol", "changed_on"]

# Monthly frozen baskets: the portfolio's composition at each month start, held
# untouched. Comparing it against the real index answers "did my trading help?".
COHORT_COLS = ["cohort_month", "snapshot_date", "isin", "symbol", "weight_pct"]
COHORT_RET_COLS = ["cohort_month", "asof", "horizon_m", "days", "frozen_ret_pct",
                   "actual_ret_pct", "delta_pp", "n_priced", "n_total", "partial"]
COHORT_HORIZONS = [1, 2, 3, 6, 12]
COHORT_MAIL_N = 6                 # cohorts shown in the mail; parquet keeps them all

# ADD/TRIM band: a weight change counts as a real trade only if it exceeds the
# drift-implied weight by BOTH an absolute (pp) and a relative margin.
ADD_TRIM_ABS_PP  = 0.5
ADD_TRIM_REL     = 0.20

# Calendar length of each return window — used to GATE windows longer than a
# stock has actually been held (a 6M number is meaningless a week after you buy).
WINDOW_DAYS = {1: 30, 2: 61, 3: 91, 6: 182, 12: 365}
MIN_ANNUALISE_DAYS = 91          # don't annualise anything shorter than a quarter

# Stagnancy is defined ONLY as "how many consecutive trading days has the close
# stayed inside ±band of today's close" — two bands, no fuzzy composite rule.
FLAT_BANDS = (0.05, 0.10)


# Goal tracking: the annual return you're aiming for, expressed as the weekly pace
# it demands. 50%/yr => (1.5)^(7/365)-1 = ~0.78%/week.
TARGET_CAGR_PCT = 50.0

# Benchmarks from data/indices/ (Phase-1 `ingest_indices_macro.py`). First is primary.
BENCHMARKS = ["NIFTY_500", "NIFTY_SMALLCAP_100"]
# Short column prefixes for the per-stock index columns, so a widened table stays
# readable: NIFTY_500 -> n500_ret_pct / vs_n500_pp.
BENCH_KEY = {"NIFTY_500": "n500", "NIFTY_SMALLCAP_100": "sc100"}
PRIMARY_KEY = BENCH_KEY[BENCHMARKS[0]]

# Allocation review: a "winner" is top-quartile pace; "underweight" is below equal
# weight. The pair flags positions whose sizing capped their impact.
WINNER_Q = 0.75

# Pace over a day or two is noise amplified by the ^(7/days) exponent — a name up
# 1% on its first day annualises to ~7.5%/week. Positions younger than this are
# shown with their numbers but never FLAGGED fast / underweight / below-target,
# and are excluded from the quantile that defines "fast".
MIN_PACE_DAYS = 5
REVIEW_MIN_DAYS    = 10   # a decision must be this old before REVIEW-NOW judges it
REVIEW_MIN_MOVE_PP = 3.0  # ...and have moved this much; kills ~0% quartile noise

# ADD/TRIM band: a weight change counts as a real trade only if it exceeds the
# drift-implied weight by BOTH an absolute (pp) and a relative margin.
ADD_TRIM_ABS_PP  = 0.5
ADD_TRIM_REL     = 0.20

# Calendar length of each return window — used to GATE windows longer than a
# stock has actually been held (a 6M number is meaningless a week after you buy).
WINDOW_DAYS = {1: 30, 2: 61, 3: 91, 6: 182, 12: 365}
MIN_ANNUALISE_DAYS = 91          # don't annualise anything shorter than a quarter

# Stagnancy is defined ONLY as "how many consecutive trading days has the close
# stayed inside ±band of today's close" — two bands, no fuzzy composite rule.
FLAT_BANDS = (0.05, 0.10)

# Goal tracking: the annual return you're aiming for, expressed as the weekly pace
# it demands. 50%/yr => (1.5)^(7/365)-1 = ~0.78%/week.
TARGET_CAGR_PCT = 50.0

# Benchmarks from data/indices/ (Phase-1 `ingest_indices_macro.py`). First is primary.
BENCHMARKS = ["NIFTY_500", "NIFTY_SMALLCAP_100"]
# Short column prefixes for the per-stock index columns, so a widened table stays
# readable: NIFTY_500 -> n500_ret_pct / vs_n500_pp.
BENCH_KEY = {"NIFTY_500": "n500", "NIFTY_SMALLCAP_100": "sc100"}
PRIMARY_KEY = BENCH_KEY[BENCHMARKS[0]]

# Allocation review: a "winner" is top-quartile pace; "underweight" is below equal
# weight. The pair flags positions whose sizing capped their impact.
WINNER_Q = 0.75

# Pace over a day or two is noise amplified by the ^(7/days) exponent — a name up
# 1% on its first day annualises to ~7.5%/week. Positions younger than this are
# shown with their numbers but never FLAGGED fast / underweight / below-target,
# and are excluded from the quantile that defines "fast".
MIN_PACE_DAYS = 5
REVIEW_MIN_DAYS    = 10   # a decision must be this old before REVIEW-NOW judges it
REVIEW_MIN_MOVE_PP = 3.0  # ...and have moved this much; kills ~0% quartile noise


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


# ── Return maths: CAGR, XIRR, flat-streak ────────────────────────────────────

def cagr_pct(total_ret_pct, days) -> float | None:
    """Constant annual rate that turns start into end over `days`.
        CAGR = (End/Start)^(365/days) - 1
    Only meaningful for periods >= ~a quarter; annualising a week of noise is
    misleading, so short windows return None by design."""
    if total_ret_pct is None or not days or days < MIN_ANNUALISE_DAYS:
        return None
    try:
        return float(((1.0 + total_ret_pct / 100.0) ** (365.0 / days) - 1.0) * 100.0)
    except Exception:
        return None


def pace_wk_pct(total_ret_pct, days) -> float | None:
    """Compounded % PER WEEK — the one yardstick that makes different holding
    periods comparable:  pace = (1 + ret)^(7/days) - 1

        +100% over 30d  -> ~17.6%/wk        +100% over 365d -> ~1.34%/wk
        +50%  over 365d -> ~0.78%/wk  (the 50%-a-year goal, in weekly terms)

    Unlike CAGR this stays legible on short windows (a 2%-in-2-days move reads as
    ~7%/wk, not a 4-digit annualised number), so it is safe to rank on."""
    if total_ret_pct is None or days is None or days < 1:
        return None
    try:
        base = 1.0 + float(total_ret_pct) / 100.0
        if base <= 0:
            return None
        return float((base ** (7.0 / float(days)) - 1.0) * 100.0)
    except Exception:
        return None


def required_wk_pct(target_annual_pct: float = TARGET_CAGR_PCT) -> float:
    """Weekly pace an annual return target demands. 50%/yr -> ~0.78%/wk."""
    return float(((1.0 + target_annual_pct / 100.0) ** (7.0 / 365.0) - 1.0) * 100.0)


def xirr_pct(flows) -> float | None:
    """Money-weighted annual rate: the r solving  Σ CFi / (1+r)^(daysi/365) = 0.
    No closed form — solved by bisection. flows = [(date, amount)], negatives are
    money in. NOTE: with a weightage-only export the only visible flows are
    (-100 at inception, +level today), so this necessarily equals since-inception
    CAGR; it diverges only once real dated contributions exist."""
    flows = [(pd.Timestamp(d), float(a)) for d, a in flows if a is not None]
    if len(flows) < 2 or not (any(a < 0 for _, a in flows) and any(a > 0 for _, a in flows)):
        return None
    t0 = min(d for d, _ in flows)

    def npv(r):
        return sum(a / ((1.0 + r) ** ((d - t0).days / 365.0)) for d, a in flows)

    lo, hi = -0.9999, 10.0
    f_lo, f_hi = npv(lo), npv(hi)
    if f_lo * f_hi > 0:
        return None
    for _ in range(200):
        mid = (lo + hi) / 2.0
        f_mid = npv(mid)
        if abs(f_mid) < 1e-10:
            return float(mid * 100.0)
        if f_lo * f_mid < 0:
            hi, f_hi = mid, f_mid
        else:
            lo, f_lo = mid, f_mid
    return float((lo + hi) / 2.0 * 100.0)


def flat_days(ser: pd.Series | None, band: float = FLAT_BANDS[0]) -> int | None:
    """Consecutive trading days, counting back from the latest bar, that the close
    stayed inside ±band of today's close — i.e. 'how long has this gone nowhere'."""
    if ser is None or len(ser) < 2:
        return None
    last = float(ser.iloc[-1])
    if last == 0:
        return None
    n = 0
    for v in ser.iloc[::-1]:
        if abs(float(v) / last - 1.0) <= band:
            n += 1
        else:
            break
    return n


def load_benchmark(drive, root_id, name: str) -> pd.Series | None:
    """Close series for a Phase-1 index parquet (data/indices/<NAME>.parquet).
    Same columns as OHLCV, so it reuses the identical reader."""
    try:
        idx_id = _folder(drive, root_id, "data", "indices")
        df = _read_drive_parquet(drive, idx_id, f"{name}.parquet")
        if df is None or df.empty or "close" not in df.columns:
            return None
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"])
        return df.sort_values("date").set_index("date")["close"].astype(float)
    except Exception as e:
        log(f"  benchmark {name}: unavailable ({str(e)[:60]})")
        return None


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


# ── ISIN continuity across corporate actions ─────────────────────────────────

def build_isin_alias(snaps: pd.DataFrame) -> pd.DataFrame:
    """Detect ISIN changes so a corporate action doesn't read as a sell + a buy.

    TDPOWERSYS moved INE419M01027 -> INE419M01035 on 2026-08-26. Nothing was
    traded, but keying identity on ISIN made the old one's disappearance a SELL of
    a 9.65% position and the new one's arrival a BUY.

    Rule: on snapshot date D, if ISIN A was present at D-1 and absent at D, while
    ISIN B was absent at D-1 and present at D, and both carry the same non-blank
    symbol, then B is the same position as A. Returns the alias table; the map is
    applied by `apply_isin_alias` before spells/decisions/cohorts are derived.

    Deliberately conservative: it fires only on a same-day swap of a shared symbol,
    and the result is persisted so a wrong alias is visible rather than silent."""
    if snaps.empty:
        return pd.DataFrame(columns=ALIAS_COLS)
    dates = sorted(snaps["snapshot_date"].unique())
    by_date = {d: snaps[snaps["snapshot_date"] == d] for d in dates}
    rows = []
    for prev_d, cur_d in zip(dates, dates[1:]):
        prev, cur = by_date[prev_d], by_date[cur_d]
        gone = set(prev["isin"]) - set(cur["isin"])
        new = set(cur["isin"]) - set(prev["isin"])
        if not gone or not new:
            continue
        psym = {r.isin: str(r.symbol).strip().upper() for r in prev.itertuples()}
        csym = {r.isin: str(r.symbol).strip().upper() for r in cur.itertuples()}
        for a in gone:
            s = psym.get(a, "")
            if not s or s == "NAN":
                continue
            match = [b for b in new if csym.get(b, "") == s]
            if len(match) == 1:                       # unambiguous swap only
                rows.append(dict(old_isin=a, new_isin=match[0], symbol=s,
                                 changed_on=cur_d))
    return pd.DataFrame(rows, columns=ALIAS_COLS)


def apply_isin_alias(df: pd.DataFrame, alias: pd.DataFrame) -> pd.DataFrame:
    """Rewrite every new_isin back to its original isin, so one company keeps one
    identity for its whole life. Chains are followed (A->B->C all become A)."""
    if df is None or df.empty or alias is None or alias.empty or "isin" not in df.columns:
        return df
    m = dict(zip(alias["new_isin"], alias["old_isin"]))
    def _root(i, seen=None):
        seen = seen or set()
        while i in m and i not in seen:
            seen.add(i); i = m[i]
        return i
    out = df.copy()
    out["isin"] = out["isin"].map(_root)
    return out


# ── Holding spells (contiguous ownership runs) ───────────────────────────────

def build_spells(snaps: pd.DataFrame, decs: pd.DataFrame) -> pd.DataFrame:
    """Split the snapshot ledger into CONTIGUOUS ownership runs, one row per
    (isin, spell): spell_start .. spell_end.

    A position can be sold and bought back. Anchoring "since first seen" on the
    EARLIEST-ever snapshot would then measure across the months it wasn't owned —
    so every per-stock window in this report anchors on the CURRENT spell instead.
    A spell breaks the moment the ISIN is missing from a snapshot date, which is the
    same observation `derive_decisions` turns into a SELL event; `weight_at_exit`
    is read off that SELL row, falling back to the last weight actually observed.

    status = HELD (spell_end None, still in the latest snapshot) or SOLD."""
    if snaps.empty:
        return pd.DataFrame(columns=SPELL_COLS)

    all_dates = sorted(snaps["snapshot_date"].unique())
    exit_w = {}
    if decs is not None and not decs.empty:
        for r in decs[decs["action"] == "SELL"].itertuples():
            exit_w[(r.isin, r.event_date)] = r.weight_before

    rows = []
    for isin, g in snaps.groupby("isin"):
        seen = dict(zip(g["snapshot_date"], g["weight_pct"]))
        sym = str(g["symbol"].iloc[-1])
        nm = str(g["name"].iloc[-1])

        def _close(start, end_date, status):
            last_w = seen.get(max(d for d in seen if d < end_date)) \
                if (end_date and any(d < end_date for d in seen)) else seen.get(start)
            rows.append(dict(
                isin=isin, symbol=sym, name=nm, spell_start=start, spell_end=end_date,
                status=status, weight_at_entry=seen.get(start),
                weight_at_exit=(np.nan if status == "HELD"
                                else exit_w.get((isin, end_date), last_w))))

        start = None
        for d in all_dates:
            if d in seen and start is None:
                start = d
            elif d not in seen and start is not None:
                _close(start, d, "SOLD")       # `d` = the snapshot it vanished on
                start = None
        if start is not None:
            _close(start, None, "HELD")

    return pd.DataFrame(rows, columns=SPELL_COLS)


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
    # The index starts at the FIRST REAL SNAPSHOT and never before it. An earlier
    # build back-projected today's weights over ~12M of price history; that number
    # was a backtest of the current book, not this portfolio's record, so it is gone.
    # Consequence by design: windows longer than the tracked history stay blank
    # until enough real days accrue.
    start = pd.Timestamp(snaps["snapshot_date"].min())
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


PERIOD_OFFSETS = {"1D": relativedelta(days=1), "1W": relativedelta(weeks=1),
                  "1M": relativedelta(months=1), "2M": relativedelta(months=2),
                  "3M": relativedelta(months=3), "6M": relativedelta(months=6),
                  "12M": relativedelta(months=12)}


def index_period_returns(idx: pd.DataFrame, benches: dict | None = None) -> dict:
    """Total return per horizon off the index level series, plus weekly pace, the
    annualised (CAGR) view, since-inception XIRR, and the same windows for each
    benchmark with the PF-minus-benchmark alpha.

    A window is reported ONLY if the tracked history actually covers it — with the
    back-projection removed, anything older than the first snapshot is blank rather
    than silently measured off a shorter span."""
    if idx.empty:
        return {}
    s = idx.set_index(pd.to_datetime(idx["date"]))["pf_index"].sort_index()
    last = float(s.iloc[-1]); end = s.index[-1]
    start_ts, start_lvl = s.index[0], float(s.iloc[0])
    benches = benches or {}

    def window_ret(ser, base_ts):
        """% move of `ser` from base_ts -> end, or None if base_ts predates it."""
        if ser is None or ser.empty or base_ts < ser.index[0]:
            return None
        p0, p1 = ser.asof(base_ts), ser.asof(end)
        if pd.isna(p0) or pd.isna(p1) or not p0:
            return None
        return float((p1 / p0 - 1) * 100)

    ret, cagr, pace, days = {}, {}, {}, {}
    bret = {b: {} for b in benches}
    alpha = {b: {} for b in benches}
    for lbl, off in PERIOD_OFFSETS.items():
        base_ts = end - off
        r = window_ret(s, base_ts)
        ret[lbl] = r
        days[lbl] = (end - base_ts).days if r is not None else None
        cagr[lbl] = cagr_pct(r, days[lbl])
        pace[lbl] = pace_wk_pct(r, days[lbl])
        for b, ser in benches.items():
            br = window_ret(ser, base_ts) if r is not None else None
            bret[b][lbl] = br
            alpha[b][lbl] = (r - br) if (r is not None and br is not None) else None

    si_days = (end - start_ts).days
    ret["SI"] = float((last / start_lvl - 1) * 100) if start_lvl else None
    days["SI"] = si_days
    cagr["SI"] = cagr_pct(ret["SI"], si_days)
    pace["SI"] = pace_wk_pct(ret["SI"], si_days)
    for b, ser in benches.items():
        br = window_ret(ser, start_ts)
        bret[b]["SI"] = br
        alpha[b]["SI"] = (ret["SI"] - br) if (ret["SI"] is not None and br is not None) else None

    return {
        "level": last, "start_level": start_lvl,
        "start_date": start_ts.strftime("%Y-%m-%d"), "end_date": end.strftime("%Y-%m-%d"),
        "si_days": si_days, "ret": ret, "cagr": cagr, "pace": pace, "days": days,
        "bret": bret, "alpha": alpha, "bench_names": list(benches),
        # Only visible cashflows in a weightage-only book: money in at inception,
        # value out today. Gated to >= a quarter — annualising a fortnight is noise.
        "xirr": (xirr_pct([(start_ts, -start_lvl), (end, last)])
                 if si_days >= MIN_ANNUALISE_DAYS else None),
    }


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


def compute_holdings_view(pf, snaps, movers, price_cache, drive, ohlcv_id,
                          asof: date, bench_map=None, spells=None) -> pd.DataFrame:
    """Per-holding table where EVERY number is measured from the day the position
    actually entered the tracked book — never earlier, never from a common date the
    stock wasn't held on.

        base_date   = first snapshot we hold (start of PF reporting)
        first_seen  = start of the CURRENT holding spell for this ISIN (see
                      build_spells) — for a name sold and bought back this is the
                      re-entry, not the original purchase, so the window never spans
                      months the position wasn't owned
        anchor_date = max(base_date, first_seen)

    Each benchmark in `bench_map` is measured over that SAME anchor→asof window, so
    `<key>_ret_pct` answers "what the index did over exactly the days I held this"
    and `vs_<key>_pp` is the like-for-like gap.

    So a name added on 6-Aug is measured from 6-Aug, not from the base date. Because
    that leaves every stock on a different-length window, the cross-stock comparator
    is `pace_wk_pct` (compounded %/week), which is window-neutral; the raw
    `ret_since_pct` is the honest "what this position actually did for me".

    Contribution answers the allocation question: `contrib_pp` is the points of PF
    growth the position actually delivered (weight x return), so a big winner held
    small shows a high return next to a small contribution."""
    if pf.empty:
        return pd.DataFrame(columns=HOLD_COLS)
    ret_cols = list(FEAT_RET_COLS.values())
    v = pf[["symbol", "isin", "name", "weight_pct"]].merge(
        movers[["isin"] + ret_cols], on="isin", how="left")

    base_date = snaps["snapshot_date"].min() if not snaps.empty else str(asof)
    # Current spell only — never the earliest-ever appearance (see build_spells).
    if spells is not None and not spells.empty:
        first_map = {r.isin: r.spell_start
                     for r in spells[spells["status"] == "HELD"].itertuples()}
    else:
        first_map = (snaps.groupby("isin")["snapshot_date"].min().to_dict()
                     if not snaps.empty else {})
    bench_map = bench_map or {}
    end_ts = pd.Timestamp(asof)
    req_wk = required_wk_pct()

    rows = []
    for r in v.itertuples():
        first_seen = str(first_map.get(r.isin, str(asof)))
        anchor = max(first_seen, str(base_date))       # never measure before we held it
        anchor_ts = pd.Timestamp(anchor)
        days_held = max((end_ts - anchor_ts).days, 0)

        ser = load_close_series(drive, ohlcv_id, r.symbol, price_cache)
        since = fwd_return(ser, anchor_ts, end_ts)
        pace = pace_wk_pct(since, days_held)

        # Each index measured over THIS stock's own holding window — like for like.
        bcols = {}
        for bname, bser in bench_map.items():
            k = BENCH_KEY.get(bname, bname.lower())
            b = fwd_return(bser, anchor_ts, end_ts)
            bcols[f"{k}_ret_pct"] = b if b is not None else np.nan
            bcols[f"vs_{k}_pp"] = ((since - b) if (since is not None and b is not None)
                                   else np.nan)

        wt = float(r.weight_pct) if pd.notna(r.weight_pct) else np.nan
        contrib = (wt / 100.0 * since) if (pd.notna(wt) and since is not None) else np.nan

        # Trailing market windows are CONTEXT only — they pre-date the holding and
        # are never used for the PF median or the ranking.
        wins = {col: (float(getattr(r, col)) if pd.notna(getattr(r, col, np.nan)) else np.nan)
                for col in ret_cols}

        rows.append(dict(
            symbol=r.symbol, isin=r.isin, name=r.name, weight_pct=wt,
            first_seen=first_seen, anchor_date=anchor, days_held=days_held,
            full_window=bool(anchor == str(base_date)),
            ret_since_pct=since if since is not None else np.nan,
            pace_wk_pct=pace if pace is not None else np.nan,
            # Kept as the primary-benchmark alias so existing readers don't shift.
            vs_bench_pp=bcols.get(f"vs_{PRIMARY_KEY}_pp", np.nan),
            contrib_pp=contrib,
            flat_5d=flat_days(ser, FLAT_BANDS[0]), flat_10d=flat_days(ser, FLAT_BANDS[1]),
            **bcols, **wins))

    h = pd.DataFrame(rows)
    # A benchmark parquet can be missing on Drive — keep the schema stable regardless.
    for k in BENCH_KEY.values():
        for c in (f"{k}_ret_pct", f"vs_{k}_pp"):
            if c not in h.columns:
                h[c] = np.nan

    # Trailing-window fail counts: each window judged against ITS OWN median, so the
    # comparison inside a column is always like-for-like.
    neg = pd.Series(0, index=h.index); below = pd.Series(0, index=h.index)
    for col in ret_cols:
        c = h[col]
        neg += (c <= 0).fillna(False).astype(int)
        if c.notna().sum() >= 4:
            below += (c < c.median()).fillna(False).astype(int)
    h["neg_windows"], h["below_med_windows"] = neg, below

    # THE book median = median weekly PACE. Window-neutral, so a 2-day-old position
    # and a since-base one sit on the same scale.
    med_pace = h["pace_wk_pct"].median()
    h["vs_median_pp"] = h["pace_wk_pct"] - med_pace

    tot = h["contrib_pp"].sum(skipna=True)
    h["contrib_share_pct"] = (h["contrib_pp"] / tot * 100.0) if tot else np.nan

    # Allocation cost, measured against EQUAL weight (100/N), not the median weight.
    # On a long-tailed book the median position is tiny (0.5% here), so a median
    # counterfactual makes every gap look negligible; equal weight is the neutral
    # "no view on sizing" baseline and gives an honest sense of scale.
    # This is arithmetic, not advice.
    eq_wt = 100.0 / len(h) if len(h) else np.nan
    gap_wt = (eq_wt - h["weight_pct"]).clip(lower=0)
    h["left_on_table_pp"] = np.where(h["ret_since_pct"] > 0,
                                     gap_wt / 100.0 * h["ret_since_pct"], np.nan)

    # Rank gap: performs like #3 but contributes like #40 => sizing is the reason.
    h["contrib_rank"] = h["contrib_pp"].rank(ascending=False, method="min")

    # Only positions old enough for pace to mean anything get rated (see MIN_PACE_DAYS).
    ratable = h["days_held"] >= MIN_PACE_DAYS
    h["below_target"] = ratable & h["pace_wk_pct"].notna() & (h["pace_wk_pct"] < req_wk)
    rated_pace = h.loc[ratable, "pace_wk_pct"]
    pace_q = rated_pace.quantile(WINNER_Q) if rated_pace.notna().any() else np.nan

    def _flags(row):
        f = []
        young = row["days_held"] < MIN_PACE_DAYS
        if young:
            f.append(f"new({int(row['days_held'])}d)")
        fast = (not young) and pd.notna(pace_q) and row["pace_wk_pct"] >= pace_q
        if fast:
            f.append("fast")
        if fast and pd.notna(eq_wt) and row["weight_pct"] < eq_wt:
            f.append("underweight-winner")
        if row["below_target"]:
            f.append("below-target")
        if pd.notna(row["ret_since_pct"]) and row["ret_since_pct"] < 0:
            f.append("negative")
        if row["flat_10d"] and row["flat_10d"] >= 30:
            f.append(f"flat{int(row['flat_10d'])}d")
        return ",".join(f)

    h["flags"] = h.apply(_flags, axis=1)
    h = h.sort_values("pace_wk_pct", ascending=False, na_position="last").reset_index(drop=True)
    h["rank"] = np.arange(1, len(h) + 1)
    return h[HOLD_COLS]


def matched_bench_returns(hold: pd.DataFrame) -> dict:
    """Weight-average each holding's own-window return against the index's return over
    THE SAME days:

        matched_pf    = Σ wᵢ·rᵢ / Σ wᵢ     (rᵢ over stock i's anchor→asof)
        matched_bench = Σ wᵢ·bᵢ / Σ wᵢ     (bᵢ over the identical window)

    READ THIS CAREFULLY before using the number: at whole-portfolio level a
    fully-invested book's "index adjusted for holding days" collapses to the plain
    index over the tracked span, because the weights renormalise to 1 every day. This
    weighted version differs from the plain index ONLY because it re-weights the
    index's sub-periods by WHEN each name was entered. So it isolates entry timing —
    it is not a second source of alpha, and the plain-index alpha in section 1
    remains the headline. Positions are dropped from a benchmark's average when
    either leg is missing, so both legs always cover the same names."""
    out = {}
    if hold is None or hold.empty:
        return out
    for bname, k in BENCH_KEY.items():
        bcol = f"{k}_ret_pct"
        if bcol not in hold.columns:
            continue
        d = hold[["weight_pct", "ret_since_pct", bcol]].dropna()
        w = d["weight_pct"].sum()
        if d.empty or w <= 0:
            continue
        pf_r = float((d["weight_pct"] * d["ret_since_pct"]).sum() / w)
        bn_r = float((d["weight_pct"] * d[bcol]).sum() / w)
        out[bname] = {"pf": pf_r, "bench": bn_r, "alpha": pf_r - bn_r, "n": int(len(d))}
    return out


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

def compute_exits_view(spells: pd.DataFrame, price_cache: dict, drive, ohlcv_id,
                       idx: pd.DataFrame, results: pd.DataFrame, asof: date,
                       bench_map=None, lookback_m: int = 12) -> pd.DataFrame:
    """Closed positions, answering "what would I have if I hadn't sold?" in the only
    honest baseline a weightage-only book has: your OWN portfolio.

    Selling didn't put the money under a mattress — it went back into the book. So the
    counterfactual for a sold name is the PF index over the identical days, not zero:

        opportunity_pp = weight_at_sell/100 × (since_sell_pct − pf_since_sell_pct)

    Positive = points of PF growth given up by selling; negative = points dodged.
    Three windows are reported per exit, all off dates the ledger actually records:
    `realised_pct` (entry→sell, what the position made), `since_sell_pct`
    (sell→today, what it did once you were out) and `full_hold_pct` (entry→today,
    where you'd be had you never sold). Each benchmark is measured over the same
    sell→today window, so the index comparison is like for like."""
    cols = EXIT_COLS
    if spells is None or spells.empty:
        return pd.DataFrame(columns=cols)
    sold = spells[(spells["status"] == "SOLD") & spells["spell_end"].notna()].copy()
    if sold.empty:
        return pd.DataFrame(columns=cols)
    cutoff = pd.Timestamp(asof) - relativedelta(months=lookback_m)
    sold = sold[pd.to_datetime(sold["spell_end"]) >= cutoff]
    if sold.empty:
        return pd.DataFrame(columns=cols)

    bench_map = bench_map or {}
    end_ts = pd.Timestamp(asof)
    pf_ser = None
    if idx is not None and not idx.empty and "pf_index" in idx.columns:
        pf_ser = (idx.assign(_d=pd.to_datetime(idx["date"]))
                     .set_index("_d")["pf_index"].astype(float).sort_index())

    yoy_map = {}
    if not results.empty and {"symbol", "yoy_pct"} <= set(results.columns):
        pcol = "period" if "period" in results.columns else None
        for sym, g in results.groupby("symbol"):
            if pcol:
                g = g.sort_values(pcol)
            yoy_map[sym] = pd.to_numeric(g["yoy_pct"], errors="coerce").dropna().iloc[-1] \
                if g["yoy_pct"].notna().any() else None
    rows = []
    for ev in sold.itertuples():
        ser = load_close_series(drive, ohlcv_id, ev.symbol, price_cache)
        start_ts, sell_ts = pd.Timestamp(ev.spell_start), pd.Timestamp(ev.spell_end)

        realised = fwd_return(ser, start_ts, sell_ts)
        since = fwd_return(ser, sell_ts, end_ts)
        full = fwd_return(ser, start_ts, end_ts)
        pf_since = fwd_return(pf_ser, sell_ts, end_ts) if pf_ser is not None else None

        wt = float(ev.weight_at_exit) if pd.notna(ev.weight_at_exit) else np.nan
        opp = (wt / 100.0 * (since - pf_since)) \
            if (pd.notna(wt) and since is not None and pf_since is not None) else np.nan

        bcols = {}
        for bname, bser in bench_map.items():
            k = BENCH_KEY.get(bname, bname.lower())
            b = fwd_return(bser, sell_ts, end_ts)
            bcols[f"{k}_since_sell_pct"] = b if b is not None else np.nan
            bcols[f"vs_{k}_pp"] = ((since - b) if (since is not None and b is not None)
                                   else np.nan)

        # Verdict is judged against the book the money moved INTO, not against zero —
        # a name up 8% while the PF did 20% was still the right thing to sell.
        yoy = yoy_map.get(ev.symbol)
        gap = (since - pf_since) if (since is not None and pf_since is not None) else None
        if since is None:
            v = "n/a"
        elif gap is None:
            v = "validated exit" if since <= 0 else ("regret — ran up" if since >= 15
                                                     else "neutral")
        elif gap <= -5 or (gap <= 0 and yoy is not None and yoy < 0):
            v = "validated exit"          # lagged the book you moved into
        elif gap >= 10:
            v = "regret — ran up"
        else:
            v = "neutral"

        rows.append(dict(
            symbol=ev.symbol, isin=ev.isin, name=ev.name,
            first_seen=ev.spell_start, sell_date=ev.spell_end,
            days_held=max((sell_ts - start_ts).days, 0),
            days_since_sell=max((end_ts - sell_ts).days, 0),
            weight_at_sell=wt,
            realised_pct=realised if realised is not None else np.nan,
            since_sell_pct=since if since is not None else np.nan,
            full_hold_pct=full if full is not None else np.nan,
            pf_since_sell_pct=pf_since if pf_since is not None else np.nan,
            opportunity_pp=opp, latest_yoy_pct=yoy, verdict=v, **bcols))

    out = pd.DataFrame(rows)
    for c in cols:
        if c not in out.columns:
            out[c] = np.nan
    # Biggest regret first — the exits that cost the most PF growth.
    return out[cols].sort_values("opportunity_pp", ascending=False, na_position="last")


# ── Monthly cohorts: the frozen-basket counterfactual ────────────────────────

def build_cohorts(snaps: pd.DataFrame, cohorts: pd.DataFrame) -> tuple[pd.DataFrame, bool]:
    """Freeze the book's composition at each month start. APPEND-ONLY — a cohort is
    a historical fact and is never recomputed once written.

    The cohort date is the FIRST SNAPSHOT ON/AFTER the 1st, because there is no
    snapshot every day. Since snapshot dates are sorted, the first date seen for a
    month IS that snapshot. The earliest cohort therefore starts at inception
    (2026-07-23), which doubles as the since-inception frozen basket."""
    cols = COHORT_COLS
    if snaps.empty:
        return (cohorts if cohorts is not None else pd.DataFrame(columns=cols)), False
    have = set(cohorts["cohort_month"]) if cohorts is not None and not cohorts.empty else set()
    rows = []
    for d in sorted(snaps["snapshot_date"].unique()):
        m = str(d)[:7]
        if m in have:
            continue
        have.add(m)
        for r in snaps[snaps["snapshot_date"] == d].itertuples():
            rows.append(dict(cohort_month=m, snapshot_date=d, isin=r.isin,
                             symbol=r.symbol, weight_pct=r.weight_pct))
    if not rows:
        return cohorts, False
    out = pd.concat([cohorts if cohorts is not None else pd.DataFrame(columns=cols),
                     pd.DataFrame(rows, columns=cols)], ignore_index=True)
    new_months = sorted({r["cohort_month"] for r in rows})
    log(f"  Cohorts: froze {len(new_months)} new basket(s) — {', '.join(new_months)}")
    return out[cols], True


def _frozen_return(members: pd.DataFrame, start_ts, end_ts, price_cache,
                   drive, ohlcv_id) -> tuple[float | None, int, int]:
    """Buy-and-hold return of a frozen basket: put wᵢ into each name at the cohort
    date and never touch it again, so

        return = Σ wᵢ·rᵢ / Σ wᵢ

    which is exactly the value of that basket today versus its cost. Names without
    a price are dropped from BOTH sides of the ratio, so the number is never
    silently diluted — `n_priced` vs `n_total` exposes how many were dropped."""
    num = den = 0.0
    n_priced = 0
    for r in members.itertuples():
        w = float(r.weight_pct) if pd.notna(r.weight_pct) else 0.0
        if w <= 0:
            continue
        ret = fwd_return(load_close_series(drive, ohlcv_id, r.symbol, price_cache),
                         start_ts, end_ts)
        if ret is None:
            continue
        num += w * ret
        den += w
        n_priced += 1
    return ((num / den) if den > 0 else None), n_priced, len(members)


def compute_cohort_returns(cohorts: pd.DataFrame, idx: pd.DataFrame, price_cache: dict,
                           drive, ohlcv_id, asof: date) -> pd.DataFrame:
    """Frozen basket vs what the portfolio ACTUALLY did, per cohort × horizon.

        delta_pp = actual − frozen

    Positive means the trading you did since that month start beat leaving the book
    alone; negative means doing nothing would have won. Fully derived from the
    cohorts + prices + pf_index, so it is regenerated (overwritten) every run.

    Horizons that have not elapsed produce NO ROW — the mail renders those cells
    blank rather than showing a zero that reads as "no difference"."""
    if cohorts is None or cohorts.empty:
        return pd.DataFrame(columns=COHORT_RET_COLS)
    pf_ser = None
    if idx is not None and not idx.empty and "pf_index" in idx.columns:
        pf_ser = (idx.assign(_d=pd.to_datetime(idx["date"]))
                     .set_index("_d")["pf_index"].astype(float).sort_index())
    end_ts = pd.Timestamp(asof)
    rows = []
    for m, g in cohorts.groupby("cohort_month"):
        start = str(g["snapshot_date"].iloc[0])
        start_ts = pd.Timestamp(start)
        windows = [(h, start_ts + relativedelta(months=h), False) for h in COHORT_HORIZONS]
        windows = [w for w in windows if w[1] <= end_ts]
        windows.append((np.nan, end_ts, True))          # the live, partial window
        for h, win_end, partial in windows:
            frozen, n_priced, n_total = _frozen_return(
                g, start_ts, win_end, price_cache, drive, ohlcv_id)
            actual = fwd_return(pf_ser, start_ts, win_end) if pf_ser is not None else None
            rows.append(dict(
                cohort_month=m, asof=str(asof), horizon_m=h,
                days=int((win_end - start_ts).days),
                frozen_ret_pct=frozen if frozen is not None else np.nan,
                actual_ret_pct=actual if actual is not None else np.nan,
                delta_pp=((actual - frozen) if (actual is not None and frozen is not None)
                          else np.nan),
                n_priced=n_priced, n_total=n_total, partial=partial))
    return pd.DataFrame(rows, columns=COHORT_RET_COLS)


# ── HTML digest ──────────────────────────────────────────────────────────────

def _pct(v, signed=True):
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return "—"
    col = "#1a7a3c" if v >= 0 else "#c0392b"
    return f"<span style='color:{col}'>{v:+.1f}%</span>" if signed else f"{v:.1f}%"


# The tables repeat the same per-cell padding ~2,000 times; that alone is ~40 KB and
# pushes the body past Gmail's ~102 KB clip, which silently truncates the mail.
#
# An earlier version hoisted these into a <style> block. That was wrong twice over:
# `mailer.send_email` sends the body as a BARE FRAGMENT (no <html>/<head>), so clients
# that drop stray <style> lost every bit of padding AND colour; and `_strip_html`
# strips tags but not tag CONTENTS, so the raw CSS became the first 400 characters of
# the plain-text alternative — which is exactly what Gmail shows as the inbox preview
# line. Hence: no <style> anywhere.
#
# Instead the padding moves to the table's `cellpadding` attribute — plain HTML that
# no mail client strips — and only colour stays inline, where it always survives.
_TABLE_OPEN = ("<table style='border-collapse:collapse",
               "<table cellpadding='4' cellspacing='0' style='border-collapse:collapse")
_STYLE_TRIM = [
    ("align='right' style='padding:2px 6px;color:#888'",   "align='right' style='color:#888'"),
    ("align='right' style='padding:2px 8px;color:#888'",   "align='right' style='color:#888'"),
    ("align='right' style='padding:2px 6px;color:#777'",   "align='right' style='color:#777'"),
    ("align='center' style='padding:2px 6px;color:#777'",  "align='center' style='color:#777'"),
    ("align='center' style='padding:3px 10px;color:#555'", "align='center' style='color:#555'"),
    ("align='right' style='padding:2px 6px'",   "align='right'"),
    ("align='right' style='padding:2px 8px'",   "align='right'"),
    ("align='center' style='padding:2px 6px'",  "align='center'"),
    ("align='center' style='padding:3px 10px'", "align='center'"),
    ("align='left' style='padding:3px 8px'",    "align='left'"),
    ("style='padding:2px 6px;font-size:11px'",  "style='font-size:11px'"),
    ("style='padding:2px 8px'", ""),
    ("style='padding:2px 6px'", ""),
    ("style='padding:3px 8px'", ""),
    (" style='padding:3px 10px;text-align:center'", " align='center'"),
]


def shrink_html(html: str) -> str:
    """Drop the repeated per-cell padding (now carried by the table's cellpadding) so
    the body stays under Gmail's clip limit. Colour stays inline. Emits no <style>:
    see the note above for why that broke both the layout and the preview line."""
    html = html.replace(*_TABLE_OPEN)
    for literal, repl in sorted(_STYLE_TRIM, key=lambda x: -len(x[0])):
        html = html.replace(literal, repl)
    return html.replace("<td >", "<td>").replace("<th >", "<th>")


def build_html(asof, wcol, pf, idx, periods, contrib, movers, score, relook, decs,
               hold=None, matched=None, cohort_ret=None):
    H = []
    H.append(f"<h2 style='margin:0'>PF Decision Tracker — {asof}</h2>")
    H.append(f"<p style='color:#666;margin:2px 0 14px'>{len(pf)} holdings · "
             f"weightage col: <code>{esc(str(wcol))}</code> · judged vs your own book's median · "
             f"detailed tables in the attached Excel</p>")

    # REVIEW NOW — recent buys sinking / recent sells that ran up (short horizon).
    warn, seen = [], set()
    if not score.empty:
        elapsed = (pd.Timestamp(asof) - pd.to_datetime(score["event_date"])).dt.days
        recent = pd.to_datetime(score["event_date"]) >= (pd.Timestamp(asof) - relativedelta(months=3))
        # A decision needs to have BREATHED before we judge it: at least
        # REVIEW_MIN_DAYS elapsed and a move big enough to mean something.
        # Without this, a trim made yesterday lands in a quartile on a ~0% move.
        sub = score[recent & score["horizon_m"].isin([1, 2])
                    & (elapsed >= REVIEW_MIN_DAYS)
                    & (score["fwd_ret_pct"].abs() >= REVIEW_MIN_MOVE_PP)]
        for r in sub.sort_values("horizon_m").itertuples():
            key = (r.symbol, r.action)
            if key in seen:
                continue                      # one line per decision, not per horizon
            if r.action in ("BUY", "ADD") and r.quartile == 4:
                seen.add(key)
                warn.append(f"⚠️ <b>{esc(r.symbol)}</b> buy is bottom-quartile at {r.horizon_m}M "
                            f"({_pct(r.fwd_ret_pct)} vs book {_pct(r.book_median_pct)})")
            if r.action in ("SELL", "TRIM") and r.quartile == 1:
                seen.add(key)
                warn.append(f"↗️ <b>{esc(r.symbol)}</b> {'sold' if r.action == 'SELL' else 'trimmed'} "
                            f"but ran up at {r.horizon_m}M ({_pct(r.fwd_ret_pct)}) — reconsider")
    if warn:
        H.append("<div style='background:#fff7e6;border:1px solid #ffd591;border-radius:6px;"
                 "padding:10px 12px;margin:8px 0'><b>REVIEW NOW</b><ul style='margin:6px 0'>"
                 + "".join(f"<li>{w}</li>" for w in dict.fromkeys(warn)) + "</ul></div>")

    req_wk = required_wk_pct()

    # 1) PF index — base date, horizon ladder, benchmark + alpha
    if periods:
        ret, cg, pc = periods["ret"], periods["cagr"], periods.get("pace", {})
        bret, alpha = periods.get("bret", {}), periods.get("alpha", {})
        bnames = periods.get("bench_names", [])
        order = ["1D", "1W", "1M", "2M", "3M", "6M", "12M", "SI"]
        dash = "<span style='color:#bbb'>—</span>"

        def row(label, src, bold=False, grey=False):
            style = "padding:3px 10px" + (";color:#555" if grey else "")
            cells = "".join(
                f"<td align='center' style='{style}'>"
                f"{_pct(src.get(k)) if src.get(k) is not None else dash}</td>" for k in order)
            lab = f"<b>{label}</b>" if bold else label
            return f"<tr><td align='left' style='padding:3px 8px'>{lab}</td>{cells}</tr>"

        head = "".join(f"<th style='padding:3px 10px;text-align:center'>{k}</th>" for k in order)
        body = [row("PF return", ret, bold=True),
                row("PF pace (%/wk)", pc, grey=True),
                row("CAGR (annualised)", cg, grey=True)]
        for b in bnames:
            body.append(row(b.replace("_", " "), bret.get(b, {}), grey=True))
            body.append(row(f"Alpha vs {b.replace('_',' ')}", alpha.get(b, {})))

        H.append(
            f"<h3>1 · How the PF is doing</h3>"
            f"<p style='margin:4px 0'>Index set to <b>100.0</b> on <b>{periods['start_date']}</b> "
            f"— the first day portfolio reporting exists — and <b>{periods['level']:.1f}</b> today: "
            f"<b>{_pct(ret.get('SI'))}</b> over <b>{periods['si_days']} tracked days</b>.</p>"
            f"<table style='border-collapse:collapse;font-size:13px'>"
            f"<tr><th align='left' style='padding:3px 8px'></th>{head}</tr>"
            f"{''.join(body)}</table>")
        if periods.get("xirr") is not None:
            H.append(f"<p style='margin:6px 0'><b>XIRR: {_pct(periods['xirr'])}</b></p>")
        H.append(
            f"<div style='background:#f6f8fa;border-left:3px solid #ccc;padding:8px 12px;"
            f"margin:8px 0;font-size:12px;color:#555'>"
            f"<b>Base date {periods['start_date']} — nothing before it is shown.</b> "
            f"Windows longer than {periods['si_days']} days are blank because that history "
            f"isn't tracked yet; they fill in as days accrue.<br>"
            f"• <b>Return</b> — change in the weighted index (each stock moves it in proportion "
            f"to its <i>Portfolio Weight</i>).<br>"
            f"• <b>Pace</b> = (1+return)<sup>7/days</sup> − 1 — the same return expressed as "
            f"%/week, so windows of different lengths compare fairly. "
            f"{TARGET_CAGR_PCT:.0f}%/yr needs <b>{req_wk:.2f}%/week</b>.<br>"
            f"• <b>CAGR</b> = (End ÷ Start)<sup>365/days</sup> − 1. Blank under a quarter — "
            f"annualising a fortnight produces silly numbers.<br>"
            f"• <b>Alpha</b> — PF return minus the index over the identical window.</div>")

        # Holding-window-matched view: each stock's own entry date, weight-averaged.
        if matched:
            lines = []
            for b, m in matched.items():
                lines.append(
                    f"<li>{esc(b.replace('_', ' '))}: your book <b>{_pct(m['pf'])}</b> vs index "
                    f"<b>{_pct(m['bench'])}</b> over the same days → "
                    f"<b>{_pct(m['alpha'])}</b> ({m['n']} holdings)</li>")
            H.append(
                "<p style='margin:10px 0 2px'><b>Index-adjusted for the days you actually "
                "held each name</b></p>"
                "<ul style='margin:2px 0;font-size:13px'>" + "".join(lines) + "</ul>"
                "<div style='background:#f6f8fa;border-left:3px solid #ccc;padding:8px 12px;"
                "margin:4px 0 8px;font-size:12px;color:#555'>Every holding's return is paired "
                "with the index over <i>its own</i> entry date → today, then both are averaged "
                "by weight — so a name bought last week is compared against last week's index, "
                "not the year's. Because the book is always fully invested, this differs from "
                "the plain Alpha row above only through <b>entry timing</b>; it is not extra "
                "alpha, and the row above stays the headline number.</div>")

        if contrib:
            up = contrib[0]; dn = contrib[-1]
            H.append(f"<p style='color:#666'>Today's biggest lift: <b>{esc(up[0])}</b> "
                     f"({_pct(up[1])} of PF) · biggest drag: <b>{esc(dn[0])}</b> ({_pct(dn[1])})</p>")

    # 2) Every holding, ranked — measured from the day it entered the book
    if hold is not None and not hold.empty:
        base_d = periods.get("start_date", "base") if periods else "base"
        n_full = int(hold["full_window"].sum())
        H.append("<h3>2 · Every holding, ranked</h3>")
        H.append(f"<p style='color:#888;font-size:12px'>Every stock is measured "
                 f"<b>from the day it entered the book</b> — {n_full} of {len(hold)} have been "
                 f"held since the base date ({base_d}); the rest start from when they were added, "
                 f"so a name bought last week shows last week's return, not the market's. "
                 f"Ranked by <b>pace</b> (%/week) because that is the only fair way to compare "
                 f"different holding lengths. <b>vs med</b> = pace minus your book's median pace. "
                 f"<b>Idx</b> = what NIFTY 500 did over that stock's <i>own</i> holding window, so "
                 f"<b>vs idx</b> is a like-for-like gap over identical days (smallcap column is in "
                 f"the workbook). "
                 f"<b>Contrib</b> = points of PF growth actually delivered (weight × return). "
                 f"Trailing market windows (1M/3M/12M) and the flat-day counts are in "
                 f"the workbook — they pre-date most holdings and were only ever context, "
                 f"and dropping them keeps this mail under Gmail's clip limit.</p>")
        H.append("<table style='border-collapse:collapse;font-size:12px'>"
                 "<tr><th style='padding:2px 6px'>#</th><th align='left'>Symbol</th>"
                 "<th style='padding:2px 6px'>Wt%</th><th style='padding:2px 6px'>From</th>"
                 "<th style='padding:2px 6px'>Days</th><th style='padding:2px 6px'>Return</th>"
                 "<th style='padding:2px 6px'>%/wk</th><th style='padding:2px 6px'>vs med</th>"
                 "<th style='padding:2px 6px'>Idx</th><th style='padding:2px 6px'>vs idx</th>"
                 "<th style='padding:2px 6px'>Contrib</th>"
                 "<th style='padding:2px 6px'>±5%</th>"
                 "<th style='padding:2px 6px'>±10%</th>"
                 "<th align='left' style='padding:2px 6px'>Flags</th></tr>")
        for r in hold.itertuples():
            f5 = "—" if r.flat_5d is None or pd.isna(r.flat_5d) else f"{int(r.flat_5d)}"
            f10 = "—" if r.flat_10d is None or pd.isna(r.flat_10d) else f"{int(r.flat_10d)}"
            fl = str(r.flags or "").replace("underweight-winner",
                                            "<span style='color:#1a7f37'>underweight-winner</span>")
            fl = fl.replace("negative", "<span style='color:#c0392b'>negative</span>")
            H.append(
                f"<tr><td align='center' style='padding:2px 6px'>{r.rank}</td>"
                f"<td>{esc(r.symbol)}</td>"
                f"<td align='right' style='padding:2px 6px'>"
                f"{'' if pd.isna(r.weight_pct) else format(r.weight_pct, '.1f')}</td>"
                f"<td align='center' style='padding:2px 6px;color:#777'>{esc(r.anchor_date, 10)}</td>"
                f"<td align='right' style='padding:2px 6px;color:#777'>{r.days_held}</td>"
                f"<td align='right' style='padding:2px 6px'><b>{_pct(r.ret_since_pct)}</b></td>"
                f"<td align='right' style='padding:2px 6px'>{_pct(r.pace_wk_pct)}</td>"
                f"<td align='right' style='padding:2px 6px'>{_pct(r.vs_median_pp)}</td>"
                f"<td align='right' style='padding:2px 6px;color:#888'>"
                f"{_pct(r.n500_ret_pct)}</td>"
                f"<td align='right' style='padding:2px 6px'>{_pct(r.vs_bench_pp)}</td>"
                f"<td align='right' style='padding:2px 6px'>{_pct(r.contrib_pp)}</td>"
                f"<td align='right' style='padding:2px 6px'>{f5}</td>"
                f"<td align='right' style='padding:2px 6px'>{f10}</td>"
                f"<td style='padding:2px 6px;font-size:11px'>{fl}</td></tr>")
        H.append("</table>")

    # 3) Allocation — did sizing match performance?
    if hold is not None and not hold.empty:
        H.append("<h3>3 · Where the growth came from (allocation)</h3>")
        eq_wt = 100.0 / len(hold)
        uw = hold[hold["flags"].str.contains("underweight-winner", na=False)].copy()
        uw = uw.sort_values("left_on_table_pp", ascending=False).head(12)
        H.append(f"<p style='color:#888;font-size:12px'>Equal weight across "
                 f"{len(hold)} holdings would be <b>{eq_wt:.2f}%</b>. These are top-quartile "
                 f"performers held <b>below</b> that — they earned well but were too small to "
                 f"move the PF. <b>Perf#/Con#</b> contrasts where the stock ranks on pace "
                 f"versus where it ranks on actual contribution; a wide gap means sizing, not "
                 f"the stock, is the limit. <b>Cost</b> = growth foregone had it merely been "
                 f"equal-weighted. Arithmetic, not a recommendation.</p>"
                 f"<p style='color:#aaa;font-size:11px'>Contributions use today's weights, so "
                 f"they sum to slightly more than the index — the index also chains daily "
                 f"compounding, weight changes and names you've since sold.</p>")
        if uw.empty:
            H.append("<p style='color:#888'>No underweight winners — sizing is tracking "
                     "performance.</p>")
        else:
            H.append("<table style='border-collapse:collapse;font-size:12px'>"
                     "<tr><th align='left'>Symbol</th><th style='padding:2px 8px'>Wt%</th>"
                     "<th style='padding:2px 8px'>Return</th><th style='padding:2px 8px'>%/wk</th>"
                     "<th style='padding:2px 8px'>Perf#/Con#</th>"
                     "<th style='padding:2px 8px'>Contrib</th>"
                     "<th style='padding:2px 8px'>Cost of sizing</th></tr>")
            for r in uw.itertuples():
                cr = "—" if pd.isna(r.contrib_rank) else f"{int(r.contrib_rank)}"
                H.append(f"<tr><td>{esc(r.symbol)}</td>"
                         f"<td align='right' style='padding:2px 8px'>{r.weight_pct:.2f}</td>"
                         f"<td align='right' style='padding:2px 8px'>{_pct(r.ret_since_pct)}</td>"
                         f"<td align='right' style='padding:2px 8px'>{_pct(r.pace_wk_pct)}</td>"
                         f"<td align='center' style='padding:2px 8px;color:#777'>"
                         f"{r.rank} → {cr}</td>"
                         f"<td align='right' style='padding:2px 8px'>{_pct(r.contrib_pp)}</td>"
                         f"<td align='right' style='padding:2px 8px;color:#1a7f37'>"
                         f"<b>{_pct(r.left_on_table_pp)}</b></td></tr>")
            H.append("</table>")

        # The mirror image: big positions that aren't earning their weight.
        ow = hold[(hold["weight_pct"] > eq_wt) & (hold["vs_median_pp"] < 0)].copy()
        ow = ow.sort_values("contrib_pp").head(10)
        if not ow.empty:
            H.append(f"<p style='color:#888;font-size:12px;margin-top:10px'>"
                     f"<b>Big but slow</b> — above-median weight, below-median pace. These "
                     f"occupy the room the names above would need.</p>")
            H.append("<table style='border-collapse:collapse;font-size:12px'>"
                     "<tr><th align='left'>Symbol</th><th style='padding:2px 8px'>Wt%</th>"
                     "<th style='padding:2px 8px'>Return</th><th style='padding:2px 8px'>%/wk</th>"
                     "<th style='padding:2px 8px'>vs med</th>"
                     "<th style='padding:2px 8px'>Contrib</th></tr>")
            for r in ow.itertuples():
                H.append(f"<tr><td>{esc(r.symbol)}</td>"
                         f"<td align='right' style='padding:2px 8px'>{r.weight_pct:.1f}</td>"
                         f"<td align='right' style='padding:2px 8px'>{_pct(r.ret_since_pct)}</td>"
                         f"<td align='right' style='padding:2px 8px'>{_pct(r.pace_wk_pct)}</td>"
                         f"<td align='right' style='padding:2px 8px;color:#c0392b'>"
                         f"{_pct(r.vs_median_pp)}</td>"
                         f"<td align='right' style='padding:2px 8px'>{_pct(r.contrib_pp)}</td></tr>")
            H.append("</table>")

    # 4) Pace vs goal + stagnancy
    if hold is not None and not hold.empty:
        H.append(f"<h3>4 · Off the pace &amp; going nowhere</h3>")
        slow = hold[hold["below_target"]].copy().sort_values("pace_wk_pct")
        H.append(f"<p style='color:#888;font-size:12px'>A <b>{TARGET_CAGR_PCT:.0f}%/year</b> "
                 f"target needs <b>{req_wk:.2f}%/week</b>. "
                 f"{len(slow)} of {len(hold)} holdings are below that pace since they entered "
                 f"the book. <b>±5% / ±10%</b> = consecutive trading days the close has stayed "
                 f"inside that band of today's price — the cleanest read on dead money.</p>")
        if not slow.empty:
            H.append("<table style='border-collapse:collapse;font-size:12px'>"
                     "<tr><th align='left'>Symbol</th><th style='padding:2px 8px'>Wt%</th>"
                     "<th style='padding:2px 8px'>Return</th><th style='padding:2px 8px'>%/wk</th>"
                     "<th style='padding:2px 8px'>Short by</th>"
                     "<th style='padding:2px 8px'>±5% days</th>"
                     "<th style='padding:2px 8px'>±10% days</th></tr>")
            for r in slow.head(25).itertuples():
                short = (req_wk - r.pace_wk_pct) if pd.notna(r.pace_wk_pct) else None
                f5 = "—" if r.flat_5d is None or pd.isna(r.flat_5d) else f"{int(r.flat_5d)}"
                f10 = "—" if r.flat_10d is None or pd.isna(r.flat_10d) else f"{int(r.flat_10d)}"
                H.append(f"<tr><td>{esc(r.symbol)}</td>"
                         f"<td align='right' style='padding:2px 8px'>"
                         f"{'' if pd.isna(r.weight_pct) else format(r.weight_pct, '.1f')}</td>"
                         f"<td align='right' style='padding:2px 8px'>{_pct(r.ret_since_pct)}</td>"
                         f"<td align='right' style='padding:2px 8px'>{_pct(r.pace_wk_pct)}</td>"
                         f"<td align='right' style='padding:2px 8px;color:#c0392b'>"
                         f"{'—' if short is None else format(short, '.2f')}</td>"
                         f"<td align='right' style='padding:2px 8px'>{f5}</td>"
                         f"<td align='right' style='padding:2px 8px'>{f10}</td></tr>")
            H.append("</table>")
            if len(slow) > 25:
                H.append(f"<p style='color:#888;font-size:12px'>…and {len(slow)-25} more "
                         f"in the attached workbook.</p>")

    # 5) Movers
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
        H.append("<h3>5 · Movers (trailing, as of today)</h3>")
        H.append(mover_table(ranked.head(5), rc, "Top 5 by 3M"))
        H.append(mover_table(ranked.tail(5).iloc[::-1], rc, "Bottom 5 by 3M"))

    # 3) Decision correctness (matured hit-rates)
    if not score.empty:
        H.append("<h3>6 · Are my decisions correct? (vs own-book median)</h3>")
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

    # 4) Sold-stock relook — what selling actually cost, vs your own book
    if not relook.empty:
        cost = relook["opportunity_pp"].dropna()
        net = float(cost.sum()) if len(cost) else None
        H.append("<h3>7 · Relook: stocks you sold — what if you hadn't?</h3>")
        H.append(
            "<p style='color:#888;font-size:12px'>Selling didn't park the money — it went "
            "back into this book. So each exit is judged against <b>your own PF</b> over the "
            "identical days, not against zero. <b>Made</b> = the return while you held it. "
            "<b>Since sell</b> = what it did once you were out. <b>Your PF</b> = what the book "
            "returned over those same days. <b>Cost</b> = weight at sale × (since sell − your "
            "PF): positive means selling gave up that many points of PF growth, negative means "
            "it dodged them. <b>Held on</b> = where the position would stand today had you "
            "never sold.</p>")
        if net is not None:
            verb = "cost" if net > 0 else "saved"
            col = "#c0392b" if net > 0 else "#1a7a3c"
            H.append(f"<p style='margin:4px 0'>Net across {len(cost)} exits: selling "
                     f"<b style='color:{col}'>{verb} {abs(net):.2f} pp</b> of portfolio "
                     f"growth versus holding on.</p>")
        H.append("<table style='border-collapse:collapse;font-size:12px'>"
                 "<tr><th align='left'>Sold</th><th style='padding:2px 8px'>On</th>"
                 "<th style='padding:2px 8px'>Wt%</th><th style='padding:2px 8px'>Days out</th>"
                 "<th style='padding:2px 8px'>Made</th><th style='padding:2px 8px'>Since sell</th>"
                 "<th style='padding:2px 8px'>Your PF</th><th style='padding:2px 8px'>Idx</th>"
                 "<th style='padding:2px 8px'>Cost</th><th style='padding:2px 8px'>Held on</th>"
                 "<th style='padding:2px 8px'>YoY</th><th align='left'>Verdict</th></tr>")
        for r in relook.itertuples():
            cc = "#c0392b" if (pd.notna(r.opportunity_pp) and r.opportunity_pp > 0) else "#1a7a3c"
            H.append(
                f"<tr><td>{esc(r.symbol)}</td>"
                f"<td style='padding:2px 8px;color:#777'>{esc(str(r.sell_date), 10)}</td>"
                f"<td align='right' style='padding:2px 8px'>"
                f"{'—' if pd.isna(r.weight_at_sell) else format(r.weight_at_sell, '.1f')}</td>"
                f"<td align='right' style='padding:2px 8px;color:#777'>{r.days_since_sell}</td>"
                f"<td align='right' style='padding:2px 8px'>{_pct(r.realised_pct)}</td>"
                f"<td align='right' style='padding:2px 8px'><b>{_pct(r.since_sell_pct)}</b></td>"
                f"<td align='right' style='padding:2px 8px;color:#888'>"
                f"{_pct(r.pf_since_sell_pct)}</td>"
                f"<td align='right' style='padding:2px 8px;color:#888'>"
                f"{_pct(r.n500_since_sell_pct)}</td>"
                f"<td align='right' style='padding:2px 8px;color:{cc}'>"
                f"<b>{'—' if pd.isna(r.opportunity_pp) else format(r.opportunity_pp, '+.2f')}"
                f"</b></td>"
                f"<td align='right' style='padding:2px 8px'>{_pct(r.full_hold_pct)}</td>"
                f"<td align='right' style='padding:2px 8px;color:#888'>"
                f"{_pct(r.latest_yoy_pct)}</td>"
                f"<td style='padding:2px 8px'>{esc(r.verdict)}</td></tr>")
        H.append("</table>")

    # 8) Monthly cohorts — has trading beaten leaving the book alone?
    if cohort_ret is not None and not cohort_ret.empty:
        H.append("<h3>8 · Trading vs doing nothing (monthly frozen baskets)</h3>")

        # The earliest cohort is inception, so this line works from day one.
        first = cohort_ret[cohort_ret["partial"]].sort_values("cohort_month")
        if not first.empty:
            r0 = first.iloc[0]
            if pd.notna(r0["delta_pp"]):
                verb = "added" if r0["delta_pp"] >= 0 else "cost"
                col = "#1a7a3c" if r0["delta_pp"] >= 0 else "#c0392b"
                H.append(
                    f"<p style='margin:4px 0'>Your book as it stood on "
                    f"<b>{esc(str(r0['cohort_month']))}</b>, frozen and never traded, would be "
                    f"<b>{_pct(r0['frozen_ret_pct'])}</b>. You actually did "
                    f"<b>{_pct(r0['actual_ret_pct'])}</b> over the same {int(r0['days'])} days "
                    f"— so trading has <b style='color:{col}'>{verb} "
                    f"{abs(r0['delta_pp']):.2f} pp</b>.</p>")

        H.append(
            "<p style='color:#888;font-size:12px'>On the first snapshot of each month the "
            "book's composition is photographed and left to run untouched. Each cell is "
            "<b>what you actually made minus what that frozen photo made</b> over the same "
            "window. <span style='color:#1a7a3c'>Green = your buying and selling earned its "
            "keep</span>; <span style='color:#c0392b'>red = you'd have done better leaving it "
            "alone</span>. Blank means that horizon hasn't elapsed yet — not zero.</p>")

        order = [("1M", 1), ("2M", 2), ("3M", 3), ("6M", 6), ("12M", 12)]
        dash = "<span style='color:#bbb'>—</span>"   # section 1's copy is scoped to it
        head = "".join(f"<th style='padding:3px 10px'>{lbl}</th>" for lbl, _ in order)
        rows = []
        for m in sorted(cohort_ret["cohort_month"].unique(), reverse=True)[:COHORT_MAIL_N]:
            g = cohort_ret[cohort_ret["cohort_month"] == m]
            cells = []
            for _, h in order:
                v = g[(~g["partial"]) & (g["horizon_m"] == h)]["delta_pp"]
                cells.append(f"<td align='center' style='padding:3px 10px'>"
                             f"{_pct(v.iloc[0]) if len(v) and pd.notna(v.iloc[0]) else dash}</td>")
            liv = g[g["partial"]]
            now = (f"{_pct(liv.iloc[0]['delta_pp'])} "
                   f"<span style='color:#aaa'>({int(liv.iloc[0]['days'])}d)</span>"
                   if len(liv) and pd.notna(liv.iloc[0]["delta_pp"]) else dash)
            rows.append(f"<tr><td align='left' style='padding:3px 8px'><b>{esc(str(m))}</b></td>"
                        f"{''.join(cells)}"
                        f"<td align='center' style='padding:3px 10px'>{now}</td></tr>")
        H.append("<table style='border-collapse:collapse;font-size:13px'>"
                 f"<tr><th align='left' style='padding:3px 8px'>Frozen on</th>{head}"
                 "<th style='padding:3px 10px'>Now</th></tr>"
                 + "".join(rows) + "</table>")

        n_months = cohort_ret["cohort_month"].nunique()
        if n_months < 3:
            H.append(f"<p style='color:#888;font-size:12px'>Only {n_months} month of tracked "
                     f"history so far — the grid fills in as months accrue. Full history is in "
                     f"the workbook's <i>Cohort Returns</i> sheet.</p>")

    H.append("<p style='color:#aaa;font-size:11px;margin-top:16px'>Signals only — human-in-the-loop. "
             "Decision dates are snapshot-observation dates. Weightage-only: no rupee P&L.</p>")
    return shrink_html("".join(H))


def build_excel_bytes(pf, idx, movers, score, relook, decs, hold=None,
                      spells=None, cohorts=None, cohort_ret=None) -> bytes:
    """Detailed tables as a multi-sheet .xlsx (the email is the summary; this is the
    drill-down). One sheet per view; header frozen + light auto-width."""
    holdings = pf[["isin", "symbol", "name", "weight_pct"]].copy()
    hold = hold if hold is not None else pd.DataFrame(columns=HOLD_COLS)
    if not hold.empty:
        problems = hold[(hold["neg_windows"] >= 3) | (hold["below_med_windows"] >= 3)
                        | (hold["below_target"])]
        slow = hold[hold["below_target"]].sort_values("pace_wk_pct")
        underweight = hold[hold["flags"].str.contains("underweight-winner", na=False)] \
            .sort_values("left_on_table_pp", ascending=False)
    else:
        problems = slow = underweight = hold
    sheets = {
        "Holdings Today":     holdings,
        "Ranked Holdings":    hold,
        "Underweight Winners": underweight,
        "Off The Pace":       slow,
        "Problem Stocks":     problems,
        "Movers":             movers,
        "Decision Scorecard": score,
        "Exits":              relook,
        "Holding Spells":     (spells if spells is not None
                               else pd.DataFrame(columns=SPELL_COLS)),
        "Cohorts":            (cohorts if cohorts is not None
                               else pd.DataFrame(columns=COHORT_COLS)),
        "Cohort Returns":     (cohort_ret if cohort_ret is not None
                               else pd.DataFrame(columns=COHORT_RET_COLS)),
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
    ap.add_argument("--no-write", action="store_true",
                    help="Send the email but persist NOTHING to Drive — a real preview "
                         "mail that leaves the ledgers untouched (the mirror of --no-mail).")
    ap.add_argument("--asof", default=None, help="Recompute as of YYYY-MM-DD (default: today).")
    ap.add_argument("--rebuild-decisions", action="store_true",
                    help="Re-derive the WHOLE decision ledger from the snapshots. Needed "
                         "once after an ISIN alias is found, to clear events a corporate "
                         "action faked (decisions are derived; snapshots are the truth).")
    args = ap.parse_args()

    asof = (datetime.strptime(args.asof, "%Y-%m-%d").date() if args.asof else date.today())
    dry = args.dry_run
    # --no-write still emails, but nothing reaches Drive: the ledgers, the snapshot
    # and every derived table are left exactly as they were.
    persist = not (dry or args.no_write)
    mode = "  [DRY-RUN]" if dry else ("  [PREVIEW — no Drive writes]" if args.no_write else "")
    log("=" * 64)
    log(f"PF Decision Tracker — asof {asof}{mode}")
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
    cohorts = load_parquet(drive, out_id, "pf_cohorts.parquet", COHORT_COLS)
    price_cache: dict = {}

    # 3. Snapshot capture (change-gated) + decision diff.
    snaps, changed = append_snapshot(drive, out_id, snaps, pf, asof, target["name"],
                                     not persist)

    # A corporate action swaps a company's ISIN. Collapse the new ISIN back onto the
    # old one BEFORE anything is derived, so the swap reads as one continuous holding
    # instead of a sell followed by a buy. Applied to today's parsed book too, so the
    # snapshot just captured lines up with the history.
    alias = build_isin_alias(snaps)
    if not alias.empty:
        for a in alias.itertuples():
            log(f"  ISIN change: {a.symbol} {a.old_isin} -> {a.new_isin} on {a.changed_on} "
                f"— treated as the SAME holding, not a sell+buy.")
        snaps = apply_isin_alias(snaps, alias)
        decs = apply_isin_alias(decs, alias)
        cohorts = apply_isin_alias(cohorts, alias)
        pf = apply_isin_alias(pf, alias)
        if persist:
            save_parquet(drive, out_id, "pf_isin_alias.parquet", alias)

    decs_before = len(decs)
    if args.rebuild_decisions:
        # Events a corporate action faked are already in the ledger. Decisions are
        # DERIVED data, so the honest repair is to regenerate them from the snapshots
        # rather than hand-patch rows.
        log("  --rebuild-decisions: re-deriving the full ledger from snapshots…")
        decs = pd.DataFrame(columns=DEC_COLS)
        all_dates = sorted(snaps["snapshot_date"].unique())
        for i in range(len(all_dates)):
            decs = derive_decisions(snaps[snaps["snapshot_date"].isin(all_dates[:i + 1])],
                                    decs, asof, price_cache, drive, ohlcv_id)
        log(f"  Rebuilt {len(decs)} decision event(s) (was {decs_before}).")
    else:
        decs = derive_decisions(snaps, decs, asof, price_cache, drive, ohlcv_id)
    if persist and (args.rebuild_decisions or len(decs) != decs_before):
        save_parquet(drive, out_id, "pf_decisions.parquet", decs)

    # 4. Compute everything. Benchmarks come from Phase-1 data/indices/.
    benches = {}
    for b in BENCHMARKS:
        ser = load_benchmark(drive, root_id, b)
        if ser is not None and not ser.empty:
            benches[b] = ser
    log(f"  Benchmarks: {', '.join(benches) if benches else 'none available'}")

    # Ownership runs — every per-stock window anchors on the CURRENT spell, so a
    # re-bought name is never measured across the months it wasn't owned.
    spells = build_spells(snaps, decs)
    if not spells.empty:
        multi = spells.groupby("isin").size()
        multi = multi[multi > 1]
        if len(multi):
            old_first = snaps.groupby("isin")["snapshot_date"].min().to_dict()
            cur = {r.isin: r.spell_start
                   for r in spells[spells["status"] == "HELD"].itertuples()}
            moved = [(i, old_first[i], cur[i]) for i in multi.index
                     if i in cur and cur[i] != old_first.get(i)]
            log(f"  {len(multi)} ISIN(s) with >1 holding spell; "
                f"{len(moved)} held name(s) re-anchored:")
            for i, o, n in moved[:10]:
                log(f"    {i}: anchor {o} -> {n}")
        else:
            log("  No re-bought names — anchors identical to the earliest-snapshot rule.")

    idx, contrib = compute_index(snaps, price_cache, drive, ohlcv_id, asof)
    periods = index_period_returns(idx, benches)
    if periods:
        log(f"  Index base {periods['start_date']} = 100.0 -> {periods['level']:.1f} "
            f"({periods['si_days']} tracked days)")
    movers = compute_movers(drive, root_id, pf)
    hold = compute_holdings_view(pf, snaps, movers, price_cache, drive, ohlcv_id, asof,
                                 bench_map=benches, spells=spells)
    matched = matched_bench_returns(hold)
    for b, m in matched.items():
        log(f"  Holding-window matched vs {b}: PF {m['pf']:+.2f}% vs {m['bench']:+.2f}% "
            f"-> {m['alpha']:+.2f} pp ({m['n']} holdings)")
    score = score_decisions(decs, snaps, price_cache, drive, ohlcv_id, asof)
    results = _read_drive_parquet(drive, index_id, "results.parquet")
    relook = compute_exits_view(spells, price_cache, drive, ohlcv_id, idx, results,
                                asof, bench_map=benches)
    if not relook.empty:
        net = relook["opportunity_pp"].dropna().sum()
        log(f"  {len(relook)} exit(s) in the last 12M; net cost of selling "
            f"{net:+.2f} pp vs holding on.")

    # Monthly frozen baskets — the cumulative verdict on trading. Cohorts are
    # append-only history; their returns are derived and regenerated every run.
    cohorts, cohorts_new = build_cohorts(snaps, cohorts)
    if persist and cohorts_new:
        save_parquet(drive, out_id, "pf_cohorts.parquet", cohorts)
    cohort_ret = compute_cohort_returns(cohorts, idx, price_cache, drive, ohlcv_id, asof)
    if not cohort_ret.empty:
        live = cohort_ret[cohort_ret["partial"]]
        for r in live.itertuples():
            log(f"  Cohort {r.cohort_month}: frozen {r.frozen_ret_pct:+.2f}% vs actual "
                f"{r.actual_ret_pct:+.2f}% over {r.days}d -> trading "
                f"{'added' if r.delta_pp >= 0 else 'cost'} {abs(r.delta_pp):.2f} pp "
                f"({r.n_priced}/{r.n_total} priced)")

    # 5. Persist derived tables (overwrite; idempotent).
    if not persist:
        log("  Drive writes suppressed — ledgers and derived tables left untouched.")
    if persist:
        save_parquet(drive, out_id, "pf_index.parquet", idx)
        save_parquet(drive, out_id, "pf_decision_scorecard.parquet", score)
        save_parquet(drive, out_id, "pf_movers.parquet", movers)
        save_parquet(drive, out_id, "pf_holdings_view.parquet", hold)
        save_parquet(drive, out_id, "pf_exits.parquet", relook)
        save_parquet(drive, out_id, "pf_spells.parquet", spells)
        save_parquet(drive, out_id, "pf_cohort_returns.parquet", cohort_ret)
        log("  Wrote pf_index / pf_decision_scorecard / pf_movers / pf_holdings_view "
            "/ pf_exits / pf_spells / pf_cohort_returns.")

    # 6. Digest (HTML findings) + Excel workbook (detailed tables).
    html = build_html(asof, wcol, pf, idx, periods, contrib, movers, score, relook,
                      decs, hold, matched, cohort_ret)
    # Gmail clips a message body past ~102 KB — surface the size instead of finding
    # out from a truncated mail.
    kb = len(html.encode("utf-8")) / 1024.0
    log(f"  HTML body {kb:.1f} KB{'  ** near Gmail 102 KB clip **' if kb > 92 else ''}")
    si = (periods.get("ret") or {}).get("SI") if periods else None
    if periods and si is not None:
        subject = (f"PF Tracker — {asof} · index {periods['level']:.1f} "
                   f"({si:+.1f}% since base {periods['start_date']}) · {len(pf)} holdings")
    else:
        subject = f"PF Tracker — {asof} ({len(pf)} holdings)"
    if args.no_write:
        subject = "[PREVIEW] " + subject
    xlsx = build_excel_bytes(pf, idx, movers, score, relook, decs, hold, spells,
                             cohorts, cohort_ret)
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
