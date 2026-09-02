r"""
signal_tracker.py — did the calls actually work?

Phase 1 has never kept score. Every day it produced a list, and the next day it
produced another one; nothing recorded whether yesterday's names went up, which
engine was worth listening to, or whether the conviction score predicted
anything at all. Without that, no weight in this pipeline can be more than a
guess — which is exactly what guru's SYSTEM_BACKTEST Calibration sheet exposed
on the research side, where a model predicting 21-37% returns delivered -0.9 to
+4.9%.

WHAT IT DOES
  Reads signals/aggregated/open_signals.csv — the entry and stop FROZEN on the
  day each (symbol, family) first fired — joins the price history since, and
  asks of every call: did it hit its target, hit its stop, or go stale?

  Everything is expressed in R, where 1R = entry - stop, the amount risked. A
  +2R winner and a -1R loser are then directly comparable across stocks priced
  Rs 40 and Rs 4,000, which raw percentages are not.

THE POINT ABOUT SELL CALLS
  A sell can only be judged against NOT selling. So every closed signal also
  records what it would have returned had it simply been held to a fixed
  horizon. exit_vs_hold is the difference: positive means the exit added value,
  negative means it cut a winner short. Without that column "which engine gives
  the best sell calls" is unanswerable. This is the same frame guru's
  exits_summary.xlsx used, where every trailing stop tested LOST 4-5pp against
  simply holding.

RELIABILITY
  Three measures, per score and per ranking term:
    decile lift   top-decile minus bottom-decile realised R. Near zero means the
                  score is decoration, however sensible it looks.
    rank IC       Spearman correlation between score and outcome, averaged over
                  days, with the share of days it was positive.
    win rate / median R by decile, with n ALWAYS shown.

Outputs (all under signals/analysis/):
    signal_outcomes.csv    append-only ledger, one row per closed signal
    tracker_summary.csv    per-family scorecard for the day

Usage:
    python scripts/signal_tracker.py --dry-run      # compute + report, no writes
    python scripts/signal_tracker.py                # update the ledger

NOT YET BUILT: seeding from the ~70 dated snapshots in signals/aggregated/.
Those snapshots predate open_signals.csv and so carry no frozen entry/stop; a
replay would have to reconstruct both from the price history on each snapshot
date. Until that exists the ledger starts empty and fills as calls close, so the
reliability section stays silent (it refuses to report below n=20) rather than
publishing a number drawn from a handful of trades.
"""
from __future__ import annotations

import argparse
import io
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd

_SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(_SCRIPTS_DIR), ".env"))

# Defaults. TARGET_R is deliberately a parameter, not a belief: the whole point
# of this script is to measure the MFE distribution and let that choose it.
TARGET_R = 2.0
MAX_HOLD_DAYS = 63          # ~3 months of trading; beyond this a call is stale
HOLD_HORIZON_DAYS = 63      # the "what if I had just held" benchmark


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


# ─────────────────────────── the outcome engine ─────────────────────────────
# Pure functions on a price frame: no Drive, no network, so they can be tested
# offline (scripts/tests/test_phase1_engine.py).

def evaluate_signal(bars: pd.DataFrame, entry: float, stop: float,
                    target_r: float = TARGET_R,
                    max_hold_days: int = MAX_HOLD_DAYS,
                    hold_horizon_days: int = HOLD_HORIZON_DAYS) -> dict | None:
    """Score one call against the bars that followed it.

    `bars` must be sorted ascending and start ON the signal date. Returns None
    when the signal is unusable (no risk defined, or no price history yet),
    which is reported rather than silently dropped.
    """
    if bars is None or len(bars) == 0:
        return None
    try:
        entry = float(entry)
        stop = float(stop)
    except (TypeError, ValueError):
        return None
    risk = entry - stop
    if not np.isfinite(risk) or risk <= 0:
        # A stop at or above entry is not a trade; it is bad data. Counted as
        # invalid so a systematic upstream break shows up rather than shrinking
        # the sample quietly.
        return {"status": "invalid", "reason": "stop >= entry"}

    target = entry + target_r * risk
    high = pd.to_numeric(bars["high"], errors="coerce").to_numpy()
    low = pd.to_numeric(bars["low"], errors="coerce").to_numpy()
    close = pd.to_numeric(bars["close"], errors="coerce").to_numpy()
    n = len(close)

    stop_idx = next((i for i in range(n) if low[i] <= stop), None)
    tgt_idx = next((i for i in range(n) if high[i] >= target), None)

    # Whichever came FIRST decides. On the same bar we assume the stop, because
    # from daily bars alone the order within the day is unknowable and assuming
    # the good outcome would flatter every result.
    if stop_idx is not None and (tgt_idx is None or stop_idx <= tgt_idx):
        status, exit_idx, realised_r = "stopped", stop_idx, -1.0
    elif tgt_idx is not None:
        status, exit_idx, realised_r = "target_hit", tgt_idx, float(target_r)
    elif n - 1 >= max_hold_days:
        status, exit_idx = "expired", max_hold_days
        realised_r = float((close[exit_idx] - entry) / risk)
    else:
        status, exit_idx = "open", n - 1
        realised_r = float((close[exit_idx] - entry) / risk)

    seg_hi, seg_lo = high[:exit_idx + 1], low[:exit_idx + 1]
    hold_idx = min(hold_horizon_days, n - 1)
    hold_r = float((close[hold_idx] - entry) / risk)

    return {
        "status": status,
        "days_held": int(exit_idx),
        "r_multiple": round(realised_r, 3),
        "mfe_r": round(float((np.nanmax(seg_hi) - entry) / risk), 3),
        "mae_r": round(float((np.nanmin(seg_lo) - entry) / risk), 3),
        # What simply holding would have returned over the same benchmark
        # window. exit_vs_hold is the ONLY way to score a sell decision.
        "hold_r": round(hold_r, 3),
        "hold_days": int(hold_idx),
        "exit_vs_hold_r": round(realised_r - hold_r, 3),
        "risk_per_share": round(risk, 4),
        "target_price": round(target, 2),
        "closed": status in ("stopped", "target_hit", "expired"),
    }


def summarise_by(df: pd.DataFrame, key: str) -> pd.DataFrame:
    """Per-family scorecard. Buy skill and sell skill are scored separately:
    r_multiple says whether the PICK was good, exit_vs_hold_r whether the EXIT
    was."""
    if df.empty:
        return pd.DataFrame()
    closed = df[df["closed"]] if "closed" in df.columns else df
    rows = []
    for k, g in df.groupby(key):
        gc = closed[closed[key] == k] if len(closed) else g.iloc[0:0]
        rows.append({
            key: k,
            "n_open": int((~g["closed"]).sum()) if "closed" in g else 0,
            "n_closed": len(gc),
            "win_rate_pct": (round(float((gc["r_multiple"] > 0).mean() * 100), 1)
                             if len(gc) else np.nan),
            "median_r": round(float(gc["r_multiple"].median()), 3) if len(gc) else np.nan,
            "median_mfe_r": round(float(gc["mfe_r"].median()), 3) if len(gc) else np.nan,
            "median_mae_r": round(float(gc["mae_r"].median()), 3) if len(gc) else np.nan,
            # positive = exiting beat holding; negative = it cut winners short
            "median_exit_vs_hold_r": (round(float(gc["exit_vs_hold_r"].median()), 3)
                                      if len(gc) else np.nan),
            "pct_target_hit": (round(float((gc["status"] == "target_hit").mean() * 100), 1)
                               if len(gc) else np.nan),
            "pct_stopped": (round(float((gc["status"] == "stopped").mean() * 100), 1)
                            if len(gc) else np.nan),
        })
    return pd.DataFrame(rows).sort_values("median_r", ascending=False)


def reliability(df: pd.DataFrame, score_col: str,
                outcome_col: str = "r_multiple", n_deciles: int = 10) -> dict:
    """Does this score actually separate outcomes?

    decile_lift is the headline: top decile minus bottom decile realised R. If
    it is ~0 the score is decoration, no matter how reasonable its construction.
    rank_ic is Spearman between score and outcome — the standard quant measure,
    reported with n because a strong IC on 12 observations means nothing.
    """
    d = df[[score_col, outcome_col]].dropna()
    out = {"score": score_col, "n": len(d), "decile_lift": np.nan,
           "rank_ic": np.nan, "top_decile_median_r": np.nan,
           "bottom_decile_median_r": np.nan}
    if len(d) < 20:
        out["note"] = f"n={len(d)} too small to read"
        return out
    # Spearman == Pearson on the ranks. Computed that way deliberately:
    # pandas' method="spearman" imports scipy, which is not in requirements.txt
    # and would fail at runtime in CI.
    out["rank_ic"] = round(float(d[score_col].rank().corr(d[outcome_col].rank())), 4)
    try:
        q = pd.qcut(d[score_col], n_deciles, labels=False, duplicates="drop")
    except ValueError:
        out["note"] = "score has too few distinct values to decile"
        return out
    top, bot = d[q == q.max()], d[q == q.min()]
    out["top_decile_median_r"] = round(float(top[outcome_col].median()), 3)
    out["bottom_decile_median_r"] = round(float(bot[outcome_col].median()), 3)
    out["decile_lift"] = round(out["top_decile_median_r"]
                               - out["bottom_decile_median_r"], 3)
    out["n_top"], out["n_bottom"] = len(top), len(bot)
    return out


# ─────────────────────────────── Drive I/O ──────────────────────────────────

def _drive():
    from _extractor_base import get_drive
    return get_drive()


def _folder(drive, *names):
    from _extractor_base import get_or_create_subfolder
    fid = os.environ["GDRIVE_FOLDER_ID"]
    for n in names:
        fid = get_or_create_subfolder(drive, fid, n)
    return fid


def _read_csv(drive, folder_id, name):
    from _extractor_base import find_file, download_bytes
    fid = find_file(drive, folder_id, name)
    if not fid:
        return None
    return pd.read_csv(io.BytesIO(download_bytes(drive, fid)))


def _write_csv(drive, folder_id, name, df):
    from _extractor_base import find_file, upload_bytes
    upload_bytes(drive, folder_id, name,
                 df.to_csv(index=False).encode(), "text/csv")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true",
                    help="compute and report, write nothing to Drive")
    ap.add_argument("--target-r", type=float, default=TARGET_R,
                    help="target as a multiple of risk (default 2.0). The MFE "
                         "distribution this script reports is what should set it")
    ap.add_argument("--max-hold-days", type=int, default=MAX_HOLD_DAYS)
    ap.add_argument("--limit", type=int, default=0,
                    help="cap symbols processed (testing)")
    args = ap.parse_args()

    print("Signal tracker — did the calls work?")
    print("-" * 60)
    drive = _drive()
    agg_id = _folder(drive, "signals", "aggregated")

    opens = _read_csv(drive, agg_id, "open_signals.csv")
    if opens is None or opens.empty:
        print("signals/aggregated/open_signals.csv not found or empty.")
        print("Run aggregate_signals.py at least once first — it writes the "
              "entry/stop memory this reads.")
        return 0
    log(f"open signals: {len(opens):,} rows, "
        f"{opens['symbol'].nunique():,} symbols")

    ohlcv_id = _folder(drive, "data", "ohlcv")
    from _extractor_base import find_file, download_bytes

    syms = sorted(opens["symbol"].astype(str).unique())
    if args.limit:
        syms = syms[:args.limit]

    results, n_missing, n_invalid = [], 0, 0
    for i, sym in enumerate(syms, 1):
        fid = find_file(drive, ohlcv_id, f"{sym}.parquet")
        if not fid:
            n_missing += 1
            continue
        try:
            px = pd.read_parquet(io.BytesIO(download_bytes(drive, fid)))
            px["date"] = pd.to_datetime(px["date"], errors="coerce")
            px = px.dropna(subset=["date"]).sort_values("date")
        except Exception as e:
            log(f"  {sym}: unreadable ({str(e)[:60]})")
            n_missing += 1
            continue
        for _, r in opens[opens["symbol"].astype(str) == sym].iterrows():
            start = pd.to_datetime(r.get("first_date"), errors="coerce")
            if pd.isna(start):
                continue
            bars = px[px["date"] >= start]
            res = evaluate_signal(bars, r.get("entry_at_signal"),
                                  r.get("stop_at_signal"),
                                  target_r=args.target_r,
                                  max_hold_days=args.max_hold_days)
            if res is None:
                n_missing += 1
                continue
            if res.get("status") == "invalid":
                n_invalid += 1
                continue
            res.update({
                "symbol": sym, "family": r.get("family"),
                "first_date": str(start.date()),
                "entry_at_signal": r.get("entry_at_signal"),
                "stop_at_signal": r.get("stop_at_signal"),
                "zone_at_signal": r.get("zone_at_signal"),
                "conviction_at_signal": r.get("conviction_at_signal"),
                "n_families_at_signal": r.get("n_families_at_signal"),
                "n_events_at_signal": r.get("n_events_at_signal"),
            })
            results.append(res)
        if i % 200 == 0:
            log(f"  {i}/{len(syms)} symbols")

    out = pd.DataFrame(results)
    log(f"evaluated {len(out):,} calls | {n_missing:,} with no usable price "
        f"history | {n_invalid:,} with stop >= entry (bad data, not dropped "
        f"silently)")
    if out.empty:
        print("nothing to score yet.")
        return 0

    print()
    print("=" * 60)
    print("PER-ENGINE SCORECARD  (R = the amount risked per trade)")
    print("=" * 60)
    summ = summarise_by(out, "family")
    print(summ.to_string(index=False))
    print()
    print("  median_r            did the PICK work")
    print("  median_exit_vs_hold did the EXIT work — negative means the exit")
    print("                      cut winners short vs simply holding")

    closed = out[out["closed"]]
    print()
    print("=" * 60)
    print(f"RELIABILITY OF EACH SCORE   (closed calls only, n={len(closed)})")
    print("=" * 60)
    if len(closed) < 20:
        print(f"  n={len(closed)} — too few closed calls to say anything yet.")
        print("  This fills in as calls close; --replay seeds it from history.")
    else:
        rel = [reliability(closed, c) for c in
               ("conviction_at_signal", "n_families_at_signal",
                "n_events_at_signal") if c in closed.columns]
        print(pd.DataFrame(rel).to_string(index=False))
        print()
        print("  decile_lift ~ 0 means the score does not separate outcomes,")
        print("  however sensible it looks. n is shown because a strong")
        print("  correlation on a handful of trades means nothing.")

    print()
    print("MFE distribution — what the target should be set from:")
    print(out["mfe_r"].describe(percentiles=[.25, .5, .6, .75, .9]).round(2).to_string())

    if args.dry_run:
        print()
        log("DRY RUN — nothing written to Drive")
        return 0

    an_id = _folder(drive, "signals", "analysis")
    prev = _read_csv(drive, an_id, "signal_outcomes.csv")
    ledger = closed.copy()
    if prev is not None and len(prev):
        key = ["symbol", "family", "first_date"]
        prev = prev[~prev.set_index(key).index.isin(ledger.set_index(key).index)]
        ledger = pd.concat([prev, ledger], ignore_index=True)
    _write_csv(drive, an_id, "signal_outcomes.csv", ledger)
    _write_csv(drive, an_id, "tracker_summary.csv", summ)
    log(f"wrote signals/analysis/signal_outcomes.csv ({len(ledger):,} rows) "
        f"and tracker_summary.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
