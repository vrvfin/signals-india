"""
Phase 3 — T7: standalone company-wise fraud tracker.

Combines the two fraud engines into ONE auditable per-company signal with
history — the "worst engine wins" so nothing is averaged away:

    investigative_fraud.parquet  -> exchange surveillance / news / SEBI / NFRA
                                    (grade 0-4 -> 0-100)
    fraud_risk.parquet           -> T2 forensic accounting rules (0-100)

    fraud_score = max(investigative_grade/4*100, fraud_risk_score)
    band:  RED >=70 · ALERT >=45 · WATCH >=20 · (score 0/<20 untracked)

Outputs (NEW parquets — no existing readers touched):
  company_repo/_index/fraud_tracker.parquet  (+.csv)
      one row per CURRENTLY tracked company (fraud_score >= 20); watchlist
      semantics: first_flagged_at survives across runs, last_changed_at moves
      only when the score moves, trend vs ~7 days ago from history.
  company_repo/_index/fraud_tracker_history.parquet
      one row per tracked company per day (plus a score-0 clearance row the
      day a company drops off) -> time-series for the app page.

No Gemini, no mail (the nightly fraud-scan digest already alerts). Runs in
t4_nightly.yml after the scorecard, when both inputs are fresh.

Usage:
    python scripts/build_fraud_tracker.py --dry-run
    python scripts/build_fraud_tracker.py --names "SUZLON"
    python scripts/build_fraud_tracker.py --local --local-dir .t4_local
"""

from __future__ import annotations

import argparse
import io
import os
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from _extractor_base import (
    get_drive, get_or_create_subfolder, find_file, download_bytes, upload_bytes,
)

TRACKER_COLS = [
    "isin", "symbol", "company_name", "fraud_score", "band",
    "score_driver", "reason",
    "investigative_grade", "grade_reason", "forensic_score",
    "n_forensic_flags", "forensic_flags",
    "news_hits", "sebi_actions", "nfra_actions",
    "first_flagged_at", "last_changed_at", "trend", "score_7d_ago",
    "as_of", "computed_at",
]
HISTORY_COLS = ["isin", "symbol", "as_of", "fraud_score", "band",
                "investigative_grade", "forensic_score"]

TRACK_MIN = 20.0          # below this a company is not on the tracker
BANDS = [(70.0, "RED"), (45.0, "ALERT"), (TRACK_MIN, "WATCH")]


def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")


def band_of(score: float) -> str:
    for cut, name in BANDS:
        if score >= cut:
            return name
    return "CLEAN"


def explain(inv_pts: float, grade_reason: str,
            for_pts: float, forensic_flags: str) -> tuple[str, str]:
    """(score_driver, one-line reason) — which engine set the score and why.
    Both engines are named whenever both contributed >= TRACK_MIN, so a
    surveillance RED with forensic smoke still shows the forensic part."""
    driver = ("both" if inv_pts == for_pts
              else "investigative" if inv_pts > for_pts else "forensic")
    parts = []
    if inv_pts > 0 and (inv_pts >= TRACK_MIN or inv_pts >= for_pts):
        parts.append(f"surveillance({inv_pts:.0f}): {grade_reason or 'flagged'}")
    if for_pts > 0 and (for_pts >= TRACK_MIN or for_pts >= inv_pts):
        parts.append(f"forensics({for_pts:.0f}): {forensic_flags or 'flagged'}")
    return driver, " | ".join(parts)


# ------------------------------------------------------------------ #
#  Storage (same Store pattern as the other T4/T5 scripts)            #
# ------------------------------------------------------------------ #

from _t4_store import Store




# ------------------------------------------------------------------ #
#  Core (pure — unit-testable offline)                                 #
# ------------------------------------------------------------------ #

def build_tracker(inv: pd.DataFrame | None, fraud: pd.DataFrame | None,
                  prev: pd.DataFrame | None, hist: pd.DataFrame | None,
                  as_of: str, now_iso: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (snapshot_df, history_df_updated)."""
    by_sym: dict[str, dict] = {}

    if inv is not None and not inv.empty:
        for _, r in inv.iterrows():
            sym = str(r.get("symbol", "")).upper()
            if not sym:
                continue
            grade = pd.to_numeric(r.get("investigative_grade"), errors="coerce")
            grade = int(grade) if pd.notna(grade) else 0
            by_sym[sym] = {
                "isin": str(r.get("isin", "")),
                "symbol": sym,
                "company_name": str(r.get("company_name", "")),
                "investigative_grade": grade,
                "grade_reason": str(r.get("grade_reason", "")),
                "news_hits": int(pd.to_numeric(r.get("news_hits"),
                                               errors="coerce") or 0),
                "sebi_actions": int(pd.to_numeric(r.get("sebi_actions"),
                                                  errors="coerce") or 0),
                "nfra_actions": int(pd.to_numeric(r.get("nfra_actions"),
                                                  errors="coerce") or 0),
                "forensic_score": 0.0, "n_forensic_flags": 0, "forensic_flags": "",
            }

    if fraud is not None and not fraud.empty:
        for _, r in fraud.iterrows():
            sym = str(r.get("symbol", "")).upper()
            if not sym:
                continue
            fs = float(pd.to_numeric(r.get("fraud_risk_score"), errors="coerce") or 0)
            e = by_sym.setdefault(sym, {
                "isin": str(r.get("isin", "")), "symbol": sym,
                "company_name": str(r.get("company_name", "")),
                "investigative_grade": 0, "grade_reason": "",
                "news_hits": 0, "sebi_actions": 0, "nfra_actions": 0,
            })
            e["forensic_score"] = fs
            e["n_forensic_flags"] = int(pd.to_numeric(r.get("n_forensic_flags"),
                                                      errors="coerce") or 0)
            e["forensic_flags"] = str(r.get("forensic_flags", ""))

    # ---- previous snapshot: watchlist carry-forward ----
    prev_by_sym: dict[str, dict] = {}
    if prev is not None and not prev.empty:
        for _, r in prev.iterrows():
            prev_by_sym[str(r.get("symbol", "")).upper()] = r.to_dict()

    # ---- ~7d-ago score from history (nearest as_of <= today-7) ----
    week_ago: dict[str, float] = {}
    if hist is not None and not hist.empty:
        h = hist.copy()
        cutoff = (pd.Timestamp(as_of) - timedelta(days=7)).strftime("%Y-%m-%d")
        h = h[h["as_of"].astype(str) <= cutoff].sort_values("as_of")
        for _, r in h.iterrows():
            week_ago[str(r["symbol"]).upper()] = float(r["fraud_score"])

    rows, hist_rows = [], []
    for sym, e in by_sym.items():
        inv_pts = e["investigative_grade"] / 4 * 100
        for_pts = e.get("forensic_score", 0.0)
        score = round(max(inv_pts, for_pts), 1)
        if score < TRACK_MIN:
            continue
        driver, reason = explain(inv_pts, e["grade_reason"],
                                 for_pts, e.get("forensic_flags", ""))
        pv = prev_by_sym.get(sym, {})
        prev_score = pd.to_numeric(pv.get("fraud_score"), errors="coerce")
        first = str(pv.get("first_flagged_at") or "") or as_of
        changed = (str(pv.get("last_changed_at") or "") or as_of)
        if pd.notna(prev_score) and float(prev_score) != score:
            changed = as_of
        s7 = week_ago.get(sym)
        if pv:
            trend = ("UP" if pd.notna(prev_score) and score > float(prev_score)
                     else "DOWN" if pd.notna(prev_score) and score < float(prev_score)
                     else "FLAT")
        else:
            trend = "NEW"
        rows.append({
            "isin": e["isin"], "symbol": sym, "company_name": e["company_name"],
            "fraud_score": score, "band": band_of(score),
            "score_driver": driver, "reason": reason,
            "investigative_grade": e["investigative_grade"],
            "grade_reason": e["grade_reason"],
            "forensic_score": e.get("forensic_score", 0.0),
            "n_forensic_flags": e.get("n_forensic_flags", 0),
            "forensic_flags": e.get("forensic_flags", ""),
            "news_hits": e["news_hits"], "sebi_actions": e["sebi_actions"],
            "nfra_actions": e["nfra_actions"],
            "first_flagged_at": first, "last_changed_at": changed,
            "trend": trend,
            "score_7d_ago": s7 if s7 is not None else None,
            "as_of": as_of, "computed_at": now_iso,
        })
        hist_rows.append({"isin": e["isin"], "symbol": sym, "as_of": as_of,
                          "fraud_score": score, "band": band_of(score),
                          "investigative_grade": e["investigative_grade"],
                          "forensic_score": e.get("forensic_score", 0.0)})

    # clearance rows: tracked yesterday, clean today -> one score-0 history row
    tracked_now = {r["symbol"] for r in rows}
    for sym, pv in prev_by_sym.items():
        if sym not in tracked_now:
            hist_rows.append({"isin": str(pv.get("isin", "")), "symbol": sym,
                              "as_of": as_of, "fraud_score": 0.0, "band": "CLEAN",
                              "investigative_grade": 0, "forensic_score": 0.0})

    snap = (pd.DataFrame(rows, columns=TRACKER_COLS)
            .sort_values("fraud_score", ascending=False).reset_index(drop=True))
    new_h = pd.DataFrame(hist_rows, columns=HISTORY_COLS)
    if hist is not None and not hist.empty:
        keep = hist[hist["as_of"].astype(str) != as_of]      # idempotent rerun
        out_h = pd.concat([keep, new_h], ignore_index=True)
    else:
        out_h = new_h
    return snap, out_h


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--names", type=str, default=None,
                    help="Comma list — restrict the snapshot (history untouched).")
    ap.add_argument("--local", action="store_true")
    ap.add_argument("--local-dir", type=str, default=None)
    ap.add_argument("--dry-run", action="store_true",
                    help="Compute and print; write nothing.")
    args = ap.parse_args()

    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
    local_dir = Path(args.local_dir) if args.local_dir else \
        Path(__file__).resolve().parent.parent / ".t4_local"
    store = Store(args.local, local_dir)
    as_of = datetime.now().strftime("%Y-%m-%d")
    now_iso = datetime.now().isoformat(timespec="seconds")
    log(f"build_fraud_tracker — mode={'LOCAL' if args.local else 'DRIVE'} "
        f"{'(dry-run)' if args.dry_run else ''}")

    inv = store.read_parquet(["company_repo", "_index", "investigative_fraud.parquet"])
    fraud = store.read_parquet(["company_repo", "_index", "fraud_risk.parquet"])
    if (inv is None or inv.empty) and (fraud is None or fraud.empty):
        log("Neither investigative_fraud nor fraud_risk parquet present — nothing to do.")
        return
    prev = store.read_parquet(["company_repo", "_index", "fraud_tracker.parquet"])
    hist = store.read_parquet(["company_repo", "_index", "fraud_tracker_history.parquet"])

    snap, out_h = build_tracker(inv, fraud, prev, hist, as_of, now_iso)
    if args.names:
        wanted = {s.strip().upper() for s in args.names.split(",") if s.strip()}
        snap = snap[snap["symbol"].isin(wanted)]
    dist = snap["band"].value_counts().to_dict() if not snap.empty else {}
    log(f"tracked: {len(snap)} companies {dist}; history rows: {len(out_h)}")

    if args.dry_run:
        for _, r in snap.head(20).iterrows():
            print(f"  {r['symbol']:<12} {r['fraud_score']:>5.0f} {r['band']:<6} "
                  f"{r['trend']:<5} {str(r['reason'])[:110]}")
        return

    store.write_df(["company_repo", "_index", "fraud_tracker.parquet"], snap)
    store.write_df(["company_repo", "_index", "fraud_tracker.csv"], snap)
    store.write_df(["company_repo", "_index", "fraud_tracker_history.parquet"], out_h)
    log("Wrote fraud_tracker.parquet/.csv + fraud_tracker_history.parquet to _index/.")


if __name__ == "__main__":
    main()
