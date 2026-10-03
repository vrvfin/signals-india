"""
update_market_cap.py — DAILY market cap = weekly share count x latest close.

Runs at the END of Phase 1 (after compute_features has written
features/latest.parquet). Reads:
  universe/share_count.csv      (build_share_count.py, weekly)
  features/latest.parquet       symbol, close, date
  universe/market_cap.csv       the existing file (Yahoo values, or yesterday's)
Writes universe/market_cap.csv with EXACTLY the same columns as before —
symbol, market_cap_cr, mcap_segment — so no reader changes. A symbol with no
share count or no close keeps its existing row untouched (the Yahoo fallback).
Symbols are only ever added, never dropped.

compute_features joins market_cap.csv at the START of the next Phase 1, so
features carry the previous session's market cap (a one-day lag, by design of
running this at the end).

Usage:
    python scripts/update_market_cap.py --dry-run
    python scripts/update_market_cap.py
"""
from __future__ import annotations

import argparse
import io
import os
import sys

_SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

import pandas as pd
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(_SCRIPTS_DIR), ".env"))

from _extractor_base import (get_drive, get_or_create_subfolder, find_file,
                             download_bytes, upload_bytes, log)
from enrich_market_cap import assign_segment

OUT_COLS = ["symbol", "market_cap_cr", "mcap_segment"]   # unchanged contract


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true",
                    help="Compute and report the changes; write nothing.")
    ap.add_argument("--share-count-csv", default="",
                    help="Test only: read share counts from this local CSV "
                         "(e.g. build_share_count.py --dry-run output).")
    args = ap.parse_args()

    drive = get_drive()
    root = os.environ["GDRIVE_FOLDER_ID"]
    uni_id = get_or_create_subfolder(drive, root, "universe")
    feat_id = get_or_create_subfolder(drive, root, "features")

    if args.share_count_csv:
        if not args.dry_run:
            log("--share-count-csv is for testing; use it with --dry-run.")
            return 1
        sc = pd.read_csv(args.share_count_csv)
    else:
        sc_id = find_file(drive, uni_id, "share_count.csv")
        if not sc_id:
            log("universe/share_count.csv not found — run build_share_count.py; "
                "market_cap.csv left unchanged.")
            return 0
        sc = pd.read_csv(io.BytesIO(download_bytes(drive, sc_id)))
    sc["shares_cr"] = pd.to_numeric(sc["shares_cr"], errors="coerce")
    sc = sc.dropna(subset=["shares_cr"]).drop_duplicates("symbol", keep="last")

    fid = find_file(drive, feat_id, "latest.parquet")
    if not fid:
        log("features/latest.parquet missing — market_cap.csv left unchanged.")
        return 0
    feat = pd.read_parquet(io.BytesIO(download_bytes(drive, fid)),
                           columns=["symbol", "close", "date"])
    feat["close"] = pd.to_numeric(feat["close"], errors="coerce")

    mc_id = find_file(drive, uni_id, "market_cap.csv")
    old = (pd.read_csv(io.BytesIO(download_bytes(drive, mc_id)))
           if mc_id else pd.DataFrame(columns=OUT_COLS))
    old["symbol"] = old["symbol"].astype(str)
    old["market_cap_cr"] = pd.to_numeric(old["market_cap_cr"], errors="coerce")

    calc = (sc[["symbol", "shares_cr"]].astype({"symbol": str})
            .merge(feat.astype({"symbol": str}), on="symbol", how="inner"))
    calc = calc[calc["close"] > 0]
    calc["new_cap"] = calc["shares_cr"] * calc["close"]

    out = old[OUT_COLS].copy() if len(old) else pd.DataFrame(columns=OUT_COLS)
    new_map = dict(zip(calc["symbol"], calc["new_cap"]))
    before = dict(zip(out["symbol"], out["market_cap_cr"]))
    upd = out["symbol"].isin(new_map)
    out.loc[upd, "market_cap_cr"] = out.loc[upd, "symbol"].map(new_map)
    added = [s for s in new_map if s not in before]
    if added:
        out = pd.concat([out, pd.DataFrame({"symbol": added,
                                            "market_cap_cr": [new_map[s] for s in added]})],
                        ignore_index=True)
    out["market_cap_cr"] = pd.to_numeric(out["market_cap_cr"], errors="coerce").round(2)
    out["mcap_segment"] = out["market_cap_cr"].map(
        lambda v: assign_segment(None if pd.isna(v) else float(v)))
    out = out[OUT_COLS]

    kept = len(out) - int(upd.sum()) - len(added)
    log(f"market_cap.csv: {len(old):,} rows before -> {len(out):,} after | "
        f"recomputed {int(upd.sum()):,} | added {len(added):,} | kept as-is {kept:,}")
    chg = pd.DataFrame({"symbol": list(new_map),
                        "old": [before.get(s) for s in new_map],
                        "new": list(new_map.values())}).dropna(subset=["old"])
    chg = chg[chg["old"] > 0]
    chg["pct"] = (chg["new"] / chg["old"] - 1) * 100
    seg_old = dict(zip(old["symbol"], old.get("mcap_segment", pd.Series(dtype=str))))
    seg_new = dict(zip(out["symbol"], out["mcap_segment"]))
    moved = sum(1 for s in new_map if s in seg_old and seg_old[s] != seg_new.get(s))
    log(f"change vs previous value: median {chg['pct'].abs().median():.1f}% | "
        f">25% off: {int((chg['pct'].abs() > 25).sum()):,} | segment changed: {moved:,}")
    for s in ("LAURUSLABS", "TCS"):
        r = chg[chg["symbol"] == s]
        if len(r):
            log(f"  {s}: {r['old'].iloc[0]:,.0f} -> {r['new'].iloc[0]:,.0f} cr")
    if args.dry_run:
        print(chg.reindex(chg["pct"].abs().sort_values(ascending=False).index)
              .head(15).to_string(index=False))
        log("[DRY-RUN] market_cap.csv not written")
        return 0
    upload_bytes(drive, uni_id, "market_cap.csv", out.to_csv(index=False).encode(),
                 "text/csv", existing_id=mc_id)
    log("wrote universe/market_cap.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
