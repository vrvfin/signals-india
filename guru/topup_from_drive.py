r"""
TOP-UP FROM DRIVE — topup_from_drive.py  (Project Guru)

refresh_daily.py --prices tops the local 25y store up from Yahoo, and Yahoo's
BSE (.BO) feed went stale: on 2026-10-18 all 2,147 BSE files stopped at
2026-09-08 while NSE files reached 2026-09-21. Phase 1 already has fresh BSE
(and NSE) daily prices on Drive data/ohlcv/<SYMBOL>.parquet (BSE API +
bhavcopy), so this step appends Drive's newer bars to any local file that is
behind.

Rules
  * APPEND ONLY: rows dated after the local file's last bar; history untouched.
  * JUNCTION CHECK first: median Drive/local close over the shared days (up to
    the last 20). Off by more than 2% means a split/bonus one source has
    adjusted and the other has not; that file is NOT written, only listed
    (fix it with scripts/repair_split_history.py, never by a silent rescale of
    25 years). Under 5 shared days the junction cannot be checked: also listed.
  * Drive keys by trading symbol; the local store by guru_key. Mapped via
    guru/data/universe_hist.parquet (nse_symbol, then bse_scrip_id, then
    bse_symbol).
  * Changed keys are ADDED to _dump_status/refresh_log.parquet so the next
    `refresh_daily.py --technical` recomputes their metrics.

Local writes only (guru/data, gitignored). Drive is read-only here.

    python guru/topup_from_drive.py --dry-run
    python guru/topup_from_drive.py --limit 50
    python guru/topup_from_drive.py
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd

GURU = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(GURU, "data")
OHLCV = os.path.join(DATA, "ohlcv_hist")
UNIVERSE = os.path.join(DATA, "universe_hist.parquet")
STATUS = os.path.join(DATA, "_dump_status")
REFRESH_LOG = os.path.join(STATUS, "refresh_log.parquet")
TOL = 0.02          # junction tolerance (same as tech_history.join_series)
JUMP = 0.40         # |1-day move| treated as bad data (= tech_history.JUMP_MAX)
MIN_OVERLAP = 5
OVERLAP_N = 20


def log(m):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {m}", flush=True)


def _drive():
    scripts = os.path.join(os.path.dirname(GURU), "scripts")
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    import build_gallery as BG          # loads .env; Drive helpers + parquet cache
    from drive_io import ParquetCache
    return BG, BG.get_drive(), ParquetCache(enabled=True)


def key_to_symbols(u: pd.DataFrame) -> dict:
    """guru_key -> candidate Drive symbols, most likely first."""
    out = {}
    for r in u.itertuples():
        cands = []
        for c in (r.nse_symbol, r.bse_scrip_id, r.bse_symbol):
            s = str(c or "").strip().upper()
            if s and s != "NAN" and s not in cands:
                cands.append(s)
        if cands and r.guru_key not in out:
            out[str(r.guru_key)] = cands
    return out


def junction(local: pd.DataFrame, drive: pd.DataFrame):
    """(ratio, n_shared) over the last OVERLAP_N shared closes, or (None, n)."""
    ov = local[["date", "close"]].merge(drive[["date", "close"]], on="date",
                                        suffixes=("_l", "_d"))
    ov = ov[(ov["close_l"] > 0) & (ov["close_d"] > 0)].sort_values("date").tail(OVERLAP_N)
    if len(ov) < MIN_OVERLAP:
        return None, len(ov)
    return float((ov["close_d"] / ov["close_l"]).median()), len(ov)


def _norm_drive(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    if "date" not in d.columns:
        d = d.reset_index().rename(columns={d.index.name or "index": "date"})
    d.columns = [str(c).lower() for c in d.columns]
    d["date"] = pd.to_datetime(d["date"], errors="coerce").dt.tz_localize(None).dt.normalize()
    for c in ("open", "high", "low", "close", "volume"):
        d[c] = pd.to_numeric(d.get(c), errors="coerce")
    return d.dropna(subset=["date", "close"]).drop_duplicates("date", keep="last").sort_values("date")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true", help="report only; write nothing")
    ap.add_argument("--limit", type=int, default=0, help="first N behind files only")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(OHLCV, "*.parquet")))
    last = {}
    for f in files:
        try:
            d = pd.read_parquet(f, columns=["date"])
            if len(d):
                last[os.path.basename(f)[:-8]] = pd.Timestamp(d["date"].max()).normalize()
        except Exception:
            pass
    newest = max(last.values())
    log(f"local store: {len(last):,} files, newest bar {newest.date()}")

    BG, drive, cache = _drive()
    folder = BG._folder(drive, "data/ohlcv")
    listing = BG._list_folder(drive, folder)
    on_drive = {str(n)[:-8].upper() for n in (listing.keys() if isinstance(listing, dict) else listing)
                if str(n).endswith(".parquet")}
    k2s = key_to_symbols(pd.read_parquet(UNIVERSE, columns=["guru_key", "nse_symbol",
                                                           "bse_scrip_id", "bse_symbol"]))
    # candidates: anything behind the local newest bar, plus everything else —
    # Drive may be ahead of the local newest too (it usually is)
    pairs, no_sym = [], 0
    for k in sorted(last):
        sym = next((s for s in k2s.get(k, []) if s in on_drive), None)
        if sym is None:
            no_sym += 1
            continue
        pairs.append((k, sym))
    if args.limit:
        # prefer the files that are furthest behind when piloting
        pairs = sorted(pairs, key=lambda p: last[p[0]])[: args.limit]
    log(f"{len(pairs):,} local files have a Drive price file; {no_sym:,} have none "
        f"(delisted / not in Phase 1)")

    t0 = time.time()
    dmap = BG._bulk_parquet(drive, folder, sorted({s for _, s in pairs}), cache=cache,
                            what="Drive OHLCV")
    log(f"downloaded in {time.time() - t0:.0f}s")

    changed, added, blocked, unchecked, jumped, current = [], 0, [], [], [], 0
    for k, sym in pairs:
        dr = dmap.get(sym)
        if dr is None or dr.empty:
            continue
        dr = _norm_drive(dr)
        new = dr[dr["date"] > last[k]]
        if new.empty:
            current += 1
            continue
        fp = os.path.join(OHLCV, f"{k}.parquet")
        loc = pd.read_parquet(fp)
        loc["date"] = pd.to_datetime(loc["date"]).dt.normalize()
        ratio, n = junction(loc, dr)
        if ratio is None:
            unchecked.append((k, sym, n, len(new)))
            continue
        if abs(ratio - 1) > TOL:
            blocked.append((k, sym, ratio, len(new)))
            continue
        # The overlap can agree while Drive's OWN newer bars jump (a split Drive
        # applied mid-series, a unit change). Found on the 2026-10-24 pilot:
        # -99.9% and +507% days appended. Same 40% bad-data rule as
        # tech_history.JUMP_MAX and the breadth jump filter.
        seq = pd.concat([loc[["date", "close"]].tail(1), new[["date", "close"]]])["close"]
        jumps = seq.pct_change().abs().dropna()
        if (jumps > JUMP).any():
            jumped.append((k, sym, float(jumps.max()), len(new)))
            continue
        add = new[["date", "open", "high", "low", "close", "volume"]].copy()
        if "adj_basis" in loc.columns:
            add["adj_basis"] = True                    # same as refresh_daily's appends
        out = (pd.concat([loc, add], ignore_index=True)
               .drop_duplicates("date", keep="last").sort_values("date"))
        changed.append(k)
        added += len(add)
        if not args.dry_run:
            out.to_parquet(fp, index=False)

    log(f"{'WOULD UPDATE' if args.dry_run else 'UPDATED'} {len(changed):,} files, "
        f"+{added:,} bars | already current {current:,} | "
        f"BLOCKED (junction off >{TOL:.0%}) {len(blocked):,} | "
        f"unchecked (<{MIN_OVERLAP} shared days) {len(unchecked):,} | "
        f"JUMP in Drive's new bars (>{JUMP:.0%}/day) {len(jumped):,}")
    for k, sym, r, nn in blocked[:15]:
        log(f"  blocked {k} ({sym}): Drive/local close ratio {r:.4f}, {nn} new bars held back")
    if len(blocked) > 15:
        log(f"  … and {len(blocked) - 15} more blocked")
    for k, sym, j, nn in jumped[:10]:
        log(f"  jump {k} ({sym}): a {j:.0%} one-day move in Drive's new bars, {nn} held back")
    for k, sym, n, nn in unchecked[:5]:
        log(f"  unchecked {k} ({sym}): only {n} shared days, {nn} new bars held back")

    after = dict(last)
    for k in changed:
        after[k] = newest if newest else after[k]
    if args.dry_run:
        log("DRY-RUN: nothing written.")
        return
    if changed:
        os.makedirs(STATUS, exist_ok=True)
        prev = (pd.read_parquet(REFRESH_LOG)["guru_key"].tolist()
                if os.path.exists(REFRESH_LOG) else [])
        keys = sorted(set(prev) | set(changed))
        pd.DataFrame({"guru_key": keys,
                      "refreshed_at": datetime.now().isoformat(timespec="seconds")}
                     ).to_parquet(REFRESH_LOG, index=False)
        log(f"refresh_log: {len(keys):,} keys queued for refresh_daily.py --technical")
    if blocked or jumped:
        rep = os.path.join(STATUS, "topup_blocked.csv")
        rows = ([(k, s, "junction_ratio", r, n) for k, s, r, n in blocked]
                + [(k, s, "jump_in_new_bars", j, n) for k, s, j, n in jumped])
        pd.DataFrame(rows, columns=["guru_key", "symbol", "reason", "value",
                                    "new_bars_held_back"]).to_csv(rep, index=False)
        log(f"blocked list -> {rep}")


if __name__ == "__main__":
    main()
