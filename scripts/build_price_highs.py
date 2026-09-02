r"""
build_price_highs.py — one small table holding every stock's long-horizon highs.

THE PROBLEM
  Phase 1's own price store is shallow: median 2.2 years, and only 26% of
  symbols carry 5 years. Every BSE name sits at exactly 2.22 years, which is the
  hard cap in fetch_bse_only_ohlcv.py, so no BSE stock can ever satisfy a
  five-year rule. Computing a "high" from that store means a 10-year high for
  one stock and an 8-month high for the next, and the bias runs one way: LESS
  history makes "near its high" trivially easy to satisfy, so the screen fills
  with the newest names.

THE POINT
  A high is a NUMBER, not a history. Storing 25 years of bars for every stock to
  answer "how far below its high is it?" costs ~239 MB; storing the answer costs
  ~300 KB. And an all-time high is a running maximum — once known, every future
  day is one comparison against today's bar.

WHERE THE DEPTH COMES FROM
  guru/data/ohlcv_hist/ — 5,062 files built during the backtest work, median 15
  years, deepest 25.2, oldest bar 2001-07-03. 78% of it clears five years
  against Phase 1's 26%.

  This is a ONE-TIME EXTRACTION, not a runtime dependency. Phase 1 owns the
  resulting table and maintains it from its own daily feed; if the guru store
  vanished the table keeps working. That distinction is why this is acceptable
  despite the plan's "no dependency on guru's store".

MAINTENANCE, AND ITS ONE HONEST LIMITATION
  --update walks today's features and raises any high today's bar exceeds. That
  is exact for the all-time high. The N-YEAR windows decay — a high set 5.5
  years ago should drop out of the 5-year window — and that cannot be done
  incrementally without the history. So between re-seeds an N-year high can only
  move UP, never down.

  That error is in the SAFE direction: a stale (too high) figure makes a stock
  look FURTHER from its high than it is, so the screen under-fires rather than
  over-fires. `as_of` records when each window was last truly computed. Re-seed
  periodically to correct the drift.

Usage:
    python scripts/build_price_highs.py --seed --dry-run   # extract, no write
    python scripts/build_price_highs.py --seed             # one-off, from guru
    python scripts/build_price_highs.py --update           # daily, from features
"""
from __future__ import annotations

import argparse
import glob
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

REPO = os.path.dirname(_SCRIPTS_DIR)
# guru/data/ is local-only and gitignored, so it is absent from a git worktree
# and from CI. --seed is a one-off run on the machine that holds it; --guru-dir
# points at it when this checkout is not that machine's main one.
GURU_OHLCV = os.path.join(REPO, "guru", "data", "ohlcv_hist")
GURU_UNIVERSE = os.path.join(REPO, "guru", "data", "universe_hist.parquet")

WINDOWS_Y = [3, 5, 10, 20, 25]
TABLE = "price_highs.parquet"


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
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def highs_from_bars(df: pd.DataFrame, windows=WINDOWS_Y) -> dict | None:
    """Long-horizon highs for one symbol. NaN where history is insufficient —
    never a silent fallback to a shorter window, which is the whole point."""
    if df is None or df.empty:
        return None
    d = df.dropna(subset=["date"]).sort_values("date")
    if d.empty:
        return None
    last = d["date"].iloc[-1]
    first = d["date"].iloc[0]
    out = {
        "first_bar": first, "last_bar": last,
        "history_years": round((last - first).days / 365.25, 2),
        "n_bars": len(d),
        "high_all_time": float(d["high"].max()),
    }
    for y in windows:
        cutoff = last - pd.DateOffset(years=y)
        if first > cutoff:
            out[f"high_{y}y"] = np.nan          # not enough history
            continue
        w = d[d["date"] >= cutoff]
        out[f"high_{y}y"] = float(w["high"].max()) if len(w) else np.nan
    return out


def phase1_symbols() -> pd.DataFrame | None:
    """Phase 1's master_list, for resolving guru's keys onto its symbols."""
    try:
        from _extractor_base import (get_drive, get_or_create_subfolder,
                                     find_file, download_bytes)
        d = get_drive()
        uid = get_or_create_subfolder(d, os.environ["GDRIVE_FOLDER_ID"], "universe")
        fid = find_file(d, uid, "master_list.csv")
        return pd.read_csv(io.BytesIO(download_bytes(d, fid))) if fid else None
    except Exception as e:
        log(f"master_list unavailable ({str(e)[:70]}) — falling back to "
            f"nse_symbol only, which reaches ~42% of the universe")
        return None


def seed_from_guru(limit: int = 0, guru_dir: str | None = None) -> pd.DataFrame:
    """Walk the local 25-year research store and extract the numbers."""
    ohlcv_dir = guru_dir or GURU_OHLCV
    uni_path = (os.path.join(os.path.dirname(ohlcv_dir), "universe_hist.parquet")
                if guru_dir else GURU_UNIVERSE)
    files = sorted(glob.glob(os.path.join(ohlcv_dir, "*.parquet")))
    if not files:
        raise SystemExit(f"no price files at {ohlcv_dir} — nothing to seed from "
                         f"(pass --guru-dir if this checkout is not the machine's "
                         f"main one; guru/data/ is gitignored)")
    if limit:
        files = files[:limit]
    log(f"reading {len(files):,} files from {ohlcv_dir}")

    # guru keys files by its own guru_key. Mapping those to the symbols Phase 1
    # uses is the whole ballgame: matching on nse_symbol alone reaches only 42%
    # of the universe and NOTHING on BSE, because guru's BSE rows carry no NSE
    # symbol. ISIN is the reliable key — both sides have it — and a cascade of
    # isin -> nse_symbol -> bse_scrip_id reaches 88.8%.
    key2sym = {}
    if os.path.exists(uni_path):
        u = pd.read_parquet(uni_path, columns=["guru_key", "nse_symbol", "isin",
                                               "bse_scrip_id"])
        ml = phase1_symbols()
        by_isin, syms = {}, set()
        if ml is not None:
            syms = set(ml["symbol"].astype(str))
            if "isin" in ml.columns:
                by_isin = dict(zip(ml["isin"].astype(str),
                                   ml["symbol"].astype(str)))
        n_isin = n_nse = n_bse = 0
        for _, r in u.iterrows():
            sym = None
            i = str(r.get("isin"))
            if i and i in by_isin:
                sym, n_isin = by_isin[i], n_isin + 1
            elif str(r.get("nse_symbol")) in syms:
                sym, n_nse = str(r["nse_symbol"]), n_nse + 1
            elif str(r.get("bse_scrip_id")) in syms:
                sym, n_bse = str(r["bse_scrip_id"]), n_bse + 1
            if sym:
                key2sym[str(r["guru_key"])] = sym
        log(f"symbol map: {len(key2sym):,} guru_key -> Phase 1 symbol "
            f"(isin {n_isin:,} / nse_symbol {n_nse:,} / bse_scrip_id {n_bse:,})")

    rows, n_unmapped, n_bad = [], 0, 0
    for i, f in enumerate(files, 1):
        key = os.path.basename(f)[:-8]
        sym = key2sym.get(key)
        if sym is None:
            # Unresolvable: keeping the guru_key as a "symbol" would add a row
            # that joins to nothing and quietly inflates the coverage numbers.
            n_unmapped += 1
            continue
        try:
            df = pd.read_parquet(f, columns=["date", "high"])
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
        except Exception:
            n_bad += 1
            continue
        h = highs_from_bars(df)
        if h is None:
            n_bad += 1
            continue
        h["symbol"] = sym
        h["source"] = "guru_hist"
        rows.append(h)
        if i % 1000 == 0:
            log(f"  {i:,}/{len(files):,}")
    log(f"extracted {len(rows):,} | {n_unmapped:,} dropped (no Phase 1 symbol) "
        f"| {n_bad:,} unreadable")
    out = pd.DataFrame(rows)
    if len(out):
        out["as_of"] = str(pd.Timestamp.today().date())
    return out


def _drive():
    from _extractor_base import get_drive
    return get_drive()


def _index_folder(drive):
    from _extractor_base import get_or_create_subfolder
    root = os.environ["GDRIVE_FOLDER_ID"]
    return get_or_create_subfolder(
        drive, get_or_create_subfolder(drive, root, "company_repo"), "_index")


def load_table(drive) -> pd.DataFrame | None:
    from _extractor_base import find_file, download_bytes
    fid = find_file(drive, _index_folder(drive), TABLE)
    if not fid:
        return None
    return pd.read_parquet(io.BytesIO(download_bytes(drive, fid)))


def save_table(drive, df: pd.DataFrame) -> None:
    from _extractor_base import upload_bytes
    buf = io.BytesIO()
    df.to_parquet(buf, index=False)
    upload_bytes(drive, _index_folder(drive), TABLE, buf.getvalue(),
                 "application/octet-stream")


def update_from_features(table: pd.DataFrame, feat: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Raise any high that today's bar exceeds.

    Exact for high_all_time. For the N-year windows this can only move a figure
    UP; the decay (a high ageing out of the window) needs a re-seed. Stale-high
    error is conservative — the stock looks further from its high than it is."""
    if table is None or table.empty or feat is None or feat.empty:
        return table, 0
    t = table.copy()
    if "symbol" not in t.columns:
        return table, 0
    t["symbol"] = t["symbol"].astype(str)
    if "symbol" not in feat.columns or "high" not in feat.columns:
        # compute_features failed, or its schema moved. Leaving the stored highs
        # untouched is right: a high can only ever be raised, so skipping a day
        # loses nothing, whereas guessing would corrupt the table permanently.
        log("update skipped — features lack symbol/high; highs left as they are")
        return table, 0
    f = feat[["symbol", "high"]].dropna(subset=["symbol"]).drop_duplicates("symbol")
    f["symbol"] = f["symbol"].astype(str)
    m = t.merge(f.rename(columns={"high": "_today_high"}), on="symbol", how="left")
    cols = ["high_all_time"] + [f"high_{y}y" for y in WINDOWS_Y]
    raised = 0
    for c in cols:
        if c not in m.columns:
            continue
        cur = pd.to_numeric(m[c], errors="coerce")
        new = pd.to_numeric(m["_today_high"], errors="coerce")
        # only where a window already exists — a NaN means "not enough history",
        # and one day's bar does not create history
        bump = cur.notna() & new.notna() & (new > cur)
        raised += int(bump.sum())
        m.loc[bump, c] = new[bump]
    return m.drop(columns=["_today_high"]), raised


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", action="store_true",
                    help="one-off full extraction from guru/data/ohlcv_hist/")
    ap.add_argument("--update", action="store_true",
                    help="daily: raise highs that today's bar exceeded")
    ap.add_argument("--dry-run", action="store_true",
                    help="compute and report, write nothing to Drive")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--guru-dir", type=str, default=None,
                    help="path to guru/data/ohlcv_hist (it is gitignored, so a "
                         "worktree or a fresh clone will not have it)")
    args = ap.parse_args()
    if not (args.seed or args.update):
        ap.error("pass --seed or --update")

    if args.seed:
        t = seed_from_guru(args.limit, args.guru_dir)
        if t.empty:
            print("nothing extracted")
            return 1
        print()
        print("=" * 68)
        print(f"PRICE HIGHS TABLE — {len(t):,} symbols")
        print("=" * 68)
        print(t["history_years"].describe(
            percentiles=[.1, .25, .5, .75, .9]).round(2).to_string())
        print()
        print("coverage by window (NaN = not enough history, by design):")
        for y in WINDOWS_Y:
            n = int(t[f"high_{y}y"].notna().sum())
            print(f"  high_{y:>2}y : {n:>5} / {len(t):,}  ({n/len(t)*100:5.1f}%)")
        size_kb = len(t.to_parquet(index=False)) / 1024
        print(f"\ntable size: {size_kb:,.0f} KB "
              f"(vs ~239,000 KB to deepen the bar store instead)")
    else:
        drive = _drive()
        t = load_table(drive)
        if t is None:
            print(f"{TABLE} not on Drive yet — run --seed first")
            return 1
        from _extractor_base import get_or_create_subfolder, find_file, download_bytes
        root = os.environ["GDRIVE_FOLDER_ID"]
        fid = find_file(drive, get_or_create_subfolder(drive, root, "features"),
                        "latest.parquet")
        if not fid:
            print("features/latest.parquet missing")
            return 1
        feat = pd.read_parquet(io.BytesIO(download_bytes(drive, fid)))
        t, raised = update_from_features(t, feat)
        log(f"raised {raised:,} high(s) on today's bar across {len(t):,} symbols")

    if args.dry_run:
        log("DRY RUN — nothing written to Drive")
        return 0

    drive = _drive() if args.seed else drive
    save_table(drive, t)
    log(f"wrote company_repo/_index/{TABLE} ({len(t):,} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
