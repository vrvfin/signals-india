r"""
build_pattern_base_rates.py — "how did this pattern usually go for similar stocks?"

For each of the 20 tech_history pattern flags, pools every past occurrence
across the local price store (guru/data/ohlcv_hist, ~5,000 securities) and
records the 5/20/60-day forward return, grouped by size bucket:
  Largecap / Midcap / Smallcap / Microcap  (TODAY's market cap — see caveat)
  first_year  = the flag fired within a stock's first 252 trading days
  all         = every tradable stock
build_pf_history.py shows the matching row next to a holding's own record,
which is what makes a thin or brand-new holding readable.

Rules (same conventions as guru/patterns/build_library.py):
  * window: last 10 years (2016-10-01 on) — today's market, not 2003's
  * TRADABLE only: 20-day median turnover >= 1 cr/day at the event (the
    live-screen floor; untradable penny names carry a fat fake tail)
  * an event whose forward window contains a >40% one-day jump (bad data /
    unadjusted split) is dropped for that horizon
  * episodes: a flag re-firing within 10 bars counts once (as on the page)

CAVEAT (survivorship + size drift): the bucket uses each stock's market cap
TODAY, not at the time of the event, and the store holds only names still in
the universe. Both bias the rates upward; the page says so.

Output (local, gitignored): guru/data/metrics/pattern_base_rates.parquet
Read-only on Drive (market cap only). No Gemini.

    python scripts/build_pattern_base_rates.py --dry-run --limit 200
    python scripts/build_pattern_base_rates.py
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
import time

import numpy as np
import pandas as pd

_SD = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _SD)
import tech_history as T                 # noqa: E402

ROOT = os.path.dirname(_SD)
STORE = os.path.join(ROOT, "guru", "data", "ohlcv_hist")
UNIVERSE_HIST = os.path.join(ROOT, "guru", "data", "universe_hist.parquet")
OUT = os.path.join(ROOT, "guru", "data", "metrics", "pattern_base_rates.parquet")
SINCE = pd.Timestamp("2016-10-01")
MIN_TURNOVER_CR = 1.0                    # = guru/patterns/build_library.py
STORE_START = pd.Timestamp("2001-08-01")  # earlier first bar = history truncated, not a listing


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _sym_for_files():
    """price-file stem (ISIN or BSE<code>) -> symbol used by the market-cap map."""
    u = pd.read_parquet(UNIVERSE_HIST, columns=["isin", "bse_code", "nse_symbol", "bse_scrip_id"])
    out = {}
    for _, r in u.iterrows():
        sym = str(r["nse_symbol"] or "").strip() or str(r["bse_scrip_id"] or "").strip()
        if not sym or sym.lower() == "nan":
            continue
        out[str(r["isin"])] = sym.upper()
        if pd.notna(r["bse_code"]):
            out[f"BSE{str(r['bse_code']).split('.')[0]}"] = sym.upper()
    return out


def stock_events(df: pd.DataFrame) -> dict:
    """{pattern: (first_year bool[n], fwd float[n, len(HORIZONS)])} for one stock;
    NaN where the outcome is unknown or crosses a bad-data bar."""
    df = T._norm(df)
    if len(df) < 300:
        return {}
    listed_in_store = df["date"].iloc[0] > STORE_START
    first_year_end = df["date"].iloc[min(T.YEAR, len(df) - 1)] if listed_in_store else None
    keep = (df["date"] >= SINCE).to_numpy()
    if keep.sum() < 120:
        return {}
    c = df["close"].to_numpy(dtype=float)
    n = len(c)
    turn = (df["close"] * df["volume"] / 1e7).rolling(20, min_periods=10).median().to_numpy()
    bad = (df["close"].pct_change().abs() > T.JUMP_MAX).to_numpy()
    bad_cum = np.concatenate([[0], np.cumsum(bad)])          # bad bars in (p, p+h]
    fy_all = (df["date"] <= first_year_end).to_numpy() if first_year_end is not None \
        else np.zeros(n, dtype=bool)
    series = dict(T.pattern_series(df))
    # BASELINE: every 10th day, pattern or not — the "any ordinary day" a
    # flag has to beat before it means anything
    base = np.zeros(n, dtype=bool)
    base[::10] = True
    series["any_day"] = ("base", pd.Series(base))
    out = {}
    for name, (_, s) in series.items():
        pos = np.flatnonzero(s.to_numpy()) if name == "any_day" else T.episodes(s, cooldown=10)
        pos = np.asarray(pos, dtype=int)
        if len(pos) == 0:
            continue
        pos = pos[keep[pos] & (np.nan_to_num(turn[pos], nan=-1) >= MIN_TURNOVER_CR)]
        if len(pos) == 0:
            continue
        fwd = np.full((len(pos), len(T.HORIZONS)), np.nan)
        for k, h in enumerate(T.HORIZONS):
            q = pos + h
            ok = q < n
            okq = ok.copy()
            okq[ok] = (bad_cum[q[ok] + 1] - bad_cum[pos[ok] + 1]) == 0
            fwd[okq, k] = c[q[okq]] / c[pos[okq]] - 1
        out[name] = (fy_all[pos], fwd)
    return out


def aggregate(acc: dict) -> pd.DataFrame:
    """acc: {(bucket, pattern): [(stock_id int[n], fwd float[n, H])]} -> one row
    per (bucket, pattern, h)."""
    rows = []
    for (bucket, pattern), parts in acc.items():
        ids = np.concatenate([p[0] for p in parts])
        fwd = np.concatenate([p[1] for p in parts])
        for k, h in enumerate(T.HORIZONS):
            m = ~np.isnan(fwd[:, k])
            r = fwd[m, k]
            rows.append({"bucket": bucket, "pattern": pattern, "h": h,
                         "n_events": int(len(r)), "n_stocks": int(len(np.unique(ids[m]))),
                         "median": float(np.median(r)) if len(r) else np.nan,
                         "pct_up": float((r > 0).mean() * 100) if len(r) else np.nan,
                         "p10": float(np.quantile(r, 0.10)) if len(r) else np.nan,
                         "p90": float(np.quantile(r, 0.90)) if len(r) else np.nan})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true", help="compute + print, write nothing")
    ap.add_argument("--limit", type=int, default=0, help="first N price files only (testing)")
    args = ap.parse_args()

    import build_pf_history as H         # Drive market cap (read-only), same map as the page
    import build_gallery as BG
    mcap = H._mcap_map(BG.get_drive())
    file_sym = _sym_for_files()
    files = sorted(glob.glob(os.path.join(STORE, "*.parquet")))
    if args.limit:
        files = files[: args.limit]
    log(f"{len(files):,} price files; market cap known for {len(mcap):,} symbols")

    acc, n_used, n_ev, t0 = {}, 0, 0, time.time()
    for i, f in enumerate(files, 1):
        stem = os.path.splitext(os.path.basename(f))[0]
        sym = file_sym.get(stem)
        seg = H._segment(mcap.get(sym)) if sym else None
        try:
            ev = stock_events(pd.read_parquet(f))
        except Exception as e:                       # one bad file must not stop the run
            log(f"  {stem}: skipped ({type(e).__name__}: {e})")
            continue
        if ev:
            n_used += 1
        for name, (fy, fwd) in ev.items():
            sid = np.full(len(fwd), i, dtype=np.int32)
            n_ev += len(fwd)
            acc.setdefault(("all", name), []).append((sid, fwd))
            if seg:
                acc.setdefault((seg, name), []).append((sid, fwd))
            if fy.any():
                acc.setdefault(("first_year", name), []).append((sid[fy], fwd[fy]))
        if i % 500 == 0:
            log(f"  {i:,}/{len(files):,} files ({time.time() - t0:.0f}s)")
    log(f"{n_used:,} stocks contributed {n_ev:,} events ({time.time() - t0:.0f}s)")
    if not acc:
        log("Nothing computed."); return
    out = aggregate(acc)
    out["built"] = pd.Timestamp.now().normalize()
    out["since"] = SINCE
    show = out[(out["h"] == 20) & (out["bucket"].isin(["all", "first_year"]))]
    with pd.option_context("display.width", 160, "display.max_rows", 60):
        print(show[["bucket", "pattern", "n_events", "n_stocks", "median", "pct_up", "p10"]]
              .round(3).to_string(index=False))
    if args.dry_run:
        log("DRY-RUN — nothing written.")
        return
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    out.to_parquet(OUT, index=False)
    log(f"wrote {OUT}  ({len(out):,} rows)")


if __name__ == "__main__":
    main()
