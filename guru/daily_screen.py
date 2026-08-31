r"""
DAILY SCREEN — daily_screen.py  (Project Guru)

Runs the CHOSEN rules against TODAY's data and surfaces which stocks satisfy
what. Designed to run daily.

LEVERAGES PHASE 1 — does not re-fetch anything:
  prices        <- Phase 1 Drive  data/ohlcv/        (daily.yml, Mon-Fri 16:00 IST)
  market cap    <- Phase 1 Drive  universe/market_cap.csv
  fundamentals  <- Phase 1 Drive  fundamentals/statements/ + summary.parquet
                                  (fundamentals.yml, Mon 06:30 IST)
Only the last PRICE_YEARS of price history is cached locally (the longest rule
lookback is 36 months), so the cache stays small instead of the 25-year
research store.

Pipeline:
  1. universe   = market cap >= MIN_MCAP_CR
  2. sync       = pull changed Phase 1 price files into guru/data/live_ohlcv/
  3. metrics    = compute only the ~21 metrics the chosen rules actually need
  4. evaluate   = test each rule's clauses against the latest values
  5. output     = per stock: which rules fire, how many, status

Output: guru/DAILY_SCREEN_<date>.xlsx
  Conviction  — stocks ranked by how many rules they satisfy
  Coverage    — >=N unique stocks per rule (breadth list)
  By_Rule     — every (rule, stock) pair
  Status      — data freshness + counts at each funnel step

Usage:
    python guru/daily_screen.py --sync          # pull latest Phase 1 prices, then screen
    python guru/daily_screen.py                 # screen on the existing cache
    python guru/daily_screen.py --min-mcap 100 --min-per-rule 5
"""
from __future__ import annotations
import argparse, glob, io, os, sys
from datetime import datetime
import numpy as np, pandas as pd

GURU = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(GURU, "data")
LIVE = os.path.join(DATA, "live_ohlcv")
BT = os.path.join(GURU, "backtest")
SCRIPTS = os.path.join(os.path.dirname(GURU), "scripts")
if SCRIPTS not in sys.path:
    sys.path.insert(0, SCRIPTS)
from pathlib import Path
from dotenv import load_dotenv
load_dotenv(Path(GURU).parent / ".env")

RULES_X = os.path.join(os.path.dirname(GURU), "Project_Guru", "rule_template.xlsx")
CHOSEN = os.path.join(GURU, "backtest", "_chosen_rules.parquet")
PRICE_YEARS = 4          # longest rule lookback is 36m; 4y gives headroom
MIN_MCAP_CR = 100


def log(m): print(f"[{datetime.now().strftime('%H:%M:%S')}] {m}", flush=True)


# ---------------------------------------------------------------- sync
def sync_prices(symbols: set, limit: int = 0):
    """pull Phase 1 OHLCV from Drive for the given NSE symbols (incremental:
    skips files whose Drive modifiedTime is not newer than our cached copy)."""
    from _extractor_base import (get_drive, get_or_create_subfolder, download_bytes)
    d = get_drive()
    root = os.environ["GDRIVE_FOLDER_ID"]
    oid = get_or_create_subfolder(d, get_or_create_subfolder(d, root, "data"), "ohlcv")
    files, tok = [], None
    while True:
        r = d.files().list(q=f"'{oid}' in parents and trashed=false",
                           fields="nextPageToken, files(id,name,modifiedTime)",
                           pageSize=1000, pageToken=tok).execute()
        files += r.get("files", [])
        tok = r.get("nextPageToken")
        if not tok:
            break
    log(f"Phase 1 price files on Drive: {len(files):,}")
    os.makedirs(LIVE, exist_ok=True)
    stamp_p = os.path.join(DATA, "_live_sync_stamps.parquet")
    stamps = {}
    if os.path.exists(stamp_p):
        s = pd.read_parquet(stamp_p)
        stamps = dict(zip(s["name"], s["modified"]))
    want = [f for f in files if f["name"].replace(".parquet", "") in symbols]
    if limit:
        want = want[:limit]
    log(f"in-universe files to consider: {len(want):,}")
    cutoff = pd.Timestamp.now() - pd.DateOffset(years=PRICE_YEARS)
    n_new = n_skip = 0
    for i, f in enumerate(want, 1):
        if stamps.get(f["name"]) == f["modifiedTime"]:
            n_skip += 1
            continue
        try:
            df = pd.read_parquet(io.BytesIO(download_bytes(d, f["id"])))
            df["date"] = pd.to_datetime(df["date"])
            df = df[df["date"] >= cutoff]          # keep the cache lean
            if len(df) < 30:
                continue
            df.to_parquet(os.path.join(LIVE, f["name"]), index=False)
            stamps[f["name"]] = f["modifiedTime"]
            n_new += 1
        except Exception:
            continue
        if i % 250 == 0:
            log(f"  {i}/{len(want)} (updated {n_new}, unchanged {n_skip})")
            pd.DataFrame({"name": list(stamps), "modified": list(stamps.values())
                          }).to_parquet(stamp_p, index=False)
    pd.DataFrame({"name": list(stamps), "modified": list(stamps.values())
                  }).to_parquet(stamp_p, index=False)
    log(f"SYNC DONE: {n_new:,} updated, {n_skip:,} unchanged")


# ---------------------------------------------------------------- metrics
def tech_metrics(df: pd.DataFrame) -> dict:
    """the technical metrics the chosen rules need, at the LATEST date."""
    df = df.sort_values("date").reset_index(drop=True)
    c, v = df["close"], df["volume"]
    if len(c) < 60:
        return {}
    out = {"last_date": df["date"].iloc[-1], "close": float(c.iloc[-1])}
    def ret(n):
        return float(c.iloc[-1] / c.iloc[-1 - n] - 1) * 100 if len(c) > n else np.nan
    out["price_return_1m_pct"] = ret(21); out["price_return_3m_pct"] = ret(63)
    out["price_return_6m_pct"] = ret(126); out["price_return_12m_pct"] = ret(252)
    out["pct_from_all_time_high"] = float(c.iloc[-1] / c.max() - 1) * 100
    w52 = c.tail(252)
    out["pct_from_52w_high"] = float(c.iloc[-1] / w52.max() - 1) * 100
    out["recovery_pct_from_low"] = float(c.iloc[-1] / w52.min() - 1) * 100
    out["volume_ratio_20d_avg"] = (float(v.iloc[-1] / v.tail(20).mean())
                                   if v.tail(20).mean() > 0 else np.nan)
    r = c.pct_change()
    out["volatility_20d_pct"] = float(r.tail(20).std() * np.sqrt(252) * 100)
    out["single_day_return_pct"] = float(r.iloc[-1] * 100)
    ma50, ma200 = c.rolling(50).mean(), c.rolling(200).mean()
    out["ma_cross_50_200"] = int(ma50.iloc[-1] >= ma200.iloc[-1]) if len(c) >= 200 else np.nan
    tr = pd.concat([(df.high - df.low), (df.high - c.shift()).abs(),
                    (df.low - c.shift()).abs()], axis=1).max(axis=1)
    atr = tr.rolling(14).mean()
    out["atr_expansion_ratio"] = (float(atr.iloc[-1] / atr.iloc[-64])
                                  if len(atr) > 64 and atr.iloc[-64] > 0 else np.nan)
    wk = c.resample("W-FRI", on=None) if False else None
    w = df.set_index("date")["close"].resample("W-FRI").last().dropna()
    up = (w.diff() > 0)
    streak = 0
    for x in reversed(up.tolist()):
        if x:
            streak += 1
        else:
            break
    out["consecutive_up_weeks"] = streak
    out["days_since_listing"] = int((df["date"].iloc[-1] - df["date"].iloc[0]).days)
    return out


def fund_metrics(st: pd.DataFrame) -> dict:
    """fundamental metrics from a Phase 1 statements file (long format)."""
    out = {}
    if st.empty or "statement" not in st.columns:
        return out
    q = st[st.statement == "quarterly_pl"]
    if q.empty:
        return out
    w = q.pivot_table(index="period", columns="line_item", values="value", aggfunc="first")
    def dt(p):
        try:
            return pd.to_datetime(p, format="%b %Y")
        except Exception:
            return pd.NaT
    w["_d"] = [dt(p) for p in w.index]
    w = w.dropna(subset=["_d"]).sort_values("_d")
    if len(w) < 5:
        return out
    sales, npf, eps = w.get("Sales"), w.get("Net Profit"), w.get("EPS in Rs")
    if sales is not None and len(sales) > 4:
        yoy = (sales / sales.shift(4) - 1) * 100
        out["sales_yoy_pct"] = float(yoy.iloc[-1]) if pd.notna(yoy.iloc[-1]) else np.nan
        # consecutive-quarter streaks for the sustained-growth rules
        for thr in (10, 15, 20, 25):
            s = 0
            for x in reversed(yoy.dropna().tolist()):
                if x >= thr:
                    s += 1
                else:
                    break
            out[f"sales_yoy_streak_{thr}"] = s
    if npf is not None and len(npf) > 4:
        y = (npf / npf.shift(4) - 1) * 100
        out["profit_yoy_pct"] = float(y.iloc[-1]) if pd.notna(y.iloc[-1]) else np.nan
    if eps is not None and len(eps) > 4:
        y = (eps / eps.shift(4) - 1) * 100
        out["eps_yoy_pct"] = float(y.iloc[-1]) if pd.notna(y.iloc[-1]) else np.nan
        # streaks at the EXACT thresholds the chosen rules use (15/25/300)
        for thr in (15, 25, 300):
            s = 0
            for x in reversed(y.dropna().tolist()):
                if x >= thr:
                    s += 1
                else:
                    break
            out[f"eps_yoy_streak_{thr}"] = s
    if "OPM %" in w.columns and len(w) > 4:
        out["margin_yoy_change_pct"] = float(w["OPM %"].iloc[-1] - w["OPM %"].iloc[-5])
    out["latest_quarter"] = str(w.index[-1])

    # ---- interest coverage: (PBT + Interest) / Interest -------------------
    # one of the strongest validated rules (77-96% win rates) and it was silently
    # missing from the daily path, so those rules produced zero hits.
    pbt, intr = w.get("Profit before tax"), w.get("Interest")
    if pbt is not None and intr is not None and len(w) >= 4:
        p4, i4 = pbt.tail(4).sum(), intr.tail(4).sum()
        if pd.notna(i4) and i4 > 0:
            out["interest_coverage_ratio"] = float((p4 + i4) / i4)
        elif pd.notna(i4) and i4 == 0:
            out["interest_coverage_ratio"] = 999.0        # no debt cost = uncapped
        # streaks for the "for N quarters" variants
        icr_q = (pbt + intr) / intr.replace(0, np.nan)
        for thr in (5, 10):
            s = 0
            for x in reversed(icr_q.dropna().tolist()):
                if x >= thr:
                    s += 1
                else:
                    break
            out[f"icr_streak_{thr}"] = s
    # ---- TTM EPS (for PE) --------------------------------------------------
    if eps is not None and len(eps) >= 4:
        t = eps.tail(4).sum()
        if pd.notna(t) and t != 0:
            out["ttm_eps"] = float(t)

    # ---- debt / equity -----------------------------------------------------
    # Phase 1's summary.parquet HAS a debt_to_equity column but it is empty for
    # all 5,617 companies, so we compute it here: Borrowings / (Equity+Reserves).
    bs = st[st.statement == "balance_sheet"]
    if not bs.empty:
        bw = bs.pivot_table(index="period", columns="line_item", values="value",
                            aggfunc="first")
        bw["_d"] = [dt(p) for p in bw.index]
        bw = bw.dropna(subset=["_d"]).sort_values("_d")
        if len(bw):
            eq, res, bor = bw.get("Equity Capital"), bw.get("Reserves"), bw.get("Borrowings")
            if eq is not None and res is not None and bor is not None:
                nw = float(eq.iloc[-1]) + float(res.iloc[-1])
                if nw > 0:
                    out["debt_to_equity"] = float(bor.iloc[-1]) / nw

    # ---- annual: 5y profit CAGR -------------------------------------------
    a = st[st.statement == "annual_pl"]
    if not a.empty:
        aw = a.pivot_table(index="period", columns="line_item", values="value",
                           aggfunc="first")
        aw["_d"] = [dt(p) for p in aw.index]
        aw = aw.dropna(subset=["_d"]).sort_values("_d")
        npa = aw.get("Net Profit")
        if npa is not None and len(npa) >= 6:
            first, last = npa.iloc[-6], npa.iloc[-1]
            if pd.notna(first) and pd.notna(last) and first > 0 and last > 0:
                out["profit_cagr_pct_5y"] = float(((last / first) ** (1 / 5) - 1) * 100)
        sa = aw.get("Sales")
        if sa is not None and len(sa) >= 6:
            f2, l2 = sa.iloc[-6], sa.iloc[-1]
            if pd.notna(f2) and pd.notna(l2) and f2 > 0 and l2 > 0:
                out["sales_cagr_pct_5y"] = float(((l2 / f2) ** (1 / 5) - 1) * 100)
    return out


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sync", action="store_true", help="pull latest Phase 1 prices first")
    ap.add_argument("--min-mcap", type=float, default=MIN_MCAP_CR)
    ap.add_argument("--min-per-rule", type=int, default=5)
    ap.add_argument("--min-turnover", type=float, default=1.0,
                    help="minimum avg 20d traded value, Rs cr/day")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--diagnose", action="store_true",
                    help="report, per chosen rule, whether it fired and if not "
                         "WHICH clause blocked it — then exit without writing")
    ap.add_argument("--no-publish", action="store_true",
                    help="skip uploading guru_picks.parquet to Drive "
                         "(the local copy is always written)")
    args = ap.parse_args()

    status = []
    # ---- current VIX (market regime clause) ----
    global VIX_NOW
    VIX_NOW = np.nan
    for vp in (os.path.join(DATA, "macro_hist", "INDIA_VIX.parquet"),):
        if os.path.exists(vp):
            vdf = pd.read_parquet(vp, columns=["date", "close"]).sort_values("date")
            VIX_NOW = float(vdf["close"].iloc[-1])
            log(f"india_vix = {VIX_NOW:.1f} (as of {pd.to_datetime(vdf['date'].iloc[-1]).date()})")

    # ---- universe: market cap filter (Phase 1 supplies mcap) ----
    mc = pd.read_csv(os.path.join(DATA, "market_cap_current.csv"))
    mcap_col = next(c for c in mc.columns if "cap" in c.lower())
    mc[mcap_col] = pd.to_numeric(mc[mcap_col], errors="coerce")
    uni = mc[mc[mcap_col] >= args.min_mcap].copy()
    symbols = set(uni["symbol"].astype(str))
    log(f"universe: {len(mc):,} -> {len(uni):,} with mcap >= {args.min_mcap:g}cr")
    status.append(("universe with mcap", len(mc)))
    status.append((f"after mcap >= {args.min_mcap:g}cr", len(uni)))

    if args.sync:
        sync_prices(symbols, args.limit)

    # ---- metrics per stock ----
    files = glob.glob(os.path.join(LIVE, "*.parquet"))
    if args.limit:
        files = files[:args.limit]
    log(f"computing metrics for {len(files):,} cached price files")
    st_dir = os.path.join(DATA, "fundamentals_hist")
    uni_map = pd.read_parquet(os.path.join(DATA, "universe_hist.parquet"),
                              columns=["guru_key", "nse_symbol", "name"])
    sym2key = {str(s).strip(): k for k, s in zip(uni_map.guru_key, uni_map.nse_symbol)
               if isinstance(s, str)}
    sym2name = {str(s).strip(): n for n, s in zip(uni_map.name, uni_map.nse_symbol)
                if isinstance(s, str)}
    summ_p = os.path.join(DATA, "fundamentals_summary_current.parquet")
    summ = pd.read_parquet(summ_p).set_index("symbol") if os.path.exists(summ_p) else pd.DataFrame()

    rows = []
    for i, f in enumerate(files, 1):
        sym = os.path.basename(f)[:-8]
        if sym not in symbols:
            continue
        try:
            px = pd.read_parquet(f)
            m = tech_metrics(px)
            if not m:
                continue
            m["symbol"] = sym
            m["name"] = sym2name.get(sym, sym)
            m["market_cap_cr"] = float(uni.loc[uni.symbol == sym, mcap_col].iloc[0]) \
                if (uni.symbol == sym).any() else np.nan
            m["turnover_20d_cr"] = float((px["close"] * px["volume"]).tail(20).mean() / 1e7)
            key = sym2key.get(sym)
            if key:
                sp = os.path.join(st_dir, f"{key}.parquet")
                if os.path.exists(sp):
                    m.update(fund_metrics(pd.read_parquet(sp)))
            if len(summ) and sym in summ.index:
                srow = summ.loc[sym]
                for c in ("roce_pct", "roe_pct", "debt_to_equity", "pe"):
                    if c in srow.index and pd.notna(srow[c]):
                        m[c if c != "pe" else "pe_ratio"] = float(srow[c])
            # PE PERCENTILE: where today's PE sits in this stock's own history.
            # Built from the cached price series x TTM EPS (EPS moves quarterly,
            # price daily) — a stock-relative valuation gauge, not a market one.
            if m.get("ttm_eps"):
                pe_series = px["close"] / m["ttm_eps"]
                pe_series = pe_series[pe_series > 0]
                if len(pe_series) > 100:
                    cur = float(px["close"].iloc[-1] / m["ttm_eps"])
                    if cur > 0:
                        m["pe_ratio"] = cur
                        m["pe_percentile"] = float((pe_series < cur).mean() * 100)
            m["india_vix"] = VIX_NOW
            rows.append(m)
        except Exception:
            continue
        if i % 500 == 0:
            log(f"  {i}/{len(files)}")
    M = pd.DataFrame(rows)
    log(f"stocks with computable metrics: {len(M):,}")
    status.append(("stocks with metrics", len(M)))
    if M.empty:
        log("nothing to screen — run with --sync first"); return
    M.to_parquet(os.path.join(DATA, "live_metrics.parquet"), index=False)

    # ---- evaluate chosen rules ----
    chosen = pd.read_parquet(CHOSEN)
    cl = pd.read_excel(RULES_X, "Clauses")
    names = pd.read_excel(RULES_X, "Rules").set_index("rule_id")["rule_name"].to_dict()
    # BUGFIX: rules store a generic 'price_return_pct' with the WINDOW only in the
    # rule name ("price up >= 20% in 3 months"). The backtest engine resolves this;
    # the daily path did not, so 5 rules silently produced zero hits.
    import re as _re
    _WINDOWS = [1, 3, 6, 12]

    def resolve_metric(metric: str, rule_name: str) -> str:
        if metric != "price_return_pct":
            return metric
        m = _re.findall(r"(\d+)\s*month", str(rule_name), _re.I)
        n = int(m[-1]) if m else 12
        n = min(_WINDOWS, key=lambda w: abs(w - n))
        return f"price_return_{n}m_pct"

    # Pretty names for the evidence line — the raw column names are ours, not
    # something a reader should have to decode on a chart.
    _PRETTY = {"roce_pct": "ROCE", "roe_pct": "ROE", "opm_pct": "OPM",
               "npm_pct": "net margin", "debt_to_equity": "debt/equity",
               "interest_coverage_ratio": "interest cover",
               "sales_yoy_pct": "sales YoY", "eps_yoy_pct": "EPS YoY",
               "pat_yoy_pct": "PAT YoY", "price_return_1m_pct": "1M return",
               "price_return_3m_pct": "3M return", "price_return_6m_pct": "6M return",
               "price_return_12m_pct": "12M return", "market_cap_cr": "market cap",
               "turnover_20d_cr": "turnover", "pe_ratio": "P/E",
               "promoter_holding_pct": "promoter holding"}

    def _pretty(m):
        return _PRETTY.get(m, str(m).replace("_pct", "").replace("_", " "))

    hits = []
    diag = {}          # rule_id -> (verdict, detail) for --diagnose
    for rid, g in cl[cl.rule_id.isin(chosen.rule_id.unique())].groupby("rule_id"):
        mask = pd.Series(True, index=M.index)
        usable = True
        # per-clause pass counts, so a rule with zero hits can name the clause
        # that actually blocked it rather than just reporting "no matches"
        clause_pass = []
        # (display label, column to read the actual value from, operator,
        #  threshold, is_streak) — what makes this rule fire, kept so each hit
        #  can show its own numbers rather than just the rule's title.
        spec = []
        for _, c in g.iterrows():
            met, op, thr = c["metric"], str(c["operator"]).strip(), c["threshold_value"]
            met = resolve_metric(met, names.get(rid, ""))
            off = int(c.get("period_offset", 0) or 0)
            # sustained-growth clauses -> use the streak metric we computed.
            # BUGFIX: previously bucketed to the nearest LOWER precomputed
            # threshold (a 30% rule was tested at 25% = too lenient). Now we
            # require an exact bucket, else fall back to per-quarter evaluation.
            if met == "sales_yoy_pct" and off > 0:
                col = f"sales_yoy_streak_{int(float(thr))}"
                if col in M.columns:
                    _cm = (M[col] >= off + 1)
                    mask &= _cm
                    clause_pass.append((f"sales YoY ≥{int(float(thr))}% for {off+1}q",
                                        int(_cm.sum())))
                    spec.append((f"sales YoY ≥{int(float(thr))}%", col,
                                 "≥", off + 1, True))
                    continue
                usable = False
                diag[rid] = ("UNUSABLE", f"needs streak column {col!r} (not built)")
                break
            if met == "eps_yoy_pct" and off > 0:
                col = f"eps_yoy_streak_{int(float(thr))}"
                if col in M.columns:
                    _cm = (M[col] >= off + 1)
                    mask &= _cm
                    clause_pass.append((f"EPS YoY ≥{int(float(thr))}% for {off+1}q",
                                        int(_cm.sum())))
                    spec.append((f"EPS YoY ≥{int(float(thr))}%", col,
                                 "≥", off + 1, True))
                    continue
                usable = False
                diag[rid] = ("UNUSABLE", f"needs streak column {col!r} (not built)")
                break
            # sustained interest-coverage clauses -> streak metric
            if met == "interest_coverage_ratio" and off > 0:
                col = f"icr_streak_{10 if float(thr) >= 10 else 5}"
                if col in M.columns:
                    _cm = (M[col] >= off + 1)
                    mask &= _cm
                    clause_pass.append((f"interest cover for {off+1}q", int(_cm.sum())))
                    spec.append((f"interest cover ≥{10 if float(thr) >= 10 else 5}",
                                 col, "≥", off + 1, True))
                    continue
                usable = False
                diag[rid] = ("UNUSABLE", f"needs streak column {col!r} (not built)")
                break
            if met not in M.columns:
                usable = False
                diag[rid] = ("UNUSABLE", f"metric {met!r} is not computed by the daily path")
                break
            v = pd.to_numeric(M[met], errors="coerce")
            t = float(thr) if str(thr).replace(".", "").replace("-", "").isdigit() else np.nan
            if np.isnan(t):
                usable = False
                diag[rid] = ("UNUSABLE", f"threshold {thr!r} is not numeric")
                break
            _cm = ({">": v > t, ">=": v >= t, "<": v < t, "<=": v <= t,
                    "==": v == t}.get(op, v >= t)).fillna(False)
            mask &= _cm
            clause_pass.append((f"{_pretty(met)} {op} {t:g}", int(_cm.sum())))
            spec.append((_pretty(met), met,
                         {">=": "≥", "<=": "≤"}.get(op, op), t, False))
        if not usable:
            continue
        sel = M[mask]
        if sel.empty:
            worst = min(clause_pass, key=lambda x: x[1]) if clause_pass else ("?", 0)
            diag[rid] = ("NO MATCH",
                         f"blocking clause: {worst[0]} — only {worst[1]:,} stocks pass it"
                         + (f" (all clauses: {clause_pass})" if len(clause_pass) > 1 else ""))
        else:
            diag[rid] = ("FIRED", f"{len(sel):,} stocks")

        # An N-quarter rule contributes one clause PER quarter (period_offset
        # 0..N-1), all reading the same streak column. Only the longest is
        # binding, so collapse them — otherwise the evidence line repeats
        # itself eight times with a rising "needs".
        _seen, _spec = {}, []
        for item in spec:
            label, col, o, t, is_streak = item
            if not is_streak:
                _spec.append(item)
                continue
            if col not in _seen:
                _seen[col] = len(_spec)
                _spec.append(item)
            elif t > _spec[_seen[col]][3]:
                _spec[_seen[col]] = item
        spec = _spec

        def _evidence(row) -> str:
            """This stock's own numbers against the rule's thresholds — the
            'why', not just the rule's title."""
            out = []
            for label, col, o, t, is_streak in spec:
                val = pd.to_numeric(pd.Series([row.get(col)]),
                                    errors="coerce").iloc[0]
                if pd.isna(val):
                    continue
                if is_streak:
                    out.append(f"{label} for {int(val)}q running "
                               f"(rule needs {int(t)})")
                else:
                    out.append(f"{label} {val:,.1f} {o} {t:g}")
            return " · ".join(out)

        for _, s in sel.iterrows():
            hits.append({"rule_id": rid, "rule_name": names.get(rid, "")[:70],
                         "evidence": _evidence(s),
                         "symbol": s["symbol"], "name": s["name"],
                         "market_cap_cr": s.get("market_cap_cr"),
                         "turnover_20d_cr": round(float(s.get("turnover_20d_cr", np.nan)), 2),
                         "price": round(s.get("close", np.nan), 2),
                         "ret_12m_pct": round(s.get("price_return_12m_pct", np.nan), 1),
                         "latest_quarter": s.get("latest_quarter", "")})
    if args.diagnose:
        print()
        print("=" * 78)
        print("RULE DIAGNOSIS — why each chosen rule did or did not produce hits")
        print("=" * 78)
        order = {"UNUSABLE": 0, "NO MATCH": 1, "FIRED": 2}
        for rid in sorted(chosen.rule_id.unique(),
                          key=lambda r: (order.get(diag.get(r, ("?",))[0], 9), r)):
            verdict, detail = diag.get(rid, ("NOT EVALUATED", "rule_id absent from Clauses"))
            print()
            print(f"[{verdict}] {rid}")
            print(f"    {str(names.get(rid, ''))[:96]}")
            print(f"    {detail}")
        n_ok = sum(1 for v in diag.values() if v[0] == "FIRED")
        print()
        print("=" * 78)
        print(f"{n_ok} fired | "
              f"{sum(1 for v in diag.values() if v[0] == 'NO MATCH')} no match | "
              f"{sum(1 for v in diag.values() if v[0] == 'UNUSABLE')} unusable | "
              f"of {chosen.rule_id.nunique()} chosen")
        print("=" * 78)
        return

    H = pd.DataFrame(hits)
    log(f"(rule, stock) hits: {len(H):,} across {H.rule_id.nunique() if len(H) else 0} rules")
    status.append(("rules that produced hits", H.rule_id.nunique() if len(H) else 0))
    status.append(("(rule,stock) pairs (pre-liquidity)", len(H)))
    if H.empty:
        log("no hits"); return

    # ---- LIQUIDITY FLOOR ----
    # market cap alone does not make a stock tradeable: the earlier run put a
    # stock trading Rs 0.82 cr/day at #2 on conviction. Traded value is the
    # binding constraint, so filter on it explicitly.
    before = len(H)
    H = H[pd.to_numeric(H["turnover_20d_cr"], errors="coerce") >= args.min_turnover]
    log(f"liquidity >= {args.min_turnover}cr/day: {before:,} -> {len(H):,} pairs")
    status.append((f"after liquidity >= {args.min_turnover:g}cr/day", len(H)))
    if H.empty:
        log("nothing passes the liquidity floor"); return

    # ---- RULE SELECTIVITY WEIGHTS ----
    # Counting rules equally over-rewards loose rules: "EPS YoY >=15% for 2
    # quarters" fires on ~45% of the universe, while "ROCE > 40%" fires on ~3%.
    # Passing the rare one is much stronger evidence, so weight each rule by
    # how selective it is AND by its validated out-of-sample return.
    n_uni = max(len(M), 1)
    sel = H.groupby("rule_id").symbol.nunique().rename("n_hits").reset_index()
    sel["hit_rate"] = sel.n_hits / n_uni
    sel["selectivity"] = 1 - sel.hit_rate                 # rarer = higher
    perf = (chosen.groupby("rule_id")["worst_of_both"].max()
            .rename("rule_return").reset_index())
    sel = sel.merge(perf, on="rule_id", how="left")
    sel["rule_return"] = sel.rule_return.fillna(sel.rule_return.median())
    sel["weight"] = (sel.selectivity * sel.rule_return).round(2)
    H = H.merge(sel[["rule_id", "n_hits", "selectivity", "rule_return", "weight"]],
                on="rule_id", how="left")

    # ---- LIST A: conviction (selectivity-weighted, not a raw count) ----
    conv = (H.groupby(["symbol", "name"])
            .agg(conviction_score=("weight", "sum"),
                 n_rules=("rule_id", "nunique"),
                 rarest_rule_hits=("n_hits", "min"),
                 best_rule_return=("rule_return", "max"),
                 market_cap_cr=("market_cap_cr", "first"),
                 turnover_20d_cr=("turnover_20d_cr", "first"),
                 price=("price", "first"),
                 ret_12m_pct=("ret_12m_pct", "first"),
                 rules=("rule_id", lambda s: ", ".join(sorted(set(s))[:20])),
                 rule_names=("rule_name", lambda s: " | ".join(sorted(set(s))[:6])))
            .reset_index())
    conv["conviction_score"] = conv.conviction_score.round(1)
    conv = conv.sort_values(["conviction_score", "n_rules"], ascending=False)
    status.append(("unique stocks surfaced", len(conv)))

    # ---- WHY: each rule the stock passed, with the stock's OWN numbers ----
    # Heaviest (rarest x best-returning) rules first, so a truncated list still
    # shows the strongest evidence. The gallery renders this under each chart.
    WHY_MAX = 8
    _H = H.sort_values("weight", ascending=False).copy()
    _ev = _H["evidence"].fillna("").astype(str)
    _H["_line"] = np.where(_ev.str.len() > 0,
                           _H["rule_name"] + " → " + _ev, _H["rule_name"])
    why = (_H.groupby("symbol")["_line"]
           .apply(lambda s: " ¦ ".join(s.head(WHY_MAX)))
           .rename("why").reset_index())
    conv = conv.merge(why, on="symbol", how="left")

    # ---- LIST B: coverage (>=N unique stocks per rule, avoiding repeats) ----
    used, cover = set(), []
    for rid, g in H.groupby("rule_id"):
        g = g.sort_values("turnover_20d_cr", ascending=False)
        fresh = g[~g.symbol.isin(used)].head(args.min_per_rule)
        if len(fresh) < args.min_per_rule:      # top up with repeats if needed
            extra = g[~g.symbol.isin(fresh.symbol)].head(args.min_per_rule - len(fresh))
            fresh = pd.concat([fresh, extra])
        used.update(fresh.symbol)
        cover.append(fresh)
    COV = pd.concat(cover, ignore_index=True) if cover else pd.DataFrame()
    status.append(("coverage-list unique stocks", COV.symbol.nunique() if len(COV) else 0))

    stat = pd.DataFrame(status, columns=["step", "count"])
    stat.loc[len(stat)] = ["price data as of", str(M["last_date"].max().date())]
    stat.loc[len(stat)] = ["latest quarter seen",
                           M["latest_quarter"].mode().iloc[0] if "latest_quarter" in M else "n/a"]
    out = os.path.join(GURU, f"DAILY_SCREEN_{datetime.now().date()}.xlsx")
    with pd.ExcelWriter(out, engine="openpyxl") as xw:
        stat.to_excel(xw, "Status", index=False)
        conv.to_excel(xw, "Conviction", index=False)
        if len(COV):
            COV.to_excel(xw, "Coverage", index=False)
        H.sort_values(["rule_id", "turnover_20d_cr"], ascending=[True, False]
                      ).to_excel(xw, "By_Rule", index=False)
        sel.sort_values("weight", ascending=False).to_excel(xw, "Rule_Selectivity",
                                                           index=False)
    log(f"DAILY SCREEN -> {out}")
    print(stat.to_string(index=False))

    # ---- publish the picks so the gallery can render them --------------------
    # The xlsx is the human artefact; this parquet is the machine one. Same
    # split the guidance watchlist uses: a builder writes a table to Drive
    # _index/, and build_gallery.py renders it. Written locally either way, so
    # the gallery still works offline from the last run.
    picks = conv.copy()
    as_of = str(M["last_date"].max().date())
    picks["as_of"] = as_of
    picks["min_mcap_cr"] = float(args.min_mcap)
    picks["min_turnover_cr"] = float(args.min_turnover)
    os.makedirs(BT, exist_ok=True)

    # ---- FIRST-SEEN LEDGER --------------------------------------------------
    # "When did this stock come onto the list?" cannot be derived from a single
    # run, so it has to be remembered. One row per symbol, carried forward:
    # first_seen never moves, last_seen tracks the newest run it appeared in.
    # Kept next to the picks and published with them, so a CI run and a local
    # run share one history instead of each keeping a private one.
    seen_p = os.path.join(BT, "guru_seen.parquet")
    seen = pd.DataFrame(columns=["symbol", "first_seen", "last_seen", "times_seen"])
    if os.path.exists(seen_p):
        try:
            seen = pd.read_parquet(seen_p)
        except Exception as e:
            log(f"first-seen ledger unreadable, starting fresh ({e})")
    elif not args.no_publish:
        try:                                   # no local copy — try Drive's
            from _extractor_base import (get_drive, get_or_create_subfolder,
                                         find_file, download_bytes)
            _d = get_drive()
            _repo = get_or_create_subfolder(_d, os.environ["GDRIVE_FOLDER_ID"],
                                            "company_repo")
            _iid = get_or_create_subfolder(_d, _repo, "_index")
            _f = find_file(_d, _iid, "guru_seen.parquet")
            if _f:
                seen = pd.read_parquet(io.BytesIO(download_bytes(_d, _f)))
                log(f"first-seen ledger pulled from Drive ({len(seen):,} symbols)")
        except Exception as e:
            log(f"first-seen ledger not on Drive ({type(e).__name__}) — starting fresh")

    prev = dict(zip(seen.get("symbol", []), seen.get("first_seen", [])))
    cnt = dict(zip(seen.get("symbol", []), seen.get("times_seen", [])))
    cur = picks["symbol"].astype(str).tolist()
    n_new = sum(1 for s in cur if s not in prev)
    rows = [{"symbol": s, "first_seen": prev.get(s, as_of), "last_seen": as_of,
             "times_seen": int(cnt.get(s, 0)) + 1} for s in cur]
    # symbols that dropped off today keep their history untouched
    gone = [r for r in seen.to_dict("records")
            if str(r.get("symbol")) not in set(cur)]
    seen = pd.DataFrame(rows + gone)
    seen.to_parquet(seen_p, index=False)
    log(f"first-seen ledger -> {len(seen):,} symbols "
        f"({n_new:,} first appeared today)")

    picks = picks.merge(seen[["symbol", "first_seen", "times_seen"]],
                        on="symbol", how="left")
    picks["days_on_list"] = (pd.to_datetime(as_of)
                             - pd.to_datetime(picks["first_seen"],
                                              errors="coerce")).dt.days
    picks_p = os.path.join(BT, "guru_picks.parquet")
    picks.to_parquet(picks_p, index=False)
    log(f"picks table -> {picks_p} ({len(picks):,} rows)")
    if not args.no_publish:
        try:
            from _extractor_base import (get_drive, get_or_create_subfolder,
                                         upload_bytes)
            d = get_drive()
            repo = get_or_create_subfolder(d, os.environ["GDRIVE_FOLDER_ID"],
                                           "company_repo")
            iid = get_or_create_subfolder(d, repo, "_index")
            for p in (picks_p, seen_p):
                with open(p, "rb") as fh:
                    upload_bytes(d, iid, os.path.basename(p), fh.read(),
                                 "application/octet-stream")
            log("published guru_picks.parquet + guru_seen.parquet "
                "-> Drive company_repo/_index/")
        except Exception as e:
            # Never fail the screen over a publish hiccup — the xlsx and the
            # mail are the deliverables; the gallery can use the local copy.
            log(f"publish SKIPPED ({type(e).__name__}: {e})")


if __name__ == "__main__":
    main()
