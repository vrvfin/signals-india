r"""
SYSTEM v2 — system_v2.py  (Project Guru)

Rebuild of the buy/hold/sell loop after v1 failed. What changed and WHY:

  v1 EXIT was a flat -35% drawdown stop. Measured: the typical max-drawdown of
  these validated rules is -42%, and 80% of them normally draw down worse than
  -35%. So the stop sat INSIDE normal volatility -> it fired on noise, exited 77
  positions at a -38% median (3.9% win rate) and dragged the system to -33.9pp
  vs Nifty500.

  v2 EXIT = 40-WEEK EMA BREAK (weekly close below the 40wk EMA). A trend exit
  adapts to each stock's own volatility instead of imposing one % on all of
  them, and it cannot fire until the trend has actually turned.

  v2 ENTRY also uses the 40wk EMA as a regime gate: only buy when price is
  ABOVE its own 40wk EMA. Enter in an uptrend, leave when the trend breaks —
  one coherent idea rather than two unrelated ones.

NO LOOK-AHEAD (unchanged from v1): family lift and look-alike expected returns
are computed from clusters entered on/before 2018-12-31; the loop runs on 2019+.

ENTRY (all must hold):
  * >= MIN_SUPPORT supporting families firing (train-derived positive lift)
  * supporting families > warning families
  * train look-alike expected return >= MIN_EXP_RET (>= MIN_CASES cases)
  * liquidity >= LIQ_MIN shares/day
  * price ABOVE its 40-week EMA          [--no-trend-gate to disable]
EXIT (first to fire):
  * weekly close BELOW the 40-week EMA   (the trend has turned)
  * TARGET +TARGET_PCT                   [optional, --target 0 to disable]
  * TIMEOUT at MAX_MONTHS

Output: guru/SYSTEM_V2.xlsx

Usage:
    python guru/system_v2.py                      # trend gate + 40wk exit
    python guru/system_v2.py --no-trend-gate      # ablation: entry gate off
    python guru/system_v2.py --target 0           # pure trend exit, no target
"""
from __future__ import annotations
import argparse, os
from datetime import datetime
import numpy as np, pandas as pd

GURU = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(GURU, "data")
BT = os.path.join(GURU, "backtest")
OHLCV = os.path.join(DATA, "ohlcv_hist")
CLUSTERS = os.path.join(BT, "consensus_clusters.parquet")
OUT = os.path.join(GURU, "SYSTEM_V2.xlsx")

SPLIT = pd.Timestamp("2018-12-31")
LIQ_MIN = 10000
MIN_SUPPORT = 3
MIN_EXP_RET = 20.0
MIN_CASES = 50
MAX_MONTHS = 24
MIN_LIFT_N = 100
EMA_WEEKS = 40


def log(m): print(f"[{datetime.now().strftime('%H:%M:%S')}] {m}", flush=True)


def weekly_ema(px: pd.DataFrame, weeks: int = EMA_WEEKS) -> pd.DataFrame:
    """weekly close + its N-week EMA, indexed by week-ending date."""
    w = (px.set_index("date")["close"].resample("W-FRI").last().dropna()
         .to_frame("wclose"))
    w["ema"] = w["wclose"].ewm(span=weeks, adjust=False).mean()
    w["above"] = w["wclose"] > w["ema"]
    return w.reset_index()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=float, default=100.0,
                    help="take-profit %%; 0 disables it")
    ap.add_argument("--max-months", type=int, default=MAX_MONTHS)
    ap.add_argument("--min-exp-ret", type=float, default=MIN_EXP_RET)
    ap.add_argument("--no-trend-gate", action="store_true",
                    help="ablation: do NOT require price above 40wk EMA at entry")
    args = ap.parse_args()

    c = pd.read_parquet(CLUSTERS)
    c = c[c.vol_avg_3m >= LIQ_MIN].copy()
    c["entry_date"] = pd.to_datetime(c["entry_date"])
    c["famset"] = c.families.fillna("").apply(lambda s: set(x for x in s.split(",") if x))
    train = c[c.entry_date <= SPLIT]
    test = c[c.entry_date > SPLIT]
    rcol = f"ret_{args.max_months}m"
    tr = train[train[rcol].notna()]
    log(f"clusters: train {len(train):,} | test {len(test):,}")

    # ---------- TRAIN-ONLY family lift ----------
    lift = {}
    for f in sorted({x for s in tr.famset for x in s}):
        m = tr.famset.apply(lambda s: f in s)
        a, b = tr[m][rcol], tr[~m][rcol]
        if len(a) >= MIN_LIFT_N and len(b) >= MIN_LIFT_N:
            lift[f] = float((a > 0).mean() - (b > 0).mean()) * 100
    pos = {f for f, v in lift.items() if v > 0}
    neg = {f for f, v in lift.items() if v < 0}
    log(f"train-derived: {len(pos)} supporting / {len(neg)} warning families")

    tr_sets, tr_ret = tr.famset.tolist(), tr[rcol].values
    cache = {}

    def lookalike(fp: frozenset):
        if fp in cache:
            return cache[fp]
        need = min(3, len(fp))
        hits = [tr_ret[i] for i, s in enumerate(tr_sets) if len(s & fp) >= need]
        r = (float(np.median(hits)), len(hits)) if len(hits) >= MIN_CASES else (None, 0)
        cache[fp] = r
        return r

    # ---------- candidate entries ----------
    cands = []
    for _, r in test.iterrows():
        sup, wrn = r.famset & pos, r.famset & neg
        if len(sup) < MIN_SUPPORT or len(wrn) >= len(sup):
            continue
        exp, n = lookalike(frozenset(sup))
        if exp is None or exp < args.min_exp_ret:
            continue
        cands.append({"guru_key": r.guru_key, "entry_date": r.entry_date,
                      "exp_ret": exp, "n_cases": n, "n_support": len(sup),
                      "support": ",".join(sorted(sup))})
    cd = pd.DataFrame(cands)
    log(f"pre-trend candidates: {len(cd):,} on {cd.guru_key.nunique():,} stocks")

    # ---------- simulate ----------
    rows = []
    for gk, grp in cd.groupby("guru_key"):
        fp = os.path.join(OHLCV, f"{gk}.parquet")
        if not os.path.exists(fp):
            continue
        px = pd.read_parquet(fp, columns=["date", "close"]).sort_values("date")
        px["date"] = pd.to_datetime(px["date"])
        if len(px) < 250:
            continue
        wk = weekly_ema(px)
        wdates = wk["date"].values
        dts, cls = px["date"].values, px["close"].values
        for _, r in grp.iterrows():
            i = int(np.searchsorted(dts, np.datetime64(r.entry_date), side="right"))
            if i >= len(cls):
                continue
            entry = float(cls[i])
            if entry <= 0:
                continue
            wi = int(np.searchsorted(wdates, np.datetime64(r.entry_date), side="right")) - 1
            if wi < 0 or wi >= len(wk):
                continue
            # ENTRY GATE: price above its own 40wk EMA
            if not args.no_trend_gate and not bool(wk["above"].iloc[wi]):
                continue
            # walk weekly; exit on first EMA break (or target / timeout)
            #
            # ARMING RULE (critical): our fundamental signals mostly pick BEATEN-DOWN
            # stocks, which are already BELOW their 40wk EMA at entry. A naive
            # "sell when below the EMA" therefore fired in week one (measured: avg
            # exit at 0.7 months, 11 of 19 positions) — we bought and instantly sold.
            # So the trend exit only ARMS once price has first closed ABOVE the EMA.
            # Until then the thesis is still playing out and we hold.
            end_i = min(i + args.max_months * 21, len(cls) - 1)
            exit_kind, exit_ret, exit_m = "TIMEOUT", None, args.max_months
            armed = bool(wk["above"].iloc[wi])       # already in uptrend at entry
            for k in range(wi + 1, len(wk)):
                wd = wk["date"].iloc[k]
                di = int(np.searchsorted(dts, np.datetime64(wd), side="right")) - 1
                if di <= i:
                    continue
                if di > end_i:
                    break
                ret = (float(cls[di]) / entry - 1) * 100
                months = (di - i) / 21
                above = bool(wk["above"].iloc[k])
                if args.target and ret >= args.target:
                    exit_kind, exit_ret, exit_m = "TARGET", ret, round(months, 1); break
                if not armed:
                    if above:
                        armed = True                  # trend established -> exit live
                    continue
                if not above:
                    exit_kind, exit_ret, exit_m = "EMA40_BREAK", ret, round(months, 1); break
            if exit_ret is None:
                exit_ret = (float(cls[end_i]) / entry - 1) * 100
                if end_i < i + args.max_months * 21:
                    exit_kind = "DATA_END"
            jb = i + args.max_months * 21
            hold = ((float(cls[jb]) / entry - 1) * 100) if jb < len(cls) else np.nan
            rows.append({"guru_key": gk, "entry_date": r.entry_date,
                         "exp_ret_predicted": round(r.exp_ret, 1),
                         "n_support": r.n_support, "support": r.support,
                         "exit_kind": exit_kind, "exit_month": exit_m,
                         "system_ret": round(exit_ret, 1),
                         "buyhold_ret": None if pd.isna(hold) else round(hold, 1)})
    R = pd.DataFrame(rows)
    log(f"positions simulated: {len(R):,} (from {len(cd):,} candidates — "
        f"gap = missing price file / <250 bars / entry beyond data / trend gate)")
    if R.empty:
        return

    # ---------- benchmark ----------
    n5 = pd.read_parquet(os.path.join(DATA, "macro_hist", "NIFTY_500.parquet"),
                         columns=["date", "close"]).sort_values("date")
    nd, nc = n5["date"].values, n5["close"].values
    b = []
    for _, r in R.iterrows():
        i = int(np.searchsorted(nd, np.datetime64(r.entry_date), side="right"))
        j = i + args.max_months * 21
        b.append((float(nc[j]) / float(nc[i]) - 1) * 100 if j < len(nc) and i < len(nc) else np.nan)
    R["nifty500_ret"] = np.round(b, 1)

    v = R[R.system_ret.notna()]
    both = v[v.buyhold_ret.notna()]
    bx = v[v.nifty500_ret.notna()]
    summ = [
        ("positions simulated", len(v)),
        ("SYSTEM median return %", round(float(v.system_ret.median()), 1)),
        ("SYSTEM win rate %", round(float((v.system_ret > 0).mean() * 100), 1)),
        ("--- vs same picks held blindly (n=%d) ---" % len(both), ""),
        ("  SYSTEM median %", round(float(both.system_ret.median()), 1) if len(both) else None),
        ("  BUY&HOLD median %", round(float(both.buyhold_ret.median()), 1) if len(both) else None),
        ("  EXIT LOGIC adds (pp)",
         round(float(both.system_ret.median() - both.buyhold_ret.median()), 1) if len(both) else None),
        ("--- vs NIFTY500 same windows (n=%d) ---" % len(bx), ""),
        ("  SYSTEM median %", round(float(bx.system_ret.median()), 1) if len(bx) else None),
        ("  NIFTY500 median %", round(float(bx.nifty500_ret.median()), 1) if len(bx) else None),
        ("  STOCK PICKING adds (pp)",
         round(float(bx.system_ret.median() - bx.nifty500_ret.median()), 1) if len(bx) else None),
    ]
    summary = pd.DataFrame(summ, columns=["metric", "value"])
    attrib = (v.groupby("exit_kind").agg(n=("system_ret", "size"),
              median_ret=("system_ret", "median"),
              winrate=("system_ret", lambda s: round((s > 0).mean() * 100, 1)),
              avg_month=("exit_month", "mean")).round(1).reset_index()
              .sort_values("n", ascending=False))

    readme = pd.DataFrame([
        ("WHAT CHANGED FROM v1", "v1 used a flat -35% drawdown stop and lost to the "
         "index by 33.9pp. Measured cause: these rules' TYPICAL drawdown is -42% "
         "and 80% normally exceed -35%, so the stop sat inside normal volatility "
         "and sold 77 positions at a -38% median. v2 replaces it with a 40-WEEK EMA "
         "BREAK, which adapts to each stock's own volatility."),
        ("ENTRY", f">= {MIN_SUPPORT} supporting families (train-derived), supports > "
         f"warnings, train look-alike expected return >= {args.min_exp_ret}%, "
         f"liquidity >= {LIQ_MIN:,}/day, and price ABOVE its 40-week EMA "
         "(trend gate — disable with --no-trend-gate to see its contribution)."),
        ("EXIT", f"First of: weekly close BELOW the 40-week EMA; TARGET "
         f"+{args.target}%; TIMEOUT at {args.max_months} months."),
        ("NO LOOK-AHEAD", "Family lifts and look-alike returns come only from "
         "clusters entered on/before 2018-12-31; the loop runs on 2019+."),
        ("Summary", "Two separate questions: does the EXIT logic add value (vs the "
         "same picks held blindly), and does the STOCK PICKING add value (vs "
         "Nifty500 over identical windows). Both must be positive for the system "
         "to be worth running."),
        ("CAVEATS", "No costs/slippage/taxes. Equal-weighted, no portfolio capacity "
         "limit. Positions entered recently cannot complete 24 months and show as "
         "DATA_END — check n on each benchmark before trusting the comparison."),
    ], columns=["item", "explanation"])

    with pd.ExcelWriter(OUT, engine="openpyxl") as xw:
        readme.to_excel(xw, "README", index=False)
        summary.to_excel(xw, "Summary", index=False)
        attrib.to_excel(xw, "Exit_Attribution", index=False)
        v.head(5000).to_excel(xw, "Positions", index=False)
        ws = xw.book["README"]
        from openpyxl.styles import Alignment
        ws.column_dimensions["A"].width = 26
        ws.column_dimensions["B"].width = 105
        for r_ in ws.iter_rows(min_row=2):
            r_[1].alignment = Alignment(wrap_text=True, vertical="top")
    log(f"SYSTEM V2 -> {OUT}")
    print(summary.to_string(index=False))
    print()
    print(attrib.to_string(index=False))


if __name__ == "__main__":
    main()
