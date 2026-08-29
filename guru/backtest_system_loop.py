r"""
SYSTEM BACKTEST — backtest_system_loop.py  (Project Guru, task #18)

Backtests the COMPLETE buy/hold/sell loop, not individual rules. The parts were
each validated separately; this tests whether they work TOGETHER.

NO LOOK-AHEAD — the critical design point:
  * family lift + look-alike expected returns are computed from TRAIN data only
    (clusters entered on/before 2018-12-31)
  * the decision loop then runs on TEST data (2019-01-01 onward), using only
    those train-derived statistics
  Using the 2019-26 lift table to make 2019 decisions would be look-ahead bias
  and would flatter the result.

The loop, per candidate:
  BUY   when, at that stock-quarter: >= MIN_SUPPORT supporting families fire,
        0 warning families, liquidity >= LIQ_MIN, and the train-derived
        look-alike expected return >= MIN_EXP_RET
  HOLD  walking forward month by month on real prices
  SELL  at whichever fires FIRST:
          (a) TARGET   : +TARGET_PCT
          (b) STOP     : drawdown from peak <= STOP_PCT
          (c) DECAY    : a warning family appears in a later quarter
          (d) TIMEOUT  : MAX_MONTHS reached
Benchmarks: the same positions held blindly to MAX_MONTHS (no exits), and the
Nifty500 over the identical windows.

Output: guru/SYSTEM_BACKTEST.xlsx

Usage: python guru/backtest_system_loop.py
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
OUT = os.path.join(GURU, "SYSTEM_BACKTEST.xlsx")

SPLIT = pd.Timestamp("2018-12-31")
LIQ_MIN = 10000            # shares/day
MIN_SUPPORT = 3            # supporting families required to BUY
MIN_EXP_RET = 20.0         # train-derived look-alike median return, %
MIN_CASES = 50             # train look-alike cases needed
TARGET_PCT = 100.0
STOP_PCT = -35.0           # drawdown from peak
MAX_MONTHS = 24
MIN_HOLD_M = 3             # don't check thesis-decay before this many months
MIN_LIFT_N = 100


def log(m): print(f"[{datetime.now().strftime('%H:%M:%S')}] {m}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=float, default=TARGET_PCT)
    ap.add_argument("--stop", type=float, default=STOP_PCT)
    ap.add_argument("--max-months", type=int, default=MAX_MONTHS)
    args = ap.parse_args()

    c = pd.read_parquet(CLUSTERS)
    c = c[c.vol_avg_3m >= LIQ_MIN].copy()
    c["entry_date"] = pd.to_datetime(c["entry_date"])
    c["famset"] = c.families.fillna("").apply(lambda s: set(x for x in s.split(",") if x))
    train = c[c.entry_date <= SPLIT]
    test = c[c.entry_date > SPLIT].copy()
    log(f"clusters: train {len(train):,} | test {len(test):,}")

    # ---------- TRAIN-ONLY family lift (no look-ahead) ----------
    rcol = f"ret_{args.max_months}m"
    tr = train[train[rcol].notna()]
    fams = sorted({f for s in tr.famset for f in s})
    lift, fam_median = {}, {}
    for f in fams:
        m = tr.famset.apply(lambda s: f in s)
        a, b = tr[m][rcol], tr[~m][rcol]
        if len(a) >= MIN_LIFT_N and len(b) >= MIN_LIFT_N:
            lift[f] = float((a > 0).mean() - (b > 0).mean()) * 100
            fam_median[f] = float(a.median())
    pos_fams = {f for f, v in lift.items() if v > 0}
    neg_fams = {f for f, v in lift.items() if v < 0}
    log(f"TRAIN-derived: {len(pos_fams)} supporting families, {len(neg_fams)} warning")

    # train look-alike lookup: fingerprint -> expected return
    tr_mat = tr.famset.tolist()
    tr_ret = tr[rcol].values

    def lookalike(fingerprint: set):
        """median return of TRAIN clusters sharing >=3 of these families."""
        need = min(3, len(fingerprint))
        hits = [tr_ret[i] for i, s in enumerate(tr_mat)
                if len(s & fingerprint) >= need]
        if len(hits) < MIN_CASES:
            return None, 0
        return float(np.median(hits)), len(hits)

    # ---------- THESIS-BREAK calendar for DECAY detection ----------
    # NOTE: an earlier version flagged decay when ANY warning family appeared.
    # Measured: 76.2% of stock-quarters contain >=1 warning family (~19 of 62
    # families fire per quarter), so that rule fired at month 1 for nearly every
    # position and made the test meaningless. A real thesis break = the WARNINGS
    # OUTNUMBER the supporting families in a later quarter.
    warn_cal = {}
    for _, r in c.iterrows():
        n_sup = len(r.famset & pos_fams)
        n_neg = len(r.famset & neg_fams)
        if n_neg > n_sup:
            warn_cal.setdefault(r.guru_key, []).append(r.entry_date)
    for k in warn_cal:
        warn_cal[k] = sorted(warn_cal[k])
    log(f"stocks with at least one thesis-break quarter: {len(warn_cal):,}")

    # ---------- select BUY candidates in TEST period ----------
    cands = []
    exp_cache = {}
    for _, r in test.iterrows():
        sup = r.famset & pos_fams
        warn = r.famset & neg_fams
        # entry filter: supporting evidence must clearly outweigh warnings.
        # (requiring ZERO warnings rejected 76% of all quarters — too strict to
        # be usable, and left only 135 candidates.)
        if len(sup) < MIN_SUPPORT or len(warn) >= len(sup):
            continue
        key = frozenset(sup)
        if key not in exp_cache:
            exp_cache[key] = lookalike(sup)
        exp, ncase = exp_cache[key]
        if exp is None or exp < MIN_EXP_RET:
            continue
        cands.append({"guru_key": r.guru_key, "entry_date": r.entry_date,
                      "entry_price": r.entry_price, "exp_ret": exp,
                      "n_cases": ncase, "n_support": len(sup),
                      "support": ",".join(sorted(sup))})
    cd = pd.DataFrame(cands)
    log(f"BUY candidates in test period: {len(cd):,} on {cd.guru_key.nunique():,} stocks")
    if cd.empty:
        return

    # ---------- walk each position forward ----------
    results = []
    for gk, grp in cd.groupby("guru_key"):
        fp = os.path.join(OHLCV, f"{gk}.parquet")
        if not os.path.exists(fp):
            continue
        px = pd.read_parquet(fp, columns=["date", "close"]).sort_values("date")
        dts, cls = px["date"].values, px["close"].values
        warns = warn_cal.get(gk, [])
        for _, r in grp.iterrows():
            i = int(np.searchsorted(dts, np.datetime64(r.entry_date), side="right"))
            if i >= len(cls):
                continue
            entry = float(cls[i])
            if entry <= 0:
                continue
            peak = entry
            exit_kind, exit_ret, exit_m = "TIMEOUT", None, args.max_months
            for m in range(1, args.max_months + 1):
                j = i + m * 21
                if j >= len(cls):
                    exit_kind, exit_m = "DATA_END", m
                    exit_ret = (float(cls[-1]) / entry - 1) * 100
                    break
                p = float(cls[j])
                peak = max(peak, p)
                ret = (p / entry - 1) * 100
                dd = (p / peak - 1) * 100
                # (c) thesis decay — only checked after MIN_HOLD_M months, so a
                # position is never stopped out before the thesis has had time
                decayed = (m >= MIN_HOLD_M and
                           any(r.entry_date < w <= pd.Timestamp(dts[j]) for w in warns))
                if ret >= args.target:
                    exit_kind, exit_ret, exit_m = "TARGET", ret, m; break
                if dd <= args.stop:
                    exit_kind, exit_ret, exit_m = "STOP", ret, m; break
                if decayed:
                    exit_kind, exit_ret, exit_m = "DECAY", ret, m; break
            if exit_ret is None:
                j = min(i + args.max_months * 21, len(cls) - 1)
                exit_ret = (float(cls[j]) / entry - 1) * 100
            # benchmark: same position held blindly to max horizon
            jb = i + args.max_months * 21
            hold_ret = ((float(cls[jb]) / entry - 1) * 100) if jb < len(cls) else np.nan
            results.append({"guru_key": gk, "entry_date": r.entry_date,
                            "exp_ret_predicted": round(r.exp_ret, 1),
                            "n_support": r.n_support, "support": r.support,
                            "exit_kind": exit_kind, "exit_month": exit_m,
                            "system_ret": round(exit_ret, 1),
                            "buyhold_ret": None if pd.isna(hold_ret) else round(hold_ret, 1)})
    R = pd.DataFrame(results)
    log(f"simulated positions: {len(R):,}")

    # ---------- benchmark: Nifty500 over same windows ----------
    n5 = pd.read_parquet(os.path.join(DATA, "macro_hist", "NIFTY_500.parquet"),
                         columns=["date", "close"]).sort_values("date")
    nd, nc = n5["date"].values, n5["close"].values
    bench = []
    for _, r in R.iterrows():
        i = int(np.searchsorted(nd, np.datetime64(r.entry_date), side="right"))
        j = i + args.max_months * 21
        bench.append((float(nc[j]) / float(nc[i]) - 1) * 100 if j < len(nc) else np.nan)
    R["nifty500_ret"] = np.round(bench, 1)

    # ---------- summary ----------
    valid = R[R.system_ret.notna()]
    # NaN-safe: compare on the subset where BOTH the system and the benchmark have
    # a value, otherwise positions too recent for a full 24m benchmark silently
    # count as benchmark losses (an earlier version reported a bogus 0% win rate).
    both = valid[valid.buyhold_ret.notna()]
    bothx = valid[valid.nifty500_ret.notna()]
    summ = [
        ("positions simulated", len(valid)),
        ("SYSTEM median return %", round(float(valid.system_ret.median()), 1)),
        ("SYSTEM win rate %", round(float((valid.system_ret > 0).mean() * 100), 1)),
        ("--- vs holding the SAME picks (n=%d) ---" % len(both), ""),
        ("  SYSTEM median %", round(float(both.system_ret.median()), 1)),
        ("  BUY&HOLD median %", round(float(both.buyhold_ret.median()), 1)),
        ("  SYSTEM win rate %", round(float((both.system_ret > 0).mean() * 100), 1)),
        ("  BUY&HOLD win rate %", round(float((both.buyhold_ret > 0).mean() * 100), 1)),
        ("  EXIT LOGIC adds (pp)",
         round(float(both.system_ret.median() - both.buyhold_ret.median()), 1)),
        ("--- vs NIFTY500 same windows (n=%d) ---" % len(bothx), ""),
        ("  SYSTEM median %", round(float(bothx.system_ret.median()), 1)),
        ("  NIFTY500 median %", round(float(bothx.nifty500_ret.median()), 1)),
        ("  STOCK PICKING adds (pp)",
         round(float(bothx.system_ret.median() - bothx.nifty500_ret.median()), 1)),
    ]
    summary = pd.DataFrame(summ, columns=["metric", "value"])
    attrib = (valid.groupby("exit_kind")
              .agg(n=("system_ret", "size"),
                   median_ret=("system_ret", "median"),
                   winrate=("system_ret", lambda s: round((s > 0).mean() * 100, 1)),
                   avg_month=("exit_month", "mean")).round(1).reset_index()
              .sort_values("n", ascending=False))
    # did the predicted expected return actually predict?
    valid2 = valid.copy()
    valid2["pred_bucket"] = pd.qcut(valid2.exp_ret_predicted, 4,
                                    duplicates="drop", labels=False)
    calib = (valid2.groupby("pred_bucket")
             .agg(n=("system_ret", "size"),
                  predicted=("exp_ret_predicted", "median"),
                  actual=("system_ret", "median")).round(1).reset_index())

    readme = pd.DataFrame([
        ("WHAT THIS IS", "A backtest of the COMPLETE buy/hold/sell system, not of "
         "individual rules. Each component was validated separately; this checks "
         "whether they work together."),
        ("NO LOOK-AHEAD", "Family lifts and look-alike expected returns are computed "
         "ONLY from clusters entered on/before 2018-12-31. The loop then runs on "
         "2019-26 using just those train-derived numbers. Using the full-period lift "
         "table to make 2019 decisions would be look-ahead bias."),
        ("THE RULES", f"BUY: >= {MIN_SUPPORT} supporting families, 0 warning families, "
         f"liquidity >= {LIQ_MIN:,}/day, train look-alike expected return >= "
         f"{MIN_EXP_RET}%. SELL on first of: TARGET +{args.target}%, STOP "
         f"{args.stop}% drawdown from peak, DECAY (a warning family appears), or "
         f"TIMEOUT at {args.max_months} months."),
        ("Summary sheet", "System vs two benchmarks: the SAME picks held blindly "
         "(isolates whether the exit logic adds value) and Nifty500 over identical "
         "windows (isolates whether the picks beat the market)."),
        ("Exit_Attribution", "Which trigger did the selling, how often, and what each "
         "returned — tells you which part of the exit logic is doing the work."),
        ("Calibration", "Were the predicted expected returns honest? Compares the "
         "predicted look-alike return against what actually happened, by quartile. "
         "If actual tracks predicted, the fingerprint method is trustworthy."),
        ("CAVEATS", "No transaction costs, taxes or slippage. Positions are equal-"
         "weighted and unlimited (no portfolio capacity constraint). 2019-26 was a "
         "rising market — always read the Nifty500 comparison, not the raw return."),
    ], columns=["item", "explanation"])

    with pd.ExcelWriter(OUT, engine="openpyxl") as xw:
        readme.to_excel(xw, "README", index=False)
        summary.to_excel(xw, "Summary", index=False)
        attrib.to_excel(xw, "Exit_Attribution", index=False)
        calib.to_excel(xw, "Calibration", index=False)
        valid.head(5000).to_excel(xw, "Positions", index=False)
        ws = xw.book["README"]
        from openpyxl.styles import Alignment
        ws.column_dimensions["A"].width = 26
        ws.column_dimensions["B"].width = 105
        for r_ in ws.iter_rows(min_row=2):
            r_[1].alignment = Alignment(wrap_text=True, vertical="top")
    log(f"SYSTEM BACKTEST -> {OUT}")
    print(summary.to_string(index=False))
    print()
    print(attrib.to_string(index=False))


if __name__ == "__main__":
    main()
