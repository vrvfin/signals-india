r"""
FAMILY LIFT TABLE — build_family_lift.py  (Project Guru)

Measures, EMPIRICALLY, how much each rule family's PRESENCE changes the odds of
a good outcome — the input that makes stock ranking actually discriminate.

Why this exists: counting rules (or averaging their historical returns) does NOT
separate stocks — measured flat at ~42% expected return / 67% win-rate whatever
the count, because those are properties of the RULES, not the stock. What DOES
separate is WHICH families are present: e.g. FUND_EPS present -> 73.5% win-rate
vs 56.2% without (+17.3pp), while VAL_PEPEAK present -> 38.5% vs 58.5% (-20pp).

Method: from the cached consensus clusters (real stock-quarter events with real
forward returns), for each family compare outcomes of clusters WITH vs WITHOUT
that family. Liquid clusters only (avg vol >= 10k/day) so illiquid stale names
don't distort it. Computed per horizon.

Output: guru/backtest/family_lift.parquet + guru/FAMILY_LIFT.xlsx

Usage: python guru/build_family_lift.py
"""
from __future__ import annotations
import os
from datetime import datetime
import numpy as np, pandas as pd

GURU = os.path.dirname(os.path.abspath(__file__))
BT = os.path.join(GURU, "backtest")
CLUSTERS = os.path.join(BT, "consensus_clusters.parquet")
OUT_PQ = os.path.join(BT, "family_lift.parquet")
OUT_X = os.path.join(GURU, "FAMILY_LIFT.xlsx")
LIQ_MIN = 10000
MIN_N = 150
HZ = [3, 6, 12, 24, 36]


def log(m): print(f"[{datetime.now().strftime('%H:%M:%S')}] {m}", flush=True)


def main():
    c = pd.read_parquet(CLUSTERS)
    liq = c[c.vol_avg_3m >= LIQ_MIN].copy()
    log(f"clusters {len(c):,} -> liquid {len(liq):,}")
    fams = sorted({f for s in liq.families.dropna() for f in s.split(",")})
    log(f"families: {len(fams)}")
    rows = []
    for H in HZ:
        col = f"ret_{H}m"
        d = liq[liq[col].notna()]
        if len(d) < MIN_N:
            continue
        base_win = float((d[col] > 0).mean()) * 100
        for fam in fams:
            m = d.families.str.contains(fam, regex=False, na=False)
            a, b = d[m][col], d[~m][col]
            if len(a) < MIN_N or len(b) < MIN_N:
                continue
            wa, wb = float((a > 0).mean()) * 100, float((b > 0).mean()) * 100
            rows.append({"horizon": f"{H}M", "family": fam,
                         "n_with": len(a), "n_without": len(b),
                         "winrate_with": round(wa, 1), "winrate_without": round(wb, 1),
                         "lift_pp": round(wa - wb, 1),
                         "median_with": round(float(a.median()), 1),
                         "median_without": round(float(b.median()), 1),
                         "median_lift": round(float(a.median() - b.median()), 1),
                         "baseline_winrate": round(base_win, 1)})
    lift = pd.DataFrame(rows)
    lift.to_parquet(OUT_PQ, index=False)

    readme = pd.DataFrame([
        ("WHAT", "How much each rule FAMILY's presence changes the odds, measured on "
         "real historical stock-quarter events (not rule averages)."),
        ("WHY IT MATTERS", "Counting rules does not discriminate: stocks passing 10 "
         "vs 20 families both showed ~42% expected return / 67% win-rate. WHICH "
         "families fire is what separates outcomes."),
        ("lift_pp", "Win-rate when this family is present MINUS win-rate when it is "
         "absent, in percentage points. Positive = presence improves the odds. "
         "NEGATIVE = presence is a WARNING (e.g. valuation-at-peak, oversold)."),
        ("median_lift", "Same comparison for median return."),
        ("How it is used", "build_stock_scorecard.py --weighted turns these lifts "
         "into a per-stock score: sum of lift_pp for the families currently firing. "
         "That score DOES separate stocks, unlike a raw rule count."),
        ("Liquidity", f"Only clusters with avg volume >= {LIQ_MIN:,}/day are used, so "
         "illiquid stale-price names do not distort the measurement."),
        ("CAVEAT", "Lift is measured on 2019-26-heavy data in a rising market, and "
         "reflects correlation, not proven causation. A negative-lift family is not "
         "necessarily 'bad' — TECH_REVERSAL has a high median return but a LOW "
         "win-rate (a high-variance lottery profile), so it scores negatively on "
         "odds while still producing big individual winners."),
    ], columns=["item", "explanation"])
    with pd.ExcelWriter(OUT_X, engine="openpyxl") as xw:
        readme.to_excel(xw, "README", index=False)
        for H in HZ:
            d = lift[lift.horizon == f"{H}M"].sort_values("lift_pp", ascending=False)
            if not d.empty:
                d.to_excel(xw, f"Lift_{H}M", index=False)
        ws = xw.book["README"]
        from openpyxl.styles import Alignment
        ws.column_dimensions["A"].width = 26
        ws.column_dimensions["B"].width = 105
        for row in ws.iter_rows(min_row=2):
            row[1].alignment = Alignment(wrap_text=True, vertical="top")
    log(f"family lift -> {OUT_PQ} and {OUT_X}")
    d24 = lift[lift.horizon == "24M"].sort_values("lift_pp", ascending=False)
    log(f"  24M: {len(d24)} families scored | best {d24.iloc[0].family} "
        f"({d24.iloc[0].lift_pp:+.1f}pp) | worst {d24.iloc[-1].family} "
        f"({d24.iloc[-1].lift_pp:+.1f}pp)")


if __name__ == "__main__":
    main()
