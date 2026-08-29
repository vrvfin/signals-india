r"""
#11 — build_decisions.py  (Project Guru)  BUY / HOLD / SELL screener

Scans the whole universe as of the latest data and emits, per stock x horizon-
class, a BUY when the stock CURRENTLY satisfies a validated survivor rule (one
that HELD out-of-sample), ranked by that rule's out-of-sample median return and
win-rate. Each BUY carries a recommended EXIT (the rule's cohort drawdown stop
from #7) so it doubles as the hold/sell plan.

Definitions
  live signal : the rule's most recent trigger for a stock is within the recency
                window (technical rules 30d, fundamental/combined 120d = ~1 qtr).
  horizon-class : SHORT (<=6M) / MID (6-24M) / LONG (>=24M), from the rule's best
                  validated horizon. A stock can be BUY-MID but not BUY-SHORT ->
                  the output is a stock x horizon-class MATRIX (mixed verdicts).
  SELL/HOLD : true SELL needs a position; in screener mode each BUY ships with its
              exit rule (cohort-DD stop level + take-profit), so once held the
              SELL is mechanical. An 'extended' caution flags don't-chase entries.

Additive: reads existing triggers/validation/exits; writes only backtest/decisions/.

Usage: python guru/build_decisions.py
       python guru/build_decisions.py --min-valid-median 20   # stricter BUY bar
"""
from __future__ import annotations
import argparse, glob, os
from datetime import datetime
import numpy as np, pandas as pd

GURU_DIR = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(GURU_DIR, "data")
BT = os.path.join(GURU_DIR, "backtest")
DEC_DIR = os.path.join(BT, "decisions")
HZCLASS = {1: "SHORT", 2: "SHORT", 3: "SHORT", 6: "SHORT",
           12: "MID", 18: "MID", 24: "MID", 36: "LONG", 48: "LONG",
           60: "LONG", 84: "LONG", 120: "LONG"}


def log(m): print(f"[{datetime.now().strftime('%H:%M:%S')}] {m}", flush=True)


def rule_recency_days(rid: str) -> int:
    fam = rid.split("_")[0]
    return 30 if fam == "TECH" and not rid.startswith("COMBO") else 120


def load_validation() -> pd.DataFrame:
    fs = [f for f in glob.glob(os.path.join(BT, "validation", "*.parquet"))
          if not os.path.basename(f).startswith("_")]
    v = pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True)
    return v[(v.testability == "TESTABLE") & (v.rule_verdict == "HOLDS")]


def load_exit_stops() -> dict:
    """cohort drawdown stop per rule from #7 exits."""
    fs = glob.glob(os.path.join(BT, "exits", "*.parquet"))
    out = {}
    for f in fs:
        d = pd.read_parquet(f, columns=["rule_id", "cohort_dd_stop"])
        if len(d):
            out[d["rule_id"].iloc[0]] = float(d["cohort_dd_stop"].iloc[0])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-winrate", type=float, default=58.0,
                    help="BUY only on rules with out-of-sample win-rate >= this "
                         "(edge over a coin flip; drift-proof, unlike median)")
    ap.add_argument("--top", type=int, default=200, help="shortlist size")
    args = ap.parse_args()
    os.makedirs(DEC_DIR, exist_ok=True)

    val = load_validation()
    stops = load_exit_stops()
    uni = pd.read_parquet(os.path.join(DATA, "universe_hist.parquet"),
                          columns=["guru_key", "name", "nse_symbol"]).set_index("guru_key")
    # BUY rules = survivors whose out-of-sample WIN-RATE clears the bar. Win-rate
    # (not median) is the selectivity lever: long-horizon medians are inflated by
    # 19y of market drift, but a >58% hit-rate is a real edge over a coin flip.
    vb = val[val["valid_winrate"] >= args.min_winrate].copy()
    best_h = vb.sort_values("valid_winrate", ascending=False).drop_duplicates("rule_id")
    surv = set(best_h["rule_id"])
    log(f"survivor BUY rules (valid win-rate >= {args.min_winrate}%): {len(surv)}")

    # AS_OF = latest trigger date across survivors
    as_of = pd.Timestamp("2000-01-01")
    live = []
    for rid in surv:
        tf = os.path.join(BT, "triggers", f"{rid}.parquet")
        if not os.path.exists(tf):
            continue
        t = pd.read_parquet(tf, columns=["guru_key", "symbol", "base_date",
                                         "entry_price", "clause_snapshot"])
        t["base_date"] = pd.to_datetime(t["base_date"])
        as_of = max(as_of, t["base_date"].max())
        # most recent trigger per stock for this rule
        t = t.sort_values("base_date").drop_duplicates("guru_key", keep="last")
        win = rule_recency_days(rid)
        cut = t["base_date"].max() - pd.Timedelta(days=win)
        t = t[t["base_date"] >= cut]
        if t.empty:
            continue
        rstats = best_h[best_h.rule_id == rid].iloc[0]
        t["rule_id"] = rid
        t["horizon"] = rstats["horizon"]
        t["hz_class"] = HZCLASS.get(int(str(rstats["horizon"])[:-1]), "MID")
        t["valid_median_ret"] = rstats["valid_median_ret"]
        t["valid_winrate"] = rstats["valid_winrate"]
        t["n_valid"] = rstats["n_valid"]
        t["exit_stop_pct"] = stops.get(rid, np.nan)
        live.append(t)
    if not live:
        log("no live signals"); return
    L = pd.concat(live, ignore_index=True)
    L["name"] = L["guru_key"].map(uni["name"])
    L["nse_symbol"] = L["guru_key"].map(uni["nse_symbol"])
    log(f"AS_OF {as_of.date()} | {len(L)} live (stock,rule) signals on "
        f"{L['guru_key'].nunique()} stocks")

    # ---- BUY screen: one row per (stock, rule), ranked by win-rate ----
    buy = L.sort_values("valid_winrate", ascending=False)[[
        "name", "nse_symbol", "guru_key", "rule_id", "hz_class", "horizon",
        "valid_winrate", "valid_median_ret", "n_valid", "base_date",
        "entry_price", "exit_stop_pct", "clause_snapshot"]]

    # ---- stock x horizon-class matrix + CONVICTION score ----
    # conviction = sum over firing rules of (win-rate - 50), i.e. total edge-over-
    # coin-flip currently pointing at this stock. Rewards consensus of high-hit-rate
    # rules, and is drift-proof (win-rate, not return).
    rows = []
    for gk, g in L.groupby("guru_key"):
        rec = {"name": g["name"].iloc[0], "nse_symbol": g["nse_symbol"].iloc[0],
               "guru_key": gk, "n_rules_firing": g["rule_id"].nunique(),
               "conviction": round(float((g["valid_winrate"] - 50).clip(lower=0).sum()), 1)}
        for hc in ("SHORT", "MID", "LONG"):
            sub = g[g["hz_class"] == hc]
            if sub.empty:
                rec[f"{hc}"] = "-"
            else:
                b = sub.sort_values("valid_winrate", ascending=False).iloc[0]
                rec[f"{hc}"] = "BUY"
                rec[f"{hc}_win"] = round(float(b["valid_winrate"]), 1)
                rec[f"{hc}_median"] = round(float(b["valid_median_ret"]), 1)
                rec[f"{hc}_rule"] = b["rule_id"]
                rec[f"{hc}_stop%"] = (None if pd.isna(b["exit_stop_pct"])
                                      else round(float(b["exit_stop_pct"]), 1))
        rows.append(rec)
    matrix = pd.DataFrame(rows).sort_values(
        ["conviction", "n_rules_firing"], ascending=False)

    matrix_top = matrix.head(args.top)
    readme = pd.DataFrame([
        ("AS OF", str(as_of.date()) + " (latest data). Re-run after new prices / "
         "results to refresh."),
        ("Ranking = CONVICTION", "Stocks ranked by conviction = total edge-over-coin-"
         "flip of all validated rules firing on them (sum of win-rate-50 across "
         "firing rules). High conviction = many high-hit-rate rules agree. This is "
         "drift-proof; we deliberately do NOT rank by long-horizon return because 19y "
         "of market rise inflates it (a 36M 'BUY' on absolute return = almost every "
         "stock). Fixing that properly needs excess-vs-index (task #8, pending)."),
        ("BUY_screen", "Every (stock, validated-rule) signal currently active: the "
         "rule's most recent trigger for that stock is within its recency window "
         "(technical 30d, fundamental/combined 120d). Ranked by the rule's "
         "OUT-OF-SAMPLE median return. valid_* = 2019-26 validation stats. "
         "clause_snapshot = why it fired (actual metric values)."),
        ("Stock_matrix", "One row per stock x horizon-class (SHORT<=6M / MID 6-24M / "
         "LONG>=24M). A stock can be BUY at one horizon and '-' at another - mixed "
         "verdicts shown side by side. Cell shows the strongest active rule's "
         "validated median, win-rate, and recommended stop."),
        ("exit_stop_pct / stop%", "The recommended drawdown STOP for that rule (from "
         "#7 exits: the drawdown its historical stocks typically endured). Once you "
         "BUY, exit if the position falls below this from its peak. Alternatives: "
         "take-profit at +100% (max median) or a 25% trailing stop (max disaster "
         "avoidance) - see exits_summary.xlsx."),
        ("SELL / HOLD", "True SELL needs a position. In screener mode each BUY ships "
         "with its exit rule above; once you hold, SELL is mechanical when the stop "
         "or target hits. HOLD = you own it and neither has fired."),
        ("Caveats", "Signals from validated survivor rules only (~830 that held out-"
         "of-sample). Not investment advice; backtest-derived, excludes costs/"
         "liquidity. Cross-check n_valid (sample size) and liquidity before acting."),
    ], columns=["item", "explanation"])

    out = os.path.join(DEC_DIR, f"buy_hold_sell_{as_of.date()}.xlsx")
    with pd.ExcelWriter(out, engine="openpyxl") as xw:
        readme.to_excel(xw, "README", index=False)
        matrix_top.to_excel(xw, "Top_conviction", index=False)
        buy.head(3000).to_excel(xw, "BUY_screen", index=False)
        matrix.to_excel(xw, "Stock_matrix_all", index=False)
    buy.to_parquet(os.path.join(DEC_DIR, f"buy_signals_{as_of.date()}.parquet"),
                   index=False)
    log(f"decisions -> {out}")
    log(f"  BUY candidates: {matrix.shape[0]} stocks | "
        f"SHORT {(matrix.get('SHORT')=='BUY').sum()} "
        f"MID {(matrix.get('MID')=='BUY').sum()} "
        f"LONG {(matrix.get('LONG')=='BUY').sum()}")


if __name__ == "__main__":
    main()
