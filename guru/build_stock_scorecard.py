r"""
STOCK SCORECARD — build_stock_scorecard.py  (Project Guru)

Flips the analysis from "per rule" to "PER STOCK":
  which stocks currently meet the criteria, HOW MANY validated rules each one
  passes, WHICH rules exactly, and what return profile those rules imply
  (median / best / worst / max / min), plus liquidity.

Every return number is the rule's OUT-OF-SAMPLE (2019-26 validation) result —
never the in-sample figure.

Key design choices (learned from earlier analysis):
  * rules are collapsed to FAMILIES for the "diversity" count, because with 830
    rules many are the same idea at different thresholds (sales_yoy>=25/30/40).
    n_rules tells you raw count; n_families tells you independent evidence.
  * a stock is "live" if a rule triggered within RECENCY days (technical rules
    move fast, fundamentals are quarterly — see rule_recency_days()).
  * liquidity (avg 20d turnover) is shown because the best-returning rules
    historically concentrate in illiquid microcaps.

Output: guru/STOCK_SCORECARD.xlsx
  README | Scorecard_<H> per horizon | Rule_Detail (stock x rule long format)

Usage:
    python guru/build_stock_scorecard.py                 # default 120d window
    python guru/build_stock_scorecard.py --days 90 --min-families 3
"""
from __future__ import annotations
import argparse, glob, os, re
from datetime import datetime
import numpy as np, pandas as pd

GURU = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(GURU, "data")
BT = os.path.join(GURU, "backtest")
OHLCV = os.path.join(DATA, "ohlcv_hist")
OUT = os.path.join(GURU, "STOCK_SCORECARD.xlsx")
RULES_X = os.path.join(os.path.dirname(GURU), "Project_Guru", "rule_template.xlsx")
HZ = ["3M", "6M", "12M", "24M", "36M"]


def log(m): print(f"[{datetime.now().strftime('%H:%M:%S')}] {m}", flush=True)


def family(rid: str) -> str:
    return re.sub(r"_?\d+.*$", "", rid)


def rule_recency_days(rid: str) -> int:
    """technical signals decay fast; fundamental/combo are quarter-anchored."""
    return 45 if rid.startswith("TECH") else 120


def load_validation() -> pd.DataFrame:
    fs = [f for f in glob.glob(os.path.join(BT, "validation", "*.parquet"))
          if not os.path.basename(f).startswith("_")]
    v = pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True)
    return v[(v.testability == "TESTABLE") & (v.rule_verdict == "HOLDS")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=0,
                    help="override recency window for ALL rules (0 = per-rule default)")
    ap.add_argument("--min-families", type=int, default=1,
                    help="only report stocks with at least this many distinct families")
    ap.add_argument("--min-liquidity", type=float, default=0.0,
                    help="minimum avg 20d traded value in Rs crore/day")
    ap.add_argument("--min-score", type=float, default=None,
                    help="minimum lift-weighted score (see build_family_lift.py)")
    ap.add_argument("--top", type=int, default=0, help="keep only top N per horizon")
    args = ap.parse_args()

    # empirical family lift: how much each family's PRESENCE shifts the odds.
    # This is what makes the ranking discriminate — a raw rule count does not.
    lift_p = os.path.join(BT, "family_lift.parquet")
    lift_tbl = pd.read_parquet(lift_p) if os.path.exists(lift_p) else pd.DataFrame()
    if lift_tbl.empty:
        log("WARNING: no family_lift.parquet — run build_family_lift.py first; "
            "scores will be zero")

    val = load_validation()
    # drawdown lives in the scorecard fragments, not the validation files
    sc_files = glob.glob(os.path.join(BT, "scores", "*.parquet"))
    sc = pd.concat([pd.read_parquet(f) for f in sc_files], ignore_index=True)
    val = val.merge(sc[["rule_id", "horizon", "median_max_drawdown_pct"]],
                    on=["rule_id", "horizon"], how="left")
    log(f"validated survivor rule-horizons: {len(val)} "
        f"({val.rule_id.nunique()} rules)")
    names = pd.read_excel(RULES_X, "Rules")[["rule_id", "rule_name"]]
    name_map = dict(zip(names.rule_id, names.rule_name))
    uni = pd.read_parquet(os.path.join(DATA, "universe_hist.parquet"),
                          columns=["guru_key", "name", "nse_symbol"]).set_index("guru_key")

    # ---- collect LIVE triggers per rule ----
    live = []
    rules = sorted(val.rule_id.unique())
    for i, rid in enumerate(rules, 1):
        tf = os.path.join(BT, "triggers", f"{rid}.parquet")
        if not os.path.exists(tf):
            continue
        t = pd.read_parquet(tf, columns=["guru_key", "base_date"])
        t["base_date"] = pd.to_datetime(t["base_date"])
        # most recent trigger per stock for this rule
        t = t.sort_values("base_date").drop_duplicates("guru_key", keep="last")
        win = args.days if args.days else rule_recency_days(rid)
        cut = t["base_date"].max() - pd.Timedelta(days=win)
        t = t[t["base_date"] >= cut]
        if t.empty:
            continue
        t["rule_id"] = rid
        live.append(t)
        if i % 200 == 0:
            log(f"  scanned {i}/{len(rules)} rules")
    L = pd.concat(live, ignore_index=True)
    L["family"] = L["rule_id"].map(family)
    log(f"live (stock,rule) signals: {len(L):,} on {L.guru_key.nunique():,} stocks")

    # ---- liquidity per stock (avg 20d turnover, Rs cr) ----
    liq = {}
    for gk in L.guru_key.unique():
        fp = os.path.join(OHLCV, f"{gk}.parquet")
        if not os.path.exists(fp):
            continue
        px = pd.read_parquet(fp, columns=["date", "close", "volume"]).tail(20)
        if len(px):
            liq[gk] = float((px["close"] * px["volume"]).mean() / 1e7)
    log(f"liquidity computed for {len(liq):,} stocks")

    # ---- per horizon: join each live signal to that rule's validated stats ----
    sheets = {}
    detail_rows = []
    for H in HZ:
        vh = val[val.horizon == H][["rule_id", "valid_median_ret", "valid_winrate",
                                    "n_valid", "median_max_drawdown_pct"]]
        j = L.merge(vh, on="rule_id", how="inner")
        if j.empty:
            continue
        lmap = {}
        if not lift_tbl.empty:
            lh = lift_tbl[lift_tbl.horizon == H]
            lmap = dict(zip(lh.family, lh.lift_pp))
        rows = []
        for gk, g in j.groupby("guru_key"):
            if g["family"].nunique() < args.min_families:
                continue
            fams_here = sorted(g.family.unique())
            pos = [f for f in fams_here if lmap.get(f, 0) > 0]
            neg = [f for f in fams_here if lmap.get(f, 0) < 0]
            score = float(sum(lmap.get(f, 0.0) for f in fams_here))
            best = g.loc[g["valid_median_ret"].idxmax()]
            worst = g.loc[g["valid_median_ret"].idxmin()]
            rules_sorted = g.sort_values("valid_median_ret", ascending=False)
            rows.append({
                "name": uni.loc[gk, "name"] if gk in uni.index else gk,
                "nse_symbol": uni.loc[gk, "nse_symbol"] if gk in uni.index else None,
                "guru_key": gk,
                "lift_score": round(score, 1),
                "n_positive_families": len(pos),
                "n_negative_families": len(neg),
                "warning_families": ", ".join(neg) if neg else "",
                "n_rules_passing": int(g.rule_id.nunique()),
                "n_families": int(g.family.nunique()),
                "liquidity_20d_cr": round(liq.get(gk, np.nan), 2),
                # RETURN PROFILE across all rules this stock passes
                "median_expected_ret": round(float(g.valid_median_ret.median()), 1),
                "mean_expected_ret": round(float(g.valid_median_ret.mean()), 1),
                "max_ret_best_rule": round(float(g.valid_median_ret.max()), 1),
                "min_ret_worst_rule": round(float(g.valid_median_ret.min()), 1),
                "avg_winrate": round(float(g.valid_winrate.mean()), 1),
                "best_winrate": round(float(g.valid_winrate.max()), 1),
                "median_drawdown": round(float(g.median_max_drawdown_pct.median()), 1),
                "best_rule": best["rule_id"],
                "best_rule_name": name_map.get(best["rule_id"], "")[:70],
                "worst_rule": worst["rule_id"],
                "worst_rule_name": name_map.get(worst["rule_id"], "")[:70],
                "families_list": ", ".join(sorted(g.family.unique())),
                "all_rules": ", ".join(rules_sorted.rule_id.tolist()[:40]),
                "top5_rule_names": " | ".join(
                    name_map.get(r, "")[:45] for r in rules_sorted.rule_id.head(5)),
            })
            if H == "24M":
                for _, r in rules_sorted.iterrows():
                    detail_rows.append({
                        "name": uni.loc[gk, "name"] if gk in uni.index else gk,
                        "guru_key": gk, "rule_id": r["rule_id"],
                        "rule_name": name_map.get(r["rule_id"], ""),
                        "family": r["family"], "trigger_date": r["base_date"],
                        "valid_median_ret": r["valid_median_ret"],
                        "valid_winrate": r["valid_winrate"],
                        "n_valid": r["n_valid"]})
        d = pd.DataFrame(rows)
        if d.empty:
            continue
        # SHARP LIST: apply liquidity / score filters, rank by lift score
        if args.min_liquidity:
            d = d[d.liquidity_20d_cr >= args.min_liquidity]
        if args.min_score is not None:
            d = d[d.lift_score >= args.min_score]
        d = d.sort_values(["lift_score", "avg_winrate"], ascending=False)
        if args.top:
            d = d.head(args.top)
        sheets[f"Scorecard_{H}"] = d
        log(f"  {H}: {len(d):,} stocks pass (>= {args.min_families} families)")

    readme = pd.DataFrame([
        ("PURPOSE", "PER-STOCK view: which stocks currently meet validated criteria, "
         "how many rules they pass, exactly which rules, and the return profile "
         "those rules imply."),
        ("Scorecard_<H> sheets", "One row per stock, for that holding period "
         "(3/6/12/24/36 months). Sorted by n_families then expected return."),
        ("lift_score  <-- RANK BY THIS", "Sum of the empirical lift (in percentage "
         "points of win-rate) of every family currently firing on this stock, from "
         "FAMILY_LIFT.xlsx. Families that historically IMPROVE the odds add to it; "
         "families that historically HURT (valuation-at-peak, oversold, crash-"
         "reversal) subtract. This is the ranking column — unlike a raw rule count, "
         "which was measured to be flat (~42% return / 67% win-rate) regardless of "
         "how many rules fired."),
        ("n_positive_families / n_negative_families / warning_families",
         "How many firing families help vs hurt the odds, and the names of the "
         "harmful ones. A stock can pass 20 rules and still be a poor candidate if "
         "several are warning signals — check warning_families before acting."),
        ("n_rules_passing", "How many individual validated rules currently fire on "
         "this stock. NOTE: deliberately NOT the ranking column (see lift_score)."),
        ("n_families", "How many DISTINCT rule families — this is the meaningful "
         "diversity number. 830 rules collapse to 62 families; many rules are the "
         "same idea at different thresholds (sales_yoy>=25/30/40), so 12 rules "
         "from 1 family is ONE piece of evidence, not twelve."),
        ("median_expected_ret", "Median of the out-of-sample median returns of all "
         "rules firing on this stock — the typical expectation if this stock "
         "behaves like the historical average for these rules."),
        ("max_ret_best_rule / best_rule", "The most optimistic rule firing on this "
         "stock and its historical out-of-sample median return, plus which rule "
         "that is (best_rule_name)."),
        ("min_ret_worst_rule / worst_rule", "The least optimistic rule firing, i.e. "
         "the pessimistic case among the signals present."),
        ("avg_winrate / best_winrate", "Average and best out-of-sample win-rate "
         "across the firing rules (% of historical triggers that were profitable)."),
        ("median_drawdown", "Typical worst peak-to-trough fall these rules' "
         "positions experienced within the horizon — the expected pain."),
        ("liquidity_20d_cr", "Average daily traded value over the last 20 sessions, "
         "in Rs crore. CRITICAL: the highest-returning rules historically "
         "concentrate in illiquid microcaps. Filter this before acting."),
        ("families_list / all_rules / top5_rule_names", "The exact families and "
         "rule IDs firing, plus readable names of the 5 highest-expectation rules."),
        ("Rule_Detail", "Long format (one row per stock x rule) at the 24M horizon, "
         "for filtering/pivoting in Excel."),
        ("RECENCY", "A signal counts as live if the rule triggered within 45 days "
         "(technical rules) or 120 days (fundamental/combo, which are quarter-"
         "anchored). Override with --days."),
        ("CAVEAT", "Expected returns are historical out-of-sample averages for the "
         "RULE, not a forecast for THIS stock. 2019-26 was a strong market. No "
         "transaction costs. Check liquidity and n_valid (sample size) before acting."),
    ], columns=["item", "explanation"])

    with pd.ExcelWriter(OUT, engine="openpyxl") as xw:
        readme.to_excel(xw, "README", index=False)
        for k, d in sheets.items():
            d.to_excel(xw, k, index=False)
        if detail_rows:
            pd.DataFrame(detail_rows).to_excel(xw, "Rule_Detail", index=False)
        ws = xw.book["README"]
        from openpyxl.styles import Alignment
        ws.column_dimensions["A"].width = 30
        ws.column_dimensions["B"].width = 105
        for row in ws.iter_rows(min_row=2):
            row[1].alignment = Alignment(wrap_text=True, vertical="top")
    log(f"STOCK SCORECARD -> {OUT}")


if __name__ == "__main__":
    main()
