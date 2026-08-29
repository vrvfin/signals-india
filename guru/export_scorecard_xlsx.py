r"""
export_scorecard_xlsx.py — snapshot guru/backtest/scorecard.parquet to Excel.

Re-run ANYTIME while the backtest grinds: it exports whatever rules are done.
Output: guru/backtest/scorecard.xlsx
  sheet 'Scorecard'   one row per (rule, horizon) — sorted by median return
                      (v1 binding decision: absolute returns, no small-n drops)
  sheet 'Best_by_horizon'  top-30 rules per horizon by median return
  sheet 'Rules_meta'  rule names/categories for lookup

Usage:  python guru/export_scorecard_xlsx.py
"""
from __future__ import annotations

import os

import pandas as pd

GURU_DIR = os.path.dirname(os.path.abspath(__file__))
SCORE = os.path.join(GURU_DIR, "backtest", "scorecard.parquet")
OUT = os.path.join(GURU_DIR, "backtest", "scorecard.xlsx")
XLSX = os.path.join(os.path.dirname(GURU_DIR), "Project_Guru", "rule_template.xlsx")


def _load_scores() -> pd.DataFrame:
    """legacy single-file scorecard (pre-parallel) + per-rule fragments; a
    fragment wins over the legacy rows for the same rule. Partially-written
    fragments (parallel workers mid-flush) are skipped harmlessly."""
    import glob
    frames = []
    if os.path.exists(SCORE):
        frames.append(pd.read_parquet(SCORE))
    frag_dir = os.path.join(GURU_DIR, "backtest", "scores")
    for f in glob.glob(os.path.join(frag_dir, "*.parquet")):
        try:
            frames.append(pd.read_parquet(f))
        except Exception:
            continue
    if not frames:
        return pd.DataFrame()
    sc = pd.concat(frames, ignore_index=True)
    # fragments appended last win the dedup
    return sc.drop_duplicates(subset=["rule_id", "horizon"], keep="last")


def main() -> None:
    sc = _load_scores()
    if sc.empty:
        print("no scorecard yet — run the backtest first")
        return
    meta = pd.read_excel(XLSX, "Rules")[["rule_id", "rule_name", "category"]]
    sc = sc.merge(meta, on="rule_id", how="left")
    front = ["rule_id", "rule_name", "category", "horizon", "n_companies",
             "n_triggers", "n_episodes", "median_return_pct", "mean_return_pct",
             "min_return_pct", "max_return_pct", "p25_return_pct", "p75_return_pct",
             "success_prob_pos", "success_prob_2x", "success_prob_5x",
             "success_prob_10x", "median_max_drawdown_pct", "pct_dropped_big",
             "sustain_ratio_median"]
    sc = sc[[c for c in front if c in sc.columns]]
    sc = sc.sort_values("median_return_pct", ascending=False)

    best = (sc[sc["n_triggers"] > 0]
            .sort_values(["horizon", "median_return_pct"], ascending=[True, False])
            .groupby("horizon", as_index=False).head(30))

    readme = _readme_rows()
    with pd.ExcelWriter(OUT, engine="openpyxl") as xw:
        pd.DataFrame(readme, columns=["item", "explanation"]).to_excel(
            xw, "README", index=False)
        sc.to_excel(xw, "Scorecard", index=False)
        best.to_excel(xw, "Best_by_horizon", index=False)
        meta.to_excel(xw, "Rules_meta", index=False)
        ws = xw.book["README"]
        ws.column_dimensions["A"].width = 34
        ws.column_dimensions["B"].width = 130
        from openpyxl.styles import Alignment, Font
        for row in ws.iter_rows(min_row=2):
            row[0].font = Font(bold=row[1].value == "")
            row[1].alignment = Alignment(wrap_text=True, vertical="top")
    print(f"exported {sc['rule_id'].nunique()} rules x horizons "
          f"({len(sc)} rows) -> {OUT}")


def _readme_rows() -> list[tuple[str, str]]:
    return [
        ("HOW TO READ THIS WORKBOOK", ""),
        ("The one-line story", "Each row of 'Scorecard' answers: if you had bought EVERY stock that satisfied this rule, on the day the market learned it, and held for exactly <horizon>, what would have happened? Sorted by median_return_pct (your v1 decision: absolute returns)."),
        ("Reading order", "1) Pick a horizon you care about (e.g. 24M). 2) Filter Scorecard to that horizon or use Best_by_horizon. 3) Compare rules on median_return_pct AND success_prob_pos AND n_triggers together - never on one column alone. 4) For any interesting rule, replay real examples: python guru\\plot_strategy.py --rule <RULE_ID> --pick best (or worst)."),
        ("The three output layers", "This Excel is layer 3 (aggregates). Layer 1 = guru\\backtest\\triggers\\<RULE>.parquet: every individual entry with the exact metric values that fired (the WHY). Layer 2 = guru\\backtest\\paths\\<RULE>.parquet: month-by-month return/peak/drawdown after each entry (the MOVEMENT). Everything links via rule_id -> trigger_id."),
        ("", ""),
        ("HOW THE MATHS WORKED", ""),
        ("Trigger", "For fundamental rules: every quarter where the rule's conditions held (each clause checked at its quarter-offset from the anchor). The trigger DATE is the actual BSE/NSE filing timestamp of the LAST quarter needed - never the quarter-end - so the backtest only uses information the market actually had (no look-ahead)."),
        ("Entry price", "Next trading day's OPEN after the trigger timestamp (you can't buy the close of a filing that landed at 6pm). All returns are computed from this entry."),
        ("Return at horizon H", "ret = (close after H*21 trading days / entry_price) - 1, in %. Prices are split/bonus/dividend-adjusted (total-return basis). If a stock has fewer than H months of subsequent history (delisted, or triggered recently), it simply drops out of that horizon's row - visible as n_triggers shrinking at longer horizons."),
        ("Technical rules", "Trigger = first day the condition turns true (edge), with a 63-trading-day re-arm so one sustained signal isn't counted daily. Pre-2009 technical triggers are flagged low-confidence (data quality)."),
        ("Combined rules", "Fundamental conditions anchor the event; the technical clause must then fire within its offset window AFTER the filing; entry happens on the tape-confirmation day."),
        ("Median not mean", "median_return_pct is the headline because one 100-bagger drags the MEAN of a losing rule positive. mean_return_pct is shown for comparison - a big mean/median gap = lottery-ticket profile."),
        ("", ""),
        ("EXACT COLUMN MEANINGS (Scorecard sheet)", ""),
        ("rule_id / rule_name / category", "Which strategy (matches rule_template.xlsx). Category: fundamental_growth / technical / combined / quality / valuation / ownership."),
        ("horizon", "Holding period this row measures: 1M..120M months after entry. Same rule appears once per horizon - that's the rule x time view you asked for."),
        ("n_companies", "Distinct companies that ever triggered and were measurable at this horizon. CONTEXT ONLY - by your binding decision, small n is never filtered out; judge confidence yourself (1 company at 100% is a valid rare find)."),
        ("n_triggers", "Total (company, date) events measured at this horizon. One company can trigger many times."),
        ("n_episodes", "Triggers deduplicated to one per company-quarter - guards against one company inflating a rule by triggering repeatedly in adjacent windows."),
        ("min/max_return_pct", "Worst and best single trigger at this horizon."),
        ("mean/median_return_pct", "Average and middle outcome. Median is the sort key and headline number."),
        ("p25/p75_return_pct", "25th/75th percentile returns - the middle-half range. Wide spread = inconsistent rule; narrow = repeatable."),
        ("success_prob_pos", "% of triggers with a POSITIVE return at this horizon. 50% = coin flip."),
        ("success_prob_2x/5x/10x", "% of triggers that were at/above +100% / +400% / +900% at this horizon (checkpoint value, not peak along the way)."),
        ("median_max_drawdown_pct", "The median trigger's WORST peak-to-trough fall within the horizon - the pain you'd have had to sit through. -53 means the typical position halved at some point."),
        ("pct_dropped_big", "% of triggers that fell 50%+ below entry at some point within the horizon (even if they recovered)."),
        ("sustain_ratio_median", "Median of (return at horizon / best return reached up to it). Near 1 = gains held. Near 0 = stocks peaked then round-tripped (the 'made 3x then gave it back' pattern)."),
        ("", ""),
        ("HONEST CAVEATS", ""),
        ("Survivorship", "Delisted companies ARE included where price history exists (760 dead names have prices; deep-dead pre-2010 names mostly don't). Long-horizon numbers still skew slightly optimistic."),
        ("Quarterly depth", "Fundamental rules have dense data from ~2008 (NSE names) but BSE-only microcaps only from 2023 - a rule's trigger span matters; check n_triggers at long horizons."),
        ("No costs", "Returns exclude brokerage/slippage/taxes, and assume the next-day open was tradeable (upper-circuit days can make that impossible for hot microcaps)."),
        ("Not yet applied", "No index-relative comparison (v2), no train/validation overfitting split yet, no liquidity/mcap filters (deliberately unconstrained v1 - filters come as post-hoc views)."),
    ]


if __name__ == "__main__":
    main()
