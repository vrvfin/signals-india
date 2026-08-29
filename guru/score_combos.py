r"""
Feature C scorer — rank generated combos, FLAG overfit (user choice: full-history
score with a train-vs-validation gap flag).

Reads backtest/combos/{triggers,paths}/, computes per combo x horizon:
  full-history median return + win-rate (big-sample view)
  train (<=2018) vs validation (>2018) median + win-rate
  overfit_gap = train_median - valid_median   (large positive = overfit risk)
  verdict: SOLID (valid median > 0 and gap small) / FADED (valid still + but gap
           big) / OVERFIT (valid median <= 0 while train +)
Output: guru/backtest/combos_summary.xlsx  (README + Ranked + Best_thesis)

Usage: python guru/score_combos.py
"""
from __future__ import annotations
import glob, os
from datetime import datetime
import pandas as pd, numpy as np

GURU_DIR = os.path.dirname(os.path.abspath(__file__))
CB = os.path.join(GURU_DIR, "backtest", "combos")
OUT = os.path.join(GURU_DIR, "backtest", "combos_summary.xlsx")
GEN = os.path.join(GURU_DIR, "generated_combos.xlsx")
SPLIT = pd.Timestamp("2018-12-31")
HZ = [3, 6, 12, 24, 36]


def log(m): print(f"[{datetime.now().strftime('%H:%M:%S')}] {m}", flush=True)


def main():
    tfs = glob.glob(os.path.join(CB, "triggers", "*.parquet"))
    if not tfs:
        log("no combo backtest output yet"); return
    meta = pd.read_excel(GEN, "Rules").set_index("rule_id")
    rows = []
    for tf in tfs:
        rid = os.path.basename(tf)[:-8]
        trig = pd.read_parquet(tf, columns=["trigger_id", "base_date"])
        trig["base_date"] = pd.to_datetime(trig["base_date"])
        paths = pd.read_parquet(os.path.join(CB, "paths", f"{rid}.parquet"),
                                columns=["trigger_id", "month", "ret_pct"])
        for hm in HZ:
            p = paths[paths.month == hm]
            d = trig.merge(p, on="trigger_id", how="inner")
            if len(d) < 20:
                continue
            tr = d[d.base_date <= SPLIT]["ret_pct"]
            va = d[d.base_date > SPLIT]["ret_pct"]
            row = {"rule_id": rid, "horizon": f"{hm}M", "n_total": len(d),
                   "full_median": round(float(d.ret_pct.median()), 1),
                   "full_winrate": round(float((d.ret_pct > 0).mean()) * 100, 1),
                   "n_train": int(len(tr)), "n_valid": int(len(va)),
                   "dims": str(meta.loc[rid, "rule_name"]).split("]")[0].strip("combo[")
                   if rid in meta.index else "",
                   "members": str(meta.loc[rid, "rule_name"]).split(":", 1)[-1].strip()
                   if rid in meta.index else ""}
            if len(tr) >= 15 and len(va) >= 15:
                row["train_median"] = round(float(tr.median()), 1)
                row["valid_median"] = round(float(va.median()), 1)
                row["valid_winrate"] = round(float((va > 0).mean()) * 100, 1)
                row["overfit_gap"] = round(row["train_median"] - row["valid_median"], 1)
                if row["valid_median"] <= 0:
                    row["verdict"] = "OVERFIT"
                elif row["overfit_gap"] > row["valid_median"]:
                    row["verdict"] = "FADED"
                else:
                    row["verdict"] = "SOLID"
            else:
                row["verdict"] = "UNTESTABLE_THIN"
            rows.append(row)
    df = pd.DataFrame(rows)
    ranked = df.sort_values("valid_median", ascending=False, na_position="last")
    solid = df[df.verdict == "SOLID"].sort_values("valid_median", ascending=False)
    thesis = solid[solid.dims.str.contains("technical", na=False)
                   & solid.dims.str.contains("growth|quality", na=False, regex=True)]
    readme = pd.DataFrame([
        ("What", "Generated multidimensional combos (survivor rules crossed across "
         "dimensions), backtested on full history, flagged for overfit."),
        ("verdict", "SOLID = validation median > 0 and train->valid drop is smaller "
         "than the validation median (edge persists). FADED = still positive out-of-"
         "sample but a big train->valid drop. OVERFIT = validation median <=0 while "
         "train was positive (in-sample mirage). UNTESTABLE_THIN = <15 triggers in a "
         "window."),
        ("overfit_gap", "train_median - valid_median (in pp). Large positive = the "
         "in-sample result did not carry out-of-sample."),
        ("Best_thesis", "SOLID combos that cross a fundamental (growth/quality) with a "
         "technical entry - the 'good business + good entry' rules."),
        ("Note", "Original 2,088-rule results are untouched (backtest/combos/ is a "
         "separate tree). These combos are candidates, not confirmed - treat SOLID "
         "as the shortlist to inspect via plot_strategy.py."),
    ], columns=["item", "explanation"])
    with pd.ExcelWriter(OUT, engine="openpyxl") as xw:
        readme.to_excel(xw, "README", index=False)
        ranked.to_excel(xw, "Ranked", index=False)
        if not solid.empty:
            solid.to_excel(xw, "Solid", index=False)
        if not thesis.empty:
            thesis.to_excel(xw, "Best_thesis", index=False)
    v = df.verdict.value_counts().to_dict()
    log(f"scored {df.rule_id.nunique()} combos | verdicts {v} -> {OUT}")


if __name__ == "__main__":
    main()
