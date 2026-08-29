r"""
Feature C — build_combos.py  (Project Guru)

Generates multidimensional combo rules by crossing VALIDATED survivor rules, then
writes them as a Rules/Clauses workbook the existing engine can backtest.

Seed selection (dimension-appropriate bar, user 2026-07):
  * technical/timing rules : HOLD out-of-sample AND beat >=80% of random timing
  * fundamental/quality/valuation/ownership : HOLD out-of-sample (a selection
    edge, so the timing-noise bar does not apply)
Cross ONLY across distinct dimensions (a growth x a quality x a technical, never
two near-duplicate growth clauses). Combo size 2..MAX_CLAUSES.
PRUNE before writing using funnel_facts co-occurrence: skip any combo whose
member conditions co-occur in < MIN_COOCCUR company-quarters historically
(they would produce ~0 triggers).

Output: guru/generated_combos.xlsx  (Rules + Clauses, same schema as rule_template)
Then:   python backtest_engine.py --rules-xlsx generated_combos.xlsx --outdir combos
        (backtests them into guru/backtest/combos/, original run untouched)

Usage:
    python guru/build_combos.py --dry-run     # seed + generation counts only
    python guru/build_combos.py               # write generated_combos.xlsx
"""
from __future__ import annotations

import argparse
import glob
import itertools
import os
import re
from datetime import datetime

import numpy as np
import pandas as pd

GURU_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(GURU_DIR, "data")
BT_DIR = os.path.join(GURU_DIR, "backtest")
VAL_DIR = os.path.join(BT_DIR, "validation")
FACTS_DIR = os.path.join(DATA_DIR, "funnel_facts")
SRC_XLSX = os.path.join(os.path.dirname(GURU_DIR), "Project_Guru", "rule_template.xlsx")
OUT_XLSX = os.path.join(GURU_DIR, "generated_combos.xlsx")

MAX_CLAUSES = 4
MIN_COOCCUR = 30          # company-quarters the combo's conditions must co-occur
MAX_COMBOS = 600          # curated set (each combo ~200s to backtest; quality>quantity)
TECH_NOISE_BAR = 80.0     # % of random a technical seed must beat


def log(m): print(f"[{datetime.now().strftime('%H:%M:%S')}] {m}", flush=True)


def dimension(fam: str) -> str:
    if fam.startswith("FUND") or fam == "MARGIN":
        return "growth"
    if fam.startswith("QUAL"):
        return "quality"
    if fam.startswith("VAL"):
        return "valuation"
    if fam.startswith("OWN"):
        return "ownership"
    if fam.startswith("MCAPVAR"):
        return "size"
    if fam.startswith("COMBO"):
        return "combo"        # already multi-dim; not used as a seed
    return "technical"


def select_seeds() -> pd.DataFrame:
    fs = [f for f in glob.glob(os.path.join(VAL_DIR, "*.parquet"))
          if not os.path.basename(f).startswith("_")]
    v = pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True)
    n = pd.read_parquet(os.path.join(VAL_DIR, "_noise_floor.parquet")
                        ).set_index("rule_id")["beats_pct_of_random"]
    # best out-of-sample horizon per rule among HOLDS
    h = v[(v.testability == "TESTABLE") & (v.rule_verdict == "HOLDS")].copy()
    if h.empty:
        return pd.DataFrame()
    best = h.sort_values("valid_median_ret", ascending=False).drop_duplicates("rule_id")
    best["fam"] = best["rule_id"].str.replace(r"_?\d+.*", "", regex=True)
    best["dim"] = best["fam"].map(dimension)
    best["beats_random"] = best["rule_id"].map(n)
    tech = best["dim"] == "technical"
    keep_tech = tech & (best["beats_random"] >= TECH_NOISE_BAR)
    keep_fund = (~tech) & (best["dim"] != "combo")
    seeds = best[keep_tech | keep_fund].copy()
    return seeds


def load_clauses() -> dict:
    cl = pd.read_excel(SRC_XLSX, "Clauses")
    return {rid: g.sort_values("clause_order").to_dict("records")
            for rid, g in cl.groupby("rule_id")}


# ---- co-occurrence check via funnel facts ----
_FACTS = None
def facts():
    global _FACTS
    if _FACTS is None:
        fs = glob.glob(os.path.join(FACTS_DIR, "*.parquet"))
        log(f"loading funnel facts for co-occurrence pruning ({len(fs)} files)…")
        _FACTS = pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True)
    return _FACTS


def clause_mask(F, metric, op, thr):
    op = str(op).strip()
    if metric in F.columns:
        v = F[metric]
    elif f"qmax_{metric}" in F.columns:
        v = None
    else:
        return None
    if v is not None:
        if op == "between":
            lo, hi = [float(x) for x in str(thr).split(",")]
            return (v >= lo) & (v <= hi)
        t = float(thr)
        return {">": v > t, ">=": v >= t, "<": v < t, "<=": v <= t, "==": v == t}[op]
    qmax, qmin = F[f"qmax_{metric}"], F[f"qmin_{metric}"]
    if op in (">", ">="):
        return qmax >= float(thr)
    if op in ("<", "<="):
        return qmin <= float(thr)
    if op == "==":
        return (qmax >= float(thr)) & (qmin <= float(thr))
    if op == "between":
        lo, hi = [float(x) for x in str(thr).split(",")]
        return (qmax >= lo) & (qmin <= hi)
    return None


def combo_cooccur(F, clause_lists) -> int:
    m = pd.Series(True, index=F.index)
    for cl in clause_lists:
        for c in cl:
            mk = clause_mask(F, c["metric"], c["operator"], c["threshold_value"])
            if mk is None:
                return -1                     # gap metric -> untestable
            m &= mk.fillna(False)
    return int(m.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--max-combos", type=int, default=MAX_COMBOS)
    args = ap.parse_args()

    seeds = select_seeds()
    if seeds.empty:
        log("no seeds — run validation first"); return
    log(f"seeds: {len(seeds)} rules")
    log("  by dimension: " + str(seeds["dim"].value_counts().to_dict()))
    clauses = load_clauses()

    # group seeds by dimension; rank within dim by out-of-sample median
    by_dim = {d: g.sort_values("valid_median_ret", ascending=False)["rule_id"].tolist()
              for d, g in seeds.groupby("dim")}
    dims = [d for d in ("growth", "quality", "valuation", "ownership", "size",
                        "technical") if d in by_dim]
    # keep only the BEST seeds per dimension (highest out-of-sample median) so the
    # 600 combos are crosses of genuinely strong primitives, not filler
    CAP = {"technical": 15, "growth": 12, "quality": 10, "valuation": 8,
           "ownership": 4, "size": 6}
    for d in dims:
        by_dim[d] = by_dim[d][:CAP.get(d, 8)]

    F = None if args.dry_run else facts()
    generated, pruned_zero, pruned_size = [], 0, 0
    combo_id = 0

    # build a candidate ITERATOR per dimension-pair/triple, then ROUND-ROBIN across
    # them so every cross-type is represented (not just the first). Pairs that
    # include 'technical' are the "good business + good entry" thesis -> listed
    # first so they get budget priority.
    dim_combos = []
    for k in (2, 3):
        dim_combos += list(itertools.combinations(dims, k))
    dim_combos.sort(key=lambda dc: (0 if "technical" in dc else 1, len(dc)))

    def gen_for(dcombo):
        for pick in itertools.product(*[by_dim[d] for d in dcombo]):
            yield dcombo, pick
    iters = [gen_for(dc) for dc in dim_combos]

    exhausted = [False] * len(iters)
    while len(generated) < args.max_combos and not all(exhausted):
        for i, it in enumerate(iters):
            if exhausted[i] or len(generated) >= args.max_combos:
                continue
            try:
                dcombo, pick = next(it)
            except StopIteration:
                exhausted[i] = True
                continue
            clause_lists = [clauses[r] for r in pick]
            if sum(len(c) for c in clause_lists) > MAX_CLAUSES:
                pruned_size += 1
                continue
            co = None
            if not args.dry_run:
                co = combo_cooccur(F, clause_lists)
                if co < MIN_COOCCUR:
                    pruned_zero += 1
                    continue
            combo_id += 1
            generated.append({"dims": "+".join(dcombo), "members": pick,
                              "clause_lists": clause_lists, "cooccur": co,
                              "id": combo_id})
    log(f"generated {len(generated)} combos "
        f"(pruned: {pruned_size} oversize, {pruned_zero} low-cooccurrence)")

    if args.dry_run:
        for g in generated[:8]:
            log(f"  DEEPGEN_{g['id']:04d} [{g['dims']}] <- {g['members']}")
        return

    # ---- write generated_combos.xlsx ----
    rule_rows, clause_rows = [], []
    for g in generated:
        rid = f"DEEPGEN_{g['id']:04d}"
        rule_rows.append({"rule_id": rid,
                          "rule_name": f"combo[{g['dims']}]: " + " + ".join(g["members"]),
                          "category": "combined",
                          "anchor_mode": "first_trigger",
                          "base_date_rule": "last_clause",
                          "cooccur_quarters": g["cooccur"],
                          "source": "generated (survivor cross)"})
        order = 0
        for cl in g["clause_lists"]:
            for c in cl:
                clause_rows.append({"rule_id": rid, "clause_order": order,
                                    "period_offset": c.get("period_offset", 0),
                                    "metric": c["metric"], "operator": c["operator"],
                                    "threshold_value": c["threshold_value"]})
                order += 1
    with pd.ExcelWriter(OUT_XLSX, engine="openpyxl") as xw:
        pd.DataFrame(rule_rows).to_excel(xw, "Rules", index=False)
        cdf = pd.DataFrame(clause_rows)
        for c in cdf.columns:
            if c.startswith("=") if isinstance(c, str) else False:
                pass
        cdf.to_excel(xw, "Clauses", index=False)
        # guard: any "==" operator cell would be read as a formula by Excel
        ws = xw.book["Clauses"]
        for row in ws.iter_rows(min_row=2):
            for cell in row:
                if isinstance(cell.value, str) and cell.value.startswith("="):
                    cell.data_type = "s"
    log(f"wrote {len(rule_rows)} combos -> {OUT_XLSX}")
    log("next: python guru/backtest_engine.py --rules-xlsx "
        f"{OUT_XLSX} --outdir combos")


if __name__ == "__main__":
    main()
