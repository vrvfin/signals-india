r"""
Feature A — funnel_analysis.py  (Project Guru, RESUMABLE, 2 phases)

Answers: "how many stocks passed clause X alone; of those, how many also pass
each further clause of the rule; and how do all clauses co-occur?" over the
FULL universe (user decision), at QUARTER grain.

PHASE 1  --build-facts   (one-time, resumable per company)
    For every company with metric data, build a quarter-grain facts row set:
      * all fundamental metrics as-of that quarter (quarterly_unified + metrics)
      * every technical metric aggregated within the quarter as BOTH qmax_<m>
        and qmin_<m>  (a daily condition 'm >= t' passed the quarter iff
        qmax_m >= t; 'm <= t' iff qmin_m <= t; '== v' iff qmax >= v >= qmin)
    -> guru/data/funnel_facts/<guru_key>.parquet
PHASE 2  --run           (per rule, resumable; fast vector ops on the facts)
    For each rule:
      1. UNIVARIATE: company-quarters passing each clause ALONE (total + by year)
      2. SEQUENTIAL: pass clause0 -> also clause1 -> ... (count + % retained)
      3. CO-OCCURRENCE: anchor clause i -> % of its passers that also pass j
    -> guru/backtest/funnels/<rule_id>.parquet   (long: view/clause/value rows)
    -> guru/backtest/funnels_summary.xlsx        (README + biggest rules)

Usage:
    python guru/funnel_analysis.py --build-facts            # phase 1 (resumes)
    python guru/funnel_analysis.py --run                    # phase 2 (resumes)
    python guru/funnel_analysis.py --run --rules COMBO_001  # specific rules
    python guru/funnel_analysis.py --status
"""
from __future__ import annotations

import argparse
import glob
import os
from datetime import datetime

import numpy as np
import pandas as pd

GURU_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(GURU_DIR, "data")
FACTS_DIR = os.path.join(DATA_DIR, "funnel_facts")
BT_DIR = os.path.join(GURU_DIR, "backtest")
FUNNEL_DIR = os.path.join(BT_DIR, "funnels")
FACTS_LEDGER = os.path.join(DATA_DIR, "_dump_status", "funnel_facts_ledger.parquet")
RUN_LEDGER = os.path.join(BT_DIR, "_funnel_ledger.parquet")
OUT_XLSX = os.path.join(BT_DIR, "funnels_summary.xlsx")
XLSX = os.path.join(os.path.dirname(GURU_DIR), "Project_Guru", "rule_template.xlsx")

QUNI = os.path.join(DATA_DIR, "metrics", "quarterly_unified")
FMET = os.path.join(DATA_DIR, "metrics", "fundamental")
TMET = os.path.join(DATA_DIR, "metrics", "technical")


def log(m): print(f"[{datetime.now().strftime('%H:%M:%S')}] {m}", flush=True)


# ---------------- phase 1: facts ----------------

def build_company_facts(gk: str) -> pd.DataFrame | None:
    frames = []
    qp = os.path.join(QUNI, f"{gk}.parquet")
    if os.path.exists(qp):
        q = pd.read_parquet(qp)
        q = q[q["period_end"].notna()].copy()
        q["quarter"] = pd.PeriodIndex(q["period_end"], freq="Q").astype(str)
        drop = ["guru_key", "period_end", "announcement_date", "filing_date",
                "source", "basis", "sales_label"]
        frames.append(q.drop(columns=[c for c in drop if c in q.columns],
                             errors="ignore").set_index("quarter"))
    fp = os.path.join(FMET, f"{gk}.parquet")
    if os.path.exists(fp):
        f = pd.read_parquet(fp)
        f = f[f["grain"] == "quarterly"].copy()
        if not f.empty:
            f["quarter"] = pd.PeriodIndex(pd.to_datetime(f["period_end_date"]),
                                          freq="Q").astype(str)
            keep = [c for c in f.columns if c not in
                    ("guru_key", "grain", "period", "period_end_date",
                     "announcement_date", "base_date_estimated", "quarter")]
            f2 = f[["quarter"] + keep].set_index("quarter")
            f2 = f2[[c for c in f2.columns
                     if not frames or c not in frames[0].columns]]
            frames.append(f2)
    tp = os.path.join(TMET, f"{gk}.parquet")
    if os.path.exists(tp):
        t = pd.read_parquet(tp)
        t["quarter"] = pd.PeriodIndex(t["date"], freq="Q").astype(str)
        num = t.select_dtypes("number")
        g = num.groupby(t["quarter"])
        qmax = g.max().add_prefix("qmax_")
        qmin = g.min().add_prefix("qmin_")
        frames.append(pd.concat([qmax, qmin], axis=1))
    if not frames:
        return None
    # a duplicated quarter index in any source breaks axis=1 concat
    frames = [f[~f.index.duplicated(keep="last")] for f in frames]
    out = pd.concat(frames, axis=1)
    out = out.loc[:, ~out.columns.duplicated()]
    out["guru_key"] = gk
    return out.reset_index().rename(columns={"index": "quarter"})


def phase_facts(args):
    uni = pd.read_parquet(os.path.join(DATA_DIR, "universe_hist.parquet"),
                          columns=["guru_key"])
    keys = sorted(set(uni["guru_key"]))
    if os.path.exists(FACTS_LEDGER):
        led = pd.read_parquet(FACTS_LEDGER)
        new = [k for k in keys if k not in set(led["guru_key"])]
        if new:
            led = pd.concat([led, pd.DataFrame({"guru_key": new,
                             "status": "pending"})], ignore_index=True)
    else:
        led = pd.DataFrame({"guru_key": keys, "status": "pending"})
    todo = led[led["status"] == "pending"]
    if args.limit:
        todo = todo.head(args.limit)
    log(f"funnel facts: {len(todo)} of {len(led)} companies")
    os.makedirs(FACTS_DIR, exist_ok=True)
    n_ok = n_empty = 0
    for i, (li, r) in enumerate(todo.iterrows(), 1):
        try:
            f = build_company_facts(r["guru_key"])
            if f is None or f.empty:
                led.at[li, "status"] = "empty"; n_empty += 1
            else:
                for c in f.columns:
                    if f[c].dtype == "float64":
                        f[c] = f[c].astype("float32")
                f.to_parquet(os.path.join(FACTS_DIR, f"{r['guru_key']}.parquet"),
                             index=False)
                led.at[li, "status"] = "done"; n_ok += 1
        except Exception as e:
            led.at[li, "status"] = "error"
            if "error" not in led.columns:
                led["error"] = ""
            led.at[li, "error"] = str(e)[:150]
        if i % 300 == 0:
            os.makedirs(os.path.dirname(FACTS_LEDGER), exist_ok=True)
            led.to_parquet(FACTS_LEDGER, index=False)
            log(f"  {i}/{len(todo)} (ok={n_ok} empty={n_empty})")
    os.makedirs(os.path.dirname(FACTS_LEDGER), exist_ok=True)
    led.to_parquet(FACTS_LEDGER, index=False)
    log(f"FACTS COMPLETE: ok={n_ok} empty={n_empty}")


# ---------------- phase 2: funnels ----------------

_FACTS = None
def load_all_facts() -> pd.DataFrame:
    global _FACTS
    if _FACTS is None:
        fs = glob.glob(os.path.join(FACTS_DIR, "*.parquet"))
        log(f"loading funnel facts ({len(fs)} companies)…")
        _FACTS = pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True)
        log(f"  facts: {len(_FACTS):,} company-quarters, {len(_FACTS.columns)} cols")
    return _FACTS


def clause_mask(F: pd.DataFrame, metric: str, op: str, thr) -> pd.Series | None:
    """quarter-grain pass mask for one clause over the facts frame."""
    op = str(op).strip()
    if metric in F.columns:                       # fundamental / quarterly metric
        v = F[metric]
        if op == "between":
            lo, hi = [float(x) for x in str(thr).split(",")]
            return (v >= lo) & (v <= hi)
        t = float(thr)
        return {">": v > t, ">=": v >= t, "<": v < t, "<=": v <= t,
                "==": v == t}[op]
    qmax, qmin = f"qmax_{metric}", f"qmin_{metric}"
    if qmax in F.columns:                         # technical daily -> quarter agg
        if op in (">", ">="):
            v = F[qmax]; t = float(thr)
            return v > t if op == ">" else v >= t
        if op in ("<", "<="):
            v = F[qmin]; t = float(thr)
            return v < t if op == "<" else v <= t
        if op == "==":
            t = float(thr)
            return (F[qmax] >= t) & (F[qmin] <= t)
        if op == "between":
            lo, hi = [float(x) for x in str(thr).split(",")]
            return (F[qmax] >= lo) & (F[qmin] <= hi)
    return None


def funnel_rule(rid: str, clauses: pd.DataFrame, F: pd.DataFrame) -> pd.DataFrame:
    rows = []
    masks = []
    yr = F["quarter"].str[:4]
    for _, c in clauses.iterrows():
        m = clause_mask(F, c["metric"], c["operator"], c["threshold_value"])
        label = f"{c['metric']} {c['operator']} {c['threshold_value']}"
        if m is None:
            rows.append({"view": "univariate", "clause": label, "measure": "events",
                         "value": np.nan, "note": "metric not in facts (gap)"})
            masks.append((label, None))
            continue
        m = m.fillna(False)
        masks.append((label, m))
        rows.append({"view": "univariate", "clause": label, "measure": "events",
                     "value": int(m.sum())})
        rows.append({"view": "univariate", "clause": label, "measure": "companies",
                     "value": int(F.loc[m, "guru_key"].nunique())})
        by = m.groupby(yr).sum()
        for y, v in by[by > 0].items():
            rows.append({"view": "univariate_by_year", "clause": label,
                         "measure": y, "value": int(v)})
    # sequential funnel
    cum = None
    for label, m in masks:
        if m is None:
            continue
        cum = m if cum is None else (cum & m)
        rows.append({"view": "sequential", "clause": label, "measure": "events",
                     "value": int(cum.sum())})
    # co-occurrence
    valid = [(l, m) for l, m in masks if m is not None]
    for li_, mi in valid:
        ni = int(mi.sum())
        if ni == 0:
            continue
        for lj, mj in valid:
            if li_ == lj:
                continue
            rows.append({"view": "cooccurrence", "clause": f"{li_} -> {lj}",
                         "measure": "pct_also_pass",
                         "value": round(100 * float((mi & mj).sum()) / ni, 2)})
    out = pd.DataFrame(rows)
    out["rule_id"] = rid
    return out


def phase_run(args):
    rules = pd.read_excel(XLSX, "Rules")
    clauses = pd.read_excel(XLSX, "Clauses")
    only = [x.strip() for x in args.rules.split(",") if x.strip()] if args.rules else None
    ids = [r for r in rules["rule_id"] if not only or r in only]
    if os.path.exists(RUN_LEDGER):
        led = pd.read_parquet(RUN_LEDGER)
        new = [r for r in ids if r not in set(led["rule_id"])]
        if new:
            led = pd.concat([led, pd.DataFrame({"rule_id": new,
                             "status": "pending"})], ignore_index=True)
    else:
        led = pd.DataFrame({"rule_id": ids, "status": "pending"})
    todo = led[led["status"] == "pending"]
    if only:
        todo = todo[todo["rule_id"].isin(only)]
    if args.limit:
        todo = todo.head(args.limit)
    log(f"funnel run: {len(todo)} rules")
    F = load_all_facts()
    os.makedirs(FUNNEL_DIR, exist_ok=True)
    for i, (li, r) in enumerate(todo.iterrows(), 1):
        try:
            cl = clauses[clauses["rule_id"] == r["rule_id"]].sort_values("clause_order")
            out = funnel_rule(r["rule_id"], cl, F)
            out.to_parquet(os.path.join(FUNNEL_DIR, f"{r['rule_id']}.parquet"),
                           index=False)
            led.at[li, "status"] = "done"
        except Exception as e:
            led.at[li, "status"] = "error"
            log(f"  {r['rule_id']} ERROR {str(e)[:80]}")
        if i % 50 == 0:
            led.to_parquet(RUN_LEDGER, index=False)
            log(f"  {i}/{len(todo)}")
    led.to_parquet(RUN_LEDGER, index=False)
    log("FUNNEL RUN COMPLETE")
    print(led["status"].value_counts().to_dict())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build-facts", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--rules", type=str, default="")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--status", action="store_true")
    args = ap.parse_args()
    if args.status:
        for name, p in [("facts", FACTS_LEDGER), ("runs", RUN_LEDGER)]:
            if os.path.exists(p):
                l = pd.read_parquet(p)
                print(name, l["status"].value_counts().to_dict())
            else:
                print(name, "not started")
        return
    if args.build_facts:
        phase_facts(args)
    if args.run:
        phase_run(args)
    if not args.build_facts and not args.run:
        phase_facts(args)
        phase_run(args)


if __name__ == "__main__":
    main()
