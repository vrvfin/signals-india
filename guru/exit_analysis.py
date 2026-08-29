r"""
#7 Phase 1a — exit_analysis.py  (Project Guru)

For each VALIDATED survivor rule, find the best EXIT among path-based exits, judged
OUT-OF-SAMPLE (validation-window triggers only). Reads the already-stored monthly
paths (month, ret_pct, peak_ret_pct, drawdown_pct) — no re-scan. Additive: writes
only backtest/exits/ + backtest/exits_summary.xlsx.

Exits tested (all read from the stored path):
  HOLD                    baseline: hold to the horizon
  TRAIL_15/20/25/30       exit the first month drawdown from peak <= -X%
  COHORT_DD               exit when drawdown breaches THIS rule's own historical
                          median max-drawdown (stop tuned to what its stocks endure)
  TIME_6/12/18            exit at N months if return still < +10% (dead money)
  TARGET_50/100           exit the first month return >= +X% (lock the gain)

Per (rule, exit, horizon): median realized return vs HOLD, % of peak captured,
% of kaputs avoided, % of sustained-winners cut short. Best exit per rule = the
one that most beats HOLD on validation median without cutting >40% of winners.

Usage:
    python guru/exit_analysis.py --limit 20      # pilot
    python guru/exit_analysis.py --shard 1/4     # parallel
    python guru/exit_analysis.py                 # all survivors + summary
    python guru/exit_analysis.py --status
"""
from __future__ import annotations
import argparse, glob, os
from datetime import datetime
import numpy as np, pandas as pd

GURU_DIR = os.path.dirname(os.path.abspath(__file__))
BT = os.path.join(GURU_DIR, "backtest")
VAL = os.path.join(BT, "validation")
EXIT_DIR = os.path.join(BT, "exits")
OUT_XLSX = os.path.join(BT, "exits_summary.xlsx")
SPLIT = pd.Timestamp("2018-12-31")
HZ = [3, 6, 12, 24, 36]
KAPUT = -50.0            # ret at horizon <= this = disaster
WINNER = 100.0          # ret at horizon >= this = sustained winner


def log(m): print(f"[{datetime.now().strftime('%H:%M:%S')}] {m}", flush=True)


def survivors() -> list[str]:
    fs = [f for f in glob.glob(os.path.join(VAL, "*.parquet"))
          if not os.path.basename(f).startswith("_")]
    v = pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True)
    return sorted(v[(v.testability == "TESTABLE") &
                    (v.rule_verdict == "HOLDS")].rule_id.unique())


def exit_month(path: pd.DataFrame, kind: str, cohort_dd: float):
    """path = one trigger's rows sorted by month. Returns the exit month (int) or
    None to hold to horizon."""
    m = path["month"].values
    ret = path["ret_pct"].values
    dd = path["drawdown_pct"].values
    if kind == "HOLD":
        return None
    if kind.startswith("TRAIL_"):
        x = -float(kind.split("_")[1])
        hit = np.where(dd <= x)[0]
        return int(m[hit[0]]) if len(hit) else None
    if kind == "COHORT_DD":
        hit = np.where(dd <= cohort_dd)[0]
        return int(m[hit[0]]) if len(hit) else None
    if kind.startswith("TIME_"):
        n = int(kind.split("_")[1])
        idx = np.where(m == n)[0]
        if len(idx) and ret[idx[0]] < 10:
            return n
        return None
    if kind.startswith("TARGET_"):
        x = float(kind.split("_")[1])
        hit = np.where(ret >= x)[0]
        return int(m[hit[0]]) if len(hit) else None
    return None


def realized(path: pd.DataFrame, ex_month, horizon: int) -> float:
    """return actually captured: at the exit month if it fires <= horizon, else
    the return at the horizon."""
    sub = path[path["month"] <= horizon]
    if sub.empty:
        return np.nan
    if ex_month is not None and ex_month <= horizon:
        r = sub[sub["month"] == ex_month]["ret_pct"]
        if len(r):
            return float(r.iloc[0])
    return float(sub[sub["month"] == sub["month"].max()]["ret_pct"].iloc[0])


EXITS = (["HOLD"] + [f"TRAIL_{x}" for x in (15, 20, 25, 30)] + ["COHORT_DD"]
         + [f"TIME_{n}" for n in (6, 12, 18)] + [f"TARGET_{x}" for x in (50, 100)])


def _exit_month_vec(months, RET, DD, kind, cohort_dd):
    """VECTORISED first-exit month per trigger. RET/DD are (n_trig x n_month)
    matrices aligned to `months`. Returns int array of exit-month (or a large
    sentinel = 'never') per trigger."""
    NEVER = 10**6
    n = RET.shape[0]
    if kind == "HOLD":
        return np.full(n, NEVER)
    if kind.startswith("TRAIL_"):
        mask = DD <= -float(kind.split("_")[1])
    elif kind == "COHORT_DD":
        mask = DD <= cohort_dd
    elif kind.startswith("TARGET_"):
        mask = RET >= float(kind.split("_")[1])
    elif kind.startswith("TIME_"):
        nmo = int(kind.split("_")[1])
        out = np.full(n, NEVER)
        if nmo in months:
            ci = list(months).index(nmo)
            hit = RET[:, ci] < 10
            out[hit] = nmo
        return out
    else:
        return np.full(n, NEVER)
    any_hit = mask.any(axis=1)
    first = np.where(any_hit, months[mask.argmax(axis=1)], NEVER)
    return first


def analyse_rule(rid: str) -> pd.DataFrame:
    tf = os.path.join(BT, "triggers", f"{rid}.parquet")
    pf = os.path.join(BT, "paths", f"{rid}.parquet")
    if not (os.path.exists(tf) and os.path.exists(pf)):
        return pd.DataFrame()
    trig = pd.read_parquet(tf, columns=["trigger_id", "base_date"])
    trig["base_date"] = pd.to_datetime(trig["base_date"])
    val_ids = set(trig[trig.base_date > SPLIT]["trigger_id"])
    if len(val_ids) < 20:
        return pd.DataFrame()
    paths = pd.read_parquet(pf, columns=["trigger_id", "month", "ret_pct",
                                         "drawdown_pct"])
    paths = paths[paths["trigger_id"].isin(val_ids)]
    # pivot to trigger x month matrices (the whole speed-up)
    ret = paths.pivot_table(index="trigger_id", columns="month", values="ret_pct")
    dd = paths.pivot_table(index="trigger_id", columns="month", values="drawdown_pct")
    months = np.array(sorted(ret.columns))
    ret = ret[months]; dd = dd[months]
    RET = ret.values; DD = dd.values
    col = {int(m): i for i, m in enumerate(months)}
    cohort_dd = float(np.nanmedian(np.nanmin(DD, axis=1)))
    # forward-fill returns across months so 'realized at exit' is well-defined even
    # if a trigger lacks that exact month row (delisted mid-path keeps last value)
    RETf = pd.DataFrame(RET).ffill(axis=1).values

    def realized_vec(ex_m, H):
        """return captured per trigger holding up to H with optional earlier exit."""
        if H not in col:
            return None
        hcol = col[H]
        # exit column = first month <= H where exit fired, else H
        use = np.minimum(ex_m, H)
        use = np.where(use > H, H, use)
        # map month value -> column index (months are monotonic)
        ci = np.searchsorted(months, use, side="left")
        ci = np.clip(ci, 0, hcol)
        ci = np.minimum(ci, hcol)
        return RETf[np.arange(len(ci)), ci]

    rows = []
    for H in HZ:
        if H not in col:
            continue
        hold = realized_vec(np.full(RET.shape[0], 10**6), H)
        valid = ~np.isnan(hold)
        if valid.sum() < 20:
            continue
        hold_v = hold[valid]
        is_kaput = hold_v <= KAPUT
        is_winner = hold_v >= WINNER
        for ex in EXITS:
            em = _exit_month_vec(months, RET, DD, ex, cohort_dd)
            rz = realized_vec(em, H)[valid]
            kap_av = (float((rz[is_kaput] > KAPUT).mean()) * 100 if is_kaput.any() else np.nan)
            win_cut = (float((rz[is_winner] < WINNER).mean()) * 100 if is_winner.any() else np.nan)
            rows.append({"rule_id": rid, "horizon": f"{H}M", "exit": ex,
                         "n": int(valid.sum()),
                         "median_ret": round(float(np.nanmedian(rz)), 1),
                         "hold_median": round(float(np.nanmedian(hold_v)), 1),
                         "vs_hold": round(float(np.nanmedian(rz) - np.nanmedian(hold_v)), 1),
                         "winrate": round(float((rz > 0).mean()) * 100, 1),
                         "pct_kaputs_avoided": None if np.isnan(kap_av) else round(kap_av, 1),
                         "pct_winners_cut": None if np.isnan(win_cut) else round(win_cut, 1),
                         "cohort_dd_stop": round(cohort_dd, 1)})
    return pd.DataFrame(rows)


def build_summary():
    fs = glob.glob(os.path.join(EXIT_DIR, "*.parquet"))
    if not fs:
        log("no exit results yet"); return
    a = pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True)
    meta = pd.read_excel(os.path.join(os.path.dirname(GURU_DIR), "Project_Guru",
                                      "rule_template.xlsx"), "Rules")[["rule_id", "rule_name"]]
    a = a.merge(meta, on="rule_id", how="left")
    a["fam"] = a["rule_id"].str.replace(r"_?\d+.*", "", regex=True)
    # best exit per (rule,horizon): beats hold, cuts < 40% of winners
    cand = a[(a.exit != "HOLD") & (a.vs_hold > 0) &
             ((a.pct_winners_cut.isna()) | (a.pct_winners_cut < 40))]
    best = cand.sort_values("vs_hold", ascending=False).drop_duplicates(["rule_id", "horizon"])
    # which exit wins most often, per family (the generalizable answer)
    famx = (best.groupby(["fam", "exit"]).size().reset_index(name="times_best")
            .sort_values(["fam", "times_best"], ascending=[True, False]))
    readme = pd.DataFrame([
        ("What", "Best EXIT per validated rule, judged out-of-sample (2019-26 "
         "triggers). Exits read the stored monthly path; 'vs_hold' = median return "
         "with the exit minus just holding to the horizon."),
        ("Exits", "TRAIL_X = sell on X% drop from peak. COHORT_DD = sell when "
         "drawdown breaches this rule's own historical median max-drawdown. TIME_N = "
         "sell at N months if still < +10%. TARGET_X = lock in at +X%."),
        ("How to read", "Positive vs_hold = the exit ADDS return over holding. "
         "pct_kaputs_avoided = of the positions that would have ended in disaster, "
         "how many the exit rescued. pct_winners_cut = of the eventual big winners, "
         "how many the exit sold too early (the cost). A good exit = high vs_hold, "
         "high kaputs_avoided, low winners_cut."),
        ("Best_per_rule", "The single best exit for each rule x horizon."),
        ("Best_per_family", "Which exit is most often best within each strategy "
         "family - the generalizable 'use this exit for this kind of rule' answer."),
    ], columns=["item", "explanation"])
    with pd.ExcelWriter(OUT_XLSX, engine="openpyxl") as xw:
        readme.to_excel(xw, "README", index=False)
        best.sort_values("vs_hold", ascending=False).to_excel(xw, "Best_per_rule", index=False)
        famx.to_excel(xw, "Best_per_family", index=False)
        a.to_excel(xw, "All_detail", index=False)
    log(f"summary: {a.rule_id.nunique()} rules -> {OUT_XLSX}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--shard", type=str, default="")
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--summary-only", action="store_true")
    args = ap.parse_args()
    os.makedirs(EXIT_DIR, exist_ok=True)
    if args.summary_only:
        build_summary(); return
    ids = survivors()
    if args.shard:
        k, n = [int(x) for x in args.shard.split("/")]
        ids = [r for i, r in enumerate(ids) if i % n == k - 1]
    done = {os.path.basename(f)[:-8] for f in glob.glob(os.path.join(EXIT_DIR, "*.parquet"))}
    if args.status:
        print(f"{len(done)} / {len(survivors())} survivors analysed")
        return
    todo = [r for r in ids if r not in done]
    if args.limit:
        todo = todo[:args.limit]
    log(f"exit analysis: {len(todo)} rules")
    for i, rid in enumerate(todo, 1):
        try:
            out = analyse_rule(rid)
            if not out.empty:
                out.to_parquet(os.path.join(EXIT_DIR, f"{rid}.parquet"), index=False)
        except Exception as e:
            log(f"  {rid} ERROR {str(e)[:70]}")
        if i % 100 == 0:
            log(f"  {i}/{len(todo)}")
    if not args.shard:
        build_summary()
    else:
        log("shard complete — run --summary-only for the xlsx")


if __name__ == "__main__":
    main()
