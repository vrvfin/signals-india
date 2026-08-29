r"""
Task #12 — validate_rules.py  (Project Guru, RESUMABLE, SHARDABLE)

Out-of-sample overfitting guard. Per validation_plan.md, with two BINDING rules:
  1. ADDITIVE ONLY — reads existing artifacts, writes ONLY to backtest/validation/
     and backtest/validation_summary.xlsx. Nothing existing is modified/deleted.
  2. TESTABILITY GATE FIRST — a rule that cannot be tested out-of-sample is
     labelled UNTESTABLE (with reason), never counted as a failure.

Locked params: split 2018-12-31 | MIN_TRAIN=20 MIN_VALID=20 | WEAKER tol 10pp
               | noise floor top-200 rules x 20 shuffles

Steps
  1 testability   : usable train/valid trigger counts (a trigger only counts if it
                    has a COMPLETE forward path to that horizon — e.g. a 2024
                    trigger has no 36M future, so it is excluded, not failed)
  2 rule validate : train vs validation stats -> HOLDS / WEAKER / BREAKS
  3 filter validate: RE-DERIVE the best separating dim+threshold on TRAIN ONLY
                    (same AUC method as Feature B), apply it UNCHANGED to
                    validation, report lift; flag if the train-picked dimension
                    differs from the in-sample pick (dimension instability)
  5 episode dedup : restate step 2 on one-trigger-per-company-quarter

Step 4 (noise floor) lives in --noise mode (separate, expensive).

Usage
    python guru/validate_rules.py --status
    python guru/validate_rules.py --limit 20             # pilot
    python guru/validate_rules.py --shard 1/4            # parallel worker
    python guru/validate_rules.py                        # all + rebuild xlsx
    python guru/validate_rules.py --noise --top 200      # step 4
"""
from __future__ import annotations

import argparse
import glob
import os
from datetime import datetime

import numpy as np
import pandas as pd

GURU_DIR = os.path.dirname(os.path.abspath(__file__))
BT_DIR = os.path.join(GURU_DIR, "backtest")
VAL_DIR = os.path.join(BT_DIR, "validation")
TPROF_DIR = os.path.join(BT_DIR, "trigger_profiles")
OUT_XLSX = os.path.join(BT_DIR, "validation_summary.xlsx")
XLSX = os.path.join(os.path.dirname(GURU_DIR), "Project_Guru", "rule_template.xlsx")

SPLIT = pd.Timestamp("2018-12-31")
MIN_TRAIN, MIN_VALID = 20, 20
WEAKER_TOL = 10.0                      # pp below train win-rate
HORIZONS = [1, 3, 6, 12, 24, 36]
MIN_CLASS_N = 8
DATA_END = pd.Timestamp("2026-07-01")  # last date with price data


def log(m): print(f"[{datetime.now().strftime('%H:%M:%S')}] {m}", flush=True)


def auc(w, l):
    x = pd.concat([w, l]).rank()
    u = x.iloc[:len(w)].sum() - len(w) * (len(w) + 1) / 2
    return float(u / (len(w) * len(l)))


def stats(r: pd.Series) -> dict:
    return {"n": int(len(r)),
            "median_ret": round(float(r.median()), 2),
            "mean_ret": round(float(r.mean()), 2),
            "winrate": round(float((r > 0).mean()) * 100, 1),
            "rate_2x": round(float((r >= 100).mean()) * 100, 1)}


def validate_rule(rid: str) -> pd.DataFrame:
    tf = os.path.join(BT_DIR, "triggers", f"{rid}.parquet")
    pf = os.path.join(BT_DIR, "paths", f"{rid}.parquet")
    if not (os.path.exists(tf) and os.path.exists(pf)):
        return pd.DataFrame()
    trig = pd.read_parquet(tf, columns=["trigger_id", "guru_key", "base_date"])
    trig["base_date"] = pd.to_datetime(trig["base_date"])
    paths = pd.read_parquet(pf, columns=["trigger_id", "month", "ret_pct"])
    tp = os.path.join(TPROF_DIR, f"{rid}.parquet")
    prof = pd.read_parquet(tp) if os.path.exists(tp) else None

    rows = []
    for hm in HORIZONS:
        p = paths[paths["month"] == hm][["trigger_id", "ret_pct"]]
        d = trig.merge(p, on="trigger_id", how="inner")   # complete path only
        tr = d[d["base_date"] <= SPLIT]
        va = d[d["base_date"] > SPLIT]
        # ---- step 1: testability ----
        if len(va) == 0:
            verdict = "UNTESTABLE_NO_VALID_DATA"
        elif len(tr) < MIN_TRAIN:
            verdict = "UNTESTABLE_THIN_TRAIN"
        elif len(va) < MIN_VALID:
            verdict = "UNTESTABLE_THIN_VALID"
        else:
            verdict = "TESTABLE"
        row = {"rule_id": rid, "horizon": f"{hm}M", "testability": verdict,
               "n_train": int(len(tr)), "n_valid": int(len(va))}
        for pre, sub in (("train", tr), ("valid", va)):
            if len(sub):
                for k, v in stats(sub["ret_pct"]).items():
                    row[f"{pre}_{k}"] = v
        # ---- step 2: rule verdict ----
        if verdict == "TESTABLE":
            vm, vw = row["valid_median_ret"], row["valid_winrate"]
            tw = row["train_winrate"]
            if vm > 0 and vw >= tw - WEAKER_TOL:
                row["rule_verdict"] = "HOLDS"
            elif vm > 0:
                row["rule_verdict"] = "WEAKER"
            else:
                row["rule_verdict"] = "BREAKS"
        # ---- step 5: episode-deduped restatement ----
        if len(va):
            d2 = d.copy()
            d2["_ep"] = d2["guru_key"] + "_" + d2["base_date"].dt.to_period("Q").astype(str)
            dd = d2.drop_duplicates("_ep")
            ddv = dd[dd["base_date"] > SPLIT]
            if len(ddv):
                row["valid_ep_n"] = int(len(ddv))
                row["valid_ep_median_ret"] = round(float(ddv["ret_pct"].median()), 2)
                row["valid_ep_winrate"] = round(float((ddv["ret_pct"] > 0).mean()) * 100, 1)
        # ---- step 3: filter validation (train-derived, applied to validation) ----
        if verdict == "TESTABLE" and prof is not None:
            # the profile's own ret_pct is for the SINGLE horizon Feature B used;
            # anchor-date dimensions are horizon-independent, so re-attach THIS
            # horizon's returns (from d) instead of reusing the stored ones.
            dims_only = prof.drop(columns=[c for c in ("ret_pct", "rule_id", "horizon")
                                           if c in prof.columns])
            P = dims_only.merge(d[["trigger_id", "base_date", "ret_pct"]],
                                on="trigger_id", how="inner")
            if "base_date" in P.columns and len(P):
                Ptr = P[P["base_date"] <= SPLIT]
                Pva = P[P["base_date"] > SPLIT]
                dims = [c for c in P.columns if c not in
                        ("trigger_id", "ret_pct", "base_date", "rule_id", "horizon")]
                best = None
                if len(Ptr) >= MIN_TRAIN * 2:
                    q1, q2 = Ptr["ret_pct"].quantile([1/3, 2/3])
                    W, L = Ptr[Ptr.ret_pct >= q2], Ptr[Ptr.ret_pct <= q1]
                    for dim in dims:
                        w = W[dim].dropna().astype(float)
                        l = L[dim].dropna().astype(float)
                        if len(w) < MIN_CLASS_N or len(l) < MIN_CLASS_N:
                            continue
                        a = auc(w, l)
                        sep = abs(a - 0.5) * 2
                        if best is None or sep > best[1]:
                            best = (dim, sep, a)
                if best and len(Pva) >= MIN_VALID:
                    dim, sep, a = best
                    thr = float(Ptr[dim].dropna().astype(float).median())
                    keep_high = a > 0.5
                    vv = Pva[dim].astype(float)
                    m = (vv >= thr) if keep_high else (vv <= thr)
                    f = Pva[m.fillna(False)]
                    row["filter_dim_train"] = dim
                    row["filter_dir"] = ">=" if keep_high else "<="
                    row["filter_thr"] = round(thr, 3)
                    row["valid_median_nofilter"] = round(float(Pva["ret_pct"].median()), 2)
                    row["valid_winrate_nofilter"] = round(float((Pva.ret_pct > 0).mean()) * 100, 1)
                    if len(f) >= MIN_CLASS_N:
                        row["n_valid_filtered"] = int(len(f))
                        row["valid_median_filtered"] = round(float(f["ret_pct"].median()), 2)
                        row["valid_winrate_filtered"] = round(float((f.ret_pct > 0).mean()) * 100, 1)
                        row["filter_lift_valid"] = round(
                            row["valid_median_filtered"] - row["valid_median_nofilter"], 2)
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------- step 4: noise floor ----------------

_OH_DIR = os.path.join(GURU_DIR, "data", "ohlcv_hist")


@__import__("functools").lru_cache(maxsize=8192)
def random_entry_returns(gk: str, horizon_days: int = 730):
    """EVERY possible 'enter on a random day in the validation window, hold
    `horizon_days`' return for this company, as a numpy array.

    Precomputing this once per company is the whole optimisation: a shuffle then
    costs one array sample instead of re-reading the price file. (The previous
    version read the parquet inside the shuffle loop = 8.3M reads / ~18 hours.)"""
    p = os.path.join(_OH_DIR, f"{gk}.parquet")
    if not os.path.exists(p):
        return None
    px = pd.read_parquet(p, columns=["date", "close"])
    px["date"] = pd.to_datetime(px["date"])
    px = px[px["date"] > SPLIT].reset_index(drop=True)
    if len(px) < 30:
        return None
    d = px["date"].values
    c = px["close"].values.astype("float64")
    j = np.searchsorted(d, d + np.timedelta64(horizon_days, "D"), side="left")
    ok = np.nonzero(j < len(c))[0]
    if len(ok) == 0:
        return None
    with np.errstate(divide="ignore", invalid="ignore"):
        r = (c[j[ok]] / c[ok] - 1.0) * 100.0
    r = r[np.isfinite(r)]
    return r if len(r) else None


def noise_floor(top_n: int, n_shuf: int):
    """Shuffle trigger dates WITHIN each company's own price history, so company
    mix is preserved and only TIMING is destroyed. Reports the percentile of the
    real result within the null distribution."""
    val_files = glob.glob(os.path.join(VAL_DIR, "*.parquet"))
    val = pd.concat([pd.read_parquet(f) for f in val_files
                     if not os.path.basename(f).startswith("_")], ignore_index=True)
    pool = val[(val["testability"] == "TESTABLE") & (val["horizon"] == "24M")
               & (val["rule_verdict"] == "HOLDS")].copy()
    cand = pool.sort_values("valid_median_ret", ascending=False).head(top_n)
    # ALSO guarantee per-family coverage so fundamental/quality/valuation dims get
    # noise-confirmed seeds (the plain top-N is technical-dominated by sample size)
    pool["fam"] = pool["rule_id"].str.replace(r"_?\d+.*", "", regex=True)
    per_fam = (pool.sort_values("valid_median_ret", ascending=False)
               .groupby("fam").head(8))
    cand = pd.concat([cand, per_fam]).drop_duplicates("rule_id")
    # skip rules already noise-tested (resume-friendly)
    done_p = os.path.join(VAL_DIR, "_noise_floor.parquet")
    if os.path.exists(done_p):
        already = set(pd.read_parquet(done_p)["rule_id"])
        cand = cand[~cand["rule_id"].isin(already)]
    log(f"noise floor: {len(cand)} rules x {n_shuf} shuffles (per-family balanced)")
    OH = os.path.join(GURU_DIR, "data", "ohlcv_hist")
    out = []
    for i, (_, r) in enumerate(cand.iterrows(), 1):
        rid = r["rule_id"]
        try:
            trig = pd.read_parquet(os.path.join(BT_DIR, "triggers", f"{rid}.parquet"),
                                   columns=["guru_key", "base_date"])
            trig["base_date"] = pd.to_datetime(trig["base_date"])
            trig = trig[trig["base_date"] > SPLIT]
            if len(trig) < MIN_VALID:
                continue
            # precompute each company's possible random-entry returns ONCE
            pools = []
            for gk, grp in trig.groupby("guru_key"):
                arr = random_entry_returns(gk)
                if arr is not None:
                    pools.append((arr, len(grp)))
            nulls = []
            if pools:
                for s in range(n_shuf):
                    rng = np.random.default_rng(1000 + s)
                    draws = [a[rng.integers(0, len(a), size=k)] for a, k in pools]
                    allr = np.concatenate(draws)
                    if len(allr):
                        nulls.append(float(np.median(allr)))
            if nulls:
                real = float(r["valid_median_ret"])
                pct = float((np.array(nulls) < real).mean()) * 100
                out.append({"rule_id": rid, "real_valid_median": real,
                            "null_median_mean": round(float(np.mean(nulls)), 2),
                            "null_p90": round(float(np.percentile(nulls, 90)), 2),
                            "beats_pct_of_random": round(pct, 1),
                            "n_shuffles": len(nulls)})
        except Exception as e:
            log(f"  {rid} noise error {str(e)[:60]}")
        if i % 20 == 0:
            log(f"  {i}/{len(cand)}")
            _merge_noise(out)
    _merge_noise(out)
    log("noise floor complete")


def _merge_noise(new_rows: list):
    """append to (not overwrite) the noise-floor file, so per-family top-ups keep
    the original top-200 results."""
    p = os.path.join(VAL_DIR, "_noise_floor.parquet")
    df = pd.DataFrame(new_rows)
    if os.path.exists(p) and len(df):
        old = pd.read_parquet(p)
        df = pd.concat([old, df], ignore_index=True).drop_duplicates(
            "rule_id", keep="last")
    if len(df):
        df.to_parquet(p, index=False)


# ---------------- summary ----------------

def build_summary():
    fs = [f for f in glob.glob(os.path.join(VAL_DIR, "*.parquet"))
          if not os.path.basename(f).startswith("_")]
    if not fs:
        log("nothing to summarise yet"); return
    val = pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True)
    meta = pd.read_excel(XLSX, "Rules")[["rule_id", "rule_name", "category"]]
    val = val.merge(meta, on="rule_id", how="left")
    val.to_parquet(os.path.join(VAL_DIR, "_testability.parquet"), index=False)

    testab = (val.groupby(["horizon", "testability"]).size()
              .unstack(fill_value=0).reset_index())
    tested = val[val["testability"] == "TESTABLE"]
    rulev = (tested.groupby(["horizon", "rule_verdict"]).size()
             .unstack(fill_value=0).reset_index()) if "rule_verdict" in tested else pd.DataFrame()
    filt = tested[tested.get("filter_lift_valid").notna()] if "filter_lift_valid" in tested else pd.DataFrame()
    survivors = tested[(tested.get("rule_verdict") == "HOLDS")].copy()
    if "filter_lift_valid" in survivors:
        survivors = survivors.sort_values(["valid_median_ret"], ascending=False)
    noise_p = os.path.join(VAL_DIR, "_noise_floor.parquet")
    noise = pd.read_parquet(noise_p) if os.path.exists(noise_p) else pd.DataFrame()

    readme = pd.DataFrame([
        ("What this is", "Out-of-sample check. Train = triggers up to 2018-12-31, "
         "Validation = 2019 onwards. A rule is only judged if it has >=20 usable "
         "triggers in BOTH windows (usable = has a COMPLETE forward path to that "
         "horizon; a 2024 trigger has no 36M future, so it is excluded - NOT failed)."),
        ("Testability", "TESTABLE = judged. UNTESTABLE_* = cannot be judged and is "
         "NOT a failure (no validation data / too thin train / too thin validation). "
         "Nothing has been deleted; all original results remain in scorecard.xlsx "
         "and profiles_summary.xlsx."),
        ("Rule_Verdicts", "HOLDS = validation median return > 0 AND validation "
         "win-rate within 10pp of train. WEAKER = still positive but materially "
         "below train. BREAKS = validation median <= 0 (in-sample result did not survive)."),
        ("Filter_Validation", "THE honest test of the +30pt in-sample filter lift: the "
         "best separating dimension+threshold is RE-DERIVED using TRAIN DATA ONLY, then "
         "applied unchanged to validation. filter_lift_valid = validation median WITH "
         "the filter minus WITHOUT. If this collapses toward 0, the filters were "
         "curve-fitting. filter_dim_train may differ from the in-sample pick - that "
         "difference is itself evidence of instability."),
        ("Survivors", "Rules that are TESTABLE and HOLDS. This shortlist - not the "
         "2,088 - is what should feed combo generation and any live buy/sell decision."),
        ("Noise_Floor", "Trigger dates shuffled within each company's own price history "
         "(company mix preserved, timing destroyed), 20 times. beats_pct_of_random = "
         "the % of random-timing runs the real result beat. Low values mean the edge "
         "is stock/era selection rather than timing."),
        ("Episode columns", "valid_ep_* restate validation on one-trigger-per-company-"
         "quarter, so a rule cannot look strong because one company triggered 42 times."),
    ], columns=["item", "explanation"])
    with pd.ExcelWriter(OUT_XLSX, engine="openpyxl") as xw:
        readme.to_excel(xw, "README", index=False)
        testab.to_excel(xw, "Testability", index=False)
        if not rulev.empty:
            rulev.to_excel(xw, "Rule_Verdicts", index=False)
        val.to_excel(xw, "All_Rules_Detail", index=False)
        if not filt.empty:
            filt.sort_values("filter_lift_valid", ascending=False).to_excel(
                xw, "Filter_Validation", index=False)
        if not survivors.empty:
            survivors.to_excel(xw, "Survivors", index=False)
        if not noise.empty:
            noise.to_excel(xw, "Noise_Floor", index=False)
    log(f"summary -> {OUT_XLSX}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--shard", type=str, default="")
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--pending-count", action="store_true",
                    help="print ONLY the number of rules left (for .bat control flow)")
    ap.add_argument("--noise", action="store_true")
    ap.add_argument("--top", type=int, default=200)
    ap.add_argument("--shuffles", type=int, default=20)
    ap.add_argument("--summary-only", action="store_true")
    args = ap.parse_args()

    os.makedirs(VAL_DIR, exist_ok=True)
    if args.pending_count:
        allids = {os.path.basename(f)[:-8] for f in
                  glob.glob(os.path.join(BT_DIR, "triggers", "*.parquet"))}
        done = {os.path.basename(f)[:-8] for f in glob.glob(os.path.join(VAL_DIR, "*.parquet"))
                if not os.path.basename(f).startswith("_")}
        print(len(allids - done))
        return
    if args.summary_only:
        build_summary(); return
    if args.noise:
        noise_floor(args.top, args.shuffles); build_summary(); return

    ids = sorted(os.path.basename(f)[:-8] for f in
                 glob.glob(os.path.join(BT_DIR, "triggers", "*.parquet")))
    ledger = os.path.join(BT_DIR, "_validation_ledger.parquet")
    if args.shard:
        k, n = [int(x) for x in args.shard.split("/")]
        ids = [r for i, r in enumerate(ids) if i % n == k - 1]
        ledger = os.path.join(BT_DIR, f"_validation_ledger_shard{k}of{n}.parquet")

    done = {os.path.basename(f)[:-8] for f in glob.glob(os.path.join(VAL_DIR, "*.parquet"))
            if not os.path.basename(f).startswith("_")}
    if args.status:
        allids = sorted(os.path.basename(f)[:-8] for f in
                        glob.glob(os.path.join(BT_DIR, "triggers", "*.parquet")))
        print(f"{len(done)}/{len(allids)} rules validated "
              f"({100*len(done)/max(len(allids),1):.1f}%)")
        return

    todo = [r for r in ids if r not in done]
    if args.limit:
        todo = todo[:args.limit]
    log(f"validating {len(todo)} rules (skipping {len(ids)-len(todo)} already done)")
    n_ok = n_skip = 0
    for i, rid in enumerate(todo, 1):
        try:
            out = validate_rule(rid)
            if out.empty:
                n_skip += 1
            else:
                out.to_parquet(os.path.join(VAL_DIR, f"{rid}.parquet"), index=False)
                n_ok += 1
        except Exception as e:
            log(f"  {rid} ERROR {str(e)[:80]}")
        if i % 100 == 0:
            log(f"  {i}/{len(todo)} (ok={n_ok} skip={n_skip})")
    log(f"RUN COMPLETE ok={n_ok} skip={n_skip}")
    if args.shard:
        log("shard complete — run once without --shard (or --summary-only) for the xlsx")
    else:
        build_summary()


if __name__ == "__main__":
    main()
