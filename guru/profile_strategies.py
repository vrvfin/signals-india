r"""
Feature B — profile_strategies.py  (Project Guru, RESUMABLE)

B1  Leaderboards: best rules per horizon {1,3,6,12,24,36}M by median return and
    by success probability — straight from the score fragments, nothing filtered
    (n_episodes shown so the reader judges confidence; user's no-drop rule).

B2  Winner/loser anchor-date profiling, for every rule with
    success_prob_pos > 50% at its best horizon <= 36M:
      * split that rule's triggers into WINNERS vs LOSERS under BOTH definitions
        (user decision): (a) top vs bottom return tercile, (b) >=2x vs negative
      * for each trigger, build the ANCHOR-DATE PROFILE (~25 dims as-of entry:
        size, liquidity, valuation, quality, technicals, ownership, regime)
      * per dimension: median(winners) vs median(losers) + AUC separation score
      * suggest the top avoid-worst filter and measure its LIFT (median return &
        win-rate with vs without the filter)
    Sampling: rules with >8,000 triggers are profiled on an 8,000-trigger sample
    stratified by year (separation statistics are stable at this n; noted in output).

Outputs
    backtest/profiles/<rule>.parquet          per-dim separation table (both splits)
    backtest/profiles_summary.xlsx            README + Leaderboards + Top_Separators + Filter_Lift
    backtest/_profiles_ledger.parquet         resume ledger

Usage
    python guru/profile_strategies.py --leaderboards-only
    python guru/profile_strategies.py --limit 10
    python guru/profile_strategies.py                 # full, resumes
    python guru/profile_strategies.py --status
"""
from __future__ import annotations

import argparse
import glob
import os
from datetime import datetime
from functools import lru_cache

import numpy as np
import pandas as pd

GURU_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(GURU_DIR, "data")
BT_DIR = os.path.join(GURU_DIR, "backtest")
PROF_DIR = os.path.join(BT_DIR, "profiles")
LIFT_DIR = os.path.join(BT_DIR, "lifts")            # per-rule filter-lift fragment
TPROF_DIR = os.path.join(BT_DIR, "trigger_profiles")  # per-trigger anchor profiles
LEDGER = os.path.join(BT_DIR, "_profiles_ledger.parquet")
OUT_XLSX = os.path.join(BT_DIR, "profiles_summary.xlsx")

LB_HORIZONS = ["1M", "3M", "6M", "12M", "24M", "36M"]
SAMPLE_CAP = 8000
MIN_CLASS_N = 8          # need at least this many winners AND losers to profile

TECH_DIMS = ["pct_from_52w_high", "price_vs_ma_200", "rsi_14", "volatility_20d_pct",
             "beta_1y", "momentum_score", "rel_strength_vs_index_pct",
             "volume_ratio_20d_avg", "days_since_listing", "price_vs_ma_50",
             "drawdown_from_high_pct", "price_return_12m_pct", "price_return_3m_pct"]
FUND_DIMS = ["market_cap_cr", "pe_ratio", "pe_percentile", "ps_ratio",
             "sales_yoy_pct", "profit_yoy_pct", "net_margin_pct",
             "roe_pct", "roce_pct", "debt_to_equity",
             "promoter_holding_pct", "fii_holding_pct"]
REGIME_DIMS = ["nifty500_above_200dma", "india_vix"]
EXTRA_DIMS = ["turnover_20d_cr", "close_price"]
ALL_DIMS = TECH_DIMS + FUND_DIMS + REGIME_DIMS + EXTRA_DIMS


def log(m): print(f"[{datetime.now().strftime('%H:%M:%S')}] {m}", flush=True)


# ---------------- data access (cached per company) ----------------

@lru_cache(maxsize=96)
def tech_f(gk):
    p = os.path.join(DATA_DIR, "metrics", "technical", f"{gk}.parquet")
    if not os.path.exists(p):
        return None
    df = pd.read_parquet(p)
    keep = ["date", "close"] + [c for c in TECH_DIMS if c in df.columns]
    return df[keep].sort_values("date").reset_index(drop=True)


@lru_cache(maxsize=256)
def fund_f(gk):
    p = os.path.join(DATA_DIR, "metrics", "fundamental", f"{gk}.parquet")
    q = os.path.join(DATA_DIR, "metrics", "quarterly_unified", f"{gk}.parquet")
    frames = []
    if os.path.exists(q):
        d = pd.read_parquet(q)
        d = d[d["announcement_date"].notna()]
        keep = ["announcement_date"] + [c for c in FUND_DIMS if c in d.columns]
        frames.append(d[keep].rename(columns={"announcement_date": "asof"}))
    if os.path.exists(p):
        d = pd.read_parquet(p)
        d = d[d["announcement_date"].notna()]
        keep = ["announcement_date"] + [c for c in FUND_DIMS if c in d.columns]
        frames.append(d[keep].rename(columns={"announcement_date": "asof"}))
    if not frames:
        return None
    out = pd.concat(frames, ignore_index=True)
    out["asof"] = pd.to_datetime(out["asof"]).dt.normalize()
    return out.sort_values("asof").reset_index(drop=True)


@lru_cache(maxsize=256)
def ohlcv_f(gk):
    p = os.path.join(DATA_DIR, "ohlcv_hist", f"{gk}.parquet")
    if not os.path.exists(p):
        return None
    d = pd.read_parquet(p, columns=["date", "close", "volume"]).sort_values("date")
    d["turnover_20d_cr"] = (d["close"] * d["volume"]).rolling(20).mean() / 1e7
    return d[["date", "turnover_20d_cr"]].reset_index(drop=True)


_REG = None
def regime_f():
    global _REG
    if _REG is None:
        _REG = pd.read_parquet(os.path.join(DATA_DIR, "metrics", "regime.parquet"),
                               columns=["date"] + REGIME_DIMS).sort_values("date")
    return _REG


def asof_row(df, datecol, when):
    if df is None:
        return None
    i = df[datecol].searchsorted(when, side="right") - 1
    return df.iloc[i] if i >= 0 else None


def profile_trigger(gk, base_date):
    out = {}
    t = tech_f(gk)
    r = asof_row(t, "date", base_date) if t is not None else None
    if r is not None:
        for c in TECH_DIMS:
            if c in r.index:
                out[c] = r[c]
        out["close_price"] = r.get("close")
    f = fund_f(gk)
    r = asof_row(f, "asof", pd.Timestamp(base_date).normalize()) if f is not None else None
    if r is not None:
        for c in FUND_DIMS:
            if c in r.index and not pd.isna(r[c]):
                out[c] = r[c]
    o = ohlcv_f(gk)
    r = asof_row(o, "date", base_date) if o is not None else None
    if r is not None:
        out["turnover_20d_cr"] = r["turnover_20d_cr"]
    r = asof_row(regime_f(), "date", base_date)
    if r is not None:
        for c in REGIME_DIMS:
            out[c] = r[c]
    return out


# ---------------- separation stats ----------------

def auc(w: pd.Series, l: pd.Series) -> float:
    """rank-based AUC: P(winner value > loser value). 0.5 = no separation."""
    x = pd.concat([w, l]).rank()
    rw = x.iloc[:len(w)]
    u = rw.sum() - len(w) * (len(w) + 1) / 2
    return float(u / (len(w) * len(l)))


def profile_rule(rid: str, horizon: str) -> tuple[pd.DataFrame, dict]:
    trig = pd.read_parquet(os.path.join(BT_DIR, "triggers", f"{rid}.parquet"))
    paths = pd.read_parquet(os.path.join(BT_DIR, "paths", f"{rid}.parquet"))
    hm = int(horizon[:-1])
    p = paths[paths["month"] == hm][["trigger_id", "ret_pct"]]
    d = trig.merge(p, on="trigger_id", how="inner")
    if len(d) < MIN_CLASS_N * 2:
        return pd.DataFrame(), {}
    sampled = False
    if len(d) > SAMPLE_CAP:
        d["_yr"] = pd.to_datetime(d["base_date"]).dt.year
        d = (d.groupby("_yr", group_keys=False)
              .apply(lambda g: g.sample(max(1, int(SAMPLE_CAP * len(g) / len(d))),
                                        random_state=7)))
        sampled = True
    profs = []
    for _, r in d.iterrows():
        pr = profile_trigger(r["guru_key"], pd.Timestamp(r["base_date"]))
        pr["trigger_id"] = r["trigger_id"]; pr["ret_pct"] = r["ret_pct"]
        profs.append(pr)
    P = pd.DataFrame(profs)
    if P.empty or "ret_pct" not in P.columns:
        return pd.DataFrame(), {}
    splits = {}
    q1, q2 = P["ret_pct"].quantile([1 / 3, 2 / 3])
    splits["tercile"] = (P["ret_pct"] >= q2, P["ret_pct"] <= q1)
    splits["absolute"] = (P["ret_pct"] >= 100, P["ret_pct"] < 0)
    rows = []
    for sname, (wmask, lmask) in splits.items():
        W, L = P[wmask], P[lmask]
        if len(W) < MIN_CLASS_N or len(L) < MIN_CLASS_N:
            continue
        for dim in ALL_DIMS:
            if dim not in P.columns:
                continue
            w = W[dim].dropna().astype(float)
            l = L[dim].dropna().astype(float)
            if len(w) < MIN_CLASS_N or len(l) < MIN_CLASS_N:
                continue
            a = auc(w, l)
            rows.append({"rule_id": rid, "horizon": horizon, "split": sname,
                         "dim": dim, "n_win": len(w), "n_loss": len(l),
                         "median_winner": round(float(w.median()), 3),
                         "median_loser": round(float(l.median()), 3),
                         "auc": round(a, 4),
                         "separation": round(abs(a - 0.5) * 2, 4)})
    # persist the per-trigger anchor profiles: expensive to rebuild, and reused by
    # the filter/lift math, Feature C and the buy/hold/sell layer.
    os.makedirs(TPROF_DIR, exist_ok=True)
    P.assign(rule_id=rid, horizon=horizon).to_parquet(
        os.path.join(TPROF_DIR, f"{rid}.parquet"), index=False)

    sep = pd.DataFrame(rows)
    lift = {}
    if not sep.empty:
        s = sep[sep["split"] == "tercile"]
        if not s.empty:
            top = s.sort_values("separation", ascending=False).iloc[0]
            dim = top["dim"]
            keep_high = top["auc"] > 0.5
            vals = P[dim].astype(float)
            thr = float(vals.dropna().median())
            mask = (vals >= thr) if keep_high else (vals <= thr)
            f = P[mask.fillna(False)]
            if len(f) >= MIN_CLASS_N:
                lift = {"rule_id": rid, "horizon": horizon, "filter_dim": dim,
                        "filter_dir": ">=" if keep_high else "<=",
                        "filter_thr": round(thr, 3),
                        "auc": float(top["auc"]),
                        "n_before": len(P), "n_after": int(len(f)),
                        "median_ret_before": round(float(P["ret_pct"].median()), 2),
                        "median_ret_after": round(float(f["ret_pct"].median()), 2),
                        "winrate_before": round(float((P["ret_pct"] > 0).mean()) * 100, 1),
                        "winrate_after": round(float((f["ret_pct"] > 0).mean()) * 100, 1),
                        "sampled": sampled}
    return sep, lift


# ---------------- B1 leaderboards ----------------

def load_scores() -> pd.DataFrame:
    frags = glob.glob(os.path.join(BT_DIR, "scores", "*.parquet"))
    sc = pd.concat([pd.read_parquet(f) for f in frags], ignore_index=True)
    meta = pd.read_excel(os.path.join(os.path.dirname(GURU_DIR), "Project_Guru",
                                      "rule_template.xlsx"), "Rules")
    return sc.merge(meta[["rule_id", "rule_name", "category"]], on="rule_id", how="left")


def leaderboards(sc: pd.DataFrame) -> dict[str, pd.DataFrame]:
    out = {}
    cols = ["rule_id", "rule_name", "category", "n_episodes", "n_companies",
            "median_return_pct", "success_prob_pos", "success_prob_2x",
            "median_max_drawdown_pct", "sustain_ratio_median"]
    for h in LB_HORIZONS:
        d = sc[(sc["horizon"] == h) & (sc["n_triggers"] > 0)]
        out[f"Top_{h}_med"] = d.sort_values(
            "median_return_pct", ascending=False)[cols].head(50)
        out[f"Top_{h}_win"] = d.sort_values(
            ["success_prob_pos", "median_return_pct"], ascending=False)[cols].head(50)
    return out


# ---------------- main ----------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--leaderboards-only", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--retry-errors", action="store_true")
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--shard", type=str, default="",
                    help="K/N parallel partition (own ledger per shard)")
    args = ap.parse_args()

    sc = load_scores()
    small = sc[sc["horizon"].isin(LB_HORIZONS) & (sc["n_triggers"] > 0)]
    best = (small.sort_values("success_prob_pos", ascending=False)
            .drop_duplicates("rule_id"))
    sel = best[best["success_prob_pos"] > 50][["rule_id", "horizon"]]
    log(f"rules with >50% win-rate at best <=36M horizon: {len(sel)}")

    ledger_path = LEDGER
    if args.shard:
        k, n = [int(x) for x in args.shard.split("/")]
        sel = sel.sort_values("rule_id").reset_index(drop=True)
        sel = sel[sel.index % n == k - 1]
        ledger_path = os.path.join(BT_DIR, f"_profiles_ledger_shard{k}of{n}.parquet")

    # union of every profiles ledger: skip rules finished anywhere
    done_any = set()
    for f in glob.glob(os.path.join(BT_DIR, "_profiles_ledger*.parquet")):
        l = pd.read_parquet(f)
        done_any |= set(l.loc[l["status"].isin(["done", "too_small"]), "rule_id"])

    if os.path.exists(ledger_path):
        led = pd.read_parquet(ledger_path)
        new = sel[~sel["rule_id"].isin(led["rule_id"])]
        if not new.empty:
            add = new.copy(); add["status"] = "pending"; add["error"] = ""
            led = pd.concat([led, add], ignore_index=True)
    else:
        led = sel.copy(); led["status"] = "pending"; led["error"] = ""
    led.loc[led["status"].eq("pending") & led["rule_id"].isin(done_any),
            "status"] = "done_elsewhere"

    if args.status:
        allled = []
        for f in glob.glob(os.path.join(BT_DIR, "_profiles_ledger*.parquet")):
            allled.append(pd.read_parquet(f))
        a = pd.concat(allled)
        # explicit priority: a rule is FINISHED if ANY ledger says so. Alphabetical
        # sorting wrongly let a stale 'pending' outrank a shard's 'too_small'.
        prio = {"done": 0, "too_small": 1, "error": 2, "pending": 3,
                "done_elsewhere": 4}
        a["_p"] = a["status"].map(prio).fillna(3)
        a = a.sort_values("_p").drop_duplicates("rule_id", keep="first")
        a = a[a["status"] != "done_elsewhere"]
        vc = a["status"].value_counts().to_dict()
        fin = int(a["status"].isin(["done", "too_small"]).sum())
        print(f"{fin}/{len(a)} finished ({100*fin/max(len(a),1):.1f}%)  {vc}")
        return

    lbs = leaderboards(sc)

    todo_mask = led["status"].eq("pending")
    if args.retry_errors:
        todo_mask |= led["status"].eq("error")
    todo = led[todo_mask]
    if args.limit:
        todo = todo.head(args.limit)
    if args.leaderboards_only:
        todo = todo.head(0)
    log(f"profiling this run: {len(todo)} rules")

    os.makedirs(PROF_DIR, exist_ok=True)
    lifts = []
    for i, (li, r) in enumerate(todo.iterrows(), 1):
        try:
            sep, lift = profile_rule(r["rule_id"], r["horizon"])
            if sep.empty:
                led.at[li, "status"] = "too_small"
            else:
                sep.to_parquet(os.path.join(PROF_DIR, f"{r['rule_id']}.parquet"),
                               index=False)
                led.at[li, "status"] = "done"
                if lift:
                    lifts.append(lift)
                    # PERSIST per-rule so the summary can always be rebuilt after a
                    # resumed/parallel run (previously lift was in-memory only and
                    # vanished when the rebuild pass processed 0 rules).
                    os.makedirs(LIFT_DIR, exist_ok=True)
                    pd.DataFrame([lift]).to_parquet(
                        os.path.join(LIFT_DIR, f"{r['rule_id']}.parquet"), index=False)
            led.at[li, "error"] = ""
        except Exception as e:
            led.at[li, "status"] = "error"; led.at[li, "error"] = str(e)[:200]
        if i % 10 == 0:
            led.to_parquet(ledger_path, index=False)
            log(f"  {i}/{len(todo)} profiled")
    led.to_parquet(ledger_path, index=False)

    if args.shard:
        # workers must not collide writing the shared summary workbook; rebuild
        # it once afterwards by running this script WITHOUT --shard (fast).
        log("shard complete — run once without --shard to rebuild the summary xlsx")
        return

    # ------- summary workbook -------
    sep_files = glob.glob(os.path.join(PROF_DIR, "*.parquet"))
    top_seps = []
    for f in sep_files:
        s = pd.read_parquet(f)
        s = s[s["split"] == "tercile"].sort_values("separation", ascending=False).head(5)
        top_seps.append(s)
    seps = pd.concat(top_seps, ignore_index=True) if top_seps else pd.DataFrame()
    # read persisted lift fragments (this run's + every previous/parallel run's)
    lift_files = glob.glob(os.path.join(LIFT_DIR, "*.parquet"))
    lifts_df = (pd.concat([pd.read_parquet(f) for f in lift_files], ignore_index=True)
                .drop_duplicates("rule_id", keep="last")
                if lift_files else pd.DataFrame(lifts))
    readme = pd.DataFrame([
        ("Leaderboards", "Top-50 rules per horizon (1/3/6/12/24/36M), ranked by median "
         "return (Top_<H>_med) and separately by win-rate (Top_<H>_win). Nothing "
         "filtered; judge n_episodes yourself."),
        ("Top_Separators", "For every >50%-win-rate rule: the 5 anchor-date dimensions "
         "that best separate its winners from its losers (tercile split). auc>0.5 = "
         "winners have HIGHER values of the dim; separation = |auc-0.5|*2 (0=none, 1=perfect)."),
        ("Filter_Lift", "The single strongest separator turned into an avoid-worst filter "
         "(keep triggers on the winner side of the median). Shows median return and "
         "win-rate BEFORE vs AFTER the filter, and how many triggers survive. 'sampled' "
         "= rule profiled on an 8,000-trigger stratified sample."),
        ("Method", "Winner/loser split computed BOTH ways (top-vs-bottom tercile AND "
         ">=2x-vs-negative); per-rule parquet in backtest/profiles/ holds both. Profiles "
         "are as-of the trigger's base_date (no look-ahead). AUC is rank-based."),
    ], columns=["sheet", "explanation"])
    with pd.ExcelWriter(OUT_XLSX, engine="openpyxl") as xw:
        readme.to_excel(xw, "README", index=False)
        for name, df in lbs.items():
            df.to_excel(xw, name[:31], index=False)
        if not seps.empty:
            meta = sc[["rule_id", "rule_name"]].drop_duplicates()
            seps.merge(meta, on="rule_id", how="left").to_excel(
                xw, "Top_Separators", index=False)
        if not lifts_df.empty:
            lifts_df.to_excel(xw, "Filter_Lift", index=False)
    log(f"summary -> {OUT_XLSX}")
    print(led["status"].value_counts().to_dict())


if __name__ == "__main__":
    main()
