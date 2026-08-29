r"""
Feature D — build_consensus.py  (Project Guru)

Tests: "when MULTIPLE validated rules fire on the SAME stock around the SAME
time, does that agreement predict a better outcome than one rule alone?" This
is EMPIRICAL consensus (rules that actually co-fired in real history), not
Feature C's synthetic clause-crossing — much cheaper, and a direct test of
whether stacking real signals helps.

Method
------
1. Pool every trigger from every VALIDATED survivor rule (HOLDS out-of-sample):
   (rule_id, guru_key, base_date).
2. Cluster = (stock, calendar quarter). Two rules firing on the same stock in
   the same ~91-day quarter count as "agreeing" (approximates the 90-day window
   without the chaining problem a fixed rolling window has when one rule alone
   re-fires every quarter for years).
3. entry_date = earliest trigger date in the cluster (no look-ahead); n_rules =
   distinct rules that fired in it.
4. Forward return computed from the real price series at each horizon (this is
   an endpoint return, not a full monthly path — kept lean on purpose).
5. Outputs:
   - Consensus_Buckets: n_rules-agreeing bucket x horizon -> win-rate, median
     return (THE hypothesis test: does the bucket average rise with agreement?)
   - Company_Reliability: per company, at high agreement (>=3 rules), how often
     did that agreement actually pay off vs not (win-rate + median return) —
     "which companies' consensus can be trusted, which can't".
   - Rule_Pair_Combos: which PAIRS of real rules co-firing most often precede a
     win — an empirical combo table.
   - Example_Clusters: real instances for a spot-check.

Additive: reads existing triggers/validation; writes only guru/CONSENSUS_ANALYSIS.xlsx.

Usage: python guru/build_consensus.py
"""
from __future__ import annotations
import glob, os, itertools
from datetime import datetime
import numpy as np, pandas as pd

GURU = os.path.dirname(os.path.abspath(__file__))
BT = os.path.join(GURU, "backtest")
DATA = os.path.join(GURU, "data")
OHLCV = os.path.join(DATA, "ohlcv_hist")
OUT = os.path.join(GURU, "CONSENSUS_ANALYSIS.xlsx")
HZ = [3, 6, 12, 24, 36]
MIN_PAIR_COUNT = 15
BUCKETS = [(1, 1, "1 rule"), (2, 2, "2 rules"), (3, 3, "3 rules"),
           (4, 5, "4-5 rules"), (6, 10, "6-10 rules"), (11, 9999, "11+ rules")]


def log(m): print(f"[{datetime.now().strftime('%H:%M:%S')}] {m}", flush=True)


def survivors() -> list[str]:
    fs = [f for f in glob.glob(os.path.join(BT, "validation", "*.parquet"))
          if not os.path.basename(f).startswith("_")]
    v = pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True)
    return sorted(v[(v.testability == "TESTABLE") & (v.rule_verdict == "HOLDS")]
                  .rule_id.unique())


def pool_triggers(rules: list[str]) -> pd.DataFrame:
    frames = []
    for i, rid in enumerate(rules, 1):
        tf = os.path.join(BT, "triggers", f"{rid}.parquet")
        if not os.path.exists(tf):
            continue
        t = pd.read_parquet(tf, columns=["guru_key", "base_date"])
        t["rule_id"] = rid
        frames.append(t)
        if i % 200 == 0:
            log(f"  pooled {i}/{len(rules)} rule files")
    return pd.concat(frames, ignore_index=True)


def rule_family(rid: str) -> str:
    """Collapse near-duplicate rules to their FAMILY. Critical: with 830 rules,
    many are the same idea at different thresholds (sales_yoy>=25/30/40), so a raw
    rule count makes '11+ agreeing' the default state (measured: ~10 rules fire on
    the average stock-quarter). Counting distinct FAMILIES measures real
    diversity of evidence instead of redundancy."""
    import re
    return re.sub(r"_?\d+.*$", "", rid)


def build_clusters(pool: pd.DataFrame) -> pd.DataFrame:
    pool = pool.copy()
    pool["base_date"] = pd.to_datetime(pool["base_date"])
    pool["quarter"] = pool["base_date"].dt.to_period("Q")
    pool["family"] = pool["rule_id"].map(rule_family)
    g = pool.groupby(["guru_key", "quarter"])
    clusters = g.agg(entry_date=("base_date", "min"),
                     n_rules=("rule_id", "nunique"),
                     n_families=("family", "nunique"),
                     families=("family", lambda s: sorted(set(s))),
                     rules=("rule_id", lambda s: sorted(set(s)))).reset_index()
    return clusters


def forward_returns(clusters: pd.DataFrame) -> pd.DataFrame:
    out_rows = []
    for gk, grp in clusters.groupby("guru_key"):
        fp = os.path.join(OHLCV, f"{gk}.parquet")
        if not os.path.exists(fp):
            continue
        px = pd.read_parquet(fp, columns=["date", "open", "close", "volume"]).sort_values("date")
        dates = px["date"].values
        opens = px["open"].values
        closes = px["close"].values
        vols = px["volume"].values
        for _, r in grp.iterrows():
            i = int(np.searchsorted(dates, np.datetime64(r["entry_date"]), side="right"))
            if i >= len(px):
                continue
            entry = float(opens[i]) if opens[i] > 0 else float(closes[i])
            if entry <= 0:
                continue
            # liquidity proxy: avg daily volume over the ~3 months after entry —
            # lets us tell apart a genuine flat outcome from a stale/no-trade stock
            j3 = min(i + 63, len(px) - 1)
            vol_avg_3m = float(np.nanmean(vols[i:j3 + 1])) if j3 > i else float(vols[i])
            row = {"guru_key": gk, "quarter": str(r["quarter"]),
                   "entry_date": r["entry_date"], "n_rules": r["n_rules"],
                   "n_families": r["n_families"], "families": ",".join(r["families"]),
                   "rules": ",".join(r["rules"]), "entry_price": round(entry, 2),
                   "vol_avg_3m": round(vol_avg_3m, 0)}
            for H in HZ:
                j = i + H * 21
                row[f"ret_{H}m"] = (float(closes[j]) / entry - 1) * 100 if j < len(px) else np.nan
            out_rows.append(row)
    return pd.DataFrame(out_rows)


def bucket_label(n: int) -> str:
    for lo, hi, lab in BUCKETS:
        if lo <= n <= hi:
            return lab
    return "11+ rules"


CACHE = os.path.join(BT, "consensus_clusters.parquet")


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--use-cache", action="store_true",
                    help="skip the ~30min pooling/price pass and reuse the "
                         "already-persisted backtest/consensus_clusters.parquet")
    args = ap.parse_args()

    if args.use_cache and os.path.exists(CACHE):
        log(f"using cached cluster table: {CACHE}")
        fr = pd.read_parquet(CACHE)
        fr["families"] = fr["families"].apply(lambda s: s.split(",") if isinstance(s, str) else [])
        fr["rules"] = fr["rules"].apply(lambda s: s.split(",") if isinstance(s, str) else [])
    else:
        surv = survivors()
        log(f"validated survivor rules: {len(surv)}")
        pool = pool_triggers(surv)
        log(f"pooled triggers: {len(pool):,} across {pool.guru_key.nunique()} stocks")
        clusters = build_clusters(pool)
        log(f"consensus clusters (stock x quarter): {len(clusters):,} "
            f"| avg rules/cluster {clusters.n_rules.mean():.1f} | max {clusters.n_rules.max()}")
        fr = forward_returns(clusters)
        log(f"clusters with computable returns: {len(fr):,}")
        # persist the raw per-cluster table so any future re-slice (liquidity
        # cuts, different bucket definitions) is instant instead of a ~30min re-run
        fr.to_parquet(CACHE, index=False)
    # BUCKET BY FAMILY DIVERSITY (not raw rule count — see rule_family docstring)
    FAM_BUCKETS = [(1, 1, "1 family"), (2, 2, "2 families"), (3, 3, "3 families"),
                   (4, 4, "4 families"), (5, 6, "5-6 families"),
                   (7, 999, "7+ families")]

    def fam_bucket(n):
        for lo, hi, lab in FAM_BUCKETS:
            if lo <= n <= hi:
                return lab
        return "7+ families"

    fr["bucket"] = fr["n_families"].apply(fam_bucket)
    bucket_order = [b[2] for b in FAM_BUCKETS]

    # ---- 1. THE HYPOTHESIS TEST: does DIVERSE agreement predict better outcomes? ----
    rows = []
    for H in HZ:
        col = f"ret_{H}m"
        d = fr[fr[col].notna()]
        for b in bucket_order:
            sub = d[d.bucket == b]
            if len(sub) < 10:
                continue
            rows.append({"horizon": f"{H}M", "n_families_agreeing": b,
                        "n_clusters": len(sub),
                        "avg_n_rules": round(float(sub.n_rules.mean()), 1),
                        "median_return_pct": round(float(sub[col].median()), 1),
                        "mean_return_pct": round(float(sub[col].mean()), 1),
                        "winrate_pct": round(float((sub[col] > 0).mean()) * 100, 1),
                        "rate_2x_pct": round(float((sub[col] >= 100).mean()) * 100, 1),
                        "pct_flat_zero_ret": round(float((sub[col].abs() < 0.01).mean()) * 100, 1)})
    buckets_df = pd.DataFrame(rows)

    # LIQUIDITY-FILTERED version: excludes thinly-traded stocks (avg vol < 10,000
    # shares/day over the 3 months after entry), so the comparison isn't confounded
    # by stale/illiquid names dominating the low-agreement buckets.
    LIQ_MIN = 10000
    rows2 = []
    liq = fr[fr["vol_avg_3m"] >= LIQ_MIN]
    for H in HZ:
        col = f"ret_{H}m"
        d = liq[liq[col].notna()]
        for b in bucket_order:
            sub = d[d.bucket == b]
            if len(sub) < 10:
                continue
            rows2.append({"horizon": f"{H}M", "n_families_agreeing": b,
                         "n_clusters": len(sub),
                         "median_return_pct": round(float(sub[col].median()), 1),
                         "winrate_pct": round(float((sub[col] > 0).mean()) * 100, 1),
                         "rate_2x_pct": round(float((sub[col] >= 100).mean()) * 100, 1)})
    buckets_liquid_df = pd.DataFrame(rows2)
    log(f"liquidity-filtered clusters (vol>={LIQ_MIN}/day): {len(liq):,} of {len(fr):,}")

    # ---- 2. per-company reliability at HIGH consensus (>=3 rules), 24M ----
    H = 24
    col = f"ret_{H}m"
    hi = fr[(fr.n_families >= 4) & fr[col].notna()]
    uni = pd.read_parquet(os.path.join(DATA, "universe_hist.parquet"),
                          columns=["guru_key", "name", "nse_symbol"]).set_index("guru_key")
    comp_rows = []
    for gk, g in hi.groupby("guru_key"):
        if len(g) < 2:
            continue
        nm = uni.loc[gk, "name"] if gk in uni.index else gk
        comp_rows.append({"name": nm, "nse_symbol": uni.loc[gk, "nse_symbol"] if gk in uni.index else None,
                          "n_high_consensus_events": len(g),
                          "avg_n_rules_agreeing": round(float(g.n_rules.mean()), 1),
                          "median_return_pct": round(float(g[col].median()), 1),
                          "winrate_pct": round(float((g[col] > 0).mean()) * 100, 1),
                          "pct_declined_despite_consensus": round(float((g[col] < 0).mean()) * 100, 1)})
    company_df = pd.DataFrame(comp_rows)
    best_companies = company_df.sort_values(["winrate_pct", "median_return_pct"],
                                             ascending=False).head(150)
    worst_companies = company_df.sort_values(["pct_declined_despite_consensus", ],
                                              ascending=False).head(150)

    # ---- 3. rule-PAIR combos: which pairs co-firing predict a win ----
    pair_rows = []
    multi = fr[(fr.n_rules >= 2) & fr[col].notna()]
    from collections import defaultdict
    pair_rets = defaultdict(list)
    for _, r in multi.iterrows():
        for a, b in itertools.combinations(r["rules"], 2):
            pair_rets[(a, b)].append(r[col])
    for (a, b), rets in pair_rets.items():
        if len(rets) >= MIN_PAIR_COUNT:
            arr = np.array(rets)
            pair_rows.append({"rule_a": a, "rule_b": b, "n_cooccur": len(arr),
                              "median_return_pct": round(float(np.median(arr)), 1),
                              "winrate_pct": round(float((arr > 0).mean()) * 100, 1)})
    pairs_df = pd.DataFrame(pair_rows).sort_values("winrate_pct", ascending=False)

    # ---- 4. example clusters for spot-checking ----
    ex = fr[fr.n_rules >= 4].copy()
    ex["name"] = ex["guru_key"].map(uni["name"])
    ex["rules_str"] = ex["rules"].apply(lambda l: ", ".join(l[:8]) + (" …" if len(l) > 8 else ""))
    ex_out = ex.sort_values(col, ascending=False)[
        ["name", "quarter", "entry_date", "n_rules", "rules_str",
         "ret_3m", "ret_6m", "ret_12m", "ret_24m", "ret_36m"]].head(300)

    readme = pd.DataFrame([
        ("PURPOSE", "Tests whether MULTIPLE already-validated rules agreeing on the "
         "same stock, at the same time, predicts a better outcome than one rule "
         "alone — using REAL historical co-firing (not synthetic combos)."),
        ("METHOD", "Pooled every trigger from all out-of-sample-validated survivor "
         "rules. Grouped by (stock, calendar quarter) as the 'agreement window' "
         "(~91 days, avoids one repeating rule creating one giant multi-year "
         "cluster). n_rules = distinct rules firing in that stock-quarter. Entry = "
         "earliest trigger date in the cluster; returns computed from real prices, "
         "endpoint-only (not a full path/drawdown, to keep this analysis lean)."),
        ("Consensus_Buckets", "THE HEADLINE ANSWER: median return and win-rate by "
         "how many distinct rule FAMILIES agreed, per horizon. IMPORTANT: we count "
         "FAMILIES, not raw rules — with 830 rules many are the same idea at "
         "different thresholds (sales_yoy>=25/30/40 all fire on one event), so a "
         "raw count made '11+ rules agreeing' the default state (~10 rules fire on "
         "the average stock-quarter) and overstated the effect. Distinct families "
         "= genuinely independent evidence (e.g. growth + quality + technical). "
         "pct_flat_zero_ret flags buckets dominated by illiquid stocks whose price "
         "never moved — a data-quality caution, not a real 0% outcome."),
        ("Consensus_Buckets_LiquidOnly", "SAME test, but restricted to clusters "
         "with average volume >= 10,000 shares/day over the 3 months after entry — "
         "removes the illiquid/stale-price confound so this is the CLEANER read on "
         "whether agreement itself predicts success."),
        ("Best_Company_Consensus", "Companies with >=2 high-consensus (3+ rules) "
         "events, ranked by win-rate — stocks where 'the market agreeing' has "
         "historically been a reliable signal."),
        ("Worst_Company_Consensus", "Same population, ranked by how often it "
         "DECLINED despite high consensus — stocks where even multiple rules "
         "agreeing did not protect you. Useful as a caution list."),
        ("Rule_Pair_Combos", "Specific PAIRS of validated rules that, when they "
         "co-fire on the same stock-quarter (n_cooccur>=15 times historically), "
         "show the highest win-rate at 24M — the empirical answer to 'which real "
         "signal combinations work best together'."),
        ("Example_Clusters", "Real historical instances with 4+ rules agreeing, for "
         "spot-checking (which stock, which quarter, which rules, what happened)."),
        ("CAVEAT", "Endpoint returns only (no drawdown/path here — see "
         "MASTER_SHORTLIST for that, per-rule). 2019-26 bull-market drift applies "
         "the same as elsewhere. High-consensus events are correlated with popular/"
         "well-covered stocks; check liquidity separately."),
    ], columns=["item", "explanation"])

    with pd.ExcelWriter(OUT, engine="openpyxl") as xw:
        readme.to_excel(xw, "README", index=False)
        buckets_df.to_excel(xw, "Consensus_Buckets", index=False)
        buckets_liquid_df.to_excel(xw, "Consensus_Buckets_LiquidOnly", index=False)
        best_companies.to_excel(xw, "Best_Company_Consensus", index=False)
        worst_companies.to_excel(xw, "Worst_Company_Consensus", index=False)
        pairs_df.head(300).to_excel(xw, "Rule_Pair_Combos", index=False)
        ex_out.to_excel(xw, "Example_Clusters", index=False)
        ws = xw.book["README"]
        from openpyxl.styles import Alignment
        ws.column_dimensions["A"].width = 28
        ws.column_dimensions["B"].width = 105
        for row in ws.iter_rows(min_row=2):
            row[1].alignment = Alignment(wrap_text=True, vertical="top")
    log(f"CONSENSUS ANALYSIS -> {OUT}")
    log(f"  buckets: {len(buckets_df)} rows | companies scored: {len(company_df)} "
        f"| rule pairs: {len(pairs_df)}")


if __name__ == "__main__":
    main()
