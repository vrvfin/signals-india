r"""
FINAL SHORTLIST — build_shortlist_final.py  (Project Guru)

Ranks candidate stocks by EXPECTED RETURN — a real, backtested number, not an
abstract score — then shows the conviction behind it and flags what is NEW.

How the expected return is derived (this is the important part):
  We do NOT average rule returns (measured: that is flat ~42% for every stock,
  because it is a property of the RULES, not the stock). Instead we do an
  EMPIRICAL FINGERPRINT MATCH:
    * each stock has a set of currently-firing rule families ("fingerprint")
    * we find every historical liquid stock-quarter that shared at least
      MIN_OVERLAP of those same families
    * expected_return = the MEDIAN ACTUAL FORWARD RETURN those historical
      look-alikes went on to deliver, and hit_rate = how often they were positive
  So "expected_return 38%" literally means: historically, stocks that looked
  like this one returned 38% at the median over that horizon.

Conviction columns explain WHY: how many look-alike cases back the number, how
many rules/families fire, which ones, and any warning families.

NEW flag: compares against the previous run's snapshot, so re-running surfaces
"first come to limelight" names — plus days_since_first_signal for freshness.

Output: guru/FINAL_SHORTLIST.xlsx  (+ .snapshot.parquet for NEW detection)

Usage:
    python guru/build_shortlist_final.py                       # 24M, liq>=1cr
    python guru/build_shortlist_final.py --horizon 12 --top 100
"""
from __future__ import annotations
import argparse, glob, os, re
from datetime import datetime
import numpy as np, pandas as pd

GURU = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(GURU, "data")
BT = os.path.join(GURU, "backtest")
OHLCV = os.path.join(DATA, "ohlcv_hist")
OUT = os.path.join(GURU, "FINAL_SHORTLIST.xlsx")
SNAP = os.path.join(BT, "shortlist_snapshot.parquet")
RULES_X = os.path.join(os.path.dirname(GURU), "Project_Guru", "rule_template.xlsx")
CLUSTERS = os.path.join(BT, "consensus_clusters.parquet")
LIQ_MIN_HIST = 10000          # historical cluster liquidity floor (shares/day)
MIN_OVERLAP = 3               # families a look-alike must share
MIN_CASES = 30                # look-alike cases needed to trust the number


def log(m): print(f"[{datetime.now().strftime('%H:%M:%S')}] {m}", flush=True)


def family(rid: str) -> str:
    return re.sub(r"_?\d+.*$", "", rid)


def recency_days(rid: str) -> int:
    return 45 if rid.startswith("TECH") else 120


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizon", type=int, default=24, choices=[3, 6, 12, 24, 36])
    ap.add_argument("--min-liquidity", type=float, default=1.0, help="Rs cr/day")
    ap.add_argument("--top", type=int, default=150)
    args = ap.parse_args()
    H = args.horizon
    rcol = f"ret_{H}m"

    # ---------- validated rules ----------
    fs = [f for f in glob.glob(os.path.join(BT, "validation", "*.parquet"))
          if not os.path.basename(f).startswith("_")]
    v = pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True)
    val = v[(v.testability == "TESTABLE") & (v.rule_verdict == "HOLDS")]
    vh = val[val.horizon == f"{H}M"][["rule_id", "valid_median_ret", "valid_winrate"]]
    names = pd.read_excel(RULES_X, "Rules")[["rule_id", "rule_name"]]
    name_map = dict(zip(names.rule_id, names.rule_name))
    uni = pd.read_parquet(os.path.join(DATA, "universe_hist.parquet"),
                          columns=["guru_key", "name", "nse_symbol"]).set_index("guru_key")
    lift_tbl = pd.read_parquet(os.path.join(BT, "family_lift.parquet"))
    lh = lift_tbl[lift_tbl.horizon == f"{H}M"]
    lift_map = dict(zip(lh.family, lh.lift_pp))

    # ---------- live signals ----------
    live = []
    rules = sorted(val.rule_id.unique())
    for i, rid in enumerate(rules, 1):
        tf = os.path.join(BT, "triggers", f"{rid}.parquet")
        if not os.path.exists(tf):
            continue
        t = pd.read_parquet(tf, columns=["guru_key", "base_date"])
        t["base_date"] = pd.to_datetime(t["base_date"])
        t = t.sort_values("base_date").drop_duplicates("guru_key", keep="last")
        t = t[t["base_date"] >= t["base_date"].max() - pd.Timedelta(days=recency_days(rid))]
        if len(t):
            t["rule_id"] = rid
            live.append(t)
        if i % 250 == 0:
            log(f"  scanned {i}/{len(rules)} rules")
    L = pd.concat(live, ignore_index=True)
    L["family"] = L["rule_id"].map(family)
    L = L.merge(vh, on="rule_id", how="inner")
    log(f"live signals: {len(L):,} on {L.guru_key.nunique():,} stocks")

    # ---------- liquidity (current) ----------
    liq = {}
    for gk in L.guru_key.unique():
        fp = os.path.join(OHLCV, f"{gk}.parquet")
        if not os.path.exists(fp):
            continue
        px = pd.read_parquet(fp, columns=["date", "close", "volume"]).tail(20)
        if len(px):
            liq[gk] = float((px["close"] * px["volume"]).mean() / 1e7)
    L["liq"] = L["guru_key"].map(liq)
    L = L[L["liq"] >= args.min_liquidity]
    log(f"after liquidity >= {args.min_liquidity} cr/day: {L.guru_key.nunique():,} stocks")

    # ---------- historical look-alike engine ----------
    cl = pd.read_parquet(CLUSTERS)
    cl = cl[(cl.vol_avg_3m >= LIQ_MIN_HIST) & cl[rcol].notna()].copy()
    fam_list = sorted({f for s in cl.families.dropna() for f in s.split(",")})
    fidx = {f: i for i, f in enumerate(fam_list)}
    log(f"historical liquid clusters: {len(cl):,} | families: {len(fam_list)}")
    # one-hot matrix of historical fingerprints
    CM = np.zeros((len(cl), len(fam_list)), dtype=np.int8)
    for r, s in enumerate(cl.families.values):
        for f in s.split(","):
            j = fidx.get(f)
            if j is not None:
                CM[r, j] = 1
    hist_ret = cl[rcol].values

    # ---------- per-stock ----------
    rows = []
    for gk, g in L.groupby("guru_key"):
        fams = sorted(g.family.unique())
        pos = [f for f in fams if lift_map.get(f, 0) > 0]
        neg = [f for f in fams if lift_map.get(f, 0) < 0]
        # fingerprint = the POSITIVE families (what makes it attractive)
        vec = np.zeros(len(fam_list), dtype=np.int8)
        for f in pos:
            j = fidx.get(f)
            if j is not None:
                vec[j] = 1
        if vec.sum() == 0:
            continue
        overlap = CM @ vec
        need = min(MIN_OVERLAP, int(vec.sum()))
        m = overlap >= need
        n_cases = int(m.sum())
        if n_cases < MIN_CASES:
            continue
        sample = hist_ret[m]
        rules_sorted = g.sort_values("valid_median_ret", ascending=False)
        rows.append({
            "name": uni.loc[gk, "name"] if gk in uni.index else gk,
            "symbol": uni.loc[gk, "nse_symbol"] if gk in uni.index else None,
            "guru_key": gk,
            # ---- HEADLINE: what look-alikes actually returned ----
            "expected_return_pct": round(float(np.median(sample)), 1),
            "hit_rate_pct": round(float((sample > 0).mean() * 100), 1),
            "upside_p75_pct": round(float(np.percentile(sample, 75)), 1),
            "downside_p25_pct": round(float(np.percentile(sample, 25)), 1),
            "n_lookalike_cases": n_cases,
            # ---- CONVICTION ----
            "n_rules_firing": int(g.rule_id.nunique()),
            "n_families": len(fams),
            "n_supporting_families": len(pos),
            "n_warning_families": len(neg),
            "liquidity_cr_day": round(float(g.liq.iloc[0]), 2),
            "latest_signal": g.base_date.max().date().isoformat(),
            "days_since_latest": int((g.base_date.max() - g.base_date.min()).days),
            # ---- WHY ----
            "supporting_families": ", ".join(pos),
            "warning_families": ", ".join(neg),
            "top_rules": " | ".join(name_map.get(r, "")[:50]
                                    for r in rules_sorted.rule_id.head(6)),
            "all_rule_ids": ", ".join(rules_sorted.rule_id.tolist()[:50]),
        })
    d = pd.DataFrame(rows)
    if d.empty:
        log("no candidates"); return
    d = d.sort_values(["expected_return_pct", "hit_rate_pct"], ascending=False)

    # ---------- NEW detection vs previous snapshot ----------
    prev = set()
    if os.path.exists(SNAP):
        try:
            prev = set(pd.read_parquet(SNAP)["guru_key"])
        except Exception:
            pass
    d["is_NEW"] = np.where(d["guru_key"].isin(prev), "", "*** NEW ***") if prev else "(first run)"
    top = d.head(args.top)
    d[["guru_key"]].to_parquet(SNAP, index=False)   # snapshot for next run

    readme = pd.DataFrame([
        ("HOW TO USE", "Sorted by expected_return_pct — highest first. Read across: "
         "the return, then the conviction columns (how many look-alike cases, how "
         "many rules/families), then WHY (which families and rules fire)."),
        ("expected_return_pct  <-- RANK", "The MEDIAN ACTUAL RETURN that historical "
         f"look-alike stocks delivered over {H} months. A look-alike = a real "
         "historical stock-quarter that shared at least 3 of this stock's currently-"
         "firing supporting families. So '38%' means: historically, stocks that "
         "looked like this returned 38% at the median. This is a backtested "
         "outcome, NOT an average of rule statistics (that approach was measured "
         "and found flat/useless at ~42% for every stock)."),
        ("hit_rate_pct", "Of those look-alike cases, the % that ended positive."),
        ("upside_p75 / downside_p25", "The 75th and 25th percentile outcomes of the "
         "look-alikes — the realistic good case and bad case, not the extremes."),
        ("n_lookalike_cases", "How many historical cases back this estimate. "
         "Minimum 30 to appear. More = more reliable."),
        ("CONVICTION: n_rules_firing / n_families / n_supporting_families",
         "How much evidence currently points at this stock. Note that rule COUNT "
         "alone does not predict returns (measured) — it is which families, and "
         "the look-alike history, that matter. Use these as supporting context."),
        ("n_warning_families / warning_families", "Families that HISTORICALLY HURT "
         "the odds when present (e.g. valuation-at-peak, oversold, crash-reversal). "
         "Prefer candidates with 0-1 warnings. Check this before acting."),
        ("supporting_families / top_rules / all_rule_ids", "Exactly which families "
         "and rules are firing right now — the conviction narrative."),
        ("is_NEW", "*** NEW *** = this stock was not on the previous run's list. "
         "Re-run daily/weekly to surface fresh entrants ('first come to limelight'). "
         "The snapshot is stored at backtest/shortlist_snapshot.parquet."),
        ("liquidity_cr_day", "Avg traded value over the last 20 sessions (Rs crore). "
         f"Filtered to >= {args.min_liquidity} cr here. Raise it for larger size."),
        ("CAVEATS", "Look-alike history is 2019-26-heavy (a rising market) and "
         "matches on family fingerprint only — not sector, size or valuation. It is "
         "an evidence-weighted expectation, not a forecast. No transaction costs. "
         "Not investment advice."),
    ], columns=["item", "explanation"])

    with pd.ExcelWriter(OUT, engine="openpyxl") as xw:
        readme.to_excel(xw, "README", index=False)
        top.to_excel(xw, f"Shortlist_{H}M", index=False)
        newonly = top[top.is_NEW.astype(str).str.contains("NEW")]
        if not newonly.empty:
            newonly.to_excel(xw, "NEW_entrants", index=False)
        d.to_excel(xw, "All_candidates", index=False)
        ws = xw.book["README"]
        from openpyxl.styles import Alignment
        ws.column_dimensions["A"].width = 34
        ws.column_dimensions["B"].width = 105
        for row in ws.iter_rows(min_row=2):
            row[1].alignment = Alignment(wrap_text=True, vertical="top")
    log(f"FINAL SHORTLIST -> {OUT}")
    log(f"  candidates {len(d)} | showing top {len(top)} | "
        f"expected return range {top.expected_return_pct.min():.0f}% to "
        f"{top.expected_return_pct.max():.0f}%")


if __name__ == "__main__":
    main()
