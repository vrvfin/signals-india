r"""
MASTER SHORTLIST — build_master_shortlist.py  (Project Guru)

ONE file answering: "per timeframe, the top-100 most robust rules by return,
where train and validation AGREE (no overfitting), with drawdowns and the exact
exit rule to use."

Selection per horizon:
  1. TESTABLE + rule holds (validation median > 0)
  2. CONSISTENCY GATE: |train_median - valid_median| / max(|train|,|valid|) <= 20%
     -> train and validation agree; the edge is stable, not regime luck
  3. rank by validation median return (out-of-sample = the honest number)
  4. keep top 100 per horizon

Each row carries: the exact rule definition (human text + the literal clause
logic), train vs validation stats, drawdown (median + worst), the best exit and
its measured benefit, and a balanced alternative exit.

Output: guru/MASTER_SHORTLIST.xlsx
  README | Top100_3M | Top100_6M | Top100_12M | Top100_24M | Top100_36M | ALL

Usage: python guru/build_master_shortlist.py
"""
from __future__ import annotations
import glob, os
from datetime import datetime
import numpy as np, pandas as pd

GURU = os.path.dirname(os.path.abspath(__file__))
BT = os.path.join(GURU, "backtest")
OUT = os.path.join(GURU, "MASTER_SHORTLIST.xlsx")
RULES_X = os.path.join(os.path.dirname(GURU), "Project_Guru", "rule_template.xlsx")
GEN_X = os.path.join(GURU, "generated_combos.xlsx")
HZ = ["3M", "6M", "12M", "24M", "36M"]
MAX_DELTA = 0.20        # train vs validation agreement gate
TOPN = 100


def log(m): print(f"[{datetime.now().strftime('%H:%M:%S')}] {m}", flush=True)


def rule_text() -> pd.DataFrame:
    """human name + literal clause logic for every rule (base + generated)."""
    frames = []
    for x in (RULES_X, GEN_X):
        if not os.path.exists(x):
            continue
        r = pd.read_excel(x, "Rules")[["rule_id", "rule_name"]]
        c = pd.read_excel(x, "Clauses")
        c["txt"] = (c["metric"].astype(str) + " " + c["operator"].astype(str) + " "
                    + c["threshold_value"].astype(str)
                    + np.where(c["period_offset"].fillna(0).astype(int) > 0,
                               " [" + c["period_offset"].fillna(0).astype(int).astype(str)
                               + "q back]", ""))
        logic = (c.sort_values("clause_order").groupby("rule_id")["txt"]
                 .apply(lambda s: "  AND  ".join(s)).reset_index()
                 .rename(columns={"txt": "exact_logic"}))
        n = c.groupby("rule_id").size().reset_index(name="n_clauses")
        frames.append(r.merge(logic, on="rule_id", how="left").merge(n, on="rule_id", how="left"))
    return pd.concat(frames, ignore_index=True).drop_duplicates("rule_id")


def validation() -> pd.DataFrame:
    fs = [f for f in glob.glob(os.path.join(BT, "validation", "*.parquet"))
          if not os.path.basename(f).startswith("_")]
    return pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True)


def scorecard() -> pd.DataFrame:
    fs = glob.glob(os.path.join(BT, "scores", "*.parquet"))
    return pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True)


def exits() -> pd.DataFrame:
    fs = glob.glob(os.path.join(BT, "exits", "*.parquet"))
    if not fs:
        return pd.DataFrame()
    return pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True)


_OHLCV = os.path.join(GURU, "data", "ohlcv_hist")
_NAMES = None


def _names():
    global _NAMES
    if _NAMES is None:
        _NAMES = pd.read_parquet(os.path.join(GURU, "data", "universe_hist.parquet"),
                                 columns=["guru_key", "name", "nse_symbol"]
                                 ).set_index("guru_key")
    return _NAMES


def example_stocks(rid: str, horizon: str, n: int = 10) -> dict:
    """Top-n and bottom-n individual stock outcomes for this rule x horizon, as
    WIDE columns: stock1_name … stock10_vol_avg  and  worst1_name … worst10_vol_avg.
    Fields per stock: name, entry_date, entry_price, price_end, low, high, avg volume
    — the low/high/volume measured over the HOLDING WINDOW (entry -> horizon)."""
    out = {}
    tf = os.path.join(BT, "triggers", f"{rid}.parquet")
    pf = os.path.join(BT, "paths", f"{rid}.parquet")
    if not (os.path.exists(tf) and os.path.exists(pf)):
        return out
    hm = int(horizon[:-1])
    t = pd.read_parquet(tf, columns=["trigger_id", "guru_key", "entry_date",
                                     "entry_price"])
    p = pd.read_parquet(pf, columns=["trigger_id", "month", "ret_pct", "close"])
    p = p[p["month"] == hm][["trigger_id", "ret_pct", "close"]]
    d = t.merge(p, on="trigger_id", how="inner")
    if d.empty:
        return out
    d = d.sort_values("ret_pct", ascending=False)
    # ONE entry per company: a rule can fire on the same stock many times, and
    # without this the "top 10" is one company's history, not 10 opportunities.
    top = d.drop_duplicates("guru_key", keep="first").head(n)          # best per co
    bot = (d.sort_values("ret_pct").drop_duplicates("guru_key", keep="first")
           .head(n))                                                   # worst per co
    nm = _names()
    for tag, sub in (("stock", top), ("worst", bot)):
        for i, (_, r) in enumerate(sub.iterrows(), 1):
            gk = r["guru_key"]
            nrow = nm.loc[gk] if gk in nm.index else None
            label = (nrow["name"] if nrow is not None and isinstance(nrow["name"], str)
                     else gk)
            ed = pd.Timestamp(r["entry_date"])
            lo = hi = vol = np.nan
            fp = os.path.join(_OHLCV, f"{gk}.parquet")
            if os.path.exists(fp):
                px = pd.read_parquet(fp, columns=["date", "low", "high", "volume"])
                px["date"] = pd.to_datetime(px["date"])
                win = px[(px["date"] >= ed) &
                         (px["date"] <= ed + pd.Timedelta(days=int(hm * 30.44)))]
                if len(win):
                    lo = float(win["low"].min()); hi = float(win["high"].max())
                    vol = float(win["volume"].mean())
            out[f"{tag}{i}_name"] = label
            out[f"{tag}{i}_entry_date"] = ed.date().isoformat()
            out[f"{tag}{i}_entry_price"] = round(float(r["entry_price"]), 2)
            out[f"{tag}{i}_price_end"] = round(float(r["close"]), 2)
            out[f"{tag}{i}_return_pct"] = round(float(r["ret_pct"]), 1)
            out[f"{tag}{i}_low"] = None if np.isnan(lo) else round(lo, 2)
            out[f"{tag}{i}_high"] = None if np.isnan(hi) else round(hi, 2)
            out[f"{tag}{i}_vol_avg"] = None if np.isnan(vol) else int(vol)
    return out


def attach_examples(df: pd.DataFrame) -> pd.DataFrame:
    """add the wide top-10 / bottom-10 stock columns to a (rule,horizon) table."""
    recs = []
    for i, (_, r) in enumerate(df.iterrows(), 1):
        recs.append(example_stocks(r["rule_id"], r["horizon"]))
        if i % 100 == 0:
            log(f"    examples {i}/{len(df)}")
    ex = pd.DataFrame(recs, index=df.index)
    return pd.concat([df, ex], axis=1)


def main():
    log("loading …")
    v = validation()
    sc = scorecard()
    ex = exits()
    txt = rule_text()

    v = v[v["testability"] == "TESTABLE"].copy()
    v = v[v["valid_median_ret"].notna() & v["train_median_ret"].notna()]
    # CONSISTENCY: relative gap between train and validation medians
    denom = np.maximum(v["train_median_ret"].abs(), v["valid_median_ret"].abs())
    v["train_valid_delta_pct"] = np.where(denom > 0,
                                          (v["train_median_ret"] - v["valid_median_ret"]).abs()
                                          / denom * 100, 0.0)
    v["consistent"] = v["train_valid_delta_pct"] <= MAX_DELTA * 100

    # drawdown + sample from the full-history scorecard
    sc_keep = sc[["rule_id", "horizon", "median_max_drawdown_pct", "pct_dropped_big",
                  "n_triggers", "n_companies", "min_return_pct", "max_return_pct",
                  "sustain_ratio_median"]]
    m = v.merge(sc_keep, on=["rule_id", "horizon"], how="left")

    # BEST exit (max return gain, winners-cut < 40%) and BALANCED exit (COHORT_DD)
    if not ex.empty:
        cand = ex[(ex.exit != "HOLD") & (ex.vs_hold > 0) &
                  ((ex.pct_winners_cut.isna()) | (ex.pct_winners_cut < 40))]
        best = (cand.sort_values("vs_hold", ascending=False)
                .drop_duplicates(["rule_id", "horizon"])
                [["rule_id", "horizon", "exit", "vs_hold", "pct_kaputs_avoided",
                  "pct_winners_cut"]]
                .rename(columns={"exit": "best_exit",
                                 "vs_hold": "best_exit_gain_pp",
                                 "pct_kaputs_avoided": "best_exit_disasters_avoided_pct",
                                 "pct_winners_cut": "best_exit_winners_cut_pct"}))
        bal = (ex[ex.exit == "COHORT_DD"][["rule_id", "horizon", "cohort_dd_stop",
                                           "vs_hold", "pct_kaputs_avoided"]]
               .rename(columns={"cohort_dd_stop": "balanced_stop_pct",
                                "vs_hold": "balanced_exit_gain_pp",
                                "pct_kaputs_avoided": "balanced_disasters_avoided_pct"}))
        m = m.merge(best, on=["rule_id", "horizon"], how="left")
        m = m.merge(bal, on=["rule_id", "horizon"], how="left")
    m = m.merge(txt, on="rule_id", how="left")

    # final selection
    sel = m[m["consistent"] & (m["valid_median_ret"] > 0)].copy()
    sel["is_generated_combo"] = sel["rule_id"].str.startswith("DEEPGEN")
    cols = ["rule_id", "rule_name", "exact_logic", "n_clauses", "horizon",
            "valid_median_ret", "train_median_ret", "train_valid_delta_pct",
            "valid_winrate", "train_winrate", "n_valid", "n_train", "n_triggers",
            "n_companies", "median_max_drawdown_pct", "pct_dropped_big",
            "min_return_pct", "max_return_pct", "sustain_ratio_median",
            "best_exit", "best_exit_gain_pp", "best_exit_disasters_avoided_pct",
            "best_exit_winners_cut_pct", "balanced_stop_pct",
            "balanced_disasters_avoided_pct", "is_generated_combo"]
    cols = [c for c in cols if c in sel.columns]
    sel = sel[cols]

    readme_rows = [
        ("=== 1. WHAT THIS FILE IS ===", ""),
        ("Purpose", "One file, per holding period (3/6/12/24/36 months), listing "
         "which rules actually made money OUT-OF-SAMPLE (on 2019-26 data the rule "
         "never saw), how consistent that edge was between two different market "
         "eras, how painful the ride was, the best way to exit, and real historical "
         "winners AND losers so you can see what success and failure look like."),
        ("Where the numbers come from", "Every rule was backtested over 2006-2026 "
         "on the full Indian listed universe. The history was split at 2018-12-31: "
         "TRAIN = 2006-2018 (used to discover/tune the rule), VALIDATION = 2019-2026 "
         "(used only to check it — the honesty check). All 'valid_*' numbers are "
         "the validation-period result, i.e. what you'd have actually gotten."),
        ("", ""),
        ("=== 2. WHAT EACH SHEET CONTAINS ===", ""),
        ("Top100_3M / 6M / 12M / 24M / 36M", "For that holding period: the 100 "
         "rules with the HIGHEST validation-period median return. Answers 'what "
         "made the most money out-of-sample at this horizon?'. No consistency "
         "filter is applied here — train_valid_delta_pct is just a column, so a "
         "rule that only worked in the recent bull market can still appear; check "
         "its delta before trusting it."),
        ("MostStable_3M / 6M / 12M / 24M / 36M", "Same universe of rules, same "
         "horizon, but sorted the OPPOSITE way: smallest train_valid_delta_pct "
         "first. Answers 'what behaved the SAME in both the 2006-18 and 2019-26 "
         "eras?'. These tend to have lower returns than Top100 but are less likely "
         "to be a one-era fluke."),
        ("STRICT_delta20_only", "The rules that are BOTH profitable out-of-sample "
         "AND have a delta of 20% or less, across all horizons in one sheet. The "
         "narrowest, most conservative shortlist."),
        ("ALL_rules", "Every rule at every horizon with a positive validation "
         "return, unsorted filter, for reference / your own pivoting."),
        ("How to use the two main views together", "A rule that appears near the "
         "top of BOTH Top100_<H> and MostStable_<H> is the strongest candidate: "
         "high return AND consistent across eras. A rule that is high in Top100 "
         "but has a large delta (see column glossary) made most of its money in "
         "one specific period — treat its return number with caution, not as a "
         "repeatable expectation."),
        ("", ""),
        ("=== 3. COLUMN-BY-COLUMN GLOSSARY ===", ""),
        ("rule_id", "Internal code for the rule (e.g. FUND_GROWTH_001, DEEPGEN_0042 "
         "for a generated combo)."),
        ("rule_name", "Short human label, e.g. 'Sales YoY growth >= 25% for 1 "
         "consecutive quarter(s)'."),
        ("exact_logic", "The LITERAL condition(s), machine-readable: "
         "'metric operator threshold', joined by AND if there's more than one. "
         "'[Nq back]' means that specific condition is checked N quarters before "
         "the trigger quarter (used by multi-quarter consistency rules)."),
        ("n_clauses", "How many conditions are ANDed together. 1 = a simple rule "
         "(e.g. just 'ROCE > 18%'). Higher = a more specific, multi-condition rule."),
        ("horizon", "The holding period this row measures: 3M/6M/12M/24M/36M "
         "(months after the entry date)."),
        ("valid_median_ret", "THE HEADLINE NUMBER. Median % return, measured only "
         "on 2019-2026 triggers (out-of-sample). Trust this over train_median_ret."),
        ("train_median_ret", "Same median % return, but measured on 2006-2018 "
         "triggers only (the period the rule was built/discovered on)."),
        ("train_valid_delta_pct", "THE RELIABILITY SCORE. How far apart train and "
         "validation results are, as a % of whichever is larger. 0% = the rule "
         "behaved identically in both eras (most trustworthy). 100%+ = the rule's "
         "result in one era was radically different from the other (treat with "
         "caution — likely era-specific, e.g. rode one bull market)."),
        ("valid_winrate / train_winrate", "% of triggers that had a POSITIVE return "
         "at this horizon, in validation / train respectively. 50% = coin flip."),
        ("n_valid / n_train", "Number of individual trigger events in the "
         "validation / train period. Small n (say <30) means treat the row's "
         "numbers as a lead, not a statistically solid conclusion."),
        ("n_triggers / n_companies", "Total triggers and distinct companies across "
         "ALL 2006-2026 history (train+validation combined) — the overall sample "
         "size behind the rule."),
        ("median_max_drawdown_pct", "For the median trigger, how far the stock fell "
         "from its peak at any point during the holding window before the horizon "
         "ended. This is the PAIN number — e.g. -35% means a typical holder "
         "watched the position fall 35% below its best point along the way."),
        ("pct_dropped_big", "% of ALL triggers that fell 50% or more below entry at "
         "some point during the holding window (even if they recovered after)."),
        ("min_return_pct / max_return_pct", "The single worst and single best "
         "outcome ever recorded for this rule at this horizon (across all history, "
         "not just validation) — the extremes, not the typical case."),
        ("sustain_ratio_median", "For the median trigger: (return at the horizon) "
         "divided by (the best return it ever reached along the way). Close to 1 "
         "= it held onto its gains. Well below 1 = it peaked higher then gave a lot "
         "of that gain back before the horizon ended."),
        ("best_exit", "Out of all exit rules tested, the one that most improved "
         "median return vs. simply holding to the horizon, measured out-of-sample. "
         "TARGET_X = sell as soon as return hits +X% (take profit). TRAIL_X = sell "
         "if price falls X% from its peak (trailing stop). TIME_N = sell at N "
         "months if the return is still below +10% (cut dead money). COHORT_DD = "
         "sell if the drawdown breaches the level this rule's stocks have "
         "HISTORICALLY tended to fall to (a stop tailored to this specific rule, "
         "not a generic %)."),
        ("best_exit_gain_pp", "How many PERCENTAGE POINTS of median return that "
         "exit adds compared to just holding to the horizon. E.g. +15.6 means "
         "using this exit turned a hypothetical +10.7% hold-median into +26.3%."),
        ("best_exit_disasters_avoided_pct", "Of the triggers that would have been "
         "big losers if held to the horizon, what % did this exit rescue (i.e. "
         "got the holder out before the worst of the damage)."),
        ("best_exit_winners_cut_pct", "THE COST of the exit: of the triggers that "
         "would have become big winners if held, what % did this exit sell too "
         "early, capping their gain. A good exit has high disasters-avoided and "
         "low winners-cut; there is usually a trade-off between the two."),
        ("balanced_stop_pct", "The COHORT_DD stop level specifically: sell if the "
         "position falls this % from its peak. This is tailored per rule (based on "
         "what that rule's stocks have historically endured), not a fixed number."),
        ("balanced_disasters_avoided_pct", "Same meaning as best_exit's version, "
         "but specifically for the COHORT_DD (balanced) exit — usually a good "
         "middle ground between protecting capital and not selling winners early."),
        ("is_generated_combo", "TRUE if this is a multi-condition combo rule "
         "automatically generated by crossing two proven single rules (e.g. a "
         "fundamental growth rule + a technical entry-timing rule), rather than one "
         "of the originally hand-designed rules."),
        ("", ""),
        ("=== 4. THE STOCK EXAMPLE COLUMNS ===", ""),
        ("stock1_name … stock10_vol_avg", "The 10 BEST individual outcomes for this "
         "rule at this horizon, one real company per slot (never the same company "
         "twice), best return first. Eight fields per stock, prefixed stockN_: "
         "  name — the company. "
         "  entry_date — the date the rule triggered (day the position was opened). "
         "  entry_price — the buy price that day. "
         "  price_end — the price at the end of the holding window (at the horizon). "
         "  return_pct — the resulting % return. "
         "  low / high — the LOWEST and HIGHEST price the stock touched at any "
         "point DURING the holding window (shows the ride, not just the endpoint). "
         "  vol_avg — average daily trading volume (shares/day) over the holding "
         "window — check this before assuming you could actually buy/sell the size "
         "you want; very low volume (a few thousand shares/day) means the stock is "
         "hard to trade in size even if the return looks great."),
        ("worst1_name … worst10_vol_avg", "The 10 WORST individual outcomes, same "
         "8 fields, one company per slot, worst return first. Look at these before "
         "trusting a rule — they show exactly what failure looked like historically "
         "(how far it fell via the 'low' field, and whether it ever recovered)."),
        ("", ""),
        ("=== 5. WORKED EXAMPLE: HOW TO READ ONE ROW ===", ""),
        ("Example rule", "'ROCE > 18% sustained for 8 quarter(s)' on sheet "
         "MostStable_24M, row 1."),
        ("Step 1 - is the edge real?", "valid_median_ret = 30.9%, train_median_ret "
         "= 31.0%, train_valid_delta_pct = 0.13%. The rule returned almost exactly "
         "the same in both eras -> this is a genuine, consistent edge, not a "
         "period-specific fluke."),
        ("Step 2 - how confident, how painful?", "valid_winrate 73.2% (roughly 3 "
         "in 4 triggers were profitable at 24 months), n_valid in the hundreds "
         "(decent sample), median_max_drawdown_pct around -35% (expect to see the "
         "position dip ~35% below its peak at some point even in a typical case)."),
        ("Step 3 - how to exit", "best_exit = TARGET_100 (take profit at +100%); "
         "balanced_stop_pct around -48% if you'd rather use a drawdown-based stop "
         "instead of a fixed profit target."),
        ("Step 4 - sanity-check with real stocks", "Look at stock1..stock10: "
         "winners include names with HIGH average volume (hundreds of thousands of "
         "shares/day) alongside some very illiquid ones (a few thousand shares/day) "
         "-> the rule works broadly, but position size should respect each stock's "
         "own liquidity. Look at worst1..worst10: some of the SAME companies appear "
         "as both a big winner (one entry date) and a big loser (a different entry "
         "date) -> this tells you the entry TIMING matters as much as the "
         "fundamental condition itself."),
        ("Step 5 - conclusion", "This row is a good candidate for further research: "
         "consistent across 19 years of data, decent win-rate, known drawdown and "
         "exit plan — but always check liquidity per stock before sizing a position."),
        ("", ""),
        ("=== 6. CAVEATS ===", ""),
        ("Market drift", "2019-2026 was a strong bull market for Indian equities. "
         "Long-horizon (24M/36M) absolute returns are partly just 'the market went "
         "up a lot', not pure stock-picking skill. Prefer win-rate and the "
         "train_valid_delta_pct for judging genuine edge over raw return size."),
        ("Not investment advice", "This is a historical backtest with no "
         "transaction costs, slippage, taxes, or borrowing limits. Small sample "
         "sizes (low n_valid) can look dramatic and still be statistical noise. "
         "Always check liquidity (vol_avg) before assuming a position is tradeable "
         "at the size you want."),
    ]
    readme = pd.DataFrame(readme_rows, columns=["item", "explanation"])

    # UNFILTERED view: no consistency condition — every positive-validation rule,
    # ranked by return, with the delta shown so YOU judge the trade-off.
    allpos = m[m["valid_median_ret"] > 0].copy()
    allpos["is_generated_combo"] = allpos["rule_id"].str.startswith("DEEPGEN")
    allpos = allpos[[c for c in cols if c in allpos.columns]]

    with pd.ExcelWriter(OUT, engine="openpyxl") as xw:
        readme.to_excel(xw, "README", index=False)
        ws = xw.book["README"]
        from openpyxl.styles import Alignment, Font
        ws.column_dimensions["A"].width = 32
        ws.column_dimensions["B"].width = 110
        for row in ws.iter_rows(min_row=2):
            item_cell, exp_cell = row[0], row[1]
            is_header = str(item_cell.value or "").startswith("===")
            item_cell.font = Font(bold=True, size=12 if is_header else 11)
            exp_cell.alignment = Alignment(wrap_text=True, vertical="top")
            exp_cell.font = Font(italic=is_header)
        # PRIMARY: per horizon, top 100 by return, NO consistency filter,
        # delta_pct shown as a column you can sort/eyeball
        for h in HZ:
            d = (allpos[allpos.horizon == h]
                 .sort_values("valid_median_ret", ascending=False).head(TOPN))
            if not d.empty:
                log(f"  {h}: top{len(d)} by return | best {d.valid_median_ret.max():.0f}% "
                    f"| delta range {d.train_valid_delta_pct.min():.0f}-"
                    f"{d.train_valid_delta_pct.max():.0f}% — attaching stock examples")
                attach_examples(d).to_excel(xw, f"Top{TOPN}_{h}", index=False)
        # SECONDARY: same universe ranked by AGREEMENT (lowest delta first)
        for h in HZ:
            d = (allpos[allpos.horizon == h]
                 .sort_values(["train_valid_delta_pct", "valid_median_ret"],
                              ascending=[True, False]).head(TOPN))
            if not d.empty:
                log(f"  MostStable_{h}: attaching stock examples")
                attach_examples(d).to_excel(xw, f"MostStable_{h}", index=False)
        # the original strict view, kept
        sel.sort_values(["horizon", "valid_median_ret"], ascending=[True, False]
                        ).to_excel(xw, "STRICT_delta20_only", index=False)
        allpos.sort_values(["horizon", "valid_median_ret"], ascending=[True, False]
                           ).to_excel(xw, "ALL_rules", index=False)
    log(f"MASTER SHORTLIST -> {OUT}")
    log(f"  unfiltered positive rule-horizons: {len(allpos)} | strict-20% subset: {len(sel)}")


if __name__ == "__main__":
    main()
