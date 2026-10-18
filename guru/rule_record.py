r"""
RULE RECORD — rule_record.py  (Project Guru, O6 2026-10-17)

Each live rule's BACKTEST RECORD, for the Guru page's "why it was picked" and
"what to expect" (user decisions 2026-10-17: expectation = the stock's STRONGEST
rule; reference stop = that rule's typical 3-month dip).

Reads only artefacts the backtest already wrote (local, gitignored guru/backtest/):
    _chosen_rules.parquet               live rules + worst_of_both (validated OOS return)
    scorecard.xlsx / Scorecard          per rule x horizon: n, median, p25/p75, win rate,
                                        median max drawdown
    validation_summary.xlsx / All_Rules_Detail   out-of-sample verdict + win rate
Nothing is re-computed and nothing is written. A missing file -> {} (page renders
without records). Small-n rules keep their record and show their n (binding: never
drop or hide a rule for small sample size).

    python guru/rule_record.py --self-test
    python guru/rule_record.py --show TECH_UPWEEKS_003
"""
from __future__ import annotations

import argparse
import html
import os
import sys

import pandas as pd

GURU = os.path.dirname(os.path.abspath(__file__))
BT = os.path.join(GURU, "backtest")
HORIZONS = ("1M", "3M")                 # ~20 and ~60 trading days
STOP_HORIZON = "3M"
_VERDICT = {"HOLDS": ("held up out-of-sample", "#1a7a3a"),
            "WEAKER": ("weaker out-of-sample", "#b8860b"),
            "BREAKS": ("did NOT hold out-of-sample", "#c0392b")}


def _num(v):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if f != f else f


def load_records(bt_dir: str = BT) -> dict:
    """{rule_id: record} for the live (chosen) rules. {} when artefacts are missing."""
    chosen_p = os.path.join(bt_dir, "_chosen_rules.parquet")
    score_p = os.path.join(bt_dir, "scorecard.xlsx")
    valid_p = os.path.join(bt_dir, "validation_summary.xlsx")
    if not (os.path.exists(chosen_p) and os.path.exists(score_p)):
        return {}
    chosen = pd.read_parquet(chosen_p)
    ids = set(chosen["rule_id"].astype(str))
    sc = pd.read_excel(score_p, "Scorecard")
    want_h = set(HORIZONS) | {STOP_HORIZON} | set(chosen["horizon"].astype(str))
    sc = sc[sc["rule_id"].astype(str).isin(ids) & sc["horizon"].astype(str).isin(want_h)]
    va = pd.DataFrame()
    if os.path.exists(valid_p):
        va = pd.read_excel(valid_p, "All_Rules_Detail")
        va = va[va["rule_id"].astype(str).isin(ids)]
    return build_records(chosen, sc, va)


def build_records(chosen: pd.DataFrame, sc: pd.DataFrame, va: pd.DataFrame) -> dict:
    """Pure: the three frames -> {rule_id: record}."""
    out = {}
    for rid, g in chosen.groupby(chosen["rule_id"].astype(str)):
        # The horizon the rule was CHOSEN for (its best validated return). A rule
        # chosen for 24M can be flat at 1-3M (TECH_PAIR_059: 0% at 3M, +41% at 24M),
        # so its verdict and headline outcome are read at THIS horizon.
        best = g.loc[pd.to_numeric(g["worst_of_both"], errors="coerce").idxmax()]
        rec = {"rule_id": rid, "name": str(g["rule_name"].iloc[0]),
               "oos_return": _num(g["worst_of_both"].max()), "h": {},
               "best_h": str(best["horizon"]), "verdict": "", "valid_winrate": None}
        for _, r in sc[sc["rule_id"].astype(str) == rid].iterrows():
            rec["h"][str(r["horizon"])] = {
                "win": _num(r.get("success_prob_pos")),
                "median": _num(r.get("median_return_pct")),
                "p25": _num(r.get("p25_return_pct")), "p75": _num(r.get("p75_return_pct")),
                "dip": _num(r.get("median_max_drawdown_pct")),
                "n_ep": _num(r.get("n_episodes")), "n_co": _num(r.get("n_companies"))}
        if va is not None and not va.empty:
            v = va[(va["rule_id"].astype(str) == rid)
                   & (va["horizon"].astype(str) == rec["best_h"])]
            if not v.empty:
                rec["verdict"] = str(v["rule_verdict"].iloc[0] or "").strip().upper()
                rec["valid_winrate"] = _num(v["valid_winrate"].iloc[0])
        out[rid] = rec
    return out


def name_index(records: dict) -> dict:
    """{rule_name (stripped, lower): rule_id} — the picks' 'why' lines carry names."""
    return {r["name"].strip().lower(): rid for rid, r in records.items()}


def strongest(rule_ids, records: dict):
    """The stock's best-validated rule: highest validated out-of-sample return
    (the same number daily_screen weights conviction by). None if none known."""
    known = [records[r] for r in rule_ids if r in records]
    known = [r for r in known if r["oos_return"] is not None] or known
    if not known:
        return None
    return max(known, key=lambda r: (r["oos_return"] or float("-inf")))


def _pct(v, sign=True):
    if v is None:
        return "–"
    if sign:
        return f"{v:+.1f}%" if abs(v) < 10 else f"{v:+.0f}%"
    return f"{v:.0f}%"


def _shown_h(rec: dict) -> list:
    """1M, 3M, then the horizon the rule is validated for (if different), in order."""
    order = ["1M", "2M", "3M", "6M", "12M", "18M", "24M", "36M", "48M", "60M", "84M", "120M"]
    hs = list(HORIZONS) + ([rec.get("best_h")] if rec.get("best_h") not in HORIZONS else [])
    return sorted([h for h in hs if h in rec["h"]], key=lambda h: order.index(h)
                  if h in order else 99)


def _n(v):
    return "–" if v is None else f"{int(v):,}"


def record_line(rec: dict) -> str:
    """Compact HTML: verdict + 1M / 3M outcome of past signals of this rule."""
    if not rec:
        return ""
    bits = []
    vtxt, vcol = _VERDICT.get(rec.get("verdict", ""), ("", "#777"))
    if vtxt:
        bits.append(f'<span style="color:{vcol};font-weight:600">chosen for '
                    f'{rec.get("best_h", "?")} · {vtxt}</span>')
    for h in _shown_h(rec):
        x = rec["h"][h]
        bits.append(f"{h}: {_pct(x['win'], False)} rose, median {_pct(x['median'])}, "
                    f"typical dip {_pct(x['dip'])}")
    x = rec["h"].get(STOP_HORIZON) or next(iter(rec["h"].values()), None)
    if x:
        bits.append(f"n={_n(x['n_ep'])} signals / {_n(x['n_co'])} cos")
    if not bits:
        return ""
    return ('<div style="color:#666;font-size:11px;margin:1px 0 0 0">'
            + " · ".join(bits) + "</div>")


def expect_box(rec: dict, price) -> str:
    """'What to expect' from the strongest rule + the reference stop. '' if unknown."""
    if not rec or not rec["h"]:
        return ""
    rows = []
    for h in _shown_h(rec):
        x = rec["h"][h]
        tag = (" <span style='color:#7a5a00'>(validated)</span>"
               if h == rec.get("best_h") else "")
        rows.append(
            f"<tr><td style='padding:1px 8px 1px 0'><b>{h}</b>{tag}</td>"
            f"<td style='padding:1px 8px'>{_pct(x['win'], False)} of past signals rose</td>"
            f"<td style='padding:1px 8px'>median {_pct(x['median'])}</td>"
            f"<td style='padding:1px 8px'>middle half {_pct(x['p25'])} to {_pct(x['p75'])}</td>"
            f"<td style='padding:1px 8px'>typical worst dip {_pct(x['dip'])}</td></tr>")
    if not rows:
        return ""
    stop = ""
    s = rec["h"].get(STOP_HORIZON) or {}
    p, dip = _num(price), s.get("dip")
    if p and dip is not None and dip < 0:
        lvl = p * (1 + dip / 100.0)
        stop = (f"<div style='margin-top:4px'><b>Reference stop ≈ ₹{lvl:,.2f}</b> "
                f"<span style='color:#666'>(₹{p:,.2f} × (1 {dip:+.0f}%) — this rule's "
                f"median worst dip within {STOP_HORIZON} of a signal. Half of past signals "
                f"dipped deeper and some still recovered, so it is a reference level, "
                f"not a sell rule)</span></div>")
    x = rec["h"].get(STOP_HORIZON) or next(iter(rec["h"].values()))
    return ("<div style='background:#fff8e6;border:1px solid #f0d9a0;border-radius:6px;"
            "padding:6px 10px;margin:3px 0;font-size:12px'>"
            "<div style='font-weight:700;color:#7a5a00'>What to expect</div>"
            f"<div style='color:#555'>Based on its strongest rule, "
            f"<b>{html.escape(rec['name'])}</b> (chosen for its {rec.get('best_h', '?')} "
            f"record — expect it to work over that span, not days): what happened after past signals "
            f"(n={_n(x.get('n_ep'))} past signals). Backtest, before costs — not a forecast.</div>"
            f"<table style='border-collapse:collapse;margin-top:3px'>{''.join(rows)}</table>"
            f"{stop}</div>")


def _self_test() -> int:
    fails = 0

    def check(name, ok):
        nonlocal fails
        print(("  ok   " if ok else "  FAIL ") + name)
        fails += 0 if ok else 1

    chosen = pd.DataFrame([
        {"rule_id": "R1", "horizon": "3M", "rule_name": "12+ consecutive up weeks",
         "worst_of_both": 9.66, "valid_winrate": 59.0},
        {"rule_id": "R2", "horizon": "3M", "rule_name": "Rare combo", "worst_of_both": 20.0,
         "valid_winrate": 70.0}])
    sc = pd.DataFrame([
        {"rule_id": "R1", "horizon": "1M", "n_companies": 1086, "n_episodes": 1407,
         "median_return_pct": 9.34, "p25_return_pct": -6.18, "p75_return_pct": 37.76,
         "success_prob_pos": 64.7, "median_max_drawdown_pct": -2.84},
        {"rule_id": "R1", "horizon": "3M", "n_companies": 1079, "n_episodes": 1390,
         "median_return_pct": 10.11, "p25_return_pct": -16.27, "p75_return_pct": 58.98,
         "success_prob_pos": 60.4, "median_max_drawdown_pct": -18.31},
        {"rule_id": "R2", "horizon": "3M", "n_companies": 1, "n_episodes": 2,
         "median_return_pct": 40.0, "p25_return_pct": 30.0, "p75_return_pct": 50.0,
         "success_prob_pos": 100.0, "median_max_drawdown_pct": -5.0}])
    va = pd.DataFrame([{"rule_id": "R1", "horizon": "3M", "rule_verdict": "HOLDS",
                        "valid_winrate": 59.0},
                       {"rule_id": "R3", "horizon": "3M", "rule_verdict": "BREAKS",
                        "valid_winrate": 48.0},
                       {"rule_id": "R3", "horizon": "24M", "rule_verdict": "HOLDS",
                        "valid_winrate": 65.5}])
    chosen = pd.concat([chosen, pd.DataFrame([
        {"rule_id": "R3", "horizon": "24M", "rule_name": "ATR expansion after listing",
         "worst_of_both": 32.0, "valid_winrate": 65.5}])], ignore_index=True)
    sc = pd.concat([sc, pd.DataFrame([
        {"rule_id": "R3", "horizon": "3M", "n_companies": 4191, "n_episodes": 10998,
         "median_return_pct": 0.0, "p25_return_pct": -14, "p75_return_pct": 23,
         "success_prob_pos": 47.8, "median_max_drawdown_pct": -17.0},
        {"rule_id": "R3", "horizon": "24M", "n_companies": 4000, "n_episodes": 10525,
         "median_return_pct": 41.37, "p25_return_pct": -30, "p75_return_pct": 150,
         "success_prob_pos": 64.2, "median_max_drawdown_pct": -45.0}])], ignore_index=True)
    recs = build_records(chosen, sc, va)
    check("records built for every live rule", set(recs) == {"R1", "R2", "R3"})
    check("verdict read at the CHOSEN horizon (24M HOLDS, not 3M BREAKS)",
          recs["R3"]["best_h"] == "24M" and recs["R3"]["verdict"] == "HOLDS")
    b3 = expect_box(recs["R3"], 100.0)
    check("expect box adds the validated horizon row", "24M" in b3 and "(validated)" in b3
          and "+41%" in b3)
    check("stop still from the 3M dip", "₹83.00" in b3)
    check("1M and 3M captured", set(recs["R1"]["h"]) == {"1M", "3M"})
    check("verdict from the 3M validation row", recs["R1"]["verdict"] == "HOLDS")
    check("strongest = highest validated OOS return", strongest(["R1", "R2", "R3"], recs)["rule_id"] == "R3")
    check("unknown rule ids ignored", strongest(["X"], recs) is None)
    line = record_line(recs["R1"])
    check("record line has verdict, win rate, median, dip, n",
          all(s in line for s in ("held up out-of-sample", "65% rose", "median +9.3%",
                                  "typical dip -18%", "n=1,390")))
    box = expect_box(recs["R1"], 100.0)
    check("stop = price x (1 + 3M median dip)", "₹81.69" in box)
    check("expect box shows the middle-half range", "-16% to +59%" in box)
    small = expect_box(recs["R2"], 50.0)
    check("small-n rule is SHOWN with its n, never hidden", "n=2 past signals" in small)
    check("no price -> no stop, box still shown", "Reference stop" not in expect_box(recs["R1"], None)
          and "What to expect" in expect_box(recs["R1"], None))
    check("name index maps names to ids", name_index(recs)["12+ consecutive up weeks"] == "R1")
    print(f"\n{'ALL PASS' if not fails else f'{fails} FAILED'}")
    return 1 if fails else 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--show", default="")
    ap.add_argument("--backtest-dir", default=BT)
    a = ap.parse_args()
    if a.self_test:
        sys.exit(_self_test())
    recs = load_records(a.backtest_dir)
    print(f"{len(recs)} live rules with a record")
    if a.show:
        import json
        print(json.dumps(recs.get(a.show, {}), indent=1, default=str))
