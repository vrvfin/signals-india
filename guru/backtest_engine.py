r"""
P6+P7 — backtest_engine.py  (Project Guru, STANDALONE, RESUMABLE per rule)

Reads the Rules/Clauses DSL from Project_Guru/rule_template.xlsx, finds every
historical (company, trigger) event, and stores the FULL DECISION TRAIL — not
just aggregates — so any strategy can later be replayed/charted:

guru/backtest/triggers/<rule_id>.parquet    one row per trigger event
    trigger_id, guru_key, symbol, anchor_period, base_date (announcement/tape),
    entry_date (next trading day), entry_price (that day's open),
    clause_snapshot (JSON: every clause's metric, offset, op, threshold AND the
    actual value that passed), mcap_cr_at_trigger, close_at_trigger,
    base_date_estimated, mixed_basis_flag
guru/backtest/paths/<rule_id>.parquet       one row per trigger per month
    trigger_id, month (1..120), date, close, ret_pct (vs entry),
    peak_ret_pct, drawdown_pct   <- the movement record for charts/exits
guru/backtest/scorecard.parquet             Rule_Horizon_Scorecard rows
    (rule_id x horizon grain, per the xlsx schema; absolute returns v1,
     NEVER drops small-n; episode-deduped counts alongside raw)
guru/backtest/_ledger.parquet               resume ledger (per rule_id)

Engine semantics (spec §5/§8): rolling anchor = every quarter where clause 0
passes and all offsets pass; base date = announcement date of the highest-
offset fundamental clause (no look-ahead); technical clauses in combined rules
must fire within their offset window AFTER the base date (entry = that day).
Pure-technical rules are edge-triggered episodes with a 63-trading-day re-arm.

Usage:
    python guru/backtest_engine.py --dry-run
    python guru/backtest_engine.py --rules FUND_GROWTH_001,TECH_MOM_001
    python guru/backtest_engine.py --limit 10          # first N pending rules
    python guru/backtest_engine.py                     # full, resumes
    python guru/backtest_engine.py --status
"""
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from functools import lru_cache

import numpy as np
import pandas as pd

GURU_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(GURU_DIR, "data")
BT_DIR = os.path.join(GURU_DIR, "backtest")
TRIG_DIR = os.path.join(BT_DIR, "triggers")
PATH_DIR = os.path.join(BT_DIR, "paths")
LEDGER = os.path.join(BT_DIR, "_ledger.parquet")
SCORE = os.path.join(BT_DIR, "scorecard.parquet")       # legacy (pre-parallel)
SCORE_DIR = os.path.join(BT_DIR, "scores")              # per-rule fragments
XLSX = os.path.join(os.path.dirname(GURU_DIR), "Project_Guru", "rule_template.xlsx")

QUNI_DIR = os.path.join(DATA_DIR, "metrics", "quarterly_unified")
FMET_DIR = os.path.join(DATA_DIR, "metrics", "fundamental")
TECH_DIR = os.path.join(DATA_DIR, "metrics", "technical")
OHLCV_DIR = os.path.join(DATA_DIR, "ohlcv_hist")
REGIME = os.path.join(DATA_DIR, "metrics", "regime.parquet")

HORIZONS_M = [1, 2, 3, 6, 12, 18, 24, 36, 48, 60, 84, 120]
TDAYS_PER_M = 21
REARM_DAYS = 63                 # technical episode re-arm (one quarter)
PRICE_RELIABLE_FROM = pd.Timestamp("2009-01-01")

# metric domains
REGIME_COLS = {"nifty500_above_200dma", "india_vix"}
QUNI_COLS = {"sales_yoy_pct", "profit_yoy_pct", "eps_yoy_pct", "sales_qoq_pct",
             "profit_qoq_pct", "net_margin_pct", "margin_yoy_change_pct",
             "net_profit_cr"}          # served by quarterly_unified (deep)


def log(m): print(f"[{datetime.now().strftime('%H:%M:%S')}] {m}", flush=True)


# ---------------- rule loading ----------------

import re as _re
_AVAIL_RET_WINDOWS = [1, 2, 3, 6, 9, 12, 18, 24, 36]


def _resolve_price_return(rule_name: str, clauses: list[dict]) -> None:
    """The workbook uses generic 'price_return_pct' with the window only in the
    rule NAME ('over trailing 3 month(s)', 'in 6 months'); the technical store
    has concrete price_return_{N}m_pct columns. Resolve in-place."""
    for c in clauses:
        if c["metric"] != "price_return_pct":
            continue
        m = _re.findall(r"(\d+)\s*month", str(rule_name), _re.I)
        n = int(m[-1]) if m else 12
        n = min(_AVAIL_RET_WINDOWS, key=lambda w: abs(w - n))
        c["metric"] = f"price_return_{n}m_pct"


def load_rules(only: list[str] | None = None) -> list[dict]:
    rules = pd.read_excel(XLSX, "Rules")
    clauses = pd.read_excel(XLSX, "Clauses")
    out = []
    for _, r in rules.iterrows():
        rid = r["rule_id"]
        if only and rid not in only:
            continue
        cl = clauses[clauses["rule_id"] == rid].sort_values("clause_order")
        if cl.empty:
            continue
        recs = cl.to_dict("records")
        _resolve_price_return(r.get("rule_name", ""), recs)
        out.append({"rule_id": rid, "name": r.get("rule_name", ""),
                    "category": r.get("category", ""),
                    "anchor_mode": r.get("anchor_mode", "first_trigger"),
                    "base_date_rule": r.get("base_date_rule", "last_clause"),
                    "clauses": recs})
    return out


def _op(series: pd.Series, op: str, thr) -> pd.Series:
    if op == "between":
        lo, hi = [float(x) for x in str(thr).split(",")]
        return (series >= lo) & (series <= hi)
    thr = float(thr)
    return {"" : series > thr, ">": series > thr, ">=": series >= thr,
            "<": series < thr, "<=": series <= thr,
            "==": series == thr}[op.strip()]


# ---------------- per-company data frames (cached) ----------------

@lru_cache(maxsize=256)
def fund_frame(gk: str) -> pd.DataFrame | None:
    """quarterly grid: unified quarterly + ownership/valuation extras + annual
    metrics broadcast as-of (latest annual period_end <= quarter end)."""
    qp = os.path.join(QUNI_DIR, f"{gk}.parquet")
    if not os.path.exists(qp):
        return None
    q = pd.read_parquet(qp)
    q = q[q["period_end"].notna()].sort_values("period_end").reset_index(drop=True)
    fp = os.path.join(FMET_DIR, f"{gk}.parquet")
    if os.path.exists(fp):
        fm = pd.read_parquet(fp)
        extras = fm[fm["grain"] == "quarterly"]
        keep = [c for c in extras.columns if c not in q.columns
                and c not in ("grain", "period", "guru_key", "period_end_date")]
        if not extras.empty and keep:
            e = extras[["period_end_date"] + keep].rename(
                columns={"period_end_date": "period_end"})
            e["period_end"] = pd.to_datetime(e["period_end"])
            q = q.merge(e, on="period_end", how="left")
        ann = fm[fm["grain"] == "annual"].sort_values("period_end_date")
        acols = [c for c in ann.columns if c not in q.columns
                 and c not in ("grain", "period", "guru_key", "announcement_date",
                               "base_date_estimated", "period_end_date")]
        if not ann.empty and acols:
            a = ann[["period_end_date"] + acols].rename(
                columns={"period_end_date": "period_end"}).dropna(subset=["period_end"])
            a["period_end"] = pd.to_datetime(a["period_end"])
            q = pd.merge_asof(q.sort_values("period_end"), a.sort_values("period_end"),
                              on="period_end", direction="backward")
    return q.reset_index(drop=True)


@lru_cache(maxsize=64)
def tech_frame(sym_key: str) -> pd.DataFrame | None:
    p = os.path.join(TECH_DIR, f"{sym_key}.parquet")
    if not os.path.exists(p):
        return None
    return pd.read_parquet(p)


@lru_cache(maxsize=512)
def price_frame(gk: str) -> pd.DataFrame | None:
    p = os.path.join(OHLCV_DIR, f"{gk}.parquet")
    if not os.path.exists(p):
        return None
    return pd.read_parquet(p, columns=["date", "open", "close"]).sort_values(
        "date").reset_index(drop=True)


_REGIME = None
def regime() -> pd.DataFrame:
    global _REGIME
    if _REGIME is None:
        _REGIME = pd.read_parquet(REGIME).sort_values("date").reset_index(drop=True)
    return _REGIME


# ---------------- trigger evaluation ----------------

def eval_fund_rule(gk: str, rule: dict) -> list[dict]:
    """all-fundamental (and regime) clauses: quarterly anchor + offsets."""
    q = fund_frame(gk)
    if q is None or len(q) < 2:
        return []
    fcl = [c for c in rule["clauses"] if c["metric"] not in REGIME_COLS]
    rcl = [c for c in rule["clauses"] if c["metric"] in REGIME_COLS]
    masks = {}
    for c in fcl:
        m = c["metric"]
        if m not in q.columns:
            return []                                  # gap metric -> no triggers
        masks[id(c)] = _op(q[m], str(c["operator"]), c["threshold_value"])
    n = len(q)
    trigs = []
    max_off = max(int(c["period_offset"]) for c in fcl)
    for i in range(n):
        ok = True
        for c in fcl:
            j = i + int(c["period_offset"])
            if j >= n or not bool(masks[id(c)].iloc[j]):
                ok = False; break
        if not ok:
            continue
        j_last = i + (max_off if rule["base_date_rule"] == "last_clause" else 0)
        base = q["announcement_date"].iloc[min(j_last, n - 1)]
        if pd.isna(base):
            continue
        base = pd.Timestamp(base)
        if rcl:                                        # regime gate at base date
            rg = regime()
            row = rg[rg["date"] <= base].tail(1)
            if row.empty:
                continue
            if not all(bool(_op(row[c["metric"]], str(c["operator"]),
                                c["threshold_value"]).iloc[0]) for c in rcl):
                continue
        snap = [{"metric": c["metric"], "offset": int(c["period_offset"]),
                 "op": str(c["operator"]), "thr": str(c["threshold_value"]),
                 "value": (None if pd.isna(v := q[c["metric"]].iloc[min(i + int(c['period_offset']), n-1)])
                           else round(float(v), 3))} for c in fcl]
        trigs.append({"anchor_period": str(q["period_end"].iloc[i].date()),
                      "base_date": base,
                      "base_date_estimated": bool(q.get("base_date_estimated",
                                                        pd.Series([False]*n)).iloc[min(j_last, n-1)]),
                      "mixed_basis_flag": bool(q.get("yoy_mixed_basis",
                                                     pd.Series([False]*n)).iloc[i]),
                      "clause_snapshot": json.dumps(snap)})
    return trigs


def eval_tech_rule(sym: str, rule: dict) -> list[dict]:
    """pure technical/regime: daily AND of clauses, offsets shift setup back
    by 63*offset trading days; edge-triggered episodes with re-arm."""
    t = tech_frame(sym)
    if t is None or len(t) < 60:
        return []
    cond = pd.Series(True, index=t.index)
    snap_meta = []
    for c in rule["clauses"]:
        m = c["metric"]
        if m in REGIME_COLS:
            rg = regime()[["date", m]]
            s = t[["date"]].merge(rg, on="date", how="left")[m].ffill()
        elif m in t.columns:
            s = t[m]
        else:
            return []
        mk = _op(s, str(c["operator"]), c["threshold_value"]).fillna(False)
        off = int(c["period_offset"])
        if off > 0:
            mk = mk.shift(off * REARM_DAYS).fillna(False)
        cond &= mk
        snap_meta.append((c, s))
    edge = cond & ~cond.shift(1, fill_value=False)
    idxs = list(np.where(edge.values)[0])
    trigs, last = [], -10**9
    for i in idxs:
        if i - last < REARM_DAYS:
            continue
        last = i
        d = t["date"].iloc[i]
        if d < PRICE_RELIABLE_FROM:
            lowconf = True
        else:
            lowconf = False
        snap = [{"metric": c["metric"], "offset": int(c["period_offset"]),
                 "op": str(c["operator"]), "thr": str(c["threshold_value"]),
                 "value": (None if pd.isna(v := s.iloc[max(i - int(c['period_offset'])*REARM_DAYS, 0)])
                           else round(float(v), 3))} for c, s in snap_meta]
        trigs.append({"anchor_period": "", "base_date": pd.Timestamp(d),
                      "base_date_estimated": lowconf, "mixed_basis_flag": False,
                      "clause_snapshot": json.dumps(snap)})
    return trigs


def eval_combined_rule(gk: str, sym: str, rule: dict) -> list[dict]:
    """fundamental anchor -> technical confirmation within offset window."""
    fcl = [c for c in rule["clauses"]
           if c["metric"] not in REGIME_COLS and _domain(c["metric"]) == "fund"]
    tcl = [c for c in rule["clauses"]
           if c["metric"] not in REGIME_COLS and _domain(c["metric"]) == "tech"]
    rcl = [c for c in rule["clauses"] if c["metric"] in REGIME_COLS]
    sub = dict(rule); sub["clauses"] = fcl + rcl
    base_trigs = eval_fund_rule(gk, sub) if fcl else []
    if not base_trigs or not tcl:
        return base_trigs if not tcl else []
    t = tech_frame(sym)
    if t is None:
        return []
    out = []
    for bt in base_trigs:
        base = bt["base_date"]
        ok_day = None
        for c in tcl:
            m = c["metric"]
            if m not in t.columns:
                return []
            off = max(int(c["period_offset"]), 1)
            win = t[(t["date"] > base)
                    & (t["date"] <= base + pd.Timedelta(days=int(off * 95)))]
            mk = _op(win[m], str(c["operator"]), c["threshold_value"]).fillna(False)
            if not mk.any():
                ok_day = None; break
            d = win.loc[mk.idxmax(), "date"]
            ok_day = d if ok_day is None else max(ok_day, d)
        if ok_day is None:
            continue
        snap = json.loads(bt["clause_snapshot"])
        row_at = t[t["date"] == ok_day]
        for c in tcl:
            v = row_at[c["metric"]].iloc[0] if not row_at.empty and c["metric"] in row_at else None
            snap.append({"metric": c["metric"], "offset": int(c["period_offset"]),
                         "op": str(c["operator"]), "thr": str(c["threshold_value"]),
                         "value": None if v is None or pd.isna(v) else round(float(v), 3)})
        nb = dict(bt)
        nb["base_date"] = pd.Timestamp(ok_day)          # entry = tape confirmation
        nb["clause_snapshot"] = json.dumps(snap)
        out.append(nb)
    return out


_TECH_HINTS = ("price", "volume", "rsi", "ma_", "bb_", "atr", "beta", "volatility",
               "momentum", "rel_strength", "consecutive", "drawdown", "recovery",
               "pct_from", "pct_position", "days_since", "single_day", "monthly_return")


def _domain(metric: str) -> str:
    m = metric.lower()
    if any(h in m for h in _TECH_HINTS) and not m.startswith("price_to_book"):
        # market_cap_cr lives in fund store
        if m == "market_cap_cr":
            return "fund"
        return "tech"
    return "fund"


def rule_kind(rule: dict) -> str:
    doms = {(_domain(c["metric"]) if c["metric"] not in REGIME_COLS else "regime")
            for c in rule["clauses"]}
    if doms <= {"fund", "regime"}:
        return "fund"
    if doms <= {"tech", "regime"}:
        return "tech"
    return "combined"


# ---------------- paths + scorecard ----------------

def build_paths(gk: str, trigs: list[dict], rule_id: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    px = price_frame(gk)
    tr_rows, path_rows = [], []
    if px is None or px.empty:
        return pd.DataFrame(), pd.DataFrame()
    dates = px["date"].values
    for k, bt in enumerate(trigs):
        base = bt["base_date"]
        i = int(np.searchsorted(dates, np.datetime64(base), side="right"))
        if i >= len(px):
            continue
        entry_date = px["date"].iloc[i]
        entry = float(px["open"].iloc[i]) if px["open"].iloc[i] > 0 else float(px["close"].iloc[i])
        if entry <= 0:
            continue
        tid = f"{rule_id}|{gk}|{str(pd.Timestamp(base).date())}|{k}"
        closes = px["close"].values
        peak = -1e18
        for m in range(1, 121):
            j = i + m * TDAYS_PER_M
            if j >= len(px):
                break
            ret = (closes[j] / entry - 1) * 100
            peak = max(peak, ret)
            path_rows.append({"trigger_id": tid, "month": m,
                              "date": px["date"].iloc[j],
                              "close": round(float(closes[j]), 3),
                              "ret_pct": round(float(ret), 2),
                              "peak_ret_pct": round(float(peak), 2),
                              "drawdown_pct": round(float((closes[j]/max(closes[i:j+1].max(),1e-9)-1)*100), 2)})
        tr_rows.append({"trigger_id": tid, "guru_key": gk,
                        "anchor_period": bt["anchor_period"],
                        "base_date": pd.Timestamp(base),
                        "entry_date": entry_date, "entry_price": round(entry, 3),
                        "close_at_trigger": round(float(closes[max(i-1, 0)]), 3),
                        "base_date_estimated": bt["base_date_estimated"],
                        "mixed_basis_flag": bt["mixed_basis_flag"],
                        "clause_snapshot": bt["clause_snapshot"]})
    return pd.DataFrame(tr_rows), pd.DataFrame(path_rows)


def scorecard_rows(rule_id: str, trig: pd.DataFrame, paths: pd.DataFrame) -> list[dict]:
    out = []
    if trig.empty:
        return out
    trig["_ep"] = trig["guru_key"] + "_" + pd.to_datetime(trig["base_date"]).dt.to_period("Q").astype(str)
    ep_by_tid = dict(zip(trig["trigger_id"], trig["_ep"]))
    for hm in HORIZONS_M:
        p = paths[paths["month"] == hm]
        if p.empty:
            out.append({"rule_id": rule_id, "horizon": f"{hm}M", "n_companies": 0,
                        "n_triggers": 0, "n_episodes": 0})
            continue
        m = p.merge(trig[["trigger_id", "guru_key"]], on="trigger_id", how="left")
        r = p["ret_pct"]
        # episodes deduped WITHIN this horizon's measurable set, so the invariant
        # n_companies <= n_episodes <= n_triggers always holds (bug fix 2026-07-xx)
        n_episodes = p["trigger_id"].map(ep_by_tid).nunique()
        upto = paths[paths["month"] <= hm].groupby("trigger_id")
        dd = upto["drawdown_pct"].min()
        peak = upto["peak_ret_pct"].max()
        fin = r.set_axis(p["trigger_id"])
        sustain = (fin / peak.reindex(fin.index).replace(0, np.nan)).clip(-5, 5)
        out.append({"rule_id": rule_id, "horizon": f"{hm}M",
                    "n_companies": int(m["guru_key"].nunique()),
                    "n_triggers": int(len(p)), "n_episodes": int(n_episodes),
                    "min_return_pct": round(float(r.min()), 2),
                    "max_return_pct": round(float(r.max()), 2),
                    "mean_return_pct": round(float(r.mean()), 2),
                    "median_return_pct": round(float(r.median()), 2),
                    "p25_return_pct": round(float(r.quantile(.25)), 2),
                    "p75_return_pct": round(float(r.quantile(.75)), 2),
                    "success_prob_pos": round(float((r > 0).mean()) * 100, 1),
                    "success_prob_2x": round(float((r >= 100).mean()) * 100, 1),
                    "success_prob_5x": round(float((r >= 400).mean()) * 100, 1),
                    "success_prob_10x": round(float((r >= 900).mean()) * 100, 1),
                    "median_max_drawdown_pct": round(float(dd.median()), 2),
                    "pct_dropped_big": round(float((dd <= -50).mean()) * 100, 1),
                    "sustain_ratio_median": round(float(sustain.median()), 3)})
    return out


# ---------------- main ----------------

def run_rule(rule: dict, uni: pd.DataFrame) -> tuple[int, int]:
    kind = rule_kind(rule)
    all_t, all_p = [], []
    n_company_errors = 0
    for _, ur in uni.iterrows():
        gk, sym = ur["guru_key"], ur["sym_key"]
        try:
            if kind == "fund":
                trigs = eval_fund_rule(gk, rule)
            elif kind == "tech":
                trigs = eval_tech_rule(sym, rule) if sym else []
            else:
                trigs = eval_combined_rule(gk, sym, rule) if sym else []
            if not trigs:
                continue
            td, pd_ = build_paths(gk, trigs, rule["rule_id"])
            if not td.empty:
                td["symbol"] = ur.get("nse_symbol") if isinstance(
                    ur.get("nse_symbol"), str) and ur.get("nse_symbol").strip() else gk
                all_t.append(td); all_p.append(pd_)
        except Exception as e:
            # NEVER swallow silently: >1% company errors must abort the rule
            n_company_errors += 1
            if n_company_errors <= 3:
                log(f"    company error {gk}: {str(e)[:90]}")
            if n_company_errors > max(20, len(uni) // 100):
                raise RuntimeError(
                    f"too many company errors ({n_company_errors}) — aborting "
                    f"rule; last: {str(e)[:120]}") from e
    if not all_t:
        return 0, 0
    trig = pd.concat(all_t, ignore_index=True)
    paths = pd.concat(all_p, ignore_index=True)
    os.makedirs(TRIG_DIR, exist_ok=True); os.makedirs(PATH_DIR, exist_ok=True)
    trig.drop(columns=["_ep"], errors="ignore").to_parquet(
        os.path.join(TRIG_DIR, f"{rule['rule_id']}.parquet"), index=False)
    paths.to_parquet(os.path.join(PATH_DIR, f"{rule['rule_id']}.parquet"), index=False)
    rows = scorecard_rows(rule["rule_id"], trig, paths)
    # PARALLEL-SAFE: each rule writes its OWN scorecard fragment (no shared-file
    # read-modify-write race across workers). The Excel export concatenates them.
    os.makedirs(SCORE_DIR, exist_ok=True)
    pd.DataFrame(rows).to_parquet(
        os.path.join(SCORE_DIR, f"{rule['rule_id']}.parquet"), index=False)
    return len(trig), len(paths)


def _all_ledgers() -> pd.DataFrame:
    """union of the solo ledger + every shard ledger (for status/skip logic)."""
    frames = []
    if os.path.exists(LEDGER):
        frames.append(pd.read_parquet(LEDGER))
    import glob as _g
    for f in _g.glob(os.path.join(BT_DIR, "_ledger_shard*.parquet")):
        frames.append(pd.read_parquet(f))
    if not frames:
        return pd.DataFrame(columns=["rule_id", "status", "triggers", "error"])
    allled = pd.concat(frames, ignore_index=True)
    # a rule counts as processed if ANY ledger finished it
    order = {"done": 0, "no_triggers": 1, "error": 2, "pending": 3}
    allled["_o"] = allled["status"].map(order).fillna(3)
    return (allled.sort_values("_o").drop_duplicates("rule_id", keep="first")
            .drop(columns="_o"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--rules", type=str, default="")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--retry-errors", action="store_true")
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--shard", type=str, default="",
                    help="K/N: run only rules where index %% N == K-1 "
                         "(deterministic partition; each shard has its own ledger)")
    ap.add_argument("--rebuild-scores", action="store_true",
                    help="recompute scorecard fragments from existing triggers/paths "
                         "(fast; fixes aggregates without re-running the search)")
    ap.add_argument("--rules-xlsx", type=str, default="",
                    help="alternate Rules/Clauses workbook (e.g. generated combos)")
    ap.add_argument("--outdir", type=str, default="",
                    help="route triggers/paths/scores/ledger under backtest/<outdir>/ "
                         "so a generated run never touches the original")
    args = ap.parse_args()

    # optional overrides — keep generated-combo runs fully separate from the
    # validated original backtest (additive, no pollution)
    global XLSX, TRIG_DIR, PATH_DIR, SCORE_DIR, LEDGER, SCORE
    if args.rules_xlsx:
        XLSX = args.rules_xlsx
    if args.outdir:
        base = os.path.join(BT_DIR, args.outdir)
        TRIG_DIR = os.path.join(base, "triggers")
        PATH_DIR = os.path.join(base, "paths")
        SCORE_DIR = os.path.join(base, "scores")
        LEDGER = os.path.join(base, "_ledger.parquet")
        SCORE = os.path.join(base, "scorecard.parquet")
        os.makedirs(base, exist_ok=True)

    if args.rebuild_scores:
        import glob as _g
        os.makedirs(SCORE_DIR, exist_ok=True)
        tfiles = _g.glob(os.path.join(TRIG_DIR, "*.parquet"))
        log(f"rebuilding scores for {len(tfiles)} rules from stored triggers/paths")
        for i, tf in enumerate(tfiles, 1):
            rid = os.path.basename(tf)[:-8]
            try:
                trig = pd.read_parquet(tf)
                paths = pd.read_parquet(os.path.join(PATH_DIR, f"{rid}.parquet"))
                rows = scorecard_rows(rid, trig, paths)
                pd.DataFrame(rows).to_parquet(
                    os.path.join(SCORE_DIR, f"{rid}.parquet"), index=False)
            except Exception as e:
                log(f"  {rid} rebuild error {str(e)[:60]}")
            if i % 200 == 0:
                log(f"  {i}/{len(tfiles)}")
        # drop the legacy monolithic scorecard so fragments are the single source
        if os.path.exists(SCORE):
            os.replace(SCORE, SCORE + ".bak")
        log("rebuild complete — scores/ fragments are now authoritative")
        return

    only = [x.strip() for x in args.rules.split(",") if x.strip()] or None
    rules = load_rules(only)
    os.makedirs(BT_DIR, exist_ok=True)

    ledger_path = LEDGER
    if args.shard:
        k, n = [int(x) for x in args.shard.split("/")]
        rules = [r for i, r in enumerate(sorted(rules, key=lambda x: x["rule_id"]))
                 if i % n == k - 1]
        ledger_path = os.path.join(os.path.dirname(LEDGER),
                                   f"_ledger_shard{k}of{n}.parquet")

    done_elsewhere = set()
    prev = _all_ledgers()
    if not prev.empty:
        done_elsewhere = set(prev.loc[prev["status"].isin(["done", "no_triggers"]),
                                      "rule_id"])
    if os.path.exists(ledger_path):
        led = pd.read_parquet(ledger_path)
        new = [r["rule_id"] for r in rules if r["rule_id"] not in set(led["rule_id"])]
        if new:
            led = pd.concat([led, pd.DataFrame({"rule_id": new, "status": "pending",
                             "triggers": 0, "error": ""})], ignore_index=True)
    else:
        led = pd.DataFrame({"rule_id": [r["rule_id"] for r in rules],
                            "status": "pending", "triggers": 0, "error": ""})
    # rules already finished by the solo run / another shard: mark done here too
    seed = led["status"].eq("pending") & led["rule_id"].isin(done_elsewhere)
    led.loc[seed, "status"] = "done_elsewhere"

    if args.status:
        allled = _all_ledgers()
        vc = allled["status"].value_counts().to_dict()
        done = int((~allled["status"].isin(["pending"])).sum())
        print(f"rules: {done}/{len(allled)} processed "
              f"({100*done/max(len(allled),1):.1f}%) | {vc} "
              f"| total triggers: {int(allled['triggers'].sum()):,}")
        return

    uni = pd.read_parquet(os.path.join(DATA_DIR, "universe_hist.parquet"),
                          columns=["guru_key", "nse_symbol"])
    have_q = {f[:-8] for f in os.listdir(QUNI_DIR)} if os.path.isdir(QUNI_DIR) else set()
    have_t = {f[:-8] for f in os.listdir(TECH_DIR)} if os.path.isdir(TECH_DIR) else set()
    uni["sym_key"] = uni["guru_key"].where(uni["guru_key"].isin(have_t))
    # technical store files are keyed by guru_key (same as ohlcv) — confirmed
    uni = uni[uni["guru_key"].isin(have_q | have_t)].reset_index(drop=True)
    log(f"universe in scope: {len(uni)} companies")

    todo_mask = led["status"].eq("pending")
    if args.retry_errors:
        todo_mask |= led["status"].eq("error")
    todo_ids = led.loc[todo_mask, "rule_id"].tolist()
    if args.limit:
        todo_ids = todo_ids[: args.limit]
    rules_by_id = {r["rule_id"]: r for r in rules}
    log(f"rules to run: {len(todo_ids)} of {len(led)}")
    if args.dry_run:
        for rid in todo_ids[:10]:
            r = rules_by_id.get(rid)
            print(f"   {rid} [{rule_kind(r) if r else '?'}] {str(r['name'])[:60] if r else ''}")
        return

    for i, rid in enumerate(todo_ids, 1):
        r = rules_by_id.get(rid)
        if r is None:
            continue
        li = led.index[led["rule_id"] == rid][0]
        t0 = datetime.now()
        try:
            nt, npth = run_rule(r, uni)
            led.at[li, "status"] = "done" if nt else "no_triggers"
            led.at[li, "triggers"] = nt
            led.at[li, "error"] = ""
            log(f"[{i}/{len(todo_ids)}] {rid} [{rule_kind(r)}]: {nt} triggers "
                f"({(datetime.now()-t0).seconds}s)")
        except Exception as e:
            led.at[li, "status"] = "error"; led.at[li, "error"] = str(e)[:200]
            log(f"[{i}/{len(todo_ids)}] {rid} ERROR {str(e)[:80]}")
        led.to_parquet(ledger_path, index=False)
    log("RUN COMPLETE")
    print(led["status"].value_counts().to_dict())


if __name__ == "__main__":
    main()
