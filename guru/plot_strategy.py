r"""
plot_strategy.py — replay any backtested trigger as a chart (Project Guru).

For a rule (strategy) and one of its triggers, draws the full decision context:
  panel 1: price (log), 50/200 DMA, entry marker, horizon checkpoints,
           the stored forward path (peak / drawdown shading)
  panel 2: quarterly sales + PAT bars with YoY line (fundamental history)
  panel 3: the clause snapshot that fired (the WHY of the entry)
Saves PNG to guru/backtest/charts/<rule>__<symbol>__<date>.png

Usage:
    python guru/plot_strategy.py --rule FUND_GROWTH_001                  # list triggers
    python guru/plot_strategy.py --rule FUND_GROWTH_001 --pick best     # top by 24M ret
    python guru/plot_strategy.py --rule FUND_GROWTH_001 --pick worst
    python guru/plot_strategy.py --rule FUND_GROWTH_001 --trigger <trigger_id>
    python guru/plot_strategy.py --rule FUND_GROWTH_001 --symbol TCS
"""
from __future__ import annotations

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

GURU_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(GURU_DIR, "data")
BT_DIR = os.path.join(GURU_DIR, "backtest")
CHART_DIR = os.path.join(BT_DIR, "charts")


def load_trigger(rule: str, trigger_id: str | None, symbol: str | None, pick: str | None):
    t = pd.read_parquet(os.path.join(BT_DIR, "triggers", f"{rule}.parquet"))
    p = pd.read_parquet(os.path.join(BT_DIR, "paths", f"{rule}.parquet"))
    if trigger_id:
        row = t[t["trigger_id"] == trigger_id]
    elif symbol:
        row = t[t["symbol"].astype(str).str.upper() == symbol.upper()].head(1)
    elif pick in ("best", "worst"):
        h = p[p["month"] == 24]
        h = h[h["trigger_id"].isin(t["trigger_id"])]
        h = h.sort_values("ret_pct", ascending=(pick == "worst"))
        row = t[t["trigger_id"] == h["trigger_id"].iloc[0]] if not h.empty else t.head(1)
    else:
        return t, p, None
    if row.empty:
        raise SystemExit("trigger not found")
    return t, p, row.iloc[0]


_NAME = None
def company_name(gk: str, fallback: str) -> str:
    """resolve a human company NAME from universe_hist (not ISIN)."""
    global _NAME
    if _NAME is None:
        u = pd.read_parquet(os.path.join(DATA_DIR, "universe_hist.parquet"),
                            columns=["guru_key", "name", "nse_symbol"])
        _NAME = u.set_index("guru_key")
    if gk in _NAME.index:
        nm = _NAME.loc[gk, "name"]
        if isinstance(nm, str) and nm.strip():
            return nm.strip()
    return fallback


def _safe(s: str) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in str(s))


def render(rule: str, row: pd.Series, p: pd.DataFrame, out_dir: str,
           tag: str = "") -> str:
    gk = row["guru_key"]
    name = company_name(gk, str(row.get("symbol", gk)))
    entry_date = pd.Timestamp(row["entry_date"])
    path = p[p["trigger_id"] == row["trigger_id"]].sort_values("month")

    px = pd.read_parquet(os.path.join(DATA_DIR, "ohlcv_hist", f"{gk}.parquet"))
    px["date"] = pd.to_datetime(px["date"])
    lo = entry_date - pd.Timedelta(days=730)
    hi = min(entry_date + pd.Timedelta(days=1830), px["date"].max() + pd.Timedelta(days=5))
    w = px[(px["date"] >= lo) & (px["date"] <= hi)].reset_index(drop=True)
    w["ma50"] = px.set_index("date")["close"].rolling(50).mean().reindex(w["date"]).values
    w["ma200"] = px.set_index("date")["close"].rolling(200).mean().reindex(w["date"]).values

    qf = os.path.join(DATA_DIR, "metrics", "quarterly_unified", f"{gk}.parquet")
    q = pd.read_parquet(qf) if os.path.exists(qf) else pd.DataFrame()
    if not q.empty:
        q = q[(q["period_end"] >= lo) & (q["period_end"] <= hi)]
    snap = json.loads(row["clause_snapshot"])

    fig = plt.figure(figsize=(15, 10))
    gs = fig.add_gridspec(3, 1, height_ratios=[3, 1.4, 0.8], hspace=0.28)
    ax = fig.add_subplot(gs[0])
    ax.plot(w["date"], w["close"], lw=1.2, color="#1f77b4", label="close (adj)")
    ax.plot(w["date"], w["ma50"], lw=0.8, color="#ff7f0e", label="50DMA")
    ax.plot(w["date"], w["ma200"], lw=0.8, color="#2ca02c", label="200DMA")
    ax.set_yscale("log")
    ax.axvline(entry_date, color="red", lw=1.2, ls="--")
    ax.scatter([entry_date], [row["entry_price"]], marker="^", s=140, color="red",
               zorder=5, label=f"ENTRY {row['entry_price']}")
    for hm in (6, 12, 24, 36):
        pr = path[path["month"] == hm]
        if not pr.empty:
            d = pd.Timestamp(pr["date"].iloc[0]); c = pr["close"].iloc[0]
            ax.scatter([d], [c], marker="o", s=45, color="purple", zorder=5)
            ax.annotate(f"{hm}M: {pr['ret_pct'].iloc[0]:+.0f}%", (d, c),
                        textcoords="offset points", xytext=(4, 8), fontsize=8)
    pk = path.loc[path["peak_ret_pct"].idxmax()] if not path.empty else None
    if pk is not None:
        ax.annotate(f"peak {pk['peak_ret_pct']:+.0f}%",
                    (pd.Timestamp(pk['date']), pk['close']), color="darkgreen",
                    fontsize=9, fontweight="bold",
                    textcoords="offset points", xytext=(4, -12))
    ax.legend(loc="upper left", fontsize=8)
    sym = str(row.get("symbol", "")) if str(row.get("symbol", "")) != gk else ""
    ttl = f"{rule}  |  {name}" + (f" ({sym})" if sym else "")
    ax.set_title(f"{ttl}  |  entry {entry_date.date()} @ {row['entry_price']}"
                 + (f"   [{tag}]" if tag else ""), fontsize=12, fontweight="bold")

    ax2 = fig.add_subplot(gs[1], sharex=ax)
    if not q.empty:
        ax2.bar(q["period_end"] - pd.Timedelta(days=22), q["sales_cr"], width=40,
                color="#9ecae1", label="sales (cr)")
        ax2.bar(q["period_end"] + pd.Timedelta(days=22), q["pat_cr"], width=40,
                color="#3182bd", label="PAT (cr)")
        ax2.legend(loc="upper left", fontsize=8)
        ax3 = ax2.twinx()
        ax3.plot(q["period_end"], q["sales_yoy_pct"], color="darkred", lw=1.1,
                 marker=".", label="sales YoY%")
        ax3.axhline(0, color="grey", lw=0.5); ax3.legend(loc="upper right", fontsize=8)
        ax2.axvline(entry_date, color="red", lw=1.0, ls="--")
    ax2.set_ylabel("quarterly")

    ax4 = fig.add_subplot(gs[2]); ax4.axis("off")
    lines = ["WHY (clause snapshot at trigger):"]
    for c in snap:
        lines.append(f"   {c['metric']} [offset {c['offset']}] {c['op']} {c['thr']}"
                     f"   ->   actual = {c['value']}")
    flags = []
    if row.get("base_date_estimated"): flags.append("base_date ESTIMATED")
    if row.get("mixed_basis_flag"): flags.append("mixed basis YoY")
    if flags: lines.append("   flags: " + ", ".join(flags))
    ax4.text(0.01, 0.9, "\n".join(lines), fontsize=9, family="monospace",
             va="top", transform=ax4.transAxes)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

    os.makedirs(out_dir, exist_ok=True)
    fname = f"{_safe(name)}__{entry_date.date()}" + (f"__{tag}" if tag else "") + ".png"
    out = os.path.join(out_dir, fname)
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return out


def pick_row(t: pd.DataFrame, p: pd.DataFrame, which: str, horizon_m: int = 24):
    h = p[p["month"] == horizon_m]
    h = h[h["trigger_id"].isin(t["trigger_id"])]
    if h.empty:
        return t.iloc[0] if which != "median" else t.iloc[len(t)//2]
    h = h.sort_values("ret_pct")
    if which == "best":
        tid = h["trigger_id"].iloc[-1]
    elif which == "worst":
        tid = h["trigger_id"].iloc[0]
    else:  # median
        tid = h["trigger_id"].iloc[len(h)//2]
    return t[t["trigger_id"] == tid].iloc[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rule", default=None)
    ap.add_argument("--rules", default=None, help="comma-separated rule ids (batch)")
    ap.add_argument("--trigger", default=None)
    ap.add_argument("--symbol", default=None)
    ap.add_argument("--pick", default=None, choices=[None, "best", "worst", "median"])
    ap.add_argument("--picks", default="best,worst,median",
                    help="which examples per rule in batch mode")
    ap.add_argument("--horizon", type=int, default=24, help="ranking horizon (months)")
    args = ap.parse_args()

    # ---- batch mode: comma-separated rules -> per-rule folders ----
    if args.rules:
        rule_ids = [r.strip() for r in args.rules.split(",") if r.strip()]
        picks = [x.strip() for x in args.picks.split(",") if x.strip()]
        for rid in rule_ids:
            tf = os.path.join(BT_DIR, "triggers", f"{rid}.parquet")
            if not os.path.exists(tf):
                print(f"  {rid}: no triggers yet (rule not run) — skipped")
                continue
            t = pd.read_parquet(tf)
            p = pd.read_parquet(os.path.join(BT_DIR, "paths", f"{rid}.parquet"))
            if t.empty:
                print(f"  {rid}: 0 triggers — skipped"); continue
            out_dir = os.path.join(CHART_DIR, rid)
            for which in picks:
                try:
                    row = pick_row(t, p, which, args.horizon)
                    out = render(rid, row, p, out_dir, tag=which)
                    print(f"  {rid} [{which}] -> {out}")
                except Exception as e:
                    print(f"  {rid} [{which}] ERROR {str(e)[:80]}")
        return

    # ---- single mode ----
    t, p, row = load_trigger(args.rule, args.trigger, args.symbol, args.pick)
    if row is None:
        print(f"{args.rule}: {len(t)} triggers. Examples:")
        print(t[["trigger_id", "symbol", "base_date", "entry_price"]].head(15).to_string(index=False))
        return
    out = render(args.rule, row, p, os.path.join(CHART_DIR, args.rule),
                 tag=(args.pick or ""))
    print("saved:", out)


if __name__ == "__main__":
    main()
