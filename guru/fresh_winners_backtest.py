r"""
FRESH WINNERS BACKTEST — fresh_winners_backtest.py  (Project Guru, open task O7)

Does "strong results + a strong chart in the same few days" pick stocks that keep going?
Backtest BEFORE any mail (plan .claude/plans/o7-fresh-winners-backtest.md).

    STRONG RESULTS  sales YoY > 30% AND PAT YoY > 30% AND EPS YoY > 30%     (user 2026-10-18)
                    and year-ago PAT > 0 (a loss turning into a profit is not "growth" -
                    the same rule as scripts/results_reaction.yoy_of)
    TECHNICAL       on one of the first 3 trading days from the announcement:
                    (A) close >  highest close of the prior 250 days (a new 52-week high), or
                    (B) close >  highest close of the prior 60 days AND volume >= 2x the prior
                        50-day average                                  (60-day base: user)
    ENTRY / EXIT    next day's open after the trigger; close 21 / 63 / 126 trading days later

Controls reported alongside: strong results with NO chart filter (entry the day after the
announcement) and A vs B separately; split by period (<=2022 vs >=2023) and by whether the
announcement date was recorded or estimated. Absolute returns (binding Guru v1); excess vs
Nifty 50 as an extra column. Nothing is dropped for small n - n is always shown.

Local, read-only inputs: guru/data/metrics/quarterly_unified, guru/data/ohlcv_hist,
guru/data/macro_hist/NIFTY_50.parquet. Output: guru/backtest/fresh_winners/ (gitignored).

    python guru/fresh_winners_backtest.py --self-test
    python guru/fresh_winners_backtest.py --limit 200      # quick sample
    python guru/fresh_winners_backtest.py                  # all companies
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

GURU = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(GURU, "data")
OUT = os.path.join(GURU, "backtest", "fresh_winners")


def _paths(data_dir: str, out_dir: str | None):
    return (os.path.join(data_dir, "metrics", "quarterly_unified"),
            os.path.join(data_dir, "ohlcv_hist"),
            os.path.join(data_dir, "macro_hist", "NIFTY_50.parquet"),
            out_dir or os.path.join(os.path.dirname(data_dir), "backtest", "fresh_winners"))

STRONG_YOY = 30.0
WINDOW_DAYS = 3
BASE_DAYS = 60
HIGH_DAYS = 250
VOL_MULT = 2.0
VOL_DAYS = 50
MIN_TURNOVER_CR = 1.0          # Guru daily screen's liquidity floor (Rs cr/day, 20d avg)
HORIZONS = {"1M": 21, "3M": 63, "6M": 126}
SPLIT_DATE = pd.Timestamp("2023-01-01")


def strong_events(q: pd.DataFrame) -> pd.DataFrame:
    """Quarters with strong results and a positive year-ago PAT. Pure."""
    if q is None or q.empty:
        return pd.DataFrame()
    q = q.sort_values("period_end").reset_index(drop=True)
    q["pat_yago"] = q["pat_cr"].shift(4)
    pe = pd.to_datetime(q["period_end"])
    # the row 4 back must really be the same quarter a year earlier
    ok_gap = (pe - pe.shift(4)).dt.days.between(350, 380)
    m = ((q["sales_yoy_pct"] > STRONG_YOY) & (q["pat_yoy_pct"] > STRONG_YOY)
         & (q["eps_yoy_pct"] > STRONG_YOY) & (q["pat_yago"] > 0) & ok_gap
         & q["announcement_date"].notna())
    return q[m].copy()


def technical_flags(px: pd.DataFrame) -> pd.DataFrame:
    """Per trading day: A (52w high), B (60d base breakout on 2x vol), 20d turnover. Pure."""
    px = px.sort_values("date").reset_index(drop=True)
    c, v = px["close"].astype(float), px["volume"].astype(float)
    hi250 = c.shift(1).rolling(HIGH_DAYS, min_periods=200).max()
    hi60 = c.shift(1).rolling(BASE_DAYS, min_periods=BASE_DAYS).max()
    vol50 = v.shift(1).rolling(VOL_DAYS, min_periods=40).mean()
    px["A"] = c > hi250          # a NEW high: equalling the old one is not a breakout
    px["B"] = (c > hi60) & (v >= VOL_MULT * vol50)
    px["turnover_cr"] = (c * v).shift(1).rolling(20, min_periods=15).mean() / 1e7
    return px


def _ret(px, e, h):
    x = e + h
    if e >= len(px) or x >= len(px):
        return None
    o = float(px["open"].iat[e])
    return None if o <= 0 else (float(px["close"].iat[x]) / o - 1) * 100.0


def _nifty_ret(nifty: pd.Series, d0, d1):
    a, b = nifty[nifty.index <= d0], nifty[nifty.index <= d1]
    if a.empty or b.empty:
        return None
    return (float(b.iloc[-1]) / float(a.iloc[-1]) - 1) * 100.0


def events_for(key: str, q: pd.DataFrame, px: pd.DataFrame, nifty: pd.Series) -> list[dict]:
    """Trigger + control events for one company. Pure (no I/O)."""
    ev = strong_events(q)
    if ev.empty or px is None or px.empty:
        return []
    px = technical_flags(px)
    dates = pd.to_datetime(px["date"]).values
    out = []
    for _, r in ev.iterrows():
        d0 = np.datetime64(pd.Timestamp(r["announcement_date"]))
        i0 = int(np.searchsorted(dates, d0))          # first trading day on/after d0
        if i0 >= len(px):
            continue
        base = {"key": key, "period_end": r["period_end"], "announcement": r["announcement_date"],
                "est_date": bool(r.get("base_date_estimated", False)),
                "sales_yoy": r["sales_yoy_pct"], "pat_yoy": r["pat_yoy_pct"],
                "eps_yoy": r["eps_yoy_pct"]}
        # control: no chart filter, enter the day after the announcement day
        legs = [("control", i0, "")]
        for k in range(i0, min(i0 + WINDOW_DAYS, len(px))):
            a, b = bool(px["A"].iat[k]), bool(px["B"].iat[k])
            if a or b:
                legs.append(("trigger", k, "A+B" if a and b else "A" if a else "B"))
                break
        for kind, k, which in legs:
            e = k + 1
            if e >= len(px):
                continue
            row = dict(base, kind=kind, which=which,
                       trigger_date=pd.Timestamp(dates[k]).date(),
                       turnover_cr=px["turnover_cr"].iat[k])
            for h, n in HORIZONS.items():
                ret = _ret(px, e, n)
                row[f"ret_{h}"] = ret
                if ret is not None and nifty is not None and e + n < len(px):
                    nr = _nifty_ret(nifty, pd.Timestamp(dates[e]), pd.Timestamp(dates[e + n]))
                    row[f"xs_{h}"] = None if nr is None else ret - nr
                else:
                    row[f"xs_{h}"] = None
            out.append(row)
    return out


def summarise(ev: pd.DataFrame) -> pd.DataFrame:
    """One row per (group, horizon): n, companies, hit rate, median, p25/p75, p10, excess."""
    rows = []
    if ev.empty:
        return pd.DataFrame()
    ev = ev.copy()
    ev["liquid"] = ev["turnover_cr"] >= MIN_TURNOVER_CR
    ev["period"] = np.where(pd.to_datetime(ev["announcement"]) >= SPLIT_DATE,
                            ">=2023 (out-of-sample)", "<=2022")
    groups = [("control: strong results, no chart filter", ev["kind"] == "control"),
              ("TRIGGER: strong results + chart (A or B)", ev["kind"] == "trigger"),
              ("  trigger via A (52-week high)", (ev["kind"] == "trigger") & ev["which"].str.contains("A")),
              ("  trigger via B (60d base, 2x volume)", (ev["kind"] == "trigger") & ev["which"].str.contains("B"))]
    for liq_name, liq in (("liquid (>= Rs 1 cr/day)", ev["liquid"]), ("all", ev["liquid"] | ~ev["liquid"])):
        for gname, gm in groups:
            for per in ("all periods", "<=2022", ">=2023 (out-of-sample)"):
                pm = ev["period"].notna() if per == "all periods" else ev["period"] == per
                for dq in ("all dates", "recorded dates only"):
                    dm = ev["est_date"].notna() if dq == "all dates" else ~ev["est_date"]
                    sub = ev[gm & liq & pm & dm]
                    for h in HORIZONS:
                        r = pd.to_numeric(sub[f"ret_{h}"], errors="coerce").dropna()
                        x = pd.to_numeric(sub[f"xs_{h}"], errors="coerce").dropna()
                        rows.append({"universe": liq_name, "group": gname, "period": per,
                                     "dates": dq, "horizon": h, "n": len(r),
                                     "companies": sub.loc[r.index, "key"].nunique(),
                                     "hit_rate_pct": round((r > 0).mean() * 100, 1) if len(r) else None,
                                     "median_pct": round(r.median(), 2) if len(r) else None,
                                     "p25_pct": round(r.quantile(.25), 2) if len(r) else None,
                                     "p75_pct": round(r.quantile(.75), 2) if len(r) else None,
                                     "p10_pct": round(r.quantile(.10), 2) if len(r) else None,
                                     "median_vs_nifty_pct": round(x.median(), 2) if len(x) else None})
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="first N companies only (quick sample)")
    ap.add_argument("--data-dir", default=DATA,
                    help="guru data folder (the local, gitignored guru/data)")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        return _self_test()
    QDIR, PDIR, NIFTY, OUT = _paths(a.data_dir, a.out_dir)
    keys = sorted(f[:-8] for f in os.listdir(QDIR) if f.endswith(".parquet") and len(f) > 8)
    keys = [k for k in keys if os.path.exists(os.path.join(PDIR, f"{k}.parquet"))]
    if a.limit:
        keys = keys[:a.limit]
    nf = pd.read_parquet(NIFTY)
    nifty = nf.assign(date=pd.to_datetime(nf["date"])).set_index("date")["close"].astype(float).sort_index()
    allev = []
    for i, k in enumerate(keys, 1):
        try:
            q = pd.read_parquet(os.path.join(QDIR, f"{k}.parquet"))
            px = pd.read_parquet(os.path.join(PDIR, f"{k}.parquet"), columns=["date", "open", "high",
                                                                              "low", "close", "volume"])
            allev += events_for(k, q, px, nifty)
        except Exception as e:                              # noqa: BLE001
            print(f"  {k}: skipped ({str(e)[:60]})")
        if i % 500 == 0:
            print(f"  {i}/{len(keys)} companies, {len(allev)} event rows")
    ev = pd.DataFrame(allev)
    os.makedirs(OUT, exist_ok=True)
    ev.to_parquet(os.path.join(OUT, "events.parquet"), index=False)
    s = summarise(ev)
    with pd.ExcelWriter(os.path.join(OUT, "summary.xlsx")) as xw:
        s.to_excel(xw, "Summary", index=False)
        ev.to_excel(xw, "Events", index=False)
    head = s[(s["universe"].str.startswith("liquid")) & (s["dates"] == "all dates")
             & (s["period"] == "all periods")]
    print(f"\n{len(keys)} companies, {len(ev)} event rows "
          f"({int((ev['kind'] == 'trigger').sum()) if len(ev) else 0} triggers)")
    print(head[["group", "horizon", "n", "companies", "hit_rate_pct", "median_pct",
                "p25_pct", "p75_pct", "median_vs_nifty_pct"]].to_string(index=False))
    print(f"\nwrote {OUT}")
    return 0


def _self_test() -> int:
    ok = fail = 0

    def check(name, cond):
        nonlocal ok, fail
        ok, fail = (ok + 1, fail) if cond else (ok, fail + 1)
        print(f"  {'ok  ' if cond else 'FAIL'} {name}")

    pe = pd.to_datetime(["2024-06-30", "2024-09-30", "2024-12-31", "2025-03-31", "2025-06-30"])
    q = pd.DataFrame({"period_end": pe, "announcement_date": pe + pd.Timedelta(days=40),
                      "base_date_estimated": False, "pat_cr": [10, 11, 12, 13, 15],
                      "sales_yoy_pct": [0, 0, 0, 0, 40], "pat_yoy_pct": [0, 0, 0, 0, 50],
                      "eps_yoy_pct": [0, 0, 0, 0, 45]})
    se = strong_events(q)
    check("strong = all three > 30% with positive year-ago PAT", len(se) == 1)
    q2 = q.assign(pat_cr=[-5, 11, 12, 13, 15])
    check("loss a year ago -> not strong", strong_events(q2).empty)
    n = 600
    d = pd.bdate_range("2024-01-01", periods=n)
    close = np.full(n, 100.0)
    ann = pd.Timestamp(q["announcement_date"].iloc[-1])
    i0 = int(np.searchsorted(d.values, np.datetime64(ann)))
    close[i0 + 1:] = 130.0                                    # new high the day after
    vol = np.full(n, 1e6)
    px = pd.DataFrame({"date": d, "open": close, "high": close, "low": close, "close": close,
                       "volume": vol})
    nifty = pd.Series(100.0, index=d)
    ev = events_for("X", q, px, nifty)
    trig = [e for e in ev if e["kind"] == "trigger"]
    check("52-week high on day 2 of the window -> trigger via A",
          len(trig) == 1 and "A" in trig[0]["which"])
    check("control leg always present", any(e["kind"] == "control" for e in ev))
    check("entry next open, 1M return computed", trig and trig[0]["ret_1M"] == 0.0)
    flat = px.assign(close=100.0, open=100.0)
    check("no new high and no volume -> no trigger",
          not [e for e in events_for("X", q, flat, nifty) if e["kind"] == "trigger"])
    s = summarise(pd.DataFrame(ev))
    check("summary has n and hit rate per horizon", {"n", "hit_rate_pct", "median_pct"} <= set(s.columns))
    print(f"\nfresh_winners_backtest self-test: {ok} passed, {fail} failed")
    return 1 if fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
