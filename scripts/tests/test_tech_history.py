r"""
test_tech_history.py — offline checks for scripts/tech_history.py

Plain python, no pytest (repo convention, see test_phase1_engine.py).
Synthetic series only: no Drive, no network.

  1. counting machinery: run lengths, episodes (cooldown), forward stats, percentile
  2. join_series: Drive wins, split-mismatch rescale, too-small overlap -> Drive only
  3. bad-data guard: a >40% one-day jump is not a "big day"
  4. personality counts match hand counts
  5. unusual: streak / MA-run counts match an independent loop; too-new -> nothing
  6. patterns: golden cross + 52w-high-on-volume fire where built; NO look-ahead
  7. vs_market: beta, outperformance -> RS 52w high + Leading; stage

Run:
    python scripts/tests/test_tech_history.py
Exit code = number of failures.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import tech_history as T  # noqa: E402

_FAILS: list[str] = []
_N = [0]


def check(name, cond, detail=""):
    _N[0] += 1
    if cond:
        print(f"  ok   {name}")
    else:
        print(f"  FAIL {name}  {detail}")
        _FAILS.append(name)


def mk(close, start="2015-01-01", vol=None):
    close = np.asarray(close, dtype=float)
    d = pd.bdate_range(start, periods=len(close))
    return pd.DataFrame({"date": d, "open": close, "high": close * 1.01,
                         "low": close * 0.99, "close": close,
                         "volume": np.full(len(close), 1000.0) if vol is None else vol})


def walk(n, seed=1, drift=0.0003, sd=0.02):
    rng = np.random.default_rng(seed)
    return 100 * np.cumprod(1 + rng.normal(drift, sd, n))


print("1) counting machinery")
rl = T.run_lengths(pd.Series([True, True, False, False, False, True])).tolist()
check("run_lengths", rl == [1, 2, 1, 2, 3, 1], rl)
ep = T.episodes(pd.Series([1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1], dtype=bool), cooldown=5).tolist()
check("episodes cooldown merges a cluster", ep == [0, 10], ep)
fs = T.forward_stats(pd.Series([100.0, 100, 200, 50]), [0, 1, 2], horizons=(2,))[2]
check("forward_stats only uses known outcomes", fs["n"] == 2, fs)
check("forward_stats median/up/worst", abs(fs["median"] - 0.25) < 1e-9
      and fs["pct_up"] == 50 and abs(fs["worst"] + 0.5) < 1e-9, fs)
check("pct_rank", T.pct_rank(pd.Series([1, 2, 3, 4]), 3) == 62.5)

print("2) join_series")
loc = mk(np.full(60, 200.0))                              # pre-split basis
drv = mk(np.full(60, 100.0), start=loc["date"].iloc[30])  # 30 shared days
j, note = T.join_series(loc, drv)
check("join length", len(j) == 90, len(j))
check("local rescaled to Drive basis", abs(j["close"].iloc[0] - 100) < 1e-6
      and "rescaled" in note, note)
check("Drive wins on shared dates", (j["close"].iloc[30:] == 100).all())
j2, note2 = T.join_series(mk(np.full(60, 100.0)), mk(np.full(60, 100.0),
                                                     start=loc["date"].iloc[50]))
check("small overlap -> Drive only", len(j2) == 60 and "Drive only" in note2, note2)
j3, note3 = T.join_series(mk(np.full(60, 100.0)), mk(np.full(60, 100.0), start=loc["date"].iloc[30]))
check("matching basis -> no rescale", "rescaled" not in note3 and len(j3) == 90, note3)

print("3) bad-data guard")
c = np.full(50, 100.0); c[25:] = 170.0                     # +70% overnight = missed split
r = T.returns(mk(c))
check("jump > 40% is NaN, not a big day", pd.isna(r.iloc[25]))

print("4) personality")
c = [100.0]
for i in range(1, 600):
    c.append(c[-1] * (1.04 if i % 50 == 0 else 1.001))     # 11 days of +4%
p = T.personality(mk(c))
n_big = round(p["big_days_per_year"]["+3"] * (599 / T.YEAR))
check("+3% day count", n_big == 11, n_big)
check("never a -3% day", p["big_days_per_year"]["-3"] == 0)
check("always above 50-day after warm-up", p["ma"][50]["pct_time_above"] > 99,
      p["ma"][50]["pct_time_above"])

print("5) unusual")
base = list(walk(500, seed=3))
for _ in range(7):
    base.append(base[-1] * 0.99)                           # 7 straight down closes
df = mk(base)
facts = {f["key"]: f for f in T.unusual(df)}
st = facts.get("streak")
check("streak found", st is not None and st["value"] >= 7 and st["side"] == "bear",
      st and (st["value"], st["side"]))
# independent count of past down streaks that reached the current length
rr = df["close"].pct_change()
sg = np.sign(rr).to_numpy()
L = st["value"]
runs, cur = [], 0
for i in range(1, len(sg) - L):                            # strictly before today's run
    cur = cur + 1 if sg[i] == -1 and (i > 1 and sg[i - 1] == -1) else (1 if sg[i] == -1 else 0)
    if cur == L:
        runs.append(i)
check("streak episodes = independent count", st["n"] == len(runs), (st["n"], len(runs)))

ma50 = df["close"].rolling(50).mean()
state = (df["close"] > ma50).where(ma50.notna())
f50 = facts["ma50_run"]
# independent: past runs of the same state, their full lengths
lens, prev, cnt = [], None, 0
vals = state.tolist()
cur_len = f50["value"]
start = len(vals) - cur_len
for i, v in enumerate(vals[:start]):
    if pd.isna(v):
        if prev is not None and cnt:
            lens.append((prev, cnt))
        prev, cnt = None, 0
        continue
    if v == prev:
        cnt += 1
    else:
        if prev is not None:
            lens.append((prev, cnt))
        prev, cnt = v, 1
if prev is not None:
    lens.append((prev, cnt))
above_now = bool(vals[-1])
longer = sum(1 for s, n in lens if s == above_now and n > cur_len)
check("MA50 'longer before' = independent count", f50["longer_before"] == longer,
      (f50["longer_before"], longer))
check("too new -> no unusual facts", T.unusual(mk(walk(100))) == [])
vv = np.full(400, 1000.0); vv[150] = 100.0; vv[300] = 50000.0; vv[-1] = 100.0
vf = {f["key"]: f for f in T.unusual(mk(walk(400, seed=5), vol=vv))}["volume"]
check("quiet day counts past QUIET days (not busy ones)", vf["n"] == 1 and vf["value"] < 0.2,
      (vf["n"], vf["value"]))
check("median 'normal' ignores one block-deal day", abs(vf["value"] - 0.1) < 1e-9, vf["value"])

print("6) patterns")
c = list(np.linspace(200, 100, 300)) + list(np.linspace(100, 260, 200))
gc = T.pattern_series(mk(c))["golden_cross_50_200"][1]
check("golden cross fires exactly once", int(gc.sum()) == 1, int(gc.sum()))
s = pd.Series(c)
m50, m200 = s.rolling(50).mean(), s.rolling(200).mean()
want = int(np.flatnonzero(((m50 > m200) & (m50.shift() <= m200.shift())).to_numpy())[0])
check("golden cross on the right bar", int(np.flatnonzero(gc.to_numpy())[0]) == want)
c = np.full(320, 100.0); c[310:] = 106.0
v = np.full(320, 1000.0); v[310] = 4000.0
hb = T.pattern_series(mk(c, vol=v))["high_52w_on_volume"][1]
check("52w high on volume fires on the breakout bar", bool(hb.iloc[310]) and not hb.iloc[:310].any())

rng = np.random.default_rng(7)
dfw = mk(walk(700, seed=7), vol=rng.uniform(500, 3000, 700))
full = T.pattern_series(dfw)
bad = []
for t in (260, 333, 450, 512, 699):
    cut = T.pattern_series(dfw.iloc[: t + 1])
    for k, (_, ser) in full.items():
        if bool(ser.iloc[t]) != bool(cut[k][1].iloc[t]):
            bad.append((k, t))
check("no look-ahead: full history == history cut at t", not bad, bad[:5])

print("7) vs_market + stage")
idx_c = walk(600, seed=11)
idx = pd.Series(idx_c, index=pd.bdate_range("2015-01-01", periods=600))
ir = pd.Series(idx_c).pct_change().fillna(0).to_numpy()
st_c = 100 * np.cumprod(1 + 1.5 * ir)
b = T.beta(mk(st_c), idx)
check("beta 1.5", b is not None and abs(b - 1.5) < 0.01, b)
out_c = idx_c * np.linspace(1, 1.8, 600)                   # steady outperformer
vm = T.vs_market(mk(out_c), idx, "Nifty 50")
check("outperformer: RS at 52w high", vm["rs_52w_high"] is True, vm["rs_52w_high"])
check("outperformer: 12M excess > 0", vm["excess"]["12M"] > 0, vm["excess"])
check("outperformer: Leading", vm["quadrant"] == "Leading", vm.get("quadrant"))
check("rising series = Stage 2", T.stage(mk(np.linspace(100, 200, 400))) == "Stage 2 uptrend")
check("falling series = Stage 4", T.stage(mk(np.linspace(200, 100, 400))) == "Stage 4 downtrend")

v = T.verdict([{"key": "a", "pct": 99, "text": "x"}, {"key": "b", "pct": 50, "text": "y"}],
              {"fired": []}, [vm])
check("verdict keeps extremes, drops the middle", [f["key"] for f in v] == ["a", "rs_high"],
      [f["key"] for f in v])

print(f"\n{_N[0] - len(_FAILS)}/{_N[0]} checks passed")
sys.exit(len(_FAILS))
