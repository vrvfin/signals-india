r"""
test_chart_ma_filter.py — offline checks for the 20 DMA chart filter and the 10-day SMA.

Plain python, no pytest (repo convention). Synthetic data only: no Drive, no network.

  1. chart_ma_filter.hidden: above kept; at or below hidden; PF never hidden; a blank,
     a missing stock, an unknown PF list, n=0 or a missing column hide nothing
  2. drop_cards keeps each card's row number (_card_id)
  3. build_html: a dropped card leaves a gap in the anchors, the others keep their row
     numbers and payload keys; an empty list still renders
  4. the PAT growth and guidance tables keep a hidden row, unlinked, with the grey note
  5. compute_features_one: sma_10 / price_vs_ma_10 are the plain 10-day average
  6. the 10 DMA line: right values from the first visible bar, daily charts only

Run:
    python scripts/tests/test_chart_ma_filter.py
Exit code = number of failures.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
os.environ.setdefault("GDRIVE_FOLDER_ID", "ROOT")

import chart_ma_filter as MAF  # noqa: E402
import build_gallery as BG      # noqa: E402
import compute_features as CF   # noqa: E402

FAILS = []


def chk(label, got, want):
    if got != want:
        FAILS.append(f"{label}: got {got!r}, want {want!r}")


# 1. the rule
feats = pd.DataFrame({"symbol": ["UP", "FLAT", "DOWN", "PFDOWN", "BLANK", "low"],
                      "price_vs_ma_20": [3.2, 0.0, -1.5, -9.0, None, -2.0],
                      "price_vs_ma_10": [-0.5, 1.0, 2.0, -1.0, 1.0, 1.0]})
syms = ["UP", "FLAT", "DOWN", "PFDOWN", "BLANK", "NOFEAT", "", None, "Low"]
chk("rule", MAF.hidden(syms, feats, {"PFDOWN"}), {"FLAT", "DOWN", "LOW"})
chk("10-day column", MAF.hidden(syms, feats, set(), n=10), {"UP", "PFDOWN"})
chk("PF unknown = nothing hidden", MAF.hidden(syms, feats, None), set())
chk("n=0 = off", MAF.hidden(syms, feats, set(), n=0), set())
chk("no such column = nothing", MAF.hidden(syms, feats, set(), n=50), set())
chk("no features = nothing", MAF.hidden(syms, pd.DataFrame(), set()), set())
chk("note", MAF.note(3), " — 3 charts hidden (below 20 DMA, not in PF)")
chk("no note when none hidden", MAF.note(0), "")
chk("table tag names the average", "below 10 DMA" in MAF.table_tag(10), True)

# 2. row numbers survive the drop
r = pd.DataFrame({"symbol": ["A", "B", "C", "D"]})
kept = MAF.drop_cards(r, {"B", "D"})
chk("drop keeps row numbers", (kept["symbol"].tolist(), kept["_card_id"].tolist()),
    (["A", "C"], [0, 2]))
chk("nothing hidden = every card, numbered", MAF.drop_cards(r, set())["_card_id"].tolist(),
    [0, 1, 2, 3])
chk("case-insensitive", MAF.drop_cards(r, {"b"})["symbol"].tolist(), ["A", "B", "C", "D"])


# 3. renderer
def bars(n=80, start=100.0):
    d = pd.bdate_range("2026-07-01", periods=n)
    c = np.linspace(start, start * 1.2, n)
    return pd.DataFrame({"date": d, "open": c, "high": c * 1.01, "low": c * 0.99,
                         "close": c, "volume": 1000})


E = pd.DataFrame()
cards = BG.Cards(E, {}, E, E, E, research=E, news={})
ranked = MAF.drop_cards(pd.DataFrame({"symbol": ["AAA", "BBB", "CCC"],
                                      "_exch": "NSE", "n_strategies": 0}), {"BBB"})
html = BG.build_html(ranked, {"AAA": bars(), "CCC": bars()}, cards, {}, 120)
chk("kept cards keep their anchors", ('id="ch0"' in html, 'id="ch2"' in html), (True, True))
chk("hidden card has no anchor", 'id="ch1"' in html, False)
chk("card number = table row", ">3. <b>CCC</b>" in html, True)
chk("payload keyed by row", ('"0":{' in html, '"2":{' in html), (True, True))
plain = BG.build_html(pd.DataFrame({"symbol": ["AAA", "CCC"], "_exch": "NSE",
                                    "n_strategies": 0}),
                      {"AAA": bars(), "CCC": bars()}, cards, {}, 120)
chk("without _card_id: numbered as before", ('id="ch1"' in plain, ">2. <b>CCC</b>" in plain),
    (True, True))
try:
    empty = BG.build_html(MAF.drop_cards(pd.DataFrame({"symbol": ["AAA"]}), {"AAA"}),
                          {}, cards, {}, 120)
    chk("every card hidden still renders a page", "<html" in empty.lower(), True)
except Exception as e:  # noqa: BLE001
    FAILS.append(f"empty page: {type(e).__name__}: {e}")

# 4. tables keep the row
pg = pd.DataFrame({"isin": ["I1", "I2"], "slug": ["AAA", "BBB"], "company_name": ["A", "B"],
                   "quarter": "Q2 FY27", "reported": "2026-11-10", "pat": [30.0, 30.0],
                   "pat_prev": [20.0, 20.0], "pat_yearago": [10.0, 10.0],
                   "yoy_pct": [200.0, 200.0], "qoq_pct": [50.0, 50.0], "source": "results",
                   "symbol": ["AAA", "BBB"], "in_pf": None, "both": "",
                   "_ma_tag": ["", MAF.table_tag()]})
t = BG._pg_table(pg)
chk("PAT table: both rows", (">AAA</a>" in t, ">BBB</span>" in t), (True, True))
chk("PAT table: hidden row unlinked + noted", ('href="#ch1"' in t, "below 20 DMA" in t),
    (False, True))
wl = pd.DataFrame({"isin": ["I1", "I2"], "nse_symbol": ["AAA", "BBB"], "quarter": "Q2FY27",
                   "cagr_pct": [60.0, 70.0], "score_metric": "revenue",
                   "date_added": "2026-11-01", "_ma_tag": [MAF.table_tag(), ""]})
t = BG._wl_table(wl)
chk("guidance table: hidden row unlinked + noted",
    ('href="#ch0"' in t, 'href="#ch1"' in t, "below 20 DMA" in t), (False, True, True))

# 5. the 10-day simple average in Phase 1 features
df = bars(260)
df.loc[df.index[-3:], "close"] = [130.0, 128.0, 125.0]
f = CF.compute_features_one("TEST", df)
want10 = float(df["close"].iloc[-10:].mean())
chk("sma_10 = mean of the last 10 closes", round(f["sma_10"], 6), round(want10, 6))
chk("price_vs_ma_10", round(f["price_vs_ma_10"], 6),
    round((df["close"].iloc[-1] / want10 - 1) * 100, 6))
chk("sma_20 unchanged", round(f["sma_20"], 6), round(float(df["close"].iloc[-20:].mean()), 6))
chk("price_vs_ma_20 still there", "price_vs_ma_20" in f and "price_vs_ma_200" in f, True)

# 6. the 10 DMA line on the charts
hist = bars(60)
hist["close"] = np.arange(1.0, 61.0)                  # 1..60 -> 10-day mean = close - 4.5
shown = [str(d)[:10] for d in hist["date"].iloc[-5:]]
line = BG._sma_line(hist, shown)
chk("10 DMA from the whole history", [x["value"] for x in line], [51.5, 52.5, 53.5, 54.5, 55.5])
chk("10 DMA: under 10 bars = no point", BG._sma_line(hist.head(9), shown), [])
chk("10 DMA: nothing to draw", (BG._sma_line(pd.DataFrame(), shown), BG._sma_line(hist, [])),
    ([], []))
daily = BG.build_html(pd.DataFrame({"symbol": ["AAA"], "_exch": "NSE", "n_strategies": 0}),
                      {"AAA": bars(80)}, cards, {}, 60)
chk("daily chart carries the line + legend", ('"m10":[{' in daily, "━ 10 DMA" in daily),
    (True, True))
weekly = BG.build_html(pd.DataFrame({"symbol": ["AAA"], "_exch": "NSE", "n_strategies": 0}),
                       {"AAA": bars(300)}, cards, {}, 60, resample="W-FRI")
chk("weekly chart: no 10-day line", ('"m10"' in weekly, "━ 10 DMA" in weekly), (False, False))

for x in FAILS:
    print("FAIL " + x)
print(f"{'FAILED' if FAILS else 'OK'}: {len(FAILS)} failure(s)")
sys.exit(len(FAILS))
