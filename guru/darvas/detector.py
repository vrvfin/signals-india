r"""
Darvas box staircase detector (pure pandas/numpy, safe to import anywhere).

Classic Nicolas Darvas rules, walked forward bar by bar over the lookback window:
  TOP     a high that is not exceeded for CONFIRM days afterwards
  BOTTOM  the lowest low after the top, not undercut for CONFIRM days
          (a new high above the top before the bottom confirms restarts the hunt)
  BOX     active from the top's bar until
            close > top    -> BREAKOUT  (box done; the hunt for the next top starts)
            close < bottom -> BROKEN    (box broken; the hunt restarts)

This is a DIFFERENT definition from scripts/strategy_darvas.find_darvas_box (LIVE,
untouched): that one finds today's single tight 5-20 day range for a signal; this
one returns the HISTORY of boxes so a chart can draw the staircase.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

CONFIRM = 3          # Darvas's own rule: 3 days without a new high / new low
LOOKBACK = 252       # one trading year
NEAR_TOP_PCT = 2.0   # "at the top of the box" = within 2% of the box top


def find_boxes(df: pd.DataFrame, lookback: int = LOOKBACK,
               confirm: int = CONFIRM) -> list[dict]:
    """-> list of boxes, oldest first:
    {start, end, top, bottom, status: done|broken|active, breakout_date}
    plus a trailing {status: 'forming', top, start} when a top is confirmed but
    its bottom is not yet."""
    d = df.sort_values("date").tail(lookback).reset_index(drop=True)
    if len(d) < confirm * 3:
        return []
    dates = pd.to_datetime(d["date"]).dt.strftime("%Y-%m-%d").tolist()
    hi = d["high"].astype(float).to_numpy()
    lo = d["low"].astype(float).to_numpy()
    cl = d["close"].astype(float).to_numpy()

    boxes: list[dict] = []
    state = "top"                    # top -> bottom -> box
    top = hi[0]; top_i = 0; since_top = 0
    bot = np.inf; since_bot = 0
    for i in range(1, len(d)):
        if state == "top":
            if hi[i] > top:
                top, top_i, since_top = hi[i], i, 0
            else:
                since_top += 1
                if since_top >= confirm:
                    state = "bottom"
                    bot = lo[top_i + 1:i + 1].min()
                    since_bot = i - (top_i + 1 + int(np.argmin(lo[top_i + 1:i + 1])))
        elif state == "bottom":
            if hi[i] > top:                       # new high: top was not the top
                state, top, top_i, since_top = "top", hi[i], i, 0
                continue
            if lo[i] < bot:
                bot, since_bot = lo[i], 0
            else:
                since_bot += 1
            if since_bot >= confirm:
                state = "box"
                boxes.append({"start": dates[top_i], "end": dates[i], "top": float(top),
                              "bottom": float(bot), "status": "active",
                              "breakout_date": None})
        else:                                      # inside a confirmed box
            b = boxes[-1]
            if cl[i] > top:
                b.update(end=dates[i - 1], status="done", breakout_date=dates[i])
                state, top, top_i, since_top = "top", hi[i], i, 0
            elif cl[i] < bot:
                b.update(end=dates[i], status="broken")
                state, top, top_i, since_top = "top", hi[i], i, 0
            else:
                b["end"] = dates[i]
    if state == "bottom":
        boxes.append({"start": dates[top_i], "end": dates[-1], "top": float(top),
                      "bottom": None, "status": "forming", "breakout_date": None})
    return boxes


def summarize(df: pd.DataFrame, boxes: list[dict]) -> dict:
    """card fields: box count, staircase length ending at the latest box, state,
    resistance (latest top) and support (latest bottom)."""
    real = [b for b in boxes if b["status"] != "forming"]
    steps = 1 if real else 0                       # ascending run ending at the latest box
    for prev, cur in zip(real[-2::-1], real[::-1]):
        if cur["bottom"] > prev["bottom"] and cur["top"] > prev["top"]:
            steps += 1
        else:
            break
    last_date = pd.to_datetime(df["date"]).max().strftime("%Y-%m-%d")
    close = float(df.sort_values("date")["close"].iloc[-1])
    state, res, sup = "no box", None, None
    if boxes:
        b = boxes[-1]
        res, sup = b["top"], b["bottom"]
        if b["status"] == "forming":
            state = "forming new box"
            if real and real[-1]["breakout_date"]:
                sup = real[-1]["top"]              # the old top is the new floor
        elif b["status"] == "active":
            state = ("at box top" if close >= b["top"] * (1 - NEAR_TOP_PCT / 100)
                     else "inside box")
        elif b["status"] == "done":                # broke out, next top not set yet
            state, res, sup = "above box (new highs)", None, b["top"]
        elif b["status"] == "broken":
            state = "box broken"
        if any(x["breakout_date"] == last_date for x in real):
            state = "breakout today"
    return {"n_boxes": len(real), "staircase": steps, "state": state,
            "resistance": None if res is None else round(res, 2),
            "support": None if sup is None else round(sup, 2)}
