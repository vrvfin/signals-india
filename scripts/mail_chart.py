r"""
mail_chart.py — a static price chart that survives an email client.

WHY (user 2026-10-10): "I need to see the chart also in the mail to make a quick
decision." The gallery's charts are JavaScript (lightweight-charts), which no mail
client runs, so a mail needs a PNG embedded inline (mailer.send_email inline_images).

One chart: ~1 year of daily candles, 50 / 200-day simple moving averages, volume, the
52-week high as a dashed line, and optional event markers (results day, pick day).
Small on purpose — ~700x380 px, tens of KB — so ten of them fit a mail comfortably.

Pure: takes a DataFrame, returns PNG bytes. Loading OHLCV from Drive is the caller's
job (build_gallery._bulk_parquet over data/ohlcv, the same files the gallery draws).

    python scripts/mail_chart.py --self-test
"""
from __future__ import annotations

import io
import sys

import pandas as pd

UP, DOWN = "#26a69a", "#ef5350"          # gallery candle colours
MA50, MA200, HI52 = "#1f77b4", "#ff7f0e", "#7a7a7a"


def _prep(ohlcv: pd.DataFrame, days: int) -> pd.DataFrame:
    if ohlcv is None or getattr(ohlcv, "empty", True) or "close" not in ohlcv.columns:
        return pd.DataFrame()
    d = ohlcv.copy()
    d["date"] = pd.to_datetime(d["date"], errors="coerce")
    for c in ("open", "high", "low", "close", "volume"):
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d.dropna(subset=["date", "close"]).sort_values("date").drop_duplicates("date")
    for c in ("open", "high", "low"):
        if c not in d.columns:
            d[c] = d["close"]
        d[c] = d[c].fillna(d["close"])
    if "volume" not in d.columns:
        d["volume"] = 0.0
    # moving averages on the FULL history, then trim — a 200-DMA computed on the
    # visible window alone would be blank for most of it
    d["ma50"] = d["close"].rolling(50, min_periods=50).mean()
    d["ma200"] = d["close"].rolling(200, min_periods=200).mean()
    d["hi252"] = d["high"].rolling(252, min_periods=20).max()
    return d.tail(days).reset_index(drop=True)


def render_png(ohlcv: pd.DataFrame, title: str = "", days: int = 250,
               markers: list[tuple] | None = None) -> bytes:
    """PNG bytes, or b"" when there is nothing to draw.

    markers: [(date, label, colour)] — e.g. ("2026-10-09", "Q2 results", "#8e44ad").
    A marker outside the window is ignored, never extends the axis.
    """
    d = _prep(ohlcv, days)
    if len(d) < 5:
        return b""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax, axv) = plt.subplots(2, 1, figsize=(7.0, 3.8), dpi=100, sharex=True,
                                  gridspec_kw={"height_ratios": [4, 1], "hspace": 0.04})
    x = range(len(d))
    up = d["close"] >= d["open"]
    col = [UP if u else DOWN for u in up]
    ax.vlines(x, d["low"], d["high"], color=col, linewidth=0.7)
    body_lo = d[["open", "close"]].min(axis=1)
    body_h = (d["close"] - d["open"]).abs().clip(lower=d["close"] * 0.0008)
    ax.bar(x, body_h, bottom=body_lo, color=col, width=0.7, linewidth=0)
    if d["ma50"].notna().any():
        ax.plot(x, d["ma50"], color=MA50, linewidth=1.1, label="50-DMA")
    if d["ma200"].notna().any():
        ax.plot(x, d["ma200"], color=MA200, linewidth=1.1, label="200-DMA")
    hi = d["hi252"].iloc[-1]
    if pd.notna(hi):
        ax.axhline(hi, color=HI52, linestyle="--", linewidth=0.8, label="52-wk high")

    for m in markers or []:
        try:
            md = pd.to_datetime(m[0])
        except Exception:
            continue
        hit = d.index[d["date"] >= md]
        if len(hit) == 0 or md < d["date"].iloc[0]:
            continue
        i = int(hit[0])
        c = m[2] if len(m) > 2 else "#8e44ad"
        ax.axvline(i, color=c, linewidth=1.0, alpha=0.8)
        near_right = i > len(d) * 0.8          # flip the label so it is not clipped
        ax.annotate(str(m[1])[:18], (i, ax.get_ylim()[1]),
                    xytext=(-3 if near_right else 3, -10), textcoords="offset points",
                    fontsize=7, color=c, ha="right" if near_right else "left")

    axv.bar(x, d["volume"], color=col, width=0.7, linewidth=0)
    axv.set_yticks([])
    last = d["close"].iloc[-1]
    ax.set_title(f"{title}   last {last:,.2f}", fontsize=9, loc="left")
    ax.legend(fontsize=7, loc="upper left", frameon=False, ncol=3)
    ax.tick_params(labelsize=7)
    ax.grid(alpha=0.15)
    # month ticks on the shared axis
    months = d["date"].dt.to_period("M")
    ticks = [i for i in range(1, len(d)) if months.iloc[i] != months.iloc[i - 1]]
    axv.set_xticks(ticks)
    axv.set_xticklabels([d["date"].iloc[i].strftime("%b") for i in ticks], fontsize=7)
    ax.set_xlim(-1, len(d))
    fig.subplots_adjust(left=0.08, right=0.99, top=0.92, bottom=0.08)
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    return buf.getvalue()


def img_tag(cid: str, alt: str = "", width: int = 640) -> str:
    """The <img> that points at an inline image sent with the same cid."""
    a = str(alt).replace('"', "")
    return (f'<img src="cid:{cid}" alt="{a}" width="{width}" '
            f'style="display:block;max-width:100%;height:auto;border:1px solid #eee">')


def _self_test() -> int:
    import numpy as np
    fails = 0

    def check(name, ok):
        nonlocal fails
        print(("  ok   " if ok else "  FAIL ") + name)
        fails += 0 if ok else 1

    n = 400
    dates = pd.bdate_range("2025-03-01", periods=n)
    rng = np.random.default_rng(7)
    close = 100 * np.exp(np.cumsum(rng.normal(0.001, 0.015, n)))
    df = pd.DataFrame({"date": dates, "open": close * 0.995, "high": close * 1.01,
                       "low": close * 0.99, "close": close,
                       "volume": rng.integers(1e4, 1e5, n)})
    png = render_png(df, "TEST", markers=[(str(dates[-20].date()), "Q2 results",
                                           "#8e44ad"), ("1990-01-01", "old", "#000")])
    check("renders a PNG", png[:8] == b"\x89PNG\r\n\x1a\n")
    check("small enough for a mail (<90 KB)", 0 < len(png) < 90_000)
    check("empty frame -> no chart", render_png(pd.DataFrame()) == b"")
    check("too few bars -> no chart", render_png(df.head(3)) == b"")
    check("string columns are coerced",
          render_png(df.astype({"close": str}), "S")[:4] == b"\x89PNG")
    check("img tag uses cid", 'src="cid:abc"' in img_tag("abc", "x"))
    print(f"\n{'ALL PASS' if not fails else f'{fails} FAILED'}")
    return 1 if fails else 0


if __name__ == "__main__":
    if "--self-test" in sys.argv:
        raise SystemExit(_self_test())
    print(__doc__)
