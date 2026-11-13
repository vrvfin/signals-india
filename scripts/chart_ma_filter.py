r"""
chart_ma_filter.py — which stock charts a gallery page hides.

User 2026-11-12: "only show chart where stock price is above its DMA ... so if stock
today is above [the] DMA it shows up ... if it goes below, it does not show up in
chart" + "lets try with 20 dma that we already calculate" + "do this for all the
charts" + "the tables can remain as is, only charts needs to change".

The rule, in one place for every chart page:
  * a stock in the PF always keeps its chart;
  * any other stock keeps its chart only if Phase 1's features table says its close
    is ABOVE its N-day simple average (price_vs_ma_<N> > 0) on the Phase 1 date;
  * a blank or missing value hides nothing (absent data must not hide a stock, as
    with the market-cap floor), and neither does a PF list that can't be read.

N is CHART_MA_N (20). Phase 1 stores price_vs_ma_{10,20,50,200}, so moving to the
10-day average is one number here, or `--chart-ma 10` on a single page.

Only cards are dropped. Each page keeps its own summary table whole; build_gallery
keeps the card numbers equal to the table row numbers (`_card_id`).

Pure helpers + two read-only Drive loaders. Writes nothing.
"""
from __future__ import annotations

import io
import os

import pandas as pd

CHART_MA_N = 20


def ma_col(n: int = CHART_MA_N) -> str:
    return f"price_vs_ma_{int(n)}"


def hidden(symbols, feats, pf_syms, n: int = CHART_MA_N) -> set:
    """Upper-case symbols whose chart is hidden.

    symbols  iterable of symbols on the page (blank ones are ignored)
    feats    features/latest.parquet as a DataFrame (needs `symbol` + price_vs_ma_<n>)
    pf_syms  set of upper-case PF symbols, or None when the PF list is unknown
    n        the average's length in days; 0 or less turns the filter off
    """
    col = ma_col(n) if n and int(n) > 0 else None
    if (col is None or pf_syms is None or feats is None or feats.empty
            or not {"symbol", col} <= set(feats.columns)):
        return set()
    val = dict(zip(feats["symbol"].astype(str).str.strip().str.upper(),
                   pd.to_numeric(feats[col], errors="coerce")))
    out = set()
    for s in symbols:
        u = str(s or "").strip().upper()
        if not u or u in ("NAN", "NONE") or u in pf_syms:
            continue
        x = val.get(u)
        if x is not None and pd.notna(x) and x <= 0:
            out.add(u)
    return out


def drop_cards(ranked: pd.DataFrame, hide: set) -> pd.DataFrame:
    """The card rows that stay. `_card_id` keeps each row's position on the full
    list, so a card's number and anchor still match the page's table row."""
    r = ranked.reset_index(drop=True).copy()
    if "_card_id" not in r.columns:
        r["_card_id"] = range(len(r))
    if not hide or "symbol" not in r.columns:
        return r
    keep = ~r["symbol"].astype(str).str.strip().str.upper().isin(hide)
    return r[keep].reset_index(drop=True)


def note(n_hidden: int, n: int = CHART_MA_N) -> str:
    """Title suffix: how many charts were hidden and why."""
    return (f" — {n_hidden} charts hidden (below {int(n)} DMA, not in PF)"
            if n_hidden else "")


def table_tag(n: int = CHART_MA_N) -> str:
    """Grey note for a table row whose chart is hidden."""
    return (f'<span style="color:#999;font-size:10.5px;font-weight:600" '
            f'title="no chart today: closed at or below its {int(n)}-day average">'
            f' below {int(n)} DMA</span>')


def pf_symbols(drive, isin2sym: dict):
    """Upper-case symbols of the PF holdings, or None if the holdings file can't be read
    (then `hidden` hides nothing rather than risk hiding a PF chart)."""
    from _extractor_base import load_portfolio_isins
    isins = load_portfolio_isins(drive, os.environ["GDRIVE_FOLDER_ID"])
    if isins is None:
        return None
    return {str(isin2sym[i]).strip().upper() for i in isins if isin2sym.get(i)}


def load_features(drive) -> pd.DataFrame:
    """features/latest.parquet from Drive (read-only), or an empty frame."""
    from _extractor_base import get_or_create_subfolder, find_file, download_bytes
    from drive_io import drive_call
    root = os.environ["GDRIVE_FOLDER_ID"]
    fid = drive_call(lambda: get_or_create_subfolder(drive, root, "features"),
                     label="features folder")
    xid = drive_call(lambda: find_file(drive, fid, "latest.parquet"),
                     label="features/latest.parquet")
    if not xid:
        return pd.DataFrame()
    raw = drive_call(lambda: download_bytes(drive, xid), label="features/latest.parquet")
    return pd.read_parquet(io.BytesIO(raw))
