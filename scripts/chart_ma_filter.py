r"""
chart_ma_filter.py — which stock charts a gallery page hides.

User 2026-11-12: "only show chart where stock price is above its DMA ... so if stock
today is above [the] DMA it shows up ... if it goes below, it does not show up in
chart" + "lets try with 20 dma that we already calculate" + "do this for all the
charts" + "the tables can remain as is, only charts needs to change".
User 2026-11-18 ("i dont think this is applied consistently" -> "Yes, do that"): the
rule now uses the BLUE line drawn on every chart (EMA20), not the 20-day simple
average, which no chart draws; and a stock Phase 1 has no row for gets its line
worked out from its own price file.

The rule, in one place for every chart page:
  * a stock in the PF always keeps its chart;
  * any other stock keeps its chart only if its close is ABOVE its N-day EMA (the
    blue line) on the Phase 1 date. Phase 1's features table gives both (close,
    ema_<N>), worked out exactly as the chart draws it: ewm(span=N, adjust=False);
  * no features row, or a blank value -> the EMA is worked out from the stock's own
    data/ohlcv/<symbol>.parquet (Phase 1 skips stocks with < 60 bars, so new listings
    never get a row). Fewer than N closes, no price file, or a read that fails ->
    the chart is KEPT (absent data must not hide a stock, as with the market-cap
    floor); so does a PF list that can't be read.

N is CHART_MA_N (20). Phase 1 stores ema_{10,20,50,100,200}, so another line is one
number here, or `--chart-ma 10` on a single page.

Only cards are dropped. Each page keeps its own summary table whole; build_gallery
keeps the card numbers equal to the table row numbers (`_card_id`).

Pure helpers + read-only Drive loaders. Writes nothing.
"""
from __future__ import annotations

import io
import os

import pandas as pd

CHART_MA_N = 20

# The Drive that load_features() last read from. hidden() uses it to look up a
# missing stock's own prices, so a caller that only does
#   hidden(symbols, load_features(drive), pf_symbols(...))
# (guru, Darvas, the private Setups / Zanger builders) gets the fallback unchanged.
_LAST_DRIVE = []


def ma_col(n: int = CHART_MA_N) -> str:
    return f"ema_{int(n)}"


def why(n: int = CHART_MA_N) -> str:
    """The reason, worded the way the chart legend names the line."""
    return f"below the EMA{int(n)} line"


def ema_last(odf, n: int = CHART_MA_N):
    """(last close, last N-day EMA) from a price frame, worked out like the chart line
    (ewm(span=N, adjust=False) on the date-sorted close); None if under N closes."""
    if odf is None or getattr(odf, "empty", True) or not {"date", "close"} <= set(odf.columns):
        return None
    c = pd.to_numeric(odf.sort_values("date")["close"], errors="coerce").dropna()
    if len(c) < int(n):
        return None
    return float(c.iloc[-1]), float(c.ewm(span=int(n), adjust=False).mean().iloc[-1])


def hidden(symbols, feats, pf_syms, n: int = CHART_MA_N, prices=None) -> set:
    """Upper-case symbols whose chart is hidden.

    symbols  iterable of symbols on the page (blank ones are ignored)
    feats    features/latest.parquet as a DataFrame (`symbol`, `close`, `ema_<n>`)
    pf_syms  set of upper-case PF symbols, or None when the PF list is unknown
    n        the line's length in days; 0 or less turns the filter off
    prices   callable symbol -> price DataFrame (date, close) or None, for stocks the
             features table has no value for; default = the Drive load_features read
    """
    if not n or int(n) <= 0 or pf_syms is None:
        return set()
    col = ma_col(n)
    val = {}
    if feats is not None and not feats.empty and {"symbol", "close", col} <= set(feats.columns):
        val = dict(zip(feats["symbol"].astype(str).str.strip().str.upper(),
                       zip(pd.to_numeric(feats["close"], errors="coerce"),
                           pd.to_numeric(feats[col], errors="coerce"))))
    if prices is None and _LAST_DRIVE:
        prices = drive_prices(_LAST_DRIVE[-1])
    out = set()
    for s in symbols:
        u = str(s or "").strip().upper()
        if not u or u in ("NAN", "NONE") or u in pf_syms:
            continue
        x = val.get(u)
        if x is None or pd.isna(x[0]) or pd.isna(x[1]):
            x = None
            if prices is not None:
                try:
                    x = ema_last(prices(str(s).strip()), n)
                except Exception:                     # noqa: BLE001  unreadable -> keep
                    x = None
        if x is not None and x[0] <= x[1]:
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
    return (f" — {n_hidden} charts hidden ({why(n)}, not in PF)"
            if n_hidden else "")


def table_tag(n: int = CHART_MA_N) -> str:
    """Grey note for a table row whose chart is hidden."""
    return (f'<span style="color:#999;font-size:10.5px;font-weight:600" '
            f'title="no chart today: closed at or below its {int(n)}-day EMA (the blue line)">'
            f' below EMA{int(n)}</span>')


def pf_symbols(drive, isin2sym: dict):
    """Upper-case symbols of the PF holdings, or None if the holdings file can't be read
    (then `hidden` hides nothing rather than risk hiding a PF chart)."""
    from _extractor_base import load_portfolio_isins
    isins = load_portfolio_isins(drive, os.environ["GDRIVE_FOLDER_ID"])
    if isins is None:
        return None
    return {str(isin2sym[i]).strip().upper() for i in isins if isin2sym.get(i)}


def drive_prices(drive):
    """-> callable symbol -> data/ohlcv/<symbol>.parquet as a DataFrame, or None when
    there is no such file. Read-only; the folder is looked up once."""
    from _extractor_base import get_or_create_subfolder, find_file, download_bytes
    from drive_io import drive_call
    memo = {}

    def folder():
        if "id" not in memo:
            fid = os.environ["GDRIVE_FOLDER_ID"]
            for p in ("data", "ohlcv"):
                fid = drive_call(lambda p=p, fid=fid: get_or_create_subfolder(drive, fid, p),
                                 label="data/ohlcv folder")
            memo["id"] = fid
        return memo["id"]

    def load(sym: str):
        name = f"{sym}.parquet"
        xid = drive_call(lambda: find_file(drive, folder(), name), label=name)
        if not xid:
            return None
        raw = drive_call(lambda: download_bytes(drive, xid), label=name)
        return pd.read_parquet(io.BytesIO(raw))

    return load


def load_features(drive) -> pd.DataFrame:
    """features/latest.parquet from Drive (read-only), or an empty frame. Also
    remembers `drive`, so hidden() can read a missing stock's own prices from it."""
    from _extractor_base import get_or_create_subfolder, find_file, download_bytes
    from drive_io import drive_call
    _LAST_DRIVE[:] = [drive]
    root = os.environ["GDRIVE_FOLDER_ID"]
    fid = drive_call(lambda: get_or_create_subfolder(drive, root, "features"),
                     label="features folder")
    xid = drive_call(lambda: find_file(drive, fid, "latest.parquet"),
                     label="features/latest.parquet")
    if not xid:
        return pd.DataFrame()
    raw = drive_call(lambda: download_bytes(drive, xid), label="features/latest.parquet")
    return pd.read_parquet(io.BytesIO(raw))
