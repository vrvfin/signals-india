r"""
SOLD GALLERY — build_sold_gallery.py

Renders every FULLY SOLD position as a standalone chart gallery, `gallery_sold.html`,
sitting alongside gallery_pf.html / gallery_guru.html and looking identical to them.
Each card carries a red vertical line on the chart at the day the position left the
book, so the shape of what happened AFTER the exit is visible at a glance.

WHY THIS IS A SEPARATE FILE, NOT A NEW --mode IN build_gallery.py
    scripts/build_gallery.py is imported here as a LIBRARY. Every card, chart,
    meta-line and bit of page furniture comes from its build_html(), so a sold card
    is byte-for-byte a gallery card. The ONE thing added to that shared file is an
    optional `marks=` kwarg (the sell-date line); absent it, every existing page
    renders exactly as before — the same additive pattern as `prelude=` / `annot=`.
    Modelled on guru/build_guru_gallery.py, which states the same contract.

INPUT (Drive, read-only — written daily by scripts/pf_decision_tracker.py)
    company_repo/_index/pf_tracker/pf_spells.parquet   status HELD | SOLD per spell
    company_repo/_index/pf_tracker/pf_exits.parquet    the return windows per exit

    FULLY SOLD = an ISIN with a SOLD spell and NO HELD spell: gone from the book and
    not bought back. One card per company, keyed on its LATEST sell date (a name can
    be sold, re-bought and sold again).

    NOTE the ledger only reaches back to the first tracker run, so this page covers
    the recorded sell history and nothing earlier. It grows on its own as the ledger
    ages — the 250 cap and the >500% band are deliberately far ahead of today's data.

USAGE
    python scripts/build_sold_gallery.py                  # top 250 by return since sold
    python scripts/build_sold_gallery.py --dry-run        # resolve + print, draw nothing
    python scripts/build_sold_gallery.py --sort per_day   # by return per day out
    python scripts/build_sold_gallery.py --max 50 --resample W
"""
from __future__ import annotations
import argparse
import os
import re
import sys
import webbrowser
from datetime import datetime

import pandas as pd

_SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(_SCRIPTS_DIR)
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

# The shared gallery, used strictly as a library.
import build_gallery as BG
from _extractor_base import log

TRACKER_DIR = "company_repo/_index/pf_tracker"
SPELLS = "pf_spells.parquet"
EXITS = "pf_exits.parquet"

# Distribution bands for `since_sell_pct`. Half-open [lo, hi); None = unbounded.
# Deliberately reaching far past today's data so the page needs no edit as the
# ledger grows — an empty band renders as a plain 0.
BANDS = [("< 0%", None, 0.0),
         ("0 – 50%", 0.0, 50.0),
         ("50 – 100%", 50.0, 100.0),
         ("100 – 200%", 100.0, 200.0),
         ("200 – 500%", 200.0, 500.0),
         ("> 500%", 500.0, None)]

# Sort key -> (column, label). "abs" ranks on |since_sell_pct| (biggest move either
# way); every other key ranks on the raw number, descending — raw meaning the
# stock's own return, NOT benchmark-relative.
SORTS = {"since": ("since_sell_pct", "return since sold"),
         "per_day": ("per_day_pct", "return per day out"),
         "realised": ("realised_pct", "realised return (entry → sell)"),
         "full": ("full_hold_pct", "return had you never sold"),
         "abs": ("_abs_since", "absolute size of the move since sold")}

# The ledger's own `symbol` column holds the company NAME when an ISIN never
# resolved to a ticker, so it is only trusted as a fallback when it LOOKS like one.
_TICKER = re.compile(r"^[A-Z0-9][A-Z0-9&_.-]{0,19}$")


# ─────────────────────────────── formatting ─────────────────────────────────

def _n1(v):
    """Scalar -> float or NaN, never raising."""
    return pd.to_numeric(pd.Series([v]), errors="coerce").iloc[0]


def _num(s):
    return pd.to_numeric(s, errors="coerce")


def _f(v, dp=1, dash="–"):
    x = _n1(v)
    return dash if pd.isna(x) else f"{x:,.{dp}f}"


def _pct(v, dp=1):
    """Signed percent with colour — green up, red down, grey when unknown."""
    x = _n1(v)
    if pd.isna(x):
        return '<span style="color:#999">–</span>'
    col = "#0a7d33" if x > 0 else "#c62828" if x < 0 else "#555"
    return f'<span style="color:{col};font-weight:600">{x:+,.{dp}f}%</span>'


def _d(v):
    """2026-08-31 -> 31-Aug-2026; anything unparseable passes through."""
    t = pd.to_datetime(v, errors="coerce")
    return str(v or "–") if pd.isna(t) else t.strftime("%d-%b-%Y")


# ──────────────────────────────── universe ──────────────────────────────────

def load_sold(drive) -> tuple[pd.DataFrame, dict]:
    """FULLY SOLD companies, one row each, newest exit per ISIN.

    Returns (frame, diag). `diag` carries the counts the page prints about itself,
    so a thin ledger is stated on the page rather than silently looking like a
    short list."""
    fid = BG._folder(drive, TRACKER_DIR)
    spells = BG._read_parquet(drive, fid, SPELLS)
    exits = BG._read_parquet(drive, fid, EXITS)
    diag = {"spells": len(spells), "exits": len(exits), "held": 0,
            "sold_spells": 0, "rebought": 0}
    if spells.empty or "status" not in spells.columns:
        log(f"  no {SPELLS} on Drive — run scripts/pf_decision_tracker.py first")
        return pd.DataFrame(), diag

    held = set(spells.loc[spells["status"] == "HELD", "isin"].astype(str))
    sold = spells[spells["status"] == "SOLD"].copy()
    sold["isin"] = sold["isin"].astype(str)
    diag["held"] = len(held)
    diag["sold_spells"] = len(sold)
    # "Fully sold" = gone and not bought back. A name currently in the book has a
    # HELD spell, so its old exits belong to the tracker's regret table, not here.
    full = sold[~sold["isin"].isin(held)].copy()
    diag["rebought"] = int(sold["isin"].nunique() - full["isin"].nunique())
    if full.empty:
        return pd.DataFrame(), diag

    # One card per company: the LATEST exit (a name can be sold more than once).
    full["_end"] = pd.to_datetime(full["spell_end"], errors="coerce")
    full = (full.sort_values("_end").groupby("isin", as_index=False).tail(1)
                .reset_index(drop=True))
    full = full.rename(columns={"symbol": "_led_sym"})

    if not exits.empty and {"isin", "sell_date"} <= set(exits.columns):
        ex = exits.copy()
        ex["isin"] = ex["isin"].astype(str)
        ex["_end"] = pd.to_datetime(ex["sell_date"], errors="coerce")
        ex = ex.drop(columns=[c for c in ("symbol", "name", "sell_date")
                              if c in ex.columns])
        full = full.merge(ex, on=["isin", "_end"], how="left")
    for c in ("realised_pct", "since_sell_pct", "full_hold_pct", "pf_since_sell_pct",
              "n500_since_sell_pct", "vs_n500_pp", "days_held", "days_since_sell",
              "weight_at_sell", "verdict", "first_seen"):
        if c not in full.columns:
            full[c] = pd.NA

    # Days out, recomputed from the sell date when the exits view did not cover this
    # row — its own window is a 12-month lookback and drops anything older.
    today = pd.Timestamp(datetime.now().date())
    full["days_since_sell"] = _num(full["days_since_sell"]).fillna(
        (today - full["_end"]).dt.days)
    full["days_held"] = _num(full["days_held"]).fillna(
        (full["_end"] - pd.to_datetime(full["spell_start"], errors="coerce")).dt.days)

    # The normalised metric: how hard the stock moved per day you were out of it.
    # Very noisy at low day counts, so it is a column and an opt-in sort, never the
    # default. NOT annualised — pf_decision_tracker.MIN_ANNUALISE_DAYS refuses to
    # annualise under a quarter and this ledger is younger than that.
    dso = _num(full["days_since_sell"]).where(lambda s: s > 0)
    full["per_day_pct"] = _num(full["since_sell_pct"]) / dso
    full["_abs_since"] = _num(full["since_sell_pct"]).abs()
    full["sell_date"] = full["_end"].dt.strftime("%Y-%m-%d")
    return full, diag


def resolve_symbols(df: pd.DataFrame, isin2sym: dict) -> pd.DataFrame:
    """isin -> NSE symbol via the shared map, falling back to the ledger's own
    symbol when it looks like a ticker. Unresolved rows keep symbol="" and render
    as a name-only card (build_gallery.build_html handles that), so a delisted or
    off-universe exit is never silently dropped."""
    def pick(r):
        s = str(isin2sym.get(str(r["isin"]), "") or "").strip()
        if s:
            return s
        led = str(r.get("_led_sym", "") or "").strip()
        return led if _TICKER.match(led) else ""
    df["symbol"] = df.apply(pick, axis=1)
    df["_pfname"] = df["name"].astype(str)
    return df


# ──────────────────────────── page furniture ────────────────────────────────

def stats_block(df: pd.DataFrame, sort_label: str, diag: dict, n_shown: int,
                n_total: int) -> str:
    """The summary that opens the page: how the whole sold book behaved, then the
    distribution of return-since-sold in bands AND cumulatively."""
    v = _num(df["since_sell_pct"]).dropna()
    n = len(v)
    if n:
        lo_i, hi_i = v.idxmin(), v.idxmax()
        lo_s = str(df.loc[lo_i, "symbol"] or df.loc[lo_i, "name"] or "")
        hi_s = str(df.loc[hi_i, "symbol"] or df.loc[hi_i, "name"] or "")
        chips = [("Fully sold", f"{n_total}"),
                 ("With price data", f"{n}"),
                 ("Average", f"{v.mean():+,.1f}%"),
                 ("Median", f"{v.median():+,.1f}%"),
                 ("Min", f"{v.min():+,.1f}% · {lo_s}"),
                 ("Max", f"{v.max():+,.1f}% · {hi_s}")]
    else:
        chips = [("Fully sold", f"{n_total}"), ("With price data", "0")]

    chip_html = "".join(
        f'<div style="background:#fff;border:1px solid #e3e7ee;border-radius:8px;'
        f'padding:6px 12px;min-width:118px">'
        f'<div style="font-size:11px;color:#777;text-transform:uppercase;'
        f'letter-spacing:.03em">{k}</div>'
        f'<div style="font-size:16px;font-weight:800;color:#1a3d6e">{x}</div></div>'
        for k, x in chips)

    # Bands are mutually exclusive and sum to n; the cumulative column answers the
    # other reading of the same question ("how many are at or above this?").
    rows = []
    for label, lo, hi in BANDS:
        if lo is None:
            m = v < hi
        elif hi is None:
            m = v >= lo
        else:
            m = (v >= lo) & (v < hi)
        cnt = int(m.sum())
        # "At or above" reads off the band's LOWER edge, so the open-ended losing
        # band has no threshold to count from and shows a dash rather than n.
        cum = "–" if lo is None else int((v >= lo).sum())
        share = (100.0 * cnt / n) if n else 0.0
        bar = ('<div style="background:#e8eef6;border-radius:3px;height:8px;'
               'width:100px">'
               f'<div style="background:#1a3d6e;height:8px;border-radius:3px;'
               f'width:{min(share, 100):.0f}px"></div></div>')
        rows.append(
            f'<tr><td style="padding:4px 10px;border-bottom:1px solid #eee;'
            f'white-space:nowrap"><b>{label}</b></td>'
            f'<td style="padding:4px 10px;border-bottom:1px solid #eee;'
            f'text-align:right">{cnt}</td>'
            f'<td style="padding:4px 10px;border-bottom:1px solid #eee;'
            f'text-align:right;color:#666">{share:.0f}%</td>'
            f'<td style="padding:4px 10px;border-bottom:1px solid #eee">{bar}</td>'
            f'<td style="padding:4px 10px;border-bottom:1px solid #eee;'
            f'text-align:right;color:#666">{cum}</td></tr>')

    pdv = _num(df["per_day_pct"]).replace([float("inf"), float("-inf")], pd.NA).dropna()
    per_day = ""
    if len(pdv):
        per_day = (
            f'<div style="font-size:12px;color:#555;margin:8px 0 0">'
            f'<b>Return per day out</b> (return since sold ÷ days since sold) — '
            f'average {pdv.mean():+.2f}%/d · median {pdv.median():+.2f}%/d · '
            f'min {pdv.min():+.2f}%/d · max {pdv.max():+.2f}%/d. '
            f'<span style="color:#888">Noisy while days-out is small; kept for when '
            f'the ledger is long enough to normalise on.</span></div>')

    note = (f'<div style="font-size:11.5px;color:#888;margin:6px 0 0">'
            f'Sold = gone from the holdings snapshot and not bought back. '
            f'{diag["sold_spells"]} sold spells against {diag["held"]} live holdings; '
            f'{diag["rebought"]} sold name(s) excluded because they were re-bought. '
            f'Showing {n_shown} of {n_total}, ranked by {sort_label}. '
            f'Sell dates are the day a name was first seen missing from a snapshot, '
            f'so they can trail the real trade by a day or two.</div>')

    return (
        '<div style="max-width:1100px;margin:0 auto 14px">'
        '<div style="background:#fbfcfe;border:1px solid #e3e7ee;border-radius:8px;'
        'padding:10px 12px">'
        '<div style="font-size:14px;font-weight:800;color:#1a3d6e;margin-bottom:8px">'
        'Sold book — distribution of return since sold</div>'
        f'<div style="display:flex;gap:8px;flex-wrap:wrap;margin-bottom:10px">'
        f'{chip_html}</div>'
        '<table style="border-collapse:collapse;font-size:12px">'
        '<tr><th style="text-align:left;padding:4px 10px;border-bottom:2px solid #ccc">'
        'Return since sold</th>'
        '<th style="text-align:right;padding:4px 10px;border-bottom:2px solid #ccc">'
        'Stocks</th>'
        '<th style="text-align:right;padding:4px 10px;border-bottom:2px solid #ccc">'
        'Share</th>'
        '<th style="padding:4px 10px;border-bottom:2px solid #ccc"></th>'
        '<th style="text-align:right;padding:4px 10px;border-bottom:2px solid #ccc">'
        'At or above</th></tr>'
        f'{"".join(rows)}</table>{per_day}{note}</div></div>')


def ranked_table(df: pd.DataFrame) -> str:
    """The whole sold list at a glance, so the page is useful before you scroll."""
    head = ["#", "Stock", "Sold on", "Held", "Days out", "Wt at sell",
            "Realised", "Since sold", "Per day", "Never sold", "vs PF", "vs N500",
            "Verdict"]
    th = "".join(
        f'<th style="text-align:left;padding:6px 10px;border-bottom:2px solid #ccc;'
        f'font-size:12px;color:#333;white-space:nowrap">{h}</th>' for h in head)
    rows = []
    for i, (_, r) in enumerate(df.iterrows(), 1):
        nm = str(r.get("name", "") or "")
        sym = str(r.get("symbol", "") or "")
        gap, pfg = _n1(r.get("since_sell_pct")), _n1(r.get("pf_since_sell_pct"))
        vs_pf = (gap - pfg) if (pd.notna(gap) and pd.notna(pfg)) else None
        cells = [
            str(i),
            (f'<b>{nm or sym}</b>'
             + (f' <span style="color:#777">{sym}</span>' if sym and nm else '')),
            _d(r.get("sell_date")),
            f'{_f(r.get("days_held"), 0)}d',
            f'{_f(r.get("days_since_sell"), 0)}d',
            f'{_f(r.get("weight_at_sell"), 1)}%',
            _pct(r.get("realised_pct")),
            _pct(r.get("since_sell_pct")),
            _pct(r.get("per_day_pct"), 2),
            _pct(r.get("full_hold_pct")),
            _pct(vs_pf),
            _pct(r.get("vs_n500_pp")),
            f'<span style="color:#555">{r.get("verdict", "") or "–"}</span>']
        rows.append("<tr>" + "".join(
            f'<td style="padding:6px 10px;border-bottom:1px solid #eee;font-size:12px;'
            f'white-space:nowrap">{c}</td>' for c in cells) + "</tr>")
    return ('<div style="max-width:1100px;margin:0 auto 14px;overflow-x:auto">'
            '<div style="background:#fff;border:1px solid #e3e7ee;border-radius:8px;'
            'padding:8px 4px">'
            '<table style="border-collapse:collapse;width:100%">'
            f'<tr>{th}</tr>{"".join(rows)}</table></div></div>')


def annot_for(df: pd.DataFrame) -> dict:
    """The badge on each card: when it left, and how the three windows read."""
    out = {}
    for _, r in df.iterrows():
        sym = str(r.get("symbol", "") or "")
        if not sym:
            continue
        pdp = _n1(r.get("per_day_pct"))
        bits = [f'SOLD {_d(r.get("sell_date"))}',
                f'held {_f(r.get("days_held"), 0)}d',
                f'out {_f(r.get("days_since_sell"), 0)}d',
                f'realised {_f(r.get("realised_pct"))}%',
                (f'since sold {_f(r.get("since_sell_pct"))}%'
                 + (f' ({pdp:+.2f}%/d)' if pd.notna(pdp) else '')),
                f'never sold {_f(r.get("full_hold_pct"))}%']
        v = str(r.get("verdict", "") or "").strip()
        if v and v != "n/a":
            bits.append(v)
        out[sym.upper()] = (
            '<div style="background:#7f1d1d;color:#fff;border-radius:6px;'
            'padding:4px 10px;font-size:13px;font-weight:700;margin:3px 0">'
            f'🔴 {" · ".join(bits)}</div>')
    return out


def marks_for(df: pd.DataFrame, omap: dict) -> dict:
    """symbol -> the sell-date marker build_gallery draws on that chart.

    The price is the close of the last bar at or before the sell date, so a sell
    dated to a holiday or a weekend still lands on a real traded price."""
    out = {}
    for _, r in df.iterrows():
        sym = str(r.get("symbol", "") or "")
        t = str(r.get("sell_date", "") or "")
        if not sym or not t:
            continue
        price = None
        odf = omap.get(sym)
        if odf is not None and not odf.empty and {"date", "close"} <= set(odf.columns):
            d = odf.copy()
            d["date"] = pd.to_datetime(d["date"], errors="coerce")
            at = d.dropna(subset=["date"]).sort_values("date")
            at = at[at["date"] <= pd.Timestamp(t)]
            if not at.empty:
                price = _n1(at["close"].iloc[-1])
        out[sym] = {"t": t, "x": f"SOLD {_d(t)[:6]}",
                    "p": (float(price) if price is not None and pd.notna(price)
                          else None)}
    return out


# ──────────────────────────────────── main ──────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Chart gallery of fully sold positions, with the sell date "
                    "marked on every chart.")
    ap.add_argument("--max", type=int, default=250,
                    help="cap the number of cards (default 250, 0 = all)")
    ap.add_argument("--sort", choices=sorted(SORTS), default="since",
                    help="ranking column, descending (default: since = return "
                         "since sold, the stock's own return, not vs a benchmark)")
    ap.add_argument("--resample", choices=["D", "W", "M"], default="D",
                    help="chart timeframe: D=daily (default), W=weekly, M=monthly")
    ap.add_argument("--timeframe-days", type=int, default=252)
    ap.add_argument("--out", default="")
    ap.add_argument("--no-open", action="store_true")
    ap.add_argument("--no-news", action="store_true",
                    help="skip the Google-News fetch (much faster)")
    ap.add_argument("--news-days", type=int, default=30)
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--dry-run", action="store_true",
                    help="resolve inputs and print the ranked list + stats; "
                         "download no charts, write no file")
    ap.add_argument("--max-missing-pct", type=float, default=5.0)
    args = ap.parse_args()

    cache = BG.ParquetCache(enabled=not args.no_cache)
    drive = BG._drive()

    log("loading sold ledger…")
    df, diag = load_sold(drive)
    if df.empty:
        log("Nothing to render — no fully sold positions in the ledger.")
        return

    log("loading universe / grades / guidance / summary…")
    uni = BG._read_csv(drive, BG._folder(drive, "universe"), "master_list.csv")
    idx = BG._folder(drive, "company_repo/_index")
    grades = BG._read_parquet(drive, idx, "screener_grades.parquet")
    guidance = BG._read_parquet(drive, idx, "guidance_tracker.parquet")
    summ = BG._read_parquet(drive, BG._folder(drive, "fundamentals"), "summary.parquet")

    df = resolve_symbols(df, BG.isin_symbol_map(uni, grades, guidance))

    mcap_map = {}
    if not summ.empty and "symbol" in summ.columns:
        for _, r in summ.iterrows():
            v = pd.to_numeric(r.get("market_cap_cr"), errors="coerce")
            if pd.notna(v) and v > 0:
                mcap_map[str(r["symbol"]).upper()] = v

    col, sort_label = SORTS[args.sort]
    df = df.sort_values(col, ascending=False, na_position="last").reset_index(drop=True)
    n_total = len(df)
    if args.max > 0:
        df = df.head(args.max).reset_index(drop=True)
    syms = [s for s in df["symbol"].tolist() if s]
    log(f"  {n_total} fully sold; showing {len(df)} ({len(syms)} chartable, "
        f"{len(df) - len(syms)} name-only) ranked by {sort_label}")

    if args.dry_run:
        cols = ["symbol", "name", "sell_date", "days_held", "days_since_sell",
                "realised_pct", "since_sell_pct", "per_day_pct", "full_hold_pct",
                "verdict"]
        log("DRY-RUN — ranked list:")
        print(df[[c for c in cols if c in df.columns]].to_string(index=False))
        v = _num(df["since_sell_pct"]).dropna()
        if len(v):
            log(f"DRY-RUN — since-sold over {len(v)} priced names: "
                f"avg {v.mean():+.1f}% · median {v.median():+.1f}% · "
                f"min {v.min():+.1f}% · max {v.max():+.1f}%")
        else:
            log("DRY-RUN — no priced names")
        tot = 0
        for label, lo, hi in BANDS:
            m = (v < hi) if lo is None else ((v >= lo) if hi is None
                                             else ((v >= lo) & (v < hi)))
            cum = "–" if lo is None else int((v >= lo).sum())
            tot += int(m.sum())
            log(f"    {label:<12} {int(m.sum()):>4}   (at or above: {cum})")
        log(f"    {'TOTAL':<12} {tot:>4}   (must equal {len(v)})")
        log("DRY-RUN — nothing written, no charts downloaded.")
        return

    _rs_map = {"D": (None, ""), "W": ("W-FRI", " · Weekly"), "M": ("ME", " · Monthly")}
    resample, tf_title = _rs_map[args.resample]
    out_default = "gallery_sold.html"
    if args.resample != "D" and not args.out:
        out_default = out_default.replace(".html", f"_{args.resample.lower()}.html")
    out_path = args.out or os.path.join(ROOT, out_default)

    log(f"downloading OHLCV for {len(syms)} names…")
    failed = set()
    omap = BG._bulk_parquet(drive, BG._folder(drive, "data/ohlcv"), syms, cache=cache,
                            what="OHLCV", max_missing_pct=args.max_missing_pct,
                            failed_out=failed)
    log("downloading statements…")
    stmts = BG._bulk_parquet(drive, BG._folder(drive, "fundamentals/statements"), syms,
                             cache=cache, what="statements",
                             max_missing_pct=args.max_missing_pct)
    log("loading gf1 / announcements / research…")
    gf1 = BG._read_parquet(drive, idx, "gf1_guidance_statements.parquet")
    ann = BG._read_parquet(drive, idx, "announcement_ledger.parquet")
    res_idx = BG._read_parquet(drive, idx, "research_index.parquet")

    name_by = {}
    if not grades.empty and {"symbol", "company_name"} <= set(grades.columns):
        name_by = {str(r["symbol"]).upper(): str(r.get("company_name", "") or "")
                   for _, r in grades.iterrows()}
    news_map = ({} if args.no_news
                else BG._fetch_news_map(syms, name_by, days=args.news_days))

    cards = BG.Cards(grades, stmts, guidance, gf1, ann, research=res_idx, news=news_map)
    prelude = (stats_block(df, sort_label, diag, len(df), n_total)
               + ranked_table(df))
    log("assembling HTML…")
    html = BG.build_html(df, omap, cards, mcap_map, args.timeframe_days,
                         title=f"🔴 Sold stocks — what happened next{tf_title}",
                         annot=annot_for(df), resample=resample, prelude=prelude,
                         failed_syms=failed, marks=marks_for(df, omap))
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    log(f"wrote {out_path}  ({len(html) / 1e6:.1f} MB, {len(syms)} charts)")
    if not args.no_open:
        webbrowser.open("file://" + os.path.abspath(out_path))
        log("opened in default browser.")


if __name__ == "__main__":
    main()
