r"""
build_pf_history.py — PF technical history page (gallery_pf_history.html).

For every PF holding: is today unusual for THIS stock, has it happened before,
what happened next, which patterns just fired, and how it stands against the
market. Every fact carries a plain-English "how" line so the logic is visible
on the page, not only in code. The maths lives in tech_history.py (tested in
scripts/tests/test_tech_history.py); this script only loads and renders.

Prices = local 25y store (guru/data/ohlcv_hist/<ISIN>.parquet, deep but
refreshed rarely) joined to Drive data/ohlcv/<SYM>.parquet (from 2016, fresh);
Drive wins on shared dates, junction checked (tech_history.join_series).
Indices via build_market_state_html._close_series (same local+Drive join).

Read-only: no Drive writes, no Gemini, no CI. Local page only.

Usage:
    python scripts/build_pf_history.py --dry-run --symbols STLTECH,SANSERA
    python scripts/build_pf_history.py              # all holdings -> page
"""
from __future__ import annotations

import argparse
import os
import sys
import webbrowser
from datetime import datetime

import numpy as np
import pandas as pd

_SD = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _SD)
import build_gallery as BG                       # noqa: E402  (loads .env)
import tech_history as T                         # noqa: E402
from drive_io import ParquetCache               # noqa: E402

ROOT = os.path.dirname(_SD)
LOCAL_OHLCV = os.path.join(ROOT, "guru", "data", "ohlcv_hist")
MACRO_LOCAL = os.path.join(ROOT, "guru", "data", "macro_hist")


def _close_series(drive, folder_id, name):
    """Index close by date: deep local guru/data/macro_hist (stale) joined to
    Drive data/indices (CI-fresh, from 2016); Drive wins on shared dates. Same
    rule as build_market_state_html._close_series, kept local so this page
    does not depend on that script's newer revision."""
    frames = []
    loc = os.path.join(MACRO_LOCAL, f"{name}.parquet")
    if os.path.exists(loc):
        try:
            frames.append(pd.read_parquet(loc, columns=["date", "close"]))
        except Exception:
            pass
    dr = BG._read_parquet(drive, folder_id, f"{name}.parquet")
    if not dr.empty and {"date", "close"} <= set(dr.columns):
        frames.append(dr[["date", "close"]])            # appended LAST, so it wins
    if not frames:
        return None
    s = pd.concat(frames, ignore_index=True)
    s["date"] = pd.to_datetime(s["date"], errors="coerce").dt.normalize()
    s["close"] = pd.to_numeric(s["close"], errors="coerce")
    s = s.dropna().drop_duplicates("date", keep="last").sort_values("date")
    return s.set_index("date")["close"]
UNIVERSE_HIST = os.path.join(ROOT, "guru", "data", "universe_hist.parquet")
log = BG.log

# size group -> its own benchmark index (Drive data/indices file stem, label)
SIZE_INDEX = {"Largecap": ("NIFTY_50", "Nifty 50"),
              "Midcap": ("NIFTY_MIDCAP_100", "Midcap 100"),
              "Smallcap": ("NIFTY_SMALLCAP_100", "Smallcap 100"),
              "Microcap": ("NIFTY_SMALLCAP_100", "Smallcap 100")}

# plain-English definition of every pattern flag (shown on the page)
PATTERN_HOW = {
    "golden_cross_50_200": "50-day average crossed ABOVE the 200-day average",
    "death_cross_50_200": "50-day average crossed BELOW the 200-day average",
    "cross_up_20_50": "20-day average crossed above the 50-day average",
    "cross_down_20_50": "20-day average crossed below the 50-day average",
    "reclaim_50": "closed back ABOVE the 50-day average after 20+ days below it",
    "lose_50": "closed BELOW the 50-day average after 20+ days above it",
    "reclaim_200": "closed back ABOVE the 200-day average after 20+ days below it",
    "lose_200": "closed BELOW the 200-day average after 20+ days above it",
    "higher_highs_lows": "last 20 days' high AND low are both above the 20 days before (uptrend structure began)",
    "lower_highs_lows": "last 20 days' high AND low are both below the 20 days before (downtrend structure began)",
    "high_52w_on_volume": "closed above the prior 52-week high on volume at least 1.5x its 50-day average",
    "low_52w": "closed below the prior 52-week low",
    "squeeze_break_up": "price range was in its tightest 10% of the past year, then closed above the upper band",
    "squeeze_break_down": "price range was in its tightest 10% of the past year, then closed below the lower band",
    "pocket_pivot": "up day on volume bigger than any down day's volume in the prior 10 days",
    "distribution_5_in_25": "5th 'distribution day' (down 0.2%+ on higher volume than the day before) within 25 sessions",
    "rsi_bear_divergence": "new 20-day closing high, but RSI below its level at the previous high (momentum fading)",
    "rsi_bull_divergence": "new 20-day closing low, but RSI above its level at the previous low (selling fading)",
    "failed_breakout": "broke above the 52-week high, then fell back below that level within 15 days",
    "close_below_21ema_after_run": "closed below the 21-day EMA after 20+ days above it (a common exit rule)",
}
PATTERN_NAME = {
    "golden_cross_50_200": "Golden cross (50/200)", "death_cross_50_200": "Death cross (50/200)",
    "cross_up_20_50": "20-day crossed above 50-day", "cross_down_20_50": "20-day crossed below 50-day",
    "reclaim_50": "Reclaimed the 50-day average", "lose_50": "Lost the 50-day average",
    "reclaim_200": "Reclaimed the 200-day average", "lose_200": "Lost the 200-day average",
    "higher_highs_lows": "Higher highs & higher lows", "lower_highs_lows": "Lower highs & lower lows",
    "high_52w_on_volume": "52-week high on volume", "low_52w": "New 52-week low",
    "squeeze_break_up": "Squeeze broke UP", "squeeze_break_down": "Squeeze broke DOWN",
    "pocket_pivot": "Pocket pivot", "distribution_5_in_25": "5 distribution days in 25",
    "rsi_bear_divergence": "Bearish RSI divergence", "rsi_bull_divergence": "Bullish RSI divergence",
    "failed_breakout": "Failed breakout", "close_below_21ema_after_run": "Closed below 21-day EMA after a run",
}


# ───────────────────────────── loading ──────────────────────────────────────
def _mcap_map(drive):
    """Same two-source market cap as build_gallery.main (summary snapshot,
    overridden by the daily share-count x close file where a count exists)."""
    out = {}
    summ = BG._read_parquet(drive, BG._folder(drive, "fundamentals"), "summary.parquet")
    if not summ.empty and "symbol" in summ.columns:
        for s, v in zip(summ["symbol"].astype(str).str.upper(),
                        pd.to_numeric(summ.get("market_cap_cr"), errors="coerce")):
            if pd.notna(v) and v > 0:
                out[s] = float(v)
    uni_dir = BG._folder(drive, "universe")
    sc = BG._read_csv(drive, uni_dir, "share_count.csv")
    mc = BG._read_csv(drive, uni_dir, "market_cap.csv")
    if not sc.empty and not mc.empty and "symbol" in sc.columns \
            and {"symbol", "market_cap_cr"} <= set(mc.columns):
        have = set(sc["symbol"].astype(str).str.upper())
        for s, v in zip(mc["symbol"].astype(str).str.upper(),
                        pd.to_numeric(mc["market_cap_cr"], errors="coerce")):
            if s in have and pd.notna(v) and v > 0:
                out[s] = float(v)
    return out


def _segment(mc):
    if mc is None or pd.isna(mc):
        return None
    return ("Largecap" if mc >= 20000 else "Midcap" if mc >= 5000
            else "Smallcap" if mc >= 500 else "Microcap")


def _holdings(drive):
    """[(symbol, isin, name)] for every PF holding — same ISIN->symbol map as
    the PF gallery (isin_symbol_map over master_list + grades + guidance)."""
    idx = BG._folder(drive, "company_repo/_index")
    uni = BG._read_csv(drive, BG._folder(drive, "universe"), "master_list.csv")
    grades = BG._read_parquet(drive, idx, "screener_grades.parquet")
    guidance = BG._read_parquet(drive, idx, "guidance_tracker.parquet")
    isin2sym = BG.isin_symbol_map(uni, grades, guidance)
    hold = BG._load_pf_holdings(drive)
    rows = []
    for _, r in hold.iterrows():
        isin = str(r["isin"])
        rows.append((isin2sym.get(isin, ""), isin, str(r.get("name", "") or "")))
    return rows


def _local_frame(isin):
    """Deep local history for one ISIN (or its BSE-code file)."""
    paths = [os.path.join(LOCAL_OHLCV, f"{isin}.parquet")]
    try:
        u = pd.read_parquet(UNIVERSE_HIST, columns=["isin", "bse_code"])
        for b in u.loc[u["isin"] == isin, "bse_code"].dropna():
            paths.append(os.path.join(LOCAL_OHLCV, f"BSE{str(b).split('.')[0]}.parquet"))
    except Exception:
        pass
    frames = [pd.read_parquet(p) for p in paths if os.path.exists(p)]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


BASE_RATES = os.path.join(ROOT, "guru", "data", "metrics", "pattern_base_rates.parquet")
_BUCKET_WORD = {"first_year": "stocks in their first year after listing",
                "Largecap": "large caps", "Midcap": "mid caps", "Smallcap": "small caps",
                "Microcap": "micro caps", "all": "all tradable stocks"}


def _load_base_rates():
    """{(bucket, pattern, h): row} from build_pattern_base_rates.py, or {}."""
    if not os.path.exists(BASE_RATES):
        log("  no pattern_base_rates.parquet — run scripts/build_pattern_base_rates.py "
            "for the 'similar stocks' lines")
        return {}, None
    br = pd.read_parquet(BASE_RATES)
    built = pd.to_datetime(br["built"].iloc[0]).date() if "built" in br else None
    return {(r.bucket, r.pattern, int(r.h)): r for r in br.itertuples()}, built


def _peer_txt(br, bucket, pattern):
    """'similar stocks' line: the pattern's pooled 20d outcome vs any ordinary
    day in the same bucket. '' when the table or the bucket row is missing."""
    if not br:
        return ""
    for b in (bucket, "all"):
        p, a = br.get((b, pattern, 20)), br.get((b, "any_day", 20))
        if p is not None and a is not None and p.n_events >= 30:
            edge = (p.pct_up - a.pct_up)
            word = ("better than" if edge >= 3 else "worse than" if edge <= -3
                    else "about the same as")
            return (f"Across {_BUCKET_WORD.get(b, b)} (since 2016, {int(p.n_events):,} cases in "
                    f"{int(p.n_stocks):,} stocks): 20 days later median {_pc(p.median)}, up "
                    f"{p.pct_up:.0f}% of the time, bad case (10th pct) {_pc(p.p10, 0)} — "
                    f"<b>{word}</b> any ordinary day ({_pc(a.median)}, up {a.pct_up:.0f}%).")
    return ""


# ───────────────────────────── text helpers ─────────────────────────────────
def _pc(x, nd=1):
    return "—" if x is None or pd.isna(x) else f"{x * 100:+.{nd}f}%"


def _fwd_txt(fwd):
    bits = []
    for h in T.HORIZONS:
        s = fwd.get(h) or {}
        if not s.get("n"):
            bits.append(f"{h}d: no outcome yet")
            continue
        bits.append(f"{h}d {_pc(s['median'])} median, up {s['pct_up']:.0f}% of the "
                    f"time, worst {_pc(s['worst'], 0)} (n={s['n']})")
    return " · ".join(bits)


def _pct_word(p):
    if p is None:
        return ""
    n = int(round(p))
    suf = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suf} percentile"


def _seen_txt(f):
    """'happened N× before', or 'longest ever' for a run with no longer past run."""
    if f.get("longer_before") == 0 and f["n"] == 0:
        return " · <b>longest ever</b> — never this long before"
    return f' · happened <b>{f["n"]}×</b> before'


FACT_HOW = {
    "move": "Counted every past {span} move at least this big in the same direction. "
            "Moves within {cd} days of each other count once. Percentile = where today's "
            "move ranks among all {span} moves in this stock's history.",
    "streak": "Counted every past run of consecutive {dir} closes that reached this length. "
              "'Longer before' = past runs that went on even longer.",
    "ma_run": "A 'run' is consecutive days closing on the same side of the {m}-day average. "
              "Counted past runs on this side that lasted at least this long; outcomes "
              "are measured from the day each past run reached today's length.",
    "stretch": "Distance between price and the {m}-day average, measured in ATRs (average "
               "daily range), so a quiet and a volatile stock are judged on the same scale. "
               "Counted past times it was stretched at least this far.",
    "volume": "Today's volume divided by the MEDIAN of the previous 50 days (a median, "
              "so one huge block-deal day does not distort 'normal'). Counted past days "
              "at least this busy (if above 1x) or at least this quiet (if below 1x).",
    "vol_regime": "Last 20 days' volatility divided by the last year's. Below 1 = calmer "
                  "than usual (squeeze), above 1 = wilder (expansion). Counted past "
                  "periods at least this extreme.",
    "range52": "Where the close sits between the 52-week low (0%) and high (100%). "
               "Counted past times it sat at least this far toward the same end.",
}


def _fact_how(f, years):
    k = f["key"]
    if k.startswith("move_"):
        span = {"move_1d": "1-day", "move_5d": "5-day", "move_20d": "20-day"}[k]
        cd = {"move_1d": 5, "move_5d": 5, "move_20d": 20}[k]
        t = FACT_HOW["move"].format(span=span, cd=cd)
    elif k == "streak":
        t = FACT_HOW["streak"].format(dir="up" if f["side"] == "bull" else "down")
    elif k.startswith("ma") and k.endswith("_run"):
        t = FACT_HOW["ma_run"].format(m=f["ma"])
    elif k.startswith("stretch_"):
        t = FACT_HOW["stretch"].format(m=k.split("_")[1])
    else:
        t = FACT_HOW.get(k, "")
    return t + f" History searched: {years}y."


# ───────────────────────────── per-stock analysis ───────────────────────────
def analyse(sym, df, note, nifty, size_idx, size_label, seg, n500):
    pers = T.personality(df, nifty)
    unus = T.unusual(df)
    pat = T.patterns(df)
    rel = [T.vs_market(df, nifty, "Nifty 50")]
    if size_idx is not None and size_label != "Nifty 50":
        rel.append(T.vs_market(df, size_idx, size_label))
    rel = [r for r in rel if r]
    return {"sym": sym, "note": note, "pers": pers, "unusual": unus, "pat": pat,
            "rel": rel, "seg": seg, "ctx": T.market_context(n500),
            "verdict": T.verdict(unus, pat, rel)}


def text_report(a):
    """Plain-text version for --dry-run."""
    p = a["pers"]
    L = [f"\n=== {a['sym']} — {p['bars']} bars ({p['years']}y from {p['first']}) — {a['seg'] or 'size ?'}",
         f"  prices: {a['note']}"]
    if p["bars"] < T.MIN_BARS_HISTORY:
        L.append("  TOO NEW for history percentiles (< 1 year of bars)")
    L.append("  VERDICT: " + (" | ".join(v.get("text") or PATTERN_NAME.get(v["key"], v["key"])
                                        for v in a["verdict"]) or "nothing extreme today"))
    for f in a["unusual"]:
        L.append(f"  - {f['text']}  [{_pct_word(f['pct'])}; n={f['n']}{' THIN' if f['thin'] else ''}"
                 + (f"; longer before={f['longer_before']}" if "longer_before" in f else "") + "]")
        L.append(f"      next: {_fwd_txt(f['fwd'])}")
    pt = a["pat"]
    L.append(f"  PATTERNS: {pt['stage']} · net score {pt['score']:+d} ({pt['bull']} bull / {pt['bear']} bear)")
    for f in pt["fired"]:
        L.append(f"  - [{f['side']}] {PATTERN_NAME[f['key']]} on {f['date']} ({f['bars_ago']}d ago); "
                 f"past n={f['n']}{' THIN' if f['thin'] else ''}")
        L.append(f"      next: {_fwd_txt(f['fwd'])}")
        if f.get("peer"):
            L.append(f"      similar: {f['peer'].replace('<b>', '').replace('</b>', '')}")
    for r in a["rel"]:
        L.append(f"  VS {r['label']}: excess {', '.join(f'{k} {v:+.1f}%' for k, v in r['excess'].items())}; "
                 f"quadrant {r.get('quadrant')}; RS 52w high={r['rs_52w_high']} low={r['rs_52w_low']}; "
                 f"beta {r['beta'] if r['beta'] is None else round(r['beta'], 2)}")
    L.append(f"  MARKET: {a['ctx']}")
    return "\n".join(L)


# ───────────────────────────── HTML ─────────────────────────────────────────
_CSS = ("font-size:12.5px;color:#222;line-height:1.5;margin:4px 0")
_HOW = "font-size:11px;color:#6b6b6b;font-style:italic;margin:1px 0 5px 14px"
_SIDE = {"bull": "#1a7a3a", "bear": "#c0392b", None: "#0b5394"}


def _box(title, color, body, bg):
    return (f'<div style="background:{bg};border-left:3px solid {color};padding:6px 10px;'
            f'margin:4px 0;{_CSS}"><b style="color:{color}">{title}</b>{body}</div>')


def html_block(a):
    p, sym = a["pers"], a["sym"]
    yrs = p["years"]
    out = []
    # header strip: history depth + how prices were assembled
    out.append(f'<div style="{_CSS}"><b>History:</b> {p["bars"]:,} trading days '
               f'({yrs}y, from {p["first"]}) · <span style="color:#777">{a["note"]}</span>'
               + (f' · <b>{a["seg"]}</b>' if a["seg"] else "") + "</div>")

    # verdict
    if a["verdict"]:
        items = "".join(
            f'<li style="color:{_SIDE.get(v.get("side"))}"><b>'
            f'{v.get("text") or PATTERN_NAME.get(v["key"], v["key"])}</b>'
            + (f' — {_pct_word(v.get("pct"))}' if v.get("pct") is not None else "")
            + (" (longest ever)" if v.get("longer_before") == 0 and v.get("n") == 0
               else f' (seen {v["n"]}× before{", thin" if v.get("thin") else ""})' if "n" in v else "")
            + "</li>" for v in a["verdict"])
        out.append(_box("⚡ What stands out today", "#e65100",
                        f'<ul style="margin:3px 0 2px 18px;padding:0">{items}</ul>'
                        f'<div style="{_HOW}">Picked automatically: facts at/beyond the 90th or '
                        f'10th percentile of this stock\'s own history, patterns that fired in '
                        f'the last 10 sessions, and a relative-strength line at a 52-week '
                        f'high/low. Most extreme first, max 3.</div>', "#fff3e0"))
    else:
        out.append(_box("⚡ What stands out today", "#e65100",
                        " — nothing extreme: every reading is inside this stock's normal range.",
                        "#fff3e0"))

    # 2) is now unusual
    if p["bars"] < T.MIN_BARS_HISTORY:
        body = (f'<div style="{_HOW}">Under one year of trading ({p["bars"]} days), so there '
                f'is not enough of its own past to say what is unusual. Patterns and the '
                f'market comparison below still apply.</div>')
    else:
        rows = []
        for f in a["unusual"]:
            extreme = f["pct"] is not None and (f["pct"] >= 90 or f["pct"] <= 10)
            rows.append(
                f'<div style="margin-top:4px">{"🔶 " if extreme else "• "}'
                f'<b style="color:{_SIDE.get(f.get("side"))}">{f["text"]}</b>'
                + (f' — {_pct_word(f["pct"])}' if f["pct"] is not None else "")
                + _seen_txt(f)
                + (' <span style="background:#b0bec5;color:#fff;padding:0 5px;border-radius:4px;'
                   'font-size:10px">thin</span>' if f["thin"] and not (f.get("longer_before") == 0 and f["n"] == 0) else "")
                + (f' · longer only {f["longer_before"]}×' if f.get("longer_before") else "")
                + f'<div style="{_CSS};margin-left:14px">What happened next: {_fwd_txt(f["fwd"])}</div>'
                + f'<div style="{_HOW}">How: {_fact_how(f, yrs)}</div></div>')
        body = "".join(rows) + (
            f'<div style="{_HOW}">🔶 = at or beyond the 90th/10th percentile of its own '
            f'history. "Thin" = fewer than {T.THIN_N} past cases, so read the outcomes as '
            f'anecdotes, not odds.</div>')
    out.append(_box("2 · Is today unusual for this stock?", "#0b5394", body, "#eef4fb"))

    # 3) patterns
    pt = a["pat"]
    rows = [f'<div><b>{pt["stage"] or "Stage: not enough data"}</b> · net pattern score '
            f'<b>{pt["score"]:+d}</b> ({pt["bull"]} bullish, {pt["bear"]} bearish fired in the '
            f'last 10 sessions)</div>',
            f'<div style="{_HOW}">Stage uses the 150-day (30-week) average: price above + '
            f'average rising = Stage 2 uptrend; above + falling = Stage 3 topping; below + '
            f'falling = Stage 4 downtrend; below + rising = Stage 1 basing.</div>']
    for f in pt["fired"]:
        rows.append(
            f'<div style="margin-top:4px"><b style="color:{_SIDE[f["side"]]}">'
            f'{"▲" if f["side"] == "bull" else "▼"} {PATTERN_NAME[f["key"]]}</b> on {f["date"]} '
            f'({f["bars_ago"]} sessions ago) · this stock did it <b>{f["n"]}×</b> before'
            + (' <span style="background:#b0bec5;color:#fff;padding:0 5px;border-radius:4px;'
               'font-size:10px">thin</span>' if f["thin"] else "")
            + f'<div style="{_CSS};margin-left:14px">What happened next (this stock): {_fwd_txt(f["fwd"])}</div>'
            + (f'<div style="{_CSS};margin-left:14px">Similar stocks: {f["peer"]}</div>' if f.get("peer") else "")
            + f'<div style="{_HOW}">Rule: {PATTERN_HOW[f["key"]]}.'
            + (" Similar stocks = every tradable stock (20-day median turnover ≥ ₹1 Cr/day) "
               "in the same group, pooled from the local price store since 2016; group = "
               "first year after listing if this stock is under a year old, else its size "
               "today. 'Any ordinary day' = every 10th day of those same stocks, flag or "
               "not — the bar a pattern must beat. Caveat: size is today's, and only "
               "stocks still listed are in the store, so both rates lean optimistic."
               if f.get("peer") else "")
            + "</div></div>")
    if not pt["fired"]:
        rows.append(f'<div style="{_HOW}">No bullish or bearish pattern fired in the last '
                    f'10 sessions.</div>')
    out.append(_box("3 · Is a pattern emerging?", "#6a1b9a", "".join(rows), "#f6effa"))

    # 4) vs market
    rows = []
    for r in a["rel"]:
        ex = " · ".join(f'{k} <b style="color:{"#1a7a3a" if v >= 0 else "#c0392b"}">{v:+.1f}%</b>'
                        for k, v in r["excess"].items())
        rsx = (" · RS line at a <b style='color:#1a7a3a'>52-week HIGH</b>" if r["rs_52w_high"]
               else " · RS line at a <b style='color:#c0392b'>52-week LOW</b>" if r["rs_52w_low"]
               else (f" · RS line at the {_pct_word(r['rs_pct_in_year'])} of its year"
                     if r["rs_pct_in_year"] is not None else ""))
        dn = r.get("on_index_down_1pct")
        dtxt = (f" · on the {dn['n']} days {r['label']} fell 1%+ in the last year it averaged "
                f"<b>{dn['stock_avg']:+.1f}%</b> vs the index {dn['index_avg']:+.1f}%" if dn else "")
        cap = (f" · up-capture {r['up_capture']:.0f}% / down-capture {r['down_capture']:.0f}%"
               if r.get("up_capture") is not None and r.get("down_capture") is not None else "")
        al = (f" · beta {r['beta']:.2f}; return beyond what beta explains: 20d "
              f"{r.get('alpha_20d', 0):+.1f}%, 60d {r.get('alpha_60d', 0):+.1f}%"
              if r.get("beta") is not None else "")
        rows.append(f'<div style="margin-top:4px"><b>vs {r["label"]}: {r.get("quadrant", "—")}</b>'
                    f' · beat/lag by {ex}{rsx}{dtxt}{cap}{al}</div>')
    rows.append(
        f'<div style="{_HOW}">Beat/lag = stock return minus index return over each window. '
        f'RS line = stock price ÷ index. Leading = RS above its 50-day average AND rising over '
        f'10 days; Weakening = above but falling; Improving = below but rising; Lagging = '
        f'below and falling. Up/down-capture = the stock\'s average move on index up/down days '
        f'as a % of the index\'s (down-capture under 100% = falls less). Size index = '
        f'Smallcap 100 for small/micro caps, Midcap 100 for mid caps (both only from May 2023).'
        f'</div>')
    if a["ctx"]:
        rows.append(f'<div><b>Market backdrop:</b> {a["ctx"]}.</div>')
    out.append(_box("4 · Stronger or weaker than the market?", "#00695c", "".join(rows), "#e8f5f2"))

    # 1) personality (reference, last)
    if p.get("pct_up_days") is not None:
        bd = p["big_days_per_year"]
        ma_txt = " · ".join(f'{m}d: {s["pct_time_above"]:.0f}% of time above'
                            for m, s in p.get("ma", {}).items())
        cr = p["corrections_10pct"]
        su, sd = p["streak_up"], p["streak_down"]
        g = p["gaps"]
        body = (
            f'<div>Up on <b>{p["pct_up_days"]:.0f}%</b> of days · typical daily move '
            f'<b>{p["median_abs_move"]:.1f}%</b> · yearly volatility {p["vol_annual"]:.0f}% · '
            f'ATR {p["atr_pct_now"] or 0:.1f}% of price (usual {p["atr_pct_median"] or 0:.1f}%)</div>'
            f'<div>Big days per year: +3% {bd["+3"]:.0f} / −3% {bd["-3"]:.0f} · +5% {bd["+5"]:.0f} / '
            f'−5% {bd["-5"]:.0f} · +10% {bd["+10"]:.1f} / −10% {bd["-10"]:.1f} · best day '
            f'{p["best_day"]:+.0f}%, worst {p["worst_day"]:+.0f}%</div>'
            f'<div>Streaks: up runs usually {su["median"] or 0:.0f}d (longest {su["max"]}), '
            f'down runs usually {sd["median"] or 0:.0f}d (longest {sd["max"]}); 5+ day runs per '
            f'year: up {su["five_plus_per_year"]:.1f}, down {sd["five_plus_per_year"]:.1f}</div>'
            f'<div>Gaps of 2%+ per year: up {g["up_per_year"]:.0f}, down {g["down_per_year"]:.0f}'
            + (f' · same-day fill: up {g["up_fill_pct"]:.0f}%' if g["up_fill_pct"] is not None else "")
            + (f', down {g["down_fill_pct"]:.0f}%' if g["down_fill_pct"] is not None else "")
            + f'</div><div>Trend: {ma_txt}</div>'
            f'<div>Drawdowns: worst {p["max_drawdown"]:.0f}%, now {p["drawdown_now"]:.0f}% · '
            f'{cr["count"]} corrections of 10%+'
            + (f', typical depth {cr["median_depth"]:.0f}%' if cr["median_depth"] is not None else "")
            + (f', typical recovery {cr["median_recovery_days"]:.0f} days' if cr["median_recovery_days"] is not None else "")
            + f' · volume spikes (2x normal) {p["volume_spikes_per_year"]:.0f}/year'
            + (f' · beta vs Nifty 50 (1y) {p["beta_1y"]:.2f}' if p.get("beta_1y") is not None else "")
            + "</div>"
            f'<div style="{_HOW}">Computed over the whole history shown above. A one-day '
            f'move over 40% is treated as bad data (e.g. an unadjusted split) and left out of '
            f'every count. A correction = a fall of 10%+ from a high before a new high; '
            f'recovery = days from its low back to a new high.</div>')
        out.append(_box("1 · How does this stock normally behave?", "#5d4037", body, "#f7f3f0"))
    return "".join(out)


# ───────────────────────────── main ─────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--symbols", default="", help="comma list; default = every PF holding")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the facts per stock; write no page")
    ap.add_argument("--timeframe-days", type=int, default=750,
                    help="bars drawn on each chart (default ~3y)")
    ap.add_argument("--out", default="")
    ap.add_argument("--no-open", action="store_true")
    args = ap.parse_args()

    drive = BG.get_drive()
    log("loading PF holdings…")
    hold = _holdings(drive)
    want = {s.strip().upper() for s in args.symbols.split(",") if s.strip()}
    if want:
        missing = want - {s.upper() for s, _, _ in hold}
        if missing:
            log(f"  not in the PF (skipped): {', '.join(sorted(missing))}")
        hold = [h for h in hold if h[0].upper() in want]
    no_sym = [n or i for s, i, n in hold if not s]
    hold = [h for h in hold if h[0]]
    log(f"  {len(hold)} holdings with a symbol"
        + (f"; {len(no_sym)} without one (not charted): {', '.join(no_sym)}" if no_sym else ""))
    if not hold:
        log("Nothing to do."); return

    mcap = _mcap_map(drive)
    log("downloading Drive prices…")
    drive_px = BG._bulk_parquet(drive, BG._folder(drive, "data/ohlcv"),
                                [s for s, _, _ in hold], cache=ParquetCache(enabled=True),
                                what="OHLCV")
    idx_folder = BG._folder(drive, "data/indices")
    nifty = _close_series(drive, idx_folder, "NIFTY_50")
    n500 = _close_series(drive, idx_folder, "NIFTY_500")
    size_series = {k: _close_series(drive, idx_folder, k)
                   for k in {"NIFTY_MIDCAP_100", "NIFTY_SMALLCAP_100"}}

    br, br_built = _load_base_rates()
    if br_built:
        log(f"  similar-stock base rates built {br_built}")
    results, joined = [], {}
    for sym, isin, name in hold:
        df, note = T.join_series(_local_frame(isin), drive_px.get(sym))
        if df.empty:
            log(f"  {sym}: no price data anywhere — skipped")
            continue
        seg = _segment(mcap.get(sym.upper()))
        stem, lbl = SIZE_INDEX.get(seg, ("NIFTY_50", "Nifty 50"))
        sidx = nifty if stem == "NIFTY_50" else size_series.get(stem)
        a = analyse(sym, df, note, nifty, sidx, lbl, seg, n500)
        a["name"] = name
        bucket = "first_year" if len(df) < T.MIN_BARS_HISTORY else (seg or "all")
        for f in a["pat"]["fired"]:
            f["peer"] = _peer_txt(br, bucket, f["key"])
        results.append(a)
        joined[sym] = df
        if args.dry_run:
            print(text_report(a))

    if args.dry_run:
        log(f"DRY-RUN — {len(results)} stocks analysed; no page written.")
        return

    # most "interesting" first: more stand-out facts, then stronger pattern score
    results.sort(key=lambda a: (-len(a["verdict"]), -abs(a["pat"]["score"])))
    ranked = pd.DataFrame({"symbol": [a["sym"] for a in results],
                           "_pfname": [a["name"] for a in results]})
    annot = {a["sym"].upper(): html_block(a) for a in results}
    E = pd.DataFrame()
    cards = BG.Cards(E, {}, E, E, E)
    prelude = (
        '<div style="max-width:1100px;margin:0 auto 12px;font-size:12.5px;color:#333;'
        'background:#fff;border:1px solid #e3e7ee;border-radius:8px;padding:8px 12px;'
        'line-height:1.55"><b>How to read this page.</b> Every number compares the stock '
        'with its OWN past. For each fact: how rare today is (percentile), how many separate '
        'times it happened before, and what the stock did 5, 20 and 60 trading days after '
        'each of those times (median return, how often it was up, and the worst case). '
        'Cards with the most stand-out facts come first. Sections run 2 → 4 (today), then '
        '1 (the stock\'s normal behaviour, for reference). Past outcomes describe this '
        'stock\'s history; they are not a forecast.</div>')
    html = BG.build_html(ranked, joined, cards, mcap, args.timeframe_days,
                         title="📜 PF technical history", annot=annot, prelude=prelude)
    out = args.out or os.path.join(ROOT, "gallery_pf_history.html")
    with open(out, "w", encoding="utf-8") as f:
        f.write(html)
    log(f"wrote {out}  ({len(results)} stocks)")
    if not args.no_open:
        webbrowser.open("file:///" + out.replace("\\", "/"))


if __name__ == "__main__":
    main()
