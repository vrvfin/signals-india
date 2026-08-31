r"""
GURU GALLERY — build_guru_gallery.py  (Project Guru)

Renders the daily screen's picks as a standalone chart gallery,
`gallery_guru.html`, sitting alongside gallery.html / gallery_additions.html /
gallery_guidance_watchlist.html and looking identical to them.

WHY THIS IS A SEPARATE FILE, NOT A NEW --mode IN build_gallery.py
    scripts/build_gallery.py is imported here as a LIBRARY and never modified.
    Every card, chart, meta-line and bit of page furniture comes from its
    build_html(), so a guru card is byte-for-byte a gallery card — but the
    guru work lives entirely in guru/, which keeps it out of the way of
    concurrent edits to the shared gallery.

INPUT
    guru/backtest/guru_picks.parquet   written by daily_screen.py
    ...or company_repo/_index/guru_picks.parquet on Drive if there is no local
    copy (so the gallery works off a CI run you never executed locally).

USAGE
    python guru/build_guru_gallery.py                 # top 100 by conviction
    python guru/build_guru_gallery.py --max 40
    python guru/build_guru_gallery.py --min-rules 8   # only high-agreement names
    python guru/build_guru_gallery.py --dry-run       # resolve inputs, draw nothing
"""
from __future__ import annotations
import argparse, io, os, sys, webbrowser
from datetime import datetime

import pandas as pd

GURU = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(GURU)
BT = os.path.join(GURU, "backtest")
SCRIPTS = os.path.join(ROOT, "scripts")
for p in (SCRIPTS, GURU):
    if p not in sys.path:
        sys.path.insert(0, p)

from pathlib import Path
from dotenv import load_dotenv
load_dotenv(Path(ROOT) / ".env")

# The shared gallery, used strictly read-only.
import build_gallery as BG

PICKS = "guru_picks.parquet"


def log(m):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {m}", flush=True)


def load_picks(drive, prefer: str = "newest") -> tuple[pd.DataFrame, str]:
    """Take whichever copy is NEWER — local or Drive.

    This matters because nothing in gallery_all.bat re-runs the screen: CI
    refreshes Drive every weekday evening, so on any day you have not screened
    locally the Drive copy is the fresh one and the local copy is yesterday's.
    Preferring 'local' blindly would quietly render stale picks.
    """
    local = os.path.join(BT, PICKS)
    l_mt = (datetime.utcfromtimestamp(os.path.getmtime(local))
            if os.path.exists(local) else None)
    idx = BG._folder(drive, "company_repo/_index")
    d_mt, fid = None, None
    try:
        from _extractor_base import find_file
        fid = find_file(BG._drive(drive) if hasattr(BG, "_drive") else drive,
                        idx, PICKS)
    except Exception:
        fid = None
    if fid:
        try:
            meta = drive.files().get(fileId=fid, fields="modifiedTime").execute()
            d_mt = datetime.strptime(meta["modifiedTime"][:19], "%Y-%m-%dT%H:%M:%S")
        except Exception:
            d_mt = None
    if prefer == "local" and l_mt:
        pick = "local"
    elif prefer == "drive" and d_mt:
        pick = "Drive"
    elif l_mt and d_mt:
        pick = "local" if l_mt >= d_mt else "Drive"
    elif l_mt:
        pick = "local"
    elif d_mt:
        pick = "Drive"
    else:
        return pd.DataFrame(), "nowhere"
    stamp = {"local": l_mt, "Drive": d_mt}[pick]
    other = {"local": d_mt, "Drive": l_mt}[pick]
    note = f"{pick} (written {stamp:%Y-%m-%d %H:%M} UTC"
    note += f"; the other copy is {other:%Y-%m-%d %H:%M} UTC)" if other else ")"
    if pick == "local":
        return pd.read_parquet(local), note
    return BG._read_parquet(drive, idx, PICKS), note


def _fmt(v, dp=1, dash="–"):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return dash
    return dash if pd.isna(f) else f"{f:,.{dp}f}"


def annot_for(picks: pd.DataFrame) -> dict:
    """The badge on each card: why this stock is here, in one line."""
    out = {}
    for _, r in picks.iterrows():
        sym = str(r.get("symbol", "") or "")
        if not sym:
            continue
        bits = []
        fs = str(r.get("first_seen", "") or "").strip()
        if fs:
            d = r.get("days_on_list")
            d = int(d) if pd.notna(pd.to_numeric(pd.Series([d]),
                                                 errors="coerce").iloc[0]) else None
            age = ("NEW today" if d == 0 else
                   f"on list {d}d (since {fs})" if d is not None else f"since {fs}")
            bits.append(age)
        bits += [f"Conviction {_fmt(r.get('conviction_score'), 0)}",
                 f"{int(r.get('n_rules') or 0)} rules"]
        br = r.get("best_rule_return")
        if pd.notna(pd.to_numeric(br, errors="coerce")):
            bits.append(f"best rule +{_fmt(br)}%")
        rare = r.get("rarest_rule_hits")
        if pd.notna(pd.to_numeric(rare, errors="coerce")):
            bits.append(f"rarest hits {int(float(rare))}")
        head = ('<div style="background:#1b4332;color:#fff;border-radius:6px 6px 0 0;'
                'padding:4px 10px;font-size:13px;font-weight:700;margin:3px 0 0 0">'
                f'🎯 {" · ".join(bits)}</div>')
        out[sym] = head + why_block(r)
    return out


def why_block(r) -> str:
    """The 'why' under the badge: every rule this stock passed, and the stock's
    OWN numbers against that rule's thresholds. Falls back to the rule names
    alone for a picks table written before evidence was captured."""
    why = str(r.get("why", "") or "").strip()
    if not why:
        names = str(r.get("rule_names", "") or "").strip()
        if not names:
            return ""
        items = [n.strip() for n in names.split("|") if n.strip()]
    else:
        items = [n.strip() for n in why.split("¦") if n.strip()]
    rows = []
    for it in items:
        if "→" in it:
            rule, ev = it.split("→", 1)
            rows.append(
                f'<li style="margin:2px 0"><span style="color:#1b4332;'
                f'font-weight:600">{rule.strip()}</span>'
                f'<span style="color:#555"> — {ev.strip()}</span></li>')
        else:
            rows.append(f'<li style="margin:2px 0"><span style="color:#1b4332;'
                        f'font-weight:600">{it}</span></li>')
    return ('<details open style="background:#f2f7f4;border:1px solid #cfe3d8;'
            'border-top:none;border-radius:0 0 6px 6px;padding:6px 10px;'
            'margin:0 0 3px 0;font-size:12px">'
            '<summary style="cursor:pointer;color:#1b4332;font-weight:600;'
            f'font-size:12px">why it was picked ({len(rows)})</summary>'
            f'<ul style="margin:6px 0 2px 0;padding-left:18px">{"".join(rows)}'
            '</ul></details>')


def prelude_for(picks: pd.DataFrame, as_of: str) -> str:
    """Summary table above the card grid — the whole shortlist at a glance,
    so the page is useful before you scroll a single chart."""
    head = ["#", "Stock", "Added", "Days", "Conviction", "Rules", "MCap (cr)",
            "Turnover (cr/d)", "12M %", "Rules satisfied"]
    th = "".join(
        f'<th style="text-align:left;padding:6px 10px;border-bottom:2px solid #ccc;'
        f'font-size:12px;color:#333;white-space:nowrap">{h}</th>' for h in head)
    rows = []
    for i, (_, r) in enumerate(picks.iterrows(), 1):
        names = str(r.get("rule_names", "") or "")
        d = pd.to_numeric(pd.Series([r.get("days_on_list")]),
                          errors="coerce").iloc[0]
        added = str(r.get("first_seen", "") or "–")
        if pd.notna(d) and int(d) == 0:
            added = f'<b style="color:#0a7d33">{added}</b>'
        cells = [str(i),
                 f'<b>{r.get("name", "") or r.get("symbol", "")}</b> '
                 f'<span style="color:#777">{r.get("symbol", "")}</span>',
                 added,
                 "–" if pd.isna(d) else str(int(d)),
                 _fmt(r.get("conviction_score"), 0),
                 str(int(r.get("n_rules") or 0)),
                 _fmt(r.get("market_cap_cr"), 0),
                 _fmt(r.get("turnover_20d_cr")),
                 _fmt(r.get("ret_12m_pct")),
                 f'<span style="color:#555;font-size:12px">{names[:220]}'
                 f'{"…" if len(names) > 220 else ""}</span>']
        rows.append("<tr>" + "".join(
            f'<td style="padding:6px 10px;border-bottom:1px solid #eee;'
            f'font-size:13px;vertical-align:top">{c}</td>' for c in cells) + "</tr>")
    return (
        '<div style="margin:0 0 18px 0;overflow-x:auto">'
        f'<div style="font-size:13px;color:#555;margin-bottom:6px">'
        f'Ranked by conviction — the sum of (rule rarity × that rule\'s validated '
        f'out-of-sample return) over every rule the stock currently satisfies. '
        f'Prices as of {as_of}. Backtest-derived, excludes costs and slippage — '
        f'research output, not investment advice.</div>'
        '<table style="border-collapse:collapse;width:100%;font-family:'
        f'-apple-system,Segoe UI,Arial">{th}{"".join(rows)}</table></div>')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max", type=int, default=100,
                    help="how many stocks to chart (0 = all; 100 default keeps "
                         "the page a sane size)")
    ap.add_argument("--min-rules", type=int, default=0,
                    help="only stocks satisfying at least this many rules")
    ap.add_argument("--min-conviction", type=float, default=0.0)
    ap.add_argument("--sort", choices=["new", "conviction"], default="new",
                    help="new = newest onto the list first (default), "
                         "conviction ties broken by score; "
                         "conviction = highest score first")
    ap.add_argument("--source", choices=["newest", "local", "drive"],
                    default="newest",
                    help="which picks table to render (default: whichever of "
                         "the local and Drive copies is newer)")
    ap.add_argument("--new-only", action="store_true",
                    help="only stocks that first appeared in the latest run")
    ap.add_argument("--timeframe-days", type=int, default=252)
    ap.add_argument("--resample", choices=["D", "W", "M"], default="D")
    ap.add_argument("--out", default="")
    ap.add_argument("--no-open", action="store_true")
    ap.add_argument("--no-news", action="store_true")
    ap.add_argument("--news-days", type=int, default=30)
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--purge-cache", action="store_true")
    ap.add_argument("--max-missing-pct", type=float, default=5.0)
    ap.add_argument("--dry-run", action="store_true",
                    help="resolve inputs and report what would be drawn")
    args = ap.parse_args()

    cache = BG.ParquetCache(enabled=not args.no_cache)
    if args.purge_cache:
        cache.purge()
        log(f"parquet cache purged ({cache.root})")

    drive = BG._drive()
    picks, src = load_picks(drive, prefer=args.source)
    if picks.empty:
        log("No guru_picks.parquet (local or Drive) — run "
            "guru/daily_screen.py first.")
        return
    as_of = str(picks["as_of"].iloc[0]) if "as_of" in picks.columns else "?"
    log(f"loaded {len(picks):,} picks from {src} (prices as of {as_of})")

    if args.min_rules > 0 and "n_rules" in picks.columns:
        picks = picks[pd.to_numeric(picks["n_rules"], errors="coerce")
                      >= args.min_rules]
    if args.min_conviction > 0 and "conviction_score" in picks.columns:
        picks = picks[pd.to_numeric(picks["conviction_score"], errors="coerce")
                      >= args.min_conviction]
    if args.new_only and "days_on_list" in picks.columns:
        picks = picks[pd.to_numeric(picks["days_on_list"], errors="coerce") == 0]
    # Newest onto the list first — a name that appeared today is the one you
    # have not looked at yet. Conviction breaks ties, so within a given day the
    # order is still strongest-first. --sort conviction restores the old order.
    if args.sort == "new" and "first_seen" in picks.columns:
        picks["_fs"] = pd.to_datetime(picks["first_seen"], errors="coerce")
        picks = picks.sort_values(["_fs", "conviction_score"],
                                  ascending=[False, False], na_position="last")
        picks = picks.drop(columns=["_fs"])
    else:
        if args.sort == "new":
            log("no first_seen column in this picks table (written before the "
                "ledger existed) — falling back to conviction order")
        picks = picks.sort_values("conviction_score", ascending=False,
                                  na_position="last")
    if args.max > 0:
        picks = picks.head(args.max)
    picks = picks.reset_index(drop=True)
    if picks.empty:
        log("Nothing left after the filters."); return

    uni = BG._read_csv(drive, BG._folder(drive, "universe"), "master_list.csv")
    exch = (dict(zip(uni["symbol"].astype(str), uni["exchange"].astype(str)))
            if not uni.empty and {"symbol", "exchange"} <= set(uni.columns) else {})

    ranked = pd.DataFrame({"symbol": picks["symbol"].astype(str)})
    ranked["_pfname"] = picks.get("name", "")
    ranked["n_strategies"] = pd.to_numeric(picks.get("n_rules"),
                                           errors="coerce").fillna(0).astype(int)
    ranked["_exch"] = ranked["symbol"].map(exch).fillna("NSE")
    syms = [s for s in ranked["symbol"].tolist() if s]

    n_new = int((pd.to_numeric(picks.get("days_on_list"), errors="coerce") == 0).sum()) \
        if "days_on_list" in picks.columns else 0
    order = ("newest onto the list first" if args.sort == "new"
             else "highest conviction first")
    title = (f"🎯 GURU SCREEN — {as_of} — {len(picks)} stocks — {order}"
             + (f" — {n_new} new today" if n_new else ""))
    prelude = prelude_for(picks, as_of)
    annot = annot_for(picks)

    out_default = "gallery_guru.html"
    _rs = {"D": (None, "", ""), "W": ("W-FRI", " · Weekly", "_weekly"),
           "M": ("ME", " · Monthly", "_monthly")}
    resample, tf_title, tf_suffix = _rs[args.resample]
    title += tf_title
    if tf_suffix and not args.out:
        out_default = out_default.replace(".html", f"{tf_suffix}.html")
    out_path = args.out or os.path.join(ROOT, out_default)

    if args.dry_run:
        log(f"DRY RUN — would chart {len(syms)} symbols -> {out_path}")
        print(picks[[c for c in ("symbol", "name", "conviction_score", "n_rules",
                                 "market_cap_cr", "turnover_20d_cr")
                     if c in picks.columns]].head(20).to_string(index=False))
        return

    idx = BG._folder(drive, "company_repo/_index")
    fund = BG._folder(drive, "fundamentals")
    log("loading grades / guidance / summary…")
    grades = BG._read_parquet(drive, idx, "screener_grades.parquet")
    guidance = BG._read_parquet(drive, idx, "guidance_tracker.parquet")
    summ = BG._read_parquet(drive, fund, "summary.parquet")
    mcap_map = {}
    if not summ.empty and "symbol" in summ.columns:
        for _, r in summ.iterrows():
            v = pd.to_numeric(r.get("market_cap_cr"), errors="coerce")
            if pd.notna(v) and v > 0:
                mcap_map[str(r["symbol"]).upper()] = v
    # the screen already knows every pick's mcap — fill anything summary lacks
    for _, r in picks.iterrows():
        s = str(r.get("symbol", "")).upper()
        v = pd.to_numeric(r.get("market_cap_cr"), errors="coerce")
        if s and s not in mcap_map and pd.notna(v) and v > 0:
            mcap_map[s] = float(v)

    log("loading gf1 / announcements / research…")
    gf1 = BG._read_parquet(drive, idx, "gf1_guidance_statements.parquet")
    ann = BG._read_parquet(drive, idx, "announcement_ledger.parquet")
    res_idx = BG._read_parquet(drive, idx, "research_index.parquet")

    log(f"downloading OHLCV for {len(syms)} names…")
    ohlcv_failed = set()
    omap = BG._bulk_parquet(drive, BG._folder(drive, "data/ohlcv"), syms,
                            cache=cache, what="OHLCV",
                            max_missing_pct=args.max_missing_pct,
                            failed_out=ohlcv_failed)
    log("downloading statements…")
    stmts = BG._bulk_parquet(drive, BG._folder(drive, "fundamentals/statements"),
                             syms, cache=cache, what="statements",
                             max_missing_pct=args.max_missing_pct)

    _name_by = {}
    if not grades.empty and {"symbol", "company_name"} <= set(grades.columns):
        _name_by = {str(r["symbol"]).upper(): str(r.get("company_name", "") or "")
                    for _, r in grades.iterrows()}
    news_map = ({} if args.no_news
                else BG._fetch_news_map(syms, _name_by, days=args.news_days))

    cards = BG.Cards(grades, stmts, guidance, gf1, ann, research=res_idx,
                     news=news_map)
    log("assembling HTML…")
    html = BG.build_html(ranked, omap, cards, mcap_map, args.timeframe_days,
                         title=title, annot=annot, resample=resample,
                         prelude=prelude, failed_syms=ohlcv_failed)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    log(f"wrote {out_path}  ({len(html) / 1e6:.1f} MB, {len(syms)} charts)")
    if not args.no_open:
        webbrowser.open("file://" + os.path.abspath(out_path))
        log("opened in default browser.")


if __name__ == "__main__":
    main()
