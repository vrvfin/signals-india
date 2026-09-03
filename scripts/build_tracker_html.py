r"""
build_tracker_html.py — did the calls actually work?  ->  gallery_tracker.html

Renders what signal_tracker.py measures: every signal the engine has made, what
happened to it, and — the part that decides whether any of this is worth
trusting — whether the SCORES predicted anything.

Reads (all written by signal_tracker.py / aggregate_signals.py):
    signals/analysis/signal_outcomes.csv   closed calls, one row each
    signals/analysis/tracker_summary.csv   per-engine scorecard
    signals/aggregated/open_signals.csv    still-running calls

EVERYTHING IS IN R, where 1R = entry - stop, the amount risked. A +2R winner and
a -1R loser then compare directly across a Rs 40 stock and a Rs 4,000 one, which
percentages do not.

THE COLUMN THAT MATTERS MOST is exit_vs_hold_r. A sell can only be judged
against NOT selling, so every closed call also records what simply holding would
have returned. Negative means the exit cut a winner short. Without it, "which
engine gives the best sell calls" is unanswerable — and guru's own exit study
found every trailing stop tested LOST 4-5pp against holding, so this is not a
hypothetical.

HONEST BY DESIGN: below MIN_N closed calls this page says so rather than drawing
a conclusion. A win rate on twelve trades is not a win rate.

Usage:
    python scripts/build_tracker_html.py
    python scripts/build_tracker_html.py --no-open
    python scripts/build_tracker_html.py --dry-run
"""
from __future__ import annotations

import argparse
import io
import os
import sys
import webbrowser
from datetime import datetime

import numpy as np
import pandas as pd

_SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(_SCRIPTS_DIR), ".env"))

try:
    sys.stdout.reconfigure(errors="replace")
    sys.stderr.reconfigure(errors="replace")
except Exception:          # pragma: no cover
    pass

from _extractor_base import (get_drive, get_or_create_subfolder, find_file,
                             download_bytes)

# Below this many closed calls, report nothing and say why. Chosen to match
# signal_tracker.reliability, which refuses to decile a smaller sample.
MIN_N = 20


def log(m):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {m}", flush=True)


def _folder(drive, path):
    fid = os.environ["GDRIVE_FOLDER_ID"]
    for part in path.split("/"):
        fid = get_or_create_subfolder(drive, fid, part)
    return fid


def _csv(drive, path, name):
    fid = find_file(drive, _folder(drive, path), name)
    if not fid:
        return pd.DataFrame()
    try:
        return pd.read_csv(io.BytesIO(download_bytes(drive, fid)))
    except Exception as e:
        log(f"  {name} unreadable ({type(e).__name__})")
        return pd.DataFrame()


def _rcol(v):
    if v is None or pd.isna(v):
        return "#888"
    return "#1a7a3a" if float(v) > 0 else ("#c0392b" if float(v) < 0 else "#888")


def _table(df, cols, fmt=None) -> str:
    if df is None or df.empty:
        return '<p style="color:#888">(nothing yet)</p>'
    fmt = fmt or {}
    head = "".join(f"<th>{c.replace('_', ' ')}</th>" for c in cols)
    rows = []
    for _, r in df.iterrows():
        tds = []
        for c in cols:
            v = r.get(c)
            if c in fmt:
                tds.append(fmt[c](v))
            elif isinstance(v, float):
                tds.append(f"<td>{'' if pd.isna(v) else f'{v:,.2f}'}</td>")
            else:
                tds.append(f"<td>{'' if v is None or (isinstance(v, float) and pd.isna(v)) else v}</td>")
        rows.append("<tr>" + "".join(tds) + "</tr>")
    return f"<table><tr>{head}</tr>{''.join(rows)}</table>"


_TPL = """<!doctype html><html><head><meta charset="utf-8">
<title>Decision tracker __DATE__</title><style>
 body{font-family:-apple-system,Segoe UI,Roboto,Arial,sans-serif;background:#f4f6f9;
      margin:0;padding:14px;color:#222}
 .wrap{max-width:1180px;margin:0 auto}
 h1{font-size:20px;margin:4px 6px} h2{font-size:14px;margin:18px 6px 6px;color:#1a3d6e}
 .card{background:#fff;border:1px solid #e3e7ee;border-radius:8px;padding:10px 12px;
       box-shadow:0 1px 3px rgba(0,0,0,.05);margin:8px 6px}
 table{border-collapse:collapse;width:100%;font-size:12px}
 th,td{padding:4px 8px;border-bottom:1px solid #eee;text-align:right}
 th{color:#666;font-weight:600;background:#fafbfd}
 th:first-child,td:first-child{text-align:left}
 .note{font-size:11px;color:#888;margin:4px 6px}
 .warn{background:#fff8e1;border-left:4px solid #b8860b;padding:8px 12px;
       border-radius:6px;font-size:13px;margin:8px 6px}
 .tiles{display:flex;gap:8px;flex-wrap:wrap;margin:6px 6px}
 .tile{flex:1;min-width:130px;background:#fff;border:1px solid #e3e7ee;
       border-radius:8px;padding:8px 10px;text-align:center}
</style></head><body><div class="wrap">
<h1>🧾 Decision tracker <span style="font-size:12px;color:#888">__DATE__</span></h1>
<div class="note">Everything is in <b>R</b> — one R is what you risked
(entry minus stop). A +2R win and a −1R loss compare directly across any two
stocks, which percentages cannot.</div>
__BANNER__
<div class="tiles">__TILES__</div>
<h2>Per-engine scorecard</h2>
<div class="card">__SUMMARY__
<div class="note"><b>median_r</b> = did the PICK work.
<b>median_exit_vs_hold_r</b> = did the EXIT work; negative means selling cut
winners short versus simply holding.</div></div>
<h2>Does the score predict anything?</h2>
<div class="card">__RELIABILITY__
<div class="note">Decile lift is top-decile minus bottom-decile realised R. Near
zero means the score does not separate outcomes, however sensible it looks.
n is always shown: a strong correlation on a handful of trades is not evidence.</div></div>
<h2>How far trades ran (MFE) — what the target should be set from</h2>
<div class="card">__MFE__</div>
<h2>Closed calls</h2>
<div class="card">__CLOSED__</div>
<h2>Still open</h2>
<div class="card">__OPEN__</div>
</div></body></html>"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-open", action="store_true")
    ap.add_argument("--dry-run", action="store_true",
                    help="compute and report, write no file")
    ap.add_argument("--out", default="gallery_tracker.html")
    args = ap.parse_args()

    print("Decision tracker page")
    print("-" * 50)
    drive = get_drive()
    out = _csv(drive, "signals/analysis", "signal_outcomes.csv")
    summ = _csv(drive, "signals/analysis", "tracker_summary.csv")
    opens = _csv(drive, "signals/aggregated", "open_signals.csv")
    log(f"closed={len(out):,} | summary rows={len(summ):,} | open={len(opens):,}")

    n_closed = len(out)
    banner = ""

    # ---- is the closed sample MATURE, not merely large? ---------------------
    # A stop can be hit in a single day; a 2R target essentially cannot. So the
    # first calls to close are almost all losers, and a win rate computed on
    # them is not a win rate — it is a measure of how long the tracker has been
    # running. Counting rows alone does not catch this: the sample reached
    # exactly MIN_N with 20 stops, 0 targets and a 0% win rate, which would have
    # rendered as a finding.
    immature = False
    if n_closed:
        stat = out.get("status")
        n_stop = int((stat == "stopped").sum()) if stat is not None else 0
        n_tgt = int((stat == "target_hit").sum()) if stat is not None else 0
        held = pd.to_numeric(out.get("days_held"), errors="coerce").dropna()
        med_held = float(held.median()) if len(held) else 0.0
        immature = (n_tgt == 0 and n_stop > 0) or med_held < 10
    if n_closed < MIN_N:
        banner = (f'<div class="warn"><b>Not enough history yet — '
                  f'{n_closed} closed call{"" if n_closed == 1 else "s"}.</b> '
                  f'Nothing on this page is a conclusion until roughly {MIN_N} '
                  f'have closed. The engine only began recording entry and stop '
                  f'prices recently, so this fills in over the coming weeks as '
                  f'trades reach their target, their stop, or their time limit. '
                  f'A win rate on {n_closed} trades is not a win rate.</div>')
    elif immature:
        banner = (f'<div class="warn"><b>The closed sample is young, and biased '
                  f'toward losses by construction.</b> {n_closed} calls have '
                  f'closed: {n_stop} stopped, {n_tgt} reached target, median '
                  f'holding {med_held:.0f} day(s).<br><br>'
                  f'A stop can be hit in one day. A 2R target usually cannot. So '
                  f'the FIRST calls to close are almost always the losers, while '
                  f'the winners are still open and uncounted. The win rate and '
                  f'median R below will look terrible for the first few weeks '
                  f"whatever the engine's real quality, and they only become "
                  f'meaningful once targets start being reached.<br><br>'
                  f'Read the <b>still-open</b> count and the MFE distribution '
                  f'instead until then.</div>')

    # ---- headline tiles ----
    def tile(label, val, sub="", col="#222"):
        return (f'<div class="tile"><div style="font-size:11px;color:#666">{label}'
                f'</div><div style="font-size:22px;font-weight:800;color:{col}">'
                f'{val}</div><div style="font-size:10px;color:#999">{sub}</div></div>')

    tiles = [tile("Open calls", f"{len(opens):,}", "being tracked"),
             tile("Closed", f"{n_closed:,}", "with an outcome")]
    if n_closed:
        r = pd.to_numeric(out.get("r_multiple"), errors="coerce").dropna()
        if len(r):
            wr = (r > 0).mean() * 100
            _sub = ("BIASED — losers close first"
                    if immature else f"of {len(r)} closed")
            _c = "#888" if immature else _rcol(wr - 50)
            tiles += [tile("Win rate", f"{wr:.0f}%", _sub, _c),
                      tile("Median R", f"{r.median():+.2f}",
                           "not yet meaningful" if immature else "per closed call",
                           "#888" if immature else _rcol(r.median()))]
        ev = pd.to_numeric(out.get("exit_vs_hold_r"), errors="coerce").dropna()
        if len(ev):
            tiles.append(tile("Exit vs hold", f"{ev.median():+.2f}R",
                              "positive = exiting helped", _rcol(ev.median())))

    # ---- per-engine ----
    scols = [c for c in ["family", "n_open", "n_closed", "win_rate_pct",
                         "median_r", "median_mfe_r", "median_mae_r",
                         "median_exit_vs_hold_r", "pct_target_hit", "pct_stopped"]
             if c in summ.columns]
    summary_html = _table(summ, scols) if scols else _table(summ, list(summ.columns))

    # ---- reliability ----
    rel_html = '<p style="color:#888">(needs closed calls)</p>'
    if immature:
        rel_html = ('<p style="color:#888">Held back: every closed call so far '
                    'is a stop, so a decile lift would only measure which stocks '
                    'fell fastest, not whether the score predicts anything. '
                    'Returns once targets start being reached.</p>')
    elif n_closed >= MIN_N:
        try:
            import importlib.util
            spec = importlib.util.spec_from_file_location(
                "st", os.path.join(_SCRIPTS_DIR, "signal_tracker.py"))
            st = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(st)
            rows = [st.reliability(out, c) for c in
                    ("conviction_at_signal", "n_families_at_signal",
                     "n_events_at_signal") if c in out.columns]
            rel = pd.DataFrame(rows)
            rel_html = _table(rel, [c for c in
                                    ["score", "n", "decile_lift", "rank_ic",
                                     "top_decile_median_r",
                                     "bottom_decile_median_r", "note"]
                                    if c in rel.columns])
        except Exception as e:
            rel_html = f'<p style="color:#c0392b">reliability failed: {e}</p>'
    else:
        rel_html = (f'<p style="color:#888">Held back until {MIN_N} calls have '
                    f'closed — currently {n_closed}. Reporting a decile lift on '
                    f'a smaller sample would be inventing a result.</p>')

    # ---- MFE ----
    mfe_html = '<p style="color:#888">(nothing yet)</p>'
    if "mfe_r" in out.columns and n_closed:
        m = pd.to_numeric(out["mfe_r"], errors="coerce").dropna()
        if len(m):
            q = m.describe(percentiles=[.25, .5, .6, .75, .9]).round(2)
            mfe_html = ("<table><tr><th>stat</th><th>R</th></tr>"
                        + "".join(f"<tr><td>{k}</td><td>{v:,.2f}</td></tr>"
                                  for k, v in q.items()) + "</table>"
                        + '<div class="note">A 2R target is reached by whatever '
                          'share of trades sits above 2.00 here. That is the '
                          'empirical basis for choosing it, rather than the '
                          'convention.</div>')

    ccols = [c for c in ["symbol", "family", "first_date", "status", "days_held",
                         "r_multiple", "mfe_r", "mae_r", "exit_vs_hold_r",
                         "entry_at_signal", "stop_at_signal"] if c in out.columns]
    closed_html = _table(out.sort_values("r_multiple", ascending=False)
                         if "r_multiple" in out.columns else out, ccols)

    ocols = [c for c in ["symbol", "family", "first_date", "times_seen",
                         "entry_at_signal", "stop_at_signal",
                         "conviction_at_signal", "n_families_at_signal"]
             if c in opens.columns]
    open_html = _table(opens.head(400), ocols)
    if len(opens) > 400:
        open_html += f'<div class="note">showing 400 of {len(opens):,}</div>'

    html = (_TPL.replace("__DATE__", str(datetime.now().date()))
                .replace("__BANNER__", banner)
                .replace("__TILES__", "".join(tiles))
                .replace("__SUMMARY__", summary_html)
                .replace("__RELIABILITY__", rel_html)
                .replace("__MFE__", mfe_html)
                .replace("__CLOSED__", closed_html)
                .replace("__OPEN__", open_html))

    if args.dry_run:
        log(f"DRY RUN — would write {args.out} ({len(html)/1024:.0f} KB)")
        return 0
    path = os.path.join(os.path.dirname(_SCRIPTS_DIR), args.out)
    io.open(path, "w", encoding="utf-8").write(html)
    log(f"wrote {path}  ({len(html)/1024:.0f} KB)")
    if not args.no_open:
        webbrowser.open("file:///" + path.replace("\\", "/"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
