r"""
pf_results_digest.py — daily earnings-season mail: PF companies that have reported
the CURRENT quarter, newest reporter first, with the 6-quarter revenue / profit /
EPS / margin table and YoY + QoQ.

The one hard rule: ONE quarter per season. A holding whose newest quarterly column
is still Q4FY26 is NOT reported under a Q1FY27 heading — it is listed as *awaiting*
until it actually files. Quarter labelling uses the Phase-1/2 results convention
(screener_scraper.current_season_key / build_gallery._qtr_label — the quarter that
just ENDED), NOT extract_concall._current_india_quarter (quarter in progress).

Cumulative, not incremental: every covered company keeps its full table in every
mail. Companies first covered today appear under "NEW TODAY", the rest under
"EARLIER THIS SEASON", both sorted newest-report-first.

Season lifecycle
  - season_quarter = current_season_key(), e.g. "Q1FY27" during Jul-Sep.
  - Mail goes out daily while any PF holding is still awaiting, plus one final mail
    on the day coverage completes. Then it goes quiet until the next season.
  - A season reset is automatic: when season_quarter flips, the ledger has no rows
    for the new key so coverage starts at zero.
  - PF membership is read LIVE every run, so adding a holding mid-season (that has
    already reported) re-opens coverage and the mail resumes.

Inputs (all already on Drive, no schema change):
  pf_tracking|portfolio holdings                -> PF ISINs (load_portfolio_isins)
  company_repo/_index/company_universe.csv      -> isin -> symbol / name
  fundamentals/statements/<SYM>.parquet         -> the quarterly numbers
  company_repo/_index/results.parquet           -> first_seen_at = declaration proxy
  company_repo/_index/processing_queue.parquet  -> results filing announcement_date

Output: company_repo/_index/pf_results_digest.parquet (coverage ledger) + the mail
(toggle 'pf_results').

Usage:
    python scripts/pf_results_digest.py --dry-run    # preview html only, no write/mail
    python scripts/pf_results_digest.py              # ledger + mail (season-gated)
    python scripts/pf_results_digest.py --force      # ignore the season-window gate
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

_SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

import pandas as pd
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(_SCRIPTS_DIR), ".env"))

from _extractor_base import (get_drive, get_or_create_subfolder, load_parquet,
                             save_parquet, log)
try:                                     # the queue's own announcement_date ->
    from backfill_company_docs import _fy_quarter_label   # quarter mapping; the
except Exception:                        # import chain pulls ingest_company_docs,
    _fy_quarter_label = None             # so degrade rather than kill the mail
from build_gallery import _bulk_parquet, _folder
from daily_brief import load_pf
from mailer import send_email, load_mail_settings, esc
from seasons import is_peak_season
import quarterly_table as QT

LEDGER_NAME = "pf_results_digest.parquet"
LEDGER_COLS = ["season_quarter", "isin", "symbol", "name", "quarter_label",
               "reported_on", "date_source", "first_covered_at", "last_mailed_at"]
MAIL_KEY = "pf_results"

# only the columns the date cascade needs (load_parquet slices to exactly these)
RESULTS_COLS = ["isin", "latest_q", "first_seen_at", "scraped_at"]
QUEUE_COLS = ["isin", "doc_type", "period", "announcement_date"]

# Gmail clips a message around 102 KB behind a "View entire message" link, which
# would cut the mail off mid-table. Past this the already-covered tail collapses
# to one line each (new reporters always keep their full table).
MAX_HTML_BYTES = 90_000

# date_source preference — a better source upgrades an already-stored date
SRC_RANK = {"screener": 0, "filing": 1, "detected": 2}
SRC_NOTE = {"screener": "", "filing": " (filing)", "detected": " (detected)"}


# ── report-date cascade ──────────────────────────────────────────────────────
def _results_dates(results: pd.DataFrame, season: str) -> dict[str, str]:
    """{isin: YYYY-MM-DD} from results.parquet first_seen_at — the declaration
    proxy strategy_pead.py and flag_growth_surge.py already rely on. Restricted to
    rows whose latest_q maps to THIS season, so a stale row can't date a company."""
    if results is None or results.empty or "latest_q" not in results.columns:
        return {}
    r = results.copy()
    r["_q"] = r["latest_q"].map(lambda v: QT.norm_q(QT.qtr_label(v)))
    r = r[r["_q"] == QT.norm_q(season)]
    if r.empty:
        return {}
    col = "first_seen_at" if "first_seen_at" in r.columns else "scraped_at"
    r["_d"] = r[col].astype(str).str.slice(0, 10)
    r = r[r["_d"].str.match(r"\d{4}-\d{2}-\d{2}", na=False)]
    if r.empty:
        return {}
    return r.groupby(r["isin"].astype(str).str.strip())["_d"].min().to_dict()


def _queue_dates(queue: pd.DataFrame, season: str) -> dict[str, str]:
    """{isin: YYYY-MM-DD} from the global processing queue — results filings for
    this period. Covers the ~150 reporters the 25-item Screener /results/latest/
    window misses each season."""
    if queue is None or queue.empty or "doc_type" not in queue.columns:
        return {}
    q = queue[queue["doc_type"].astype(str) == "results"].copy()
    if q.empty:
        return {}
    q["_d"] = q["announcement_date"].astype(str).str.slice(0, 10)
    # `period` is a T12 addition and is still blank on results rows — fall back to
    # the same announcement-date -> quarter mapping the queue itself uses.
    def _per(row):
        p = str(row.get("period") or "").strip()
        if p and p.lower() != "none":
            return QT.norm_q(p)
        return _fy_quarter_label(row["_d"]) if _fy_quarter_label else ""
    q = q[q.apply(_per, axis=1) == QT.norm_q(season)]
    if q.empty:
        return {}
    q = q[q["_d"].str.match(r"\d{4}-\d{2}-\d{2}", na=False)]
    if q.empty:
        return {}
    return q.groupby(q["isin"].astype(str).str.strip())["_d"].min().to_dict()


def resolve_date(isin, screener_d, queue_d, prev_row, today):
    """(date, source) — best available 'populated on' date. Falls back to the day
    this digest first saw the company flip, so every covered company has a date."""
    if isin in screener_d:
        return screener_d[isin], "screener"
    if isin in queue_d:
        return queue_d[isin], "filing"
    if prev_row is not None and str(prev_row.get("reported_on") or "").strip():
        return str(prev_row["reported_on"])[:10], str(prev_row.get("date_source") or "detected")
    return today, "detected"


# ── coverage ─────────────────────────────────────────────────────────────────
def build_coverage(pf, stmts_map, results, queue, ledger, season, today):
    """(covered_rows, awaiting_rows). `covered` = the company's newest quarterly_pl
    column IS the season quarter — the no-mixing guard."""
    prev = {}
    if ledger is not None and not ledger.empty:
        for _, r in ledger[ledger["season_quarter"].astype(str)
                           == season].iterrows():
            prev[str(r["isin"]).strip()] = r

    screener_d = _results_dates(results, season)
    queue_d = _queue_dates(queue, season)
    log(f"report dates available: screener={len(screener_d)} queue={len(queue_d)}")

    covered, awaiting = [], []
    for isin, sym, name in pf:
        stmts = stmts_map.get(sym)
        lbl = QT.latest_quarter_label(stmts)
        if lbl is None or QT.norm_q(lbl) != QT.norm_q(season):
            awaiting.append({"isin": isin, "symbol": sym, "name": name,
                             "quarter_label": lbl or "",
                             "reason": "no quarterly data" if lbl is None else f"still {lbl}"})
            continue
        p = prev.get(isin)
        rep, src = resolve_date(isin, screener_d, queue_d, p, today)
        # a better source upgrades a date stored earlier as "detected"
        if p is not None and str(p.get("date_source") or "") in SRC_RANK:
            if SRC_RANK[src] > SRC_RANK[str(p["date_source"])]:
                rep, src = str(p["reported_on"])[:10], str(p["date_source"])
        covered.append({
            "season_quarter": season, "isin": isin, "symbol": sym, "name": name,
            "quarter_label": lbl, "reported_on": rep, "date_source": src,
            "first_covered_at": (str(p["first_covered_at"])[:10]
                                 if p is not None and str(p.get("first_covered_at") or "").strip()
                                 else today),
            "last_mailed_at": (str(p.get("last_mailed_at") or "") if p is not None else ""),
            "stmts": stmts,
        })
    covered.sort(key=lambda r: (r["reported_on"], r["symbol"]), reverse=True)
    awaiting.sort(key=lambda r: r["symbol"])
    return covered, awaiting


# ── html ─────────────────────────────────────────────────────────────────────
_WRAP = "font-family:Arial,Helvetica,sans-serif;font-size:13px;color:#222"


def _fmt_day(d: str) -> str:
    try:
        return datetime.strptime(str(d)[:10], "%Y-%m-%d").strftime("%d %b")
    except Exception:
        return str(d)[:10]


def _company_block(row) -> str:
    table = QT.quarterly_table_html(row["stmts"])
    if not table:
        return ""
    sym, name = esc(row["symbol"], 24), esc(row["name"], 70)
    when = _fmt_day(row["reported_on"]) + SRC_NOTE.get(row["date_source"], "")
    return (
        "<div style='margin:0 0 14px 0'>"
        f"<div style='font-size:14px;font-weight:700;color:#111'>{name} "
        f"<span style='color:#888;font-weight:400'>· {sym}</span></div>"
        f"<div style='color:#888;font-size:11px;margin:0 0 3px'>"
        f"{esc(row['quarter_label'], 12)} · populated {esc(when, 24)}</div>"
        f"{table}</div>"
    )


def _compact_rows(rows) -> str:
    """One line per company — the fallback for already-covered names when the full
    -table mail would exceed Gmail's clip threshold. Same numbers, no history."""
    head = ("<tr style='color:#666'>"
            + "".join(f"<td style='{QT._HD};{a}'>{h}</td>" for h, a in
                      (("Company", "text-align:left"), ("Reported", ""),
                       ("Revenue", ""), ("Rev YoY", ""), ("Net Profit", ""),
                       ("PAT YoY", ""), ("NPM %", "")))
            + "</tr>")
    body = ""
    for r in rows:
        h = QT.headline(r["stmts"])
        if not h:
            continue
        def _v(d, dp=0):
            v = d.get("value")
            return "—" if v is None else format(v, f",.{dp}f")
        def _g(d):
            g = d.get("yoy")
            if g is None:
                return "<td style='color:#bbb'>—</td>"
            col = QT.UP if g >= 0 else QT.DOWN
            return f"<td style='font-weight:700;color:{col}'>{g:+.0f}%</td>"
        body += (f"<tr><td style='text-align:left'><b>{esc(r['symbol'], 24)}</b></td>"
                 f"<td>{_fmt_day(r['reported_on'])}</td>"
                 f"<td>{_v(h['revenue'])}</td>{_g(h['revenue'])}"
                 f"<td>{_v(h['pat'])}</td>{_g(h['pat'])}"
                 f"<td>{_v(h['npm'], 1)}</td></tr>")
    if not body:
        return ""
    return (f"<table cellpadding='4' cellspacing='0' style='{QT._TBL}'>{head}{body}</table>")


def build_html(covered, awaiting, season, today, pf_n, in_window,
               compact_old: bool = False) -> str:
    new = [r for r in covered if str(r["first_covered_at"])[:10] == today]
    old = [r for r in covered if str(r["first_covered_at"])[:10] != today]
    season_lbl = esc(f"{season[:2]} {season[2:]}", 12)   # Q1FY27 -> "Q1 FY27"
    done = len(covered) >= pf_n and pf_n > 0

    head = (f"<h2 style='margin:0 0 2px'>📊 PF results — {season_lbl}</h2>"
            f"<div style='color:#888;font-size:12px;margin:0 0 8px'>"
            f"{len(covered)} of {pf_n} holdings reported · as of {today}"
            + ("" if in_window else " · season window closed") + "</div>")
    if done:
        head += ("<div style='background:#e8f5e9;border-left:3px solid #1a7a3a;"
                 "padding:6px 10px;margin:0 0 10px;font-size:12px'>"
                 f"✅ All {pf_n} PF holdings have reported {season_lbl} — "
                 "this is the final mail for this season.</div>")

    def _sec(title, rows, colour, blocks):
        if not blocks:
            return ""
        return (f"<h3 style='margin:14px 0 6px;font-size:13px;color:{colour};"
                f"border-bottom:2px solid {colour};padding-bottom:2px'>"
                f"{title} ({len(rows)})</h3>{blocks}")

    # new reporters ALWAYS keep the full table; only the already-covered tail
    # collapses, and only when the full mail would be clipped by Gmail
    old_blocks = (_compact_rows(old) if compact_old
                  else "".join(_company_block(r) for r in old))
    body = (_sec("🆕 NEW TODAY", new, "#1a7a3a",
                 "".join(_company_block(r) for r in new))
            + _sec("EARLIER THIS SEASON", old, "#34495e", old_blocks))
    if compact_old and old:
        body += ("<div style='color:#999;font-size:11px;margin:2px 0 0'>"
                 "Earlier reporters shown as one line each — the full-table mail "
                 "would exceed Gmail's size limit and be clipped.</div>")

    foot = ""
    if awaiting:
        names = ", ".join(esc(r["symbol"] or r["name"], 24) for r in awaiting[:40])
        more = f" +{len(awaiting) - 40} more" if len(awaiting) > 40 else ""
        foot = ("<hr style='border:0;border-top:1px solid #ddd;margin:12px 0 6px'>"
                f"<div style='color:#666;font-size:12px'><b>Awaiting {season_lbl}</b> "
                f"({len(awaiting)} of {pf_n}): {names}{more}</div>")
        if not in_window:
            foot += ("<div style='color:#999;font-size:11px;margin-top:4px'>"
                     f"Season closed — {len(awaiting)} never reported.</div>")
    foot += ("<div style='color:#aaa;font-size:11px;margin-top:8px'>"
             "Revenue / Net Profit in ₹ Cr, EPS in ₹, margins in %. "
             "YoY = same quarter last year; QoQ = previous quarter; "
             "margin deltas in percentage points. Source: Screener quarterly P&amp;L."
             "</div>")
    return f"<div style='{_WRAP}'>{head}{body}{foot}</div>"


# ── main ─────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true",
                    help="preview html only — no Drive write, no mail")
    ap.add_argument("--force", action="store_true",
                    help="ignore the season-window gate and the already-complete stop")
    args = ap.parse_args()

    drive = get_drive()
    root_id = os.environ["GDRIVE_FOLDER_ID"]
    repo_id = get_or_create_subfolder(drive, root_id, "company_repo")
    index_id = get_or_create_subfolder(drive, repo_id, "_index")

    season = QT.season_quarter()
    today = datetime.now().strftime("%Y-%m-%d")
    in_window = is_peak_season(datetime.now())
    log(f"season quarter={season} · peak-season={in_window} · today={today}")

    if not in_window and not args.force and not args.dry_run:
        log("outside the results-season window — no mail. (--force to override)")
        return

    pf = load_pf(drive, root_id, index_id)
    if not pf:
        log("no PF holdings resolved — nothing to do.")
        return
    syms = sorted({s for _, s, _ in pf if s})
    log(f"PF holdings: {len(pf)} ({len(syms)} symbols)")

    stmts_map = _bulk_parquet(drive, _folder(drive, "fundamentals/statements"), syms)
    log(f"statements downloaded: {sum(1 for v in stmts_map.values() if v is not None and not v.empty)}/{len(syms)}")

    # load_parquet returns df[cols] — the column list is mandatory, never []
    results = load_parquet(drive, index_id, "results.parquet", RESULTS_COLS)
    queue = load_parquet(drive, index_id, "processing_queue.parquet", QUEUE_COLS)
    ledger = load_parquet(drive, index_id, LEDGER_NAME, LEDGER_COLS)

    covered, awaiting = build_coverage(pf, stmts_map, results, queue, ledger, season, today)
    new_today = [r for r in covered if str(r["first_covered_at"])[:10] == today]
    never_mailed = [r for r in covered if not str(r["last_mailed_at"] or "").strip()]
    log(f"covered={len(covered)} awaiting={len(awaiting)} "
        f"new_today={len(new_today)} never_mailed={len(never_mailed)}")

    if not covered:
        log(f"no PF holding has reported {season} yet — no mail.")
        return

    html = build_html(covered, awaiting, season, today, len(pf), in_window)
    if len(html.encode("utf-8")) > MAX_HTML_BYTES:
        html = build_html(covered, awaiting, season, today, len(pf), in_window,
                          compact_old=True)
        log(f"full-table mail exceeded {MAX_HTML_BYTES // 1024} KB — "
            f"earlier reporters collapsed to one line each "
            f"({len(html.encode('utf-8')) / 1024:.0f} KB)")

    if args.dry_run:
        show = pd.DataFrame([{k: r[k] for k in
                              ("symbol", "quarter_label", "reported_on", "date_source",
                               "first_covered_at")} for r in covered])
        print(show.to_string(index=False))
        if awaiting:
            print("\nAWAITING:")
            print(pd.DataFrame(awaiting)[["symbol", "quarter_label", "reason"]]
                  .to_string(index=False))
        prev = Path(__file__).resolve().parent.parent / "pf_results_digest_preview.html"
        prev.write_text(html, encoding="utf-8")
        print(f"\nDRY RUN — preview saved to {prev.name} "
              f"({len(html.encode('utf-8')) / 1024:.0f} KB); no Drive write, no mail.")
        return

    # stop rule: mail daily while incomplete; one final mail when coverage completes
    complete = len(covered) >= len(pf)
    if complete and not never_mailed and not args.force:
        log("season complete and already mailed — no mail. (--force to override)")
        return

    if not load_mail_settings(drive, index_id).get(MAIL_KEY, True):
        log(f"{MAIL_KEY} mail toggled OFF — skipped (ledger unchanged).")
        return

    season_lbl = f"{season[:2]} {season[2:]}"
    subject = (f"📊 PF results {season_lbl} — {len(covered)}/{len(pf)} reported"
               + (f" · {len(new_today)} new" if new_today else "")
               + (" · complete" if complete else ""))
    sent = send_email(subject, html)
    # ascii-only in log lines — local console is cp1252, emoji crash print()
    log(f"Email {'sent' if sent else 'FAILED'}: "
        f"{subject.encode('ascii', 'ignore').decode().strip()}")

    # The ledger advances ONLY on a delivered mail. Stamping last_mailed_at after a
    # failed send would mark these companies as already reported to the user and
    # demote them out of "NEW TODAY" on the next run — they must stay new until a
    # mail actually goes out.
    if not sent:
        log("mail not sent — ledger NOT advanced; these stay new for the next run.")
        return

    out = ledger[ledger["season_quarter"].astype(str) != season] if not ledger.empty \
        else pd.DataFrame(columns=LEDGER_COLS)
    rows = [{k: r[k] for k in LEDGER_COLS if k != "last_mailed_at"} for r in covered]
    for r in rows:
        r["last_mailed_at"] = datetime.now().isoformat(timespec="seconds")
    out = pd.concat([out, pd.DataFrame(rows, columns=LEDGER_COLS)], ignore_index=True)
    save_parquet(drive, index_id, LEDGER_NAME, out)
    log(f"wrote _index/{LEDGER_NAME} ({len(out)} rows, {len(rows)} this season)")


if __name__ == "__main__":
    main()
