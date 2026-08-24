r"""
run_guidance_progress_mail.py — WEEKLY "are they getting there?" mail.

Reads guidance_progress.parquet (built by build_guidance_progress.py) and answers,
per company: this is what management promised, this is how much has landed, and did
it move closer THIS WEEK.

Sections
  1  Moved this week      non-zero delta_week, biggest movement first  <- the point
  2  Order-book scoreboard guided inflow vs Rs booked to date
  3  Losing the thread     BEHIND / AT_RISK past the horizon's halfway mark
  4  Newly there           crossed 100% since the last snapshot
  5  Coverage              what is NOT measurable, and why

Every existing mail in this repo is daily; this is the first weekly one, so it runs
in its own workflow rather than the 19:00 pead -> t4_nightly chain.

Toggle: 'guidance_progress' in mail_settings.json (app sidebar / toggle_mail.bat).

Usage:
    python scripts/run_guidance_progress_mail.py --dry-run
    python scripts/run_guidance_progress_mail.py --pf-only
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime

_SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

import pandas as pd
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(_SCRIPTS_DIR), ".env"))

from _extractor_base import (get_drive, get_or_create_subfolder,  # noqa: E402
                             load_parquet, log)
from build_guidance_progress import HIST_COLS, HIST_NAME, OUT_NAME, PROGRESS_COLS  # noqa: E402
from mailer import esc, load_mail_settings, send_email  # noqa: E402

MAIL_KEY = "guidance_progress"
MAX_HTML_BYTES = 90_000          # Gmail clips near 102 KB
PREVIEW = "guidance_progress_preview.html"

# rows per section — enough to be useful, few enough to stay under the clip
CAP_MOVED, CAP_ORDERS, CAP_RISK, CAP_DONE = 30, 25, 25, 15

STATUS_CHIP = {
    "ACHIEVED": ("#1a7a3a", "#fff"),
    "AHEAD":    ("#d4edda", "#155724"),
    "ON_TRACK": ("#e2f0d9", "#2d5016"),
    "BEHIND":   ("#fff3cd", "#856404"),
    "AT_RISK":  ("#f8d7da", "#721c24"),
    "NO_DATA":  ("#eee", "#666"),
}
TD = ("padding:6px 10px;border:1px solid #ddd;font-size:13px;"
      "font-family:Arial,sans-serif;")
TH = TD + "background:#34495e;color:#fff;text-align:left;"
P = "font-family:Arial,sans-serif;font-size:13px"
MUTED = "font-size:11px;color:#999;font-family:Arial,sans-serif"


def _n(v):
    try:
        f = float(v)
        return None if pd.isna(f) else f
    except (TypeError, ValueError):
        return None


def _cr(v) -> str:
    f = _n(v)
    if f is None:
        return "—"
    if abs(f) >= 1000:
        return f"{f:,.0f}"
    return f"{f:,.1f}"


def _pct(v, signed: bool = False) -> str:
    f = _n(v)
    if f is None:
        return "—"
    return f"{'+' if signed and f >= 0 else ''}{f:,.1f}%"


def _guided(r) -> str:
    """The promise, in its own units."""
    v = _n(r.guided_value)
    if v is None:
        return esc(r.guided_text, 40)
    return f"₹{_cr(v)} cr" if r.guided_unit == "INR_cr" else f"{v:,.1f}%"


def _chip(status: str) -> str:
    bg, fg = STATUS_CHIP.get(status, ("#eee", "#333"))
    return (f"<span style='background:{bg};color:{fg};padding:2px 8px;"
            f"border-radius:9px;font-size:11px;white-space:nowrap'>{status}</span>")


def _co(r) -> str:
    sym = esc(r.symbol or r.isin, 14)
    link = f"https://www.screener.in/company/{sym}/" if r.symbol else "#"
    badge = "💼 " if r.in_pf else ""
    return (f"{badge}<a href='{link}' style='color:#1a237e'><b>{sym}</b></a>")


def _table(headers: list[str], body: list[str]) -> str:
    return ("<table style='border-collapse:collapse;width:100%'><tr>"
            + "".join(f"<th style='{TH}'>{h}</th>" for h in headers) + "</tr>"
            + "".join(body) + "</table>")


def _h(title: str, note: str = "") -> str:
    return (f"<h3 style='font-family:Arial,sans-serif;font-size:15px;margin:22px 0 6px'>"
            f"{title}</h3>"
            + (f"<p style='{MUTED};margin:0 0 8px'>{note}</p>" if note else ""))


# ------------------------------------------------------------------ #
#  sections                                                            #
# ------------------------------------------------------------------ #

def sec_moved(df: pd.DataFrame) -> str:
    d = df[df["delta_week"].notna() & (df["delta_week"].abs() >= 0.1)].copy()
    if d.empty:
        return (_h("1 · Moved this week")
                + f"<p style='{P}'>No commitment changed measurably this week. "
                  f"This is expected in the weeks between results — the financial "
                  f"feeds only move when companies report, while order wins can "
                  f"land any day.</p>")
    d = d.reindex(d["delta_week"].abs().sort_values(ascending=False).index)
    body = []
    for r in d.head(CAP_MOVED).itertuples(index=False):
        dv = _n(r.delta_week) or 0
        col = "#1a7a3a" if dv > 0 else "#c0392b"
        body.append(
            "<tr>"
            f"<td style='{TD}'>{_co(r)}</td>"
            f"<td style='{TD}'>{esc(r.metric, 12)}</td>"
            f"<td style='{TD}'>{esc(r.target_period, 10)}</td>"
            f"<td style='{TD}'>{_guided(r)}</td>"
            f"<td style='{TD}'>{_cr(r.actual_to_date)}</td>"
            f"<td style='{TD}'>{_pct(r.pct_of_target)}</td>"
            f"<td style='{TD}color:{col}'><b>{_pct(dv, signed=True)}</b></td>"
            f"<td style='{TD}'>{_chip(r.status)}</td>"
            "</tr>")
    return (_h("1 · Moved this week",
               "Change in how far along each promise is, versus last Monday's "
               "snapshot. This is the only section that needs two weeks of history "
               "to populate.")
            + _table(["Company", "Metric", "Horizon", "Guided", "So far",
                      "% of target", "Δ week", "Status"], body))


def sec_orders(df: pd.DataFrame) -> str:
    d = df[df["metric"] == "order_book"].copy()
    d["_b"] = pd.to_numeric(d["actual_to_date"], errors="coerce").fillna(-1)
    d = d[(d["_b"] >= 0) | d["pct_of_target"].notna()]
    if d.empty:
        return ""
    d = d.sort_values(["_b", "guided_value"], ascending=False)
    body = []
    for r in d.head(CAP_ORDERS).itertuples(index=False):
        incomplete = "INCOMPLETE" in str(r.actual_source)
        note = (" <span style='color:#999;font-size:11px'>(partial window)</span>"
                if incomplete else "")
        body.append(
            "<tr>"
            f"<td style='{TD}'>{_co(r)}</td>"
            f"<td style='{TD}'>{esc(r.target_period, 10)}</td>"
            f"<td style='{TD}'><b>₹{_cr(r.guided_value)} cr</b></td>"
            f"<td style='{TD}'>₹{_cr(r.actual_to_date)} cr{note}</td>"
            f"<td style='{TD}'>{_pct(r.pct_of_target)}</td>"
            f"<td style='{TD}'>{_pct(r.time_pct)}</td>"
            f"<td style='{TD}'>{_chip(r.status)}</td>"
            "</tr>")
    n_part = int(d["actual_source"].astype(str).str.contains("INCOMPLETE").sum())
    note = ("Guided order inflow against rupees actually booked, summed from BSE "
            "order-win filings.")
    if n_part:
        note += (f" {n_part} row(s) marked <i>partial window</i>: the announcement "
                 f"ledger starts after that horizon opened, so the booked figure is "
                 f"a floor and no pace verdict is given.")
    return _h("2 · Order-book scoreboard", note) + _table(
        ["Company", "Horizon", "Guided inflow", "Booked so far", "% of target",
         "% elapsed", "Status"], body)


def sec_risk(df: pd.DataFrame) -> str:
    d = df[df["status"].isin(["BEHIND", "AT_RISK"])
           & (pd.to_numeric(df["time_pct"], errors="coerce") >= 50)]
    if d.empty:
        return ""
    d = d.sort_values(["status", "pct_of_target"])
    body = []
    for r in d.head(CAP_RISK).itertuples(index=False):
        body.append(
            "<tr>"
            f"<td style='{TD}'>{_co(r)}</td>"
            f"<td style='{TD}'>{esc(r.metric, 12)}</td>"
            f"<td style='{TD}'>{esc(r.target_period, 10)}</td>"
            f"<td style='{TD}'>{_guided(r)}</td>"
            f"<td style='{TD}'>{_cr(r.actual_to_date)}</td>"
            f"<td style='{TD}'>{_pct(r.pct_of_target)}</td>"
            f"<td style='{TD}'>{r.periods_elapsed}/{r.periods_total}</td>"
            f"<td style='{TD}'>{_chip(r.status)}</td>"
            "</tr>")
    return _h("3 · Losing the thread",
              "At least halfway through the horizon and not keeping up. Quarters "
              "elapsed is shown so a company that simply has not reported yet is "
              "visible as such — Screener lags the filing by weeks.") + _table(
        ["Company", "Metric", "Horizon", "Guided", "So far", "% of target",
         "Qtrs", "Status"], body)


def sec_done(df: pd.DataFrame) -> str:
    d = df[df["status"] == "ACHIEVED"]
    if d.empty:
        return ""
    d = d.sort_values("pct_of_target", ascending=False)
    body = []
    for r in d.head(CAP_DONE).itertuples(index=False):
        body.append(
            "<tr>"
            f"<td style='{TD}'>{_co(r)}</td>"
            f"<td style='{TD}'>{esc(r.metric, 12)}</td>"
            f"<td style='{TD}'>{esc(r.target_period, 10)}</td>"
            f"<td style='{TD}'>{_guided(r)}</td>"
            f"<td style='{TD}'>{_cr(r.actual_to_date)}</td>"
            f"<td style='{TD}'><b>{_pct(r.pct_of_target)}</b></td>"
            "</tr>")
    return _h("4 · Already there",
              "Guided number met or beaten with the horizon still open.") + _table(
        ["Company", "Metric", "Horizon", "Guided", "Delivered", "% of target"], body)


def sec_coverage(df: pd.DataFrame) -> str:
    """What is NOT measurable, and why — so the gaps are visible, not silent."""
    nd = df[df["status"] == "NO_DATA"]
    if nd.empty:
        return ""
    by_metric = nd["metric"].value_counts().head(8)
    lines = " · ".join(f"{esc(m, 16)} {int(n)}" for m, n in by_metric.items())
    return (_h("5 · Not measurable",
               "Commitments being tracked but with no actuals feed to score them "
               "against. Shown so the gaps stay visible instead of silently "
               "shrinking the universe.")
            + f"<p style='{P}'><b>{len(nd):,}</b> of {len(df):,} commitments "
              f"({len(nd) / len(df) * 100:.0f}%) cannot be scored.<br>"
              f"By metric: {lines}.</p>"
              f"<p style='{MUTED}'>Volume, capacity and working-capital targets have "
              f"no actuals source anywhere in the repo. Capex is annual-only "
              f"(Screener publishes the balance sheet once a year), so it cannot be "
              f"scored mid-year. The rest are prose or unit targets with no "
              f"comparable number.</p>")


def build_html(df: pd.DataFrame, week: str, hist_weeks: int) -> str:
    meas = df[df["status"] != "NO_DATA"]
    n_pf = int(df["in_pf"].sum())
    head = (
        f"<div style='max-width:760px'>"
        f"<p style='{P}'>Tracking <b>{len(df):,} open commitments</b> across "
        f"<b>{df['isin'].nunique()} companies</b> — <b>{len(meas):,}</b> with a "
        f"live actuals feed"
        + (f", 💼 {n_pf} in the portfolio" if n_pf else "") + f".<br>"
        f"Week of <b>{week}</b>. Progress is measured while the horizon is still "
        f"open: how much has landed versus how much of the period has gone.</p>")
    if hist_weeks < 2:
        head += (f"<p style='{MUTED}'>This is snapshot week {hist_weeks} — the "
                 f"“moved this week” column needs a prior Monday to compare "
                 f"against and will populate from next week.</p>")
    parts = [head, sec_moved(df), sec_orders(df), sec_risk(df), sec_done(df),
             sec_coverage(df)]
    foot = (f"<p style='{MUTED}'>A rate target (growth %, margin %) is judged "
            f"directly against the guided rate; a cumulative target (₹ revenue, "
            f"₹ order inflow) is judged against elapsed time. Sources: "
            f"guidance_tracker + mgmt_credibility (promises), financials_3stmt "
            f"(reported numbers), announcement_ledger (order wins). "
            f"Toggle this mail in the app sidebar (📧 Email toggles).</p></div>")

    # Budgeted packing: drop whole sections from the bottom before the Gmail clip,
    # never a hard mid-table cut (pf_results_digest pattern).
    html = "\n".join([p for p in parts if p]) + foot
    while len(html.encode("utf-8")) > MAX_HTML_BYTES and len(parts) > 2:
        parts.pop()
        html = ("\n".join([p for p in parts if p])
                + f"<p style='{MUTED}'>Some sections were dropped to stay under "
                  f"Gmail's size limit.</p>" + foot)
    return html


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true",
                    help=f"build + write {PREVIEW}, send nothing")
    ap.add_argument("--pf-only", action="store_true",
                    help="restrict to portfolio holdings")
    args = ap.parse_args()

    drive = get_drive()
    root = os.environ["GDRIVE_FOLDER_ID"]
    idx = get_or_create_subfolder(
        drive, get_or_create_subfolder(drive, root, "company_repo"), "_index")

    df = load_parquet(drive, idx, OUT_NAME, PROGRESS_COLS)
    if df.empty:
        log(f"{OUT_NAME} is empty — run build_guidance_progress.py first. No mail.")
        return 0
    # A parquet round trip leaves these object-dtype with None in them, so .abs()
    # and comparisons raise. Coerce once here rather than at every use site.
    for c in ("guided_value", "actual_to_date", "pct_of_target", "time_pct",
              "pace_ratio", "delta_week", "periods_elapsed", "periods_total"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["in_pf"] = df["in_pf"].fillna(False).astype(bool)
    hist = load_parquet(drive, idx, HIST_NAME, HIST_COLS)
    hist_weeks = hist["week_start"].nunique() if not hist.empty else 0

    if args.pf_only:
        df = df[df["in_pf"]]
        log(f"--pf-only -> {len(df)} rows")
        if df.empty:
            log("nothing in the portfolio has an open commitment. No mail.")
            return 0

    week = str(df["as_of"].max())[:10]
    html = build_html(df, week, hist_weeks)
    kb = len(html.encode("utf-8")) / 1024
    meas = int((df["status"] != "NO_DATA").sum())
    moved = int(df["delta_week"].notna().sum())
    log(f"rows={len(df)} companies={df['isin'].nunique()} measurable={meas} "
        f"moved={moved} history_weeks={hist_weeks} html={kb:.1f} KB")

    subject = (f"📊 Guidance progress — {meas:,} tracked commitments, "
               f"{df['isin'].nunique()} companies")
    if moved:
        subject += f" · {moved} moved"

    if args.dry_run:
        out = os.path.join(os.path.dirname(_SCRIPTS_DIR), PREVIEW)
        with open(out, "w", encoding="utf-8") as fh:
            fh.write(html)
        log(f"DRY RUN - preview written to {out}; no email sent.")
        return 0

    if not load_mail_settings(drive, idx).get(MAIL_KEY, True):
        log(f"{MAIL_KEY} mail toggled OFF - skipped.")
        return 0
    sent = send_email(subject, html)
    # ascii-only in log lines - the local console is cp1252 and emoji crash print()
    log(f"Email {'sent' if sent else 'FAILED'}: "
        f"{subject.encode('ascii', 'ignore').decode().strip()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
