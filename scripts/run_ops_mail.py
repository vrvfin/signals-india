r"""
run_ops_mail.py — morning ops digest (08:30 IST, NO Gemini).

One mail answering: did everything run last night, is the data fresh, what did
each build produce, and did the user flag anything for review?

  1. WORKFLOW RUNS  — GitHub Actions runs in the last ~26h (needs GITHUB_TOKEN,
     auto-present in CI; section skipped gracefully when absent locally).
  2. DATA FRESHNESS — key parquets: exists / rows / latest timestamp
     (✅ <30h · ⚠️ <54h · ❌ stale or missing).
  3. SAMPLES        — top scorecard names, fraud-tracker movers w/ reason,
     latest catalyst notes, today's PEAD verdicts, market health.
  4. REVIEW FLAGS   — open rows from _index/review_flags.csv (the app sidebar
     "🚩 Flag for review" box writes here; Claude reads it each session).

Toggle: 'ops_digest' in mail_settings.json. Workflow: ops_mail.yml (03:00 UTC).

Usage:
    python scripts/run_ops_mail.py --dry-run
"""
from __future__ import annotations

import argparse
import html as html_mod
import io
import os
import sys
from datetime import date, datetime, timedelta

_SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(_SCRIPTS_DIR), ".env"))

from _extractor_base import (get_drive, get_or_create_subfolder, find_file,
                             download_bytes, log)
from mailer import send_email, load_mail_settings

# Timestamps are stored in UTC (CI runners); the ops mail reports in IST for the reader.
_IST = timedelta(hours=5, minutes=30)

# (label, path_parts, timestamp column candidates)
FRESHNESS = [
    ("results (Screener scrape)",  ["company_repo", "_index", "results.parquet"],            ["scraped_at"]),
    ("pead_flags",                 ["company_repo", "_index", "pead_flags.parquet"],         ["as_of"]),
    ("guidance_tracker (Table_A)", ["company_repo", "_index", "guidance_tracker.parquet"],   ["processed_at"]),
    ("gf1_guidance_statements",    ["company_repo", "_index", "gf1_guidance_statements.parquet"], ["processed_at"]),
    ("mgmt_credibility (T1)",      ["company_repo", "_index", "mgmt_credibility.parquet"],   ["processed_at"]),
    ("financials_3stmt (T2)",      ["company_repo", "_index", "financials_3stmt.parquet"],   ["scraped_at"]),
    ("valuation (T4.1)",           ["company_repo", "_index", "valuation.parquet"],          ["computed_at"]),
    ("fraud_risk (T4.2)",          ["company_repo", "_index", "fraud_risk.parquet"],         ["computed_at"]),
    ("investigative_fraud (T4.4)", ["company_repo", "_index", "investigative_fraud.parquet"],["checked_at"]),
    ("company_scorecard (T4)",     ["company_repo", "_index", "company_scorecard.parquet"],  ["computed_at"]),
    ("catalyst_index (T5)",        ["company_repo", "_index", "catalyst_index.parquet"],     ["computed_at"]),
    ("fraud_tracker (T7)",         ["company_repo", "_index", "fraud_tracker.parquet"],      ["computed_at"]),
]


from mailer import esc as _esc_base


def _esc(s, n=140) -> str:
    return _esc_base(s, n)


def _read(drive, root, path_parts):
    fid = root
    for p in path_parts[:-1]:
        fid = get_or_create_subfolder(drive, fid, p)
    f = find_file(drive, fid, path_parts[-1])
    if not f:
        return None
    raw = download_bytes(drive, f)
    name = path_parts[-1]
    try:
        if name.endswith(".csv"):
            return pd.read_csv(io.BytesIO(raw))
        return pd.read_parquet(io.BytesIO(raw))
    except Exception:
        return None


# ---------------- section 1: workflow runs ----------------

def workflow_runs_html() -> tuple[str, int, int]:
    """(html, n_ok, n_failed) for runs created in the last ~26h."""
    token = os.environ.get("GITHUB_TOKEN", "")
    repo = os.environ.get("GH_REPO", "")
    if not token or not repo:
        return "<p><i>Workflow status: GITHUB_TOKEN/GH_REPO not set (local run).</i></p>", 0, 0
    since = (datetime.utcnow() - timedelta(hours=26)).strftime("%Y-%m-%dT%H:%M")
    try:
        r = requests.get(
            f"https://api.github.com/repos/{repo}/actions/runs",
            params={"per_page": 50, "created": f">={since}"},
            headers={"Authorization": f"Bearer {token}",
                     "Accept": "application/vnd.github+json"}, timeout=30)
        runs = r.json().get("workflow_runs", [])
    except Exception as e:
        return f"<p><i>Workflow status fetch failed: {_esc(e, 80)}</i></p>", 0, 0
    ok = failed = 0
    rows = []
    for x in runs:
        concl = str(x.get("conclusion") or x.get("status") or "?")
        good = concl == "success"
        ok += good
        failed += concl in ("failure", "timed_out", "cancelled")
        icon = "✅" if good else ("🏃" if concl in ("in_progress", "queued") else "❌")
        rows.append(f"<tr><td>{icon} {_esc(x.get('name'), 30)}</td>"
                    f"<td>{_esc(concl, 12)}</td>"
                    f"<td>{_esc(str(x.get('run_started_at', ''))[:16], 16)}</td>"
                    f"<td><a href='{_esc(x.get('html_url'), 120)}'>log</a></td></tr>")
    if not rows:
        return "<p><i>No workflow runs in the last 26h.</i></p>", 0, 0
    return ("<table border=1 cellpadding=4 cellspacing=0>"
            "<tr><th>Workflow</th><th>Result</th><th>Started (UTC)</th><th></th></tr>"
            + "".join(rows) + "</table>", ok, failed)


# ---------------- section 2: freshness ----------------

def freshness_html(drive, root) -> tuple[str, int]:
    rows, n_bad = [], 0
    now = datetime.now()
    for label, parts, ts_cols in FRESHNESS:
        df = _read(drive, root, parts)
        if df is None or df.empty:
            rows.append(f"<tr><td>❌ {_esc(label, 40)}</td><td>missing</td><td>-</td></tr>")
            n_bad += 1
            continue
        latest = None
        for c in ts_cols:
            if c in df.columns:
                latest = pd.to_datetime(df[c], errors="coerce").max()
                break
        if latest is None or pd.isna(latest):
            icon, age = "⚠️", "no timestamp"
        else:
            h = (now - latest.to_pydatetime().replace(tzinfo=None)).total_seconds() / 3600
            icon = "✅" if h < 30 else ("⚠️" if h < 54 else "❌")
            n_bad += icon == "❌"
            age = f"{h:.0f}h ago"
        rows.append(f"<tr><td>{icon} {_esc(label, 40)}</td>"
                    f"<td align=right>{len(df):,}</td><td>{age}</td></tr>")
    return ("<table border=1 cellpadding=4 cellspacing=0>"
            "<tr><th>Dataset</th><th>Rows</th><th>Latest</th></tr>"
            + "".join(rows) + "</table>", n_bad)


# ---------------- section 2b: coverage ----------------

# Buckets are measured relative to the LATEST available market bar (not calendar
# today) — so the morning before a run, or over a weekend, does not show everything
# as "stale". "current" = as fresh as the freshest session we have.
_BUCKET_ORDER = ["current", "1 back", "<=5 back", "<=1mo", ">1mo", "no data"]


def _freshness_table(df: pd.DataFrame, group_col: str, group_label: str) -> str:
    """Crosstab of last-bar-age buckets per group (exchange / segment)."""
    piv = pd.crosstab(df[group_col].fillna("(none)"), df["bucket"])
    for b in _BUCKET_ORDER:
        if b not in piv.columns:
            piv[b] = 0
    piv = piv[_BUCKET_ORDER]
    piv["total"] = piv.sum(axis=1)
    head = ("<tr><th>" + group_label + "</th>"
            + "".join(f"<th>{b}</th>" for b in _BUCKET_ORDER)
            + "<th>total</th></tr>")
    body = []
    for g, r in piv.sort_values("total", ascending=False).iterrows():
        tds = "".join(f"<td align=right>{int(r[b]):,}</td>" for b in _BUCKET_ORDER)
        body.append(f"<tr><td>{_esc(g, 16)}</td>{tds}"
                    f"<td align=right><b>{int(r['total']):,}</b></td></tr>")
    return ("<table border=1 cellpadding=4 cellspacing=0>"
            + head + "".join(body) + "</table>")


def coverage_html(drive, root) -> tuple[str, int]:
    """Stock coverage + freshness, bifurcated by exchange and market-cap segment.
    Per-company 'last data date' = features/latest.parquet date (its latest bar);
    names with no features row = 'no data'. Returns (html, n_bad)."""
    ml = _read(drive, root, ["universe", "master_list.csv"])
    feat = _read(drive, root, ["features", "latest.parquet"])
    sc = _read(drive, root, ["company_repo", "_index", "company_scorecard.parquet"])
    total = len(ml) if ml is not None else 0
    if feat is None or feat.empty or "date" not in feat.columns or total == 0:
        return ("<p>❌ Coverage: master_list or features/latest.parquet "
                "missing — cannot compute.</p>", 1)

    ml = ml[["symbol", "exchange"]].copy()
    ml["symbol"] = ml["symbol"].astype(str).str.upper()
    f = feat[["symbol", "date"]].copy()
    f["symbol"] = f["symbol"].astype(str).str.upper()
    df = ml.merge(f, on="symbol", how="left")
    if sc is not None and not sc.empty and "mcap_segment" in sc.columns:
        s = sc[["symbol", "mcap_segment"]].copy()
        s["symbol"] = s["symbol"].astype(str).str.upper()
        df = df.merge(s, on="symbol", how="left")
    else:
        df["mcap_segment"] = None

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    today = pd.Timestamp(datetime.now().date())
    latest_bar = df["date"].max()
    # Age is measured against the LATEST AVAILABLE bar, not calendar today, so the
    # morning before the daily run (or a weekend) doesn't read as 0% / all-stale.
    age = (latest_bar - df["date"]).dt.days

    def _bucket(a, has):
        if not has:
            return "no data"
        if a <= 0:
            return "current"
        if a <= 1:
            return "1 back"
        if a <= 5:
            return "<=5 back"
        if a <= 30:
            return "<=1mo"
        return ">1mo"

    df["bucket"] = [_bucket(a, pd.notna(d)) for a, d in zip(age, df["date"])]
    df["seg"] = df["mcap_segment"].where(
        df["mcap_segment"].isin(["Largecap", "Midcap", "Smallcap", "Microcap"]),
        "Unknown/BSE")

    featured = int((df["bucket"] != "no data").sum())
    current = int((df["date"] == latest_bar).sum())   # current to the latest session
    missing = int((df["bucket"] == "no data").sum())

    # Is the data itself stale? (pipeline stopped producing fresh bars). Weekends
    # mean the latest session can be 2-3 calendar days old, so allow up to 5.
    latest_bar_age = (today - latest_bar).days if pd.notna(latest_bar) else 999

    def pct(x):
        return f"{(100.0 * x / total):.1f}%" if total else "n/a"

    # Hard red only on a GENUINE break: the freshest bar we have is itself stale
    # (daily run not producing), or coverage collapsed below half the universe.
    # NOT on calendar-today==0, which is normal every morning pre-run.
    n_bad = 1 if (latest_bar_age > 5 or featured < 0.5 * total) else 0
    fresh_icon = "✅" if latest_bar_age <= 5 else "❌"

    summary = (
        "<table border=1 cellpadding=4 cellspacing=0>"
        "<tr><th>Metric</th><th>Count</th><th>% of total</th></tr>"
        f"<tr><td>Total stocks (master_list)</td><td align=right>{total:,}</td><td>100%</td></tr>"
        f"<tr><td>With data (featured)</td><td align=right>{featured:,}</td><td>{pct(featured)}</td></tr>"
        f"<tr><td>{fresh_icon} Latest bar {str(latest_bar)[:10]} "
        f"({latest_bar_age}d old)</td><td colspan=2>data freshness</td></tr>"
        f"<tr><td>Current to that bar</td>"
        f"<td align=right>{current:,}</td><td>{pct(current)}</td></tr>"
        f"<tr><td>Missing (no data)</td><td align=right>{missing:,}</td><td>{pct(missing)}</td></tr>"
        "</table>")

    html = (summary
            + "<p style='margin:6px 0 2px'><b>Last-data freshness by exchange:</b></p>"
            + _freshness_table(df, "exchange", "exchange")
            + "<p style='margin:6px 0 2px'><b>by market-cap segment:</b></p>"
            + _freshness_table(df, "seg", "segment")
            + "<p style='font-size:11px;color:#999'>Per-company 'last data' = latest "
              "bar in features/latest. BSE/.BO names typically lag a few days "
              "(Yahoo feed delay) — expected, not a failure.</p>")
    return html, n_bad


# ---------------- section 2c: market-cap freshness + coverage ----------------

def mcap_health_html(drive, root) -> tuple[str, int]:
    """Market-cap source health: is the data fresh and how complete is
    market_cap_cr. Both enrich steps in fundamentals.yml are continue-on-error,
    so a missed weekly run or a Screener-cookie expiry mid-run can leave mcap
    stale while CI still shows green — this catches that. Returns (html, n_bad)."""
    rows, n_bad = [], 0
    now = datetime.now()

    def _row(label, df, ts_col):
        nonlocal n_bad
        if df is None or df.empty or "market_cap_cr" not in df.columns:
            n_bad += 1
            return (f"<tr><td>❌ {_esc(label, 44)}</td><td>missing</td>"
                    f"<td>-</td><td>-</td></tr>")
        total = len(df)
        mc = pd.to_numeric(df["market_cap_cr"], errors="coerce")
        have = int(mc.notna().sum())
        nan_pct = 100 * (total - have) / total if total else 100
        # Freshness — weekly job, so measured in DAYS
        if ts_col is None:
            fresh_icon, age_txt = "", "—"
        elif ts_col in df.columns and pd.notna(
                pd.to_datetime(df[ts_col], errors="coerce").max()):
            latest = pd.to_datetime(df[ts_col], errors="coerce").max()
            days = (now - latest.to_pydatetime().replace(tzinfo=None)).total_seconds() / 86400
            fresh_icon = "✅" if days < 9 else ("⚠️" if days < 16 else "❌")
            age_txt = f"{days:.0f}d ago"
            n_bad += fresh_icon == "❌"
        else:
            fresh_icon, age_txt = "⚠️", "no timestamp"
        # Coverage — NaN-rate of market_cap_cr
        cov_icon = "✅" if nan_pct < 10 else ("⚠️" if nan_pct < 25 else "❌")
        n_bad += cov_icon == "❌"
        return (f"<tr><td>{_esc(label, 44)}</td>"
                f"<td>{fresh_icon} {age_txt}</td>"
                f"<td align=right>{have:,}/{total:,}</td>"
                f"<td>{cov_icon} {nan_pct:.0f}% NaN</td></tr>")

    rows.append(_row("Screener (summary.parquet, NSE+BSE)",
                     _read(drive, root, ["fundamentals", "summary.parquet"]), "fetched_at"))
    rows.append(_row("yfinance (market_cap.csv, NSE)",
                     _read(drive, root, ["universe", "market_cap.csv"]), None))

    html = ("<table border=1 cellpadding=4 cellspacing=0>"
            "<tr><th>Market-cap source</th><th>Freshness</th>"
            "<th>Have/Total</th><th>Coverage</th></tr>"
            + "".join(rows) + "</table>"
            "<p style='font-size:11px;color:#999'>Refreshed weekly by "
            "fundamentals.yml (Mon 06:30 IST). Stale (❌ &gt;16d) = a weekly run "
            "was missed or the Screener cookie expired mid-run — both enrich steps "
            "are continue-on-error, so CI can show green while mcap silently ages.</p>")
    return html, n_bad


# ---------------- section 3: samples ----------------

def _tbl(df: pd.DataFrame, cols: list[str], fmt: dict | None = None) -> str:
    cols = [c for c in cols if c in df.columns]
    head = "".join(f"<th>{_esc(c, 20)}</th>" for c in cols)
    body = []
    for _, r in df.iterrows():
        tds = []
        for c in cols:
            v = r[c]
            if fmt and c in fmt and pd.notna(v):
                v = fmt[c].format(v)
            tds.append(f"<td>{_esc(v, 90)}</td>")
        body.append("<tr>" + "".join(tds) + "</tr>")
    return (f"<table border=1 cellpadding=4 cellspacing=0><tr>{head}</tr>"
            + "".join(body) + "</table>")


def samples_html(drive, root) -> str:
    parts = []
    sc = _read(drive, root, ["company_repo", "_index", "company_scorecard.parquet"])
    if sc is not None and not sc.empty and "composite_score" in sc.columns:
        top = (sc[pd.to_numeric(sc["data_completeness_pct"], errors="coerce") >= 50]
               .sort_values("composite_score", ascending=False).head(5))
        parts.append("<p><b>🏆 Scorecard — top 5 (completeness ≥50%):</b></p>"
                     + _tbl(top, ["symbol", "composite_score",
                                  "data_completeness_pct", "score_technical",
                                  "score_fundamental", "score_investigative"],
                            {"composite_score": "{:.0f}"}))
    ft = _read(drive, root, ["company_repo", "_index", "fraud_tracker.parquet"])
    if ft is not None and not ft.empty:
        parts.append("<p><b>🕵️ Fraud tracker — top 5 by score:</b></p>"
                     + _tbl(ft.head(5), ["symbol", "fraud_score", "band",
                                         "trend", "reason"],
                            {"fraud_score": "{:.0f}"}))
    cat = _read(drive, root, ["company_repo", "_index", "catalyst_index.parquet"])
    if cat is not None and not cat.empty:
        latest = cat.sort_values("as_of", ascending=False).head(3)
        parts.append("<p><b>💡 Latest catalyst notes:</b></p>"
                     + _tbl(latest, ["symbol", "as_of", "catalyst_type", "headline"]))
    pead = _read(drive, root, ["company_repo", "_index", "pead_flags.parquet"])
    if pead is not None and not pead.empty and "as_of" in pead.columns:
        today_f = pead[pead["as_of"].astype(str) >= (date.today() - timedelta(days=1)).isoformat()]
        if not today_f.empty:
            parts.append("<p><b>📊 Fresh PEAD verdicts:</b></p>"
                         + _tbl(today_f.head(5), ["symbol", "quarter", "metric",
                                                  "guided_value", "actual_value",
                                                  "verdict"]))
    ms = _read(drive, root, ["data", "market_state", "latest.parquet"])
    if ms is not None and not ms.empty:
        r = ms.iloc[0]
        parts.append(f"<p><b>📈 Market:</b> health "
                     f"{_esc(r.get('health_score'), 6)}/100 · "
                     f"regime {_esc(r.get('regime'), 10)} ({_esc(r.get('date'), 10)})</p>")
    return "".join(parts) or "<p><i>No sample data available yet.</i></p>"


# ---------------- section 4: review flags ----------------

def flags_html(drive, root) -> str:
    fl = _read(drive, root, ["company_repo", "_index", "review_flags.csv"])
    if fl is None or fl.empty:
        return "<p>🚩 Review flags: none open.</p>"
    open_f = fl[fl.get("status", pd.Series(dtype=str)).astype(str).str.lower()
                != "done"] if "status" in fl.columns else fl
    if open_f.empty:
        return "<p>🚩 Review flags: none open.</p>"
    rows = "".join(f"<tr><td>{_esc(r.get('ts', ''), 16)}</td>"
                   f"<td>{_esc(r.get('flag', ''), 200)}</td></tr>"
                   for _, r in open_f.tail(10).iterrows())
    return (f"<p><b>🚩 Open review flags ({len(open_f)}):</b> Claude reads these "
            f"each session.</p><table border=1 cellpadding=4 cellspacing=0>"
            f"<tr><th>When</th><th>Flag</th></tr>{rows}</table>")


# ---------------- section 2d: document processing (backfill + live) ----------

def docs_processed_html(drive, root) -> tuple[str, int]:
    """How many documents the pipeline has processed — the ONE global queue
    (processing_queue.parquet) broken down by doc_type x status, plus how many
    were summarised in the last 24h, and the Stage-0 page-check coverage
    (companies whose Screener page we have actually swept per doc_type).
    Returns (html, done_last_24h)."""
    q = _read(drive, root, ["company_repo", "_index", "processing_queue.parquet"])
    if q is None or q.empty:
        return "<p><i>processing_queue.parquet missing — nothing processed yet.</i></p>", 0
    for c in ("status", "doc_type", "processed_at", "source"):
        if c not in q.columns:
            q[c] = ""
    pc = _read(drive, root, ["company_repo", "_index", "backfill_pagecheck.parquet"])

    # processed_at is stored in UTC (extractors run on UTC CI). Convert to IST for the
    # report (user reads in IST); the 24h count is identical, just IST-framed.
    now = pd.Timestamp(datetime.utcnow()) + _IST
    cutoff = now - pd.Timedelta(hours=24)
    # 7-day ROLLING error window (user 2026-07-09): the old cumulative error count
    # only ever grows (error rows stay 'error' forever) so it read as alarming while
    # the pipeline was actually healthy. Rolling reflects current health.
    cutoff7 = now - pd.Timedelta(days=7)
    st_all = q["status"].astype(str)
    proc_all = pd.to_datetime(q["processed_at"], errors="coerce") + _IST
    done24_total = int(((st_all == "done") & (proc_all >= cutoff)).sum())
    err24_total = int(((st_all == "error") & (proc_all >= cutoff)).sum())
    done7_total = int(((st_all == "done") & (proc_all >= cutoff7)).sum())
    err7_total = int(((st_all == "error") & (proc_all >= cutoff7)).sum())
    rate7 = err7_total / max(1, done7_total + err7_total) * 100

    rows = []
    for dt, g in q.groupby(q["doc_type"].astype(str)):
        st = g["status"].astype(str)
        pr = pd.to_datetime(g["processed_at"], errors="coerce") + _IST   # UTC->IST
        done = int((st == "done").sum())
        done24 = int(((st == "done") & (pr >= cutoff)).sum())
        pend = int((st == "pending").sum())
        done7 = int(((st == "done") & (pr >= cutoff7)).sum())
        err7 = int(((st == "error") & (pr >= cutoff7)).sum())
        e7pct = err7 / max(1, done7 + err7) * 100
        exp = int(st.isin(["expired", "download_failed"]).sum())
        pcn = 0
        if pc is not None and not pc.empty:
            pcn = (pc[pc["doc_type"].astype(str) == str(dt)]["key"]
                   .astype(str).nunique())
        rows.append((str(dt), done, done24, pend, err7, e7pct, exp, pcn))
    rows.sort(key=lambda r: -r[1])

    body = "".join(
        f"<tr><td>{_esc(dt, 16)}</td>"
        f"<td align=right>{done:,}</td>"
        f"<td align=right>{'+' + format(done24, ',') if done24 else '0'}</td>"
        f"<td align=right>{pend:,}</td>"
        f"<td align=right>{err7:,} ({e7pct:.1f}%)</td>"
        f"<td align=right>{exp:,}</td>"
        f"<td align=right>{pcn:,}</td></tr>"
        for dt, done, done24, pend, err7, e7pct, exp, pcn in rows)
    total_done = int((st_all == "done").sum())
    head = (f"<p><b>📄 {done24_total:,} docs summarised in the last 24h</b> "
            f"(+{err24_total:,} errored) · 7-day error rate "
            f"<b>{rate7:.1f}%</b> ({err7_total:,}/{done7_total + err7_total:,}) · "
            f"{total_done:,} done all-time (one global queue; IST).</p>")
    table = ("<table border=1 cellpadding=4 cellspacing=0>"
             "<tr><th>doc_type</th><th>done</th><th>+24h</th><th>pending</th>"
             "<th>err 7d</th><th>expired</th><th>page-checked cos</th></tr>"
             + body + "</table>"
             "<p style='font-size:11px;color:#999'>err 7d = errors in the last 7 days "
             "(rolling; % of that window's attempts) — all-time errors stay in the "
             "queue but aren't reported here. expired = PDF aged out (2-day "
             "retention) before summarising — re-fetched automatically. "
             "page-checked = companies whose Screener page we have swept for that "
             "doc_type (Stage-0 denominator). Full tier/depth %: "
             "company_repo/_index/coverage_report.md.</p>")
    return head + table, done24_total


# ---------------- section 2e: structured signals (tabulation) ----------------

# (label, path_parts, key column for distinct-company count)
_STRUCT_TABLES = [
    ("AR guidance",        ["company_repo", "_index", "ar_guidance.parquet"],      "isin"),
    ("AR red flags",       ["company_repo", "_index", "ar_red_flags.parquet"],     "isin"),
    ("PPT guidance",       ["company_repo", "_index", "ppt_guidance.parquet"],     "isin"),
    ("PPT highlights",     ["company_repo", "_index", "ppt_highlights.parquet"],   "isin"),
    ("Rating drivers",     ["company_repo", "_index", "rating_drivers.parquet"],   "isin"),
    ("Rating concerns",    ["company_repo", "_index", "rating_concerns.parquet"],  "isin"),
    ("Rating sensitivity", ["company_repo", "_index", "rating_sensitivity.parquet"], "isin"),
]


def structured_signals_html(drive, root) -> str:
    """Row counts + distinct companies for every Stage-2/3 tabulation table, plus the
    catalyst event-tag mix — so the daily mail shows tabulation actually producing
    rows across concall/AR/presentation/rating/announcements."""
    rows = []
    for label, parts, keycol in _STRUCT_TABLES:
        df = _read(drive, root, parts)
        if df is None or df.empty:
            rows.append(f"<tr><td>{_esc(label, 22)}</td><td align=right>0</td>"
                        f"<td align=right>-</td><td>-</td></tr>")
            continue
        n = len(df)
        nco = df[keycol].astype(str).nunique() if keycol in df.columns else "-"
        # one short sample cell (first non-null text field after the keys)
        textcol = next((c for c in ("driver", "concern", "flag_type", "metric",
                                    "statement", "trigger", "category")
                        if c in df.columns), None)
        sample = _esc(df[textcol].dropna().iloc[0], 46) if textcol and df[textcol].notna().any() else ""
        rows.append(f"<tr><td>{_esc(label, 22)}</td><td align=right>{n:,}</td>"
                    f"<td align=right>{nco}</td><td>{sample}</td></tr>")
    tbl = ("<table border=1 cellpadding=4 cellspacing=0>"
           "<tr><th>Table</th><th>rows</th><th>companies</th><th>sample</th></tr>"
           + "".join(rows) + "</table>")

    # Catalyst event tags (announcement_ledger): event_type / direction mix.
    led = _read(drive, root, ["company_repo", "_index", "announcement_ledger.parquet"])
    cat_html = ""
    if led is not None and not led.empty and "event_type" in led.columns:
        et = led["event_type"].astype(str).value_counts().head(8).to_dict()
        dr = (led["direction"].astype(str).value_counts().to_dict()
              if "direction" in led.columns else {})
        cat_html = (f"<p style='margin:6px 0 2px'><b>Catalyst event tags</b> "
                    f"({len(led):,} announcements):</p>"
                    f"<p>event_type: {_esc(et, 200)}<br>direction: {_esc(dr, 120)}</p>")
    return tbl + cat_html


# ---------------- section 2f: Gemini key x model usage (24h) ----------------

def gemini_usage_html(drive, root) -> str:
    """Per-(key, model) attribution over the last 24h from gemini_usage.parquet:
    how many summaries each bucket produced and WHY others stopped — rpm_cool =
    PerMinute (the known RPM cap), overload_503 = model 503s, state dead_today =
    PerDay quota. This is the empirical read on the throughput bottleneck."""
    df = _read(drive, root, ["company_repo", "_index", "gemini_usage.parquet"])
    if df is None or df.empty:
        return ("<p><i>gemini_usage.parquet not present yet — fills after the next "
                "extract run (concall/AR/rating/presentation).</i></p>")
    df = df.copy()
    df["ts"] = pd.to_datetime(df["ts"], errors="coerce") + _IST   # UTC->IST for report
    day = df[df["ts"] >= (pd.Timestamp(datetime.utcnow()) + _IST - timedelta(hours=24))]
    if day.empty:
        return "<p><i>No extract activity logged in the last 24h.</i></p>"
    for c in ("ok", "fail", "rpm_cool", "overload_503"):
        day[c] = pd.to_numeric(day[c], errors="coerce").fillna(0).astype(int)

    tot_ok = int(day["ok"].sum())
    tot_rpm = int(day["rpm_cool"].sum())
    tot_503 = int(day["overload_503"].sum())
    perday = day[day["state"].astype(str) == "dead_today"]
    n_perday = perday[["key_idx", "model"]].drop_duplicates().shape[0]
    n_keys = day["key_idx"].nunique()
    n_models = day["model"].nunique()

    # Data-driven verdict (was a hardcoded "RPM>>PerDay" claim — wrong when comparable).
    if tot_rpm >= 2 * max(n_perday, 1):
        verdict = "rate-limit (RPM/min) dominated"
    elif n_perday >= 2 * max(tot_rpm, 1):
        verdict = "daily-quota (PerDay) dominated → add projects"
    else:
        verdict = "BOTH RPM + daily-quota limited"
    head = (f"<p><b>🔑 {tot_ok:,} Gemini calls in 24h</b> across {n_keys} keys × {n_models} "
            f"models · RPM-cooldowns={tot_rpm:,} · 503s={tot_503:,} · "
            f"PerDay-dead buckets={n_perday}. <i>({verdict}.)</i></p>")

    # per-model rollup
    pm = (day.groupby("model")
             .agg(summaries=("ok", "sum"), rpm=("rpm_cool", "sum"),
                  s503=("overload_503", "sum"), keys=("key_idx", "nunique"))
             .reset_index().sort_values("summaries", ascending=False))
    mrows = "".join(
        f"<tr><td>{_esc(r['model'],26)}</td><td align=right>{int(r['summaries']):,}</td>"
        f"<td align=right>{int(r['keys'])}</td><td align=right>{int(r['rpm']):,}</td>"
        f"<td align=right>{int(r['s503']):,}</td></tr>" for _, r in pm.iterrows())
    mtbl = ("<p style='margin:6px 0 2px'><b>by model:</b></p>"
            "<table border=1 cellpadding=4 cellspacing=0><tr><th>model</th>"
            "<th>summaries</th><th>keys</th><th>RPM-cool</th><th>503</th></tr>"
            + mrows + "</table>")

    # per-key rollup (top 15 by summaries)
    pk = (day.groupby("key_idx")
             .agg(summaries=("ok", "sum"), rpm=("rpm_cool", "sum"),
                  s503=("overload_503", "sum"),
                  models=("model", "nunique"))
             .reset_index().sort_values("summaries", ascending=False).head(15))
    krows = "".join(
        f"<tr><td>key{int(r['key_idx'])}</td><td align=right>{int(r['summaries']):,}</td>"
        f"<td align=right>{int(r['models'])}</td><td align=right>{int(r['rpm']):,}</td>"
        f"<td align=right>{int(r['s503']):,}</td></tr>" for _, r in pk.iterrows())
    ktbl = ("<p style='margin:6px 0 2px'><b>by key (top 15):</b></p>"
            "<table border=1 cellpadding=4 cellspacing=0><tr><th>key</th>"
            "<th>summaries</th><th>models</th><th>RPM-cool</th><th>503</th></tr>"
            + krows + "</table>")
    return head + mtbl + ktbl


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    drive = get_drive()
    root = os.environ["GDRIVE_FOLDER_ID"]
    repo_id = get_or_create_subfolder(drive, root, "company_repo")
    index_id = get_or_create_subfolder(drive, repo_id, "_index")

    wf_html, ok, failed = workflow_runs_html()
    fresh_html, n_bad = freshness_html(drive, root)
    cov_html, n_cov_bad = coverage_html(drive, root)
    mcap_html, n_mcap_bad = mcap_health_html(drive, root)
    docs_html, docs_24h = docs_processed_html(drive, root)
    body = (f"<h3>Workflow runs (last 26h)</h3>{wf_html}"
            f"<h3>Document processing</h3>{docs_html}"
            f"<h3>Structured signals (tabulation)</h3>{structured_signals_html(drive, root)}"
            f"<h3>Gemini key×model usage (24h)</h3>{gemini_usage_html(drive, root)}"
            f"<h3>Stock coverage</h3>{cov_html}"
            f"<h3>Market-cap freshness</h3>{mcap_html}"
            f"<h3>Data freshness</h3>{fresh_html}"
            f"<h3>Samples</h3>{samples_html(drive, root)}"
            f"{flags_html(drive, root)}"
            f"<p style='font-size:11px;color:#999'>Toggle this mail in the app "
            f"sidebar (📧 Email toggles) or toggle_mail.bat.</p>")
    status = "🔴" if (failed or n_bad or n_cov_bad or n_mcap_bad) else "🟢"
    subject = (f"{status} Ops digest — {ok} ok / {failed} failed / "
               f"{n_bad} stale / docs +{docs_24h}/24h / "
               f"coverage {'⚠️' if n_cov_bad else 'ok'} / "
               f"mcap {'⚠️' if n_mcap_bad else 'ok'} — {date.today()}")
    log(f"ops digest: {ok} ok, {failed} failed, {n_bad} stale, "
        f"docs_24h={docs_24h}, coverage_bad={n_cov_bad}, mcap_bad={n_mcap_bad}")

    if args.dry_run:
        prev = os.path.join(os.path.dirname(_SCRIPTS_DIR), "ops_digest_preview.html")
        with open(prev, "w", encoding="utf-8") as f:
            f.write(body)
        log(f"DRY-RUN — preview saved to {prev}; no email.")
        return
    if not load_mail_settings(drive, index_id).get("ops_digest", True):
        log("ops_digest mail toggled OFF — skipped.")
        return
    send_email(subject, body)


if __name__ == "__main__":
    main()
