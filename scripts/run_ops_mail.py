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


def _esc(s, n=140) -> str:
    return html_mod.escape(str(s)[:n])


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
    body = (f"<h3>Workflow runs (last 26h)</h3>{wf_html}"
            f"<h3>Data freshness</h3>{fresh_html}"
            f"<h3>Samples</h3>{samples_html(drive, root)}"
            f"{flags_html(drive, root)}"
            f"<p style='font-size:11px;color:#999'>Toggle this mail in the app "
            f"sidebar (📧 Email toggles) or toggle_mail.bat.</p>")
    status = "🔴" if (failed or n_bad) else "🟢"
    subject = (f"{status} Ops digest — {ok} ok / {failed} failed / "
               f"{n_bad} stale — {date.today()}")
    log(f"ops digest: {ok} ok, {failed} failed, {n_bad} stale datasets")

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
