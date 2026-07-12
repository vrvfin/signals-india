r"""
check_infra_health.py — CODE-RED infra monitor (user 2026-07-12; NO Gemini).

The problem it fixes: the ops digest is perpetually "🔴 43 ok / 4 failed / 2 stale",
so a GENUINE break (e.g. the Screener cookie expiring for 20 days) looks identical to
routine noise — alert fatigue. This script is the opposite: it stays SILENT when
healthy and only fires when something is actually broken, names the EXACT key/cookie,
and gives click-by-click remediation. CRIT also pushes to your phone (ntfy) because
email delivery itself can't be trusted.

Three signal sources, correlated (definitive, no HTML-scraping guesswork):
  1. DATA FRESHNESS — critical parquets past a hard age threshold (Drive).
  2. LIVE PROBES    — Drive token refresh; Gemini 24h success (gemini_usage.parquet).
  3. LOG SCAN       — downloads the last ~20 workflow-run logs (GitHub API) and greps
                      a curated failure-signature dictionary (the "read ALL logs" ask).

Severity: CRIT (code red) > WARN. Mail fires only when severity != OK. CRIT ignores the
'infra_health' toggle (a code red can't be silenced) and also sends an ntfy push.

Usage:
    python scripts/check_infra_health.py --dry-run   # build preview, print issues, no send
    python scripts/check_infra_health.py             # send only if broken
    python scripts/check_infra_health.py --always     # send even when healthy (heartbeat)
"""
from __future__ import annotations

import argparse
import io
import os
import sys
import zipfile
from datetime import datetime, timedelta
from pathlib import Path

_SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(_SCRIPTS_DIR), ".env"))

from _extractor_base import (get_drive, get_or_create_subfolder, find_file,
                             download_bytes, log)
from mailer import send_email, load_mail_settings, esc

REPO = os.environ.get("GH_REPO", "vrvfin/signals-india")
_SECRETS_URL = f"https://github.com/{REPO}/settings/secrets/actions"

# ---- remediation playbooks (exact steps) ----
FIX_SCREENER = (
    "1. Open https://www.screener.in and LOG IN.<br>"
    "2. DevTools (F12) → Application → Cookies → screener.in → copy the "
    "<b>sessionid</b> value.<br>"
    f"3. <a href='{_SECRETS_URL}'>Repo → Settings → Secrets → Actions</a> → "
    "<b>SCREENER_SESSION_COOKIE</b> → Update → paste → Save.<br>"
    "4. Re-run: <code>gh workflow run pead.yml</code> (and fundamentals.yml on Mon).")
FIX_GDRIVE = (
    "The Google Drive OAuth token was revoked/expired. Regenerate it locally "
    "(the OAuth flow writes a fresh token), then update the "
    f"<b>GDRIVE_OAUTH_TOKEN_JSON</b> secret at <a href='{_SECRETS_URL}'>Actions "
    "secrets</a>. Nothing on Drive updates until this is fixed.")
FIX_GEMINI = (
    "No Gemini summaries succeeded in 24h. Either every free-tier bucket hit its "
    "daily quota (add a NEW Google-Cloud PROJECT's keys — more keys from the same "
    "project add ZERO quota) or the keys are invalid. Check FREE_POOL / GEMINI_API_KEY "
    f"at <a href='{_SECRETS_URL}'>Actions secrets</a>.")
FIX_GMAIL = (
    "Mailer could not authenticate. Verify <b>GMAIL_USER</b> + "
    "<b>GMAIL_APP_PASSWORD</b> (a Google App Password, not the account password) and "
    f"that <b>NOTIFY_EMAIL</b> is the inbox you read, at "
    f"<a href='{_SECRETS_URL}'>Actions secrets</a>. Also check Gmail Spam/Promotions.")

# label, path parts, ts column, CRIT-threshold hours, severity-if-stale, name, fix
FRESHNESS = [
    ("results (Screener scrape)", ["company_repo", "_index", "results.parquet"],
     "scraped_at", 48, "CRIT", "SCREENER_SESSION_COOKIE", FIX_SCREENER),
    ("financials_3stmt", ["company_repo", "_index", "financials_3stmt.parquet"],
     "scraped_at", 96, "WARN", "SCREENER_SESSION_COOKIE", FIX_SCREENER),
    ("fundamentals/summary (mcap)", ["fundamentals", "summary.parquet"],
     "fetched_at", 16 * 24, "WARN", "SCREENER_SESSION_COOKIE", FIX_SCREENER),
]

# regex-free substring signatures -> (severity, issue-key, human, fix)
LOG_SIGNATURES = [
    ("No results scraped. Check the Screener cookie", "CRIT", "screener_cookie",
     "Screener results scrape returned nothing", FIX_SCREENER),
    ("invalid_grant", "CRIT", "gdrive_token", "Google OAuth token rejected", FIX_GDRIVE),
    ("Token has been expired or revoked", "CRIT", "gdrive_token",
     "Google OAuth token expired/revoked", FIX_GDRIVE),
    ("GMAIL_USER / GMAIL_APP_PASSWORD not set", "WARN", "gmail",
     "Mailer skipped — Gmail creds missing", FIX_GMAIL),
    ("RESOURCE_EXHAUSTED", "WARN", "gemini_quota", "Gemini daily quota exhausted", FIX_GEMINI),
    ("Check the Screener cookie", "WARN", "screener_cookie",
     "A step warned about the Screener cookie", FIX_SCREENER),
]


class Issue:
    def __init__(self, key, severity, title, detail, fix):
        self.key, self.severity = key, severity
        self.title, self.detail, self.fix = title, detail, fix


def _read(drive, root, parts):
    fid = root
    for p in parts[:-1]:
        fid = get_or_create_subfolder(drive, fid, p)
    f = find_file(drive, fid, parts[-1])
    if not f:
        return None
    try:
        return pd.read_parquet(io.BytesIO(download_bytes(drive, f)))
    except Exception:
        return None


# ---------------- freshness ----------------

def check_freshness(drive, root) -> list[Issue]:
    issues, now = [], datetime.now()
    for label, parts, ts, thr_h, sev, key, fix in FRESHNESS:
        df = _read(drive, root, parts)
        if df is None or df.empty or ts not in df.columns:
            issues.append(Issue(key, sev, f"{label}: MISSING / unreadable",
                                f"{'/'.join(parts)} absent or has no '{ts}' column.", fix))
            continue
        latest = pd.to_datetime(df[ts], errors="coerce").max()
        if pd.isna(latest):
            continue
        age_h = (now - latest.to_pydatetime().replace(tzinfo=None)).total_seconds() / 3600
        if age_h > thr_h:
            issues.append(Issue(
                key, sev, f"{label}: STALE {age_h/24:.1f} days",
                f"Newest row is <b>{latest:%Y-%m-%d %H:%M}</b> "
                f"({age_h:.0f}h old, threshold {thr_h}h). Everything downstream "
                f"(PEAD, screener grades, scorecard, growth-surge, combined-strength) "
                f"is running on stale numbers.", fix))
    return issues


# ---------------- gemini health ----------------

def check_gemini(drive, root) -> list[Issue]:
    df = _read(drive, root, ["company_repo", "_index", "gemini_usage.parquet"])
    if df is None or df.empty or "ts" not in df.columns:
        return []
    df = df.copy()
    df["ts"] = pd.to_datetime(df["ts"], errors="coerce")
    day = df[df["ts"] >= datetime.utcnow() - timedelta(hours=24)]
    if day.empty:
        return []                       # no extract activity in window = not a fault
    ok = pd.to_numeric(day.get("ok"), errors="coerce").fillna(0).sum()
    if ok == 0:
        return [Issue("gemini", "WARN", "Gemini: 0 successful calls in 24h",
                      "Extraction attempted but every key×model bucket failed "
                      "(quota or invalid keys).", FIX_GEMINI)]
    return []


# ---------------- workflow runs + log scan ----------------

def _gh(path, token, **params):
    r = requests.get(f"https://api.github.com/repos/{REPO}/{path}",
                     headers={"Authorization": f"Bearer {token}",
                              "Accept": "application/vnd.github+json"},
                     params=params, timeout=30)
    r.raise_for_status()
    return r.json()


def check_workflows_and_logs(token: str) -> tuple[list[Issue], list[str]]:
    """Failed runs (24h) + curated log-signature hits across recent run logs."""
    issues, notes = [], []
    if not token:
        return [], ["log scan skipped (no GITHUB_TOKEN — local run)."]
    since = (datetime.utcnow() - timedelta(hours=26)).strftime("%Y-%m-%dT%H:%M")
    try:
        runs = _gh("actions/runs", token, per_page=40, created=f">={since}").get(
            "workflow_runs", [])
    except Exception as e:
        return [], [f"workflow list failed: {str(e)[:80]}"]

    failed = [x for x in runs if str(x.get("conclusion")) in
              ("failure", "timed_out", "cancelled")]
    if failed:
        names = {}
        for x in failed:
            names[str(x.get("name"))] = names.get(str(x.get("name")), 0) + 1
        detail = ", ".join(f"{n}×{c}" for n, c in sorted(names.items(),
                                                         key=lambda kv: -kv[1]))
        issues.append(Issue("workflow_fail", "WARN",
                            f"{len(failed)} workflow run(s) failed in 26h", detail,
                            "Open the run logs (GitHub → Actions) — this scan lists the "
                            "known signatures below; anything else is a code/CI fault."))

    # scan the last ~18 run logs for signatures (zip per run)
    seen = {}
    for x in runs[:18]:
        rid = x.get("id")
        try:
            resp = requests.get(
                f"https://api.github.com/repos/{REPO}/actions/runs/{rid}/logs",
                headers={"Authorization": f"Bearer {token}"}, timeout=45)
            if resp.status_code != 200:
                continue
            zf = zipfile.ZipFile(io.BytesIO(resp.content))
            text = "\n".join(zf.read(n).decode("utf-8", "replace")
                             for n in zf.namelist() if n.endswith(".txt"))
        except Exception:
            continue
        for sig, sev, key, human, fix in LOG_SIGNATURES:
            if sig in text and key not in seen:
                seen[key] = Issue(key, sev, human,
                                  f"Signature <code>{esc(sig, 60)}</code> seen in "
                                  f"workflow <b>{esc(x.get('name'), 30)}</b>.", fix)
    issues.extend(seen.values())
    notes.append(f"log scan: {len(runs)} runs listed, {min(len(runs),18)} logs read.")
    return issues, notes


# ---------------- assemble + send ----------------

def _dedupe(issues: list[Issue]) -> list[Issue]:
    """One issue per key; keep the most severe."""
    rank = {"CRIT": 2, "WARN": 1}
    best = {}
    for i in issues:
        if i.key not in best or rank[i.severity] > rank[best[i.key].severity]:
            best[i.key] = i
    return sorted(best.values(), key=lambda i: -rank[i.severity])


def build_html(issues: list[Issue], notes: list[str], sev: str) -> str:
    banner = ("🔴 <b>CODE RED — critical infra failure</b>" if sev == "CRIT"
              else "🟠 <b>Infra warnings</b>" if sev == "WARN"
              else "🟢 <b>All systems healthy</b>")
    out = [f"<div style='max-width:720px;font-family:Arial,sans-serif'>"
           f"<p style='font-size:15px'>{banner} — {datetime.now():%Y-%m-%d %H:%M}</p>"]
    for i in issues:
        color = "#c0392b" if i.severity == "CRIT" else "#e67e22"
        out.append(
            f"<div style='border-left:4px solid {color};padding:6px 10px;margin:10px 0;"
            f"background:#faf3f0'>"
            f"<div style='font-size:14px'><b>{'🔴' if i.severity=='CRIT' else '🟠'} "
            f"{esc(i.title, 90)}</b></div>"
            f"<div style='font-size:12.5px;color:#333;margin:4px 0'>{i.detail}</div>"
            f"<div style='font-size:12.5px;color:#111;margin-top:4px'>"
            f"<b>Fix:</b><br>{i.fix}</div></div>")
    if not issues:
        out.append("<p>No critical secrets/cookies broken, data fresh, no failing "
                   "workflows, no known bad signatures in recent logs.</p>")
    if notes:
        out.append("<p style='font-size:11px;color:#999'>"
                   + " · ".join(esc(n, 120) for n in notes) + "</p>")
    out.append("</div>")
    return "\n".join(out)


def _ntfy(title: str, msg: str) -> bool:
    topic = os.environ.get("NTFY_TOPIC", "").strip()
    if not topic:
        return False
    try:
        requests.post(f"https://ntfy.sh/{topic}", data=msg.encode("utf-8"),
                      headers={"Title": title, "Priority": "urgent",
                               "Tags": "rotating_light",
                               "Click": f"https://github.com/{REPO}/actions"}, timeout=15)
        return True
    except Exception as e:
        log(f"ntfy push failed (non-fatal): {str(e)[:60]}")
        return False


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--always", action="store_true",
                    help="Send even when healthy (heartbeat).")
    args = ap.parse_args()

    print("Infra health monitor — code-red on genuine breakage only")
    print("-" * 60)
    issues, notes = [], []

    # Drive is the gateway; if it fails that is itself a CRIT we can still alert on.
    drive = root = None
    try:
        drive = get_drive()
        root = os.environ["GDRIVE_FOLDER_ID"]
        issues += check_freshness(drive, root)
        issues += check_gemini(drive, root)
    except Exception as e:
        issues.append(Issue("gdrive_token", "CRIT", "Google Drive access FAILED",
                            f"get_drive() raised: <code>{esc(e, 100)}</code>. Nothing "
                            f"can read or write Drive until this is fixed.", FIX_GDRIVE))

    wf_issues, wf_notes = check_workflows_and_logs(os.environ.get("GITHUB_TOKEN", ""))
    issues += wf_issues
    notes += wf_notes

    issues = _dedupe(issues)
    sev = "CRIT" if any(i.severity == "CRIT" for i in issues) else \
          "WARN" if issues else "OK"
    for i in issues:
        log(f"  [{i.severity}] {i.title}")
    log(f"overall severity: {sev} ({len(issues)} issue(s))")

    html = build_html(issues, notes, sev)
    if args.dry_run:
        prev = Path(__file__).resolve().parent.parent / "infra_health_preview.html"
        prev.write_text(html, encoding="utf-8")
        print(f"\nDRY RUN — preview saved to {prev.name}; no mail, no push.")
        return

    if sev == "OK" and not args.always:
        log("healthy — no alert sent (silence = green).")
        return

    n_crit = sum(1 for i in issues if i.severity == "CRIT")
    subject = (f"🔴 CODE RED — {n_crit} critical infra issue(s)" if sev == "CRIT"
               else f"🟠 Infra warnings — {len(issues)} issue(s)" if sev == "WARN"
               else "🟢 Infra healthy (heartbeat)")

    # CRIT bypasses the toggle — a code red must not be silenceable.
    toggled_on = True
    if drive is not None and root is not None:
        idx = get_or_create_subfolder(
            drive, get_or_create_subfolder(drive, root, "company_repo"), "_index")
        toggled_on = load_mail_settings(drive, idx).get("infra_health", True)
    if sev == "CRIT" or toggled_on:
        sent = send_email(subject, html)
        log(f"Email {'sent' if sent else 'FAILED'}: "
            f"{subject.encode('ascii', 'ignore').decode().strip()}")
    else:
        log("infra_health toggled OFF and severity=WARN — email skipped.")

    if sev == "CRIT":
        top = "; ".join(i.title for i in issues if i.severity == "CRIT")[:180]
        pushed = _ntfy(f"CODE RED — {n_crit} infra issue(s)", top)
        log(f"ntfy phone push: {'sent' if pushed else 'not configured'}")


if __name__ == "__main__":
    main()
