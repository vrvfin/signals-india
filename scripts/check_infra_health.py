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


def _fix(secret: str, where: str, fmt: str, steps: list[str]) -> str:
    """Uniform remediation block: exact SECRET name, WHERE to update (link),
    the expected VALUE FORMAT (with an example), then click-by-click steps."""
    return (
        f"&nbsp;&nbsp;• <b>Secret:</b> <code>{secret}</code><br>"
        f"&nbsp;&nbsp;• <b>Where:</b> <a href='{_SECRETS_URL}'>{_SECRETS_URL}</a> "
        f"→ <code>{secret}</code> → <b>Update</b><br>"
        f"&nbsp;&nbsp;• <b>Value format:</b> {where and where + '<br>'}{fmt}<br>"
        f"&nbsp;&nbsp;• <b>Steps:</b><br>"
        + "".join(f"&nbsp;&nbsp;&nbsp;&nbsp;{i}. {s}<br>" for i, s in enumerate(steps, 1)))


FIX_SCREENER = _fix(
    "SCREENER_SESSION_COOKIE",
    "Paste <u>only the cookie VALUE</u> — NOT <code>sessionid=…</code>, NOT the whole "
    "cookie header, NOT the <code>csrftoken</code>. No surrounding quotes or spaces.",
    "a ~32–40 char letters+digits string, e.g. <code>abcd1234efgh5678ijkl9012mnop3456</code> "
    "(format: <code>[a-z0-9]{32,}</code>)",
    ["Open https://www.screener.in and LOG IN (confirm you see your account, not 'Login').",
     "F12 → Application (Chrome) / Storage (Firefox) → Cookies → https://www.screener.in",
     "Click the row named <b>sessionid</b> and copy its <b>Value</b> column only.",
     "Update the secret above with that value, Save.",
     "Verify: <code>gh workflow run pead.yml</code> → the scrape log should say "
     "'page 1: N companies', NOT 'No results scraped'."])
FIX_GDRIVE = _fix(
    "GDRIVE_OAUTH_TOKEN_JSON",
    "The full OAuth token JSON (one line), starting with <code>{&quot;token&quot;:</code> "
    "or <code>{&quot;refresh_token&quot;:</code>.",
    "a JSON object: <code>{&quot;token&quot;:&quot;ya29…&quot;,&quot;refresh_token&quot;"
    ":&quot;1//…&quot;,&quot;client_id&quot;:…}</code>",
    ["Re-run the local OAuth flow (any get_drive() call) — it writes a fresh token file.",
     "Copy the ENTIRE contents of that token JSON (single line).",
     "Update the secret above, Save. Nothing on Drive updates until this is fixed."])
FIX_GEMINI = _fix(
    "FREE_POOL / GEMINI_API_KEY",
    "Comma-separated API keys, or one key per <code>FREE_POOL_1..18</code>.",
    "keys look like <code>AQ.…</code> (newer) or <code>AIza…</code> (older)",
    ["If all buckets hit daily quota: add a key from a NEW Google-Cloud PROJECT "
     "(more keys from the SAME project add ZERO quota).",
     "If keys are invalid: regenerate at aistudio.google.com and update the secret."])
FIX_GMAIL = _fix(
    "GMAIL_USER / GMAIL_APP_PASSWORD / NOTIFY_EMAIL",
    "GMAIL_APP_PASSWORD is a 16-char Google <u>App Password</u>, not your login password.",
    "GMAIL_USER=<code>you@gmail.com</code> · GMAIL_APP_PASSWORD=<code>abcd efgh ijkl mnop</code> "
    "(16 chars, spaces optional) · NOTIFY_EMAIL=<code>inbox-you-read@…</code>",
    ["Create an App Password: Google Account → Security → 2-Step → App passwords.",
     "Confirm NOTIFY_EMAIL is the inbox you actually read; check Spam/Promotions."])
FIX_DRIVE = (
    "&nbsp;&nbsp;<b>1. IMMEDIATE relief:</b> "
    "<code>python scripts/cleanup_company_docs.py --retain-days 1</code> — permanently "
    "deletes processed raw PDFs older than 1 day (they are re-fetchable).<br>"
    "&nbsp;&nbsp;<b>2. Root cause:</b> the annual_report backfill pulls ~800 huge AR PDFs/day "
    "(~13 GB) — far more than a 16 GB free account holds even with 2-day retention.<br>"
    "&nbsp;&nbsp;<b>3. Permanent fix:</b> lower the AR backfill fetch cap and/or delete each "
    "AR PDF immediately after extraction (ARs are 30–300 MB each; the 2-day buffer is not "
    "worth 13 GB), or move the repo to a Google account with more storage.")
FIX_CHANNEL = (
    "&nbsp;&nbsp;A code-red can only reach you if the alert channels are configured. "
    f"Set the missing secret at <a href='{_SECRETS_URL}'>{_SECRETS_URL}</a>. "
    "NTFY_TOPIC = an unguessable topic string you also SUBSCRIBE to in the ntfy phone app. "
    "NOTIFY_EMAIL = the inbox you actually read.")

# label, path parts, ts column, CRIT-threshold hours, severity-if-stale, issue-key, fix
# results + financials_3stmt share key "screener_cookie" (same root cause + the log
# signature) so they collapse into ONE code-red row. mcap is a distinct key (a missed
# weekly run can stale it independently of the cookie).
FRESHNESS = [
    ("results (Screener scrape)", ["company_repo", "_index", "results.parquet"],
     "scraped_at", 48, "CRIT", "screener_cookie", FIX_SCREENER),
    ("financials_3stmt", ["company_repo", "_index", "financials_3stmt.parquet"],
     "scraped_at", 96, "WARN", "screener_cookie", FIX_SCREENER),
    ("fundamentals/summary (mcap)", ["fundamentals", "summary.parquet"],
     "fetched_at", 16 * 24, "WARN", "mcap_stale", FIX_SCREENER),
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

def _gb(x) -> str:
    try:
        return f"{int(x) / 1e9:.2f} GB"
    except (TypeError, ValueError):
        return "?"


def check_drive_storage(drive) -> list[Issue]:
    """Account storage %: WARN >=80%, CRIT >=92% (writes fail near 100%). When over,
    enumerate PDFs to attribute the bloat (usually the annual_report backfill)."""
    try:
        sq = drive.about().get(fields="storageQuota").execute().get("storageQuota", {})
        limit = int(sq.get("limit", 0) or 0)
        usage = int(sq.get("usage", 0) or 0)
    except Exception as e:
        return [Issue("drive_full", "WARN", "Drive storage: quota unreadable",
                      f"about().get failed: {esc(e, 80)}", FIX_DRIVE)]
    if not limit:
        return []
    pct = 100 * usage / limit
    if pct < 80:
        return []
    sev = "CRIT" if pct >= 92 else "WARN"
    # attribute: sum PDF bytes + count + top doc_type prefix
    pdf_n = pdf_sz = 0
    top: dict = {}
    try:
        page = None
        while True:
            r = drive.files().list(
                q="mimeType='application/pdf' and trashed=false",
                fields="nextPageToken,files(name,quotaBytesUsed)",
                pageSize=1000, pageToken=page).execute()
            for f in r.get("files", []):
                s = int(f.get("quotaBytesUsed", 0) or 0)
                pdf_n += 1
                pdf_sz += s
                pre = str(f.get("name", "")).split("__")[0][:20]
                top[pre] = top.get(pre, 0) + s
            page = r.get("nextPageToken")
            if not page:
                break
    except Exception:
        pass
    big = max(top.items(), key=lambda kv: kv[1]) if top else ("", 0)
    detail = (f"Drive is <b>{pct:.0f}% full</b> ({_gb(usage)} / {_gb(limit)}). "
              f"Raw PDFs = <b>{_gb(pdf_sz)}</b> across {pdf_n:,} files; the biggest "
              f"bucket is <b>{esc(big[0], 20)}</b> ({_gb(big[1])}). "
              + ("<b>Writes will start FAILING near 100% — data loss risk.</b>"
                 if sev == "CRIT" else "Approaching the limit."))
    return [Issue("drive_full", sev, f"Drive storage {pct:.0f}% full", detail, FIX_DRIVE)]


def check_alert_channels() -> list[Issue]:
    """Meta-check: can a code-red actually REACH the user? A silently-unset NTFY_TOPIC
    or NOTIFY_EMAIL means the alert itself is dead — the very failure mode that let the
    cookie hide 20 days. WARN (not CRIT, so it can't loop on its own missing email)."""
    out = []
    if not os.environ.get("NTFY_TOPIC", "").strip():
        out.append(Issue("channel_ntfy", "WARN", "Phone alerts OFF (NTFY_TOPIC unset)",
                         "Code-reds are email-only. If email is filtered, you get nothing.",
                         FIX_CHANNEL))
    if not os.environ.get("NOTIFY_EMAIL", "").strip() and \
            not os.environ.get("GMAIL_USER", "").strip():
        out.append(Issue("channel_email", "WARN", "Alert email not configured",
                         "NOTIFY_EMAIL and GMAIL_USER are both unset — no mail can send.",
                         FIX_CHANNEL))
    return out


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
        issues += check_drive_storage(drive)
        issues += check_gemini(drive, root)
    except Exception as e:
        issues.append(Issue("gdrive_token", "CRIT", "Google Drive access FAILED",
                            f"get_drive() raised: <code>{esc(e, 100)}</code>. Nothing "
                            f"can read or write Drive until this is fixed.", FIX_GDRIVE))

    issues += check_alert_channels()
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
