"""
mailer.py — tiny shared Gmail sender (Phase 3). Reuses the SAME env config as
notify_deepdive.py (GMAIL_USER / GMAIL_APP_PASSWORD / NOTIFY_EMAIL). Import-safe.

    from mailer import send_email
    send_email("Subject", "<b>html</b>", "plain fallback")

Per-mail on/off toggles live in company_repo/_index/mail_settings.json on Drive
(written by the Streamlit app's "Email toggles" sidebar). Senders check:

    from mailer import load_mail_settings
    if load_mail_settings(drive, index_id).get("catalyst", True): send_email(...)

Missing file / unknown key = ON (mails only go quiet when explicitly toggled off).
"""
from __future__ import annotations

import json
import os
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

GMAIL_USER = os.environ.get("GMAIL_USER", "")
GMAIL_PASS = os.environ.get("GMAIL_APP_PASSWORD", "")
NOTIFY_EMAIL = os.environ.get("NOTIFY_EMAIL", GMAIL_USER)

SETTINGS_NAME = "mail_settings.json"
# ALL mails arrive in ONE evening batch from 19:00 IST (user rule 2026-06-12):
# pead fires at 19:00, t4_nightly chains off its completion.
MAIL_KEYS = {                      # key -> human label (app.py shows these)
    "pead_guidance":   "Results last 24h vs guidance",
    "pead_tomorrow":   "Results tomorrow calendar",
    "fraud_scan":      "Fraud scan findings",
    "catalyst":        "Catalyst notes digest",
    "guidance_digest": "Concall guidance table + 🚀 >30% flags (last 24h)",
    "ops_digest":      "Ops digest: runs pass/fail + freshness + samples",
    "ar_focus":        "AR digest: focus/defocus lists (PDF attached)",
    "pf_digest":       "PF daily digest: filings/research/flags/docs/community",
    "pf_results":      "PF results digest: quarterly revenue/profit/EPS/margin (season)",
    "growth_surge":    "Results surge: >100% PAT YoY, consistent vs one-off (last 24h)",
    "combined_strength": "Combined strength: strong tech + results + guidance",
    "pf_docs_digest":  "PF new documents: AR/concall/presn/rating/announcement + summary",
    "guidance_watchlist": "Guidance watchlist: names entering/leaving the >50% CAGR list",
    "pf_decision":     "PF decision scorecard: index + movers + buy/sell/hold verdicts + sold relook",
    "infra_health":    "Infra code-red: broken keys/cookies + fix steps (CRIT bypasses this)",
    "quarter_teardown": "Quarterly teardown: clean PAT + divergence engine + red-flag register + deck",
    "mgmt_interviews": "Management interviews (PORTFOLIO): MD/CEO/CFO on TV + print, link + summary",
    "mgmt_interviews_watchlist": "Management interviews (WATCHLIST): same, for non-held names",
    # the only WEEKLY mail — runs Monday in its own workflow, not the 19:00 batch
    "guidance_progress": "WEEKLY guidance progress: how far along each promise is + what moved this week",
}


def esc(s, n=120) -> str:
    """HTML-escape + truncate — the one escape helper every mail builder uses."""
    import html
    return html.escape(str(s)[:n])


def default_mail_settings() -> dict:
    return {k: True for k in MAIL_KEYS}


def load_mail_settings(drive, index_id) -> dict:
    """Read company_repo/_index/mail_settings.json -> {key: bool}. Any failure
    (file absent, Drive hiccup) returns all-ON so a toggle bug never silences
    mails accidentally."""
    settings = default_mail_settings()
    try:
        from _extractor_base import find_file, download_bytes
        fid = find_file(drive, index_id, SETTINGS_NAME)
        if fid:
            data = json.loads(download_bytes(drive, fid).decode("utf-8"))
            settings.update({k: bool(v) for k, v in data.items() if k in settings})
    except Exception as e:
        print(f"mailer: could not read {SETTINGS_NAME} ({str(e)[:60]}) — all ON.")
    return settings


def send_email(subject: str, html_body: str, plain_body: str = "",
               to: str | None = None,
               attachments: list[tuple[str, bytes, str]] | None = None,
               inline_images: list[tuple[str, bytes]] | None = None) -> bool:
    """Send one email. attachments = [(filename, bytes, mime_subtype)], e.g.
    ("digest.pdf", pdf_bytes, "pdf"). inline_images = [(cid, png_bytes)], shown where
    the html has <img src="cid:..."> (mail_chart.img_tag). Returns True on success."""
    if not GMAIL_USER or not GMAIL_PASS:
        print("mailer: GMAIL_USER / GMAIL_APP_PASSWORD not set — skipping email.")
        return False
    recipient = to or NOTIFY_EMAIL
    msg = _build_message(html_body, plain_body, attachments, inline_images)
    msg["Subject"] = subject
    msg["From"] = GMAIL_USER
    msg["To"] = recipient
    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
            smtp.login(GMAIL_USER, GMAIL_PASS)
            smtp.send_message(msg)
        print(f"mailer: sent '{subject[:50]}' -> {recipient}")
        return True
    except Exception as e:
        print(f"mailer: send failed ({type(e).__name__}: {str(e)[:100]})")
        return False


def _build_message(html_body: str, plain_body: str = "",
                   attachments: list[tuple[str, bytes, str]] | None = None,
                   inline_images: list[tuple[str, bytes]] | None = None):
    """The MIME tree. Without inline_images it is exactly what it always was:
    alternative, or mixed(alternative + attachments). With them, the alternative part
    is wrapped in multipart/related together with the images, which is how a mail
    client resolves <img src="cid:...">."""
    alt = MIMEMultipart("alternative")
    alt.attach(MIMEText(plain_body or _strip_html(html_body), "plain"))
    alt.attach(MIMEText(html_body, "html"))
    body = alt
    if inline_images:
        from email.mime.image import MIMEImage
        body = MIMEMultipart("related")
        body.attach(alt)
        for cid, data in inline_images:
            if not data:
                continue
            img = MIMEImage(data, _subtype="png")
            img.add_header("Content-ID", f"<{cid}>")
            img.add_header("Content-Disposition", "inline", filename=f"{cid}.png")
            body.attach(img)
    if attachments:
        from email.mime.application import MIMEApplication
        msg = MIMEMultipart("mixed")
        msg.attach(body)
        for fname, data, sub in attachments:
            part = MIMEApplication(data, _subtype=sub)
            part.add_header("Content-Disposition", "attachment", filename=fname)
            msg.attach(part)
        return msg
    return body


def _self_test() -> int:
    """Offline: the MIME tree, with and without inline images. Never sends."""
    fails = 0

    def check(name, ok):
        nonlocal fails
        print(("  ok   " if ok else "  FAIL ") + name)
        fails += 0 if ok else 1

    m = _build_message("<b>x</b>")
    check("plain mail unchanged: multipart/alternative", m.get_content_type()
          == "multipart/alternative")
    m = _build_message("<b>x</b>", attachments=[("a.pdf", b"%PDF", "pdf")])
    check("attachments unchanged: mixed > alternative",
          m.get_content_type() == "multipart/mixed"
          and m.get_payload()[0].get_content_type() == "multipart/alternative")
    png = b"\x89PNG\r\n\x1a\n" + b"0" * 20
    m = _build_message('<img src="cid:c1">', inline_images=[("c1", png), ("c2", b"")])
    parts = m.get_payload()
    check("inline: related > alternative + image", m.get_content_type()
          == "multipart/related" and parts[1].get_content_type() == "image/png")
    check("inline: Content-ID set", parts[1]["Content-ID"] == "<c1>")
    check("inline: empty image skipped", len(parts) == 2)
    m = _build_message("x", attachments=[("a.pdf", b"%PDF", "pdf")],
                       inline_images=[("c1", png)])
    check("inline + attachment: mixed > related",
          m.get_payload()[0].get_content_type() == "multipart/related")
    print(f"\n{'ALL PASS' if not fails else f'{fails} FAILED'}")
    return 1 if fails else 0


def _strip_html(html: str) -> str:
    import re
    return re.sub(r"<[^>]+>", "", html).replace("&nbsp;", " ")


if __name__ == "__main__":
    import sys
    if "--self-test" in sys.argv:
        raise SystemExit(_self_test())
