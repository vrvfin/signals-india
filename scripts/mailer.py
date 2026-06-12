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
MAIL_KEYS = {                      # key -> human label (app.py shows these)
    "pead_guidance":   "Results last 24h vs guidance (daily 20:00 IST)",
    "pead_tomorrow":   "Results tomorrow calendar (daily 20:00 IST)",
    "fraud_scan":      "Fraud scan findings (nightly 21:30 IST)",
    "catalyst":        "Catalyst notes digest (nightly 21:30 IST)",
    "guidance_digest": "Concall guidance table + 🚀 >30% flags, last 24h (daily ~20:00 IST)",
    "ops_digest":      "Ops digest: runs pass/fail + freshness + samples (daily 08:30 IST)",
    "ar_focus":        "AR digest: focus/defocus lists from fresh annual reports (per 4h slot)",
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
               attachments: list[tuple[str, bytes, str]] | None = None) -> bool:
    """Send one email. attachments = [(filename, bytes, mime_subtype)], e.g.
    ("digest.pdf", pdf_bytes, "pdf"). Returns True on success."""
    if not GMAIL_USER or not GMAIL_PASS:
        print("mailer: GMAIL_USER / GMAIL_APP_PASSWORD not set — skipping email.")
        return False
    recipient = to or NOTIFY_EMAIL
    alt = MIMEMultipart("alternative")
    alt.attach(MIMEText(plain_body or _strip_html(html_body), "plain"))
    alt.attach(MIMEText(html_body, "html"))
    if attachments:
        from email.mime.application import MIMEApplication
        msg = MIMEMultipart("mixed")
        msg.attach(alt)
        for fname, data, sub in attachments:
            part = MIMEApplication(data, _subtype=sub)
            part.add_header("Content-Disposition", "attachment", filename=fname)
            msg.attach(part)
    else:
        msg = alt
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


def _strip_html(html: str) -> str:
    import re
    return re.sub(r"<[^>]+>", "", html).replace("&nbsp;", " ")
