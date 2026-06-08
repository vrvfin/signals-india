"""
mailer.py — tiny shared Gmail sender (Phase 3). Reuses the SAME env config as
notify_deepdive.py (GMAIL_USER / GMAIL_APP_PASSWORD / NOTIFY_EMAIL). Import-safe.

    from mailer import send_email
    send_email("Subject", "<b>html</b>", "plain fallback")
"""
from __future__ import annotations

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


def send_email(subject: str, html_body: str, plain_body: str = "",
               to: str | None = None) -> bool:
    """Send one email. Returns True on success, False if creds missing / failed."""
    if not GMAIL_USER or not GMAIL_PASS:
        print("mailer: GMAIL_USER / GMAIL_APP_PASSWORD not set — skipping email.")
        return False
    recipient = to or NOTIFY_EMAIL
    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = GMAIL_USER
    msg["To"] = recipient
    msg.attach(MIMEText(plain_body or _strip_html(html_body), "plain"))
    msg.attach(MIMEText(html_body, "html"))
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
