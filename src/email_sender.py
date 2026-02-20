from __future__ import annotations

import json
import os
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path

from dotenv import load_dotenv

from src.logger import setup_logger

load_dotenv()

log = setup_logger("email_sender")

CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587


def _load_subscribers() -> list[dict]:
    subs_path = CONFIG_DIR / "subscribers.json"
    with open(subs_path) as f:
        data = json.load(f)
    active = [s for s in data["subscribers"] if s.get("active", True)]
    return active


def send_email(
    subject: str,
    html_body: str,
    recipients: list[dict] | None = None,
) -> dict:
    """Send an HTML email via Gmail SMTP.

    recipients: list of dicts with 'email' and 'name' keys.
                If None, loads from config/subscribers.json.

    Returns dict with 'sent', 'failed' counts and 'details' list.
    """
    if recipients is None:
        recipients = _load_subscribers()

    if not recipients:
        log.warning("No recipients — skipping email send")
        return {"sent": 0, "failed": 0, "details": []}

    sender_address = os.environ.get("GMAIL_ADDRESS", "")
    app_password = os.environ.get("GMAIL_APP_PASSWORD", "")

    if not sender_address or not app_password:
        raise RuntimeError(
            "GMAIL_ADDRESS and GMAIL_APP_PASSWORD must be set in environment"
        )

    results = {"sent": 0, "failed": 0, "details": []}

    try:
        server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
        server.ehlo()
        server.starttls()
        server.login(sender_address, app_password)

        for recipient in recipients:
            email = recipient["email"]
            name = recipient.get("name", email)

            try:
                msg = MIMEMultipart("alternative")
                msg["Subject"] = subject
                msg["From"] = f"ArXiv AI Digest <{sender_address}>"
                msg["To"] = f"{name} <{email}>"

                msg.attach(MIMEText(html_body, "html"))

                server.sendmail(sender_address, email, msg.as_string())
                log.info(f"Email sent to {name} ({email})")
                results["sent"] += 1
                results["details"].append({"email": email, "status": "sent"})

            except Exception as e:
                log.error(f"Failed to send to {email}: {e}")
                results["failed"] += 1
                results["details"].append(
                    {"email": email, "status": "failed", "error": str(e)}
                )

        server.quit()

    except smtplib.SMTPAuthenticationError as e:
        log.error(f"SMTP authentication failed: {e}")
        raise RuntimeError(
            "Gmail authentication failed. Check GMAIL_ADDRESS and GMAIL_APP_PASSWORD."
        ) from e
    except Exception as e:
        log.error(f"SMTP connection error: {e}")
        raise

    log.info(
        f"Email delivery complete: {results['sent']} sent, {results['failed']} failed"
    )
    return results
