# services/notifier.py
"""Counselor Alert Notification Service.
Dispatches in-app and optional email notifications to authorized counselors when urgent safety indicators are detected.
Protects patient privacy by never transmitting raw journal reflections over email.
"""

import os
import smtplib
import logging
from email.mime.text import MIMEText
from datetime import datetime, timezone
from extensions import db
from models.notification import CounselorNotification
from models.user import User

logger = logging.getLogger(__name__)


def is_smtp_configured() -> bool:
    """Checks whether external SMTP settings are configured in environment variables."""
    return bool(
        os.getenv("SMTP_HOST") and
        os.getenv("SMTP_USER") and
        os.getenv("SMTP_PASSWORD")
    )


def dispatch_counselor_alert(
    alert_id: int,
    user_id: int,
    risk_level: str = "critical",
    reason: str = "Urgent safety risk detected"
) -> int:
    """Dispatches notifications to all authorized counselors for a given safety alert.
    Returns the count of successfully created notifications.
    """
    if not alert_id:
        return 0

    try:
        # Check if already dispatched to prevent duplicate spam
        existing = CounselorNotification.query.filter_by(alert_id=alert_id).first()
        if existing:
            logger.info(f"Alert #{alert_id} already has dispatched notifications. Skipping duplicate dispatch.")
            return 0

        # Query all active counselors and admins
        counselors = User.query.filter(
            (User.role.in_(["counselor", "admin"])) | (User.is_admin == True)
        ).all()

        if not counselors:
            logger.warning("No counselors or administrators registered in the system to receive alerts.")
            # Record a system notification record so the alert is not lost
            fallback_notif = CounselorNotification(
                alert_id=alert_id,
                channel="in_app",
                status="delivered",
                subject=f"Urgent Risk Alert: Patient #{user_id}",
                error_message="No specific counselor assigned. Broadcast alert."
            )
            db.session.add(fallback_notif)
            db.session.commit()
            return 1

        smtp_ready = is_smtp_configured()
        dispatched_count = 0

        for c in counselors:
            # 1. In-App Notification Record
            in_app_notif = CounselorNotification(
                alert_id=alert_id,
                counselor_id=c.id,
                channel="in_app",
                recipient_email=c.email,
                status="delivered",
                subject=f"Urgent Risk Indicator: Patient #{user_id} ({risk_level.upper()})",
                dispatched_at=datetime.now(timezone.utc)
            )
            db.session.add(in_app_notif)
            dispatched_count += 1

            # 2. Optional SMTP Email Notification (Strict Privacy: No raw diary texts)
            if smtp_ready:
                try:
                    host = os.getenv("SMTP_HOST")
                    port = int(os.getenv("SMTP_PORT", "587"))
                    sender = os.getenv("SMTP_FROM", os.getenv("SMTP_USER"))
                    recipient = c.email

                    body = (
                        f"CAREAI CLINICAL SAFETY ALERT\n\n"
                        f"An urgent risk indicator has been flagged for patient review:\n"
                        f"- Patient ID: #{user_id}\n"
                        f"- Risk Level: {risk_level.upper()}\n"
                        f"- Trigger Category: {reason}\n"
                        f"- Timestamp: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}\n\n"
                        f"Please sign in to the authorized CareAI Counselor Portal to review this case.\n\n"
                        f"[PRIVACY NOTICE]: Raw personal journal texts and transcripts are withheld from email notifications in accordance with patient data protection protocols."
                    )

                    msg = MIMEText(body)
                    msg["Subject"] = f"[CareAI Alert] Urgent Patient Risk Indicator: #{user_id}"
                    msg["From"] = sender
                    msg["To"] = recipient

                    with smtplib.SMTP(host, port, timeout=10) as server:
                        if os.getenv("SMTP_USE_TLS", "true").lower() == "true":
                            server.starttls()
                        server.login(os.getenv("SMTP_USER"), os.getenv("SMTP_PASSWORD"))
                        server.sendmail(sender, [recipient], msg.as_string())

                    email_notif = CounselorNotification(
                        alert_id=alert_id,
                        counselor_id=c.id,
                        channel="email",
                        recipient_email=c.email,
                        status="delivered",
                        subject=msg["Subject"],
                        dispatched_at=datetime.now(timezone.utc)
                    )
                    db.session.add(email_notif)
                except Exception as mail_err:
                    logger.warning(f"Failed to dispatch external email to {c.email}: {mail_err}")
                    email_notif = CounselorNotification(
                        alert_id=alert_id,
                        counselor_id=c.id,
                        channel="email",
                        recipient_email=c.email,
                        status="failed",
                        subject=f"[CareAI Alert] Urgent Patient Risk Indicator: #{user_id}",
                        error_message=str(mail_err)[:250],
                        dispatched_at=datetime.now(timezone.utc)
                    )
                    db.session.add(email_notif)
            else:
                # Log that external email is not configured (graceful fallback)
                logger.info(f"External SMTP not configured. Recorded in-app notification for counselor {c.email}.")

        db.session.commit()
        return dispatched_count
    except Exception as e:
        db.session.rollback()
        logger.error(f"Error dispatching counselor alerts: {e}")
        return 0
