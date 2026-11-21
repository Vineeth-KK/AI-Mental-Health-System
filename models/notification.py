# models/notification.py
"""Counselor notification model for tracking alert dispatches."""
from extensions import db
from datetime import datetime

class CounselorNotification(db.Model):
    __tablename__ = "counselor_notifications"
    id = db.Column(db.Integer, primary_key=True)
    alert_id = db.Column(db.Integer, db.ForeignKey("alerts.id"), nullable=False)
    counselor_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=True)
    channel = db.Column(db.String(32), default="in_app", nullable=False)  # in_app, email
    recipient_email = db.Column(db.String(120), nullable=True)
    status = db.Column(db.String(32), default="delivered", nullable=False)  # pending, delivered, failed, not_configured
    subject = db.Column(db.String(255), nullable=True)
    error_message = db.Column(db.Text, nullable=True)
    dispatched_at = db.Column(db.DateTime, default=datetime.utcnow)

    alert = db.relationship("SafetyAlert", backref=db.backref("notifications", lazy=True, cascade="all, delete-orphan"))
    counselor = db.relationship("User", backref=db.backref("received_notifications", lazy=True))

    def to_dict(self):
        return {
            "id": self.id,
            "alert_id": self.alert_id,
            "counselor_id": self.counselor_id,
            "channel": self.channel,
            "recipient_email": self.recipient_email,
            "status": self.status,
            "subject": self.subject,
            "dispatched_at": self.dispatched_at.isoformat() if self.dispatched_at else None
        }

    def __repr__(self):
        return f"<CounselorNotification #{self.id} alert={self.alert_id} status={self.status}>"
