# models/alert.py
"""Safety alert model for risk notifications."""
from extensions import db
from datetime import datetime

class SafetyAlert(db.Model):
    __tablename__ = "alerts"
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False)
    message = db.Column(db.String(255), nullable=False)
    risk_level = db.Column(db.String(32), default="critical") # moderate, high, critical
    action_taken = db.Column(db.String(255), nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    resolved = db.Column(db.Boolean, default=False)
    status = db.Column(db.String(32), default="unresolved", nullable=False) # unresolved, in_review, resolved
    counselor_notes = db.Column(db.Text, nullable=True)
    reviewed_by = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=True)
    reviewed_at = db.Column(db.DateTime, nullable=True)

    user = db.relationship("User", foreign_keys=[user_id], backref=db.backref("alerts", lazy=True))
    reviewer = db.relationship("User", foreign_keys=[reviewed_by], backref=db.backref("reviewed_alerts", lazy=True))

    def __repr__(self):
        return f"<Alert u{self.user_id} resolved={self.resolved}>"
