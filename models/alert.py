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

    user = db.relationship("User", backref=db.backref("alerts", lazy=True))

    def __repr__(self):
        return f"<Alert u{self.user_id} resolved={self.resolved}>"
