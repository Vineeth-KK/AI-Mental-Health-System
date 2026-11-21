# models/checkin.py
"""Check-in model for daily mental health entries."""
from extensions import db
from datetime import datetime

class CheckIn(db.Model):
    __tablename__ = "checkins"
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False)
    mood = db.Column(db.Integer, nullable=False)            # 1-5 scale
    anxiety = db.Column(db.Integer, nullable=False)         # 1-5
    stress = db.Column(db.Integer, nullable=False)          # 1-5
    sleep_quality = db.Column(db.Integer, nullable=False)  # 1-10
    isolation_score = db.Column(db.Integer, nullable=False)  # 1-10
    distress_score = db.Column(db.Float, nullable=True)     # 0.0 - 1.0
    distress_level = db.Column(db.String(20), nullable=True) # low, moderate, high, critical
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

    user = db.relationship("User", backref=db.backref("checkins", lazy=True))

    def __repr__(self):
        return f"<CheckIn u{self.user_id} {self.timestamp}>"
