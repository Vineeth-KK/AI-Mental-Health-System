# models/case_note.py
"""Case summary and notes model for authorized human counselors."""
from extensions import db
from datetime import datetime

class CaseSummary(db.Model):
    __tablename__ = "case_summaries"
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False)
    counselor_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=True)
    title = db.Column(db.String(255), default="Longitudinal Wellbeing & Trauma Case Synthesis")
    overview = db.Column(db.Text, nullable=True)
    reported_concerns = db.Column(db.Text, nullable=True)
    recurring_patterns = db.Column(db.Text, nullable=True)
    changes_over_time = db.Column(db.Text, nullable=True)
    historical_events = db.Column(db.Text, nullable=True)
    risk_indicators = db.Column(db.Text, nullable=True)
    recommended_review = db.Column(db.Text, nullable=True)
    full_content = db.Column(db.Text, nullable=False)
    counselor_edits = db.Column(db.Text, nullable=True)
    status = db.Column(db.String(32), default="draft", nullable=False)  # draft, reviewed, approved
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    patient = db.relationship("User", foreign_keys=[user_id], backref=db.backref("case_summaries", lazy=True))
    counselor = db.relationship("User", foreign_keys=[counselor_id], backref=db.backref("authored_summaries", lazy=True))

    def to_dict(self):
        return {
            "id": self.id,
            "user_id": self.user_id,
            "counselor_id": self.counselor_id,
            "title": self.title,
            "overview": self.overview,
            "reported_concerns": self.reported_concerns,
            "recurring_patterns": self.recurring_patterns,
            "changes_over_time": self.changes_over_time,
            "historical_events": self.historical_events,
            "risk_indicators": self.risk_indicators,
            "recommended_review": self.recommended_review,
            "full_content": self.full_content,
            "counselor_edits": self.counselor_edits,
            "status": self.status,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None
        }

    def __repr__(self):
        return f"<CaseSummary #{self.id} patient_u{self.user_id} status={self.status}>"
