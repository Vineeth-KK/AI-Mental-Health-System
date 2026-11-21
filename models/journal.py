# models/journal.py
"""Journal entry model for optional user reflections."""
from extensions import db
from datetime import datetime

class JournalEntry(db.Model):
    __tablename__ = "journals"
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False)
    content = db.Column(db.Text, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    user = db.relationship("User", backref=db.backref("journals", lazy=True))

    def __repr__(self):
        return f"<Journal u{self.user_id} {self.created_at}>"
