# models/chat.py
"""Chat message model for conversation history persistence."""
from extensions import db
from datetime import datetime

class ChatMessage(db.Model):
    __tablename__ = "chat_messages"
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False)
    role = db.Column(db.String(16), nullable=False)  # "user" or "assistant"
    content = db.Column(db.Text, nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow, index=True)
    is_safety_override = db.Column(db.Boolean, default=False)

    user = db.relationship("User", backref=db.backref("chat_messages", lazy=True, cascade="all, delete-orphan"))

    def to_dict(self):
        return {
            "id": self.id,
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "is_safety_override": self.is_safety_override
        }

    def __repr__(self):
        return f"<ChatMessage u{self.user_id} {self.role}: {self.content[:30]}>"
