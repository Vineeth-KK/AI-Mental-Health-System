# models/user.py
"""User model for the mental health system."""
from extensions import db
from werkzeug.security import generate_password_hash, check_password_hash
from flask_login import UserMixin
from datetime import datetime

class User(UserMixin, db.Model):
    __tablename__ = "users"
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(64), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(256), nullable=False)
    registered_on = db.Column(db.DateTime, default=datetime.utcnow)
    is_admin = db.Column(db.Boolean, default=False)
    role = db.Column(db.String(20), default="user", nullable=False)  # "user", "counselor", "admin"

    def set_password(self, password: str) -> None:
        self.password_hash = generate_password_hash(password)

    def check_password(self, password: str) -> bool:
        if not self.password_hash or not password:
            return False
        return check_password_hash(self.password_hash, password)

    def is_counselor(self) -> bool:
        """Returns True if the user is an authorized counselor or admin."""
        return self.role in ["counselor", "admin"] or self.is_admin

    def is_counselor_or_admin(self) -> bool:
        """Alias for role-based counselor access check."""
        return self.is_counselor()

    def __repr__(self):
        return f"<User {self.username}>"

