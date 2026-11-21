# extensions.py
"""Initialize Flask extensions used across the app."""

from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager
from flask_wtf import CSRFProtect

# SQLAlchemy instance (will be linked to app in app factory)
db = SQLAlchemy()

# Login manager for user session handling
login_manager = LoginManager()
login_manager.login_view = "auth.login"

# CSRF protection for forms and API endpoints
csrf = CSRFProtect()
