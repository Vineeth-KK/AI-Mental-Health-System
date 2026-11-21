# services/db_migration.py
"""Additive schema migration utility.
Safely and idempotently verifies database tables and columns exist without deleting or altering existing data.
Works transparently on both SQLite and MySQL.
"""

import logging
from sqlalchemy import inspect, text
from extensions import db

logger = logging.getLogger(__name__)

def run_additive_migrations(app):
    """Safely checks and adds any missing columns or tables needed for new features."""
    with app.app_context():
        # First ensure all declared tables exist
        db.create_all()

        inspector = inspect(db.engine)
        existing_tables = inspector.get_table_names()

        with db.engine.connect() as conn:
            # 1. Check chat_messages for conversational distress columns
            if "chat_messages" in existing_tables:
                chat_cols = [c["name"] for c in inspector.get_columns("chat_messages")]
                if "distress_score" not in chat_cols:
                    conn.execute(text("ALTER TABLE chat_messages ADD COLUMN distress_score FLOAT"))
                    logger.info("Added distress_score column to chat_messages")
                if "sentiment" not in chat_cols:
                    conn.execute(text("ALTER TABLE chat_messages ADD COLUMN sentiment VARCHAR(32)"))
                    logger.info("Added sentiment column to chat_messages")
                if "detected_emotions" not in chat_cols:
                    conn.execute(text("ALTER TABLE chat_messages ADD COLUMN detected_emotions VARCHAR(255)"))
                    logger.info("Added detected_emotions column to chat_messages")

            # 2. Check users for role column
            if "users" in existing_tables:
                user_cols = [c["name"] for c in inspector.get_columns("users")]
                if "role" not in user_cols:
                    conn.execute(text("ALTER TABLE users ADD COLUMN role VARCHAR(20) DEFAULT 'user'"))
                    logger.info("Added role column to users")

            # 3. Check alerts for counselor review columns
            if "alerts" in existing_tables:
                alert_cols = [c["name"] for c in inspector.get_columns("alerts")]
                if "status" not in alert_cols:
                    conn.execute(text("ALTER TABLE alerts ADD COLUMN status VARCHAR(32) DEFAULT 'unresolved'"))
                    logger.info("Added status column to alerts")
                if "counselor_notes" not in alert_cols:
                    conn.execute(text("ALTER TABLE alerts ADD COLUMN counselor_notes TEXT"))
                    logger.info("Added counselor_notes column to alerts")
                if "reviewed_by" not in alert_cols:
                    conn.execute(text("ALTER TABLE alerts ADD COLUMN reviewed_by INTEGER"))
                    logger.info("Added reviewed_by column to alerts")
                if "reviewed_at" not in alert_cols:
                    conn.execute(text("ALTER TABLE alerts ADD COLUMN reviewed_at DATETIME"))
                    logger.info("Added reviewed_at column to alerts")

            conn.commit()
            logger.info("Additive database migrations verified successfully.")

        # Auto-seed default demo accounts on empty database in production/dev (e.g. Render fresh deploy)
        if not app.config.get("TESTING"):
            try:
                from models.user import User
                if User.query.first() is None:
                    demo_user = User(username="DemoUser", email="demo@example.com", role="user")
                    demo_user.set_password("password123")
                    counselor_user = User(username="CounselorSarah", email="counselor@careai.com", role="counselor")
                    counselor_user.set_password("password123")
                    admin_user = User(username="AdminCareAI", email="admin@careai.com", role="admin", is_admin=True)
                    admin_user.set_password("password123")
                    db.session.add_all([demo_user, counselor_user, admin_user])
                    db.session.commit()
                    logger.info("Automatically seeded default demo accounts for fresh database.")
            except Exception as seed_err:
                logger.warning(f"Initial user auto-seeding skipped: {seed_err}")
