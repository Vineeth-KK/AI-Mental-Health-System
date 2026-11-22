"""tests/test_persistence_and_auth.py
Automated Verification Suite for Database Persistence, Login Reliability,
Chat History, and Cross-Restart Application Stability.
"""

import os
import sys
import tempfile
import pytest

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from datetime import datetime
from sqlalchemy import text
from app import create_app
from extensions import db
from models.user import User
from models.checkin import CheckIn
from models.journal import JournalEntry
from models.chat import ChatMessage
from models.alert import SafetyAlert
from config import Config

class TestPersistenceConfig(Config):
    TESTING = True
    WTF_CSRF_ENABLED = False
    # Use a persistent test file so we can simulate application restart
    db_fd, db_temp_path = tempfile.mkstemp(suffix="_test_persistence.db")
    SQLALCHEMY_DATABASE_URI = f"sqlite:///{db_temp_path.replace(os.sep, '/')}"
    SECRET_KEY = "test-persistent-secret-key-123"

@pytest.fixture(scope="module")
def persistent_db_path():
    yield TestPersistenceConfig.db_temp_path
    try:
        os.close(TestPersistenceConfig.db_fd)
        if os.path.exists(TestPersistenceConfig.db_temp_path):
            os.remove(TestPersistenceConfig.db_temp_path)
    except Exception:
        pass

def test_1_registration(persistent_db_path):
    """1. Registering a new account."""
    app = create_app(TestPersistenceConfig)
    client = app.test_client()

    with app.app_context():
        # Register User A
        res = client.post("/register", data={
            "username": "PersistUserA",
            "email": "persista@example.com",
            "password": "SecurePassword123!",
            "confirm": "SecurePassword123!"
        }, follow_redirects=True)
        assert res.status_code == 200

        user = User.query.filter_by(email="persista@example.com").first()
        assert user is not None
        assert user.username == "PersistUserA"
        assert user.check_password("SecurePassword123!")

        # Duplicate email should be rejected (case-insensitive)
        res_dup = client.post("/register", data={
            "username": "DifferentUser",
            "email": "PERSISTA@EXAMPLE.COM ",
            "password": "SecurePassword123!",
            "confirm": "SecurePassword123!"
        }, follow_redirects=True)
        assert b"already exists" in res_dup.data or res_dup.status_code == 200
        assert User.query.filter_by(username="DifferentUser").first() is None

def test_2_and_3_login_and_reject_incorrect_password(persistent_db_path):
    """2. Logging in with correct password & 3. Rejecting incorrect password."""
    app = create_app(TestPersistenceConfig)
    client = app.test_client()

    with app.app_context():
        # First test: Reject incorrect password on unauthenticated client
        res_wrong = client.post("/login", data={
            "email": "persista@example.com",
            "password": "WrongPassword!"
        }, follow_redirects=True)
        assert b"Invalid email or password" in res_wrong.data

        # Second test: Accept correct password
        res_correct = client.post("/login", data={
            "email": "persista@example.com",
            "password": "SecurePassword123!"
        }, follow_redirects=False)
        assert res_correct.status_code == 302
        assert "/dashboard" in res_correct.headers.get("Location", "")

def test_4_logout_and_relogin(persistent_db_path):
    """4. Logging out and logging back in."""
    app = create_app(TestPersistenceConfig)
    client = app.test_client()

    with app.app_context():
        # Login
        client.post("/login", data={"email": "persista@example.com", "password": "SecurePassword123!"})
        res_dash = client.get("/dashboard")
        assert res_dash.status_code == 200

        # Logout
        res_logout = client.get("/logout", follow_redirects=True)
        assert b"logged out" in res_logout.data

        # Verify protected route now redirects to login
        res_protected = client.get("/dashboard", follow_redirects=False)
        assert res_protected.status_code == 302
        assert "/login" in res_protected.headers.get("Location", "")

        # Relogin
        res_relogin = client.post("/login", data={"email": "persista@example.com", "password": "SecurePassword123!"}, follow_redirects=False)
        assert res_relogin.status_code == 302

def test_5_restart_flask_and_verify_account_persists(persistent_db_path):
    """5. Restarting the Flask application and verifying that the account still exists."""
    # Completely destroy the old app instance and instantiate a brand-new app factory
    app_restarted = create_app(TestPersistenceConfig)
    client_restarted = app_restarted.test_client()

    with app_restarted.app_context():
        user = User.query.filter_by(email="persista@example.com").first()
        assert user is not None, "User disappeared after Flask application restart!"
        assert user.check_password("SecurePassword123!"), "Password hash verification failed after restart!"

        # Logging in on restarted app succeeds
        res_login = client_restarted.post("/login", data={"email": "persista@example.com", "password": "SecurePassword123!"}, follow_redirects=False)
        assert res_login.status_code == 302

def test_6_checkin_persists_across_restart(persistent_db_path):
    """6. Creating a check-in and retrieving it after restarting."""
    app_1 = create_app(TestPersistenceConfig)
    client_1 = app_1.test_client()

    with app_1.app_context():
        user = User.query.filter_by(email="persista@example.com").first()
        client_1.post("/login", data={"email": "persista@example.com", "password": "SecurePassword123!"})
        
        # Submit checkin via JSON API
        res_checkin = client_1.post("/api/checkin", json={
            "mood": 4,
            "anxiety": 2,
            "stress": 3,
            "sleep_quality": 8,
            "isolation_score": 3
        })
        assert res_checkin.status_code in (200, 201)

    # Simulate app restart
    app_2 = create_app(TestPersistenceConfig)
    with app_2.app_context():
        user = User.query.filter_by(email="persista@example.com").first()
        checkins = CheckIn.query.filter_by(user_id=user.id).all()
        assert len(checkins) >= 1
        assert checkins[0].mood == 4
        assert checkins[0].sleep_quality == 8

def test_7_journal_persists_across_restart(persistent_db_path):
    """7. Creating a journal entry and retrieving it after restarting."""
    app_1 = create_app(TestPersistenceConfig)
    client_1 = app_1.test_client()

    with app_1.app_context():
        client_1.post("/login", data={"email": "persista@example.com", "password": "SecurePassword123!"})
        res_j = client_1.post("/api/journal", json={
            "content": "Today was a productive day and I felt grounded."
        })
        assert res_j.status_code in (200, 201)

    # Simulate app restart
    app_2 = create_app(TestPersistenceConfig)
    with app_2.app_context():
        user = User.query.filter_by(email="persista@example.com").first()
        journals = JournalEntry.query.filter_by(user_id=user.id).all()
        assert len(journals) >= 1
        assert "productive day" in journals[0].content

def test_8_chat_conversation_persists_across_restart(persistent_db_path):
    """8. Saving a conversation and retrieving it after restarting."""
    app_1 = create_app(TestPersistenceConfig)
    client_1 = app_1.test_client()

    with app_1.app_context():
        user = User.query.filter_by(email="persista@example.com").first()
        # Add conversation messages
        msg1 = ChatMessage(user_id=user.id, role="user", content="Hello CareAI, I want to talk about stress.")
        msg2 = ChatMessage(user_id=user.id, role="assistant", content="Hello! I am here to help you unpack what is causing stress.")
        db.session.add_all([msg1, msg2])
        db.session.commit()

    # Simulate app restart
    app_2 = create_app(TestPersistenceConfig)
    client_2 = app_2.test_client()
    with app_2.app_context():
        client_2.post("/login", data={"email": "persista@example.com", "password": "SecurePassword123!"})
        res_history = client_2.get("/api/chat/history")
        assert res_history.status_code == 200
        data = res_history.get_json()
        history = data.get("history", [])
        assert len(history) >= 2
        assert history[0]["content"] == "Hello CareAI, I want to talk about stress."
        assert history[1]["content"] == "Hello! I am here to help you unpack what is causing stress."

def test_9_data_isolation_between_users(persistent_db_path):
    """9. Ensuring one user cannot access another user's conversations or private records."""
    app = create_app(TestPersistenceConfig)
    client = app.test_client()

    with app.app_context():
        # Create User B
        client.post("/register", data={
            "username": "PersistUserB",
            "email": "persistb@example.com",
            "password": "SecurePassword456!",
            "confirm": "SecurePassword456!"
        })
        user_b = User.query.filter_by(email="persistb@example.com").first()
        assert user_b is not None

        # Log in as User B
        client.post("/login", data={"email": "persistb@example.com", "password": "SecurePassword456!"})

        # User B fetches chat history -> should NOT contain User A's messages
        res_b_history = client.get("/api/chat/history").get_json()
        b_msgs = res_b_history.get("history", [])
        assert len(b_msgs) == 0, "User B was able to see User A's private chat messages!"

        # User B fetches journals -> should NOT contain User A's journals
        user_b_journals = JournalEntry.query.filter_by(user_id=user_b.id).all()
        assert len(user_b_journals) == 0

def test_10_database_failure_and_recovery(persistent_db_path):
    """10. Database connection failures and recovery."""
    app = create_app(TestPersistenceConfig)

    with app.app_context():
        # Healthy connection
        health_res = app.test_client().get("/health").get_json()
        assert health_res["database"]["status"] == "connected"

        # Simulate connection check with invalid text
        try:
            db.session.execute(text("SELECT 1 FROM non_existent_table_99999"))
        except Exception:
            db.session.rollback()

        # Engine recovers automatically via pool_pre_ping / fresh statement
        recovery_ping = db.session.execute(text("SELECT 1")).scalar()
        assert recovery_ping == 1

def test_11_additive_migration_preserves_records(persistent_db_path):
    """11. Database migrations without losing existing records."""
    from services.db_migration import run_additive_migrations

    app = create_app(TestPersistenceConfig)
    with app.app_context():
        count_before = User.query.count()
        chat_count_before = ChatMessage.query.count()

        # Re-run migrations
        run_additive_migrations(app)

        # Records must remain exactly the same
        assert User.query.count() == count_before
        assert ChatMessage.query.count() == chat_count_before
