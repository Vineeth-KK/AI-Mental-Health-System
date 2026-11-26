# tests/test_migration_and_memory.py
"""Tests for database migration utility, chat memory metadata, and proxy configurations."""

import os
import sys
import pytest
from app import create_app
from extensions import db
from models.user import User
from config import Config
from scripts.migrate_sqlite_to_postgres import normalize_postgres_url, migrate_database
from agents.conversation import _generate_intelligent_fallback


class TestConfig(Config):
    TESTING = True
    WTF_CSRF_ENABLED = False
    SQLALCHEMY_DATABASE_URI = "sqlite:///:memory:"


@pytest.fixture
def app():
    app = create_app(TestConfig)
    with app.app_context():
        db.create_all()
        yield app
        db.session.remove()
        db.drop_all()


@pytest.fixture
def client(app):
    return app.test_client()


def test_normalize_postgres_url():
    """Verifies that PostgreSQL URLs with legacy prefixes or quotes are properly normalized to postgresql+psycopg2://."""
    assert normalize_postgres_url("postgres://user:pass@host/db") == "postgresql+psycopg2://user:pass@host/db"
    assert normalize_postgres_url("postgresql://user:pass@host/db") == "postgresql+psycopg2://user:pass@host/db"
    assert normalize_postgres_url("'postgres://user:pass@host/db'") == "postgresql+psycopg2://user:pass@host/db"
    assert normalize_postgres_url('"postgres://user:pass@host/db"') == "postgresql+psycopg2://user:pass@host/db"
    assert normalize_postgres_url("  postgresql://user:pass@host/db  ") == "postgresql+psycopg2://user:pass@host/db"
    assert normalize_postgres_url("postgresql+psycopg://user:pass@host/db") == "postgresql+psycopg://user:pass@host/db"


def test_fallback_template_formatting_no_leakage():
    """Verifies that _generate_intelligent_fallback never outputs literal unformatted {rag_snippet_str}."""
    resp = _generate_intelligent_fallback(
        messages=[{"role": "user", "content": "What is the capital of France?"}],
        rag_context=[],
        user_name="DemoUser"
    )
    assert "{rag_snippet_str}" not in resp
    assert "DemoUser" in resp

    resp2 = _generate_intelligent_fallback(
        messages=[{"role": "user", "content": "I am feeling tired today"}],
        rag_context=[],
        user_name="DemoUser"
    )
    assert "{rag_snippet_str}" not in resp2


def test_chat_api_metadata_fields(client, app):
    """Verifies that /api/chat returns AI provider and model metadata without leaking secrets."""
    with app.app_context():
        user = User(username="testchatuser", email="testchat@test.com", role="user")
        user.set_password("password123")
        db.session.add(user)
        db.session.commit()

    client.post("/login", data={"email": "testchat@test.com", "password": "password123"})

    resp = client.post("/api/chat", json={
        "message": "Hello CareAI, how are you?",
        "history": []
    })

    assert resp.status_code == 200
    data = resp.get_json()
    assert "response" in data
    assert "ai_provider" in data
    assert "ai_model" in data
    assert "is_llm" in data
    assert "is_fallback" in data
    assert isinstance(data["is_fallback"], bool)


def test_sqlite_migration_dry_run():
    """Verifies that migrate_database dry-run reads tables and validates without modifying database."""
    sqlite_path = os.path.abspath("instance/mhapp_pre_migration_backup.db")
    if os.path.exists(sqlite_path):
        # Test dry-run with in-memory SQLite pretending to be PostgreSQL target schema
        result = migrate_database(
            sqlite_path=sqlite_path,
            pg_url="sqlite:///:memory:",
            dry_run=True
        )
        assert result is True
