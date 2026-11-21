# tests/test_journal_ai.py
"""Comprehensive verification of the Reflective Journal AI Analysis pipeline.
Tests 6 distinct user reflection scenarios and verifies resolution of:
1. Generic boilerplate feedback -> rich, personalized, context-aware analysis.
2. [object Object] sentiment display bug -> clean standardized JSON schema.
3. User boundary respect -> empty suggestions when user asks for no advice.
"""

import pytest
from app import create_app
from extensions import db
from models.user import User
from models.journal import JournalEntry
from agents.journal_agent import analyze_journal_entry, detect_advice_opt_out


from config import Config


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


@pytest.fixture
def auth_user(app):
    with app.app_context():
        u = User(username="sam", email="sam@example.com")
        u.set_password("password123")
        db.session.add(u)
        db.session.commit()
        return u


def test_scenario_1_positive_day():
    """Scenario 1: Positive day with gratitude and social connection."""
    text = "I had a wonderful day today. I spent time with my friends and felt happy."
    result = analyze_journal_entry(text, user_name="Sam")
    
    assert result["sentiment"] == "Positive"
    assert any(e in ["Gratitude", "Connection", "Joy", "Happy"] for e in result["emotions"])
    assert len(result["wellbeing_feedback"]) > 40
    # Must NOT be the old generic message
    assert "Your journal reflection has been safely recorded" not in result["wellbeing_feedback"]
    assert len(result["suggestions"]) > 0


def test_scenario_2_academic_stress():
    """Scenario 2: Academic exam stress and concentration difficulty."""
    text = "I have been feeling stressed because of my upcoming examinations. I am unable to concentrate."
    result = analyze_journal_entry(text, user_name="Sam")
    
    assert result["sentiment"] == "Distressed"
    assert any("Stress" in e or "Fatigue" in e or "Academic" in e or "Exam" in e for e in result["emotions"])
    assert "exam" in result["key_observation"].lower() or "concentrat" in result["key_observation"].lower()
    assert len(result["suggestions"]) > 0


def test_scenario_3_loneliness():
    """Scenario 3: Feeling lonely despite being surrounded by others."""
    text = "I feel lonely even when I am surrounded by people."
    result = analyze_journal_entry(text, user_name="Sam")
    
    assert result["sentiment"] in ["Distressed", "Reflective"]
    assert any(any(k in e.lower() for k in ["lonel", "disconnect", "isolat", "frustrat"]) for e in result["emotions"])
    assert len(result["wellbeing_feedback"]) > 50


def test_scenario_4_ordinary_day():
    """Scenario 4: Ordinary day with no major positive or negative events."""
    text = "Today was an ordinary day. Nothing particularly good or bad happened."
    result = analyze_journal_entry(text, user_name="Sam")
    
    assert result["sentiment"] in ["Neutral", "Reflective"]
    assert any(any(k in e.lower() for k in ["calm", "routine", "neutral", "content", "peace", "ordinary"]) for e in result["emotions"])
    assert len(result["wellbeing_feedback"]) > 40


def test_scenario_5_frustration():
    """Scenario 5: Frustration when plans fail to materialize."""
    text = "I am frustrated because things are not going according to my plans."
    result = analyze_journal_entry(text, user_name="Sam")
    
    assert result["sentiment"] in ["Distressed", "Overwhelmed", "Reflective"]
    assert any(any(k in e.lower() for k in ["frustrat", "disappoint"]) for e in result["emotions"])
    assert len(result["suggestions"]) > 0


def test_scenario_6_no_advice_requested():
    """Scenario 6: User explicitly requests NO advice / just documenting thoughts."""
    text = "I don't want advice right now. I just wanted to write down what happened today."
    assert detect_advice_opt_out(text) is True
    
    result = analyze_journal_entry(text, user_name="Sam")
    assert result["sentiment"] in ["Reflective", "Neutral", "Distressed"]
    # CRITICAL: Suggestions must be empty to respect user's boundary
    assert result["suggestions"] == []
    assert len(result["wellbeing_feedback"]) > 40


def test_api_journal_endpoint_and_sentiment_schema(client, auth_user):
    """End-to-end API test verifying data schema, eliminating [object Object] bug."""
    client.post("/login", data={"email": "sam@example.com", "password": "password123"})
    
    resp = client.post("/api/journal", json={
        "content": "I had a wonderful day today. I spent time with my friends and felt happy."
    })
    
    assert resp.status_code == 201
    data = resp.get_json()
    
    assert data["success"] is True
    assert "analysis" in data
    assert isinstance(data["analysis"], dict)
    
    # Sentiment label must be a readable string (never "[object Object]")
    assert isinstance(data["sentiment_label"], str)
    assert data["sentiment_label"] != "[object Object]"
    assert data["analysis"]["sentiment"] != "[object Object]"
    
    # Sentiment dict preserved for test_system.py backward compatibility
    assert isinstance(data["sentiment"], dict)
    assert "sentiment" in data["sentiment"]
    
    # Check that feedback is non-generic
    assert len(data["response"]) > 40
    assert "Your journal reflection has been safely recorded" not in data["response"]
