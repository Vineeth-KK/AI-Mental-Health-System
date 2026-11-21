# tests/test_system.py
"""Comprehensive Test Suite for AI-Powered Dynamic Mental Health Monitoring System.
Covers:
- Registration, login, authentication route protection
- Check-in submission, retrieval, and validation
- Journal creation, retrieval, and sentiment analysis
- Strict user data isolation
- Deterministic distress score normalization
- All six agent node executions
- LangGraph workflow routing and safety-priority escalation
- Monitoring trend calculations and rolling averages
- ChromaDB indexing and RAG retrieval
- RAG-grounded recommendations
- AI Chat functionality
- Dashboard views and API time-series responses
- Health check and error handling
"""

import pytest
import os
import sys

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app import create_app
from extensions import db
from models.user import User
from models.checkin import CheckIn
from models.journal import JournalEntry
from models.alert import SafetyAlert
from config import Config
from agents.distress import calculate_distress_score, analyze_journal_sentiment, distress_agent_node
from agents.safety import evaluate_safety, evaluate_text_safety, safety_agent_node
from agents.monitoring import analyze_trends, monitoring_agent_node
from agents.recommendation import generate_recommendations, recommendation_agent_node
from agents.conversation import generate_llm_response, conversation_agent_node
from agents.orchestrator import orchestrator
from services.rag import rag_service

class TestConfig(Config):
    TESTING = True
    SQLALCHEMY_DATABASE_URI = "sqlite:///:memory:"
    WTF_CSRF_ENABLED = False
    SECRET_KEY = "test-secret-key"

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
    """Creates and returns a test user."""
    with app.app_context():
        user = User(username="testuser", email="test@example.com")
        user.set_password("securepassword123")
        db.session.add(user)
        db.session.commit()
        return user.id

# -------------------------------------------------------------
# 1. Deterministic Distress Score Normalization Tests
# -------------------------------------------------------------
def test_distress_score_best_case():
    """Best possible case: mood=10, anxiety=1, stress=1, isolation=1, sleep=10 -> score must be 0.0."""
    score, level, breakdown = calculate_distress_score(10, 1, 1, 1, 10)
    assert score == 0.0
    assert level == "low"
    assert breakdown["mood_component"] == 0.0
    assert breakdown["anxiety_component"] == 0.0

def test_distress_score_worst_case():
    """Worst possible case: mood=1, anxiety=10, stress=10, isolation=10, sleep=1 -> score must be 1.0."""
    score, level, breakdown = calculate_distress_score(1, 10, 10, 10, 1)
    assert score == 1.0
    assert level == "critical"
    assert breakdown["mood_component"] == 0.30
    assert breakdown["anxiety_component"] == 0.25

def test_distress_score_categories():
    """Verifies non-overlapping low, moderate, high, and critical categories."""
    # Low: < 0.35
    s_low, lvl_low, _ = calculate_distress_score(8, 2, 2, 2, 8)
    assert s_low < 0.35
    assert lvl_low == "low"

    # Moderate: 0.35 <= score < 0.65
    s_mod, lvl_mod, _ = calculate_distress_score(5, 5, 5, 5, 5)
    assert 0.35 <= s_mod < 0.65
    assert lvl_mod == "moderate"

    # High: 0.65 <= score < 0.85
    s_high, lvl_high, _ = calculate_distress_score(2, 8, 8, 7, 3)
    assert 0.65 <= s_high < 0.85
    assert lvl_high == "high"

    # Critical: >= 0.85
    s_crit, lvl_crit, _ = calculate_distress_score(1, 9, 9, 9, 2)
    assert s_crit >= 0.85
    assert lvl_crit == "critical"

# -------------------------------------------------------------
# 2. Safety Agent & Priority Escalation Tests
# -------------------------------------------------------------
def test_safety_agent_critical_keyword_trigger():
    """Urgent self-harm statements must trigger priority override and crisis helplines."""
    alert = evaluate_safety("I want to end my life, please help")
    assert alert["is_urgent"] is True
    assert alert["risk_level"] == "critical"
    assert "988" in alert["crisis_resources"]
    assert "14416" in alert["crisis_resources"]

def test_safety_agent_critical_score_trigger():
    """Critical numeric distress score (>= 0.85) must trigger safety intervention."""
    alert = evaluate_safety(text="I just feel completely drained", distress_score=0.88)
    assert alert["is_urgent"] is True
    assert alert["risk_level"] == "critical"
    assert "0.88" in alert["flag_reason"]

def test_safety_agent_safe_text():
    """Everyday text must be flagged as safe without priority override."""
    alert = evaluate_safety("Had a nice walk in the park today and felt peaceful")
    assert alert["is_urgent"] is False
    assert alert["risk_level"] == "safe"
    assert alert["crisis_resources"] is None

# -------------------------------------------------------------
# 3. Monitoring Agent Trend Analysis Tests
# -------------------------------------------------------------
def test_monitoring_agent_improving_trend():
    """Distress decreasing over time should be classified as improving."""
    checkins = [
        {"timestamp": "2026-11-01", "distress_score": 0.70, "mood": 3, "anxiety": 7, "stress": 7, "sleep_quality": 3, "isolation_score": 7},
        {"timestamp": "2026-11-02", "distress_score": 0.55, "mood": 5, "anxiety": 5, "stress": 5, "sleep_quality": 5, "isolation_score": 5},
        {"timestamp": "2026-11-03", "distress_score": 0.35, "mood": 7, "anxiety": 3, "stress": 3, "sleep_quality": 7, "isolation_score": 3},
        {"timestamp": "2026-11-04", "distress_score": 0.20, "mood": 9, "anxiety": 2, "stress": 2, "sleep_quality": 8, "isolation_score": 2},
    ]
    trends = analyze_trends(checkins)
    assert trends["trend"] == "improving"
    assert trends["slope"] < 0
    assert trends["rolling_averages"]["distress_score_7d"] > 0

def test_monitoring_agent_worsening_trend():
    """Distress increasing over time should be classified as worsening."""
    checkins = [
        {"timestamp": "2026-11-01", "distress_score": 0.15, "mood": 9, "anxiety": 2, "stress": 2, "sleep_quality": 8, "isolation_score": 2},
        {"timestamp": "2026-11-02", "distress_score": 0.40, "mood": 6, "anxiety": 5, "stress": 5, "sleep_quality": 6, "isolation_score": 4},
        {"timestamp": "2026-11-03", "distress_score": 0.65, "mood": 3, "anxiety": 7, "stress": 8, "sleep_quality": 4, "isolation_score": 7},
        {"timestamp": "2026-11-04", "distress_score": 0.85, "mood": 1, "anxiety": 9, "stress": 9, "sleep_quality": 2, "isolation_score": 9},
    ]
    trends = analyze_trends(checkins)
    assert trends["trend"] == "worsening"
    assert trends["slope"] > 0

# -------------------------------------------------------------
# 4. RAG Retrieval & Recommendation Agent Tests
# -------------------------------------------------------------
def test_rag_retrieval_and_recommendations():
    """Verifies that RAG queries return grounded snippets and recommendations are matched."""
    results = rag_service.query("breathing techniques for anxiety", n_results=2)
    assert len(results) > 0
    assert any("breath" in r["text"].lower() or "anxiety" in r["text"].lower() for r in results)

    exercises, rag_context = generate_recommendations(
        distress_level="high",
        current_checkin={"anxiety": 8, "stress": 7, "mood": 3, "sleep_quality": 4, "isolation_score": 6}
    )
    assert len(exercises) > 0
    assert any("grounding" in e["title"].lower() or "breathing" in e["title"].lower() for e in exercises)

# -------------------------------------------------------------
# 5. Conversation Agent Tests
# -------------------------------------------------------------
def test_conversation_agent_empathetic_fallback():
    """Verifies conversation agent generates empathetic, non-diagnostic response."""
    reply = generate_llm_response(
        messages=[{"role": "user", "content": "I am feeling extremely anxious about my upcoming exams"}],
        user_name="Alex"
    )
    assert "Alex" in reply
    assert len(reply) > 50
    # Must not diagnose
    assert "diagnos" not in reply.lower()

# -------------------------------------------------------------
# 6. LangGraph Multi-Agent Workflow Tests
# -------------------------------------------------------------
def test_orchestrator_checkin_flow():
    """Runs a complete checkin workflow through all nodes in LangGraph."""
    state = orchestrator.handle_checkin(
        user_id=1,
        user_name="Sam",
        checkin_data={"mood": 7, "anxiety": 3, "stress": 3, "sleep_quality": 8, "isolation_score": 2}
    )
    assert "distress_score" in state
    assert state["distress_level"] == "low"
    assert "trend_analysis" in state
    assert "recommendations" in state
    assert state["priority_override"] is False
    assert len(state["response"]) > 0

def test_orchestrator_safety_override_flow():
    """Verifies safety-priority routing overrides normal chat when suicidal ideation is detected."""
    state = orchestrator.handle_chat(
        user_id=1,
        user_name="Sam",
        messages=[{"role": "user", "content": "I feel like I want to commit suicide tonight"}]
    )
    assert state["priority_override"] is True
    assert "988" in state["response"]
    assert "14416" in state["response"]

# -------------------------------------------------------------
# 7. User Registration and Authentication Tests
# -------------------------------------------------------------
def test_user_registration(client, app):
    """User registration creates new user with hashed password."""
    resp = client.post("/register", data={
        "username": "newuser",
        "email": "newuser@example.com",
        "password": "password123",
        "confirm": "password123"
    }, follow_redirects=True)
    assert resp.status_code == 200

    with app.app_context():
        user = User.query.filter_by(username="newuser").first()
        assert user is not None
        assert user.check_password("password123") is True
        assert user.check_password("wrongpassword") is False

def test_user_login_and_logout(client, auth_user):
    """User can log in with valid credentials and log out."""
    login_resp = client.post("/login", data={
        "email": "test@example.com",
        "password": "securepassword123"
    }, follow_redirects=True)
    assert login_resp.status_code == 200
    assert b"Dashboard" in login_resp.data

    logout_resp = client.get("/logout", follow_redirects=True)
    assert logout_resp.status_code == 200
    assert b"Login" in logout_resp.data

def test_unauthenticated_routes_protected(client):
    """Protected routes must redirect unauthenticated users to login."""
    for path in ["/dashboard", "/checkin", "/journal", "/chat"]:
        resp = client.get(path)
        assert resp.status_code == 302
        assert "/login" in resp.headers["Location"]

# -------------------------------------------------------------
# 8. Check-In & Journal API with Isolation Tests
# -------------------------------------------------------------
def test_checkin_api_and_data_isolation(client, app):
    """Tests creating check-in and verifying data isolation between two distinct users."""
    with app.app_context():
        u1 = User(username="user1", email="u1@test.com")
        u1.set_password("pass123")
        u2 = User(username="user2", email="u2@test.com")
        u2.set_password("pass123")
        db.session.add_all([u1, u2])
        db.session.commit()
        u1_id = u1.id
        u2_id = u2.id

    # Log in as user 1
    client.post("/login", data={"email": "u1@test.com", "password": "pass123"})
    post_resp = client.post("/api/checkin", json={
        "mood": 8, "anxiety": 2, "stress": 3, "sleep_quality": 8, "isolation_score": 2
    })
    assert post_resp.status_code == 201
    json_data = post_resp.get_json()
    assert "distress_score" in json_data
    assert json_data["distress_level"] == "low"

    # User 1 should see 1 check-in
    get_resp1 = client.get("/api/checkins")
    assert len(get_resp1.get_json()) == 1

    # Log out, log in as user 2
    client.get("/logout")
    client.post("/login", data={"email": "u2@test.com", "password": "pass123"})
    get_resp2 = client.get("/api/checkins")
    # User 2 must see 0 check-ins (STRICT ISOLATION)
    assert len(get_resp2.get_json()) == 0

def test_journal_api_and_sentiment(client, auth_user):
    """Tests journal creation and sentiment response."""
    client.post("/login", data={"email": "test@example.com", "password": "securepassword123"})
    resp = client.post("/api/journal", json={
        "content": "Today was a really tough day. I felt exhausted and anxious about everything."
    })
    assert resp.status_code == 201
    data = resp.get_json()
    assert data["sentiment"]["sentiment"] == "distressed"

def test_chat_api_endpoint(client, auth_user):
    """Tests chat API endpoint response."""
    client.post("/login", data={"email": "test@example.com", "password": "securepassword123"})
    resp = client.post("/api/chat", json={
        "message": "Can you give me a grounding exercise for anxiety?"
    })
    assert resp.status_code == 200
    data = resp.get_json()
    assert len(data["response"]) > 0
    assert "recommendations" in data

def test_dashboard_api_endpoint(client, auth_user):
    """Tests dashboard API returns structured time-series data."""
    client.post("/login", data={"email": "test@example.com", "password": "securepassword123"})
    # Log one checkin
    client.post("/api/checkin", json={
        "mood": 7, "anxiety": 3, "stress": 3, "sleep_quality": 8, "isolation_score": 2
    })
    resp = client.get("/api/dashboard")
    assert resp.status_code == 200
    data = resp.get_json()
    assert "labels" in data
    assert "distress_score" in data
    assert len(data["distress_score"]) == 1

def test_health_endpoint(client):
    """Tests /health endpoint for deployment health monitoring."""
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["status"] == "healthy"
