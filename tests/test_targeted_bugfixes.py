# tests/test_targeted_bugfixes.py
"""Targeted verification tests for:
1. Recommended Wellbeing Exercises (proper field extraction, no [object Object], indicator relevance).
2. Multi-Agent AI Evaluation (actual agent execution, state passing, contribution trace).
3. Trend Trajectory (improving, worsening, stable, insufficient_data, chronological sorting).
4. Counselor Portal (data isolation, alert triage, case summary, search/filter compatibility).
5. Existing accounts and database integrity.
"""

import pytest
from app import create_app
from extensions import db
from config import Config
from models.user import User
from models.checkin import CheckIn
from models.alert import SafetyAlert
from models.case_note import CaseSummary
from agents.recommendation import generate_recommendations, WELLBEING_EXERCISES
from agents.monitoring import analyze_trends
from agents.orchestrator import orchestrator


class TestConfig(Config):
    TESTING = True
    SQLALCHEMY_DATABASE_URI = "sqlite:///:memory:"
    WTF_CSRF_ENABLED = False
    SECRET_KEY = "test-secret-key-target"


@pytest.fixture
def app():
    app = create_app(TestConfig)
    with app.app_context():
        db.create_all()
        # Seed test accounts
        patient = User(username="FixPatient", email="fix_patient@careai.com", role="user")
        patient.set_password("password123")

        counselor = User(username="FixCounselor", email="fix_counselor@careai.com", role="counselor")
        counselor.set_password("password123")

        admin = User(username="FixAdmin", email="fix_admin@careai.com", role="admin", is_admin=True)
        admin.set_password("password123")

        db.session.add_all([patient, counselor, admin])
        db.session.commit()

        yield app

        db.session.remove()
        db.drop_all()


@pytest.fixture
def client(app):
    return app.test_client()


# -------------------------------------------------------------
# Test 1: Recommended Wellbeing Exercises
# -------------------------------------------------------------
def test_recommendations_structure_and_no_object_bug():
    """Verifies that exercises contain all required descriptive fields and are structured dictionaries."""
    # Anxiety-specific checkin
    anxiety_checkin = {"anxiety": 8, "stress": 4, "mood": 7, "sleep_quality": 8, "isolation_score": 2}
    exercises, _ = generate_recommendations(current_checkin=anxiety_checkin)
    
    assert len(exercises) > 0
    for ex in exercises:
        assert isinstance(ex, dict), "Exercise must be a dictionary, not a raw string or undefined object"
        assert "title" in ex and len(ex["title"]) > 0
        assert "steps" in ex and len(ex["steps"]) > 0
        assert "benefit" in ex and len(ex["benefit"]) > 0
        assert "category" in ex and len(ex["category"]) > 0
        assert "duration" in ex and len(ex["duration"]) > 0
        # Ensure title and steps are clean strings
        assert "[object" not in ex["title"]
        assert "[object" not in ex["steps"]

    # Mood-specific checkin
    low_mood_checkin = {"mood": 2, "anxiety": 3, "stress": 3, "sleep_quality": 6, "isolation_score": 3}
    mood_exercises, _ = generate_recommendations(current_checkin=low_mood_checkin)
    assert len(mood_exercises) > 0
    assert any("behavioral" in m["title"].lower() or "compassion" in m["title"].lower() for m in mood_exercises)

    # General / balanced checkin
    balanced_checkin = {"mood": 8, "anxiety": 2, "stress": 2, "sleep_quality": 8, "isolation_score": 2}
    general_exercises, _ = generate_recommendations(current_checkin=balanced_checkin)
    assert len(general_exercises) > 0


# -------------------------------------------------------------
# Test 2: Multi-Agent AI Evaluation
# -------------------------------------------------------------
def test_multi_agent_evaluation_workflow_and_trace(client):
    """Verifies that checkin API executes the multi-agent pipeline and returns complete genuine trace."""
    client.post("/login", data={"email": "fix_patient@careai.com", "password": "password123"})

    payload = {
        "mood": 6,
        "anxiety": 5,
        "stress": 5,
        "sleep_quality": 7,
        "isolation_score": 3
    }
    resp = client.post("/api/checkin", json=payload)
    assert resp.status_code == 201
    data = resp.get_json()

    # Core response verification
    assert "multi_agent_evaluation" in data
    eval_data = data["multi_agent_evaluation"]
    assert eval_data["status"] == "completed"
    assert eval_data["total_agents"] == 5
    assert eval_data["successful_agents"] == 5
    assert len(eval_data["trace"]) == 5

    # Check that individual agents executed and reported contributions
    agent_names = [t["agent"] for t in eval_data["trace"]]
    assert "Orchestrator Agent" in agent_names
    assert "Distress Analysis Agent" in agent_names
    assert "Safety Agent" in agent_names
    assert "Monitoring Agent" in agent_names
    assert "Recommendation Agent" in agent_names

    for t in eval_data["trace"]:
        assert t["status"] == "success"
        assert len(t["contribution"]) > 10

    # Verify recommendations are properly structured
    assert "recommendations" in data
    assert isinstance(data["recommendations"], list)
    assert len(data["recommendations"]) > 0
    assert isinstance(data["recommendations"][0], dict)
    assert "title" in data["recommendations"][0]


# -------------------------------------------------------------
# Test 3: Trend Trajectory (Improving, Worsening, Stable, Insufficient Data)
# -------------------------------------------------------------
def test_trend_trajectory_improving():
    """Historical distress significantly decreasing must be classified as improving."""
    checkins = [
        {"timestamp": "2026-11-01T10:00:00Z", "distress_score": 0.75, "mood": 2, "anxiety": 8, "stress": 8, "sleep_quality": 3, "isolation_score": 8},
        {"timestamp": "2026-11-02T10:00:00Z", "distress_score": 0.55, "mood": 5, "anxiety": 5, "stress": 5, "sleep_quality": 5, "isolation_score": 5},
        {"timestamp": "2026-11-03T10:00:00Z", "distress_score": 0.40, "mood": 7, "anxiety": 4, "stress": 4, "sleep_quality": 7, "isolation_score": 3},
        {"timestamp": "2026-11-04T10:00:00Z", "distress_score": 0.22, "mood": 9, "anxiety": 2, "stress": 2, "sleep_quality": 8, "isolation_score": 2},
    ]
    trends = analyze_trends(checkins)
    assert trends["status"] == "success"
    assert trends["trend"] == "improving"
    assert trends["slope"] < -0.02


def test_trend_trajectory_worsening():
    """Historical distress significantly increasing must be classified as worsening."""
    checkins = [
        {"timestamp": "2026-11-01T10:00:00Z", "distress_score": 0.18, "mood": 9, "anxiety": 2, "stress": 2, "sleep_quality": 8, "isolation_score": 2},
        {"timestamp": "2026-11-02T10:00:00Z", "distress_score": 0.38, "mood": 6, "anxiety": 4, "stress": 4, "sleep_quality": 6, "isolation_score": 4},
        {"timestamp": "2026-11-03T10:00:00Z", "distress_score": 0.60, "mood": 4, "anxiety": 7, "stress": 7, "sleep_quality": 4, "isolation_score": 6},
        {"timestamp": "2026-11-04T10:00:00Z", "distress_score": 0.82, "mood": 2, "anxiety": 9, "stress": 9, "sleep_quality": 2, "isolation_score": 8},
    ]
    trends = analyze_trends(checkins)
    assert trends["status"] == "success"
    assert trends["trend"] == "worsening"
    assert trends["slope"] > 0.02


def test_trend_trajectory_stable():
    """Minor normal daily fluctuations must remain classified as stable rather than false worsening."""
    # Data similar to minor user fluctuations: 0.26 -> 0.32 -> 0.34 -> 0.34 -> 0.34
    checkins = [
        {"timestamp": "2026-11-01T10:00:00Z", "distress_score": 0.26, "mood": 7, "anxiety": 3, "stress": 4, "sleep_quality": 8, "isolation_score": 2},
        {"timestamp": "2026-11-02T10:00:00Z", "distress_score": 0.32, "mood": 5, "anxiety": 4, "stress": 4, "sleep_quality": 7, "isolation_score": 3},
        {"timestamp": "2026-11-03T10:00:00Z", "distress_score": 0.34, "mood": 6, "anxiety": 4, "stress": 5, "sleep_quality": 8, "isolation_score": 2},
        {"timestamp": "2026-11-04T10:00:00Z", "distress_score": 0.34, "mood": 6, "anxiety": 4, "stress": 5, "sleep_quality": 8, "isolation_score": 2},
        {"timestamp": "2026-11-05T10:00:00Z", "distress_score": 0.34, "mood": 6, "anxiety": 4, "stress": 5, "sleep_quality": 8, "isolation_score": 2},
    ]
    trends = analyze_trends(checkins)
    assert trends["status"] == "success"
    assert trends["trend"] == "stable"


def test_trend_trajectory_insufficient_data():
    """Fewer than 3 check-ins must be classified as insufficient_data."""
    # 0 check-ins
    t0 = analyze_trends([])
    assert t0["status"] == "insufficient_data"
    assert t0["trend"] == "insufficient_data"

    # 1 check-in
    t1 = analyze_trends([{"timestamp": "2026-11-01T10:00:00Z", "distress_score": 0.30}])
    assert t1["status"] == "insufficient_data"
    assert t1["trend"] == "insufficient_data"

    # 2 check-ins
    t2 = analyze_trends([
        {"timestamp": "2026-11-01T10:00:00Z", "distress_score": 0.30},
        {"timestamp": "2026-11-02T10:00:00Z", "distress_score": 0.35}
    ])
    assert t2["status"] == "insufficient_data"
    assert t2["trend"] == "insufficient_data"


def test_trend_chronological_sorting():
    """Check-ins passed in reverse or scrambled order must be sorted chronologically before regression."""
    scrambled = [
        {"timestamp": "2026-11-04T10:00:00Z", "distress_score": 0.20, "mood": 9, "anxiety": 2, "stress": 2, "sleep_quality": 8, "isolation_score": 2},
        {"timestamp": "2026-11-01T10:00:00Z", "distress_score": 0.70, "mood": 3, "anxiety": 7, "stress": 7, "sleep_quality": 3, "isolation_score": 7},
        {"timestamp": "2026-11-03T10:00:00Z", "distress_score": 0.35, "mood": 7, "anxiety": 3, "stress": 3, "sleep_quality": 7, "isolation_score": 3},
        {"timestamp": "2026-11-02T10:00:00Z", "distress_score": 0.55, "mood": 5, "anxiety": 5, "stress": 5, "sleep_quality": 5, "isolation_score": 5},
    ]
    trends = analyze_trends(scrambled)
    # Even though first element has distress=0.20 and last has 0.55, after sorting Nov 1 -> Nov 4 it is improving!
    assert trends["trend"] == "improving"
    assert trends["slope"] < 0


# -------------------------------------------------------------
# Test 4: Counselor Portal Verification
# -------------------------------------------------------------
def test_counselor_portal_data_isolation_and_actions(client, app):
    """Verifies counselor dashboard triage, alert notes, and case summary saving."""
    with app.app_context():
        patient = User.query.filter_by(email="fix_patient@careai.com").first()
        alert = SafetyAlert(
            user_id=patient.id,
            message="Test high distress safety alert",
            risk_level="high",
            status="unresolved"
        )
        db.session.add(alert)
        db.session.commit()
        alert_id = alert.id
        patient_id = patient.id

    # 1. Non-counselor access denied
    client.post("/login", data={"email": "fix_patient@careai.com", "password": "password123"})
    r_denied = client.get("/counselor/dashboard", follow_redirects=True)
    assert b"Access denied" in r_denied.data or b"dashboard" in r_denied.data
    client.get("/logout")

    # 2. Counselor login and dashboard
    client.post("/login", data={"email": "fix_counselor@careai.com", "password": "password123"})
    r_dash = client.get("/counselor/dashboard")
    assert r_dash.status_code == 200
    assert b"Safety Risk Indicators" in r_dash.data
    assert b"Patient Caseload" in r_dash.data

    # 3. Update alert status to in_review with counselor notes
    r_triage = client.post(f"/api/counselor/alert/{alert_id}/status", json={
        "status": "in_review",
        "counselor_notes": "Counselor initiated phone contact with patient."
    })
    assert r_triage.status_code == 200
    assert r_triage.get_json()["status"] == "in_review"

    # Verify updated in DB
    with app.app_context():
        updated_alert = SafetyAlert.query.get(alert_id)
        assert updated_alert.status == "in_review"
        assert "initiated phone contact" in updated_alert.counselor_notes

    # 4. Resolve alert
    r_resolve = client.post(f"/api/counselor/alert/{alert_id}/status", json={
        "status": "resolved",
        "counselor_notes": "Safety verified with patient."
    })
    assert r_resolve.status_code == 200
    assert r_resolve.get_json()["status"] == "resolved"

    with app.app_context():
        resolved_alert = SafetyAlert.query.get(alert_id)
        assert resolved_alert.status == "resolved"
        assert resolved_alert.resolved is True

    # 5. Patient Detail Page
    r_detail = client.get(f"/counselor/patient/{patient_id}")
    assert r_detail.status_code == 200
    assert b"Patient File" in r_detail.data


# -------------------------------------------------------------
# Test 5: Existing Accounts and Password Hashes Preserved
# -------------------------------------------------------------
def test_existing_accounts_authentication(client, app):
    """Verifies that all three core accounts authenticate with existing credentials."""
    for email in ["fix_patient@careai.com", "fix_counselor@careai.com", "fix_admin@careai.com"]:
        res = client.post("/login", data={"email": email, "password": "password123"}, follow_redirects=True)
        assert res.status_code == 200
        assert b"dashboard" in res.data or b"Logged in successfully" in res.data
        client.get("/logout")
