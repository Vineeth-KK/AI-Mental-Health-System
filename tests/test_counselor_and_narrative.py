# tests/test_counselor_and_narrative.py
"""Comprehensive verification of the new longitudinal distress tracking,
crisis narrative synthesis, counselor alert notification, and case summary features.
"""

import pytest
from datetime import datetime, timezone, timedelta
from app import create_app
from extensions import db
from models.user import User
from models.checkin import CheckIn
from models.journal import JournalEntry
from models.chat import ChatMessage
from models.alert import SafetyAlert
from models.notification import CounselorNotification
from models.case_note import CaseSummary
from agents.distress import analyze_conversational_distress
from agents.monitoring import analyze_trends, fetch_user_conversational_history
from agents.narrative_agent import synthesize_crisis_narrative
from agents.case_summary_agent import generate_case_summary
from services.notifier import dispatch_counselor_alert


@pytest.fixture
def app():
    app = create_app()
    app.config["TESTING"] = True
    app.config["WTF_CSRF_ENABLED"] = False
    app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///:memory:"
    with app.app_context():
        db.create_all()
        yield app
        db.session.remove()
        db.drop_all()


@pytest.fixture
def client(app):
    return app.test_client()


@pytest.fixture
def test_users(app):
    with app.app_context():
        patient = User(username="patient1", email="patient@test.com", role="user")
        patient.set_password("pass123")
        counselor = User(username="counselor1", email="counselor@test.com", role="counselor")
        counselor.set_password("pass123")
        other_patient = User(username="patient2", email="patient2@test.com", role="user")
        other_patient.set_password("pass123")

        db.session.add_all([patient, counselor, other_patient])
        db.session.commit()
        return {
            "patient_id": patient.id,
            "counselor_id": counselor.id,
            "other_patient_id": other_patient.id
        }


def test_conversational_distress_tracking(client, test_users):
    """Verifies that conversational chat messages with distress markers extract signals, while casual greetings do not."""
    # 1. Message with distress cues
    distressed_analysis = analyze_conversational_distress("I have been feeling so anxious and stressed about my upcoming exams. I cannot sleep.")
    assert distressed_analysis is not None
    assert distressed_analysis["sentiment"] == "distressed"
    assert distressed_analysis["distress_score"] > 0.40
    assert "Anxiety" in distressed_analysis["detected_emotions"] or "Stress" in distressed_analysis["detected_emotions"]

    # 2. Casual greeting (should NOT generate distress score)
    greeting_analysis = analyze_conversational_distress("hello there, good morning!")
    assert greeting_analysis is None

    # 3. Informational question without first-person emotion (should NOT generate distress score)
    info_analysis = analyze_conversational_distress("what is cognitive behavioral therapy?")
    assert info_analysis is None

    # 4. Verify API persistence
    client.post("/login", data={"email": "patient@test.com", "password": "pass123"})
    resp = client.post("/api/chat", json={"message": "I am feeling overwhelmed and deeply exhausted today."})
    assert resp.status_code == 200

    last_msg = ChatMessage.query.filter_by(user_id=test_users["patient_id"], role="user").order_by(ChatMessage.timestamp.desc()).first()
    assert last_msg is not None
    assert last_msg.distress_score is not None
    assert last_msg.distress_score >= 0.40
    assert last_msg.sentiment == "distressed"


def test_monitoring_agent_receives_conversational_signals(app, test_users):
    """Verifies that the Monitoring Agent fetches and incorporates conversational distress signals into trend analysis."""
    with app.app_context():
        u_id = test_users["patient_id"]
        # Add conversational signal
        cm = ChatMessage(
            user_id=u_id,
            role="user",
            content="I am having trouble sleeping and feeling anxious",
            distress_score=0.62,
            sentiment="distressed",
            detected_emotions="Anxiety, Sleep Disruption"
        )
        # Add daily checkin
        ci = CheckIn(
            user_id=u_id,
            mood=5, anxiety=6, stress=6, sleep_quality=5, isolation_score=5,
            distress_score=0.55, distress_level="moderate"
        )
        db.session.add_all([cm, ci])
        db.session.commit()

        conv_history = fetch_user_conversational_history(u_id)
        assert len(conv_history) == 1
        assert conv_history[0]["distress_score"] == 0.62

        trends = analyze_trends(
            checkins=[{"distress_score": 0.55, "mood": 5, "anxiety": 6, "stress": 6, "sleep_quality": 5, "isolation_score": 5}],
            conversational_signals=conv_history
        )
        assert "conversational_distress" in trends
        assert trends["conversational_distress"] is not None
        assert trends["conversational_distress"]["count"] == 1
        assert trends["conversational_distress"]["avg_distress"] == 0.62


def test_crisis_narrative_synthesis_agent(app, test_users):
    """Verifies that the Narrative Synthesis Agent evaluates longitudinal entries and distinguishes facts from themes without hallucination."""
    with app.app_context():
        u_id = test_users["patient_id"]
        # Seed 2 journal entries with specific concerns
        j1 = JournalEntry(user_id=u_id, content="Upcoming exam preparation is creating huge stress. I find it difficult to concentrate.")
        j2 = JournalEntry(user_id=u_id, content="Another day feeling exam anxiety and lack of concentration.")
        db.session.add_all([j1, j2])
        db.session.commit()

        narrative = synthesize_crisis_narrative(user_id=u_id)
        assert narrative["status"] == "success"
        assert len(narrative["narrative_summary"]) > 50
        assert isinstance(narrative["explicitly_reported_concerns"], list)
        assert len(narrative["explicitly_reported_concerns"]) > 0
        assert any("exam" in c.lower() or "academic" in c.lower() for c in narrative["explicitly_reported_concerns"])
        assert "trajectory" in narrative

        # Verify insufficient data handling for user with 0 records
        sparse_narrative = synthesize_crisis_narrative(user_id=test_users["other_patient_id"])
        assert sparse_narrative["status"] == "insufficient_data"


def test_counselor_authorization_and_access_control(client, test_users):
    """Verifies strict RBAC: Unauthorized regular users cannot access counselor routes, while authorized counselors can."""
    # 1. Unauthenticated request -> redirects to login
    resp = client.get("/counselor/dashboard")
    assert resp.status_code == 302
    assert "/login" in resp.headers["Location"]

    # 2. Regular user logged in -> redirected with permission error
    client.post("/login", data={"email": "patient@test.com", "password": "pass123"})
    resp_user = client.get("/counselor/dashboard", follow_redirects=False)
    assert resp_user.status_code == 302
    assert "/dashboard" in resp_user.headers["Location"]

    # Regular user attempting patient review -> blocked
    resp_detail = client.get(f"/counselor/patient/{test_users['patient_id']}", follow_redirects=False)
    assert resp_detail.status_code == 302
    assert "/dashboard" in resp_detail.headers["Location"]

    # 3. Counselor logged in -> 200 OK
    client.get("/logout")
    client.post("/login", data={"email": "counselor@test.com", "password": "pass123"})
    resp_counselor = client.get("/counselor/dashboard")
    assert resp_counselor.status_code == 200
    assert "Counselor" in resp_counselor.text or "Triage" in resp_counselor.text

    resp_patient_view = client.get(f"/counselor/patient/{test_users['patient_id']}")
    assert resp_patient_view.status_code == 200
    assert "Patient File" in resp_patient_view.text


def test_alert_notification_dispatch(app, test_users):
    """Verifies that high-risk safety alerts dispatch in-app notifications to authorized counselors, avoiding duplicate dispatches."""
    with app.app_context():
        p_id = test_users["patient_id"]
        c_id = test_users["counselor_id"]

        alert = SafetyAlert(
            user_id=p_id,
            message="Critical self-harm marker detected: 'kill myself'",
            risk_level="critical"
        )
        db.session.add(alert)
        db.session.commit()

        # Dispatch alert
        dispatched_count = dispatch_counselor_alert(
            alert_id=alert.id,
            user_id=p_id,
            risk_level="critical",
            reason="Critical self-harm indicator"
        )
        assert dispatched_count >= 1

        # Check notification record
        notif = CounselorNotification.query.filter_by(alert_id=alert.id).first()
        assert notif is not None
        assert notif.counselor_id == c_id
        assert notif.status in ["delivered", "pending", "not_configured"]
        # Raw diary text should NOT be leaked in subject
        assert "kill myself" not in notif.subject

        # Verify duplicate dispatch prevention
        dup_count = dispatch_counselor_alert(alert_id=alert.id, user_id=p_id)
        assert dup_count == 0


def test_case_summary_generator_and_persistence(client, app, test_users):
    """Verifies that the Case Summary Generator produces 7-section structured output and can be saved and approved."""
    with app.app_context():
        u_id = test_users["patient_id"]
        c_id = test_users["counselor_id"]
        j = JournalEntry(user_id=u_id, content="Upcoming exam preparation is creating huge stress. I find it difficult to concentrate.")
        ci = CheckIn(user_id=u_id, mood=4, anxiety=8, stress=8, sleep_quality=4, isolation_score=6, distress_score=0.72, distress_level="high")
        db.session.add_all([j, ci])
        db.session.commit()

        # 1. Direct agent generation
        summary = generate_case_summary(user_id=u_id, counselor_id=c_id)
        assert summary["status"] == "success"
        assert len(summary["overview"]) > 20
        assert len(summary["reported_concerns"]) > 0
        assert "disclaimer" in summary

    # 2. Counselor API test
    client.post("/login", data={"email": "counselor@test.com", "password": "pass123"})
    gen_resp = client.post(f"/api/counselor/case-summary/generate/{u_id}")
    assert gen_resp.status_code == 200
    gen_data = gen_resp.get_json()
    assert "overview" in gen_data
    assert "reported_concerns" in gen_data

    # 3. Save case summary via API
    save_resp = client.post("/api/counselor/case-summary/save", json={
        "user_id": u_id,
        "overview": gen_data["overview"],
        "reported_concerns": gen_data["reported_concerns"],
        "recurring_patterns": gen_data.get("recurring_patterns", ""),
        "changes_over_time": gen_data.get("changes_over_time", ""),
        "historical_events": gen_data.get("historical_events", ""),
        "risk_indicators": gen_data.get("risk_indicators", ""),
        "recommended_review": gen_data.get("recommended_review", ""),
        "counselor_edits": "Reviewed in clinical session on Nov 21.",
        "status": "approved"
    })
    assert save_resp.status_code == 200
    save_data = save_resp.get_json()
    assert save_data["success"] is True

    # Verify saved in database
    with app.app_context():
        saved_record = CaseSummary.query.filter_by(user_id=u_id).first()
        assert saved_record is not None
        assert saved_record.status == "approved"
        assert "Reviewed in clinical session" in saved_record.counselor_edits


def test_alert_triage_status_update(client, app, test_users):
    """Verifies that counselors can update alert triage status and add clinical notes."""
    with app.app_context():
        p_id = test_users["patient_id"]
        alert = SafetyAlert(
            user_id=p_id,
            message="Severe anxiety report",
            risk_level="high",
            status="unresolved"
        )
        db.session.add(alert)
        db.session.commit()
        alert_id = alert.id

    client.post("/login", data={"email": "counselor@test.com", "password": "pass123"})
    resp = client.post(f"/api/counselor/alert/{alert_id}/status", json={
        "status": "in_review",
        "counselor_notes": "Spoke with student by phone. Safety plan confirmed."
    })
    assert resp.status_code == 200
    assert resp.get_json()["status"] == "in_review"

    with app.app_context():
        updated = SafetyAlert.query.get(alert_id)
        assert updated.status == "in_review"
        assert "Safety plan confirmed" in updated.counselor_notes
        assert updated.reviewed_by == test_users["counselor_id"]
        assert updated.reviewed_at is not None
