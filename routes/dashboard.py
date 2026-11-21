# routes/dashboard.py
"""Dashboard routes for dynamic mental health trend monitoring and Chart.js visualizations."""

from flask import Blueprint, render_template, jsonify
from flask_login import login_required, current_user
from models.checkin import CheckIn
from models.journal import JournalEntry
from models.alert import SafetyAlert
from agents.monitoring import analyze_trends, fetch_user_checkin_history
from agents.distress import calculate_distress_score
from services.rag import rag_service

dashboard_bp = Blueprint("dashboard", __name__)

@dashboard_bp.route("/dashboard")
@login_required
def dashboard():
    """Renders the comprehensive mental health monitoring dashboard."""
    # Isolated checkins for current user
    checks = (
        CheckIn.query.filter_by(user_id=current_user.id)
        .order_by(CheckIn.timestamp.asc())
        .limit(30)
        .all()
    )
    
    # Process checkin history for trend analysis
    history_dicts = []
    for c in checks:
        history_dicts.append({
            "id": c.id,
            "mood": c.mood,
            "anxiety": c.anxiety,
            "stress": c.stress,
            "sleep_quality": c.sleep_quality,
            "isolation_score": c.isolation_score,
            "distress_score": c.distress_score,
            "distress_level": c.distress_level,
            "timestamp": c.timestamp.strftime("%b %d") if c.timestamp else ""
        })

    trend_data = analyze_trends(history_dicts)
    latest_checkin = checks[-1] if checks else None

    # Recent alerts
    alerts = (
        SafetyAlert.query.filter_by(user_id=current_user.id)
        .order_by(SafetyAlert.created_at.desc())
        .limit(5)
        .all()
    )

    # Recent journals
    journals_count = JournalEntry.query.filter_by(user_id=current_user.id).count()

    # RAG educational resources for user
    resources = rag_service.query("stress reduction mindfulness and emotional balance", n_results=3)

    return render_template(
        "dashboard/dashboard.html",
        latest_checkin=latest_checkin,
        trend_data=trend_data,
        alerts=alerts,
        total_checkins=len(checks),
        total_journals=journals_count,
        resources=resources
    )

@dashboard_bp.route("/api/dashboard", methods=["GET"])
@login_required
def dashboard_data():
    """Returns JSON time-series metrics and trend analytics for Chart.js."""
    checks = (
        CheckIn.query.filter_by(user_id=current_user.id)
        .order_by(CheckIn.timestamp.asc())
        .limit(30)
        .all()
    )

    labels = [c.timestamp.strftime("%b %d") if c.timestamp else f"#{i+1}" for i, c in enumerate(checks)]
    
    # Ensure distress score is computed for each check-in
    distress_scores = []
    history_dicts = []
    for c in checks:
        score = c.distress_score
        if score is None:
            score, _, _ = calculate_distress_score(c.mood, c.anxiety, c.stress, c.isolation_score, c.sleep_quality)
        distress_scores.append(round(score, 3))
        history_dicts.append({
            "mood": c.mood,
            "anxiety": c.anxiety,
            "stress": c.stress,
            "sleep_quality": c.sleep_quality,
            "isolation_score": c.isolation_score,
            "distress_score": score,
            "timestamp": c.timestamp.isoformat() if c.timestamp else ""
        })

    trends = analyze_trends(history_dicts)

    data = {
        "labels": labels,
        "mood": [c.mood for c in checks],
        "anxiety": [c.anxiety for c in checks],
        "stress": [c.stress for c in checks],
        "sleep_quality": [c.sleep_quality for c in checks],
        "isolation_score": [c.isolation_score for c in checks],
        "distress_score": distress_scores,
        "trends": trends
    }
    return jsonify(data), 200
