# routes/counselor.py
"""Routes for authorized human counselors: dashboard, patient reviews, alerts triage, and case summaries."""

from functools import wraps
from flask import Blueprint, request, jsonify, render_template, redirect, url_for, flash, abort
from flask_login import login_required, current_user
from datetime import datetime, timezone
from extensions import db
from models.user import User
from models.alert import SafetyAlert
from models.checkin import CheckIn
from models.journal import JournalEntry
from models.chat import ChatMessage
from models.case_note import CaseSummary
from models.notification import CounselorNotification
from agents.monitoring import analyze_trends, fetch_user_checkin_history, fetch_user_conversational_history
from agents.narrative_agent import synthesize_crisis_narrative
from agents.case_summary_agent import generate_case_summary

counselor_bp = Blueprint("counselor", __name__)


def counselor_required(f):
    """Decorator ensuring only authenticated users with counselor or admin privileges can access clinical routes."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not current_user.is_authenticated:
            return redirect(url_for("auth.login", next=request.url))
        is_auth = (
            getattr(current_user, "is_counselor", lambda: False)() or
            getattr(current_user, "is_admin", False) or
            getattr(current_user, "role", "user") in ["counselor", "admin"]
        )
        if not is_auth:
            flash("Access denied: You do not have human counselor authorization privileges.", "danger")
            return redirect(url_for("dashboard.dashboard"))
        return f(*args, **kwargs)
    return decorated_function


@counselor_bp.route("/counselor/dashboard", methods=["GET"])
@login_required
@counselor_required
def counselor_dashboard():
    """Renders the central counselor portal with unresolved alerts and patient caseload."""
    alerts = SafetyAlert.query.order_by(SafetyAlert.created_at.desc()).all()
    patients = User.query.filter(User.role == "user").order_by(User.registered_on.desc()).all()

    # Precompute statistics
    unresolved_count = sum(1 for a in alerts if getattr(a, "status", "unresolved") == "unresolved")
    in_review_count = sum(1 for a in alerts if getattr(a, "status", "unresolved") == "in_review")
    resolved_count = sum(1 for a in alerts if getattr(a, "status", "unresolved") == "resolved")

    # Patient risk summaries
    patient_summaries = []
    for p in patients:
        recent_checkins = fetch_user_checkin_history(p.id, limit=7)
        conv_signals = fetch_user_conversational_history(p.id, limit=7)
        p_alerts = [a for a in alerts if a.user_id == p.id]
        trends = analyze_trends(recent_checkins, conv_signals)
        patient_summaries.append({
            "patient": p,
            "distress_score": trends.get("current_score", "N/A"),
            "trend": trends.get("trend", "stable"),
            "slope": trends.get("slope", 0.0),
            "alerts_count": len(p_alerts),
            "unresolved_alerts": sum(1 for a in p_alerts if getattr(a, "status", "unresolved") != "resolved")
        })

    return render_template(
        "counselor/dashboard.html",
        alerts=alerts,
        patient_summaries=patient_summaries,
        unresolved_count=unresolved_count,
        in_review_count=in_review_count,
        resolved_count=resolved_count
    )


@counselor_bp.route("/counselor/patient/<int:user_id>", methods=["GET"])
@login_required
@counselor_required
def patient_detail(user_id):
    """Detailed clinical review page for a specific patient."""
    patient = User.query.get_or_404(user_id)
    alerts = SafetyAlert.query.filter_by(user_id=user_id).order_by(SafetyAlert.created_at.desc()).all()
    checkins = fetch_user_checkin_history(user_id, limit=20)
    conv_signals = fetch_user_conversational_history(user_id, limit=15)
    journals = JournalEntry.query.filter_by(user_id=user_id).order_by(JournalEntry.created_at.desc()).limit(10).all()
    case_summaries = CaseSummary.query.filter_by(user_id=user_id).order_by(CaseSummary.created_at.desc()).all()
    trends = analyze_trends(checkins, conv_signals)

    return render_template(
        "counselor/patient_detail.html",
        patient=patient,
        alerts=alerts,
        checkins=checkins,
        conv_signals=conv_signals,
        journals=journals,
        case_summaries=case_summaries,
        trends=trends
    )


@counselor_bp.route("/api/counselor/alert/<int:alert_id>/status", methods=["POST"])
@login_required
@counselor_required
def update_alert_status(alert_id):
    """Updates review notes and resolution status for a safety alert."""
    alert = SafetyAlert.query.get_or_404(alert_id)
    data = request.get_json() or {}

    new_status = data.get("status", "in_review")
    if new_status not in ["unresolved", "in_review", "resolved"]:
        return jsonify({"error": "Invalid status."}), 400

    alert.status = new_status
    if "counselor_notes" in data:
        alert.counselor_notes = str(data["counselor_notes"]).strip()
    alert.resolved = (new_status == "resolved")
    alert.reviewed_by = current_user.id
    alert.reviewed_at = datetime.now(timezone.utc)

    db.session.commit()
    return jsonify({
        "success": True,
        "message": f"Alert #{alert_id} marked as {new_status}.",
        "alert_id": alert.id,
        "status": alert.status
    }), 200


@counselor_bp.route("/api/counselor/case-summary/generate/<int:user_id>", methods=["GET", "POST"])
@login_required
@counselor_required
def api_generate_case_summary(user_id):
    """Generates an AI clinical case summary on demand for an authorized counselor."""
    patient = User.query.get(user_id)
    if not patient:
        return jsonify({"error": f"Patient #{user_id} not found."}), 404

    summary_data = generate_case_summary(user_id=user_id, counselor_id=current_user.id)
    return jsonify(summary_data), 200


@counselor_bp.route("/api/counselor/case-summary/save", methods=["POST"])
@login_required
@counselor_required
def api_save_case_summary():
    """Saves or updates an edited/approved clinical case summary."""
    data = request.get_json() or {}
    user_id = data.get("user_id")
    if not user_id:
        return jsonify({"error": "user_id is required."}), 400

    summary_id = data.get("id")
    if summary_id:
        record = CaseSummary.query.get(summary_id)
        if not record:
            return jsonify({"error": "Summary record not found."}), 404
    else:
        record = CaseSummary(user_id=user_id, counselor_id=current_user.id)
        db.session.add(record)

    record.title = data.get("title", f"Clinical Case Summary – Patient #{user_id}")
    record.overview = data.get("overview", "")
    record.reported_concerns = data.get("reported_concerns", "")
    record.recurring_patterns = data.get("recurring_patterns", "")
    record.changes_over_time = data.get("changes_over_time", "")
    record.historical_events = data.get("historical_events", "")
    record.risk_indicators = data.get("risk_indicators", "")
    record.recommended_review = data.get("recommended_review", "")
    record.full_content = data.get("full_content", "")
    record.counselor_edits = data.get("counselor_edits", "")
    record.status = data.get("status", "reviewed")
    record.updated_at = datetime.now(timezone.utc)

    db.session.commit()
    return jsonify({
        "success": True,
        "message": "Clinical case summary saved successfully.",
        "id": record.id,
        "status": record.status
    }), 200


@counselor_bp.route("/api/counselor/case-summary/<int:summary_id>", methods=["GET"])
@login_required
@counselor_required
def api_get_case_summary(summary_id):
    """Retrieves an existing saved case summary."""
    record = CaseSummary.query.get_or_404(summary_id)
    return jsonify(record.to_dict()), 200


@counselor_bp.route("/api/counselor/narrative/<int:user_id>", methods=["GET", "POST"])
@login_required
@counselor_required
def api_get_narrative_synthesis(user_id):
    """Generates on-demand narrative synthesis across multiple entries for an authorized counselor."""
    synthesis = synthesize_crisis_narrative(user_id=user_id)
    return jsonify(synthesis), 200
