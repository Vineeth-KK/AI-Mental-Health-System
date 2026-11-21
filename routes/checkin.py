# routes/checkin.py
"""Routes for daily mental health check-ins."""

from flask import Blueprint, request, jsonify, render_template
from flask_login import login_required, current_user
from extensions import db
from models.checkin import CheckIn
from agents.distress import calculate_distress_score
from agents.orchestrator import orchestrator

checkin_bp = Blueprint("checkin", __name__)

@checkin_bp.route("/checkin", methods=["GET"])
@login_required
def checkin_page():
    """Renders the interactive check-in web page."""
    recent_checkins = (
        CheckIn.query.filter_by(user_id=current_user.id)
        .order_by(CheckIn.timestamp.desc())
        .limit(10)
        .all()
    )
    return render_template("checkin/checkin.html", recent_checkins=recent_checkins)

@checkin_bp.route("/api/checkin", methods=["POST"])
@login_required
def create_checkin():
    """API endpoint to record check-in and run multi-agent evaluation."""
    data = request.get_json()
    required_fields = ["mood", "anxiety", "stress", "sleep_quality", "isolation_score"]
    if not data or not all(f in data for f in required_fields):
        return jsonify({"error": "Missing required fields"}), 400

    try:
        mood = int(data["mood"])
        anxiety = int(data["anxiety"])
        stress = int(data["stress"])
        sleep_quality = int(data["sleep_quality"])
        isolation_score = int(data["isolation_score"])
    except (ValueError, TypeError):
        return jsonify({"error": "Invalid metric values. Must be integers between 1 and 10."}), 400

    # Calculate deterministic distress score
    score, level, breakdown = calculate_distress_score(mood, anxiety, stress, isolation_score, sleep_quality)

    checkin = CheckIn(
        user_id=current_user.id,
        mood=mood,
        anxiety=anxiety,
        stress=stress,
        sleep_quality=sleep_quality,
        isolation_score=isolation_score,
        distress_score=score,
        distress_level=level
    )
    db.session.add(checkin)
    db.session.commit()

    # Invoke Multi-Agent Workflow via Orchestrator
    agent_result = orchestrator.handle_checkin(
        user_id=current_user.id,
        user_name=current_user.username,
        checkin_data={
            "mood": mood,
            "anxiety": anxiety,
            "stress": stress,
            "sleep_quality": sleep_quality,
            "isolation_score": isolation_score,
            "distress_score": score,
            "distress_level": level
        }
    )

    return jsonify({
        "message": "Check-in recorded and evaluated",
        "id": checkin.id,
        "distress_score": score,
        "distress_level": level,
        "distress_breakdown": breakdown,
        "trend_analysis": agent_result.get("trend_analysis", {}),
        "recommendations": agent_result.get("recommendations", []),
        "safety_alert": agent_result.get("safety_alert", {}),
        "priority_override": agent_result.get("priority_override", False),
        "response": agent_result.get("response", "")
    }), 201

@checkin_bp.route("/api/checkins", methods=["GET"])
@login_required
def list_checkins():
    """API endpoint to list user's historical check-ins."""
    checks = (
        CheckIn.query.filter_by(user_id=current_user.id)
        .order_by(CheckIn.timestamp.desc())
        .limit(30)
        .all()
    )
    result = [
        {
            "id": c.id,
            "mood": c.mood,
            "anxiety": c.anxiety,
            "stress": c.stress,
            "sleep_quality": c.sleep_quality,
            "isolation_score": c.isolation_score,
            "distress_score": c.distress_score,
            "distress_level": c.distress_level,
            "timestamp": c.timestamp.isoformat() if c.timestamp else "",
        }
        for c in checks
    ]
    return jsonify(result), 200
