# routes/chat.py
"""Routes for AI Chat interface and multi-agent conversation."""

from flask import Blueprint, request, jsonify, render_template
from flask_login import login_required, current_user
from agents.orchestrator import orchestrator
from models.checkin import CheckIn

chat_bp = Blueprint("chat", __name__)

@chat_bp.route("/chat", methods=["GET"])
@login_required
def chat_page():
    """Renders the AI chat interface."""
    return render_template("chat/chat.html")

@chat_bp.route("/api/chat", methods=["POST"])
@login_required
def chat():
    """API endpoint for multi-agent coordinated chat."""
    data = request.get_json()
    if not data or "message" not in data or not str(data["message"]).strip():
        return jsonify({"error": "Message cannot be empty"}), 400

    user_msg = str(data["message"]).strip()
    history = data.get("history", [])

    # Format message history
    formatted_messages = []
    for item in history:
        if isinstance(item, dict) and "role" in item and "content" in item:
            formatted_messages.append({"role": item["role"], "content": item["content"]})
    
    # Add current user message
    formatted_messages.append({"role": "user", "content": user_msg})

    # Retrieve user's latest checkin for context
    latest_checkin = (
        CheckIn.query.filter_by(user_id=current_user.id)
        .order_by(CheckIn.timestamp.desc())
        .first()
    )
    checkin_dict = None
    if latest_checkin:
        checkin_dict = {
            "mood": latest_checkin.mood,
            "anxiety": latest_checkin.anxiety,
            "stress": latest_checkin.stress,
            "sleep_quality": latest_checkin.sleep_quality,
            "isolation_score": latest_checkin.isolation_score,
            "distress_score": latest_checkin.distress_score,
            "distress_level": latest_checkin.distress_level
        }

    # Execute Multi-Agent Workflow
    agent_result = orchestrator.handle_chat(
        user_id=current_user.id,
        user_name=current_user.username,
        messages=formatted_messages,
        current_checkin=checkin_dict
    )

    return jsonify({
        "response": agent_result.get("response", "I am here to support you."),
        "priority_override": agent_result.get("priority_override", False),
        "safety_alert": agent_result.get("safety_alert", {}),
        "recommendations": agent_result.get("recommendations", []),
        "rag_context": [
            {"topic": r.get("topic"), "text": r.get("text")[:150] + "..."}
            for r in agent_result.get("rag_context", [])
        ]
    }), 200
