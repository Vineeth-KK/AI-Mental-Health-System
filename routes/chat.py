# routes/chat.py
"""Routes for AI Chat interface and multi-agent conversation with persistent conversation memory."""

from flask import Blueprint, request, jsonify, render_template
from flask_login import login_required, current_user
from extensions import db
from models.chat import ChatMessage
from models.checkin import CheckIn
from agents.orchestrator import orchestrator
from agents.distress import analyze_conversational_distress

chat_bp = Blueprint("chat", __name__)

@chat_bp.route("/chat", methods=["GET"])
@login_required
def chat_page():
    """Renders the AI chat interface with past conversation history."""
    past_messages = (
        ChatMessage.query.filter_by(user_id=current_user.id)
        .order_by(ChatMessage.timestamp.asc())
        .limit(50)
        .all()
    )
    return render_template(
        "chat/chat.html",
        initial_messages=[m.to_dict() for m in past_messages]
    )

@chat_bp.route("/api/chat/history", methods=["GET"])
@login_required
def get_history():
    """Returns stored conversation history for the current user."""
    messages = (
        ChatMessage.query.filter_by(user_id=current_user.id)
        .order_by(ChatMessage.timestamp.asc())
        .limit(50)
        .all()
    )
    return jsonify({"history": [m.to_dict() for m in messages]}), 200

@chat_bp.route("/api/chat/history", methods=["DELETE"])
@login_required
def clear_history():
    """Clears stored conversation history for the current user."""
    ChatMessage.query.filter_by(user_id=current_user.id).delete()
    db.session.commit()
    return jsonify({"success": True, "message": "Conversation history cleared."}), 200

@chat_bp.route("/api/chat", methods=["POST"])
@login_required
def chat():
    """API endpoint for multi-agent coordinated chat with persistent context."""
    data = request.get_json()
    if not data or "message" not in data or not str(data["message"]).strip():
        return jsonify({"error": "Message cannot be empty"}), 400

    user_msg = str(data["message"]).strip()

    # 1. Retrieve active conversation history for context-aware multi-turn continuity
    from datetime import datetime, timedelta
    
    client_history = data.get("history", [])
    recent_db_msgs = (
        ChatMessage.query.filter_by(user_id=current_user.id)
        .order_by(ChatMessage.timestamp.desc())
        .limit(14)
        .all()
    )
    recent_db_msgs.reverse()

    formatted_messages = []
    # Prioritize active tab conversation session for context-aware multi-turn continuity
    if isinstance(client_history, list) and client_history:
        for h in client_history[-6:]:
            if isinstance(h, dict) and h.get("content"):
                role = h.get("role", "user")
                if role in ("user", "assistant"):
                    formatted_messages.append({"role": role, "content": str(h["content"]).strip()})
    elif recent_db_msgs:
        # Use recent DB messages only if within an active recent session window (30 minutes)
        last_time = recent_db_msgs[-1].timestamp
        now_time = datetime.utcnow()
        if last_time and (now_time - last_time).total_seconds() < 1800:
            for msg in recent_db_msgs[-6:]:
                if msg.role in ("user", "assistant"):
                    formatted_messages.append({"role": msg.role, "content": msg.content})

    # Add the new user message to the workflow input
    formatted_messages.append({"role": "user", "content": user_msg})

    # 2. Persist user message to DB with calibrated distress analysis
    distress_signal = analyze_conversational_distress(user_msg)
    user_chat_record = ChatMessage(
        user_id=current_user.id,
        role="user",
        content=user_msg,
        distress_score=distress_signal.get("distress_score") if distress_signal else None,
        sentiment=distress_signal.get("sentiment") if distress_signal else None,
        detected_emotions=distress_signal.get("detected_emotions") if distress_signal else None
    )
    db.session.add(user_chat_record)
    db.session.commit()

    # 3. Retrieve user's latest checkin for state context
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

    # 4. Execute Multi-Agent Workflow
    agent_result = orchestrator.handle_chat(
        user_id=current_user.id,
        user_name=current_user.username,
        messages=formatted_messages,
        current_checkin=checkin_dict
    )

    reply_content = agent_result.get("response", "I am here to support you.")
    is_safety_override = agent_result.get("priority_override", False)

    # 5. Persist assistant reply to DB
    assistant_chat_record = ChatMessage(
        user_id=current_user.id,
        role="assistant",
        content=reply_content,
        is_safety_override=is_safety_override
    )
    db.session.add(assistant_chat_record)
    db.session.commit()

    return jsonify({
        "response": reply_content,
        "priority_override": is_safety_override,
        "safety_alert": agent_result.get("safety_alert", {}),
        "recommendations": agent_result.get("recommendations", []),
        "rag_context": [
            {"topic": r.get("topic"), "text": r.get("text")[:150] + "..."}
            for r in agent_result.get("rag_context", [])
        ]
    }), 200
