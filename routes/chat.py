# routes/chat.py
"""Routes for AI Chat interface and multi-agent conversation with persistent conversation memory."""

import logging
from datetime import datetime
from flask import Blueprint, request, jsonify, render_template
from flask_login import login_required, current_user
from extensions import db
from models.chat import ChatMessage
from models.checkin import CheckIn
from agents.orchestrator import orchestrator
from agents.distress import analyze_conversational_distress

logger = logging.getLogger(__name__)
chat_bp = Blueprint("chat", __name__)

@chat_bp.route("/chat", methods=["GET"])
@login_required
def chat_page():
    """Renders the AI chat interface with past conversation history."""
    try:
        past_messages = (
            ChatMessage.query.filter_by(user_id=current_user.id)
            .order_by(ChatMessage.timestamp.asc())
            .limit(100)
            .all()
        )
    except Exception as e:
        logger.error(f"Error fetching chat page messages for user {current_user.id}: {e}")
        past_messages = []

    return render_template(
        "chat/chat.html",
        initial_messages=[m.to_dict() for m in past_messages]
    )

@chat_bp.route("/api/chat/history", methods=["GET"])
@login_required
def get_history():
    """Returns stored conversation history for the current user with pagination."""
    limit = request.args.get("limit", 50, type=int)
    offset = request.args.get("offset", 0, type=int)
    # Sanitize limit
    limit = max(1, min(limit, 200))
    offset = max(0, offset)

    try:
        total = ChatMessage.query.filter_by(user_id=current_user.id).count()
        messages = (
            ChatMessage.query.filter_by(user_id=current_user.id)
            .order_by(ChatMessage.timestamp.asc())
            .offset(offset)
            .limit(limit)
            .all()
        )
        return jsonify({
            "history": [m.to_dict() for m in messages],
            "total": total,
            "limit": limit,
            "offset": offset
        }), 200
    except Exception as e:
        logger.error(f"Error retrieving history for user {current_user.id}: {e}")
        return jsonify({"error": "Failed to retrieve conversation history", "history": []}), 500

@chat_bp.route("/api/chat/history", methods=["DELETE"])
@login_required
def clear_history():
    """Clears stored conversation history for the current user."""
    try:
        deleted_count = ChatMessage.query.filter_by(user_id=current_user.id).delete()
        db.session.commit()
        logger.info(f"Cleared {deleted_count} messages for user {current_user.id}")
        return jsonify({"success": True, "message": "Conversation history cleared."}), 200
    except Exception as e:
        db.session.rollback()
        logger.error(f"Error clearing history for user {current_user.id}: {e}")
        return jsonify({"error": "Failed to clear history"}), 500

@chat_bp.route("/api/chat", methods=["POST"])
@login_required
def chat():
    """API endpoint for multi-agent coordinated chat with persistent context."""
    data = request.get_json()
    if not data or "message" not in data or not str(data["message"]).strip():
        return jsonify({"error": "Message cannot be empty"}), 400

    user_msg = str(data["message"]).strip()

    # 1. Retrieve persistent conversation history for context-aware multi-turn continuity
    client_history = data.get("history", [])
    formatted_messages = []

    # If client passed active session history, prioritize it
    if isinstance(client_history, list) and client_history:
        for h in client_history[-10:]:
            if isinstance(h, dict) and h.get("content"):
                role = h.get("role", "user")
                if role in ("user", "assistant"):
                    formatted_messages.append({"role": role, "content": str(h["content"]).strip()})
    else:
        # Otherwise, load recent persistent conversation history from the database
        try:
            recent_db_msgs = (
                ChatMessage.query.filter_by(user_id=current_user.id)
                .order_by(ChatMessage.timestamp.desc())
                .limit(10)
                .all()
            )
            recent_db_msgs.reverse()
            for msg in recent_db_msgs:
                if msg.role in ("user", "assistant") and msg.content:
                    formatted_messages.append({"role": msg.role, "content": msg.content})
        except Exception as e:
            logger.warning(f"Could not load past DB messages for context: {e}")

    # Append the new user message to the conversational context
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
    try:
        db.session.add(user_chat_record)
        db.session.commit()
    except Exception as e:
        db.session.rollback()
        logger.error(f"Error persisting user chat message: {e}", exc_info=True)

    # 3. Retrieve user's latest checkin for state context
    checkin_dict = None
    try:
        latest_checkin = (
            CheckIn.query.filter_by(user_id=current_user.id)
            .order_by(CheckIn.timestamp.desc())
            .first()
        )
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
    except Exception as e:
        logger.warning(f"Could not load latest checkin for user {current_user.id}: {e}")

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
    try:
        db.session.add(assistant_chat_record)
        db.session.commit()
    except Exception as e:
        db.session.rollback()
        logger.error(f"Error persisting assistant chat message: {e}", exc_info=True)

    ai_meta = agent_result.get("ai_metadata") or {}
    return jsonify({
        "response": reply_content,
        "priority_override": is_safety_override,
        "ai_provider": ai_meta.get("provider", "unknown"),
        "ai_model": ai_meta.get("model", "unknown"),
        "is_llm": ai_meta.get("is_llm", False),
        "is_fallback": ai_meta.get("is_fallback", False),
        "is_error": ai_meta.get("is_error", False),
        "fallback_reason": ai_meta.get("fallback_reason"),
        "safety_alert": agent_result.get("safety_alert") or {},
        "recommendations": agent_result.get("recommendations") or [],
        "rag_context": [
            {"topic": r.get("topic"), "text": r.get("text")[:150] + "..."}
            for r in (agent_result.get("rag_context") or [])
            if isinstance(r, dict) and r.get("text")
        ]
    }), 200
