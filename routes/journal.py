# routes/journal.py
"""Routes for journal entries and sentiment/safety evaluation."""

from flask import Blueprint, request, jsonify, render_template
from flask_login import login_required, current_user
from extensions import db
from models.journal import JournalEntry
from agents.orchestrator import orchestrator

journal_bp = Blueprint("journal", __name__)

@journal_bp.route("/journal", methods=["GET"])
@login_required
def journal_page():
    """Renders the journal entry interface and history."""
    entries = (
        JournalEntry.query.filter_by(user_id=current_user.id)
        .order_by(JournalEntry.created_at.desc())
        .limit(20)
        .all()
    )
    return render_template("journal/journal.html", entries=entries)

@journal_bp.route("/api/journal", methods=["POST"])
@login_required
def create_journal():
    """API endpoint to save journal and run multi-agent evaluation."""
    data = request.get_json()
    if not data or "content" not in data or not str(data["content"]).strip():
        return jsonify({"error": "Content cannot be empty"}), 400

    content = str(data["content"]).strip()
    entry = JournalEntry(user_id=current_user.id, content=content)
    db.session.add(entry)
    db.session.commit()

    # Invoke multi-agent workflow
    agent_result = orchestrator.handle_journal(
        user_id=current_user.id,
        user_name=current_user.username,
        content=content
    )

    sentiment_data = agent_result.get("journal_sentiment", {})
    analysis_data = agent_result.get("journal_analysis") or {}

    sentiment_label = analysis_data.get("sentiment")
    if not sentiment_label:
        if isinstance(sentiment_data, dict):
            sentiment_label = sentiment_data.get("sentiment", "Neutral").capitalize()
        else:
            sentiment_label = str(sentiment_data).capitalize()

    response_text = analysis_data.get("wellbeing_feedback") or agent_result.get("response", "")

    return jsonify({
        "success": True,
        "message": "Journal saved and evaluated",
        "id": entry.id,
        "sentiment": sentiment_data,
        "sentiment_label": sentiment_label,
        "analysis": analysis_data,
        "safety_alert": agent_result.get("safety_alert", {}),
        "priority_override": agent_result.get("priority_override", False),
        "recommendations": agent_result.get("recommendations", []),
        "response": response_text
    }), 201

@journal_bp.route("/api/journals", methods=["GET"])
@login_required
def list_journals():
    """API endpoint to list user's journal entries."""
    entries = (
        JournalEntry.query.filter_by(user_id=current_user.id)
        .order_by(JournalEntry.created_at.desc())
        .limit(30)
        .all()
    )
    result = [
        {
            "id": e.id,
            "content": e.content,
            "created_at": e.created_at.isoformat() if e.created_at else "",
        }
        for e in entries
    ]
    return jsonify(result), 200
