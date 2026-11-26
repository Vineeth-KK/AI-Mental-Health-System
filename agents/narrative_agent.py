# agents/narrative_agent.py
"""Crisis Narrative Synthesis Agent.
Synthesizes longitudinal data across historical journal reflections, conversational distress signals,
and daily check-in trends to construct an objective, factual, non-diagnostic narrative arc over time.
Integrates with local Ollama LLM (llama3.2) with a factual, hallucination-resistant structured schema
and an intelligent deterministic fallback engine.
"""

import json
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone, timedelta
from flask import has_app_context
import httpx

from extensions import db
from models.journal import JournalEntry
from models.chat import ChatMessage
from models.checkin import CheckIn
from agents.monitoring import analyze_trends, fetch_user_checkin_history, fetch_user_conversational_history
from services.ai_provider import ai_provider, extract_json_from_text

logger = logging.getLogger(__name__)

NARRATIVE_SYSTEM_PROMPT = """You are CareAI's Crisis Narrative Synthesis Agent, an objective, evidence-grounded clinical support assistant.
Your task is to synthesize longitudinal personal wellbeing data for a user across multiple historical journal entries, conversational chat signals, and daily check-ins.

CRITICAL REPORTING CONSTRAINTS:
1. STRICT FACTUAL GROUNDING: Rely ONLY on the information explicitly provided in the user's timeline. NEVER invent traumatic events, family conflicts, physical abuse, or psychiatric diagnoses.
2. DISTINGUISH CATEGORIES RIGOROUSLY:
   - "explicitly_reported_concerns": Issues the user directly and literally stated in their entries.
   - "recurring_themes": Patterns that appear across multiple separate entries over time.
   - "ai_observations": Gentle, tentative observations regarding trajectory, clearly marked with uncertainty.
3. SINGLE NEGATIVE STATEMENTS: A single negative remark is NOT evidence of chronic trauma or crisis. Report it neutrally as an isolated disclosure.
4. INSUFFICIENT DATA: If fewer than 2 distinct entries exist, state clearly that data is insufficient for pattern synthesis.
5. NO DIAGNOSES: Never apply clinical DSM/ICD psychiatric labels.

Respond ONLY with a valid JSON object matching this schema:
{
  "status": "success",
  "narrative_summary": "2-3 paragraphs factual synthesis with dates and direct references",
  "explicitly_reported_concerns": ["Concern 1", "Concern 2"],
  "recurring_themes": ["Theme 1", "Theme 2"],
  "trajectory": "improving|stable|escalating",
  "ai_observations": "Gentle non-diagnostic observations on changes over time",
  "data_points_evaluated": 5
}
"""


def fetch_user_journal_history(user_id: int, days_limit: int = 30, entry_limit: int = 10) -> List[Dict[str, Any]]:
    """Fetches user's past journal entries within time window, strictly isolated by user_id."""
    if not has_app_context() or not user_id:
        return []
    cutoff = datetime.utcnow() - timedelta(days=days_limit)
    records = (
        JournalEntry.query.filter(
            JournalEntry.user_id == user_id,
            JournalEntry.created_at >= cutoff
        )
        .order_by(JournalEntry.created_at.asc())
        .limit(entry_limit)
        .all()
    )
    return [
        {
            "id": r.id,
            "content": r.content,
            "date": r.created_at.strftime("%Y-%m-%d %H:%M") if r.created_at else "Unknown date"
        }
        for r in records
    ]


def _generate_fallback_narrative(
    journals: List[Dict[str, Any]],
    conversational_signals: List[Dict[str, Any]],
    trend_data: Dict[str, Any]
) -> Dict[str, Any]:
    """Deterministic, factual narrative synthesis engine when Ollama is offline.
    Never invents facts; extracts literal terms and combines with regression slope.
    """
    total_data_points = len(journals) + len(conversational_signals)

    if total_data_points < 2:
        return {
            "status": "insufficient_data",
            "narrative_summary": "Insufficient longitudinal records to establish a recurring narrative arc. At least two historical reflections or check-in signals are required.",
            "explicitly_reported_concerns": ["Limited historical data logged."],
            "recurring_themes": ["Initial self-monitoring"],
            "trajectory": "insufficient_data",
            "ai_observations": "Single or no entries logged. Ongoing check-ins will enable pattern identification.",
            "data_points_evaluated": total_data_points
        }

    # Extract literal user-reported concerns
    reported_concerns = []
    recurring_themes = []

    all_texts = [j["content"] for j in journals] + [c["content"] for c in conversational_signals]
    combined_text = " ".join(all_texts).lower()

    # Identify literal mentions
    topic_keywords = [
        ("exam", "Academic stress and exam preparation"),
        ("stud", "Study workload and academic pressure"),
        ("concentrat", "Difficulty concentrating"),
        ("sleep", "Sleep disruption / insomnia"),
        ("tired", "Physical and mental exhaustion"),
        ("lonel", "Feelings of loneliness and social disconnection"),
        ("friend", "Social dynamics and interpersonal relationships"),
        ("plan", "Frustration with plans not going as expected"),
        ("work", "Work-related stress and responsibilities"),
        ("happy", "Positive moments and social enjoyment"),
        ("grateful", "Practicing gratitude and appreciation")
    ]

    theme_counts = {}
    for kw, label in topic_keywords:
        hits = sum(1 for t in all_texts if kw in t.lower())
        if hits >= 1 and label not in reported_concerns:
            reported_concerns.append(label)
        if hits >= 2:
            recurring_themes.append(f"{label} (mentioned in {hits} separate entries)")
            theme_counts[label] = hits

    if not recurring_themes and reported_concerns:
        recurring_themes = ["Isolated self-reflections without repeated chronic distress markers"]

    # Trajectory assessment based on monitoring regression slope
    slope = trend_data.get("slope", 0.0)
    if slope < -0.015:
        trajectory = "improving"
        traj_desc = "shows a steady downward drift in distress indicators, reflecting positive emotional adaptation."
    elif slope > 0.015:
        trajectory = "escalating"
        traj_desc = "indicates an upward creep in distress levels over recent data points, warranting gentle supportive review."
    else:
        trajectory = "stable"
        traj_desc = "indicates a relatively steady, balanced emotional baseline across recorded entries."

    summary = (
        f"Longitudinal synthesis based on {len(journals)} journal reflection(s) and {len(conversational_signals)} conversational distress signal(s). "
        f"The user's overall mental capacity trajectory {traj_desc} "
    )
    if reported_concerns:
        summary += f"Explicit concerns documented by the user include: {'; '.join(reported_concerns[:3])}. "
    if recurring_themes:
        summary += f"Recurring thematic patterns identified across entries include: {'; '.join(recurring_themes[:2])}."

    ai_obs = (
        f"Data points indicate the user actively engages in reflective self-monitoring. "
        f"Slope coefficient is {slope:.4f}. No unverified external trauma is inferred."
    )

    return {
        "status": "success",
        "narrative_summary": summary,
        "explicitly_reported_concerns": reported_concerns[:5],
        "recurring_themes": recurring_themes[:4],
        "trajectory": trajectory,
        "ai_observations": ai_obs,
        "data_points_evaluated": total_data_points
    }


def synthesize_crisis_narrative(
    user_id: int,
    time_window_days: int = 30,
    ollama_model: str = "llama3.2",
    ollama_host: str = "http://127.0.0.1:11434"
) -> Dict[str, Any]:
    """Gathers multi-source longitudinal history and generates an evidence-grounded crisis narrative synthesis."""
    if not user_id:
        return {"status": "error", "message": "Valid user_id required."}

    # 1. Fetch multi-source records isolated by user_id
    journals = fetch_user_journal_history(user_id, days_limit=time_window_days, entry_limit=10)
    checkins = fetch_user_checkin_history(user_id, limit=20)
    conv_signals = fetch_user_conversational_history(user_id, limit=15)
    trend_data = analyze_trends(checkins, conv_signals)

    total_records = len(journals) + len(conv_signals)
    if total_records < 2:
        return _generate_fallback_narrative(journals, conv_signals, trend_data)

    # 2. Build factual chronology payload for Ollama
    chronology_lines = []
    for j in journals:
        chronology_lines.append(f"[Journal on {j['date']}]: \"{j['content']}\"")
    for c in conv_signals:
        em = f" (Emotions: {c['detected_emotions']})" if c.get('detected_emotions') else ""
        chronology_lines.append(f"[Chat on {c['timestamp'][:16]}]: \"{c['content']}\"{em}")

    timeline_str = "\n".join(chronology_lines)
    trend_summary = f"Numeric Trend: {trend_data.get('trend', 'stable')} (regression slope: {trend_data.get('slope', 0.0)})."

    prompt = (
        f"Timeline of Self-Reported User Entries (Last {time_window_days} Days):\n"
        f"{timeline_str}\n\n"
        f"{trend_summary}\n\n"
        "Synthesize this factual timeline according to your instructions. Respond in strict JSON format."
    )

    # 3. Attempt synthesis via centralized AI provider (Ollama in local dev, Groq in cloud prod)
    ai_resp = ai_provider.chat_completion(
        messages=[{"role": "user", "content": prompt}],
        system_prompt=NARRATIVE_SYSTEM_PROMPT,
        json_mode=True,
        temperature=0.3,
        override_model=ollama_model if (ollama_model and ollama_model != "llama3.2") else None,
        override_base_url=ollama_host if (ollama_host and ollama_host != "http://127.0.0.1:11434") else None
    )

    if not ai_resp.is_fallback:
        parsed = ai_resp.parsed_json or extract_json_from_text(ai_resp.content)
        if parsed and "narrative_summary" in parsed:
            return {
                "status": "success",
                "narrative_summary": str(parsed.get("narrative_summary", "")).strip(),
                "explicitly_reported_concerns": list(parsed.get("explicitly_reported_concerns", []))[:5],
                "recurring_themes": list(parsed.get("recurring_themes", []))[:4],
                "trajectory": str(parsed.get("trajectory", "stable")),
                "ai_observations": str(parsed.get("ai_observations", "")).strip(),
                "data_points_evaluated": total_records,
                "is_ai_generated": True,
                "ai_provider": ai_resp.provider,
                "ai_model": ai_resp.model
            }

    logger.info(f"AI Provider ({ai_resp.provider}) used fallback: {ai_resp.fallback_reason}. Using intelligent fallback.")
    fallback_res = _generate_fallback_narrative(journals, conv_signals, trend_data)
    fallback_res["is_ai_generated"] = False
    fallback_res["ai_provider"] = "fallback"
    return fallback_res
