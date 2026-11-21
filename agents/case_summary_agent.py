# agents/case_summary_agent.py
"""Case Summary Generator Agent.
Generates structured, factual, non-diagnostic longitudinal trauma and wellbeing case summaries
for authorized human counselors using the local Ollama LLM (llama3.2) and an intelligent fallback engine.
"""

import json
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone
import httpx

from extensions import db
from models.user import User
from models.alert import SafetyAlert
from models.case_note import CaseSummary
from agents.monitoring import analyze_trends, fetch_user_checkin_history, fetch_user_conversational_history
from agents.narrative_agent import fetch_user_journal_history, synthesize_crisis_narrative

logger = logging.getLogger(__name__)

SUMMARY_SYSTEM_PROMPT = """You are CareAI's Clinical Case Summary Generator, an AI assistant supporting licensed human mental health counselors.
Your task is to compile a rigorous, factual, structured case summary for a human counselor based strictly on self-reported patient records.

STRICT GUIDELINES:
1. STRICT FACTUAL ACCURACY: Base every statement strictly on the provided timeline of check-ins, journal entries, conversational messages, and alerts.
2. ZERO HALLUCINATION: Do NOT fabricate traumatic backstories, childhood events, or medical conditions not explicitly mentioned by the patient.
3. NO CLINICAL DIAGNOSES: Do NOT assign formal psychiatric disorders (e.g. PTSD, Bipolar, Borderline, Major Depressive Disorder).
4. CLEAR DISTINCTIONS: Clearly distinguish between:
   - What the patient explicitly reported.
   - The observed data trajectory (distress scores, rolling averages).
   - Tentative AI observations offered for counselor consideration.

Respond ONLY with a valid JSON object matching the following 7-section clinical schema:
{
  "title": "Longitudinal Wellbeing & Clinical Case Synthesis",
  "overview": "Concise summary of patient engagement, timeline span, and primary self-reported baseline.",
  "reported_concerns": "Bullet points or paragraph of explicit issues and stressors stated by the patient.",
  "recurring_patterns": "Observed repeating emotional themes (e.g., anxiety spikes, insomnia patterns, social friction).",
  "changes_over_time": "Directional shift in distress levels, coping indicators, and capacity trajectory.",
  "historical_events": "Chronological factual occurrences explicitly documented by the patient.",
  "risk_indicators": "Safety alerts, crisis markers, elevated distress scores (or state that none were flagged).",
  "recommended_review": "Suggested focal points, questions, and areas for the human counselor's upcoming session.",
  "disclaimer": "This summary is an AI-assisted synthesis of self-reported entries and does not constitute a formal psychiatric diagnosis."
}
"""


def _generate_fallback_summary(
    patient: User,
    journals: List[Dict[str, Any]],
    checkins: List[Dict[str, Any]],
    conv_signals: List[Dict[str, Any]],
    alerts: List[SafetyAlert],
    trends: Dict[str, Any]
) -> Dict[str, Any]:
    """Deterministic, structured clinical fallback summary when Ollama is offline."""
    slope = trends.get("slope", 0.0)
    current_score = trends.get("current_score", "N/A")
    trend_direction = trends.get("trend", "stable")

    # Overview
    overview = (
        f"Case summary compiled for Patient #{patient.id} ({patient.username}). "
        f"Records include {len(checkins)} daily check-in(s), {len(journals)} journal reflection(s), "
        f"{len(conv_signals)} conversational distress signal(s), and {len(alerts)} safety alert(s). "
        f"Current numerical distress score is {current_score} with an overall '{trend_direction}' trajectory."
    )

    # Reported concerns
    all_texts = [j["content"] for j in journals] + [c["content"] for c in conv_signals]
    combined_str = " ".join(all_texts).lower()

    concerns = []
    if any(k in combined_str for k in ["exam", "study", "academic"]):
        concerns.append("Academic workload and upcoming examination pressure.")
    if any(k in combined_str for k in ["concentrat", "focus"]):
        concerns.append("Subjective concentration difficulties during periods of high stress.")
    if any(k in combined_str for k in ["sleep", "insomnia", "tired"]):
        concerns.append("Sleep disruption and physical/cognitive fatigue.")
    if any(k in combined_str for k in ["lonel", "isolated", "alone"]):
        concerns.append("Feelings of loneliness and interpersonal isolation.")
    if any(k in combined_str for k in ["plan", "frustrat"]):
        concerns.append("Frustration when plans and expectations do not materialize.")

    if not concerns:
        concerns_text = "Patient has logged routine reflections without chronic distress themes."
    else:
        concerns_text = " • " + "\n • ".join(concerns)

    # Recurring patterns
    if len(journals) >= 2 or len(conv_signals) >= 2:
        recurring_patterns = (
            f"Analysis of {len(all_texts)} narrative entries shows themes centering around stress management "
            f"and self-monitoring. Longitudinal regression slope is {slope:.4f}."
        )
    else:
        recurring_patterns = "Insufficient historical entries to establish persistent multi-week cyclical patterns."

    # Changes over time
    if slope < -0.015:
        changes = f"Favorable trend: Distress scores show a consistent downward trajectory (slope: {slope:.4f})."
    elif slope > 0.015:
        changes = f"Elevated trend: Distress scores have trended upward (slope: {slope:.4f}), indicating escalating stress."
    else:
        changes = f"Stable baseline: Distress scores have remained relatively steady (slope: {slope:.4f})."

    # Historical events
    if journals:
        recent_entry = journals[-1]
        historical_events = f"Most recent self-reported reflection on {recent_entry.get('date')}: \"{recent_entry.get('content')[:140]}...\""
    else:
        historical_events = "No specific historical life events explicitly disclosed in journal records."

    # Risk indicators
    if alerts:
        risk_list = [f"Alert #{a.id} on {a.created_at.strftime('%Y-%m-%d')}: {a.message} (Status: {a.status})" for a in alerts]
        risk_indicators = " • " + "\n • ".join(risk_list)
    else:
        risk_indicators = "No critical crisis alerts or safety triggers logged in system history."

    # Recommended review
    recommendations = (
        "1. Explore the patient's coping strategies for their explicitly stated concerns.\n"
        "2. Review sleep hygiene and daily routine pacing.\n"
        "3. Evaluate social support systems and validate emotional experiences."
    )

    full_markdown = (
        f"# Clinical Case Summary: Patient #{patient.id} ({patient.username})\n\n"
        f"**Date Compiled:** {datetime.now(timezone.utc).strftime('%B %d, %Y')}\n\n"
        f"### 1. Case Overview\n{overview}\n\n"
        f"### 2. Reported Concerns\n{concerns_text}\n\n"
        f"### 3. Recurring Emotional Patterns\n{recurring_patterns}\n\n"
        f"### 4. Significant Changes Over Time\n{changes}\n\n"
        f"### 5. Relevant Historical Events\n{historical_events}\n\n"
        f"### 6. Existing Risk Indicators & Safety Alerts\n{risk_indicators}\n\n"
        f"### 7. Recommended Areas for Human Review\n{recommendations}\n\n"
        f"---\n*Disclaimer: This summary is an AI-assisted synthesis of self-reported entries and does not constitute a formal psychiatric diagnosis.*"
    )

    return {
        "status": "success",
        "title": f"Clinical Case Summary – Patient #{patient.id}",
        "overview": overview,
        "reported_concerns": concerns_text,
        "recurring_patterns": recurring_patterns,
        "changes_over_time": changes,
        "historical_events": historical_events,
        "risk_indicators": risk_indicators,
        "recommended_review": recommendations,
        "full_content": full_markdown,
        "disclaimer": "This summary is an AI-assisted synthesis of self-reported entries and does not constitute a formal psychiatric diagnosis."
    }


def generate_case_summary(
    user_id: int,
    counselor_id: Optional[int] = None,
    ollama_model: str = "llama3.2",
    ollama_host: str = "http://127.0.0.1:11434"
) -> Dict[str, Any]:
    """Generates a structured, evidence-grounded case summary for a patient."""
    patient = User.query.get(user_id)
    if not patient:
        return {"status": "error", "message": f"Patient with ID {user_id} not found."}

    # Fetch longitudinal multi-source data
    journals = fetch_user_journal_history(user_id, days_limit=60, entry_limit=15)
    checkins = fetch_user_checkin_history(user_id, limit=30)
    conv_signals = fetch_user_conversational_history(user_id, limit=20)
    alerts = SafetyAlert.query.filter_by(user_id=user_id).order_by(SafetyAlert.created_at.desc()).limit(10).all()
    trends = analyze_trends(checkins, conv_signals)

    # Format factual payload
    timeline_items = []
    for c in checkins[-7:]:
        timeline_items.append(f"[Checkin {c['timestamp'][:10]}]: Mood={c['mood']}/10, Anxiety={c['anxiety']}/10, Stress={c['stress']}/10, DistressScore={c['distress_score']}")
    for j in journals[-5:]:
        timeline_items.append(f"[Journal {j['date'][:10]}]: \"{j['content']}\"")
    for cs in conv_signals[-5:]:
        timeline_items.append(f"[Chat {cs['timestamp'][:10]}]: \"{cs['content']}\" (Distress: {cs['distress_score']})")

    alerts_str = f"{len(alerts)} alerts logged."
    if alerts:
        alerts_str += " Recent: " + "; ".join([f"[{a.created_at.strftime('%Y-%m-%d')}]: {a.message}" for a in alerts[:3]])

    prompt = (
        f"Patient Profile: ID #{patient.id}, Username: {patient.username}\n"
        f"Timeline of Historical Entries:\n" + "\n".join(timeline_items) + "\n\n"
        f"Numeric Trend: {trends.get('trend')} (Slope: {trends.get('slope')}, Current: {trends.get('current_score')})\n"
        f"Alerts History: {alerts_str}\n\n"
        "Generate a structured, non-hallucinated clinical summary according to your instructions. Respond in strict JSON."
    )

    # Attempt Ollama LLM generation
    try:
        url = f"{ollama_host.rstrip('/')}/api/chat"
        payload = {
            "model": ollama_model,
            "messages": [
                {"role": "system", "content": SUMMARY_SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            "stream": False,
            "format": "json",
            "options": {"temperature": 0.3, "top_p": 0.9, "num_predict": 750}
        }
        with httpx.Client(timeout=40.0) as client:
            resp = client.post(url, json=payload)
            if resp.status_code == 200:
                raw_json = resp.json().get("message", {}).get("content", "")
                parsed = json.loads(raw_json)
                if parsed and "overview" in parsed:
                    overview = str(parsed.get("overview", "")).strip()
                    concerns = str(parsed.get("reported_concerns", "")).strip()
                    patterns = str(parsed.get("recurring_patterns", "")).strip()
                    changes = str(parsed.get("changes_over_time", "")).strip()
                    history = str(parsed.get("historical_events", "")).strip()
                    risks = str(parsed.get("risk_indicators", "")).strip()
                    recs = str(parsed.get("recommended_review", "")).strip()
                    title = str(parsed.get("title", f"Clinical Case Summary – Patient #{patient.id}")).strip()

                    full_md = (
                        f"# {title}\n\n"
                        f"**Patient:** #{patient.id} ({patient.username}) | **Date Compiled:** {datetime.now(timezone.utc).strftime('%B %d, %Y')}\n\n"
                        f"### 1. Case Overview\n{overview}\n\n"
                        f"### 2. Reported Concerns\n{concerns}\n\n"
                        f"### 3. Recurring Emotional Patterns\n{patterns}\n\n"
                        f"### 4. Significant Changes Over Time\n{changes}\n\n"
                        f"### 5. Relevant Historical Events\n{history}\n\n"
                        f"### 6. Existing Risk Indicators & Safety Alerts\n{risks}\n\n"
                        f"### 7. Recommended Areas for Human Review\n{recs}\n\n"
                        f"---\n*Disclaimer: This summary is an AI-assisted synthesis of self-reported entries and does not constitute a formal psychiatric diagnosis.*"
                    )

                    return {
                        "status": "success",
                        "title": title,
                        "overview": overview,
                        "reported_concerns": concerns,
                        "recurring_patterns": patterns,
                        "changes_over_time": changes,
                        "historical_events": history,
                        "risk_indicators": risks,
                        "recommended_review": recs,
                        "full_content": full_md,
                        "disclaimer": "This summary is an AI-assisted synthesis of self-reported entries and does not constitute a formal psychiatric diagnosis."
                    }
    except Exception as e:
        logger.info(f"Ollama case summary generation bypassed or timed out ({e}). Using intelligent fallback.")

    return _generate_fallback_summary(patient, journals, checkins, conv_signals, alerts, trends)
