# agents/safety.py
"""Agent 5: Safety Agent.
Performs deterministic safety evaluations across user journal reflections,
chat messages, and structured distress scores.
If urgent safety risks are detected, sets priority_override=True, overrides
standard responses, and provides verified emergency helpline resources.
"""

import re
from typing import Dict, Any, Tuple, Optional
from flask import has_app_context
from agents.state import AgentState
from extensions import db
from models.alert import SafetyAlert

# High-risk trigger phrases indicating severe self-harm or suicidal ideation
CRITICAL_SAFETY_PATTERNS = [
    r"\b(suicide|suicidal|kill myself|end my life|want to die|better off dead)\b",
    r"\b(hang myself|slit my wrist|overdose|jump off|shoot myself)\b",
    r"\b(no reason to live|don'?t want to live anymore|goodbye world|can'?t go on)\b",
    r"\b(hurt myself|self[- ]harm|cutting myself|take all my pills)\b",
    r"\b(plan to end it|ending it tonight|saying goodbye forever)\b"
]

# Moderate concern indicators
MODERATE_SAFETY_PATTERNS = [
    r"\b(hopeless|unbearable pain|drowning in despair|can'?t take this pain)\b",
    r"\b(completely alone|nobody cares about me|worthless|burden to everyone)\b",
    r"\b(panic attack|hyperventilating|losing control|going crazy)\b"
]

CRISIS_RESOURCES_MESSAGE = """⚠️ **Immediate Support Available — You Are Not Alone**

If you are experiencing severe distress or thoughts of self-harm, please know that compassionate, confidential support is available right now:

- **United States & Canada**: Call or text **988** (available 24/7, toll-free) or text **HOME** to **741741**.
- **India**: Call **14416** (Tele-MANAS, 24/7 National Mental Health Helpline) or **+91 9999 666 555** (Vandrevala Foundation).
- **United Kingdom**: Call **111** (NHS Mental Health) or **116 123** (Samaritans).
- **International Directory**: Visit [Befrienders Worldwide](https://www.befrienders.org) to find immediate local crisis centres.

Please reach out to a trusted professional, doctor, or family member, or visit your nearest emergency room. Your life and wellbeing have value, and help is available."""

def evaluate_text_safety(text: str) -> Tuple[bool, str, str]:
    """Evaluates text for urgent safety concerns.
    Returns: (is_urgent, risk_level, reason)
    """
    if not text or not text.strip():
        return False, "safe", ""

    # Check critical patterns first
    for pattern in CRITICAL_SAFETY_PATTERNS:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return True, "critical", f"Critical self-harm or crisis indicator detected: '{match.group(0)}'"

    # Check moderate concern patterns
    for pattern in MODERATE_SAFETY_PATTERNS:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return False, "moderate", f"Elevated distress marker detected: '{match.group(0)}'"

    return False, "safe", ""

def evaluate_safety(
    text: Optional[str] = None,
    distress_score: Optional[float] = None,
    distress_level: Optional[str] = None
) -> Dict[str, Any]:
    """Combines text safety assessment and numeric distress score threshold."""
    is_urgent, risk_level, reason = evaluate_text_safety(text or "")

    # Score-based safety check: Critical distress score (>= 0.85) triggers safety intervention
    if distress_score is not None and distress_score >= 0.85:
        is_urgent = True
        risk_level = "critical"
        score_reason = f"Critical composite distress score ({distress_score:.2f} >= 0.85)"
        reason = f"{reason}; {score_reason}" if reason else score_reason
    elif distress_level == "critical":
        is_urgent = True
        risk_level = "critical"
        score_reason = "Critical check-in level reported"
        reason = f"{reason}; {score_reason}" if reason else score_reason

    recommended_action = "Escalate to emergency crisis resources" if is_urgent else ("Monitor wellbeing" if risk_level != "safe" else "Standard workflow")

    return {
        "is_urgent": is_urgent,
        "risk_level": risk_level,
        "flag_reason": reason,
        "recommended_action": recommended_action,
        "crisis_resources": CRISIS_RESOURCES_MESSAGE if is_urgent else None
    }

def record_safety_alert_if_urgent(user_id: Optional[int], alert_data: Dict[str, Any]) -> Optional[int]:
    """Persists SafetyAlert to database when urgent safety risk is identified."""
    if not user_id or not alert_data.get("is_urgent") or not has_app_context():
        return None
    try:
        alert = SafetyAlert(
            user_id=user_id,
            message=alert_data.get("flag_reason", "Urgent safety risk detected")[:250],
            risk_level=alert_data.get("risk_level", "critical"),
            action_taken="Emergency crisis resources displayed and safety override activated."
        )
        db.session.add(alert)
        db.session.commit()

        # Trigger counselor alert dispatch
        try:
            from services.notifier import dispatch_counselor_alert
            dispatch_counselor_alert(
                alert_id=alert.id,
                user_id=user_id,
                risk_level=alert_data.get("risk_level", "critical"),
                reason=alert_data.get("flag_reason", "Urgent safety risk detected")
            )
        except Exception:
            pass

        return alert.id
    except Exception:
        db.session.rollback()
        return None

def safety_agent_node(state: AgentState) -> Dict[str, Any]:
    """LangGraph node function for Agent 5: Safety Agent."""
    # Gather potential text sources
    texts = []
    if state.get("journal_text"):
        texts.append(state["journal_text"])
    
    messages = state.get("messages", [])
    if messages:
        # Check last user message
        last_user_msg = next((m["content"] for m in reversed(messages) if m.get("role") == "user"), "")
        if last_user_msg:
            texts.append(last_user_msg)

    combined_text = " ".join(texts)
    distress_score = state.get("distress_score")
    distress_level = state.get("distress_level")

    alert = evaluate_safety(combined_text, distress_score, distress_level)
    user_id = state.get("user_id")

    if alert["is_urgent"]:
        record_safety_alert_if_urgent(user_id, alert)
        return {
            "safety_alert": alert,
            "priority_override": True,
            "response": alert["crisis_resources"]
        }

    return {
        "safety_alert": alert,
        "priority_override": False
    }
