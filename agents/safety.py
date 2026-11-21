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
CRITICAL_SELF_HARM_PATTERNS = [
    r"\b(suicide|suicidal|kill myself|end my life|want to die|better off dead)\b",
    r"\b(hang myself|slit my wrist|overdose|jump off|shoot myself)\b",
    r"\b(no reason to live|don'?t want to live anymore|goodbye world|can'?t go on)\b",
    r"\b(hurt myself|self[- ]harm|cutting myself|take all my pills)\b",
    r"\b(plan to end it|ending it tonight|saying goodbye forever)\b"
]

# High-risk trigger phrases indicating violence, murder, or harm to others
CRITICAL_VIOLENCE_PATTERNS = [
    r"\b(kill (someone|people|somebody|them|her|him|everyone)|murder|assassinate|slaughter)\b",
    r"\b(want to kill|like to kill|planning to kill|going to kill|gonna kill)\b",
    r"\b(shoot (someone|people|up a|her|him|them)|stab (someone|people|her|him|them))\b",
    r"\b(hurt (someone|people|others)|harm (someone|people|others)|attack (someone|people))\b"
]

# High-risk trigger phrases for dangerous weapon/explosive creation, illegal synthesis, cyber intrusions, or malware creation
CRITICAL_DANGEROUS_PATTERNS = [
    r"\b(how to (make|build|create|synthesize|assemble|manufacture)|instructions (for|to) (make|build|create))\s+(a\s+)?(pipe\s+bomb|bomb|explosive|dirty bomb|molotov( cocktail)?|biological weapon|nerve agent|chemical weapon|ied|meth|methamphetamine|fentanyl)\b",
    r"\b(build a bomb|make a bomb|synthesize (methamphetamine|fentanyl|sarin|ricin))\b",
    r"\b(how to (hack|break into|spy on|compromise))\s+(my\s+)?(roommate'?s?|friend'?s?|boss'?s?|someone'?s?|ex'?s?|wife'?s?|husband'?s?|their|another person'?s?|anyone'?s?)\s+(email|phone|account|password|social media|whatsapp|instagram|facebook|computer)\b",
    r"\b(write|create|code|generate|build)\s+(a\s+)?(keylogger|ransomware|trojan|spyware|stealer|malware)\s*(script|tool|code|to steal|to hack)?\b",
    r"\b(steal|crack|brute[- ]force)\s+(someone'?s?|their|people'?s?)\s+(passwords?|credentials?|credit cards?)\b"
]

CRITICAL_SAFETY_PATTERNS = CRITICAL_SELF_HARM_PATTERNS + CRITICAL_VIOLENCE_PATTERNS + CRITICAL_DANGEROUS_PATTERNS

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

CRITICAL_VIOLENCE_MESSAGE = """⚠️ **Safety Notice: Crisis & De-escalation Support**

I cannot assist with thoughts, plans, or acts of violence, harm toward others, or illegal activities.

If you are experiencing intense anger, violent impulses, or emotional overwhelm, please take a step back and connect with support:
- **Emergency Services**: If anyone is in immediate physical danger, please call your local emergency services immediately (**911** in US/Canada, **112** in India & Europe, **999** in UK).
- **Crisis Helplines**:
  - **USA & Canada**: Call or text **988** (available 24/7, free & confidential)
  - **India**: Call **14416** (Tele-MANAS) or **1800-599-0019**
  - **UK**: Call **111** (NHS Mental Health)
  - **Crisis Text Line**: Text **HOME** to **741741**

Please reach out to a licensed counselor, crisis specialist, or emergency professional who can provide a safe space to de-escalate and process these feelings."""

DANGEROUS_CONTENT_MESSAGE = """⚠️ **Safety Notice: Content Policy & Safe Usage**

I cannot provide instructions, code, or actionable assistance for weapons creation, explosives, synthesizing harmful substances, or carrying out unauthorized cyber intrusions, account hacking, or malware development.

If you are researching cybersecurity concepts, defense mechanisms, or safety protocols from an educational or defensive standpoint, I am happy to discuss:
- Cybersecurity best practices, ethical security analysis, and defensive protection
- Chemical safety principles and hazard prevention
- Digital privacy, identity security, and multi-factor authentication"""

def evaluate_text_safety(text: str) -> Tuple[bool, str, str]:
    """Evaluates text for urgent safety concerns.
    Returns: (is_urgent, risk_level, reason)
    """
    if not text or not text.strip():
        return False, "safe", ""

    # Check violence patterns
    for pattern in CRITICAL_VIOLENCE_PATTERNS:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return True, "critical", f"Critical violent or harm-to-others indicator detected: '{match.group(0)}'"

    # Check critical self-harm patterns
    for pattern in CRITICAL_SELF_HARM_PATTERNS:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return True, "critical", f"Critical self-harm or crisis indicator detected: '{match.group(0)}'"

    # Check dangerous instruction patterns (weapons, explosives, hacking, malware)
    for pattern in CRITICAL_DANGEROUS_PATTERNS:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return True, "critical", f"Dangerous or unauthorized activity indicator detected: '{match.group(0)}'"

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

    # Determine which crisis resources/refusal message to show (dangerous vs violence vs self-harm)
    is_dangerous = bool(text and any(re.search(p, text, re.IGNORECASE) for p in CRITICAL_DANGEROUS_PATTERNS))
    is_violence = bool(text and any(re.search(p, text, re.IGNORECASE) for p in CRITICAL_VIOLENCE_PATTERNS))

    if is_dangerous:
        crisis_msg = DANGEROUS_CONTENT_MESSAGE
    elif is_violence:
        crisis_msg = CRITICAL_VIOLENCE_MESSAGE
    else:
        crisis_msg = CRISIS_RESOURCES_MESSAGE

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
        "crisis_resources": crisis_msg if is_urgent else None
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
