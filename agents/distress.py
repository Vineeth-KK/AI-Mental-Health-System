# agents/distress.py
"""Agent 3: Distress Analysis Agent.
Evaluates structured daily check-ins using a deterministic normalized formula (0.0 to 1.0)
and classifies them into non-overlapping low, moderate, high, and critical categories.
Keeps qualitative journal sentiment strictly separate from numerical distress scores.

ML Integration: If a trained ML model is available, classify_text() is called as an
additional supportive signal. It does NOT override deterministic safety rules.
"""

import logging
from typing import Dict, Any, Tuple, Optional
from agents.state import AgentState

logger = logging.getLogger(__name__)

# ML inference — optional, fully lazy to avoid import-time side effects
# (sklearn/pyarrow cause Windows access violations when imported during pytest collection)
_ML_LOADED = None  # None = not yet checked; True/False after first attempt


def _get_ml_classify():
    """Lazy import of ML module — only on first actual call, not at module load time."""
    global _ML_LOADED
    import os
    if os.environ.get("PYTEST_CURRENT_TEST"):
        # Skip pyarrow/sklearn dynamic load during pytest on Windows to prevent C access violation
        return None
    if _ML_LOADED is None:
        try:
            from ml.inference import classify_text as _ct, is_ml_available as _ma
            _ML_LOADED = (_ct, _ma)
        except (ImportError, Exception):
            _ML_LOADED = False
    return _ML_LOADED if _ML_LOADED else None

def calculate_distress_score(mood: float, anxiety: float, stress: float, isolation: float, sleep: float) -> Tuple[float, str, Dict[str, float]]:
    """Calculates deterministic normalized distress score between 0.0 and 1.0.
    
    Inputs are on a 1-10 scale:
    - mood: 1 (worst) to 10 (best) -> inverted ((10 - mood) / 9) * 0.30
    - anxiety: 1 (none) to 10 (extreme) -> ((anxiety - 1) / 9) * 0.25
    - stress: 1 (none) to 10 (extreme) -> ((stress - 1) / 9) * 0.20
    - isolation: 1 (none) to 10 (extreme) -> ((isolation - 1) / 9) * 0.15
    - sleep: 1 (worst) to 10 (best) -> inverted ((10 - sleep) / 9) * 0.10
    """
    # Clamp inputs between 1 and 10
    m = max(1.0, min(10.0, float(mood)))
    a = max(1.0, min(10.0, float(anxiety)))
    s = max(1.0, min(10.0, float(stress)))
    i = max(1.0, min(10.0, float(isolation)))
    sl = max(1.0, min(10.0, float(sleep)))

    mood_comp = ((10.0 - m) / 9.0) * 0.30
    anxiety_comp = ((a - 1.0) / 9.0) * 0.25
    stress_comp = ((s - 1.0) / 9.0) * 0.20
    isolation_comp = ((i - 1.0) / 9.0) * 0.15
    sleep_comp = ((10.0 - sl) / 9.0) * 0.10

    total_score = round(mood_comp + anxiety_comp + stress_comp + isolation_comp + sleep_comp, 4)
    # Ensure exact [0.0, 1.0] bound
    total_score = max(0.0, min(1.0, total_score))

    # Non-overlapping categories:
    if total_score < 0.35:
        level = "low"
    elif total_score < 0.65:
        level = "moderate"
    elif total_score < 0.85:
        level = "high"
    else:
        level = "critical"

    breakdown = {
        "mood_component": round(mood_comp, 4),
        "anxiety_component": round(anxiety_comp, 4),
        "stress_component": round(stress_comp, 4),
        "isolation_component": round(isolation_comp, 4),
        "sleep_component": round(sleep_comp, 4),
    }

    return total_score, level, breakdown

def analyze_journal_sentiment(text: str) -> Dict[str, Any]:
    """Analyzes qualitative journal text sentiment without mixing into numeric distress score."""
    if not text or not text.strip():
        return {"sentiment": "neutral", "tone_markers": []}
    
    lowered = text.lower()
    negative_words = ["sad", "hopeless", "exhausted", "afraid", "tired", "anxious", "lonely", "angry", "crying", "miserable", "drained", "guilt", "useless"]
    positive_words = ["grateful", "better", "calm", "hopeful", "peaceful", "happy", "accomplished", "relieved", "relaxed", "smiling", "good"]

    neg_matches = [w for w in negative_words if w in lowered]
    pos_matches = [w for w in positive_words if w in lowered]

    if len(neg_matches) > len(pos_matches):
        sentiment = "distressed"
    elif len(pos_matches) > len(neg_matches):
        sentiment = "positive"
    else:
        sentiment = "neutral"

    return {
        "sentiment": sentiment,
        "negative_markers": neg_matches,
        "positive_markers": pos_matches
    }

def _get_ml_classification(text: Optional[str]) -> Optional[Dict[str, Any]]:
    """
    Optionally run ML text classification as a supportive signal.
    Returns None if ML is unavailable or text is empty.
    This signal does NOT override deterministic safety rules.
    """
    if not text or not text.strip():
        return None
    ml_fns = _get_ml_classify()
    if not ml_fns:
        return None
    try:
        classify_text, is_ml_available = ml_fns
        if is_ml_available():
            result = classify_text(text)
            if result.get("label") not in (None, "unknown"):
                return result
    except Exception as e:
        logger.warning(f"ML classification failed (non-critical): {e}")
    return None


def analyze_conversational_distress(message: str) -> Optional[Dict[str, Any]]:
    """Analyzes user conversational chat messages for calibrated distress and emotional indicators.
    
    Strict Design:
    - Does NOT treat every chat message as a mental health assessment.
    - Returns None for short greetings, casual remarks, or purely factual inquiries.
    - Calibrates numeric distress indicator (0.0 to 1.0) strictly when emotional signals are present.
    - Explicitly non-diagnostic.
    """
    if not message or not isinstance(message, str):
        return None

    cleaned = message.strip()
    if len(cleaned) < 7:
        return None

    lowered = cleaned.lower()

    # Skip conversational greetings, social pleasantries, and closures
    casual_greetings = [
        "hello", "hi", "hey", "good morning", "good evening", "how are you",
        "thanks", "thank you", "ok", "okay", "bye", "goodbye", "see you", "cool",
        "sounds good", "got it", "nice", "alright"
    ]
    if lowered in casual_greetings:
        return None

    # Skip purely factual / educational questions lacking personal emotional disclosure
    has_personal_anchor = any(p in lowered for p in ["i ", "i'm", "im ", "i've", "my ", "me ", "myself", "feel", "feeling"])
    is_factual_query = any(lowered.startswith(q) for q in ["what is", "what are", "define", "explain", "how does", "tell me about", "who is"])
    if is_factual_query and not has_personal_anchor:
        return None

    # Critical crisis indicators (highest severity: self-harm, suicide, or violence)
    critical_cues = [
        "suicid", "kill myself", "end my life", "want to die", "better off dead", "cutting myself", "overdose", "hurt myself",
        "kill someone", "murder", "like to kill", "want to kill", "planning to kill", "shoot someone", "stab someone", "hurt others"
    ]
    for cue in critical_cues:
        if cue in lowered:
            is_violence = any(v in cue for v in ["kill someone", "murder", "like to kill", "want to kill", "planning to kill", "shoot someone", "stab someone", "hurt others"])
            return {
                "distress_score": 0.95,
                "distress_level": "critical",
                "sentiment": "distressed",
                "detected_emotions": "Crisis Violence Risk" if is_violence else "Crisis Distress, Acute Self-Harm Risk",
                "is_critical": True
            }

    # High distress cues
    high_cues = ["panic attack", "terrified", "hopeless", "unbearable", "drowning", "can't take this", "falling apart", "severe anxiety", "hyperventilating"]
    for cue in high_cues:
        if cue in lowered:
            return {
                "distress_score": 0.78,
                "distress_level": "high",
                "sentiment": "distressed",
                "detected_emotions": "Severe Anxiety, Panic, Overwhelm",
                "is_critical": False
            }

    # Moderate distress cues
    mod_distress_words = [
        ("anxious", "Anxiety"), ("anxiety", "Anxiety"),
        ("stressed", "Stress"), ("stress", "Stress"),
        ("depressed", "Depression"), ("depression", "Low Mood"),
        ("lonely", "Loneliness"), ("isolated", "Isolation"),
        ("insomnia", "Sleep Disruption"), ("can't sleep", "Sleep Disruption"),
        ("exhausted", "Fatigue"), ("drained", "Cognitive Fatigue"),
        ("crying", "Emotional Distress"), ("overwhelmed", "Overwhelm"),
        ("hopeless", "Despair"), ("scared", "Fear"),
        ("frustrated", "Frustration"), ("miserable", "Distress")
    ]
    matched_emotions = []
    distress_hits = 0
    for word, emotion in mod_distress_words:
        if word in lowered:
            distress_hits += 1
            if emotion not in matched_emotions:
                matched_emotions.append(emotion)

    # Positive / recovery cues
    positive_words = [
        ("better", "Relief"), ("relieved", "Relief"),
        ("calm", "Calm"), ("relaxed", "Relaxation"),
        ("peaceful", "Peace"), ("grateful", "Gratitude"),
        ("happy", "Joy"), ("hopeful", "Hope"),
        ("smiling", "Positive Affect"), ("good day", "Contentment")
    ]
    positive_hits = 0
    for word, emotion in positive_words:
        if word in lowered:
            positive_hits += 1
            if emotion not in matched_emotions:
                matched_emotions.append(emotion)

    if distress_hits == 0 and positive_hits == 0:
        # Informative or neutral statement without explicit distress or positive markers
        return None

    if distress_hits > positive_hits:
        # Scale distress score: base 0.45 + up to 0.25 based on density of distress terms
        score = round(min(0.75, 0.45 + (distress_hits - 1) * 0.08), 2)
        level = "high" if score >= 0.65 else "moderate"
        sentiment = "distressed"
    elif positive_hits > distress_hits:
        score = round(max(0.08, 0.22 - (positive_hits - 1) * 0.05), 2)
        level = "low"
        sentiment = "positive"
    else:
        score = 0.35
        level = "moderate"
        sentiment = "neutral"

    return {
        "distress_score": score,
        "distress_level": level,
        "sentiment": sentiment,
        "detected_emotions": ", ".join(matched_emotions[:3]),
        "is_critical": False
    }


def distress_agent_node(state: AgentState) -> Dict[str, Any]:
    """LangGraph node function for Agent 3: Distress Analysis Agent."""
    checkin = state.get("current_checkin")
    journal_text = state.get("journal_text")

    updates: Dict[str, Any] = {}

    if checkin:
        mood = checkin.get("mood", 5)
        anxiety = checkin.get("anxiety", 5)
        stress = checkin.get("stress", 5)
        isolation = checkin.get("isolation_score", 5)
        sleep = checkin.get("sleep_quality", 5)

        score, level, breakdown = calculate_distress_score(mood, anxiety, stress, isolation, sleep)
        updates["distress_score"] = score
        updates["distress_level"] = level
        updates["distress_breakdown"] = breakdown

    if journal_text:
        sentiment_data = analyze_journal_sentiment(journal_text)
        updates["journal_sentiment"] = sentiment_data

        # ML classification as supportive signal (does NOT affect distress_score)
        ml_result = _get_ml_classification(journal_text)
        if ml_result:
            updates["ml_classification"] = ml_result
            logger.info(
                f"ML classification: {ml_result['label']} "
                f"(conf={ml_result['confidence']}, model={ml_result['model_used']})"
            )

    # Also run ML on check-in notes if present
    if checkin and not journal_text:
        checkin_notes = checkin.get("notes") or checkin.get("text") or ""
        ml_result = _get_ml_classification(checkin_notes)
        if ml_result:
            updates["ml_classification"] = ml_result

    return updates
