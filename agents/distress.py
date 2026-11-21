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
    if _ML_LOADED is None:
        try:
            from ml.inference import classify_text as _ct, is_ml_available as _ma
            _ML_LOADED = (_ct, _ma)
        except ImportError:
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
