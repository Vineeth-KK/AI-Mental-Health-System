# agents/monitoring.py
"""Agent 4: Monitoring Agent.
Retrieves historical check-ins from the database, calculates rolling averages,
performs regression / slope analysis on distress trends over time,
and classifies trajectory into 'improving', 'stable', or 'worsening'.
"""

from typing import List, Dict, Any, Optional
from flask import has_app_context
import numpy as np
from datetime import datetime, timezone, timedelta
from agents.state import AgentState
from models.checkin import CheckIn
from models.chat import ChatMessage
from agents.distress import calculate_distress_score

def analyze_trends(
    checkins: List[Dict[str, Any]],
    conversational_signals: Optional[List[Dict[str, Any]]] = None
) -> Dict[str, Any]:
    """Analyzes a chronological list of check-in dictionaries to compute
    rolling metrics, slopes, and overall mental health trajectory.
    Optionally incorporates longitudinal conversational distress signals.
    """
    if not checkins:
        conv_summary = None
        if conversational_signals:
            c_scores = [c["distress_score"] for c in conversational_signals if c.get("distress_score") is not None]
            if c_scores:
                conv_summary = {
                    "count": len(c_scores),
                    "avg_distress": round(float(np.mean(c_scores)), 3),
                    "latest_distress": round(float(c_scores[-1]), 3)
                }
        return {
            "status": "insufficient_data",
            "trend": "insufficient_data",
            "summary": "At least 3 daily check-ins are required to establish a reliable trend trajectory.",
            "rolling_averages": {},
            "data_points_count": 0,
            "conversational_distress": conv_summary
        }

    # Ensure chronological sorting by timestamp or id
    def _sort_key(c):
        ts = c.get("timestamp")
        if isinstance(ts, datetime):
            return ts.timestamp()
        if isinstance(ts, str) and ts:
            try:
                return datetime.fromisoformat(ts.replace("Z", "+00:00")).timestamp()
            except Exception:
                pass
        return float(c.get("id") or 0)

    try:
        sorted_checkins = sorted(checkins, key=_sort_key)
    except Exception:
        sorted_checkins = list(checkins)

    # Ensure checkins have calculated distress scores
    processed = []
    for c in sorted_checkins:
        score = c.get("distress_score")
        if score is None:
            m = c.get("mood", 5)
            a = c.get("anxiety", 5)
            s = c.get("stress", 5)
            i = c.get("isolation_score", 5)
            sl = c.get("sleep_quality", 5)
            score, _, _ = calculate_distress_score(m, a, s, i, sl)
        
        processed.append({
            "timestamp": c.get("timestamp"),
            "mood": float(c.get("mood", 5)),
            "anxiety": float(c.get("anxiety", 5)),
            "stress": float(c.get("stress", 5)),
            "sleep_quality": float(c.get("sleep_quality", 5)),
            "isolation_score": float(c.get("isolation_score", 5)),
            "distress_score": float(score)
        })

    scores = [p["distress_score"] for p in processed]
    count = len(scores)

    # Rolling averages
    last_7 = scores[-7:] if count >= 7 else scores
    last_30 = scores[-30:] if count >= 30 else scores

    avg_7 = float(np.mean(last_7))
    avg_30 = float(np.mean(last_30))
    current_score = scores[-1]

    # Metrics rolling averages
    metric_averages = {
        "mood": round(float(np.mean([p["mood"] for p in processed[-7:]])), 2),
        "anxiety": round(float(np.mean([p["anxiety"] for p in processed[-7:]])), 2),
        "stress": round(float(np.mean([p["stress"] for p in processed[-7:]])), 2),
        "sleep": round(float(np.mean([p["sleep_quality"] for p in processed[-7:]])), 2),
        "isolation": round(float(np.mean([p["isolation_score"] for p in processed[-7:]])), 2),
        "distress_score_7d": round(avg_7, 3),
        "distress_score_30d": round(avg_30, 3)
    }

    # Linear regression slope and trajectory evaluation
    # Requires at least 3 data points to establish a genuine longitudinal trajectory
    if count >= 3:
        x = np.arange(count)
        y = np.array(scores)
        slope, _ = np.polyfit(x, y, 1)
        slope = float(slope)

        # Net displacement between early points and recent points
        half = max(1, count // 2)
        early_mean = float(np.mean(scores[:half]))
        recent_mean = float(np.mean(scores[-half:]))
        net_delta = recent_mean - early_mean
        
        # In distress score: higher is worse (0.0=best, 1.0=critical).
        # Negative slope & negative delta means distress is decreasing -> improving
        # Positive slope & positive delta means distress is increasing -> worsening
        if (slope <= -0.02 and net_delta <= -0.04) or (slope <= -0.035):
            trend = "improving"
            insight = "Your distress levels show a steady downward trend over recent entries. Well done on taking positive steps!"
        elif (slope >= 0.02 and net_delta >= 0.04) or (slope >= 0.035):
            trend = "worsening"
            insight = "Your indicators show an upward creep in stress or anxiety recently. Consider exploring the suggested calming activities."
        else:
            trend = "stable"
            insight = "Your mental wellbeing indicators have been relatively steady and balanced across recent entries."
        status = "success"
    else:
        slope = 0.0
        trend = "insufficient_data"
        status = "insufficient_data"
        insight = f"Logged {count} check-in{'s' if count != 1 else ''}. At least 3 daily check-ins are required to establish a meaningful personal trajectory."

    # Conversational distress synthesis if signals present
    conv_distress = None
    if conversational_signals:
        c_scores = [c["distress_score"] for c in conversational_signals if c.get("distress_score") is not None]
        if c_scores:
            emotions = []
            for c in conversational_signals:
                if c.get("detected_emotions"):
                    emotions.extend([e.strip() for e in str(c["detected_emotions"]).split(",") if e.strip()])
            conv_distress = {
                "count": len(c_scores),
                "avg_distress": round(float(np.mean(c_scores)), 3),
                "recent_distress": round(float(c_scores[-1]), 3),
                "recurring_emotions": list(dict.fromkeys(emotions))[:4]
            }

    return {
        "status": status,
        "trend": trend,
        "slope": round(slope, 4),
        "current_score": round(current_score, 3),
        "rolling_averages": metric_averages,
        "summary": insight,
        "data_points_count": count,
        "conversational_distress": conv_distress
    }

def fetch_user_checkin_history(user_id: int, limit: int = 30) -> List[Dict[str, Any]]:
    """Fetches user's checkin history from database isolated by user_id."""
    if not has_app_context() or not user_id:
        return []
    records = (
        CheckIn.query.filter_by(user_id=user_id)
        .order_by(CheckIn.timestamp.asc())
        .limit(limit)
        .all()
    )
    history = []
    for r in records:
        history.append({
            "id": r.id,
            "mood": r.mood,
            "anxiety": r.anxiety,
            "stress": r.stress,
            "sleep_quality": r.sleep_quality,
            "isolation_score": r.isolation_score,
            "distress_score": r.distress_score,
            "distress_level": r.distress_level,
            "timestamp": r.timestamp.isoformat() if r.timestamp else datetime.now(timezone.utc).isoformat()
        })
    return history

def fetch_user_conversational_history(user_id: int, limit: int = 30) -> List[Dict[str, Any]]:
    """Fetches user's conversational distress signals from chat_messages table."""
    if not has_app_context() or not user_id:
        return []
    try:
        records = (
            ChatMessage.query.filter(
                ChatMessage.user_id == user_id,
                ChatMessage.role == "user",
                ChatMessage.distress_score.isnot(None)
            )
            .order_by(ChatMessage.timestamp.asc())
            .limit(limit)
            .all()
        )
        return [
            {
                "id": r.id,
                "content": r.content,
                "distress_score": r.distress_score,
                "sentiment": r.sentiment,
                "detected_emotions": r.detected_emotions,
                "timestamp": r.timestamp.isoformat() if r.timestamp else datetime.now(timezone.utc).isoformat()
            }
            for r in records
        ]
    except Exception:
        return []

def monitoring_agent_node(state: AgentState) -> Dict[str, Any]:
    """LangGraph node function for Agent 4: Monitoring Agent."""
    user_id = state.get("user_id")
    historical = state.get("historical_checkins", [])

    # If historical data not passed in state, query DB if user_id present
    if not historical and user_id:
        try:
            historical = fetch_user_checkin_history(user_id)
        except Exception:
            historical = []

    conversational_signals = []
    if user_id:
        try:
            conversational_signals = fetch_user_conversational_history(user_id)
        except Exception:
            conversational_signals = []

    # If current checkin was just submitted, append to historical for live trend update
    current = state.get("current_checkin")
    if current:
        historical_copy = list(historical)
        item = dict(current)
        if "distress_score" in state:
            item["distress_score"] = state["distress_score"]
        if "timestamp" not in item:
            item["timestamp"] = datetime.now(timezone.utc).isoformat()
        historical_copy.append(item)
        trends = analyze_trends(historical_copy, conversational_signals)
    else:
        trends = analyze_trends(historical, conversational_signals)

    return {"trend_analysis": trends}
