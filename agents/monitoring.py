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
from agents.distress import calculate_distress_score

def analyze_trends(checkins: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyzes a chronological list of check-in dictionaries to compute
    rolling metrics, slopes, and overall mental health trajectory.
    """
    if not checkins:
        return {
            "status": "insufficient_data",
            "trend": "stable",
            "summary": "No historical check-in data recorded yet.",
            "rolling_averages": {},
            "data_points_count": 0
        }

    # Ensure checkins have calculated distress scores
    processed = []
    for c in checkins:
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

    # Linear regression slope if at least 2 data points
    if count >= 2:
        x = np.arange(count)
        y = np.array(scores)
        slope, _ = np.polyfit(x, y, 1)
        slope = float(slope)
        
        # In distress score: higher is worse.
        # Negative slope means distress is decreasing -> improving
        # Positive slope means distress is increasing -> worsening
        if slope < -0.015:
            trend = "improving"
            insight = "Your distress levels show a steady downward trend over recent entries. Well done on taking positive steps!"
        elif slope > 0.015:
            trend = "worsening"
            insight = "Your indicators show an upward creep in stress or anxiety recently. Consider exploring the suggested calming activities."
        else:
            trend = "stable"
            insight = "Your mental wellbeing indicators have been relatively steady over recent entries."
    else:
        slope = 0.0
        trend = "stable"
        insight = "Initial check-in logged. Regular daily check-ins will help identify meaningful personal patterns."

    return {
        "status": "success",
        "trend": trend,
        "slope": round(slope, 4),
        "current_score": round(current_score, 3),
        "rolling_averages": metric_averages,
        "summary": insight,
        "data_points_count": count
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
        trends = analyze_trends(historical_copy)
    else:
        trends = analyze_trends(historical)

    return {"trend_analysis": trends}
