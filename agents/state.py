# agents/state.py
"""Shared AgentState definition for the LangGraph multi-agent coordination system."""

from typing import TypedDict, List, Dict, Any, Optional

class AgentState(TypedDict, total=False):
    user_id: int
    user_name: str
    action_type: str  # 'checkin', 'journal', 'chat', 'full_assessment'
    messages: List[Dict[str, str]]
    current_checkin: Optional[Dict[str, Any]]
    historical_checkins: List[Dict[str, Any]]
    journal_text: Optional[str]
    journal_sentiment: Optional[Dict[str, Any]]
    distress_score: Optional[float]
    distress_level: Optional[str]  # 'low', 'moderate', 'high', 'critical'
    distress_breakdown: Optional[Dict[str, Any]]
    safety_alert: Optional[Dict[str, Any]]
    trend_analysis: Optional[Dict[str, Any]]
    rag_context: Optional[List[Dict[str, Any]]]
    recommendations: Optional[List[Dict[str, Any]]]
    response: Optional[str]
    priority_override: bool
    timestamp: str
    ml_classification: Optional[Dict[str, Any]]  # Supportive ML signal (not clinical diagnosis)
