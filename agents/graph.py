# agents/graph.py
"""LangGraph coordination engine for the six-agent AI system."""

import logging
from typing import Dict, Any, Literal
from datetime import datetime, timezone
from langgraph.graph import StateGraph, END

from agents.state import AgentState
from agents.distress import distress_agent_node
from agents.monitoring import monitoring_agent_node
from agents.safety import safety_agent_node
from agents.recommendation import recommendation_agent_node
from agents.conversation import conversation_agent_node
from agents.journal_agent import journal_agent_node

logger = logging.getLogger(__name__)

# --- Orchestrator Entry Node ---
def orchestrator_entry_node(state: AgentState) -> Dict[str, Any]:
    """Agent 1: Orchestrator entry node. Validates and initializes shared state."""
    updates: Dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "priority_override": False
    }
    if "action_type" not in state:
        if state.get("current_checkin"):
            updates["action_type"] = "checkin"
        elif state.get("journal_text"):
            updates["action_type"] = "journal"
        elif state.get("messages"):
            updates["action_type"] = "chat"
        else:
            updates["action_type"] = "full_assessment"
    return updates

# --- Safety Override Node ---
def safety_override_node(state: AgentState) -> Dict[str, Any]:
    """Safety priority termination node: overrides all responses with crisis resources."""
    alert = state.get("safety_alert", {})
    crisis_msg = alert.get("crisis_resources") or "Immediate emergency assistance is recommended. Please contact 988 or 14416."
    return {
        "response": crisis_msg,
        "priority_override": True
    }

# --- Orchestrator Synthesis Node ---
def orchestrator_synthesis_node(state: AgentState) -> Dict[str, Any]:
    """Agent 1: Orchestrator synthesis node. Combines outputs of all active agents into a unified result."""
    action = state.get("action_type", "")
    distress_level = state.get("distress_level")
    distress_score = state.get("distress_score")
    trends = state.get("trend_analysis", {})
    recommendations = state.get("recommendations", [])
    response = state.get("response")

    # If action was checkin/assessment and conversation agent wasn't called, synthesize a supportive summary
    if action in ["checkin", "full_assessment"] and not response:
        score_txt = f"{distress_score:.2f}" if distress_score is not None else "N/A"
        level_txt = distress_level.capitalize() if distress_level else "Evaluated"
        trend_summary = trends.get("summary", "")

        rec_titles = [r["title"] for r in recommendations[:2]] if recommendations else []
        rec_str = f" Recommended activities for you: {', '.join(rec_titles)}." if rec_titles else ""

        response = (
            f"Thank you for logging your daily check-in. Your current distress score is {score_txt} ({level_txt}). "
            f"{trend_summary}{rec_str}"
        )

    elif action == "journal":
        analysis = state.get("journal_analysis", {})
        if analysis and analysis.get("wellbeing_feedback"):
            response = analysis["wellbeing_feedback"]
        elif not response:
            response = "Your reflection has been safely recorded. Putting thoughts into words is a grounding practice."

    return {
        "response": response,
        "priority_override": state.get("priority_override", False)
    }

# --- Conditional Edge Functions ---
def route_from_entry(state: AgentState) -> Literal["distress_agent", "safety_agent"]:
    """Routes to distress agent if check-in metrics or journal text present; otherwise directly to safety."""
    action = state.get("action_type")
    if action in ["checkin", "journal", "full_assessment"] or state.get("current_checkin") or state.get("journal_text"):
        return "distress_agent"
    return "safety_agent"

def route_from_safety(state: AgentState) -> Literal["safety_override", "conversation_agent", "monitoring_agent"]:
    """Enforces safety priority routing. If urgent, immediately overrides."""
    if state.get("priority_override"):
        return "safety_override"
    if state.get("action_type") == "chat":
        return "conversation_agent"
    return "monitoring_agent"

# --- Build LangGraph Workflow ---
def create_mental_health_graph():
    builder = StateGraph(AgentState)

    # Add Nodes
    builder.add_node("orchestrator_entry", orchestrator_entry_node)
    builder.add_node("distress_agent", distress_agent_node)
    builder.add_node("safety_agent", safety_agent_node)
    builder.add_node("safety_override", safety_override_node)
    builder.add_node("monitoring_agent", monitoring_agent_node)
    builder.add_node("recommendation_agent", recommendation_agent_node)
    builder.add_node("conversation_agent", conversation_agent_node)
    builder.add_node("journal_agent", journal_agent_node)
    builder.add_node("orchestrator_synthesis", orchestrator_synthesis_node)

    # Set Entry Point
    builder.set_entry_point("orchestrator_entry")

    # Conditional Routing from Entry
    builder.add_conditional_edges(
        "orchestrator_entry",
        route_from_entry,
        {
            "distress_agent": "distress_agent",
            "safety_agent": "safety_agent"
        }
    )

    # Distress Agent always passes to Safety Agent (so safety agent can evaluate calculated score)
    builder.add_edge("distress_agent", "safety_agent")

    # Conditional Routing from Safety Agent
    builder.add_conditional_edges(
        "safety_agent",
        route_from_safety,
        {
            "safety_override": "safety_override",
            "conversation_agent": "recommendation_agent", # RAG context is fetched before conversation
            "monitoring_agent": "monitoring_agent"
        }
    )

    # Safety Override terminates immediately at END
    builder.add_edge("safety_override", END)

    # Flow for checkin/journal: monitoring -> recommendation -> synthesis
    builder.add_edge("monitoring_agent", "recommendation_agent")
    
    # Route after recommendation: chat -> conversation_agent, journal -> journal_agent, checkin -> synthesis
    def route_after_recommendation(state: AgentState) -> Literal["conversation_agent", "journal_agent", "orchestrator_synthesis"]:
        if state.get("action_type") == "chat":
            return "conversation_agent"
        elif state.get("action_type") == "journal" or state.get("journal_text"):
            return "journal_agent"
        return "orchestrator_synthesis"

    builder.add_conditional_edges(
        "recommendation_agent",
        route_after_recommendation,
        {
            "conversation_agent": "conversation_agent",
            "journal_agent": "journal_agent",
            "orchestrator_synthesis": "orchestrator_synthesis"
        }
    )

    builder.add_edge("conversation_agent", "orchestrator_synthesis")
    builder.add_edge("journal_agent", "orchestrator_synthesis")
    builder.add_edge("orchestrator_synthesis", END)

    return builder.compile()

# Compile global multi-agent graph
mental_health_graph = create_mental_health_graph()
