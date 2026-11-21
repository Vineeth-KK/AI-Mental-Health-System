# agents/orchestrator.py
"""Agent 1: Orchestrator Agent.
Central interface to the multi-agent workflow. Coordinates execution of all 6 agents,
manages AgentState across LangGraph nodes, and provides clean high-level methods
for Flask routes.
"""

from typing import Dict, Any, List, Optional
from agents.graph import mental_health_graph
from agents.state import AgentState

class Orchestrator:
    def __init__(self):
        self.graph = mental_health_graph

    def run_workflow(self, initial_state: AgentState) -> AgentState:
        """Executes the coordinated LangGraph workflow."""
        result = self.graph.invoke(initial_state)
        return result

    def handle_checkin(
        self,
        user_id: int,
        user_name: str,
        checkin_data: Dict[str, Any],
        historical_checkins: Optional[List[Dict[str, Any]]] = None
    ) -> AgentState:
        """Processes a daily check-in through Distress, Safety, Monitoring, and Recommendation agents."""
        state: AgentState = {
            "user_id": user_id,
            "user_name": user_name,
            "action_type": "checkin",
            "current_checkin": checkin_data,
            "historical_checkins": historical_checkins or [],
            "messages": [],
            "priority_override": False
        }
        return self.run_workflow(state)

    def handle_journal(
        self,
        user_id: int,
        user_name: str,
        content: str
    ) -> AgentState:
        """Processes a journal entry through Safety, Sentiment, and Recommendation agents."""
        state: AgentState = {
            "user_id": user_id,
            "user_name": user_name,
            "action_type": "journal",
            "journal_text": content,
            "messages": [],
            "priority_override": False
        }
        return self.run_workflow(state)

    def handle_chat(
        self,
        user_id: int,
        user_name: str,
        messages: List[Dict[str, str]],
        current_checkin: Optional[Dict[str, Any]] = None
    ) -> AgentState:
        """Processes a chat turn through Safety, RAG Recommendation, and Conversation agents."""
        state: AgentState = {
            "user_id": user_id,
            "user_name": user_name,
            "action_type": "chat",
            "messages": messages,
            "current_checkin": current_checkin,
            "priority_override": False
        }
        return self.run_workflow(state)

    def handle_narrative_synthesis(self, user_id: int, time_window_days: int = 30) -> Dict[str, Any]:
        """Invokes the Narrative Synthesis Agent to evaluate longitudinal trajectory across multiple sessions."""
        from agents.narrative_agent import synthesize_crisis_narrative
        return synthesize_crisis_narrative(user_id=user_id, time_window_days=time_window_days)

# Singleton orchestrator instance
orchestrator = Orchestrator()
