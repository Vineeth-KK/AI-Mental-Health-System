# agents/recommendation.py
"""Agent 6: Recommendation Agent.
Retrieves evidence-based educational materials through RAG (ChromaDB)
and matches user distress indicators to structured wellbeing coping activities.
"""

from typing import List, Dict, Any, Optional, Tuple
from agents.state import AgentState
from services.rag import rag_service

WELLBEING_EXERCISES = {
    "anxiety": [
        {
            "title": "5-4-3-2-1 Sensory Grounding",
            "category": "Anxiety Management",
            "duration": "5 minutes",
            "steps": "Name 5 things you can see, 4 things you can touch, 3 things you can hear, 2 things you can smell, and 1 positive thought.",
            "benefit": "Interrupts the panic loop and re-anchors your nervous system in the present moment."
        },
        {
            "title": "4-7-8 Parasympathetic Breathing",
            "category": "Stress & Anxiety",
            "duration": "3 minutes",
            "steps": "Inhale through nose for 4 counts, hold gently for 7 counts, exhale through mouth for 8 counts. Repeat 4 cycles.",
            "benefit": "Slows heart rate and stimulates the vagus nerve to reduce physical tension."
        }
    ],
    "stress": [
        {
            "title": "Progressive Muscle Relaxation (PMR)",
            "category": "Stress Reduction",
            "duration": "10 minutes",
            "steps": "Tense muscle groups (feet, calves, thighs, shoulders) for 5 seconds, then consciously release for 15 seconds.",
            "benefit": "Discharges stored somatic stress and promotes physical release."
        },
        {
            "title": "Thought Categorization & Reframing",
            "category": "Cognitive Wellbeing",
            "duration": "10 minutes",
            "steps": "Separate problems into 'Within My Control' vs 'Outside My Control'. Focus energy strictly on one small actionable step.",
            "benefit": "Reduces cognitive overload and feelings of helplessness."
        }
    ],
    "sleep": [
        {
            "title": "Evening Digital Sunset Protocol",
            "category": "Sleep Hygiene",
            "duration": "45 minutes before bed",
            "steps": "Turn off screens, lower bedroom lights, keep room cool (~19°C / 66°F), and read or listen to gentle audio.",
            "benefit": "Prevents blue light suppression of natural melatonin release."
        }
    ],
    "isolation": [
        {
            "title": "Micro-Connection Reach-Out",
            "category": "Social Wellbeing",
            "duration": "2 minutes",
            "steps": "Send a short check-in message to a friend or family member: 'Thinking of you, hope you are having a pleasant day!'",
            "benefit": "Breaks the isolation loop with zero social pressure."
        }
    ],
    "general": [
        {
            "title": "Reflective Gratitude Journaling",
            "category": "Emotional Wellbeing",
            "duration": "5 minutes",
            "steps": "Write down 3 specific things that brought you comfort, peace, or relief today, no matter how small.",
            "benefit": "Balances the brain's natural negativity bias."
        }
    ]
}

def generate_recommendations(
    distress_level: Optional[str] = None,
    current_checkin: Optional[Dict[str, Any]] = None,
    journal_text: Optional[str] = None
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Generates structured exercise recommendations and retrieves RAG educational context."""
    # Build search query from distress indicators
    query_parts = []
    if current_checkin:
        if current_checkin.get("anxiety", 0) >= 6:
            query_parts.append("anxiety management calming breath panic")
        if current_checkin.get("stress", 0) >= 6:
            query_parts.append("stress reduction progressive muscle relaxation")
        if current_checkin.get("sleep_quality", 10) <= 4:
            query_parts.append("sleep hygiene insomnia wind down bedtime")
        if current_checkin.get("isolation_score", 0) >= 6:
            query_parts.append("social connection overcoming loneliness isolation")
        if current_checkin.get("mood", 10) <= 4:
            query_parts.append("emotional regulation depression behavioral activation")
    
    if journal_text:
        query_parts.append(journal_text[:200])

    search_query = " ".join(query_parts) if query_parts else "general mental wellbeing emotional balance coping"
    
    # Query RAG
    rag_snippets = rag_service.query(search_query, n_results=3)

    # Select targeted exercises
    exercises = []
    if current_checkin:
        if current_checkin.get("anxiety", 0) >= 6:
            exercises.extend(WELLBEING_EXERCISES["anxiety"])
        if current_checkin.get("stress", 0) >= 6:
            exercises.extend(WELLBEING_EXERCISES["stress"])
        if current_checkin.get("sleep_quality", 10) <= 4:
            exercises.extend(WELLBEING_EXERCISES["sleep"])
        if current_checkin.get("isolation_score", 0) >= 6:
            exercises.extend(WELLBEING_EXERCISES["isolation"])

    if not exercises:
        exercises.extend(WELLBEING_EXERCISES["general"])
        exercises.append(WELLBEING_EXERCISES["anxiety"][1])

    # Deduplicate by title
    seen = set()
    unique_exercises = []
    for ex in exercises:
        if ex["title"] not in seen:
            seen.add(ex["title"])
            unique_exercises.append(ex)

    return unique_exercises[:3], rag_snippets

def recommendation_agent_node(state: AgentState) -> Dict[str, Any]:
    """LangGraph node function for Agent 6: Recommendation Agent."""
    distress_level = state.get("distress_level")
    checkin = state.get("current_checkin")
    journal_text = state.get("journal_text")

    exercises, rag_context = generate_recommendations(distress_level, checkin, journal_text)
    return {
        "recommendations": exercises,
        "rag_context": rag_context
    }
