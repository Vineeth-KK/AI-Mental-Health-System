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
    "mood": [
        {
            "title": "Behavioral Activation & Micro-Movement",
            "category": "Mood Support",
            "duration": "10 minutes",
            "steps": "Choose one small, low-effort positive action (e.g. stepping outside into daylight, gentle stretching, or making a warm cup of herbal tea) and engage with it fully.",
            "benefit": "Gently breaks cognitive inertia and stimulates natural dopamine without overwhelming your emotional energy."
        },
        {
            "title": "Mindful Self-Compassion Break",
            "category": "Emotional Wellbeing",
            "duration": "5 minutes",
            "steps": "Place a hand gently over your chest, acknowledge that difficult feelings are a normal part of life, and offer yourself words of patience and kindness.",
            "benefit": "Calms inner self-criticism and soothes emotional fatigue."
        }
    ],
    "general": [
        {
            "title": "Deep Breathing & Body Reset",
            "category": "Daily Relaxation",
            "duration": "3 minutes",
            "steps": "Practice slow nasal breathing for 3 minutes to settle your mind and reset physical tension.",
            "benefit": "Promotes parasympathetic balance and mental clarity."
        },
        {
            "title": "Reflective Gratitude Journaling",
            "category": "Emotional Wellbeing",
            "duration": "5 minutes",
            "steps": "Write down 3 specific things that brought you comfort, peace, or relief today, no matter how small.",
            "benefit": "Balances the brain's natural negativity bias."
        },
        {
            "title": "Mindful Walking Observation",
            "category": "Mindfulness",
            "duration": "10 minutes",
            "steps": "Take a short walk and pay gentle attention to your physical surroundings: notice 3 distinct colors and ambient sounds.",
            "benefit": "Grounds awareness in immediate reality and interrupts rumination."
        }
    ]
}

def generate_recommendations(
    distress_level: Optional[str] = None,
    current_checkin: Optional[Dict[str, Any]] = None,
    journal_text: Optional[str] = None,
    messages: Optional[List[Dict[str, str]]] = None
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Generates structured exercise recommendations and retrieves RAG educational context."""
    # Build search query from distress indicators or user chat message
    query_parts = []
    
    # In chat context, prioritize the user's explicit question/message
    last_user_msg = ""
    if messages:
        last_user_msg = next((m.get("content", "") for m in reversed(messages) if m.get("role") == "user"), "")

    is_casual_chat = False
    if last_user_msg:
        lowered = last_user_msg.lower()
        casual_greetings = ["hi", "hello", "hey", "good morning", "good evening", "how are you", "bored", "what's up"]
        if any(lowered == g or lowered.startswith(g + " ") for g in casual_greetings) and len(lowered.split()) <= 4:
            is_casual_chat = True
        else:
            query_parts.append(last_user_msg[:250])

    if not is_casual_chat:
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

    search_query = " ".join(query_parts).strip() if query_parts else ""
    
    # Query RAG if a substantive query exists
    rag_snippets = []
    if search_query and not is_casual_chat:
        raw_snippets = rag_service.query(search_query, n_results=3)
        # Keep only snippets with sufficient semantic relevance (>= 0.40)
        rag_snippets = [s for s in raw_snippets if s.get("relevance_score", 0) >= 0.40]

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
        if current_checkin.get("mood", 10) <= 4:
            exercises.extend(WELLBEING_EXERCISES["mood"])

    if not exercises:
        exercises.extend(WELLBEING_EXERCISES["general"])

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
    messages = state.get("messages", [])

    exercises, rag_context = generate_recommendations(distress_level, checkin, journal_text, messages)
    return {
        "recommendations": exercises,
        "rag_context": rag_context
    }
