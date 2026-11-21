# agents/conversation.py
"""Agent 2: Conversation Agent.
Provides empathetic, non-diagnostic psychological wellbeing conversation.
Integrates with Ollama (configurable local LLM) with a robust, context-grounded
fallback when Ollama is offline or in cloud deployment environments.
"""

import os
import json
import logging
from typing import List, Dict, Any, Optional
import httpx
from agents.state import AgentState

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are CareAI, a warm, compassionate, and supportive mental health companion.
Your goals:
1. Listen actively, validate the user's emotional experience, and offer warm empathy.
2. Ground your suggestions in positive coping strategies (e.g. mindfulness, gentle breathing, self-compassion).
3. STRICT ETHICAL BOUNDARY: You are an academic wellbeing assistant, NOT a doctor, therapist, or medical practitioner. Do NOT diagnose conditions, prescribe medications, or claim clinical authority.
4. Keep answers supportive, encouraging, concise (2-4 paragraphs), and easy to read.
"""

def generate_llm_response(
    messages: List[Dict[str, str]],
    rag_context: Optional[List[Dict[str, Any]]] = None,
    user_name: Optional[str] = None,
    ollama_model: str = "llama3.2",
    ollama_host: str = "http://127.0.0.1:11434"
) -> str:
    """Generates an empathetic response via Ollama if available, with intelligent fallback."""
    # Build contextual prompt
    context_str = ""
    if rag_context:
        snippets = [f"- {item.get('topic', 'Guidance')}: {item.get('text', '')}" for item in rag_context]
        context_str = "\nTrusted Educational Context:\n" + "\n".join(snippets)

    # Format messages for Ollama API
    formatted_messages = [
        {"role": "system", "content": SYSTEM_PROMPT + (f"\nAddress the user as {user_name}." if user_name else "") + context_str}
    ]
    for msg in messages:
        formatted_messages.append({
            "role": msg.get("role", "user"),
            "content": msg.get("content", "")
        })

    # Try local Ollama first
    try:
        url = f"{ollama_host.rstrip('/')}/api/chat"
        payload = {
            "model": ollama_model,
            "messages": formatted_messages,
            "stream": False,
            "options": {
                "temperature": 0.7,
                "top_p": 0.9
            }
        }
        with httpx.Client(timeout=10.0) as client:
            resp = client.post(url, json=payload)
            if resp.status_code == 200:
                data = resp.json()
                content = data.get("message", {}).get("content", "")
                if content and content.strip():
                    return content.strip()
    except Exception as e:
        logger.debug(f"Ollama local inference unavailable ({e}), using safe contextual responder.")

    # Contextual, empathetic fallback response
    last_user_msg = messages[-1].get("content", "") if messages else "Hello"
    return _generate_contextual_fallback(last_user_msg, rag_context, user_name)

def _generate_contextual_fallback(
    user_msg: str,
    rag_context: Optional[List[Dict[str, Any]]] = None,
    user_name: Optional[str] = None
) -> str:
    """Provides a genuine, warm, non-diagnostic response grounded in mental health best practices."""
    name_prefix = f"Hi {user_name}, " if user_name else "Hello, "
    lowered = user_msg.lower()

    if any(w in lowered for w in ["anxious", "anxiety", "panic", "worried", "nervous", "scared"]):
        advice = (
            f"{name_prefix}I hear how much anxiety you're carrying right now. Feeling overwhelmed or physically tense is deeply exhausting, and it is completely understandable that you're looking for calm.\n\n"
            "When anxious thoughts race, our bodies often enter fight-or-flight mode. A fast way to ground yourself is the **4-7-8 breathing exercise**: gently inhale through your nose for 4 seconds, hold for 7 seconds, and exhale smoothly through your mouth for 8 seconds. Even two minutes of this tells your nervous system that you are in a safe space.\n\n"
            "Take this one breath at a time. What part of your day is feeling most challenging right now?"
        )
    elif any(w in lowered for w in ["stress", "stressed", "workload", "pressure", "burnout", "overwhelmed"]):
        advice = (
            f"{name_prefix}It sounds like you have a tremendous amount on your shoulders. Chronic stress can make even simple tasks feel like climbing a mountain.\n\n"
            "One helpful practice is **Progressive Muscle Relaxation**: consciously unclench your jaw, drop your shoulders away from your ears, and let your hands rest loosely. We often store high tension in our upper body without realizing it.\n\n"
            "Remember that you don't have to resolve everything today. Would it help to break down what's currently in front of you into one tiny, manageable step?"
        )
    elif any(w in lowered for w in ["sleep", "insomnia", "tired", "awake", "can't sleep", "nightmare"]):
        advice = (
            f"{name_prefix}Dealing with sleep difficulties is so draining. Sleep and emotional wellbeing are closely linked, and a rough night makes everything feel harder.\n\n"
            "For tonight, try a **digital wind-down**: dim your lights and step away from bright phone or laptop screens at least 45 minutes before getting in bed. If your mind is racing with tomorrow's to-do list, jotting those thoughts down on a physical notepad helps externalize them so your brain can rest.\n\n"
            "Be patient with yourself tonight; rest comes gradually when we release the pressure to fall asleep immediately."
        )
    elif any(w in lowered for w in ["sad", "depressed", "down", "lonely", "alone", "unhappy", "cry"]):
        advice = (
            f"{name_prefix}Thank you for sharing that with me. Acknowledging feeling low or disconnected takes courage, and your feelings are completely valid.\n\n"
            "When mood drops, our instinct is often to isolate, which can reinforce sadness. Even a tiny micro-action—opening a window for fresh air, making a warm cup of herbal tea, or sending a brief message to someone you care about—can bring a small measure of comfort.\n\n"
            "I'm here to listen. If you'd like to share more about what has been weighing on you, I am right here."
        )
    else:
        # General supportive response with RAG enhancement
        rag_tip = ""
        if rag_context and len(rag_context) > 0:
            top_snippet = rag_context[0].get("text", "")
            if len(top_snippet) > 180:
                top_snippet = top_snippet[:180] + "..."
            rag_tip = f"\n\nHere is an evidence-based tip: {top_snippet}"

        advice = (
            f"{name_prefix}Thank you for checking in with me today. Reflecting on your thoughts and emotions is a wonderful and proactive habit for your mental wellbeing.\n\n"
            "I am here to support you whenever you want to unpack a thought, talk through a feeling, or practice a relaxation technique.{rag_tip}\n\n"
            "How have you been feeling overall today?"
        )

    return advice

def conversation_agent_node(state: AgentState) -> Dict[str, Any]:
    """LangGraph node function for Agent 2: Conversation Agent."""
    messages = state.get("messages", [])
    rag_context = state.get("rag_context", [])
    user_name = state.get("user_name", "")
    
    response = generate_llm_response(
        messages=messages,
        rag_context=rag_context,
        user_name=user_name
    )
    return {"response": response}
