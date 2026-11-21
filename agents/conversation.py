# agents/conversation.py
"""Agent 2: Conversation Agent.
Provides empathetic, context-aware, non-diagnostic psychological wellbeing conversation.
Integrates directly with local Ollama LLM (llama3.2) on RTX 3070 Ti, backed by a
dynamic, intent-aware, context-sensitive fallback engine when Ollama is offline.
"""

import os
import json
import logging
from typing import List, Dict, Any, Optional
import httpx
from agents.state import AgentState

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are CareAI, a deeply compassionate, highly intelligent, and evidence-grounded mental health companion.

CORE BEHAVIOR & CONVERSATIONAL INTELLIGENCE:
1. DIRECT RELEVANCE: Directly address the user's exact question, statement, or concern. Never respond with a generic disclaimer or boilerplate paragraph.
2. CONVERSATION CONTINUITY & MEMORY:
   - Carefully review all prior messages in the conversation.
   - Recognize and resolve follow-up questions (e.g., if the user previously mentioned "I am stressed about my exams" and now asks "What can I do about it?", understand that "it" refers to exam stress).
   - If the user adds new details (e.g., "I am also unable to sleep properly"), connect the new symptom with previous context (e.g., how exam stress disrupts sleep).
   - If the user asks "What should I do tonight?", provide an actionable evening wind-down plan tailored to their specific situation.
3. INTENT RECOGNITION & TONE ADAPTATION:
   - Informational / Educational (e.g., "What is stress?", "Difference between anxiety and stress"): Provide clear, engaging, well-structured psychological education with physical and mental distinctions.
   - Personal Emotional Sharing (e.g., "I feel lonely even around people"): Validate their feelings warmly, explore the underlying dynamics, and offer supportive perspective.
   - Practical / Action-Oriented (e.g., "Give me a simple daily routine", "What can I do?"): Provide concrete, realistic, numbered or bulleted steps that are easy to follow.
   - Casual / Boredom: Be warm, friendly, and engaging; suggest stimulating, mindful, or creative activities without clinical pathologizing.
4. VARIETY & FRESHNESS:
   - Avoid repeating the same phrases, sentence openings, or coping techniques across conversation turns.
   - Do NOT start every response with "I'm sorry you're feeling this way" or "Thank you for sharing." Vary your openings naturally.
5. ETHICAL BOUNDARY & SAFETY:
   - You are an academic wellbeing assistant, NOT a licensed doctor or psychotherapist.
   - Do NOT provide medical diagnoses or prescribe medications.
   - If severe distress or self-harm is mentioned, safety workflows will handle escalation.
6. FORMAT: Clear, readable, concise (2 to 4 focused paragraphs or clean bullet points). Use formatting (bolding, lists) to make guidance approachable.
"""

def generate_llm_response(
    messages: List[Dict[str, str]],
    rag_context: Optional[List[Dict[str, Any]]] = None,
    user_name: Optional[str] = None,
    ollama_model: str = "llama3.2",
    ollama_host: str = "http://127.0.0.1:11434"
) -> str:
    """Generates an empathetic, context-aware response via local Ollama LLM, with intelligent fallback."""
    # Build contextual prompt
    context_str = ""
    if rag_context:
        snippets = [f"- [{item.get('topic', 'Guidance')}]: {item.get('text', '')}" for item in rag_context if item.get('text')]
        if snippets:
            context_str = "\n\nTrusted Evidence-Based Knowledge Base Snippets:\n" + "\n".join(snippets[:2])

    system_instruction = SYSTEM_PROMPT
    if user_name:
        system_instruction += f"\nThe user's name is {user_name}. Always address the user warmly by their name ({user_name}) in your response."
    if context_str:
        system_instruction += context_str

    # Format messages for Ollama API
    formatted_messages = [
        {"role": "system", "content": system_instruction}
    ]
    for msg in messages:
        if isinstance(msg, dict) and msg.get("content"):
            formatted_messages.append({
                "role": msg.get("role", "user"),
                "content": str(msg.get("content", "")).strip()
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
                "top_p": 0.9,
                "num_predict": 512,
                "num_ctx": 4096
            }
        }
        with httpx.Client(timeout=35.0) as client:
            resp = client.post(url, json=payload)
            if resp.status_code == 200:
                data = resp.json()
                content = data.get("message", {}).get("content", "")
                if content and content.strip():
                    logger.info(f"Generated dynamic response via Ollama ({ollama_model}).")
                    return content.strip()
            else:
                logger.warning(f"Ollama returned status {resp.status_code}: {resp.text}")
    except Exception as e:
        logger.info(f"Ollama local inference unavailable or timed out ({e}). Using intelligent fallback engine.")

    # Contextual, intent-aware dynamic fallback
    return _generate_intelligent_fallback(messages, rag_context, user_name)


def _generate_intelligent_fallback(
    messages: List[Dict[str, str]],
    rag_context: Optional[List[Dict[str, Any]]] = None,
    user_name: Optional[str] = None
) -> str:
    """Intelligent, multi-turn context-aware fallback engine.
    Ensures non-repetitive, intent-specific responses when Ollama is offline.
    """
    if not messages:
        return "Hello! I am CareAI, your wellbeing companion. What's on your mind today?"

    last_user_msg = ""
    for m in reversed(messages):
        if m.get("role") == "user":
            last_user_msg = m.get("content", "").strip()
            break

    user_text = last_user_msg.lower()
    name_str = f"{user_name}, " if user_name else ""

    # Gather prior user topics for conversation continuity
    prior_user_msgs = [m.get("content", "").lower() for m in messages[:-1] if m.get("role") == "user"]
    prior_context = " ".join(prior_user_msgs)

    # 1. Informational: Difference between stress and anxiety
    if "difference" in user_text and ("stress" in user_text and "anxiety" in user_text):
        return (
            f"Understanding the difference between stress and anxiety is a great step toward managing both:\n\n"
            "• **Stress** is primarily a reaction to an external trigger or demand (such as an exam, work deadline, or conflict). When the stressor is resolved, stress usually subsides. Physical signs often include muscle tension, headaches, or irritability.\n\n"
            "• **Anxiety**, by contrast, is an internal, persistent state of apprehension or dread that can linger even in the absence of an immediate threat. It often involves excessive, uncontrollable 'what-if' thoughts and physical symptoms like a racing heart or shallow breathing.\n\n"
            "In short: *Stress is about the current situation; anxiety is about anticipated fear.*\n\n"
            "Which of these feels closer to what you've been noticing in yourself lately?"
        )

    # 2. Informational: What is stress?
    if user_text in ["what is stress?", "what is stress", "define stress", "explain stress"] or (
        "what" in user_text and "stress" in user_text and len(user_text.split()) <= 6
    ):
        return (
            "Stress is your body and mind's natural biological response to perceived challenges, demands, or changes.\n\n"
            "When faced with a challenge, your brain triggers a release of stress hormones (cortisol and adrenaline). In short bursts, acute stress can actually sharpen focus and help you meet a goal. However, **chronic stress**—when the pressure continues without relief—drains your emotional energy and causes physical fatigue.\n\n"
            "Common signs of stress include:\n"
            "• Physical: Clenched jaw, shoulder tightness, shallow breathing, digestive issues\n"
            "• Mental: Rushing thoughts, difficulty focusing, feeling irritable or easily overwhelmed\n\n"
            "Is there a particular source of stress you are navigating right now?"
        )

    # 3. Follow-up: "What should I do tonight?"
    if "tonight" in user_text or ("what should i do" in user_text and "night" in user_text):
        has_exam_context = "exam" in prior_context or "test" in prior_context or "study" in prior_context
        has_sleep_context = "sleep" in prior_context or "awake" in prior_context or "insomnia" in prior_context

        if has_exam_context and has_sleep_context:
            return (
                f"Given your upcoming exams and sleep difficulties, here is a realistic, step-by-step plan for tonight:\n\n"
                "1. **Set a Hard Study Cutoff:** Close all study materials at least 60 minutes before bedtime. Cramming late into the night raises cortisol and keeps your brain in high-alert mode.\n"
                "2. **The 'Brain Dump' Exercise:** Take 5 minutes to write down whatever you need to review tomorrow on a physical notepad, then physically close it. This signals to your brain that tomorrow's work is scheduled and safe to leave behind.\n"
                "3. **Dim the Lights & Disconnect:** Turn your phone to 'Do Not Disturb' and keep bedroom lighting low. Read a light book or listen to calming audio rather than scrolling screens.\n"
                "4. **No-Pressure Resting:** If you lie down and don't fall asleep in 20 minutes, don't battle the clock. Get up, sit in a dimly lit chair, and do slow 4-7-8 breathing until drowsiness naturally returns.\n\n"
                "You've worked hard today. Prioritizing rest tonight will do more for your exam performance than another hour of tired reading."
            )
        elif has_exam_context:
            return (
                f"For tonight, the most productive thing you can do for your exams is protect your mental stamina:\n\n"
                "1. **Wrap up studying early:** Stop active revision at least an hour before sleep.\n"
                "2. **Decompress:** Take a warm shower, stretch your neck and shoulders, and step away from digital study portals.\n"
                "3. **Sleep is memory consolidation:** Your brain organizes facts during sleep. Treat resting tonight as an active part of your exam prep.\n\n"
                "How are your study materials organized for tomorrow?"
            )
        else:
            return (
                f"Here is a restful evening plan for tonight, {name_str}:\n\n"
                "1. **Digital Sunset:** Dim screens and turn on warm room lighting 45 minutes before sleep.\n"
                "2. **Somatic Release:** Spend 5 minutes gently rolling your shoulders and releasing tension in your jaw.\n"
                "3. **Pleasant Mental Anchor:** Listen to relaxing ambient music, read a chapter of an enjoyable book, or sip a warm cup of caffeine-free tea.\n\n"
                "Take the evening one calm moment at a time. What would feel most comforting right now?"
            )

    # 4. Contextual Follow-up: "What can I do about it?" / "How do I fix it?"
    if any(q in user_text for q in ["what can i do", "what should i do", "how to handle", "how can i cope", "what do i do"]) and len(user_text.split()) <= 8:
        if "exam" in prior_context or "test" in prior_context:
            return (
                "Here are three practical, immediate strategies to regain control over exam stress:\n\n"
                "1. **Chunk It Down (The 25/5 Pomodoro Rule):** When a whole syllabus feels overwhelming, commit to studying for just 25 focused minutes, followed by a 5-minute complete break. It lowers the barrier to getting started.\n"
                "2. **Focus on 'Controllables':** You cannot control the exact test questions, but you can control your review schedule, taking practice questions, and your hydration and sleep.\n"
                "3. **Active Reset:** Whenever anxiety spikes at your desk, push your chair back, plant both feet firmly on the floor, and take 3 deep, audible sighs. It physically resets your nervous system.\n\n"
                "Which subject or chapter feels heaviest on your mind right now?"
            )
        elif "lonel" in prior_context:
            return (
                "To gently address feelings of loneliness, here are small, compassionate steps you can take:\n\n"
                "1. **Micro-Connections with Zero Pressure:** Send a simple meme or a 1-sentence note to someone you trust ('Saw this and thought of you!').\n"
                "2. **Shared Solitude Spaces:** Visit a quiet coffee shop, library, or public park. Being around others without having to perform socially often eases the weight of isolation.\n"
                "3. **Creative Self-Connection:** Write or express how you're feeling without editing. Often, feeling understood by yourself is the foundation for reconnecting with others.\n\n"
                "Would you like to explore reaching out to someone, or focus on self-care first?"
            )
        else:
            return (
                "Here are three grounded steps you can take right now:\n\n"
                "1. **Pause and Breathe:** Take 3 slow, diaphragmatic breaths—in through the nose, out smoothly through the mouth.\n"
                "2. **Identify the Single Next Step:** Don't try to solve the entire challenge at once. What is the single smallest action you can take in the next 15 minutes?\n"
                "3. **Release Self-Judgment:** Acknowledge that having a tough day is part of being human. Give yourself permission to go at a gentler pace.\n\n"
                "Would you like to talk through one specific part of what you're facing?"
            )

    # 5. Academic / Exam Stress
    if any(w in user_text for w in ["exam", "examination", "studying", "syllabus", "test", "finals", "grades"]):
        return (
            f"Exam pressure can be immensely heavy, {name_str}and it's completely understandable to feel stressed when expectations are high.\n\n"
            "When exam anxiety takes over, the brain often falls into **catastrophizing**—convincing you that a single exam defines your entire future. In reality, exam stress is simply your brain trying to mobilize energy, but it often gives you too much adrenaline at once.\n\n"
            "Try this quick strategy right now:\n"
            "• **Divide your remaining study topics into 3 piles:** 'Confident', 'Needs Review', and 'Unfamiliar'. Start with 'Needs Review' to build momentum.\n"
            "• **Protect 10-minute movement breaks:** Physical movement helps clear excess adrenaline.\n\n"
            "How many days do you have until your next exam, and what subject is it?"
        )

    # 6. Sleep-related questions
    if any(w in user_text for w in ["sleep", "insomnia", "sleepless", "awake", "can't sleep", "cant sleep", "tired but", "falling asleep"]):
        has_exam = "exam" in prior_context or "test" in prior_context or "study" in prior_context
        connection_note = " especially with exams on your horizon, " if has_exam else " "
        return (
            f"Struggling to sleep{connection_note}is exhausting because tired minds generate even more worry.\n\n"
            "The most common reason for nighttime sleeplessness is **cognitive hyperarousal**—your body is in bed, but your nervous system is still on duty. Here are evidence-based ways to break that loop:\n\n"
            "• **The 20-Minute Rule:** If you've been lying in bed unable to sleep for about 20 minutes, get up. Staying in bed tossing and turning trains your brain to associate the bed with frustration. Sit in a comfortable chair in dim light and do something quiet until your eyelids feel heavy.\n"
            "• **Cool Temperature:** Keep your room cool (around 18-19°C / 65-68°F). A drop in core body temperature is a biological trigger for sleep onset.\n"
            "• **Cognitive Shuffling:** Instead of thinking about your to-do list, picture random words starting with a specific letter (e.g. apple, anchor, airplane). It occupies the verbal brain and induces alpha brainwaves.\n\n"
            "Are you having trouble falling asleep initially, or are you waking up in the middle of the night?"
        )

    # 7. Loneliness (even around people)
    if any(w in user_text for w in ["lonely", "loneliness", "isolated", "disconnect", "nobody cares", "around people"]):
        return (
            f"Feeling lonely even when surrounded by other people is a deeply painful and surprisingly common experience, {name_str}.\n\n"
            "It often happens because of a gap between **physical proximity** and **emotional resonance**. You can be in a room full of friends or family, but if you don't feel truly seen, understood, or able to share your authentic thoughts, it actually heightens the sense of isolation.\n\n"
            "A few gentle perspectives to hold:\n"
            "• You are not broken for feeling this way. It usually indicates that you crave deeper, more genuine connection rather than just superficial presence.\n"
            "• You don't have to connect with everyone. Finding even one person or space where you can speak without a mask makes a profound difference.\n\n"
            "What do you feel is keeping you from feeling connected to the people around you right now?"
        )

    # 8. Practical Daily Routine Request
    if any(w in user_text for w in ["routine", "daily routine", "schedule", "plan my day", "daily habits", "habit"]):
        return (
            "Here is a balanced, evidence-based daily wellbeing routine designed to reduce chronic stress without feeling restrictive:\n\n"
            "🌅 **Morning Anchor (First 30 Minutes):**\n"
            "• Hydrate with a full glass of water.\n"
            "• Get 5-10 minutes of natural outdoor daylight in your eyes to calibrate your circadian rhythm.\n"
            "• Avoid checking stressful social media or news feeds for the first 20 minutes.\n\n"
            "☀️ **Midday Reset:**\n"
            "• Take a dedicated 15-minute screen-free lunch.\n"
            "• Do a 2-minute posture check: unclench your jaw, drop your shoulders, and take three deep breaths.\n\n"
            "🌙 **Evening Decompression (Last Hour):**\n"
            "• Write a 3-item 'Done List' celebrating what you accomplished, no matter how small.\n"
            "• Dim bright overhead lights.\n"
            "• Digital sunset: Put your phone out of arm's reach 30 minutes before sleep.\n\n"
            "Which part of this routine would be easiest for you to try out tomorrow?"
        )

    # 9. Casual / Boredom
    if any(w in user_text for w in ["bored", "boredom", "nothing to do", "entertain me", "what can i do when i feel bored"]):
        return (
            "Boredom is actually a fascinating mental state! It's your brain's way of signaling that it has spare cognitive capacity and is looking for meaningful engagement.\n\n"
            "Here are four refreshing ways to shift that energy:\n"
            "1. **Micro-Learning Sprint:** Spend 15 minutes learning something completely outside your normal routine (e.g. origami, a chess tactic, a new language phrase, or a bizarre historical fact).\n"
            "2. **Sensory Change of Pace:** Step outside for a 10-minute walk without headphones, noticing 5 things you've never paid attention to on your street.\n"
            "3. **Creative Dopamine Reset:** Sketch, organize one drawer, or cook a creative snack from whatever is in your kitchen.\n"
            "4. **Reconnect:** Call or message a friend you haven't spoken to in a while just to see how they're doing.\n\n"
            "Do you feel like doing something creative, something active, or something relaxing?"
        )

    # 10. General Emotional Distress / Overwhelm (Enhanced with RAG)
    rag_snippet_str = ""
    if rag_context:
        for item in rag_context:
            t = item.get("text", "")
            if len(t) > 60:
                rag_snippet_str = f"\n\n*Helpful insight from wellbeing resources:* {t[:220]}..."
                break

    return (
        f"Thank you for sharing that with me, {name_str}. Whatever you are experiencing right now, take a moment to acknowledge that your feelings are valid and you don't have to carry them entirely on your own.\n\n"
        "When things feel cloudy or heavy, simplifying your focus is the kindest step:\n"
        "• Notice your posture and let go of any tension you are holding in your shoulders or forehead.\n"
        "• Ask yourself: *What is the one small thing that would give me a little more comfort in the next hour?*{rag_snippet_str}\n\n"
        "I'm right here with you. What part of your situation would you like to unpack first?"
    )


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
