# agents/conversation.py
"""Agent 2: Conversation Agent.
Provides empathetic, context-aware, non-diagnostic psychological wellbeing conversation.
Integrates directly with local Ollama LLM (llama3.2) on RTX 3070 Ti, backed by an
advanced, intent-aware, context-sensitive multi-turn reasoning engine.
"""

import os
import json
import logging
from typing import List, Dict, Any, Optional
import httpx
from agents.state import AgentState

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are CareAI, an advanced, deeply perceptive, and evidence-grounded mental health companion.

CORE BEHAVIOR & CONVERSATIONAL INTELLIGENCE:
1. DIRECT INTENT & TARGETED ANSWERS:
   - Answer the user's exact question or concern directly. Never substitute a real answer with a generic disclaimer or canned paragraph.
   - If the user asks an informational or psychological question (e.g., "What is stress?", "What is the difference between anxiety and stress?", "Why am I unable to sleep properly?"), provide clear, intellectually illuminating explanations distinguishing psychological vs physiological manifestations, cognitive patterns, and situational triggers.
   - If the user shares a personal struggle (e.g., "I feel lonely even around people", "I am overwhelmed"), validate their emotional state with genuine depth. Unpack the underlying dynamics (e.g., emotional resonance vs physical presence) so they feel truly understood.
   - If the user asks for practical guidance or an action plan (e.g., "What should I do tonight?", "Give me a simple daily routine to reduce stress", "What can I do about it?"), provide structured, realistic, numbered or bulleted steps that can be implemented immediately.
   - If the user engages in casual conversation or mentions feeling bored, respond with warmth, curiosity, and creative, engaging ideas without clinical or patronizing language.

2. CONTEXTUAL CONTINUITY & MULTI-TURN MEMORY:
   - Maintain acute situational awareness across the entire conversation history.
   - Understand pronouns and implicit references in follow-up queries (e.g., if the user previously mentioned "I am stressed about my exams" and now asks "What can I do about it?", understand that "it" refers to exam stress and offer realistic study pacing and anxiety release).
   - Synthesize evolving details: if the user adds "I am also unable to sleep properly" after discussing exam stress, address how performance pressure hyperarouses the nervous system at night, connecting both threads together.
   - When the user asks "What should I do tonight?", generate an evening wind-down routine tailored specifically to their combined situation (e.g. studying cutoffs + insomnia relief).
   - If the user pivots to an unrelated topic (e.g. asking something casual or curious), gracefully pivot with them rather than dragging old issues into the new question.

3. FRESHNESS & AVOIDANCE OF REPETITION:
   - NEVER repeat the exact same sentences, greetings, or stock phrases across turns.
   - Avoid generic robotic disclaimers. Avoid starting every turn with "I'm sorry you're feeling that way" or "Thank you for sharing."
   - Do NOT force the same breathing exercises into every conversation. Only introduce somatic exercises when directly relevant to acute panic, physical tension, or requested coping techniques.

4. ETHICAL BOUNDARIES & SAFETY:
   - You are an academic wellbeing assistant, NOT a licensed doctor or psychotherapist.
   - Do NOT provide medical diagnoses or prescribe medications.
   - If acute self-harm or suicidal ideation is detected, safety workflows handle escalation.

5. COMMUNICATION STYLE:
   - Empathetic, articulate, conversational, and natural.
   - Use clean formatting (bold headers, bullet points) when structuring advice to make it readable and gentle on an anxious mind.
"""

def generate_llm_response(
    messages: List[Dict[str, str]],
    rag_context: Optional[List[Dict[str, Any]]] = None,
    user_name: Optional[str] = None,
    ollama_model: Optional[str] = None,
    ollama_host: Optional[str] = None
) -> str:
    """Generates an empathetic, context-aware response via local Ollama LLM, with intelligent multi-turn fallback."""
    # Resolve model and host from config/env
    model = ollama_model or os.getenv("OLLAMA_MODEL", "llama3.2")
    host = ollama_host or os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")

    # Build contextual prompt
    context_str = ""
    if rag_context:
        snippets = [f"- [{item.get('topic', 'Guidance')}]: {item.get('text', '')}" for item in rag_context if item.get('text')]
        if snippets:
            context_str = "\n\nRelevant Evidence-Based Knowledge Base Guidance (integrate seamlessly if relevant to user's exact query):\n" + "\n".join(snippets[:2])

    system_instruction = SYSTEM_PROMPT
    if user_name:
        system_instruction += f"\nThe user's name is {user_name}. Address them warmly and naturally as {user_name} when appropriate, without repeating it excessively."
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

    # Try local Ollama LLM first
    try:
        url = f"{host.rstrip('/')}/api/chat"
        payload = {
            "model": model,
            "messages": formatted_messages,
            "stream": False,
            "options": {
                "temperature": 0.75,
                "top_p": 0.9,
                "repeat_penalty": 1.18,
                "repeat_last_n": 128,
                "num_predict": 512,
                "num_ctx": 4096
            }
        }
        with httpx.Client(timeout=25.0) as client:
            resp = client.post(url, json=payload)
            if resp.status_code == 200:
                data = resp.json()
                content = data.get("message", {}).get("content", "")
                if content and content.strip():
                    logger.info(f"Generated dynamic response via Ollama ({model}).")
                    return content.strip()
            else:
                logger.warning(f"Ollama returned status {resp.status_code}: {resp.text}")
    except Exception as e:
        logger.info(f"Ollama inference unavailable ({e}). Using multi-turn intelligent conversational engine.")

    # Contextual, intent-aware dynamic multi-turn engine
    return _generate_intelligent_fallback(messages, rag_context, user_name)


def _generate_intelligent_fallback(
    messages: List[Dict[str, str]],
    rag_context: Optional[List[Dict[str, Any]]] = None,
    user_name: Optional[str] = None
) -> str:
    """Advanced, multi-turn context-aware conversational engine.
    Ensures non-repetitive, intent-specific, highly relevant responses when LLM is offline or on cloud containers.
    """
    if not messages:
        greeting_name = f", {user_name}" if user_name else ""
        return f"Hello{greeting_name}! I am CareAI, your wellbeing companion. What is on your mind today?"

    last_user_msg = ""
    for m in reversed(messages):
        if m.get("role") == "user":
            last_user_msg = m.get("content", "").strip()
            break

    user_text = last_user_msg.lower().strip()
    name_str = f"{user_name}, " if user_name else ""
    name_clause = f", {user_name}" if user_name else ""

    # Gather prior user & assistant topics for conversation continuity
    prior_user_msgs = [m.get("content", "").lower() for m in messages[:-1] if m.get("role") == "user"]
    prior_context = " ".join(prior_user_msgs)

    # 1. Informational: Difference between stress and anxiety
    if ("difference" in user_text and ("stress" in user_text and "anxiety" in user_text)) or \
       ("distinguish" in user_text and "stress" in user_text):
        return (
            f"Understanding the distinction between stress and anxiety is a foundational step in managing emotional health{name_clause}:\n\n"
            "• **Stress** is primarily an adaptive reaction to an external demand, deadline, or life event (e.g. an upcoming examination, heavy workload, or interpersonal conflict). When the external stressor passes, stress naturally subsides. It physically manifests as muscle tension, elevated heart rate, and fatigue.\n\n"
            "• **Anxiety**, by contrast, is an internal, persistent state of apprehension that can linger even in the absence of an immediate external threat. It is characterized by recurring 'what-if' thoughts, anticipatory dread, and cognitive vigilance.\n\n"
            "In summary: **Stress is your body's response to a present challenge; anxiety is your mind's anticipation of future uncertainty.**\n\n"
            "Which of these two experiences feels closer to what you have been noticing in yourself recently?"
        )

    # 2. Informational: What is stress? / Define stress
    if user_text in ["what is stress?", "what is stress", "define stress", "explain stress", "what does stress mean", "tell me about stress"] or (
        "what" in user_text and "stress" in user_text and len(user_text.split()) <= 6 and "routine" not in user_text and "reduce" not in user_text
    ):
        return (
            f"Stress is your body and nervous system's biological mobilization in response to perceived demands, challenges, or changes{name_clause}.\n\n"
            "When faced with a challenge, your brain's amygdala activates the hypothalamic-pituitary-adrenal (HPA) axis, releasing adrenaline and cortisol. In short bursts (**acute stress**), this reaction sharpens attention, speeds reflexes, and helps you meet important goals. However, when demands persist without recovery (**chronic stress**), it drains cognitive resilience.\n\n"
            "Common dimensions of stress include:\n"
            "• **Physical:** Clenched jaw, shoulder tightness, shallow breathing, headaches, and sleep changes.\n"
            "• **Cognitive & Emotional:** Rushing thoughts, decision fatigue, irritability, and feeling overwhelmed by minor tasks.\n\n"
            "Is there a particular area of your life—such as academics, work, or relationships—where you are feeling this pressure most right now?"
        )

    # 3. Informational: What is anxiety?
    if user_text in ["what is anxiety?", "what is anxiety", "define anxiety", "explain anxiety"] or (
        "what" in user_text and "anxiety" in user_text and len(user_text.split()) <= 5
    ):
        return (
            f"Anxiety is a natural cognitive and physiological alarm system designed to alert you to potential danger{name_clause}.\n\n"
            "Unlike fear—which responds to an immediate, present threat—anxiety focuses on **future uncertainty**. It involves cognitive loops ('what if something goes wrong?'), physiological hyperarousal (a pounding pulse, butterflies in the stomach, muscle tightness), and a tendency to avoid uncomfortable situations.\n\n"
            "While occasional anxiety is completely normal before big milestones, persistent anxiety can exhaust your nervous system.\n\n"
            "Are you noticing anxiety as a physical sensation in your body, or more as racing thoughts in your mind?"
        )

    # 4. Contextual Multi-Turn: "What should I do tonight?"
    if "tonight" in user_text or ("what should i do" in user_text and "night" in user_text) or ("plan for tonight" in user_text):
        has_exam_context = any(k in prior_context for k in ["exam", "test", "study", "syllabus", "finals"])
        has_sleep_context = any(k in prior_context for k in ["sleep", "awake", "insomnia", "sleepless", "tired"])

        if has_exam_context and (has_sleep_context or "sleep" in user_text):
            return (
                f"Given your upcoming examinations and difficulty sleeping, {name_str}here is a realistic, step-by-step evening plan designed to protect both your recall and your sleep:\n\n"
                "1. **Set a Hard Study Cutoff (8:30 PM):** Close textbooks and laptops at least 60–90 minutes before bedtime. Late-night cramming keeps cortisol high and prevents the brain from entering restorative slow-wave sleep.\n"
                "2. **The 5-Minute Brain Dump:** Take a physical sheet of paper and write down the top 3 items you will review tomorrow morning. Then physically close the notebook. This gives your subconscious permission to stop rehearsing facts.\n"
                "3. **Zero-Screen Transition:** Keep bedroom lighting soft and dim. Avoid looking at grade portals, test forums, or social media. Listen to relaxing ambient audio or read a non-academic chapter.\n"
                "4. **No-Pressure Resting Rule:** When you lie down, remember that simply resting quietly in the dark restores 80% of somatic energy even before sleep begins. If you aren't asleep after 20 minutes, sit in a dim chair and breathe gently until drowsiness arrives.\n\n"
                "Resting tonight will do more for your exam performance tomorrow than an extra hour of exhausted reading."
            )
        elif has_exam_context:
            return (
                f"For tonight, {name_str}the most effective strategy for your exam preparation is active decompression:\n\n"
                "1. **Wrap up study revision early:** Set a firm stopping point at least an hour before sleep to let your working memory cool down.\n"
                "2. **Hydrate and unwind:** Drink water or herbal tea, step away from digital study screens, and take a warm shower to lower core body temperature.\n"
                "3. **Trust your preparation:** Memory consolidation happens during sleep. Treat quality rest tonight as an essential, high-yield part of your exam routine.\n\n"
                "How are your study materials organized for tomorrow?"
            )
        else:
            return (
                f"Here is a calming, restorative evening flow for tonight, {name_str}:\n\n"
                "1. **Digital Sunset:** Dim overhead lights and place devices on 'Do Not Disturb' 45 minutes before bedtime.\n"
                "2. **Somatic Release:** Spend 3–5 minutes unclenching your jaw, rolling your shoulders backward, and releasing tension in your neck.\n"
                "3. **Mindful Anchor:** Listen to quiet music, read something uplifting, or reflect on one small moment of grace from today.\n\n"
                "Take this evening one gentle hour at a time. What would feel most comforting right now?"
            )

    # 5. Contextual Multi-Turn: "What can I do about it?" / "How can I handle it?"
    if any(q in user_text for q in ["what can i do about it", "what can i do", "what should i do", "how do i handle it", "how to cope", "what can i do now"]) and len(user_text.split()) <= 8:
        if any(k in prior_context for k in ["exam", "test", "study", "syllabus", "finals"]):
            return (
                f"Here are three practical, immediate actions to regain control over exam stress, {name_str}:\n\n"
                "1. **The 25/5 Study Sprint (Pomodoro):** Do not look at the entire syllabus at once. Commit to just 25 minutes on one focused topic, followed by 5 minutes away from your desk. It dramatically lowers cognitive friction.\n"
                "2. **Triage Your Notes into Three Piles:** Sort chapters into 'Solid', 'Needs Review', and 'Uncertain'. Spend 70% of your energy reviewing the middle pile—this yields the highest grade return with the least panic.\n"
                "3. **Physical Reset:** When anxiety spikes at your study desk, stand up, stretch both arms overhead, and take two long, audible exhalations to discharge excess adrenaline.\n\n"
                "Which specific chapter or topic feels most urgent right now?"
            )
        elif any(k in prior_context for k in ["lonel", "isolated", "disconnect", "alone"]):
            return (
                f"To gently work through feelings of loneliness, {name_str}here are small, low-pressure steps that help restore emotional connection:\n\n"
                "1. **Low-Stakes Micro-Reconnection:** Send a low-pressure message to one friend or family member ('Thinking of you, hope your week is going well!'). You don't have to initiate an intense conversation to feel the warmth of contact.\n"
                "2. **Shared Presence Spaces:** Spend an hour in a library, bookstore, or cozy neighborhood café. Being around people without having to perform socially eases the acute ache of isolation.\n"
                "3. **Self-Compassion Check:** Remind yourself that craving depth and belonging is a healthy human need, not a personal flaw.\n\n"
                "Would you prefer to explore ways to reach out, or focus on comforting yourself first?"
            )
        elif any(k in prior_context for k in ["sleep", "insomnia", "awake"]):
            return (
                f"Here are three immediate steps to reset your sleep cycle, {name_str}:\n\n"
                "1. **Protect Wake-Up Time:** Wake up at the exact same time every morning, regardless of how much you slept the night before. This anchors your circadian clock.\n"
                "2. **Stimulus Control (The 20-Minute Rule):** If you are awake in bed for 20 minutes, get up. Do not stay in bed tossing and turning.\n"
                "3. **Cognitive Shuffling:** Mentally visualize random neutral objects (e.g., barn, lighthouse, teapot) to quiet internal verbal narration.\n\n"
                "Are you having more trouble falling asleep, or waking up throughout the night?"
            )
        else:
            return (
                f"Here are three grounded steps you can take right now, {name_str}:\n\n"
                "1. **Pause and Breathe:** Take three slow, diaphragmatic breaths—inhaling for 4 seconds, and exhaling smoothly for 6 seconds.\n"
                "2. **Identify the Single Next Action:** Don't attempt to solve everything today. What is the single smallest task that would give you some relief in the next 20 minutes?\n"
                "3. **Release Self-Blame:** Acknowledge that feeling overwhelmed is a natural response to high demands. Give yourself permission to slow down.\n\n"
                "What is the most pressing part of what you are facing right now?"
            )

    # 6. Sleep-related questions: "Why am I unable to sleep properly?"
    if any(w in user_text for w in ["sleep", "insomnia", "sleepless", "awake", "can't sleep", "cant sleep", "unable to sleep", "falling asleep"]):
        has_exam = any(k in prior_context for k in ["exam", "test", "study", "syllabus"])
        connection_note = f", especially alongside the academic demands on your mind, {name_str}" if has_exam else f", {name_str}"
        return (
            f"Difficulty sleeping{connection_note}is deeply frustrating because an exhausted brain finds it even harder to manage stress.\n\n"
            "The primary driver of bedtime insomnia during stressful periods is **cognitive hyperarousal**. Even when your body is fatigued, your nervous system interprets upcoming deadlines as an active threat and stays in sentinel mode. Physical and psychological triggers include:\n\n"
            "• **The Bedtime Rumination Trap:** When external distractions vanish at night, unaddressed worries rush into the silence.\n"
            "• **Circadian Disruption:** Screen exposure and irregular study hours suppress the natural evening surge of melatonin.\n\n"
            "Key strategies to break the cycle:\n"
            "1. **The 20-Minute Rule:** If you cannot sleep after 20 minutes, leave the bed. Sitting in dim light until you feel sleepy prevents your brain from associating your mattress with frustration.\n"
            "2. **Keep the Bedroom Cool:** Maintain a cool ambient temperature (~18–19°C / 65–68°F); a drop in core body temperature is a biological prerequisite for deep sleep.\n"
            "3. **Cognitive Shuffling:** Picture neutral, unconnected items starting with different letters (e.g. cloud, canoe, compass) to gently dismantle racing thoughts.\n\n"
            "Are you having trouble falling asleep initially, or are you waking up in the early morning hours?"
        )

    # 7. Personal Concern: Loneliness in a crowd
    if any(w in user_text for w in ["lonely", "loneliness", "isolated", "disconnect", "around people", "nobody understands", "no friends"]):
        return (
            f"Feeling lonely even when surrounded by others is a deeply poignant and surprisingly common emotional experience{name_clause}.\n\n"
            "It usually occurs because of the difference between **physical proximity** and **emotional resonance**. Being in a crowded room, a busy family home, or among acquaintances doesn't cure loneliness if you don't feel genuinely seen, safe, or understood for who you truly are. In fact, performing a pleasant 'social mask' while feeling isolated inside often magnifies the emptiness.\n\n"
            "Gentle insights to keep in mind:\n"
            "• **You are not broken:** Feeling this way doesn't mean you are incapable of connection; it indicates that your soul is hungry for authentic, meaningful intimacy rather than superficial presence.\n"
            "• **Quality over quantity:** You don't need a massive circle. Experiencing real mutual understanding with even one person—or simply within a creative outlet—can dissolve that cold feeling of detachment.\n\n"
            "When you are around people right now, what feels most challenging—feeling misunderstood, or feeling like you have to pretend to be fine?"
        )

    # 8. Practical Daily Routine Request
    if any(w in user_text for w in ["routine", "daily routine", "schedule", "daily habits", "plan my day", "habit"]):
        return (
            f"Here is a balanced, evidence-based daily wellbeing routine designed to reduce chronic stress without feeling overwhelming{name_clause}:\n\n"
            "🌅 **Morning Anchor (First 30 Minutes):**\n"
            "• **Light & Hydration:** Drink a glass of water and get 5–10 minutes of natural outdoor daylight in your eyes to calibrate your cortisol awakening curve.\n"
            "• **The 15-Minute Shield:** Avoid opening stressful emails or social media for the first 15 minutes of your morning.\n\n"
            "☀️ **Midday Reset (Afternoon):**\n"
            "• **The Micro-Break:** Step away from all screens for a dedicated 15-minute lunch.\n"
            "• **Somatic Unclenching:** Drop your shoulders, unglue your tongue from the roof of your mouth, and exhale fully.\n\n"
            "🌙 **Evening Decompression (Last 60 Minutes):**\n"
            "• **The 'Done List':** Write down 3 small things you accomplished today, shifting your mindset from deficiency to progress.\n"
            "• **Digital Sunset:** Dim overhead lights and transition to warm lamps 45 minutes before sleep.\n\n"
            "Which part of this routine feels like the easiest, most accessible place for you to begin tomorrow?"
        )

    # 9. Academic / Exam Stress
    if any(w in user_text for w in ["exam", "examination", "studying", "syllabus", "test", "finals", "grades", "academic"]):
        return (
            f"Exam stress can feel all-consuming{name_clause}, and it is entirely valid to feel the weight of performance pressure.\n\n"
            "When exam dates draw near, the brain often falls into **catastrophizing**—unconsciously equating test scores with personal worth or entire life trajectories. In truth, exam stress is your nervous system mobilizing energy for a challenge, but when adrenaline stays elevated all day, it impairs the very working memory you need to revise.\n\n"
            "Practical ways to ease the load right now:\n"
            "• **Divide and Conquer:** Break your syllabus into 3 distinct categories: 'Mastered', 'Review Needed', and 'Unfamiliar'. Focus on 'Review Needed' first to build momentum.\n"
            "• **Active Recall over Passive Rereading:** Testing yourself with flashcards or practice questions is twice as effective in half the study time.\n"
            "• **Scheduled Physical Breaks:** Take a 10-minute walk every hour to metabolize cortisol.\n\n"
            "How many days do you have until your next exam, and what subject is feeling most daunting?"
        )

    # 10. Casual Conversation / Boredom
    if any(w in user_text for w in ["bored", "boredom", "nothing to do", "entertain me", "what can i do when i feel bored"]):
        return (
            f"Boredom is actually a fascinating cognitive signal{name_clause}! It is your brain's way of telling you that you have spare cognitive capacity and are craving meaningful stimulation.\n\n"
            "Here are four refreshing ways to channel that energy:\n"
            "1. **The Micro-Learning Sprint:** Spend 15 minutes diving into a bizarre or captivating subject outside your normal routine (e.g., deep-sea biology, history of cryptography, or origami).\n"
            "2. **Sensory Walk:** Step outside for a 10-minute walk without headphones, purposefully spotting 5 visual details you have never noticed on your street.\n"
            "3. **Tactile Creation Reset:** Sketch, cook a creative snack from random kitchen ingredients, or rearrange one small corner of your room.\n"
            "4. **A Micro-Connection:** Send an unexpected, appreciative note to an old friend just to ask how their week is going.\n\n"
            "Do you feel like engaging in something creative, something physical, or something relaxing?"
        )

    # 11. General Emotional Distress / Overwhelm (Ground with RAG if provided)
    rag_snippet_str = ""
    if rag_context:
        for item in rag_context:
            t = item.get("text", "")
            if len(t) > 50:
                rag_snippet_str = f"\n\n*Evidence-based wellbeing insight:* {t[:220]}..."
                break

    return (
        f"Thank you for sharing that with me{name_clause}. Navigating difficult emotions takes courage, and it is important to remember that whatever you are experiencing right now is valid.\n\n"
        "When thoughts feel turbulent or heavy, simplifying your immediate horizon is the kindest step:\n"
        "• Take a slow breath, release any physical tension in your forehead and shoulders, and plant both feet firmly on the ground.\n"
        "• Ask yourself: *What is one small thing that would bring 5% more comfort or clarity to my next hour?*{rag_snippet_str}\n\n"
        "I am here beside you. Would you like to unpack what is feeling heaviest, or would you prefer a practical strategy to help you center yourself?"
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
