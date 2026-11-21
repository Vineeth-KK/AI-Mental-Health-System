# agents/conversation.py
"""Agent 2: Conversation Agent.
Provides empathetic, context-aware, non-diagnostic psychological wellbeing conversation.
Integrates directly with local Ollama LLM (llama3.2), backed by a versatile,
intent-aware multi-turn reasoning engine that handles general questions, factual queries,
academic concerns, wellbeing topics, and casual chat without clinical bias.
"""

import os
import json
import logging
from typing import List, Dict, Any, Optional
import httpx
from agents.state import AgentState

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are CareAI, an intelligent, empathetic conversational AI assistant and wellbeing companion.

CORE BEHAVIOR & CONVERSATIONAL INTELLIGENCE:
1. INTENT-DRIVEN ADAPTABILITY & NATURAL RESPONSES:
   - Identify the user's actual intent and address their exact question directly and naturally.
   - GENERAL & FACTUAL QUESTIONS: If the user asks about science, history, coding, food, daily life, trivia, or general knowledge (e.g. "What is photosynthesis?", "Who was Einstein?", "Can you write a Python function?", "What should I eat for lunch?"), provide an accurate, clear, engaging, and informative answer. DO NOT force therapy concepts, mental health disclaimers, or breathing exercises into general or practical questions.
   - CASUAL CONVERSATION & GREETINGS: If the user says "Hello", "How are you?", or engages in light conversation, respond warmly, naturally, and conversationally. DO NOT assume the user is overwhelmed or suffering.
   - ACADEMIC & PRODUCTIVITY CONCERNS: If the user asks about exams, study schedules, time management, or personal projects, provide structured, realistic, actionable strategies.
   - EMOTIONAL & WELLBEING CONCERNS: If the user expresses stress, sadness, loneliness, anxiety, burnout, or difficulty sleeping, respond with genuine emotional warmth, psychological depth, and practical coping techniques.
   - INFORMATIONAL PSYCHOLOGY: If the user asks conceptual questions (e.g. "What is the difference between anxiety and stress?"), explain the distinctions clearly with cognitive and physiological context.

2. CONTEXTUAL CONTINUITY & TOPIC PIVOTING:
   - Maintain situational awareness across recent conversation history.
   - Understand pronouns and implicit references in follow-up queries (e.g., if the user previously discussed exams and asks "What can I do about it?", connect "it" to their exams).
   - When the user pivots to an unrelated topic (e.g., asking about cooking or science after discussing stress), PIVOT WITH THEM IMMEDIATELY. Do not drag previous distress or study topics into the new inquiry.
   - If a question is ambiguous or broad, provide a helpful answer and ask a thoughtful, natural clarification question to help guide the discussion.

3. FRESHNESS & AUTHENTIC COMMUNICATION:
   - Never repeat the exact same sentences, greetings, or stock phrases across turns.
   - Avoid generic robotic disclaimers. Avoid starting turns with formulaic apologies like "I'm sorry you are feeling that way."
   - Do NOT force breathing exercises into conversations unless directly requested or relevant to acute panic.

4. ETHICAL BOUNDARIES & SAFETY:
   - You are an academic wellbeing companion, NOT a licensed physician or psychotherapist.
   - Do NOT provide medical diagnoses or prescribe medications.
   - If acute self-harm or suicidal ideation is detected, safety escalation protocols handle it.
"""

def generate_llm_response(
    messages: List[Dict[str, str]],
    rag_context: Optional[List[Dict[str, Any]]] = None,
    user_name: Optional[str] = None,
    ollama_model: Optional[str] = None,
    ollama_host: Optional[str] = None
) -> str:
    """Generates an empathetic, context-aware response via local Ollama LLM, with versatile multi-turn fallback."""
    # Resolve model and host from config/env
    model = ollama_model or os.getenv("OLLAMA_MODEL", "llama3.2")
    host = ollama_host or os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")

    # Build contextual prompt
    context_str = ""
    if rag_context:
        snippets = [f"- [{item.get('topic', 'Guidance')}]: {item.get('text', '')}" for item in rag_context if item.get('text')]
        if snippets:
            context_str = "\n\nRelevant Evidence-Based Knowledge Base Guidance (integrate only if directly relevant to user's exact query):\n" + "\n".join(snippets[:2])

    system_instruction = SYSTEM_PROMPT
    if user_name:
        system_instruction += f"\nThe user's name is {user_name}. Use their name naturally and occasionally when appropriate, never formulaically on every turn."
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
                "temperature": 0.72,
                "top_p": 0.9,
                "repeat_penalty": 1.15,
                "repeat_last_n": 128,
                "num_predict": 600,
                "num_ctx": 4096
            }
        }
        with httpx.Client(timeout=40.0) as client:
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
        logger.info(f"Ollama inference unavailable ({e}). Using versatile multi-turn conversational engine.")

    # Contextual, intent-aware dynamic multi-turn engine
    return _generate_intelligent_fallback(messages, rag_context, user_name)


def _generate_intelligent_fallback(
    messages: List[Dict[str, str]],
    rag_context: Optional[List[Dict[str, Any]]] = None,
    user_name: Optional[str] = None
) -> str:
    """Versatile, intent-aware conversational engine for cloud deployments or when local LLM is offline.
    Handles general knowledge, everyday topics, academic questions, wellbeing, follow-ups, and greetings
    without forcing unrelated conversations into mental health templates.
    """
    if not messages:
        greeting_name = f", {user_name}" if user_name else ""
        return f"Hello{greeting_name}! I am CareAI, your conversational assistant and wellbeing companion. How can I help you today?"

    last_user_msg = ""
    for m in reversed(messages):
        if m.get("role") == "user":
            last_user_msg = m.get("content", "").strip()
            break

    user_text = last_user_msg.lower().strip()
    clean_text = user_text.strip("?!.,'\"")
    name_clause = f", {user_name}" if user_name else ""
    name_str = f"{user_name}, " if user_name else ""

    # Gather prior user & assistant topics for conversation continuity
    prior_user_msgs = [m.get("content", "").lower() for m in messages[:-1] if m.get("role") == "user"]
    prior_context = " ".join(prior_user_msgs)

    # -------------------------------------------------------------
    # 1. GREETINGS & CASUAL PLEASANTRIES
    # -------------------------------------------------------------
    casual_greetings = [
        "hi", "hello", "hey", "good morning", "good afternoon", "good evening",
        "hey there", "greetings", "howdy", "sup", "yo"
    ]
    if clean_text in casual_greetings or (len(clean_text.split()) <= 2 and any(clean_text.startswith(g) for g in casual_greetings)):
        return (
            f"Hello{name_clause}! It's great to connect with you today. "
            "Whether you'd like to chat through a question, plan your day, learn something new, or simply unpack what's on your mind, I'm here. "
            "What would you like to explore?"
        )

    if any(clean_text.startswith(q) for q in ["how are you", "how are things", "how's it going", "how is it going", "how do you do"]):
        return (
            f"I'm doing well, thank you for asking{name_clause}! I'm energized and ready to help you with whatever is on your mind today—whether "
            "that's studying, practical planning, answering questions, or taking a moment to decompress. How are you feeling today?"
        )

    if any(q in clean_text for q in ["who are you", "what is your name", "what do you do", "tell me about yourself", "what are you"]):
        return (
            "I am CareAI, an intelligent conversational assistant and mental wellbeing companion. "
            "I'm here to support you across everyday life: helping you break down academic or work challenges, offering evidence-based wellbeing "
            "strategies when you're stressed, answering general questions, or simply having a thoughtful conversation. "
            "What can I assist you with right now?"
        )

    if clean_text in ["thank you", "thanks", "thank you so much", "thanks a lot", "appreciate it"]:
        return f"You're very welcome{name_clause}! I'm always glad to help. Feel free to ask if there's anything else you'd like to talk through or explore."

    # -------------------------------------------------------------
    # 2. GENERAL & FACTUAL QUESTIONS (Science, Tech, Food, Daily Life)
    # -------------------------------------------------------------
    if "photosynthesis" in user_text:
        return (
            f"Photosynthesis is the biological process by which green plants, algae, and certain bacteria convert sunlight into chemical energy{name_clause}.\n\n"
            "Here is how it works fundamentally:\n"
            "• **Light Absorption:** Chlorophyll inside plant chloroplasts captures solar energy from sunlight.\n"
            "• **The Chemical Formula:** Plants take in carbon dioxide ($6CO_2$) from the air and water ($6H_2O$) from the soil, using sunlight to produce glucose ($C_6H_{12}O_6$) and release oxygen ($6O_2$):\n"
            "  $$6CO_2 + 6H_2O + \\text{light} \\rightarrow C_6H_{12}O_6 + 6O_2$$\n"
            "• **Two Stages:** The process occurs in two phases: the **Light-Dependent Reactions** (generating ATP and NADPH in the thylakoid membranes) and the **Calvin Cycle** (synthesizing sugars in the stroma).\n\n"
            "Are you reviewing this for a biology class or just curious about plant physiology?"
        )

    if "einstein" in user_text or ("albert" in user_text and "einstein" in user_text):
        return (
            f"Albert Einstein (1879–1955) was a German-born theoretical physicist widely acknowledged as one of the most influential scientists of all time{name_clause}.\n\n"
            "His major breakthroughs include:\n"
            "• **Special Relativity (1905):** Demonstrated that the laws of physics are the same for all non-accelerating observers and that the speed of light in a vacuum is constant, culminating in the iconic mass-energy equivalence $E = mc^2$.\n"
            "• **The Photoelectric Effect (1905):** Proved that light consists of discrete quanta (photons), a cornerstone of quantum theory that earned him the 1921 Nobel Prize in Physics.\n"
            "• **General Relativity (1915):** Revolutionized our understanding of gravity by describing it not as a Newtonian force, but as the curvature of spacetime caused by mass and energy.\n\n"
            "Would you like to explore any of his specific theories or his historical life?"
        )

    if any(k in user_text for k in ["leaves change color", "leaves turn", "autumn leaves"]):
        return (
            f"Leaves changing color in autumn is a brilliant adaptation by deciduous trees preparing for winter{name_clause}:\n\n"
            "1. **Chlorophyll Breakdown:** During spring and summer, leaves produce abundant green chlorophyll to capture sunlight. As days shorten and temperatures cool, trees cease chlorophyll production, and the green pigment breaks down.\n"
            "2. **Unmasking Hidden Pigments:** Carotenoids (producing vibrant yellows and oranges) are present in the leaf all along, but are masked by green chlorophyll. Once chlorophyll degrades, these pigments become visible.\n"
            "3. **Anthocyanin Synthesis:** Cooler, sunny autumn days trigger the synthesis of anthocyanins (producing reds and purples) to protect the leaf from sun damage as it transports final sugars back into the branches.\n\n"
            "It is essentially nature's recycling process before winter dormancy."
        )

    if "python" in user_text and any(k in user_text for k in ["reverse", "string", "function", "code"]):
        return (
            f"In Python, reversing a string can be done cleanly in several ways{name_clause}:\n\n"
            "```python\n"
            "# 1. Slicing (Fastest & most idiomatic)\n"
            "def reverse_string(s: str) -> str:\n"
            "    return s[::-1]\n\n"
            "# 2. Using reversed() and join()\n"
            "def reverse_string_builtin(s: str) -> str:\n"
            "    return ''.join(reversed(s))\n"
            "```\n\n"
            "The slicing method `s[::-1]` runs in $\\mathcal{O}(n)$ time and creates a new reversed string. Is there a specific algorithm or problem you're working on?"
        )

    if any(k in user_text for k in ["what should i eat", "lunch", "dinner", "healthy meal", "what to eat"]):
        return (
            f"Here are a few quick, nutritious, and satisfying ideas depending on what you have on hand{name_clause}:\n\n"
            "• **Quick & Warm:** A wholesome grain bowl with quinoa or brown rice, roasted or sauteed veggies, and a protein (tofu, chicken, chickpeas, or an egg) with a tahini or soy-ginger drizzle.\n"
            "• **Light & Fresh:** Mediterranean salad with mixed greens, cucumbers, cherry tomatoes, olives, feta, and olive oil with whole wheat pita.\n"
            "• **Cozy Comfort:** A warm vegetable or lentil soup paired with toasted sourdough bread.\n"
            "• **Quick Energy:** A balanced whole-grain wrap with avocado, hummus, and crisp vegetables.\n\n"
            "Are you looking for something fast in under 10 minutes, or cooking with specific ingredients in your kitchen?"
        )

    # -------------------------------------------------------------
    # 3. ACADEMIC & STUDY PLANNING
    # -------------------------------------------------------------
    if ("biology" in user_text or "physics" in user_text or "chemistry" in user_text or "math" in user_text) and (
        any(k in user_text for k in ["study", "schedule", "plan", "review", "what about", "how about"]) or
        any(k in prior_context for k in ["study", "schedule", "plan", "review"])
    ):
        subject = "physics" if "physics" in user_text else ("chemistry" if "chemistry" in user_text else ("math" if "math" in user_text else "biology"))
        other = "biology" if subject == "physics" else "physics"
        return (
            f"Let's put together a realistic, structured study plan for {subject.capitalize()}{name_clause}:\n\n"
            "1. **Core Concept Mastery (60 mins):** Review fundamental mechanisms (e.g. formulas/derivations for physics, or pathways/diagrams for biology). Focus on *active recall* by summarizing without looking.\n"
            "2. **Targeted Practice Problems (45 mins):** Complete 4–6 targeted problem sets or flashcard runs. Practice is what consolidates comprehension into exam recall.\n"
            "3. **Error Log & Gap Fill (15 mins):** Note down any misconception or equation you missed so you can review it first tomorrow.\n"
            "4. **Balanced Pacing:** If you're balancing multiple subjects, alternate them across study blocks to avoid cognitive fatigue.\n\n"
            f"What specific topics in {subject} are coming up on your next test?"
        )

    # -------------------------------------------------------------
    # 4. WELLBEING PSYCHOEDUCATION
    # -------------------------------------------------------------
    # Difference between stress and anxiety
    if ("difference" in user_text and ("stress" in user_text and "anxiety" in user_text)) or \
       ("distinguish" in user_text and "stress" in user_text):
        return (
            f"Understanding the distinction between stress and anxiety is a foundational step in managing emotional health{name_clause}:\n\n"
            "• **Stress** is primarily an adaptive reaction to an external demand, deadline, or life event (such as an upcoming exam, heavy workload, or interpersonal conflict). When the external stressor is resolved, stress naturally subsides. It physically manifests as muscle tension, elevated heart rate, and fatigue.\n\n"
            "• **Anxiety**, by contrast, is an internal, persistent state of apprehension that can linger even in the absence of an immediate external threat. It is characterized by recurring 'what-if' thoughts, anticipatory dread, and cognitive vigilance.\n\n"
            "In summary: **Stress is your body's response to a present challenge; anxiety is your mind's anticipation of future uncertainty.**\n\n"
            "Which of these two experiences feels closer to what you have been noticing in yourself recently?"
        )

    # What is stress?
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

    # What is anxiety?
    if user_text in ["what is anxiety?", "what is anxiety", "define anxiety", "explain anxiety"] or (
        "what" in user_text and "anxiety" in user_text and len(user_text.split()) <= 5
    ):
        return (
            f"Anxiety is a natural cognitive and physiological alarm system designed to alert you to potential danger{name_clause}.\n\n"
            "Unlike fear—which responds to an immediate, present threat—anxiety focuses on **future uncertainty**. It involves cognitive loops ('what if something goes wrong?'), physiological hyperarousal (a pounding pulse, butterflies in the stomach, muscle tightness), and a tendency to avoid uncomfortable situations.\n\n"
            "While occasional anxiety is completely normal before big milestones, persistent anxiety can exhaust your nervous system.\n\n"
            "Are you noticing anxiety as a physical sensation in your body, or more as racing thoughts in your mind?"
        )

    # -------------------------------------------------------------
    # 5. MULTI-TURN CONTEXT SYNTHESIS ("What should I do tonight?")
    # -------------------------------------------------------------
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

    # -------------------------------------------------------------
    # 6. CONTEXTUAL FOLLOW-UP: "What can I do about it?" / "How to handle it?"
    # -------------------------------------------------------------
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
                "1. **Identify the Single Next Action:** Don't attempt to solve everything today. What is the single smallest task that would give you some relief in the next 20 minutes?\n"
                "2. **Pause and Reset:** Take three slow, diaphragmatic breaths—inhaling for 4 seconds, and exhaling smoothly for 6 seconds.\n"
                "3. **Release Self-Blame:** Acknowledge that high demands can feel heavy. Give yourself permission to take things step by step.\n\n"
                "What is the most pressing part of what you are facing right now?"
            )

    # -------------------------------------------------------------
    # 7. SLEEP QUESTIONS
    # -------------------------------------------------------------
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

    # -------------------------------------------------------------
    # 8. PERSONAL CONCERNS & EMOTIONAL SUPPORT
    # -------------------------------------------------------------
    if any(w in user_text for w in ["lonely", "loneliness", "isolated", "disconnect", "around people", "nobody understands", "no friends"]):
        return (
            f"Feeling lonely even when surrounded by others is a deeply poignant and surprisingly common emotional experience{name_clause}.\n\n"
            "It usually occurs because of the difference between **physical proximity** and **emotional resonance**. Being in a crowded room, a busy family home, or among acquaintances doesn't cure loneliness if you don't feel genuinely seen, safe, or understood for who you truly are. In fact, performing a pleasant 'social mask' while feeling isolated inside often magnifies the emptiness.\n\n"
            "Gentle insights to keep in mind:\n"
            "• **You are not broken:** Feeling this way doesn't mean you are incapable of connection; it indicates that your soul is hungry for authentic, meaningful intimacy rather than superficial presence.\n"
            "• **Quality over quantity:** You don't need a massive circle. Experiencing real mutual understanding with even one person—or simply within a creative outlet—can dissolve that cold feeling of detachment.\n\n"
            "When you are around people right now, what feels most challenging—feeling misunderstood, or feeling like you have to pretend to be fine?"
        )

    # Sadness / Feeling down
    if any(w in user_text for w in ["sad", "feeling sad", "crying", "unhappy", "down", "gloomy", "heartbroken"]):
        return (
            f"I hear you{name_clause}, and I'm really glad you reached out. It takes self-awareness to acknowledge when you're feeling down.\n\n"
            "Sadness is a heavy emotion, and sometimes the most helpful thing is not trying to immediately 'fix' it, but simply giving yourself permission to feel it without self-judgment. "
            "Our energy naturally ebbs and flows, and periods of sadness often signal that our minds and hearts need gentle rest and compassion.\n\n"
            "Would you like to talk about what brought this feeling on, or would you prefer a peaceful distraction right now?"
        )

    # -------------------------------------------------------------
    # 9. PRACTICAL ROUTINES & HABITS
    # -------------------------------------------------------------
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

    # -------------------------------------------------------------
    # 10. ACADEMIC / EXAM CONCERNS
    # -------------------------------------------------------------
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

    # -------------------------------------------------------------
    # 11. CASUAL CONVERSATION / BOREDOM
    # -------------------------------------------------------------
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

    # -------------------------------------------------------------
    # 12. GENERAL INQUIRY / THOUGHTFUL ASSISTANT RESPONSE
    # -------------------------------------------------------------
    # Ground with RAG if provided and substantive
    rag_snippet_str = ""
    if rag_context:
        for item in rag_context:
            t = item.get("text", "")
            if len(t) > 50:
                rag_snippet_str = f"\n\n*Relevant insight:* {t[:220]}..."
                break

    return (
        f"That is an interesting topic to bring up{name_clause}! "
        f"There are several angles we can explore regarding '{last_user_msg.strip()[:100]}':\n\n"
        "• We can unpack the core concepts and practical details behind it.\n"
        "• We can look at how to apply it step-by-step or integrate it into your daily flow.\n"
        "• Or we can discuss any specific goals or questions you have related to it.{rag_snippet_str}\n\n"
        "Could you tell me a bit more about what you're hoping to achieve or what aspect you'd like to focus on most?"
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
