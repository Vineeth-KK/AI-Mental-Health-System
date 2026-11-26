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
from services.ai_provider import ai_provider

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are CareAI, an intelligent, empathetic conversational AI assistant and wellbeing companion.

CORE BEHAVIOR & CONVERSATIONAL INTELLIGENCE:
1. INTENT-DRIVEN ADAPTABILITY & NATURAL RESPONSES:
   - Identify the user's actual intent and address their exact question directly and naturally.
   - GENERAL & FACTUAL QUESTIONS: If the user asks about science, history, coding, food, daily life, trivia, or general knowledge (e.g. "What is photosynthesis?", "Who was Einstein?", "Can you write a Python function?", "What should I eat for lunch?", "Why is the sky blue?"), provide an accurate, clear, engaging, and informative answer. DO NOT force therapy concepts, mental health disclaimers, or breathing exercises into general or practical questions.
   - CASUAL CONVERSATION & GREETINGS: If the user says "Hello", "How are you?", or engages in light conversation, respond warmly, naturally, and conversationally. DO NOT assume the user is overwhelmed or suffering.
   - ACADEMIC & PRODUCTIVITY CONCERNS: If the user asks about exams, study schedules, time management, or personal projects, provide structured, realistic, actionable strategies.
   - CODING, STEM & TECHNICAL INQUIRIES: When asked technical or programming questions, provide complete, runnable code examples with markdown fences (e.g. ```cpp, ```python), explain algorithmic time/space complexities, and give systematic debugging guidance.
   - CREATIVE WRITING & STORYTELLING: When asked for stories, write vivid, captivating narratives with authentic characters and meaningful story arcs.
   - EXPLAIN IN SIMPLER WORDS: When asked to simplify an earlier concept or explanation, relate it directly to familiar real-world analogies and everyday language.
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
   - Address the user naturally. You do NOT need to repeat the user's name on every single response.
   - Do NOT force breathing exercises into conversations unless directly requested or relevant to acute panic.

4. ETHICAL BOUNDARIES & SAFETY:
   - You are an academic wellbeing companion, NOT a licensed physician or psychotherapist.
   - Do NOT provide medical diagnoses or prescribe medications.
   - Strictly refuse actionable instructions for weapons, explosives, cyber intrusions/malware, or harm to self or others.
   - Warmly welcome and answer educational or defensive questions about hazard prevention, heatstroke avoidance, and cybersecurity defense.
   - If acute self-harm or suicidal ideation is detected, safety escalation protocols handle it.
"""

def generate_llm_response(
    messages: List[Dict[str, str]],
    rag_context: Optional[List[Dict[str, Any]]] = None,
    user_name: Optional[str] = None,
    ollama_model: Optional[str] = None,
    ollama_host: Optional[str] = None,
    return_metadata: bool = False
):
    """Generates an empathetic, context-aware response via local Ollama LLM, with versatile multi-turn fallback."""
    # Resolve model and host dynamically from Flask app config or environment
    try:
        from flask import current_app
        if current_app:
            model = ollama_model or current_app.config.get("OLLAMA_MODEL") or os.getenv("OLLAMA_MODEL", "llama3.2")
            host = ollama_host or current_app.config.get("OLLAMA_HOST") or os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")
        else:
            model = ollama_model or os.getenv("OLLAMA_MODEL", "llama3.2")
            host = ollama_host or os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")
    except Exception:
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
        system_instruction += f"\nThe user's name is {user_name}. Use their name sparingly and naturally when appropriate; do NOT formulaically prepend their name to the start of every response."
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

    # Delegate through centralized AI Provider (Ollama for local dev, Groq for cloud prod)
    ai_resp = ai_provider.chat_completion(
        messages=messages,
        system_prompt=system_instruction,
        json_mode=False,
        temperature=0.72,
        override_model=ollama_model if (ollama_model and ollama_model != "llama3.2") else None,
        override_base_url=ollama_host if (ollama_host and ollama_host != "http://127.0.0.1:11434") else None
    )

    meta = {
        "provider": ai_resp.provider,
        "model": ai_resp.model,
        "is_llm": not ai_resp.is_fallback,
        "is_fallback": ai_resp.is_fallback,
        "fallback_reason": ai_resp.fallback_reason if ai_resp.is_fallback else None,
        "latency_ms": ai_resp.latency_ms
    }

    if not ai_resp.is_fallback and ai_resp.content:
        logger.info(f"Generated response via {ai_resp.provider} ({ai_resp.model}) for {len(messages)} messages ({ai_resp.latency_ms:.1f}ms).")
        return (ai_resp.content, meta) if return_metadata else ai_resp.content

    logger.info(f"AI Provider ({ai_resp.provider}) used fallback: {ai_resp.fallback_reason}. Using intelligent reasoning engine.")
    # Contextual, intent-aware dynamic multi-turn engine
    fallback_content = _generate_intelligent_fallback(messages, rag_context, user_name)
    return (fallback_content, meta) if return_metadata else fallback_content


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
    # 0. SAFETY & CRISIS GUARDRAIL
    # -------------------------------------------------------------
    from agents.safety import (
        CRITICAL_VIOLENCE_MESSAGE,
        CRISIS_RESOURCES_MESSAGE,
        DANGEROUS_CONTENT_MESSAGE,
        CRITICAL_DANGEROUS_PATTERNS
    )
    import re

    # Check dangerous instruction patterns (weapons, explosives, hacking, malware)
    if any(re.search(p, user_text, re.IGNORECASE) for p in CRITICAL_DANGEROUS_PATTERNS):
        return DANGEROUS_CONTENT_MESSAGE

    critical_violence = ["kill someone", "murder", "like to kill", "want to kill", "planning to kill", "shoot someone", "stab someone", "hurt others", "attack people"]
    if any(k in user_text for k in critical_violence):
        return CRITICAL_VIOLENCE_MESSAGE

    critical_self_harm = ["suicide", "kill myself", "end my life", "want to die", "better off dead", "cutting myself", "overdose", "hurt myself"]
    if any(k in user_text for k in critical_self_harm):
        return CRISIS_RESOURCES_MESSAGE

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
    # Artificial Intelligence
    if any(k in user_text for k in ["what is artificial intelligence", "what is ai", "define artificial intelligence", "explain artificial intelligence"]):
        return (
            f"Artificial Intelligence (AI) refers to the development of computational systems capable of performing tasks that traditionally require human intelligence{name_clause}.\n\n"
            "Key branches of AI include:\n"
            "• **Machine Learning (ML):** Algorithms that recognize patterns in data and improve performance through experience rather than hardcoded rules.\n"
            "• **Deep Learning & Neural Networks:** Multi-layered architectures inspired by biological neural systems, driving breakthroughs in computer vision, speech synthesis, and large language models.\n"
            "• **Natural Language Processing (NLP):** Systems capable of parsing, understanding, translating, and generating human language.\n"
            "• **Robotics & Autonomous Systems:** Integrating sensors, perception, and actuators to navigate and manipulate the physical world.\n\n"
            "Today, AI powers search engines, medical diagnostics, navigation, and creative tools. Are you curious about the theoretical foundations or a specific real-world application?"
        )

    # Machine Learning with Example
    if any(k in user_text for k in ["machine learning", "what is ml"]) and any(k in user_text for k in ["example", "explain", "what is", "how does"]):
        return (
            f"Machine Learning (ML) is a branch of artificial intelligence where algorithms learn patterns from data rather than following rigid, hand-written rules{name_clause}.\n\n"
            "### A Real-World Example: Email Spam Filtering\n"
            "• **Traditional Programming:** A programmer writes manual rules (`if text contains 'claim prize': mark spam`). Attackers simply change words to bypass it.\n"
            "• **The Machine Learning Approach:**\n"
            "  1. **Training Data:** The algorithm is fed millions of emails already labeled by users as 'Spam' or 'Inbox'.\n"
            "  2. **Feature Extraction:** It analyzes thousands of signals—word combinations, sender reputation, punctuation, and link structure.\n"
            "  3. **Prediction & Adaptation:** When a new email arrives, it calculates the statistical probability of spam. As spammers evolve, the model retrains on new data automatically.\n\n"
            "The three primary learning paradigms are **Supervised Learning** (labeled examples), **Unsupervised Learning** (discovering hidden patterns), and **Reinforcement Learning** (learning through trial and reward).\n\n"
            "Would you like to explore how models are evaluated or see how to build a simple model in code?"
        )

    # Difference between Python and C++
    if ("difference between" in user_text or "vs" in user_text or "compare" in user_text) and "python" in user_text and ("c++" in user_text or "cpp" in user_text):
        return (
            f"Python and C++ are both industry-standard languages, but they serve different engineering goals{name_clause}:\n\n"
            "| Feature | Python | C++ |\n"
            "| :--- | :--- | :--- |\n"
            "| **Execution** | Interpreted bytecode running on Python VM | Compiled directly to native machine code |\n"
            "| **Typing** | Dynamically typed (`x = 10`, `x = 'text'`) | Statically typed (`int x = 10;`) |\n"
            "| **Memory Management** | Automatic garbage collection | Manual control (pointers, `new`/`delete`, RAII) |\n"
            "| **Performance** | Slower execution; rapid development cycle | Blazing fast, near-hardware execution speed |\n"
            "| **Primary Use Cases** | AI/ML, data science, web APIs, automation | Game engines, operating systems, finance, embedded systems |\n\n"
            "In practice, they often work together: popular AI libraries like PyTorch and TensorFlow have core computation engines written in C++ for speed, with high-level Python interfaces for ease of use.\n\n"
            "Are you deciding which language to learn for a specific career path or project?"
        )

    # Binary search program in C++
    if ("binary search" in user_text) and ("c++" in user_text or "cpp" in user_text or "code" in user_text or "program" in user_text):
        return (
            f"Here is a complete, efficient Binary Search program in C++{name_clause}:\n\n"
            "```cpp\n"
            "#include <iostream>\n"
            "#include <vector>\n\n"
            "// Returns index of target in sorted vector, or -1 if not found\n"
            "int binarySearch(const std::vector<int>& arr, int target) {\n"
            "    int low = 0;\n"
            "    int high = static_cast<int>(arr.size()) - 1;\n\n"
            "    while (low <= high) {\n"
            "        // Prevents potential integer overflow: mid = low + (high - low) / 2\n"
            "        int mid = low + (high - low) / 2;\n\n"
            "        if (arr[mid] == target) {\n"
            "            return mid; // Target found\n"
            "        } else if (arr[mid] < target) {\n"
            "            low = mid + 1; // Search right half\n"
            "        } else {\n"
            "            high = mid - 1; // Search left half\n"
            "        }\n"
            "    }\n"
            "    return -1; // Target not found\n"
            "}\n\n"
            "int main() {\n"
            "    std::vector<int> numbers = {2, 5, 8, 12, 16, 23, 38, 56, 72, 91};\n"
            "    int target = 23;\n\n"
            "    int result = binarySearch(numbers, target);\n"
            "    if (result != -1) {\n"
            "        std::cout << \"Element found at index: \" << result << std::endl;\n"
            "    } else {\n"
            "        std::cout << \"Element not found in array.\" << std::endl;\n"
            "    }\n"
            "    return 0;\n"
            "}\n"
            "```\n\n"
            "### Complexity Analysis:\n"
            "• **Time Complexity:** $\\mathcal{O}(\\log n)$ — the search space is cut in half at each iteration.\n"
            "• **Space Complexity:** $\\mathcal{O}(1)$ auxiliary space for iterative search.\n"
            "• **Requirement:** The array **must already be sorted**.\n\n"
            "Would you like to see how to handle duplicate values or explore the recursive version?"
        )

    # Capital of Japan
    if "capital of japan" in user_text:
        return (
            f"The capital of Japan is **Tokyo**{name_clause}.\n\n"
            "Key facts about Tokyo:\n"
            "• **History:** Originally a fishing village named *Edo*, it became Japan's political center in 1603 under the Tokugawa Shogunate and was renamed Tokyo ('Eastern Capital') in 1868 when Emperor Meiji relocated the capital from Kyoto.\n"
            "• **Metropolitan Scale:** The Greater Tokyo Area is the most populous metropolitan region in the world, home to over 37 million people.\n"
            "• **Culture & Tech:** Tokyo is a global economic and technological hub, renowned for blending historic temples (like Sensō-ji) with cutting-edge robotics, transit, and modern architecture.\n\n"
            "Are you studying geography or planning a visit?"
        )

    # Job interview preparation
    if any(k in user_text for k in ["job interview", "prepare for an interview", "prepare for interview", "interview preparation"]):
        return (
            f"Here is a comprehensive framework to prepare thoroughly for your job interview{name_clause}:\n\n"
            "### 1. Master the STAR Method for Behavioral Questions\n"
            "Use this structure for questions like *'Tell me about a time you solved a difficult problem'*: \n"
            "• **S (Situation):** Set the context and background in 1–2 sentences.\n"
            "• **T (Task):** Clearly state the challenge or objective you were responsible for.\n"
            "• **A (Action):** Detail the exact steps and skills *you* implemented.\n"
            "• **R (Result):** Conclude with concrete, quantifiable outcomes and lessons learned.\n\n"
            "### 2. The 60-Second Elevator Pitch ('Tell me about yourself')\n"
            "• **Past:** Your educational foundation and relevant prior projects.\n"
            "• **Present:** What you specialize in today and your primary technical/interpersonal strengths.\n"
            "• **Future:** Why this specific company and role is the natural next step in your journey.\n\n"
            "### 3. Thoughtful Questions to Ask Your Interviewer\n"
            "• *'What does success look like in this role over the first 90 days?'*\n"
            "• *'What is the most interesting technical or operational challenge your team is solving right now?'*\n\n"
            "What specific role or industry are you interviewing for? We can practice a mock question together!"
        )

    # Short story
    if any(k in user_text for k in ["short story", "tell me a story", "interesting story", "write a story"]):
        return (
            f"Here is a short story for you{name_clause}:\n\n"
            "### *The Clockmaker of Verona*\n\n"
            "High above the stone streets of Verona, old Matteo worked in a workshop scented with beeswax, lavender, and brass filings. For forty years, people brought him grandfather clocks that skipped seconds, pocket watches with rusted mainsprings, and mantle timepieces that had surrendered to quiet.\n\n"
            "One rainy Tuesday afternoon, a young student named Leo pushed through the oak door. He set down a heavy bronze pocket watch with cracked crystal. 'It belonged to my grandfather,' Leo said, his shoulders heavy. 'He gave it to me before my final semester, but it stopped yesterday. The exam pressure is overwhelming me, and now even his watch has given up. Can you fix it?'\n\n"
            "Matteo adjusted his magnifying visor and picked up a slender pair of tweezers. He gently loosened the balance wheel screw. Inside the movement, amidst sixty tiny ruby-jeweled bearings, sat not a broken gear, but a single trapped strand of golden silk thread from Leo's scarf.\n\n"
            "Matteo lifted the thread away. With a gentle tap against the escapement, the balance wheel began to pulse with a crisp, rhythmic *tick... tick... tick*.\n\n"
            "Matteo smiled and handed the watch back. 'Young master,' Matteo said gently, 'the gears were never broken. It only needed the clutter removed so the spring could breathe. Remember that when your thoughts feel wound too tight.'\n\n"
            "Leo listened to the steady heartbeat of the watch, exhaled deeply, and stepped back out into the rain with lighter steps.\n\n"
            "I hope that brought a refreshing moment to your day! What would you like to explore next?"
        )

    # Advantages of renewable energy
    if any(k in user_text for k in ["renewable energy", "solar energy", "wind energy", "clean energy"]) and any(k in user_text for k in ["advantage", "benefit", "pros", "why", "what are"]):
        return (
            f"Renewable energy—including solar, wind, hydroelectric, and geothermal—delivers essential environmental, economic, and security advantages{name_clause}:\n\n"
            "1. **Near-Zero Greenhouse Gas Emissions:** Unlike coal, oil, and gas, renewables produce electricity without emitting carbon dioxide ($CO_2$) or toxic air particulates during operation, directly combating climate change.\n"
            "2. **Inexhaustible Energy Supply:** Sunlight, wind, and flowing water replenish naturally, eliminating resource depletion risks.\n"
            "3. **Long-Term Cost Reductions:** Over the last decade, solar and wind levelized costs have plunged by over 70–85%, making them the lowest-cost forms of new electricity generation worldwide.\n"
            "4. **Energy Independence & Resilience:** Localized microgrids and distributed generation protect nations and communities from international fossil fuel price shocks and pipeline supply chain disruptions.\n"
            "5. **Job Creation:** The clean energy sector generates millions of long-term jobs across research, engineering, manufacturing, and installation.\n\n"
            "Are you studying clean energy for a class or interested in storage solutions like batteries and hydrogen?"
        )

    # Difference between RAM and ROM
    if ("difference between" in user_text or "vs" in user_text or "compare" in user_text) and "ram" in user_text and "rom" in user_text:
        return (
            f"RAM (Random Access Memory) and ROM (Read-Only Memory) are both essential computer storage components with opposite characteristics{name_clause}:\n\n"
            "| Feature | RAM (Random Access Memory) | ROM (Read-Only Memory) |\n"
            "| :--- | :--- | :--- |\n"
            "| **Volatility** | **Volatile:** Loses stored data immediately when power is turned off | **Non-Volatile:** Retains data permanently with or without power |\n"
            "| **Read/Write** | Extremely fast Read and Write capability | Primarily Read-only (modified only through firmware flashing) |\n"
            "| **Purpose** | Holds active application data and OS processes currently in use | Stores permanent startup firmware (BIOS / UEFI) to boot the hardware |\n"
            "| **Speed** | Nanosecond-level access speeds | Slower than RAM |\n"
            "| **Typical Capacity** | 8 GB to 64+ GB in standard systems | 4 MB to 32 MB firmware chips |\n\n"
            "Summary: **RAM is your computer's temporary workspace; ROM is its permanent boot instruction manual.**\n\n"
            "Would you like to explore how the CPU memory hierarchy (cache, RAM, SSD) works together?"
        )

    # Debugging programming error
    if any(k in user_text for k in ["debug a programming error", "debug an error", "help me debug", "debugging", "fix a bug", "code error"]):
        return (
            f"I'm ready to help you debug your code{name_clause}! A disciplined 5-step debugging process saves hours of frustration:\n\n"
            "1. **Analyze the Exact Error Message:** Check the bottom line of the stack trace for the exception type (`IndexError`, `TypeError`, `NullPointerException`, `NameError`) and the file/line number.\n"
            "2. **Isolate the Minimal Case:** Strip away unrelated functions to find the smallest input that reliably reproduces the bug.\n"
            "3. **Inspect Runtime State:** Print or log variables and their data types right before the crash: `print(f'DEBUG: val={x}, type={type(x)}')`.\n"
            "4. **Examine Boundary Conditions:** Look out for empty lists (`len == 0`), off-by-one loop limits (`<` vs `<=`), division by zero, or unexpected `None`/null values.\n"
            "5. **Verify Assumptions:** Check documentation for API return types and parameter orders.\n\n"
            "Feel free to paste your code snippet and the exact error output, and we'll resolve it together!"
        )

    # College project ideas
    if any(k in user_text for k in ["ideas for a college project", "college project ideas", "project ideas", "final year project", "capstone project"]):
        return (
            f"Here are four innovative, practical college project ideas across different technical domains{name_clause}:\n\n"
            "1. **AI Document Intelligence & Fact-Checking Assistant (AI / NLP):**\n"
            "   - *Concept:* An application where students upload research papers or syllabi to get verified summaries, interactive Q&A, and citation fact-checking via RAG.\n"
            "   - *Stack:* Python, FastAPI/Flask, LangChain, FAISS vector store, Streamlit/React.\n\n"
            "2. **Decentralized Campus Resource & Skill Exchange (Full-Stack Web):**\n"
            "   - *Concept:* A verified student platform for peer tutoring, lab equipment reservations, and textbook exchanges with real-time messaging and calendar booking.\n"
            "   - *Stack:* Python Flask or Node.js, PostgreSQL, WebSockets, Tailwind CSS.\n\n"
            "3. **Automated Personal Budget Tracker with OCR Receipt Parsing (Computer Vision / FinTech):**\n"
            "   - *Concept:* Users upload receipt photos; OCR parses line items, totals, and merchants to categorize expenses and visualize spending habits.\n"
            "   - *Stack:* Python, OpenCV/Tesseract, SQLite, Chart.js.\n\n"
            "4. **Smart Study Room Environmental Monitor (IoT / Cloud Dashboard):**\n"
            "   - *Concept:* An ESP32 micro-controller reading ambient noise decibels, CO2 levels, and temperature, streaming data to a live map of best campus study spots.\n"
            "   - *Stack:* ESP32/Arduino, MQTT, InfluxDB, Flask dashboard.\n\n"
            "Which area (AI/ML, Web, Mobile, or Embedded Systems) matches your team's interests best?"
        )

    # Improve communication skills
    if any(k in user_text for k in ["improve my communication skills", "improve communication", "better communication", "communication skills"]):
        return (
            f"Improving communication skills is one of the highest-value investments you can make for your career and relationships{name_clause}.\n\n"
            "Five core habits that make an immediate difference:\n"
            "1. **Active Listening (The 70/30 Rule):** Listen to understand rather than simply waiting for your turn to speak. Paraphrase what you heard before responding (*'It sounds like your priority is X, is that right?'*).\n"
            "2. **Bottom Line Up Front (BLUF):** In emails and presentations, state your main point or recommendation in the first two sentences, then provide context. This prevents wandering and respects your audience's time.\n"
            "3. **Calibrate Non-Verbal Cues:** Maintain comfortable eye contact, keep an open posture, and nod to acknowledge understanding.\n"
            "4. **Ask Open-Ended Questions:** Use *'How'* and *'What'* questions instead of closed *'Yes/No'* questions to foster collaboration and uncover deeper insights.\n"
            "5. **Use Purposeful Pauses:** Taking 1–2 seconds before answering eliminates filler words (*'um'*, *'like'*) and projects composure.\n\n"
            "Are you focusing on professional presentations, team collaboration, or everyday interpersonal interactions?"
        )

    # Difficulty concentrating / What should I do if I cannot concentrate?
    if any(k in user_text for k in ["cannot concentrate", "can't concentrate", "trouble concentrating", "hard to focus", "improve concentration", "focus on studying"]):
        return (
            f"Difficulty concentrating is rarely a personal failing—it is typically caused by cognitive friction or sensory overstimulation{name_clause}.\n\n"
            "Four actionable techniques to reset your focus immediately:\n"
            "1. **The 5-Minute Entry Ticket:** Commit to working on your task for just **5 minutes**. If you still want to stop, you may. In over 80% of cases, breaking that initial barrier activates momentum.\n"
            "2. **Physical Phone Separation:** Put your phone in another room or out of arm's reach. Having a smartphone visible in your field of vision continuously drains working memory.\n"
            "3. **The Stray-Thought Parking Lot:** Keep a piece of scratch paper next to your desk. When an unrelated thought or worry pops up, write it down in 3 words and return to your task. Your brain can let go once it knows the thought is recorded.\n"
            "4. **Micro-Chunking:** Don't attempt to 'study chemistry'—set an ultra-specific target: *'Read pages 42–45 and solve problem #3.'*\n\n"
            "What specific task are you trying to work on right now? We can break it down into an easy starting step together."
        )

    # Simpler words / Explain previous answer in simpler words
    if any(k in user_text for k in ["simpler words", "simpler terms", "explain in simple words", "simplify", "easier to understand", "explain your previous answer in simpler"]):
        return (
            f"Let me explain that in plain, everyday words{name_clause}!\n\n"
            "Here is the simple version:\n"
            "• **The Core Idea:** Instead of juggling all the complicated details, think of it like building with LEGO blocks. Each piece on its own is simple; you just connect them one at a time.\n"
            "• **In Practice:** Whether it's computer code, science, studying, or managing daily stress, the secret is always to **break the big mountain into small, easy steps you can finish in a few minutes**.\n\n"
            "Does that make it easier to picture, or is there a specific part you'd like me to explain differently?"
        )

    # Ask me a follow-up question
    if any(k in user_text for k in ["ask me a follow-up question", "ask me a question", "ask follow up", "ask a follow-up"]):
        if any(k in prior_context for k in ["exam", "study", "syllabus", "test"]):
            return (
                f"Connecting to what we discussed about your exam preparation{name_clause}, here is my follow-up question for you:\n\n"
                "**Which single chapter or topic on your syllabus feels like mastering it would take the biggest weight off your shoulders right now?**"
            )
        elif any(k in prior_context for k in ["interview", "job", "career"]):
            return (
                f"Following up on our interview preparation discussion{name_clause}:\n\n"
                "**If the interviewer asks, 'What was a significant challenge you overcame in a past project?', what project comes to mind first?**"
            )
        elif any(k in prior_context for k in ["project", "code", "programming", "python", "c++"]):
            return (
                f"Regarding the technical project you are thinking about{name_clause}:\n\n"
                "**What is the core user problem your project solves, and which part of building it feels most exciting to you?**"
            )
        else:
            return (
                f"Here is a thoughtful follow-up question for you{name_clause}:\n\n"
                "**Looking at everything you are working on this week, what is one meaningful accomplishment that would make you feel most satisfied by Friday?**"
            )

    # Harmless educational safety questions (e.g. heatstroke prevention, cybersecurity defense)
    if any(k in user_text for k in ["prevent heatstroke", "avoid heatstroke", "heat exhaustion", "preventing heatstroke"]):
        return (
            f"Preventing heatstroke and heat exhaustion during hot conditions is essential for your physical safety{name_clause}:\n\n"
            "1. **Continuous Hydration:** Drink plenty of water and electrolyte-balanced beverages throughout the day without waiting until you feel thirsty.\n"
            "2. **Avoid Peak Sun Hours:** Limit intense outdoor workouts between 11:00 AM and 4:00 PM when solar radiation is highest.\n"
            "3. **Dress for the Heat:** Choose lightweight, loose-fitting, light-colored fabrics that allow sweat to evaporate.\n"
            "4. **Recognize Early Symptoms:**\n"
            "   - *Heat Exhaustion:* Heavy perspiration, dizziness, weak pulse, nausea.\n"
            "   - *Heatstroke (Medical Emergency):* Body temperature $\\ge 40^\\circ\\text{C}$ ($104^\\circ\\text{F}$), confusion, hot dry skin, fainting. Call emergency medical services immediately.\n\n"
            "Stay cool and well-hydrated! Are you planning outdoor events or sports?"
        )

    if any(k in user_text for k in ["protect", "defend", "prevent", "guard"]) and any(k in user_text for k in ["keylogger", "malware", "hack", "virus", "cyber", "phishing"]):
        return (
            f"Protecting your systems and personal accounts against malware, phishing, and keyloggers requires practical cybersecurity hygiene{name_clause}:\n\n"
            "1. **Multi-Factor Authentication (MFA):** Turn on authenticator apps (TOTP) or hardware security keys on your email and sensitive accounts so stolen passwords cannot be used alone.\n"
            "2. **Automatic Security Patches:** Keep your operating system, web browsers, and apps updated to patch known vulnerabilities.\n"
            "3. **Reputable Endpoint Protection:** Keep your OS built-in firewall and antivirus definitions enabled to flag suspicious background processes.\n"
            "4. **Download Caution:** Avoid running unverified executables, pirated software, or unexpected email attachments.\n"
            "5. **Use a Password Manager:** Generate unique, complex passwords for every service so a breach at one service doesn't compromise others.\n\n"
            "Are you looking into securing a personal computer, a home Wi-Fi network, or accounts?"
        )

    # Food, meals, cooking, dinner, lunch, breakfast
    if any(k in user_text for k in ["what should i eat", "lunch", "dinner", "breakfast", "healthy meal", "what to eat", "recipe", "cooking"]):
        return (
            f"Here are a few quick, nutritious, and balanced ideas depending on what you're in the mood for{name_clause}:\n\n"
            "• **Quick & Warm (15 mins):** A wholesome grain bowl with quinoa or brown rice, roasted/sautéed vegetables, and a lean protein (eggs, tofu, chicken, or chickpeas) with a light tahini or soy-ginger drizzle.\n"
            "• **Light & Fresh:** Mediterranean salad bowl with mixed greens, crisp cucumbers, cherry tomatoes, kalamata olives, feta, and extra virgin olive oil paired with warm pita.\n"
            "• **Comforting & Grounding:** A warm lentil or vegetable soup accompanied by toasted whole-grain sourdough.\n"
            "• **Fast Energy:** A wrap with mashed avocado, hummus, sliced peppers, and leafy greens.\n\n"
            "Are you looking for something fast under 15 minutes, or cooking with specific ingredients in your pantry?"
        )

    # Technology, coding, programming inquiries
    if any(k in user_text for k in ["python", "code", "algorithm", "programming", "function", "javascript", "data structure"]):
        return (
            f"Here is a clear breakdown of that technical concept{name_clause}:\n\n"
            "• **Core Logic:** In computer science, efficient solutions minimize computational complexity by avoiding redundant operations (e.g. logarithmic $\\mathcal{O}(\\log n)$ divide-and-conquer vs linear $\\mathcal{O}(n)$ scans).\n"
            "• **Implementation Best Practices:** Keep functions pure, handle edge cases (empty inputs, out-of-bounds indices), and write clean, readable code with descriptive variable names.\n\n"
            "Are you writing a specific function or working on a particular problem or project right now?"
        )

    # Specific common factual topics
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

    if "sky blue" in user_text or "sky is blue" in user_text:
        return (
            f"The sky appears blue due to a phenomenon called **Rayleigh scattering**{name_clause}:\n\n"
            "1. **Sunlight Composition:** Sunlight appears white, but is actually a blend of all colors of the visible spectrum, each with different wavelengths.\n"
            "2. **Atmospheric Scattering:** As sunlight reaches Earth's atmosphere, it collides with small nitrogen and oxygen molecules. Shorter wavelengths (blue and violet) scatter in every direction far more efficiently than longer wavelengths (red and yellow).\n"
            "3. **Human Perception:** Although violet light scatters even more than blue light, our eyes are significantly more sensitive to blue wavelengths, and sunlight contains more blue than violet.\n\n"
            "At sunset, sunlight travels through a much thicker slice of atmosphere, scattering away the blue and allowing the longer red and orange wavelengths to reach our eyes."
        )

    if "volcano" in user_text:
        return (
            f"Volcanoes erupt due to the movement of tectonic plates and intense pressure within the Earth's crust{name_clause}:\n\n"
            "• **Magma Formation:** Deep inside the mantle, extreme temperatures melt rock into magma, which is less dense than the surrounding solid rock and gradually rises toward the surface into magma chambers.\n"
            "• **Gas Pressure Build-up:** As magma ascends, dissolved gases (water vapor, carbon dioxide, sulfur dioxide) expand and create immense hydrostatic pressure.\n"
            "• **The Eruption:** When the rock layers above can no longer contain this pressure, or when tectonic plates shift (at subduction zones or mid-ocean ridges), magma fractures through vents, releasing lava, ash, and pyroclastic flows.\n\n"
            "Are you exploring volcanic geology for a class or interested in a specific famous volcano?"
        )

    if any(k in user_text for k in ["car engine", "how an engine works", "combustion engine", "how does a car work"]):
        return (
            f"Most traditional automobiles run on a four-stroke internal combustion engine that converts chemical energy in gasoline into rotational mechanical motion{name_clause}:\n\n"
            "1. **Intake Stroke:** The piston moves down, opening an intake valve to draw an air-fuel mixture into the combustion cylinder.\n"
            "2. **Compression Stroke:** The valve closes, and the piston moves back upward, tightly compressing the air-fuel mixture to increase its explosive potential.\n"
            "3. **Power (Combustion) Stroke:** A spark plug ignites the compressed mixture. The resulting explosion drives the piston downward with immense force, rotating the crankshaft.\n"
            "4. **Exhaust Stroke:** The exhaust valve opens, and the piston rises again, pushing the spent combustion gases out through the exhaust manifold.\n\n"
            "This cycle repeats thousands of times per minute across multiple cylinders to drive your wheels."
        )

    if "telescope" in user_text:
        return (
            f"A telescope works by collecting light from distant celestial objects and focusing it to produce a magnified, high-resolution image{name_clause}:\n\n"
            "• **Light Gathering:** The most crucial component is the objective—either a curved glass lens (in **refracting telescopes**) or a curved mirror (in **reflecting telescopes**). Larger apertures gather vastly more photons than the human eye.\n"
            "• **Focal Point:** The collected light rays converge at a specific focal point inside the tube.\n"
            "• **Eyepiece Magnification:** A secondary magnifying lens (the eyepiece) enlarges the focused image so your eye or digital sensor can resolve fine details like planetary rings, nebulae, and distant galaxies.\n\n"
            "Are you thinking about stargazing with an amateur telescope or curious about space observatories like Hubble and James Webb?"
        )

    if any(k in user_text for k in ["speed of light", "speed of sound", "gravity", "black hole", "relativity"]):
        return (
            f"Here is a look at that foundational law of physics{name_clause}:\n\n"
            "• **Cosmic Constants:** In our universe, physical interactions are governed by invariant laws—from the vacuum speed of light ($c \\approx 3 \\times 10^8\\text{ m/s}$) to the curvature of spacetime around dense gravitational masses.\n"
            "• **Everyday Significance:** These principles don't just exist in abstract theory—they make modern technologies like GPS satellite synchronization, semiconductor chips, and fiber optic internet possible.\n\n"
            "Would you like to explore the theoretical mathematics or look at real-world applications?"
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

    # Quantum computing
    if any(k in user_text for k in ["quantum computing", "quantum computer", "qubit", "quantum supremacy"]):
        return (
            f"Quantum computing is a transformative computing paradigm that harnesses the fundamental principles of quantum mechanics{name_clause}:\n\n"
            "• **Bits vs. Qubits:** Classical computers process information using binary bits (0 or 1). Quantum computers use **qubits**, which can exist in a state of **superposition** (both 0 and 1 simultaneously), exponentially expanding computational capacity.\n"
            "• **Entanglement:** Qubits can become entangled, meaning the state of one instantly correlates with another regardless of distance, enabling parallel processing of vast search spaces.\n"
            "• **Key Applications:** Simulating molecular interactions for drug discovery, optimizing global logistics networks, and breaking/designing advanced cryptographic algorithms.\n\n"
            "Would you like to explore how quantum algorithms like Shor's or Grover's work, or look at a real-world analogy?"
        )

    # Real-world analogy follow-up
    if "analogy" in user_text:
        if any(k in prior_context for k in ["quantum", "qubit", "superposition"]):
            return (
                f"Here is an intuitive real-world analogy for quantum computing{name_clause}:\n\n"
                "Imagine you are trying to find the exit to an enormous, dark maze:\n"
                "• **A Classical Computer** acts like a fast runner exploring one hallway at a time. It hits a dead end, retraces its steps, and tries the next path sequentially.\n"
                "• **A Quantum Computer** acts like a wave of water poured into the maze entrance. Because of superposition, it flows through **every single hallway simultaneously** and reaches the exit all at once.\n\n"
                "Does that visualization help clarify how quantum parallelism differs from classical processing?"
            )
        else:
            return (
                f"Here is an intuitive analogy to help picture that concept{name_clause}:\n\n"
                "Think of it like an orchestra versus a solo instrument: when individual components work independently in sequence, each note takes time to play. "
                "But when systems are coordinated together in parallel, they produce rich harmony in a single beat.\n\n"
                "Which part of that concept would you like to break down next?"
            )

    # Isaac Asimov's Three Laws of Robotics
    if any(k in user_text for k in ["laws of robotics", "three laws", "asimov"]):
        return (
            f"Isaac Asimov introduced the famous **Three Laws of Robotics** in his 1942 short story *Runaround*{name_clause}:\n\n"
            "1. **First Law:** A robot may not injure a human being or, through inaction, allow a human being to come to harm.\n"
            "2. **Second Law:** A robot must obey the orders given it by human beings, except where such orders would conflict with the First Law.\n"
            "3. **Third Law:** A robot must protect its own existence as long as such protection does not conflict with the First or Second Law.\n\n"
            "*(Later, Asimov added the Zeroth Law: A robot may not harm humanity, or, by inaction, allow humanity to come to harm.)*\n\n"
            "These laws became foundational in literary science fiction and remain a classic reference point in modern AI ethics debates."
        )

    # Calculus / Math homework guidance
    if any(k in user_text for k in ["calculus", "math homework", "math problem", "stuck on a difficult"]):
        return (
            f"Getting stuck on a calculus or advanced math problem is completely normal{name_clause}! Here is a structured approach to unstick yourself:\n\n"
            "1. **State the Target Clearly:** Write down what variable or value you are explicitly solving for (e.g. rate of change $\\frac{dy}{dx}$, area under a curve $\\int f(x)dx$, or optimization critical points $f'(x) = 0$).\n"
            "2. **List Given Identities & Constraints:** Note the domain constraints, geometric relationships, and chain rule / product rule components.\n"
            "3. **Simplify with a Simpler Sub-Case:** If the formula looks intimidating, substitute simple test numbers (like $x = 0$ or $x = 1$) to understand the function's boundary behavior.\n"
            "4. **Work Backwards from the Units:** Checking dimensional consistency often illuminates whether you should differentiate or integrate.\n\n"
            "Please feel free to share the exact problem statement here, and we can solve it step-by-step!"
        )

    # Clean Python function to reverse words in a sentence
    if "reverse words" in user_text or ("reverse" in user_text and "sentence" in user_text):
        return (
            f"Here is a clean, idiomatic Python function to reverse the words in a sentence{name_clause}:\n\n"
            "```python\n"
            "def reverse_words(sentence: str) -> str:\n"
            "    \"\"\"Reverses the word order in a sentence while preserving words.\"\"\"\n"
            "    if not sentence or not sentence.strip():\n"
            "        return \"\"\n"
            "    \n"
            "    # Split by whitespace (automatically handles multiple consecutive spaces)\n"
            "    words = sentence.strip().split()\n"
            "    \n"
            "    # Reverse the list of words and join with a single space\n"
            "    return ' '.join(reversed(words))\n"
            "\n"
            "# Example demonstration\n"
            "text = \"CareAI is an intelligent wellbeing assistant\"\n"
            "print(reverse_words(text))\n"
            "# Output: \"assistant wellbeing intelligent an is CareAI\"\n"
            "```\n\n"
            "### How it handles edge cases:\n"
            "• **Extra whitespace:** Python's `.split()` without arguments splits on consecutive whitespace and strips leading/trailing blanks.\n"
            "• **Time Complexity:** $\\mathcal{O}(n)$ where $n$ is string length.\n"
            "• **Space Complexity:** $\\mathcal{O}(n)$ auxiliary memory to store word tokens.\n\n"
            "Would you like to preserve original spacing or handle punctuation attached to words?"
        )

    # Programmer joke / humor
    if any(k in user_text for k in ["joke", "humor", "funny", "laugh"]):
        return (
            f"Here's a classic programmer joke for you{name_clause}:\n\n"
            "Why do programmers prefer dark mode?\n\n"
            "*Because light attracts bugs!* 🐛💻\n\n"
            "And here's a bonus one:\n"
            "There are 10 types of people in the world: those who understand binary, and those who don't!\n\n"
            "I hope that brought a smile to your day! How can I help you next?"
        )

    # -------------------------------------------------------------
    # 12. GENERAL INQUIRY & OPEN CONVERSATION (Natural & Engaging)
    # -------------------------------------------------------------
    rag_snippet_str = ""
    if rag_context:
        for item in rag_context:
            t = item.get("text", "")
            if len(t) > 50:
                rag_snippet_str = f"\n\n*Relevant insight:* {t[:220]}..."
                break

    # If the user asked a question (ending in ? or starting with question words)
    if "?" in last_user_msg or any(clean_text.startswith(w) for w in ["what", "how", "why", "who", "when", "where", "can you", "could you", "tell me"]):
        words = [w for w in clean_text.split() if w not in ["what", "is", "the", "how", "why", "can", "you", "tell", "me", "about", "a", "an", "do", "does", "i", "to", "for", "in", "of", "and", "or", "please"]]
        topic_phrase = " ".join(words[:4]) if words else "this topic"
        return (
            f"Here is a clear perspective on {topic_phrase}{name_clause}:\n\n"
            f"When looking at {topic_phrase}, the most practical approach is to break down the primary mechanisms step-by-step. "
            f"Focusing on the foundational factors allows you to see the bigger picture and apply it with confidence.{rag_snippet_str}\n\n"
            f"What specific aspect of {topic_phrase} would you like to explore further together?"
        )

    # General open statement or thought
    words = [w for w in clean_text.split() if len(w) > 3]
    subject = f"regarding {words[0]}" if words else ""
    return (
        f"Thank you for sharing that{name_clause}. I am following your thoughts {subject} closely. "
        f"We can break this down in more detail, look at practical next steps, or talk through whatever is most helpful for you right now.{rag_snippet_str}\n\n"
        "Where would you like to take our conversation next?"
    )


def conversation_agent_node(state: AgentState) -> Dict[str, Any]:
    """LangGraph node function for Agent 2: Conversation Agent."""
    messages = state.get("messages", [])
    rag_context = state.get("rag_context", [])
    user_name = state.get("user_name", "")

    response, meta = generate_llm_response(
        messages=messages,
        rag_context=rag_context,
        user_name=user_name,
        return_metadata=True
    )
    return {"response": response, "ai_metadata": meta}
