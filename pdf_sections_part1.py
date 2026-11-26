# pdf_sections_part1.py
"""Sections 1 through 5 of CareAI Technical Documentation."""

from reportlab.platypus import Paragraph, Spacer, PageBreak, KeepTogether, Table, TableStyle
from reportlab.lib import colors
from pdf_helpers import (
    p, p_bold, h1, h2, h3, bullet, callout, code_box, create_table,
    USABLE_WIDTH, STYLES
)

def build_cover_and_toc():
    story = []

    # Cover Page Banner / Title
    story.append(Spacer(1, 20))
    story.append(Paragraph("CareAI: AI-Powered Dynamic Mental Health Monitoring System", STYLES["CoverTitle"]))
    story.append(Paragraph(
        "Comprehensive Technical Analysis, Distributed Multi-Agent Architecture, Machine Learning Pipeline &amp; Full System Documentation",
        STYLES["CoverSubtitle"]
    ))

    # Meta Table
    meta_data = [
        ["Project Title", "CareAI – Dynamic Mental Health Capacity &amp; Supportive Guidance System"],
        ["Lead Author / Developer", "Vineeth K K (Sem 7 LLM &amp; AI Engineering Project)"],
        ["Architecture Pattern", "Coordinated Multi-Agent System (LangGraph) + RAG + Dual-Tier DB + Fine-Tuned DistilBERT"],
        ["Primary Technologies", "Python 3.12, Flask, LangGraph, Ollama (Llama 3.2), ChromaDB, HuggingFace, PostgreSQL/SQLite"],
        ["System Status", "v2.4 - Persistent Production Architecture (Render &amp; Localhost Verified)"],
        ["Test Coverage", "20/20 Test Suites Passing (100% Pass Rate across Core Scenarios)"],
        ["Documentation Date", "November 2026 | Comprehensive Academic &amp; Engineering Release"]
    ]
    t_meta = create_table(["Project Dimension", "Engineering Specification"], meta_data, col_widths=[140, 364])
    story.append(t_meta)
    story.append(Spacer(1, 15))

    story.append(callout(
        "Mandatory Clinical Safety Disclaimer",
        "CareAI is an academic research and self-monitoring system designed for educational, self-reflective, and emotional support. "
        "It does not provide clinical diagnosis, psychiatric evaluations, or medical prescriptions. In situations of acute psychological distress or "
        "self-harm ideation, the deterministic Safety Agent executes an unconditional priority override to present verified 24/7 crisis hotlines: "
        "Tele-MANAS (14416 / 1800 891 4416 in India), 988 Suicide &amp; Crisis Lifeline (US &amp; Canada), and 111 (UK).",
        kind="warning"
    ))

    story.append(PageBreak())

    # Table of Contents
    story.append(h1("Table of Contents"))
    story.append(Spacer(1, 6))

    toc_items = [
        ("Section 1", "Project Overview &amp; Problem Statement", "Purpose, real-world necessity, core objectives, clinical triage gap."),
        ("Section 2", "Complete System Architecture", "Multi-tier design, LangGraph state engine, component communication, data flow."),
        ("Section 3", "Complete Technology Stack", "Consolidated stack matrix, Python, Flask, LangGraph, Ollama, ChromaDB, DistilBERT."),
        ("Section 4", "Detailed Module-by-Module Explanation", "Exhaustive breakdown of all 15 Python modules, agents, services, and route controllers."),
        ("Section 5", "Complete Working Mechanism &amp; Workflow", "Lifecycle from boot to user interaction, mathematical scoring, RAG retrieval, synthesis."),
        ("Section 6", "AI/ML and Algorithmic Deep-Dive", "DistilBERT fine-tuning vs. TF-IDF Logistic Regression, HuggingFace dataset, metrics."),
        ("Section 7", "Database &amp; Data Management Architecture", "Dual-tier SQLite/PostgreSQL, schema tables, additive migrations, relational integrity."),
        ("Section 8", "API &amp; Backend Specification", "Complete RESTful endpoint catalog, methods, request/response JSON payloads, auth."),
        ("Section 9", "Frontend Architecture &amp; User Experience", "Mira Life dark-mode design system, Bootstrap 5.3, Chart.js time-series charts, templates."),
        ("Section 10", "Project Folder Structure", "Exhaustive inventory of every file and directory in the project repository."),
        ("Section 11", "Installation, Setup &amp; Deployment Guide", "Local setup, virtualenv, dependencies, Render cloud deployment, Docker, WSGI."),
        ("Section 12", "Security, Privacy &amp; Performance Engineering", "Scrypt hashing, CSRF, user isolation, RBAC, lazy loading, connection pooling."),
        ("Section 13", "Technical Challenges &amp; Implemented Solutions", "Hallucination guards, advice opt-out, cloud RAM limits, memory leaks, Windows PyArrow."),
        ("Section 14", "Advantages, Limitations &amp; Future Roadmap", "Architectural strengths, current boundaries, IoT/Wearables, Voice biomarkers, EHR."),
        ("Section 15", "Technical Interview Preparation Handbook", "35+ rigorous technical questions and answers spanning architecture, ML, DB, and ethics."),
        ("Section 16", "Final Project Summary &amp; Elevator Pitches", "Technical executive summary, 2-minute pitch, 5-minute academic viva, layman explanation.")
    ]

    t_toc = create_table(["Section", "Title", "Scope of Coverage"], toc_items, col_widths=[60, 180, 264])
    story.append(t_toc)
    story.append(Spacer(1, 15))

    return story


def build_section_1():
    story = []
    story.append(h1("1. Project Overview"))

    story.append(h2("1.1 Project Title and Purpose"))
    story.append(p(
        "The <b>AI-Powered Dynamic Mental Health Monitoring System (CareAI)</b> is a full-stack, distributed "
        "multi-agent clinical support and personal wellbeing platform. It bridges the critical operational gap between "
        "infrequent clinical therapy appointments and the day-to-day psychological experiences of individuals. "
        "By synthesizing daily quantitative check-ins, unconstrained reflective journals, conversational interactions, "
        "and supervised text classification, CareAI provides real-time, longitudinal visibility into an individual's "
        "mental health trajectory."
    ))

    story.append(h2("1.2 Problem Statement &amp; Real-World Necessity"))
    story.append(p(
        "Modern mental healthcare systems face three catastrophic systemic failure modes: severe clinical shortages, "
        "pervasive societal stigma, and longitudinal data fragmentation. Traditional psychological interventions rely on episodic "
        "consultations conducted weeks or months apart. During these sessions, clinicians must rely on subjective, recall-biased "
        "patient retrospectives (e.g., 'How have you felt over the past month?'). Patients often struggle to accurately recall "
        "fluctuations in anxiety, sleep deterioration, or escalating distress."
    ))
    story.append(p(
        "Furthermore, an estimated 70% of individuals experiencing psychological distress in developing and developed nations "
        "never seek professional care due to financial barriers, geographic distance, or fear of judgment. When crises occur "
        "(such as acute suicidal ideation or overwhelming panic), traditional health systems fail to provide immediate triage, "
        "leading to preventable tragedies. CareAI addresses this by offering a zero-stigma, private, continuous monitoring companion "
        "that mathematically computes distress trajectories and alerts human counselors before crises turn fatal."
    ))

    story.append(h2("1.3 Objectives and Expected Outcomes"))
    story.append(bullet("<b>Objective 1 (Empirical Longitudinal Tracking):</b> Record structured daily psychological parameters (Mood, Anxiety, Stress, Sleep Quality, Social Isolation) on standardized 1–10 scales, computing a deterministic normalized composite distress score (0.00 to 1.00)."))
    story.append(bullet("<b>Objective 2 (Linear Trajectory Regression):</b> Apply continuous mathematical regression analysis over rolling 7-day and 30-day windows to categorize trajectory into 'Improving', 'Stable', or 'Worsening', detecting chronic decline before clinical escalation."))
    story.append(bullet("<b>Objective 3 (Fail-Safe Safety Override):</b> Enforce a deterministic, multi-pattern lexical and score-based tripwire that intercepts self-harm, suicidal ideation, violence, and extreme distress, overriding chat responses with localized emergency resources."))
    story.append(bullet("<b>Objective 4 (Grounded Non-Hallucinatory Interventions):</b> Leverage ChromaDB vector search and Sentence-Transformers embeddings over curated psychological protocols (PMR, 5-4-3-2-1 grounding, sleep hygiene) to prevent generic or unsafe LLM advice."))
    story.append(bullet("<b>Objective 5 (Empathetic Adaptive Dialogue):</b> Deliver a versatile, intent-driven conversational companion powered by local Small Language Models (Ollama Llama 3.2), capable of topic switching, technical tutoring, and casual conversation without forcing clinical therapy scripts."))
    story.append(bullet("<b>Objective 6 (Clinical Human-in-the-Loop Triage):</b> Provide authorized licensed counselors with a dedicated portal featuring automated, hallucination-resistant clinical case summaries, longitudinal crisis narratives, and alert management."))

    story.append(h2("1.4 Key Features and Functionalities"))
    story.append(p(
        "The CareAI platform provides six core user-facing and clinical capabilities:"
    ))
    feat_data = [
        ["Feature Module", "Target User", "Core Engineering Implementation"],
        ["Dynamic Monitoring Dashboard", "Individual User", "KPI scorecards, rolling averages, linear trend badges, interactive Chart.js time-series plots."],
        ["Interactive Daily Check-In", "Individual User", "Dual-label range sliders (1-10), instant client-side preview, server-side normalization formula."],
        ["Reflective Journal Analysis", "Individual User", "Free-form reflection space, advice opt-out detector, Ollama sentiment tagger, gentle validation."],
        ["Conversational AI Companion", "Individual User", "Multi-turn context continuity, anaphora resolution, rapid topic pivoting, emergency tripwire."],
        ["Clinical Counselor Portal", "Licensed Counselors", "Caseload triage table, unresolved alert workflow, clinical case notes, audit timestamps."],
        ["Automated Case Synthesis", "Licensed Counselors", "Longitudinal crisis narrative synthesis, factual grounding without hallucination or clinical DSM labeling."]
    ]
    story.append(create_table(["Feature Module", "Target User", "Implementation Detail"], feat_data, col_widths=[110, 85, 309]))
    story.append(Spacer(1, 10))

    story.append(h2("1.5 Real-World Applications and Use Cases"))
    story.append(p(
        "<b>1. Higher Education &amp; University Campuses:</b> Deployed as a student wellbeing portal where academic stress, "
        "exam pressure, and dorm isolation are monitored continuously, providing immediate grounding techniques and notifying campus counselors of severe declines.<br/>"
        "<b>2. Corporate Employee Assistance Programs (EAPs):</b> Integrated into enterprise intranets to mitigate occupational burnout, "
        "chronic stress, and sleep deprivation while preserving strict worker anonymity.<br/>"
        "<b>3. Outpatient Psychiatric &amp; Psychological Care:</b> Deployed by therapy clinics between outpatient sessions, "
        "supplying therapists with objective, quantified time-series charts and AI-synthesized narrative timelines before each weekly consultation.<br/>"
        "<b>4. Tele-Health &amp; Crisis Helpline Triage:</b> Integrated into regional crisis triage pipelines to route high-acuity individuals "
        "directly to human tele-counselors while providing instant grounding exercises to waiting callers."
    ))

    story.append(Spacer(1, 15))
    return story


def build_section_2():
    story = []
    story.append(h1("2. Complete System Architecture"))

    story.append(h2("2.1 High-Level Architectural Paradigm"))
    story.append(p(
        "CareAI is engineered according to a <b>Tiered Micro-Modular Hybrid AI Architecture</b>. "
        "Rather than delegating all responsibilities to an unconstrained, opaque Large Language Model (which risks clinical hallucinations, "
        "stochastic variability, and non-deterministic safety enforcement), CareAI decouples cognitive responsibilities into "
        "distinct deterministic, heuristic, vector-retrieval, deep-learning, and generative components orchestrated by a stateful graph."
    ))

    story.append(h2("2.2 System Architecture Diagram"))
    diag_text = """
+===================================================================================================+
|                                    USER PRESENTATION LAYER                                        |
|  [Bootstrap 5.3 Dark Theme | Responsive Grid | Chart.js Visualizer | CSRF-Guarded Fetch API]      |
+=================================================^=================================================+
                                                  | HTTP / JSON (Session Auth Cookie)
+=================================================v=================================================+
|                              FLASK APPLICATION GATEWAY & CORE ENGINE                              |
|  - App Factory (app.py)       - Blueprints (Auth, Checkin, Chat, Journal, Dashboard, Counselor)   |
|  - Session / CSRF Security    - RBAC Authorization Decorators (@counselor_required)               |
+=================================================^=================================================+
                                                  | Internal Python Method Dispatch
+=================================================v=================================================+
|                            LANGGRAPH MULTI-AGENT ORCHESTRATION ENGINE                             |
|                                                                                                   |
|    +-----------------------------+                                                                |
|    |  Agent 1: Orchestrator Node | <--- Initializes AgentState (TypedDict)                        |
|    +--------------+--------------+                                                                |
|                   |                                                                               |
|         +---------+---------+                                                                     |
|         |                   |                                                                     |
|         v                   v                                                                     |
|  +--------------+    +--------------+                                                             |
|  |   Agent 3:   |    |   Agent 5:   |                                                             |
|  |   Distress   |    |    Safety    |                                                             |
|  |   Analysis   |    |    Agent     |                                                             |
|  +-------+------+    +-------+------+                                                             |
|          |                   |                                                                    |
|          +-------->+<--------+                                                                    |
|                    |                                                                              |
|                    v                                                                              |
|           [Critical Override?] -- YES --> [Safety Override Node] ---> END (Helpline 988/14416)    |
|                    | NO                                                                           |
|                    v                                                                              |
|         +---------------------+                                                                   |
|         |       Agent 4:      |                                                                   |
|         |   Monitoring Agent  |                                                                   |
|         +----------+----------+                                                                   |
|                    |                                                                              |
|                    v                                                                              |
|         +---------------------+                                                                   |
|         |       Agent 6:      | <--- ChromaDB Vector Index (all-MiniLM-L6-v2)                     |
|         | Recommendation Agent|                                                                   |
|         +----------+----------+                                                                   |
|                    |                                                                              |
|         +----------+----------+--------------------+                                              |
|         |                     |                    |                                              |
|         v                     v                    v                                              |
|  +--------------+      +--------------+     +--------------+                                      |
|  |   Agent 2:   |      | Reflective   |     | Orchestrator |                                      |
|  | Conversation |      | Journal Node |     |  Synthesis   |                                      |
|  +-------+------+      +-------+------+     +-------+------+                                      |
|          |                     |                    |                                             |
|          +-------------------->+<-------------------+                                             |
|                                |                                                                  |
|                                v                                                                  |
|                               END                                                                 |
+===================================================================================================+
|                                    PERSISTENCE & STORAGE TIER                                     |
|  - Canonical SQLite (instance/mhapp.db) / Cloud PostgreSQL (Render) via SQLAlchemy ORM            |
|  - ChromaDB Local Vector Store (instance/chroma_db) with Sentence-Transformers Embeddings         |
|  - Fine-Tuned DistilBERT Classifier (models_saved/distilbert) / TF-IDF Logistic Regression        |
+===================================================================================================+
"""
    story.append(code_box(diag_text))
    story.append(Spacer(1, 10))

    story.append(h2("2.3 Component Responsibilities and Layer Decoupling"))
    story.append(p(
        "<b>1. Presentation Layer:</b> Built with HTML5, Jinja2 templating, Bootstrap 5.3 dark-mode styling, and Chart.js. "
        "Interacts via asynchronous REST API endpoints protected by CSRF tokens injected into meta tags.<br/>"
        "<b>2. Application Gateway Layer:</b> Flask Application Factory (`create_app`) instantiates modular Blueprints, "
        "configures Werkzeug scrypt hashing, manages 14-day persistent cookies, and enforces Role-Based Access Control (RBAC).<br/>"
        "<b>3. Multi-Agent Orchestration Tier:</b> LangGraph coordinates execution across a typed state graph (`AgentState`), "
        "ensuring that every node receives structured inputs, performs isolated computation, and commits updates deterministically.<br/>"
        "<b>4. Specialized AI Agent Tier:</b> Six discrete agents handle Distress Evaluation, Longitudinal Monitoring, "
        "Safety Interception, RAG Vector Retrieval, Conversational Reasoning, and Journal Analysis.<br/>"
        "<b>5. Machine Learning &amp; Deep Learning Tier:</b> A standalone PyTorch/DistilBERT transformer and scikit-learn pipeline "
        "trained on 48,950 clinical text entries provide non-diagnostic supportive text classification.<br/>"
        "<b>6. Persistence Tier:</b> Dual-tier SQLAlchemy data persistence supporting zero-config canonical SQLite locally "
        "and production PostgreSQL in the cloud, accompanied by an additive schema migration engine."
    ))

    story.append(h2("2.4 Inter-Component Communication Protocols"))
    story.append(p(
        "Communication across system boundaries follows strict protocol standards:<br/>"
        "• <b>Frontend to Backend:</b> Asynchronous JSON over HTTPS using standard HTTP verbs (`GET`, `POST`). Requests carry `X-CSRFToken` headers and `careai_session` cookies.<br/>"
        "• <b>Backend to LangGraph Agents:</b> Direct synchronous function invocation passing a shared dictionary (`AgentState`). Nodes return partial dictionary updates merged by LangGraph.<br/>"
        "• <b>Agents to Local LLM (Ollama):</b> HTTP REST requests over `http://127.0.0.1:11434/api/generate` (or Cloudflare tunnel in cloud deployments) with JSON schemas, 8-second timeouts, and automatic fallback.<br/>"
        "• <b>Agents to Vector Store:</b> In-process ChromaDB Python API using cosine similarity embeddings generated by `sentence-transformers/all-MiniLM-L6-v2`.<br/>"
        "• <b>Backend to Database:</b> SQLAlchemy 2.0 object-relational mapping using connection pooling (`pool_pre_ping=True`, `pool_recycle=300`)."
    ))

    story.append(h2("2.5 Complete End-to-End Data Flow"))
    story.append(p(
        "<b>Step 1 (Input Acquisition):</b> The user inputs daily scores via the check-in sliders (1–10) or submits free-text via chat or journal.<br/>"
        "<b>Step 2 (State Ingestion):</b> Flask route captures the payload, resolves `current_user.id`, persists the raw entry to the database, and calls `orchestrator.handle_checkin()` or `handle_chat()`.<br/>"
        "<b>Step 3 (Distress Calculation):</b> Agent 3 computes the normalized formula, generating a score between 0.00 and 1.00 and assigning a category.<br/>"
        "<b>Step 4 (Safety Triaging):</b> Agent 5 scans for lexical trigger phrases and checks if the composite score exceeds 0.85. If triggered, execution short-circuits immediately to `safety_override_node`.<br/>"
        "<b>Step 5 (Historical Regression):</b> If safe, Agent 4 queries historical check-ins, calculates 7d/30d rolling averages, and determines linear slope.<br/>"
        "<b>Step 6 (RAG Retrieval):</b> Agent 6 maps distress indicators to ChromaDB and retrieves relevant grounding exercises.<br/>"
        "<b>Step 7 (Conversational / Feedback Synthesis):</b> Agent 2 or the Journal Agent executes an LLM prompt grounded in the RAG context and user input.<br/>"
        "<b>Step 8 (Persistence &amp; Response):</b> Results are committed to database tables (`checkins`, `chat_messages`, `alerts`), and a structured JSON payload is returned to the frontend."
    ))

    story.append(Spacer(1, 15))
    return story


def build_section_3():
    story = []
    story.append(h1("3. Complete Technology Stack"))

    story.append(h2("3.1 Consolidated Technology Stack Matrix"))
    stack_data = [
        ["Category", "Technology", "Version", "Role &amp; Architectural Location"],
        ["Language", "Python", "3.12 / 3.13", "Primary backend, data science, and orchestration programming language."],
        ["Web Framework", "Flask", "3.0.3", "Lightweight WSGI web server framework, routing, request dispatching."],
        ["WSGI Server", "Gunicorn", "26.0.0", "Production pre-forking WSGI HTTP server binding `wsgi:app` on Linux/Render."],
        ["Security / Forms", "Flask-WTF", "1.2.1", "Cross-Site Request Forgery (CSRF) protection and form validation."],
        ["Authentication", "Flask-Login", "0.6.3", "User session lifecycle management, login persistence, `@login_required`."],
        ["Cryptography", "Werkzeug", "3.1.3", "Secure password hashing using salted `scrypt` (`generate_password_hash`)."],
        ["Agent Engine", "LangGraph", "0.0.20+", "Stateful cyclical multi-agent graph orchestration and conditional routing."],
        ["Agent Core", "LangChain-Core", "0.1.53+", "Base abstractions, message structures, and runnable state schemas."],
        ["Local LLM", "Ollama", "0.1.6+", "Local Small Language Model inference engine running `llama3.2` (3B parameters)."],
        ["Vector Database", "ChromaDB", "0.5.0+", "Persistent on-disk vector database storing mental health protocols."],
        ["Embeddings", "Sentence-Transformers", "5.4.1", "`all-MiniLM-L6-v2` generating 384-dimensional dense semantic vectors."],
        ["Deep Learning", "PyTorch", "2.11.0", "Deep neural network framework executing DistilBERT transformer models."],
        ["Transformers", "HuggingFace Transformers", "5.8.0", "DistilBERT model loading, tokenization, and fine-tuning."],
        ["Machine Learning", "Scikit-Learn", "1.7.2", "TF-IDF vectorization, Logistic Regression classifier, stratified cross-validation."],
        ["Data Processing", "Pandas &amp; NumPy", "2.3.3 / 2.2.4", "Tabular dataset wrangling, linear regression slopes, rolling averages."],
        ["ORM", "Flask-SQLAlchemy", "3.1.1", "Database object-relational mapping, connection pooling, model schemas."],
        ["Database Drivers", "psycopg2-binary / pymysql", "2.9.9 / 1.1.1", "Production PostgreSQL and MySQL database drivers."],
        ["Frontend UI", "Bootstrap", "5.3.3", "Responsive grid, dark-mode glassmorphic components, modal dialogs."],
        ["Visual Icons", "Bootstrap Icons", "1.11.3", "Comprehensive vector icon set embedded throughout UI components."],
        ["Data Viz", "Chart.js", "4.x (CDN)", "Client-side interactive multi-line charts rendering longitudinal distress."],
        ["Automated Testing", "PyTest", "9.2.0", "Comprehensive test runner executing 20+ suites across unit and system tests."],
        ["Cloud Deploy", "Render Cloud", "Blueprint v1", "Infrastructure-as-Code (`render.yaml`) auto-deploying Web + PostgreSQL."]
    ]
    story.append(create_table(["Category", "Technology", "Version", "Role &amp; Architectural Location"], stack_data, col_widths=[75, 95, 55, 279]))
    story.append(Spacer(1, 10))

    story.append(h2("3.2 In-Depth Analysis of Individual Technologies"))

    story.append(h3("1. Python (v3.12 / v3.13)"))
    story.append(p(
        "<b>What it is:</b> A high-level, dynamically typed multi-paradigm programming language.<br/>"
        "<b>Why used:</b> Python is the undisputed gold standard for AI, machine learning, and NLP engineering, offering "
        "seamless integration across PyTorch, HuggingFace, ChromaDB, and Flask.<br/>"
        "<b>Where used:</b> Across the entire backend codebase: `app.py`, `agents/*.py`, `ml/*.py`, `routes/*.py`, `services/*.py`.<br/>"
        "<b>Role &amp; Interoperability:</b> Executes business logic, interfaces with C-level mathematical libraries (NumPy, PyTorch), "
        "and serves HTTP endpoints via Flask."
    ))

    story.append(h3("2. Flask &amp; Werkzeug (v3.0.3 / v3.1.3)"))
    story.append(p(
        "<b>What it is:</b> A modular, micro web framework and WSGI utility library.<br/>"
        "<b>Why used:</b> Provides clean routing and Blueprint modularity without the rigid overhead of Django, allowing "
        "direct integration with custom background AI loops.<br/>"
        "<b>Where used:</b> Application factory in `app.py`, route Blueprints in `routes/`, extension initialization in `extensions.py`.<br/>"
        "<b>Role &amp; Interoperability:</b> Handles HTTP request lifecycles, parses JSON payloads, sets secure cookies, and "
        "dispatches requests to LangGraph workflows."
    ))

    story.append(h3("3. LangGraph &amp; LangChain-Core (v0.0.20+ / v0.1.53+)"))
    story.append(p(
        "<b>What it is:</b> An orchestration framework for building stateful, multi-agent cyclical graphs.<br/>"
        "<b>Why used:</b> Unlike simple linear chains, mental health monitoring requires dynamic routing, conditional branching "
        "(e.g., immediate safety override vs. normal conversation), and shared immutable state accumulation.<br/>"
        "<b>Where used:</b> `agents/graph.py`, `agents/state.py`, `agents/orchestrator.py`.<br/>"
        "<b>Role &amp; Interoperability:</b> Manages the global `AgentState` TypedDict, executing agents as coordinated nodes "
        "connected by conditional edges."
    ))

    story.append(h3("4. Ollama &amp; Llama 3.2 (3B)"))
    story.append(p(
        "<b>What it is:</b> A lightweight, high-performance local inference server running quantized Small Language Models.<br/>"
        "<b>Why used:</b> Transmitting sensitive patient reflections to proprietary cloud APIs (such as OpenAI or Anthropic) "
        "violates patient privacy and incurs recurring API costs. Local SLMs preserve 100% data sovereignty.<br/>"
        "<b>Where used:</b> `agents/conversation.py`, `agents/journal_agent.py`, `agents/case_summary_agent.py`.<br/>"
        "<b>Role &amp; Interoperability:</b> Receives structured prompts via HTTP REST and generates empathetic dialogue, "
        "sentiment tags, and clinical syntheses, backed by a deterministic fallback engine if Ollama is unreachable."
    ))

    story.append(h3("5. ChromaDB &amp; Sentence-Transformers"))
    story.append(p(
        "<b>What it is:</b> An open-source, embedded vector database paired with dense neural sentence embeddings.<br/>"
        "<b>Why used:</b> Powers Retrieval-Augmented Generation (RAG) by converting evidence-based clinical coping protocols "
        "into 384-dimensional mathematical vectors, enabling semantic retrieval without model fine-tuning.<br/>"
        "<b>Where used:</b> `services/rag.py`, `agents/recommendation.py`, indexed in `instance/chroma_db`.<br/>"
        "<b>Role &amp; Interoperability:</b> Queries indexed documents (`knowledge_base/*.txt`) using cosine similarity, injecting "
        "exact protocol steps into LLM prompts and dashboard recommendation widgets."
    ))

    story.append(h3("6. DistilBERT &amp; Scikit-Learn (Text Classification Engine)"))
    story.append(p(
        "<b>What it is:</b> A distilled, 66-million parameter transformer model (`distilbert-base-uncased`) and classical "
        "TF-IDF Logistic Regression pipeline.<br/>"
        "<b>Why used:</b> Provides fast, reproducible, supervised text classification over user disclosures into 4 classes "
        "(Normal, Depression, Anxiety, Suicidal) to serve as a supportive analytical signal.<br/>"
        "<b>Where used:</b> `ml/train_bert.py`, `ml/train_lr.py`, `ml/inference.py`, `models_saved/distilbert/`.<br/>"
        "<b>Role &amp; Interoperability:</b> Delivers non-diagnostic inference via `classify_text()` within `agents/distress.py`, "
        "achieving 82.69% accuracy on the test partition."
    ))

    story.append(Spacer(1, 15))
    return story


def build_section_4():
    story = []
    story.append(h1("4. Detailed Module-by-Module Explanation"))

    story.append(p(
        "CareAI is partitioned into 15 logical modules across application core, agents, ML pipeline, routes, services, and models:"
    ))

    modules = [
        ("Module 1: Core Application Factory", "app.py, config.py, extensions.py, wsgi.py",
         "Initializes extensions (SQLAlchemy, LoginManager, CSRFProtect), configures environment variables, "
         "registers the 6 route Blueprints, exposes the `/health` diagnostic ping, and triggers safe schema migrations.",
         "Input: Environment variables (.env). Output: Configured Flask WSGI application instance."),

        ("Module 2: State Definition & Graph Orchestrator", "agents/state.py, agents/graph.py, agents/orchestrator.py",
         "Defines `AgentState` TypedDict containing user demographics, action types, distress metrics, RAG context, and override flags. "
         "Builds the StateGraph with conditional edges and provides the `Orchestrator` wrapper singleton.",
         "Input: User action & text. Output: Synthesized response dictionary with distress scores and recommendations."),

        ("Module 3: Distress Analysis Agent", "agents/distress.py",
         "Computes the normalized deterministic formula across 5 check-in metrics. Evaluates conversational sentiment and "
         "lazy-loads the ML inference classifier as a secondary signal.",
         "Input: Mood, anxiety, stress, sleep, isolation (1-10). Output: Normalized score (0.00-1.00), category, breakdown."),

        ("Module 4: Longitudinal Monitoring Agent", "agents/monitoring.py",
         "Extracts chronological check-in history from the database, computes 7-day and 30-day rolling averages, "
         "and performs linear regression over distress scores to classify trajectory as Improving, Stable, or Worsening.",
         "Input: Historical check-in records. Output: Rolling averages, regression slope, trajectory label, summary string."),

        ("Module 5: Deterministic Safety Agent", "agents/safety.py",
         "Executes dual-stage safety verification: evaluates compiled regex patterns for suicidal ideation, self-harm, "
         "violence, and weapon creation, and checks if distress score >= 0.85. Tripping sets `priority_override=True` and logs a SafetyAlert.",
         "Input: Text disclosures & distress score. Output: Safety boolean, risk level, action taken, verified crisis hotlines."),

        ("Module 6: Recommendation Agent", "agents/recommendation.py",
         "Maps dominant distress metrics to structured wellbeing exercises (5-4-3-2-1 Grounding, 4-7-8 Breathing, PMR) "
         "and queries ChromaDB RAG for evidence-based coping protocols.",
         "Input: Distress level & breakdown. Output: Ranked list of recommended evidence-based wellbeing exercises."),

        ("Module 7: Conversational AI Companion", "agents/conversation.py",
         "Maintains empathetic dialogue, multi-turn context memory, and anaphora resolution. Identifies user intent "
         "(scientific facts, code debugging, academic schedules, or emotional support) and pivots seamlessly without robotic therapy disclaimers.",
         "Input: Message history & user persona. Output: Context-aware, natural, empathetic response string."),

        ("Module 8: Reflective Journal Analysis Agent", "agents/journal_agent.py",
         "Analyzes private user journal reflections. Features `detect_advice_opt_out()` to respect venting boundaries, "
         "classifying sentiment (Positive, Reflective, Distressed, Overwhelmed) and returning gentle validation.",
         "Input: Raw journal reflection text. Output: Structured JSON containing sentiment, nuances, and feedback."),

        ("Module 9: Clinical Crisis Narrative Agent", "agents/narrative_agent.py",
         "Synthesizes longitudinal journal entries, chat messages, and check-in trends into an objective, factual timeline "
         "for counselors, strictly preventing psychiatric diagnostic labels or fabricated trauma backstories.",
         "Input: Patient longitudinal history. Output: Factual chronological clinical narrative synthesis."),

        ("Module 10: Clinical Case Summary Generator", "agents/case_summary_agent.py",
         "Generates a 7-section structured clinical summary (Reported Concerns, Recurring Patterns, Changes Over Time, Risk Indicators) "
         "for licensed counselors, backed by an intelligent deterministic fallback engine.",
         "Input: Patient database record & history. Output: Formatted 7-section clinical case summary object."),

        ("Module 11: Machine Learning & NLP Pipeline", "ml/preprocess.py, ml/train_lr.py, ml/train_bert.py, ml/inference.py",
         "Cleans raw HuggingFace dataset (48,950 samples), trains TF-IDF Logistic Regression baseline, fine-tunes DistilBERT transformer, "
         "evaluates confusion matrices, and exposes `classify_text()` with confidence scoring.",
         "Input: Raw text string. Output: Label (Normal, Depression, Anxiety, Suicidal) and confidence score."),

        ("Module 12: Knowledge Base & RAG Service", "services/rag.py, knowledge_base/*.txt",
         "Maintains ChromaDB vector collection `mental_health_knowledge`, embedding curated guidelines using Sentence-Transformers. "
         "Includes memory-efficient in-memory fallback for constrained cloud containers (Render 512MB RAM).",
         "Input: Query string. Output: Top-k semantically relevant evidence-based text chunks."),

        ("Module 13: Counselor Notification Service", "services/notifier.py",
         "Dispatches in-app and optional SMTP email notifications to authorized human counselors when high-acuity alerts occur, "
         "preserving privacy by excluding raw journal contents from email bodies.",
         "Input: Alert ID, user ID, risk level. Output: Count of dispatched notifications."),

        ("Module 14: Additive Schema Migration Engine", "services/db_migration.py",
         "Inspects database tables at startup and safely adds missing columns (e.g. `distress_score` in chat, `role` in users) "
         "without dropping tables, altering types, or wiping persistent user data across SQLite, PostgreSQL, and MySQL.",
         "Input: SQLAlchemy engine metadata. Output: Idempotent additive schema updates."),

        ("Module 15: Web Controller Routing Layer", "routes/auth.py, routes/dashboard.py, routes/checkin.py, routes/chat.py, routes/journal.py, routes/counselor.py",
         "Implements Flask route controllers, enforces session authentication, validates form inputs with WTForms, and "
         "renders Jinja2 templates or returns JSON API responses.",
         "Input: HTTP client requests. Output: HTML web views or JSON payloads.")
    ]

    for mod_title, mod_files, mod_desc, mod_io in modules:
        story.append(h2(mod_title))
        story.append(p_bold(f"Files: {mod_files}"))
        story.append(p(mod_desc))
        story.append(callout("Input / Output Interface", mod_io, kind="note"))
        story.append(Spacer(1, 4))

    story.append(Spacer(1, 10))
    return story


def build_section_5():
    story = []
    story.append(h1("5. Complete Working Mechanism &amp; Workflow"))

    story.append(h2("5.1 Complete Step-by-Step System Lifecycle"))
    story.append(p(
        "<b>Phase 1: Boot &amp; Environment Initialization:</b><br/>"
        "1. The WSGI server (`gunicorn wsgi:app`) or CLI (`python run.py`) calls `create_app()`.<br/>"
        "2. `config.py` loads `.env`, detects cloud presence (`RENDER`), and resolves the database connection string. If `DATABASE_URL` exists, PostgreSQL is configured with connection pooling (`pool_size=10`, `pool_recycle=300`). Otherwise, an absolute canonical SQLite path (`instance/mhapp.db`) is locked.<br/>"
        "3. Database tables are verified and `services/db_migration.py` executes additive schema migrations.<br/>"
        "4. `services/rag.py` lazily initializes ChromaDB. If running on Render with 512MB RAM limits, an in-memory keyword retrieval fallback is activated to prevent out-of-memory container crashes.<br/>"
        "5. Blueprints are registered and the HTTP server binds port 5000."
    ))

    story.append(p(
        "<b>Phase 2: User Onboarding &amp; Authentication:</b><br/>"
        "1. The user registers at `/register`. Email is lowercased and stripped of whitespace.<br/>"
        "2. `User.set_password()` hashes the password using Werkzeug's `scrypt` algorithm with salt, storing a 256-character hash.<br/>"
        "3. Upon login at `/login`, `check_password_hash()` verifies credentials, and `login_user(user, remember=True)` issues a 14-day persistent cookie (`careai_session`) with `HttpOnly` and `SameSite=Lax`."
    ))

    story.append(p(
        "<b>Phase 3: Daily Check-In Processing:</b><br/>"
        "1. The user visits `/checkin`, adjusts the five dual-label sliders (1 to 10), and clicks 'Submit Daily Check-In'.<br/>"
        "2. Client JavaScript computes an instant visual preview and dispatches a POST request with CSRF token to `/api/checkin`.<br/>"
        "3. Route handler verifies integers (1–10) and calls `orchestrator.handle_checkin()`.<br/>"
        "4. LangGraph initiates `orchestrator_entry_node`, setting `action_type = 'checkin'`.<br/>"
        "5. `distress_agent_node` executes the deterministic formula, producing `distress_score` and `distress_level`.<br/>"
        "6. `safety_agent_node` checks if score >= 0.85. If safe, execution flows to `monitoring_agent_node`.<br/>"
        "7. `monitoring_agent_node` extracts historical check-ins for `current_user.id`, calculating rolling averages and regression slope.<br/>"
        "8. `recommendation_agent_node` matches distress indicators to ChromaDB RAG and wellbeing exercises.<br/>"
        "9. `orchestrator_synthesis_node` compiles a personalized supportive summary.<br/>"
        "10. Database commits a new `CheckIn` record and returns JSON containing score, trajectory, and coping recommendations."
    ))

    story.append(p(
        "<b>Phase 4: Conversational AI Interaction:</b><br/>"
        "1. The user navigates to `/chat` and types a query.<br/>"
        "2. `POST /api/chat` persists the user message to `chat_messages` table and invokes `orchestrator.handle_chat()`.<br/>"
        "3. LangGraph routes through `safety_agent_node`. Regex patterns scan for self-harm or violence.<br/>"
        "4. If safe, `recommendation_agent_node` fetches contextual wellbeing protocols, and `conversation_agent_node` formats a prompt with past conversation history for Ollama Llama 3.2.<br/>"
        "5. Ollama generates a response. If Ollama is offline or times out (8s), the built-in intelligent fallback engine delivers a high-quality context-aware answer.<br/>"
        "6. Response is saved as an assistant message in `chat_messages` and streamed to the frontend."
    ))

    story.append(p(
        "<b>Phase 5: Emergency Escalation &amp; Counselor Triage:</b><br/>"
        "1. If a message contains explicit crisis keywords (e.g. 'kill myself', 'want to die'), `safety_agent_node` trips `priority_override = True`.<br/>"
        "2. Normal conversation is suppressed. The system returns verified emergency resources (Tele-MANAS 14416 / 988 Lifeline).<br/>"
        "3. A `SafetyAlert` record is committed with `status = 'unresolved'` and `risk_level = 'critical'`.<br/>"
        "4. `notifier.py` dispatches in-app notifications to all registered counselors.<br/>"
        "5. A counselor logs into `/counselor/dashboard`, inspects the patient's timeline, reviews longitudinal charts, generates an AI Case Summary, and updates the alert status to 'In Review' or 'Resolved'."
    ))

    story.append(h2("5.2 End-to-End Workflow Diagram"))
    flow_text = """
 [User Action] ---> Check-in / Chat / Journal
        |
        v
 [Flask Route] ---> Authenticate Session (user_id = current_user.id) + Check CSRF
        |
        v
 [DB Persistence] -> Record raw entry in checkins / chat_messages / journals
        |
        v
 [Orchestrator] ---> Initialize AgentState (TypedDict)
        |
        +---> [Distress Agent] ---> Compute Deterministic Formula (0.00 - 1.00)
        |                                    |
        |                                    v
        +---> [Safety Agent] -----> Scan Lexical Triggers + Check Distress >= 0.85
                                             |
                     +-----------------------+-----------------------+
                     |                                               |
             [SAFETY TRIPPED]                                  [SAFETY PASSED]
                     |                                               |
                     v                                               v
        [Create SafetyAlert in DB]                    [Monitoring Agent: 7d/30d Slope]
                     |                                               |
        [Dispatch Counselor Alert]                                   v
                     |                                [Recommendation Agent: ChromaDB RAG]
                     v                                               |
        [Safety Override Response]                                   v
        (988 / 14416 Crisis Helplines)                [Conversation / Journal Agent (Ollama)]
                     |                                               |
                     v                                               v
                   [END]                                  [Orchestrator Synthesis]
                                                                     |
                                                                     v
                                                            [Return JSON to Client]
                                                                     |
                                                                     v
                                                       [Update Charts & UI Stream]
"""
    story.append(code_box(flow_text))
    story.append(Spacer(1, 15))

    return story
