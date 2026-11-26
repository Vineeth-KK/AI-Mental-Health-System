# pdf_sections_part2.py
"""Sections 6 through 10 of CareAI Technical Documentation."""

from reportlab.platypus import Paragraph, Spacer, PageBreak, KeepTogether, Table, TableStyle
from reportlab.lib import colors
from pdf_helpers import (
    p, p_bold, h1, h2, h3, bullet, callout, code_box, create_table,
    USABLE_WIDTH, STYLES
)

def build_section_6():
    story = []
    story.append(h1("6. AI/ML, NLP, and Deep Learning Algorithmic Deep-Dive"))

    story.append(h2("6.1 Hybrid AI Algorithmic Rationale"))
    story.append(p(
        "A central design innovation of CareAI is its <b>Decoupled Hybrid AI Paradigm</b>. "
        "In healthcare applications, relying entirely on black-box LLMs or single neural networks is dangerous: "
        "they exhibit hallucinations, unpredictable stochastic behavior, and non-deterministic safety enforcement. "
        "CareAI mitigates this by combining four distinct algorithmic tiers:"
    ))
    story.append(bullet("<b>1. Deterministic Mathematical Scoring:</b> Formula-based distress calculation guaranteeing 100% reproducible results."))
    story.append(bullet("<b>2. Strict Lexical Safety Tripwires:</b> Compiled regular expressions that immediately trigger crisis resources with zero LLM latency."))
    story.append(bullet("<b>3. Dense Vector RAG:</b> ChromaDB similarity search over clinical coping literature, grounding recommendations in empirical science."))
    story.append(bullet("<b>4. Supervised Deep Learning NLP:</b> A fine-tuned DistilBERT transformer providing non-diagnostic text classification across 4 mental health states."))

    story.append(h2("6.2 Deterministic Distress Normalization Formula"))
    story.append(p(
        "Daily check-in distress is evaluated across five fundamental psychological metrics, each scored on an integer scale from 1 to 10. "
        "To combine these disparate dimensions into a single continuous distress metric, positive indicators (Mood, Sleep Quality) "
        "are inverted, and negative indicators (Anxiety, Stress, Social Isolation) are aligned directly:"
    ))

    formula_box = """
Distress Score = [ ((10 - Mood) / 9) * 0.30 ] +
                 [ ((Anxiety - 1) / 9) * 0.25 ] +
                 [ ((Stress - 1) / 9) * 0.20 ] +
                 [ ((Isolation - 1) / 9) * 0.15 ] +
                 [ ((10 - Sleep) / 9) * 0.10 ]
"""
    story.append(code_box(formula_box))
    story.append(p(
        "<b>Component Weight Rationale:</b><br/>"
        "• <b>Mood (0.30 weight):</b> Inverted `(10 - Mood) / 9`. Valence is the single most predictive indicator of acute depressive state.<br/>"
        "• <b>Anxiety (0.25 weight):</b> Direct `(Anxiety - 1) / 9`. Measures somatic and cognitive hyper-arousal.<br/>"
        "• <b>Stress (0.20 weight):</b> Direct `(Stress - 1) / 9`. Measures acute external environmental pressure.<br/>"
        "• <b>Social Isolation (0.15 weight):</b> Direct `(Isolation - 1) / 9`. Captures perceived social disconnection, a primary risk factor in clinical depression.<br/>"
        "• <b>Sleep Quality (0.10 weight):</b> Inverted `(10 - Sleep) / 9`. Captures restorative physiological baseline."
    ))

    story.append(p("The resulting score strictly ranges from <b>0.00</b> (perfect wellbeing) to <b>1.00</b> (extreme crisis), partitioned into non-overlapping bands:"))
    cat_data = [
        ["Distress Level", "Score Range", "Clinical / Operational Meaning", "System Response"],
        ["Low", "0.00 - < 0.35", "Healthy equilibrium; mild transient fluctuations.", "Reinforce positive habits, general wellbeing tips."],
        ["Moderate", "0.35 - < 0.65", "Noticeable emotional or physiological tension.", "Suggest targeted relaxation (4-7-8 breathing, PMR)."],
        ["High", "0.65 - < 0.85", "Substantial distress; coping resources strained.", "Prioritize grounding exercises (5-4-3-2-1), offer counselor connect."],
        ["Critical", "≥ 0.85", "Acute psychological crisis; immediate risk.", "Trigger Safety Override, display 24/7 hotlines, log SafetyAlert."]
    ]
    story.append(create_table(["Level", "Score", "Clinical Meaning", "System Response"], cat_data, col_widths=[60, 65, 175, 204]))
    story.append(Spacer(1, 10))

    story.append(h2("6.3 Longitudinal Trend &amp; Linear Regression Slope Analysis"))
    story.append(p(
        "Single-day scores can be noisy due to acute daily events. In `agents/monitoring.py`, the system calculates rolling "
        "averages (7-day and 30-day) and applies <b>Ordinary Least Squares (OLS) Linear Regression</b> over historical check-in distress scores "
        "$y_i$ against chronological day index $x_i$:"
    ))
    slope_box = """
Slope (m) = [ Σ (x_i - x_mean) * (y_i - y_mean) ] / [ Σ (x_i - x_mean)^2 ]
"""
    story.append(code_box(slope_box))
    story.append(p(
        "<b>Trajectory Classification Rules:</b><br/>"
        "• <b>Improving:</b> `Slope < -0.01` (Distress is systematically decreasing over time).<br/>"
        "• <b>Stable:</b> `-0.01 <= Slope <= +0.01` (Distress is plateauing within normal variance).<br/>"
        "• <b>Worsening:</b> `Slope > +0.01` (Distress is systematically escalating; flags proactive clinical attention)."
    ))

    story.append(h2("6.4 Supervised Machine Learning Pipeline (`ml/`)"))
    story.append(p(
        "<b>Dataset Acquisition &amp; Exploration:</b><br/>"
        "The system incorporates the `nayma11/Mental-Health_Text-Classification_Dataset` downloaded via HuggingFace Datasets. "
        "The corpus contains <b>48,950 annotated text samples</b> reflecting real-world disclosures from online support forums. "
        "Raw labels are normalized into four mutually exclusive target classes:"
    ))
    dist_data = [
        ["Class Name", "Label ID", "Target Semantic Scope", "Training Samples (Split)"],
        ["Normal", "0", "Neutral, healthy everyday disclosures, non-distressed thoughts.", "18,153 samples (37.1%)"],
        ["Depression", "1", "Feelings of hopelessness, chronic sadness, lack of interest.", "14,273 samples (29.2%)"],
        ["Anxiety", "2", "Panic attacks, nervousness, anticipatory worry, somatic tension.", "5,327 samples (10.9%)"],
        ["Suicidal", "3", "Explicit or implicit suicidal ideation, desire to end life.", "11,197 samples (22.8%)"]
    ]
    story.append(create_table(["Class", "ID", "Semantic Scope", "Dataset Distribution"], dist_data, col_widths=[75, 30, 260, 139]))
    story.append(Spacer(1, 10))

    story.append(p(
        "<b>Preprocessing Pipeline (`ml/preprocess.py`):</b><br/>"
        "1. Text cleaning: lowercase normalization, URL removal, stripping of unprintable control characters.<br/>"
        "2. Stratified Data Split: 70% Train (34,265 samples), 15% Validation (7,342 samples), and 15% Test (7,343 samples).<br/>"
        "3. Exact balance preserved across splits using `sklearn.model_selection.train_test_split(stratify=y, random_state=42)`."
    ))

    story.append(h3("Model 1: Baseline TF-IDF + Logistic Regression (`ml/train_lr.py`)"))
    story.append(p(
        "• <b>Feature Extraction:</b> `TfidfVectorizer(max_features=10000, ngram_range=(1,2), sublinear_tf=True)`.<br/>"
        "• <b>Classifier:</b> `LogisticRegression(C=1.0, max_iter=1000, class_weight='balanced')`.<br/>"
        "• <b>Artifacts:</b> Saved as `models_saved/lr_model.pkl` (1.6 MB) and `models_saved/tfidf_vectorizer.pkl` (1.9 MB)."
    ))

    story.append(h3("Model 2: Fine-Tuned DistilBERT Transformer (`ml/train_bert.py`)"))
    story.append(p(
        "• <b>Base Model:</b> `distilbert-base-uncased` (6 transformer layers, 768 hidden dimensions, 12 attention heads, 66M parameters).<br/>"
        "• <b>Optimization:</b> AdamW optimizer with learning rate 2e-5, linear warmup over first 10% steps, and weight decay 0.01.<br/>"
        "• <b>Loss Function:</b> Weighted Cross-Entropy Loss to handle class imbalance (`compute_class_weight`).<br/>"
        "• <b>Training Infrastructure:</b> 3 epochs, batch size 16, mixed precision FP16 on NVIDIA RTX 3070 Ti Laptop GPU.<br/>"
        "• <b>Artifacts:</b> Saved weights in Safetensors format (`models_saved/distilbert/model.safetensors`, 267.8 MB)."
    ))

    story.append(h2("6.5 Quantitative Model Evaluation &amp; Comparison"))
    story.append(p("The following evaluation metrics were rigorously recorded on the identical held-out 7,343-sample test set:"))

    eval_data = [
        ["Evaluation Metric", "Logistic Regression + TF-IDF", "Fine-Tuned DistilBERT", "Performance Delta"],
        ["Overall Test Accuracy", "79.18%", "82.69%", "+3.51% (Superior)"],
        ["Macro F1-Score", "0.7741", "0.8163", "+0.0422 (Superior)"],
        ["Weighted F1-Score", "0.7890", "0.8266", "+0.0376 (Superior)"],
        ["Macro Precision", "0.7676", "0.8098", "+0.0422 (Superior)"],
        ["Macro Recall", "0.7859", "0.8260", "+0.0401 (Superior)"],
        ["Normal Class F1", "0.9098 (P: 0.888, R: 0.933)", "0.9360 (P: 0.940, R: 0.932)", "+0.0262"],
        ["Depression Class F1", "0.6986 (P: 0.763, R: 0.644)", "0.7459 (P: 0.791, R: 0.705)", "+0.0473"],
        ["Anxiety Class F1", "0.7707 (P: 0.714, R: 0.837)", "0.8351 (P: 0.794, R: 0.881)", "+0.0644"],
        ["Suicidal Class F1", "0.7172 (P: 0.706, R: 0.729)", "0.7483 (P: 0.714, R: 0.786)", "+0.0311"]
    ]
    story.append(create_table(["Metric", "Logistic Regression", "DistilBERT", "Delta"], eval_data, col_widths=[120, 140, 144, 100]))
    story.append(Spacer(1, 10))

    story.append(p(
        "<b>Model Selection Decision:</b> DistilBERT was selected as the active primary inference model due to its higher Macro F1 (0.8163 vs 0.7741) "
        "and notably higher recall on high-risk suicidal disclosures (78.6% vs 72.9%). If GPU or memory constraints prevent DistilBERT initialization, "
        "`ml/inference.py` gracefully degrades to the lightweight Logistic Regression model."
    ))

    story.append(h2("6.6 Vector RAG Knowledge Retrieval Pipeline"))
    story.append(p(
        "The RAG engine (`services/rag.py`) maintains an embedded ChromaDB collection (`mental_health_knowledge`). "
        "Curated knowledge documents covering Anxiety Management, Sleep Protocols, Stress Reduction, and Emotional Regulation "
        "are ingested, chunked into 500-token blocks, and converted into 384-dimensional dense vectors using `sentence-transformers/all-MiniLM-L6-v2`. "
        "When a user exhibits high anxiety or sleep issues, cosine similarity retrieves the top-2 matching coping steps to inject "
        "into the LLM prompt or UI recommendation cards."
    ))

    story.append(Spacer(1, 15))
    return story


def build_section_7():
    story = []
    story.append(h1("7. Database and Data Management Architecture"))

    story.append(h2("7.1 Dual-Tier Database Architecture"))
    story.append(p(
        "CareAI implements a production-grade <b>Dual-Tier Database Strategy</b> designed for zero-configuration local development "
        "and high-concurrency cloud deployments:<br/>"
        "• <b>Local Development Tier (Canonical SQLite):</b> Resolved to an absolute canonical filesystem path (`instance/mhapp.db`), "
        "guaranteeing persistence across local restarts without duplicate fragmented databases.<br/>"
        "• <b>Cloud Production Tier (Managed PostgreSQL):</b> Attached automatically via the `DATABASE_URL` environment variable "
        "on Render. Utilizes SQLAlchemy connection pooling (`pool_pre_ping=True`, `pool_recycle=300`, `pool_size=10`, `max_overflow=20`) "
        "to prevent broken pipe exceptions on managed cloud database sleep intervals.<br/>"
        "• <b>Enterprise MySQL Support:</b> Full PyMySQL connection string support when `MYSQL_*` environment variables are present."
    ))

    story.append(h2("7.2 Complete Database Schemas and Tables"))
    story.append(p("The CareAI schema comprises 7 relational tables defined via SQLAlchemy declarative models:"))

    schema_data = [
        ["Table Name", "Primary Key", "Foreign Keys", "Key Columns &amp; Constraints", "Purpose"],
        ["users", "id (INT)", "None", "username VARCHAR(64) UNIQUE, email VARCHAR(120) UNIQUE, password_hash VARCHAR(256), role VARCHAR(20), is_admin BOOL", "User accounts, authentication hashes, and counselor role flags."],
        ["checkins", "id (INT)", "user_id -> users.id", "mood INT, anxiety INT, stress INT, sleep_quality INT, isolation_score INT, distress_score FLOAT, distress_level VARCHAR(20), timestamp DATETIME", "Daily quantitative wellbeing metrics and normalized scores."],
        ["journals", "id (INT)", "user_id -> users.id", "content TEXT, created_at DATETIME", "Free-form reflective journal entries and personal reflections."],
        ["chat_messages", "id (INT)", "user_id -> users.id", "role VARCHAR(16), content TEXT, timestamp DATETIME, is_safety_override BOOL, distress_score FLOAT, sentiment VARCHAR(32), detected_emotions VARCHAR(255)", "Full multi-turn conversational history and distress signals."],
        ["alerts", "id (INT)", "user_id -> users.id, reviewed_by -> users.id", "message VARCHAR(255), risk_level VARCHAR(32), action_taken VARCHAR(255), status VARCHAR(32), counselor_notes TEXT, reviewed_at DATETIME", "High-acuity safety alerts flagged for human counselor review."],
        ["case_summaries", "id (INT)", "patient_id -> users.id, counselor_id -> users.id", "title VARCHAR(255), summary_json TEXT, created_at DATETIME, notes TEXT", "Synthesized longitudinal clinical case syntheses for counselors."],
        ["counselor_notifications", "id (INT)", "counselor_id -> users.id, alert_id -> alerts.id", "is_read BOOL, email_sent BOOL, created_at DATETIME", "In-app and email notification tracking dispatched to counselors."]
    ]
    story.append(create_table(["Table", "PK", "FKs", "Columns &amp; Constraints", "Purpose"], schema_data, col_widths=[65, 45, 95, 175, 124]))
    story.append(Spacer(1, 10))

    story.append(h2("7.3 Relational Entity-Relationship (ER) Architecture"))
    er_text = """
+--------------------+            +------------------------+
|       users        | 1        * |        checkins        |
|--------------------|<-----------|------------------------|
| id (PK)            |            | id (PK), user_id (FK)  |
| username, email    |            | mood, anxiety, stress  |
| password_hash      |            | distress_score, level  |
| role, is_admin     |            +------------------------+
+---------+----------+
          | 1        * +------------------------+
          +----------->|        journals        |
          |            |------------------------|
          |            | id (PK), user_id (FK)  |
          |            | content, created_at    |
          |            +------------------------+
          | 1        * +------------------------+
          +----------->|     chat_messages      |
          |            |------------------------|
          |            | id (PK), user_id (FK)  |
          |            | role, content, distress|
          |            +------------------------+
          | 1        * +------------------------+
          +----------->|         alerts         |
          |            |------------------------|
          |            | id (PK), user_id (FK)  |
          |            | risk_level, status     |
          |            +-----------+------------+
          | 1                      | 1
          |                        | *
          | 1        * +-----------v------------+
          +----------->| counselor_notifications|
                       |------------------------|
                       | id (PK), alert_id (FK) |
                       | counselor_id (FK)      |
                       +------------------------+
"""
    story.append(code_box(er_text))
    story.append(Spacer(1, 10))

    story.append(h2("7.4 Additive Schema Migration Engine (`services/db_migration.py`)"))
    story.append(p(
        "A critical engineering requirement was preventing database table drops or data loss during feature deployments. "
        "The custom additive migration engine inspects existing database column metadata at application boot using `sqlalchemy.inspect`: "
        "if new columns (such as `distress_score` in `chat_messages` or `role` in `users`) are absent, `ALTER TABLE ... ADD COLUMN` statements "
        "are safely executed within an idempotent transaction. Existing patient check-in records are never overwritten or deleted."
    ))

    story.append(Spacer(1, 15))
    return story


def build_section_8():
    story = []
    story.append(h1("8. API and Backend Specification"))

    story.append(h2("8.1 Complete RESTful API Catalog"))
    story.append(p("The backend exposes 18 HTTP endpoints grouped by functional domain:"))

    api_data = [
        ["Endpoint Route", "HTTP Method", "Auth Required", "Request Payload", "Response Summary"],
        ["/", "GET", "Public", "None", "Redirects to /dashboard if logged in; else /login."],
        ["/health", "GET", "Public", "None", "JSON status, DB ping diagnostic, model availability."],
        ["/login", "GET / POST", "Public", "username_or_email, password, remember", "HTML form on GET; session cookie and redirect on POST."],
        ["/register", "GET / POST", "Public", "username, email, password, confirm", "HTML form on GET; account creation on POST."],
        ["/logout", "GET", "Authenticated", "None", "Clears session cookie, redirects to /login."],
        ["/dashboard", "GET", "Authenticated", "None", "Renders dashboard view with KPI metrics &amp; chart scripts."],
        ["/checkin", "GET", "Authenticated", "None", "Renders interactive daily check-in slider view."],
        ["/api/checkin", "POST", "Authenticated", "JSON: {mood, anxiety, stress, sleep, isolation}", "Returns computed score, level, trajectory, coping exercises."],
        ["/api/checkin/history", "GET", "Authenticated", "Query: ?limit=30", "Returns historical time-series array of user check-ins."],
        ["/journal", "GET", "Authenticated", "None", "Renders reflective journal writing interface and history."],
        ["/api/journal", "POST", "Authenticated", "JSON: {content}", "Saves journal entry; returns sentiment &amp; validation feedback."],
        ["/chat", "GET", "Authenticated", "None", "Renders AI conversational companion interface."],
        ["/api/chat", "POST", "Authenticated", "JSON: {message}", "Returns assistant response, distress score, and safety flag."],
        ["/api/chat/history", "GET", "Authenticated", "Query: ?limit=50&amp;offset=0", "Returns paginated conversational history for current user."],
        ["/counselor/dashboard", "GET", "Counselor/Admin", "None", "Renders clinical caseload overview and unresolved alerts."],
        ["/counselor/patient/<id>", "GET", "Counselor/Admin", "None", "Detailed patient clinical dossier, history, and narrative."],
        ["/counselor/alert/<id>/status", "POST", "Counselor/Admin", "JSON: {status, notes}", "Updates alert state to in_review or resolved with clinical note."],
        ["/counselor/patient/<id>/case-summary", "POST", "Counselor/Admin", "JSON: {notes}", "Generates and saves structured 7-section clinical case summary."]
    ]
    story.append(create_table(["Route", "Method", "Auth", "Payload", "Response"], api_data, col_widths=[105, 50, 65, 120, 164]))
    story.append(Spacer(1, 10))

    story.append(h2("8.2 Detailed Endpoint Specifications &amp; JSON Schemas"))

    story.append(h3("1. Check-In Submission API (`POST /api/checkin`)"))
    checkin_schema = """
Request Payload:
{
  "mood": 3,
  "anxiety": 8,
  "stress": 7,
  "sleep_quality": 4,
  "isolation_score": 6
}

Response (200 OK):
{
  "distress_score": 0.63,
  "distress_level": "moderate",
  "distress_breakdown": {
    "mood_component": 0.233,
    "anxiety_component": 0.194,
    "stress_component": 0.133,
    "isolation_component": 0.083,
    "sleep_component": 0.067
  },
  "trend": "improving",
  "recommendations": [
    {
      "title": "5-4-3-2-1 Sensory Grounding",
      "category": "Anxiety Management",
      "steps": "Name 5 things you can see, 4 you can touch..."
    }
  ]
}
"""
    story.append(code_box(checkin_schema))
    story.append(Spacer(1, 6))

    story.append(h3("2. Conversational Chat API (`POST /api/chat`)"))
    chat_schema = """
Request Payload:
{
  "message": "I have three exams tomorrow and I can't concentrate at all."
}

Response (200 OK):
{
  "response": "Exam crunch can trigger intense sensory overload. Let's break this down into three manageable 45-minute study sprints...",
  "priority_override": false,
  "distress_score": 0.52,
  "detected_emotions": "Academic Stress, Overwhelm"
}
"""
    story.append(code_box(chat_schema))
    story.append(Spacer(1, 15))

    return story


def build_section_9():
    story = []
    story.append(h1("9. Frontend Architecture and User Experience"))

    story.append(h2("9.1 'Mira Life' Inspired Design System"))
    story.append(p(
        "CareAI features a proprietary modern dark-mode aesthetic inspired by the 'Mira Life' design language. "
        "Mental health applications must never feel clinical, sterile, or institutional; rather, they should project "
        "warmth, calm, and security. The UI uses deep obsidian backgrounds (`#0a0b0e`), warm amber highlights (`#ffbc7d`), "
        "electric sapphire accents (`#0b84fe`), and restorative emerald tones (`#10b981`)."
    ))

    design_tokens = [
        ["Token Variable", "Hex Code", "Role &amp; Application"],
        ["--bg-deep", "#0a0b0e", "Deep obsidian background canvas with radial blur gradients."],
        ["--card-surface", "#15161c", "Elevated card surface with subtle glassmorphic transparency."],
        ["--accent-warm", "#ffbc7d", "Mira warm amber/peach for primary branding and emotional comfort."],
        ["--accent-blue", "#0b84fe", "Electric sapphire for technical buttons, active states, and navigation."],
        ["--accent-emerald", "#10b981", "Restorative green for improving trajectories and low-risk badges."],
        ["--accent-rose", "#f43f5e", "Alert rose for critical distress warnings and emergency helplines."],
        ["--text-primary", "#ffffff", "Crisp white headers and primary scorecard metrics."],
        ["--text-muted", "#969ea6", "Subdued text for secondary descriptions and timestamps."]
    ]
    story.append(create_table(["CSS Variable", "Hex Value", "Design Token Purpose"], design_tokens, col_widths=[100, 75, 329]))
    story.append(Spacer(1, 10))

    story.append(h2("9.2 Page-by-Page Architectural Breakdown"))
    story.append(p(
        "<b>1. Dynamic Monitoring Dashboard (`templates/dashboard/dashboard.html`):</b><br/>"
        "• Features 5 real-time KPI scorecards displaying current Mood, Anxiety, Stress, Sleep, and Isolation.<br/>"
        "• Trajectory Indicator Badge displaying 'Improving' (Green), 'Stable' (Blue), or 'Worsening' (Amber/Red) calculated via OLS linear regression.<br/>"
        "• Multi-dataset Chart.js time-series graph plotting daily distress alongside rolling averages.<br/>"
        "• Dynamic coping cards displaying personalized RAG recommendations with step-by-step instructions.<br/>"
        "<b>2. Daily Check-In Interface (`templates/checkin/checkin.html`):</b><br/>"
        "• 5 customized range sliders with dual numeric/semantic labels (e.g. 1='Severe Depression', 10='Vibrant &amp; Joyful').<br/>"
        "• Live client-side JavaScript calculation previewing distress scores before submission.<br/>"
        "<b>3. Conversational Companion (`templates/chat/chat.html`):</b><br/>"
        "• Auto-scrolling chat window with typewriter-style text streaming and user/assistant message separation.<br/>"
        "• Quick-starter prompt pills ('Quick Breathing Exercise', 'Study Schedule Help', 'Why do I feel anxious?').<br/>"
        "• Emergency helpline overlay card rendered immediately if safety tripwires are breached.<br/>"
        "<b>4. Reflective Journal (`templates/journal/journal.html`):</b><br/>"
        "• Full-page rich reflection text area with character counter and advice opt-out banner.<br/>"
        "• Accordion history listing past reflections with AI-tagged sentiment badges ('Positive', 'Reflective', 'Distressed').<br/>"
        "<b>5. Counselor Clinical Portal (`templates/counselor/dashboard.html` &amp; `patient_detail.html`):</b><br/>"
        "• Triage table displaying all active unresolved safety alerts with severity badges and timestamps.<br/>"
        "• Patient caseload filter with instant search by username or risk level.<br/>"
        "• Detailed patient clinical dossier containing full longitudinal history, regression charts, and AI case summaries."
    ))

    story.append(Spacer(1, 15))
    return story


def build_section_10():
    story = []
    story.append(h1("10. Complete Project Folder Structure"))

    story.append(p(
        "The CareAI repository is organized into distinct functional directories ensuring strict separation of concerns:"
    ))

    tree_str = """
AI-Mental-Health-System/
├── app.py                     # Flask application factory, blueprint registration, /health route
├── config.py                  # Production/local configuration, DB pooling, cookie security
├── extensions.py              # Singleton extensions: SQLAlchemy, LoginManager, CSRFProtect
├── run.py                     # Local development entry point (debug=True, port=5000)
├── wsgi.py                    # Production WSGI entry point for Gunicorn
├── Procfile                   # Process file for cloud container deployment
├── render.yaml                # Infrastructure-as-Code Blueprint for Render web + PostgreSQL
├── Dockerfile                 # Containerized Docker build file (Python 3.12-slim)
├── requirements.txt           # Python dependency specification
├── seed_users.py              # CLI utility seeding default patient, counselor, and admin accounts
├── agents/                    # LangGraph Multi-Agent System directory
│   ├── __init__.py            # Module exports
│   ├── state.py               # Shared AgentState TypedDict definition
│   ├── graph.py               # LangGraph StateGraph builder, conditional edges, synthesis node
│   ├── orchestrator.py        # Orchestrator agent entry point and action dispatch wrapper
│   ├── distress.py            # Agent 3: Deterministic distress calculation & ML text classification
│   ├── monitoring.py          # Agent 4: 7d/30d rolling averages, OLS linear regression slope
│   ├── safety.py              # Agent 5: Dual-stage lexical regex and score-based crisis override
│   ├── recommendation.py      # Agent 6: ChromaDB RAG wellbeing exercise matcher
│   ├── conversation.py        # Agent 2: Empathetic multi-turn conversational companion (Ollama)
│   ├── journal_agent.py       # Reflective journal analysis agent with advice opt-out detection
│   ├── narrative_agent.py     # Crisis narrative synthesis agent for clinical counselors
│   └── case_summary_agent.py  # 7-section clinical case summary generator for counselors
├── ml/                        # Supervised Machine Learning Pipeline
│   ├── preprocess.py          # HuggingFace dataset download, text cleaning, stratified split
│   ├── train_lr.py            # TF-IDF vectorizer + Logistic Regression baseline training
│   ├── train_bert.py          # DistilBERT transformer fine-tuning with PyTorch & HuggingFace
│   ├── evaluate.py            # Confusion matrix generation and comparative evaluation
│   ├── inference.py           # Public classify_text() inference module with fallback
│   └── dataset_explore.py     # Dataset analysis and exploratory statistics generator
├── dataset/                   # Dataset storage directory
│   ├── raw/                   # Raw CSV datasets
│   └── processed/             # Cleaned datasets, column maps, and label mappings
├── models/                    # SQLAlchemy database models
│   ├── user.py                # User model with scrypt password hashing & counselor flags
│   ├── checkin.py             # Daily check-in model (mood, anxiety, stress, sleep, isolation)
│   ├── journal.py             # Reflective journal model
│   ├── chat.py                # Conversational chat message model with distress attributes
│   ├── alert.py               # Safety alert model for counselor review workflows
│   ├── case_note.py           # Clinical case summaries and counselor notes
│   └── notification.py        # Counselor alert dispatch notifications
├── routes/                    # Flask route controller Blueprints
│   ├── auth.py                # Registration, login, logout, Flask-Login user loader
│   ├── dashboard.py           # Monitoring dashboard view and time-series aggregation
│   ├── checkin.py             # Check-in submission and history endpoints
│   ├── chat.py                # AI conversational companion API and chat interface
│   ├── journal.py             # Reflective journal recording and sentiment API
│   └── counselor.py           # Clinical counselor portal and patient detail routes
├── services/                  # Business logic and external infrastructure services
│   ├── rag.py                 # ChromaDB vector index and Sentence-Transformers retrieval
│   ├── notifier.py            # In-app and SMTP email counselor alert dispatcher
│   ├── db_migration.py        # Safe additive schema migration utility
│   └── db.py                  # Standalone database engine session helper
├── knowledge_base/            # Curated evidence-based clinical wellbeing texts
│   ├── anxiety_management.txt # 5-4-3-2-1 grounding, parasympathetic breathing
│   ├── crisis_support.txt     # International crisis hotlines (988, 14416, 111)
│   ├── emotional_wellbeing.txt# Cognitive reframing, emotional regulation
│   ├── sleep_hygiene.txt      # Stimulus control, sleep restriction protocols
│   ├── social_connection.txt  # Overcoming isolation, interpersonal effectiveness
│   └── stress_management.txt  # Progressive muscle relaxation (PMR), somatic release
├── static/                    # Static assets
│   ├── css/style.css          # Mira Life dark-mode design system stylesheet
│   └── img/                   # Vector SVGs (logo, bot avatar, support icons) and banners
├── templates/                 # Jinja2 HTML templates
│   ├── base.html              # Core layout with glassmorphic navbar and footer
│   ├── auth/                  # login.html, register.html
│   ├── dashboard/             # dashboard.html
│   ├── checkin/               # checkin.html
│   ├── chat/                  # chat.html
│   ├── journal/               # journal.html
│   └── counselor/             # dashboard.html, patient_detail.html
└── tests/                     # Automated test suites
    ├── test_system.py         # Full end-to-end integration test suite
    ├── test_careai_20_scenarios.py # 20 scenario benchmark test runner
    ├── test_conversational_ai.py   # Multi-turn conversation and intent testing
    ├── test_counselor_and_narrative.py # Clinical portal and narrative verification
    ├── test_persistence_and_auth.py   # Database persistence and session tests
    └── test_targeted_bugfixes.py      # Regression test suite for specific edge cases
"""
    story.append(code_box(tree_str))
    story.append(Spacer(1, 15))

    return story
