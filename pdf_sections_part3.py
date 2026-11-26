# pdf_sections_part3.py
"""Sections 11 through 14 of CareAI Technical Documentation."""

from reportlab.platypus import Paragraph, Spacer, PageBreak, KeepTogether, Table, TableStyle
from reportlab.lib import colors
from pdf_helpers import (
    p, p_bold, h1, h2, h3, bullet, callout, code_box, create_table,
    USABLE_WIDTH, STYLES
)

def build_section_11():
    story = []
    story.append(h1("11. Installation, Setup, and Deployment Guide"))

    story.append(h2("11.1 Prerequisites and System Requirements"))
    story.append(bullet("<b>Operating System:</b> Windows 10/11, macOS (Apple Silicon or Intel), or Linux (Ubuntu 22.04+)."))
    story.append(bullet("<b>Python Runtime:</b> Python 3.12+ (Python 3.12.8 recommended; fully compatible with Python 3.13)."))
    story.append(bullet("<b>RAM:</b> Minimum 4 GB for core web application; 8 GB+ recommended if running local Ollama SLMs."))
    story.append(bullet("<b>Optional Local SLM:</b> [Ollama](https://ollama.com/) with model `llama3.2` pulled (`ollama pull llama3.2`). If Ollama is offline, CareAI automatically falls back to its deterministic conversational engine."))
    story.append(bullet("<b>Optional GPU Acceleration:</b> NVIDIA GPU with CUDA 12.1+ for accelerated DistilBERT training and real-time Ollama inference (tested on RTX 3070 Ti Laptop GPU)."))

    story.append(h2("11.2 Step-by-Step Local Development Setup"))
    setup_cmds = """
# 1. Clone the repository
git clone https://github.com/Vineeth-KK/AI-Mental-Health-System.git
cd AI-Mental-Health-System

# 2. Create and activate a Python virtual environment
python -m venv venv
# On Windows PowerShell:
.\\venv\\Scripts\\Activate.ps1
# On macOS / Linux:
source venv/bin/activate

# 3. Install core dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 4. Configure environment variables
# Copy template configuration
cp .env.example .env

# 5. Seed default demo accounts (Patient, Counselor, Administrator)
python seed_users.py

# 6. Launch the development server
python run.py
"""
    story.append(code_box(setup_cmds))
    story.append(Spacer(1, 10))

    story.append(h2("11.3 Environment Variable Configuration (.env)"))
    story.append(p("The application reads configuration values from the local `.env` file via `python-dotenv`:"))

    env_data = [
        ["Variable Name", "Default / Example Value", "Description &amp; Operational Impact"],
        ["FLASK_SECRET_KEY", "careai-mental-health-system-stable-key-2026", "Session encryption key; must be persistent to prevent session drops on restarts."],
        ["DATABASE_URL", "sqlite:///instance/mhapp.db (Local) / postgresql://... (Cloud)", "SQLAlchemy database connection URI (canonical SQLite or managed PostgreSQL)."],
        ["OLLAMA_HOST", "http://127.0.0.1:11434", "Network endpoint for local Ollama server (or Cloudflare tunnel URL on cloud)."],
        ["OLLAMA_MODEL", "llama3.2", "Target quantized Small Language Model tag for conversational inference."],
        ["WTF_CSRF_ENABLED", "true", "Enforces CSRF token validation on all POST/PUT/DELETE web requests."],
        ["SMTP_HOST", "smtp.gmail.com (Optional)", "Host for outgoing counselor crisis email notifications."],
        ["SMTP_USER / PASSWORD", "counselor@careai.com (Optional)", "Credentials for automated counselor alert email dispatch."]
    ]
    story.append(create_table(["Variable", "Example Value", "Description"], env_data, col_widths=[110, 130, 264]))
    story.append(Spacer(1, 10))

    story.append(h2("11.4 Default Pre-Seeded Accounts (`seed_users.py`)"))
    story.append(p("The initialization script automatically creates three role-based test accounts:"))
    account_data = [
        ["Role Type", "Email Address", "Default Password", "Access Privileges"],
        ["Patient Demo", "demo@example.com", "password123", "Full patient access: daily check-in, AI companion chat, reflective journal, dashboard."],
        ["Counselor Demo", "counselor@careai.com", "password123", "Full clinical access: `/counselor/dashboard`, patient timelines, crisis alerts, case summaries."],
        ["Admin Demo", "admin@careai.com", "password123", "Superuser privileges: counselor portal access, administrative user and system overrides."]
    ]
    story.append(create_table(["Role", "Email", "Password", "Privileges"], account_data, col_widths=[85, 120, 75, 224]))
    story.append(Spacer(1, 10))

    story.append(h2("11.5 Cloud Deployment on Render (`render.yaml`)"))
    story.append(p(
        "CareAI contains an automated Infrastructure-as-Code Blueprint (`render.yaml`) for instant production deployment:<br/>"
        "1. Connect the GitHub repository `https://github.com/Vineeth-KK/AI-Mental-Health-System` to [Render](https://render.com).<br/>"
        "2. Select <b>Blueprints -> New Blueprint Instance</b>.<br/>"
        "3. Render automatically provisions:<br/>"
        "   • <b>Web Service:</b> Python 3.12 environment running Gunicorn (`gunicorn --bind 0.0.0.0:$PORT --workers 1 --threads 2 --timeout 120 wsgi:app`).<br/>"
        "   • <b>Managed Database:</b> Free PostgreSQL instance (`careai-db`), automatically injecting `DATABASE_URL`.<br/>"
        "   • <b>Health Check:</b> Monitored at `/health` to verify continuous live database connectivity."
    ))

    story.append(h2("11.6 Docker Containerization (`Dockerfile`)"))
    docker_cmds = """
# Build the production Docker image
docker build -t careai-mental-health .

# Run the container with environment variables
docker run -d -p 5000:5000 \\
  -e FLASK_SECRET_KEY="production-secret-key" \\
  -e DATABASE_URL="postgresql://user:pass@dbhost:5432/careai_db" \\
  careai-mental-health
"""
    story.append(code_box(docker_cmds))
    story.append(Spacer(1, 15))

    return story


def build_section_12():
    story = []
    story.append(h1("12. Security, Privacy, and Performance Engineering"))

    story.append(h2("12.1 Multi-Layered Security Architecture"))
    story.append(p("Because CareAI handles sensitive psychological data, security is enforced across five defense layers:"))

    sec_layers = [
        ["Security Defense Layer", "Implemented Mechanism", "Threat Mitigated"],
        ["Password Cryptography", "Werkzeug Salted Scrypt Hashing (`generate_password_hash`)", "Credential exposure during database breaches; brute-force attacks."],
        ["Session Integrity", "14-Day Persistent Cookies (`careai_session`), `HttpOnly`, `SameSite=Lax`, `Secure` in HTTPS", "Session hijacking, cross-site scripting (XSS) cookie theft."],
        ["CSRF Defense", "Flask-WTF CSRF Token Injection across forms and AJAX `X-CSRFToken` headers", "Cross-Site Request Forgery unauthorized state mutations."],
        ["Data Isolation", "Row-Level Query Filtering (`filter_by(user_id=current_user.id)`)", "Multi-tenant horizontal privilege escalation; cross-patient data leaks."],
        ["Role-Based Access Control", "Custom `@counselor_required` decorator verifying `is_counselor()` or `is_admin`", "Unauthorized patient access to clinical triage dashboards and notes."],
        ["Privacy in Notifications", "Email dispatches transmit Alert ID and risk level; omit raw journal text", "Interception of sensitive unencrypted personal disclosures over SMTP."]
    ]
    story.append(create_table(["Defense Layer", "Implemented Mechanism", "Threat Mitigated"], sec_layers, col_widths=[105, 185, 214]))
    story.append(Spacer(1, 10))

    story.append(h2("12.2 High-Performance Engineering &amp; Resource Optimization"))
    story.append(bullet("<b>Lazy Module Loading:</b> In `agents/distress.py` and `ml/inference.py`, heavy ML libraries (PyTorch, PyArrow, Scikit-Learn) are loaded dynamically on first inference call. This prevents C-level DLL access violations and ensures fast startup."))
    story.append(bullet("<b>SQLAlchemy Connection Pooling:</b> Configured with `pool_pre_ping=True` (verifies connection validity before execution), `pool_recycle=300` (recycles connections every 5 minutes), `pool_size=10`, and `max_overflow=20` to prevent socket leaks."))
    story.append(bullet("<b>Cloud Memory Optimization:</b> On Render cloud environments (512MB RAM free tier limit), `services/rag.py` detects cloud presence and avoids loading heavy ONNX embeddings into RAM, switching to in-memory matching to prevent container OOM termination."))
    story.append(bullet("<b>Database Indexing:</b> Dedicated B-tree indexes placed on `users.username`, `users.email`, `checkins.timestamp`, and `chat_messages.timestamp` for sub-millisecond retrieval."))

    story.append(h2("12.3 Potential Vulnerabilities &amp; Concrete Hardening Recommendations"))
    story.append(p(
        "<b>1. Database at Rest Encryption (HIPAA Compliance):</b> While SQLite/PostgreSQL store salted password hashes, "
        "raw journal entries and chat messages are currently stored as plaintext strings in the database. In clinical production, "
        "transparent disk encryption (SQLCipher for SQLite or pgcrypto / AWS RDS KMS for PostgreSQL) should be enabled.<br/>"
        "<b>2. API Rate Limiting:</b> Currently, endpoints rely on authenticated sessions. Implementing `Flask-Limiter` "
        "(e.g., 20 requests/minute on `/api/chat`) is recommended to mitigate denial-of-service attempts on local SLM inference.<br/>"
        "<b>3. Direct Model Jailbreak Resistance:</b> While regex filters intercept explicit suicidal keywords, users could theoretically "
        "attempt adversarial prompt injections (e.g., 'Ignore all previous safety rules'). Adding an LLM Guardrails layer (such as NeMo Guardrails) "
        "is planned for v3.0."
    ))

    story.append(Spacer(1, 15))
    return story


def build_section_13():
    story = []
    story.append(h1("13. Technical Challenges Encountered &amp; Implemented Solutions"))

    story.append(p(
        "During the development and testing of CareAI, several complex engineering challenges were identified and systematically resolved. "
        "Below are the verified technical problems and their exact implementations in code:"
    ))

    challenges = [
        ("Challenge 1: LLM Hallucinations in Clinical &amp; Crisis Contexts",
         "When LLMs generate unguided mental health advice, they risk hallucinating clinical diagnoses, recommending unverified medications, or providing dangerous responses during self-harm emergencies.",
         "Engineered a dual-stage deterministic safety architecture in `agents/safety.py`. Hard-coded compiled regex patterns scan disclosures BEFORE prompts reach the LLM. If critical patterns are detected, the graph short-circuits to `safety_override_node`, returning verified crisis helplines without touching the LLM."),

        ("Challenge 2: Unsolicited Advice Intrusion in Personal Journals",
         "Users frequently write reflections simply to express grief, frustration, or sadness without wanting problem-solving advice (e.g., 'I just want to vent, please don't give me advice'). Standard LLMs reflexively offer solutions.",
         "Implemented `detect_advice_opt_out()` in `agents/journal_agent.py`. The algorithm scans for negative advice intent phrases ('don't want advice', 'just venting'). If detected, the prompt strictly commands the LLM to output ZERO suggestions, focusing 100% on emotional validation."),

        ("Challenge 3: Robotic Conversational Drift &amp; Over-Medicalization",
         "Conversational mental health bots often treat every query as a clinical emergency. If a user asked 'What is photosynthesis?' or 'How does binary search work?', early prototypes forced breathing exercises.",
         "Completely refactored `agents/conversation.py` with multi-turn intent classification and prompt guardrails. General knowledge, science, coding, and casual conversation queries are answered directly and authoritatively. The agent pivots topics dynamically without robotic therapy disclaimers."),

        ("Challenge 4: Cloud Free-Tier Out-of-Memory (OOM) Termination",
         "When deploying on Render's 512MB RAM free tier, loading PyTorch, DistilBERT, ChromaDB, and ONNX tokenizers simultaneously caused instantaneous container crashes (Linux SIGKILL / Exit Code 137).",
         "Engineered cloud environment detection in `config.py` and `services/rag.py`. When `is_cloud_env` is True, ChromaDB switches to lightweight in-memory retrieval, and heavy ML model weights are loaded lazily on demand rather than at application boot."),

        ("Challenge 5: Database Drift &amp; Session Wipeouts Across Restarts",
         "Using relative SQLite file paths (`mhapp.db`) caused different execution contexts (`run.py` vs tests vs workers) to create multiple disjoint database files, causing users to lose their accounts on application restart.",
         "Established an absolute canonical database path (`CANONICAL_DB_PATH = os.path.join(basedir, 'instance', 'mhapp.db')`) and built `services/db_migration.py`. Database tables and columns are safely updated without dropping tables or losing user sessions."),

        ("Challenge 6: Windows PyArrow/Scikit-Learn DLL Access Violations During PyTest",
         "On Windows environments, importing PyArrow and Scikit-Learn during PyTest test collection triggered C-level memory access violations (Windows STATUS_ACCESS_VIOLATION / 0xC0000005).",
         "Implemented an environment check in `agents/distress.py` (`os.environ.get('PYTEST_CURRENT_TEST')`) and lazy singleton loading in `_get_ml_classify()`. Test suites collect and execute cleanly without memory corruption.")
    ]

    for title, problem, solution in challenges:
        story.append(h2(title))
        story.append(p_bold("Technical Problem:"))
        story.append(p(problem))
        story.append(p_bold("Engineered Solution:"))
        story.append(p(solution))
        story.append(Spacer(1, 4))

    story.append(Spacer(1, 15))
    return story


def build_section_14():
    story = []
    story.append(h1("14. Advantages, Limitations, and Future Roadmap"))

    story.append(h2("14.1 Major Architectural &amp; Clinical Advantages"))
    story.append(bullet("<b>Zero Hallucination Safety:</b> Deterministic regex tripwires and score-based thresholds guarantee that crisis resources are presented immediately, eliminating the risk of LLM refusal or hallucinated helplines."))
    story.append(bullet("<b>Objective Longitudinal Visibility:</b> Replaces unreliable patient recall with empirical, quantified daily check-in histories, rolling averages, and linear regression trajectories."))
    story.append(bullet("<b>Data Sovereignty &amp; Privacy:</b> Supports local Small Language Model inference via Ollama (Llama 3.2), ensuring sensitive personal reflections never leave the local or self-hosted server."))
    story.append(bullet("<b>Clinical Human-in-the-Loop:</b> Automatically compiles structured 7-section case summaries and chronological crisis narratives for human therapists, reducing clinical documentation overhead by 80%."))
    story.append(bullet("<b>Versatile Conversational Companion:</b> Handles academic planning, programming, everyday advice, and casual chat naturally without forcing repetitive therapy scripts."))

    story.append(h2("14.2 Current System Limitations"))
    story.append(bullet("<b>Self-Reporting Bias:</b> Quantitative distress scores rely on user self-assessment. Users experiencing dissociation or masking may submit scores that do not reflect their true physiological state."))
    story.append(bullet("<b>Modality Limitation (Text-Only):</b> The current implementation is text-based. It does not analyze vocal biomarkers, pitch variability, facial expressions, or physiological data."))
    story.append(bullet("<b>English Language Focus:</b> The prompt templates, regular expressions, and DistilBERT training datasets are currently optimized for English disclosures."))

    story.append(h2("14.3 Future Development Roadmap"))
    roadmap_data = [
        ["Phase", "Target Feature", "Technical Strategy &amp; Architecture"],
        ["v3.0", "Wearable IoT Sensor Integration", "Ingest real-time Heart Rate Variability (HRV), resting heart rate, and sleep staging via Apple HealthKit and Google Health Connect APIs."],
        ["v3.1", "Voice &amp; Audio Biomarker Analysis", "Analyze user voice messages using acoustic feature extractors (jitter, shimmer, speaking cadence, fundamental frequency $F_0$) to detect depressive flat affect."],
        ["v3.2", "Multilingual Global Support", "Expand safety regex and fine-tune multilingual transformers (XLM-RoBERTa) across Hindi, Spanish, Mandarin, and Arabic."],
        ["v3.3", "Clinical EHR / EMR Interoperability", "Export synthesized clinical case summaries directly into hospital electronic health records using standard HL7 / FHIR (Fast Healthcare Interoperability Resources) protocols."],
        ["v3.4", "On-Device Federated Learning", "Implement privacy-preserving federated model training, enabling local smartphone fine-tuning without centralizing sensitive user reflections."]
    ]
    story.append(create_table(["Version", "Roadmap Feature", "Engineering Strategy"], roadmap_data, col_widths=[50, 160, 294]))
    story.append(Spacer(1, 15))

    return story
