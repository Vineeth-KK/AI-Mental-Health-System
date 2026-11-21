# AI-Powered Dynamic Mental Health Monitoring System (CareAI)

CareAI is an intelligent, multi-agent mental health monitoring and supportive guidance web platform. It integrates **Flask**, **LangGraph**, **Ollama**, **ChromaDB RAG**, and **Chart.js** to track longitudinal mental wellbeing indicators, analyze structured daily check-ins, assess emergency psychological safety, and provide evidence-based coping exercises.

> **Disclaimer**: *CareAI is an academic research and self-monitoring prototype designed for educational and emotional support. It does not provide clinical diagnosis, psychiatric evaluations, or medical prescriptions. In emergencies, users are immediately directed to qualified human healthcare professionals and 24/7 crisis helplines (988 in US/Canada, 14416 in India, 111 in UK).*

---

## Architecture Overview

```
                      +-----------------------------+
                      |   User / Browser Frontend   |
                      |  (Bootstrap 5 Dark + Icons) |
                      +--------------+--------------+
                                     |
                                     v
                      +-----------------------------+
                      |   Flask Application Factory |
                      |    (CSRF, Session, Auth)    |
                      +--------------+--------------+
                                     |
                                     v
+-------------------------------------------------------------------------+
|                  LangGraph Coordinated Multi-Agent System               |
|                                                                         |
|  +--------------------+         +------------------------------------+  |
|  | Agent 1:           | ------> | Agent 3: Distress Analysis Agent   |  |
|  | Orchestrator Agent |         | - Deterministic Formula (0.0–1.0)  |  |
|  +---------+----------+         | - Low / Mod / High / Critical      |  |
|            |                    +-----------------+------------------+  |
|            |                                      |                     |
|            v                                      v                     |
|  +--------------------+         +------------------------------------+  |
|  | Agent 5:           | <-------+ Agent 4: Monitoring Agent          |  |
|  | Safety Agent       |         | - Rolling Averages (7d, 30d)       |  |
|  | - Lexical Triggers |         | - Regression Slope & Trajectory    |  |
|  | - Score Check      |         +-----------------+------------------+  |
|  | - Emergency Override         +------------------------------------+  |
|  +---------+----------+ ------> | Agent 6: Recommendation Agent      |  |
|            |                    | - ChromaDB Vector Store RAG        |  |
|            v                    | - Grounded Wellbeing Exercises     |  |
|  +--------------------+         +-----------------+------------------+  |
|  | Agent 2:           | <-------------------------+                     |
|  | Conversation Agent |                                                 |
|  | - Empathetic Ollama|                                                 |
|  | - Contextual Guard |                                                 |
|  +--------------------+                                                 |
+-------------------------------------------------------------------------+
```

---

## The Six Coordinated Agents

1. **Agent 1: Orchestrator Agent (`agents/orchestrator.py`)**
   - Main entry point for the LangGraph workflow.
   - Initializes shared `AgentState`.
   - Coordinates agent transitions and conditionally routes tasks.
   - Enforces safety-priority routing when urgent risks are flagged.

2. **Agent 2: Conversation Agent (`agents/conversation.py`)**
   - Provides empathetic, active-listening conversations.
   - Configurable for local Ollama models (`llama3.2`, `mistral`, `phi3`) with automatic intelligent fallback when Ollama is offline or in cloud deployments.
   - Strictly non-diagnostic and non-prescriptive.

3. **Agent 3: Distress Analysis Agent (`agents/distress.py`)**
   - Evaluates 5 core structured metrics on a 1–10 scale:
     - Mood (1–10)
     - Anxiety (1–10)
     - Stress (1–10)
     - Sleep Quality (1–10)
     - Social Isolation (1–10)
   - Computes deterministic normalized distress score between `0.00` and `1.00`:
     $$\text{Score} = \left(\frac{10 - \text{mood}}{9}\right) \times 0.30 + \left(\frac{\text{anxiety} - 1}{9}\right) \times 0.25 + \left(\frac{\text{stress} - 1}{9}\right) \times 0.20 + \left(\frac{\text{isolation} - 1}{9}\right) \times 0.15 + \left(\frac{10 - \text{sleep}}{9}\right) \times 0.10$$
   - Non-overlapping classifications:
     - **Low**: `< 0.35`
     - **Moderate**: `0.35` to `< 0.65`
     - **High**: `0.65` to `< 0.85`
     - **Critical**: `≥ 0.85`

4. **Agent 4: Monitoring Agent (`agents/monitoring.py`)**
   - Retrieves historical check-ins from the database isolated by user ID.
   - Computes 7-day and 30-day rolling averages across each metric.
   - Calculates linear regression slope over distress trajectory to identify **Improving**, **Stable**, or **Worsening** patterns.

5. **Agent 5: Safety Agent (`agents/safety.py`)**
   - Dual-stage safety verification:
     1. Deterministic regex & lexical keyword evaluation for self-harm / suicidal thoughts.
     2. Score-based trigger for Critical composite distress ($\ge 0.85$).
   - If triggered, sets `priority_override = True`, overrides ordinary chat, logs a database `SafetyAlert`, and immediately displays 24/7 crisis helplines:
     - **US & Canada**: 988 Suicide & Crisis Lifeline
     - **India**: 14416 (Tele-MANAS) / +91 9999 666 555 (Vandrevala Foundation)
     - **UK**: 111 / 116 123 (Samaritans)

6. **Agent 6: Recommendation Agent (`agents/recommendation.py`)**
   - Queries the ChromaDB RAG vector index using Sentence-Transformers embeddings.
   - Recommends evidence-based wellbeing practices (5-4-3-2-1 Sensory Grounding, 4-7-8 Breathing, Progressive Muscle Relaxation, Digital Sunset Sleep Protocol).

---

## Dynamic Monitoring & RAG Pipeline

- **Knowledge Base (`knowledge_base/`)**: Curated educational materials covering stress reduction, sleep hygiene, anxiety de-escalation, emotional regulation, social connection, and crisis guidance.
- **Vector Database**: Persistent ChromaDB collection embedded using `all-MiniLM-L6-v2`.
- **Dynamic Charts**: Chart.js displays longitudinal trends for all 5 metrics alongside the composite distress trajectory.

---

## Local Development & Setup

### Prerequisites
- Python 3.12+ (or 3.13)
- (Optional) [Ollama](https://ollama.com/) for local LLM inference

### Installation
```bash
git clone https://github.com/Vineeth-KK/AI-Mental-Health-System.git
cd AI-Mental-Health-System

pip install -r requirements.txt
```

### Running the Application
```bash
python run.py
```
Open **`http://127.0.0.1:5000`** in your browser.

### Running Automated Tests
```bash
pytest tests/ -v
```
All 20 test suites cover authentication, user data isolation, deterministic formulas, LangGraph coordination, safety override, RAG retrieval, and dashboard APIs.

---

## Cloud Deployment

### 1. Render Deployment
The repository includes `render.yaml` and `Procfile`.
1. Sign in to [Render](https://render.com).
2. Create **New Web Service** and select this repository: `https://github.com/Vineeth-KK/AI-Mental-Health-System`.
3. Render automatically reads `render.yaml` and deploys using Gunicorn on port `5000`.

### 2. Docker Deployment
A production `Dockerfile` is included:
```bash
docker build -t careai-mental-health .
docker run -p 5000:5000 careai-mental-health
```

---

## Security & Privacy
- **Strict User Isolation**: Queries enforce `user_id = current_user.id`.
- **Credential Protection**: `.env` and SQLite instances are strictly `.gitignore`d.
- **CSRF Protection**: Flask-WTF protects all forms and API endpoints.
- **Password Security**: Werkzeug password hashing with salted bcrypt.
