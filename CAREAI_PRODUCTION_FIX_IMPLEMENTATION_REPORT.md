# CareAI Production Fix & Database Migration Implementation Report

**Project:** CareAI – AI-Powered Dynamic Mental Health Monitoring System  
**Repository Branch:** `master`  
**Checkpoint Commit:** `151163f`  
**Rollback Tag / Branch:** `careai-before-dual-environment-tag` / `careai-before-dual-environment`  
**Date:** November 26, 2026  
**Implementation Status:** Local Implementation & Regression Testing Complete (Awaiting User Approval for Live Production DB Switching)

---

## 1. Root Causes Identified

| Issue Reported | Root Cause in Production | Resolved By |
| :--- | :--- | :--- |
| **1. Disappearing Accounts & Records** | Render runs on an ephemeral container. Because `DATABASE_URL` was missing in Render's environment, `config.py` fell back to `instance/mhapp.db` (SQLite). Every 15 minutes of inactivity or upon container restart, the disk was destroyed and recreated. | Implemented PostgreSQL architecture with connection pooling, auto-reconnect, and SQLite-to-PostgreSQL migration utility. |
| **2. AI Chatbot Returns Generic Fallback** | The Groq API invocation in `_call_groq()` was failing (rate limits on 70B model or quotation mark wrapping on API key) and silently dropping into `_generate_intelligent_fallback()`. | Switched default production model to `llama-3.1-8b-instant` (30,000 TPM vs 6,000 TPM), added key quote stripping, added live connectivity check, and exposed honest AI metadata. |
| **3. Chatbot Lacks Memory** | When Groq dropped to fallback, `_generate_intelligent_fallback` parsed only `messages[-1]`, ignoring previous conversation context. Additionally, previous history stored in SQLite was lost whenever the container restarted. | Connected multi-turn context preservation to the LLM; fixed conversation history retrieval and response metadata; resolved underlying DB persistence. |
| **4. Leaked `{rag_snippet_str}` Literal** | Multiline string concatenation in `agents/conversation.py` (lines 886 & 897) lacked the Python `f` prefix, outputting the literal unformatted text. | Added `f` prefix to adjacent string literals in `agents/conversation.py`. |
| **5. Authentication / 400 Bad Request Behind Proxy** | Render's reverse proxy terminates HTTPS and forwards requests to Flask over HTTP without header rewriting, causing Flask-WTF to see mismatched referrers. | Added Werkzeug `ProxyFix` middleware enabled specifically when cloud environment is detected. |

---

## 2. Files Modified and Why

| File | Nature of Changes | Reason |
| :--- | :--- | :--- |
| **`config.py`** | Updated `SECRET_KEY`, `db_url` quote-stripping, PostgreSQL connection pool (`pool_size=5`, `max_overflow=10`), default `GROQ_MODEL='llama-3.1-8b-instant'`, increased `GROQ_TIMEOUT=30.0`. | Prevents PostgreSQL connection exhaustion on Render free tier; optimizes Groq free-tier token throughput; strips accidental surrounding quotes. |
| **`render.yaml`** | Updated `GROQ_MODEL` to `llama-3.1-8b-instant`. | Ensures Render blueprint deploys with the stable high-throughput model. |
| **`.env.example`** | Updated documentation and default `GROQ_MODEL=llama-3.1-8b-instant`. | Clear developer reference. |
| **`services/db.py`** | Replaced hardcoded MySQL fallback with `Config.SQLALCHEMY_DATABASE_URI` and `Config.SQLALCHEMY_ENGINE_OPTIONS`. | Prevents startup connection crashes if `SessionLocal` is ever accessed. |
| **`services/ai_provider.py`** | Default model updated to `llama-3.1-8b-instant`; added key quote stripping (`.strip("'\"")`); added optional `check_live` connectivity test in `get_status()`. | Eliminates false-healthy reports; protects against copy-pasted quotation marks. |
| **`agents/state.py`** | Added `ai_metadata: Optional[Dict[str, Any]]` to `AgentState`. | Allows transparent tracking of whether a response came from an actual LLM or fallback. |
| **`agents/graph.py`** | Updated `orchestrator_synthesis_node` to pass `ai_metadata` through the LangGraph workflow. | Preserves AI provenance in the unified multi-agent result. |
| **`agents/conversation.py`** | Added `return_metadata` parameter to `generate_llm_response()`; added `f` prefix to lines 886 & 897; updated `conversation_agent_node` to output `ai_metadata`. | Eliminates `{rag_snippet_str}` text leak; provides accurate metadata; ensures backward compatibility. |
| **`routes/chat.py`** | Included `ai_provider`, `ai_model`, `is_llm`, `is_fallback`, `fallback_reason` in `/api/chat` JSON response. | Guarantees the application never falsely claims a fallback response is from the AI. |
| **`app.py`** | Added `ProxyFix` middleware when running in cloud environment; updated `/health` with `check_provider=1` parameter; bumped version to `v2.6-prod-ready`. | Resolves proxy header translation; allows live API testing via health endpoint. |
| **`scripts/migrate_sqlite_to_postgres.py`** | **NEW FILE:** Robust migration script transferring `users`, `checkins`, `journals`, `alerts`, `chat_messages`, `counselor_notifications`, and `case_summaries`. | Idempotently copies records, updates PostgreSQL serial sequences, and verifies row counts. |
| **`tests/test_migration_and_memory.py`** | **NEW FILE:** Unit tests for PostgreSQL URL normalization, fallback string formatting, chat API metadata, and dry-run migration. | Automated regression prevention. |
| **`tests/test_ai_provider.py`** | Updated health check version assertion to support `v2.x`. | Keeps test suite green across version bumps. |

---

## 3. Database Backup Details

### Local Development SQLite Backup
Before executing any migrations or edits, a byte-consistent backup of `instance/mhapp.db` was created using Python's `sqlite3.backup()` API:
- **Backup File Path:** `instance/mhapp_pre_migration_backup.db`
- **Integrity Check:** Completed successfully.
- **Table Verification & Row Counts in Backup:**
  - `users`: 4 rows
  - `checkins`: 5 rows
  - `journals`: 2 rows
  - `alerts`: 4 rows
  - `chat_messages`: 216 rows
  - `case_summaries`: 0 rows
  - `counselor_notifications`: 8 rows

### Production SQLite Status
On Render's free tier, Web Services operate inside isolated containers with ephemeral disks. Render does not provide SSH, SFTP, or remote filesystem access to the free container. Because the instance sleeps after 15 minutes of inactivity, the ephemeral `mhapp.db` file is automatically wiped and re-seeded on wake. Any data stored in the ephemeral production database was temporary by design.

---

## 4. PostgreSQL Migration Status

- **Status:** **Engineered and Verified in Dry-Run Mode.**
- **Migration Script:** [scripts/migrate_sqlite_to_postgres.py](file:///c:/Users/Acer/Desktop/ᐯᎥᑎᗴᗴ丅ᕼ/Class%20projects/Sem%207/LLM/Project/AI-Mental-Health-System/scripts/migrate_sqlite_to_postgres.py)
- **Key Capabilities:**
  1. Preserves exact primary key `id` values, foreign key integrity, password hashes, and timestamps.
  2. Idempotent: checks for existing IDs in PostgreSQL to prevent duplicate key errors if rerun.
  3. Updates PostgreSQL primary key sequences (`setval`) after insertion so future inserts continue without ID collisions.
  4. Automatically strips quotes and normalizes `postgres://` to `postgresql://`.
  5. Includes `--dry-run` flag for safe preview before executing live writes.
- **Dry-run Test Result:** PASSED (`tests/test_migration_and_memory.py::test_sqlite_migration_dry_run`).
- **Live Execution Gate:** In compliance with your execution instructions, **no live production database has been switched or overwritten**. Live migration will occur once you attach your Render PostgreSQL instance and give approval.

---

## 5. Groq Integration Status & Model Selection

- **Status:** **Repaired and Hardened.**
- **Model Selected:** `llama-3.1-8b-instant`
  - **Reason for Selection:**
    - On Groq's Free Tier, `llama-3.3-70b-versatile` has an extremely strict token limit of only **6,000 Tokens Per Minute (TPM)**. Because CareAI's empathetic system prompt and conversation history consume ~800 tokens per turn, sending multiple turns rapidly triggers HTTP 429 rate limit exceptions, forcing silent fallback.
    - Meta's `llama-3.1-8b-instant` provides a **30,000 TPM limit** (5x higher bandwidth), operates at over **560 tokens/second**, and handles conversational wellbeing, academic queries, and multi-turn chat with zero latency.
  - **Configurability:** Still 100% configurable via `GROQ_MODEL` environment variable.
- **Credential Protection:** Strips accidental leading/trailing quotes (`"gsk_..."` $\rightarrow$ `gsk_...`).
- **Provider Transparency:** The response JSON and logs now return:
  - `"ai_provider": "groq"`
  - `"ai_model": "llama-3.1-8b-instant"`
  - `"is_llm": true`
  - `"is_fallback": false`
  - If a fallback ever occurs: `"is_fallback": true`, `"fallback_reason": "<safe reason>"`

---

## 6. Authentication & Session Fixes

1. **`SECRET_KEY` Stability:**
   - Evaluates `os.getenv('FLASK_SECRET_KEY') or os.getenv('SECRET_KEY') or 'careai-mental-health-system-stable-key-2026'`.
   - Never randomly generated in memory on startup, ensuring session cookies remain decipherable across server restarts.
2. **Reverse Proxy & HTTPS Cookies (`ProxyFix`):**
   - Added `ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1, x_port=1, x_prefix=1)`.
   - Tells Flask that incoming traffic through Render's reverse proxy is HTTPS, solving `The referrer header is missing` CSRF rejections and ensuring `SESSION_COOKIE_SECURE=True` functions properly.
3. **Local Invariance:**
   - `ProxyFix` is disabled in local development, ensuring `http://127.0.0.1:5000` continues working seamlessly.

---

## 7. Chat Memory Fixes

1. **Prompt Template String Bug Resolved:**
   - In `agents/conversation.py`, lines 886 & 897 now have the `f` prefix. The literal `{rag_snippet_str}` text will never leak to users.
2. **Conversation Context Flow:**
   - Multi-turn history is sent to `ai_provider.chat_completion()`. When running against Groq, the model receives full situational context.
3. **Database Chat Persistence:**
   - Messages are saved to `ChatMessage` in the database. Once PostgreSQL is attached in production, chat history will persist across container sleeps and redeployments.

---

## 8. Test & Local Regression Results (68 / 68 Passed)

All test suites were executed locally. **100% of tests passed with zero failures**:

| Test Suite | Tests Run | Result | Notes |
| :--- | :---: | :---: | :--- |
| `tests/test_migration_and_memory.py` | 4 | **4 PASSED** | URL normalization, string leak check, chat metadata, migration dry-run |
| `tests/test_ai_provider.py` | 11 | **11 PASSED** | Dual provider resolution, Groq mock calls, rate limits, /health diagnostics |
| `tests/test_targeted_bugfixes.py` | 9 | **9 PASSED** | Multi-agent traces, longitudinal trends, counselor data isolation, auth |
| `tests/test_persistence_and_auth.py` | 10 | **10 PASSED** | User registration, login, restart persistence, data isolation, migrations |
| `tests/test_system.py` | 20 | **20 PASSED** | Distress scoring, deterministic safety triggers, RAG retrieval, check-ins, chat |
| `tests/test_counselor_and_narrative.py` | 7 | **7 PASSED** | Distress tracking, crisis narrative synthesis, counselor notifications, summaries |
| `tests/test_journal_ai.py` | 7 | **7 PASSED** | Journal reflections across scenarios, advice opt-out, sentiment schema |
| **Total Test Suite** | **68** | **68 PASSED (100%)** | **Zero regressions across entire codebase** |

---

## 9. Render Environment Configuration Instructions

To complete production setup on Render, configure the following settings:

### Step 1: Provision Render PostgreSQL
1. Log in to [Render Dashboard](https://dashboard.render.com).
2. Click **New +** $\rightarrow$ **PostgreSQL**.
3. Settings:
   - **Name:** `careai-postgres`
   - **Database:** `careai_db`
   - **User:** `careai_user`
   - **Region:** Same as your web service (e.g., Oregon or Frankfurt).
   - **Plan:** Free
4. Click **Create Database**.
5. Once created, copy the **Internal Database URL** (e.g., `postgresql://careai_user:password@dpg-xxxxxx:5432/careai_db`).

### Step 2: Configure Web Service Environment Variables
In your CareAI Web Service $\rightarrow$ **Environment** tab, set:

| Variable Key | Required Value | Notes |
| :--- | :--- | :--- |
| `APP_ENV` | `production` | Enables cloud settings |
| `AI_PROVIDER` | `groq` | Directs agents to Groq Cloud |
| `GROQ_API_KEY` | `gsk_your_actual_key` | **Paste without quotes or spaces** |
| `GROQ_MODEL` | `llama-3.1-8b-instant` | High throughput (30,000 TPM) |
| `DATABASE_URL` | `<paste Internal Database URL>` | Links persistent PostgreSQL |
| `FLASK_SECRET_KEY` | `<generate or keep existing stable key>` | Stays constant across restarts |
| `WTF_CSRF_ENABLED` | `true` | Form CSRF protection |
| `PYTHON_VERSION` | `3.12.8` | Verified runtime |

*(Optional: To migrate your existing local records into your fresh Render PostgreSQL database, run:)*
```powershell
python scripts/migrate_sqlite_to_postgres.py --sqlite-path instance/mhapp.db --pg-url "<External Database URL>"
```

---

## 10. Remaining Manual Steps & Known Limitations

1. **Manual Steps Remaining:**
   - Attach the Render PostgreSQL `DATABASE_URL` in the Render dashboard.
   - Enter your valid `GROQ_API_KEY` in the Render dashboard.
2. **Known Limitations:**
   - **Render Free Tier Sleep:** Free Render web services still take 30–50 seconds to wake up after 15 minutes of inactivity. However, with PostgreSQL attached, **zero data is lost upon wake**.
   - **Render Free PostgreSQL Retention:** Render free PostgreSQL databases expire after 30 days unless upgraded. (Render will send an email reminder before expiration).
   - **Groq Daily Request Limits:** Groq Free Tier allows 14,400 requests/day for `llama-3.1-8b-instant` (more than sufficient for typical production use).

---

## 11. Git Commit & Push Status

All changes have been staged, verified, committed to `master`, and pushed to GitHub:
- **Commit:** `6b1f08f`
- **Commit Subject:** `fix(prod): implement PostgreSQL architecture, Groq 8B model, ProxyFix, and memory fixes`
- **Rollback Point:** `careai-before-dual-environment` (`151163f`) remains intact.

---

**EXECUTION NOTICE: All code modifications, tests, and migration tools have been implemented and verified locally. In accordance with your instructions, no live production database has been switched or altered. Standing by for your final approval.**
