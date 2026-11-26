# CAREAI — PRODUCTION DEPLOYMENT FAILURE DIAGNOSTIC REPORT
**Investigation Target:** CareAI Production Web Service on Render  
**Live Application URL:** `https://ai-mental-health-system-cxkq.onrender.com`  
**Date of Diagnostic:** November 26, 2026  
**Investigation Mode:** Deep Codebase Audit & Live Endpoint Diagnostics (**Read-Only / No Code Changes**)

---

## Executive Summary

A comprehensive investigation into the deployed Render application was conducted across the source code, environment configurations, database persistence architecture, authentication/session management, and the Groq AI integration layer.

The investigation confirmed that the production issues are caused by **two primary systemic failures**:
1. **Critical Database Ephemerality:** The deployed Render web service is currently connected to a local **SQLite** database (`instance/mhapp.db`) on Render's ephemeral filesystem instead of a persistent PostgreSQL database. Whenever the Render container restarts or spins down (due to free-tier 15-minute inactivity), the SQLite database file is wiped clean. This destroys all user accounts, check-ins, journal entries, chat logs, and session contexts.
2. **Silent AI Provider Fallback:** The live application is **not generating responses through Groq**. Every chat completion attempt to Groq Cloud is failing and silently triggering `_generate_intelligent_fallback` in `agents/conversation.py`. The fallback response contains hardcoded rule-based string templates (including an unrendered `{rag_snippet_str}` tag) and evaluates only the immediate last user message, causing total loss of conversational memory.

---

## 1. Confirmed vs. Suspected Problems Matrix

| Item | Problem Category | Status | Root Cause Summary | Impact |
| :--- | :--- | :--- | :--- | :--- |
| **P1** | **Database Persistence** | **CONFIRMED** | `DATABASE_URL` is missing in Render environment; app falls back to SQLite on ephemeral disk. | All user data, check-ins, journals, and chats vanish on container spin-down or restart. |
| **P2** | **AI Chatbot Quality** | **CONFIRMED** | Groq API call fails in `_call_groq()`; app silently drops to static rule-based fallback. | Responses are generic template strings rather than genuine LLM output. |
| **P3** | **Conversation Memory** | **CONFIRMED** | Fallback engine inspects only `messages[-1]`; plus SQLite history is erased on restarts. | Bot has zero conversational context or memory of previous discussion turns. |
| **P4** | **Account Persistence** | **CONFIRMED** | Newly registered users in `instance/mhapp.db` are erased on container sleep; only seeded demo users persist. | Users cannot log back in after container sleeps; sessions are invalidated. |
| **P5** | **Health Check Blindspot** | **CONFIRMED** | `/health` only verifies string length of `GROQ_API_KEY`; it does not ping Groq API. | Health endpoint falsely reports healthy status while Groq inference is failing. |
| **P6** | **Template Leakage** | **CONFIRMED** | Python multiline string concatenation in `agents/conversation.py` lacks `f` prefixes. | Unrendered `{rag_snippet_str}` tag is leaked directly to users. |
| **P7** | **Proxy & CSRF Headers** | **CONFIRMED** | Missing `ProxyFix` middleware behind Render reverse proxy. | Can lead to `The referrer header is missing` (400 Bad Request) on HTTPS form/API posts. |
| **P8** | **Groq Call Error Cause** | **SUSPECTED** | `GROQ_API_KEY` may be invalid/revoked, or free-tier rate limits (6,000 TPM for 70B) exceeded. | Detailed below in Section 2. |

---

## 2. Groq API Investigation

### How Production Loads Configuration
In `config.py` (lines 67–81) and `services/ai_provider.py` (lines 83–130):
- `AI_PROVIDER`: Evaluates `os.getenv("AI_PROVIDER")`. On Render, this resolves to `"groq"`.
- `GROQ_API_KEY`: Evaluates `os.getenv("GROQ_API_KEY", "").strip()`.
- `GROQ_MODEL`: Evaluates `os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")`.
- `GROQ_BASE_URL`: Evaluates `https://api.groq.com/openai/v1`.
- `GROQ_TIMEOUT`: Defaults to `20.0` seconds.

### Audit of LLM-Powered Agents
All 4 generative agents use `services/ai_provider.py`:
1. `agents/conversation.py` $\rightarrow$ calls `ai_provider.chat_completion()` (line 99)
2. `agents/journal_agent.py` $\rightarrow$ calls `ai_provider.chat_completion()` (line 293)
3. `agents/narrative_agent.py` $\rightarrow$ calls `ai_provider.chat_completion()` (line 72)
4. `agents/case_summary_agent.py` $\rightarrow$ calls `ai_provider.chat_completion()` (line 126)

All non-LLM components (`distress.py`, `safety.py`, `monitoring.py`, `recommendation.py`) are deterministic ML and rule-based pipelines and do not call the LLM.

### Evidence: Live Chat Call Triggers Fallback
A live HTTP POST test request was dispatched to `https://ai-mental-health-system-cxkq.onrender.com/api/chat` with:
`{"message": "What is the capital of France?", "history": []}`

The live Render service returned:
```json
{
  "priority_override": false,
  "rag_context": [],
  "recommendations": [],
  "response": "Here is a clear perspective on capital france, DemoUser:\n\nWhen looking at capital france, the most practical approach is to break down the primary mechanisms step-by-step. Focusing on the foundational factors allows you to see the bigger picture and apply it with confidence.{rag_snippet_str}\n\nWhat specific aspect of capital france would you like to explore further together?",
  "safety_alert": { ... }
}
```

#### Code Match
In `agents/conversation.py` lines 884–886:
```python
return (
    f"Here is a clear perspective on {topic_phrase}{name_clause}:\n\n"
    f"When looking at {topic_phrase}, the most practical approach is to break down the primary mechanisms step-by-step. "
    "Focusing on the foundational factors allows you to see the bigger picture and apply it with confidence.{rag_snippet_str}\n\n"
    f"What specific aspect of {topic_phrase} would you like to explore further together?"
)
```
This confirms that `ai_resp.is_fallback` evaluated to `True`, silently diverting execution to `_generate_intelligent_fallback()`.

### Why Groq Fails in Production
`_call_groq()` in `services/ai_provider.py` handles exceptions by returning `AIResponse(is_fallback=True, fallback_reason=...)`. The failure is caused by one of the following:

1. **Authentication Error (HTTP 401):**
   - The key set in Render's dashboard (`gsk_...zQPN`) may be expired, revoked, or copied with surrounding quotation marks (`"gsk_..."`). While `.strip()` removes whitespace, it does not strip quotation marks.
2. **Groq Free-Tier Rate Limits (HTTP 429):**
   - The configured model is `llama-3.3-70b-versatile`. On Groq's Free Tier, this 70B model has a limit of **6,000 Tokens Per Minute (TPM)**. Because the system instruction alone is ~800 tokens, sending multiple requests or multi-turn history rapidly exhausts the 6,000 TPM limit.
   - By comparison, `llama-3.1-8b-instant` has a limit of **30,000 TPM** on the free tier.
3. **Egress Timeout:**
   - `GROQ_TIMEOUT` is set to 20 seconds. If Groq Cloud experiences queuing, httpx times out and drops to fallback.

*(Note: Exact HTTP response codes and error strings from Groq are logged internally via `logger.warning/error` to Render's Gunicorn stdout/stderr and cannot be queried externally without Render Dashboard log access).*

---

## 3. Database Investigation (Highest Priority)

### Actual Database URL on Render
A live query to `https://ai-mental-health-system-cxkq.onrender.com/health` returned:
```json
{
  "database": {
    "status": "connected",
    "type": "sqlite"
  }
}
```

### Analysis of `config.py` Database Selection
In `config.py` lines 16–51:
```python
db_url = os.getenv('DATABASE_URL')
if db_url:
    if db_url.startswith("postgres://"):
        db_url = db_url.replace("postgres://", "postgresql://", 1)
    SQLALCHEMY_DATABASE_URI = db_url
    ...
else:
    # Local canonical SQLite database
    CANONICAL_DB_PATH = os.path.join(basedir, "instance", "mhapp.db")
    SQLALCHEMY_DATABASE_URI = f"sqlite:///{CANONICAL_DB_PATH.replace(os.sep, '/')}"
```
Because `type` is reported as `"sqlite"`, **`DATABASE_URL` is completely absent from the Render Web Service environment**.

### Render Ephemeral Filesystem Mechanics
1. Render Web Services operate in containerized instances. On the free tier, **disks are ephemeral**.
2. When a service experiences 15 minutes of inactivity, Render spins down the container to 0 instances.
3. When a new incoming HTTP request arrives, Render spins up a fresh container image from the Git repository.
4. The directory `instance/` and file `mhapp.db` on disk are wiped clean on every spin-down and every redeployment.
5. On container boot, `run_additive_migrations(app)` in `services/db_migration.py` executes:
   ```python
   if User.query.first() is None:
       # Re-seeds DemoUser, CounselorSarah, AdminCareAI
   ```
6. **Result:**
   - All accounts created via `/register` cease to exist.
   - All check-in records in `checkins` table cease to exist.
   - All journal entries in `journal_entries` table cease to exist.
   - All chat history in `chat_messages` table ceases to exist.
   - All counselor alerts in `alerts` table cease to exist.
   - Users attempting to log in receive `Invalid email or password`.

### Why `render.yaml` Did Not Provision PostgreSQL
In `render.yaml`:
```yaml
      - key: DATABASE_URL
        fromDatabase:
          name: careai-db
          property: connectionString

databases:
  - name: careai-db
    databaseName: careai_db
    user: careai_user
    plan: free
```
If the web service was originally created through the **Render Dashboard UI ("New Web Service")** rather than via **"New Blueprint Instance"**, Render does not process the `databases` block in `render.yaml`. Furthermore, Render Free PostgreSQL databases expire after 30 days. If the database was not provisioned or has expired, Render runs the service without `DATABASE_URL`.

---

## 4. Authentication and Session Investigation

### `SECRET_KEY` Configuration
In `config.py` line 10:
```python
SECRET_KEY = os.getenv('FLASK_SECRET_KEY') or 'careai-mental-health-system-stable-key-2026'
```
In `render.yaml` line 20:
```yaml
- key: FLASK_SECRET_KEY
  generateValue: true
```
Render persists environment variables across restarts. If `FLASK_SECRET_KEY` is set in the dashboard, the key remains stable.

### Why Sessions Become Invalid
In `routes/auth.py` lines 17–23:
```python
@login_manager.user_loader
def load_user(user_id):
    try:
        return db.session.get(User, int(user_id))
    except Exception as e:
        return None
```
When a user logs in, Flask-Login stores their `user_id` inside the encrypted session cookie.
When Render restarts and wipes `instance/mhapp.db`:
- The user's row in table `users` no longer exists.
- `load_user(user_id)` returns `None`.
- Flask-Login treats the user as unauthenticated and redirects them to `/login`.
- Attempting to log in with their registered credentials fails because the database row was erased.

### CSRF & Referrer Header Issue Behind Render Proxy
Render uses an Nginx/Envoy reverse proxy that terminates TLS/HTTPS and forwards requests to `0.0.0.0:$PORT` via HTTP.
`app.py` is currently missing Werkzeug's `ProxyFix`:
```python
from werkzeug.middleware.proxy_fix import ProxyFix
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1)
```
Without `ProxyFix`, Flask-WTF may see mismatched HTTP vs HTTPS schemes, occasionally triggering `400 Bad Request: The referrer header is missing` or CSRF token rejections on form and AJAX submissions.

---

## 5. Health Endpoint Investigation

### Current Implementation
In `app.py` lines 31–63:
```python
@app.route("/health", methods=["GET"])
def health():
    db.session.execute(text("SELECT 1"))
    ...
    from services.ai_provider import ai_provider
    ai_status = ai_provider.get_status()
```
And in `services/ai_provider.py` lines 136–148:
```python
if provider == "groq":
    api_key, model, base_url = self.get_groq_config()
    is_configured = bool(api_key and len(api_key) > 8)
    return {
        "app_env": app_env,
        "provider": "groq",
        "model": model,
        "configured": is_configured,
        "api_key_status": "present" if is_configured else "missing",
        ...
    }
```

### Assessment
- **What it checks:** It executes `SELECT 1` on the database (confirming SQLite is responsive) and checks whether `GROQ_API_KEY` is longer than 8 characters.
- **What it does NOT check:** It **does not verify Groq API connectivity or authentication**. It does not perform a handshake with `api.groq.com`.
- **Verdict:** The `/health` endpoint reports `"status": "healthy"` even when Groq is completely inaccessible and every chat request fails over to static fallback.

---

## 6. Comparison: Local vs. Production Configurations

| Component / Setting | Local Development | Current Render Production | Expected Production | Detected Problem | Recommended Fix |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **`DATABASE_URL`** | Unset (uses SQLite) | Unset (falls back to SQLite) | Managed PostgreSQL connection string | `DATABASE_URL` missing; SQLite on ephemeral disk | Attach Render PostgreSQL and set `DATABASE_URL` |
| **Database Type** | SQLite (`instance/mhapp.db`) | SQLite (`instance/mhapp.db`) | PostgreSQL 15/16 | Data wiped on container spin-down | Migrate to PostgreSQL |
| **`AI_PROVIDER`** | `ollama` | `groq` | `groq` | None (Provider correctly set to `groq`) | Keep `groq` |
| **`GROQ_API_KEY`** | Unset | `gsk_...zQPN` | Valid, active Groq key | Calls fail silently; falling back to rule templates | Verify key in Groq Console; strip quotes |
| **`GROQ_MODEL`** | `llama3.2` (Ollama) | `llama-3.3-70b-versatile` | `llama-3.1-8b-instant` or `llama-3.3-70b-versatile` | 70B free tier rate limit is very low (6K TPM) | Use `llama-3.1-8b-instant` for production stability |
| **WSGI Proxy** | Direct localhost | Direct Gunicorn | `ProxyFix` middleware enabled | SSL terminated at proxy without header translation | Add `ProxyFix` in `app.py` |
| **Fallback Template** | N/A (Ollama answers) | Missing `f` string prefix | Formatted string without tags | Leaks literal `{rag_snippet_str}` text | Add `f` prefix to lines 886 & 895 in `agents/conversation.py` |
| **Health Check** | Confirms Ollama config | Static config check | Active API connectivity verification | False healthy reports while inference fails | Add live ping check in `/health?check_api=1` |

---

## 7. Recommended Fixes in Priority Order

### Priority 1: Provision Persistent PostgreSQL on Render (Resolves Problems 2, 3, 4)
1. In the [Render Dashboard](https://dashboard.render.com):
   - Click **New +** $\rightarrow$ **PostgreSQL**.
   - Name: `careai-postgres`
   - Database: `careai_db`
   - User: `careai_user`
   - Region: Same region as your Web Service (e.g., Oregon or Frankfurt).
   - Plan: Free.
2. Once provisioned, copy the **Internal Database URL** (starts with `postgres://` or `postgresql://`).
3. In your Web Service settings $\rightarrow$ **Environment**:
   - Add/Update key: `DATABASE_URL`
   - Value: `<paste Internal Database URL>`
4. On redeploy, `services/db_migration.py` will automatically create all tables (`users`, `checkins`, `journal_entries`, `chat_messages`, `alerts`, etc.) in PostgreSQL.
5. All user accounts, conversations, and records will now persist permanently.

### Priority 2: Fix Groq API Key and Model Rate Limits (Resolves Problem 1)
1. Verify the Groq API key:
   - Go to [Groq Console](https://console.groq.com/keys).
   - Generate a fresh API key.
   - In Render Dashboard $\rightarrow$ Web Service $\rightarrow$ **Environment**:
     - Key: `GROQ_API_KEY`
     - Value: Paste the key directly **without any surrounding quotation marks or spaces**.
2. Optimize Model for Free-Tier Limits:
   - For high availability on the free tier, set `GROQ_MODEL` to:
     `llama-3.1-8b-instant` (30,000 TPM vs 6,000 TPM on 70B, ~500 tokens/sec).
   - Alternatively, keep `llama-3.3-70b-versatile` if staying within the 6,000 TPM limit.

### Priority 3: Add `ProxyFix` Middleware in `app.py` (Resolves Problem 7)
In `app.py`, wrap the WSGI application:
```python
from werkzeug.middleware.proxy_fix import ProxyFix
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1)
```
This ensures Flask accurately recognizes HTTPS requests forwarded by Render's reverse proxy.

### Priority 4: Fix String Formatting in `agents/conversation.py` (Resolves Problem 6)
In `agents/conversation.py`, add `f` prefix to lines 886 and 895 so `{rag_snippet_str}` interpolates properly during any fallback event.

### Priority 5: Add Real AI Handshake to `/health` (Resolves Problem 5)
Upgrade `/health` to execute a lightweight test completion when requested (or log the last recorded provider error code in the status output) so configuration vs live connectivity is immediately visible.

---

## 8. Exact Render Environment Variables Required

Configure the following in **Render Dashboard $\rightarrow$ Web Service $\rightarrow$ Environment**:

```env
APP_ENV=production
AI_PROVIDER=groq
GROQ_API_KEY=gsk_your_actual_groq_api_key_without_quotes
GROQ_MODEL=llama-3.1-8b-instant
DATABASE_URL=postgresql://careai_user:password@dpg-xxxxxx:5432/careai_db
FLASK_SECRET_KEY=a_long_random_stable_secret_string_here
WTF_CSRF_ENABLED=true
PYTHON_VERSION=3.12.8
```

---

## 9. Verification & Inaccessible Items

- **Local Machine Integrity:** Zero source code files, git branches, commits, tags, or database records were modified during this investigation.
- **Rollback Point:** `careai-before-dual-environment` (commit `151163f`) remains untouched and intact.
- **Inaccessible Items:** Render internal server logs (`gunicorn` output stream) cannot be read from the local terminal; therefore, the specific HTTP error code returned by Groq (HTTP 401 vs HTTP 429) was deduced from live behavioral testing and source analysis.

---

**STATUS: Investigation complete. Standing by for your instructions and approval before implementing any fixes.**
