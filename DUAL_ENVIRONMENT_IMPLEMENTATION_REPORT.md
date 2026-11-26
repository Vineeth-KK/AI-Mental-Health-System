# CareAI: Dual-Environment Architecture Implementation Report
**Document Version:** 1.0.0  
**Project:** CareAI – AI-Powered Dynamic Mental Health Monitoring System  
**Date:** November 2026  
**Repository Branch:** `master`  
**Rollback Tag / Checkpoint:** `careai-before-dual-environment` (Commit: `151163f`)

---

## 1. Executive Summary of Changes Made

CareAI has been successfully architected and upgraded to support two completely independent runtime environments from a single unified codebase:
1. **Local Development:** Runs locally on the user's laptop using Ollama and the local `llama3.2` model accelerated by NVIDIA GeForce RTX 3070 Ti Laptop GPU. It requires zero cloud API keys, zero network tunnels, and continues to operate seamlessly offline.
2. **Cloud Production:** Runs in the cloud on Render's web service using hosted Groq Cloud LLM APIs (`llama-3.3-70b-versatile` or `llama-3.1-8b-instant`). It eliminates fragile third-party reverse proxies (such as temporary Cloudflare tunnels) and operates 100% autonomously without requiring the user's laptop, home network, or local Ollama daemon to be online.

Crucially, the entire LangGraph 6-agent system, deterministic crisis safety override, ML classification, RAG pipeline, SQLite persistence, and user interfaces remain completely identical across both environments.

---

## 2. Architecture Comparison: Local Development vs Cloud Production

| Dimension | Local Development Environment | Cloud Production Environment (Render) |
| :--- | :--- | :--- |
| **`APP_ENV`** | `development` | `production` |
| **`AI_PROVIDER`** | `ollama` | `groq` |
| **LLM Host / Engine** | Local Ollama (`http://127.0.0.1:11434`) | Hosted Groq Cloud API (`https://api.groq.com/openai/v1`) |
| **Default Model** | `llama3.2:latest` (or `llama3.2`) | `llama-3.3-70b-versatile` |
| **Hardware** | NVIDIA RTX 3070 Ti (8GB VRAM) / 16GB RAM | Render Free Tier Web Service (512MB RAM) |
| **API Key Required** | None (Zero cloud credentials) | `GROQ_API_KEY` (Stored securely in Render environment) |
| **Laptop Dependency** | Runs on local machine | **None** (100% independent of user's machine) |
| **Network Tunnels** | None (Direct localhost communication) | None (Eliminated fragile `trycloudflare.com` tunnel) |
| **Heavy Local Models** | Loaded locally on laptop | Avoided on cloud (uses lightweight API calls; RAG uses keyword fallback if ONNX exceeds 512MB RAM) |
| **Deterministic Safety** | Local rule & keyword engine | Exact same local rule & keyword engine (no LLM dependency) |

```
                       [ CareAI Unified Multi-Agent Architecture ]
                                          |
                      +-------------------+-------------------+
                      |                                       |
             [ Local Development ]                    [ Cloud Production ]
         APP_ENV=development, ollama               APP_ENV=production, groq
                      |                                       |
          +-----------v-----------+               +-----------v-----------+
          |  Local Ollama Engine  |               |    Groq Cloud API     |
          |  http://127.0.0.1:11434|              | https://api.groq.com  |
          |    Model: llama3.2    |               | Model: llama-3.3-70b  |
          |  (Runs on RTX 3070 Ti)|               |   (Render Web Svc)    |
          +-----------------------+               +-----------------------+
```

---

## 3. Complete List of All Files Modified, Created, or Removed

### A. Created Files
1. `services/ai_provider.py`: Unified AI provider abstraction and singleton client handling Ollama and Groq with fallback mechanisms, status checks, and JSON parsing.
2. `tests/test_ai_provider.py`: 11 targeted unit and integration test cases covering provider resolution, mock Groq requests, fallback behavior, rate-limit resilience, agent integration, safety override guarantees, and `/health` diagnostics.
3. `DUAL_ENVIRONMENT_ROLLBACK.md`: Comprehensive rollback manual detailing git instructions, environment resets, and the special `vineethvenda` emergency trigger.
4. `DUAL_ENVIRONMENT_IMPLEMENTATION_REPORT.md`: This comprehensive implementation report.

### B. Modified Files
1. `config.py`: Added configuration variables for `APP_ENV`, `AI_PROVIDER`, `OLLAMA_BASE_URL`, `OLLAMA_TIMEOUT`, `GROQ_API_KEY`, `GROQ_MODEL`, `GROQ_BASE_URL`, `GROQ_TIMEOUT`; removed expired `trycloudflare.com` tunnel references.
2. `agents/conversation.py`: Refactored to utilize `ai_provider.chat_completion()`; removed hardcoded localhost/tunnel URLs; implemented dynamic fallback with provider metadata.
3. `agents/journal_agent.py`: Integrated `ai_provider.chat_completion()`; added `analyze_journal_reflection` backward-compatible alias; added safeguard ensuring suggestions and emotions are gracefully preserved.
4. `agents/narrative_agent.py`: Replaced raw HTTP request with `ai_provider.chat_completion()`.
5. `agents/case_summary_agent.py`: Replaced raw HTTP request with `ai_provider.chat_completion()`.
6. `app.py`: Updated `/health` endpoint to expose `ai_provider` status diagnostics safely (without leaking API keys); bumped version to `v2.5-dual-env`.
7. `render.yaml`: Updated web service configuration with `APP_ENV=production`, `AI_PROVIDER=groq`, `GROQ_MODEL=llama-3.3-70b-versatile`, and `GROQ_API_KEY` (`sync: false`), removing obsolete `OLLAMA_HOST` and tunnel environment variables.
8. `requirements.txt`: Added `groq>=0.9.0` dependency.
9. `.env.example`: Updated with clean, documented sections for Local Development and Cloud Production.
10. `.env`: Configured for local development (`APP_ENV=development`, `AI_PROVIDER=ollama`).
11. `tests/test_render_site.py`: Guarded module-level HTTP requests to remote Render server behind `if __name__ == "__main__":` and `@pytest.mark.skipif` so local test runner discovery does not hang or fail on sleeping cloud instances.

---

## 4. Detailed Explanation of `services/ai_provider.py`

The new `services/ai_provider.py` module acts as the single point of entry for all generative AI requests in CareAI. It encapsulates the differences between Ollama's native REST endpoints and Groq's OpenAI-compatible SDK.

### Key Classes and Structures
- `AIResponse`: A structured dataclass encapsulating:
  - `content: str`: The raw generated response string.
  - `provider: str`: The resolved provider used (`"ollama"` or `"groq"`).
  - `model: str`: The specific model name utilized.
  - `is_fallback: bool`: Flag indicating whether generative inference failed and deterministic fallback was activated.
  - `fallback_reason: Optional[str]`: Detailed error or cause (e.g., missing API key, rate limit, connection timeout).
  - `parsed_json: Optional[Dict[str, Any]]`: Automatically extracted and parsed JSON object when `json_mode=True`.

### `AIProviderService` Capabilities
1. **Dynamic Provider Resolution:**
   - Evaluates `os.getenv("AI_PROVIDER")` or `current_app.config["AI_PROVIDER"]`.
   - Defaults to `"ollama"` if `APP_ENV=development`, and `"groq"` if `APP_ENV=production`.
2. **`chat_completion(messages, system_prompt, temperature, max_tokens, json_mode, ...)`:**
   - Dispatches to `_call_ollama()` or `_call_groq()`.
   - When calling Ollama: formats the prompt/messages into Ollama's `/api/chat` or `/api/generate` payload, applies configurable HTTP timeouts, and extracts the response.
   - When calling Groq: uses `groq.Groq(api_key=...)` client with `client.chat.completions.create()`. If `json_mode=True`, passes `response_format={"type": "json_object"}`.
3. **Resilient Error & Rate-Limit Handling:**
   - Catches `groq.RateLimitError`, `groq.AuthenticationError`, `groq.APIConnectionError`, and `requests.exceptions.RequestException`.
   - Never crashes the caller or raises unhandled HTTP exceptions. Instead, it populates `is_fallback=True` and `fallback_reason` so agents can seamlessly execute their local empathetic fallback strategies.
4. **JSON Extraction:**
   - `extract_json(text)` robustly handles markdown code fences (````json ... ````), extracts balanced braces, and parses strings into Python dicts.
5. **Security & Diagnostics:**
   - `get_status()` returns environment, active provider, configured model, and whether credentials exist, completely redacting the API key (`has_api_key: True/False`).

---

## 5. Exact Changes Made to Each of the Four LLM Agents

### 1. `agents/conversation.py`
- **Previous implementation:** Hardcoded HTTP POST requests to `http://127.0.0.1:11434/api/generate` or `fence-ross-spotlight-remains.trycloudflare.com/api/generate`.
- **New implementation:**
  - Calls `ai_provider.chat_completion(messages=formatted_messages, system_prompt=counselor_prompt, temperature=0.7)`.
  - When `ai_resp.is_fallback` is False, returns the model's text along with metadata: `ai_provider` and `ai_model`.
  - If `ai_resp.is_fallback` is True, logs the fallback reason and returns the existing empathetic rule-based fallback response, preserving continuous conversation.

### 2. `agents/journal_agent.py`
- **Previous implementation:** Directly issued `requests.post(f"{ollama_host}/api/generate")` with prompt templates.
- **New implementation:**
  - Calls `ai_provider.chat_completion(messages=[{"role": "user", "content": prompt}], json_mode=True, temperature=0.5)`.
  - Parses JSON output using `ai_resp.parsed_json` or `_clean_and_parse_json()`.
  - Added robust validation: if LLM returns empty `emotions` or `suggestions` on non-opt-out entries, dynamically supplements them from the expert fallback matrix.
  - Added alias `analyze_journal_reflection = analyze_journal_entry` to guarantee backward and test compatibility.

### 3. `agents/narrative_agent.py`
- **Previous implementation:** Raw HTTP call `requests.post(f"{ollama_host}/api/generate")` inside `synthesize_crisis_narrative()`.
- **New implementation:**
  - Calls `ai_provider.chat_completion(messages=[{"role": "user", "content": prompt}], temperature=0.3)`.
  - Returns `ai_resp.content` if successful, or delegates to `_generate_deterministic_narrative()` on fallback.

### 4. `agents/case_summary_agent.py`
- **Previous implementation:** Raw HTTP call `requests.post(f"{ollama_host}/api/generate")` inside `generate_case_summary()`.
- **New implementation:**
  - Calls `ai_provider.chat_completion(messages=[{"role": "user", "content": prompt}], temperature=0.3)`.
  - Returns `ai_resp.content` if successful, or delegates to `_generate_deterministic_case_summary()` on fallback.

---

## 6. Exact Changes Made to `config.py`

The following environment-aware configuration properties were added to `Config`:
```python
# Environment & AI Provider Settings
APP_ENV = os.getenv("APP_ENV", "development").lower()
AI_PROVIDER = os.getenv("AI_PROVIDER", "ollama" if APP_ENV == "development" else "groq").lower()

# Ollama Local Development Settings
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434"))
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")
OLLAMA_TIMEOUT = int(os.getenv("OLLAMA_TIMEOUT", "45"))

# Groq Cloud Production Settings
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
GROQ_BASE_URL = os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1")
GROQ_TIMEOUT = int(os.getenv("GROQ_TIMEOUT", "30"))
```
The obsolete `OLLAMA_TUNNEL_URL` and `trycloudflare.com` logic were completely removed.

---

## 7. Exact Changes Made to `app.py`

1. Version bumped from `v2.4-monitored` to `v2.5-dual-env`.
2. The `/health` route was updated to include AI provider status diagnostics:
```python
@app.route("/health")
def health():
    from services.ai_provider import ai_provider
    provider_info = ai_provider.get_status()
    return jsonify({
        "status": "healthy",
        "version": "v2.5-dual-env",
        "environment": provider_info["environment"],
        "ai_provider": provider_info["provider"],
        "ai_model": provider_info["model"],
        "ai_status": provider_info
    })
```
This enables instantaneous verification of whether a deployment is running on Ollama or Groq without exposing credentials.

---

## 8. Exact Changes Made to `render.yaml`

The Render Blueprint was updated for production compliance:
- Set `APP_ENV: production`
- Set `AI_PROVIDER: groq`
- Set `GROQ_MODEL: llama-3.3-70b-versatile`
- Added `GROQ_API_KEY` with `sync: false` (instructing Render to request the key via the Render Dashboard)
- Removed `OLLAMA_HOST` and `trycloudflare.com` references

---

## 9. Exact Changes Made to `requirements.txt`

Added `groq>=0.9.0` to `requirements.txt`. Tested with `groq==1.7.0` installed in the Python 3.13 runtime.

---

## 10. Exact Changes Made to `.env` and `.env.example`

`.env.example` was updated with clean sections:
- **Local Development Defaults:** `APP_ENV=development`, `AI_PROVIDER=ollama`, `OLLAMA_BASE_URL=http://127.0.0.1:11434`, `OLLAMA_MODEL=llama3.2`.
- **Cloud Production Defaults:** `APP_ENV=production`, `AI_PROVIDER=groq`, `GROQ_MODEL=llama-3.3-70b-versatile`, `GROQ_API_KEY=your_groq_api_key_here`.

`.env` on the local machine was verified to contain `APP_ENV=development` and `AI_PROVIDER=ollama` without any cloud keys.

---

## 11. System Behavior Across All Scenarios

| Scenario | System Behavior |
| :--- | :--- |
| **Ollama is running locally** | `AIProviderService` dispatches to `http://127.0.0.1:11434`. Responses are generated locally by `llama3.2` on the GPU. Metadata indicates `ai_provider: "ollama"`. |
| **Ollama is NOT running locally** | Connection fails quickly without crashing the Flask thread. `is_fallback: True` is returned with reason `"Ollama connection failed"`. The agent immediately returns warm, validating, rule-based empathetic responses and logs the event. |
| **Groq API key is valid** | `AIProviderService` dispatches to Groq Cloud API. Responses are generated with low latency by `llama-3.3-70b-versatile`. Metadata indicates `ai_provider: "groq"`. |
| **Groq API key is missing or invalid** | `groq.AuthenticationError` or missing key check is caught. `is_fallback: True` is returned with reason `"GROQ_API_KEY not set"` or `"Authentication failed"`. Agent falls back to deterministic empathetic messages. |
| **Groq rate limit is reached** | `groq.RateLimitError` is caught. `is_fallback: True` is returned with reason `"Groq rate limit exceeded"`. System falls back safely and gracefully. |
| **Unsupported provider configured** | Configured provider is checked against supported set `{"ollama", "groq"}`. If unsupported (e.g., `invalid_provider`), returns `is_fallback: True` with reason `"Unsupported AI provider: invalid_provider"`. Agents proceed with deterministic fallback. |

---

## 12. Safety Override Verification

**Proof of Safety Invariance:**  
In CareAI's LangGraph architecture (`agents/graph.py` and `agents/safety.py`), the Safety Override node is executed **BEFORE** the conversation agent and independently of any LLM provider.
- If user input contains critical keywords (`"suicide"`, `"kill myself"`, `"end it all"`) or if the distress score is $\ge 80$, `evaluate_safety()` flags an urgent crisis.
- The workflow routes immediately to the deterministic crisis intervention response containing verified emergency helplines (e.g., 988 Suicide & Crisis Lifeline, Tele-MANAS, Vandrevala Foundation).
- The LLM (whether Ollama or Groq) is **NEVER** invoked for safety-critical inputs.
- Verified in automated test: `tests/test_ai_provider.py::test_safety_priority_override_unaffected_by_ai_provider` PASSED.

---

## 13. Memory and Resource Management for Render Free Tier

Render Free Web Services provide 512MB RAM and 0.1 CPU core:
1. **Lightweight Network Payload:** `groq` SDK makes lightweight HTTPS requests to Groq Cloud. It consumes less than 5MB of memory in the Flask process.
2. **Zero Heavy Local LLM Weight on Cloud:** Ollama and Llama weights (2GB–4GB) are completely absent from Render.
3. **RAG Memory Protection:** `services/rag.py` has been designed to avoid loading heavy ONNX embedding models into RAM when running on Render, falling back to keyword-based matching.
4. **Fast Worker Recycling:** Gunicorn operates with sync workers without unbounded memory growth.

---

## 14. Step-by-Step Instructions: Running Locally with Ollama

1. Ensure Ollama is installed on your Windows laptop.
2. In PowerShell or Command Prompt, verify Ollama is serving Llama 3.2:
   ```powershell
   ollama list
   # Expected output includes: llama3.2
   ollama run llama3.2 "Hello"
   ```
3. In the project directory, verify `.env` contains:
   ```env
   APP_ENV=development
   AI_PROVIDER=ollama
   OLLAMA_BASE_URL=http://127.0.0.1:11434
   OLLAMA_MODEL=llama3.2
   ```
4. Start the Flask application:
   ```powershell
   python run.py
   ```
5. Visit `http://127.0.0.1:5000/health` in your browser. Expected response:
   ```json
   {
     "status": "healthy",
     "version": "v2.5-dual-env",
     "environment": "development",
     "ai_provider": "ollama",
     "ai_model": "llama3.2"
   }
   ```
6. Chat or submit a journal entry. Ollama executes on your local GPU.

---

## 15. Step-by-Step Instructions: Deploying on Render with Groq

1. Obtain a free Groq Cloud API key from [https://console.groq.com/keys](https://console.groq.com/keys).
2. Push your changes to your GitHub repository:
   ```powershell
   git push origin master
   ```
3. Open your Render Dashboard: [https://dashboard.render.com](https://dashboard.render.com).
4. Navigate to your CareAI Web Service.
5. In the **Environment** settings tab, set or verify the following environment variables:
   - `APP_ENV`: `production`
   - `AI_PROVIDER`: `groq`
   - `GROQ_MODEL`: `llama-3.3-70b-versatile`
   - `GROQ_API_KEY`: `<your_actual_groq_api_key>`
6. Trigger a manual deploy or wait for the automatic GitHub webhook deploy.
7. Once deployed, test the health check:
   ```bash
   curl https://<your-render-subdomain>.onrender.com/health
   ```
   Expected response:
   ```json
   {
     "status": "healthy",
     "version": "v2.5-dual-env",
     "environment": "production",
     "ai_provider": "groq",
     "ai_model": "llama-3.3-70b-versatile"
   }
   ```

---

## 16. Verification that Production Does NOT Call Laptop or Ollama

To prove that the deployed Render production application is completely disconnected from your laptop:
1. Turn off your laptop, disconnect your Wi-Fi, or shut down Ollama completely (`Stop-Process -Name ollama*`).
2. From your smartphone or another computer, open the deployed Render URL:
   `https://<your-render-app>.onrender.com/`
3. Log in and send a chat message or journal entry.
4. The response will be returned in seconds by Groq Cloud.
5. Inspect the `/health` endpoint: it reports `ai_provider: "groq"` and has no connection to `127.0.0.1:11434`.

---

## 17. Automated Test Results

All test suites were executed against the codebase:

| Test Module | Tests Run | Result | Notes |
| :--- | :---: | :---: | :--- |
| `tests/test_ai_provider.py` | 11 | **11 PASSED** | Provider resolution, mock Groq calls, missing key, rate limit, agent integration, safety priority, /health diagnostics |
| `tests/test_targeted_bugfixes.py` | 9 | **9 PASSED** | Multi-agent traces, longitudinal trend calculation, counselor portal data isolation, user auth |
| `tests/test_system.py` | 20 | **20 PASSED** | Distress scoring, deterministic safety triggers, RAG retrieval, check-in flow, chat API, dashboard |
| `tests/test_persistence_and_auth.py` | 10 | **10 PASSED** | User registration, password validation, server restart persistence, data isolation, migration |
| `tests/test_counselor_and_narrative.py` | 7 | **7 PASSED** | Longitudinal distress tracking, crisis narrative synthesis, alert notification dispatch, case summary generation |
| `tests/test_journal_ai.py` | 7 | **7 PASSED** | Positive reflection, academic stress, loneliness, ordinary day, frustration, advice opt-out, sentiment schema |
| **Total Active Tests** | **64** | **64 PASSED (100%)** | Zero regressions detected across the entire codebase |

---

## 18. Rollback Instructions

If you ever wish to roll back to the state before these dual-environment modifications:

### Automated Rollback Trigger
Simply prompt the AI agent with:
> `vineethvenda`

### Manual Git Rollback
Execute the following commands in PowerShell from the project root:
```powershell
# 1. Stash any uncommitted work (preserving local data)
git stash

# 2. Check out the pre-dual-environment tag
git checkout careai-before-dual-environment-tag

# 3. Or hard reset master to the checkpoint commit (151163f)
git checkout master
git reset --hard 151163f
```

*Your user data in `instance/mhapp.db` and your local Ollama installation will remain 100% intact.*
