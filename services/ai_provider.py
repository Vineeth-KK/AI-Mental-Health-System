# services/ai_provider.py
"""
Centralized AI Provider Service for CareAI.
Decouples LLM-powered agents from specific inference backends, supporting:
1. Local Development: Ollama + Llama 3.2 on local workstation (zero API keys required).
2. Cloud Production: Hosted Groq LLM API (e.g. llama-3.3-70b-versatile) on Render.
3. Intelligent Fallback: Guaranteed deterministic responses if external providers are unavailable.

Safety Notice: Deterministic safety evaluations, distress scoring, monitoring regression,
ML text classification, RAG retrieval, and clinical emergency overrides run independently
and are NEVER bypassed or compromised by provider status.
"""

import os
import re
import json
import time
import logging
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple
import httpx

logger = logging.getLogger(__name__)

# Default model definitions
DEFAULT_OLLAMA_MODEL = "llama3.2"
DEFAULT_OLLAMA_BASE_URL = "http://127.0.0.1:11434"
DEFAULT_GROQ_MODEL = "llama-3.3-70b-versatile"
DEFAULT_GROQ_BASE_URL = "https://api.groq.com/openai/v1"


@dataclass
class AIResponse:
    """Standardized response from the AI provider layer."""
    content: str
    raw_content: str
    provider: str                      # "ollama" | "groq" | "fallback"
    model: str                         # e.g. "llama3.2", "llama-3.3-70b-versatile"
    is_fallback: bool = False          # False = genuine LLM inference; True = static/fallback engine
    fallback_reason: Optional[str] = None
    latency_ms: float = 0.0
    parsed_json: Optional[Dict[str, Any]] = None


def extract_json_from_text(text: str) -> Optional[Dict[str, Any]]:
    """Robustly extracts JSON from an LLM response string."""
    if not text:
        return None
    cleaned = text.strip()

    # 1. Direct JSON parse
    try:
        return json.loads(cleaned)
    except Exception:
        pass

    # 2. Extract from markdown code fences: ```json ... ```
    m = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", cleaned)
    if m:
        try:
            return json.loads(m.group(1).strip())
        except Exception:
            pass

    # 3. Extract between outer curly braces
    first_brace = cleaned.find("{")
    last_brace = cleaned.rfind("}")
    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        try:
            return json.loads(cleaned[first_brace:last_brace + 1])
        except Exception:
            pass

    return None


class AIProviderService:
    """Unified client managing Ollama (local) and Groq (cloud) LLM requests."""

    def __init__(self):
        pass

    @staticmethod
    def get_active_provider_name() -> str:
        """Resolves the active AI provider based on environment & configuration."""
        # 1. Environment variable explicit override (standard Twelve-Factor pattern)
        env_provider = os.getenv("AI_PROVIDER", "").strip().lower()
        if env_provider:
            return env_provider

        # 2. Flask app configuration
        try:
            from flask import current_app, has_app_context
            if has_app_context() and current_app:
                cfg_provider = current_app.config.get("AI_PROVIDER")
                if cfg_provider:
                    return str(cfg_provider).strip().lower()
        except Exception:
            pass

        # 3. Detect cloud vs local environment default
        app_env = os.getenv("APP_ENV", "").strip().lower()
        is_cloud = bool(
            app_env == "production" or
            os.getenv("RENDER") or
            os.getenv("DYNO")
        )
        if is_cloud:
            return "groq"
        return "ollama"

    @staticmethod
    def get_ollama_config() -> Tuple[str, str]:
        """Resolves (base_url, model) for Ollama."""
        base_url = (
            os.getenv("OLLAMA_BASE_URL") or
            os.getenv("OLLAMA_HOST") or
            DEFAULT_OLLAMA_BASE_URL
        ).rstrip("/")
        model = os.getenv("OLLAMA_MODEL") or DEFAULT_OLLAMA_MODEL
        return base_url, model

    @staticmethod
    def get_groq_config() -> Tuple[str, str, str]:
        """Resolves (api_key, model, base_url) for Groq."""
        api_key = os.getenv("GROQ_API_KEY", "").strip()
        model = os.getenv("GROQ_MODEL") or DEFAULT_GROQ_MODEL
        base_url = (os.getenv("GROQ_BASE_URL") or DEFAULT_GROQ_BASE_URL).rstrip("/")
        return api_key, model, base_url

    def get_status(self) -> Dict[str, Any]:
        """Returns safe provider diagnostics for the health check without exposing keys."""
        provider = self.get_active_provider_name()
        app_env = os.getenv("APP_ENV", "development" if provider == "ollama" else "production")

        if provider == "groq":
            api_key, model, base_url = self.get_groq_config()
            is_configured = bool(api_key and len(api_key) > 8)
            masked_key = f"{api_key[:4]}...{api_key[-4:]}" if is_configured else "not_configured"
            return {
                "app_env": app_env,
                "provider": "groq",
                "model": model,
                "configured": is_configured,
                "api_key_status": "present" if is_configured else "missing",
                "key_preview": masked_key,
                "endpoint": base_url
            }
        else:
            base_url, model = self.get_ollama_config()
            return {
                "app_env": app_env,
                "provider": "ollama",
                "model": model,
                "configured": True,
                "endpoint": base_url
            }

    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        system_prompt: Optional[str] = None,
        json_mode: bool = False,
        temperature: float = 0.7,
        timeout: Optional[float] = None,
        override_provider: Optional[str] = None,
        override_model: Optional[str] = None,
        override_base_url: Optional[str] = None
    ) -> AIResponse:
        """
        Executes a chat completion against the active AI provider (Ollama or Groq).
        Returns an AIResponse object indicating whether genuine LLM output or fallback was obtained.
        """
        provider = (override_provider or self.get_active_provider_name()).lower()
        t0 = time.time()

        # Build message payload
        formatted_messages = []
        if system_prompt:
            formatted_messages.append({"role": "system", "content": system_prompt})
        for m in messages:
            if isinstance(m, dict) and m.get("content"):
                formatted_messages.append({
                    "role": m.get("role", "user"),
                    "content": str(m.get("content", "")).strip()
                })

        if provider == "groq":
            return self._call_groq(
                formatted_messages=formatted_messages,
                json_mode=json_mode,
                temperature=temperature,
                timeout=timeout or float(os.getenv("GROQ_TIMEOUT", "20.0")),
                override_model=override_model,
                override_base_url=override_base_url,
                start_time=t0
            )
        else:
            return self._call_ollama(
                formatted_messages=formatted_messages,
                json_mode=json_mode,
                temperature=temperature,
                timeout=timeout or float(os.getenv("OLLAMA_TIMEOUT", "30.0")),
                override_model=override_model,
                override_base_url=override_base_url,
                start_time=t0
            )

    def _call_groq(
        self,
        formatted_messages: List[Dict[str, str]],
        json_mode: bool,
        temperature: float,
        timeout: float,
        override_model: Optional[str],
        override_base_url: Optional[str],
        start_time: float
    ) -> AIResponse:
        """Executes hosted LLM completion via Groq REST API."""
        api_key, default_model, default_base_url = self.get_groq_config()
        model = override_model or default_model
        base_url = override_base_url or default_base_url

        if not api_key:
            err = "GROQ_API_KEY environment variable is missing or empty"
            logger.warning(f"Groq invocation bypassed: {err}")
            return AIResponse(
                content="",
                raw_content="",
                provider="groq",
                model=model,
                is_fallback=True,
                fallback_reason=err,
                latency_ms=(time.time() - start_time) * 1000
            )

        endpoint = f"{base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        payload: Dict[str, Any] = {
            "model": model,
            "messages": formatted_messages,
            "temperature": max(0.0, min(temperature, 1.0)),
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}

        try:
            with httpx.Client(timeout=timeout) as client:
                resp = client.post(endpoint, headers=headers, json=payload)
                latency = (time.time() - start_time) * 1000

                if resp.status_code == 200:
                    data = resp.json()
                    choices = data.get("choices", [])
                    if choices:
                        content = choices[0].get("message", {}).get("content", "").strip()
                        parsed = extract_json_from_text(content) if json_mode else None
                        logger.info(f"Groq completion succeeded [{model}] ({latency:.1f}ms).")
                        return AIResponse(
                            content=content,
                            raw_content=content,
                            provider="groq",
                            model=model,
                            is_fallback=False,
                            latency_ms=latency,
                            parsed_json=parsed
                        )
                    else:
                        reason = "Groq returned 200 OK with empty choices list"
                elif resp.status_code == 429:
                    reason = f"Groq rate limit exceeded (HTTP 429): {resp.text[:120]}"
                    logger.warning(reason)
                elif resp.status_code == 401:
                    reason = "Groq authentication failed (HTTP 401): invalid API key"
                    logger.error(reason)
                else:
                    reason = f"Groq returned HTTP {resp.status_code}: {resp.text[:120]}"
                    logger.warning(reason)

        except httpx.TimeoutException:
            reason = f"Groq API call timed out after {timeout}s"
            logger.warning(reason)
        except httpx.ConnectError as ce:
            reason = f"Groq connection failed: {ce}"
            logger.warning(reason)
        except Exception as ex:
            reason = f"Groq unexpected error: {ex}"
            logger.error(reason)

        return AIResponse(
            content="",
            raw_content="",
            provider="groq",
            model=model,
            is_fallback=True,
            fallback_reason=reason,
            latency_ms=(time.time() - start_time) * 1000
        )

    def _call_ollama(
        self,
        formatted_messages: List[Dict[str, str]],
        json_mode: bool,
        temperature: float,
        timeout: float,
        override_model: Optional[str],
        override_base_url: Optional[str],
        start_time: float
    ) -> AIResponse:
        """Executes local LLM completion via Ollama REST API."""
        default_base_url, default_model = self.get_ollama_config()
        base_url = override_base_url or default_base_url
        model = override_model or default_model

        endpoint = f"{base_url}/api/chat"
        payload: Dict[str, Any] = {
            "model": model,
            "messages": formatted_messages,
            "stream": False,
            "options": {
                "temperature": max(0.0, min(temperature, 1.0)),
                "top_p": 0.9,
                "repeat_penalty": 1.15
            }
        }
        if json_mode:
            payload["format"] = "json"

        try:
            with httpx.Client(timeout=timeout) as client:
                resp = client.post(endpoint, json=payload)
                latency = (time.time() - start_time) * 1000

                if resp.status_code == 200:
                    data = resp.json()
                    content = data.get("message", {}).get("content", "").strip()
                    if content:
                        parsed = extract_json_from_text(content) if json_mode else None
                        logger.info(f"Ollama completion succeeded [{model}] ({latency:.1f}ms).")
                        return AIResponse(
                            content=content,
                            raw_content=content,
                            provider="ollama",
                            model=model,
                            is_fallback=False,
                            latency_ms=latency,
                            parsed_json=parsed
                        )
                    else:
                        reason = "Ollama returned 200 OK with empty content"
                else:
                    reason = f"Ollama returned HTTP {resp.status_code}: {resp.text[:120]}"
                    logger.warning(reason)

        except httpx.ConnectError:
            reason = f"Ollama is offline or unreachable at {base_url}"
            logger.info(reason)
        except httpx.TimeoutException:
            reason = f"Ollama generation timed out after {timeout}s"
            logger.warning(reason)
        except Exception as ex:
            reason = f"Ollama request error: {ex}"
            logger.warning(reason)

        return AIResponse(
            content="",
            raw_content="",
            provider="ollama",
            model=model,
            is_fallback=True,
            fallback_reason=reason,
            latency_ms=(time.time() - start_time) * 1000
        )


# Global singleton provider instance
ai_provider = AIProviderService()
