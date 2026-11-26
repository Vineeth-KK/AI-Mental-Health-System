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
DEFAULT_GROQ_PRIMARY_MODEL = "openai/gpt-oss-120b"
DEFAULT_GROQ_FALLBACK_MODEL = "openai/gpt-oss-20b"
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
        self._verified_models_cache: Optional[List[str]] = None
        self._models_cache_time: float = 0.0
        self._unavailable_models: set = set()

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
    def get_groq_config() -> Tuple[str, str, str, str]:
        """Resolves (api_key, primary_model, fallback_model, base_url) for Groq."""
        api_key = os.getenv("GROQ_API_KEY", "").strip().strip("'\"")
        primary_model = (os.getenv("GROQ_MODEL") or DEFAULT_GROQ_PRIMARY_MODEL).strip()
        fallback_model = (os.getenv("GROQ_FALLBACK_MODEL") or DEFAULT_GROQ_FALLBACK_MODEL).strip()
        base_url = (os.getenv("GROQ_BASE_URL") or DEFAULT_GROQ_BASE_URL).rstrip("/")
        return api_key, primary_model, fallback_model, base_url

    def get_verified_models(self, api_key: str, base_url: str, force_refresh: bool = False) -> List[str]:
        """Queries the authenticated Groq Models API to check active, accessible models."""
        now = time.time()
        if not force_refresh and self._verified_models_cache is not None and (now - self._models_cache_time) < 300:
            return self._verified_models_cache

        if not api_key:
            return []

        try:
            headers = {"Authorization": f"Bearer {api_key}"}
            with httpx.Client(timeout=8.0) as client:
                resp = client.get(f"{base_url}/models", headers=headers)
                if resp.status_code == 200:
                    data = resp.json()
                    model_items = data.get("data", [])
                    models = [m.get("id") for m in model_items if isinstance(m, dict) and m.get("id")]
                    self._verified_models_cache = models
                    self._models_cache_time = now
                    logger.info(f"Verified {len(models)} active Groq models via authenticated API.")
                    return models
                elif resp.status_code == 401:
                    logger.error("Groq authentication failed: GROQ_API_KEY is invalid or unauthorized.")
                elif resp.status_code == 403:
                    logger.error("Groq access forbidden: account or project permissions restricted.")
        except Exception as e:
            logger.warning(f"Could not verify models from Groq Models API: {e}")

        return self._verified_models_cache or []

    def resolve_groq_model(self, api_key: str, base_url: str, requested_model: Optional[str] = None) -> Tuple[str, Optional[str]]:
        """
        Determines the best available model by checking the authenticated Groq Models API.
        Returns (selected_model, fallback_candidate).
        """
        _, primary_config, fallback_config, _ = self.get_groq_config()
        primary = requested_model or primary_config
        fallback = fallback_config

        verified = self.get_verified_models(api_key, base_url)
        if not verified:
            # If verification endpoint couldn't be reached, avoid known unavailable models
            if primary in self._unavailable_models:
                return (fallback, None)
            return (primary, fallback if fallback not in self._unavailable_models else None)

        # 1. Primary model is verified and accessible
        if primary in verified and primary not in self._unavailable_models:
            return (primary, fallback if (fallback in verified and fallback not in self._unavailable_models) else None)

        # 2. Fallback model is verified and accessible
        if fallback in verified and fallback not in self._unavailable_models:
            logger.info(f"Primary model '{primary}' not available; using verified fallback '{fallback}'.")
            return (fallback, None)

        # 3. Dynamic discovery: Pick an active text chat completion model from verified list
        chat_candidates = [
            m for m in verified
            if m not in self._unavailable_models
            and not any(x in m.lower() for x in ["whisper", "guard", "embed", "safeguard", "tool", "distil", "rerank"])
        ]
        if chat_candidates:
            selected = chat_candidates[0]
            logger.warning(
                f"Configured models '{primary}' and '{fallback}' not verified in Groq account. "
                f"Dynamically selected active model: '{selected}'"
            )
            return (selected, None)

        return (primary, fallback)

    def get_status(self, check_live: bool = False) -> Dict[str, Any]:
        """Returns safe provider diagnostics for the health check without exposing keys."""
        provider = self.get_active_provider_name()
        app_env = os.getenv("APP_ENV", "development" if provider == "ollama" else "production")

        if provider == "groq":
            api_key, primary_model, fallback_model, base_url = self.get_groq_config()
            is_configured = bool(api_key and len(api_key) > 8)
            masked_key = f"{api_key[:4]}...{api_key[-4:]}" if is_configured else "not_configured"
            status_data = {
                "app_env": app_env,
                "provider": "groq",
                "model": primary_model,
                "fallback_model": fallback_model,
                "configured": is_configured,
                "api_key_status": "present" if is_configured else "missing",
                "key_preview": masked_key,
                "endpoint": base_url,
                "live_connectivity": "not_checked",
                "verified_models_count": len(self._verified_models_cache) if self._verified_models_cache else 0
            }
            if check_live and is_configured:
                verified = self.get_verified_models(api_key, base_url, force_refresh=True)
                if verified:
                    status_data["live_connectivity"] = "connected"
                    status_data["verified_models_count"] = len(verified)
                    status_data["available_models_sample"] = verified[:5]
                else:
                    status_data["live_connectivity"] = "models_fetch_failed"
            return status_data
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
        """Executes hosted LLM completion via Groq REST API with dynamic model verification."""
        api_key, _, _, default_base_url = self.get_groq_config()
        base_url = override_base_url or default_base_url

        if not api_key:
            err = "GROQ_API_KEY environment variable is missing or empty"
            logger.warning(f"Groq invocation bypassed: {err}")
            return AIResponse(
                content="",
                raw_content="",
                provider="groq",
                model=override_model or DEFAULT_GROQ_PRIMARY_MODEL,
                is_fallback=True,
                fallback_reason=err,
                latency_ms=(time.time() - start_time) * 1000
            )

        model, fallback_candidate = self.resolve_groq_model(api_key, base_url, override_model)

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
                elif resp.status_code == 404:
                    self._unavailable_models.add(model)
                    if fallback_candidate and fallback_candidate != model and fallback_candidate not in self._unavailable_models:
                        logger.warning(
                            f"Groq model '{model}' not accessible (HTTP 404). "
                            f"Retrying once with verified fallback '{fallback_candidate}'..."
                        )
                        payload["model"] = fallback_candidate
                        retry_resp = client.post(endpoint, headers=headers, json=payload)
                        retry_latency = (time.time() - start_time) * 1000
                        if retry_resp.status_code == 200:
                            retry_data = retry_resp.json()
                            retry_choices = retry_data.get("choices", [])
                            if retry_choices:
                                content = retry_choices[0].get("message", {}).get("content", "").strip()
                                parsed = extract_json_from_text(content) if json_mode else None
                                logger.info(f"Groq fallback completion succeeded [{fallback_candidate}] ({retry_latency:.1f}ms).")
                                return AIResponse(
                                    content=content,
                                    raw_content=content,
                                    provider="groq",
                                    model=fallback_candidate,
                                    is_fallback=False,
                                    latency_ms=retry_latency,
                                    parsed_json=parsed
                                )
                        elif retry_resp.status_code == 404:
                            self._unavailable_models.add(fallback_candidate)
                        reason = f"Groq fallback '{fallback_candidate}' failed (HTTP {retry_resp.status_code}): {retry_resp.text[:120]}"
                        logger.warning(reason)
                    else:
                        reason = f"Groq model '{model}' unavailable (HTTP 404) and no verified fallback available."
                        logger.warning(reason)
                elif resp.status_code == 401:
                    reason = "Groq authentication failed (HTTP 401): invalid or unauthorized API key"
                    logger.error(reason)
                elif resp.status_code == 403:
                    reason = f"Groq access forbidden (HTTP 403): account permissions restricted: {resp.text[:120]}"
                    logger.error(reason)
                elif resp.status_code == 429:
                    reason = f"Groq rate limit exceeded (HTTP 429): {resp.text[:120]}"
                    logger.warning(reason)
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
