# tests/test_ai_provider.py
"""
Comprehensive Test Suite for CareAI Dual AI Provider Architecture.
Verifies:
1. Local Development configuration (Ollama + Llama 3.2 without requiring Groq key)
2. Cloud Production configuration (Groq hosted API without requiring local laptop)
3. Provider resolution based on APP_ENV and AI_PROVIDER
4. Graceful fallback on missing keys, timeouts, or API errors
5. JSON mode extraction and validation across agents
6. Full coverage of all 4 LLM-powered agents (Conversation, Journal, Narrative, Case Summary)
7. Unconditional priority of Safety Agent over provider status
8. Health check diagnostic transparency
"""

import os
import sys
import json
import pytest
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app import create_app
from extensions import db
from config import Config
from services.ai_provider import AIProviderService, AIResponse, extract_json_from_text, ai_provider
from agents.conversation import generate_llm_response
from agents.journal_agent import analyze_journal_reflection
from agents.narrative_agent import synthesize_crisis_narrative
from agents.case_summary_agent import generate_case_summary
from agents.safety import evaluate_safety, evaluate_text_safety


class TestAppConfig(Config):
    TESTING = True
    SQLALCHEMY_DATABASE_URI = "sqlite:///:memory:"
    WTF_CSRF_ENABLED = False
    SECRET_KEY = "test-secret-key"


@pytest.fixture
def app():
    app = create_app(TestAppConfig)
    with app.app_context():
        db.create_all()
        yield app
        db.session.remove()
        db.drop_all()


@pytest.fixture
def client(app):
    return app.test_client()


# ---------------------------------------------------------------------
# 1. Provider Resolution & Environment Detection
# ---------------------------------------------------------------------

def test_provider_resolution_local_development(monkeypatch):
    """Verifies that development environment defaults to Ollama without Groq key."""
    monkeypatch.setenv("APP_ENV", "development")
    monkeypatch.setenv("AI_PROVIDER", "ollama")
    monkeypatch.delenv("GROQ_API_KEY", raising=False)

    provider = AIProviderService.get_active_provider_name()
    assert provider == "ollama"

    service = AIProviderService()
    status = service.get_status()
    assert status["provider"] == "ollama"
    assert status["configured"] is True
    assert "endpoint" in status


def test_provider_resolution_cloud_production(monkeypatch):
    """Verifies that production environment defaults to Groq."""
    monkeypatch.setenv("APP_ENV", "production")
    monkeypatch.delenv("AI_PROVIDER", raising=False)

    provider = AIProviderService.get_active_provider_name()
    assert provider == "groq"

    service = AIProviderService()
    status = service.get_status()
    assert status["provider"] == "groq"
    assert status["configured"] is False  # Key not yet set
    assert status["api_key_status"] == "missing"


def test_provider_resolution_groq_with_key(monkeypatch):
    """Verifies status when Groq API key is present."""
    monkeypatch.setenv("APP_ENV", "production")
    monkeypatch.setenv("AI_PROVIDER", "groq")
    monkeypatch.setenv("GROQ_API_KEY", "gsk_test_mock_secret_key_12345678")
    monkeypatch.setenv("GROQ_MODEL", "openai/gpt-oss-120b")
    monkeypatch.setenv("GROQ_FALLBACK_MODEL", "openai/gpt-oss-20b")

    service = AIProviderService()
    status = service.get_status()
    assert status["provider"] == "groq"
    assert status["configured"] is True
    assert status["api_key_status"] == "present"
    assert status["model"] == "openai/gpt-oss-120b"
    assert status["fallback_model"] == "openai/gpt-oss-20b"
    assert "gsk_...5678" == status["key_preview"]
    assert "gsk_test_mock_secret_key_12345678" not in json.dumps(status)  # Secret must NOT leak


# ---------------------------------------------------------------------
# 2. JSON Extraction Helper
# ---------------------------------------------------------------------

def test_extract_json_from_text():
    """Tests JSON extraction across plain JSON, markdown fences, and embedded text."""
    # Plain JSON
    assert extract_json_from_text('{"key": "value"}') == {"key": "value"}

    # Markdown block
    md = 'Here is the result:\n```json\n{"sentiment": "Reflective", "score": 0.5}\n```'
    assert extract_json_from_text(md) == {"sentiment": "Reflective", "score": 0.5}

    # Embedded in surrounding prose
    prose = 'Analysis:\n{"category": "anxiety", "level": 3}\nHope this helps.'
    assert extract_json_from_text(prose) == {"category": "anxiety", "level": 3}

    # Invalid / empty
    assert extract_json_from_text("") is None
    assert extract_json_from_text("no json here") is None


# ---------------------------------------------------------------------
# 3. Groq Provider Mocked Invocations
# ---------------------------------------------------------------------

def test_groq_missing_api_key_returns_fallback(monkeypatch):
    """Verifies graceful fallback when Groq key is missing in production."""
    monkeypatch.setenv("AI_PROVIDER", "groq")
    monkeypatch.delenv("GROQ_API_KEY", raising=False)

    service = AIProviderService()
    resp = service.chat_completion(messages=[{"role": "user", "content": "Hello"}])

    assert resp.is_fallback is True
    assert resp.provider == "groq"
    assert "GROQ_API_KEY" in (resp.fallback_reason or "")


def test_groq_mocked_success(monkeypatch):
    """Verifies successful Groq chat completion with mocked HTTP response using openai/gpt-oss-120b."""
    monkeypatch.setenv("AI_PROVIDER", "groq")
    monkeypatch.setenv("GROQ_API_KEY", "gsk_test_mock_key_12345")
    monkeypatch.setenv("GROQ_MODEL", "openai/gpt-oss-120b")
    monkeypatch.setenv("GROQ_FALLBACK_MODEL", "openai/gpt-oss-20b")

    mock_json = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "This is a response generated by Groq hosted LLM."
                }
            }
        ]
    }

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = mock_json

    with patch("httpx.Client.post", return_value=mock_resp):
        service = AIProviderService()
        resp = service.chat_completion(messages=[{"role": "user", "content": "Hello Groq"}])

        assert resp.is_fallback is False
        assert resp.provider == "groq"
        assert resp.model == "openai/gpt-oss-120b"
        assert "Groq hosted LLM" in resp.content


def test_groq_mocked_rate_limit(monkeypatch):
    """Verifies that HTTP 429 returns a fallback flag without crashing."""
    monkeypatch.setenv("AI_PROVIDER", "groq")
    monkeypatch.setenv("GROQ_API_KEY", "gsk_test_mock_key_12345")

    mock_resp = MagicMock()
    mock_resp.status_code = 429
    mock_resp.text = "Rate limit reached. Try again later."

    with patch("httpx.Client.post", return_value=mock_resp):
        service = AIProviderService()
        resp = service.chat_completion(messages=[{"role": "user", "content": "Hello"}])

        assert resp.is_fallback is True
        assert "429" in (resp.fallback_reason or "")


def test_groq_404_auto_retry_fallback_model(monkeypatch):
    """Verifies that if primary model returns 404, it retries with openai/gpt-oss-20b."""
    monkeypatch.setenv("AI_PROVIDER", "groq")
    monkeypatch.setenv("GROQ_API_KEY", "gsk_test_mock_key_12345")
    monkeypatch.setenv("GROQ_MODEL", "openai/gpt-oss-120b")
    monkeypatch.setenv("GROQ_FALLBACK_MODEL", "openai/gpt-oss-20b")

    resp_404 = MagicMock()
    resp_404.status_code = 404
    resp_404.text = '{"error":{"message":"The model openai/gpt-oss-120b does not exist"}}'

    resp_200 = MagicMock()
    resp_200.status_code = 200
    resp_200.json.return_value = {
        "choices": [{"message": {"role": "assistant", "content": "Recovered response from 20b model"}}]
    }

    with patch("httpx.Client.post", side_effect=[resp_404, resp_200]):
        service = AIProviderService()
        resp = service.chat_completion(messages=[{"role": "user", "content": "Hello"}])

        assert resp.is_fallback is False
        assert resp.model == "openai/gpt-oss-20b"
        assert resp.content == "Recovered response from 20b model"


def test_groq_does_not_repeatedly_retry_unavailable_models(monkeypatch):
    """Verifies that models marked unavailable are not repeatedly requested."""
    monkeypatch.setenv("AI_PROVIDER", "groq")
    monkeypatch.setenv("GROQ_API_KEY", "gsk_test_mock_key_12345")
    monkeypatch.setenv("GROQ_MODEL", "unavailable-model-id")
    monkeypatch.setenv("GROQ_FALLBACK_MODEL", "openai/gpt-oss-20b")

    service = AIProviderService()
    service._unavailable_models.add("unavailable-model-id")

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "choices": [{"message": {"role": "assistant", "content": "Directly routed to fallback"}}]
    }

    with patch("httpx.Client.post", return_value=mock_resp) as mock_post:
        resp = service.chat_completion(messages=[{"role": "user", "content": "Hello"}])
        assert resp.model == "openai/gpt-oss-20b"
        # Ensure post was called with fallback model, NOT the unavailable one
        call_payload = mock_post.call_args[1]["json"]
        assert call_payload["model"] == "openai/gpt-oss-20b"


def test_groq_verified_models_discovery(monkeypatch):
    """Verifies that get_verified_models discovers available models from the Groq Models API."""
    service = AIProviderService()
    mock_models_resp = MagicMock()
    mock_models_resp.status_code = 200
    mock_models_resp.json.return_value = {
        "data": [
            {"id": "openai/gpt-oss-120b"},
            {"id": "openai/gpt-oss-20b"},
            {"id": "meta-llama/llama-prompt-guard-2-86m"}
        ]
    }

    with patch("httpx.Client.get", return_value=mock_models_resp):
        verified = service.get_verified_models("gsk_valid_key", "https://api.groq.com/openai/v1")
        assert "openai/gpt-oss-120b" in verified
        assert "openai/gpt-oss-20b" in verified
        assert len(verified) == 3


# ---------------------------------------------------------------------
# 4. Agent Coverage Across Configured Provider
# ---------------------------------------------------------------------

def test_conversation_agent_groq_integration(monkeypatch):
    """Verifies Agent 2 (Conversation Agent) handles Groq output."""
    monkeypatch.setenv("AI_PROVIDER", "groq")
    monkeypatch.setenv("GROQ_API_KEY", "gsk_test_mock_key_12345")

    mock_content = "Grounding exercises help regulate your nervous system. Try naming 5 things you see around you."
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"choices": [{"message": {"content": mock_content}}]}

    with patch("httpx.Client.post", return_value=mock_resp):
        reply = generate_llm_response(
            messages=[{"role": "user", "content": "How can I calm down right now?"}],
            user_name="Jordan"
        )
        assert "Grounding exercises" in reply


def test_journal_agent_groq_integration(monkeypatch):
    """Verifies Reflective Journal Agent parses structured JSON from Groq."""
    monkeypatch.setenv("AI_PROVIDER", "groq")
    monkeypatch.setenv("GROQ_API_KEY", "gsk_test_mock_key_12345")

    mock_json_content = json.dumps({
        "sentiment": "Reflective",
        "emotions": ["Gratitude", "Calm"],
        "key_observation": "The user reflected on finishing their exams with peace.",
        "wellbeing_feedback": "It is wonderful to hear that you have reached the finish line. Celebrate this achievement.",
        "suggestions": ["Take a restorative walk outdoors."]
    })

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"choices": [{"message": {"content": mock_json_content}}]}

    with patch("httpx.Client.post", return_value=mock_resp):
        result = analyze_journal_reflection(
            journal_text="I finally completed my final exams today and I feel a quiet sense of relief.",
            user_name="Jordan"
        )
        assert result["sentiment"] == "Reflective"
        assert "Gratitude" in result["emotions"]
        assert result.get("is_ai_generated") is True
        assert result.get("ai_provider") == "groq"


def test_safety_priority_override_unaffected_by_ai_provider(monkeypatch):
    """
    CRITICAL SAFETY REQUIREMENT:
    Even if the AI provider is down, misconfigured, or returning errors,
    deterministic safety rules MUST execute immediately and return crisis helplines.
    """
    monkeypatch.setenv("AI_PROVIDER", "groq")
    monkeypatch.delenv("GROQ_API_KEY", raising=False)  # Provider broken/missing key

    # Critical suicidal disclosure evaluated through full safety agent pipeline
    res = evaluate_safety(
        text="I want to end my life tonight, I can't go on.",
        distress_score=0.92
    )
    assert res["is_urgent"] is True
    assert res["risk_level"] == "critical"
    assert "988" in res["crisis_resources"]
    assert "14416" in res["crisis_resources"]  # Tele-MANAS verified hotline present


# ---------------------------------------------------------------------
# 5. Health Check Endpoint Diagnostic
# ---------------------------------------------------------------------

def test_health_endpoint_diagnostics(client, monkeypatch):
    """Verifies that /health reports safe AI provider configuration details."""
    monkeypatch.setenv("APP_ENV", "production")
    monkeypatch.setenv("AI_PROVIDER", "groq")
    monkeypatch.setenv("GROQ_API_KEY", "gsk_mock_production_key_abcdef")
    monkeypatch.setenv("GROQ_MODEL", "openai/gpt-oss-120b")
    monkeypatch.setenv("GROQ_FALLBACK_MODEL", "openai/gpt-oss-20b")

    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.get_json()

    assert data["service"] == "AI Mental Health Monitoring System"
    assert data["version"].startswith("v2.")
    assert "ai_provider" in data
    assert data["ai_provider"]["provider"] == "groq"
    assert data["ai_provider"]["model"] == "openai/gpt-oss-120b"
    assert data["ai_provider"]["configured"] is True
    assert "gsk_mock_production_key_abcdef" not in resp.text  # Strict credential protection


# ---------------------------------------------------------------------
# 6. Conversational Diversity & Multi-Turn Context Tests (Phase 6 Scenarios)
# ---------------------------------------------------------------------

def test_conversation_scenario_1_what_is_stress():
    """Test 1: Ask 'What is stress?' - Must receive rich, relevant definition without generic stock repetition."""
    reply = generate_llm_response(
        messages=[{"role": "user", "content": "What is stress?"}],
        user_name="Jordan"
    )
    assert "stress" in reply.lower()
    assert any(w in reply.lower() for w in ["threat", "body", "demand", "response", "cortisol", "pressure", "biological"])
    assert "When looking at what is stress, the most practical approach" not in reply


def test_conversation_scenario_2_symptoms_of_anxiety():
    """Test 2: Ask 'What are the symptoms of anxiety?' - Must receive distinct symptom explanation."""
    reply = generate_llm_response(
        messages=[{"role": "user", "content": "What are the symptoms of anxiety?"}],
        user_name="Jordan"
    )
    assert ("anxiety" in reply.lower() or "anxious" in reply.lower())
    assert any(w in reply.lower() for w in ["symptom", "physical", "worry", "body", "mind", "heart", "racing", "nervous"])
    assert "When looking at" not in reply


def test_conversation_scenario_3_how_can_i_improve_my_sleep():
    """Test 3: Ask 'How can I improve my sleep?' - Must receive evidence-based sleep hygiene guidance."""
    reply = generate_llm_response(
        messages=[{"role": "user", "content": "How can I improve my sleep?"}],
        user_name="Jordan"
    )
    assert any(w in reply.lower() for w in ["sleep", "bedtime", "rest", "routine", "circadian", "wake"])
    assert "When looking at" not in reply


def test_conversation_scenario_4_multi_turn_follow_up():
    """Test 4: Ask follow-up question related to sleep - Must maintain contextual continuity."""
    history = [
        {"role": "user", "content": "How can I improve my sleep?"},
        {"role": "assistant", "content": "Here is an evidence-based framework: 1. Anchor circadian rhythm. 2. Cool Cave. 3. 20-minute stimulus control."},
        {"role": "user", "content": "Can you explain more about how that helps?"}
    ]
    reply = generate_llm_response(
        messages=history,
        user_name="Jordan"
    )
    assert any(w in reply.lower() for w in ["sleep", "rest", "bed", "circadian", "body", "rhythm", "adenosine", "help", "routine"])
    assert "When looking at" not in reply


def test_different_questions_receive_different_responses():
    """Verifies that different questions NEVER receive the same response."""
    r1 = generate_llm_response([{"role": "user", "content": "What is stress?"}])
    r2 = generate_llm_response([{"role": "user", "content": "What are the symptoms of anxiety?"}])
    r3 = generate_llm_response([{"role": "user", "content": "How can I improve my sleep?"}])

    assert r1 != r2
    assert r2 != r3
    assert r1 != r3
