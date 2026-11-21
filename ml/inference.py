"""
ML Inference Module — Public Interface for CareAI Agents
Provides classify_text() as a supportive signal for existing AI agents.

IMPORTANT: These predictions are supportive analytical signals only.
They are NOT clinical diagnoses and must not override Safety Agent rules.
"""

import os
import json
import pickle
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
MODELS_DIR = PROJECT_ROOT / "models_saved"
LABEL_NAMES = {0: "Normal", 1: "Depression", 2: "Anxiety", 3: "Suicidal"}

# Module-level singletons (lazy loaded)
_lr_model = None
_tfidf_vectorizer = None
_bert_model = None
_bert_tokenizer = None
_active_model = None  # "distilbert" | "lr" | None
_initialized = False

# Defer all heavy imports (sklearn, torch, etc.) to first use to avoid
# Windows/pyarrow access violations when module is imported at test collection time.


def _load_lr_model():
    """Load TF-IDF + Logistic Regression model."""
    global _lr_model, _tfidf_vectorizer
    lr_path = MODELS_DIR / "lr_model.pkl"
    tfidf_path = MODELS_DIR / "tfidf_vectorizer.pkl"

    if not lr_path.exists() or not tfidf_path.exists():
        return False

    try:
        with open(tfidf_path, "rb") as f:
            _tfidf_vectorizer = pickle.load(f)
        with open(lr_path, "rb") as f:
            _lr_model = pickle.load(f)
        logger.info("ML: Logistic Regression model loaded successfully")
        return True
    except Exception as e:
        logger.warning(f"ML: Failed to load LR model: {e}")
        return False


def _load_bert_model():
    """Load fine-tuned DistilBERT model."""
    global _bert_model, _bert_tokenizer
    bert_dir = MODELS_DIR / "distilbert"

    if not (bert_dir / "config.json").exists():
        return False

    try:
        import torch
        from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification

        _bert_tokenizer = DistilBertTokenizerFast.from_pretrained(str(bert_dir))
        _bert_model = DistilBertForSequenceClassification.from_pretrained(str(bert_dir))
        _bert_model.eval()
        logger.info("ML: DistilBERT model loaded successfully")
        return True
    except Exception as e:
        logger.warning(f"ML: Failed to load DistilBERT: {e}")
        return False


def _initialize():
    """Initialize models on first use (lazy loading)."""
    global _active_model, _initialized
    if _initialized:
        return

    _initialized = True

    # Try DistilBERT first (better model)
    if _load_bert_model():
        _active_model = "distilbert"
        return

    # Fallback to Logistic Regression
    if _load_lr_model():
        _active_model = "lr"
        return

    _active_model = None
    logger.warning("ML: No trained models found. Run ml/train_lr.py first.")


def is_ml_available() -> bool:
    """Check if any ML model is available for classification."""
    _initialize()
    return _active_model is not None


def classify_text(text: str) -> dict:
    """
    Classify text into mental health category.

    Args:
        text: Input text to classify (journal entry, chat message, etc.)

    Returns:
        dict with keys:
            - label: predicted class ("Normal", "Depression", "Anxiety", "Suicidal")
            - confidence: confidence score [0, 1]
            - all_scores: dict of {label: score} for all classes
            - model_used: "distilbert" | "lr" | "none"
            - disclaimer: always present, warning about clinical use

    Example:
        >>> result = classify_text("I feel hopeless and don't want to live")
        >>> result["label"]  # "Suicidal"
        >>> result["confidence"]  # 0.92
    """
    _initialize()

    DISCLAIMER = (
        "This is a supportive ML signal only, not a clinical diagnosis. "
        "Safety decisions are handled by the Safety Agent independently."
    )

    if not text or not isinstance(text, str) or len(text.strip()) < 3:
        return {
            "label": "unknown",
            "confidence": 0.0,
            "all_scores": {},
            "model_used": "none",
            "disclaimer": DISCLAIMER,
            "error": "Input text is too short or empty"
        }

    if _active_model == "distilbert":
        return _classify_bert(text, DISCLAIMER)
    elif _active_model == "lr":
        return _classify_lr(text, DISCLAIMER)
    else:
        return {
            "label": "unknown",
            "confidence": 0.0,
            "all_scores": {},
            "model_used": "none",
            "disclaimer": DISCLAIMER,
            "error": "No ML model available. Run training scripts first."
        }


def _classify_bert(text: str, disclaimer: str) -> dict:
    """Classify using DistilBERT."""
    try:
        import torch
        import torch.nn.functional as F

        inputs = _bert_tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=128,
            padding=True
        )
        with torch.no_grad():
            outputs = _bert_model(**inputs)
            probs = F.softmax(outputs.logits, dim=1).squeeze(0)

        probs_list = probs.numpy().tolist()
        predicted_id = int(probs.argmax().item())
        confidence = round(probs_list[predicted_id], 4)
        label = LABEL_NAMES.get(predicted_id, "unknown")
        all_scores = {LABEL_NAMES[i]: round(p, 4) for i, p in enumerate(probs_list)}

        return {
            "label": label,
            "confidence": confidence,
            "all_scores": all_scores,
            "model_used": "distilbert",
            "disclaimer": disclaimer
        }
    except Exception as e:
        logger.error(f"ML: DistilBERT inference failed: {e}. Falling back to LR.")
        # Fallback to LR
        if _load_lr_model():
            return _classify_lr(text, disclaimer)
        return {
            "label": "unknown",
            "confidence": 0.0,
            "all_scores": {},
            "model_used": "none",
            "disclaimer": disclaimer,
            "error": str(e)
        }


def _classify_lr(text: str, disclaimer: str) -> dict:
    """Classify using Logistic Regression + TF-IDF."""
    try:
        import numpy as np

        X = _tfidf_vectorizer.transform([text])
        probs = _lr_model.predict_proba(X)[0]
        predicted_id = int(np.argmax(probs))
        confidence = round(float(probs[predicted_id]), 4)
        label = LABEL_NAMES.get(predicted_id, "unknown")
        all_scores = {LABEL_NAMES[i]: round(float(p), 4) for i, p in enumerate(probs)}

        return {
            "label": label,
            "confidence": confidence,
            "all_scores": all_scores,
            "model_used": "lr",
            "disclaimer": disclaimer
        }
    except Exception as e:
        logger.error(f"ML: LR inference failed: {e}")
        return {
            "label": "unknown",
            "confidence": 0.0,
            "all_scores": {},
            "model_used": "none",
            "disclaimer": disclaimer,
            "error": str(e)
        }


def get_model_info() -> dict:
    """Return information about the currently loaded model."""
    _initialize()
    info = {
        "active_model": _active_model,
        "is_available": _active_model is not None,
        "label_names": list(LABEL_NAMES.values()),
    }

    # Load metrics if available
    if _active_model == "distilbert":
        metrics_path = MODELS_DIR / "distilbert_metrics.json"
    elif _active_model == "lr":
        metrics_path = MODELS_DIR / "lr_metrics.json"
    else:
        metrics_path = None

    if metrics_path and metrics_path.exists():
        try:
            with open(metrics_path) as f:
                metrics = json.load(f)
            info["test_accuracy"] = metrics.get("test_accuracy")
            info["test_f1_macro"] = metrics.get("test_f1_macro")
        except Exception:
            pass

    return info


if __name__ == "__main__":
    # Quick smoke test
    print("ML Inference Module — Smoke Test")
    print(f"Available: {is_ml_available()}")
    print(f"Model info: {get_model_info()}")

    test_cases = [
        "I feel so hopeless and don't want to live anymore",
        "I had a great day today, feeling very happy",
        "I'm constantly worried and can't stop overthinking everything",
        "I feel empty inside and nothing brings me joy anymore",
    ]

    for text in test_cases:
        result = classify_text(text)
        print(f"\nText: {text[:60]}...")
        print(f"  Label: {result['label']} (confidence: {result['confidence']})")
        print(f"  Model: {result['model_used']}")
        print(f"  Scores: {result['all_scores']}")
