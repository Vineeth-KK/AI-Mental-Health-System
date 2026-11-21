"""
Phase 5: Model Evaluation & Comparison Report
Loads both trained models and generates a comprehensive comparison report.
"""

import os
import json
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, accuracy_score, f1_score,
    precision_score, recall_score, confusion_matrix
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

PROJECT_ROOT = Path(__file__).parent.parent
CLEAN_CSV = PROJECT_ROOT / "dataset" / "processed" / "mental_health_clean.csv"
MODELS_DIR = PROJECT_ROOT / "models_saved"
BERT_DIR = MODELS_DIR / "distilbert"
LABEL_NAMES = ["Normal", "Depression", "Anxiety", "Suicidal"]
REPORT_PATH = MODELS_DIR / "model_comparison_report.md"


def load_test_set():
    df = pd.read_csv(CLEAN_CSV)
    X = df["text"].astype(str)
    y = df["label_id"].astype(int)
    _, X_temp, _, y_temp = train_test_split(X, y, test_size=0.30, stratify=y, random_state=42)
    _, X_test, _, y_test = train_test_split(X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=42)
    return X_test.tolist(), y_test.tolist()


def evaluate_lr(X_test, y_test):
    lr_path = MODELS_DIR / "lr_model.pkl"
    tfidf_path = MODELS_DIR / "tfidf_vectorizer.pkl"
    if not lr_path.exists() or not tfidf_path.exists():
        return None

    with open(tfidf_path, "rb") as f:
        tfidf = pickle.load(f)
    with open(lr_path, "rb") as f:
        lr = pickle.load(f)

    X_tfidf = tfidf.transform(X_test)
    y_pred = lr.predict(X_tfidf)

    present_labels = sorted(set(y_test))
    present_names = [LABEL_NAMES[i] for i in present_labels]
    report = classification_report(y_test, y_pred, labels=present_labels, target_names=present_names, zero_division=0, output_dict=True)

    return {
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "f1_macro": round(f1_score(y_test, y_pred, average="macro", zero_division=0), 4),
        "f1_weighted": round(f1_score(y_test, y_pred, average="weighted", zero_division=0), 4),
        "precision_macro": round(precision_score(y_test, y_pred, average="macro", zero_division=0), 4),
        "recall_macro": round(recall_score(y_test, y_pred, average="macro", zero_division=0), 4),
        "per_class": report,
        "y_pred": y_pred,
        "y_test": y_test
    }


def evaluate_bert(X_test, y_test):
    if not (BERT_DIR / "config.json").exists():
        return None

    import torch
    from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification

    device = torch.device("cpu")
    tokenizer = DistilBertTokenizerFast.from_pretrained(str(BERT_DIR))
    model = DistilBertForSequenceClassification.from_pretrained(str(BERT_DIR))
    model = model.to(device)
    model.eval()

    all_preds = []
    batch_size = 32
    with torch.no_grad():
        for i in range(0, len(X_test), batch_size):
            batch_texts = X_test[i:i+batch_size]
            encoding = tokenizer(
                batch_texts, padding=True, truncation=True,
                max_length=128, return_tensors="pt"
            )
            outputs = model(**{k: v.to(device) for k, v in encoding.items()})
            preds = outputs.logits.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy().tolist())

    y_pred = all_preds
    present_labels = sorted(set(y_test))
    present_names = [LABEL_NAMES[i] for i in present_labels]
    report = classification_report(y_test, y_pred, labels=present_labels, target_names=present_names, zero_division=0, output_dict=True)

    return {
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "f1_macro": round(f1_score(y_test, y_pred, average="macro", zero_division=0), 4),
        "f1_weighted": round(f1_score(y_test, y_pred, average="weighted", zero_division=0), 4),
        "precision_macro": round(precision_score(y_test, y_pred, average="macro", zero_division=0), 4),
        "recall_macro": round(recall_score(y_test, y_pred, average="macro", zero_division=0), 4),
        "per_class": report,
        "y_pred": y_pred,
        "y_test": y_test
    }


def generate_comparison_report(lr_results, bert_results):
    lines = [
        "# CareAI ML — Model Comparison Report",
        "",
        "## Overview",
        "This report compares two text classification models trained on the",
        "nayma11/Mental-Health_Text-Classification_Dataset (4 classes: Normal, Depression, Anxiety, Suicidal).",
        "",
        "**Important disclaimer:** These ML predictions are supportive analytical signals only.",
        "They are NOT clinical diagnoses and must not override the existing Safety Agent rules.",
        "",
        "## Model Summary",
        "",
        "| Metric | Logistic Regression | DistilBERT |",
        "|--------|--------------------:|----------:|",
    ]

    lr_acc = lr_results["accuracy"] if lr_results else "N/A"
    bert_acc = bert_results["accuracy"] if bert_results else "N/A"
    lr_f1m = lr_results["f1_macro"] if lr_results else "N/A"
    bert_f1m = bert_results["f1_macro"] if bert_results else "N/A"
    lr_f1w = lr_results["f1_weighted"] if lr_results else "N/A"
    bert_f1w = bert_results["f1_weighted"] if bert_results else "N/A"
    lr_prec = lr_results["precision_macro"] if lr_results else "N/A"
    bert_prec = bert_results["precision_macro"] if bert_results else "N/A"
    lr_rec = lr_results["recall_macro"] if lr_results else "N/A"
    bert_rec = bert_results["recall_macro"] if bert_results else "N/A"

    lines += [
        f"| Accuracy | {lr_acc} | {bert_acc} |",
        f"| F1 (Macro) | {lr_f1m} | {bert_f1m} |",
        f"| F1 (Weighted) | {lr_f1w} | {bert_f1w} |",
        f"| Precision (Macro) | {lr_prec} | {bert_prec} |",
        f"| Recall (Macro) | {lr_rec} | {bert_rec} |",
        "",
    ]

    # Determine best model
    if lr_results and bert_results:
        if bert_results["f1_macro"] >= lr_results["f1_macro"]:
            best = "DistilBERT"
            reason = f"DistilBERT achieves F1 macro {bert_results['f1_macro']} vs LR {lr_results['f1_macro']}"
        else:
            best = "Logistic Regression"
            reason = f"LR achieves F1 macro {lr_results['f1_macro']} vs DistilBERT {bert_results['f1_macro']}"
    elif bert_results:
        best = "DistilBERT"
        reason = "LR model not evaluated"
    else:
        best = "Logistic Regression"
        reason = "DistilBERT model not evaluated"

    lines += [
        f"## Recommended Model: **{best}**",
        f"{reason}",
        "",
        "## Integration Notes",
        "- The inference module (`ml/inference.py`) uses the best model automatically.",
        "- Fallback: if DistilBERT fails to load, Logistic Regression is used.",
        "- ML classification result is added as `ml_classification` field in agent state.",
        "- Safety Agent continues to apply its own rules independently.",
        "",
        "## Per-Class Performance",
        "",
        "### Logistic Regression",
    ]
    if lr_results:
        for cls_name in ["Normal", "Depression", "Anxiety", "Suicidal"]:
            if cls_name in lr_results["per_class"]:
                cls = lr_results["per_class"][cls_name]
                lines.append(f"- **{cls_name}**: P={cls['precision']:.3f} R={cls['recall']:.3f} F1={cls['f1-score']:.3f} (n={cls['support']})")
    else:
        lines.append("Not available")

    lines += ["", "### DistilBERT"]
    if bert_results:
        for cls_name in ["Normal", "Depression", "Anxiety", "Suicidal"]:
            if cls_name in bert_results["per_class"]:
                cls = bert_results["per_class"][cls_name]
                lines.append(f"- **{cls_name}**: P={cls['precision']:.3f} R={cls['recall']:.3f} F1={cls['f1-score']:.3f} (n={cls['support']})")
    else:
        lines.append("Not available")

    lines += [
        "",
        "## Files",
        "- `lr_model.pkl` — Logistic Regression model",
        "- `tfidf_vectorizer.pkl` — TF-IDF vectorizer",
        "- `lr_metrics.json` — LR evaluation metrics",
        "- `distilbert/` — DistilBERT model weights + tokenizer",
        "- `distilbert_metrics.json` — DistilBERT evaluation metrics",
        "- `lr_confusion_matrix.png` — LR confusion matrix",
        "- `distilbert_confusion_matrix.png` — DistilBERT confusion matrix",
        "",
        "## Usage",
        "```python",
        "from ml.inference import classify_text, is_ml_available",
        "",
        "if is_ml_available():",
        '    result = classify_text("I feel hopeless and don\'t want to live")',
        '    print(result)  # {"label": "Suicidal", "confidence": 0.92, ...}',
        "```",
    ]

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"\n[OK] Comparison report saved: {REPORT_PATH}")
    return best


if __name__ == "__main__":
    print("Loading test set...")
    X_test, y_test = load_test_set()
    print(f"Test set size: {len(X_test):,}")

    print("\nEvaluating Logistic Regression...")
    lr_results = evaluate_lr(X_test, y_test)
    if lr_results:
        print(f"  LR Test Accuracy: {lr_results['accuracy']:.4f} | F1 macro: {lr_results['f1_macro']:.4f}")
    else:
        print("  LR model not found, skipping.")

    print("\nEvaluating DistilBERT...")
    bert_results = evaluate_bert(X_test, y_test)
    if bert_results:
        print(f"  BERT Test Accuracy: {bert_results['accuracy']:.4f} | F1 macro: {bert_results['f1_macro']:.4f}")
    else:
        print("  DistilBERT model not found, skipping.")

    best = generate_comparison_report(lr_results, bert_results)
    print(f"\n[DONE] Recommended model: {best}")
