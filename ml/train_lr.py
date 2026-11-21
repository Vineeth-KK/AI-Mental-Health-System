"""
Phase 3: Logistic Regression Training with TF-IDF
Trains a TF-IDF + Logistic Regression classifier on the cleaned mental health dataset.
"""

import os
import json
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    classification_report, accuracy_score, f1_score,
    precision_score, recall_score, confusion_matrix
)
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

PROJECT_ROOT = Path(__file__).parent.parent
CLEAN_CSV = PROJECT_ROOT / "dataset" / "processed" / "mental_health_clean.csv"
MODELS_DIR = PROJECT_ROOT / "models_saved"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

LR_MODEL_PATH = MODELS_DIR / "lr_model.pkl"
TFIDF_PATH = MODELS_DIR / "tfidf_vectorizer.pkl"
LR_METRICS_PATH = MODELS_DIR / "lr_metrics.json"
LABEL_NAMES = ["Normal", "Depression", "Anxiety", "Suicidal"]


def load_data():
    if not CLEAN_CSV.exists():
        raise FileNotFoundError(f"Clean CSV not found: {CLEAN_CSV}. Run preprocess.py first.")
    df = pd.read_csv(CLEAN_CSV)
    print(f"Loaded clean dataset: {len(df):,} samples")
    print(f"Label distribution:\n{df['label'].value_counts()}")
    return df


def train_lr(df):
    print("\n" + "=" * 60)
    print("PHASE 3: Logistic Regression Training")
    print("=" * 60)

    X = df["text"].astype(str)
    y = df["label_id"].astype(int)

    # Stratified split: 70% train, 15% val, 15% test
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=42
    )

    print(f"\nSplit sizes:")
    print(f"  Train : {len(X_train):,}")
    print(f"  Val   : {len(X_val):,}")
    print(f"  Test  : {len(X_test):,}")

    # TF-IDF vectorizer
    print("\nFitting TF-IDF vectorizer...")
    tfidf = TfidfVectorizer(
        max_features=50000,
        ngram_range=(1, 2),
        sublinear_tf=True,
        min_df=2,
        analyzer="word",
        strip_accents="unicode",
        token_pattern=r"(?u)\b[a-zA-Z]\w+\b"
    )
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_val_tfidf = tfidf.transform(X_val)
    X_test_tfidf = tfidf.transform(X_test)

    print(f"  TF-IDF matrix shape: {X_train_tfidf.shape}")

    # Logistic Regression
    print("\nTraining Logistic Regression...")
    lr = LogisticRegression(
        C=1.0,
        max_iter=1000,
        class_weight="balanced",
        solver="lbfgs",
        random_state=42,
        n_jobs=-1
    )
    lr.fit(X_train_tfidf, y_train)
    print("  Training complete.")

    # Evaluate on validation set
    y_val_pred = lr.predict(X_val_tfidf)
    val_acc = accuracy_score(y_val, y_val_pred)
    val_f1_macro = f1_score(y_val, y_val_pred, average="macro")
    val_f1_weighted = f1_score(y_val, y_val_pred, average="weighted")
    print(f"\n--- Validation Results ---")
    print(f"Accuracy    : {val_acc:.4f}")
    print(f"F1 (macro)  : {val_f1_macro:.4f}")
    print(f"F1 (weighted): {val_f1_weighted:.4f}")

    # Evaluate on test set
    y_test_pred = lr.predict(X_test_tfidf)
    test_acc = accuracy_score(y_test, y_test_pred)
    test_f1_macro = f1_score(y_test, y_test_pred, average="macro")
    test_f1_weighted = f1_score(y_test, y_test_pred, average="weighted")
    test_precision_macro = precision_score(y_test, y_test_pred, average="macro", zero_division=0)
    test_recall_macro = recall_score(y_test, y_test_pred, average="macro", zero_division=0)

    print(f"\n--- Test Results ---")
    print(f"Accuracy      : {test_acc:.4f}")
    print(f"F1 (macro)    : {test_f1_macro:.4f}")
    print(f"F1 (weighted) : {test_f1_weighted:.4f}")
    print(f"Precision (macro): {test_precision_macro:.4f}")
    print(f"Recall (macro)   : {test_recall_macro:.4f}")

    # Per-class report
    present_labels = sorted(y_test.unique())
    present_names = [LABEL_NAMES[i] for i in present_labels if i < len(LABEL_NAMES)]
    print(f"\n--- Classification Report (Test) ---")
    print(classification_report(
        y_test, y_test_pred,
        labels=present_labels,
        target_names=present_names,
        zero_division=0
    ))

    # Confusion matrix
    cm = confusion_matrix(y_test, y_test_pred, labels=present_labels)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=present_names, yticklabels=present_names, ax=ax
    )
    ax.set_title("Logistic Regression — Confusion Matrix (Test Set)")
    ax.set_ylabel("True Label")
    ax.set_xlabel("Predicted Label")
    plt.tight_layout()
    cm_path = MODELS_DIR / "lr_confusion_matrix.png"
    plt.savefig(cm_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"\n[OK] Confusion matrix saved: {cm_path}")

    # Save model artifacts
    with open(LR_MODEL_PATH, "wb") as f:
        pickle.dump(lr, f)
    with open(TFIDF_PATH, "wb") as f:
        pickle.dump(tfidf, f)
    print(f"[OK] LR model saved    : {LR_MODEL_PATH}")
    print(f"[OK] TF-IDF saved      : {TFIDF_PATH}")

    # Save metrics JSON
    per_class_report = classification_report(
        y_test, y_test_pred,
        labels=present_labels,
        target_names=present_names,
        zero_division=0,
        output_dict=True
    )
    metrics = {
        "model": "LogisticRegression + TF-IDF",
        "train_size": len(X_train),
        "val_size": len(X_val),
        "test_size": len(X_test),
        "val_accuracy": round(val_acc, 4),
        "val_f1_macro": round(val_f1_macro, 4),
        "val_f1_weighted": round(val_f1_weighted, 4),
        "test_accuracy": round(test_acc, 4),
        "test_f1_macro": round(test_f1_macro, 4),
        "test_f1_weighted": round(test_f1_weighted, 4),
        "test_precision_macro": round(test_precision_macro, 4),
        "test_recall_macro": round(test_recall_macro, 4),
        "per_class": per_class_report
    }
    with open(LR_METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[OK] Metrics saved     : {LR_METRICS_PATH}")

    return metrics


if __name__ == "__main__":
    df = load_data()
    metrics = train_lr(df)
    print(f"\n[DONE] Phase 3 complete. Test Accuracy: {metrics['test_accuracy']:.4f}")
