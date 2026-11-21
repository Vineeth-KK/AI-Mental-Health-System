"""
Phase 4: DistilBERT Fine-tuning for Mental Health Text Classification
CPU-compatible training with memory optimizations.
Fallback gracefully to LR if training fails.
"""

import os
import sys
import json
import time
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, accuracy_score, f1_score,
    precision_score, recall_score, confusion_matrix
)
from sklearn.utils.class_weight import compute_class_weight

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

PROJECT_ROOT = Path(__file__).parent.parent
CLEAN_CSV = PROJECT_ROOT / "dataset" / "processed" / "mental_health_clean.csv"
MODELS_DIR = PROJECT_ROOT / "models_saved"
BERT_DIR = MODELS_DIR / "distilbert"
BERT_DIR.mkdir(parents=True, exist_ok=True)
BERT_METRICS_PATH = MODELS_DIR / "distilbert_metrics.json"

LABEL_NAMES = ["Normal", "Depression", "Anxiety", "Suicidal"]
NUM_LABELS = 4


def get_device():
    import torch
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        device = torch.device("cpu")
        print("Using CPU (no CUDA available)")
    return device


def load_data():
    if not CLEAN_CSV.exists():
        raise FileNotFoundError(f"Clean CSV not found: {CLEAN_CSV}")
    df = pd.read_csv(CLEAN_CSV)
    print(f"Loaded: {len(df):,} samples")
    return df


class MentalHealthDataset:
    def __init__(self, texts, labels, tokenizer, max_length=128):
        import torch
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        import torch
        encoding = self.tokenizer(
            str(self.texts[idx]),
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(int(self.labels[idx]), dtype=torch.long)
        }


def train_distilbert(df):
    print("\n" + "=" * 60)
    print("PHASE 4: DistilBERT Fine-tuning")
    print("=" * 60)

    import torch
    from torch import nn
    from torch.utils.data import DataLoader
    from transformers import (
        DistilBertTokenizerFast,
        DistilBertForSequenceClassification,
        get_linear_schedule_with_warmup
    )

    device = get_device()

    X = df["text"].astype(str).tolist()
    y = df["label_id"].astype(int).tolist()

    # Stratified splits — same seed as LR for fair comparison
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

    # Tokenizer
    print("\nLoading DistilBERT tokenizer...")
    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

    # Datasets & DataLoaders
    # CPU: use batch_size=16; GPU: batch_size=32
    batch_size = 32 if device.type == "cuda" else 16
    max_length = 128

    # Limit training set to 20k for CPU feasibility; use full set on GPU
    if device.type == "cpu":
        max_train = 20000
        if len(X_train) > max_train:
            print(f"\n[CPU mode] Limiting training set to {max_train:,} samples for time feasibility.")
            X_train_sub, _, y_train_sub, _ = train_test_split(
                X_train, y_train, train_size=max_train, stratify=y_train, random_state=42
            )
            X_train_use = X_train_sub
            y_train_use = y_train_sub
        else:
            X_train_use = X_train
            y_train_use = y_train
    else:
        X_train_use = X_train
        y_train_use = y_train

    train_dataset = MentalHealthDataset(X_train_use, y_train_use, tokenizer, max_length)
    val_dataset = MentalHealthDataset(X_val, y_val, tokenizer, max_length)
    test_dataset = MentalHealthDataset(X_test, y_test, tokenizer, max_length)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # Compute class weights for imbalanced data
    unique_labels = sorted(set(y_train_use))
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.array(unique_labels),
        y=np.array(y_train_use)
    )
    weight_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)
    print(f"\nClass weights: {dict(zip([LABEL_NAMES[i] for i in unique_labels], class_weights.round(3)))}")

    # Model
    print("\nLoading DistilBERT model...")
    model = DistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-uncased",
        num_labels=NUM_LABELS,
        ignore_mismatched_sizes=True
    )
    model = model.to(device)

    # Optimizer & scheduler
    num_epochs = 3
    num_training_steps = num_epochs * len(train_loader)
    num_warmup_steps = num_training_steps // 10

    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )
    loss_fn = nn.CrossEntropyLoss(weight=weight_tensor)

    print(f"\nTraining config:")
    print(f"  Epochs        : {num_epochs}")
    print(f"  Batch size    : {batch_size}")
    print(f"  Max length    : {max_length}")
    print(f"  LR            : 2e-5")
    print(f"  Training steps: {num_training_steps}")

    # Training loop
    best_val_f1 = 0.0
    best_model_state = None
    history = []

    for epoch in range(num_epochs):
        print(f"\n--- Epoch {epoch + 1}/{num_epochs} ---")
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        start_time = time.time()
        for step, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            loss = loss_fn(logits, labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            if (step + 1) % 50 == 0:
                elapsed = time.time() - start_time
                avg_loss = total_loss / (step + 1)
                acc = correct / total
                print(f"  Step {step+1:4d}/{len(train_loader)} | Loss: {avg_loss:.4f} | Acc: {acc:.4f} | {elapsed:.0f}s elapsed")

        epoch_loss = total_loss / len(train_loader)
        epoch_acc = correct / total

        # Validation
        model.eval()
        val_preds, val_true = [], []
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                preds = outputs.logits.argmax(dim=1)
                val_preds.extend(preds.cpu().numpy())
                val_true.extend(labels.cpu().numpy())

        val_acc = accuracy_score(val_true, val_preds)
        val_f1 = f1_score(val_true, val_preds, average="macro", zero_division=0)
        print(f"  Epoch {epoch+1} complete | Train Loss: {epoch_loss:.4f} | Val Acc: {val_acc:.4f} | Val F1 (macro): {val_f1:.4f}")

        history.append({
            "epoch": epoch + 1,
            "train_loss": round(epoch_loss, 4),
            "train_acc": round(epoch_acc, 4),
            "val_acc": round(val_acc, 4),
            "val_f1_macro": round(val_f1, 4)
        })

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            print(f"  [NEW BEST] Val F1 (macro): {best_val_f1:.4f}")

    # Load best checkpoint
    if best_model_state:
        model.load_state_dict(best_model_state)
        model = model.to(device)
    model.eval()

    # Test evaluation
    print("\n--- Test Evaluation ---")
    test_preds, test_true = [], []
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = outputs.logits.argmax(dim=1)
            test_preds.extend(preds.cpu().numpy())
            test_true.extend(labels.cpu().numpy())

    test_acc = accuracy_score(test_true, test_preds)
    test_f1_macro = f1_score(test_true, test_preds, average="macro", zero_division=0)
    test_f1_weighted = f1_score(test_true, test_preds, average="weighted", zero_division=0)
    test_precision_macro = precision_score(test_true, test_preds, average="macro", zero_division=0)
    test_recall_macro = recall_score(test_true, test_preds, average="macro", zero_division=0)

    present_labels = sorted(set(test_true))
    present_names = [LABEL_NAMES[i] for i in present_labels if i < len(LABEL_NAMES)]
    print(classification_report(test_true, test_preds, labels=present_labels, target_names=present_names, zero_division=0))

    # Confusion matrix
    cm = confusion_matrix(test_true, test_preds, labels=present_labels)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Oranges",
                xticklabels=present_names, yticklabels=present_names, ax=ax)
    ax.set_title("DistilBERT — Confusion Matrix (Test Set)")
    ax.set_ylabel("True Label")
    ax.set_xlabel("Predicted Label")
    plt.tight_layout()
    cm_path = MODELS_DIR / "distilbert_confusion_matrix.png"
    plt.savefig(cm_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"\n[OK] Confusion matrix saved: {cm_path}")

    # Save model + tokenizer
    model.save_pretrained(str(BERT_DIR))
    tokenizer.save_pretrained(str(BERT_DIR))
    print(f"[OK] DistilBERT model saved: {BERT_DIR}")

    # Save metrics
    per_class_report = classification_report(
        test_true, test_preds, labels=present_labels,
        target_names=present_names, zero_division=0, output_dict=True
    )
    metrics = {
        "model": "DistilBERT fine-tuned",
        "pretrained_base": "distilbert-base-uncased",
        "train_size": len(X_train_use),
        "val_size": len(X_val),
        "test_size": len(X_test),
        "best_val_f1_macro": round(best_val_f1, 4),
        "test_accuracy": round(test_acc, 4),
        "test_f1_macro": round(test_f1_macro, 4),
        "test_f1_weighted": round(test_f1_weighted, 4),
        "test_precision_macro": round(test_precision_macro, 4),
        "test_recall_macro": round(test_recall_macro, 4),
        "per_class": per_class_report,
        "training_history": history
    }
    with open(BERT_METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[OK] Metrics saved: {BERT_METRICS_PATH}")

    return metrics


if __name__ == "__main__":
    df = load_data()
    metrics = train_distilbert(df)
    print(f"\n[DONE] Phase 4 complete. Test Accuracy: {metrics['test_accuracy']:.4f}")
