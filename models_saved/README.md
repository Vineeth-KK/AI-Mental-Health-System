# CareAI ML Models — README

## Overview
This directory contains trained ML models for mental health text classification.
These models classify text into 4 categories:

| ID | Label | Description |
|----|-------|-------------|
| 0  | Normal | No significant mental health concerns |
| 1  | Depression | Depressive symptom indicators |
| 2  | Anxiety | Anxiety disorder indicators |
| 3  | Suicidal | Suicidal ideation indicators |

> **IMPORTANT DISCLAIMER:** These models produce supportive analytical signals only.
> They are NOT clinical diagnoses and must not be used as the sole basis for
> any clinical or safety decision. All safety decisions are handled by the
> Safety Agent (agents/safety.py) independently using rule-based logic.

---

## Files

| File | Description |
|------|-------------|
| `lr_model.pkl` | Trained Logistic Regression classifier |
| `tfidf_vectorizer.pkl` | Fitted TF-IDF vectorizer (50k features, bigrams) |
| `lr_metrics.json` | LR evaluation metrics on held-out test set |
| `distilbert/` | Fine-tuned DistilBERT model (weights + tokenizer) |
| `distilbert_metrics.json` | DistilBERT evaluation metrics |
| `lr_confusion_matrix.png` | LR confusion matrix visualization |
| `distilbert_confusion_matrix.png` | DistilBERT confusion matrix visualization |
| `model_comparison_report.md` | Head-to-head comparison of both models |

---

## Usage

### Python API (via ml/inference.py)

```python
from ml.inference import classify_text, is_ml_available, get_model_info

# Check if any model is available
if is_ml_available():
    result = classify_text("I feel hopeless and don't want to live")
    print(result)
    # {
    #   "label": "Suicidal",
    #   "confidence": 0.92,
    #   "all_scores": {"Normal": 0.02, "Depression": 0.04, "Anxiety": 0.02, "Suicidal": 0.92},
    #   "model_used": "distilbert",
    #   "disclaimer": "This is a supportive ML signal only..."
    # }

# Get model info
info = get_model_info()
print(info)  # {"active_model": "distilbert", "test_accuracy": 0.93, ...}
```

### Model Loading (manual)

#### Logistic Regression
```python
import pickle

with open("models_saved/tfidf_vectorizer.pkl", "rb") as f:
    tfidf = pickle.load(f)
with open("models_saved/lr_model.pkl", "rb") as f:
    lr = pickle.load(f)

X = tfidf.transform(["I feel very sad and hopeless"])
pred = lr.predict(X)[0]
proba = lr.predict_proba(X)[0]
# pred: 1 (Depression), proba: [0.05, 0.85, 0.07, 0.03]
```

#### DistilBERT
```python
import torch
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification

tokenizer = DistilBertTokenizerFast.from_pretrained("models_saved/distilbert")
model = DistilBertForSequenceClassification.from_pretrained("models_saved/distilbert")
model.eval()

inputs = tokenizer("I feel very sad", return_tensors="pt", truncation=True, max_length=128)
with torch.no_grad():
    logits = model(**inputs).logits
    probs = torch.softmax(logits, dim=1)
    pred = probs.argmax().item()
```

---

## Reproducing Training

```bash
# 1. Download and explore dataset
python ml/dataset_explore.py

# 2. Clean and preprocess
python ml/preprocess.py

# 3. Train Logistic Regression
python ml/train_lr.py

# 4. Fine-tune DistilBERT (time-intensive, ~30-120 min on CPU)
python ml/train_bert.py

# 5. Evaluate and compare
python ml/evaluate.py
```

---

## Dataset
- Source: https://huggingface.co/datasets/nayma11/Mental-Health_Text-Classification_Dataset
- 4 classes: Normal, Depression, Anxiety, Suicidal
- Cleaned data: `dataset/processed/mental_health_clean.csv`
- Label map: `dataset/processed/label_map.json`
