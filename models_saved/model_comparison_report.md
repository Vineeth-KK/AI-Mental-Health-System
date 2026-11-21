# CareAI ML — Model Comparison Report

## Overview
This report compares two text classification models trained on the
nayma11/Mental-Health_Text-Classification_Dataset (4 classes: Normal, Depression, Anxiety, Suicidal).

**Important disclaimer:** These ML predictions are supportive analytical signals only.
They are NOT clinical diagnoses and must not override the existing Safety Agent rules.

## Model Summary

| Metric | Logistic Regression | DistilBERT |
|--------|--------------------:|----------:|
| Accuracy | 0.7918 | 0.8269 |
| F1 (Macro) | 0.7741 | 0.8163 |
| F1 (Weighted) | 0.789 | 0.8266 |
| Precision (Macro) | 0.7676 | 0.8098 |
| Recall (Macro) | 0.7859 | 0.826 |

## Recommended Model: **DistilBERT**
DistilBERT achieves F1 macro 0.8163 vs LR 0.7741

## Integration Notes
- The inference module (`ml/inference.py`) uses the best model automatically.
- Fallback: if DistilBERT fails to load, Logistic Regression is used.
- ML classification result is added as `ml_classification` field in agent state.
- Safety Agent continues to apply its own rules independently.

## Per-Class Performance

### Logistic Regression
- **Normal**: P=0.888 R=0.933 F1=0.910 (n=2723.0)
- **Depression**: P=0.763 R=0.644 F1=0.699 (n=2141.0)
- **Anxiety**: P=0.714 R=0.837 F1=0.771 (n=799.0)
- **Suicidal**: P=0.706 R=0.729 F1=0.717 (n=1680.0)

### DistilBERT
- **Normal**: P=0.940 R=0.932 F1=0.936 (n=2723.0)
- **Depression**: P=0.791 R=0.705 F1=0.746 (n=2141.0)
- **Anxiety**: P=0.794 R=0.881 F1=0.835 (n=799.0)
- **Suicidal**: P=0.714 R=0.786 F1=0.748 (n=1680.0)

## Files
- `lr_model.pkl` — Logistic Regression model
- `tfidf_vectorizer.pkl` — TF-IDF vectorizer
- `lr_metrics.json` — LR evaluation metrics
- `distilbert/` — DistilBERT model weights + tokenizer
- `distilbert_metrics.json` — DistilBERT evaluation metrics
- `lr_confusion_matrix.png` — LR confusion matrix
- `distilbert_confusion_matrix.png` — DistilBERT confusion matrix

## Usage
```python
from ml.inference import classify_text, is_ml_available

if is_ml_available():
    result = classify_text("I feel hopeless and don't want to live")
    print(result)  # {"label": "Suicidal", "confidence": 0.92, ...}
```