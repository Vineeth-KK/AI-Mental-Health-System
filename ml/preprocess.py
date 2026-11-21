"""
Phase 2: Data Cleaning and Preprocessing
Cleans raw dataset and prepares it for model training.
"""

import os
import re
import json
import pandas as pd
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
RAW_CSV = PROJECT_ROOT / "dataset" / "raw" / "mental_health_raw.csv"
PROCESSED_DIR = PROJECT_ROOT / "dataset" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
CLEAN_CSV = PROCESSED_DIR / "mental_health_clean.csv"
LABEL_MAP_PATH = PROCESSED_DIR / "label_map.json"
COL_MAP_PATH = PROCESSED_DIR / "column_map.json"

# Target label normalization map
LABEL_NORMALIZE = {
    # Normal variants
    "normal": "Normal",
    "healthy": "Normal",
    "not depressed": "Normal",
    "no disorder": "Normal",

    # Depression variants
    "depression": "Depression",
    "depressed": "Depression",
    "depressive": "Depression",
    "major depression": "Depression",

    # Anxiety variants
    "anxiety": "Anxiety",
    "anxious": "Anxiety",
    "anxiety disorder": "Anxiety",
    "general anxiety": "Anxiety",

    # Suicidal variants
    "suicidal": "Suicidal",
    "suicide": "Suicidal",
    "suicidality": "Suicidal",
    "self-harm": "Suicidal",
}

LABEL_ENCODING = {
    "Normal": 0,
    "Depression": 1,
    "Anxiety": 2,
    "Suicidal": 3,
}


def clean_text(text):
    """
    Minimal but effective text cleaning.
    Preserves emotionally meaningful content, negations, punctuation.
    """
    if not isinstance(text, str):
        return ""

    # Remove HTML tags
    text = re.sub(r"<[^>]+>", " ", text)

    # Remove URLs
    text = re.sub(r"http\S+|www\.\S+", "", text)

    # Remove excess whitespace but preserve sentence structure
    text = re.sub(r"\s+", " ", text).strip()

    # Remove very short fragments (likely noise)
    if len(text) < 5:
        return ""

    return text


def normalize_label(label):
    """Normalize label to canonical form."""
    if not isinstance(label, str):
        return None
    normalized = label.strip().lower()
    return LABEL_NORMALIZE.get(normalized, label.strip().title())


def run_preprocessing():
    print("=" * 60)
    print("PHASE 2: Data Cleaning and Preprocessing")
    print("=" * 60)

    if not RAW_CSV.exists():
        print(f"[ERROR] Raw CSV not found: {RAW_CSV}")
        print("        Run dataset_explore.py first.")
        return None

    # Load column map
    if COL_MAP_PATH.exists():
        with open(COL_MAP_PATH) as f:
            col_map = json.load(f)
        text_col = col_map.get("text_column")
        label_col = col_map.get("label_column")
    else:
        text_col = None
        label_col = None

    # Load raw data
    df = pd.read_csv(RAW_CSV)
    print(f"\nLoaded raw dataset: {len(df):,} rows, {len(df.columns)} columns")
    print(f"Columns: {list(df.columns)}")

    # Auto-detect columns if not mapped
    if not text_col or text_col not in df.columns:
        for col in df.columns:
            col_lower = col.lower()
            if any(x in col_lower for x in ["text", "statement", "content", "message", "post"]):
                text_col = col
                break
        if not text_col:
            # Find the longest string column
            str_cols = df.select_dtypes(include="object").columns
            if len(str_cols) > 0:
                text_col = max(str_cols, key=lambda c: df[c].fillna("").str.len().mean())

    if not label_col or label_col not in df.columns:
        for col in df.columns:
            col_lower = col.lower()
            if any(x in col_lower for x in ["label", "status", "category", "class", "target"]):
                label_col = col
                break

    print(f"\nUsing text column  : '{text_col}'")
    print(f"Using label column : '{label_col}'")

    # Step 1: Drop rows with missing text or label
    before = len(df)
    df = df.dropna(subset=[text_col, label_col])
    print(f"\n[Step 1] Dropped NaN rows: {before - len(df):,} removed | {len(df):,} remaining")

    # Step 2: Clean text
    df["text_clean"] = df[text_col].apply(clean_text)

    # Drop rows where cleaning produced empty text
    before = len(df)
    df = df[df["text_clean"].str.len() > 0]
    print(f"[Step 2] Empty text after cleaning: {before - len(df):,} removed | {len(df):,} remaining")

    # Step 3: Normalize labels
    df["label_norm"] = df[label_col].apply(normalize_label)
    valid_labels = set(LABEL_ENCODING.keys())
    before = len(df)
    df = df[df["label_norm"].isin(valid_labels)]
    print(f"[Step 3] Invalid labels removed: {before - len(df):,} | {len(df):,} remaining")
    print(f"         Label distribution after normalization:")
    for lbl, cnt in df["label_norm"].value_counts().items():
        print(f"         {lbl}: {cnt:,} ({100*cnt/len(df):.1f}%)")

    # Step 4: Remove exact duplicates (same text + label)
    before = len(df)
    df = df.drop_duplicates(subset=["text_clean", "label_norm"])
    print(f"[Step 4] Exact duplicates removed: {before - len(df):,} | {len(df):,} remaining")

    # Step 5: Encode labels
    df["label_encoded"] = df["label_norm"].map(LABEL_ENCODING)

    # Step 6: Select final columns
    df_final = df[["text_clean", "label_norm", "label_encoded"]].copy()
    df_final.columns = ["text", "label", "label_id"]
    df_final = df_final.reset_index(drop=True)

    # Save
    df_final.to_csv(CLEAN_CSV, index=False)
    print(f"\n[OK] Clean dataset saved: {CLEAN_CSV}")
    print(f"     Final shape: {df_final.shape}")

    # Save label map
    with open(LABEL_MAP_PATH, "w") as f:
        json.dump(LABEL_ENCODING, f, indent=2)
    print(f"[OK] Label map saved: {LABEL_MAP_PATH}")

    # Save reverse map
    reverse_map = {v: k for k, v in LABEL_ENCODING.items()}
    with open(PROCESSED_DIR / "label_map_reverse.json", "w") as f:
        json.dump(reverse_map, f, indent=2)

    # Final summary
    print(f"\n--- Final Dataset Summary ---")
    print(f"Total samples  : {len(df_final):,}")
    print(f"Label encoding : {LABEL_ENCODING}")
    print(f"Label distribution:")
    for lbl, cnt in df_final["label"].value_counts().items():
        print(f"  {lbl}: {cnt:,} ({100*cnt/len(df_final):.1f}%)")

    return df_final


if __name__ == "__main__":
    df = run_preprocessing()
    print("\n[DONE] Phase 2 complete.")
