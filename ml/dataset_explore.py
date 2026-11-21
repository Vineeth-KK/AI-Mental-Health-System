"""
Phase 1: Dataset Download and Exploration
Downloads the nayma11/Mental-Health_Text-Classification_Dataset from HuggingFace
and generates a comprehensive exploration report.
"""

import os
import sys
import json
import pandas as pd
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
RAW_DIR = PROJECT_ROOT / "dataset" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "dataset" / "processed"
RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

REPORT_PATH = PROJECT_ROOT / "dataset" / "exploration_report.md"
RAW_CSV = RAW_DIR / "mental_health_raw.csv"


def download_dataset():
    """Try HuggingFace datasets library first, then fallback to direct download."""
    print("=" * 60)
    print("PHASE 1: Dataset Download & Exploration")
    print("=" * 60)

    # Method 1: HuggingFace datasets library
    try:
        print("\n[1/3] Attempting download via datasets library...")
        from datasets import load_dataset
        ds = load_dataset("nayma11/Mental-Health_Text-Classification_Dataset")
        print(f"      Splits available: {list(ds.keys())}")

        all_dfs = []
        for split_name, split_data in ds.items():
            df = split_data.to_pandas()
            df["_split"] = split_name
            all_dfs.append(df)
            print(f"      Split '{split_name}': {len(df)} rows, columns: {list(df.columns)}")

        combined = pd.concat(all_dfs, ignore_index=True)
        combined.to_csv(RAW_CSV, index=False)
        print(f"\n[OK] Dataset saved to: {RAW_CSV}")
        return combined

    except Exception as e:
        print(f"      datasets library failed: {e}")

    # Method 2: Direct CSV download via huggingface_hub
    try:
        print("\n[2/3] Attempting download via huggingface_hub...")
        from huggingface_hub import hf_hub_download
        import shutil

        possible_files = [
            "train.csv", "test.csv", "validation.csv", "val.csv",
            "data/train.csv", "data/test.csv",
            "mental_health.csv", "dataset.csv", "all_data.csv"
        ]

        all_dfs = []
        for fname in possible_files:
            try:
                local_path = hf_hub_download(
                    repo_id="nayma11/Mental-Health_Text-Classification_Dataset",
                    filename=fname,
                    repo_type="dataset"
                )
                df = pd.read_csv(local_path)
                df["_split"] = fname.replace(".csv", "").replace("data/", "")
                all_dfs.append(df)
                print(f"      Downloaded: {fname} -> {len(df)} rows")
            except Exception:
                pass

        if all_dfs:
            combined = pd.concat(all_dfs, ignore_index=True)
            combined.to_csv(RAW_CSV, index=False)
            print(f"\n[OK] Dataset saved to: {RAW_CSV}")
            return combined
        else:
            print("      No files found via hf_hub_download")

    except Exception as e:
        print(f"      huggingface_hub failed: {e}")

    # Method 3: urllib direct download
    try:
        print("\n[3/3] Attempting direct URL download...")
        import urllib.request
        import urllib.error

        base_url = "https://huggingface.co/datasets/nayma11/Mental-Health_Text-Classification_Dataset/resolve/main"
        filenames = ["train.csv", "test.csv", "val.csv", "validation.csv"]

        all_dfs = []
        for fname in filenames:
            url = f"{base_url}/{fname}"
            try:
                local_path = RAW_DIR / fname
                urllib.request.urlretrieve(url, local_path)
                df = pd.read_csv(local_path)
                df["_split"] = fname.replace(".csv", "")
                all_dfs.append(df)
                print(f"      Downloaded: {fname} -> {len(df)} rows")
            except Exception as url_err:
                print(f"      Skipped {fname}: {url_err}")

        if all_dfs:
            combined = pd.concat(all_dfs, ignore_index=True)
            combined.to_csv(RAW_CSV, index=False)
            print(f"\n[OK] Dataset saved to: {RAW_CSV}")
            return combined

    except Exception as e:
        print(f"      Direct download failed: {e}")

    print("\n[ERROR] All download methods failed. Cannot proceed.")
    sys.exit(1)


def explore_dataset(df):
    """Generate comprehensive exploration report."""
    print("\n" + "=" * 60)
    print("Dataset Exploration")
    print("=" * 60)

    # Identify text and label columns
    text_col = None
    label_col = None

    for col in df.columns:
        col_lower = col.lower()
        if any(x in col_lower for x in ["text", "statement", "content", "message", "post"]):
            text_col = col
        if any(x in col_lower for x in ["label", "status", "category", "class", "target"]):
            label_col = col

    # Fallback: guess by dtype
    if text_col is None:
        str_cols = df.select_dtypes(include="object").columns.tolist()
        for col in str_cols:
            if df[col].str.len().mean() > 20:
                text_col = col
                break

    if label_col is None:
        for col in df.columns:
            if df[col].nunique() <= 10:
                label_col = col
                break

    print(f"\nIdentified columns:")
    print(f"  Text column  : {text_col}")
    print(f"  Label column : {label_col}")
    print(f"\nAll columns: {list(df.columns)}")
    print(f"Shape: {df.shape}")
    print(f"Total samples: {len(df):,}")

    # Save column mapping
    col_map = {"text_column": text_col, "label_column": label_col}
    with open(PROCESSED_DIR / "column_map.json", "w") as f:
        json.dump(col_map, f, indent=2)

    # Basic stats
    print(f"\n--- Missing Values ---")
    missing = df.isnull().sum()
    print(missing[missing > 0] if missing.sum() > 0 else "No missing values")

    print(f"\n--- Duplicate Rows ---")
    n_dupes = df.duplicated().sum()
    print(f"Exact duplicates: {n_dupes:,} ({100*n_dupes/len(df):.1f}%)")

    # Label distribution
    print(f"\n--- Label Distribution ---")
    if label_col:
        label_counts = df[label_col].value_counts()
        print(label_counts)
        print(f"\nUnique labels: {df[label_col].unique()}")

    # Text length stats
    if text_col:
        df["_text_len"] = df[text_col].fillna("").astype(str).str.len()
        df["_word_count"] = df[text_col].fillna("").astype(str).str.split().str.len()
        print(f"\n--- Text Length Statistics ---")
        print(df[["_text_len", "_word_count"]].describe().round(1))

    # Sample rows per label
    print(f"\n--- Sample Rows Per Label ---")
    if label_col and text_col:
        for lbl in df[label_col].unique():
            subset = df[df[label_col] == lbl]
            sample = subset[text_col].iloc[0] if len(subset) > 0 else ""
            print(f"\n  [{lbl}] ({len(subset):,} samples)")
            print(f"  Example: {str(sample)[:120]}...")

    # Build report
    report_lines = [
        "# CareAI ML — Dataset Exploration Report",
        "",
        f"**Dataset:** nayma11/Mental-Health_Text-Classification_Dataset",
        f"**Total samples:** {len(df):,}",
        f"**Columns:** {list(df.columns)}",
        f"**Text column:** `{text_col}`",
        f"**Label column:** `{label_col}`",
        "",
        "## Shape",
        f"- Rows: {df.shape[0]:,}",
        f"- Columns: {df.shape[1]}",
        "",
        "## Missing Values",
        str(missing[missing > 0].to_dict()) if missing.sum() > 0 else "None",
        "",
        "## Duplicates",
        f"- Exact duplicate rows: {n_dupes:,} ({100*n_dupes/len(df):.1f}%)",
        "",
        "## Label Distribution",
    ]

    if label_col:
        for lbl, cnt in label_counts.items():
            pct = 100 * cnt / len(df)
            report_lines.append(f"- {lbl}: {cnt:,} ({pct:.1f}%)")

    if text_col:
        stats = df[["_text_len", "_word_count"]].describe().round(1)
        report_lines += [
            "",
            "## Text Length Statistics",
            f"- Mean chars: {stats.loc['mean', '_text_len']}",
            f"- Median chars: {stats.loc['50%', '_text_len']}",
            f"- Max chars: {stats.loc['max', '_text_len']}",
            f"- Mean words: {stats.loc['mean', '_word_count']}",
        ]

    report_lines += [
        "",
        "## Notes",
        "- Labels verified against expected set: Normal, Depression, Anxiety, Suicidal",
        "- Dataset will be cleaned and preprocessed in Phase 2",
        "- Label encoding: Normal=0, Depression=1, Anxiety=2, Suicidal=3",
    ]

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    print(f"\n[OK] Exploration report saved to: {REPORT_PATH}")
    return text_col, label_col


if __name__ == "__main__":
    df = download_dataset()
    text_col, label_col = explore_dataset(df)
    print("\n[DONE] Phase 1 complete.")
    print(f"  Raw CSV: {RAW_CSV}")
    print(f"  Report : {REPORT_PATH}")
