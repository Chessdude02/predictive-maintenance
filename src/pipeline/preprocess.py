import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple


def load_raw_file(filepath: Path) -> pd.DataFrame:
    """Load a single raw bearing data file."""
    df = pd.read_csv(filepath, sep="\t", header=None)
    df.columns = ["bearing_1", "bearing_2", "bearing_3", "bearing_4"]
    return df


def extract_features(df: pd.DataFrame) -> dict:
    """
    Extract time-domain features from a single file.
    Each file = one time window (20480 samples).
    We reduce it to a feature vector.
    """
    features = {}
    for col in df.columns:
        x = df[col].values
        features[f"{col}_rms"] = np.sqrt(np.mean(x**2))
        features[f"{col}_mean"] = np.mean(x)
        features[f"{col}_std"] = np.std(x)
        features[f"{col}_peak"] = np.max(np.abs(x))
        features[f"{col}_kurtosis"] = pd.Series(x).kurtosis()
        features[f"{col}_skewness"] = pd.Series(x).skew()
    return features


def assign_label(file_index: int, total_files: int) -> int:
    """
    Assign binary label based on file position.
    0 = healthy, 1 = failure
    Excludes ambiguous degradation zone (70-90% of run).
    Returns -1 for excluded zone.
    """
    pct = file_index / total_files
    if pct <= 0.70:
        return 0  # healthy
    elif pct >= 0.90:
        return 1  # failure
    else:
        return -1  # exclude


def build_feature_dataset(data_dir: Path) -> pd.DataFrame:
    """
    Process all raw files into a feature matrix with labels.
    Excludes ambiguous degradation zone.
    """
    files = sorted(data_dir.iterdir())
    total = len(files)
    records = []

    for i, f in enumerate(files):
        label = assign_label(i, total)
        if label == -1:
            continue
        df = load_raw_file(f)
        features = extract_features(df)
        features["label"] = label
        features["timestamp"] = f.name
        records.append(features)

    return pd.DataFrame(records)

if __name__ == "__main__":
    DATA_DIR = Path("data/raw")
    OUTPUT_DIR = Path("data/processed")
    OUTPUT_DIR.mkdir(exist_ok=True)

    print("Building feature dataset...")
    dataset = build_feature_dataset(DATA_DIR)

    print(f"Dataset shape: {dataset.shape}")
    print(f"Label distribution:\n{dataset['label'].value_counts()}")

    dataset.to_csv(OUTPUT_DIR / "features.csv", index=False)
    print("Saved to data/processed/features.csv")