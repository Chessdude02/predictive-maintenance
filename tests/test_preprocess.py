import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from src.pipeline.preprocess import (
    load_raw_file,
    extract_features,
    assign_label,
    build_feature_dataset
)


def test_load_raw_file(tmp_path):
    """Test that loader produces correct shape and column names."""
    # Create a fake data file
    fake_data = "\t".join(["0.1", "0.2", "0.3", "0.4"]) + "\n"
    fake_data = fake_data * 20480
    fake_file = tmp_path / "test_file"
    fake_file.write_text(fake_data)

    df = load_raw_file(fake_file)
    assert df.shape == (20480, 4)
    assert list(df.columns) == ["bearing_1", "bearing_2", "bearing_3", "bearing_4"]


def test_extract_features_keys():
    """Test that feature extractor returns all expected keys."""
    df = pd.DataFrame(
        np.random.randn(20480, 4),
        columns=["bearing_1", "bearing_2", "bearing_3", "bearing_4"]
    )
    features = extract_features(df)
    expected_suffixes = ["rms", "mean", "std", "peak", "kurtosis", "skewness"]
    bearings = ["bearing_1", "bearing_2", "bearing_3", "bearing_4"]

    for b in bearings:
        for s in expected_suffixes:
            assert f"{b}_{s}" in features, f"Missing feature: {b}_{s}"


def test_extract_features_rms():
    """Test RMS calculation is correct."""
    df = pd.DataFrame(
        np.ones((20480, 4)),
        columns=["bearing_1", "bearing_2", "bearing_3", "bearing_4"]
    )
    features = extract_features(df)
    assert abs(features["bearing_1_rms"] - 1.0) < 1e-6


def test_assign_label_healthy():
    assert assign_label(0, 100) == 0
    assert assign_label(69, 100) == 0


def test_assign_label_failure():
    assert assign_label(90, 100) == 1
    assert assign_label(99, 100) == 1


def test_assign_label_excluded():
    assert assign_label(75, 100) == -1
    assert assign_label(89, 100) == -1