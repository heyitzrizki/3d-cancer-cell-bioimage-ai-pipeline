"""Tests for Milestone 3 cell-level feature extraction."""

import numpy as np
import pandas as pd
import pytest

from bioimage_pipeline.features import (
    extract_region_features,
    summarize_feature_table,
)


def test_extract_region_features_from_2d_mask() -> None:
    labels = np.zeros((6, 6), dtype=np.int32)
    labels[1:3, 1:3] = 1
    labels[4:6, 2:5] = 2

    features = extract_region_features(
        labels,
        metadata={"dataset": "toy", "sequence": "01", "method": "test"},
    )

    assert len(features) == 2
    assert {"centroid_y", "centroid_x", "bbox_min_y", "bbox_max_x"} <= set(
        features.columns
    )
    assert "centroid_z" not in features.columns
    assert features["area"].tolist() == [4.0, 6.0]
    assert features["dataset"].tolist() == ["toy", "toy"]


def test_extract_region_features_from_3d_mask() -> None:
    labels = np.zeros((4, 5, 6), dtype=np.int32)
    labels[1:3, 1:4, 2:5] = 1

    features = extract_region_features(labels)

    assert len(features) == 1
    assert features.loc[0, "area"] == 18.0
    assert {"centroid_z", "centroid_y", "centroid_x"} <= set(features.columns)
    assert {"bbox_min_z", "bbox_max_z"} <= set(features.columns)


def test_intensity_features_are_calculated() -> None:
    labels = np.zeros((4, 4), dtype=np.int32)
    labels[1:3, 1:3] = 1
    intensity = np.zeros((4, 4), dtype=np.float32)
    intensity[1:3, 1:3] = np.array([[1, 2], [3, 4]])

    features = extract_region_features(labels, intensity)

    assert features.loc[0, "mean_intensity"] == 2.5
    assert features.loc[0, "min_intensity"] == 1.0
    assert features.loc[0, "max_intensity"] == 4.0
    assert features.loc[0, "integrated_intensity"] == 10.0


def test_intensity_shape_mismatch_raises_clear_error() -> None:
    labels = np.zeros((4, 4), dtype=np.int32)
    intensity = np.zeros((3, 4), dtype=np.float32)

    with pytest.raises(ValueError, match="must have the same shape"):
        extract_region_features(labels, intensity)


def test_empty_mask_returns_expected_columns() -> None:
    labels = np.zeros((3, 4, 5), dtype=np.int32)

    features = extract_region_features(labels)

    assert features.empty
    assert {
        "dataset",
        "sequence",
        "frame",
        "label",
        "area",
        "centroid_z",
        "centroid_y",
        "centroid_x",
        "bbox_min_z",
        "bbox_max_x",
    } <= set(features.columns)


def test_summarize_feature_table_returns_expected_columns() -> None:
    features = pd.DataFrame(
        {
            "label": [1, 2],
            "area": [4.0, 6.0],
            "mean_intensity": [2.0, 4.0],
            "frame": ["000", "000"],
        }
    )

    summary = summarize_feature_table(features)

    assert summary.columns.tolist() == [
        "feature",
        "count",
        "mean",
        "std",
        "min",
        "median",
        "max",
    ]
    area_summary = summary.loc[summary["feature"] == "area"].iloc[0]
    assert area_summary["count"] == 2
    assert area_summary["mean"] == 5.0
    assert area_summary["median"] == 5.0
