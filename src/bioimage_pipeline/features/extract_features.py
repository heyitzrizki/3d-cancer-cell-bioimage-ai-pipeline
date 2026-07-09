"""Cell-level morphology and intensity feature extraction."""

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from skimage.measure import label, regionprops_table

from bioimage_pipeline.data import load_tiff_image

METADATA_COLUMNS = (
    "dataset",
    "sequence",
    "frame",
    "source_image_file",
    "mask_file",
    "method",
)
SUMMARY_COLUMNS = ("feature", "count", "mean", "std", "min", "median", "max")


def _coordinate_columns(ndim: int) -> tuple[list[str], list[str]]:
    if ndim == 2:
        centroid_columns = ["centroid_y", "centroid_x"]
        bbox_columns = ["bbox_min_y", "bbox_min_x", "bbox_max_y", "bbox_max_x"]
    elif ndim == 3:
        centroid_columns = ["centroid_z", "centroid_y", "centroid_x"]
        bbox_columns = [
            "bbox_min_z",
            "bbox_min_y",
            "bbox_min_x",
            "bbox_max_z",
            "bbox_max_y",
            "bbox_max_x",
        ]
    else:
        raise ValueError(f"Expected a 2D or 3D label image, received {ndim} dimensions.")
    return centroid_columns, bbox_columns


def _expected_columns(ndim: int, include_intensity: bool) -> list[str]:
    centroid_columns, bbox_columns = _coordinate_columns(ndim)
    columns = [*METADATA_COLUMNS, "label", "area", *centroid_columns, *bbox_columns]
    if include_intensity:
        columns.extend(
            [
                "mean_intensity",
                "max_intensity",
                "min_intensity",
                "integrated_intensity",
            ]
        )
    return columns


def _prepare_label_image(label_image: np.ndarray) -> np.ndarray:
    labels = np.asarray(label_image)
    _coordinate_columns(labels.ndim)
    if labels.size == 0:
        raise ValueError("Cannot extract features from an empty image array.")
    if np.any(labels < 0):
        raise ValueError("Label images cannot contain negative labels.")
    if labels.dtype == bool:
        return label(labels, connectivity=labels.ndim).astype(np.int32)
    if not np.issubdtype(labels.dtype, np.integer):
        if not np.all(np.equal(labels, np.floor(labels))):
            raise ValueError("Label images must contain integer label values.")
        labels = labels.astype(np.int32)
    return labels


def _metadata_value(value: Any) -> Any:
    return str(value) if isinstance(value, Path) else value


def extract_region_features(
    label_image: np.ndarray,
    intensity_image: np.ndarray | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> pd.DataFrame:
    """Extract one row of morphology and optional intensity features per object."""
    labels = _prepare_label_image(label_image)
    intensity = None if intensity_image is None else np.asarray(intensity_image)
    if intensity is not None and intensity.shape != labels.shape:
        raise ValueError(
            "intensity_image and label_image must have the same shape: "
            f"{intensity.shape} != {labels.shape}."
        )

    include_intensity = intensity is not None
    expected_columns = _expected_columns(labels.ndim, include_intensity)
    centroid_columns, bbox_columns = _coordinate_columns(labels.ndim)
    properties = ["label", "area", "centroid", "bbox"]
    if include_intensity:
        properties.extend(["intensity_mean", "intensity_max", "intensity_min"])

    table = regionprops_table(
        labels,
        intensity_image=intensity,
        properties=tuple(properties),
    )
    features = pd.DataFrame(table)

    rename_map = {
        **{
            f"centroid-{index}": name
            for index, name in enumerate(centroid_columns)
        },
        **{
            f"bbox-{index}": name
            for index, name in enumerate(bbox_columns)
        },
        "intensity_mean": "mean_intensity",
        "intensity_max": "max_intensity",
        "intensity_min": "min_intensity",
    }
    features = features.rename(columns=rename_map)
    if include_intensity:
        features["integrated_intensity"] = features["mean_intensity"] * features["area"]

    metadata_values = dict(metadata or {})
    for column in METADATA_COLUMNS:
        features[column] = _metadata_value(metadata_values.get(column))

    return features.reindex(columns=expected_columns)


def extract_features_from_mask_file(
    mask_path: str | Path,
    intensity_path: str | Path | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> pd.DataFrame:
    """Load a labeled mask and optional intensity image, then extract features."""
    mask_file = Path(mask_path).expanduser().resolve(strict=False)
    intensity_file = (
        Path(intensity_path).expanduser().resolve(strict=False)
        if intensity_path is not None
        else None
    )
    labels = load_tiff_image(mask_file)
    intensity = load_tiff_image(intensity_file) if intensity_file is not None else None

    combined_metadata = {
        "mask_file": str(mask_file),
        "source_image_file": str(intensity_file) if intensity_file else None,
    }
    combined_metadata.update(metadata or {})
    return extract_region_features(labels, intensity, combined_metadata)


def summarize_feature_table(features_df: pd.DataFrame) -> pd.DataFrame:
    """Summarize numeric measurement columns in a feature table."""
    excluded_columns = {*METADATA_COLUMNS, "label"}
    numeric_columns = [
        column
        for column in features_df.select_dtypes(include=[np.number]).columns
        if column not in excluded_columns
    ]
    rows: list[dict[str, float | str | int]] = []
    for column in numeric_columns:
        values = pd.to_numeric(features_df[column], errors="coerce").dropna()
        if values.empty:
            continue
        rows.append(
            {
                "feature": column,
                "count": int(values.count()),
                "mean": float(values.mean()),
                "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                "min": float(values.min()),
                "median": float(values.median()),
                "max": float(values.max()),
            }
        )
    return pd.DataFrame(rows, columns=SUMMARY_COLUMNS)


def save_feature_table(features_df: pd.DataFrame, output_path: str | Path) -> Path:
    """Save a cell-level feature table as CSV."""
    path = Path(output_path).expanduser().resolve(strict=False)
    if path.suffix.lower() != ".csv":
        raise ValueError("Feature table output path must use the .csv extension.")
    path.parent.mkdir(parents=True, exist_ok=True)
    features_df.to_csv(path, index=False)
    return path


def save_feature_summary(summary_df: pd.DataFrame, output_path: str | Path) -> Path:
    """Save a feature summary table as CSV."""
    return save_feature_table(summary_df, output_path)
