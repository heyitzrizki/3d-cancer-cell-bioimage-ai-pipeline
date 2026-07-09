"""Matplotlib visualizations for extracted cell-level features."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def _prepare_output_path(output_path: str | Path) -> Path:
    path = Path(output_path).expanduser().resolve(strict=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _save_empty_figure(output_path: str | Path, title: str) -> Path:
    path = _prepare_output_path(output_path)
    figure, axis = plt.subplots(figsize=(7, 5))
    axis.text(0.5, 0.5, "No data available", ha="center", va="center")
    axis.set_title(title)
    axis.set_axis_off()
    figure.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    return path


def plot_feature_distribution(
    features_df: pd.DataFrame,
    column: str,
    output_path: str | Path,
    bins: int = 30,
) -> Path:
    """Save a histogram for one numeric feature."""
    if bins < 1:
        raise ValueError("bins must be at least 1.")
    if features_df.empty:
        return _save_empty_figure(output_path, f"{column} Distribution")
    if column not in features_df.columns:
        raise ValueError(f"Feature column not found: {column}")

    values = pd.to_numeric(features_df[column], errors="coerce").dropna()
    if values.empty:
        return _save_empty_figure(output_path, f"{column} Distribution")

    path = _prepare_output_path(output_path)
    figure, axis = plt.subplots(figsize=(7, 5))
    axis.hist(values, bins=bins)
    axis.set_title(f"{column.replace('_', ' ').title()} Distribution")
    axis.set_xlabel(column.replace("_", " ").title())
    axis.set_ylabel("Object Count")
    figure.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    return path


def plot_area_vs_intensity(
    features_df: pd.DataFrame,
    output_path: str | Path,
) -> Path:
    """Save a scatter plot of object area against mean intensity."""
    required_columns = {"area", "mean_intensity"}
    if features_df.empty or not required_columns.issubset(features_df.columns):
        return _save_empty_figure(output_path, "Area vs Mean Intensity")

    plot_data = features_df[["area", "mean_intensity"]].apply(
        pd.to_numeric,
        errors="coerce",
    ).dropna()
    if plot_data.empty:
        return _save_empty_figure(output_path, "Area vs Mean Intensity")

    path = _prepare_output_path(output_path)
    figure, axis = plt.subplots(figsize=(7, 5))
    axis.scatter(plot_data["area"], plot_data["mean_intensity"], alpha=0.6)
    axis.set_title("Area vs Mean Intensity")
    axis.set_xlabel("Area / Voxel Count")
    axis.set_ylabel("Mean Intensity")
    figure.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    return path


def plot_feature_summary_bar(
    summary_df: pd.DataFrame,
    output_path: str | Path,
) -> Path:
    """Save a bar chart of mean values from a feature summary table."""
    if summary_df.empty or not {"feature", "mean"}.issubset(summary_df.columns):
        return _save_empty_figure(output_path, "Feature Summary")

    path = _prepare_output_path(output_path)
    figure, axis = plt.subplots(figsize=(10, 5))
    axis.bar(summary_df["feature"], summary_df["mean"])
    axis.set_title("Mean Feature Values")
    axis.set_ylabel("Mean")
    axis.tick_params(axis="x", rotation=45)
    figure.tight_layout()
    figure.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    return path
