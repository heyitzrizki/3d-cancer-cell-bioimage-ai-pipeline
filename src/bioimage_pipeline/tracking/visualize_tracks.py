"""Matplotlib visualizations for baseline cell tracks."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def _prepare_path(output_path: str | Path) -> Path:
    path = Path(output_path).expanduser().resolve(strict=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _save_empty(output_path: str | Path, title: str) -> Path:
    path = _prepare_path(output_path)
    figure, axis = plt.subplots(figsize=(7, 5))
    axis.text(0.5, 0.5, "No data available", ha="center", va="center")
    axis.set_title(title)
    axis.set_axis_off()
    figure.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    return path


def plot_tracks_projection(
    tracks_df: pd.DataFrame,
    output_path: str | Path,
) -> Path:
    """Plot x/y track trajectories, ignoring z in 3D tables."""
    required = {"track_id", "centroid_y", "centroid_x"}
    if tracks_df.empty or not required.issubset(tracks_df.columns):
        return _save_empty(output_path, "Cell Track Projection")

    path = _prepare_path(output_path)
    figure, axis = plt.subplots(figsize=(8, 8))
    for _, track in tracks_df.groupby("track_id", sort=False):
        ordered = track.sort_values("frame")
        axis.plot(
            ordered["centroid_x"],
            ordered["centroid_y"],
            marker="o",
            markersize=3,
            linewidth=1,
        )
    axis.set_title("Cell Track Projection")
    axis.set_xlabel("Centroid X")
    axis.set_ylabel("Centroid Y")
    axis.invert_yaxis()
    axis.set_aspect("equal", adjustable="box")
    figure.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    return path


def plot_speed_distribution(
    tracks_df: pd.DataFrame,
    output_path: str | Path,
) -> Path:
    """Plot the distribution of defined frame-to-frame speeds."""
    if tracks_df.empty or "speed" not in tracks_df.columns:
        return _save_empty(output_path, "Speed Distribution")
    speeds = pd.to_numeric(tracks_df["speed"], errors="coerce").dropna()
    if speeds.empty:
        return _save_empty(output_path, "Speed Distribution")

    path = _prepare_path(output_path)
    figure, axis = plt.subplots(figsize=(7, 5))
    axis.hist(speeds, bins=30)
    axis.set_title("Speed Distribution")
    axis.set_xlabel("Centroid Distance per Frame")
    axis.set_ylabel("Observation Count")
    figure.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    return path


def plot_track_duration_distribution(
    track_summary_df: pd.DataFrame,
    output_path: str | Path,
) -> Path:
    """Plot track duration measured as number of observed frames."""
    if track_summary_df.empty or "track_length" not in track_summary_df.columns:
        return _save_empty(output_path, "Track Duration Distribution")
    durations = pd.to_numeric(
        track_summary_df["track_length"],
        errors="coerce",
    ).dropna()
    if durations.empty:
        return _save_empty(output_path, "Track Duration Distribution")

    path = _prepare_path(output_path)
    bins = range(1, int(durations.max()) + 2)
    figure, axis = plt.subplots(figsize=(7, 5))
    axis.hist(durations, bins=bins, align="left", rwidth=0.8)
    axis.set_title("Track Duration Distribution")
    axis.set_xlabel("Track Length (Frames)")
    axis.set_ylabel("Track Count")
    axis.set_xticks(sorted(durations.unique()))
    figure.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    return path
