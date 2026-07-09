"""Centroid-based cell tracking and temporal analysis."""

from .tracker import (
    add_temporal_features,
    build_tracks,
    compute_centroid_distance_matrix,
    detect_coordinate_columns,
    link_frames_nearest_neighbor,
    summarize_tracks,
)
from .visualize_tracks import (
    plot_speed_distribution,
    plot_track_duration_distribution,
    plot_tracks_projection,
)

__all__ = [
    "add_temporal_features",
    "build_tracks",
    "compute_centroid_distance_matrix",
    "detect_coordinate_columns",
    "link_frames_nearest_neighbor",
    "plot_speed_distribution",
    "plot_track_duration_distribution",
    "plot_tracks_projection",
    "summarize_tracks",
]
