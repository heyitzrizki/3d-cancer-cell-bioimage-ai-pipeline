"""Transparent centroid-based tracking for consecutive microscopy frames."""

from collections.abc import Sequence

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

LINK_COLUMNS = ("current_index", "next_index", "distance")
TEMPORAL_COLUMNS = (
    "displacement_from_previous",
    "speed",
    "cumulative_displacement",
    "track_length",
)
TRACK_SUMMARY_COLUMNS = (
    "dataset",
    "sequence",
    "method",
    "track_id",
    "start_frame",
    "end_frame",
    "track_length",
    "total_displacement",
    "mean_speed",
    "max_speed",
    "mean_area",
    "mean_intensity",
)


def detect_coordinate_columns(features_df: pd.DataFrame) -> list[str]:
    """Detect complete 3D or 2D centroid coordinate columns."""
    coordinates_3d = ["centroid_z", "centroid_y", "centroid_x"]
    coordinates_2d = ["centroid_y", "centroid_x"]
    if all(column in features_df.columns for column in coordinates_3d):
        return coordinates_3d
    if all(column in features_df.columns for column in coordinates_2d):
        return coordinates_2d
    raise ValueError(
        "Feature table must contain centroid_y and centroid_x, "
        "with optional centroid_z for 3D tracking."
    )


def _coordinate_array(
    features_df: pd.DataFrame,
    coordinate_columns: Sequence[str],
) -> np.ndarray:
    missing = [column for column in coordinate_columns if column not in features_df.columns]
    if missing:
        raise ValueError(f"Missing centroid columns: {', '.join(missing)}")
    coordinates = features_df.loc[:, coordinate_columns].apply(
        pd.to_numeric,
        errors="coerce",
    )
    if coordinates.isna().any().any():
        raise ValueError("Centroid coordinates must be numeric and non-missing.")
    return coordinates.to_numpy(dtype=float)


def compute_centroid_distance_matrix(
    current_features: pd.DataFrame,
    next_features: pd.DataFrame,
    coordinate_columns: Sequence[str],
) -> np.ndarray:
    """Compute pairwise Euclidean centroid distances between two frames."""
    current_coordinates = _coordinate_array(current_features, coordinate_columns)
    next_coordinates = _coordinate_array(next_features, coordinate_columns)
    if len(current_coordinates) == 0 or len(next_coordinates) == 0:
        return np.empty((len(current_coordinates), len(next_coordinates)), dtype=float)
    return cdist(current_coordinates, next_coordinates, metric="euclidean")


def link_frames_nearest_neighbor(
    current_features: pd.DataFrame,
    next_features: pd.DataFrame,
    max_distance: float = 50.0,
) -> pd.DataFrame:
    """Link two frames with thresholded Hungarian centroid matching."""
    if max_distance < 0:
        raise ValueError("max_distance must be non-negative.")
    if current_features.empty or next_features.empty:
        return pd.DataFrame(columns=LINK_COLUMNS)

    coordinate_columns = detect_coordinate_columns(current_features)
    if not all(column in next_features.columns for column in coordinate_columns):
        raise ValueError("Current and next frames must use the same centroid columns.")

    distances = compute_centroid_distance_matrix(
        current_features,
        next_features,
        coordinate_columns,
    )
    penalty = max(float(distances.max(initial=0.0)), max_distance) + 1_000_000.0
    thresholded_cost = np.where(distances <= max_distance, distances, penalty)
    current_positions, next_positions = linear_sum_assignment(thresholded_cost)

    rows = []
    for current_position, next_position in zip(current_positions, next_positions):
        distance = float(distances[current_position, next_position])
        if distance <= max_distance:
            rows.append(
                {
                    "current_index": current_features.index[current_position],
                    "next_index": next_features.index[next_position],
                    "distance": distance,
                }
            )
    return pd.DataFrame(rows, columns=LINK_COLUMNS)


def _frame_sort_value(value: object) -> tuple[int, float | str]:
    try:
        return 0, float(value)
    except (TypeError, ValueError):
        return 1, str(value)


def _frames_are_consecutive(current_frame: object, next_frame: object) -> bool:
    try:
        return float(next_frame) - float(current_frame) == 1
    except (TypeError, ValueError):
        return True


def build_tracks(
    features_df: pd.DataFrame,
    max_distance: float = 50.0,
) -> pd.DataFrame:
    """Assign persistent track IDs across consecutive frames."""
    if max_distance < 0:
        raise ValueError("max_distance must be non-negative.")
    if features_df.empty:
        result = features_df.copy()
        result["track_id"] = pd.Series(dtype="int64")
        return result
    if "frame" not in features_df.columns:
        raise ValueError("Feature table must contain a frame column.")
    detect_coordinate_columns(features_df)

    result = features_df.copy().reset_index(drop=True)
    result["track_id"] = pd.Series(pd.NA, index=result.index, dtype="Int64")
    group_columns = [
        column for column in ("dataset", "sequence") if column in result.columns
    ]
    grouped = (
        result.groupby(group_columns, dropna=False, sort=False)
        if group_columns
        else [(None, result)]
    )

    next_track_id = 1
    for _, group in grouped:
        frames = sorted(group["frame"].drop_duplicates(), key=_frame_sort_value)
        previous_frame: object | None = None
        previous_rows: pd.DataFrame | None = None

        for frame in frames:
            frame_rows = result.loc[group.index[group["frame"] == frame]]
            if previous_rows is None or not _frames_are_consecutive(previous_frame, frame):
                for row_index in frame_rows.index:
                    result.at[row_index, "track_id"] = next_track_id
                    next_track_id += 1
            else:
                links = link_frames_nearest_neighbor(
                    previous_rows,
                    frame_rows,
                    max_distance=max_distance,
                )
                matched_next_indices: set[int] = set()
                for link_row in links.itertuples(index=False):
                    track_id = result.at[link_row.current_index, "track_id"]
                    result.at[link_row.next_index, "track_id"] = track_id
                    matched_next_indices.add(int(link_row.next_index))

                for row_index in frame_rows.index:
                    if row_index not in matched_next_indices:
                        result.at[row_index, "track_id"] = next_track_id
                        next_track_id += 1

            previous_frame = frame
            previous_rows = result.loc[frame_rows.index]

    result["track_id"] = result["track_id"].astype(int)
    return result


def add_temporal_features(tracks_df: pd.DataFrame) -> pd.DataFrame:
    """Add displacement, speed, cumulative distance, and track length."""
    if tracks_df.empty:
        result = tracks_df.copy()
        for column in TEMPORAL_COLUMNS:
            dtype = "int64" if column == "track_length" else "float64"
            result[column] = pd.Series(dtype=dtype)
        return result
    if "track_id" not in tracks_df.columns:
        raise ValueError("Tracking table must contain a track_id column.")
    if "frame" not in tracks_df.columns:
        raise ValueError("Tracking table must contain a frame column.")
    coordinate_columns = detect_coordinate_columns(tracks_df)

    result = tracks_df.copy()
    for column in TEMPORAL_COLUMNS:
        result[column] = np.nan
    for _, track in result.groupby("track_id", sort=False):
        sorted_indices = sorted(
            track.index,
            key=lambda index: _frame_sort_value(result.at[index, "frame"]),
        )
        coordinates = _coordinate_array(result.loc[sorted_indices], coordinate_columns)
        displacements = np.full(len(sorted_indices), np.nan, dtype=float)
        if len(sorted_indices) > 1:
            displacements[1:] = np.linalg.norm(np.diff(coordinates, axis=0), axis=1)
        cumulative = np.nancumsum(displacements)

        result.loc[sorted_indices, "displacement_from_previous"] = displacements
        result.loc[sorted_indices, "speed"] = displacements
        result.loc[sorted_indices, "cumulative_displacement"] = cumulative
        result.loc[sorted_indices, "track_length"] = len(sorted_indices)

    result["track_length"] = result["track_length"].astype(int)
    return result


def summarize_tracks(tracks_df: pd.DataFrame) -> pd.DataFrame:
    """Create one summary row per track."""
    if tracks_df.empty:
        return pd.DataFrame(columns=TRACK_SUMMARY_COLUMNS)
    if "track_id" not in tracks_df.columns:
        raise ValueError("Tracking table must contain a track_id column.")

    rows: list[dict[str, object]] = []
    for track_id, track in tracks_df.groupby("track_id", sort=True):
        ordered = track.loc[
            sorted(track.index, key=lambda index: _frame_sort_value(track.at[index, "frame"]))
        ]
        speeds = (
            pd.to_numeric(ordered["speed"], errors="coerce").dropna()
            if "speed" in ordered.columns
            else pd.Series(dtype=float)
        )
        cumulative = (
            pd.to_numeric(ordered["cumulative_displacement"], errors="coerce").dropna()
            if "cumulative_displacement" in ordered.columns
            else pd.Series(dtype=float)
        )
        rows.append(
            {
                "dataset": ordered["dataset"].iloc[0] if "dataset" in ordered else None,
                "sequence": ordered["sequence"].iloc[0] if "sequence" in ordered else None,
                "method": ordered["method"].iloc[0] if "method" in ordered else None,
                "track_id": int(track_id),
                "start_frame": ordered["frame"].iloc[0],
                "end_frame": ordered["frame"].iloc[-1],
                "track_length": len(ordered),
                "total_displacement": float(cumulative.iloc[-1]) if not cumulative.empty else 0.0,
                "mean_speed": float(speeds.mean()) if not speeds.empty else np.nan,
                "max_speed": float(speeds.max()) if not speeds.empty else np.nan,
                "mean_area": (
                    float(pd.to_numeric(ordered["area"], errors="coerce").mean())
                    if "area" in ordered
                    else np.nan
                ),
                "mean_intensity": (
                    float(
                        pd.to_numeric(
                            ordered["mean_intensity"],
                            errors="coerce",
                        ).mean()
                    )
                    if "mean_intensity" in ordered
                    else np.nan
                ),
            }
        )
    return pd.DataFrame(rows, columns=TRACK_SUMMARY_COLUMNS)
