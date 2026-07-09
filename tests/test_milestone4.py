"""Tests for Milestone 4 centroid-based tracking."""

import numpy as np
import pandas as pd

from bioimage_pipeline.tracking import (
    add_temporal_features,
    build_tracks,
    compute_centroid_distance_matrix,
    link_frames_nearest_neighbor,
    summarize_tracks,
)


def test_centroid_distance_matrix() -> None:
    current = pd.DataFrame({"centroid_y": [0.0, 3.0], "centroid_x": [0.0, 4.0]})
    following = pd.DataFrame({"centroid_y": [0.0], "centroid_x": [0.0]})

    distances = compute_centroid_distance_matrix(
        current,
        following,
        ["centroid_y", "centroid_x"],
    )

    np.testing.assert_allclose(distances, np.array([[0.0], [5.0]]))


def test_nearest_neighbor_frame_linking() -> None:
    current = pd.DataFrame(
        {"centroid_y": [0.0, 10.0], "centroid_x": [0.0, 10.0]},
        index=[4, 5],
    )
    following = pd.DataFrame(
        {"centroid_y": [9.0, 1.0], "centroid_x": [10.0, 0.0]},
        index=[8, 9],
    )

    links = link_frames_nearest_neighbor(current, following, max_distance=5.0)

    assert set(zip(links["current_index"], links["next_index"])) == {(4, 9), (5, 8)}


def test_max_distance_rejects_far_matches() -> None:
    current = pd.DataFrame({"centroid_y": [0.0], "centroid_x": [0.0]})
    following = pd.DataFrame({"centroid_y": [10.0], "centroid_x": [0.0]})

    links = link_frames_nearest_neighbor(current, following, max_distance=5.0)

    assert links.empty


def test_build_tracks_across_three_frames() -> None:
    features = pd.DataFrame(
        {
            "frame": [0, 1, 2],
            "label": [1, 2, 3],
            "centroid_y": [0.0, 1.0, 2.0],
            "centroid_x": [0.0, 0.0, 0.0],
        }
    )

    tracks = build_tracks(features, max_distance=2.0)

    assert tracks["track_id"].nunique() == 1
    assert tracks["track_id"].tolist() == [1, 1, 1]


def test_unmatched_object_gets_new_track() -> None:
    features = pd.DataFrame(
        {
            "frame": [0, 1, 1],
            "label": [1, 2, 3],
            "centroid_y": [0.0, 1.0, 100.0],
            "centroid_x": [0.0, 0.0, 100.0],
        }
    )

    tracks = build_tracks(features, max_distance=5.0)

    first_track = tracks.loc[tracks["frame"] == 0, "track_id"].item()
    next_track_ids = tracks.loc[tracks["frame"] == 1, "track_id"].tolist()
    assert first_track in next_track_ids
    assert len(set(next_track_ids)) == 2


def test_temporal_features() -> None:
    tracks = pd.DataFrame(
        {
            "track_id": [1, 1, 1],
            "frame": [0, 1, 2],
            "centroid_y": [0.0, 3.0, 6.0],
            "centroid_x": [0.0, 4.0, 8.0],
        }
    )

    temporal = add_temporal_features(tracks)

    assert np.isnan(temporal.loc[0, "speed"])
    np.testing.assert_allclose(temporal.loc[1:, "speed"], [5.0, 5.0])
    np.testing.assert_allclose(temporal["cumulative_displacement"], [0.0, 5.0, 10.0])
    assert temporal["track_length"].tolist() == [3, 3, 3]


def test_summarize_tracks_columns_and_values() -> None:
    tracks = pd.DataFrame(
        {
            "track_id": [1, 1],
            "frame": [0, 1],
            "centroid_y": [0.0, 3.0],
            "centroid_x": [0.0, 4.0],
            "area": [10.0, 14.0],
            "mean_intensity": [2.0, 4.0],
        }
    )
    temporal = add_temporal_features(tracks)

    summary = summarize_tracks(temporal)

    assert {
        "track_id",
        "start_frame",
        "end_frame",
        "track_length",
        "total_displacement",
        "mean_speed",
        "max_speed",
        "mean_area",
        "mean_intensity",
    } <= set(summary.columns)
    assert summary.loc[0, "track_length"] == 2
    assert summary.loc[0, "total_displacement"] == 5.0
    assert summary.loc[0, "mean_speed"] == 5.0
    assert summary.loc[0, "mean_area"] == 12.0


def test_empty_dataframes_are_handled() -> None:
    empty_coordinates = pd.DataFrame(columns=["centroid_y", "centroid_x"])

    distances = compute_centroid_distance_matrix(
        empty_coordinates,
        empty_coordinates,
        ["centroid_y", "centroid_x"],
    )
    links = link_frames_nearest_neighbor(empty_coordinates, empty_coordinates)
    tracks = build_tracks(pd.DataFrame())
    temporal = add_temporal_features(tracks)
    summary = summarize_tracks(temporal)

    assert distances.shape == (0, 0)
    assert links.empty
    assert tracks.empty and "track_id" in tracks.columns
    assert temporal.empty and "speed" in temporal.columns
    assert summary.empty and "track_length" in summary.columns
