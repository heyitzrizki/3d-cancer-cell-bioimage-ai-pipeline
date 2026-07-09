"""Run centroid-based Hungarian cell tracking on a feature table."""

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from bioimage_pipeline.tracking import (
    add_temporal_features,
    build_tracks,
    detect_coordinate_columns,
    plot_speed_distribution,
    plot_track_duration_distribution,
    plot_tracks_projection,
    summarize_tracks,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--features-file",
        type=Path,
        default=Path("reports") / "milestone_3" / "features" / "mda231_cell_features.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports") / "milestone_4",
    )
    parser.add_argument("--max-distance", type=float, default=50.0)
    parser.add_argument("--dataset-name", default="mda231")
    args = parser.parse_args()
    if args.max_distance < 0:
        parser.error("--max-distance must be non-negative.")
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.dataset_name):
        parser.error("--dataset-name may contain only letters, numbers, hyphens, and underscores.")
    return args


def _portable_path(path: Path) -> str:
    try:
        return path.resolve(strict=False).relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(path.resolve(strict=False))


def _format_value(value: float) -> str:
    return "N/A" if pd.isna(value) else f"{value:.4f}"


def _write_markdown_summary(
    output_path: Path,
    *,
    features_file: Path,
    tracks: pd.DataFrame,
    track_summary: pd.DataFrame,
    max_distance: float,
    track_file: Path,
    summary_file: Path,
) -> Path:
    mean_track_length = pd.to_numeric(
        track_summary.get("track_length"),
        errors="coerce",
    ).mean()
    median_track_length = pd.to_numeric(
        track_summary.get("track_length"),
        errors="coerce",
    ).median()
    mean_speed = pd.to_numeric(tracks.get("speed"), errors="coerce").mean()

    content = f"""# Tracking Summary

- Input feature file: `{_portable_path(features_file)}`
- Number of frames: {tracks["frame"].nunique() if "frame" in tracks else 0}
- Number of objects: {len(tracks)}
- Number of tracks: {len(track_summary)}
- Mean track length: {_format_value(mean_track_length)}
- Median track length: {_format_value(median_track_length)}
- Mean speed: {_format_value(mean_speed)}
- Maximum matching distance: {max_distance:.4f}
- Cell tracks: `{_portable_path(track_file)}`
- Track summary: `{_portable_path(summary_file)}`

## Limitations

- Centroid matching is a transparent baseline, not a state-of-the-art tracker.
- Merged or split cells can break track identity.
- Cell division is not modeled.
- Appearance-based matching is not included.
- Distances are measured in voxel coordinates without physical voxel-size calibration.
"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(content, encoding="utf-8")
    return output_path


def main() -> None:
    args = parse_args()
    features_file = args.features_file.expanduser().resolve(strict=False)
    output_dir = args.output_dir.expanduser().resolve(strict=False)
    if not features_file.is_file():
        raise FileNotFoundError(f"Feature table not found: {features_file}")

    features = pd.read_csv(features_file)
    required_columns = {"frame", "label"}
    missing = sorted(required_columns - set(features.columns))
    if missing:
        raise ValueError(f"Missing required feature columns: {', '.join(missing)}")
    coordinate_columns = detect_coordinate_columns(features)
    print(f"Coordinate columns: {', '.join(coordinate_columns)}")

    tracks = add_temporal_features(
        build_tracks(features, max_distance=args.max_distance)
    )
    track_summary = summarize_tracks(tracks)

    tracks_dir = output_dir / "tracks"
    figures_dir = output_dir / "figures"
    tracks_dir.mkdir(parents=True, exist_ok=True)
    track_file = tracks_dir / f"{args.dataset_name}_cell_tracks.csv"
    summary_file = tracks_dir / f"{args.dataset_name}_track_summary.csv"
    tracks.to_csv(track_file, index=False)
    track_summary.to_csv(summary_file, index=False)

    figure_paths = [
        plot_tracks_projection(
            tracks,
            figures_dir / "cell_tracks_projection.png",
        ),
        plot_speed_distribution(
            tracks,
            figures_dir / "speed_distribution.png",
        ),
        plot_track_duration_distribution(
            track_summary,
            figures_dir / "track_duration_distribution.png",
        ),
    ]
    markdown_path = _write_markdown_summary(
        output_dir / "tracking_summary.md",
        features_file=features_file,
        tracks=tracks,
        track_summary=track_summary,
        max_distance=args.max_distance,
        track_file=track_file,
        summary_file=summary_file,
    )

    print(f"Frames: {features['frame'].nunique()}")
    print(f"Objects: {len(tracks)}")
    print(f"Tracks: {len(track_summary)}")
    print(f"Track table: {track_file}")
    print(f"Track summary: {summary_file}")
    for figure_path in figure_paths:
        print(f"Figure: {figure_path}")
    print(f"Markdown summary: {markdown_path}")


if __name__ == "__main__":
    main()
