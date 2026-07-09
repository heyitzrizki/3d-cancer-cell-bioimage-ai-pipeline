"""Extract cell-level features from labeled segmentation masks."""

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from bioimage_pipeline.data import find_tiff_files, load_tiff_image
from bioimage_pipeline.features import (
    extract_region_features,
    plot_area_vs_intensity,
    plot_feature_distribution,
    save_feature_summary,
    save_feature_table,
    summarize_feature_table,
)
from bioimage_pipeline.preprocessing import preprocess_volume

_FRAME_PATTERN = re.compile(r"\d+")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", required=True, type=Path, help="CTC dataset root.")
    parser.add_argument("--sequence", default="01", help="Sequence identifier, such as 01.")
    parser.add_argument(
        "--mask-dir",
        type=Path,
        default=Path("reports") / "milestone_2" / "predicted_masks",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports") / "milestone_3",
    )
    parser.add_argument("--method", default="otsu")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument(
        "--use-preprocessed",
        action="store_true",
        help="Use on-the-fly preprocessed images for intensity features.",
    )
    args = parser.parse_args()
    if args.max_frames is not None and args.max_frames < 1:
        parser.error("--max-frames must be at least 1.")
    return args


def _frame_id(path: Path) -> str | None:
    match = _FRAME_PATTERN.search(path.stem)
    return match.group(0).zfill(3) if match else None


def _format_value(value: float | None) -> str:
    return "N/A" if value is None or pd.isna(value) else f"{value:.4f}"


def _portable_path(path: Path) -> str:
    try:
        return path.resolve(strict=False).relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(path.resolve(strict=False))


def _write_markdown_summary(
    output_path: Path,
    *,
    dataset: str,
    sequence: str,
    method: str,
    processed_masks: int,
    features: pd.DataFrame,
    feature_path: Path,
    summary_path: Path,
    intensity_source: str,
) -> Path:
    area_values = pd.to_numeric(features.get("area"), errors="coerce").dropna()
    intensity_values = (
        pd.to_numeric(features["mean_intensity"], errors="coerce").dropna()
        if "mean_intensity" in features.columns
        else pd.Series(dtype=float)
    )
    mean_area = float(area_values.mean()) if not area_values.empty else None
    median_area = float(area_values.median()) if not area_values.empty else None
    mean_intensity = (
        float(intensity_values.mean()) if not intensity_values.empty else None
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    content = f"""# Feature Extraction Summary

- Dataset: `{dataset}`
- Sequence: `{sequence}`
- Segmentation method: `{method}`
- Intensity source: `{intensity_source}`
- Processed masks: {processed_masks}
- Extracted objects: {len(features)}
- Mean area / voxel count: {_format_value(mean_area)}
- Median area / voxel count: {_format_value(median_area)}
- Mean intensity: {_format_value(mean_intensity)}
- Cell-level features: `{_portable_path(feature_path)}`
- Feature summary: `{_portable_path(summary_path)}`

These features are derived from classical baseline segmentation masks. They are exploratory outputs, not final validated biological measurements.
"""
    output_path.write_text(content, encoding="utf-8")
    return output_path


def main() -> None:
    args = parse_args()
    dataset_dir = args.dataset_dir.expanduser().resolve(strict=False)
    mask_dir = args.mask_dir.expanduser().resolve(strict=False)
    output_dir = args.output_dir.expanduser().resolve(strict=False)
    sequence = str(args.sequence).zfill(2)
    sequence_dir = dataset_dir / sequence

    if not sequence_dir.is_dir():
        raise NotADirectoryError(f"Sequence directory not found: {sequence_dir}")
    if not mask_dir.is_dir():
        raise NotADirectoryError(f"Mask directory not found: {mask_dir}")

    raw_by_frame = {
        frame_id: path
        for path in find_tiff_files(sequence_dir, recursive=False)
        if (frame_id := _frame_id(path)) is not None
    }
    mask_by_frame = {
        frame_id: path
        for path in find_tiff_files(mask_dir, recursive=False)
        if (frame_id := _frame_id(path)) is not None
    }
    matched_frames = sorted(set(raw_by_frame) & set(mask_by_frame))
    if args.max_frames is not None:
        matched_frames = matched_frames[: args.max_frames]
    if not matched_frames:
        raise FileNotFoundError("No matching raw-image and mask frame numbers were found.")

    unmatched_masks = sorted(set(mask_by_frame) - set(raw_by_frame))
    for frame in unmatched_masks:
        print(f"Warning: no raw image found for mask frame {frame}; skipping it.")

    frame_tables: list[pd.DataFrame] = []
    processed_masks = 0
    for frame in matched_frames:
        raw_path = raw_by_frame[frame]
        mask_path = mask_by_frame[frame]
        print(f"Processing frame {frame}: {mask_path.name} + {raw_path.name}")

        label_image = load_tiff_image(mask_path)
        raw_image = load_tiff_image(raw_path)
        intensity_image = raw_image
        if raw_image.shape != label_image.shape:
            print(
                f"Warning: shape mismatch for frame {frame}: raw {raw_image.shape}, "
                f"mask {label_image.shape}. Extracting morphology-only features."
            )
            intensity_image = None
        elif args.use_preprocessed:
            intensity_image = preprocess_volume(raw_image)

        metadata = {
            "dataset": dataset_dir.name,
            "sequence": sequence,
            "frame": frame,
            "source_image_file": _portable_path(raw_path),
            "mask_file": _portable_path(mask_path),
            "method": args.method,
        }
        frame_tables.append(
            extract_region_features(
                label_image,
                intensity_image=intensity_image,
                metadata=metadata,
            )
        )
        processed_masks += 1

    features = pd.concat(frame_tables, ignore_index=True, sort=False)
    summary = summarize_feature_table(features)

    features_dir = output_dir / "features"
    figures_dir = output_dir / "figures"
    feature_path = save_feature_table(
        features,
        features_dir / "mda231_cell_features.csv",
    )
    summary_path = save_feature_summary(
        summary,
        features_dir / "mda231_feature_summary.csv",
    )

    figure_paths = [
        plot_feature_distribution(
            features,
            "area",
            figures_dir / "cell_area_distribution.png",
        )
    ]
    has_intensity = (
        "mean_intensity" in features.columns
        and features["mean_intensity"].notna().any()
    )
    if has_intensity:
        figure_paths.extend(
            [
                plot_feature_distribution(
                    features,
                    "mean_intensity",
                    figures_dir / "mean_intensity_distribution.png",
                ),
                plot_area_vs_intensity(
                    features,
                    figures_dir / "area_vs_intensity.png",
                ),
            ]
        )

    markdown_path = _write_markdown_summary(
        output_dir / "feature_extraction_summary.md",
        dataset=dataset_dir.name,
        sequence=sequence,
        method=args.method,
        processed_masks=processed_masks,
        features=features,
        feature_path=feature_path,
        summary_path=summary_path,
        intensity_source="preprocessed" if args.use_preprocessed else "raw",
    )

    print(f"Processed masks: {processed_masks}")
    print(f"Extracted objects: {len(features)}")
    print(f"Feature table: {feature_path}")
    print(f"Feature summary: {summary_path}")
    for figure_path in figure_paths:
        print(f"Figure: {figure_path}")
    print(f"Markdown summary: {markdown_path}")


if __name__ == "__main__":
    main()
