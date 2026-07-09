"""Inspect a CTC dataset and save sample image visualizations."""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from bioimage_pipeline.data import (
    find_tiff_files,
    get_image_statistics,
    list_ctc_sequences,
    load_tiff_image,
    summarize_dataset_structure,
)
from bioimage_pipeline.visualization import (
    plot_max_intensity_projection,
    plot_slice,
    save_figure,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", required=True, type=Path, help="CTC dataset directory.")
    parser.add_argument("--output-dir", required=True, type=Path, help="Figure output directory.")
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Optionally limit the sorted sample-file candidates.",
    )
    args = parser.parse_args()
    if args.max_files is not None and args.max_files < 1:
        parser.error("--max-files must be at least 1.")
    return args


def _save_sample_figures(image, sample_path: Path, output_dir: Path) -> list[Path]:
    if image.ndim == 2:
        display_image = image
        slice_title = f"2D Image: {sample_path.name}"
    elif image.ndim == 3:
        middle_z = image.shape[0] // 2
        display_image = image[middle_z]
        slice_title = f"Middle Z-Slice ({middle_z}): {sample_path.name}"
    else:
        print(
            f"Warning: expected a 2D image or 3D volume, "
            f"but {sample_path.name} has shape {image.shape}. Skipping figures."
        )
        return []

    slice_figure, slice_axis = plt.subplots(figsize=(7, 7))
    plot_slice(display_image, ax=slice_axis, title=slice_title)
    slice_path = save_figure(
        slice_figure,
        "sample_z_slice.png",
        output_dir=output_dir,
    )
    plt.close(slice_figure)

    projection_figure, projection_axis = plt.subplots(figsize=(7, 7))
    plot_max_intensity_projection(
        image,
        ax=projection_axis,
        title=f"Max Intensity Projection: {sample_path.name}",
    )
    projection_path = save_figure(
        projection_figure,
        "sample_max_intensity_projection.png",
        output_dir=output_dir,
    )
    plt.close(projection_figure)
    return [slice_path, projection_path]


def main() -> None:
    args = parse_args()
    dataset_dir = args.dataset_dir.expanduser().resolve(strict=False)
    output_dir = args.output_dir.expanduser().resolve(strict=False)

    summary = summarize_dataset_structure(dataset_dir)
    print(f"Dataset path: {dataset_dir}")
    if summary.empty:
        print("No CTC sequence or ground-truth folders were detected.")
    else:
        print(summary.to_string(index=False))

    sequences = list_ctc_sequences(dataset_dir)
    if not sequences:
        print("Warning: no numeric sequence folders were found; no sample was loaded.")
        return

    sample_files = find_tiff_files(sequences[0], recursive=True)
    if args.max_files is not None:
        sample_files = sample_files[: args.max_files]
    if not sample_files:
        print(f"Warning: no TIFF files were found in first sequence: {sequences[0]}")
        return

    sample_path = sample_files[0]
    image = load_tiff_image(sample_path)
    statistics = get_image_statistics(image)
    print(f"Sample TIFF: {sample_path}")
    print(f"Shape: {statistics['shape']}")
    print(f"Dimensions: {statistics['ndim']}D")
    print(f"Dtype: {statistics['dtype']}")
    print(
        "Intensity range: "
        f"{statistics['min_intensity']} to {statistics['max_intensity']}"
    )
    print(f"Mean intensity: {statistics['mean_intensity']:.4f}")

    saved_paths = _save_sample_figures(image, sample_path, output_dir)
    for path in saved_paths:
        print(f"Saved figure: {path}")


if __name__ == "__main__":
    main()
