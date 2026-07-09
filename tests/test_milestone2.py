"""Tests for Milestone 2 preprocessing, segmentation, and metrics."""

import json
import subprocess
import sys
from pathlib import Path

import matplotlib
import numpy as np
import tifffile

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from bioimage_pipeline.data import (
    get_segmentation_gt_slice_index,
    match_prediction_to_gt_frame,
)
from bioimage_pipeline.evaluation import (
    dice_coefficient,
    iou_score,
    precision_recall_f1,
)
from bioimage_pipeline.preprocessing import denoise_image, normalize_intensity
from bioimage_pipeline.segmentation import label_components, segment_volume_baseline
from bioimage_pipeline.visualization import overlay_mask_on_image


def test_intensity_normalization_and_constant_image() -> None:
    image = np.arange(9, dtype=np.uint16).reshape(3, 3)
    normalized = normalize_intensity(image, method="minmax")
    constant = normalize_intensity(np.full((3, 3), 7, dtype=np.uint16))

    assert normalized.dtype == np.float32
    assert normalized.shape == image.shape
    assert normalized.min() == 0.0
    assert normalized.max() == 1.0
    assert np.all(constant == 0)


def test_empty_mask_metrics_are_safe() -> None:
    empty = np.zeros((4, 4), dtype=bool)
    classification_metrics = precision_recall_f1(empty, empty)

    assert dice_coefficient(empty, empty) == 1.0
    assert iou_score(empty, empty) == 1.0
    assert np.isnan(classification_metrics["precision"])
    assert np.isnan(classification_metrics["recall"])
    assert np.isnan(classification_metrics["f1"])


def test_dice_and_iou_on_toy_masks() -> None:
    y_true = np.array([[1, 1], [0, 0]])
    y_pred = np.array([[1, 0], [1, 0]])

    assert dice_coefficient(y_true, y_pred) == 0.5
    assert iou_score(y_true, y_pred) == 1 / 3


def test_connected_component_labeling() -> None:
    mask = np.zeros((8, 8), dtype=bool)
    mask[1:3, 1:3] = True
    mask[5:7, 5:7] = True

    labels = label_components(mask)

    assert labels.shape == mask.shape
    assert labels.max() == 2


def test_baseline_segmentation_preserves_input_shape() -> None:
    image = np.zeros((3, 16, 16), dtype=np.float32)
    image[:, 4:12, 4:12] = 1.0

    binary_mask, labeled_mask = segment_volume_baseline(
        image,
        method="otsu",
        config={"min_size": 4},
    )

    assert binary_mask.shape == image.shape
    assert labeled_mask.shape == image.shape
    assert binary_mask.dtype == bool


def test_per_axis_denoising_can_avoid_blurring_across_z() -> None:
    image = np.zeros((3, 9, 9), dtype=np.float32)
    image[1, 4, 4] = 1.0

    denoised = denoise_image(image, sigma=(0.0, 1.0, 1.0))

    assert np.all(denoised[0] == 0)
    assert np.all(denoised[2] == 0)
    assert denoised[1].max() > 0


def test_sparse_gt_frame_and_slice_matching(tmp_path: Path) -> None:
    prediction = tmp_path / "t000.tif"
    matching_gt = tmp_path / "man_seg_000_013.tif"
    other_gt = tmp_path / "man_seg_001_007.tif"

    matches = match_prediction_to_gt_frame(
        [prediction],
        [matching_gt, other_gt],
    )

    assert matches[prediction.resolve()] == [matching_gt.resolve()]
    assert get_segmentation_gt_slice_index(matching_gt) == 13


def test_overlay_uses_visible_mask_color_range() -> None:
    image = np.zeros((8, 8), dtype=np.float32)
    mask = np.zeros((8, 8), dtype=bool)
    mask[2:6, 2:6] = True

    figure, axis = plt.subplots()
    overlay_mask_on_image(image, mask, ax=axis)

    assert len(axis.images) == 2
    assert axis.images[1].get_clim() == (0.0, 1.0)
    plt.close(figure)


def test_baseline_cli_writes_full_volume_evaluation(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "dataset"
    sequence_dir = dataset_dir / "01"
    gt_dir = dataset_dir / "01_GT" / "SEG"
    output_dir = tmp_path / "output"
    sequence_dir.mkdir(parents=True)
    gt_dir.mkdir(parents=True)

    image = np.zeros((3, 32, 32), dtype=np.uint16)
    image[:, 8:24, 8:24] = 1000
    gt_mask = np.zeros_like(image)
    gt_mask[:, 8:24, 8:24] = 1
    tifffile.imwrite(sequence_dir / "t000.tif", image, photometric="minisblack")
    tifffile.imwrite(gt_dir / "man_seg000.tif", gt_mask, photometric="minisblack")

    project_root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [
            sys.executable,
            str(project_root / "scripts" / "run_baseline_segmentation.py"),
            "--dataset-dir",
            str(dataset_dir),
            "--output-dir",
            str(output_dir),
            "--max-frames",
            "1",
            "--min-size",
            "4",
            "--save-masks",
        ],
        cwd=project_root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    metrics_path = output_dir / "metrics" / "baseline_segmentation_metrics.json"
    with metrics_path.open(encoding="utf-8") as file:
        records = json.load(file)
    assert records[0]["evaluation_scope"] == "full_volume"
    assert records[0]["evaluated_gt_files"] == 1
    assert (output_dir / "predicted_masks" / "mask000.tif").is_file()
