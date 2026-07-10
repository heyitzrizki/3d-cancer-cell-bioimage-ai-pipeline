"""Tests for Milestone 6 model comparison utilities."""

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import tifffile
import torch

from bioimage_pipeline.deep_learning import (
    CTCAnnotatedPlaneDataset,
    UNet2D,
    save_checkpoint,
)
from bioimage_pipeline.evaluation.model_comparison import (
    compare_methods_on_dataset,
    load_unet_from_checkpoint,
    summarize_comparison,
)


def _make_toy_ctc_dataset(root: Path) -> None:
    sequence_dir = root / "01"
    gt_dir = root / "01_GT" / "SEG"
    sequence_dir.mkdir(parents=True)
    gt_dir.mkdir(parents=True)

    volume = np.zeros((3, 32, 32), dtype=np.uint16)
    volume[1, 10:22, 10:22] = 2000
    gt_mask = np.zeros((32, 32), dtype=np.uint16)
    gt_mask[10:22, 10:22] = 1

    tifffile.imwrite(sequence_dir / "t000.tif", volume, photometric="minisblack")
    tifffile.imwrite(gt_dir / "man_seg_000_001.tif", gt_mask, photometric="minisblack")


def _save_toy_checkpoint(path: Path) -> Path:
    model = UNet2D(base_channels=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    return save_checkpoint(
        path,
        model,
        optimizer,
        epoch=1,
        best_validation_dice=0.0,
        config={
            "base_channels": 2,
            "in_channels": 1,
            "out_channels": 1,
            "sequence_ids": ["01"],
        },
    )


def test_compare_methods_on_dataset_returns_rows_and_overlays(tmp_path: Path) -> None:
    _make_toy_ctc_dataset(tmp_path)
    checkpoint_path = _save_toy_checkpoint(tmp_path / "checkpoint.pt")
    dataset = CTCAnnotatedPlaneDataset(tmp_path, sequence_ids=["01"])
    model = load_unet_from_checkpoint(checkpoint_path, device="cpu")

    metrics, overlays = compare_methods_on_dataset(
        dataset,
        unet_model=model,
        device="cpu",
        min_size=4,
        overlay_dir=tmp_path / "figures",
        max_overlays=1,
    )
    summary = summarize_comparison(metrics)

    assert set(metrics["method"]) == {"classical_otsu", "unet_2d"}
    assert len(metrics) == 2
    assert set(summary["method"]) == {"classical_otsu", "unet_2d"}
    assert overlays and overlays[0].is_file()
    assert metrics["dice"].between(0, 1).all()
    assert metrics["iou"].between(0, 1).all()


def test_compare_segmentation_methods_cli_without_checkpoint(tmp_path: Path) -> None:
    _make_toy_ctc_dataset(tmp_path)
    output_dir = tmp_path / "reports"

    result = subprocess.run(
        [
            sys.executable,
            "scripts/compare_segmentation_methods.py",
            "--dataset-dir",
            str(tmp_path),
            "--output-dir",
            str(output_dir),
            "--sequence-ids",
            "01",
            "--min-size",
            "4",
            "--max-overlays",
            "1",
            "--device",
            "cpu",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    metrics_path = output_dir / "metrics" / "segmentation_method_comparison.csv"
    summary_path = output_dir / "metrics" / "segmentation_method_summary.csv"
    markdown_path = output_dir / "milestone_6_summary.md"
    metrics = pd.read_csv(metrics_path)

    assert "classical baseline only" in result.stdout
    assert metrics_path.is_file()
    assert summary_path.is_file()
    assert markdown_path.is_file()
    assert (output_dir / "figures" / "comparison_overlay_000.png").is_file()
    assert metrics["method"].tolist() == ["classical_otsu"]
    assert metrics.loc[0, "dice"] == pytest.approx(1.0)
