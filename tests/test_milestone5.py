"""CPU-only tests for Milestone 5 2D U-Net components."""

from pathlib import Path

import numpy as np
import pytest
import tifffile
import torch

from bioimage_pipeline.deep_learning import (
    CTCAnnotatedPlaneDataset,
    UNet2D,
    binary_dice_score,
    binary_iou_score,
    load_checkpoint,
    parse_ctc_segmentation_filename,
    save_checkpoint,
)


def test_parse_sparse_and_dense_ctc_segmentation_names() -> None:
    assert parse_ctc_segmentation_filename("man_seg_000_013.tif") == (0, 13)
    assert parse_ctc_segmentation_filename("man_seg011.tif") == (11, None)
    with pytest.raises(ValueError, match="Unsupported"):
        parse_ctc_segmentation_filename("mask000.tif")


def test_annotated_plane_dataset_extracts_matching_z_slice(tmp_path: Path) -> None:
    sequence_dir = tmp_path / "01"
    gt_dir = tmp_path / "01_GT" / "SEG"
    sequence_dir.mkdir()
    gt_dir.mkdir(parents=True)
    volume = np.zeros((3, 16, 16), dtype=np.uint16)
    volume[1, 4:12, 4:12] = 1000
    gt_mask = np.zeros((16, 16), dtype=np.uint16)
    gt_mask[4:12, 4:12] = 7
    tifffile.imwrite(
        sequence_dir / "t000.tif",
        volume,
        photometric="minisblack",
    )
    tifffile.imwrite(gt_dir / "man_seg_000_001.tif", gt_mask)

    dataset = CTCAnnotatedPlaneDataset(tmp_path, sequence_ids=["01"])
    image, mask, metadata = dataset[0]

    assert len(dataset) == 1
    assert image.shape == (1, 16, 16)
    assert mask.shape == (1, 16, 16)
    assert image.dtype == torch.float32
    assert mask.dtype == torch.float32
    assert image.min() >= 0 and image.max() <= 1
    assert set(torch.unique(mask).tolist()) == {0.0, 1.0}
    assert metadata["frame"] == 0
    assert metadata["z_index"] == 1


def test_unet_forward_preserves_spatial_shape() -> None:
    model = UNet2D(base_channels=4)
    inputs = torch.randn(2, 1, 31, 33)

    outputs = model(inputs)

    assert outputs.shape == (2, 1, 31, 33)


def test_torch_dice_and_iou_on_toy_tensors() -> None:
    targets = torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]])
    logits = torch.where(targets > 0, torch.tensor(10.0), torch.tensor(-10.0))

    assert binary_dice_score(logits, targets).item() == pytest.approx(1.0)
    assert binary_iou_score(logits, targets).item() == pytest.approx(1.0)


def test_checkpoint_save_and_load(tmp_path: Path) -> None:
    model = UNet2D(base_channels=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    checkpoint_path = save_checkpoint(
        tmp_path / "checkpoint.pt",
        model,
        optimizer,
        epoch=3,
        best_validation_dice=0.75,
        config={"base_channels": 2},
    )
    restored_model = UNet2D(base_channels=2)
    restored_optimizer = torch.optim.Adam(restored_model.parameters(), lr=0.001)

    checkpoint = load_checkpoint(
        checkpoint_path,
        restored_model,
        restored_optimizer,
        device="cpu",
    )

    assert checkpoint["epoch"] == 3
    assert checkpoint["best_validation_dice"] == 0.75
    assert checkpoint["config"]["base_channels"] == 2
    for original, restored in zip(model.parameters(), restored_model.parameters()):
        assert torch.equal(original, restored)
