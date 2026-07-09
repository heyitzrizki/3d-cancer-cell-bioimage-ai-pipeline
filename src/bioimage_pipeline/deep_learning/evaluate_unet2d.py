"""Evaluation helpers for annotated 2D planes."""

from typing import Any

import pandas as pd
import torch
from torch import nn

from .train_unet2d import binary_dice_score, binary_iou_score


def _metadata_value(metadata: dict[str, Any], key: str, index: int) -> Any:
    value = metadata.get(key)
    if isinstance(value, torch.Tensor):
        return value[index].item()
    if isinstance(value, (list, tuple)):
        return value[index]
    return value


def evaluate_loader(
    model: nn.Module,
    data_loader: Any,
    device: torch.device,
) -> pd.DataFrame:
    """Return one Dice/IoU row per annotated sample."""
    model.eval()
    rows: list[dict[str, Any]] = []
    with torch.no_grad():
        for images, masks, metadata in data_loader:
            images = images.to(device)
            masks = masks.to(device)
            logits = model(images)
            for index in range(images.size(0)):
                sample_logits = logits[index : index + 1]
                sample_mask = masks[index : index + 1]
                rows.append(
                    {
                        "dataset": _metadata_value(metadata, "dataset", index),
                        "sequence": _metadata_value(metadata, "sequence", index),
                        "frame": _metadata_value(metadata, "frame", index),
                        "z_index": _metadata_value(metadata, "z_index", index),
                        "raw_path": _metadata_value(metadata, "raw_path", index),
                        "gt_path": _metadata_value(metadata, "gt_path", index),
                        "dice": float(binary_dice_score(sample_logits, sample_mask)),
                        "iou": float(binary_iou_score(sample_logits, sample_mask)),
                    }
                )
    return pd.DataFrame(rows)
