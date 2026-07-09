"""Inference helpers for 2D U-Net segmentation."""

import torch
from torch import nn


def predict_mask(
    model: nn.Module,
    image: torch.Tensor,
    device: str | torch.device = "cpu",
    threshold: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return binary prediction and sigmoid probability tensors on CPU."""
    model.eval()
    image_batch = image.unsqueeze(0) if image.ndim == 3 else image
    if image_batch.ndim != 4:
        raise ValueError("image must have shape [C,H,W] or [N,C,H,W].")
    with torch.no_grad():
        probabilities = torch.sigmoid(model(image_batch.to(device)))
    binary_mask = probabilities >= threshold
    return binary_mask.cpu(), probabilities.cpu()
