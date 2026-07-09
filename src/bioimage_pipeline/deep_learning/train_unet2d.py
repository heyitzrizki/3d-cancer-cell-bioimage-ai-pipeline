"""Training utilities for the compact 2D U-Net."""

import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn


def set_random_seed(seed: int = 42) -> None:
    """Set Python, NumPy, and PyTorch random seeds."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def binary_dice_score(
    logits: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
) -> torch.Tensor:
    """Compute mean binary Dice from logits."""
    predictions = torch.sigmoid(logits) >= threshold
    targets_binary = targets > 0.5
    dimensions = tuple(range(1, predictions.ndim))
    intersection = (predictions & targets_binary).sum(dim=dimensions).float()
    denominator = predictions.sum(dim=dimensions) + targets_binary.sum(dim=dimensions)
    score = torch.where(
        denominator > 0,
        2.0 * intersection / denominator.float(),
        torch.ones_like(intersection),
    )
    return score.mean()


def binary_iou_score(
    logits: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
) -> torch.Tensor:
    """Compute mean binary IoU from logits."""
    predictions = torch.sigmoid(logits) >= threshold
    targets_binary = targets > 0.5
    dimensions = tuple(range(1, predictions.ndim))
    intersection = (predictions & targets_binary).sum(dim=dimensions).float()
    union = (predictions | targets_binary).sum(dim=dimensions)
    score = torch.where(
        union > 0,
        intersection / union.float(),
        torch.ones_like(intersection),
    )
    return score.mean()


def _run_epoch(
    model: nn.Module,
    data_loader: Any,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_dice = 0.0
    total_iou = 0.0
    total_samples = 0

    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for batch in data_loader:
            images, masks = batch[0].to(device), batch[1].to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, masks)
            if training:
                loss.backward()
                optimizer.step()

            batch_size = images.size(0)
            total_loss += float(loss.detach()) * batch_size
            total_dice += float(binary_dice_score(logits.detach(), masks)) * batch_size
            total_iou += float(binary_iou_score(logits.detach(), masks)) * batch_size
            total_samples += batch_size

    if total_samples == 0:
        return {"loss": float("nan"), "dice": float("nan"), "iou": float("nan")}
    return {
        "loss": total_loss / total_samples,
        "dice": total_dice / total_samples,
        "iou": total_iou / total_samples,
    }


def train_one_epoch(
    model: nn.Module,
    data_loader: Any,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> dict[str, float]:
    """Train for one epoch and return loss, Dice, and IoU."""
    return _run_epoch(model, data_loader, criterion, device, optimizer)


def validate_one_epoch(
    model: nn.Module,
    data_loader: Any,
    criterion: nn.Module,
    device: torch.device,
) -> dict[str, float]:
    """Evaluate for one epoch without updating model weights."""
    return _run_epoch(model, data_loader, criterion, device, optimizer=None)


def save_checkpoint(
    checkpoint_path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    *,
    epoch: int = 0,
    best_validation_dice: float = float("nan"),
    config: dict[str, Any] | None = None,
) -> Path:
    """Save model state and training metadata."""
    path = Path(checkpoint_path).expanduser().resolve(strict=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer else None,
        "epoch": epoch,
        "best_validation_dice": best_validation_dice,
        "config": config or {},
    }
    torch.save(payload, path)
    return path


def load_checkpoint(
    checkpoint_path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    device: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Load a checkpoint into a model and optional optimizer."""
    path = Path(checkpoint_path).expanduser().resolve(strict=False)
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None and checkpoint.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint
