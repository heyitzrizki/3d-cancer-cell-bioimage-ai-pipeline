"""Train a compact 2D U-Net on official sparse CTC SEG planes."""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset, random_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from bioimage_pipeline.deep_learning import (
    CTCAnnotatedPlaneDataset,
    UNet2D,
    evaluate_loader,
    load_checkpoint,
    predict_mask,
    save_checkpoint,
    set_random_seed,
    train_one_epoch,
    validate_one_epoch,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("data") / "raw" / "Fluo-C3DL-MDA231",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports") / "milestone_5",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("checkpoints") / "milestone_5",
    )
    parser.add_argument("--sequence-ids", nargs="+", default=["01", "02"])
    parser.add_argument("--val-fraction", type=float, default=0.25)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--base-channels", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()
    if not 0 < args.val_fraction < 1:
        parser.error("--val-fraction must be between 0 and 1.")
    for name in ("epochs", "batch_size", "base_channels"):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be at least 1.")
    if args.learning_rate <= 0:
        parser.error("--learning-rate must be positive.")
    if args.max_samples is not None and args.max_samples < 2:
        parser.error("--max-samples must be at least 2.")
    return args


def _resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    return device


def _portable_path(path: Path) -> str:
    try:
        return path.resolve(strict=False).relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(path.resolve(strict=False))


def _save_loss_curve(history: pd.DataFrame, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(7, 5))
    axis.plot(history["epoch"], history["train_loss"], label="Train")
    axis.plot(history["epoch"], history["val_loss"], label="Validation")
    axis.set_xlabel("Epoch")
    axis.set_ylabel("BCEWithLogitsLoss")
    axis.set_title("2D U-Net Training Loss")
    axis.legend()
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    return output_path


def _save_prediction_overlay(
    model: nn.Module,
    validation_dataset: Subset,
    device: torch.device,
    output_path: Path,
) -> Path:
    image, mask, _ = validation_dataset[0]
    prediction, _ = predict_mask(model, image, device=device)
    image_plane = image.squeeze().numpy()
    mask_plane = mask.squeeze().numpy()
    prediction_plane = prediction.squeeze().numpy()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(image_plane, cmap="gray")
    axes[0].set_title("Input")
    axes[1].imshow(image_plane, cmap="gray")
    axes[1].imshow(
        np.ma.masked_where(mask_plane == 0, mask_plane),
        cmap="Reds",
        alpha=0.45,
        vmin=0,
        vmax=1,
    )
    axes[1].set_title("Official GT")
    axes[2].imshow(image_plane, cmap="gray")
    axes[2].imshow(
        np.ma.masked_where(prediction_plane == 0, prediction_plane),
        cmap="Reds",
        alpha=0.45,
        vmin=0,
        vmax=1,
    )
    axes[2].set_title("Prediction")
    for axis in axes:
        axis.set_axis_off()
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    return output_path


def _write_summary(
    output_path: Path,
    *,
    dataset_dir: Path,
    sequences: list[str],
    sample_count: int,
    train_size: int,
    val_size: int,
    epochs: int,
    batch_size: int,
    device: torch.device,
    best_dice: float,
    best_iou: float,
    checkpoint_path: Path,
) -> Path:
    content = f"""# 2D U-Net Training Summary

- Dataset: `{dataset_dir.name}`
- Sequences: `{", ".join(sequences)}`
- Annotated 2D samples: {sample_count}
- Training samples: {train_size}
- Validation samples: {val_size}
- Epochs: {epochs}
- Batch size: {batch_size}
- Device: `{device}`
- Best validation Dice: {best_dice:.6f}
- Validation IoU at best Dice: {best_iou:.6f}
- Best checkpoint: `{_portable_path(checkpoint_path)}`

## Limitations

- Training uses sparse annotated 2D planes.
- This is not full 3D segmentation.
- The supervised dataset is limited.
- Hyperparameters have not been systematically searched.
"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(content, encoding="utf-8")
    return output_path


def main() -> None:
    args = parse_args()
    set_random_seed(args.seed)
    device = _resolve_device(args.device)
    dataset_dir = args.dataset_dir.expanduser().resolve(strict=False)
    output_dir = args.output_dir.expanduser().resolve(strict=False)
    checkpoint_dir = args.checkpoint_dir.expanduser().resolve(strict=False)

    full_dataset = CTCAnnotatedPlaneDataset(dataset_dir, args.sequence_ids)
    if len(full_dataset) < 2:
        raise ValueError("At least two compatible annotated planes are required.")
    dataset = (
        Subset(full_dataset, range(min(args.max_samples, len(full_dataset))))
        if args.max_samples is not None
        else full_dataset
    )
    sample_count = len(dataset)
    val_size = max(1, round(sample_count * args.val_fraction))
    val_size = min(val_size, sample_count - 1)
    train_size = sample_count - val_size
    generator = torch.Generator().manual_seed(args.seed)
    train_dataset, validation_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=generator,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    model = UNet2D(base_channels=args.base_channels).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = nn.BCEWithLogitsLoss()
    checkpoint_path = checkpoint_dir / "unet_2d_best.pt"
    history_rows = []
    best_dice = -1.0
    best_iou = float("nan")

    print(f"Device: {device}")
    print(f"Annotated samples: {sample_count} ({train_size} train, {val_size} validation)")
    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
        )
        validation_metrics = validate_one_epoch(
            model,
            validation_loader,
            criterion,
            device,
        )
        history_rows.append(
            {
                "epoch": epoch,
                "train_loss": train_metrics["loss"],
                "train_dice": train_metrics["dice"],
                "train_iou": train_metrics["iou"],
                "val_loss": validation_metrics["loss"],
                "val_dice": validation_metrics["dice"],
                "val_iou": validation_metrics["iou"],
            }
        )
        print(
            f"Epoch {epoch:03d}/{args.epochs}: "
            f"train_loss={train_metrics['loss']:.4f}, "
            f"val_loss={validation_metrics['loss']:.4f}, "
            f"val_dice={validation_metrics['dice']:.4f}"
        )
        if validation_metrics["dice"] > best_dice:
            best_dice = validation_metrics["dice"]
            best_iou = validation_metrics["iou"]
            save_checkpoint(
                checkpoint_path,
                model,
                optimizer,
                epoch=epoch,
                best_validation_dice=best_dice,
                config={
                    "base_channels": args.base_channels,
                    "in_channels": 1,
                    "out_channels": 1,
                    "sequence_ids": [str(value).zfill(2) for value in args.sequence_ids],
                    "normalization_lower": 1.0,
                    "normalization_upper": 99.0,
                },
            )

    history = pd.DataFrame(history_rows)
    metrics_dir = output_dir / "metrics"
    figures_dir = output_dir / "figures"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    history_path = metrics_dir / "unet_2d_training_history.csv"
    history.to_csv(history_path, index=False)

    load_checkpoint(checkpoint_path, model, device=device)
    validation_metrics = evaluate_loader(model, validation_loader, device)
    validation_metrics_path = metrics_dir / "unet_2d_metrics.csv"
    validation_metrics.to_csv(validation_metrics_path, index=False)
    loss_curve_path = _save_loss_curve(
        history,
        figures_dir / "training_loss_curve.png",
    )
    overlay_path = _save_prediction_overlay(
        model,
        validation_dataset,
        device,
        figures_dir / "unet_2d_prediction_overlay.png",
    )
    summary_path = _write_summary(
        output_dir / "unet_2d_summary.md",
        dataset_dir=dataset_dir,
        sequences=[str(value).zfill(2) for value in args.sequence_ids],
        sample_count=sample_count,
        train_size=train_size,
        val_size=val_size,
        epochs=args.epochs,
        batch_size=args.batch_size,
        device=device,
        best_dice=best_dice,
        best_iou=best_iou,
        checkpoint_path=checkpoint_path,
    )

    print(f"Best validation Dice: {best_dice:.4f}")
    print(f"Best validation IoU: {best_iou:.4f}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"History: {history_path}")
    print(f"Validation metrics: {validation_metrics_path}")
    print(f"Loss curve: {loss_curve_path}")
    print(f"Prediction overlay: {overlay_path}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
