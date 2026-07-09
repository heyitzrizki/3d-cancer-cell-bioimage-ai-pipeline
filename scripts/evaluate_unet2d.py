"""Evaluate a trained 2D U-Net on official CTC annotated planes."""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

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
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", required=True, type=Path)
    parser.add_argument("--checkpoint-path", required=True, type=Path)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports") / "milestone_5_eval",
    )
    parser.add_argument("--sequence-ids", nargs="+", default=["01", "02"])
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def _resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    return device


def _save_overlay(
    image: torch.Tensor,
    target: torch.Tensor,
    prediction: torch.Tensor,
    output_path: Path,
) -> Path:
    image_plane = image.squeeze().numpy()
    target_plane = target.squeeze().numpy()
    prediction_plane = prediction.squeeze().numpy()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 3, figsize=(15, 5))
    for axis, overlay, title in zip(
        axes,
        (None, target_plane, prediction_plane),
        ("Input", "Official GT", "Prediction"),
    ):
        axis.imshow(image_plane, cmap="gray")
        if overlay is not None:
            axis.imshow(
                np.ma.masked_where(overlay == 0, overlay),
                cmap="Reds",
                alpha=0.45,
                vmin=0,
                vmax=1,
            )
        axis.set_title(title)
        axis.set_axis_off()
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    return output_path


def main() -> None:
    args = parse_args()
    device = _resolve_device(args.device)
    checkpoint_path = args.checkpoint_path.expanduser().resolve(strict=False)
    output_dir = args.output_dir.expanduser().resolve(strict=False)
    checkpoint_metadata = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )
    config = checkpoint_metadata.get("config", {})
    model = UNet2D(
        in_channels=int(config.get("in_channels", 1)),
        out_channels=int(config.get("out_channels", 1)),
        base_channels=int(config.get("base_channels", 16)),
    ).to(device)
    load_checkpoint(checkpoint_path, model, device=device)

    dataset = CTCAnnotatedPlaneDataset(args.dataset_dir, args.sequence_ids)
    if not dataset:
        raise ValueError("No compatible annotated planes were found.")
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    metrics = evaluate_loader(model, loader, device)
    metrics_dir = output_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = metrics_dir / "unet_2d_evaluation_metrics.csv"
    metrics.to_csv(metrics_path, index=False)

    figure_paths = []
    for index in range(min(3, len(dataset))):
        image, target, _ = dataset[index]
        prediction, _ = predict_mask(model, image, device=device)
        figure_paths.append(
            _save_overlay(
                image,
                target,
                prediction,
                output_dir / "figures" / f"unet_2d_overlay_{index:03d}.png",
            )
        )

    print(f"Device: {device}")
    print(f"Annotated samples: {len(dataset)}")
    print(f"Mean Dice: {metrics['dice'].mean():.4f}")
    print(f"Mean IoU: {metrics['iou'].mean():.4f}")
    print(f"Metrics: {metrics_path}")
    for figure_path in figure_paths:
        print(f"Overlay: {figure_path}")


if __name__ == "__main__":
    main()
