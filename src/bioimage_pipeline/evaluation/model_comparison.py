"""Compare classical segmentation and 2D U-Net predictions on CTC GT planes."""

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from bioimage_pipeline.deep_learning import (
    CTCAnnotatedPlaneDataset,
    UNet2D,
    load_checkpoint,
    predict_mask,
)
from bioimage_pipeline.evaluation import (
    dice_coefficient,
    iou_score,
    object_count_error,
    precision_recall_f1,
)
from bioimage_pipeline.segmentation import segment_volume_baseline


def _tensor_to_plane(tensor: torch.Tensor) -> np.ndarray:
    """Convert a [1, H, W] or [H, W] tensor to a NumPy plane."""
    array = tensor.detach().cpu().squeeze().numpy()
    if array.ndim != 2:
        raise ValueError(f"Expected a 2D plane after squeezing, received {array.shape}.")
    return array


def _metric_row(
    *,
    method: str,
    metadata: dict[str, Any],
    target: np.ndarray,
    prediction: np.ndarray,
    prediction_labels: np.ndarray | None = None,
) -> dict[str, Any]:
    """Build one comparison row from target and predicted binary masks."""
    target_binary = np.asarray(target) != 0
    prediction_binary = np.asarray(prediction) != 0
    classification = precision_recall_f1(target_binary, prediction_binary)
    labels = (
        prediction_labels
        if prediction_labels is not None
        else prediction_binary.astype(np.uint8)
    )
    return {
        "dataset": metadata["dataset"],
        "sequence": str(metadata["sequence"]).zfill(2),
        "frame": int(metadata["frame"]),
        "z_index": int(metadata["z_index"]),
        "method": method,
        "evaluation_scope": "annotated_2d_planes",
        "dice": dice_coefficient(target_binary, prediction_binary),
        "iou": iou_score(target_binary, prediction_binary),
        "precision": classification["precision"],
        "recall": classification["recall"],
        "f1": classification["f1"],
        "object_count_error": object_count_error(target_binary, labels),
        "gt_foreground_pixels": int(target_binary.sum()),
        "predicted_foreground_pixels": int(prediction_binary.sum()),
        "raw_path": metadata["raw_path"],
        "gt_path": metadata["gt_path"],
    }


def resolve_device(requested: str) -> torch.device:
    """Resolve a requested device string into a safe PyTorch device."""
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    return device


def load_unet_from_checkpoint(
    checkpoint_path: str | Path,
    device: str | torch.device = "cpu",
) -> UNet2D:
    """Load a 2D U-Net using architecture metadata saved in the checkpoint."""
    checkpoint_path = Path(checkpoint_path).expanduser().resolve(strict=False)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    torch_device = torch.device(device)
    checkpoint = torch.load(checkpoint_path, map_location=torch_device, weights_only=False)
    config = checkpoint.get("config", {})
    model = UNet2D(
        in_channels=int(config.get("in_channels", 1)),
        out_channels=int(config.get("out_channels", 1)),
        base_channels=int(config.get("base_channels", 16)),
    ).to(torch_device)
    load_checkpoint(checkpoint_path, model, device=torch_device)
    model.eval()
    return model


def evaluate_classical_sample(
    image: torch.Tensor,
    target: torch.Tensor,
    metadata: dict[str, Any],
    *,
    method: str = "otsu",
    min_size: int = 64,
) -> tuple[dict[str, Any], np.ndarray]:
    """Evaluate one classical segmentation prediction on an annotated plane."""
    image_plane = _tensor_to_plane(image)
    target_plane = _tensor_to_plane(target)
    binary_mask, labeled_mask = segment_volume_baseline(
        image_plane,
        method=method,
        config={"min_size": min_size},
    )
    row = _metric_row(
        method=f"classical_{method}",
        metadata=metadata,
        target=target_plane,
        prediction=binary_mask,
        prediction_labels=labeled_mask,
    )
    return row, np.asarray(binary_mask, dtype=bool)


def evaluate_unet_sample(
    model: torch.nn.Module,
    image: torch.Tensor,
    target: torch.Tensor,
    metadata: dict[str, Any],
    *,
    device: str | torch.device = "cpu",
    threshold: float = 0.5,
) -> tuple[dict[str, Any], np.ndarray]:
    """Evaluate one 2D U-Net prediction on an annotated plane."""
    target_plane = _tensor_to_plane(target)
    prediction, _ = predict_mask(model, image, device=device, threshold=threshold)
    prediction_plane = _tensor_to_plane(prediction)
    row = _metric_row(
        method="unet_2d",
        metadata=metadata,
        target=target_plane,
        prediction=prediction_plane,
    )
    return row, np.asarray(prediction_plane, dtype=bool)


def summarize_comparison(metrics: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-plane metrics into one summary row per method."""
    metric_columns = ["dice", "iou", "precision", "recall", "f1", "object_count_error"]
    summary = (
        metrics.groupby("method", dropna=False)
        .agg(
            samples=("dice", "count"),
            **{
                f"mean_{column}": (column, "mean")
                for column in metric_columns
            },
        )
        .reset_index()
    )
    return summary


def save_comparison_overlay(
    *,
    image: torch.Tensor,
    target: torch.Tensor,
    classical_prediction: np.ndarray,
    unet_prediction: np.ndarray | None,
    output_path: str | Path,
) -> Path:
    """Save Input | GT | Classical | U-Net overlay for one annotated plane."""
    image_plane = _tensor_to_plane(image)
    target_plane = _tensor_to_plane(target)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    panels: list[tuple[str, np.ndarray | None]] = [
        ("Input", None),
        ("Official GT", target_plane != 0),
        ("Classical", classical_prediction != 0),
    ]
    if unet_prediction is not None:
        panels.append(("2D U-Net", unet_prediction != 0))

    figure, axes = plt.subplots(1, len(panels), figsize=(5 * len(panels), 5))
    if len(panels) == 1:
        axes = [axes]
    for axis, (title, overlay) in zip(axes, panels):
        axis.imshow(image_plane, cmap="gray")
        if overlay is not None:
            axis.imshow(
                np.ma.masked_where(~overlay, overlay),
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


def compare_methods_on_dataset(
    dataset: CTCAnnotatedPlaneDataset,
    *,
    classical_method: str = "otsu",
    min_size: int = 64,
    unet_model: torch.nn.Module | None = None,
    device: str | torch.device = "cpu",
    threshold: float = 0.5,
    max_samples: int | None = None,
    overlay_dir: str | Path | None = None,
    max_overlays: int = 3,
) -> tuple[pd.DataFrame, list[Path]]:
    """Evaluate available methods on a CTC annotated-plane dataset."""
    if len(dataset) == 0:
        raise ValueError("No compatible annotated planes were found.")
    sample_count = len(dataset) if max_samples is None else min(max_samples, len(dataset))
    rows: list[dict[str, Any]] = []
    overlay_paths: list[Path] = []

    for index in range(sample_count):
        image, target, metadata = dataset[index]
        classical_row, classical_prediction = evaluate_classical_sample(
            image,
            target,
            metadata,
            method=classical_method,
            min_size=min_size,
        )
        rows.append(classical_row)

        unet_prediction: np.ndarray | None = None
        if unet_model is not None:
            unet_row, unet_prediction = evaluate_unet_sample(
                unet_model,
                image,
                target,
                metadata,
                device=device,
                threshold=threshold,
            )
            rows.append(unet_row)

        if overlay_dir is not None and index < max_overlays:
            overlay_paths.append(
                save_comparison_overlay(
                    image=image,
                    target=target,
                    classical_prediction=classical_prediction,
                    unet_prediction=unet_prediction,
                    output_path=Path(overlay_dir) / f"comparison_overlay_{index:03d}.png",
                )
            )

    return pd.DataFrame(rows), overlay_paths


def write_comparison_summary(
    output_path: str | Path,
    *,
    dataset_dir: str | Path,
    checkpoint_path: str | Path | None,
    metrics: pd.DataFrame,
    summary: pd.DataFrame,
    overlay_paths: list[Path],
) -> Path:
    """Write a concise Markdown summary for Milestone 6."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_text = str(checkpoint_path) if checkpoint_path else "not provided"
    method_lines = [
        (
            f"- `{row.method}`: {int(row.samples)} samples, "
            f"mean Dice={row.mean_dice:.6f}, mean IoU={row.mean_iou:.6f}"
        )
        for row in summary.itertuples(index=False)
    ]
    overlay_lines = [f"- `{path}`" for path in overlay_paths] or ["- No overlays saved."]
    content = f"""# Milestone 6: Model Evaluation and Baseline Comparison

- Dataset: `{Path(dataset_dir).name}`
- Checkpoint: `{checkpoint_text}`
- Evaluation scope: sparse official 2D SEG planes only
- Per-sample rows: {len(metrics)}

## Method Summary

{chr(10).join(method_lines)}

## Saved Overlays

{chr(10).join(overlay_lines)}

## Notes

- Classical segmentation and 2D U-Net are evaluated on the same annotated planes.
- Metrics are binary foreground metrics and do not validate full 3D instance tracking.
- Results depend on the checkpoint and should be regenerated whenever training changes.
"""
    output_path.write_text(content, encoding="utf-8")
    return output_path
