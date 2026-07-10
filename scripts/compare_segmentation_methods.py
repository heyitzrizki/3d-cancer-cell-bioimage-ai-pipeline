"""Compare classical segmentation and 2D U-Net on official CTC SEG planes."""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from bioimage_pipeline.deep_learning import CTCAnnotatedPlaneDataset
from bioimage_pipeline.evaluation.model_comparison import (
    compare_methods_on_dataset,
    load_unet_from_checkpoint,
    resolve_device,
    summarize_comparison,
    write_comparison_summary,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", required=True, type=Path)
    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        default=None,
        help="Optional trained 2D U-Net checkpoint. If omitted, only classical baseline is evaluated.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports") / "milestone_6",
    )
    parser.add_argument("--sequence-ids", nargs="+", default=["01", "02"])
    parser.add_argument(
        "--classical-method",
        choices=("otsu", "adaptive", "watershed"),
        default="otsu",
    )
    parser.add_argument("--min-size", type=int, default=64)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-overlays", type=int, default=3)
    args = parser.parse_args()
    if args.min_size < 0:
        parser.error("--min-size must be non-negative.")
    if not 0 <= args.threshold <= 1:
        parser.error("--threshold must be between 0 and 1.")
    if args.max_samples is not None and args.max_samples < 1:
        parser.error("--max-samples must be at least 1.")
    if args.max_overlays < 0:
        parser.error("--max-overlays must be non-negative.")
    return args


def main() -> None:
    args = parse_args()
    dataset_dir = args.dataset_dir.expanduser().resolve(strict=False)
    output_dir = args.output_dir.expanduser().resolve(strict=False)
    device = resolve_device(args.device)
    checkpoint_path = (
        args.checkpoint_path.expanduser().resolve(strict=False)
        if args.checkpoint_path is not None
        else None
    )

    dataset = CTCAnnotatedPlaneDataset(dataset_dir, args.sequence_ids)
    if len(dataset) == 0:
        raise ValueError("No compatible annotated planes were found.")

    unet_model = None
    if checkpoint_path is None:
        print("Warning: --checkpoint-path was not provided; evaluating classical baseline only.")
    elif not checkpoint_path.is_file():
        print(f"Warning: checkpoint not found ({checkpoint_path}); evaluating classical baseline only.")
    else:
        unet_model = load_unet_from_checkpoint(checkpoint_path, device=device)

    metrics, overlay_paths = compare_methods_on_dataset(
        dataset,
        classical_method=args.classical_method,
        min_size=args.min_size,
        unet_model=unet_model,
        device=device,
        threshold=args.threshold,
        max_samples=args.max_samples,
        overlay_dir=output_dir / "figures",
        max_overlays=args.max_overlays,
    )
    summary = summarize_comparison(metrics)

    metrics_dir = output_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = metrics_dir / "segmentation_method_comparison.csv"
    summary_path = metrics_dir / "segmentation_method_summary.csv"
    metrics.to_csv(metrics_path, index=False)
    summary.to_csv(summary_path, index=False)
    markdown_path = write_comparison_summary(
        output_dir / "milestone_6_summary.md",
        dataset_dir=dataset_dir,
        checkpoint_path=checkpoint_path,
        metrics=metrics,
        summary=summary,
        overlay_paths=overlay_paths,
    )

    print(f"Device: {device}")
    print(f"Annotated samples available: {len(dataset)}")
    print(f"Per-sample metrics: {metrics_path}")
    print(f"Method summary: {summary_path}")
    print(f"Markdown summary: {markdown_path}")
    for row in summary.itertuples(index=False):
        print(
            f"{row.method}: samples={int(row.samples)}, "
            f"mean_dice={row.mean_dice:.4f}, mean_iou={row.mean_iou:.4f}"
        )
    for overlay_path in overlay_paths:
        print(f"Overlay: {overlay_path}")


if __name__ == "__main__":
    main()
