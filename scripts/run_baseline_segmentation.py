"""Run a classical segmentation baseline on a small number of CTC frames."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from bioimage_pipeline.data import (
    find_segmentation_gt_files,
    find_tiff_files,
    get_segmentation_gt_slice_index,
    load_gt_mask,
    load_tiff_image,
    match_prediction_to_gt_frame,
)
from bioimage_pipeline.evaluation import (
    dice_coefficient,
    iou_score,
    object_count_error,
    precision_recall_f1,
)
from bioimage_pipeline.preprocessing import preprocess_volume
from bioimage_pipeline.segmentation import segment_volume_baseline
from bioimage_pipeline.visualization import save_side_by_side_segmentation

METRIC_NAMES = ("dice", "iou", "precision", "recall", "f1", "object_count_error")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", required=True, type=Path, help="CTC dataset root.")
    parser.add_argument("--sequence", default="01", help="Sequence identifier, such as 01.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports") / "milestone_2",
        help="Directory for overlays, masks, and metrics.",
    )
    parser.add_argument(
        "--method",
        choices=("otsu", "adaptive", "watershed"),
        default="otsu",
    )
    parser.add_argument("--max-frames", type=int, default=5)
    parser.add_argument("--min-size", type=int, default=64)
    parser.add_argument("--save-masks", action="store_true")
    args = parser.parse_args()
    if args.max_frames < 1:
        parser.error("--max-frames must be at least 1.")
    if args.min_size < 0:
        parser.error("--min-size must be non-negative.")
    return args


def _evaluate_matched_gt(
    binary_mask: np.ndarray,
    labeled_mask: np.ndarray,
    gt_paths: list[Path],
) -> dict[str, object] | None:
    true_binary_parts: list[np.ndarray] = []
    predicted_binary_parts: list[np.ndarray] = []
    count_errors: list[int] = []
    evaluation_scopes: set[str] = set()

    for gt_path in gt_paths:
        gt_mask = load_gt_mask(gt_path)
        if gt_mask.shape == binary_mask.shape:
            predicted_binary = binary_mask
            predicted_labels = labeled_mask
            evaluation_scopes.add("full_volume")
        elif binary_mask.ndim == 3 and gt_mask.ndim == 2:
            z_index = get_segmentation_gt_slice_index(gt_path)
            if z_index is None or not 0 <= z_index < binary_mask.shape[0]:
                print(f"Warning: cannot identify a valid z-slice for GT file {gt_path.name}.")
                continue
            predicted_binary = binary_mask[z_index]
            predicted_labels = labeled_mask[z_index]
            evaluation_scopes.add("annotated_2d_planes")
        else:
            print(
                f"Warning: skipping GT file {gt_path.name}; shape {gt_mask.shape} "
                f"does not align with prediction shape {binary_mask.shape}."
            )
            continue

        if gt_mask.shape != predicted_binary.shape:
            print(
                f"Warning: skipping GT file {gt_path.name}; shape {gt_mask.shape} "
                f"does not match evaluated prediction shape {predicted_binary.shape}."
            )
            continue

        true_binary_parts.append((gt_mask != 0).ravel())
        predicted_binary_parts.append((predicted_binary != 0).ravel())
        count_errors.append(object_count_error(gt_mask, predicted_labels))

    if not true_binary_parts:
        return None

    y_true = np.concatenate(true_binary_parts)
    y_pred = np.concatenate(predicted_binary_parts)
    classification_metrics = precision_recall_f1(y_true, y_pred)
    scope = (
        next(iter(evaluation_scopes))
        if len(evaluation_scopes) == 1
        else "mixed"
    )
    return {
        "evaluation_scope": scope,
        "evaluated_gt_files": len(true_binary_parts),
        "dice": dice_coefficient(y_true, y_pred),
        "iou": iou_score(y_true, y_pred),
        **classification_metrics,
        "object_count_error": float(np.mean(count_errors)),
    }


def _save_metrics(metrics: pd.DataFrame, output_dir: Path) -> tuple[Path, Path]:
    metrics_dir = output_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    csv_path = metrics_dir / "baseline_segmentation_metrics.csv"
    json_path = metrics_dir / "baseline_segmentation_metrics.json"

    metrics.to_csv(csv_path, index=False)
    json_records = (
        metrics.astype(object)
        .where(pd.notna(metrics), None)
        .to_dict(orient="records")
    )
    with json_path.open("w", encoding="utf-8") as file:
        json.dump(json_records, file, indent=2, allow_nan=False)
    return csv_path, json_path


def _frame_id_from_path(path: Path) -> str:
    digits = "".join(character for character in path.stem if character.isdigit())
    return digits or path.stem


def main() -> None:
    args = parse_args()
    dataset_dir = args.dataset_dir.expanduser().resolve(strict=False)
    output_dir = args.output_dir.expanduser().resolve(strict=False)
    sequence_id = str(args.sequence).zfill(2)
    sequence_dir = dataset_dir / sequence_id
    if not sequence_dir.is_dir():
        raise NotADirectoryError(f"Sequence directory not found: {sequence_dir}")

    frame_files = find_tiff_files(sequence_dir, recursive=False)[: args.max_frames]
    if not frame_files:
        raise FileNotFoundError(f"No TIFF frames found in sequence: {sequence_dir}")

    gt_files = find_segmentation_gt_files(dataset_dir, sequence_id)
    gt_matches = match_prediction_to_gt_frame(frame_files, gt_files)
    if not gt_files:
        print(
            f"Warning: no segmentation GT files found for sequence {sequence_id}; "
            "overlays will be saved and evaluation skipped."
        )

    overlays_dir = output_dir / "overlays"
    masks_dir = output_dir / "predicted_masks"
    rows: list[dict[str, object]] = []

    print(f"Dataset: {dataset_dir}")
    print(f"Sequence: {sequence_id}")
    print(f"Method: {args.method}")
    print(f"Frames selected: {len(frame_files)}")

    for frame_path in frame_files:
        frame_id = _frame_id_from_path(frame_path)
        print(f"Processing {frame_path.name}...")
        raw_image = load_tiff_image(frame_path)
        processed_image = preprocess_volume(raw_image)
        binary_mask, labeled_mask = segment_volume_baseline(
            processed_image,
            method=args.method,
            config={"min_size": args.min_size},
        )

        overlay_path = overlays_dir / f"{frame_path.stem}_{args.method}_overlay.png"
        save_side_by_side_segmentation(
            raw_image,
            processed_image,
            binary_mask,
            overlay_path,
        )

        mask_path: Path | None = None
        if args.save_masks:
            masks_dir.mkdir(parents=True, exist_ok=True)
            mask_path = masks_dir / f"mask{frame_id}.tif"
            output_dtype = np.uint16 if labeled_mask.max() <= np.iinfo(np.uint16).max else np.uint32
            tifffile.imwrite(
                mask_path,
                labeled_mask.astype(output_dtype),
                photometric="minisblack",
            )

        matched_gt = gt_matches.get(frame_path.resolve(), [])
        frame_metrics = _evaluate_matched_gt(binary_mask, labeled_mask, matched_gt)
        if not matched_gt:
            print(
                f"Warning: no matching segmentation GT for {frame_path.name}; "
                "evaluation skipped for this frame."
            )
        elif frame_metrics is None:
            print(
                f"Warning: matched GT for {frame_path.name} was incompatible; "
                "evaluation skipped for this frame."
            )
        row: dict[str, object] = {
            "dataset": dataset_dir.name,
            "sequence": sequence_id,
            "frame": frame_id,
            "source_file": str(frame_path),
            "method": args.method,
            "gt_available": frame_metrics is not None,
            "evaluation_scope": (
                frame_metrics["evaluation_scope"]
                if frame_metrics is not None
                else "not_evaluated"
            ),
            "evaluated_gt_files": (
                frame_metrics["evaluated_gt_files"]
                if frame_metrics is not None
                else 0
            ),
            "matched_gt_files": ";".join(path.name for path in matched_gt),
            "overlay_file": str(overlay_path),
            "mask_file": str(mask_path) if mask_path else "",
        }
        row.update(
            {
                metric_name: frame_metrics[metric_name]
                for metric_name in METRIC_NAMES
            }
            if frame_metrics is not None
            else {metric_name: float("nan") for metric_name in METRIC_NAMES}
        )
        rows.append(row)

    metrics = pd.DataFrame(rows)
    csv_path, json_path = _save_metrics(metrics, output_dir)

    evaluated = metrics[metrics["gt_available"]]
    if evaluated.empty:
        print("Warning: no compatible GT masks were available; evaluation was skipped.")
    else:
        print(f"Evaluated frames: {len(evaluated)}")
        scope_counts = evaluated["evaluation_scope"].value_counts()
        for scope, count in scope_counts.items():
            print(f"Evaluation scope {scope}: {count} frame(s)")
        for metric_name in METRIC_NAMES:
            values = pd.to_numeric(evaluated[metric_name], errors="coerce").dropna()
            if not values.empty:
                print(f"Mean {metric_name}: {values.mean():.4f}")

    print(f"Overlays: {overlays_dir}")
    if args.save_masks:
        print(f"Predicted masks: {masks_dir}")
    print(f"Metrics CSV: {csv_path}")
    print(f"Metrics JSON: {json_path}")


if __name__ == "__main__":
    main()
