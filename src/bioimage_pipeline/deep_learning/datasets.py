"""PyTorch datasets for sparse CTC annotated 2D planes."""

import re
import warnings
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import tifffile
import torch
import torch.nn.functional as functional
from torch.utils.data import Dataset

from bioimage_pipeline.data import (
    find_segmentation_gt_files,
    find_tiff_files,
    load_gt_mask,
    load_tiff_image,
)
from bioimage_pipeline.preprocessing import normalize_intensity

_SEGMENTATION_NAME_PATTERN = re.compile(
    r"^man_seg_?(\d+)(?:_(\d+))?$",
    re.IGNORECASE,
)
_RAW_FRAME_PATTERN = re.compile(r"^t(\d+)$", re.IGNORECASE)


def parse_ctc_segmentation_filename(path: str | Path) -> tuple[int, int | None]:
    """Parse frame and optional z indices from a CTC SEG filename."""
    stem = Path(path).stem
    match = _SEGMENTATION_NAME_PATTERN.fullmatch(stem)
    if not match:
        raise ValueError(f"Unsupported CTC segmentation filename: {Path(path).name}")
    frame_index = int(match.group(1))
    z_index = int(match.group(2)) if match.group(2) is not None else None
    return frame_index, z_index


def _tiff_shape(path: Path) -> tuple[int, ...]:
    with tifffile.TiffFile(path) as tiff:
        return tuple(int(value) for value in tiff.series[0].shape)


class CTCAnnotatedPlaneDataset(Dataset):
    """Create supervised image/mask pairs from official sparse CTC SEG planes."""

    def __init__(
        self,
        dataset_dir: str | Path,
        sequence_ids: Sequence[str | int] = ("01", "02"),
        *,
        resize: tuple[int, int] | None = None,
        crop_size: tuple[int, int] | None = None,
        transform: Callable[
            [torch.Tensor, torch.Tensor],
            tuple[torch.Tensor, torch.Tensor],
        ]
        | None = None,
        normalization_lower: float = 1.0,
        normalization_upper: float = 99.0,
    ) -> None:
        self.dataset_dir = Path(dataset_dir).expanduser().resolve(strict=False)
        if not self.dataset_dir.is_dir():
            raise NotADirectoryError(f"Dataset directory not found: {self.dataset_dir}")
        self.sequence_ids = [str(sequence).zfill(2) for sequence in sequence_ids]
        self.resize = resize
        self.crop_size = crop_size
        self.transform = transform
        self.normalization_lower = normalization_lower
        self.normalization_upper = normalization_upper
        self.samples = self._build_sample_index()

    def _raw_frames(self, sequence: str) -> dict[int, Path]:
        sequence_dir = self.dataset_dir / sequence
        if not sequence_dir.is_dir():
            warnings.warn(f"Sequence directory not found: {sequence_dir}", stacklevel=2)
            return {}
        frames: dict[int, Path] = {}
        for path in find_tiff_files(sequence_dir, recursive=False):
            match = _RAW_FRAME_PATTERN.fullmatch(path.stem)
            if match:
                frames[int(match.group(1))] = path
        return frames

    def _build_sample_index(self) -> list[dict[str, Any]]:
        samples: list[dict[str, Any]] = []
        for sequence in self.sequence_ids:
            raw_frames = self._raw_frames(sequence)
            gt_files = find_segmentation_gt_files(self.dataset_dir, sequence)
            if not gt_files:
                warnings.warn(
                    f"No official SEG ground truth found for sequence {sequence}.",
                    stacklevel=2,
                )
                continue

            for gt_path in gt_files:
                try:
                    frame_index, z_index = parse_ctc_segmentation_filename(gt_path)
                except ValueError as error:
                    warnings.warn(str(error), stacklevel=2)
                    continue
                raw_path = raw_frames.get(frame_index)
                if raw_path is None:
                    warnings.warn(
                        f"No raw frame t{frame_index:03d}.tif for {gt_path.name}; skipping.",
                        stacklevel=2,
                    )
                    continue

                raw_shape = _tiff_shape(raw_path)
                gt_shape = _tiff_shape(gt_path)
                valid = False
                if len(raw_shape) == 3 and len(gt_shape) == 2:
                    valid = z_index is not None and 0 <= z_index < raw_shape[0]
                elif len(raw_shape) == 2 and len(gt_shape) == 2:
                    valid = raw_shape == gt_shape

                if not valid:
                    warnings.warn(
                        f"Incompatible raw/GT sample {raw_path.name} {raw_shape} and "
                        f"{gt_path.name} {gt_shape}; skipping.",
                        stacklevel=2,
                    )
                    continue
                if gt_shape != raw_shape[-2:]:
                    warnings.warn(
                        f"Spatial shape mismatch for {gt_path.name}: "
                        f"{gt_shape} != {raw_shape[-2:]}; skipping.",
                        stacklevel=2,
                    )
                    continue

                samples.append(
                    {
                        "dataset": self.dataset_dir.name,
                        "sequence": sequence,
                        "frame": frame_index,
                        "z_index": z_index,
                        "raw_path": raw_path,
                        "gt_path": gt_path,
                    }
                )
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    @staticmethod
    def _center_crop(
        image: torch.Tensor,
        mask: torch.Tensor,
        crop_size: tuple[int, int],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        crop_height, crop_width = crop_size
        _, height, width = image.shape
        if crop_height > height or crop_width > width:
            raise ValueError(
                f"crop_size {crop_size} exceeds image size {(height, width)}."
            )
        top = (height - crop_height) // 2
        left = (width - crop_width) // 2
        slices = (slice(None), slice(top, top + crop_height), slice(left, left + crop_width))
        return image[slices], mask[slices]

    def __getitem__(
        self,
        index: int,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        sample = self.samples[index]
        raw = load_tiff_image(sample["raw_path"])
        gt = load_gt_mask(sample["gt_path"])
        if raw.ndim == 3:
            image_plane = raw[sample["z_index"]]
        else:
            image_plane = raw
        if image_plane.shape != gt.shape:
            raise ValueError(
                f"Image plane and GT mask shapes do not match: "
                f"{image_plane.shape} != {gt.shape}."
            )

        normalized = normalize_intensity(
            image_plane,
            method="percentile",
            lower=self.normalization_lower,
            upper=self.normalization_upper,
        )
        image_tensor = torch.from_numpy(normalized).unsqueeze(0).float()
        mask_tensor = torch.from_numpy((np.asarray(gt) != 0).astype(np.float32)).unsqueeze(0)

        if self.crop_size is not None:
            image_tensor, mask_tensor = self._center_crop(
                image_tensor,
                mask_tensor,
                self.crop_size,
            )
        if self.resize is not None:
            image_tensor = functional.interpolate(
                image_tensor.unsqueeze(0),
                size=self.resize,
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
            mask_tensor = functional.interpolate(
                mask_tensor.unsqueeze(0),
                size=self.resize,
                mode="nearest",
            ).squeeze(0)
        if self.transform is not None:
            image_tensor, mask_tensor = self.transform(image_tensor, mask_tensor)

        metadata = {
            "dataset": sample["dataset"],
            "sequence": sample["sequence"],
            "frame": sample["frame"],
            "z_index": sample["z_index"] if sample["z_index"] is not None else -1,
            "raw_path": str(sample["raw_path"]),
            "gt_path": str(sample["gt_path"]),
        }
        return image_tensor, mask_tensor, metadata
