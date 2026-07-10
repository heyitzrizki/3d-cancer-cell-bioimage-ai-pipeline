# 3D Cancer Cell Bioimage AI Pipeline

## Project Overview

This repository provides a reproducible foundation for analyzing 3D+time fluorescence microscopy data from the Cell Tracking Challenge. It separates version-controlled code and configuration from large datasets and model artifacts, and supports both local development and Google Colab workflows.

The primary dataset is **Fluo-C3DL-MDA231**. **Fluo-C3DH-A549** is reserved for later external validation.

## Motivation

Reliable cancer-cell bioimage analysis requires more than a model: data organization, preprocessing, evaluation, experiment configuration, and reproducible execution are equally important. This project develops those components incrementally, beginning with an interpretable classical image-analysis baseline before introducing deep learning.

## Dataset

The project targets these public Cell Tracking Challenge datasets:

- **Fluo-C3DL-MDA231**: main development and MVP dataset.
- **Fluo-C3DH-A549**: external-validation dataset for later phases.

The utilities recognize common CTC layouts such as `01/`, `02/`, `01_GT/SEG/`, `01_GT/TRA/`, `02_GT/SEG/`, and `02_GT/TRA/`.

No raw microscopy data is tracked in this repository. Obtain the datasets from the official Cell Tracking Challenge source and follow its terms of use.

## Repository Structure

```text
.
├── colab/                  # Colab setup and experiment notebooks
├── configs/                # Path and dataset experiment configurations
├── data/                   # Documentation only; raw data is external
├── notebooks/              # Local exploratory and baseline notebooks
├── reports/                # Small figures and metric summaries
├── scripts/                # Command-line entry points
├── src/bioimage_pipeline/  # Reusable Python package
└── tests/                  # Lightweight validation tests
```

## Data Storage Strategy

Code, small configuration files, notebooks, and documentation belong in the local/GitHub repository. Raw datasets, processed volumes, predictions, checkpoints, and logs belong outside Git, preferably in Google Drive.

The Colab configuration uses:

```text
/content/drive/MyDrive/self learning/Self Project/3d-cancer-cell-bioimage-ai-pipeline-data
```

Paths are stored in YAML and handled with `pathlib`, so spaces in Google Drive folder names are safe. Copy an example path configuration to a non-example filename if you need machine-specific values; local configuration files matching `configs/paths.*.yaml` are ignored.

## Local Setup

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
```

Activate the environment, then install the project:

```bash
python -m pip install --upgrade pip
python -m pip install -e .
```

Copy `configs/paths.local.example.yaml` to `configs/paths.local.yaml`, replace the placeholders with paths on your machine, and inspect a dataset:

```bash
python scripts/inspect_dataset.py --dataset-dir "PATH/TO/Fluo-C3DL-MDA231"
```

Local datasets under `data/raw/` are ignored for safety. Keep large data there for local exploration or use the configured external data root.

## Google Colab Setup

Open `colab/00_colab_setup.ipynb`, replace `YOUR_USERNAME` in the repository URL, and run the cells in order. The notebook mounts Drive, clones the code repository, installs dependencies, defines `DATA_ROOT`, creates the external directory tree, and inspects the primary dataset.

The equivalent setup command is:

```bash
python scripts/setup_drive_dirs.py \
  --data-root "/content/drive/MyDrive/self learning/Self Project/3d-cancer-cell-bioimage-ai-pipeline-data"
```

## Milestone 1: Dataset Inspection and Visualization

Milestone 1 adds reusable utilities for discovering and loading TIFF images, summarizing Cell Tracking Challenge sequence and annotation folders, calculating basic image statistics, and visualizing 2D images or 3D volumes. It does not perform preprocessing, segmentation, or model training.

For local exploration, place the ignored datasets at:

```text
data/raw/Fluo-C3DL-MDA231
data/raw/Fluo-C3DH-A549
```

Inspect the primary dataset and print a pandas DataFrame summary:

```bash
python scripts/inspect_dataset.py \
  --dataset-dir data/raw/Fluo-C3DL-MDA231
```

Inspect the dataset, load its first sequence image, and save a middle z-slice and max intensity projection:

```bash
python scripts/explore_dataset.py \
  --dataset-dir data/raw/Fluo-C3DL-MDA231 \
  --output-dir reports/figures
```

The exploration notebook at `notebooks/01_data_exploration.ipynb` applies the same workflow to both datasets. Generated figures remain local unless deliberately selected for version control.

## Milestone 2: Classical Segmentation Baseline

A classical baseline establishes an interpretable reference before deep learning. It exposes data-quality, preprocessing, annotation-coverage, and evaluation issues while keeping runtime and model complexity low.

The baseline applies percentile intensity normalization, Gaussian denoising, smooth background subtraction, and final rescaling. Gaussian settings accept either one scalar or explicit per-axis values such as `(sigma_z, sigma_y, sigma_x)`, allowing later experiments to account for anisotropic voxel spacing without hardcoding unverified spacing assumptions. Otsu thresholding is the default segmentation method, followed by small-object removal, small-hole filling, and connected-component labeling. Adaptive thresholding and watershed are available as optional comparisons.

When compatible CTC segmentation ground truth is available, the runner calculates Dice, IoU, precision, recall, F1, and absolute object-count error. MDA231's sparsely annotated 2D GT planes are evaluated only against their corresponding prediction planes. The metrics output records `evaluation_scope` and `evaluated_gt_files`, so sparse-plane results cannot be mistaken for full-volume evaluation. Missing or incompatible GT is reported clearly and does not prevent overlays or predictions from being saved.

Run the default five-frame baseline:

```bash
python scripts/run_baseline_segmentation.py \
  --dataset-dir data/raw/Fluo-C3DL-MDA231 \
  --sequence 01 \
  --output-dir reports/milestone_2 \
  --method otsu \
  --max-frames 5 \
  --min-size 64 \
  --save-masks
```

Expected outputs:

```text
reports/milestone_2/
├── overlays/
├── predicted_masks/
└── metrics/
    ├── baseline_segmentation_metrics.csv
    └── baseline_segmentation_metrics.json
```

The CSV and JSON files contain measured values only for frames with compatible GT. Frames without evaluation retain clear missing values rather than fabricated results.

## Milestone 3: Cell-Level Feature Extraction

Feature extraction converts segmented objects into structured, analysis-ready rows. This makes it possible to inspect morphology and fluorescence distributions, identify segmentation outliers, and prepare interpretable inputs for later statistical or machine-learning work.

Each nonzero label in a predicted mask becomes one table row. Extracted measurements include pixel area or voxel count, 2D or 3D centroid coordinates, bounding-box coordinates, and—when an aligned intensity image is available—mean, minimum, maximum, and integrated intensity. Every row also records dataset, sequence, frame, source image, mask, and segmentation method.

Run feature extraction on the current five-frame baseline masks:

```bash
python scripts/extract_features.py \
  --dataset-dir data/raw/Fluo-C3DL-MDA231 \
  --sequence 01 \
  --mask-dir reports/milestone_2/predicted_masks \
  --output-dir reports/milestone_3 \
  --method otsu \
  --max-frames 5
```

Add `--use-preprocessed` to calculate intensity features from the current preprocessing pipeline instead of raw fluorescence intensities.

Expected outputs:

```text
reports/milestone_3/
├── feature_extraction_summary.md
├── features/
│   ├── mda231_cell_features.csv
│   └── mda231_feature_summary.csv
└── figures/
    ├── cell_area_distribution.png
    ├── mean_intensity_distribution.png
    └── area_vs_intensity.png
```

These measurements inherit all errors and assumptions from the classical segmentation baseline. They are exploratory baseline-derived features, not final validated biological measurements.

## Milestone 4: Basic Cell Tracking

Tracking connects segmented cells over time so motion, persistence, and temporal changes can be analyzed instead of treating every frame independently. This milestone provides a transparent baseline for the 3D+time workflow.

For each pair of consecutive frames, the tracker calculates Euclidean distances between object centroids and uses Hungarian assignment to find a one-to-one matching with minimum total distance. Matches beyond a configurable maximum distance are rejected, while unmatched objects receive new persistent track IDs. The resulting table includes frame-to-frame displacement, speed at a frame interval of one, cumulative displacement, and track length.

Run tracking on the current MDA231 feature table:

```bash
python scripts/run_tracking.py \
  --features-file reports/milestone_3/features/mda231_cell_features.csv \
  --output-dir reports/milestone_4 \
  --max-distance 50 \
  --dataset-name mda231
```

Expected outputs:

```text
reports/milestone_4/
├── tracking_summary.md
├── tracks/
│   ├── mda231_cell_tracks.csv
│   └── mda231_track_summary.csv
└── figures/
    ├── cell_tracks_projection.png
    ├── speed_distribution.png
    └── track_duration_distribution.png
```

This centroid-only baseline does not model cell division, appearance, merges, or splits. Segmentation errors can break identity, and voxel-coordinate distances are not yet calibrated to physical spacing.

## Milestone 5: 2D U-Net Segmentation

Deep learning is introduced only after establishing inspectable classical baselines, evaluation conventions, and data-quality checks. The first learned model is a compact 2D U-Net because the official MDA231 segmentation ground truth consists of sparse annotated 2D planes extracted from 3D volumes. Training a 3D network directly on those sparse labels would require additional sampling and supervision design.

`CTCAnnotatedPlaneDataset` reads only official masks from `01_GT/SEG/` and `02_GT/SEG/`. It parses each GT filename, locates the matching raw time-point volume, extracts the annotated z-plane, converts nonzero GT labels to binary foreground, and normalizes the raw image to `[0, 1]`. Predicted masks from Milestone 2 are never used as training targets.

Install the optional deep-learning dependencies:

```bash
python -m pip install -e ".[deep-learning,dev]"
```

Run a small local CPU smoke test:

```bash
python scripts/train_unet2d.py \
  --dataset-dir data/raw/Fluo-C3DL-MDA231 \
  --output-dir reports/milestone_5 \
  --checkpoint-dir checkpoints/milestone_5 \
  --sequence-ids 01 02 \
  --epochs 1 \
  --batch-size 1 \
  --base-channels 4 \
  --device cpu \
  --max-samples 4
```

For full GPU training, open `colab/02_train_2d_unet_colab.ipynb`. It mounts Google Drive and writes reports and checkpoints directly under the configured external data root.

Expected training outputs:

```text
reports/milestone_5/
├── unet_2d_summary.md
├── metrics/
│   ├── unet_2d_training_history.csv
│   └── unet_2d_metrics.csv
└── figures/
    ├── training_loss_curve.png
    └── unet_2d_prediction_overlay.png
```

The model is trained on a limited number of sparse 2D annotations, without hyperparameter search or full-volume 3D supervision. The next model extension is a patch-based 3D U-Net with an annotation strategy appropriate for volumetric training.

## Milestone 6: Model Evaluation and Baseline Comparison

Milestone 6 compares the interpretable classical segmentation baseline against the trained 2D U-Net on the same official sparse CTC SEG planes. This makes the project easier to explain as an experiment: one shared evaluation target, two segmentation approaches, and side-by-side metrics/overlays.

The comparison uses binary foreground metrics:

- Dice
- IoU
- precision
- recall
- F1
- object-count error

It evaluates only sparse annotated 2D planes from `01_GT/SEG/` and `02_GT/SEG/`. It does not claim full-volume 3D performance.

Run locally after copying or training a checkpoint:

```bash
python scripts/compare_segmentation_methods.py \
  --dataset-dir data/raw/Fluo-C3DL-MDA231 \
  --checkpoint-path checkpoints/milestone_5/unet_2d_best.pt \
  --output-dir reports/milestone_6 \
  --sequence-ids 01 02 \
  --classical-method otsu \
  --device auto
```

Run in Colab using the external Google Drive data root:

```bash
python scripts/compare_segmentation_methods.py \
  --dataset-dir "/content/drive/MyDrive/self learning/Self Project/3d-cancer-cell-bioimage-ai-pipeline-data/data/raw/Fluo-C3DL-MDA231" \
  --checkpoint-path "/content/drive/MyDrive/self learning/Self Project/3d-cancer-cell-bioimage-ai-pipeline-data/checkpoints/milestone_5/unet_2d_best.pt" \
  --output-dir "/content/drive/MyDrive/self learning/Self Project/3d-cancer-cell-bioimage-ai-pipeline-data/reports/milestone_6" \
  --sequence-ids 01 02 \
  --classical-method otsu \
  --device auto
```

If `--checkpoint-path` is omitted or missing, the script still evaluates the classical baseline and prints a warning that U-Net comparison was skipped.

Expected outputs:

```text
reports/milestone_6/
├── milestone_6_summary.md
├── metrics/
│   ├── segmentation_method_comparison.csv
│   └── segmentation_method_summary.csv
└── figures/
    ├── comparison_overlay_000.png
    ├── comparison_overlay_001.png
    └── comparison_overlay_002.png
```

The local notebook `notebooks/06_model_comparison.ipynb` documents the same workflow. Keep large checkpoints and raw prediction artifacts outside Git; small summary CSVs or selected figures can be copied into version control later only if they are intentionally chosen for portfolio presentation.

## Pipeline Roadmap

1. Explore volume dimensions, intensity distributions, annotations, and sequence structure.
2. Normalize and preprocess microscopy volumes.
3. Establish classical segmentation baselines.
4. Evaluate predictions against available segmentation ground truth.
5. Extract interpretable morphology and intensity features.
6. Train a 2D U-Net baseline.
7. Compare classical and learned segmentation baselines.
8. Train a patch-based 3D U-Net.
9. Add end-to-end experiment management.

## Current MVP Scope

The current MVP covers project setup, dataset inspection, data exploration, preprocessing, classical segmentation, sparse-plane evaluation, feature extraction, centroid tracking, 2D U-Net training, and baseline-vs-U-Net comparison.

## Future Work

Planned extensions include patch-based 3D U-Net experiments, stronger validation on Fluo-C3DH-A549, DVC-backed data versioning, Docker environments, and Airflow orchestration. These should be introduced only after the current evaluation protocol and artifact storage strategy are stable.
