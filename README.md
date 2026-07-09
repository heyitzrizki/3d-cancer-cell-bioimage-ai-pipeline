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

## Pipeline Roadmap

1. Explore volume dimensions, intensity distributions, annotations, and sequence structure.
2. Normalize and preprocess microscopy volumes.
3. Establish classical segmentation baselines.
4. Evaluate predictions against available segmentation ground truth.
5. Extract interpretable morphology and intensity features.
6. Train a 2D U-Net baseline.
7. Train a patch-based 3D U-Net.
8. Add cell tracking and end-to-end experiment management.

## Current MVP Scope

The initial MVP covers project setup, dataset inspection, data exploration, preprocessing, classical segmentation, evaluation, and feature extraction. It does not yet train neural networks or report experimental results.

## Future Work

Planned extensions include 2D U-Net training, patch-based 3D U-Net experiments, temporal cell tracking, DVC-backed data versioning, Docker environments, and Airflow orchestration. These will be introduced only after the MVP pipeline and evaluation protocol are stable.
