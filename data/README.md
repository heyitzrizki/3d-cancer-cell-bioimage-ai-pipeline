# Data Directory

Raw Cell Tracking Challenge datasets must be stored outside this Git repository. The recommended location is the Google Drive data root documented in the main README.

Expected external layout:

```text
3d-cancer-cell-bioimage-ai-pipeline-data/
├── raw/
│   ├── Fluo-C3DL-MDA231/
│   └── Fluo-C3DH-A549/
├── interim/
├── processed/
├── masks/
├── features/
├── predictions/
├── checkpoints/
├── logs/
└── reports/
```

Do not commit raw volumes, annotations, processed volumes, predictions, or model checkpoints. This directory is retained only for documentation and optional small metadata files.
