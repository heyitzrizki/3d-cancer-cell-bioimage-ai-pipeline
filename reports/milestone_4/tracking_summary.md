# Tracking Summary

- Input feature file: `reports/milestone_3/features/mda231_cell_features.csv`
- Number of frames: 5
- Number of objects: 160
- Number of tracks: 39
- Mean track length: 4.1026
- Median track length: 5.0000
- Mean speed: 6.3453
- Maximum matching distance: 50.0000
- Cell tracks: `reports/milestone_4/tracks/mda231_cell_tracks.csv`
- Track summary: `reports/milestone_4/tracks/mda231_track_summary.csv`

## Limitations

- Centroid matching is a transparent baseline, not a state-of-the-art tracker.
- Merged or split cells can break track identity.
- Cell division is not modeled.
- Appearance-based matching is not included.
- Distances are measured in voxel coordinates without physical voxel-size calibration.
