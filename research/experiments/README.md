# Phase 2 Experiment Package

This package holds the fixed, redistributable experiment inputs for the GIGNN patent reassessment.

## License and provenance

- Data generation: deterministic synthetic spatial lattice created by
  `research/scripts/generate_phase2_dataset.py`.
- License: MIT, same as the GIGNN repository (`LICENSE`).
- External remote-sensing or commercial GIS raw data are intentionally not used.
- Clean rebuild command:

```bash
.venv/bin/python research/scripts/generate_phase2_dataset.py --output-dir research/experiments
```

Rebuilding must recreate identical files and hashes. The generator writes:

- `dataset.csv`: node attributes, ordered labels, spatial groups, and coordinates
- `schema.json`: column contract, class order, and spatial-unit ID rules
- `manifest.json`: source metadata, generator parameters, file hashes, and package version

## Dataset contract

| Column | Meaning |
| --- | --- |
| `node_id` | Stable spatial-unit identity (`unit-000` ...). |
| `x`, `y` | Synthetic lattice coordinates. |
| `spatial_group` | 3x3 region labels used for spatial-block holdout. |
| `primary` | Region-centered continuous feature. |
| `secondary` | Weakly informative continuous feature. |
| `noise` | Non-informative continuous feature. |
| `label` | Ordered target in `["low", "medium", "high"]`. |

The fixed class order is `["low", "medium", "high"]`. Partition strategies must keep
`spatial_group` wholly inside one of train/validation/test when using spatial holdout.
