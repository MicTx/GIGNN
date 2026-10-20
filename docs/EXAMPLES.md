# Examples

The repository currently includes one bundled executable example.

The example exercises the current notebook-aligned GIGNN estimator path, including automatic selection of `num_neighbors_list` when the auto workflow uses `GIGNNClassifier`.

## `examples/quickstart.py`

Purpose:

- Build a small numeric feature table.
- Create simple point-like geometry objects.
- Standardize a spatial dataset.
- Fit `GIGNNAutoClassifier`.
- Run scenario simulation.
- Predict on the training rows.
- Build adjacency and Gaussian edge attributes.
- Compute generalization diagnostics and an accuracy table.
- Create histogram, training curve, and accuracy comparison figures.

Run:

```bash
.venv/bin/python examples/quickstart.py
```

Expected output includes lines similar to:

```text
selected features: [...]
selected graph: knn
best model params: {...}
generalization accuracy: ...
standardized features: [...]
adjacency nodes: ... edge attributes: (...)
predictions: [...]
changed in simulation: [...]
bias types: [...]
plotting: generated histogram, training-curve, and accuracy-comparison figures
```

The quickstart does not save figure files. It exercises plotting return paths in memory.

## Extending Examples

When adding examples, keep each example focused on one user story:

| Example Type | Recommended Focus |
| --- | --- |
| Auto workflow | `GIGNNAutoClassifier.fit`, reports, prediction, simulation. |
| Explicit graph | `edge_index` input and `GIGNNClassifier`. |
| GeoDetector only | Discretization, factor q ranking, interactions, selector transform. |
| Generalization | `standardize_spatial_dataset`, diagnostics, bias plots. |
| Plotting | Figure generation and optional `output_path`. |

Examples should use small synthetic data unless real data is intentionally included and licensed.
