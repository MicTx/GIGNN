# Data Contract

This document defines the input and boundary contracts enforced by the current codebase.

## Feature Table

`X` must be a non-empty pandas `DataFrame`.

Requirements:

- At least one row and one feature column.
- Every feature column must convert to numeric values with `pandas.to_numeric`.
- All feature values must be finite.
- Prediction input must use the same feature columns in the same order as training input for `GIGNNClassifier`.

GIGNN copies input frames before validation and does not intentionally mutate user-provided `DataFrame` objects.

Research artifact export treats raw tables as sensitive by default. `export_research_artifacts(..., include_raw=False)` records input schema, row/column summary, and a SHA-256 fingerprint but does not write `raw/raw_frame.csv`. Set `include_raw=True` only when duplicating the source table into the artifact bundle is acceptable for the research workflow. Fingerprints and optional raw CSV output use the same sanitized canonical UTF-8 bytes with `\n` line endings.

## Labels

`y` may be a pandas `Series` or another iterable.

Requirements:

- Length must match `X`.
- Must be non-empty.
- Must not contain missing values.
- Values may be strings or non-contiguous integers.

Labels are encoded internally with `LabelEncoder`. The encoder is immutable-style: call `encoder = LabelEncoder().fit(y)` before `encoder.transform(y)`.

## Geometry

Geometry is required when graph edges are not supplied explicitly.

For `knn` and `distance` graph strategies, each item must expose either:

- `item.x` and `item.y`
- `item.centroid.x` and `item.centroid.y`

The resolved coordinates must be finite numeric values. NaN and infinity are rejected with `ValidationError`.

For `contiguity`, geometry objects should expose callable `touches(other)` or `intersects(other)` methods.

Geometry length must match the number of rows when `num_nodes` is provided.

## Edge Index

`edge_index` is a PyG-style directed edge list.

Accepted shapes:

- `(2, num_edges)`
- `(num_edges, 2)`

Requirements:

- 2D array-like input.
- Integer node indices.
- At least one edge.
- Node indices in `[0, num_nodes - 1]`.
- Edge count no greater than `MAX_EDGES`.

When both `edge_index` and `geometry` are provided, explicit `edge_index` validation takes priority in direct graph resolution. In the automatic workflow, explicit edges become a graph candidate and geometry-derived graphs may also be evaluated.

## Graph Strategies

| Strategy | Required Parameters | Behavior |
| --- | --- | --- |
| `knn` | `k >= 1` | Each node connects to the `min(k, n-1)` nearest other nodes by Euclidean centroid distance. Edges are directed; equal-distance candidates are resolved by ascending target index. The normal path queries only `k+1` neighbors, expanding only the boundary-distance tie set. |
| `distance` | `threshold > 0` | Each node connects in both directions to other nodes whose Euclidean centroid distance is `<= threshold`. Self-loops are excluded. |
| `contiguity` | Geometry with `touches` or `intersects` | Directed edges are emitted for touching or intersecting geometries. |

Graph construction raises `ValidationError` when no edges are produced.

## Edge Attributes And Weights

`GIGNNClassifier` computes Gaussian edge attributes from the feature table used for the current fit or prediction call:

```text
edge_attr[e, f] = exp(-((x_source,f - x_target,f)^2) / (2 * sigma^2))
```

The default `edge_weight_sigma` is `2.0` (Eq. (2) of the paper) and expects GeoDetector stratum-code features; on raw measurement scales the kernel collapses and `fit` emits `EdgeWeightWarning`. Per-feature edge attributes are aggregated into scalar `edge_weight` values with `edge_weight_mode="product"` by default, which makes the scalar weight exactly `exp(-||x_i - x_j||^2 / (2 sigma^2))`. Supported aggregation modes are `product`, `mean`, `min`, and `max`.

`build_pyg_data(..., edge_attr=..., edge_weight=...)` preserves these arrays on the returned PyG `Data` object. `edge_attr` must have shape `(num_edges, num_edge_features)` or `(num_edges,)`; `edge_weight` must have shape `(num_edges,)`.

## Multi-Hop Neighbor Sampling

`GIGNNClassifier.num_neighbors_list` controls notebook-style per-hop neighbor sampling. The default `(6, 12)` mirrors the notebook's `NUM_NEIGHBORS_LIST = [6, 12]`.

Requirements:

- Values must be positive integers.
- Length must equal `len(hidden_channels) + 1`.
- Each value is capped by `MAX_GRAPH_NEIGHBORS`.
- Nodes without neighbors sample themselves so prediction can still produce one output per input row.

## Auto Workflow Inputs

`GIGNNAutoClassifier.fit` requires at least one of:

- `geometry`
- `edge_index`

Search grid limits and parameter-level schemas are checked before fitting and again when loading automatic checkpoints. Detector, graph, and model grids reject unknown keys, empty candidate lists, boolean-as-integer values, non-finite numeric candidates, and values outside resource limits. Optional graph values such as `threshold=None` remain valid and round-trip through `.gignn` checkpoints. `max_grid_candidates` must be a non-boolean positive integer no greater than `MAX_FEATURES`; it bounds the actual workload as `detector_candidates + graph_candidates * model_candidates` in both runtime and checkpoint validation.

`validation_fraction` and `test_fraction` must be positive and sum to less than one. The default random mode creates disjoint train/validation/test partitions; `split_labels` can provide an explicit external split, and `spatial_groups` keeps each group wholly inside one partition. `split_labels` and `spatial_groups` are mutually exclusive. Spatial groups are indivisible, so requested fractions may be unattainable; GIGNN emits a `UserWarning` when the realized row fraction differs by more than 10 percentage points and includes both row and group fractions in the message.

Non-numeric ordinal targets require `class_order`, unless `y` is an ordered pandas categorical whose declared categories exactly match the observed labels. Numeric targets default to ascending numeric order. Unordered strings and incomplete or duplicate class orders are rejected.

For geometry-derived candidates, `knn`, `distance`, and `contiguity` graphs are constructed separately inside each partition. Explicit edges are filtered and locally reindexed per partition. Failed no-edge candidates are recorded and skipped; fitting fails only when every candidate is unavailable.

## Prediction Inputs

For `GIGNNAutoClassifier.predict`, `predict_proba`, and `simulate`:

- If `edge_index` is provided, it is used for prediction.
- If no `edge_index` is provided and the fitted selected strategy is not `edge_index`, `geometry` is used to rebuild prediction edges with the selected graph strategy.
- If the fitted selected strategy is `edge_index`, `geometry` alone does not replace the fitted graph; the stored fitted `edge_index_` is reused unless a new `edge_index` is explicitly provided.
- `simulate` requires baseline and scenario frames to have the same row count, feature-column order, and pandas row index. When the fitted workflow has `node_ids`, the caller must also provide the exact fitted ID order.
- Scenario topology is compared as a canonical directed edge multiset, so edge-column ordering alone does not change topology identity.

Reusing a stored edge index is only appropriate when prediction rows are compatible with the training graph. Resetting both frame indices without supplying `node_ids` removes row-identity information that the library cannot reconstruct; callers that require identity enforcement must fit and simulate with explicit IDs.

## Generalization Diagnostics

`generalization_diagnostics` requires observed and predicted labels with equal length and no missing values.

Bias is computed as (paper Fig. 10 convention):

```text
Bias = predicted_code - observed_code
```

Bias labels are produced as:

- `Correct` when bias is zero.
- `Overestimated (...)` when bias is positive.
- `Underestimated (...)` when bias is negative.

Provide `class_order` when labels are ordinal, mixed-type, or otherwise not safely sortable. With `edge_index`, `generalization_diagnostics` also tests the spatial autocorrelation of the bias with `morans_i` (fields on `GeneralizationDiagnostics.bias_moran`); `morans_i(values, edge_index)` requires at least three finite observations and an edge index connecting at least two distinct nodes.

## Resource Limits

| Constant | Value | Purpose |
| --- | --- | --- |
| `MAX_AUTO_GRAPH_NODES` | `1_000` | Prevent accidental quadratic automatic graph construction. |
| `MAX_EDGES` | `1_000_000` | Cap edge-index size. |
| `MAX_FEATURES` | `10_000` | Cap checkpoint feature count. |
| `MAX_CLASSES` | `10_000` | Cap checkpoint class count. |
| `MAX_GRAPH_NEIGHBORS` | `1_000` | Cap KNN neighbor count. |
| `MAX_HIDDEN_CHANNEL` | `4_096` | Cap hidden width. |
| `MAX_HIDDEN_LAYERS` | `8` | Cap hidden depth. |
| `MAX_TRAINING_EPOCHS` | `10_000` | Cap training loop length. |
| `MAX_CHECKPOINT_BYTES` | `100 MiB` | Cap direct and automatic checkpoint file size; automatic archives also cap total expanded member size. |
| `MAX_CHECKPOINT_TENSORS` | `128` | Cap state-dict tensor count. |
| `MAX_CHECKPOINT_TENSOR_ELEMENTS` | `10_000_000` | Cap total checkpoint tensor elements. |
