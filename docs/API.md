# Public API Reference

This reference describes the public and semi-public APIs visible in the current codebase. Stable user entry points are the top-level `gignn` exports plus the documented `gignn.geodetector` and `gignn.graph` helpers. Lower-level `training`, `models`, `io`, and `validation` APIs are useful for advanced integration but may change before `1.0`.

For the current audit of whether the notebook GIGNN core is implemented, see [Implementation Status](IMPLEMENTATION_STATUS.md).

## Top-Level `gignn` Exports

| Name | Kind | Purpose |
| --- | --- | --- |
| `GIGNNAutoClassifier` | Dataclass estimator | High-level workflow that searches feature discovery, graph strategy, and model configuration. |
| `GIGNNClassifier` | Dataclass estimator | Direct notebook-aligned GIGNN node classifier with explicit graph/model control. |
| `GIGNN` | Torch module | Weighted multi-hop neighbor aggregation model matching the notebook core. |
| `multihop_sampling` | Function | Build notebook-style sampled node lists from an adjacency dictionary and per-hop neighbor counts. |
| `standardize_spatial_dataset` | Function | Build a standardized spatial-unit dataset and report from a tabular frame. |
| `generalization_diagnostics` | Function | Compute accuracy, prediction bias table, classification report, and optional bias Moran's I. |
| `morans_i` | Function | Global Moran's I over the graph (`esda.Moran`-compatible semantics). |
| `cross_validate` | Function | Transductive k-fold evaluation with pooled out-of-fold predictions and optional bias Moran's I (paper §5.2 protocol). |
| `model_accuracy_table` | Function | Convert model accuracy values into a comparison table. |
| `export_research_artifacts` | Function | Export a fitted auto workflow as a raw/preprocessing/process/results research artifact bundle. |
| `ResearchArtifactBundle` | Frozen dataclass | Holds artifact root, manifest path, and logical file mapping. |
| `NATURE_PALETTE` | Lazy-loaded constant | Nature-style semantic plotting palette. |
| `NATURE_COLOR_CYCLE` | Lazy-loaded constant | Default publication plotting color cycle. |
| `apply_publication_style` | Lazy-loaded function | Apply publication-ready Matplotlib rcParams without import-time side effects. |
| `infer_publication_layout` | Lazy-loaded function | Infer figure size, font size, spacing, legend placement, and mark sizes from data complexity. |
| `add_panel_label` | Lazy-loaded function | Add bold axes-relative panel labels for multi-panel figures. |
| `save_publication_figure` | Lazy-loaded function | Save figures as SVG/PNG/PDF-style publication outputs. |
| `plot_classification_map` | Lazy-loaded function | Plot a thematic class or prediction map. |
| `plot_prediction_comparison` | Lazy-loaded function | Plot observed and predicted maps side by side. |
| `plot_observed_prediction_bias` | Lazy-loaded function | Plot observed, predicted, and bias panels. |
| `plot_accuracy_comparison` | Lazy-loaded function | Plot model accuracy bars. |
| `plot_discretization_histogram` | Lazy-loaded function | Plot feature histogram with optional discretization intervals. |
| `plot_training_curves` | Lazy-loaded function | Plot training loss and train/test accuracy curves. |
| `plot_metric_boxplot` | Lazy-loaded function | Plot metric distributions, optionally grouped. |
| `plot_interaction_network` | Lazy-loaded function | Plot GeoDetector factor interaction network. |
| `StandardizedSpatialDataset` | Frozen dataclass | Holds standardized frame, features, target, IDs, and geometry. |
| `SpatialDatasetReport` | Frozen dataclass | Reports selected columns, row count, and filled missing values. |
| `GeneralizationDiagnostics` | Frozen dataclass | Holds accuracy, bias frame, classification report, and optional `bias_moran`. |
| `MoranResult` | Frozen dataclass | Moran's I statistic, expected value, normality and permutation inference. |
| `CrossValidationReport` | Frozen dataclass | Per-fold metrics plus pooled out-of-fold labels/predictions and `bias_moran`. |
| `GIGNNError` | Exception | Base package exception. |
| `EdgeWeightWarning` | Warning | Gaussian edge weights collapsed; neighbors stop contributing. |
| `ValidationError` | Exception | Input contract or resource limit violation. |
| `NotFittedError` | Exception | Estimator or transformer used before fitting. |

## `GIGNNAutoClassifier`

Signature:

```python
GIGNNAutoClassifier(
    detector_grid={"discretizer": ["quantiles"], "max_bins": [3]},
    graph_grid={"strategy": ["knn"], "k": [8]},
    model_grid={"hidden_channels": [(64,)], "learning_rate": [0.01], "max_epochs": [100]},
    top_k_features=None,
    random_state=None,
    validation_fraction=0.2,
    test_fraction=0.2,
    max_grid_candidates=100,
)
```

Methods:

| Method | Returns | Notes |
| --- | --- | --- |
| `fit(X, y, *, geometry=None, edge_index=None, split_labels=None, spatial_groups=None, class_order=None, node_ids=None)` | `self` | Requires `geometry` or `edge_index`; searches detector, graph, and model candidates with disjoint train/validation/test partitions. |
| `predict(X, *, geometry=None, edge_index=None)` | `np.ndarray` | Returns labels in original label space. |
| `predict_proba(X, *, geometry=None, edge_index=None)` | `np.ndarray` | Returns shape `(n_rows, n_classes)`. |
| `simulate(X_scenario, *, baseline_X, geometry=None, edge_index=None, node_ids=None, scenario_edge_index=None)` | `SimulationReport` | Compares baseline and scenario under locked row identity and topology. |
| `save(path)` | `None` | Saves the full fitted automatic workflow as a bounded `.gignn` ZIP containing JSON metadata and NPY arrays; no pickle loader is used. |
| `load(path)` | `GIGNNAutoClassifier` | Validates archive members, sizes, hashes, schema/version, arrays, and model state before restoring the selector, ordinal schema, graph/model configuration, reports, training losses, and fitted weights. |
| `export_research_artifacts(output_dir, **kwargs)` | `ResearchArtifactBundle` | Writes a reproducibility bundle for the fitted auto workflow. |

Learned attributes:

| Attribute | Description |
| --- | --- |
| `selector_` | Fitted `GeoDetectorSelector`. |
| `estimator_` | Fitted lower-level `GIGNNClassifier`. |
| `selected_features_` | Feature names selected by GeoDetector ranking. |
| `edge_index_` | Selected graph edge index. |
| `discovery_report_` | `SpatialRelationshipReport`. |
| `graph_report_` | `GraphSelectionReport`. |
| `model_selection_report_` | `ModelSelectionReport`. |
| `validation_report_` | `GeneralizationReport` computed only from validation rows and used for candidate selection; includes `partition="validation"` and immutable source indices. |
| `generalization_report_` | `GeneralizationReport` computed only from independent test rows after selection; includes `partition="test"` and immutable source indices. |

Errors:

- Raises `ValidationError` if neither graph input nor geometry is provided.
- Raises `ValidationError` for unknown grid keys, invalid parameter values, non-finite numeric candidates, or excessive search budgets. Candidate workload is counted as detector candidates plus graph-by-model combinations, and `max_grid_candidates` is validated identically at fit and checkpoint load time.
- Runtime search and `.gignn` checkpoint loading use the same parameter-level grid contract; optional values such as `graph_grid={"threshold": [None]}` round-trip unchanged.
- Raises `NotFittedError` when prediction methods are called before `fit`.

## `GIGNNClassifier`

Signature:

```python
GIGNNClassifier(
    hidden_channels=(128,),
    num_neighbors_list=(6, 12),
    graph_builder="knn",
    k_neighbors=8,
    distance_threshold=None,
    edge_weight_sigma=2.0,
    edge_weight_mode="product",
    normalize_edge_weights=False,
    inference="full",
    max_epochs=100,
    learning_rate=0.01,
    weight_decay=5e-4,
    dropout=0.0,
    random_state=None,
    device="cpu",
    class_order=None,
)
```

- `edge_weight_mode="product"` reproduces Eq. (2) exactly: the product of per-feature Gaussian kernels is `exp(-||x_i - x_j||^2 / (2 sigma^2))`. `mean`, `min`, `max` are non-paper alternatives.
- `normalize_edge_weights=False` applies weights as Eq. (2) message scaling inside the mean over neighbors; `True` divides by the weight sum (kernel-weighted average).
- `inference="full"` (default) evaluates deterministic full-neighborhood aggregation — the expectation of the sampled estimator; `"sampled"` reproduces the notebook's stochastic `multihop_sampling` inference.
- `fit` accepts `train_mask=`: a boolean array restricting the loss to masked nodes while every node stays in the graph (transductive training for the paper's k-fold protocol); labels outside the mask may be missing.
- When the median edge weight collapses below `1e-3` — typical when raw-scale rather than stratum-code features are fed to the `sigma=2` kernel — `fit` emits `EdgeWeightWarning`; discretize with `GeoDetectorSelector` first or raise `edge_weight_sigma`.

Methods:

| Method | Returns | Notes |
| --- | --- | --- |
| `fit(X, y, *, geometry=None, edge_index=None, train_mask=None)` | `self` | Builds or validates graph, computes Gaussian edge attributes/weights, converts to PyG data, trains weighted multi-hop GIGNN; `train_mask` restricts the loss to masked nodes. |
| `predict(X, *, geometry=None, edge_index=None)` | `np.ndarray` | Returns original labels. |
| `predict_proba(X, *, geometry=None, edge_index=None)` | `np.ndarray` | Softmax probabilities. |
| `score(X, y, *, geometry=None, edge_index=None)` | `float` | Accuracy score. |
| `save(path)` | `None` | Writes checkpoint. Requires fitted estimator. |
| `load(path)` | `GIGNNClassifier` | Loads checkpoint and validates schema/resources. |

Learned attributes:

| Attribute | Description |
| --- | --- |
| `training_result_` | `TrainingResult` with per-epoch losses. |
| `model_` | Fitted `GIGNN`. |
| `label_encoder_` | Fitted immutable-style label encoder. |
| `n_features_in_` | Number of training feature columns. |
| `feature_names_in_` | Training feature columns in order. |
| `classes_` | Original label classes. |

Prediction requires the same feature names in the same order as training.

`num_neighbors_list` follows the notebook semantics: each entry is the number of sampled neighbors for one hop. The default `(6, 12)` mirrors the notebook's `NUM_NEIGHBORS_LIST = [6, 12]` for one hidden layer plus output layer. Its length must equal `len(hidden_channels) + 1`.

Sampling uses two explicit RNG modes. An integer `random_state` reconstructs an independent `np.random.default_rng(seed)` on each call, so equal seeds produce equal neighborhoods. `random_state=None` consumes NumPy's global RNG stream; consecutive calls advance the stream, while `set_random_seed(seed)` or `np.random.seed(seed)` can replay the complete sequence.

## Research Artifact Export

Module: `gignn.artifacts`

Primary API:

```python
from gignn import export_research_artifacts

bundle = export_research_artifacts(
    model,
    "artifacts/run_001",
    raw_frame=X,
    geometry=geometry,
    include_raw=False,
    provenance={"study_area": "example"},
)
```

Equivalent fitted-model convenience method:

```python
bundle = model.export_research_artifacts("artifacts/run_001", raw_frame=X, geometry=geometry)
```

The export writes a directory with `manifest.json`, `README.md`, and four evidence layers:

| Layer | Key files |
| --- | --- |
| `raw/` | `input_schema.json`, `input_summary.json`, `input_fingerprint.json`, optional `raw_frame.csv` when `include_raw=True` |
| `preprocessing/` | `feature_summary.csv`, `missing_values.csv`, `fill_summary.json`, `target_distribution.csv`, train/validation/test `split_indices.csv`, `selected_columns.json` |
| `process/` | `factor_scores.csv`, `interaction_scores.csv`, `discretization_bins.csv`, `graph_candidates.csv`, `selected_edges.csv`, `model_candidates.csv`, `training_loss.csv` |
| `results/` | independent-test `metrics.json`, `validation_metrics.json`, `per_class_report.csv`, `predictions.csv`, `probabilities.csv`, optional `simulation_report.csv`, `figures_manifest.csv` |

`include_raw=False` is the default and does not write `raw/raw_frame.csv`; schema, summary, and fingerprint are still recorded for reproducibility. `overwrite=False` is the default and raises `ValidationError` when the output directory already contains files. Use `overwrite=True` only when replacing an existing artifact bundle intentionally.

## GeoDetector API

Module: `gignn.geodetector`

| Name | Signature / Kind | Purpose |
| --- | --- | --- |
| `spatial_discretization` | `spatial_discretization(data_frame, continuous_columns, method="quantiles", num_intervals=4)` | Discretize continuous columns and return metadata. |
| `factor_detector` | `factor_detector(df, y, factors, alpha=0.05)` | Compute q statistics and p-values for factors. |
| `geodetector_q` | `geodetector_q(y_values, factor_values)` | Compute one q statistic. |
| `interaction_detector` | `interaction_detector(df, y, factors)` | Compute pairwise interaction q values and interaction type. |
| `GeoDetectorSelector` | `GeoDetectorSelector(discretizer="natural_breaks", max_bins=5, top_k=None, alpha=None, strategy="interaction", top_interactions=10)` | Fit/transform feature selector. |
| `DiscretizationMetadata` | Frozen dataclass | Stores method and bin boundaries. |

`natural_breaks` and `fisher_jenks` use a deterministic in-package Fisher-Jenks dynamic program (exact optimum, vectorized); `maximum_breaks` is a separate widest-gap method. `interaction_detector` also reports `p_value_interaction` (ANOVA F on the combined stratification).

`GeoDetectorSelector` defaults implement the paper's §2.2.1/Fig. 8 process: `strategy="interaction"` ranks factors by degree centrality in the top-`top_interactions` interaction network, breaking ties by factor q (reproduce the case study with `top_k=3`); `strategy="factor"` ranks by the factor-detector q alone. Fitted attributes `interactions_` and `interaction_degree_` hold the sorted interaction table and per-factor degree. `target_to_numeric` requires numeric or ordered-categorical targets; unordered non-numeric targets with more than two classes raise `ValidationError` because the variance-based q is undefined for them.

## Graph API

Module: `gignn.graph`

| Name | Signature / Kind | Purpose |
| --- | --- | --- |
| `build_spatial_edges` | `build_spatial_edges(geometry=None, *, edge_index=None, num_nodes=None, strategy="knn", k=8, threshold=None, max_nodes=1000, max_edges=1000000)` | Validate explicit edges or build finite-coordinate geometry edges. KNN ties use ascending target index; distance thresholds are inclusive and emit both directions. |
| `build_pyg_data` | `build_pyg_data(X, y, edge_index, *, edge_attr=None, edge_weight=None)` | Build `torch_geometric.data.Data`, preserving optional edge attributes and edge weights. |
| `build_adjacency_dict` | `build_adjacency_dict(edge_index, *, num_nodes)` | Build deduplicated bidirectional adjacency dict. |
| `calculate_gaussian_edge_attributes` | `calculate_gaussian_edge_attributes(node_features, edge_index, *, sigma=2.0)` | Compute Gaussian source-target feature attributes. |
| `calculate_edge_weights` | `calculate_edge_weights(edge_attributes, *, mode="product")` | Aggregate per-feature edge attributes into model-consumable scalar edge weights; `product` is the exact Eq. (2) vector kernel. |
| `filter_isolated_nodes` | `filter_isolated_nodes(adjacency_dict)` | Remove isolated nodes and return retained original indices. |

## Generalization API

| Name | Returns | Purpose |
| --- | --- | --- |
| `standardize_spatial_dataset` | `(StandardizedSpatialDataset, SpatialDatasetReport)` | Normalize a spatial-unit frame for modeling. |
| `generalization_diagnostics` | `GeneralizationDiagnostics` | Compare observed/predicted labels and produce bias diagnostics; `edge_index` adds `bias_moran`. `Bias` is predicted minus observed class code — positive means overestimated (paper Fig. 10). |
| `morans_i` | `MoranResult` | `morans_i(values, edge_index, permutations=999, random_state=None)`; row-standardized symmetrized contiguity, `p_norm`/`z_norm` under the normality assumption, folded one-tailed `p_sim`/`z_sim` from permutations. |
| `cross_validate` | `CrossValidationReport` | `cross_validate(estimator_factory, X, y, n_splits=5, random_state=None, class_order=None, geometry=None, edge_index=None, moran_permutations=999, factory_kwargs=None)`; transductive stratified folds, pooled out-of-fold predictions, optional bias Moran's I. |
| `model_accuracy_table` | `pd.DataFrame` | Build a plotting/reporting table from model accuracy values. |

## Plotting API

All plotting functions return Matplotlib `Figure`/`Axes` objects and save only when `output_path` is provided. Publication helpers are lazy-loaded from `gignn` and do not modify Matplotlib rcParams at import time.

| Helper | Purpose |
| --- | --- |
| `NATURE_PALETTE` | Semantic Nature-style colors for methods, directional effects, and neutral support. |
| `NATURE_COLOR_CYCLE` | Default publication color order derived from `NATURE_PALETTE`. |
| `apply_publication_style(font_size=12, axes_linewidth=1.2, use_tex=False)` | Apply sans-serif font stack, editable SVG text, frameless legends, and minimal spines. |
| `infer_publication_layout(row_count, category_count=1, panel_count=1, label_lengths=None, geometry_aspect=None)` | Infer adaptive `figsize`, font sizes, spacing, legend location, marker size, and bar width. |
| `add_panel_label(ax, label, ...)` | Add bold panel labels using axes-relative coordinates. |
| `save_publication_figure(figure, output_path, formats=("svg", "png"), dpi=300, pad=2, close=False)` | Save one figure to multiple publication formats and optionally close it. |

| Function | Required columns / input | Returns |
| --- | --- | --- |
| `plot_classification_map` | `geo_frame`, `column` | `(Figure, Axes)` |
| `plot_prediction_comparison` | `truth_column`, `prediction_column` | `(Figure, np.ndarray)` |
| `plot_observed_prediction_bias` | `observed_column`, `prediction_column`, `bias_column` | `(Figure, np.ndarray)` |
| `plot_accuracy_comparison` | accuracy table columns | `(Figure, Axes)` |
| `plot_discretization_histogram` | data frame and numeric column | `(Figure, Axes)` |
| `plot_training_curves` | loss, train accuracy, test accuracy columns | `(Figure, np.ndarray)` |
| `plot_metric_boxplot` | metric/value/group columns | `(Figure, Axes)` |
| `plot_interaction_network` | source, target, weight columns | `(Figure, Axes)` |

## Advanced Modules

### `gignn.validation`

| Name | Purpose |
| --- | --- |
| `validate_feature_table` | Validate finite numeric feature frame. |
| `validate_X_y` | Validate feature/label alignment. |
| `validate_edge_index` | Validate and normalize edge index. |
| `LabelEncoder` | Immutable-style label encoder. |

### `gignn.training`

| Name | Purpose |
| --- | --- |
| `TrainingConfig` | Frozen training configuration. |
| `TrainingResult` | Frozen training loss report. |
| `train_node_classifier` | Fixed-epoch node classification training loop. |
| `calculate_class_weights` | Inverse-frequency class weights. |
| `set_random_seed` | Seed Python, NumPy, and Torch. |

### `gignn.models`

| Name | Purpose |
| --- | --- |
| `GIGNN` | Notebook-aligned weighted multi-hop GIGNN module. |
| `NeighborAggregator` / `GCN` | Building blocks for the notebook-aligned GIGNN module. |
| `multihop_sampling` | Multi-hop sampling helper: integer seeds use an independent `default_rng(seed)`; `None` consumes NumPy's global RNG stream. |
| `GraphSAGEClassifier` | Compatibility multi-layer GraphSAGE node classifier. |

### `gignn.io`

| Name | Purpose |
| --- | --- |
| `save_checkpoint` | Normalize NumPy scalar values, reject non-weights-only-safe payloads before writing, fsync the temporary file, atomically replace the target, and fsync the parent directory when supported. |
| `load_checkpoint` | Load checkpoint on CPU with `weights_only=True`, file-size guard, and stable-final-path symlink rejection. |

## Report Dataclasses

See [Reports](REPORTS.md) for field-level detail on:

- `SpatialRelationshipReport`
- `GraphSelectionReport`
- `ModelSelectionReport`
- `GeneralizationReport`
- `SimulationReport`
