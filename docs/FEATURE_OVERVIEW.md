# Feature Overview

This document maps GIGNN capabilities from the current codebase. It is organized by user-visible layers rather than by source-file order.

For the implementation completeness audit against the notebook GIGNN core, see [Implementation Status](IMPLEMENTATION_STATUS.md).

## 1. Data Preparation

### 1.1 Spatial Dataset Standardization

API: `gignn.standardize_spatial_dataset`

Purpose:

- Accept a pandas `DataFrame` containing target, optional ID, optional geometry, and feature columns.
- Resolve feature columns explicitly or by excluding target/ID/geometry columns.
- Convert features to finite numeric values.
- Optionally fill missing feature values before validation.
- Return immutable dataclasses describing standardized data and preprocessing decisions.

Outputs:

- `StandardizedSpatialDataset.frame`: reset-index copy of the full frame.
- `StandardizedSpatialDataset.features`: numeric feature table.
- `StandardizedSpatialDataset.target`: target series with missing labels rejected.
- `StandardizedSpatialDataset.ids`: optional ID series.
- `StandardizedSpatialDataset.geometry`: optional geometry series.
- `SpatialDatasetReport`: selected columns, row count, and missing-value fill counts.

### 1.2 Input Validation

API: `gignn.validation.validate_feature_table`, `validate_X_y`, `validate_edge_index`, `LabelEncoder`

Purpose:

- Enforce non-empty finite numeric feature tables.
- Enforce aligned `X` and `y` lengths.
- Reject missing labels.
- Normalize PyG-style edge indices from `(2, num_edges)` or `(num_edges, 2)`.
- Encode labels into contiguous class indices and decode predictions back to original labels.

## 2. GeoDetector Feature Discovery

### 2.1 Spatial Discretization

API: `gignn.geodetector.spatial_discretization`

Supported methods:

- `quantiles`
- `equal_interval`
- `natural_breaks`
- `fisher_jenks`
- `maximum_breaks`
- `box_plot`
- `std_mean`

The current implementation validates numeric finite inputs and returns `DiscretizationMetadata` with method and bins. `natural_breaks` and `fisher_jenks` use an exact in-package Fisher-Jenks dynamic program; `maximum_breaks` is a distinct widest-gap method.

### 2.2 Factor Detection

API: `gignn.geodetector.factor_detector`, `geodetector_q`

Purpose:

- Compute GeoDetector q statistics for candidate factors.
- Convert non-numeric targets to categorical codes.
- Compute approximate F-test p-values for q statistics.
- Optionally filter factors by `alpha`.

### 2.3 Interaction Detection

API: `gignn.geodetector.interaction_detector`

Purpose:

- Evaluate pairwise factor interactions.
- Report individual q values and combined interaction q values.
- Classify interactions as `nonlinear_enhance`, `bi_enhance`, `independent`, or `weaken`.

### 2.4 Feature Selection

API: `gignn.geodetector.GeoDetectorSelector`

Purpose:

- Fit discretization metadata on training features.
- Rank factors using GeoDetector scores — `strategy="interaction"` (default) ranks by degree centrality in the top-`top_interactions` interaction network per the paper's §2.2.1/Fig. 8; `strategy="factor"` ranks by factor q alone.
- Select all significant factors or `top_k` factors.
- Transform future data using stored discretization metadata and selected features.

## 3. Graph Construction

### 3.1 Explicit Edge Validation

API: `gignn.graph.build_spatial_edges(edge_index=..., num_nodes=...)`

Purpose:

- Validate user-provided graph edges.
- Accept either `(2, num_edges)` or `(num_edges, 2)` layout.
- Enforce integer node indices, non-empty edges, in-range nodes, and maximum edge limits.

### 3.2 Geometry-Based Graphs

API: `gignn.graph.build_spatial_edges(geometry=..., strategy=...)`

Strategies:

- `knn`: connect each node to its `k` nearest neighbors by centroid coordinates.
- `distance`: connect nodes within a positive distance threshold.
- `contiguity`: connect geometries whose `touches` or `intersects` methods return true.

Geometry objects must expose numeric `x/y` or `centroid.x/centroid.y` for `knn` and `distance` strategies. Automatic graph construction is guarded by `MAX_AUTO_GRAPH_NODES`.

### 3.3 Graph Utilities

APIs:

- `build_adjacency_dict`: converts edge index to a deduplicated adjacency dictionary.
- `calculate_gaussian_edge_attributes`: computes Gaussian edge attributes from source/target node feature differences.
- `calculate_edge_weights`: aggregates per-feature Gaussian attributes into scalar edge weights.
- `filter_isolated_nodes`: removes isolated nodes and returns retained node indices.
- `build_pyg_data`: builds `torch_geometric.data.Data` with `x`, `y`, `edge_index`, optional `edge_attr`, optional `edge_weight`, and a label encoder.

## 4. GNN Model and Training

### 4.1 Notebook-Aligned GIGNN

Internal model: `gignn.models.GIGNN`

Purpose:

- Match the core notebook structure with `NeighborAggregator`, `GCN`, and `GIGNN` modules.
- Use `num_neighbors_list` to express per-hop neighbor sampling, such as notebook `[6, 12]`.
- Consume Gaussian edge weights during neighbor aggregation.
- Return output logits for node classification.

`gignn.models.GraphSAGEClassifier` remains available as a lower-level compatibility model, but `GIGNNClassifier` now uses the weighted multi-hop `GIGNN` by default.

This is a notebook-core implementation boundary. It does not claim full reproduction of every paper dataset, baseline experiment, plot, or saved artifact.

### 4.2 Training Utilities

APIs: `TrainingConfig`, `TrainingResult`, `train_node_classifier`, `calculate_class_weights`, `set_random_seed`

Purpose:

- Train a node classifier for a fixed number of epochs.
- Use class-balanced cross-entropy weights.
- Use Adam optimizer with configurable learning rate and weight decay.
- Seed Python, NumPy, and Torch when `random_state` is provided.
- Return per-epoch training losses.

## 5. Estimators

### 5.1 `GIGNNClassifier`

Purpose:

- Direct, sklearn-style estimator for users who already know their graph and model configuration.
- Handles graph resolution, Gaussian edge attributes/weights, PyG conversion, weighted multi-hop GIGNN training, prediction, probability prediction, scoring, and checkpoint save/load.

Main methods:

- `fit(X, y, geometry=None, edge_index=None)`
- `predict(X, geometry=None, edge_index=None)`
- `predict_proba(X, geometry=None, edge_index=None)`
- `score(X, y, geometry=None, edge_index=None)`
- `save(path)`
- `load(path)`

Learned attributes:

- `model_`
- `label_encoder_`
- `n_features_in_`
- `feature_names_in_`
- `classes_`
- `training_result_`

### 5.2 `GIGNNAutoClassifier`

Purpose:

- High-level workflow estimator that searches feature discovery parameters, graph strategies, and model parameters.
- Fits a selected lower-level `GIGNNClassifier`.
- Emits reports for every major decision stage.

Main stages:

1. Validate that `geometry` or `edge_index` is available.
2. Resolve disjoint train/validation/test partitions by random holdout, explicit external labels, or indivisible spatial groups.
3. Fit ordinal target encoding and GeoDetector metadata on training rows only.
4. Build train, validation, and test graphs independently for every geometry candidate; filter and locally reindex explicit edges per partition.
5. Select graph/model candidates using `validation_report_`, evaluate the selected candidate once using `generalization_report_`, and then fit the deployment estimator on the full graph.
6. Store selected features, selected edge index, estimator, reports, and bounded save/load state.
7. Support prediction, probability prediction, and identity/topology-locked scenario simulation.

Learned attributes:

- `selector_`
- `estimator_`
- `selected_features_`
- `edge_index_`
- `discovery_report_`
- `graph_report_`
- `model_selection_report_`
- `validation_report_`
- `generalization_report_`

## 6. Reports and Diagnostics

### 6.1 Automatic Workflow Reports

Dataclasses:

- `SpatialRelationshipReport`
- `GraphSelectionReport`
- `ModelSelectionReport`
- `GeneralizationReport`
- `SimulationReport`

These reports expose feature selection, graph selection, model selection, validation metrics, and scenario changes.

### 6.2 Generalization Diagnostics

APIs: `generalization_diagnostics`, `model_accuracy_table`

Purpose:

- Compare observed and predicted labels.
- Produce accuracy, classification report, and bias table; bias is `predicted_code - observed_code` (positive = overestimation, paper Fig. 10).
- Label bias as `Correct`, `Overestimated (...)`, or `Underestimated (...)` based on ordered class codes.
- Test spatial autocorrelation of residuals with `morans_i` (`edge_index` also adds `bias_moran` to diagnostics).
- Run transductive k-fold evaluation with `cross_validate`, pooling out-of-fold predictions and their Moran's I.
- Build accuracy comparison tables for plotting or reporting.

## 7. Plotting

Top-level plotting exports are loaded lazily from `gignn.plotting`.

Functions:

- `plot_classification_map`
- `plot_prediction_comparison`
- `plot_observed_prediction_bias`
- `plot_accuracy_comparison`
- `plot_discretization_histogram`
- `plot_training_curves`
- `plot_metric_boxplot`
- `plot_interaction_network`

All plotting functions validate required columns, return Matplotlib objects, and only write files when `output_path` is provided.

## 8. Persistence and Safety Limits

### 8.1 Checkpoints

APIs: `GIGNNClassifier.save`, `GIGNNClassifier.load`, `GIGNNAutoClassifier.save`, `GIGNNAutoClassifier.load`

Purpose:

- Store estimator parameters, state dict, classes, feature count, and feature names.
- Load direct-estimator checkpoints on CPU using `torch.load(..., weights_only=True)`.
- Store automatic workflows as bounded `.gignn` ZIP archives with JSON metadata and NPY arrays loaded using `allow_pickle=False`.
- Validate file and expanded size, archive members, array hashes/dtypes/shapes, schema and package version, class/feature counts, tensor counts, and tensor sizes before reconstruction.

### 8.2 Resource Limits

Key guards:

- `MAX_AUTO_GRAPH_NODES = 1_000`
- `MAX_EDGES = 1_000_000`
- `MAX_FEATURES = 10_000`
- `MAX_CLASSES = 10_000`
- `MAX_GRAPH_NEIGHBORS = 1_000`
- `MAX_HIDDEN_CHANNEL = 4_096`
- `MAX_HIDDEN_LAYERS = 8`
- `MAX_TRAINING_EPOCHS = 10_000`
- `MAX_CHECKPOINT_BYTES = 100 MiB`
- `MAX_CHECKPOINT_TENSORS = 128`
- `MAX_CHECKPOINT_TENSOR_ELEMENTS = 10_000_000`
