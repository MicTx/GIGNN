# GIGNN Implementation Status

This document audits the package against the published paper — Ma, X., Zhang, Z., & Song, Y. (2025). Geographically informed graph neural networks. *Spatial Statistics*, 69, 100920. https://doi.org/10.1016/j.spasta.2025.100920 — and against the reference notebook `research/notebooks/space-prediction-generalization.ipynb`, which preserves the inference-stage code of the case study.

## Short Answer

The package implements the GIGNN mechanism the paper *describes*, end to end:

- GeoDetector factor and interaction detectors (Eq. (1)), Jenks natural breaks, and the interaction-network factor selection of §2.2.1 / Fig. 8.
- Stratum re-quantification (codes 1..n) as node features (§2.2.2).
- Gaussian kernel edge attributes with `sigma = 2`, consumed as edge weights that scale neighbor messages (Eq. (2), Eq. (4)).
- GraphSAGE-style `NeighborAggregator` + `GCN` layers (Eq. (3)-(5)), `num_neighbors_list=(6, 12)`, one 128-unit hidden layer, per-batch neighbor resampling during training, and deterministic full-neighborhood inference.
- Transductive masked training (`train_mask`) and the residual diagnostics of §5.2.2: bias = predicted − observed code (Fig. 10 convention) and Moran's I (validated to match `esda.Moran` exactly).

Not implemented, by scope:

- Reproduction of the paper's headline numbers (84.1 % accuracy 2016, 81.0 % generalization 2021, Moran's I = 0.089/0.014): they require the Greater Perth SA2/ABS/VIIRS dataset, which is not distributed.
- The simulation dataset generator (Eq. (6)-(8)), the GAT/GIN/ChebNet/MPNN/RF baselines, and the k-fold driver (five rounds × 1000 epochs is supported via `fit(..., train_mask=...)` but no driver function is packaged).
- Raw data ingestion, hosted services, automatic publishing — see [Current Boundary](#current-boundary).

## Paper-to-Code Matrix

| Paper element | Implementation | Evidence | Status / caveat |
| --- | --- | --- | --- |
| §2.2.1 / Eq. (1) GeoDetector q, factor detector | `geodetector/factor.py`: `geodetector_q`, `factor_detector` with ANOVA F p-value; `target_to_numeric` accepts numeric and *ordered* categorical targets only | `tests/test_geodetector.py` | Implemented |
| Eq. (1) Jenks natural breaks, 5 strata for Xs | `discretization.py`: exact Fisher–Jenks DP (`natural_breaks`/`fisher_jenks`); `maximum_breaks` is a separate widest-gap method. `GeoDetectorSelector` defaults `discretizer="natural_breaks"`, `max_bins=5` | `tests/test_geodetector.py` | Implemented; the DP is deterministic — the notebook's `mapclassify.NaturalBreaks` optimizes the same objective by a randomized heuristic, so cutpoints may differ slightly |
| §2.2.2 re-quantification, codes 1..n | `spatial_discretization` labels strata 1..n ascending by interval position; Jenks strata are value-ordered, so for Jenks this equals the paper's "ascending group-level statistic" relabeling | `tests/test_geodetector.py` | Implemented |
| §2.2.1 interaction detector | `interaction_detector`: `q_interaction`, `p_value_interaction`, and the standard five-type classification with numerical tolerance | `tests/test_geodetector.py` | Implemented |
| §4.2 / Fig. 8 interaction-network selection ("top three key factors from the interaction detection network with high centrality degree and Q value"; network = top 10 interactions) | `GeoDetectorSelector(strategy="interaction", top_interactions=10, top_k=...)`: factors ranked by degree in the top-10 interaction network, ties by factor q | `tests/test_geodetector.py` | Implemented; "high centrality degree and Q value" is under-specified in the paper — operationalized as degree desc, then q desc; reproduce the case study with `top_k=3` |
| §2.2.2 / Eq. (2) Gaussian kernel, sigma=2, on re-quantified features | `calculate_gaussian_edge_attributes` (per-feature `exp(-(Δx_k)²/2σ²)`) + `calculate_edge_weights(mode="product")` = `exp(-‖xi−xj‖²/2σ²)` exactly; `mean`/`min`/`max` are non-paper alternatives. Collapse of near-zero weights raises `EdgeWeightWarning` | `tests/test_model_training.py::test_eq2_product_edge_weight_equals_vector_gaussian_kernel` | Implemented; requires stratum-code features — feed `GeoDetectorSelector` output, not raw scales |
| §4.2 step 1 topology (notebook: buffer(100) + intersects; simulation §3: queen contiguity on hexagons) | `build_spatial_edges`: `knn` (default), `distance`, `contiguity` (touches/intersects), or user `edge_index` | `tests/test_graph_builders.py` | Generalized: the notebook's exact buffer adjacency is reproducible via `geometry` + `contiguity` on buffered polygons or an explicit `edge_index`; no `buffer` convenience wrapper |
| §2.2.3 / Eq. (3)-(5) NA + GCN layers, k-order aggregation | `models/gignn.py`: `NeighborAggregator`, `GCN`, `GIGNN`; weighted mean `A(v) = (1/\|N\|)Σ w_uv h_u` with `normalize_edge_weights=False` | `tests/test_model_training.py::test_gignn_forward_uses_edge_weights`, `test_unnormalized_mean_limits_increment_of_dissimilar_neighbors` | Implemented — and *stricter than the notebook*: see §Divergences below |
| Notebook hyperparameters: `HIDDEN_DIM=[128, out]`, `NUM_NEIGHBORS_LIST=[6,12]`, LR 0.01, EPOCHS 1000, batch 64 | `GIGNNClassifier` defaults `hidden_channels=(128,)`, `num_neighbors_list=(6, 12)`, `learning_rate=0.01`; `max_epochs` defaults to 100 (paper values available via params) | `tests/test_estimator.py` | Implemented; per-node mini-batching is not packaged — training is full-graph |
| §5.2.1 5-fold CV protocol | Transductive `fit(..., train_mask=...)`: all nodes stay in the graph, only masked labels enter the loss | `tests/test_model_training.py::test_train_node_classifier_ignores_labels_outside_mask` | Mechanism implemented; the fold driver is not packaged. Note: running `GeoDetectorSelector` on all labels *before* CV leaks target information into every fold — the paper's workflow does this; fold-internal selection requires orchestrating it yourself |
| §5.2.2 / Fig. 10 bias analysis | `generalization_diagnostics`: bias = predicted − observed code, positive = overestimation | `tests/test_generalization_standard.py` | Implemented; diverges from the notebook — see §Divergences |
| §5.2.2 Moran's I of residuals | `morans_i(values, edge_index)`: row-standardized contiguity, normality and permutation inference | `tests/test_generalization_standard.py` (exact match with `esda.Moran` 2.9.0 reference values) | Implemented |

## Divergences Between Paper, Notebook, and Package

These are inconsistencies found in the paper/notebook evidence itself. The package follows the *paper's stated design*; none of these are silently copied.

1. **Edge attributes never reached the published model.** The notebook computes `edge_attr` and `edge_attr_diff` and stores them in `Data`, but `NeighborAggregator.forward(neighbor_feature)` takes no weight argument — aggregation is an unweighted mean. The package scales neighbor messages by the Eq. (2) weights, as the paper describes.
2. **Notebook inference is stochastic.** `multihop_sampling` runs unseeded inside `torch.no_grad()`, so the reported 84.1 %/81.0 % figures are draws from a sampled estimator. The package defaults to deterministic full-neighborhood inference (`inference="full"`), the expectation of that estimator; `inference="sampled"` reproduces notebook behavior.
3. **Bias sign.** The notebook computes `observed − predicted` but labels positive values "Overestimated", while the paper's Fig. 10 caption defines positive bias as `predicted − observed`. Under Fig. 10's own definition the reported "25 overestimated / 9 underestimated" counts swap. The package implements the Fig. 10 convention.
4. **Training loop absent from the notebook.** Cells 49–53 only *load* checkpoints; optimizer, loss function, and how class weights entered the loss are not preserved in the artifact. The package uses Adam + class-weighted cross-entropy (weights by inverse class distribution, as computed in cell 43).
5. **Baseline graphs differ from the target graph.** The target model uses buffer/intersection adjacency while control models use a different `edge_index`, confounding model-vs-model comparisons in the paper's tables.

## Current Boundary

The implemented core is suitable for local research workflows over aligned spatial-unit indicator tables. Users still provide cleaned feature tables, labels, and geometry or graph edges.

Out of scope for the current package:

- raw imagery ingestion and remote data acquisition
- the published case-study datasets and checkpoint files
- a packaged k-fold driver and the Eq. (6)-(8) simulation generator
- exact notebook filesystem paths
- hosted inference or training services
- automatic package publishing

## Regression Commands

Use these commands when changing implementation or documentation around GIGNN core alignment:

```bash
.venv/bin/python -m pytest tests/test_geodetector.py tests/test_graph_builders.py tests/test_model_training.py tests/test_estimator.py tests/test_auto.py tests/test_generalization_standard.py
.venv/bin/python -m pytest
.venv/bin/python examples/quickstart.py
.venv/bin/python -m build
.venv/bin/python -m black --check gignn tests examples
```
