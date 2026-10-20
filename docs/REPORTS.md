# Reports

GIGNN reports are plain dataclasses and pandas tables exposed after fitting or diagnostics. They are intended for inspection, notebooks, and downstream documentation.

For publication-grade reproducibility, `export_research_artifacts` collects these reports into a directory bundle with `raw/`, `preprocessing/`, `process/`, and `results/` evidence layers. The bundle is separate from model checkpoints: checkpoints restore fitted models, while research artifacts document how a run was computed.

## Automatic Workflow Reports

`GIGNNAutoClassifier.fit` creates five main reports and one simulation report returned by `simulate`.

## `SpatialRelationshipReport`

Owner: `GIGNNAutoClassifier.discovery_report_`

Fields:

| Field | Type | Description |
| --- | --- | --- |
| `selected_features` | `list[str]` | Features selected by `GeoDetectorSelector`. |
| `factor_scores` | `pd.DataFrame` | Factor q scores and p-values indexed by factor name. |
| `interaction_scores` | `pd.DataFrame` | Pairwise GeoDetector interaction scores. |
| `discretizer` | `str` | Selected discretization method. |
| `max_bins` | `int` | Selected maximum bin count. |

Typical usage:

```python
model.discovery_report_.selected_features
model.discovery_report_.factor_scores
model.discovery_report_.interaction_scores
```

## `GraphSelectionReport`

Owner: `GIGNNAutoClassifier.graph_report_`

Fields:

| Field | Type | Description |
| --- | --- | --- |
| `selected_strategy` | `str` | Selected graph strategy, such as `edge_index`, `knn`, `distance`, or `contiguity`. |
| `selected_params` | `dict[str, object]` | Parameters used by the selected graph strategy. |
| `candidates` | `pd.DataFrame` | Candidate graph metrics, partition edge audit fields, failure reasons, and validation scores. |
| `selected_edge_index` | `np.ndarray` | Full-data deployment edge index used by the final model. |

Candidate tables include full-data graph metrics plus available `train_*`, `validation_*`, and `test_*` edge counts, node counts, and canonical SHA-256 edge hashes. Partition hashes describe the local selection/evaluation graphs; they do not imply that the final full-data deployment graph is independent of holdout geometry.

## `ModelSelectionReport`

Owner: `GIGNNAutoClassifier.model_selection_report_`

Fields:

| Field | Type | Description |
| --- | --- | --- |
| `best_params` | `dict[str, object]` | Best model parameters selected by validation accuracy. |
| `candidates` | `pd.DataFrame` | Candidate model/graph combinations and scores. |

Typical usage:

```python
model.model_selection_report_.best_params
model.model_selection_report_.candidates.sort_values("validation_accuracy", ascending=False)
```

## `GeneralizationReport`

Owner: `GIGNNAutoClassifier.validation_report_` and `GIGNNAutoClassifier.generalization_report_`

Fields:

| Field | Type | Description |
| --- | --- | --- |
| `accuracy` | `float` | Accuracy on the report's own partition. |
| `macro_f1` | `float` | Macro-averaged F1. |
| `weighted_f1` | `float` | Weighted F1. |
| `per_class` | `pd.DataFrame` | Per-class metrics from scikit-learn classification report. |

`validation_report_` is produced from the validation partition used for candidate selection. `generalization_report_` is produced only from the independent test partition after selection and does not influence the selected candidate. For external observed/predicted comparisons, use `generalization_diagnostics`.

## `SimulationReport`

Returned by `GIGNNAutoClassifier.simulate(...)`.

Fields:

| Field | Type | Description |
| --- | --- | --- |
| `baseline_predictions` | `np.ndarray` | Predicted labels for baseline features. |
| `scenario_predictions` | `np.ndarray` | Predicted labels for scenario features. |
| `baseline_probabilities` | `np.ndarray` | Class probabilities for baseline features. |
| `scenario_probabilities` | `np.ndarray` | Class probabilities for scenario features. |
| `probability_delta` | `np.ndarray` | `scenario_probabilities - baseline_probabilities`. |
| `changed_mask` | `np.ndarray` | Boolean mask where predicted labels changed. |

Both baseline and scenario features are transformed with the fitted `GeoDetectorSelector`.

## Dataset Reports

### `StandardizedSpatialDataset`

Returned by `standardize_spatial_dataset`.

| Field | Type | Description |
| --- | --- | --- |
| `frame` | `pd.DataFrame` | Reset-index copy of the full frame. |
| `features` | `pd.DataFrame` | Validated numeric feature table. |
| `target` | `pd.Series` | Target labels. |
| `ids` | `pd.Series | None` | Optional spatial-unit identifiers. |
| `geometry` | `pd.Series | None` | Optional geometry objects. |

### `SpatialDatasetReport`

Returned with `StandardizedSpatialDataset`.

| Field | Type | Description |
| --- | --- | --- |
| `id_column` | `str | None` | ID column name. |
| `target_column` | `str` | Target column name. |
| `feature_columns` | `list[str]` | Feature columns used. |
| `geometry_column` | `str | None` | Geometry column name. |
| `missing_filled` | `dict[str, int]` | Missing feature counts before fill. |
| `row_count` | `int` | Number of rows. |

## Diagnostic Reports

### `GeneralizationDiagnostics`

Returned by `generalization_diagnostics`.

| Field | Type | Description |
| --- | --- | --- |
| `accuracy` | `float` | Accuracy between observed and predicted labels. |
| `bias_frame` | `pd.DataFrame` | Row-level observed, prediction, bias, bias type, and validation label. |
| `classification_report` | `pd.DataFrame` | Scikit-learn classification report converted to a DataFrame. |
| `bias_moran` | `MoranResult \| None` | Moran's I of the bias over `edge_index`; `None` when no edge index is passed. |

`bias_frame` columns:

| Column | Description |
| --- | --- |
| `observed` | Original observed label. |
| `prediction` | Original predicted label. |
| `Bias` | `predicted_code - observed_code`; positive means the model overestimated the level (paper Fig. 10). |
| `Bias Type` | `Correct`, `Overestimated (...)`, or `Underestimated (...)`. |
| `Validation` | Text pair such as `ObservedLabel-PredictedLabel`. |

`MoranResult` fields follow `esda.Moran` semantics: `statistic`, `expected`, `variance_norm`, `z_norm`, `p_norm` (two-sided, normality assumption), and the permutation-inference `p_sim` (folded one-tailed pseudo p), `z_sim`, `permutations`; fields are NaN when values are constant.

### `CrossValidationReport`

Returned by `cross_validate` (transductive k-fold, paper §5.2.1).

| Field | Type | Description |
| --- | --- | --- |
| `fold_metrics` | `pd.DataFrame` | `fold`, `n_test`, `accuracy`, `f1_macro` per fold. |
| `out_of_fold_observed` | `pd.Series` | All labels (every node is held out exactly once). |
| `out_of_fold_predictions` | `pd.Series` | Pooled held-out predictions. |
| `bias_moran` | `MoranResult \| None` | Moran's I of out-of-fold bias; requires `edge_index`. |

Use `class_order` when label ordering has domain meaning.

## Training Reports

### `TrainingResult`

Owner: `GIGNNClassifier.training_result_`

| Field | Type | Description |
| --- | --- | --- |
| `losses` | `list[float]` | One loss value per epoch. |

`TrainingResult.losses` can be converted into a DataFrame for `plot_training_curves`.

## Research Artifact Bundle

`export_research_artifacts(model, output_dir, ...)` and `GIGNNAutoClassifier.export_research_artifacts(...)` return `ResearchArtifactBundle`.

| Field | Type | Description |
| --- | --- | --- |
| `root` | `Path` | Artifact bundle root directory. |
| `manifest_path` | `Path` | Path to `manifest.json`. |
| `files` | `dict[str, Path]` | Logical artifact names mapped to written files. |

Bundle layers:

| Layer | Purpose |
| --- | --- |
| `raw/` | Input schema, row/column summary, SHA-256 fingerprint, optional raw CSV when `include_raw=True`. |
| `preprocessing/` | Feature descriptions, missing counts, fill summary, target distribution, selected columns, and train/validation/test indices. |
| `process/` | GeoDetector factor and interaction scores, discretization bins, graph candidates, selected edge list, model candidates, training loss. |
| `results/` | Validation metrics, independent test metrics, per-class test report, predictions, probabilities, optional simulation outputs, and figure manifest. |

Raw table export is opt-in. By default the bundle records only schema, summary, and fingerprint so sensitive research data is not duplicated accidentally.
