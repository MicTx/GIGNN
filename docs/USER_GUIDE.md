# User Guide

This guide shows the intended user workflow for GIGNN: prepare a spatial-unit table, build or provide a graph, train a classifier, inspect reports, simulate feature changes, and visualize outputs.

The current package implements the notebook GIGNN core. It does not claim full reproduction of every paper experiment or external dataset; see [Implementation Status](IMPLEMENTATION_STATUS.md).

## 1. Prepare Inputs

Each row must represent one spatial unit, sample, grid cell, or region. Feature columns must be numeric and finite.

```python
import pandas as pd

X = pd.DataFrame(
    {
        "ndvi": [0.12, 0.44, 0.51, 0.83],
        "dem": [12.0, 19.0, 21.0, 35.0],
        "road_density": [1.0, 2.5, 2.1, 4.2],
    }
)
y = pd.Series(["low", "medium", "medium", "high"])
```

Labels may be strings or non-contiguous integers. GIGNN encodes labels internally and returns predictions in the original label space.

## 2. Provide Spatial Structure

GIGNN needs graph structure. You can provide it explicitly or derive it from geometry.

### Explicit `edge_index`

```python
import numpy as np

edge_index = np.array(
    [
        [0, 1, 1, 2, 2, 3],
        [1, 0, 2, 1, 3, 2],
    ]
)
```

`edge_index` may be shaped `(2, num_edges)` or `(num_edges, 2)`. Node indices must be integer, non-negative, and in range.

### Geometry

```python
class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.centroid = self

geometry = [Point(index, 0) for index in range(len(X))]
```

For `knn` and `distance` graphs, each geometry item must expose finite numeric `x/y` or `centroid.x/centroid.y`. KNN ties are resolved by ascending row index; distance graphs include pairs exactly at the threshold and emit both directions. For `contiguity`, geometry objects must provide `touches` or `intersects` methods.

## 3. Use The Automatic Workflow

`GIGNNAutoClassifier` is the recommended entry point when you want GIGNN to select features, graph strategy, and model parameters.

```python
from gignn import GIGNNAutoClassifier

model = GIGNNAutoClassifier(
    detector_grid={"discretizer": ["quantiles"], "max_bins": [3]},
    graph_grid={"strategy": ["knn"], "k": [4]},
    model_grid={"hidden_channels": [(64,)], "learning_rate": [0.01], "max_epochs": [100]},
    top_k_features=None,
    random_state=42,
)

model.fit(
    X,
    y,
    geometry=geometry,
    class_order=["low", "medium", "high"],
)
predictions = model.predict(X, geometry=geometry)
probabilities = model.predict_proba(X, geometry=geometry)
```

After fitting, inspect these attributes:

| Attribute | Meaning |
| --- | --- |
| `selected_features_` | Features selected by GeoDetector ranking. |
| `edge_index_` | Edge index selected for the final model. |
| `discovery_report_` | Feature q scores, interactions, discretizer, and selected features. |
| `graph_report_` | Graph candidates, selected strategy, and selected edge index. |
| `model_selection_report_` | Candidate model parameters and validation accuracy. |
| `validation_report_` | Validation-only metrics used for candidate selection. |
| `generalization_report_` | Independent test accuracy, macro F1, weighted F1, and per-class metrics. |

## 4. Use The Direct Estimator

Use `GIGNNClassifier` when you already know the graph and model configuration.

```python
from gignn import GIGNNClassifier

classifier = GIGNNClassifier(
    hidden_channels=(128, 64),
    num_neighbors_list=(6, 12, 12),
    graph_builder="knn",
    k_neighbors=4,
    max_epochs=100,
    learning_rate=0.01,
    random_state=42,
)

classifier.fit(X, y, geometry=geometry)
predictions = classifier.predict(X, geometry=geometry)
probabilities = classifier.predict_proba(X, geometry=geometry)
accuracy = classifier.score(X, y, geometry=geometry)
```

`num_neighbors_list` follows the notebook's per-hop neighbor sampling semantics. The default is `(6, 12)` for the default one-hidden-layer model; when you configure two hidden layers as above, provide three hop sizes.

Use an integer `random_state` when each call must rebuild the same sampled neighborhoods. With `random_state=None`, sampling consumes NumPy's global random stream; consecutive calls can differ, while `set_random_seed(seed)` can replay the sequence.

The direct estimator validates that prediction feature columns match the training columns exactly, including order.

## 5. Run Scenario Simulation

`simulate` compares baseline and scenario features with the same fitted selector and estimator.

```python
scenario = X.copy()
scenario["ndvi"] = scenario["ndvi"] + 0.1

simulation = model.simulate(scenario, baseline_X=X, geometry=geometry)

simulation.baseline_predictions
simulation.scenario_predictions
simulation.probability_delta
simulation.changed_mask
```

`probability_delta` is `scenario_probabilities - baseline_probabilities`. `changed_mask` is true where predicted classes changed.

## 6. Standardize A Spatial Dataset

```python
from gignn import standardize_spatial_dataset

dataset, report = standardize_spatial_dataset(
    geo_df,
    id_column="SA2_CODE21",
    target_column="Urbanization level",
    geometry_column="geometry",
    fill_missing=0.0,
)

X = dataset.features
y = dataset.target
geometry = dataset.geometry
```

Use explicit `feature_columns` if your frame contains non-feature numeric columns.

## 7. Diagnose Generalization

```python
from gignn import generalization_diagnostics, model_accuracy_table

diagnostics = generalization_diagnostics(
    observed=y,
    predicted=predictions,
    class_names={"low": "Low", "medium": "Medium", "high": "High"},
    class_order=["low", "medium", "high"],
)

accuracy_table = model_accuracy_table({"GIGNN": diagnostics.accuracy, "Baseline": 0.5}, year=2021)
```

Use `class_order` when labels are ordinal or not naturally sortable. Bias values are computed as `predicted_code - observed_code` (positive = overestimated, paper Fig. 10). Pass `edge_index` to also get `diagnostics.bias_moran`, the Moran's I test of residual spatial autocorrelation; `morans_i(values, edge_index)` runs it standalone, and `cross_validate` evaluates a transductive k-fold protocol.

## 8. Plot Outputs

```python
from gignn import plot_accuracy_comparison, plot_discretization_histogram, plot_training_curves

plot_discretization_histogram(X, "ndvi", intervals=[0.3, 0.6])
plot_accuracy_comparison(accuracy_table)
plot_training_curves(training_history)
```

Plotting functions return Matplotlib objects. They write files only when `output_path` is provided.

## 9. Save And Load A Model

```python
classifier.save("gignn.pt")
loaded = GIGNNClassifier.load("gignn.pt")

auto_model.save("workflow.gignn")
loaded_auto = GIGNNAutoClassifier.load("workflow.gignn")
```

Both classes enforce bounded checkpoint loading. The direct estimator uses a weights-only PyTorch checkpoint, normalizes NumPy scalar labels/configuration values, and rejects custom payload objects before replacing the target. The automatic workflow uses a separate ZIP/JSON/NPY format that does not call `torch.load`; it restores the frozen selector, ordinal label order, split indices, selected graph configuration, validation/test reports, training losses, and lower-level model weights. Runtime and checkpoint loading use the same parameter-level auto-grid validation, including support for optional `threshold=None`.
