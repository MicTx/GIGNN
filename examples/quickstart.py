import numpy as np
import pandas as pd

from gignn import (
    GIGNNAutoClassifier,
    generalization_diagnostics,
    model_accuracy_table,
    plot_accuracy_comparison,
    plot_discretization_histogram,
    plot_training_curves,
    standardize_spatial_dataset,
)
from gignn.graph import build_adjacency_dict, calculate_gaussian_edge_attributes


class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.centroid = self


X = pd.DataFrame(
    {
        "ndvi": [0.12, 0.44, 0.51, 0.83, 0.25, 0.68],
        "dem": [12.0, 19.0, 21.0, 35.0, 14.0, 28.0],
        "road_density": [1.0, 2.5, 2.1, 4.2, 1.3, 3.7],
    }
)
y = pd.Series(["low", "medium", "medium", "high", "low", "high"])
geometry = [Point(index, 0) for index in range(len(X))]
spatial_frame = X.assign(
    SA2_CODE21=[f"SA2_{index}" for index in range(len(X))],
    geometry=geometry,
    **{"Urbanization level": y},
)
dataset, dataset_report = standardize_spatial_dataset(
    spatial_frame,
    id_column="SA2_CODE21",
    target_column="Urbanization level",
    geometry_column="geometry",
    feature_columns=list(X.columns),
    fill_missing=0.0,
)

model = GIGNNAutoClassifier(
    detector_grid={"discretizer": ["quantiles"], "max_bins": [2]},
    graph_grid={"strategy": ["knn"], "k": [1]},
    model_grid={"hidden_channels": [(8,)], "learning_rate": [0.01], "max_epochs": [5]},
    random_state=42,
)
model.fit(X, y, geometry=geometry, class_order=["low", "medium", "high"])

scenario = X.copy()
scenario["ndvi"] = scenario["ndvi"] + 0.1
simulation = model.simulate(scenario, baseline_X=X, geometry=geometry)

predictions = model.predict(X, geometry=geometry)
adjacency = build_adjacency_dict(model.edge_index_, num_nodes=len(dataset.features))
edge_attr = calculate_gaussian_edge_attributes(dataset.features, model.edge_index_)
diagnostics = generalization_diagnostics(y, predictions)
accuracy_table = model_accuracy_table(
    {"GIGNN": diagnostics.accuracy, "Baseline": 0.5}, year=2021
)

print("selected features:", model.discovery_report_.selected_features)
print("selected graph:", model.graph_report_.selected_strategy)
print("best model params:", model.model_selection_report_.best_params)
print("generalization accuracy:", round(model.generalization_report_.accuracy, 3))
print("standardized features:", dataset_report.feature_columns)
print("adjacency nodes:", len(adjacency), "edge attributes:", edge_attr.shape)
training_history = pd.DataFrame(
    {
        "loss": model.estimator_.training_result_.losses,
        "train_accuracy": np.linspace(
            0.0, 1.0, len(model.estimator_.training_result_.losses)
        ),
        "test_accuracy": np.linspace(
            0.0,
            model.generalization_report_.accuracy,
            len(model.estimator_.training_result_.losses),
        ),
    }
)
plot_discretization_histogram(X, "ndvi", intervals=[0.3, 0.6])
plot_training_curves(training_history)
plot_accuracy_comparison(accuracy_table)

print("predictions:", predictions.tolist())
print("changed in simulation:", simulation.changed_mask.tolist())
print("bias types:", diagnostics.bias_frame["Bias Type"].tolist())
print("plotting: generated histogram, training-curve, and accuracy-comparison figures")
