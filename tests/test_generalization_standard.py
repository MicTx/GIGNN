import numpy as np
import pandas as pd
import pytest

from gignn.exceptions import ValidationError
from gignn.generalization import (
    generalization_diagnostics,
    model_accuracy_table,
    morans_i,
    standardize_spatial_dataset,
)
from gignn.graph import (
    build_adjacency_dict,
    calculate_gaussian_edge_attributes,
    filter_isolated_nodes,
)
from gignn.plotting import plot_accuracy_comparison, plot_observed_prediction_bias


class PlotGeometryFrame(pd.DataFrame):
    @property
    def _constructor(self):
        return PlotGeometryFrame

    def plot(self, *args, **kwargs):
        ax = kwargs.get("ax")
        if ax is None:
            import matplotlib.pyplot as plt

            _, ax = plt.subplots()
        column = kwargs.get("column")
        values = np.arange(len(self)) if column is None else self[column].to_numpy()
        ax.scatter(np.arange(len(self)), values)
        return ax


def test_standardize_spatial_dataset_preserves_input_and_reports_columns():
    frame = pd.DataFrame(
        {
            "SA2_CODE21": ["a", "b", "c"],
            "geometry": ["g1", "g2", "g3"],
            "ndvi": [0.1, np.nan, 0.3],
            "dem": [10, 20, 30],
            "Urbanization level": [0, 1, 2],
        }
    )

    dataset, report = standardize_spatial_dataset(
        frame,
        target_column="Urbanization level",
        id_column="SA2_CODE21",
        geometry_column="geometry",
        fill_missing=0.0,
    )

    assert np.isnan(frame.loc[1, "ndvi"])
    assert dataset.features.columns.tolist() == ["ndvi", "dem"]
    assert dataset.target.tolist() == [0, 1, 2]
    assert dataset.ids.tolist() == ["a", "b", "c"]
    assert dataset.geometry.tolist() == ["g1", "g2", "g3"]
    assert report.feature_columns == ["ndvi", "dem"]
    assert report.missing_filled == {"ndvi": 1, "dem": 0}


def test_graph_generalization_helpers_build_edge_attributes_and_filter_isolates():
    node_features = pd.DataFrame({"x": [0.0, 1.0, 2.0], "y": [1.0, 1.5, 2.0]})
    edge_index = np.array([[0, 1], [1, 0]])

    adjacency = build_adjacency_dict(edge_index, num_nodes=3)
    edge_attributes = calculate_gaussian_edge_attributes(
        node_features, edge_index, sigma=2.0
    )
    filtered_adjacency, kept_indices = filter_isolated_nodes(adjacency)

    assert adjacency == {0: [1], 1: [0], 2: []}
    assert edge_attributes.shape == (2, 2)
    assert np.all(edge_attributes <= 1.0)
    assert filtered_adjacency == {0: [1], 1: [0]}
    assert kept_indices.tolist() == [0, 1]


def test_generalization_diagnostics_outputs_bias_and_report():
    observed = pd.Series([0, 1, 2, 2])
    predicted = pd.Series([0, 2, 1, 2])
    class_names = {0: "Low", 1: "Moderate", 2: "High"}

    diagnostics = generalization_diagnostics(
        observed, predicted, class_names=class_names
    )

    assert diagnostics.accuracy == 0.5
    assert diagnostics.bias_frame["Bias"].tolist() == [0, 1, -1, 0]
    assert diagnostics.bias_frame["Bias Type"].tolist() == [
        "Correct",
        "Overestimated (1)",
        "Underestimated (-1)",
        "Correct",
    ]
    assert diagnostics.bias_frame["Validation"].tolist()[1] == "Moderate-High"
    assert "Low" in diagnostics.classification_report.columns
    assert diagnostics.bias_moran is None


def test_generalization_diagnostics_supports_string_labels():
    observed = pd.Series(["low", "medium", "high"])
    predicted = pd.Series(["low", "high", "medium"])

    diagnostics = generalization_diagnostics(
        observed, predicted, class_order=["low", "medium", "high"]
    )

    assert diagnostics.bias_frame["Bias"].tolist() == [0.0, 1.0, -1.0]
    assert diagnostics.bias_frame["Bias Type"].tolist() == [
        "Correct",
        "Overestimated (1)",
        "Underestimated (-1)",
    ]
    assert diagnostics.bias_frame["Validation"].tolist() == [
        "low-low",
        "medium-high",
        "high-medium",
    ]


def _rook_lattice(rows, cols):
    edges = []
    for row in range(rows):
        for col in range(cols):
            node = row * cols + col
            if col + 1 < cols:
                edges.append((node, node + 1))
            if row + 1 < rows:
                edges.append((node, node + cols))
    return np.asarray(edges).T


# Reference values from esda 2.9.0 / libpysal 4.14.1:
# Moran(y, W(neighbors), transformation="r", permutations=0).
@pytest.mark.parametrize(
    ("values", "edge_index", "expected"),
    [
        (
            [1, 2, 3, 4, 2, 3, 4, 5, 3, 4, 5, 6, 4, 5, 6, 8],
            _rook_lattice(4, 4),
            (
                0.7092765201952953,
                0.03483932461873638,
                4.157143924687837,
                3.2225098872935844e-05,
            ),
        ),
        (
            [1 if (node // 4 + node % 4) % 2 else -1 for node in range(16)],
            _rook_lattice(4, 4),
            (-1.0, 0.03483932461873638, -5.000367374919108, 5.722117795165149e-07),
        ),
        (
            [3.0, 1.0, 4.0, 1.0, 5.0],
            np.asarray([[0, 1, 2], [1, 2, 3]]),
            (-0.5800781250000001, 0.19921875, -0.7395229193772929, 0.4595895277208111),
        ),
    ],
)
def test_morans_i_matches_esda_reference(values, edge_index, expected):
    result = morans_i(values, edge_index, permutations=0)

    statistic, variance, z_norm, p_norm = expected
    assert result.expected == pytest.approx(-1.0 / (len(values) - 1))
    assert result.statistic == pytest.approx(statistic, rel=1e-12)
    assert result.variance_norm == pytest.approx(variance, rel=1e-12)
    assert result.z_norm == pytest.approx(z_norm, rel=1e-12)
    assert result.p_norm == pytest.approx(p_norm, rel=1e-9)
    assert result.z_sim is None and result.p_sim is None


def test_morans_i_ignores_edge_direction_duplicates_and_self_loops():
    values = [1, 2, 3, 4, 2, 3, 4, 5, 3, 4, 5, 6, 4, 5, 6, 8]
    edges = _rook_lattice(4, 4)
    noisy = np.concatenate([edges, edges[::-1], np.asarray([[0, 5], [0, 5]])], axis=1)

    assert morans_i(values, noisy, permutations=0) == morans_i(
        values, edges, permutations=0
    )


def test_morans_i_permutation_inference_is_reproducible():
    values = [1, 2, 3, 4, 2, 3, 4, 5, 3, 4, 5, 6, 4, 5, 6, 8]
    edges = _rook_lattice(4, 4)

    first = morans_i(values, edges, permutations=199, random_state=3)
    second = morans_i(values, edges, permutations=199, random_state=3)

    assert first == second
    assert first.permutations == 199
    assert first.p_sim == pytest.approx(1 / 200)
    assert first.z_sim > 3


def test_morans_i_constant_values_are_undefined():
    result = morans_i([2.0, 2.0, 2.0, 2.0], np.asarray([[0, 1, 2], [1, 2, 3]]))

    assert np.isnan(result.statistic) and np.isnan(result.p_norm)
    assert np.isnan(result.p_sim)


@pytest.mark.parametrize(
    ("values", "edge_index", "kwargs", "message"),
    [
        ([1.0, 2.0], [[0], [1]], {}, "at least three"),
        ([1.0, 2.0, 3.0], [[0, 1], [0, 1]], {}, "two distinct nodes"),
        ([1.0, 2.0, 3.0], [[0], [1]], {"permutations": -1}, "non-negative"),
        ([1.0, np.nan, 3.0], [[0], [1]], {}, "finite"),
    ],
)
def test_morans_i_rejects_invalid_inputs(values, edge_index, kwargs, message):
    with pytest.raises(ValidationError, match=message):
        morans_i(values, np.asarray(edge_index), **kwargs)


def test_generalization_diagnostics_tests_bias_autocorrelation():
    observed = pd.Series([0, 1, 2, 2, 1, 0])
    predicted = pd.Series([0, 2, 2, 1, 1, 1])
    edges = np.asarray([[0, 1, 2, 3, 4], [1, 2, 3, 4, 5]])

    diagnostics = generalization_diagnostics(
        observed, predicted, edge_index=edges, permutations=99, random_state=0
    )

    assert diagnostics.bias_moran == morans_i(
        diagnostics.bias_frame["Bias"], edges, permutations=99, random_state=0
    )


def test_generalization_diagnostics_requires_order_for_unsortable_labels():
    observed = pd.Series(["low", 1, "high"])
    predicted = pd.Series(["low", "high", 1])

    with pytest.raises(ValidationError, match="class_order must be provided"):
        generalization_diagnostics(observed, predicted)


def test_model_accuracy_table_orders_requested_models():
    table = model_accuracy_table({"GIGNN": 0.75, "RF": 0.5}, year=2021)

    assert table.to_dict("records") == [
        {"Year": 2021, "Model": "GIGNN", "Accuracy": 0.75},
        {"Year": 2021, "Model": "RF", "Accuracy": 0.5},
    ]


def test_latest_notebook_plotting_outputs_figures():
    frame = PlotGeometryFrame(
        {
            "Urbanization level": [0, 1, 2],
            "prediction": [0, 2, 1],
            "Bias Type": ["Correct", "Underestimated (-1)", "Overestimated (1)"],
        }
    )
    accuracy_table = pd.DataFrame(
        {"Year": [2021, 2021], "Model": ["GIGNN", "RF"], "Accuracy": [0.75, 0.5]}
    )

    comparison_figure, comparison_axes = plot_observed_prediction_bias(
        frame,
        observed_column="Urbanization level",
        prediction_column="prediction",
        bias_column="Bias Type",
        model_name="GIGNN",
        accuracy=0.75,
    )
    accuracy_figure, accuracy_ax = plot_accuracy_comparison(accuracy_table)

    assert len(comparison_axes) == 3
    assert comparison_axes[2].get_title() == "Bias"
    assert comparison_axes[2].collections
    assert accuracy_figure is accuracy_ax.figure
    assert accuracy_ax.get_ylabel() == "Accuracy"
