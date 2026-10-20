import numpy as np
import pandas as pd
import pytest

from gignn import GIGNNAutoClassifier
from gignn.exceptions import NotFittedError, ValidationError
from gignn.limits import MAX_FEATURES, MAX_HIDDEN_LAYERS


class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.centroid = self


class Interval:
    def __init__(self, left, right):
        self.left = left
        self.right = right
        self.x = (left + right) / 2
        self.y = 0
        self.centroid = self

    def touches(self, other):
        return self.right == other.left or other.right == self.left

    def intersects(self, other):
        return not (self.right < other.left or other.right < self.left)


def test_auto_classifier_discovers_relationships_and_reports(indicator_frame, labels):
    geometry = [Point(i, 0) for i in range(len(indicator_frame))]
    model = GIGNNAutoClassifier(
        detector_grid={"discretizer": ["quantiles"], "max_bins": [2]},
        graph_grid={"strategy": ["knn"], "k": [1]},
        model_grid={
            "hidden_channels": [(8,)],
            "learning_rate": [0.01],
            "max_epochs": [1],
        },
        random_state=42,
    )

    model.fit(indicator_frame, labels, geometry=geometry)

    assert model.discovery_report_.selected_features
    assert "ndvi" in model.discovery_report_.factor_scores.index
    assert model.graph_report_.selected_strategy == "knn"
    assert model.model_selection_report_.best_params["hidden_channels"] == (8,)
    assert 0.0 <= model.generalization_report_.accuracy <= 1.0


def test_graph_selection_report_copies_and_freezes_selected_edges():
    from gignn.reports import GraphSelectionReport

    edges = np.array([[0, 1], [1, 0]], dtype=np.int64)
    report = GraphSelectionReport(
        selected_strategy="edge_index",
        selected_params={},
        candidates=pd.DataFrame(),
        selected_edge_index=edges,
    )
    edges[0, 0] = 1

    assert report.selected_edge_index[0, 0] == 0
    with pytest.raises(ValueError, match="read-only"):
        report.selected_edge_index[0, 0] = 1


def test_auto_classifier_predict_and_simulate(indicator_frame, labels):
    geometry = [Point(i, 0) for i in range(len(indicator_frame))]
    model = GIGNNAutoClassifier(
        detector_grid={"discretizer": ["quantiles"], "max_bins": [2]},
        graph_grid={"strategy": ["knn"], "k": [1]},
        model_grid={
            "hidden_channels": [(8,)],
            "learning_rate": [0.01],
            "max_epochs": [1],
        },
        random_state=42,
    ).fit(indicator_frame, labels, geometry=geometry)
    scenario = indicator_frame.copy()
    scenario["ndvi"] = scenario["ndvi"] + 0.1

    predictions = model.predict(indicator_frame, geometry=geometry)
    simulation = model.simulate(scenario, baseline_X=indicator_frame, geometry=geometry)

    assert predictions.shape == (len(labels),)
    assert simulation.baseline_predictions.shape == (len(labels),)
    assert simulation.scenario_predictions.shape == (len(labels),)
    assert simulation.probability_delta.shape[0] == len(labels)


def test_auto_classifier_rejects_mismatched_feature_and_target_rows():
    frame = pd.DataFrame({"feature": [0.0, 1.0, 2.0]})

    with pytest.raises(ValidationError, match="same number of rows"):
        GIGNNAutoClassifier().fit(
            frame,
            pd.Series([0, 1]),
            geometry=[Point(index, 0) for index in range(len(frame))],
        )


def test_auto_classifier_requires_geometry_or_edge_index(indicator_frame, labels):
    model = GIGNNAutoClassifier(
        detector_grid={"discretizer": ["quantiles"], "max_bins": [2]},
        graph_grid={"strategy": ["knn"], "k": [1]},
        model_grid={
            "hidden_channels": [(8,)],
            "learning_rate": [0.01],
            "max_epochs": [1],
        },
        random_state=42,
    )

    with pytest.raises(ValidationError, match="geometry or edge_index"):
        model.fit(indicator_frame, labels)


def test_auto_classifier_predict_before_fit_raises(indicator_frame):
    model = GIGNNAutoClassifier()

    with pytest.raises(NotFittedError):
        model.predict(indicator_frame)


def test_auto_classifier_rejects_excessive_grid(indicator_frame, labels):
    geometry = [Point(i, 0) for i in range(len(indicator_frame))]
    model = GIGNNAutoClassifier(
        detector_grid={
            "discretizer": ["quantiles"] * 20,
            "max_bins": list(range(1, 21)),
        },
        graph_grid={"strategy": ["knn"], "k": [1]},
        model_grid={
            "hidden_channels": [(8,)],
            "learning_rate": [0.01],
            "max_epochs": [1],
        },
        max_grid_candidates=10,
    )

    with pytest.raises(ValidationError, match="candidate budget"):
        model.fit(indicator_frame, labels, geometry=geometry)


def test_auto_classifier_rejects_product_search_budget(indicator_frame, labels):
    geometry = [Point(index, 0) for index in range(len(indicator_frame))]
    model = GIGNNAutoClassifier(
        detector_grid={"discretizer": ["quantiles"], "max_bins": [2]},
        graph_grid={"strategy": ["knn"], "k": [1, 2, 3]},
        model_grid={
            "hidden_channels": [(8,)],
            "learning_rate": [0.01, 0.02, 0.03],
            "max_epochs": [1],
        },
        max_grid_candidates=8,
    )

    with pytest.raises(ValidationError, match="candidate budget"):
        model.fit(indicator_frame, labels, geometry=geometry)


@pytest.mark.parametrize(
    "budget",
    [True, float("nan"), float("inf"), MAX_FEATURES + 1],
)
def test_auto_classifier_rejects_invalid_candidate_budget(
    indicator_frame, labels, budget
):
    geometry = [Point(index, 0) for index in range(len(indicator_frame))]
    model = GIGNNAutoClassifier(
        detector_grid={"discretizer": ["quantiles"], "max_bins": [2]},
        graph_grid={"strategy": ["knn"], "k": [1]},
        model_grid={
            "hidden_channels": [(8,)],
            "learning_rate": [0.01],
            "max_epochs": [1],
        },
        max_grid_candidates=budget,
    )

    with pytest.raises(ValidationError, match="candidate budget"):
        model.fit(indicator_frame, labels, geometry=geometry)


def test_auto_classifier_rejects_invalid_graph_grid(indicator_frame, labels):
    geometry = [Point(i, 0) for i in range(len(indicator_frame))]
    model = GIGNNAutoClassifier(
        detector_grid={"discretizer": ["quantiles"], "max_bins": [2]},
        graph_grid={"strategy": ["knn"], "k": ["bad"]},
        model_grid={
            "hidden_channels": [(8,)],
            "learning_rate": [0.01],
            "max_epochs": [1],
        },
    )

    with pytest.raises(ValidationError, match="k must be"):
        model.fit(indicator_frame, labels, geometry=geometry)


@pytest.mark.parametrize(
    ("grid_name", "invalid_grid", "message"),
    [
        ("detector_grid", {"unknown": [1]}, "unknown keys"),
        ("detector_grid", {"max_bins": [True]}, "max_bins"),
        (
            "detector_grid",
            {"max_bins": [MAX_FEATURES + 1]},
            "max_bins.*resource limits",
        ),
        ("graph_grid", {"unknown": [1]}, "unknown keys"),
        ("graph_grid", {"k": [True]}, "k must be"),
        ("graph_grid", {"threshold": [np.nan]}, "threshold"),
        ("graph_grid", {"threshold": [{"nested": 1}]}, "threshold"),
        ("model_grid", {"unknown": [1]}, "unknown keys"),
        ("model_grid", {"hidden_channels": [()]}, "hidden_channels"),
        (
            "model_grid",
            {"hidden_channels": [[8] * (MAX_HIDDEN_LAYERS + 1)]},
            "layer resource limits",
        ),
        ("model_grid", {"learning_rate": [np.inf]}, "learning_rate"),
        ("model_grid", {"max_epochs": [True]}, "max_epochs"),
    ],
)
def test_auto_search_grids_enforce_parameter_schema(grid_name, invalid_grid, message):
    from gignn.auto import _validate_search_grids

    grids = {
        "detector_grid": {"discretizer": ["quantiles"], "max_bins": [2]},
        "graph_grid": {"strategy": ["knn"], "k": [1]},
        "model_grid": {
            "hidden_channels": [(8,)],
            "learning_rate": [0.01],
            "max_epochs": [1],
        },
    }
    grids[grid_name] = invalid_grid

    with pytest.raises(ValidationError, match=message):
        _validate_search_grids(**grids)


def test_auto_search_grid_rejects_oversized_candidate_list():
    from gignn.auto import _validate_search_grids

    with pytest.raises(ValidationError, match="resource limits"):
        _validate_search_grids(
            detector_grid={"discretizer": ["quantiles"] * (MAX_FEATURES + 1)},
            graph_grid={"strategy": ["knn"], "k": [1]},
            model_grid={
                "hidden_channels": [(8,)],
                "learning_rate": [0.01],
                "max_epochs": [1],
            },
        )


def test_auto_classifier_handles_non_default_index(indicator_frame, labels):
    frame = indicator_frame.copy()
    frame.index = [10, 20, 30, 40, 50, 60]
    geometry = [Point(i, 0) for i in range(len(frame))]
    model = GIGNNAutoClassifier(
        detector_grid={"discretizer": ["quantiles"], "max_bins": [2]},
        graph_grid={"strategy": ["knn"], "k": [1]},
        model_grid={
            "hidden_channels": [(8,)],
            "learning_rate": [0.01],
            "max_epochs": [1],
        },
        random_state=42,
    )

    model.fit(frame, labels, geometry=geometry)

    assert not model.discovery_report_.interaction_scores.empty


def test_auto_classifier_discovers_features_from_training_holdout_only(
    monkeypatch, indicator_frame, labels
):
    import gignn.auto as auto_module

    observed_lengths = []
    original_fit = auto_module.GeoDetectorSelector.fit

    def spy_fit(self, X, y):
        observed_lengths.append(len(y))
        return original_fit(self, X, y)

    monkeypatch.setattr(auto_module.GeoDetectorSelector, "fit", spy_fit)
    geometry = [Point(i, 0) for i in range(len(indicator_frame))]
    model = GIGNNAutoClassifier(
        detector_grid={"discretizer": ["quantiles"], "max_bins": [2]},
        graph_grid={"strategy": ["knn"], "k": [1]},
        model_grid={
            "hidden_channels": [(8,)],
            "learning_rate": [0.01],
            "max_epochs": [1],
        },
        validation_fraction=1 / 3,
        random_state=42,
    )

    model.fit(indicator_frame, labels, geometry=geometry)

    assert observed_lengths == [len(model.train_indices_)]
    assert len(model.validation_indices_) > 0
    assert len(model.test_indices_) > 0


def test_auto_classifier_selector_metadata_ignores_validation_and_test_features(
    monkeypatch,
):
    import gignn.auto as auto_module

    frame = pd.DataFrame(
        {
            "primary": [0.0, 1.0, 2.0, 3.0, 10.0, 11.0, 20.0, 21.0, 22.0],
            "secondary": [9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0],
        }
    )
    labels = pd.Series([0, 1, 2, 0, 1, 2, 0, 1, 2])
    split_labels = [
        "train",
        "train",
        "train",
        "train",
        "validation",
        "validation",
        "test",
        "test",
        "test",
    ]
    edges = np.array(
        [[0, 1, 2, 3, 4, 5, 6, 7, 7, 8], [1, 0, 3, 2, 5, 4, 7, 6, 8, 7]],
        dtype=np.int64,
    )

    class FakeEstimator:
        def __init__(self, **params):
            self.classes_ = np.array([0, 1, 2], dtype=object)

        def fit(self, X, y, *, edge_index):
            return self

        def predict(self, X, *, edge_index):
            return np.zeros(len(X), dtype=np.int64)

    monkeypatch.setattr(auto_module, "GIGNNClassifier", FakeEstimator)

    def fit_selector(values):
        model = GIGNNAutoClassifier(
            detector_grid={"discretizer": ["quantiles"], "max_bins": [2]},
            model_grid={
                "hidden_channels": [(8,)],
                "learning_rate": [0.01],
                "max_epochs": [1],
            },
            top_k_features=1,
            random_state=42,
        ).fit(
            values,
            labels,
            edge_index=edges,
            split_labels=split_labels,
        )
        return model.selector_

    baseline = fit_selector(frame)
    perturbed = frame.copy(deep=True)
    perturbed.loc[4:, :] = perturbed.loc[4:, :] * 10_000
    changed = fit_selector(perturbed)

    assert baseline.selected_features_ == changed.selected_features_
    pd.testing.assert_frame_equal(baseline.scores_, changed.scores_)
    for feature in baseline.discretizers_:
        np.testing.assert_array_equal(
            baseline.discretizers_[feature].bins,
            changed.discretizers_[feature].bins,
        )


def test_auto_classifier_uses_fitted_discretizer_for_prediction(
    monkeypatch, indicator_frame, labels
):
    import gignn.auto as auto_module

    geometry = [Point(i, 0) for i in range(len(indicator_frame))]
    model = GIGNNAutoClassifier(
        detector_grid={"discretizer": ["quantiles"], "max_bins": [2]},
        graph_grid={"strategy": ["knn"], "k": [1]},
        model_grid={
            "hidden_channels": [(8,)],
            "learning_rate": [0.01],
            "max_epochs": [1],
        },
        random_state=42,
    ).fit(indicator_frame, labels, geometry=geometry)
    captured = []
    original_predict = auto_module.GIGNNClassifier.predict

    def spy_predict(self, X, *, geometry=None, edge_index=None):
        captured.append(X.copy(deep=True))
        return original_predict(self, X, geometry=geometry, edge_index=edge_index)

    monkeypatch.setattr(auto_module.GIGNNClassifier, "predict", spy_predict)

    model.predict(indicator_frame * 100.0, geometry=geometry)

    assert captured
    assert set(np.unique(captured[0].to_numpy())) <= {1, 2}


def test_auto_classifier_training_knn_is_independent_of_holdout_geometry():
    frame = pd.DataFrame({"feature": [0.0, 1.0, 2.0, 3.0]})
    training_indices = np.array([0, 1, 2], dtype=np.int64)
    model = GIGNNAutoClassifier(
        graph_grid={"strategy": ["knn"], "k": [1]},
    )

    def training_edges(holdout_x):
        geometry = [Point(0, 0), Point(10, 0), Point(20, 0), Point(holdout_x, 0)]
        candidate = model._graph_candidates(
            frame,
            geometry=geometry,
            edge_index=None,
            partition_indices={"train": training_indices},
        )[0]
        return candidate["partition_edges"]["train"]

    distant_holdout_edges = training_edges(100)
    nearby_holdout_edges = training_edges(0.1)

    np.testing.assert_array_equal(distant_holdout_edges, nearby_holdout_edges)


def test_auto_classifier_builds_geometry_graphs_with_local_partition_indices():
    frame = pd.DataFrame({"feature": np.arange(6, dtype=float)})
    partitions = {
        "train": np.array([0, 2, 4], dtype=np.int64),
        "validation": np.array([1, 3, 5], dtype=np.int64),
    }

    distance_candidate = GIGNNAutoClassifier(
        graph_grid={"strategy": ["distance"], "k": [1], "threshold": [5.0]},
    )._graph_candidates(
        frame,
        geometry=[Point(index, 0) for index in range(len(frame))],
        edge_index=None,
        partition_indices=partitions,
    )[
        0
    ]
    contiguity_candidate = GIGNNAutoClassifier(
        graph_grid={"strategy": ["contiguity"], "k": [1]},
    )._graph_candidates(
        frame,
        geometry=[Interval(index, index + 2) for index in range(len(frame))],
        edge_index=None,
        partition_indices=partitions,
    )[
        0
    ]

    for candidate in (distance_candidate, contiguity_candidate):
        for partition, edges in candidate["partition_edges"].items():
            assert edges.min() == 0
            assert edges.max() < 3
            metrics = candidate["partition_metrics"][partition]
            assert metrics["num_nodes"] == 3
            assert metrics["num_edges"] == edges.shape[1]
            assert len(metrics["edge_hash"]) == 64


def test_auto_classifier_partition_graph_hashes_ignore_other_partition_geometry():
    frame = pd.DataFrame({"feature": np.arange(6, dtype=float)})
    partitions = {
        "train": np.array([0, 1, 2], dtype=np.int64),
        "validation": np.array([3, 4, 5], dtype=np.int64),
    }

    def training_hash(strategy, geometry, **params):
        candidate = GIGNNAutoClassifier(
            graph_grid={"strategy": [strategy], "k": [1], **params}
        )._graph_candidates(
            frame,
            geometry=geometry,
            edge_index=None,
            partition_indices=partitions,
        )[
            0
        ]
        return candidate["partition_metrics"]["train"]["edge_hash"]

    stable_points = [Point(index, 0) for index in range(6)]
    moved_points = stable_points[:3] + [Point(100 + index, 0) for index in range(3)]
    assert training_hash("knn", stable_points) == training_hash("knn", moved_points)
    assert training_hash("distance", stable_points, threshold=[2.0]) == training_hash(
        "distance", moved_points, threshold=[2.0]
    )

    stable_intervals = [Interval(index, index + 1) for index in range(6)]
    moved_intervals = stable_intervals[:3] + [
        Interval(100 + index, 101 + index) for index in range(3)
    ]
    assert training_hash("contiguity", stable_intervals) == training_hash(
        "contiguity", moved_intervals
    )


def test_auto_classifier_filters_explicit_edges_per_partition():
    frame = pd.DataFrame({"feature": np.arange(6, dtype=float)})
    partitions = {
        "train": np.array([0, 2, 4], dtype=np.int64),
        "validation": np.array([1, 3, 5], dtype=np.int64),
    }
    edge_index = np.array(
        [
            [0, 2, 2, 4, 1, 3, 3, 5, 0, 1, 4, 5],
            [2, 0, 4, 2, 3, 1, 5, 3, 1, 0, 5, 4],
        ],
        dtype=np.int64,
    )

    candidate = GIGNNAutoClassifier()._graph_candidates(
        frame,
        geometry=None,
        edge_index=edge_index,
        partition_indices=partitions,
    )[0]

    np.testing.assert_array_equal(
        candidate["partition_edges"]["train"],
        np.array([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=np.int64),
    )
    np.testing.assert_array_equal(
        candidate["partition_edges"]["validation"],
        np.array([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=np.int64),
    )
    assert (
        candidate["partition_metrics"]["train"]
        == candidate["partition_metrics"]["validation"]
    )


def test_auto_classifier_explicit_training_hash_ignores_cross_partition_edges():
    frame = pd.DataFrame({"feature": np.arange(6, dtype=float)})
    partitions = {
        "train": np.array([0, 1, 2], dtype=np.int64),
        "validation": np.array([3, 4, 5], dtype=np.int64),
    }
    training_edges = np.array([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=np.int64)
    validation_edges = np.array([[3, 4, 4, 5], [4, 3, 5, 4]], dtype=np.int64)
    first = np.concatenate([training_edges, validation_edges], axis=1)
    second = np.concatenate(
        [training_edges, validation_edges[:, ::-1], np.array([[0, 5], [5, 0]])],
        axis=1,
    )

    def training_hash(edges):
        candidate = GIGNNAutoClassifier()._graph_candidates(
            frame,
            geometry=None,
            edge_index=edges,
            partition_indices=partitions,
        )[0]
        return candidate["partition_metrics"]["train"]["edge_hash"]

    assert training_hash(first) == training_hash(second)


def test_auto_classifier_uses_distinct_validation_and_test_partitions(monkeypatch):
    import gignn.auto as auto_module

    frame = pd.DataFrame({"feature": np.arange(9, dtype=float)})
    labels = pd.Series(["low", "medium", "high"] * 3)
    geometry = [Point(index, 0) for index in range(len(frame))]
    split_labels = [
        "train",
        "train",
        "train",
        "train",
        "validation",
        "validation",
        "test",
        "test",
        "test",
    ]
    prediction_lengths = []

    class FakeEstimator:
        def __init__(self, **params):
            self.classes_ = np.array([0, 1, 2], dtype=object)
            self.label_encoder_ = self

        def fit(self, X, y, *, edge_index):
            return self

        def predict(self, X, *, edge_index):
            prediction_lengths.append(len(X))
            if len(X) == 2:
                return np.asarray([1, 2], dtype=np.int64)
            return np.asarray([0, 0, 0], dtype=np.int64)

        def predict_proba(self, X, *, edge_index):
            return np.full((len(X), 3), 1 / 3)

        def inverse_transform(self, encoded):
            return self.classes_[np.asarray(list(encoded), dtype=int)]

    monkeypatch.setattr(auto_module, "GIGNNClassifier", FakeEstimator)
    model = GIGNNAutoClassifier(
        detector_grid={"discretizer": ["quantiles"], "max_bins": [2]},
        graph_grid={"strategy": ["knn"], "k": [1]},
        model_grid={
            "hidden_channels": [(8,)],
            "learning_rate": [0.01],
            "max_epochs": [1],
        },
        random_state=42,
    ).fit(
        frame,
        labels,
        geometry=geometry,
        split_labels=split_labels,
        class_order=["low", "medium", "high"],
    )

    assert prediction_lengths == [2, 3]
    assert set(model.train_indices_).isdisjoint(model.validation_indices_)
    assert set(model.train_indices_).isdisjoint(model.test_indices_)
    assert set(model.validation_indices_).isdisjoint(model.test_indices_)
    assert sorted(
        np.concatenate(
            [model.train_indices_, model.validation_indices_, model.test_indices_]
        )
    ) == list(range(len(frame)))
    assert model.validation_report_.accuracy == 1.0
    assert model.validation_report_.partition == "validation"
    np.testing.assert_array_equal(
        model.validation_report_.indices, model.validation_indices_
    )
    assert model.generalization_report_.accuracy == pytest.approx(1 / 3)
    assert model.generalization_report_.partition == "test"
    np.testing.assert_array_equal(
        model.generalization_report_.indices, model.test_indices_
    )


def test_auto_classifier_uses_ascending_order_for_numeric_labels_without_class_order():
    frame = pd.DataFrame({"feature": np.arange(9, dtype=float)})
    labels = pd.Series([2, 1, 2, 1, 2, 1, 0, 0, 0])
    geometry = [Point(index, 0) for index in range(len(frame))]
    split_labels = [
        "train",
        "train",
        "train",
        "train",
        "validation",
        "validation",
        "test",
        "test",
        "test",
    ]

    model = GIGNNAutoClassifier(
        detector_grid={"discretizer": ["quantiles"], "max_bins": [2]},
        graph_grid={"strategy": ["knn"], "k": [1]},
        model_grid={
            "hidden_channels": [(8,)],
            "learning_rate": [0.01],
            "max_epochs": [1],
        },
    ).fit(
        frame,
        labels,
        geometry=geometry,
        split_labels=split_labels,
    )

    assert model.class_order_ == (0, 1, 2)


def test_auto_classifier_requires_class_order_for_string_labels(
    indicator_frame, labels
):
    geometry = [Point(index, 0) for index in range(len(indicator_frame))]

    unordered_labels = pd.Series(labels.astype(str).tolist())

    with pytest.raises(ValidationError, match="class_order"):
        GIGNNAutoClassifier(
            detector_grid={"discretizer": ["quantiles"], "max_bins": [2]},
            graph_grid={"strategy": ["knn"], "k": [1]},
            model_grid={
                "hidden_channels": [(8,)],
                "learning_rate": [0.01],
                "max_epochs": [1],
            },
        ).fit(indicator_frame, unordered_labels, geometry=geometry)


def test_auto_classifier_selects_graph_by_validation_accuracy(monkeypatch):
    import gignn.auto as auto_module

    frame = pd.DataFrame({"feature": [0.0, 0.1, 1.0, 1.1, 2.0, 2.1]})
    labels = pd.Series(["a", "a", "b", "b", "c", "c"])
    geometry = [Point(i, 0) for i in range(len(frame))]
    current_k = {"value": None}

    class FakeEstimator:
        def __init__(self, **params):
            self.params = params
            self.k = current_k["value"]
            self.label_encoder_ = self
            self.classes_ = np.array(["a", "b", "c"], dtype=object)

        def fit(self, X, y, *, edge_index):
            current_k["value"] = 2 if edge_index.shape[1] == 8 else 1
            self.classes_ = np.array(sorted(pd.unique(y)), dtype=object)
            return self

        def predict(self, X, *, edge_index):
            return np.zeros(len(X), dtype=np.int64)

        def predict_proba(self, X, *, edge_index):
            return np.full((len(X), len(self.classes_)), 1 / len(self.classes_))

        def inverse_transform(self, encoded):
            return self.classes_[np.asarray(list(encoded), dtype=int)]

    def spy_accuracy(y_true, y_pred):
        return 0.9 if current_k["value"] == 2 else 0.2

    monkeypatch.setattr(auto_module, "GIGNNClassifier", FakeEstimator)
    monkeypatch.setattr(auto_module, "accuracy_score", spy_accuracy)
    model = GIGNNAutoClassifier(
        detector_grid={"discretizer": ["quantiles"], "max_bins": [2]},
        graph_grid={"strategy": ["knn"], "k": [1, 2]},
        model_grid={
            "hidden_channels": [(8,)],
            "learning_rate": [0.01],
            "max_epochs": [1],
        },
        random_state=42,
    ).fit(
        frame,
        labels,
        geometry=geometry,
        class_order=["a", "b", "c"],
        split_labels=[
            "train",
            "train",
            "train",
            "train",
            "validation",
            "test",
        ],
    )

    assert model.graph_report_.selected_params["k"] == 2
    assert "validation_accuracy" in model.graph_report_.candidates.columns


def test_auto_classifier_passes_notebook_neighbor_list_to_estimator(monkeypatch):
    import gignn.auto as auto_module

    frame = pd.DataFrame({"feature": [0.0, 0.1, 1.0, 1.1, 2.0, 2.1]})
    labels = pd.Series(["a", "a", "b", "b", "c", "c"])
    geometry = [Point(i, 0) for i in range(len(frame))]
    seen_neighbor_lists = []

    class FakeEstimator:
        def __init__(self, **params):
            self.params = params
            self.label_encoder_ = self
            self.classes_ = np.array(["a", "b", "c"], dtype=object)
            seen_neighbor_lists.append(params.get("num_neighbors_list"))

        def fit(self, X, y, *, edge_index):
            self.classes_ = np.array(sorted(pd.unique(y)), dtype=object)
            return self

        def predict(self, X, *, edge_index):
            return np.zeros(len(X), dtype=np.int64)

        def predict_proba(self, X, *, edge_index):
            return np.full((len(X), len(self.classes_)), 1 / len(self.classes_))

        def inverse_transform(self, encoded):
            return self.classes_[np.asarray(list(encoded), dtype=int)]

    monkeypatch.setattr(auto_module, "GIGNNClassifier", FakeEstimator)
    GIGNNAutoClassifier(
        detector_grid={"discretizer": ["quantiles"], "max_bins": [2]},
        graph_grid={"strategy": ["knn"], "k": [1]},
        model_grid={
            "hidden_channels": [(8,)],
            "learning_rate": [0.01],
            "max_epochs": [1],
            "num_neighbors_list": [(6, 12)],
        },
        random_state=42,
    ).fit(
        frame,
        labels,
        geometry=geometry,
        class_order=["a", "b", "c"],
    )

    assert seen_neighbor_lists
    assert set(seen_neighbor_lists) == {(6, 12)}


def test_auto_classifier_skips_failed_graph_candidate(monkeypatch):
    import gignn.auto as auto_module

    frame = pd.DataFrame({"feature": np.arange(9, dtype=float)})
    labels = pd.Series([0, 1, 2] * 3)
    geometry = [Point(index, 0) for index in range(len(frame))]

    class FakeEstimator:
        def __init__(self, **params):
            self.classes_ = np.array([0, 1, 2], dtype=object)
            self.label_encoder_ = self

        def fit(self, X, y, *, edge_index):
            return self

        def predict(self, X, *, edge_index):
            return np.zeros(len(X), dtype=np.int64)

        def predict_proba(self, X, *, edge_index):
            return np.full((len(X), 3), 1 / 3)

        def inverse_transform(self, encoded):
            return self.classes_[np.asarray(list(encoded), dtype=int)]

    monkeypatch.setattr(auto_module, "GIGNNClassifier", FakeEstimator)
    model = GIGNNAutoClassifier(
        detector_grid={"discretizer": ["quantiles"], "max_bins": [2]},
        graph_grid={
            "strategy": ["distance"],
            "k": [1],
            "threshold": [0.1, 2.1],
        },
        model_grid={
            "hidden_channels": [(8,)],
            "learning_rate": [0.01],
            "max_epochs": [1],
        },
        random_state=42,
    ).fit(
        frame,
        labels,
        geometry=geometry,
        split_labels=[
            "train",
            "train",
            "train",
            "train",
            "validation",
            "validation",
            "test",
            "test",
            "test",
        ],
    )

    candidates = model.graph_report_.candidates.set_index("threshold")
    assert candidates.loc[0.1, "status"] == "failed"
    assert "no edges" in candidates.loc[0.1, "failure_reason"].lower()
    assert candidates.loc[2.1, "status"] == "ok"
    assert model.graph_report_.selected_params["threshold"] == 2.1


def test_auto_classifier_test_edge_failure_does_not_filter_candidate(monkeypatch):
    import gignn.auto as auto_module

    frame = pd.DataFrame({"feature": np.arange(6, dtype=float)})
    labels = pd.Series([0, 1, 2, 0, 1, 2])
    edge_index = np.array(
        [[0, 1, 2, 3], [1, 0, 3, 2]],
        dtype=np.int64,
    )

    class FakeEstimator:
        def __init__(self, **params):
            self.classes_ = np.array([0, 1, 2], dtype=object)
            self.label_encoder_ = self

        def fit(self, X, y, *, edge_index):
            return self

        def predict(self, X, *, edge_index):
            return np.zeros(len(X), dtype=np.int64)

        def predict_proba(self, X, *, edge_index):
            return np.full((len(X), 3), 1 / 3)

        def inverse_transform(self, encoded):
            return self.classes_[np.asarray(list(encoded), dtype=int)]

    monkeypatch.setattr(auto_module, "GIGNNClassifier", FakeEstimator)
    model = GIGNNAutoClassifier(
        detector_grid={"discretizer": ["quantiles"], "max_bins": [2]},
        model_grid={
            "hidden_channels": [(8,)],
            "learning_rate": [0.01],
            "max_epochs": [1],
        },
    )

    with pytest.raises(ValidationError, match="independent test"):
        model.fit(
            frame,
            labels,
            edge_index=edge_index,
            split_labels=[
                "train",
                "train",
                "validation",
                "validation",
                "test",
                "test",
            ],
        )


def test_auto_classifier_simulate_validates_identity_and_topology():
    frame = pd.DataFrame({"feature": np.arange(9, dtype=float)})
    labels = pd.Series([0, 1, 2] * 3)
    geometry = [Point(index, 0) for index in range(len(frame))]
    node_ids = [f"unit-{index}" for index in range(len(frame))]
    model = GIGNNAutoClassifier(
        detector_grid={"discretizer": ["quantiles"], "max_bins": [2]},
        graph_grid={"strategy": ["knn"], "k": [1]},
        model_grid={
            "hidden_channels": [(8,)],
            "learning_rate": [0.01],
            "max_epochs": [1],
        },
        random_state=42,
    ).fit(
        frame,
        labels,
        geometry=geometry,
        node_ids=node_ids,
        split_labels=[
            "train",
            "train",
            "train",
            "train",
            "validation",
            "validation",
            "test",
            "test",
            "test",
        ],
    )

    with pytest.raises(ValidationError, match="same row count"):
        model.simulate(
            frame.iloc[:-1],
            baseline_X=frame,
            geometry=geometry,
            node_ids=node_ids,
        )
    with pytest.raises(ValidationError, match="same feature columns"):
        model.simulate(
            frame[["feature"]].rename(columns={"feature": "renamed"}),
            baseline_X=frame,
            geometry=geometry,
            node_ids=node_ids,
        )
    reordered = frame.copy(deep=True)
    reordered.index = list(reversed(frame.index))
    with pytest.raises(ValidationError, match="same row index"):
        model.simulate(
            reordered,
            baseline_X=frame,
            geometry=geometry,
            node_ids=node_ids,
        )
    with pytest.raises(ValidationError, match="node_ids"):
        model.simulate(
            frame,
            baseline_X=frame,
            geometry=geometry,
            node_ids=list(reversed(node_ids)),
        )
    with pytest.raises(ValidationError, match="topology"):
        model.simulate(
            frame,
            baseline_X=frame,
            geometry=geometry,
            node_ids=node_ids,
            scenario_edge_index=np.array([[0, 1], [1, 0]], dtype=np.int64),
        )


def test_auto_classifier_simulate_limits_discretized_and_edge_changes(monkeypatch):
    from gignn.graph import calculate_gaussian_edge_attributes

    frame = pd.DataFrame({"feature": np.arange(9, dtype=float)})
    labels = pd.Series([0, 1, 2] * 3)
    geometry = [Point(index, 0) for index in range(len(frame))]
    model = GIGNNAutoClassifier(
        detector_grid={"discretizer": ["quantiles"], "max_bins": [2]},
        graph_grid={"strategy": ["knn"], "k": [1]},
        model_grid={
            "hidden_channels": [(8,)],
            "learning_rate": [0.01],
            "max_epochs": [1],
        },
        random_state=42,
    ).fit(
        frame,
        labels,
        geometry=geometry,
        split_labels=[
            "train",
            "train",
            "train",
            "train",
            "validation",
            "validation",
            "test",
            "test",
            "test",
        ],
    )
    captured = []
    original_predict_proba = model.estimator_.predict_proba

    def spy_predict_proba(X, *, geometry=None, edge_index=None):
        captured.append((X.copy(deep=True), np.asarray(edge_index).copy()))
        return original_predict_proba(X, geometry=geometry, edge_index=edge_index)

    monkeypatch.setattr(model.estimator_, "predict_proba", spy_predict_proba)
    bins = model.selector_.discretizers_["feature"].bins
    same_bin = frame.copy(deep=True)
    same_bin.loc[0, "feature"] = min(frame.loc[0, "feature"] + 0.1, bins[0] - 0.1)

    model.simulate(same_bin, baseline_X=frame, geometry=geometry)

    baseline_selected, edges = captured[-2]
    same_bin_selected, same_edges = captured[-1]
    pd.testing.assert_frame_equal(baseline_selected, same_bin_selected)
    np.testing.assert_array_equal(edges, same_edges)

    crossed = frame.copy(deep=True)
    crossed.loc[0, "feature"] = bins[0] + 0.1
    model.simulate(crossed, baseline_X=frame, geometry=geometry)
    crossed_selected, crossed_edges = captured[-1]
    changed_rows = np.flatnonzero(
        baseline_selected.to_numpy() != crossed_selected.to_numpy()
    )
    np.testing.assert_array_equal(changed_rows, np.array([0]))
    np.testing.assert_array_equal(edges, crossed_edges)

    baseline_attr = calculate_gaussian_edge_attributes(
        baseline_selected, edges, sigma=model.estimator_.edge_weight_sigma
    )
    crossed_attr = calculate_gaussian_edge_attributes(
        crossed_selected, crossed_edges, sigma=model.estimator_.edge_weight_sigma
    )
    changed_edges = np.flatnonzero(np.any(baseline_attr != crossed_attr, axis=1))
    incident_edges = np.flatnonzero(np.any(edges == 0, axis=0))
    assert set(changed_edges).issubset(set(incident_edges))
    assert len(changed_edges) > 0


def test_auto_classifier_simulate_accepts_equivalent_edge_order():
    frame = pd.DataFrame({"feature": np.arange(9, dtype=float)})
    labels = pd.Series([0, 1, 2] * 3)
    edges = np.array(
        [
            [0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8],
            [1, 0, 2, 1, 3, 2, 4, 3, 5, 4, 6, 5, 7, 6, 8, 7],
        ],
        dtype=np.int64,
    )
    model = GIGNNAutoClassifier(
        detector_grid={"discretizer": ["quantiles"], "max_bins": [2]},
        model_grid={
            "hidden_channels": [(8,)],
            "learning_rate": [0.01],
            "max_epochs": [1],
        },
        random_state=42,
    ).fit(
        frame,
        labels,
        edge_index=edges,
        split_labels=[
            "train",
            "train",
            "train",
            "train",
            "validation",
            "validation",
            "test",
            "test",
            "test",
        ],
    )

    report = model.simulate(
        frame,
        baseline_X=frame,
        edge_index=edges,
        scenario_edge_index=edges[:, ::-1],
    )

    np.testing.assert_array_equal(
        report.baseline_predictions, report.scenario_predictions
    )


def test_auto_classifier_rejects_holdout_without_induced_edges(indicator_frame, labels):
    edge_index = np.array([[0, 1], [1, 0]], dtype=np.int64)
    model = GIGNNAutoClassifier(
        detector_grid={"discretizer": ["quantiles"], "max_bins": [2]},
        graph_grid={"strategy": ["knn"], "k": [1]},
        model_grid={
            "hidden_channels": [(8,)],
            "learning_rate": [0.01],
            "max_epochs": [1],
        },
        random_state=42,
    )

    with pytest.raises(ValidationError, match="no induced edges"):
        model.fit(indicator_frame, labels, edge_index=edge_index)


def test_auto_classifier_rejects_excessive_model_params(indicator_frame, labels):
    geometry = [Point(i, 0) for i in range(len(indicator_frame))]
    model = GIGNNAutoClassifier(
        detector_grid={"discretizer": ["quantiles"], "max_bins": [2]},
        graph_grid={"strategy": ["knn"], "k": [1]},
        model_grid={
            "hidden_channels": [(100_000,)],
            "learning_rate": [0.01],
            "max_epochs": [1],
        },
    )

    with pytest.raises(ValidationError, match="hidden_channels"):
        model.fit(indicator_frame, labels, geometry=geometry)
