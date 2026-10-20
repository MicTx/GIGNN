import numpy as np
import pytest
import torch

from gignn import GIGNNClassifier
from gignn.exceptions import NotFittedError, ValidationError
from gignn.models import GIGNN

pytestmark = pytest.mark.filterwarnings("ignore::gignn.exceptions.EdgeWeightWarning")


def test_gignn_classifier_fit_predict_score_with_edge_index(
    indicator_frame, labels, edge_index
):
    pytest.importorskip("torch")
    pytest.importorskip("torch_geometric")
    model = GIGNNClassifier(
        hidden_channels=(8,),
        max_epochs=1,
        random_state=42,
    )

    fitted = model.fit(indicator_frame, labels, edge_index=edge_index)
    predictions = fitted.predict(indicator_frame, edge_index=edge_index)
    probabilities = fitted.predict_proba(indicator_frame, edge_index=edge_index)
    score = fitted.score(indicator_frame, labels, edge_index=edge_index)

    assert fitted is model
    assert isinstance(fitted.model_, GIGNN)
    assert fitted.num_neighbors_list == (6, 12)
    assert predictions.shape == (len(labels),)
    assert set(predictions).issubset(set(labels))
    assert probabilities.shape == (len(labels), 3)
    np.testing.assert_allclose(
        probabilities.sum(axis=1), np.ones(len(labels)), rtol=1e-5
    )
    assert 0.0 <= score <= 1.0


def test_gignn_classifier_handles_binary_labels(indicator_frame, edge_index):
    pytest.importorskip("torch")
    pytest.importorskip("torch_geometric")
    labels = ["absent", "present", "present", "absent", "present", "absent"]
    model = GIGNNClassifier(hidden_channels=(8,), max_epochs=1, random_state=42)

    model.fit(indicator_frame, labels, edge_index=edge_index)
    predictions = model.predict(indicator_frame, edge_index=edge_index)

    assert set(predictions).issubset(set(labels))


def test_gignn_classifier_preserves_declared_class_order_when_training_class_missing(
    indicator_frame, edge_index
):
    labels = [0, 2, 2, 0, 2, 0]
    model = GIGNNClassifier(
        hidden_channels=(8,),
        class_order=(0, 1, 2),
        max_epochs=1,
        random_state=42,
    )

    model.fit(indicator_frame, labels, edge_index=edge_index)
    probabilities = model.predict_proba(indicator_frame, edge_index=edge_index)

    assert model.classes_.tolist() == [0, 1, 2]
    assert probabilities.shape == (len(labels), 3)


def test_gignn_classifier_allows_notebook_neighbor_sampling_shape(
    indicator_frame, labels, edge_index
):
    pytest.importorskip("torch")
    pytest.importorskip("torch_geometric")
    model = GIGNNClassifier(
        hidden_channels=(8,),
        num_neighbors_list=(2, 3),
        max_epochs=1,
        random_state=42,
    )

    model.fit(indicator_frame, labels, edge_index=edge_index)

    assert model.model_.num_neighbors_list == (2, 3)


def test_gignn_classifier_predict_before_fit_raises(indicator_frame, edge_index):
    model = GIGNNClassifier(hidden_channels=(8,), max_epochs=1, random_state=42)

    with pytest.raises(NotFittedError):
        model.predict(indicator_frame, edge_index=edge_index)


def test_gignn_classifier_rejects_predict_feature_mismatch(
    indicator_frame, labels, edge_index
):
    pytest.importorskip("torch")
    pytest.importorskip("torch_geometric")
    model = GIGNNClassifier(hidden_channels=(8,), max_epochs=1, random_state=42)
    model.fit(indicator_frame, labels, edge_index=edge_index)

    with pytest.raises(ValidationError, match="same number of feature columns"):
        model.predict(indicator_frame.drop(columns=["ndvi"]), edge_index=edge_index)


def test_gignn_classifier_rejects_reordered_predict_columns(
    indicator_frame, labels, edge_index
):
    pytest.importorskip("torch")
    pytest.importorskip("torch_geometric")
    model = GIGNNClassifier(hidden_channels=(8,), max_epochs=1, random_state=42)
    model.fit(indicator_frame, labels, edge_index=edge_index)

    reordered = indicator_frame[["dem", "ndvi", "road_density"]]

    with pytest.raises(ValidationError, match="same feature columns in the same order"):
        model.predict(reordered, edge_index=edge_index)


def test_gignn_classifier_save_load_round_trip(
    tmp_path, indicator_frame, labels, edge_index
):
    pytest.importorskip("torch_geometric")
    path = tmp_path / "model.pt"
    model = GIGNNClassifier(hidden_channels=(8,), max_epochs=1, random_state=42)
    model.fit(indicator_frame, labels, edge_index=edge_index)
    before = model.predict(indicator_frame, edge_index=edge_index)

    model.save(path)
    loaded = GIGNNClassifier.load(path)
    after = loaded.predict(indicator_frame, edge_index=edge_index)

    np.testing.assert_array_equal(before, after)


def test_gignn_classifier_round_trips_numpy_labels_and_class_order(
    tmp_path, indicator_frame, edge_index
):
    labels = np.asarray([0, 1, 1, 2, 0, 2], dtype=np.int64)
    class_order = tuple(np.asarray([0, 1, 2], dtype=np.int64))
    model = GIGNNClassifier(
        hidden_channels=(8,),
        class_order=class_order,
        max_epochs=1,
        random_state=42,
    )
    model.fit(indicator_frame, labels, edge_index=edge_index)
    path = tmp_path / "numpy-labels.pt"

    model.save(path)
    loaded = GIGNNClassifier.load(path)

    assert loaded.classes_.tolist() == [0, 1, 2]
    assert loaded.class_order == (0, 1, 2)
    np.testing.assert_array_equal(
        loaded.predict(indicator_frame, edge_index=edge_index),
        model.predict(indicator_frame, edge_index=edge_index),
    )


def test_gignn_classifier_load_rejects_missing_checkpoint_keys(tmp_path):
    path = tmp_path / "bad.pt"
    torch.save({"params": {}}, path)

    with pytest.raises(ValidationError, match="invalid schema"):
        GIGNNClassifier.load(path)


def test_gignn_classifier_load_rejects_invalid_checkpoint_params(tmp_path):
    path = tmp_path / "bad-params.pt"
    torch.save(
        {
            "params": {
                "hidden_channels": (),
                "graph_builder": "knn",
                "k_neighbors": 8,
                "distance_threshold": None,
                "max_epochs": 1,
                "learning_rate": 0.01,
                "weight_decay": 0.0,
                "dropout": 0.0,
                "random_state": 42,
                "device": "cpu",
            },
            "state_dict": {"weight": torch.ones(1)},
            "classes": ["a", "b"],
            "n_features_in": 3,
            "feature_names_in": ["ndvi", "dem", "road_density"],
        },
        path,
    )

    with pytest.raises(ValidationError, match="hidden_channels"):
        GIGNNClassifier.load(path)


def test_gignn_classifier_load_rejects_invalid_state_dict(tmp_path):
    path = tmp_path / "bad-state.pt"
    torch.save(
        {
            "params": {
                "hidden_channels": [8],
                "graph_builder": "knn",
                "k_neighbors": 8,
                "distance_threshold": None,
                "max_epochs": 1,
                "learning_rate": 0.01,
                "weight_decay": 0.0,
                "dropout": 0.0,
                "random_state": 42,
                "device": "cpu",
            },
            "state_dict": {"weight": "not-a-tensor"},
            "classes": ["a", "b"],
            "n_features_in": 3,
            "feature_names_in": ["ndvi", "dem", "road_density"],
        },
        path,
    )

    with pytest.raises(ValidationError, match="state_dict"):
        GIGNNClassifier.load(path)


def test_gignn_classifier_load_rejects_zero_learning_rate(tmp_path):
    path = tmp_path / "bad-learning-rate.pt"
    torch.save(
        {
            "params": {
                "hidden_channels": [8],
                "graph_builder": "knn",
                "k_neighbors": 8,
                "distance_threshold": None,
                "max_epochs": 1,
                "learning_rate": 0.0,
                "weight_decay": 0.0,
                "dropout": 0.0,
                "random_state": 42,
                "device": "cpu",
            },
            "state_dict": {"weight": torch.ones(1)},
            "classes": ["a", "b"],
            "n_features_in": 3,
            "feature_names_in": ["ndvi", "dem", "road_density"],
        },
        path,
    )

    with pytest.raises(ValidationError, match="learning_rate"):
        GIGNNClassifier.load(path)


def test_gignn_classifier_rejects_excessive_training_params(
    indicator_frame, labels, edge_index
):
    model = GIGNNClassifier(hidden_channels=(100_000,), max_epochs=1)

    with pytest.raises(ValidationError, match="hidden_channels"):
        model.fit(indicator_frame, labels, edge_index=edge_index)


def test_gignn_classifier_load_rejects_excessive_class_count(tmp_path):
    path = tmp_path / "too-many-classes.pt"
    torch.save(
        {
            "params": {
                "hidden_channels": [8],
                "graph_builder": "knn",
                "k_neighbors": 8,
                "distance_threshold": None,
                "max_epochs": 1,
                "learning_rate": 0.01,
                "weight_decay": 0.0,
                "dropout": 0.0,
                "random_state": 42,
                "device": "cpu",
            },
            "state_dict": {"weight": torch.ones(1)},
            "classes": [str(index) for index in range(10_001)],
            "n_features_in": 3,
            "feature_names_in": ["ndvi", "dem", "road_density"],
        },
        path,
    )

    with pytest.raises(ValidationError, match="classes"):
        GIGNNClassifier.load(path)


def test_gignn_classifier_load_rejects_excessive_declared_model_size(tmp_path):
    path = tmp_path / "oversized-model.pt"
    torch.save(
        {
            "params": {
                "hidden_channels": [4096] * 8,
                "num_neighbors_list": [1] * 9,
                "graph_builder": "knn",
                "k_neighbors": 1,
                "distance_threshold": None,
                "edge_weight_sigma": 2.0,
                "edge_weight_mode": "mean",
                "class_order": None,
                "max_epochs": 1,
                "learning_rate": 0.01,
                "weight_decay": 0.0,
                "dropout": 0.0,
                "random_state": 42,
                "device": "cpu",
            },
            "state_dict": {"weight": torch.ones(1)},
            "classes": ["a", "b"],
            "n_features_in": 10_000,
            "feature_names_in": [f"feature-{index}" for index in range(10_000)],
        },
        path,
    )

    with pytest.raises(ValidationError, match="model architecture exceeds"):
        GIGNNClassifier.load(path)


def test_gignn_classifier_load_rejects_excessive_tensor_count(tmp_path):
    path = tmp_path / "too-many-tensors.pt"
    torch.save(
        {
            "params": {
                "hidden_channels": [8],
                "graph_builder": "knn",
                "k_neighbors": 8,
                "distance_threshold": None,
                "max_epochs": 1,
                "learning_rate": 0.01,
                "weight_decay": 0.0,
                "dropout": 0.0,
                "random_state": 42,
                "device": "cpu",
            },
            "state_dict": {str(index): torch.ones(1) for index in range(129)},
            "classes": ["a", "b"],
            "n_features_in": 3,
            "feature_names_in": ["ndvi", "dem", "road_density"],
        },
        path,
    )

    with pytest.raises(ValidationError, match="tensor count"):
        GIGNNClassifier.load(path)
