from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import warnings

import numpy as np
import pandas as pd
import torch

from gignn.exceptions import EdgeWeightWarning, NotFittedError, ValidationError
from gignn.graph import (
    build_pyg_data,
    build_spatial_edges,
    calculate_edge_weights,
    calculate_gaussian_edge_attributes,
)
from gignn.io import load_checkpoint, save_checkpoint
from gignn.limits import (
    MAX_CHECKPOINT_TENSOR_ELEMENTS,
    MAX_CHECKPOINT_TENSORS,
    MAX_CLASSES,
    MAX_FEATURES,
    MAX_GRAPH_NEIGHBORS,
    MAX_HIDDEN_CHANNEL,
    MAX_HIDDEN_LAYERS,
    MAX_TRAINING_EPOCHS,
)
from gignn.models import GIGNN
from gignn.models.gignn import INFERENCE_MODES
from gignn.training import TrainingConfig, train_node_classifier
from gignn.training.split import set_random_seed
from gignn.validation import validate_class_order, validate_feature_table

EDGE_WEIGHT_MODES = ("product", "mean", "min", "max")
EDGE_WEIGHT_COLLAPSE_THRESHOLD = 1e-3


@dataclass
class GIGNNClassifier:
    """GIGNN node classifier (Ma, Zhang & Song, 2025, Section 2.2).

    Defaults follow the paper: Eq. (2) Gaussian edge weights with ``sigma=2``
    on GeoDetector stratum codes (``edge_weight_mode="product"``), neighbor
    messages scaled by those weights and averaged over neighbors
    (``normalize_edge_weights=False``), two GraphSAGE-style layers with
    ``num_neighbors_list=(6, 12)`` and 128 hidden units, and deterministic
    full-neighborhood inference.
    """

    hidden_channels: tuple[int, ...] = (128,)
    num_neighbors_list: tuple[int, ...] = (6, 12)
    graph_builder: str = "knn"
    k_neighbors: int = 8
    distance_threshold: float | None = None
    edge_weight_sigma: float = 2.0
    edge_weight_mode: str = "product"
    normalize_edge_weights: bool = False
    inference: str = "full"
    max_epochs: int = 100
    learning_rate: float = 0.01
    weight_decay: float = 5e-4
    dropout: float = 0.0
    random_state: int | None = None
    device: str = "cpu"
    class_order: tuple[object, ...] | None = None

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series | Iterable[object],
        *,
        geometry: Sequence[object] | None = None,
        edge_index: object | None = None,
        train_mask: object | None = None,
    ) -> "GIGNNClassifier":
        """Fit on all rows, or transductively on ``train_mask`` rows.

        With ``train_mask`` every row stays in the graph and contributes
        features to message passing, but only masked rows contribute labels to
        the loss; labels outside the mask may be missing.
        """
        self._validate_training_params()
        # Seed before model construction: GIGNN weight initialization consumes
        # the global torch RNG, so seeding only inside the trainer would leave
        # init non-reproducible across repeated fits in one process.
        if self.random_state is not None:
            set_random_seed(self.random_state)
        labels, mask = _resolve_training_labels(y, train_mask, row_count=len(X))
        graph_edges = self._resolve_edge_index(
            X, geometry=geometry, edge_index=edge_index
        )
        data = self._build_data(X, labels, graph_edges, class_order=self.class_order)
        _warn_if_edge_weights_collapsed(
            data.edge_weight.numpy(), sigma=self.edge_weight_sigma
        )
        model = self._build_model(data.num_features, len(data.label_encoder.classes_))
        config = TrainingConfig(
            max_epochs=self.max_epochs,
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
            random_state=self.random_state,
            device=self.device,
        )
        self.training_result_ = train_node_classifier(
            model, data, config, train_mask=mask
        )
        self.model_ = model.to("cpu").eval()
        self.label_encoder_ = data.label_encoder
        self.n_features_in_ = data.num_features
        self.feature_names_in_ = tuple(X.columns)
        self.classes_ = data.label_encoder.classes_
        return self

    def predict(
        self,
        X: pd.DataFrame,
        *,
        geometry: Sequence[object] | None = None,
        edge_index: object | None = None,
    ) -> np.ndarray:
        probabilities = self.predict_proba(X, geometry=geometry, edge_index=edge_index)
        encoded = probabilities.argmax(axis=1)
        return self.label_encoder_.inverse_transform(encoded)

    def predict_proba(
        self,
        X: pd.DataFrame,
        *,
        geometry: Sequence[object] | None = None,
        edge_index: object | None = None,
    ) -> np.ndarray:
        self._require_fitted()
        X_validated = validate_feature_table(X)
        if X_validated.shape[1] != self.n_features_in_:
            raise ValidationError(
                "X must have the same number of feature columns used during fit."
            )
        if tuple(X_validated.columns) != self.feature_names_in_:
            raise ValidationError(
                "X must have the same feature columns in the same order used during fit."
            )
        graph_edges = self._resolve_edge_index(
            X_validated, geometry=geometry, edge_index=edge_index
        )
        data = self._build_data(
            X_validated,
            self.classes_[np.zeros(len(X_validated), dtype=int)],
            graph_edges,
            class_order=tuple(self.classes_.tolist()),
        )
        self.model_.inference = _validate_inference(self.inference)
        with torch.no_grad():
            logits = self.model_(
                data.x,
                data.edge_index,
                edge_weight=getattr(data, "edge_weight", None),
            )
            return torch.softmax(logits, dim=1).cpu().numpy()

    def score(
        self,
        X: pd.DataFrame,
        y: pd.Series | Iterable[object],
        *,
        geometry: Sequence[object] | None = None,
        edge_index: object | None = None,
    ) -> float:
        predictions = self.predict(X, geometry=geometry, edge_index=edge_index)
        expected = np.asarray(pd.Series(y).to_numpy(), dtype=object)
        predicted = np.asarray(predictions, dtype=object)
        if expected.shape[0] != predicted.shape[0]:
            raise ValidationError("y must have one label per prediction row.")
        # predict() returns an object-dtype array of original labels; sklearn's
        # accuracy_score rejects object dtype, so compare element-wise instead.
        return float((predicted == expected).mean())

    def save(self, path: str | Path) -> None:
        self._require_fitted()
        save_checkpoint(
            path,
            {
                "params": self._params(),
                "state_dict": self.model_.state_dict(),
                "classes": [_to_python_scalar(v) for v in self.classes_.tolist()],
                "n_features_in": self.n_features_in_,
                "feature_names_in": list(self.feature_names_in_),
            },
        )

    @classmethod
    def load(cls, path: str | Path) -> "GIGNNClassifier":
        checkpoint = _validate_checkpoint(load_checkpoint(path))
        model = cls(**checkpoint["params"])
        model.classes_ = np.array(checkpoint["classes"], dtype=object)
        model.n_features_in_ = checkpoint["n_features_in"]
        model.feature_names_in_ = tuple(checkpoint["feature_names_in"])
        from gignn.validation import LabelEncoder

        model.label_encoder_ = LabelEncoder(classes_=model.classes_)
        model.model_ = model._build_model(model.n_features_in_, len(model.classes_))
        model.model_.load_state_dict(checkpoint["state_dict"])
        model.model_.eval()
        return model

    def _build_model(self, n_features: int, n_classes: int) -> GIGNN:
        return GIGNN(
            in_channels=n_features,
            hidden_channels=self.hidden_channels,
            out_channels=n_classes,
            num_neighbors_list=self.num_neighbors_list,
            dropout=self.dropout,
            random_state=self.random_state,
            normalize_edge_weights=self.normalize_edge_weights,
            inference=self.inference,
        )

    def _resolve_edge_index(
        self,
        X: pd.DataFrame,
        *,
        geometry: Sequence[object] | None = None,
        edge_index: object | None = None,
    ) -> np.ndarray:
        num_nodes = len(X)
        if edge_index is not None:
            return build_spatial_edges(edge_index=edge_index, num_nodes=num_nodes)
        return build_spatial_edges(
            geometry=geometry,
            num_nodes=num_nodes,
            strategy=self.graph_builder,
            k=self.k_neighbors,
            threshold=self.distance_threshold,
        )

    def _build_data(
        self,
        X: pd.DataFrame,
        y,
        edge_index: np.ndarray,
        *,
        class_order: Sequence[object] | None = None,
    ):
        edge_attr = calculate_gaussian_edge_attributes(
            X,
            edge_index,
            sigma=self.edge_weight_sigma,
        )
        edge_weight = calculate_edge_weights(edge_attr, mode=self.edge_weight_mode)
        return build_pyg_data(
            X,
            y,
            edge_index,
            edge_attr=edge_attr,
            edge_weight=edge_weight,
            class_order=class_order,
        )

    def _require_fitted(self) -> None:
        if not hasattr(self, "model_"):
            raise NotFittedError("GIGNNClassifier must be fitted before prediction.")

    def _validate_training_params(self) -> None:
        _validate_hidden_channels(self.hidden_channels)
        _validate_num_neighbors_list(self.num_neighbors_list, self.hidden_channels)
        if (
            not isinstance(self.k_neighbors, int)
            or isinstance(self.k_neighbors, bool)
            or not 0 < self.k_neighbors <= MAX_GRAPH_NEIGHBORS
        ):
            raise ValidationError("k_neighbors must be within resource limits.")
        if self.distance_threshold is not None and self.distance_threshold <= 0:
            raise ValidationError("distance_threshold must be positive when provided.")
        if self.edge_weight_sigma <= 0:
            raise ValidationError("edge_weight_sigma must be positive.")
        if self.edge_weight_mode not in EDGE_WEIGHT_MODES:
            raise ValidationError("edge_weight_mode is invalid.")
        if not isinstance(self.normalize_edge_weights, bool):
            raise ValidationError("normalize_edge_weights must be a boolean.")
        _validate_inference(self.inference)
        if (
            not isinstance(self.max_epochs, int)
            or isinstance(self.max_epochs, bool)
            or not 0 < self.max_epochs <= MAX_TRAINING_EPOCHS
        ):
            raise ValidationError("max_epochs must be within resource limits.")
        if self.learning_rate <= 0:
            raise ValidationError("learning_rate must be positive.")
        if self.weight_decay < 0:
            raise ValidationError("weight_decay must be non-negative.")
        if not 0 <= self.dropout < 1:
            raise ValidationError("dropout must be in [0, 1).")
        _validate_device_string(self.device)

    def _params(self) -> dict[str, object]:
        return {
            "hidden_channels": self.hidden_channels,
            "num_neighbors_list": self.num_neighbors_list,
            "graph_builder": self.graph_builder,
            "k_neighbors": self.k_neighbors,
            "distance_threshold": self.distance_threshold,
            "edge_weight_sigma": self.edge_weight_sigma,
            "edge_weight_mode": self.edge_weight_mode,
            "normalize_edge_weights": self.normalize_edge_weights,
            "inference": self.inference,
            "max_epochs": self.max_epochs,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "dropout": self.dropout,
            "random_state": self.random_state,
            "device": self.device,
            "class_order": (
                tuple(_to_python_scalar(v) for v in self.class_order)
                if self.class_order is not None
                else None
            ),
        }


def _to_python_scalar(value: object) -> object:
    """Convert numpy scalar to native Python scalar for weights_only round-trip."""
    if isinstance(value, np.generic):
        return value.item()
    return value


def _validate_inference(inference: object) -> str:
    if inference not in INFERENCE_MODES:
        raise ValidationError("inference must be 'full' or 'sampled'.")
    return inference


def _resolve_training_labels(
    y: pd.Series | Iterable[object], train_mask: object | None, *, row_count: int
) -> tuple[pd.Series, np.ndarray | None]:
    labels = pd.Series(y).reset_index(drop=True)
    if train_mask is None:
        return labels, None
    mask = np.asarray(train_mask)
    if mask.dtype != bool or mask.shape != (row_count,):
        raise ValidationError(
            "train_mask must be a boolean array with one entry per row."
        )
    if len(labels) != row_count:
        raise ValidationError("X and y must have the same number of rows.")
    if not mask.any():
        raise ValidationError("train_mask must select at least one row.")
    labeled = labels[mask]
    if labeled.isna().any():
        raise ValidationError("y must not contain missing labels inside train_mask.")
    return labels.where(pd.Series(mask), labeled.iloc[0]), mask


def _warn_if_edge_weights_collapsed(edge_weight: np.ndarray, *, sigma: float) -> None:
    if edge_weight.size == 0:
        return
    median = float(np.median(edge_weight))
    if median < EDGE_WEIGHT_COLLAPSE_THRESHOLD:
        warnings.warn(
            f"Median Gaussian edge weight is {median:.3g}: with sigma={sigma:g} "
            "neighbor messages vanish and GIGNN degenerates to a per-node model. "
            "Eq. (2) is defined on GeoDetector stratum codes; discretize features "
            "first (for example with GeoDetectorSelector) or increase "
            "edge_weight_sigma.",
            EdgeWeightWarning,
            stacklevel=3,
        )


def _validate_checkpoint(checkpoint: object) -> dict[str, Any]:
    if not isinstance(checkpoint, dict):
        raise ValidationError("Checkpoint must be a dictionary.")
    required_keys = {
        "params",
        "state_dict",
        "classes",
        "n_features_in",
        "feature_names_in",
    }
    if set(checkpoint) != required_keys:
        raise ValidationError("Checkpoint has an invalid schema.")

    params = checkpoint["params"]
    if not isinstance(params, dict):
        raise ValidationError("Checkpoint params must be a dictionary.")
    validated_params = _validate_checkpoint_params(params)

    classes = checkpoint["classes"]
    if not isinstance(classes, list) or not classes or len(classes) > MAX_CLASSES:
        raise ValidationError(
            "Checkpoint classes must be a non-empty list within resource limits."
        )
    validate_class_order(classes)

    n_features_in = checkpoint["n_features_in"]
    if (
        not isinstance(n_features_in, int)
        or n_features_in <= 0
        or n_features_in > MAX_FEATURES
    ):
        raise ValidationError("Checkpoint n_features_in must be a positive integer.")

    feature_names_in = checkpoint["feature_names_in"]
    if (
        not isinstance(feature_names_in, list)
        or len(feature_names_in) != n_features_in
        or not all(isinstance(value, str) for value in feature_names_in)
    ):
        raise ValidationError("Checkpoint feature_names_in must match n_features_in.")

    state_dict = checkpoint["state_dict"]
    if not isinstance(state_dict, dict) or not state_dict:
        raise ValidationError("Checkpoint state_dict must be a non-empty dictionary.")
    if len(state_dict) > MAX_CHECKPOINT_TENSORS:
        raise ValidationError("Checkpoint state_dict exceeds tensor count limit.")
    if not all(isinstance(value, torch.Tensor) for value in state_dict.values()):
        raise ValidationError("Checkpoint state_dict values must be tensors.")
    total_elements = sum(int(value.numel()) for value in state_dict.values())
    if total_elements > MAX_CHECKPOINT_TENSOR_ELEMENTS:
        raise ValidationError("Checkpoint state_dict exceeds tensor size limit.")
    _validate_model_state_spec(
        state_dict,
        n_features_in=n_features_in,
        hidden_channels=validated_params["hidden_channels"],
        class_count=len(classes),
    )

    return {
        "params": validated_params,
        "state_dict": state_dict,
        "classes": classes,
        "n_features_in": n_features_in,
        "feature_names_in": feature_names_in,
    }


def _validate_model_state_spec(
    state_dict: dict[str, torch.Tensor],
    *,
    n_features_in: int,
    hidden_channels: object,
    class_count: int,
) -> None:
    layer_dims = (*tuple(hidden_channels), class_count)
    expected: dict[str, tuple[int, int]] = {}
    input_dim = n_features_in
    total_elements = 0
    for index, output_dim in enumerate(layer_dims):
        shape = (input_dim, output_dim)
        expected[f"gcn.{index}.weight"] = shape
        expected[f"gcn.{index}.aggregator.weight"] = shape
        total_elements += 2 * input_dim * output_dim
        input_dim = output_dim
    if total_elements > MAX_CHECKPOINT_TENSOR_ELEMENTS:
        raise ValidationError("Checkpoint model architecture exceeds resource limits.")
    if set(state_dict) != set(expected):
        raise ValidationError(
            "Checkpoint state_dict does not match model architecture."
        )
    for name, shape in expected.items():
        value = state_dict[name]
        if (
            tuple(value.shape) != shape
            or value.dtype != torch.float32
            or value.device.type != "cpu"
        ):
            raise ValidationError(
                "Checkpoint state_dict does not match model architecture."
            )


def _validate_checkpoint_params(params: dict[object, object]) -> dict[str, object]:
    required_keys = {
        "hidden_channels",
        "graph_builder",
        "k_neighbors",
        "distance_threshold",
        "max_epochs",
        "learning_rate",
        "weight_decay",
        "dropout",
        "random_state",
        "device",
    }
    optional_keys = {
        "num_neighbors_list",
        "edge_weight_sigma",
        "edge_weight_mode",
        "normalize_edge_weights",
        "inference",
        "class_order",
    }
    if not required_keys.issubset(params) or not set(params).issubset(
        required_keys | optional_keys
    ):
        raise ValidationError("Checkpoint params have an invalid schema.")
    # Absent keys come from checkpoints written before they existed; fill them
    # with the semantics those checkpoints were trained and evaluated under.
    params = {
        "num_neighbors_list": (6, 12),
        "edge_weight_sigma": 2.0,
        "edge_weight_mode": "mean",
        "normalize_edge_weights": True,
        "inference": "sampled",
        "class_order": None,
        **params,
    }

    hidden_channels = params["hidden_channels"]
    hidden_tuple = _validate_hidden_channels(hidden_channels)

    num_neighbors_tuple = _validate_num_neighbors_list(
        params["num_neighbors_list"], hidden_tuple
    )

    graph_builder = params["graph_builder"]
    if graph_builder not in {"knn", "distance", "contiguity"}:
        raise ValidationError("graph_builder is invalid.")

    k_neighbors = params["k_neighbors"]
    if (
        not isinstance(k_neighbors, int)
        or isinstance(k_neighbors, bool)
        or k_neighbors <= 0
        or k_neighbors > MAX_GRAPH_NEIGHBORS
    ):
        raise ValidationError("k_neighbors must be a positive integer.")

    distance_threshold = params["distance_threshold"]
    if distance_threshold is not None and (
        not isinstance(distance_threshold, (int, float)) or distance_threshold <= 0
    ):
        raise ValidationError("distance_threshold must be positive when provided.")

    edge_weight_sigma = params["edge_weight_sigma"]
    if not isinstance(edge_weight_sigma, (int, float)) or edge_weight_sigma <= 0:
        raise ValidationError("edge_weight_sigma must be positive.")

    edge_weight_mode = params["edge_weight_mode"]
    if edge_weight_mode not in EDGE_WEIGHT_MODES:
        raise ValidationError("edge_weight_mode is invalid.")

    normalize_edge_weights = params["normalize_edge_weights"]
    if not isinstance(normalize_edge_weights, bool):
        raise ValidationError("normalize_edge_weights must be a boolean.")
    inference = _validate_inference(params["inference"])

    class_order = params["class_order"]
    if class_order is not None:
        class_order = tuple(validate_class_order(class_order).tolist())

    max_epochs = params["max_epochs"]
    if (
        not isinstance(max_epochs, int)
        or isinstance(max_epochs, bool)
        or max_epochs <= 0
        or max_epochs > MAX_TRAINING_EPOCHS
    ):
        raise ValidationError("max_epochs must be a positive integer.")

    learning_rate = params["learning_rate"]
    if not isinstance(learning_rate, (int, float)) or learning_rate <= 0:
        raise ValidationError("learning_rate must be positive.")

    weight_decay = params["weight_decay"]
    if not isinstance(weight_decay, (int, float)) or weight_decay < 0:
        raise ValidationError("weight_decay must be non-negative.")

    dropout = params["dropout"]
    if not isinstance(dropout, (int, float)) or not 0 <= dropout < 1:
        raise ValidationError("dropout must be in [0, 1).")

    random_state = params["random_state"]
    if random_state is not None and not isinstance(random_state, int):
        raise ValidationError("random_state must be an integer or None.")

    device = params["device"]
    _validate_device_string(device)

    return {
        "hidden_channels": hidden_tuple,
        "num_neighbors_list": num_neighbors_tuple,
        "graph_builder": graph_builder,
        "k_neighbors": k_neighbors,
        "distance_threshold": (
            float(distance_threshold) if distance_threshold is not None else None
        ),
        "edge_weight_sigma": float(edge_weight_sigma),
        "edge_weight_mode": edge_weight_mode,
        "normalize_edge_weights": normalize_edge_weights,
        "inference": inference,
        "class_order": class_order,
        "max_epochs": max_epochs,
        "learning_rate": float(learning_rate),
        "weight_decay": float(weight_decay),
        "dropout": float(dropout),
        "random_state": random_state,
        "device": device,
    }


def _validate_device_string(device: object) -> str:
    if (
        not isinstance(device, str)
        or device not in {"cpu", "cuda", "mps"}
        and not _is_cuda_index(device)
    ):
        raise ValidationError("device is invalid.")
    return device


def _is_cuda_index(device: object) -> bool:
    if not isinstance(device, str) or not device.startswith("cuda:"):
        return False
    suffix = device.split(":", 1)[1]
    return suffix.isdigit()


def _validate_hidden_channels(hidden_channels: object) -> tuple[int, ...]:
    if not isinstance(hidden_channels, (tuple, list)) or not hidden_channels:
        raise ValidationError("hidden_channels must be a non-empty sequence.")
    hidden_tuple = tuple(hidden_channels)
    if len(hidden_tuple) > MAX_HIDDEN_LAYERS:
        raise ValidationError("hidden_channels exceeds layer resource limits.")
    if not all(
        isinstance(value, int)
        and not isinstance(value, bool)
        and 0 < value <= MAX_HIDDEN_CHANNEL
        for value in hidden_tuple
    ):
        raise ValidationError(
            "hidden_channels must contain positive integers within resource limits."
        )
    return hidden_tuple


def _validate_num_neighbors_list(
    num_neighbors_list: object, hidden_channels: tuple[int, ...]
) -> tuple[int, ...]:
    if not isinstance(num_neighbors_list, (tuple, list)) or not num_neighbors_list:
        raise ValidationError("num_neighbors_list must be a non-empty sequence.")
    neighbor_tuple = tuple(num_neighbors_list)
    if len(neighbor_tuple) != len(hidden_channels) + 1:
        raise ValidationError(
            "num_neighbors_list length must equal len(hidden_channels) + 1."
        )
    if not all(
        isinstance(value, int)
        and not isinstance(value, bool)
        and 0 < value <= MAX_GRAPH_NEIGHBORS
        for value in neighbor_tuple
    ):
        raise ValidationError(
            "num_neighbors_list must contain positive integers within resource limits."
        )
    return neighbor_tuple
