from __future__ import annotations

import math
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
import re
from typing import Any

import numpy as np
import pandas as pd
import torch

from gignn.auto_archive import load_payload, save_payload
from gignn.estimators import GIGNNClassifier, _validate_checkpoint
from gignn.exceptions import ValidationError
from gignn.geodetector import GeoDetectorSelector
from gignn.geodetector.discretization import DiscretizationMetadata
from gignn.geodetector.selector import SELECTION_STRATEGIES
from gignn.limits import (
    MAX_AUTO_GRAPH_NODES,
    MAX_CLASSES,
    MAX_EDGES,
    MAX_FEATURES,
    MAX_TRAINING_EPOCHS,
)
from gignn.ordinal import OrdinalTargetEncoder
from gignn.reports import (
    GeneralizationReport,
    GraphSelectionReport,
    ModelSelectionReport,
    SpatialRelationshipReport,
)
from gignn.training import TrainingResult
from gignn.validation import LabelEncoder, validate_class_order

AUTO_CHECKPOINT_SCHEMA_VERSION = "1"


def save_auto_classifier(model: Any, path: str | Path) -> None:
    model._require_fitted()
    save_payload(path, _auto_payload(model))


def load_auto_classifier(path: str | Path) -> Any:
    payload = _validate_auto_payload(load_payload(path))
    from gignn.auto import GIGNNAutoClassifier

    try:
        model = GIGNNAutoClassifier(**payload["params"])
        model.selector_ = _restore_selector(payload["selector"])
        model.estimator_ = _restore_estimator(
            payload["estimator"], payload["training_losses"]
        )
        model.discovery_report_ = _restore_discovery_report(payload["discovery_report"])
        model.graph_report_ = _restore_graph_report(payload["graph_report"])
        model.model_selection_report_ = _restore_model_report(payload["model_report"])
        model.validation_report_ = _restore_generalization_report(
            payload["validation_report"]
        )
        model.generalization_report_ = _restore_generalization_report(
            payload["generalization_report"]
        )
        model.ordinal_encoder_ = OrdinalTargetEncoder(
            classes_=tuple(payload["class_order"])
        )
        model.class_order_ = model.ordinal_encoder_.classes_
        model.target_ = pd.Series(payload["target"])
        model.train_indices_ = _tensor_to_indices(payload["train_indices"])
        model.validation_indices_ = _tensor_to_indices(payload["validation_indices"])
        model.test_indices_ = _tensor_to_indices(payload["test_indices"])
        model.node_ids_ = (
            None
            if payload["node_ids"] is None
            else np.asarray(payload["node_ids"], dtype=object)
        )
        model.selected_features_ = list(model.discovery_report_.selected_features)
        model.edge_index_ = model.graph_report_.selected_edge_index.copy()
    except ValidationError:
        raise
    except (KeyError, TypeError, ValueError, RuntimeError) as exc:
        raise ValidationError(
            "Automatic workflow checkpoint payload is invalid."
        ) from exc
    return model


def _auto_payload(model: Any) -> dict[str, Any]:
    return {
        "schema_version": AUTO_CHECKPOINT_SCHEMA_VERSION,
        "package_version": _package_version(),
        "params": _primitive(model.__dict__, keys=_constructor_keys()),
        "selector": _selector_payload(model.selector_),
        "estimator": {
            "params": model.estimator_._params(),
            "state_dict": model.estimator_.model_.state_dict(),
            "classes": model.estimator_.classes_.tolist(),
            "n_features_in": model.estimator_.n_features_in_,
            "feature_names_in": list(model.estimator_.feature_names_in_),
        },
        "training_losses": _primitive_sequence(
            model.estimator_.training_result_.losses, "training_losses"
        ),
        "discovery_report": _discovery_report_payload(model.discovery_report_),
        "graph_report": _graph_report_payload(model.graph_report_),
        "model_report": _model_report_payload(model.model_selection_report_),
        "validation_report": _generalization_report_payload(model.validation_report_),
        "generalization_report": _generalization_report_payload(
            model.generalization_report_
        ),
        "class_order": _primitive_sequence(model.class_order_, "class_order"),
        "target": _primitive_sequence(model.target_.tolist(), "target"),
        "train_indices": torch.as_tensor(
            np.asarray(model.train_indices_, dtype=np.int64).copy(), dtype=torch.int64
        ),
        "validation_indices": torch.as_tensor(
            np.asarray(model.validation_indices_, dtype=np.int64).copy(),
            dtype=torch.int64,
        ),
        "test_indices": torch.as_tensor(
            np.asarray(model.test_indices_, dtype=np.int64).copy(), dtype=torch.int64
        ),
        "node_ids": (
            None
            if model.node_ids_ is None
            else _primitive_sequence(model.node_ids_.tolist(), "node_ids")
        ),
    }


def _validate_auto_payload(payload: object) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValidationError("Automatic workflow checkpoint must be a dictionary.")
    required = {
        "schema_version",
        "package_version",
        "params",
        "selector",
        "estimator",
        "training_losses",
        "discovery_report",
        "graph_report",
        "model_report",
        "validation_report",
        "generalization_report",
        "class_order",
        "target",
        "train_indices",
        "validation_indices",
        "test_indices",
        "node_ids",
    }
    if set(payload) != required:
        raise ValidationError("Automatic workflow checkpoint has an invalid schema.")
    if payload["schema_version"] != AUTO_CHECKPOINT_SCHEMA_VERSION:
        raise ValidationError("Automatic workflow checkpoint schema is incompatible.")
    _validate_package_version(payload["package_version"])
    _validate_constructor_params(payload["params"])
    _validate_selector_payload(payload["selector"])
    _validate_checkpoint(payload["estimator"])
    _validate_training_losses(payload["training_losses"])
    _validate_discovery_report_payload(payload["discovery_report"])
    _validate_graph_report_payload(payload["graph_report"])
    _validate_model_report_payload(payload["model_report"])
    _validate_generalization_report_payload(payload["validation_report"])
    _validate_generalization_report_payload(payload["generalization_report"])
    _validate_target_payload(payload)
    _validate_index_partition(payload)
    _validate_payload_invariants(payload)
    return payload


def _validate_payload_invariants(payload: dict[str, Any]) -> None:
    selected = payload["selector"]["selected_features"]
    discovery_selected = payload["discovery_report"]["selected_features"]
    estimator = payload["estimator"]
    feature_names = estimator["feature_names_in"]
    if selected != discovery_selected or selected != feature_names:
        raise ValidationError(
            "Automatic workflow checkpoint feature metadata is inconsistent."
        )
    if estimator["n_features_in"] != len(selected):
        raise ValidationError(
            "Automatic workflow checkpoint feature count is inconsistent."
        )

    class_count = len(payload["class_order"])
    expected_classes = list(range(class_count))
    if estimator["classes"] != expected_classes:
        raise ValidationError(
            "Automatic workflow checkpoint class metadata is inconsistent."
        )

    validation_indices = _tensor_to_indices(payload["validation_indices"])
    test_indices = _tensor_to_indices(payload["test_indices"])
    validation_report = payload["validation_report"]
    test_report = payload["generalization_report"]
    if validation_report["partition"] != "validation" or not np.array_equal(
        _tensor_to_indices(validation_report["indices"]), validation_indices
    ):
        raise ValidationError(
            "Automatic workflow checkpoint validation report is inconsistent."
        )
    if test_report["partition"] != "test" or not np.array_equal(
        _tensor_to_indices(test_report["indices"]), test_indices
    ):
        raise ValidationError(
            "Automatic workflow checkpoint test report is inconsistent."
        )

    graph_report = payload["graph_report"]
    edges = graph_report["selected_edge_index"]
    target_size = len(payload["target"])
    if edges.numel() and (int(edges.min()) < 0 or int(edges.max()) >= target_size):
        raise ValidationError(
            "Automatic workflow checkpoint graph metadata is inconsistent."
        )
    for key, value in payload["model_report"]["best_params"].items():
        if estimator["params"].get(key) != value:
            raise ValidationError(
                "Automatic workflow checkpoint model metadata is inconsistent."
            )


def _validate_constructor_params(payload: Any) -> None:
    if not isinstance(payload, dict) or set(payload) != set(_constructor_keys()):
        raise ValidationError("Automatic workflow checkpoint params are invalid.")
    from gignn.auto import (
        _grid_size,
        _validate_candidate_budget,
        _validate_search_grids,
    )

    try:
        _validate_search_grids(
            detector_grid=payload["detector_grid"],
            graph_grid=payload["graph_grid"],
            model_grid=payload["model_grid"],
        )
    except ValidationError as exc:
        raise ValidationError(
            f"Automatic workflow checkpoint search grids are invalid: {exc}"
        ) from exc
    top_k = payload["top_k_features"]
    if top_k is not None and (not isinstance(top_k, int) or top_k <= 0):
        raise ValidationError(
            "Automatic workflow checkpoint top_k_features is invalid."
        )
    random_state = payload["random_state"]
    if random_state is not None and (
        not isinstance(random_state, int) or isinstance(random_state, bool)
    ):
        raise ValidationError("Automatic workflow checkpoint random_state is invalid.")
    for name in ("validation_fraction", "test_fraction"):
        value = payload[name]
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(float(value))
            or not 0 < value < 1
        ):
            raise ValidationError(f"Automatic workflow checkpoint {name} is invalid.")
    if payload["validation_fraction"] + payload["test_fraction"] >= 1:
        raise ValidationError(
            "Automatic workflow checkpoint split fractions are invalid."
        )
    budget = _validate_candidate_budget(payload["max_grid_candidates"])
    workload = _grid_size(payload["detector_grid"]) + _grid_size(
        payload["graph_grid"]
    ) * _grid_size(payload["model_grid"])
    if workload > budget:
        raise ValidationError(
            "Automatic workflow checkpoint search grid exceeds candidate budget."
        )


def _validate_search_grid(payload: Any, *, name: str) -> None:
    if not isinstance(payload, dict) or not payload or len(payload) > MAX_FEATURES:
        raise ValidationError(f"Automatic workflow checkpoint {name} is invalid.")
    for key, values in payload.items():
        if (
            not isinstance(key, str)
            or not isinstance(values, list)
            or not values
            or len(values) > MAX_FEATURES
        ):
            raise ValidationError(f"Automatic workflow checkpoint {name} is invalid.")
        for value in values:
            if isinstance(value, (list, tuple)):
                for element in value:
                    if (
                        not isinstance(element, (str, bool, int, float))
                        or element is None
                    ):
                        raise ValidationError(
                            f"Automatic workflow checkpoint {name} is invalid."
                        )
            elif not isinstance(value, (str, bool, int, float)) or value is None:
                raise ValidationError(
                    f"Automatic workflow checkpoint {name} is invalid."
                )


def _validate_target_payload(payload: dict[str, Any]) -> None:
    class_order = payload["class_order"]
    target = payload["target"]
    if (
        not isinstance(class_order, list)
        or not class_order
        or len(class_order) > MAX_CLASSES
        or not all(_is_label_scalar(value) for value in class_order)
    ):
        raise ValidationError("Automatic workflow checkpoint class order is invalid.")
    validate_class_order(class_order)
    if (
        not isinstance(target, list)
        or not target
        or len(target) > MAX_AUTO_GRAPH_NODES
        or not all(_is_label_scalar(value) for value in target)
    ):
        raise ValidationError("Automatic workflow checkpoint target is invalid.")
    if {_label_key(value) for value in target} != {
        _label_key(value) for value in class_order
    }:
        raise ValidationError(
            "Automatic workflow checkpoint target does not match class order."
        )
    node_ids = payload["node_ids"]
    if node_ids is not None and (
        not isinstance(node_ids, list)
        or len(node_ids) != len(target)
        or not all(_is_label_scalar(value) for value in node_ids)
        or len({_label_key(value) for value in node_ids}) != len(node_ids)
    ):
        raise ValidationError("Automatic workflow checkpoint node IDs are invalid.")


def _label_key(value: Any) -> tuple[type[Any], Any]:
    return type(value), value


def _is_label_scalar(value: Any) -> bool:
    return (
        value is not None
        and isinstance(value, (str, bool, int, float))
        and (not isinstance(value, float) or math.isfinite(value))
    )


def _validate_index_partition(payload: dict[str, Any]) -> None:
    target_size = len(payload["target"])
    tensors = [
        payload["train_indices"],
        payload["validation_indices"],
        payload["test_indices"],
    ]
    if any(
        not isinstance(value, torch.Tensor) or value.numel() > target_size
        for value in tensors
    ):
        raise ValidationError(
            "Automatic workflow checkpoint split indices exceed target size."
        )
    partitions = [_tensor_to_indices(value) for value in tensors]
    if any(values.size == 0 for values in partitions):
        raise ValidationError("Automatic workflow checkpoint split indices are empty.")

    seen = np.zeros(target_size, dtype=np.uint8)
    for values in partitions:
        if values.min() < 0 or values.max() >= target_size:
            raise ValidationError(
                "Automatic workflow checkpoint split indices are invalid."
            )
        if len(np.unique(values)) != values.size:
            raise ValidationError(
                "Automatic workflow checkpoint split indices contain duplicates."
            )
        if np.any(seen[values]):
            raise ValidationError(
                "Automatic workflow checkpoint split indices are invalid."
            )
        seen[values] = 1
    if not np.all(seen):
        raise ValidationError(
            "Automatic workflow checkpoint split indices are incomplete."
        )


def _validate_training_losses(payload: Any) -> None:
    if (
        not isinstance(payload, list)
        or not payload
        or len(payload) > MAX_TRAINING_EPOCHS
        or not all(
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(float(value))
            for value in payload
        )
    ):
        raise ValidationError(
            "Automatic workflow checkpoint training losses are invalid."
        )


def _selector_payload(selector: GeoDetectorSelector) -> dict[str, Any]:
    return {
        "params": {
            "discretizer": selector.discretizer,
            "max_bins": selector.max_bins,
            "top_k": selector.top_k,
            "alpha": selector.alpha,
            "strategy": selector.strategy,
            "top_interactions": selector.top_interactions,
        },
        "selected_features": list(selector.selected_features_),
        "scores": _frame_payload(selector.scores_),
        "discretizers": {
            feature: {
                "method": selector.discretizers_[feature].method,
                "bins": torch.as_tensor(
                    selector.discretizers_[feature].bins, dtype=torch.float64
                ),
            }
            for feature in selector.selected_features_
        },
    }


def _validate_selector_payload(payload: Any) -> None:
    if not isinstance(payload, dict) or set(payload) != {
        "params",
        "selected_features",
        "scores",
        "discretizers",
    }:
        raise ValidationError("Automatic workflow checkpoint selector is invalid.")
    params = payload["params"]
    legacy_keys = {"discretizer", "max_bins", "top_k", "alpha"}
    if not isinstance(params, dict) or set(params) not in (
        legacy_keys,
        legacy_keys | {"strategy", "top_interactions"},
    ):
        raise ValidationError(
            "Automatic workflow checkpoint selector params are invalid."
        )
    if "strategy" in params and params["strategy"] not in SELECTION_STRATEGIES:
        raise ValidationError(
            "Automatic workflow checkpoint selector strategy is invalid."
        )
    if "top_interactions" in params and (
        not isinstance(params["top_interactions"], int)
        or isinstance(params["top_interactions"], bool)
        or params["top_interactions"] < 1
    ):
        raise ValidationError(
            "Automatic workflow checkpoint selector top_interactions is invalid."
        )
    selected = payload["selected_features"]
    if (
        not isinstance(selected, list)
        or not selected
        or len(selected) > MAX_FEATURES
        or not all(isinstance(value, str) for value in selected)
        or len(set(selected)) != len(selected)
    ):
        raise ValidationError(
            "Automatic workflow checkpoint selected features are invalid."
        )
    _validate_frame_payload(payload["scores"])
    discretizers = payload["discretizers"]
    if not isinstance(discretizers, dict) or set(discretizers) != set(selected):
        raise ValidationError("Automatic workflow checkpoint discretizers are invalid.")
    for metadata in discretizers.values():
        if not isinstance(metadata, dict) or set(metadata) != {"method", "bins"}:
            raise ValidationError(
                "Automatic workflow checkpoint discretizer metadata is invalid."
            )
        bins = metadata["bins"]
        if (
            not isinstance(metadata["method"], str)
            or not isinstance(bins, torch.Tensor)
            or bins.ndim != 1
            or bins.numel() == 0
        ):
            raise ValidationError(
                "Automatic workflow checkpoint discretizer bins are invalid."
            )


def _restore_selector(payload: dict[str, Any]) -> GeoDetectorSelector:
    # Checkpoints saved before the interaction strategy existed were fit with
    # single-factor q ranking; restore them under that semantics.
    params = {"strategy": "factor", "top_interactions": 10, **payload["params"]}
    selector = GeoDetectorSelector(**params)
    selector.selected_features_ = list(payload["selected_features"])
    selector.scores_ = _restore_frame(payload["scores"])
    selector.discretizers_ = {
        feature: DiscretizationMetadata(
            method=metadata["method"],
            bins=np.asarray(metadata["bins"].cpu(), dtype=float),
        )
        for feature, metadata in payload["discretizers"].items()
    }
    return selector


def _restore_estimator(
    payload: dict[str, Any], training_losses: list[float]
) -> GIGNNClassifier:
    checkpoint = _validate_checkpoint(payload)
    estimator = GIGNNClassifier(**checkpoint["params"])
    estimator.classes_ = np.asarray(checkpoint["classes"], dtype=object)
    estimator.n_features_in_ = checkpoint["n_features_in"]
    estimator.feature_names_in_ = tuple(checkpoint["feature_names_in"])
    estimator.label_encoder_ = LabelEncoder(classes_=estimator.classes_)
    estimator.model_ = estimator._build_model(
        estimator.n_features_in_, len(estimator.classes_)
    )
    estimator.model_.load_state_dict(checkpoint["state_dict"])
    estimator.model_.eval()
    estimator.training_result_ = TrainingResult(
        losses=[float(value) for value in training_losses]
    )
    return estimator


def _discovery_report_payload(report: SpatialRelationshipReport) -> dict[str, Any]:
    return {
        "selected_features": list(report.selected_features),
        "factor_scores": _frame_payload(report.factor_scores),
        "interaction_scores": _frame_payload(report.interaction_scores),
        "discretizer": report.discretizer,
        "max_bins": report.max_bins,
    }


def _validate_discovery_report_payload(payload: Any) -> None:
    if not isinstance(payload, dict) or set(payload) != {
        "selected_features",
        "factor_scores",
        "interaction_scores",
        "discretizer",
        "max_bins",
    }:
        raise ValidationError(
            "Automatic workflow checkpoint discovery report is invalid."
        )
    if not isinstance(payload["selected_features"], list) or not all(
        isinstance(value, str) for value in payload["selected_features"]
    ):
        raise ValidationError(
            "Automatic workflow checkpoint discovery features are invalid."
        )
    _validate_frame_payload(payload["factor_scores"])
    _validate_frame_payload(payload["interaction_scores"])
    if not isinstance(payload["discretizer"], str) or not isinstance(
        payload["max_bins"], int
    ):
        raise ValidationError(
            "Automatic workflow checkpoint discovery configuration is invalid."
        )


def _restore_discovery_report(payload: dict[str, Any]) -> SpatialRelationshipReport:
    return SpatialRelationshipReport(
        selected_features=list(payload["selected_features"]),
        factor_scores=_restore_frame(payload["factor_scores"]),
        interaction_scores=_restore_frame(payload["interaction_scores"]),
        discretizer=payload["discretizer"],
        max_bins=payload["max_bins"],
    )


def _graph_report_payload(report: GraphSelectionReport) -> dict[str, Any]:
    return {
        "selected_strategy": report.selected_strategy,
        "selected_params": _primitive(report.selected_params),
        "candidates": _frame_payload(report.candidates),
        "selected_edge_index": torch.as_tensor(
            report.selected_edge_index.copy(), dtype=torch.int64
        ),
    }


def _validate_graph_report_payload(payload: Any) -> None:
    if not isinstance(payload, dict) or set(payload) != {
        "selected_strategy",
        "selected_params",
        "candidates",
        "selected_edge_index",
    }:
        raise ValidationError("Automatic workflow checkpoint graph report is invalid.")
    if not isinstance(payload["selected_strategy"], str) or not isinstance(
        payload["selected_params"], dict
    ):
        raise ValidationError(
            "Automatic workflow checkpoint graph selection is invalid."
        )
    _validate_frame_payload(payload["candidates"])
    edges = payload["selected_edge_index"]
    if (
        not isinstance(edges, torch.Tensor)
        or edges.ndim != 2
        or edges.shape[0] != 2
        or edges.shape[1] > MAX_EDGES
        or edges.dtype not in {torch.int32, torch.int64}
    ):
        raise ValidationError(
            "Automatic workflow checkpoint selected edges are invalid."
        )


def _restore_graph_report(payload: dict[str, Any]) -> GraphSelectionReport:
    return GraphSelectionReport(
        selected_strategy=payload["selected_strategy"],
        selected_params=dict(payload["selected_params"]),
        candidates=_restore_frame(payload["candidates"]),
        selected_edge_index=np.asarray(
            payload["selected_edge_index"].cpu(), dtype=np.int64
        ),
    )


def _model_report_payload(report: ModelSelectionReport) -> dict[str, Any]:
    return {
        "best_params": _primitive(report.best_params),
        "candidates": _frame_payload(report.candidates),
    }


def _validate_model_report_payload(payload: Any) -> None:
    if not isinstance(payload, dict) or set(payload) != {"best_params", "candidates"}:
        raise ValidationError("Automatic workflow checkpoint model report is invalid.")
    if not isinstance(payload["best_params"], dict):
        raise ValidationError(
            "Automatic workflow checkpoint best model params are invalid."
        )
    _validate_frame_payload(payload["candidates"])


def _restore_model_report(payload: dict[str, Any]) -> ModelSelectionReport:
    return ModelSelectionReport(
        best_params=dict(payload["best_params"]),
        candidates=_restore_frame(payload["candidates"]),
    )


def _generalization_report_payload(report: GeneralizationReport) -> dict[str, Any]:
    return {
        "accuracy": report.accuracy,
        "macro_f1": report.macro_f1,
        "weighted_f1": report.weighted_f1,
        "per_class": _frame_payload(report.per_class),
        "partition": report.partition,
        "indices": torch.as_tensor(
            np.asarray(report.indices, dtype=np.int64).copy(), dtype=torch.int64
        ),
    }


def _validate_generalization_report_payload(payload: Any) -> None:
    if not isinstance(payload, dict) or set(payload) != {
        "accuracy",
        "macro_f1",
        "weighted_f1",
        "per_class",
        "partition",
        "indices",
    }:
        raise ValidationError(
            "Automatic workflow checkpoint generalization report is invalid."
        )
    for name in ("accuracy", "macro_f1", "weighted_f1"):
        value = payload[name]
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(float(value))
            or not 0 <= float(value) <= 1
        ):
            raise ValidationError("Automatic workflow checkpoint metric is invalid.")
    _validate_frame_payload(payload["per_class"])
    if payload["partition"] not in {"validation", "test"}:
        raise ValidationError(
            "Automatic workflow checkpoint report partition is invalid."
        )
    indices = payload["indices"]
    if (
        not isinstance(indices, torch.Tensor)
        or indices.numel() == 0
        or indices.numel() > MAX_AUTO_GRAPH_NODES
    ):
        raise ValidationError(
            "Automatic workflow checkpoint report indices are invalid."
        )
    _tensor_to_indices(indices)


def _restore_generalization_report(payload: dict[str, Any]) -> GeneralizationReport:
    return GeneralizationReport(
        accuracy=float(payload["accuracy"]),
        macro_f1=float(payload["macro_f1"]),
        weighted_f1=float(payload["weighted_f1"]),
        per_class=_restore_frame(payload["per_class"]),
        partition=payload["partition"],
        indices=_tensor_to_indices(payload["indices"]),
    )


def _frame_payload(frame: pd.DataFrame) -> dict[str, Any]:
    return {
        "columns": [str(column) for column in frame.columns],
        "index": [_primitive_value(value) for value in frame.index.tolist()],
        "index_name": (None if frame.index.name is None else str(frame.index.name)),
        "rows": [
            [_primitive_value(value) for value in row]
            for row in frame.itertuples(index=False, name=None)
        ],
    }


def _validate_frame_payload(payload: Any) -> None:
    if not isinstance(payload, dict) or set(payload) != {
        "columns",
        "index",
        "index_name",
        "rows",
    }:
        raise ValidationError("Automatic workflow checkpoint table is invalid.")
    columns = payload["columns"]
    index = payload["index"]
    index_name = payload["index_name"]
    rows = payload["rows"]
    if (
        not isinstance(columns, list)
        or len(columns) > MAX_FEATURES
        or not all(isinstance(value, str) for value in columns)
        or len(set(columns)) != len(columns)
        or not isinstance(index, list)
        or not isinstance(rows, list)
        or len(index) != len(rows)
        or (index_name is not None and not isinstance(index_name, str))
        or len(rows) > MAX_AUTO_GRAPH_NODES
        or not all(isinstance(row, list) and len(row) == len(columns) for row in rows)
    ):
        raise ValidationError("Automatic workflow checkpoint table is invalid.")


def _restore_frame(payload: dict[str, Any]) -> pd.DataFrame:
    frame = pd.DataFrame(payload["rows"], columns=payload["columns"])
    frame.index = pd.Index(payload["index"], name=payload["index_name"])
    return frame


def _constructor_keys() -> tuple[str, ...]:
    return (
        "detector_grid",
        "graph_grid",
        "model_grid",
        "top_k_features",
        "random_state",
        "validation_fraction",
        "test_fraction",
        "max_grid_candidates",
    )


def _primitive(
    values: dict[str, Any], *, keys: tuple[str, ...] | None = None
) -> dict[str, Any]:
    selected = values if keys is None else {key: values[key] for key in keys}
    return {str(key): _primitive_value(value) for key, value in selected.items()}


def _primitive_value(value: Any) -> Any:
    if value is None or value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, np.generic):
        return _primitive_value(value.item())
    if isinstance(value, (list, tuple)):
        return [_primitive_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _primitive_value(item) for key, item in value.items()}
    raise ValidationError(
        f"Automatic workflow checkpoint contains unsupported value {type(value).__name__}."
    )


def _primitive_sequence(values: Any, name: str) -> list[Any]:
    try:
        return [_primitive_value(value) for value in values]
    except ValidationError as exc:
        raise ValidationError(
            f"{name} contains unsupported checkpoint values."
        ) from exc


def _tensor_to_indices(value: Any) -> np.ndarray:
    if (
        not isinstance(value, torch.Tensor)
        or value.ndim != 1
        or value.dtype not in {torch.int32, torch.int64}
    ):
        raise ValidationError(
            "Automatic workflow checkpoint indices must be 1D integer tensors."
        )
    return np.asarray(value.cpu(), dtype=np.int64)


def _validate_package_version(saved_version: Any) -> None:
    if not isinstance(saved_version, str):
        raise ValidationError(
            "Automatic workflow checkpoint package version is invalid."
        )
    current = _package_version()
    saved_release = _release_version(saved_version)
    current_release = _release_version(current)
    if saved_release[:2] != current_release[:2]:
        raise ValidationError(
            "Automatic workflow checkpoint package version is incompatible."
        )


def _release_version(value: str) -> tuple[int, int, int]:
    match = re.fullmatch(r"(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)", value)
    if match is None:
        raise ValidationError(
            "Automatic workflow checkpoint package version is invalid."
        )
    return tuple(int(part) for part in match.groups())


def _package_version() -> str:
    try:
        return version("gignn")
    except PackageNotFoundError:
        return "0.1.0"
