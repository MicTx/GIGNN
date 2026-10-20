from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
import hashlib
from itertools import product
import math
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split

from gignn.estimators import GIGNNClassifier
from gignn.exceptions import NotFittedError, ValidationError
from gignn.geodetector import GeoDetectorSelector, interaction_detector
from gignn.graph import build_spatial_edges
from gignn.limits import (
    MAX_FEATURES,
    MAX_GRAPH_NEIGHBORS,
    MAX_HIDDEN_CHANNEL,
    MAX_HIDDEN_LAYERS,
    MAX_TRAINING_EPOCHS,
)
from gignn.ordinal import fit_ordinal_target
from gignn.reports import (
    GeneralizationReport,
    GraphSelectionReport,
    ModelSelectionReport,
    SimulationReport,
    SpatialRelationshipReport,
)
from gignn.splitting import resolve_split_indices
from gignn.validation import validate_X_y


@dataclass
class GIGNNAutoClassifier:
    detector_grid: dict[str, list[Any]] = field(
        default_factory=lambda: {"discretizer": ["quantiles"], "max_bins": [3]}
    )
    graph_grid: dict[str, list[Any]] = field(
        default_factory=lambda: {"strategy": ["knn"], "k": [8]}
    )
    model_grid: dict[str, list[Any]] = field(
        default_factory=lambda: {
            "hidden_channels": [(64,)],
            "learning_rate": [0.01],
            "max_epochs": [100],
        }
    )
    top_k_features: int | None = None
    random_state: int | None = None
    validation_fraction: float = 0.2
    test_fraction: float = 0.2
    max_grid_candidates: int = 100

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series | Iterable[object],
        *,
        geometry: Sequence[object] | None = None,
        edge_index: object | None = None,
        split_labels: Iterable[str] | None = None,
        spatial_groups: Iterable[object] | None = None,
        class_order: Iterable[object] | None = None,
        node_ids: Iterable[object] | None = None,
    ) -> "GIGNNAutoClassifier":
        if geometry is None and edge_index is None:
            raise ValidationError("geometry or edge_index must be provided.")
        self._validate_search_budget()
        X_aligned, original_target = validate_X_y(X, y)
        encoded_target, ordinal_encoder = fit_ordinal_target(
            original_target, class_order=class_order
        )
        self.target_ = original_target.copy(deep=True)
        self.ordinal_encoder_ = ordinal_encoder
        self.class_order_ = ordinal_encoder.classes_
        split = resolve_split_indices(
            encoded_target,
            validation_fraction=self.validation_fraction,
            test_fraction=self.test_fraction,
            random_state=self.random_state,
            split_labels=split_labels,
            spatial_groups=spatial_groups,
        )
        self.train_indices_ = split.train
        self.validation_indices_ = split.validation
        self.test_indices_ = split.test
        self.node_ids_ = _validate_node_ids(node_ids, row_count=len(X_aligned))
        selected_X, selector, self.discovery_report_ = self._discover_relationships(
            X_aligned.iloc[split.train].reset_index(drop=True),
            encoded_target.iloc[split.train].reset_index(drop=True),
            X_aligned,
        )
        self.selector_ = selector
        graph_candidates = self._graph_candidates(
            selected_X,
            geometry=geometry,
            edge_index=edge_index,
            partition_indices=split.as_dict(),
        )
        (
            self.estimator_,
            self.graph_report_,
            self.model_selection_report_,
            self.validation_report_,
            self.generalization_report_,
        ) = self._select_model(
            selected_X,
            encoded_target,
            graph_candidates,
            split.train,
            split.validation,
            split.test,
        )
        self.selected_features_ = self.discovery_report_.selected_features
        self.edge_index_ = self.graph_report_.selected_edge_index
        return self

    def predict(
        self,
        X: pd.DataFrame,
        *,
        geometry: Sequence[object] | None = None,
        edge_index: object | None = None,
    ) -> np.ndarray:
        self._require_fitted()
        selected_X = self.selector_.transform(X.reset_index(drop=True))
        graph_edges = self._prediction_edges(
            selected_X, geometry=geometry, edge_index=edge_index
        )
        encoded = self.estimator_.predict(selected_X, edge_index=graph_edges)
        return self.ordinal_encoder_.inverse_transform(encoded)

    def predict_proba(
        self,
        X: pd.DataFrame,
        *,
        geometry: Sequence[object] | None = None,
        edge_index: object | None = None,
    ) -> np.ndarray:
        self._require_fitted()
        selected_X = self.selector_.transform(X.reset_index(drop=True))
        graph_edges = self._prediction_edges(
            selected_X, geometry=geometry, edge_index=edge_index
        )
        return self.estimator_.predict_proba(selected_X, edge_index=graph_edges)

    def simulate(
        self,
        X_scenario: pd.DataFrame,
        *,
        baseline_X: pd.DataFrame,
        geometry: Sequence[object] | None = None,
        edge_index: object | None = None,
        node_ids: Iterable[object] | None = None,
        scenario_edge_index: object | None = None,
    ) -> SimulationReport:
        self._require_fitted()
        _validate_simulation_frames(baseline_X, X_scenario)
        _validate_simulation_ids(
            fitted_ids=self.node_ids_,
            node_ids=node_ids,
            row_count=len(baseline_X),
        )
        baseline_selected = self.selector_.transform(baseline_X.reset_index(drop=True))
        scenario_selected = self.selector_.transform(X_scenario.reset_index(drop=True))
        graph_edges = self._prediction_edges(
            baseline_selected, geometry=geometry, edge_index=edge_index
        )
        if scenario_edge_index is not None:
            scenario_edges = build_spatial_edges(
                edge_index=scenario_edge_index,
                num_nodes=len(baseline_selected),
            )
            if _edge_hash(graph_edges) != _edge_hash(scenario_edges):
                raise ValidationError(
                    "scenario topology must match baseline topology exactly."
                )
        baseline_probabilities = self.estimator_.predict_proba(
            baseline_selected, edge_index=graph_edges
        )
        scenario_probabilities = self.estimator_.predict_proba(
            scenario_selected, edge_index=graph_edges
        )
        baseline_predictions = self.ordinal_encoder_.inverse_transform(
            baseline_probabilities.argmax(axis=1)
        )
        scenario_predictions = self.ordinal_encoder_.inverse_transform(
            scenario_probabilities.argmax(axis=1)
        )
        return SimulationReport(
            baseline_predictions=baseline_predictions,
            scenario_predictions=scenario_predictions,
            baseline_probabilities=baseline_probabilities,
            scenario_probabilities=scenario_probabilities,
            probability_delta=scenario_probabilities - baseline_probabilities,
            changed_mask=baseline_predictions != scenario_predictions,
        )

    def save(self, path: str | Any) -> None:
        from gignn.auto_persistence import save_auto_classifier

        save_auto_classifier(self, path)

    @classmethod
    def load(cls, path: str | Any) -> "GIGNNAutoClassifier":
        from gignn.auto_persistence import load_auto_classifier

        return load_auto_classifier(path)

    def export_research_artifacts(self, output_dir: str | Any, **kwargs: Any) -> Any:
        self._require_fitted()
        from gignn.artifacts import export_research_artifacts

        return export_research_artifacts(self, output_dir, **kwargs)

    def _discover_relationships(
        self, training_X: pd.DataFrame, training_y: pd.Series, full_X: pd.DataFrame
    ) -> tuple[pd.DataFrame, GeoDetectorSelector, SpatialRelationshipReport]:
        candidates = []
        for params in _grid(self.detector_grid):
            selector = GeoDetectorSelector(
                discretizer=params.get("discretizer", "quantiles"),
                max_bins=params.get("max_bins", 3),
                top_k=self.top_k_features,
                alpha=None,
                strategy="factor",
            ).fit(training_X, training_y)
            score = (
                float(selector.scores_["q"].mean())
                if not selector.scores_.empty
                else 0.0
            )
            candidates.append((score, params, selector))
        _, best_params, best_selector = max(candidates, key=lambda item: item[0])
        selected_X = best_selector.transform(full_X)
        selected_training_X = best_selector.transform(training_X)
        interactions = interaction_detector(
            pd.concat(
                [selected_training_X, pd.Series(training_y, name="__target__")], axis=1
            ),
            y="__target__",
            factors=best_selector.selected_features_,
        )
        report = SpatialRelationshipReport(
            selected_features=best_selector.selected_features_,
            factor_scores=best_selector.scores_,
            interaction_scores=interactions,
            discretizer=best_params.get("discretizer", "quantiles"),
            max_bins=best_params.get("max_bins", 3),
        )
        return selected_X, best_selector, report

    def _graph_candidates(
        self,
        X: pd.DataFrame,
        *,
        geometry: Sequence[object] | None,
        edge_index: object | None,
        partition_indices: dict[str, np.ndarray] | None = None,
    ) -> list[dict[str, object]]:
        partitions = partition_indices or {}
        candidates = []
        if edge_index is not None:
            try:
                edges = build_spatial_edges(edge_index=edge_index, num_nodes=len(X))
            except ValidationError as exc:
                candidates.append(
                    _failed_graph_candidate(
                        strategy="edge_index",
                        params={},
                        num_nodes=len(X),
                        reason=str(exc),
                    )
                )
            else:
                partition_edges, required_failure, test_failure = (
                    _resolve_explicit_partition_edges(edges, partitions)
                )
                if required_failure is not None:
                    candidates.append(
                        _failed_graph_candidate(
                            strategy="edge_index",
                            params={},
                            num_nodes=len(X),
                            reason=required_failure,
                        )
                    )
                else:
                    candidates.append(
                        {
                            "strategy": "edge_index",
                            "params": {},
                            "edge_index": edges,
                            "partition_edges": partition_edges,
                            "partition_metrics": _partition_graph_metrics(
                                partition_edges, partitions
                            ),
                            "status": "ok",
                            "failure_reason": None,
                            "test_failure_reason": test_failure,
                            **_graph_metrics(edges, len(X)),
                        }
                    )
        if geometry is not None:
            aligned_geometry = list(geometry)
            if len(aligned_geometry) != len(X):
                raise ValidationError("geometry length must match X rows.")
            for params in _grid(self.graph_grid):
                graph_params = _validate_graph_params(params)
                strategy = graph_params["strategy"]
                try:
                    edges = build_spatial_edges(
                        geometry=aligned_geometry,
                        strategy=strategy,
                        k=graph_params["k"],
                        threshold=graph_params["threshold"],
                        num_nodes=len(X),
                    )
                except ValidationError as exc:
                    candidates.append(
                        _failed_graph_candidate(
                            strategy=strategy,
                            params=graph_params,
                            num_nodes=len(X),
                            reason=str(exc),
                        )
                    )
                    continue
                partition_edges, required_failure, test_failure = (
                    _resolve_geometry_partition_edges(
                        aligned_geometry,
                        partitions,
                        strategy=strategy,
                        k=graph_params["k"],
                        threshold=graph_params["threshold"],
                    )
                )
                if required_failure is not None:
                    candidates.append(
                        _failed_graph_candidate(
                            strategy=strategy,
                            params=graph_params,
                            num_nodes=len(X),
                            reason=required_failure,
                        )
                    )
                    continue
                candidates.append(
                    {
                        "strategy": strategy,
                        "params": graph_params,
                        "edge_index": edges,
                        "partition_edges": partition_edges,
                        "partition_metrics": _partition_graph_metrics(
                            partition_edges, partitions
                        ),
                        "status": "ok",
                        "failure_reason": None,
                        "test_failure_reason": test_failure,
                        **_graph_metrics(edges, len(X)),
                    }
                )
        if not candidates:
            raise ValidationError("No graph candidates were generated.")
        return candidates

    def _select_model(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        graph_candidates: list[dict[str, object]],
        train_indices: np.ndarray,
        validation_indices: np.ndarray,
        test_indices: np.ndarray,
    ) -> tuple[
        GIGNNClassifier,
        GraphSelectionReport,
        ModelSelectionReport,
        GeneralizationReport,
        GeneralizationReport,
    ]:
        X_train = X.iloc[train_indices].reset_index(drop=True)
        y_train = y.iloc[train_indices].reset_index(drop=True)
        X_validation = X.iloc[validation_indices].reset_index(drop=True)
        y_validation = y.iloc[validation_indices].reset_index(drop=True)
        X_test = X.iloc[test_indices].reset_index(drop=True)
        y_test = y.iloc[test_indices].reset_index(drop=True)

        rows = []
        candidates = []
        for graph_candidate in graph_candidates:
            if graph_candidate["status"] != "ok":
                continue
            partition_edges = graph_candidate["partition_edges"]
            train_edges = partition_edges["train"]
            validation_edges = partition_edges["validation"]
            for params in _grid(self.model_grid):
                model_params = _validate_model_params(params)
                estimator = GIGNNClassifier(
                    hidden_channels=model_params["hidden_channels"],
                    num_neighbors_list=model_params["num_neighbors_list"],
                    learning_rate=model_params["learning_rate"],
                    max_epochs=model_params["max_epochs"],
                    random_state=self.random_state,
                    class_order=tuple(range(len(self.class_order_))),
                ).fit(X_train, y_train, edge_index=train_edges)
                predictions = np.asarray(
                    estimator.predict(X_validation, edge_index=validation_edges),
                    dtype=np.int64,
                )
                accuracy = float(accuracy_score(y_validation, predictions))
                row = {
                    "strategy": graph_candidate["strategy"],
                    **graph_candidate["params"],
                    **model_params,
                    "validation_accuracy": accuracy,
                    "num_edges": graph_candidate["num_edges"],
                    "mean_degree": graph_candidate["mean_degree"],
                    "isolated_nodes": graph_candidate["isolated_nodes"],
                }
                rows.append(row)
                candidates.append(
                    (accuracy, model_params, graph_candidate, predictions)
                )
        if not candidates:
            raise ValidationError(
                "All graph/model candidates failed: no induced edges were available "
                "for both training and validation."
            )
        best_accuracy, best_params, best_graph, best_predictions = max(
            candidates, key=lambda item: item[0]
        )
        validation_report = _generalization_report(
            y_validation,
            best_predictions,
            partition="validation",
            indices=validation_indices,
        )
        test_edges = best_graph["partition_edges"].get("test")
        if test_edges is None:
            reason = best_graph.get("test_failure_reason") or "no test edges"
            raise ValidationError(
                "Selected candidate cannot evaluate the independent test partition: "
                f"{reason}"
            )
        test_estimator = GIGNNClassifier(
            hidden_channels=best_params["hidden_channels"],
            num_neighbors_list=best_params["num_neighbors_list"],
            learning_rate=best_params["learning_rate"],
            max_epochs=best_params["max_epochs"],
            random_state=self.random_state,
            class_order=tuple(range(len(self.class_order_))),
        ).fit(X_train, y_train, edge_index=best_graph["partition_edges"]["train"])
        test_predictions = np.asarray(
            test_estimator.predict(X_test, edge_index=test_edges),
            dtype=np.int64,
        )
        generalization_report = _generalization_report(
            y_test,
            test_predictions,
            partition="test",
            indices=test_indices,
        )
        best_edge_index = best_graph["edge_index"]
        final_estimator = GIGNNClassifier(
            hidden_channels=best_params["hidden_channels"],
            num_neighbors_list=best_params["num_neighbors_list"],
            learning_rate=best_params["learning_rate"],
            max_epochs=best_params["max_epochs"],
            random_state=self.random_state,
            class_order=tuple(range(len(self.class_order_))),
        ).fit(X, y, edge_index=best_edge_index)
        graph_rows = []
        for graph_candidate in graph_candidates:
            graph_row = {
                "strategy": graph_candidate["strategy"],
                **graph_candidate["params"],
                "num_nodes": graph_candidate["num_nodes"],
                "num_edges": graph_candidate["num_edges"],
                "mean_degree": graph_candidate["mean_degree"],
                "isolated_nodes": graph_candidate["isolated_nodes"],
                "status": graph_candidate["status"],
                "failure_reason": graph_candidate["failure_reason"],
                "test_failure_reason": graph_candidate.get("test_failure_reason"),
            }
            for partition, metrics in graph_candidate.get(
                "partition_metrics", {}
            ).items():
                graph_row[f"{partition}_num_nodes"] = metrics["num_nodes"]
                graph_row[f"{partition}_num_edges"] = metrics["num_edges"]
                graph_row[f"{partition}_edge_hash"] = metrics["edge_hash"]
            matching_scores = [
                row["validation_accuracy"]
                for row in rows
                if row["strategy"] == graph_row["strategy"]
                and all(
                    row.get(key) == value
                    for key, value in graph_candidate["params"].items()
                )
            ]
            graph_row["validation_accuracy"] = (
                max(matching_scores) if matching_scores else np.nan
            )
            graph_rows.append(graph_row)
        return (
            final_estimator,
            GraphSelectionReport(
                selected_strategy=best_graph["strategy"],
                selected_params=dict(best_graph["params"]),
                candidates=pd.DataFrame(graph_rows),
                selected_edge_index=best_edge_index,
            ),
            ModelSelectionReport(
                best_params=dict(best_params), candidates=pd.DataFrame(rows)
            ),
            validation_report,
            generalization_report,
        )

    def _prediction_edges(
        self,
        X: pd.DataFrame,
        *,
        geometry: Sequence[object] | None,
        edge_index: object | None,
    ) -> np.ndarray:
        if edge_index is not None:
            return build_spatial_edges(edge_index=edge_index, num_nodes=len(X))
        if geometry is not None:
            graph_params = _validate_graph_params(self.graph_report_.selected_params)
            if self.graph_report_.selected_strategy == "edge_index":
                return build_spatial_edges(
                    edge_index=self.edge_index_, num_nodes=len(X)
                )
            return build_spatial_edges(
                geometry=geometry,
                num_nodes=len(X),
                strategy=graph_params["strategy"],
                k=graph_params["k"],
                threshold=graph_params["threshold"],
            )
        return self.edge_index_

    def _require_fitted(self) -> None:
        if not hasattr(self, "estimator_"):
            raise NotFittedError("GIGNNAutoClassifier must be fitted before use.")

    def _validate_search_budget(self) -> None:
        budget = _validate_candidate_budget(self.max_grid_candidates)
        _validate_search_grids(
            detector_grid=self.detector_grid,
            graph_grid=self.graph_grid,
            model_grid=self.model_grid,
        )
        detector_candidates = _grid_size(self.detector_grid)
        graph_candidates = _grid_size(self.graph_grid)
        model_candidates = _grid_size(self.model_grid)
        total = detector_candidates + graph_candidates * model_candidates
        if total > budget:
            raise ValidationError("Search grid exceeds candidate budget.")


def _validate_simulation_frames(
    baseline_X: pd.DataFrame, scenario_X: pd.DataFrame
) -> None:
    if len(baseline_X) != len(scenario_X):
        raise ValidationError("baseline and scenario must have the same row count.")
    if tuple(baseline_X.columns) != tuple(scenario_X.columns):
        raise ValidationError(
            "baseline and scenario must have the same feature columns in the same order."
        )
    if not baseline_X.index.equals(scenario_X.index):
        raise ValidationError("baseline and scenario must have the same row index.")


def _validate_simulation_ids(
    *,
    fitted_ids: np.ndarray | None,
    node_ids: Iterable[object] | None,
    row_count: int,
) -> None:
    if fitted_ids is None:
        if node_ids is not None:
            _validate_node_ids(node_ids, row_count=row_count)
        return
    if node_ids is None:
        raise ValidationError("node_ids are required for this fitted workflow.")
    resolved = _validate_node_ids(node_ids, row_count=row_count)
    if not np.array_equal(fitted_ids, resolved):
        raise ValidationError("node_ids must match the fitted identity order exactly.")


def _validate_node_ids(
    node_ids: Iterable[object] | None, *, row_count: int
) -> np.ndarray | None:
    if node_ids is None:
        return None
    values = pd.Series(list(node_ids), dtype=object)
    if len(values) != row_count:
        raise ValidationError("node_ids must match X rows.")
    if values.isna().any():
        raise ValidationError("node_ids must not contain missing values.")
    if values.duplicated().any():
        raise ValidationError("node_ids must be unique.")
    return values.to_numpy(dtype=object)


def _validate_candidate_budget(value: object) -> int:
    if (
        not isinstance(value, int)
        or isinstance(value, bool)
        or not 0 < value <= MAX_FEATURES
    ):
        raise ValidationError("max_grid_candidates is an invalid candidate budget.")
    return value


def _validate_search_grids(
    *,
    detector_grid: dict[str, list[Any]],
    graph_grid: dict[str, list[Any]],
    model_grid: dict[str, list[Any]],
) -> None:
    validators = {
        "detector_grid": (_validate_detector_params, {"discretizer", "max_bins"}),
        "graph_grid": (_validate_graph_params, {"strategy", "k", "threshold"}),
        "model_grid": (
            _validate_model_params,
            {
                "hidden_channels",
                "num_neighbors_list",
                "learning_rate",
                "max_epochs",
            },
        ),
    }
    for name, grid in (
        ("detector_grid", detector_grid),
        ("graph_grid", graph_grid),
        ("model_grid", model_grid),
    ):
        validator, allowed_keys = validators[name]
        if not isinstance(grid, dict) or not grid:
            raise ValidationError(f"{name} must be a non-empty dictionary.")
        unknown_keys = set(grid) - allowed_keys
        if unknown_keys:
            raise ValidationError(
                f"{name} contains unknown keys: {sorted(unknown_keys)!r}."
            )
        candidate_count = 1
        for values in grid.values():
            if not isinstance(values, list) or not values:
                raise ValidationError("Search grid values must be non-empty lists.")
            if len(values) > MAX_FEATURES:
                raise ValidationError(f"{name} exceeds candidate resource limits.")
            candidate_count *= len(values)
            if candidate_count > MAX_FEATURES:
                raise ValidationError(f"{name} exceeds candidate resource limits.")
        for params in _grid(grid):
            validator(params)


def _validate_detector_params(params: dict[str, object]) -> dict[str, object]:
    discretizer = params.get("discretizer", "quantiles")
    if discretizer not in {
        "quantiles",
        "equal_interval",
        "natural_breaks",
        "fisher_jenks",
        "maximum_breaks",
        "box_plot",
        "std_mean",
    }:
        raise ValidationError("discretizer is invalid.")
    max_bins = params.get("max_bins", 3)
    if (
        not isinstance(max_bins, int)
        or isinstance(max_bins, bool)
        or not 0 < max_bins <= MAX_FEATURES
    ):
        raise ValidationError(
            "max_bins must be a positive integer within resource limits."
        )
    return {"discretizer": discretizer, "max_bins": max_bins}


def _grid(grid: dict[str, list[Any]]) -> list[dict[str, Any]]:
    if not grid:
        raise ValidationError("Search grid must not be empty.")
    for values in grid.values():
        if not isinstance(values, list) or not values:
            raise ValidationError("Search grid values must be non-empty lists.")
    keys = list(grid)
    values = [grid[key] for key in keys]
    return [dict(zip(keys, combination)) for combination in product(*values)]


def _grid_size(grid: dict[str, list[Any]]) -> int:
    size = 1
    for values in grid.values():
        if not isinstance(values, list) or not values:
            raise ValidationError("Search grid values must be non-empty lists.")
        size *= len(values)
    return size


def _validate_graph_params(params: dict[str, object]) -> dict[str, object]:
    strategy = params.get("strategy", "knn")
    if strategy not in {"knn", "distance", "contiguity"}:
        raise ValidationError("graph strategy is invalid.")
    k = params.get("k", 8)
    if (
        not isinstance(k, int)
        or isinstance(k, bool)
        or not 0 < k <= MAX_GRAPH_NEIGHBORS
    ):
        raise ValidationError("k must be a positive integer within resource limits.")
    threshold = params.get("threshold")
    if threshold is not None and (
        not isinstance(threshold, (int, float))
        or isinstance(threshold, bool)
        or not math.isfinite(float(threshold))
        or threshold <= 0
    ):
        raise ValidationError("threshold must be positive when provided.")
    return {
        "strategy": strategy,
        "k": k,
        "threshold": float(threshold) if threshold is not None else None,
    }


def _validate_model_params(params: dict[str, object]) -> dict[str, object]:
    hidden_channels = params.get("hidden_channels", (64,))
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
    num_neighbors_list = params.get("num_neighbors_list", (6, 12))
    if (
        not isinstance(num_neighbors_list, (tuple, list))
        or len(num_neighbors_list) != len(hidden_tuple) + 1
    ):
        raise ValidationError(
            "num_neighbors_list length must equal len(hidden_channels) + 1."
        )
    neighbor_tuple = tuple(num_neighbors_list)
    if not all(
        isinstance(value, int)
        and not isinstance(value, bool)
        and 0 < value <= MAX_GRAPH_NEIGHBORS
        for value in neighbor_tuple
    ):
        raise ValidationError(
            "num_neighbors_list must contain positive integers within resource limits."
        )
    learning_rate = params.get("learning_rate", 0.01)
    if (
        not isinstance(learning_rate, (int, float))
        or isinstance(learning_rate, bool)
        or not math.isfinite(float(learning_rate))
        or learning_rate <= 0
    ):
        raise ValidationError("learning_rate must be positive.")
    max_epochs = params.get("max_epochs", 100)
    if (
        not isinstance(max_epochs, int)
        or isinstance(max_epochs, bool)
        or not 0 < max_epochs <= MAX_TRAINING_EPOCHS
    ):
        raise ValidationError(
            "max_epochs must be a positive integer within resource limits."
        )
    return {
        "hidden_channels": hidden_tuple,
        "num_neighbors_list": neighbor_tuple,
        "learning_rate": float(learning_rate),
        "max_epochs": max_epochs,
    }


def _holdout_indices(
    y: pd.Series, validation_fraction: float, random_state: int | None
) -> tuple[np.ndarray, np.ndarray]:
    if not 0 < validation_fraction < 1:
        raise ValidationError("validation_fraction must be in (0, 1).")
    indices = np.arange(len(y))
    class_count = y.nunique()
    validation_count = int(np.ceil(len(y) * validation_fraction))
    train_count = len(y) - validation_count
    can_stratify = (
        y.value_counts().min() >= 2
        and validation_count >= class_count
        and train_count >= class_count
    )
    train, validation = train_test_split(
        indices,
        test_size=validation_fraction,
        random_state=random_state,
        stratify=y if can_stratify else None,
    )
    return np.asarray(train, dtype=np.int64), np.asarray(validation, dtype=np.int64)


def _resolve_explicit_partition_edges(
    edge_index: np.ndarray,
    partitions: dict[str, np.ndarray],
) -> tuple[dict[str, np.ndarray], str | None, str | None]:
    resolved: dict[str, np.ndarray] = {}
    test_failure = None
    for name, indices in partitions.items():
        try:
            resolved[name] = _filter_edge_index(edge_index, indices)
        except ValidationError as exc:
            if name == "test":
                test_failure = str(exc)
            else:
                return {}, f"{name} partition: {exc}", None
    return resolved, None, test_failure


def _resolve_geometry_partition_edges(
    geometry: Sequence[object],
    partitions: dict[str, np.ndarray],
    *,
    strategy: str,
    k: int,
    threshold: float | None,
) -> tuple[dict[str, np.ndarray], str | None, str | None]:
    resolved: dict[str, np.ndarray] = {}
    test_failure = None
    for name, indices in partitions.items():
        try:
            resolved[name] = _build_partition_edges(
                geometry,
                indices,
                strategy=strategy,
                k=k,
                threshold=threshold,
            )
        except ValidationError as exc:
            if name == "test":
                test_failure = str(exc)
            else:
                return {}, f"{name} partition: {exc}", None
    return resolved, None, test_failure


def _failed_graph_candidate(
    *,
    strategy: str,
    params: dict[str, object],
    num_nodes: int,
    reason: str,
) -> dict[str, object]:
    return {
        "strategy": strategy,
        "params": params,
        "edge_index": None,
        "partition_edges": {},
        "partition_metrics": {},
        "status": "failed",
        "failure_reason": reason,
        "test_failure_reason": None,
        "num_nodes": num_nodes,
        "num_edges": 0,
        "mean_degree": 0.0,
        "isolated_nodes": num_nodes,
    }


def _build_partition_edges(
    geometry: Sequence[object],
    indices: np.ndarray,
    *,
    strategy: str,
    k: int,
    threshold: float | None,
) -> np.ndarray:
    partition_geometry = [geometry[int(index)] for index in indices]
    if len(partition_geometry) == 1:
        return np.array([[0], [0]], dtype=np.int64)
    partition_k = min(k, len(partition_geometry) - 1)
    return build_spatial_edges(
        geometry=partition_geometry,
        num_nodes=len(partition_geometry),
        strategy=strategy,
        k=partition_k,
        threshold=threshold,
    )


def _filter_edge_index(edge_index: np.ndarray, kept_indices: np.ndarray) -> np.ndarray:
    mapping = {int(old): new for new, old in enumerate(kept_indices.tolist())}
    edges = []
    for source, target in edge_index.T:
        source_int = int(source)
        target_int = int(target)
        if source_int in mapping and target_int in mapping:
            edges.append((mapping[source_int], mapping[target_int]))
    if edges:
        return np.asarray(edges, dtype=np.int64).T
    raise ValidationError(
        "holdout split has no induced edges for this graph candidate."
    )


def _partition_graph_metrics(
    partition_edges: dict[str, np.ndarray], partitions: dict[str, np.ndarray]
) -> dict[str, dict[str, object]]:
    return {
        name: {
            "num_nodes": int(len(partitions[name])),
            "num_edges": int(edges.shape[1]),
            "edge_hash": _edge_hash(edges),
        }
        for name, edges in partition_edges.items()
    }


def _edge_hash(edge_index: np.ndarray) -> str:
    edges = np.asarray(edge_index, dtype="<i8")
    order = np.lexsort((edges[1], edges[0]))
    canonical = np.ascontiguousarray(edges[:, order])
    return hashlib.sha256(canonical.tobytes()).hexdigest()


def _graph_metrics(edge_index: np.ndarray, num_nodes: int) -> dict[str, float | int]:
    degree = np.bincount(edge_index[0], minlength=num_nodes)
    return {
        "num_nodes": num_nodes,
        "num_edges": int(edge_index.shape[1]),
        "mean_degree": float(degree.mean()),
        "isolated_nodes": int((degree == 0).sum()),
    }


def _generalization_report(
    y_true: pd.Series,
    y_pred: np.ndarray,
    *,
    partition: str,
    indices: np.ndarray,
) -> GeneralizationReport:
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    per_class = pd.DataFrame(report).T
    return GeneralizationReport(
        accuracy=float(accuracy_score(y_true, y_pred)),
        macro_f1=float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        weighted_f1=float(
            f1_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        per_class=per_class,
        partition=partition,
        indices=indices,
    )
