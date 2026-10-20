from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SpatialRelationshipReport:
    selected_features: list[str]
    factor_scores: pd.DataFrame
    interaction_scores: pd.DataFrame
    discretizer: str
    max_bins: int


@dataclass(frozen=True)
class GraphSelectionReport:
    selected_strategy: str
    selected_params: dict[str, object]
    candidates: pd.DataFrame
    selected_edge_index: np.ndarray

    def __post_init__(self) -> None:
        edges = np.asarray(self.selected_edge_index, dtype=np.int64).copy()
        edges.flags.writeable = False
        object.__setattr__(self, "selected_edge_index", edges)


@dataclass(frozen=True)
class ModelSelectionReport:
    best_params: dict[str, object]
    candidates: pd.DataFrame


@dataclass(frozen=True)
class GeneralizationReport:
    accuracy: float
    macro_f1: float
    weighted_f1: float
    per_class: pd.DataFrame
    partition: str
    indices: np.ndarray

    def __post_init__(self) -> None:
        indices = np.asarray(self.indices, dtype=np.int64).copy()
        indices.flags.writeable = False
        object.__setattr__(self, "indices", indices)


@dataclass(frozen=True)
class SimulationReport:
    baseline_predictions: np.ndarray
    scenario_predictions: np.ndarray
    baseline_probabilities: np.ndarray
    scenario_probabilities: np.ndarray
    probability_delta: np.ndarray
    changed_mask: np.ndarray

    def __post_init__(self) -> None:
        for name in (
            "baseline_predictions",
            "scenario_predictions",
            "baseline_probabilities",
            "scenario_probabilities",
            "probability_delta",
            "changed_mask",
        ):
            arr = np.asarray(getattr(self, name)).copy()
            arr.flags.writeable = False
            object.__setattr__(self, name, arr)
