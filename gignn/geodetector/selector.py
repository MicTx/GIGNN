from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from gignn.exceptions import NotFittedError, ValidationError
from gignn.geodetector.discretization import (
    DiscretizationMetadata,
    apply_discretization,
    spatial_discretization,
)
from gignn.geodetector.factor import factor_detector
from gignn.geodetector.interaction import interaction_detector

SELECTION_STRATEGIES = ("interaction", "factor")

_RESERVED_TARGET_COLUMN = "__gignn_target__"

_INTERACTION_COLUMNS = [
    "factor_1",
    "factor_2",
    "q_factor_1",
    "q_factor_2",
    "q_a",
    "q_b",
    "q_interaction",
    "p_value_interaction",
    "interaction",
]


@dataclass
class GeoDetectorSelector:
    """Select features following the GIGNN paper's GeoDetector step.

    Defaults match the paper: Jenks natural breaks with five strata
    (Section 5.1, Eq. 1) and interaction-detector selection (Section 2.2.1),
    ranking factors by centrality degree in the network formed by the top
    ``top_interactions`` interaction results (Section 4.2, Fig. 8). The
    ``"factor"`` strategy keeps single-factor q ranking for workflows
    documented on those semantics.
    """

    discretizer: str = "natural_breaks"
    max_bins: int = 5
    top_k: int | None = None
    alpha: float | None = None
    strategy: str = "interaction"
    top_interactions: int = 10

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "GeoDetectorSelector":
        if self.strategy not in SELECTION_STRATEGIES:
            raise ValidationError(f"strategy must be one of {SELECTION_STRATEGIES!r}.")
        if (
            not isinstance(self.top_interactions, int)
            or isinstance(self.top_interactions, bool)
            or self.top_interactions < 1
        ):
            raise ValidationError("top_interactions must be a positive integer.")
        if self.top_k is not None and (
            not isinstance(self.top_k, int)
            or isinstance(self.top_k, bool)
            or self.top_k < 1
        ):
            raise ValidationError("top_k must be a positive integer.")
        target_name = _RESERVED_TARGET_COLUMN
        if target_name in X.columns:
            raise ValidationError(
                f"X must not contain the reserved column {target_name!r}."
            )
        target = pd.Series(y)
        if len(target) != len(X):
            raise ValidationError("y must have the same length as X.")
        frame = X.copy(deep=True)
        frame[target_name] = target.array
        discretized, discretizers = spatial_discretization(
            frame,
            list(X.columns),
            method=self.discretizer,
            num_intervals=self.max_bins,
        )
        scores = factor_detector(
            discretized, y=target_name, factors=list(X.columns), alpha=self.alpha
        )
        if self.strategy == "factor":
            ranking = scores["factor"].tolist()
            self.interactions_ = None
            self.interaction_degree_ = None
        else:
            candidates = scores["factor"].tolist()
            if len(candidates) >= 2:
                interactions = (
                    interaction_detector(discretized, y=target_name, factors=candidates)
                    .sort_values("q_interaction", ascending=False, kind="mergesort")
                    .reset_index(drop=True)
                )
            else:
                interactions = pd.DataFrame(columns=_INTERACTION_COLUMNS)
            self.interactions_ = interactions
            degrees = dict.fromkeys(candidates, 0)
            network = interactions.head(self.top_interactions)
            for first, second in zip(network["factor_1"], network["factor_2"]):
                degrees[first] += 1
                degrees[second] += 1
            ranking = sorted(candidates, key=degrees.__getitem__, reverse=True)
            self.interaction_degree_ = pd.Series(
                [degrees[factor] for factor in ranking],
                index=pd.Index(ranking, name="factor"),
                name="degree",
                dtype=np.int64,
            )
        if self.top_k is not None:
            ranking = ranking[: self.top_k]
        self.selected_features_ = ranking
        self.scores_ = scores.set_index("factor")
        self.discretizers_: dict[str, DiscretizationMetadata] = discretizers
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not hasattr(self, "selected_features_") or not hasattr(
            self, "discretizers_"
        ):
            raise NotFittedError("GeoDetectorSelector must be fitted before transform.")
        selected_metadata = {
            feature: self.discretizers_[feature] for feature in self.selected_features_
        }
        discretized = apply_discretization(X, selected_metadata)
        return discretized[self.selected_features_].copy(deep=True)

    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        return self.fit(X, y).transform(X)
