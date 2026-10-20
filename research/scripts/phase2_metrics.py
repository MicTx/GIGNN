"""Structural and evaluation metrics for Phase 2 ablations."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np
import pandas as pd

from gignn.graph import build_spatial_edges


def edge_hash(edge_index: np.ndarray) -> str:
    edges = np.asarray(edge_index, dtype="<i8")
    if edges.size == 0:
        return hashlib.sha256(b"").hexdigest()
    order = np.lexsort((edges[1], edges[0]))
    canonical = np.ascontiguousarray(edges[:, order])
    return hashlib.sha256(canonical.tobytes()).hexdigest()


def topology_change_rate(left: np.ndarray, right: np.ndarray) -> float:
    left_set = {_edge_key(edge) for edge in np.asarray(left, dtype=np.int64).T}
    right_set = {_edge_key(edge) for edge in np.asarray(right, dtype=np.int64).T}
    union = left_set | right_set
    if not union:
        return 0.0
    return float(len(left_set ^ right_set) / len(union))


def validation_optimism_bias(validation_score: float, test_score: float) -> float:
    return float(validation_score - test_score)


def ranking_consistency(rankings: Sequence[Sequence[str]]) -> float:
    if len(rankings) < 2:
        return 1.0
    first = tuple(rankings[0])
    matches = sum(1 for ranking in rankings if tuple(ranking) == first)
    return float(matches / len(rankings))


def mean_variance_ci(
    values: Sequence[float], *, confidence: float = 0.95
) -> dict[str, float]:
    array = np.asarray(list(values), dtype=float)
    if array.size == 0:
        raise ValueError("values must not be empty")
    mean = float(array.mean())
    variance = float(array.var(ddof=1)) if array.size > 1 else 0.0
    std = math.sqrt(variance)
    # Normal approximation is enough for the small fixed seed budget.
    z = 1.959963984540054 if confidence == 0.95 else 1.6448536269514722
    half_width = z * std / math.sqrt(array.size) if array.size > 1 else 0.0
    return {
        "mean": mean,
        "variance": variance,
        "std": std,
        "ci_low": mean - half_width,
        "ci_high": mean + half_width,
        "n": float(array.size),
    }


def selector_fingerprint(bins_by_feature: dict[str, Sequence[float]]) -> str:
    payload = {
        feature: [float(value) for value in bins]
        for feature, bins in sorted(bins_by_feature.items())
    }
    encoded = repr(payload).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def breakpoint_change_rate(
    left: dict[str, Sequence[float]], right: dict[str, Sequence[float]]
) -> float:
    """Fraction of selected features whose discretization breakpoints change."""

    features = sorted(set(left) | set(right))
    if not features:
        return 0.0
    changed = sum(
        [float(value) for value in left.get(feature, [])]
        != [float(value) for value in right.get(feature, [])]
        for feature in features
    )
    return float(changed / len(features))


def factor_score_change(
    left: dict[str, float], right: dict[str, float]
) -> dict[str, float]:
    """Per-feature q-statistic change and the mean absolute change."""

    features = sorted(set(left) | set(right))
    if not features:
        return {"mean_abs_change": 0.0, "per_feature": {}}
    per_feature = {
        feature: float(left.get(feature, 0.0)) - float(right.get(feature, 0.0))
        for feature in features
    }
    mean_abs = float(np.mean([abs(value) for value in per_feature.values()]))
    return {"mean_abs_change": mean_abs, "per_feature": per_feature}


def paired_difference_ci(
    treated: Sequence[float], baseline: Sequence[float], *, confidence: float = 0.95
) -> dict[str, float]:
    """Paired mean difference confidence interval (normal approximation)."""

    treated_array = np.asarray(list(treated), dtype=float)
    baseline_array = np.asarray(list(baseline), dtype=float)
    if treated_array.shape != baseline_array.shape:
        raise ValueError("treated and baseline must have matching shapes")
    diffs = treated_array - baseline_array
    if diffs.size == 0:
        raise ValueError("paired samples must not be empty")
    mean = float(diffs.mean())
    std = float(diffs.std(ddof=1)) if diffs.size > 1 else 0.0
    z = 1.959963984540054 if confidence == 0.95 else 1.6448536269514722
    half_width = z * std / math.sqrt(diffs.size) if diffs.size > 1 else 0.0
    return {
        "mean": mean,
        "std": std,
        "ci_low": mean - half_width,
        "ci_high": mean + half_width,
        "min": float(diffs.min()),
        "max": float(diffs.max()),
        "n_positive": float(int(np.sum(diffs > 0))),
        "n_zero": float(int(np.sum(diffs == 0))),
        "n_negative": float(int(np.sum(diffs < 0))),
        "n": float(diffs.size),
    }


def build_partition_geometry_edges(
    coordinates: pd.DataFrame,
    indices: np.ndarray,
    *,
    strategy: str = "knn",
    k: int = 2,
) -> np.ndarray:
    points = [
        _Point(
            float(coordinates.iloc[int(index)]["x"]),
            float(coordinates.iloc[int(index)]["y"]),
        )
        for index in indices
    ]
    if len(points) == 1:
        return np.array([[0], [0]], dtype=np.int64)
    local_k = min(k, len(points) - 1)
    return build_spatial_edges(
        geometry=points,
        num_nodes=len(points),
        strategy=strategy,
        k=local_k,
    )


def build_then_filter_edges(
    coordinates: pd.DataFrame,
    indices: np.ndarray,
    *,
    strategy: str = "knn",
    k: int = 2,
) -> np.ndarray:
    points = [
        _Point(float(row.x), float(row.y))
        for row in coordinates.itertuples(index=False)
    ]
    full_edges = build_spatial_edges(
        geometry=points,
        num_nodes=len(points),
        strategy=strategy,
        k=k,
    )
    mapping = {int(old): new for new, old in enumerate(indices.tolist())}
    kept = []
    for source, target in full_edges.T:
        source_int = int(source)
        target_int = int(target)
        if source_int in mapping and target_int in mapping:
            kept.append((mapping[source_int], mapping[target_int]))
    if not kept:
        return np.array([[0], [0]], dtype=np.int64)
    return np.asarray(kept, dtype=np.int64).T


def summarize_run(metrics: dict[str, Any]) -> dict[str, Any]:
    required = {
        "selector_fingerprint",
        "train_edge_hash",
        "topology_change_rate",
        "validation_accuracy",
        "test_accuracy",
        "validation_optimism_bias",
        "cross_region_accuracy",
        "candidate_ranking",
    }
    missing = required - set(metrics)
    if missing:
        raise ValueError(f"missing metrics: {sorted(missing)}")
    return dict(metrics)


def _edge_key(edge: Iterable[int]) -> tuple[int, int]:
    source, target = (int(value) for value in edge)
    return source, target


class _Point:
    def __init__(self, x: float, y: float) -> None:
        self.x = x
        self.y = y
        self.centroid = self
