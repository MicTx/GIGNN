from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from gignn.exceptions import ValidationError


@dataclass(frozen=True)
class DiscretizationMetadata:
    method: str
    bins: np.ndarray


def spatial_discretization(
    data_frame: pd.DataFrame,
    continuous_columns: str | Iterable[str],
    method: str = "quantiles",
    num_intervals: int = 4,
) -> tuple[pd.DataFrame, dict[str, DiscretizationMetadata]]:
    if isinstance(continuous_columns, str):
        columns = [continuous_columns]
    else:
        columns = list(continuous_columns)
    if num_intervals < 1:
        raise ValidationError("num_intervals must be at least 1.")

    discretized = data_frame.copy(deep=True)
    metadata: dict[str, DiscretizationMetadata] = {}
    for column in columns:
        if column not in discretized.columns:
            raise ValidationError(f"Column {column!r} does not exist.")
        values = pd.to_numeric(discretized[column], errors="raise").to_numpy(
            dtype=float
        )
        if values.size == 0:
            raise ValidationError(f"Column {column!r} is empty.")
        if not np.isfinite(values).all():
            raise ValidationError(f"Column {column!r} must contain only finite values.")
        labels, bins = _discretize_values(
            values, method=method, num_intervals=num_intervals
        )
        discretized[column] = labels
        metadata[column] = DiscretizationMetadata(method=method, bins=bins)
    return discretized, metadata


def apply_discretization(
    data_frame: pd.DataFrame,
    metadata: dict[str, DiscretizationMetadata],
) -> pd.DataFrame:
    discretized = data_frame.copy(deep=True)
    for column, column_metadata in metadata.items():
        if column not in discretized.columns:
            raise ValidationError(f"Column {column!r} does not exist.")
        values = pd.to_numeric(discretized[column], errors="raise").to_numpy(
            dtype=float
        )
        if not np.isfinite(values).all():
            raise ValidationError(f"Column {column!r} must contain only finite values.")
        discretized[column] = (
            np.searchsorted(column_metadata.bins, values, side="right").astype(np.int64)
            + 1
        )
    return discretized


def _discretize_values(
    values: np.ndarray, method: str, num_intervals: int
) -> tuple[np.ndarray, np.ndarray]:
    if method == "quantiles":
        quantiles = np.linspace(0, 1, num_intervals + 1)[1:-1]
        bins = np.unique(np.quantile(values, quantiles))
    elif method == "equal_interval":
        bins = np.linspace(values.min(), values.max(), num_intervals + 1)[1:-1]
    elif method in {"natural_breaks", "fisher_jenks"}:
        bins = _jenks_natural_breaks(values, num_intervals)
    elif method == "maximum_breaks":
        bins = _maximum_breaks(values, num_intervals)
    elif method == "box_plot":
        q1, q3 = np.quantile(values, [0.25, 0.75])
        iqr = q3 - q1
        bins = np.array([q1 - 1.5 * iqr, q1, q3, q3 + 1.5 * iqr], dtype=float)
        bins = np.unique(bins[(bins > values.min()) & (bins < values.max())])
    elif method == "std_mean":
        mean = values.mean()
        std = values.std(ddof=0)
        bins = np.array([mean - std, mean, mean + std], dtype=float)
        bins = np.unique(bins[(bins > values.min()) & (bins < values.max())])
    else:
        raise ValidationError(f"Invalid discretization method {method!r}.")

    labels = np.searchsorted(bins, values, side="right") + 1
    return labels.astype(np.int64), np.asarray(bins, dtype=float)


def _maximum_breaks(values: np.ndarray, num_intervals: int) -> np.ndarray:
    """Cut at the midpoints of the ``num_intervals - 1`` widest value gaps."""
    unique_values = np.unique(np.asarray(values, dtype=float))
    gaps = np.diff(unique_values)
    count = min(num_intervals - 1, gaps.size)
    if count <= 0:
        return np.asarray([], dtype=float)
    widest = np.sort(np.lexsort((np.arange(gaps.size), -gaps))[:count])
    return (unique_values[widest] + unique_values[widest + 1]) / 2.0


def _jenks_natural_breaks(values: np.ndarray, num_intervals: int) -> np.ndarray:
    """Exact Fisher-Jenks optimum: minimum total within-class squared deviation.

    ``mapclassify.NaturalBreaks`` (used by the reference notebook) optimizes
    the same objective with a randomized k-means heuristic; this dynamic
    program returns the global optimum deterministically.
    """
    sorted_values = np.sort(np.asarray(values, dtype=float))
    unique_values = np.unique(sorted_values)
    class_count = min(num_intervals, unique_values.size)
    if class_count <= 1:
        return np.asarray([], dtype=float)

    lower_class_limits = np.zeros(
        (sorted_values.size + 1, class_count + 1), dtype=np.int64
    )
    variance_combinations = np.full(
        (sorted_values.size + 1, class_count + 1), np.inf, dtype=float
    )

    for index in range(1, class_count + 1):
        lower_class_limits[1, index] = 1
        variance_combinations[1, index] = 0.0

    for sample_count in range(2, sorted_values.size + 1):
        # Reversed cumulative sums reproduce the scalar loop's accumulation
        # order exactly: value_sum/squared_sum for lower_index = j are built
        # from sorted_values[sample_count - 1] downwards, so the segment SSDs
        # are bitwise identical to the reference implementation.
        reversed_prefix = sorted_values[sample_count - 1 :: -1]
        value_sum = np.cumsum(reversed_prefix)
        squared_sum = np.cumsum(reversed_prefix * reversed_prefix)
        weight = np.arange(1, sample_count + 1)
        variance = squared_sum - value_sum * value_sum / weight
        # variance[offset - 1] is the SSD of the segment
        # [lower_index - 1, sample_count - 1] for lower_index =
        # sample_count - offset + 1. The scalar loop visited lower_index
        # descending and replaced only on strict <, so among exact ties the
        # largest lower_index wins: argmin over descending candidates
        # preserves that tie-break.
        candidates = (
            variance[: sample_count - 1, None]
            + variance_combinations[sample_count - 1 : 0 : -1, 1:class_count]
        )
        candidates = np.where(np.isnan(candidates), np.inf, candidates)
        best = np.argmin(candidates, axis=0)
        best_values = candidates[best, np.arange(class_count - 1)]
        current = variance_combinations[sample_count, 2 : class_count + 1]
        improved = best_values < current
        variance_combinations[sample_count, 2 : class_count + 1] = np.where(
            improved, best_values, current
        )
        lower_class_limits[sample_count, 2 : class_count + 1] = np.where(
            improved,
            sample_count - best,
            lower_class_limits[sample_count, 2 : class_count + 1],
        )

        lower_class_limits[sample_count, 1] = 1
        variance_combinations[sample_count, 1] = variance[-1]

    class_maxima = [sorted_values[-1]]
    sample_count = sorted_values.size
    for class_index in range(class_count, 1, -1):
        lower_index = lower_class_limits[sample_count, class_index]
        class_maxima.append(sorted_values[lower_index - 2])
        sample_count = lower_index - 1
    class_maxima.append(sorted_values[0])
    class_maxima = sorted(class_maxima)

    cutpoints = []
    for class_maximum in class_maxima[1:-1]:
        greater_values = sorted_values[sorted_values > class_maximum]
        if greater_values.size:
            cutpoints.append((class_maximum + greater_values[0]) / 2.0)
        else:
            cutpoints.append(class_maximum)
    return np.asarray(cutpoints, dtype=float)
