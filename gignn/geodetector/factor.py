from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
from scipy import stats

from gignn.exceptions import ValidationError


def factor_detector(
    df: pd.DataFrame,
    y: str,
    factors: Iterable[str],
    alpha: float | None = 0.05,
) -> pd.DataFrame:
    factor_names = list(factors)
    if y not in df.columns:
        raise ValidationError(f"Target column {y!r} does not exist.")
    if not factor_names:
        return pd.DataFrame(columns=["factor", "q", "p_value"])
    result = []
    y_values = target_to_numeric(df[y])
    for factor in factor_names:
        if factor not in df.columns:
            raise ValidationError(f"Factor column {factor!r} does not exist.")
        q = geodetector_q(y_values, df[factor].to_numpy())
        p_value = _q_p_value(y_values, df[factor].to_numpy(), q)
        result.append({"factor": factor, "q": q, "p_value": p_value})

    output = (
        pd.DataFrame(result).sort_values("q", ascending=False).reset_index(drop=True)
    )
    if alpha is not None:
        output = output[output["p_value"] < alpha].reset_index(drop=True)
    return output


def target_to_numeric(values: pd.Series) -> np.ndarray:
    """Numeric target for the variance-based q statistic.

    Numeric targets are used as-is and ordered categoricals by category code.
    Unordered labels have no meaningful variance beyond two classes, so they
    are rejected instead of being coded alphabetically.
    """
    if values.isna().any():
        raise ValidationError("target values must not contain missing values.")
    if isinstance(values.dtype, pd.CategoricalDtype) and values.dtype.ordered:
        return values.cat.codes.to_numpy(dtype=float)
    numeric = pd.to_numeric(values, errors="coerce")
    if numeric.notna().all():
        return numeric.to_numpy(dtype=float)
    categorical = pd.Categorical(values)
    if len(categorical.categories) > 2:
        raise ValidationError(
            "GeoDetector q is variance-based: non-numeric targets with more than "
            "two classes must be ordered categoricals or ordinal numeric codes."
        )
    return categorical.codes.astype(float)


def geodetector_q(y_values: np.ndarray, factor_values: np.ndarray) -> float:
    """Eq. (1): ``q = 1 - sum_h N_h sigma_h^2 / (N sigma^2)`` (population variances)."""
    y_array = np.asarray(y_values, dtype=float)
    factor_array = np.asarray(factor_values)
    if y_array.shape[0] != factor_array.shape[0]:
        raise ValidationError("y and factor must have the same length.")
    _require_valid_factor_values(factor_array)
    total_variance = np.var(y_array, ddof=0)
    if total_variance == 0:
        return 0.0

    weighted_variance = 0.0
    for group in pd.unique(factor_array):
        group_values = y_array[factor_array == group]
        weighted_variance += group_values.size * np.var(group_values, ddof=0)
    q = 1.0 - weighted_variance / (y_array.size * total_variance)
    return float(np.clip(q, 0.0, 1.0))


def _q_p_value(y_values: np.ndarray, factor_values: np.ndarray, q: float) -> float:
    """One-way ANOVA F test of ``q``: ``F = (N - L) q / ((L - 1)(1 - q))``.

    Returns NaN when every stratum holds a single observation (``N <= L``),
    because ``q = 1`` is then trivial and the test is undefined.
    """
    y_array = np.asarray(y_values, dtype=float)
    groups = pd.unique(factor_values)
    n = y_array.size
    group_count = len(groups)
    if group_count <= 1:
        return 0.0 if q > 0 else 1.0
    if n <= group_count:
        return float("nan")
    if q >= 1.0:
        return 0.0
    denominator = (group_count - 1) * (1 - q)
    if denominator == 0:
        return 0.0
    f_stat = ((n - group_count) * q) / denominator
    return float(stats.f.sf(f_stat, group_count - 1, n - group_count))


def _require_valid_factor_values(factor_array: np.ndarray) -> None:
    if factor_array.size == 0:
        raise ValidationError("factor must contain at least one value.")
    if np.issubdtype(factor_array.dtype, np.number):
        if not np.isfinite(factor_array.astype(float)).all():
            raise ValidationError(
                "factor values must be finite numeric values; "
                "missing or non-finite stratification values are not allowed."
            )
    elif pd.isna(factor_array).any():
        raise ValidationError("factor values must not contain missing values.")
