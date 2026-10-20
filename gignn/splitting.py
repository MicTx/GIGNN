from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable
import math
import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from gignn.exceptions import ValidationError

_SPLIT_NAMES = ("train", "validation", "test")


@dataclass(frozen=True)
class SplitIndices:
    train: np.ndarray
    validation: np.ndarray
    test: np.ndarray

    def __post_init__(self) -> None:
        for name in _SPLIT_NAMES:
            values = np.asarray(getattr(self, name), dtype=np.int64).copy()
            values.flags.writeable = False
            object.__setattr__(self, name, values)

    def as_dict(self) -> dict[str, np.ndarray]:
        return {
            "train": self.train.copy(),
            "validation": self.validation.copy(),
            "test": self.test.copy(),
        }


def resolve_split_indices(
    y: pd.Series,
    *,
    validation_fraction: float,
    test_fraction: float,
    random_state: int | None,
    split_labels: Iterable[str] | None = None,
    spatial_groups: Iterable[object] | None = None,
) -> SplitIndices:
    _validate_split_fractions(validation_fraction, test_fraction)
    if split_labels is not None and spatial_groups is not None:
        raise ValidationError(
            "split_labels and spatial_groups must not be provided together."
        )
    if split_labels is not None:
        return _split_from_labels(split_labels, row_count=len(y))
    if spatial_groups is not None:
        return _split_from_groups(
            spatial_groups,
            row_count=len(y),
            validation_fraction=validation_fraction,
            test_fraction=test_fraction,
            random_state=random_state,
        )
    return _random_split(
        y,
        validation_fraction=validation_fraction,
        test_fraction=test_fraction,
        random_state=random_state,
    )


def _validate_split_fractions(validation_fraction: float, test_fraction: float) -> None:
    values = (validation_fraction, test_fraction)
    if not all(
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and value > 0
        for value in values
    ):
        raise ValidationError("split fractions must be positive finite numbers.")
    if validation_fraction + test_fraction >= 1:
        raise ValidationError("split fractions must sum to less than 1.")


def _split_from_labels(split_labels: Iterable[str], *, row_count: int) -> SplitIndices:
    labels = pd.Series(list(split_labels), dtype="string")
    if len(labels) != row_count:
        raise ValidationError("split_labels must match X rows.")
    if labels.isna().any() or set(labels) != set(_SPLIT_NAMES):
        raise ValidationError(
            "split_labels must contain train, validation, and test rows."
        )
    return SplitIndices(
        train=_indices_for(labels, "train"),
        validation=_indices_for(labels, "validation"),
        test=_indices_for(labels, "test"),
    )


def _split_from_groups(
    spatial_groups: Iterable[object],
    *,
    row_count: int,
    validation_fraction: float,
    test_fraction: float,
    random_state: int | None,
) -> SplitIndices:
    groups = pd.Series(list(spatial_groups))
    if len(groups) != row_count:
        raise ValidationError("spatial_groups must match X rows.")
    if groups.isna().any():
        raise ValidationError("spatial_groups must not contain missing values.")
    try:
        unique_values = pd.unique(groups)
    except TypeError as exc:
        raise ValidationError("spatial_groups must be hashable values.") from exc
    unique_groups = np.asarray(list(unique_values), dtype=object)
    if len(unique_groups) < 3:
        raise ValidationError("spatial_groups must contain at least three groups.")
    rng = np.random.default_rng(random_state)
    ordered_groups = unique_groups[rng.permutation(len(unique_groups))]
    test_group_count = max(1, int(np.ceil(len(unique_groups) * test_fraction)))
    validation_group_count = max(
        1, int(np.ceil(len(unique_groups) * validation_fraction))
    )
    if test_group_count + validation_group_count >= len(unique_groups):
        raise ValidationError("split fractions leave no training spatial group.")
    test_groups = set(ordered_groups[:test_group_count])
    validation_groups = set(
        ordered_groups[test_group_count : test_group_count + validation_group_count]
    )
    test_row_fraction = float(groups.isin(test_groups).mean())
    validation_row_fraction = float(groups.isin(validation_groups).mean())
    _warn_fraction_divergence(
        name="test",
        requested=test_fraction,
        row_fraction=test_row_fraction,
        group_count=test_group_count,
        total_groups=len(unique_groups),
    )
    _warn_fraction_divergence(
        name="validation",
        requested=validation_fraction,
        row_fraction=validation_row_fraction,
        group_count=validation_group_count,
        total_groups=len(unique_groups),
    )
    labels = pd.Series(
        [
            (
                "test"
                if value in test_groups
                else "validation" if value in validation_groups else "train"
            )
            for value in groups
        ],
        dtype="string",
    )
    return _split_from_labels(labels, row_count=row_count)


def _warn_fraction_divergence(
    *,
    name: str,
    requested: float,
    row_fraction: float,
    group_count: int,
    total_groups: int,
) -> None:
    if abs(row_fraction - requested) <= 0.10:
        return
    group_fraction = group_count / total_groups
    warnings.warn(
        f"Requested {name}_fraction={requested:.2f} but realized "
        f"row_fraction={row_fraction:.2f} and group_fraction={group_fraction:.2f} "
        f"({group_count}/{total_groups} groups).",
        UserWarning,
        stacklevel=3,
    )


def _random_split(
    y: pd.Series,
    *,
    validation_fraction: float,
    test_fraction: float,
    random_state: int | None,
) -> SplitIndices:
    if len(y) < 3:
        raise ValidationError("random split requires at least three rows.")

    all_indices = np.arange(len(y), dtype=np.int64)
    train_validation, test = _safe_train_test_split(
        all_indices,
        y,
        fraction=test_fraction,
        random_state=random_state,
    )
    relative_validation = validation_fraction / (1 - test_fraction)
    train_validation_y = y.iloc[train_validation].reset_index(drop=True)
    train, validation = _safe_train_test_split(
        train_validation,
        train_validation_y,
        fraction=relative_validation,
        random_state=random_state,
    )
    return SplitIndices(train=train, validation=validation, test=test)


def _safe_train_test_split(
    indices: np.ndarray,
    y: pd.Series,
    *,
    fraction: float,
    random_state: int | None,
) -> tuple[np.ndarray, np.ndarray]:
    holdout_count = int(np.ceil(len(indices) * fraction))
    train_count = len(indices) - holdout_count
    if train_count < 1:
        raise ValidationError("split fractions leave no training rows.")
    class_count = y.nunique()
    can_stratify = (
        y.value_counts().min() >= 2
        and holdout_count >= class_count
        and train_count >= class_count
    )
    train, holdout = train_test_split(
        indices,
        test_size=fraction,
        random_state=random_state,
        stratify=y if can_stratify else None,
    )
    return np.asarray(train, dtype=np.int64), np.asarray(holdout, dtype=np.int64)


def _indices_for(labels: pd.Series, name: str) -> np.ndarray:
    return np.flatnonzero(labels.to_numpy() == name).astype(np.int64)
