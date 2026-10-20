from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from gignn.exceptions import ValidationError


@dataclass(frozen=True)
class OrdinalTargetEncoder:
    classes_: tuple[object, ...]

    def transform(self, values: Iterable[object]) -> np.ndarray:
        mapping = {
            _label_key(label): index for index, label in enumerate(self.classes_)
        }
        try:
            return np.asarray(
                [mapping[_label_key(value)] for value in values], dtype=np.int64
            )
        except (KeyError, TypeError) as exc:
            label = exc.args[0][1] if isinstance(exc, KeyError) else None
            raise ValidationError(f"Unknown label {label!r} for class_order.") from exc

    def inverse_transform(self, encoded: Iterable[int]) -> np.ndarray:
        raw_values = list(encoded)
        if any(
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            for value in raw_values
        ):
            raise ValidationError("Encoded ordinal labels must be integers.")
        values = np.asarray(raw_values, dtype=np.int64)
        if values.size == 0:
            return np.asarray([], dtype=object)
        if values.min() < 0 or values.max() >= len(self.classes_):
            raise ValidationError("Encoded ordinal labels are out of range.")
        classes = np.asarray(self.classes_, dtype=object)
        return classes[values]


def fit_ordinal_target(
    y: pd.Series | Iterable[object],
    *,
    class_order: Iterable[object] | None,
) -> tuple[pd.Series, OrdinalTargetEncoder]:
    target = pd.Series(y).reset_index(drop=True)
    if target.empty:
        raise ValidationError("y must contain at least one label.")
    if target.isna().any():
        raise ValidationError("y must not contain missing labels.")

    observed = _unique_labels(target)
    if class_order is None and isinstance(target.dtype, pd.CategoricalDtype):
        if not target.dtype.ordered:
            raise ValidationError(
                "class_order must be provided for unordered categorical labels."
            )
        classes = tuple(target.dtype.categories.tolist())
        if not _same_label_set(classes, observed):
            raise ValidationError(
                "ordered categorical labels must use every declared category."
            )
    elif class_order is None:
        if not pd.api.types.is_numeric_dtype(target.dtype):
            raise ValidationError(
                "class_order must be provided for non-numeric ordinal labels."
            )
        classes = _sorted_numeric_classes(observed)
    else:
        classes = tuple(class_order)
        if not classes:
            raise ValidationError("class_order must not be empty.")
        _validate_distinct_labels(classes, "class_order")
        if not _same_label_set(classes, observed):
            raise ValidationError(
                "class_order must contain every observed label exactly once."
            )

    _validate_distinct_labels(classes, "class_order")
    encoder = OrdinalTargetEncoder(classes_=classes)
    encoded = pd.Series(encoder.transform(target), name=target.name)
    return encoded, encoder


def _unique_labels(values: Iterable[object]) -> list[object]:
    unique: dict[tuple[type[object], object], object] = {}
    try:
        for value in values:
            key = _label_key(value)
            if key not in unique:
                unique[key] = value
    except TypeError as exc:
        raise ValidationError("ordinal labels must be hashable scalar values.") from exc
    return list(unique.values())


def _sorted_numeric_classes(observed: list[object]) -> tuple[object, ...]:
    try:
        ordered = sorted(observed, key=float)
    except (TypeError, ValueError) as exc:
        raise ValidationError(
            "numeric ordinal labels must be real numbers sortable by value."
        ) from exc
    if any(not np.isfinite(float(value)) for value in ordered):
        raise ValidationError("numeric ordinal labels must be finite.")
    return tuple(ordered)


def _validate_distinct_labels(values: tuple[object, ...], name: str) -> None:
    try:
        keys = [_label_key(value) for value in values]
        duplicates = len(set(keys)) != len(keys)
    except TypeError as exc:
        raise ValidationError(f"{name} labels must be hashable scalar values.") from exc
    if duplicates:
        raise ValidationError(f"{name} must not contain duplicate labels.")


def _same_label_set(left: Iterable[object], right: Iterable[object]) -> bool:
    try:
        return {_label_key(value) for value in left} == {
            _label_key(value) for value in right
        }
    except TypeError as exc:
        raise ValidationError("ordinal labels must be hashable scalar values.") from exc


def _label_key(value: object) -> tuple[type[object], object]:
    if isinstance(value, np.generic):
        value = value.item()
    return type(value), value
