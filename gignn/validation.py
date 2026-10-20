from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, TypeAlias

import numpy as np
import pandas as pd

from gignn.exceptions import ValidationError
from gignn.limits import MAX_EDGES, MAX_FEATURES

NumericFrame: TypeAlias = pd.DataFrame


def validate_feature_table(X: pd.DataFrame) -> NumericFrame:
    if not isinstance(X, pd.DataFrame):
        raise ValidationError("X must be a pandas DataFrame.")
    if X.empty:
        raise ValidationError("X must contain at least one row and one feature column.")
    if X.columns.duplicated().any():
        raise ValidationError("X must not contain duplicate column names.")
    if not all(isinstance(column, str) for column in X.columns):
        raise ValidationError(
            "X column names must be strings so checkpoints can round-trip."
        )
    if X.shape[1] > MAX_FEATURES:
        raise ValidationError(
            f"X must not contain more than {MAX_FEATURES} feature columns."
        )

    converted = X.copy(deep=True)
    for column in converted.columns:
        try:
            converted[column] = pd.to_numeric(converted[column], errors="raise")
        except (TypeError, ValueError) as exc:
            raise ValidationError(f"Column {column!r} must be numeric.") from exc

    values = converted.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValidationError("X must contain only finite numeric values.")
    return converted


def validate_X_y(
    X: pd.DataFrame, y: Iterable[object]
) -> tuple[NumericFrame, pd.Series]:
    X_validated = validate_feature_table(X)
    y_series = pd.Series(y).copy(deep=True)
    if len(X_validated) != len(y_series):
        raise ValidationError("X and y must have the same number of rows.")
    if y_series.empty:
        raise ValidationError("y must contain at least one label.")
    if y_series.isna().any():
        raise ValidationError("y must not contain missing labels.")
    return X_validated, y_series.reset_index(drop=True)


def validate_edge_index(
    edge_index: object, num_nodes: int, max_edges: int = MAX_EDGES
) -> np.ndarray:
    if num_nodes <= 0:
        raise ValidationError("num_nodes must be positive.")

    _guard_edge_index_size(edge_index, max_edges=max_edges)
    edges = np.asarray(edge_index)
    if edges.ndim != 2:
        raise ValidationError("edge_index must be a 2D array.")
    if edges.shape[0] != 2 and edges.shape[1] == 2:
        edges = edges.T
    if edges.shape[0] != 2:
        raise ValidationError(
            "edge_index must have shape (2, num_edges) or (num_edges, 2)."
        )
    if not np.issubdtype(edges.dtype, np.integer):
        raise ValidationError("edge_index must contain integer node indices.")
    if edges.size == 0:
        raise ValidationError("edge_index must contain at least one edge.")
    if edges.shape[1] > max_edges:
        raise ValidationError("edge_index exceeds max_edges.")
    if edges.min() < 0 or edges.max() >= num_nodes:
        raise ValidationError("edge_index contains node indices out of range.")
    return edges.astype(np.int64, copy=True)


def _guard_edge_index_size(edge_index: object, max_edges: int) -> None:
    shape = getattr(edge_index, "shape", None)
    if isinstance(shape, tuple) and len(shape) == 2:
        if shape[0] == 2 and shape[1] > max_edges:
            raise ValidationError("edge_index exceeds max_edges.")
        if shape[1] == 2 and shape[0] > max_edges:
            raise ValidationError("edge_index exceeds max_edges.")
        return
    if isinstance(edge_index, (list, tuple)):
        if len(edge_index) == 2:
            first_length = _safe_len(edge_index[0])
            second_length = _safe_len(edge_index[1])
            if first_length is not None and second_length is not None:
                if first_length > max_edges or second_length > max_edges:
                    raise ValidationError("edge_index exceeds max_edges.")
                return
        if len(edge_index) > max_edges:
            raise ValidationError("edge_index exceeds max_edges.")


def _safe_len(value: object) -> int | None:
    try:
        return len(value)  # type: ignore[arg-type]
    except TypeError:
        return None


def validate_class_order(values: Iterable[object]) -> np.ndarray:
    classes = np.asarray(list(values), dtype=object)
    if classes.size == 0:
        raise ValidationError("class_order must not be empty.")
    if pd.isna(classes).any():
        raise ValidationError("class_order must not contain missing labels.")
    for value in classes:
        if not isinstance(value, (str, bool, int, float, np.integer, np.floating)):
            raise ValidationError("class_order labels must be scalar values.")
        if isinstance(value, (float, np.floating)) and not np.isfinite(value):
            raise ValidationError("class_order labels must be finite.")
    keys = [_typed_label_key(value) for value in classes]
    if len(set(keys)) != len(keys):
        raise ValidationError("class_order must not contain duplicate labels.")
    return classes


def _typed_label_key(value: object) -> tuple[type[object], object]:
    if isinstance(value, np.generic):
        value = value.item()
    return type(value), value


@dataclass(frozen=True)
class LabelEncoder:
    classes_: np.ndarray | None = None

    def fit(self, y: Iterable[object]) -> "LabelEncoder":
        y_array = np.asarray(list(y), dtype=object)
        if y_array.size == 0:
            raise ValidationError("y must contain at least one label.")
        if pd.isna(y_array).any():
            raise ValidationError("y must not contain missing labels.")
        unique: dict[tuple[type[object], object], object] = {}
        try:
            for value in y_array:
                unique.setdefault(_typed_label_key(value), value)
        except TypeError as exc:
            raise ValidationError("y labels must be hashable values.") from exc
        try:
            values = sorted(unique.values())
        except TypeError as exc:
            raise ValidationError("y labels must be mutually sortable.") from exc
        return LabelEncoder(classes_=np.asarray(values, dtype=object))

    def transform(self, y: Iterable[object]) -> np.ndarray:
        if self.classes_ is None:
            raise ValidationError("LabelEncoder must be fitted before transform.")
        mapping = {
            _typed_label_key(label): index for index, label in enumerate(self.classes_)
        }
        try:
            return np.array(
                [mapping[_typed_label_key(label)] for label in y], dtype=np.int64
            )
        except KeyError as exc:
            raise ValidationError(f"Unknown label {exc.args[0][1]!r}.") from exc
        except TypeError as exc:
            raise ValidationError("y labels must be hashable values.") from exc

    def inverse_transform(self, encoded: Iterable[int]) -> np.ndarray:
        if self.classes_ is None:
            raise ValidationError(
                "LabelEncoder must be fitted before inverse_transform."
            )
        encoded_array = np.asarray(list(encoded), dtype=np.int64)
        if encoded_array.size == 0:
            return np.array([], dtype=object)
        if encoded_array.min() < 0 or encoded_array.max() >= len(self.classes_):
            raise ValidationError("Encoded labels contain class indices out of range.")
        return self.classes_[encoded_array]

    def fit_transform(self, y: Iterable[object]) -> np.ndarray:
        fitted = self.fit(y)
        return fitted.transform(y)
