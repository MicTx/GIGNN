from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import math
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse, stats
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import KFold, StratifiedKFold

from gignn.exceptions import ValidationError
from gignn.validation import validate_edge_index


@dataclass(frozen=True)
class StandardizedSpatialDataset:
    frame: pd.DataFrame
    features: pd.DataFrame
    target: pd.Series
    ids: pd.Series | None
    geometry: pd.Series | None


@dataclass(frozen=True)
class SpatialDatasetReport:
    id_column: str | None
    target_column: str
    feature_columns: list[str]
    geometry_column: str | None
    missing_filled: dict[str, int]
    row_count: int


@dataclass(frozen=True)
class MoranResult:
    """Global Moran's I on row-standardized binary contiguity weights.

    Field semantics follow ``esda.Moran`` (PySAL) defaults: ``z_norm`` and the
    two-sided ``p_norm`` use the variance under the normality assumption;
    ``p_sim`` is the folded one-tailed pseudo p-value
    ``(min(m, P - m) + 1) / (P + 1)`` with ``m = #{I_sim >= I}`` over ``P``
    random relabelings, and ``z_sim`` standardizes ``I`` by their distribution.
    Statistics are NaN when the values are constant.
    """

    statistic: float
    expected: float
    variance_norm: float
    z_norm: float
    p_norm: float
    permutations: int
    z_sim: float | None
    p_sim: float | None


@dataclass(frozen=True)
class GeneralizationDiagnostics:
    accuracy: float
    bias_frame: pd.DataFrame
    classification_report: pd.DataFrame
    bias_moran: MoranResult | None = None


@dataclass(frozen=True)
class CrossValidationReport:
    """Pooled out-of-fold results of transductive k-fold cross-validation.

    ``out_of_fold_predictions`` is the pooled prediction vector used for the
    residual-bias Moran's I of the paper (Section 5.2.2).
    """

    fold_metrics: pd.DataFrame
    out_of_fold_observed: pd.Series
    out_of_fold_predictions: pd.Series
    bias_moran: MoranResult | None = None


def standardize_spatial_dataset(
    frame: pd.DataFrame,
    *,
    target_column: str,
    id_column: str | None = None,
    feature_columns: list[str] | None = None,
    geometry_column: str | None = None,
    fill_missing: float | None = None,
) -> tuple[StandardizedSpatialDataset, SpatialDatasetReport]:
    if not isinstance(frame, pd.DataFrame):
        raise ValidationError("frame must be a pandas DataFrame.")
    required = [target_column]
    if id_column is not None:
        required.append(id_column)
    if geometry_column is not None:
        required.append(geometry_column)
    _require_columns(frame, required)

    resolved_features = _resolve_feature_columns(
        frame,
        target_column=target_column,
        id_column=id_column,
        geometry_column=geometry_column,
        feature_columns=feature_columns,
    )
    standardized = frame.copy(deep=True).reset_index(drop=True)
    feature_frame = standardized[resolved_features].copy(deep=True)
    missing_filled = {
        column: int(feature_frame[column].isna().sum()) for column in resolved_features
    }
    if fill_missing is not None:
        feature_frame = feature_frame.fillna(fill_missing)
    feature_frame = _validate_numeric_features(feature_frame)
    standardized.loc[:, resolved_features] = feature_frame
    target = standardized[target_column].copy(deep=True).reset_index(drop=True)
    if target.isna().any():
        raise ValidationError("target_column must not contain missing values.")
    ids = (
        standardized[id_column].copy(deep=True).reset_index(drop=True)
        if id_column is not None
        else None
    )
    geometry = (
        standardized[geometry_column].copy(deep=True).reset_index(drop=True)
        if geometry_column is not None
        else None
    )
    return (
        StandardizedSpatialDataset(
            frame=standardized,
            features=feature_frame,
            target=target,
            ids=ids,
            geometry=geometry,
        ),
        SpatialDatasetReport(
            id_column=id_column,
            target_column=target_column,
            feature_columns=resolved_features,
            geometry_column=geometry_column,
            missing_filled=missing_filled,
            row_count=len(standardized),
        ),
    )


def generalization_diagnostics(
    observed: pd.Series | list[Any] | np.ndarray,
    predicted: pd.Series | list[Any] | np.ndarray,
    *,
    class_names: dict[Any, str] | None = None,
    class_order: list[Any] | None = None,
    edge_index: object | None = None,
    permutations: int = 999,
    random_state: int | None = None,
) -> GeneralizationDiagnostics:
    """Accuracy, per-unit bias, and classification report for ordered classes.

    ``Bias`` is the predicted minus the observed class code (paper Fig. 10):
    positive values mean the model overestimated the level. With
    ``edge_index`` the spatial autocorrelation of the bias is tested with
    :func:`morans_i` (paper Section 5.2.2).
    """
    observed_series = pd.Series(observed).reset_index(drop=True)
    predicted_series = pd.Series(predicted).reset_index(drop=True)
    if len(observed_series) != len(predicted_series):
        raise ValidationError("observed and predicted must have the same length.")
    if observed_series.isna().any() or predicted_series.isna().any():
        raise ValidationError("observed and predicted must not contain missing values.")
    observed_codes, predicted_codes = _ordered_label_codes(
        observed_series,
        predicted_series,
        class_order=class_order,
        class_names=class_names,
    )
    bias = predicted_codes - observed_codes
    bias_frame = pd.DataFrame(
        {
            "observed": observed_series,
            "prediction": predicted_series,
            "Bias": bias,
            "Bias Type": [_bias_type(value) for value in bias],
            "Validation": [
                f"{_label_name(left, class_names)}-{_label_name(right, class_names)}"
                for left, right in zip(observed_series, predicted_series)
            ],
        }
    )
    report = classification_report(
        observed_series,
        predicted_series,
        output_dict=True,
        zero_division=1,
    )
    report_frame = pd.DataFrame(report).rename(
        columns=_classification_report_columns(class_names)
    )
    bias_moran = (
        None
        if edge_index is None
        else morans_i(
            bias, edge_index, permutations=permutations, random_state=random_state
        )
    )
    return GeneralizationDiagnostics(
        accuracy=float(accuracy_score(observed_series, predicted_series)),
        bias_frame=bias_frame,
        classification_report=report_frame,
        bias_moran=bias_moran,
    )


def morans_i(
    values: pd.Series | list[float] | np.ndarray,
    edge_index: object,
    *,
    permutations: int = 999,
    random_state: int | None = None,
) -> MoranResult:
    """Global Moran's I of ``values`` over the undirected graph ``edge_index``.

    Neighbors are the symmetrized edges without self-loops, weighted
    ``w_ij = 1 / deg(i)`` (row standardization); isolated nodes keep zero
    weights but count towards ``n``, as in ``esda.Moran``.
    """
    y = np.asarray(values, dtype=float).reshape(-1)
    n = y.size
    if n < 3:
        raise ValidationError("Moran's I requires at least three observations.")
    if not np.isfinite(y).all():
        raise ValidationError("values must contain only finite numbers.")
    if (
        isinstance(permutations, bool)
        or not isinstance(permutations, int)
        or permutations < 0
    ):
        raise ValidationError("permutations must be a non-negative integer.")
    weights = _row_standardized_weights(edge_index, n)
    s0 = float(weights.sum())
    symmetric = weights + weights.T
    s1 = 0.5 * float(symmetric.multiply(symmetric).sum())
    s2 = float(
        np.square(
            np.asarray(weights.sum(axis=1)).ravel()
            + np.asarray(weights.sum(axis=0)).ravel()
        ).sum()
    )
    expected = -1.0 / (n - 1)
    variance_norm = (n * n * s1 - n * s2 + 3 * s0 * s0) / (
        (n - 1) * (n + 1) * s0 * s0
    ) - expected**2
    z = y - y.mean()
    z2ss = float(z @ z)
    if z2ss == 0.0:
        nan = float("nan")
        simulated_nan = nan if permutations else None
        return MoranResult(
            nan,
            expected,
            variance_norm,
            nan,
            nan,
            permutations,
            simulated_nan,
            simulated_nan,
        )
    statistic = n / s0 * float(z @ (weights @ z)) / z2ss
    z_norm = (statistic - expected) / math.sqrt(variance_norm)
    p_norm = 2.0 * float(stats.norm.sf(abs(z_norm)))
    z_sim = p_sim = None
    if permutations:
        rng = np.random.default_rng(random_state)
        simulated = np.empty(permutations, dtype=float)
        block = max(1, 1_000_000 // n)
        for start in range(0, permutations, block):
            count = min(block, permutations - start)
            permuted = rng.permuted(np.tile(z, (count, 1)), axis=1)
            lagged = np.asarray(weights @ permuted.T).T
            simulated[start : start + count] = (
                n / s0 * np.einsum("ij,ij->i", permuted, lagged) / z2ss
            )
        larger = int((simulated >= statistic).sum())
        p_sim = (min(larger, permutations - larger) + 1.0) / (permutations + 1.0)
        spread = float(simulated.std())
        z_sim = (
            (statistic - float(simulated.mean())) / spread
            if spread > 0
            else float("nan")
        )
    return MoranResult(
        statistic, expected, variance_norm, z_norm, p_norm, permutations, z_sim, p_sim
    )


def cross_validate(
    estimator_factory: Callable[[], Any],
    X: pd.DataFrame,
    y: pd.Series | Iterable[object],
    *,
    n_splits: int = 5,
    random_state: int | None = None,
    class_order: list[Any] | None = None,
    geometry: Any = None,
    edge_index: object | None = None,
    moran_permutations: int = 999,
    factory_kwargs: dict[str, Any] | None = None,
) -> CrossValidationReport:
    """Transductive k-fold evaluation following the paper's protocol (§5.2.1).

    Each round calls ``estimator_factory(**factory_kwargs).fit(X, y,
    train_mask=..., geometry=..., edge_index=...)`` — the same graph and
    feature values for every node, but the loss restricted to the training
    folds — then pools predictions on the held-out nodes. ``geometry`` /
    ``edge_index`` are forwarded to both ``fit`` and ``predict`` so every fold
    sees the identical graph; the same ``edge_index`` also defines the
    neighborhood weights of the out-of-fold bias Moran's I (§5.2.2).

    Folds are stratified when every class has at least ``n_splits`` members,
    plain shuffled KFold otherwise. To avoid leaking label information into
    evaluation, run ``GeoDetectorSelector`` *inside* the factory (per fold)
    and mask ``y`` there — the paper's published workflow selected factors on
    all labels before splitting.
    """
    labels = pd.Series(y).reset_index(drop=True)
    row_count = len(X)
    if labels.isna().any():
        raise ValidationError("cross_validate requires complete labels.")
    if len(labels) != row_count:
        raise ValidationError("X and y must have the same number of rows.")
    if not isinstance(n_splits, int) or isinstance(n_splits, bool) or n_splits < 2:
        raise ValidationError("n_splits must be an integer >= 2.")
    if n_splits > row_count:
        raise ValidationError("n_splits must not exceed the number of rows.")
    kwargs = dict(factory_kwargs or {})
    counts = labels.value_counts()
    if counts.min() >= n_splits:
        splitter = StratifiedKFold(
            n_splits=n_splits, shuffle=True, random_state=random_state
        )
    else:
        splitter = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    observed = labels.to_numpy()
    predictions = np.empty(row_count, dtype=object)
    metrics = []
    for fold, (_, test_index) in enumerate(splitter.split(np.zeros(row_count), labels)):
        train_mask = np.ones(row_count, dtype=bool)
        train_mask[test_index] = False
        estimator = estimator_factory(**kwargs)
        estimator.fit(
            X,
            labels,
            train_mask=train_mask,
            geometry=geometry,
            edge_index=edge_index,
        )
        fold_predictions = np.asarray(
            estimator.predict(X, geometry=geometry, edge_index=edge_index),
            dtype=object,
        )
        predictions[test_index] = fold_predictions[test_index]
        # Lists keep sklearn's type_of_target happy with object-dtype labels.
        fold_observed = list(observed[test_index])
        fold_predicted = list(predictions[test_index])
        metrics.append(
            {
                "fold": fold,
                "n_test": int(test_index.size),
                "accuracy": float(accuracy_score(fold_observed, fold_predicted)),
                "f1_macro": float(
                    f1_score(
                        fold_observed,
                        fold_predicted,
                        average="macro",
                        zero_division=0,
                    )
                ),
            }
        )
    oof_observed = pd.Series(observed, name=labels.name)
    oof_predicted = pd.Series(predictions, name="prediction")
    bias_moran = None
    if edge_index is not None:
        observed_codes, predicted_codes = _ordered_label_codes(
            oof_observed, oof_predicted, class_order=class_order, class_names=None
        )
        bias_moran = morans_i(
            predicted_codes - observed_codes,
            edge_index,
            permutations=moran_permutations,
            random_state=random_state,
        )
    return CrossValidationReport(
        fold_metrics=pd.DataFrame(metrics),
        out_of_fold_observed=oof_observed,
        out_of_fold_predictions=oof_predicted,
        bias_moran=bias_moran,
    )


def model_accuracy_table(
    accuracies: dict[str, float], *, year: int | str | None = None
) -> pd.DataFrame:
    rows = []
    for model, accuracy in accuracies.items():
        try:
            value = float(accuracy)
        except (TypeError, ValueError) as exc:
            raise ValidationError(
                "model accuracies must be real numbers in [0, 1]."
            ) from exc
        if not 0 <= value <= 1:
            raise ValidationError("model accuracies must be in [0, 1].")
        row = {"Model": model, "Accuracy": value}
        if year is not None:
            row = {"Year": year, **row}
        rows.append(row)
    return pd.DataFrame(rows)


def _ordered_label_codes(
    observed: pd.Series,
    predicted: pd.Series,
    *,
    class_order: list[Any] | None,
    class_names: dict[Any, str] | None,
) -> tuple[np.ndarray, np.ndarray]:
    if class_order is None:
        if class_names is not None:
            class_order = list(class_names.keys())
        else:
            try:
                class_order = sorted(pd.unique(pd.concat([observed, predicted])))
            except TypeError as exc:
                raise ValidationError(
                    "class_order must be provided when labels are not sortable."
                ) from exc
    try:
        mapping = {label: index for index, label in enumerate(class_order)}
        observed_codes = np.asarray([mapping[label] for label in observed], dtype=float)
        predicted_codes = np.asarray(
            [mapping[label] for label in predicted], dtype=float
        )
    except KeyError as exc:
        raise ValidationError(
            f"Unknown label {exc.args[0]!r} for class_order."
        ) from exc
    except TypeError as exc:
        raise ValidationError(
            "class_order and labels must be hashable scalar values."
        ) from exc
    return observed_codes, predicted_codes


def _row_standardized_weights(edge_index: object, num_nodes: int) -> sparse.csr_matrix:
    source, target = validate_edge_index(edge_index, num_nodes=num_nodes)
    keep = source != target
    rows = np.concatenate([source[keep], target[keep]])
    columns = np.concatenate([target[keep], source[keep]])
    if rows.size == 0:
        raise ValidationError("edge_index must connect at least two distinct nodes.")
    binary = sparse.coo_matrix(
        (np.ones(rows.size), (rows, columns)), shape=(num_nodes, num_nodes)
    ).tocsr()
    binary.data[:] = 1.0
    degree = np.asarray(binary.sum(axis=1)).ravel()
    inverse = np.divide(1.0, degree, out=np.zeros_like(degree), where=degree > 0)
    return sparse.csr_matrix(sparse.diags(inverse) @ binary)


def _classification_report_columns(
    class_names: dict[Any, str] | None,
) -> dict[Any, str]:
    if class_names is None:
        return {}
    mapped: dict[Any, str] = {}
    for key, value in class_names.items():
        mapped[key] = value
        mapped[str(key)] = value
    return mapped


def _resolve_feature_columns(
    frame: pd.DataFrame,
    *,
    target_column: str,
    id_column: str | None,
    geometry_column: str | None,
    feature_columns: list[str] | None,
) -> list[str]:
    if feature_columns is not None:
        _require_columns(frame, feature_columns)
        return list(feature_columns)
    excluded = {target_column}
    if id_column is not None:
        excluded.add(id_column)
    if geometry_column is not None:
        excluded.add(geometry_column)
    return [column for column in frame.columns if column not in excluded]


def _validate_numeric_features(features: pd.DataFrame) -> pd.DataFrame:
    converted = features.copy(deep=True)
    for column in converted.columns:
        try:
            converted[column] = pd.to_numeric(converted[column], errors="raise")
        except (TypeError, ValueError) as exc:
            raise ValidationError(
                f"Feature column {column!r} must be numeric."
            ) from exc
    values = converted.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValidationError(
            "feature columns must contain only finite numeric values."
        )
    return converted


def _bias_type(value: Any) -> str:
    numeric_value = float(value)
    if numeric_value > 0:
        return f"Overestimated ({int(numeric_value) if numeric_value.is_integer() else numeric_value})"
    if numeric_value < 0:
        return f"Underestimated ({int(numeric_value) if numeric_value.is_integer() else numeric_value})"
    return "Correct"


def _label_name(value: Any, class_names: dict[Any, str] | None) -> str:
    if class_names is None:
        return str(value)
    return class_names.get(value, str(value))


def _require_columns(frame: pd.DataFrame, columns: list[str]) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValidationError(f"Missing required columns: {missing}.")
