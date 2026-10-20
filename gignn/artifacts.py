from __future__ import annotations

import hashlib
import io
import json
import math
import numbers
import shutil
from dataclasses import dataclass
from datetime import UTC, date, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from gignn.exceptions import NotFittedError, ValidationError

SCHEMA_VERSION = "1"
MANAGED_DIRECTORIES = ("raw", "preprocessing", "process", "results")
MANAGED_FILES = ("manifest.json", "README.md")
MAX_ARTIFACT_ROWS = 1_000_000
MAX_ARTIFACT_COLUMNS = 10_000
MAX_ARTIFACT_CELLS = 10_000_000
MAX_PROVENANCE_DEPTH = 20
MAX_PROVENANCE_ITEMS = 10_000
CSV_FORMULA_PREFIXES = ("=", "+", "-", "@", "\t", "\r", "\n")


@dataclass(frozen=True)
class ResearchArtifactBundle:
    root: Path
    manifest_path: Path
    files: dict[str, Path]


@dataclass(frozen=True)
class _ResultPayload:
    predictions: np.ndarray
    probabilities: np.ndarray
    ids: np.ndarray | None
    simulation_report: Any | None


def export_research_artifacts(
    model: Any,
    output_dir: str | Path,
    *,
    raw_frame: pd.DataFrame | None = None,
    include_raw: bool = False,
    ids: pd.Series | list[Any] | np.ndarray | None = None,
    geometry: list[Any] | pd.Series | None = None,
    predictions: pd.Series | list[Any] | np.ndarray | None = None,
    probabilities: pd.DataFrame | np.ndarray | None = None,
    simulation_report: Any | None = None,
    provenance: dict[str, Any] | None = None,
    overwrite: bool = False,
) -> ResearchArtifactBundle:
    _require_auto_model_fitted(model)
    output_path = Path(output_dir).expanduser()
    if output_path.is_symlink():
        raise ValidationError("Artifact output directory must not be a symlink.")
    root = output_path.resolve()
    frame = _ensure_frame(raw_frame)
    _validate_raw_frame_contract(model, frame, raw_frame=raw_frame)
    ids_array = _resolve_ids(
        ids, expected_rows=len(frame) if raw_frame is not None else None
    )
    result_payload = _resolve_result_payload(
        model,
        raw_frame=raw_frame,
        frame=frame,
        geometry=geometry,
        predictions=predictions,
        probabilities=probabilities,
        simulation_report=simulation_report,
        ids=ids_array,
    )
    _validate_provenance(provenance or {})
    safe_provenance = _json_safe(provenance or {})
    backup_root = _prepare_output_dir(root, overwrite=overwrite)
    try:
        directories = _create_directories(root)

        files: dict[str, Path] = {}
        raw_files = _write_raw_artifacts(
            directories["raw"], raw_frame=raw_frame, include_raw=include_raw
        )
        files.update(raw_files)
        files.update(
            _write_preprocessing_artifacts(
                directories["preprocessing"],
                model=model,
                raw_frame=raw_frame,
                ids=ids_array,
                geometry=geometry,
            )
        )
        files.update(_write_process_artifacts(directories["process"], model=model))
        result_files, result_summary = _write_result_artifacts(
            directories["results"],
            model=model,
            result_payload=result_payload,
        )
        files.update(result_files)
        readme_path = _write_readme(root)
        files["readme"] = readme_path
        manifest_path = _write_manifest(
            root,
            model=model,
            include_raw=include_raw,
            provenance=safe_provenance,
            files=files,
            result_summary=result_summary,
        )
        files["manifest"] = manifest_path
    except Exception:
        if root.exists():
            shutil.rmtree(root)
        if backup_root is not None and backup_root.exists():
            backup_root.rename(root)
        raise
    if backup_root is not None and backup_root.exists():
        shutil.rmtree(backup_root)
    return ResearchArtifactBundle(root=root, manifest_path=manifest_path, files=files)


def _require_auto_model_fitted(model: Any) -> None:
    required = [
        "discovery_report_",
        "graph_report_",
        "model_selection_report_",
        "generalization_report_",
        "estimator_",
    ]
    if not all(hasattr(model, name) for name in required):
        raise NotFittedError(
            "GIGNNAutoClassifier must be fitted before exporting artifacts."
        )


def _prepare_output_dir(root: Path, *, overwrite: bool) -> Path | None:
    """Validate output dir and return backup path if swapping an existing bundle."""
    if root.exists() and not root.is_dir():
        raise ValidationError("Artifact output path must be a directory.")
    if root.exists() and any(root.iterdir()) and not overwrite:
        raise ValidationError(f"Artifact output directory already exists: {root}")
    if (
        root.exists()
        and overwrite
        and any(root.iterdir())
        and not _is_managed_bundle(root)
    ):
        raise ValidationError(
            "overwrite=True can only replace an existing GIGNN artifact bundle."
        )
    if not root.exists() or not any(root.iterdir()):
        root.mkdir(parents=True, exist_ok=True)
        return None
    backup_root = root.parent / f".{root.name}.swap"
    if backup_root.exists():
        shutil.rmtree(backup_root)
    root.rename(backup_root)
    root.mkdir(parents=True, exist_ok=True)
    return backup_root


def _is_managed_bundle(root: Path) -> bool:
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file() or manifest_path.is_symlink():
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return (
        isinstance(manifest, dict)
        and manifest.get("producer") == "gignn.export_research_artifacts"
    )


def _create_directories(root: Path) -> dict[str, Path]:
    directories = {
        "raw": root / "raw",
        "preprocessing": root / "preprocessing",
        "process": root / "process",
        "results": root / "results",
    }
    for path in directories.values():
        path.mkdir(parents=True, exist_ok=True)
    return directories


def _write_raw_artifacts(
    raw_dir: Path, *, raw_frame: pd.DataFrame | None, include_raw: bool
) -> dict[str, Path]:
    files: dict[str, Path] = {}
    schema_path = raw_dir / "input_schema.json"
    summary_path = raw_dir / "input_summary.json"
    fingerprint_path = raw_dir / "input_fingerprint.json"
    frame = _ensure_frame(raw_frame)
    _write_json(schema_path, _frame_schema(frame))
    _write_json(summary_path, _frame_summary(frame))
    _write_json(fingerprint_path, _frame_fingerprint(frame))
    files["raw_schema"] = schema_path
    files["raw_summary"] = summary_path
    files["raw_fingerprint"] = fingerprint_path
    if include_raw and raw_frame is not None:
        raw_path = raw_dir / "raw_frame.csv"
        _safe_to_csv(raw_frame, raw_path)
        files["raw_frame"] = raw_path
    return files


def _write_preprocessing_artifacts(
    preprocessing_dir: Path,
    *,
    model: Any,
    raw_frame: pd.DataFrame | None,
    ids: pd.Series | list[Any] | np.ndarray | None,
    geometry: list[Any] | pd.Series | None,
) -> dict[str, Path]:
    files: dict[str, Path] = {}
    frame = _ensure_frame(raw_frame)
    feature_columns = list(getattr(model.estimator_, "feature_names_in_", []))
    feature_frame = _feature_frame(frame, feature_columns)

    feature_summary_path = preprocessing_dir / "feature_summary.csv"
    feature_summary = _feature_summary(feature_frame)
    _safe_to_csv(feature_summary, feature_summary_path, index=True)
    files["feature_summary"] = feature_summary_path

    missing_path = preprocessing_dir / "missing_values.csv"
    missing_values = _missing_values(frame)
    _safe_to_csv(missing_values, missing_path)
    files["missing_values"] = missing_path

    fill_summary_path = preprocessing_dir / "fill_summary.json"
    _write_json(fill_summary_path, _fill_summary(missing_values))
    files["fill_summary"] = fill_summary_path

    target_distribution_path = preprocessing_dir / "target_distribution.csv"
    _safe_to_csv(_target_distribution(model), target_distribution_path)
    files["target_distribution"] = target_distribution_path

    split_indices_path = preprocessing_dir / "split_indices.csv"
    _safe_to_csv(_split_indices(model, row_count=len(frame)), split_indices_path)
    files["split_indices"] = split_indices_path

    selected_columns_path = preprocessing_dir / "selected_columns.json"
    _write_json(
        selected_columns_path,
        {
            "selected_features": list(model.discovery_report_.selected_features),
            "model_feature_columns": feature_columns,
            "target_summary": _target_summary(model),
            "id_summary": _sequence_summary(ids),
            "geometry_summary": _geometry_summary(geometry),
        },
    )
    files["selected_columns"] = selected_columns_path
    return files


def _write_process_artifacts(process_dir: Path, *, model: Any) -> dict[str, Path]:
    files: dict[str, Path] = {}
    factor_scores_path = process_dir / "factor_scores.csv"
    model.discovery_report_.factor_scores.reset_index().rename(
        columns={"index": "factor"}
    ).pipe(_safe_to_csv, factor_scores_path)
    files["factor_scores"] = factor_scores_path

    interaction_scores_path = process_dir / "interaction_scores.csv"
    _safe_to_csv(model.discovery_report_.interaction_scores, interaction_scores_path)
    files["interaction_scores"] = interaction_scores_path

    discretization_bins_path = process_dir / "discretization_bins.csv"
    _safe_to_csv(_discretization_bins(model), discretization_bins_path)
    files["discretization_bins"] = discretization_bins_path

    graph_candidates_path = process_dir / "graph_candidates.csv"
    _safe_to_csv(model.graph_report_.candidates, graph_candidates_path)
    files["graph_candidates"] = graph_candidates_path

    selected_edges_path = process_dir / "selected_edges.csv"
    _safe_to_csv(
        _edge_table(model.graph_report_.selected_edge_index), selected_edges_path
    )
    files["selected_edges"] = selected_edges_path

    model_candidates_path = process_dir / "model_candidates.csv"
    _safe_to_csv(model.model_selection_report_.candidates, model_candidates_path)
    files["model_candidates"] = model_candidates_path

    training_loss_path = process_dir / "training_loss.csv"
    _safe_to_csv(_training_loss(model), training_loss_path)
    files["training_loss"] = training_loss_path
    return files


def _write_result_artifacts(
    results_dir: Path,
    *,
    model: Any,
    result_payload: _ResultPayload,
) -> tuple[dict[str, Path], dict[str, Any]]:
    files: dict[str, Path] = {}
    resolved_predictions = result_payload.predictions
    resolved_probabilities = result_payload.probabilities

    predictions_path = results_dir / "predictions.csv"
    _prediction_frame(resolved_predictions, result_payload.ids).pipe(
        _safe_to_csv, predictions_path
    )
    files["predictions"] = predictions_path

    probabilities_path = results_dir / "probabilities.csv"
    _probability_frame(model, resolved_probabilities, result_payload.ids).pipe(
        _safe_to_csv, probabilities_path
    )
    files["probabilities"] = probabilities_path

    metrics = {
        "accuracy": float(model.generalization_report_.accuracy),
        "macro_f1": float(model.generalization_report_.macro_f1),
        "weighted_f1": float(model.generalization_report_.weighted_f1),
    }
    validation_metrics = {
        "accuracy": float(model.validation_report_.accuracy),
        "macro_f1": float(model.validation_report_.macro_f1),
        "weighted_f1": float(model.validation_report_.weighted_f1),
    }
    validation_metrics_path = results_dir / "validation_metrics.json"
    _write_json(validation_metrics_path, validation_metrics)
    files["validation_metrics"] = validation_metrics_path
    metrics_path = results_dir / "metrics.json"
    _write_json(metrics_path, metrics)
    files["metrics"] = metrics_path

    per_class_path = results_dir / "per_class_report.csv"
    _safe_to_csv(model.generalization_report_.per_class, per_class_path)
    files["per_class_report"] = per_class_path

    if result_payload.simulation_report is not None:
        simulation_path = results_dir / "simulation_report.csv"
        _simulation_frame(result_payload.simulation_report, result_payload.ids).pipe(
            _safe_to_csv, simulation_path
        )
        files["simulation_report"] = simulation_path

    figures_path = results_dir / "figures_manifest.csv"
    _safe_to_csv(pd.DataFrame(columns=["name", "path", "format"]), figures_path)
    files["figures_manifest"] = figures_path
    return files, metrics


def _write_readme(root: Path) -> Path:
    path = root / "README.md"
    path.write_text(
        "# GIGNN Research Artifacts\n\n"
        "This bundle records raw, preprocessing, process, and results evidence "
        "for a fitted GIGNN automatic workflow. Raw input rows are only included "
        "when `include_raw=True`; otherwise this bundle keeps schema, summary, "
        "and fingerprint metadata.\n",
        encoding="utf-8",
    )
    return path


def _write_manifest(
    root: Path,
    *,
    model: Any,
    include_raw: bool,
    provenance: dict[str, Any],
    files: dict[str, Path],
    result_summary: dict[str, Any],
) -> Path:
    manifest_path = root / "manifest.json"
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "producer": "gignn.export_research_artifacts",
        "created_at": datetime.now(UTC).isoformat(),
        "package_version": _package_version(),
        "raw_policy": {"include_raw": include_raw},
        "provenance": _json_safe(provenance),
        "selected_features": list(model.discovery_report_.selected_features),
        "selected_graph_strategy": model.graph_report_.selected_strategy,
        "selected_graph_params": _json_safe(model.graph_report_.selected_params),
        "best_model_params": _json_safe(model.model_selection_report_.best_params),
        "metrics": result_summary,
        "files": sorted(_relative_path(path, root) for path in files.values()),
    }
    _write_json(manifest_path, manifest)
    return manifest_path


def _ensure_frame(frame: pd.DataFrame | None) -> pd.DataFrame:
    if frame is None:
        return pd.DataFrame()
    if not isinstance(frame, pd.DataFrame):
        raise ValidationError("raw_frame must be a pandas DataFrame when provided.")
    _validate_frame_size(frame)
    return frame.copy(deep=True).reset_index(drop=True)


def _validate_frame_size(frame: pd.DataFrame) -> None:
    if len(frame) > MAX_ARTIFACT_ROWS or len(frame.columns) > MAX_ARTIFACT_COLUMNS:
        raise ValidationError("raw_frame exceeds artifact export size limits.")
    if len(frame) * max(len(frame.columns), 1) > MAX_ARTIFACT_CELLS:
        raise ValidationError("raw_frame exceeds artifact export cell limit.")


def _validate_raw_frame_contract(
    model: Any, frame: pd.DataFrame, *, raw_frame: pd.DataFrame | None
) -> None:
    if raw_frame is None or frame.empty:
        return
    required = list(getattr(model.estimator_, "feature_names_in_", []))
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValidationError(
            f"raw_frame is missing fitted feature columns: {missing}."
        )


def _feature_frame(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    if frame.empty or not columns:
        return pd.DataFrame()
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValidationError(
            f"raw_frame is missing fitted feature columns: {missing}."
        )
    return frame[columns].copy(deep=True)


def _feature_summary(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    numeric = frame.apply(pd.to_numeric, errors="coerce")
    return numeric.describe(include="all")


def _missing_values(frame: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "column": list(frame.columns),
            "missing_count": [
                int(frame[column].isna().sum()) for column in frame.columns
            ],
        }
    )


def _fill_summary(missing_values: pd.DataFrame) -> dict[str, Any]:
    columns_with_missing = missing_values.loc[
        missing_values["missing_count"] > 0, "column"
    ].astype(str)
    return {
        "imputed_by_exporter": False,
        "source_missing_values_file": "missing_values.csv",
        "columns_with_missing_values": columns_with_missing.tolist(),
        "total_missing_values": int(missing_values["missing_count"].sum()),
    }


def _target_summary(model: Any) -> dict[str, Any]:
    target = getattr(model, "target_", None)
    if target is None:
        return {"present": False, "count": 0, "missing_count": None}
    target_series = pd.Series(target)
    return {
        "present": True,
        "count": int(len(target_series)),
        "missing_count": int(target_series.isna().sum()),
        "class_count": int(target_series.nunique(dropna=True)),
    }


def _sequence_summary(values: Any | None) -> dict[str, Any]:
    if values is None:
        return {"present": False, "count": 0, "missing_count": None}
    series = pd.Series(list(values))
    return {
        "present": True,
        "count": int(len(series)),
        "missing_count": int(series.isna().sum()),
    }


def _geometry_summary(geometry: list[Any] | pd.Series | None) -> dict[str, Any]:
    summary = _sequence_summary(geometry)
    if geometry is None:
        return {
            **summary,
            "has_xy": False,
            "has_centroid": False,
            "has_topology": False,
        }
    values = [value for value in list(geometry) if value is not None]
    return {
        **summary,
        "has_xy": any(hasattr(value, "x") and hasattr(value, "y") for value in values),
        "has_centroid": any(hasattr(value, "centroid") for value in values),
        "has_topology": any(
            callable(getattr(value, "touches", None))
            or callable(getattr(value, "intersects", None))
            for value in values
        ),
    }


def _target_distribution(model: Any) -> pd.DataFrame:
    target = getattr(model, "target_", None)
    if target is None:
        classes = list(getattr(model.estimator_, "classes_", []))
        return pd.DataFrame({"label": classes, "count": [np.nan] * len(classes)})
    distribution = pd.Series(target).value_counts(dropna=False).sort_index()
    return pd.DataFrame(
        {
            "label": distribution.index.astype(str),
            "count": distribution.to_numpy(dtype=int),
        }
    )


def _split_indices(model: Any, *, row_count: int) -> pd.DataFrame:
    train_indices = getattr(model, "train_indices_", np.array([], dtype=np.int64))
    validation_indices = getattr(
        model, "validation_indices_", np.arange(row_count, dtype=np.int64)
    )
    test_indices = getattr(model, "test_indices_", np.array([], dtype=np.int64))
    rows = [
        {"split": "train", "index": int(index)} for index in np.asarray(train_indices)
    ]
    rows.extend(
        {"split": "validation", "index": int(index)}
        for index in np.asarray(validation_indices)
    )
    rows.extend(
        {"split": "test", "index": int(index)} for index in np.asarray(test_indices)
    )
    return pd.DataFrame(rows, columns=["split", "index"])


def _discretization_bins(model: Any) -> pd.DataFrame:
    rows = []
    for feature, metadata in model.selector_.discretizers_.items():
        for index, value in enumerate(np.asarray(metadata.bins, dtype=float), start=1):
            rows.append(
                {
                    "feature": feature,
                    "method": metadata.method,
                    "bin_index": index,
                    "bin": float(value),
                }
            )
    return pd.DataFrame(rows, columns=["feature", "method", "bin_index", "bin"])


def _edge_table(edge_index: np.ndarray) -> pd.DataFrame:
    edge_array = np.asarray(edge_index, dtype=np.int64)
    return pd.DataFrame(
        {"source": edge_array[0].astype(int), "target": edge_array[1].astype(int)}
    )


def _training_loss(model: Any) -> pd.DataFrame:
    losses = list(getattr(model.estimator_.training_result_, "losses", []))
    return pd.DataFrame(
        {"epoch": list(range(1, len(losses) + 1)), "loss": [float(v) for v in losses]}
    )


def _resolve_result_payload(
    model: Any,
    *,
    raw_frame: pd.DataFrame | None,
    frame: pd.DataFrame,
    geometry: list[Any] | pd.Series | None,
    predictions: pd.Series | list[Any] | np.ndarray | None,
    probabilities: pd.DataFrame | np.ndarray | None,
    simulation_report: Any | None,
    ids: np.ndarray | None,
) -> _ResultPayload:
    if raw_frame is None:
        if predictions is None and probabilities is None:
            raise ValidationError(
                "raw_frame or explicit predictions and probabilities are required."
            )
        if predictions is None or probabilities is None:
            raise ValidationError(
                "explicit predictions and probabilities are required when raw_frame is omitted."
            )
    class_count = len(getattr(model.estimator_, "classes_", []))
    if simulation_report is not None:
        _validate_simulation_report(
            simulation_report,
            expected_rows=None,
            class_count=class_count,
        )
    expected_rows = _expected_result_rows(
        raw_frame=raw_frame,
        frame=frame,
        predictions=predictions,
        probabilities=probabilities,
        simulation_report=simulation_report,
        ids=ids,
    )
    resolved_predictions = _resolve_predictions(
        model,
        frame=frame,
        geometry=geometry,
        predictions=predictions,
        expected_rows=expected_rows,
    )
    resolved_probabilities = _resolve_probabilities(
        model,
        frame=frame,
        geometry=geometry,
        probabilities=probabilities,
        expected_rows=expected_rows,
        class_count=class_count,
    )
    if simulation_report is not None:
        _validate_simulation_report(
            simulation_report,
            expected_rows=expected_rows,
            class_count=class_count,
        )
    if ids is not None:
        _validate_vector_length(ids, expected_rows=expected_rows, name="ids length")
    return _ResultPayload(
        predictions=resolved_predictions,
        probabilities=resolved_probabilities,
        ids=ids,
        simulation_report=simulation_report,
    )


def _resolve_ids(
    ids: pd.Series | list[Any] | np.ndarray | None, *, expected_rows: int | None
) -> np.ndarray | None:
    if ids is None:
        return None
    resolved = np.asarray(ids, dtype=object)
    _validate_vector_length(resolved, expected_rows=expected_rows, name="ids length")
    return resolved


def _expected_result_rows(
    *,
    raw_frame: pd.DataFrame | None,
    frame: pd.DataFrame,
    predictions: pd.Series | list[Any] | np.ndarray | None,
    probabilities: pd.DataFrame | np.ndarray | None,
    simulation_report: Any | None,
    ids: np.ndarray | None,
) -> int | None:
    if raw_frame is not None:
        return len(frame)
    candidates = [
        _optional_len(predictions),
        None if probabilities is None else np.asarray(probabilities).shape[0],
        None if ids is None else len(ids),
    ]
    if simulation_report is not None:
        candidates.append(len(np.asarray(simulation_report.baseline_predictions)))
    present = [value for value in candidates if value is not None]
    if not present:
        return None
    if len(set(present)) != 1:
        raise ValidationError("result artifact row counts must match.")
    _validate_result_size(
        row_count=present[0], class_count=_result_class_count(probabilities)
    )
    return present[0]


def _result_class_count(probabilities: pd.DataFrame | np.ndarray | None) -> int:
    if probabilities is None:
        return 1
    if isinstance(probabilities, pd.DataFrame):
        return max(len(probabilities.columns), 1)
    values = np.asarray(probabilities)
    if values.ndim < 2:
        return 1
    return max(int(values.shape[1]), 1)


def _validate_result_size(*, row_count: int, class_count: int) -> None:
    if row_count > MAX_ARTIFACT_ROWS:
        raise ValidationError("result artifacts exceed artifact export row limits.")
    if row_count * max(class_count, 1) > MAX_ARTIFACT_CELLS:
        raise ValidationError("result artifacts exceed artifact export cell limit.")


def _optional_len(value: Any) -> int | None:
    if value is None:
        return None
    return len(value)


def _resolve_predictions(
    model: Any,
    *,
    frame: pd.DataFrame,
    geometry: list[Any] | pd.Series | None,
    predictions: pd.Series | list[Any] | np.ndarray | None,
    expected_rows: int | None,
) -> np.ndarray:
    if predictions is not None:
        resolved = np.asarray(predictions, dtype=object)
        _validate_vector_length(
            resolved, expected_rows=expected_rows, name="predictions length"
        )
        return resolved
    if frame.empty:
        return np.asarray([], dtype=object)
    try:
        resolved = np.asarray(model.predict(frame, geometry=geometry), dtype=object)
    except (ValidationError, RuntimeError, TypeError, ValueError) as exc:
        raise ValidationError("Could not infer predictions from raw_frame.") from exc
    _validate_vector_length(
        resolved, expected_rows=expected_rows, name="predictions length"
    )
    return resolved


def _resolve_probabilities(
    model: Any,
    *,
    frame: pd.DataFrame,
    geometry: list[Any] | pd.Series | None,
    probabilities: pd.DataFrame | np.ndarray | None,
    expected_rows: int | None,
    class_count: int,
) -> np.ndarray:
    if probabilities is not None:
        if isinstance(probabilities, pd.DataFrame):
            resolved = _probability_array_from_frame(probabilities, model)
        else:
            resolved = np.asarray(probabilities, dtype=float)
        _validate_probability_shape(
            resolved, expected_rows=expected_rows, class_count=class_count
        )
        return resolved
    if frame.empty:
        return np.empty((0, class_count))
    try:
        resolved = np.asarray(
            model.predict_proba(frame, geometry=geometry), dtype=float
        )
    except (ValidationError, RuntimeError, TypeError, ValueError) as exc:
        raise ValidationError("Could not infer probabilities from raw_frame.") from exc
    _validate_probability_shape(
        resolved, expected_rows=expected_rows, class_count=class_count
    )
    return resolved


def _prediction_frame(predictions: np.ndarray, ids: np.ndarray | None) -> pd.DataFrame:
    frame = pd.DataFrame({"prediction": predictions})
    return _with_row_identity(frame, ids)


def _probability_array_from_frame(
    probabilities: pd.DataFrame, model: Any
) -> np.ndarray:
    classes = [str(value) for value in getattr(model.estimator_, "classes_", [])]
    direct_columns = classes
    prefixed_columns = [f"probability_{value}" for value in classes]
    if probabilities.columns.tolist() == direct_columns:
        ordered = probabilities[direct_columns]
    elif probabilities.columns.tolist() == prefixed_columns:
        ordered = probabilities[prefixed_columns]
    else:
        raise ValidationError(
            "probabilities shape must use fitted class columns in order."
        )
    return ordered.to_numpy(dtype=float)


def _probability_frame(
    model: Any, probabilities: np.ndarray, ids: np.ndarray | None
) -> pd.DataFrame:
    classes = [str(value) for value in getattr(model.estimator_, "classes_", [])]
    columns = [f"probability_{value}" for value in classes]
    return _with_row_identity(pd.DataFrame(probabilities, columns=columns), ids)


def _with_row_identity(frame: pd.DataFrame, ids: np.ndarray | None) -> pd.DataFrame:
    identity = pd.DataFrame({"row_index": np.arange(len(frame), dtype=int)})
    if ids is not None:
        identity["id"] = ids
    return pd.concat([identity, frame.reset_index(drop=True)], axis=1)


def _validate_vector_length(
    values: np.ndarray, *, expected_rows: int | None, name: str
) -> None:
    if values.ndim != 1:
        raise ValidationError(f"{name} must be a one-dimensional array.")
    if expected_rows is not None and len(values) != expected_rows:
        raise ValidationError(f"{name} must match raw_frame row count.")


def _validate_probability_shape(
    values: np.ndarray, *, expected_rows: int | None, class_count: int
) -> None:
    if values.ndim != 2:
        raise ValidationError("probabilities shape must be two-dimensional.")
    if expected_rows is not None and values.shape[0] != expected_rows:
        raise ValidationError("probabilities shape must match raw_frame row count.")
    if values.shape[1] != class_count:
        raise ValidationError("probabilities shape must match fitted class count.")
    if not np.isfinite(values).all():
        raise ValidationError("probabilities must contain only finite values.")
    if ((values < 0) | (values > 1)).any():
        raise ValidationError("probabilities must be within [0, 1].")
    if values.size and not np.allclose(values.sum(axis=1), 1.0, atol=1e-6):
        raise ValidationError("probabilities rows must sum to 1.")


def _validate_simulation_report(
    simulation_report: Any,
    *,
    expected_rows: int | None,
    class_count: int,
) -> None:
    try:
        arrays = [
            np.asarray(simulation_report.baseline_predictions),
            np.asarray(simulation_report.scenario_predictions),
            np.asarray(simulation_report.changed_mask),
        ]
        probability_delta = np.asarray(simulation_report.probability_delta)
    except (AttributeError, TypeError) as exc:
        raise ValidationError(
            "simulation_report must expose baseline_predictions, "
            "scenario_predictions, changed_mask, and probability_delta."
        ) from exc
    if any(values.ndim != 1 for values in arrays):
        raise ValidationError(
            "simulation_report arrays must include 1D prediction and change arrays."
        )
    if probability_delta.ndim != 2:
        raise ValidationError(
            "simulation_report arrays must include 2D probability_delta."
        )
    if probability_delta.shape[1] != class_count:
        raise ValidationError(
            "simulation_report probability_delta must match fitted class count."
        )
    if not np.isfinite(probability_delta).all():
        raise ValidationError(
            "simulation_report probability_delta must contain only finite values."
        )
    _validate_result_size(
        row_count=probability_delta.shape[0],
        class_count=probability_delta.shape[1] + 3,
    )
    lengths = [values.shape[0] for values in arrays] + [probability_delta.shape[0]]
    if len(set(lengths)) != 1:
        raise ValidationError("simulation_report arrays must have the same length.")
    if expected_rows is not None and lengths[0] != expected_rows:
        raise ValidationError(
            "simulation_report arrays must match raw_frame row count."
        )


def _simulation_frame(simulation_report: Any, ids: np.ndarray | None) -> pd.DataFrame:
    probability_delta = np.asarray(simulation_report.probability_delta)
    delta_frame = pd.DataFrame(
        probability_delta,
        columns=[
            f"probability_delta_{index}" for index in range(probability_delta.shape[1])
        ],
    )
    frame = pd.concat(
        [
            pd.DataFrame(
                {
                    "baseline_prediction": simulation_report.baseline_predictions,
                    "scenario_prediction": simulation_report.scenario_predictions,
                    "changed": simulation_report.changed_mask,
                }
            ),
            delta_frame,
        ],
        axis=1,
    )
    return _with_row_identity(frame, ids)


def _frame_schema(frame: pd.DataFrame) -> dict[str, Any]:
    return {
        "columns": [
            {"name": str(column), "dtype": str(frame[column].dtype)}
            for column in frame.columns
        ]
    }


def _frame_summary(frame: pd.DataFrame) -> dict[str, Any]:
    return {"row_count": int(len(frame)), "column_count": int(len(frame.columns))}


def _frame_fingerprint(frame: pd.DataFrame) -> dict[str, Any]:
    csv_bytes = _canonical_csv_bytes(frame, index=False)
    return {
        "algorithm": "sha256",
        "value": hashlib.sha256(csv_bytes).hexdigest(),
    }


def _safe_to_csv(frame: pd.DataFrame, path: Path, *, index: bool = False) -> None:
    path.write_bytes(_canonical_csv_bytes(frame, index=index))


def _canonical_csv_bytes(frame: pd.DataFrame, *, index: bool) -> bytes:
    stream = io.StringIO(newline="")
    _sanitize_csv_frame(frame).to_csv(
        stream,
        index=index,
        lineterminator="\n",
    )
    return stream.getvalue().encode("utf-8")


def _sanitize_csv_frame(frame: pd.DataFrame) -> pd.DataFrame:
    sanitized = frame.copy(deep=True)
    sanitized.columns = [_sanitize_csv_value(column) for column in sanitized.columns]
    for column in sanitized.columns:
        sanitized[column] = sanitized[column].map(_sanitize_csv_value)
    return sanitized


def _sanitize_csv_value(value: Any) -> Any:
    if isinstance(value, numbers.Number) and not isinstance(value, bool):
        return value
    if isinstance(value, str):
        text = value
    elif value is None or value is pd.NA or value is pd.NaT:
        return value
    else:
        text = str(value)
    if text.startswith(CSV_FORMULA_PREFIXES):
        return f"'{text}"
    return value


def _validate_provenance(value: Any, *, depth: int = 0, items_seen: int = 0) -> int:
    if depth > MAX_PROVENANCE_DEPTH:
        raise ValidationError("provenance metadata exceeds nesting depth limit.")
    if items_seen > MAX_PROVENANCE_ITEMS:
        raise ValidationError("provenance metadata exceeds item limit.")
    if isinstance(value, dict):
        total = items_seen + len(value)
        for item in value.values():
            total = _validate_provenance(item, depth=depth + 1, items_seen=total)
        return total
    if isinstance(value, (list, tuple)):
        total = items_seen + len(value)
        for item in value:
            total = _validate_provenance(item, depth=depth + 1, items_seen=total)
        return total
    if isinstance(value, np.ndarray):
        if value.size + items_seen > MAX_PROVENANCE_ITEMS:
            raise ValidationError("provenance metadata exceeds item limit.")
        return items_seen + int(value.size)
    return items_seen + 1


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    try:
        encoded = json.dumps(
            _json_safe(payload), indent=2, sort_keys=True, allow_nan=False
        )
    except (TypeError, ValueError) as exc:
        raise ValidationError(
            f"Could not serialize JSON artifact {path.name!r}."
        ) from exc
    path.write_text(encoded, encoding="utf-8")


def _json_safe(value: Any) -> Any:
    if value is None or value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, (datetime, date, pd.Timestamp)):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def _relative_path(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root).as_posix()


def _package_version() -> str | None:
    try:
        return version("gignn")
    except PackageNotFoundError:
        return None
