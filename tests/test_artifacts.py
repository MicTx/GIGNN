import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import gignn.artifacts as artifacts
from gignn import GIGNNAutoClassifier, export_research_artifacts
from gignn.exceptions import NotFittedError, ValidationError


class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.centroid = self


class FormulaLike:
    def __str__(self):
        return "=object_formula"


def _fit_small_auto_model(indicator_frame, labels):
    geometry = [Point(i, 0) for i in range(len(indicator_frame))]
    model = GIGNNAutoClassifier(
        detector_grid={"discretizer": ["quantiles"], "max_bins": [2]},
        graph_grid={"strategy": ["knn"], "k": [1]},
        model_grid={
            "hidden_channels": [(8,)],
            "learning_rate": [0.01],
            "max_epochs": [1],
        },
        random_state=42,
    ).fit(indicator_frame, labels, geometry=geometry)
    return model, geometry


def test_export_research_artifacts_writes_four_layer_bundle(
    indicator_frame, labels, tmp_path
):
    model, geometry = _fit_small_auto_model(indicator_frame, labels)

    bundle = model.export_research_artifacts(
        tmp_path / "run-001",
        raw_frame=indicator_frame,
        geometry=geometry,
        provenance={"study_area": "synthetic"},
    )

    assert bundle.root == (tmp_path / "run-001").resolve()
    assert bundle.manifest_path == bundle.root / "manifest.json"
    assert {"raw", "preprocessing", "process", "results"} <= {
        path.name for path in bundle.root.iterdir() if path.is_dir()
    }
    assert (bundle.root / "README.md").is_file()
    manifest = json.loads(bundle.manifest_path.read_text())
    assert manifest["schema_version"] == "1"
    assert manifest["raw_policy"]["include_raw"] is False
    assert manifest["provenance"] == {"study_area": "synthetic"}
    assert "results/validation_metrics.json" in manifest["files"]
    assert "process/factor_scores.csv" in manifest["files"]
    assert (
        bundle.files["factor_scores"] == bundle.root / "process" / "factor_scores.csv"
    )


def test_export_research_artifacts_keeps_raw_private_by_default(
    indicator_frame, labels, tmp_path
):
    frame = indicator_frame.copy()
    frame.loc[1, "ndvi"] = np.nan
    filled = frame.fillna(0.0)
    model, geometry = _fit_small_auto_model(filled, labels)

    export_research_artifacts(
        model,
        tmp_path / "run-raw-private",
        raw_frame=frame,
        geometry=geometry,
        predictions=["low"] * len(frame),
        probabilities=np.full(
            (len(frame), len(model.estimator_.classes_)),
            1 / len(model.estimator_.classes_),
        ),
        ids=["unit-0", "unit-1", "unit-2", "unit-3", "unit-4", "unit-5"],
    )

    raw_dir = tmp_path / "run-raw-private" / "raw"
    preprocessing_dir = tmp_path / "run-raw-private" / "preprocessing"
    assert not (raw_dir / "raw_frame.csv").exists()
    assert (raw_dir / "input_schema.json").is_file()
    assert (raw_dir / "input_fingerprint.json").is_file()
    summary = json.loads((raw_dir / "input_summary.json").read_text())
    assert summary["row_count"] == len(frame)
    missing = pd.read_csv(preprocessing_dir / "missing_values.csv")
    assert missing.set_index("column").loc["ndvi", "missing_count"] == 1
    feature_summary = pd.read_csv(
        preprocessing_dir / "feature_summary.csv", index_col=0
    )
    assert "mean" in feature_summary.index
    target_distribution = pd.read_csv(preprocessing_dir / "target_distribution.csv")
    assert set(target_distribution["label"]) == set(labels)
    assert target_distribution["count"].sum() == len(labels)
    split_indices = pd.read_csv(preprocessing_dir / "split_indices.csv")
    assert set(split_indices["split"]) == {"train", "validation", "test"}
    selected_columns = json.loads(
        (preprocessing_dir / "selected_columns.json").read_text()
    )
    assert selected_columns["id_summary"] == {
        "present": True,
        "count": len(frame),
        "missing_count": 0,
    }
    assert selected_columns["geometry_summary"] == {
        "present": True,
        "count": len(frame),
        "missing_count": 0,
        "has_xy": True,
        "has_centroid": True,
        "has_topology": False,
    }
    assert selected_columns["target_summary"]["present"] is True
    fill_summary = json.loads((preprocessing_dir / "fill_summary.json").read_text())
    assert fill_summary["imputed_by_exporter"] is False
    assert fill_summary["source_missing_values_file"] == "missing_values.csv"
    assert fill_summary["columns_with_missing_values"] == ["ndvi"]


def test_export_research_artifacts_can_include_raw_frame(
    indicator_frame, labels, tmp_path
):
    model, geometry = _fit_small_auto_model(indicator_frame, labels)

    export_research_artifacts(
        model,
        tmp_path / "run-raw-included",
        raw_frame=indicator_frame,
        geometry=geometry,
        include_raw=True,
    )

    raw_frame = pd.read_csv(tmp_path / "run-raw-included" / "raw" / "raw_frame.csv")
    assert raw_frame.shape == indicator_frame.shape
    assert raw_frame.columns.tolist() == indicator_frame.columns.tolist()


def test_export_fingerprint_matches_written_raw_csv_bytes(
    indicator_frame, labels, tmp_path
):
    model, geometry = _fit_small_auto_model(indicator_frame, labels)
    frame = indicator_frame.copy()
    frame["formula"] = pd.Series(["=raw_formula"] * len(frame), dtype="string")
    root = tmp_path / "fingerprint"

    model.export_research_artifacts(
        root,
        raw_frame=frame,
        geometry=geometry,
        include_raw=True,
        predictions=["low"] * len(frame),
        probabilities=np.full(
            (len(frame), len(model.estimator_.classes_)),
            1 / len(model.estimator_.classes_),
        ),
    )

    fingerprint = json.loads((root / "raw" / "input_fingerprint.json").read_text())
    raw_csv = (root / "raw" / "raw_frame.csv").read_bytes()
    assert fingerprint == {
        "algorithm": "sha256",
        "value": hashlib.sha256(raw_csv).hexdigest(),
    }


def test_export_research_artifacts_writes_process_and_result_tables(
    indicator_frame, labels, tmp_path
):
    model, geometry = _fit_small_auto_model(indicator_frame, labels)

    model.export_research_artifacts(
        tmp_path / "run-process-results",
        raw_frame=indicator_frame,
        geometry=geometry,
        ids=[f"unit-{index}" for index in range(len(indicator_frame))],
    )

    root = tmp_path / "run-process-results"
    factor_scores = pd.read_csv(root / "process" / "factor_scores.csv")
    interaction_scores = pd.read_csv(root / "process" / "interaction_scores.csv")
    discretization_bins = pd.read_csv(root / "process" / "discretization_bins.csv")
    graph_candidates = pd.read_csv(root / "process" / "graph_candidates.csv")
    selected_edges = pd.read_csv(root / "process" / "selected_edges.csv")
    model_candidates = pd.read_csv(root / "process" / "model_candidates.csv")
    training_loss = pd.read_csv(root / "process" / "training_loss.csv")
    predictions = pd.read_csv(root / "results" / "predictions.csv")
    probabilities = pd.read_csv(root / "results" / "probabilities.csv")
    metrics = json.loads((root / "results" / "metrics.json").read_text())
    validation_metrics = json.loads(
        (root / "results" / "validation_metrics.json").read_text()
    )
    per_class = pd.read_csv(root / "results" / "per_class_report.csv")

    assert {"factor", "q", "p_value"} <= set(factor_scores.columns)
    assert {"factor_1", "factor_2", "q_interaction", "interaction"} <= set(
        interaction_scores.columns
    )
    assert {"feature", "method", "bin_index", "bin"} <= set(discretization_bins.columns)
    assert {"strategy", "num_edges", "validation_accuracy"} <= set(
        graph_candidates.columns
    )
    assert {"source", "target"} <= set(selected_edges.columns)
    assert {"validation_accuracy", "hidden_channels", "learning_rate"} <= set(
        model_candidates.columns
    )
    assert training_loss["loss"].notna().all()
    assert len(predictions) == len(indicator_frame)
    assert predictions.columns[:2].tolist() == ["row_index", "id"]
    assert predictions["id"].iloc[0] == "unit-0"
    assert probabilities.shape[0] == len(indicator_frame)
    assert probabilities.columns[:2].tolist() == ["row_index", "id"]
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert 0.0 <= validation_metrics["accuracy"] <= 1.0
    assert "precision" in per_class.columns


def test_loaded_auto_classifier_can_export_research_artifacts(
    indicator_frame, labels, tmp_path
):
    model, geometry = _fit_small_auto_model(indicator_frame, labels)
    checkpoint = tmp_path / "model.gignn"
    model.save(checkpoint)

    loaded = GIGNNAutoClassifier.load(checkpoint)
    loaded.export_research_artifacts(
        tmp_path / "loaded-model-artifacts",
        raw_frame=indicator_frame,
        geometry=geometry,
    )

    training_loss = pd.read_csv(
        tmp_path / "loaded-model-artifacts" / "process" / "training_loss.csv"
    )
    assert training_loss["loss"].tolist() == pytest.approx(
        model.estimator_.training_result_.losses
    )


def test_export_research_artifacts_rejects_unfitted_model(tmp_path):
    model = GIGNNAutoClassifier()

    with pytest.raises(NotFittedError):
        model.export_research_artifacts(tmp_path / "unfitted")


def test_export_research_artifacts_rejects_broken_symlink_output(
    indicator_frame, labels, tmp_path, symlink_supported
):
    model, geometry = _fit_small_auto_model(indicator_frame, labels)
    symlink_path = tmp_path / "broken-link"
    symlink_path.symlink_to(tmp_path / "missing-target")

    with pytest.raises(ValidationError, match="must not be a symlink"):
        model.export_research_artifacts(
            symlink_path,
            raw_frame=indicator_frame,
            geometry=geometry,
        )


def test_export_research_artifacts_limits_result_only_rows(
    indicator_frame, labels, tmp_path, monkeypatch
):
    model, _ = _fit_small_auto_model(indicator_frame, labels)
    monkeypatch.setattr(artifacts, "MAX_ARTIFACT_ROWS", 2)

    with pytest.raises(ValidationError, match="result artifacts exceed"):
        model.export_research_artifacts(
            tmp_path / "too-many-results",
            predictions=["low"] * len(indicator_frame),
            probabilities=np.full(
                (len(indicator_frame), len(model.estimator_.classes_)),
                1 / len(model.estimator_.classes_),
            ),
        )


def test_export_research_artifacts_limits_provenance_depth(
    indicator_frame, labels, tmp_path, monkeypatch
):
    model, geometry = _fit_small_auto_model(indicator_frame, labels)
    monkeypatch.setattr(artifacts, "MAX_PROVENANCE_DEPTH", 1)

    with pytest.raises(ValidationError, match="provenance metadata exceeds"):
        model.export_research_artifacts(
            tmp_path / "deep-provenance",
            raw_frame=indicator_frame,
            geometry=geometry,
            provenance={"nested": {"too": "deep"}},
        )

    model, geometry = _fit_small_auto_model(indicator_frame, labels)
    output_dir = tmp_path / "existing"
    output_dir.mkdir()
    (output_dir / "keep.txt").write_text("existing")

    with pytest.raises(ValidationError, match="already exists"):
        model.export_research_artifacts(
            output_dir,
            raw_frame=indicator_frame,
            geometry=geometry,
            overwrite=False,
        )


def test_export_research_artifacts_rejects_overwrite_of_non_bundle(
    indicator_frame, labels, tmp_path
):
    model, geometry = _fit_small_auto_model(indicator_frame, labels)
    output_dir = tmp_path / "project"
    (output_dir / "results").mkdir(parents=True)
    (output_dir / "results" / "keep.csv").write_text("do not delete")

    with pytest.raises(ValidationError, match="existing GIGNN artifact bundle"):
        model.export_research_artifacts(
            output_dir,
            raw_frame=indicator_frame,
            geometry=geometry,
            overwrite=True,
        )

    assert (output_dir / "results" / "keep.csv").is_file()


def test_export_research_artifacts_overwrite_removes_stale_raw_frame(
    indicator_frame, labels, tmp_path
):
    model, geometry = _fit_small_auto_model(indicator_frame, labels)
    output_dir = tmp_path / "rerun"
    model.export_research_artifacts(
        output_dir,
        raw_frame=indicator_frame,
        geometry=geometry,
        include_raw=True,
    )
    assert (output_dir / "raw" / "raw_frame.csv").is_file()

    model.export_research_artifacts(
        output_dir,
        raw_frame=indicator_frame,
        geometry=geometry,
        include_raw=False,
        overwrite=True,
    )

    assert not (output_dir / "raw" / "raw_frame.csv").exists()
    manifest = json.loads((output_dir / "manifest.json").read_text())
    assert manifest["raw_policy"]["include_raw"] is False


def test_export_research_artifacts_rejects_incompatible_raw_frame_for_inference(
    indicator_frame, labels, tmp_path
):
    model, geometry = _fit_small_auto_model(indicator_frame, labels)
    bad_frame = indicator_frame.copy()
    bad_frame.loc[0, "ndvi"] = np.nan

    with pytest.raises(ValidationError, match="Could not infer predictions"):
        model.export_research_artifacts(
            tmp_path / "bad-inference",
            raw_frame=bad_frame,
            geometry=geometry,
        )


@pytest.mark.parametrize("error_type", [RuntimeError, TypeError, ValueError])
def test_export_research_artifacts_translates_prediction_errors(
    indicator_frame, labels, tmp_path, monkeypatch, error_type
):
    model, geometry = _fit_small_auto_model(indicator_frame, labels)

    def fail_predict(*args, **kwargs):
        raise error_type("predict failed")

    monkeypatch.setattr(model, "predict", fail_predict)

    with pytest.raises(ValidationError, match="Could not infer predictions"):
        model.export_research_artifacts(
            tmp_path / f"prediction-{error_type.__name__}",
            raw_frame=indicator_frame,
            geometry=geometry,
        )


@pytest.mark.parametrize("error_type", [RuntimeError, TypeError, ValueError])
def test_export_research_artifacts_translates_probability_errors(
    indicator_frame, labels, tmp_path, monkeypatch, error_type
):
    model, geometry = _fit_small_auto_model(indicator_frame, labels)

    def fail_predict_proba(*args, **kwargs):
        raise error_type("predict_proba failed")

    monkeypatch.setattr(model, "predict_proba", fail_predict_proba)

    with pytest.raises(ValidationError, match="Could not infer probabilities"):
        model.export_research_artifacts(
            tmp_path / f"probability-{error_type.__name__}",
            raw_frame=indicator_frame,
            geometry=geometry,
            predictions=["low"] * len(indicator_frame),
        )


def test_export_research_artifacts_validates_result_shapes(
    indicator_frame, labels, tmp_path
):
    model, geometry = _fit_small_auto_model(indicator_frame, labels)

    with pytest.raises(ValidationError, match="predictions length"):
        model.export_research_artifacts(
            tmp_path / "bad-predictions",
            raw_frame=indicator_frame,
            geometry=geometry,
            predictions=["low"],
        )

    with pytest.raises(ValidationError, match="probabilities shape"):
        model.export_research_artifacts(
            tmp_path / "bad-probabilities",
            raw_frame=indicator_frame,
            geometry=geometry,
            probabilities=np.zeros((len(indicator_frame), 1)),
        )

    classes = [str(value) for value in model.estimator_.classes_]
    reversed_probabilities = pd.DataFrame(
        np.zeros((len(indicator_frame), len(classes))),
        columns=list(reversed(classes)),
    )
    with pytest.raises(ValidationError, match="probabilities shape"):
        model.export_research_artifacts(
            tmp_path / "bad-probability-columns",
            raw_frame=indicator_frame,
            geometry=geometry,
            probabilities=reversed_probabilities,
        )


def test_export_research_artifacts_writes_simulation_report(
    indicator_frame, labels, tmp_path
):
    model, geometry = _fit_small_auto_model(indicator_frame, labels)
    simulation = model.simulate(
        indicator_frame, baseline_X=indicator_frame, geometry=geometry
    )

    model.export_research_artifacts(
        tmp_path / "run-simulation",
        raw_frame=indicator_frame,
        geometry=geometry,
        simulation_report=simulation,
    )

    simulation_frame = pd.read_csv(
        tmp_path / "run-simulation" / "results" / "simulation_report.csv"
    )
    assert {"baseline_prediction", "scenario_prediction", "changed"} <= set(
        simulation_frame.columns
    )
    assert any(column.startswith("probability_delta_") for column in simulation_frame)


def test_export_research_artifacts_validates_simulation_shape(
    indicator_frame, labels, tmp_path
):
    model, geometry = _fit_small_auto_model(indicator_frame, labels)
    simulation = model.simulate(
        indicator_frame, baseline_X=indicator_frame, geometry=geometry
    )
    broken = simulation.__class__(
        baseline_predictions=simulation.baseline_predictions[:-1],
        scenario_predictions=simulation.scenario_predictions,
        baseline_probabilities=simulation.baseline_probabilities,
        scenario_probabilities=simulation.scenario_probabilities,
        probability_delta=simulation.probability_delta,
        changed_mask=simulation.changed_mask,
    )

    with pytest.raises(ValidationError, match="simulation_report arrays"):
        model.export_research_artifacts(
            tmp_path / "bad-simulation",
            raw_frame=indicator_frame,
            geometry=geometry,
            simulation_report=broken,
        )


def test_export_research_artifacts_rejects_scalar_simulation_arrays(
    indicator_frame, labels, tmp_path
):
    model, geometry = _fit_small_auto_model(indicator_frame, labels)
    broken = SimpleNamespace(
        baseline_predictions=np.asarray("low", dtype=object),
        scenario_predictions=np.asarray("low", dtype=object),
        changed_mask=np.asarray(False),
        probability_delta=np.zeros((1, len(model.estimator_.classes_))),
    )

    with pytest.raises(ValidationError, match="simulation_report arrays"):
        model.export_research_artifacts(
            tmp_path / "scalar-simulation",
            raw_frame=indicator_frame,
            geometry=geometry,
            simulation_report=broken,
        )


@pytest.mark.parametrize(
    "probability_delta",
    [
        np.full((6, 3), np.nan),
        np.zeros((6, 10_001)),
    ],
)
def test_export_research_artifacts_rejects_invalid_simulation_probability_delta(
    indicator_frame, labels, tmp_path, probability_delta
):
    model, geometry = _fit_small_auto_model(indicator_frame, labels)
    broken = SimpleNamespace(
        baseline_predictions=np.asarray(["low"] * 6, dtype=object),
        scenario_predictions=np.asarray(["low"] * 6, dtype=object),
        changed_mask=np.zeros(6, dtype=bool),
        probability_delta=probability_delta,
    )

    with pytest.raises(
        ValidationError,
        match="probability_delta|class count|artifact.*limit|finite",
    ):
        model.export_research_artifacts(
            tmp_path / "invalid-simulation-delta",
            raw_frame=indicator_frame,
            geometry=geometry,
            simulation_report=broken,
        )


def test_export_research_artifacts_sanitizes_scientific_json_metadata(
    indicator_frame, labels, tmp_path
):
    model, geometry = _fit_small_auto_model(indicator_frame, labels)

    model.export_research_artifacts(
        tmp_path / "metadata",
        raw_frame=indicator_frame,
        geometry=geometry,
        provenance={
            "timestamp": pd.Timestamp("2026-01-02"),
            "missing": pd.NA,
            "non_finite": np.nan,
        },
    )

    manifest = json.loads((tmp_path / "metadata" / "manifest.json").read_text())
    assert manifest["provenance"] == {
        "missing": None,
        "non_finite": None,
        "timestamp": "2026-01-02T00:00:00",
    }


def test_export_research_artifacts_escapes_spreadsheet_formulas(
    indicator_frame, labels, tmp_path
):
    model, geometry = _fit_small_auto_model(indicator_frame, labels)

    model.export_research_artifacts(
        tmp_path / "safe-csv",
        raw_frame=indicator_frame,
        geometry=geometry,
        ids=["=cmd"] + [f"unit-{index}" for index in range(1, len(indicator_frame))],
        predictions=[FormulaLike()] + ["low"] * (len(indicator_frame) - 1),
        probabilities=np.full(
            (len(indicator_frame), len(model.estimator_.classes_)),
            1 / len(model.estimator_.classes_),
        ),
    )

    predictions = pd.read_csv(tmp_path / "safe-csv" / "results" / "predictions.csv")
    assert predictions.loc[0, "id"] == "'=cmd"
    assert predictions.loc[0, "prediction"] == "'=object_formula"


def test_export_research_artifacts_escapes_string_dtype_raw_columns(
    indicator_frame, labels, tmp_path
):
    model, geometry = _fit_small_auto_model(indicator_frame, labels)
    frame = indicator_frame.copy()
    frame["formula"] = pd.Series(["=raw_formula"] * len(frame), dtype="string")

    model.export_research_artifacts(
        tmp_path / "safe-raw-csv",
        raw_frame=frame,
        geometry=geometry,
        include_raw=True,
        predictions=["low"] * len(frame),
        probabilities=np.full(
            (len(frame), len(model.estimator_.classes_)),
            1 / len(model.estimator_.classes_),
        ),
    )

    raw_frame = pd.read_csv(tmp_path / "safe-raw-csv" / "raw" / "raw_frame.csv")
    assert raw_frame.loc[0, "formula"] == "'=raw_formula"


def test_export_research_artifacts_requires_result_source_when_raw_frame_omitted(
    indicator_frame, labels, tmp_path
):
    model, _ = _fit_small_auto_model(indicator_frame, labels)

    with pytest.raises(ValidationError, match="raw_frame or explicit predictions"):
        model.export_research_artifacts(tmp_path / "empty-results")
    with pytest.raises(ValidationError, match="explicit predictions and probabilities"):
        model.export_research_artifacts(
            tmp_path / "missing-probabilities",
            predictions=["low"] * len(indicator_frame),
        )
    with pytest.raises(ValidationError, match="explicit predictions and probabilities"):
        model.export_research_artifacts(
            tmp_path / "missing-predictions",
            probabilities=np.full(
                (len(indicator_frame), len(model.estimator_.classes_)),
                1 / len(model.estimator_.classes_),
            ),
        )


def test_export_research_artifacts_validates_without_raw_frame(
    indicator_frame, labels, tmp_path
):
    model, _ = _fit_small_auto_model(indicator_frame, labels)

    with pytest.raises(ValidationError, match="result artifact row counts"):
        model.export_research_artifacts(
            tmp_path / "no-raw-mismatch",
            predictions=["a", "b"],
            probabilities=np.full((3, len(model.estimator_.classes_)), 0.5),
        )


def test_export_research_artifacts_rejects_missing_raw_feature_with_explicit_outputs(
    indicator_frame, labels, tmp_path
):
    model, geometry = _fit_small_auto_model(indicator_frame, labels)
    bad_frame = indicator_frame.drop(columns=["ndvi"])

    with pytest.raises(ValidationError, match="missing fitted feature columns"):
        model.export_research_artifacts(
            tmp_path / "missing-feature",
            raw_frame=bad_frame,
            geometry=geometry,
            predictions=["low"] * len(bad_frame),
            probabilities=np.full(
                (len(bad_frame), len(model.estimator_.classes_)),
                1 / len(model.estimator_.classes_),
            ),
        )


def test_export_research_artifacts_rejects_invalid_probability_values(
    indicator_frame, labels, tmp_path
):
    model, geometry = _fit_small_auto_model(indicator_frame, labels)
    bad_probabilities = np.full(
        (len(indicator_frame), len(model.estimator_.classes_)),
        0.5,
    )

    with pytest.raises(ValidationError, match="probabilities rows must sum"):
        model.export_research_artifacts(
            tmp_path / "bad-probability-values",
            raw_frame=indicator_frame,
            geometry=geometry,
            predictions=["low"] * len(indicator_frame),
            probabilities=bad_probabilities,
        )
