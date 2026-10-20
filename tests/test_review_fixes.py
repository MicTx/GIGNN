import os

import numpy as np
import pandas as pd
import pytest
import torch

from gignn import GIGNNClassifier
from gignn.exceptions import ValidationError
from gignn.generalization import standardize_spatial_dataset
from gignn.geodetector.discretization import spatial_discretization
from gignn.geodetector.factor import (
    factor_detector,
    geodetector_q,
    target_to_numeric,
)
from gignn.geodetector.interaction import _classify_interaction, interaction_detector
from gignn.geodetector.selector import GeoDetectorSelector
from gignn.graph import build_spatial_edges
from gignn.models import GraphSAGEClassifier, multihop_sampling
from gignn.models.gignn import NeighborAggregator
from gignn.ordinal import fit_ordinal_target
from gignn.plotting import save_publication_figure
from gignn.splitting import resolve_split_indices
from gignn.training import TrainingConfig, train_node_classifier
from gignn.validation import LabelEncoder, validate_feature_table

pytestmark = pytest.mark.filterwarnings("ignore::gignn.exceptions.EdgeWeightWarning")

# ---------------------------------------------------------------------------
# T1: interaction classification formula
# ---------------------------------------------------------------------------


def test_classify_interaction_independent_when_q12_equals_sum():
    # Standard definition: independent iff q12 == q1 + q2 (previously the sum
    # case fell through to bi_enhance).
    assert _classify_interaction(0.3, 0.2, 0.5) == "independent"


def test_classify_interaction_nonlinear_enhance_when_q12_exceeds_sum():
    assert _classify_interaction(0.3, 0.2, 0.7) == "nonlinear_enhance"


def test_classify_interaction_bi_enhance_between_max_and_sum():
    assert _classify_interaction(0.3, 0.2, 0.4) == "bi_enhance"


def test_classify_interaction_weaken_when_q12_at_or_below_max():
    # A fully redundant second factor leaves q12 == max(q1, q2); it must not be
    # reported as independent.
    assert _classify_interaction(0.5, 0.5, 0.5) == "weaken"
    assert _classify_interaction(0.5, 0.2, 0.3) == "weaken"


def test_interaction_detector_redundant_factor_not_independent():
    frame = pd.DataFrame(
        {
            "Y": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            "a": [1, 1, 2, 2, 3, 3, 4, 4],
        }
    )
    frame["b"] = frame["a"]
    result = interaction_detector(frame, y="Y", factors=["a", "b"])
    assert result.loc[0, "interaction"] != "independent"


def test_interaction_detector_rejects_missing_target_column():
    frame = pd.DataFrame({"Y": [1, 2, 3], "a": [1, 2, 3], "b": [3, 2, 1]})
    with pytest.raises(ValidationError, match="Target column"):
        interaction_detector(frame, y="missing", factors=["a", "b"])


def test_interaction_detector_rejects_missing_factor_column():
    frame = pd.DataFrame({"Y": [1, 2, 3], "a": [1, 2, 3]})
    with pytest.raises(ValidationError, match="Factor column"):
        interaction_detector(frame, y="Y", factors=["a", "missing"])


# ---------------------------------------------------------------------------
# T2: training reproducibility + trainer eval mode
# ---------------------------------------------------------------------------


def test_gignn_classifier_reproducible_across_repeated_fits(
    indicator_frame, edge_index
):
    pytest.importorskip("torch")
    pytest.importorskip("torch_geometric")
    labels = [0, 1, 1, 2, 0, 2]

    first = GIGNNClassifier(hidden_channels=(8,), max_epochs=3, random_state=42)
    first.fit(indicator_frame, labels, edge_index=edge_index)
    first_proba = first.predict_proba(indicator_frame, edge_index=edge_index)

    second = GIGNNClassifier(hidden_channels=(8,), max_epochs=3, random_state=42)
    second.fit(indicator_frame, labels, edge_index=edge_index)
    second_proba = second.predict_proba(indicator_frame, edge_index=edge_index)

    np.testing.assert_allclose(first_proba, second_proba)


def test_train_node_classifier_returns_model_in_eval_mode(
    indicator_frame, labels, edge_index
):
    pytest.importorskip("torch")
    pytest.importorskip("torch_geometric")
    from gignn.graph import build_pyg_data

    data = build_pyg_data(indicator_frame, labels, edge_index)
    model = GraphSAGEClassifier(
        in_channels=data.num_features, hidden_channels=(8,), out_channels=3
    )

    train_node_classifier(model, data, TrainingConfig(max_epochs=1, random_state=42))

    assert model.training is False


# ---------------------------------------------------------------------------
# T3: score() with numeric labels
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "labels",
    [
        [0, 1, 1, 2, 0, 2],
        [0.0, 1.0, 1.0, 2.0, 0.0, 2.0],
        ["low", "medium", "medium", "high", "low", "high"],
    ],
    ids=["int", "float", "string"],
)
def test_gignn_classifier_score_with_numeric_labels(
    indicator_frame, edge_index, labels
):
    pytest.importorskip("torch")
    pytest.importorskip("torch_geometric")
    model = GIGNNClassifier(hidden_channels=(8,), max_epochs=1, random_state=42)
    model.fit(indicator_frame, labels, edge_index=edge_index)

    score = model.score(indicator_frame, labels, edge_index=edge_index)

    assert 0.0 <= score <= 1.0


def test_gignn_classifier_score_rejects_label_length_mismatch(
    indicator_frame, labels, edge_index
):
    pytest.importorskip("torch")
    pytest.importorskip("torch_geometric")
    model = GIGNNClassifier(hidden_channels=(8,), max_epochs=1, random_state=42)
    model.fit(indicator_frame, labels, edge_index=edge_index)

    with pytest.raises(ValidationError, match="one label per prediction"):
        model.score(indicator_frame, ["low", "medium"], edge_index=edge_index)


# ---------------------------------------------------------------------------
# T4: edge weight handling
# ---------------------------------------------------------------------------


def test_graphsage_rejects_edge_weight(indicator_frame, labels, edge_index):
    pytest.importorskip("torch")
    pytest.importorskip("torch_geometric")
    from gignn.graph import build_pyg_data

    data = build_pyg_data(
        indicator_frame, labels, edge_index, edge_weight=[1.0] * edge_index.shape[1]
    )
    model = GraphSAGEClassifier(
        in_channels=data.num_features, hidden_channels=(8,), out_channels=3
    )

    with pytest.raises(NotImplementedError, match="edge_weight"):
        model(data.x, data.edge_index, edge_weight=data.edge_weight)


def test_gignn_rejects_mismatched_edge_weight_length(indicator_frame):
    pytest.importorskip("torch")
    from gignn.models import GIGNN

    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long)
    model = GIGNN(
        in_channels=2, hidden_channels=(4,), out_channels=2, num_neighbors_list=(2, 2)
    )
    x = torch.randn(4, 2)

    with pytest.raises(ValueError, match="one value per edge"):
        model(x, edge_index, edge_weight=torch.tensor([0.5, 0.5]))


def test_multihop_sampling_accepts_numpy_integers():
    adjacency = {0: [1, 2], 1: [0], 2: [0]}

    sampled = multihop_sampling([0, 1], np.array([2, 3]), adjacency)

    assert [nodes.shape[0] for nodes in sampled] == [2, 4, 12]


def test_multihop_sampling_rejects_bool_sample_num():
    adjacency = {0: [1], 1: [0]}
    with pytest.raises(ValueError, match="positive integers"):
        multihop_sampling([0], [True], adjacency)


def test_weighted_max_aggregation_selects_high_weight_neighbor():
    aggregator = NeighborAggregator(1, 1, aggr_method="max")
    with torch.no_grad():
        aggregator.weight.copy_(torch.ones(1, 1))
    neighbor_feature = torch.tensor([[[-1.0], [-2.0]]])
    weights = torch.tensor([[1.0, 0.4]])

    output = aggregator(neighbor_feature, weights)

    # The high-weight neighbor's raw feature (-1.0) wins, not the low-weight
    # neighbor's scaled feature (-2.0 * 0.4 = -0.8).
    assert output[0, 0].item() == pytest.approx(-1.0)


# ---------------------------------------------------------------------------
# T5: geodetector input robustness
# ---------------------------------------------------------------------------


def test_geodetector_q_rejects_nan_factor():
    y = np.array([1.0, 2.0, 3.0])
    factor = np.array([1.0, np.nan, 2.0])
    with pytest.raises(ValidationError, match="factor"):
        geodetector_q(y, factor)


def test_geodetector_q_rejects_empty_factor():
    with pytest.raises(ValidationError, match="at least one value"):
        geodetector_q(np.array([]), np.array([]))


def test_factor_detector_empty_factor_list_returns_typed_frame():
    frame = pd.DataFrame({"Y": [1, 2, 3], "x": [1, 2, 3]})
    result = factor_detector(frame, "Y", [])
    assert list(result.columns) == ["factor", "q", "p_value"]
    assert len(result) == 0


def test_spatial_discretization_rejects_empty_column():
    frame = pd.DataFrame({"a": pd.Series([], dtype=float)})
    with pytest.raises(ValidationError, match="empty"):
        spatial_discretization(frame, ["a"])


def test_selector_rejects_reserved_target_column():
    X = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0], "__gignn_target__": [0, 1, 0, 1]})
    y = pd.Series([0, 1, 0, 1])
    with pytest.raises(ValidationError, match="reserved column"):
        GeoDetectorSelector().fit(X, y)


def test_target_to_numeric_rejects_missing_values():
    with pytest.raises(ValidationError, match="missing values"):
        target_to_numeric(pd.Series([1.0, np.nan, 2.0]))


# ---------------------------------------------------------------------------
# T6: graph builders error type
# ---------------------------------------------------------------------------


def test_build_spatial_edges_rejects_tuple_geometry_with_validation_error():
    with pytest.raises(ValidationError, match="numeric x/y"):
        build_spatial_edges(
            geometry=[(0, 0), (1, 0), (2, 0)], num_nodes=3, strategy="knn", k=1
        )


# ---------------------------------------------------------------------------
# T7: splitting guards
# ---------------------------------------------------------------------------


def test_resolve_split_indices_rejects_empty_train_split():
    y = pd.Series([0, 1] * 5)
    with pytest.raises(ValidationError, match="no training rows"):
        resolve_split_indices(
            y, validation_fraction=0.75, test_fraction=0.2, random_state=0
        )


def test_resolve_split_indices_rejects_unhashable_spatial_groups():
    y = pd.Series([0, 1, 0, 1])
    with pytest.raises(ValidationError, match="hashable"):
        resolve_split_indices(
            y,
            validation_fraction=0.2,
            test_fraction=0.2,
            random_state=0,
            spatial_groups=[["a"], ["b"], ["c"], ["d"]],
        )


# ---------------------------------------------------------------------------
# T8: estimator validation asymmetry
# ---------------------------------------------------------------------------


def test_validate_feature_table_rejects_duplicate_columns():
    frame = pd.DataFrame([[1.0, 2.0]], columns=["a", "a"])
    with pytest.raises(ValidationError, match="duplicate column names"):
        validate_feature_table(frame)


def test_validate_feature_table_rejects_non_string_columns():
    frame = pd.DataFrame([[1.0, 2.0]], columns=[0, 1])
    with pytest.raises(ValidationError, match="column names must be strings"):
        validate_feature_table(frame)


def test_gignn_classifier_fit_rejects_non_string_columns(edge_index):
    frame = pd.DataFrame([[0.1, 1.0]] * 6, columns=[0, 1])
    labels = [0, 1, 1, 0, 1, 0]
    model = GIGNNClassifier(hidden_channels=(8,), max_epochs=1, random_state=42)
    with pytest.raises(ValidationError, match="column names must be strings"):
        model.fit(frame, labels, edge_index=edge_index[:, :2])


def test_gignn_classifier_fit_rejects_invalid_device(
    indicator_frame, labels, edge_index
):
    model = GIGNNClassifier(hidden_channels=(8,), max_epochs=1, device="not-a-device")
    with pytest.raises(ValidationError, match="device is invalid"):
        model.fit(indicator_frame, labels, edge_index=edge_index)


def test_checkpoint_params_accept_cuda_index_device(tmp_path):
    path = tmp_path / "cuda-device.pt"
    torch.save(
        {
            "params": {
                "hidden_channels": [8],
                "graph_builder": "knn",
                "k_neighbors": 8,
                "distance_threshold": None,
                "max_epochs": 1,
                "learning_rate": 0.01,
                "weight_decay": 0.0,
                "dropout": 0.0,
                "random_state": 42,
                "device": "cuda:0",
            },
            "state_dict": {"weight": torch.ones(1)},
            "classes": ["a", "b"],
            "n_features_in": 3,
            "feature_names_in": ["ndvi", "dem", "road_density"],
        },
        path,
    )
    # The checkpoint schema validation proceeds past the device field; reaching
    # a later state_dict error proves cuda:0 is accepted as a device.
    with pytest.raises(ValidationError, match="state_dict"):
        GIGNNClassifier.load(path)


def test_checkpoint_params_reject_bool_k_neighbors(tmp_path):
    path = tmp_path / "bool-kneighbors.pt"
    torch.save(
        {
            "params": {
                "hidden_channels": [8],
                "graph_builder": "knn",
                "k_neighbors": True,
                "distance_threshold": None,
                "max_epochs": 1,
                "learning_rate": 0.01,
                "weight_decay": 0.0,
                "dropout": 0.0,
                "random_state": 42,
                "device": "cpu",
            },
            "state_dict": {"weight": torch.ones(1)},
            "classes": ["a", "b"],
            "n_features_in": 3,
            "feature_names_in": ["ndvi", "dem", "road_density"],
        },
        path,
    )
    with pytest.raises(ValidationError, match="k_neighbors"):
        GIGNNClassifier.load(path)


def test_label_encoder_rejects_unhashable_labels():
    encoder = LabelEncoder()
    with pytest.raises(ValidationError, match="hashable"):
        encoder.fit([[1, 2], [3, 4]])


def test_fit_ordinal_target_rejects_complex_labels():
    with pytest.raises(ValidationError, match="real numbers"):
        fit_ordinal_target(pd.Series([1 + 2j, 2 + 3j]), class_order=None)


def test_fit_ordinal_target_rejects_infinite_labels():
    with pytest.raises(ValidationError, match="finite"):
        fit_ordinal_target(pd.Series([1.0, float("inf")]), class_order=None)


# ---------------------------------------------------------------------------
# T9: checkpoint write safety
# ---------------------------------------------------------------------------


def test_save_checkpoint_is_atomic_on_failure(tmp_path):
    from gignn.io import load_checkpoint, save_checkpoint

    path = tmp_path / "model.pt"
    save_checkpoint(path, {"value": torch.ones(4)})

    class Unpicklable:
        def __reduce__(self):
            raise RuntimeError("boom")

    with pytest.raises(Exception):
        save_checkpoint(path, {"value": torch.ones(4), "bad": Unpicklable()})

    recovered = load_checkpoint(path)
    torch.testing.assert_close(recovered["value"], torch.ones(4))


def test_save_checkpoint_rejects_symlink_path(tmp_path, symlink_supported):
    from gignn.io import save_checkpoint

    target = tmp_path / "victim.bin"
    target.write_bytes(b"original")
    link = tmp_path / "model.pt"
    os.symlink(target, link)

    with pytest.raises(ValidationError, match="symlink"):
        save_checkpoint(link, {"value": torch.ones(1)})
    assert target.read_bytes() == b"original"


# ---------------------------------------------------------------------------
# T10: report / artifacts / plotting robustness
# ---------------------------------------------------------------------------


def test_standardize_spatial_dataset_honors_falsy_integer_column_names():
    frame = pd.DataFrame({0: ["a", "b", "c"], 1: [1, 0, 1], 2: [0.5, 0.6, 0.7]})
    dataset, report = standardize_spatial_dataset(frame, target_column=1, id_column=0)
    assert dataset.ids is not None
    assert dataset.ids.tolist() == ["a", "b", "c"]
    assert report.id_column == 0


def test_standardize_spatial_dataset_rejects_non_numeric_feature():
    frame = pd.DataFrame({"target": [0, 1], "feature": ["x", "y"]})
    with pytest.raises(ValidationError, match="numeric"):
        standardize_spatial_dataset(frame, target_column="target")


def test_is_managed_bundle_handles_non_dict_manifest(tmp_path):
    from gignn.artifacts import _is_managed_bundle

    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "manifest.json").write_text("[1, 2, 3]", encoding="utf-8")

    assert _is_managed_bundle(bundle) is False


def test_save_publication_figure_rejects_empty_formats(tmp_path):
    matplotlib = pytest.importorskip("matplotlib")
    import matplotlib.pyplot as plt

    figure, _ = plt.subplots()
    with pytest.raises(ValidationError, match="at least one format"):
        save_publication_figure(figure, tmp_path / "fig", formats=())
    plt.close(figure)


# ---------------------------------------------------------------------------
# Round-2 fixes
# ---------------------------------------------------------------------------


# -- T1: _validate_frame_payload boolean ordering ----------------------------


def test_validate_frame_payload_rejects_non_list_rows_without_type_error():
    """rows=5 (non-list) must raise ValidationError, not raw TypeError."""
    from gignn.auto_persistence import _validate_frame_payload

    payload = {
        "columns": ["a"],
        "index": [0],
        "index_name": None,
        "rows": 5,
    }
    with pytest.raises(ValidationError, match="table is invalid"):
        _validate_frame_payload(payload)


def test_validate_frame_payload_rejects_non_list_index_without_type_error():
    """index=5 (non-list) must raise ValidationError, not raw TypeError."""
    from gignn.auto_persistence import _validate_frame_payload

    payload = {
        "columns": ["a"],
        "index": 5,
        "index_name": None,
        "rows": [[1]],
    }
    with pytest.raises(ValidationError, match="table is invalid"):
        _validate_frame_payload(payload)


# -- T2: atomic export preserves old bundle on failure ----------------------


def test_export_artifacts_preserves_bundle_on_overwrite_failure(
    indicator_frame, labels, tmp_path, monkeypatch
):
    """If overwrite=True export fails mid-write, the old bundle must survive."""
    from gignn import GIGNNAutoClassifier, export_research_artifacts
    from tests.test_artifacts import Point, _fit_small_auto_model

    model, geometry = _fit_small_auto_model(indicator_frame, labels)
    output_dir = tmp_path / "run"
    model.export_research_artifacts(
        output_dir,
        raw_frame=indicator_frame,
        geometry=geometry,
    )
    old_manifest = (output_dir / "manifest.json").read_text()

    # Simulate a write failure during the second export
    import gignn.artifacts as artifacts_mod

    original_write_manifest = artifacts_mod._write_manifest

    def failing_write_manifest(*args, **kwargs):
        raise RuntimeError("simulated write failure")

    monkeypatch.setattr(artifacts_mod, "_write_manifest", failing_write_manifest)

    with pytest.raises(RuntimeError, match="simulated write failure"):
        model.export_research_artifacts(
            output_dir,
            raw_frame=indicator_frame,
            geometry=geometry,
            overwrite=True,
        )

    monkeypatch.setattr(artifacts_mod, "_write_manifest", original_write_manifest)
    assert (output_dir / "manifest.json").read_text() == old_manifest
    assert (output_dir / "process" / "factor_scores.csv").is_file()


# -- T3: metadata zip-bomb length check -------------------------------------


def test_auto_archive_rejects_oversized_metadata_content(tmp_path):
    """metadata whose actual decompressed content exceeds the limit must be rejected."""
    import json
    import zipfile

    from gignn.exceptions import ValidationError

    # Craft a zip where file_size in central dir is small but actual content is large
    big_content = "x" * (5 * 1024 * 1024 + 100)  # > 5 MB
    path = tmp_path / "bomb.gignn"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("metadata.json", json.dumps({"format": "gignn-auto"}))
        archive.writestr("arrays/array-0000.npy", big_content)

    from gignn.auto_archive import load_payload

    with pytest.raises(ValidationError):
        load_payload(path)


# -- T4: generalization TypeError guards ------------------------------------


def test_model_accuracy_table_rejects_none_accuracy():
    from gignn.generalization import model_accuracy_table

    with pytest.raises(ValidationError, match="real numbers"):
        model_accuracy_table({"model_a": None})


def test_model_accuracy_table_rejects_string_accuracy():
    from gignn.generalization import model_accuracy_table

    with pytest.raises(ValidationError, match="real numbers"):
        model_accuracy_table({"model_a": "high"})


def test_ordered_label_codes_rejects_unhashable_class_order():
    from gignn.generalization import _ordered_label_codes

    observed = pd.Series([0, 1])
    predicted = pd.Series([0, 0])
    with pytest.raises(ValidationError, match="hashable scalar"):
        _ordered_label_codes(
            observed, predicted, class_order=[[0, 1]], class_names=None
        )


# -- T5: random_state bool + within-partition duplicates --------------------


def test_auto_checkpoint_rejects_bool_random_state():
    from gignn.auto_persistence import _validate_constructor_params

    params = {
        "detector_grid": {"discretizer": ["quantiles"], "max_bins": [2]},
        "graph_grid": {"strategy": ["knn"], "k": [1]},
        "model_grid": {
            "hidden_channels": [(8,)],
            "learning_rate": [0.01],
            "max_epochs": [1],
        },
        "top_k_features": None,
        "random_state": True,
        "validation_fraction": 0.2,
        "test_fraction": 0.2,
        "max_grid_candidates": 100,
    }
    with pytest.raises(ValidationError, match="random_state"):
        _validate_constructor_params(params)


def test_auto_checkpoint_rejects_within_partition_duplicate_indices():
    from gignn.auto_persistence import _validate_index_partition

    payload = {
        "target": [0, 1, 2, 3],
        "train_indices": torch.tensor([0, 0, 1], dtype=torch.int64),
        "validation_indices": torch.tensor([2], dtype=torch.int64),
        "test_indices": torch.tensor([3], dtype=torch.int64),
    }
    with pytest.raises(ValidationError, match="duplicates"):
        _validate_index_partition(payload)


# -- T6: boxplot tick_labels + figure close ---------------------------------


def test_plot_metric_boxplot_no_deprecation_warning(tmp_path):
    """boxplot must not use deprecated labels= kwarg (triggers error under filterwarnings)."""
    matplotlib = pytest.importorskip("matplotlib")
    import matplotlib.pyplot as plt

    from gignn.plotting import plot_metric_boxplot

    metrics = pd.DataFrame({"metric": ["acc"] * 3, "value": [0.8, 0.9, 0.85]})
    figure, ax = plot_metric_boxplot(
        metrics,
        metric_column="metric",
        value_column="value",
        group_column=None,
        output_path=tmp_path / "box.png",
    )
    plt.close(figure)


def test_save_figure_closes_figure_when_close_true(tmp_path):
    from gignn.plotting import _save_figure
    import matplotlib.pyplot as plt

    figure, _ = plt.subplots()
    _save_figure(figure, tmp_path / "fig.png", dpi=72, close=True)
    assert not plt.fignum_exists(figure.number)


def test_save_figure_keeps_figure_when_close_false(tmp_path):
    from gignn.plotting import _save_figure
    import matplotlib.pyplot as plt

    figure, _ = plt.subplots()
    _save_figure(figure, tmp_path / "fig2.png", dpi=72, close=False)
    assert plt.fignum_exists(figure.number)
    plt.close(figure)


# -- T7: SimulationReport immutability + dead code cleanup ------------------


def test_simulation_report_arrays_are_read_only():
    from gignn.reports import SimulationReport

    report = SimulationReport(
        baseline_predictions=np.array([0, 1]),
        scenario_predictions=np.array([1, 1]),
        baseline_probabilities=np.array([[0.5, 0.5], [0.5, 0.5]]),
        scenario_probabilities=np.array([[0.4, 0.6], [0.5, 0.5]]),
        probability_delta=np.array([[-0.1, 0.1], [0.0, 0.0]]),
        changed_mask=np.array([True, False]),
    )
    for name in (
        "baseline_predictions",
        "scenario_predictions",
        "baseline_probabilities",
        "scenario_probabilities",
        "probability_delta",
        "changed_mask",
    ):
        assert not getattr(report, name).flags.writeable


def test_validate_edge_weight_accepts_list_and_array():
    from gignn.graph.data import _validate_edge_weight

    weights = _validate_edge_weight([1.0, 2.0, 3.0], num_edges=3)
    assert weights.shape == (3,)
    weights2 = _validate_edge_weight(np.array([0.5, 0.5]), num_edges=2)
    assert weights2.shape == (2,)


# ===========================================================================
# Round-3 fixes: RNG, checkpoint, splitting, cKDTree, plotting, grid, archive,
# fingerprint, simulation report, predict fallback
# ===========================================================================


# -- R3-T1: multihop_sampling random_state=None is stochastic ---------------


def test_multihop_sampling_none_consumes_replayable_global_rng_sequence():
    adjacency = {0: [1, 2, 3], 1: [0, 2, 3], 2: [0, 1, 3], 3: [0, 1, 2]}

    np.random.seed(123)
    first_run = [
        multihop_sampling([0], [8], adjacency, random_state=None)[1] for _ in range(2)
    ]
    np.random.seed(123)
    replay = [
        multihop_sampling([0], [8], adjacency, random_state=None)[1] for _ in range(2)
    ]

    np.testing.assert_array_equal(first_run[0], replay[0])
    np.testing.assert_array_equal(first_run[1], replay[1])
    assert [values.tolist() for values in first_run] == [
        [3, 1, 1, 2, 3, 2, 3, 3],
        [2, 2, 2, 3, 2, 1, 2, 3],
    ]


def test_multihop_sampling_int_is_reproducible():
    """random_state=42 must produce identical samples across calls."""
    adjacency = {0: [1, 2, 3], 1: [0, 2, 3], 2: [0, 1, 3], 3: [0, 1, 2]}
    first = multihop_sampling([0], [5], adjacency, random_state=42)
    second = multihop_sampling([0], [5], adjacency, random_state=42)
    np.testing.assert_array_equal(first[1], second[1])


# -- R3-T2: checkpoint numpy-scalar coercion + symlink guard + fsync --------


def test_to_python_scalar_converts_numpy_scalars():
    from gignn.estimators import _to_python_scalar

    assert _to_python_scalar(np.int64(42)) == 42
    assert type(_to_python_scalar(np.int64(42))) is int
    assert _to_python_scalar(np.float64(3.14)) == pytest.approx(3.14)
    assert type(_to_python_scalar(np.float64(3.14))) is float
    assert _to_python_scalar(np.bool_(True)) is True
    assert _to_python_scalar("hello") == "hello"
    assert _to_python_scalar(42) == 42


def test_estimator_save_load_with_numpy_int64_classes(
    indicator_frame, labels, edge_index, tmp_path
):
    """Save/load round-trip must succeed when classes_ holds numpy int64 scalars."""
    pytest.importorskip("torch")
    pytest.importorskip("torch_geometric")
    model = GIGNNClassifier(hidden_channels=(8,), max_epochs=1, random_state=42)
    model.fit(indicator_frame, labels, edge_index=edge_index)
    # Simulate classes_ containing numpy int64 scalars (dtype=object preserves them)
    n_classes = len(model.classes_)
    model.classes_ = np.array([np.int64(i) for i in range(n_classes)], dtype=object)
    path = tmp_path / "numpy_classes.pt"
    model.save(path)
    loaded = GIGNNClassifier.load(path)
    assert loaded.classes_.tolist() == list(range(n_classes))


def test_load_checkpoint_rejects_symlink_path(tmp_path, symlink_supported):
    from gignn.io import load_checkpoint, save_checkpoint

    target = tmp_path / "real.pt"
    save_checkpoint(target, {"value": torch.ones(1)})

    link = tmp_path / "link.pt"
    os.symlink(target, link)

    with pytest.raises(ValidationError, match="symlink"):
        load_checkpoint(link)


# -- R3-T3: group split proportion divergence warning ----------------------


def test_group_split_warns_on_proportion_divergence():
    """With 3 groups and test_fraction=0.2, realized 1/3 diverges >10pp."""
    import warnings

    y = pd.Series([0, 1] * 10)
    groups = pd.Series(["a", "b", "c"] * 6 + ["a", "b"])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        resolve_split_indices(
            y,
            validation_fraction=0.2,
            test_fraction=0.2,
            random_state=0,
            spatial_groups=groups,
        )
    assert any("realized" in str(w.message) for w in caught)


# -- R3-T4: cKDTree KNN/distance edges ------------------------------------


def test_knn_edges_produces_expected_edge_count():
    from gignn.graph.builders import _knn_edges

    points = [(0.0, 0.0), (1.0, 0.0), (2.0, 0.0), (3.0, 0.0)]
    edges = _knn_edges(points, k=2)
    assert edges.shape[1] == 8  # 4 nodes * 2 neighbors


def test_distance_edges_produces_expected_edge_count():
    from gignn.graph.builders import _distance_edges

    points = [(0.0, 0.0), (1.0, 0.0), (2.0, 0.0)]
    edges = _distance_edges(points, threshold=1.5)
    assert edges.shape[1] == 4  # (0,1), (1,0), (1,2), (2,1)


def test_knn_edges_handles_k_larger_than_n():
    from gignn.graph.builders import _knn_edges

    points = [(0.0, 0.0), (1.0, 0.0)]
    edges = _knn_edges(points, k=5)
    assert edges.shape[1] == 2  # (0,1), (1,0) — capped at N-1


# -- R3-T5: plot_interaction_network rejects negative weights --------------


def test_plot_interaction_network_rejects_negative_weights():
    matplotlib = pytest.importorskip("matplotlib")
    import matplotlib.pyplot as plt

    from gignn.plotting import plot_interaction_network

    interactions = pd.DataFrame(
        {"factor_1": ["a", "b"], "factor_2": ["b", "c"], "q_interaction": [-0.5, 0.8]}
    )
    with pytest.raises(ValidationError, match="non-negative"):
        plot_interaction_network(interactions)
    plt.close("all")


# -- R3-T6: search-grid element type validation ----------------------------


def test_validate_search_grid_rejects_none_element():
    from gignn.auto_persistence import _validate_search_grid

    with pytest.raises(ValidationError, match="invalid"):
        _validate_search_grid({"k": [None]}, name="test grid")


def test_validate_search_grid_rejects_nested_dict_element():
    from gignn.auto_persistence import _validate_search_grid

    with pytest.raises(ValidationError, match="invalid"):
        _validate_search_grid({"k": [{"x": 1}]}, name="test grid")


def test_validate_search_grid_accepts_flat_list_values():
    from gignn.auto_persistence import _validate_search_grid

    _validate_search_grid(
        {"hidden_channels": [[8], [16]], "lr": [0.01, 0.001]},
        name="test grid",
    )


# -- R3-T9: simulation_report missing attributes raises ValidationError ---


def test_validate_simulation_report_rejects_missing_attributes():
    from gignn.artifacts import _validate_simulation_report

    class FakeReport:
        pass

    with pytest.raises(ValidationError, match="simulation_report must expose"):
        _validate_simulation_report(FakeReport(), expected_rows=None, class_count=1)


# -- R3-T10: broadened predict/proba exception catch ----------------------


def test_resolve_predictions_runtime_error_becomes_validation_error():
    from gignn.artifacts import _resolve_predictions

    class FakeModel:
        def predict(self, frame, geometry=None):
            raise RuntimeError("model crashed")

    frame = pd.DataFrame({"a": [1.0, 2.0]})
    with pytest.raises(ValidationError, match="Could not infer predictions"):
        _resolve_predictions(
            FakeModel(),
            frame=frame,
            geometry=None,
            predictions=None,
            expected_rows=2,
        )


def test_resolve_probabilities_runtime_error_becomes_validation_error():
    from gignn.artifacts import _resolve_probabilities

    class FakeModel:
        def predict_proba(self, frame, geometry=None):
            raise RuntimeError("model crashed")

    frame = pd.DataFrame({"a": [1.0, 2.0]})
    with pytest.raises(ValidationError, match="Could not infer probabilities"):
        _resolve_probabilities(
            FakeModel(),
            frame=frame,
            geometry=None,
            probabilities=None,
            expected_rows=2,
            class_count=3,
        )
