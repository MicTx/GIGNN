import json
import subprocess
import sys
import warnings
import zipfile

import numpy as np
import pandas as pd
import pytest
import torch

from gignn import GIGNNAutoClassifier
from gignn.exceptions import ValidationError


class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.centroid = self


def _rewrite_checkpoint_metadata(path, update):
    with zipfile.ZipFile(path) as source:
        members = {
            info.filename: source.read(info)
            for info in source.infolist()
            if info.filename != "metadata.json"
        }
        metadata = json.loads(source.read("metadata.json"))
    update(metadata)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as target:
        target.writestr("metadata.json", json.dumps(metadata, sort_keys=True))
        for name, content in members.items():
            target.writestr(name, content)


def _fitted_auto_model(*, graph_grid=None):
    frame = pd.DataFrame({"feature": np.arange(9, dtype=float)})
    labels = pd.Series([0, 1, 2] * 3)
    geometry = [Point(index, 0) for index in range(len(frame))]
    model = GIGNNAutoClassifier(
        detector_grid={"discretizer": ["quantiles"], "max_bins": [2]},
        graph_grid=graph_grid or {"strategy": ["knn"], "k": [1]},
        model_grid={
            "hidden_channels": [(8,)],
            "learning_rate": [0.01],
            "max_epochs": [1],
        },
        random_state=42,
    ).fit(
        frame,
        labels,
        geometry=geometry,
        split_labels=[
            "train",
            "train",
            "train",
            "train",
            "validation",
            "validation",
            "test",
            "test",
            "test",
        ],
    )
    return model, frame, geometry


def test_auto_classifier_round_trips_knn_grid_with_none_threshold(tmp_path):
    model, frame, geometry = _fitted_auto_model(
        graph_grid={"strategy": ["knn"], "k": [1], "threshold": [None]}
    )
    path = tmp_path / "none-threshold.gignn"

    model.save(path)
    loaded = GIGNNAutoClassifier.load(path)

    np.testing.assert_array_equal(
        loaded.predict(frame, geometry=geometry),
        model.predict(frame, geometry=geometry),
    )


def test_auto_classifier_save_load_round_trip(tmp_path, monkeypatch):
    model, frame, geometry = _fitted_auto_model()
    before = model.predict(frame, geometry=geometry)
    path = tmp_path / "auto.gignn"

    model.save(path)
    assert zipfile.is_zipfile(path)
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        assert names[0] == "metadata.json"
        assert all(
            name == "metadata.json" or name.startswith("arrays/") for name in names
        )
    monkeypatch.setattr(
        torch,
        "load",
        lambda *args, **kwargs: pytest.fail(
            "automatic checkpoints must not use torch.load"
        ),
    )
    loaded = GIGNNAutoClassifier.load(path)
    after = loaded.predict(frame, geometry=geometry)

    np.testing.assert_array_equal(before, after)
    assert loaded.class_order_ == model.class_order_
    assert loaded.graph_report_.selected_strategy == "knn"
    assert loaded.selector_.selected_features_ == model.selector_.selected_features_
    assert (
        loaded.estimator_.training_result_.losses
        == model.estimator_.training_result_.losses
    )
    for feature in model.selector_.selected_features_:
        np.testing.assert_array_equal(
            loaded.selector_.discretizers_[feature].bins,
            model.selector_.discretizers_[feature].bins,
        )


def test_auto_classifier_save_load_round_trip_with_top_k_and_ordinal_metadata(
    tmp_path,
):
    frame = pd.DataFrame(
        {
            "primary": np.arange(12, dtype=float),
            "secondary": np.arange(12, dtype=float)[::-1],
        }
    )
    labels = pd.Series(["low", "medium", "high"] * 4)
    geometry = [Point(index, 0) for index in range(len(frame))]
    node_ids = [f"unit-{index}" for index in range(len(frame))]
    model = GIGNNAutoClassifier(
        detector_grid={"discretizer": ["quantiles"], "max_bins": [2]},
        graph_grid={"strategy": ["knn"], "k": [1]},
        model_grid={
            "hidden_channels": [(8,)],
            "learning_rate": [0.01],
            "max_epochs": [1],
        },
        top_k_features=1,
        random_state=42,
    ).fit(
        frame,
        labels,
        geometry=geometry,
        class_order=["low", "medium", "high"],
        node_ids=node_ids,
        split_labels=[
            "train",
            "train",
            "train",
            "train",
            "train",
            "train",
            "validation",
            "validation",
            "validation",
            "test",
            "test",
            "test",
        ],
    )
    path = tmp_path / "top-k.gignn"

    model.save(path)
    loaded = GIGNNAutoClassifier.load(path)

    np.testing.assert_array_equal(
        loaded.predict(frame, geometry=geometry),
        model.predict(frame, geometry=geometry),
    )
    assert loaded.class_order_ == ("low", "medium", "high")
    np.testing.assert_array_equal(loaded.node_ids_, np.asarray(node_ids, dtype=object))
    assert loaded.selector_.selected_features_ == model.selector_.selected_features_
    assert set(loaded.selector_.discretizers_) == set(
        loaded.selector_.selected_features_
    )
    pd.testing.assert_frame_equal(loaded.selector_.scores_, model.selector_.scores_)
    assert loaded.validation_report_.partition == "validation"
    np.testing.assert_array_equal(
        loaded.validation_report_.indices, model.validation_indices_
    )
    assert loaded.generalization_report_.partition == "test"
    np.testing.assert_array_equal(
        loaded.generalization_report_.indices, model.test_indices_
    )


def test_auto_classifier_loads_in_fresh_process(tmp_path):
    model, frame, geometry = _fitted_auto_model()
    path = tmp_path / "fresh-process.gignn"
    model.save(path)
    script = """
import json
import pandas as pd
from gignn import GIGNNAutoClassifier

class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.centroid = self

frame = pd.DataFrame({"feature": list(range(9))})
geometry = [Point(index, 0) for index in range(len(frame))]
model = GIGNNAutoClassifier.load(PATH)
print(json.dumps(model.predict(frame, geometry=geometry).tolist()))
""".replace("PATH", repr(str(path)))

    result = subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        text=True,
    )

    assert json.loads(result.stdout) == model.predict(frame, geometry=geometry).tolist()


def test_auto_classifier_selector_payload_preserves_strategy(tmp_path):
    model, _, _ = _fitted_auto_model()
    path = tmp_path / "strategy.gignn"

    model.save(path)
    loaded = GIGNNAutoClassifier.load(path)

    assert loaded.selector_.strategy == model.selector_.strategy == "factor"
    assert loaded.selector_.top_interactions == model.selector_.top_interactions == 10


def test_auto_classifier_loads_legacy_selector_params(tmp_path):
    model, _, _ = _fitted_auto_model()
    path = tmp_path / "legacy-selector.gignn"
    model.save(path)

    def strip_new_params(metadata):
        params = metadata["payload"]["selector"]["params"]
        del params["strategy"]
        del params["top_interactions"]

    _rewrite_checkpoint_metadata(path, strip_new_params)

    loaded = GIGNNAutoClassifier.load(path)
    assert loaded.selector_.strategy == "factor"
    assert loaded.selector_.top_interactions == 10


def test_auto_classifier_load_rejects_invalid_selector_strategy(tmp_path):
    model, _, _ = _fitted_auto_model()
    path = tmp_path / "bad-strategy.gignn"
    model.save(path)
    _rewrite_checkpoint_metadata(
        path,
        lambda metadata: metadata["payload"]["selector"]["params"].update(
            strategy="bogus"
        ),
    )

    with pytest.raises(ValidationError, match="selector strategy"):
        GIGNNAutoClassifier.load(path)


def test_auto_classifier_load_rejects_invalid_top_interactions(tmp_path):
    model, _, _ = _fitted_auto_model()
    path = tmp_path / "bad-top-interactions.gignn"
    model.save(path)
    _rewrite_checkpoint_metadata(
        path,
        lambda metadata: metadata["payload"]["selector"]["params"].update(
            top_interactions=0
        ),
    )

    with pytest.raises(ValidationError, match="top_interactions"):
        GIGNNAutoClassifier.load(path)


def test_auto_classifier_load_rejects_incompatible_package_version(tmp_path):
    model, _, _ = _fitted_auto_model()
    path = tmp_path / "incompatible.gignn"
    model.save(path)
    _rewrite_checkpoint_metadata(
        path, lambda metadata: metadata["payload"].update(package_version="999.0.0")
    )

    with pytest.raises(ValidationError, match="package version"):
        GIGNNAutoClassifier.load(path)


def test_auto_classifier_load_rejects_inconsistent_feature_metadata(tmp_path):
    model, _, _ = _fitted_auto_model()
    path = tmp_path / "inconsistent.gignn"
    model.save(path)

    def make_inconsistent(metadata):
        metadata["payload"]["estimator"]["feature_names_in"] = ["unexpected"]

    _rewrite_checkpoint_metadata(path, make_inconsistent)

    with pytest.raises(ValidationError, match="inconsistent"):
        GIGNNAutoClassifier.load(path)


def test_auto_classifier_load_rejects_malformed_package_version(tmp_path):
    model, _, _ = _fitted_auto_model()
    path = tmp_path / "malformed-version.gignn"
    model.save(path)
    _rewrite_checkpoint_metadata(
        path, lambda metadata: metadata["payload"].update(package_version="0.1")
    )

    with pytest.raises(ValidationError, match="package version"):
        GIGNNAutoClassifier.load(path)


def test_auto_checkpoint_rejects_split_longer_than_target():
    from gignn.auto_persistence import _validate_index_partition

    payload = {
        "target": [0, 1, 2],
        "train_indices": torch.arange(4, dtype=torch.int64),
        "validation_indices": torch.tensor([0], dtype=torch.int64),
        "test_indices": torch.tensor([1], dtype=torch.int64),
    }

    with pytest.raises(ValidationError, match="split indices exceed target size"):
        _validate_index_partition(payload)


def test_auto_checkpoint_rejects_excessive_candidate_budget():
    from gignn.auto_persistence import _validate_constructor_params
    from gignn.limits import MAX_FEATURES

    params = {
        "detector_grid": {"discretizer": ["quantiles"], "max_bins": [2]},
        "graph_grid": {"strategy": ["knn"], "k": [1]},
        "model_grid": {
            "hidden_channels": [[8]],
            "learning_rate": [0.01],
            "max_epochs": [1],
        },
        "top_k_features": None,
        "random_state": 42,
        "validation_fraction": 0.2,
        "test_fraction": 0.2,
        "max_grid_candidates": MAX_FEATURES + 1,
    }

    with pytest.raises(ValidationError, match="candidate budget"):
        _validate_constructor_params(params)


def test_auto_checkpoint_rejects_workload_above_saved_budget():
    from gignn.auto_persistence import _validate_constructor_params

    params = {
        "detector_grid": {"discretizer": ["quantiles"], "max_bins": [2]},
        "graph_grid": {"strategy": ["knn"], "k": [1, 2]},
        "model_grid": {
            "hidden_channels": [[8]],
            "learning_rate": [0.01, 0.02],
            "max_epochs": [1],
        },
        "top_k_features": None,
        "random_state": 42,
        "validation_fraction": 0.2,
        "test_fraction": 0.2,
        "max_grid_candidates": 4,
    }

    with pytest.raises(ValidationError, match="exceeds candidate budget"):
        _validate_constructor_params(params)


def test_auto_classifier_load_rejects_invalid_schema(tmp_path):
    path = tmp_path / "invalid-auto.gignn"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(
            "metadata.json",
            json.dumps({"format": "gignn-auto", "schema_version": "999"}),
        )

    with pytest.raises(ValidationError, match="schema"):
        GIGNNAutoClassifier.load(path)


def test_auto_classifier_load_rejects_truncated_zip_tail(tmp_path):
    model, _, _ = _fitted_auto_model()
    path = tmp_path / "truncated-tail.gignn"
    model.save(path)
    content = path.read_bytes()
    path.write_bytes(content[:-7])

    with pytest.raises(ValidationError, match="archive is invalid"):
        GIGNNAutoClassifier.load(path)


def test_auto_classifier_load_rejects_unexpected_archive_member(tmp_path):
    path = tmp_path / "unexpected-member.gignn"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(
            "metadata.json",
            json.dumps(
                {
                    "format": "gignn-auto",
                    "schema_version": "1",
                    "arrays": {},
                    "payload": {},
                }
            ),
        )
        archive.writestr("../outside.txt", "unexpected")

    with pytest.raises(ValidationError, match="member"):
        GIGNNAutoClassifier.load(path)


def test_auto_classifier_load_rejects_array_digest_mismatch(tmp_path):
    model, _, _ = _fitted_auto_model()
    path = tmp_path / "tampered.gignn"
    model.save(path)
    with zipfile.ZipFile(path) as source:
        members = {info.filename: source.read(info) for info in source.infolist()}
    array_name = next(name for name in members if name.startswith("arrays/"))
    members[array_name] += b"tampered"
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as target:
        for name, content in members.items():
            target.writestr(name, content)

    with pytest.raises(ValidationError, match="digest"):
        GIGNNAutoClassifier.load(path)


def test_auto_classifier_load_rejects_duplicate_archive_members(tmp_path):
    path = tmp_path / "duplicate.gignn"
    metadata = json.dumps(
        {
            "format": "gignn-auto",
            "schema_version": "1",
            "arrays": {},
            "payload": {},
        }
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr("metadata.json", metadata)
            archive.writestr("metadata.json", metadata)

    with pytest.raises(ValidationError, match="duplicate archive members"):
        GIGNNAutoClassifier.load(path)


def test_auto_classifier_load_rejects_excessive_archive_members(tmp_path):
    path = tmp_path / "too-many-members.gignn"
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as archive:
        for index in range(4_097):
            archive.writestr(f"empty-{index}", b"")

    with pytest.raises(ValidationError, match="member count"):
        GIGNNAutoClassifier.load(path)


def test_auto_classifier_checkpoint_rejects_symlink_paths(tmp_path, symlink_supported):
    model, _, _ = _fitted_auto_model()
    target = tmp_path / "target.gignn"
    target.write_bytes(b"sentinel")
    link = tmp_path / "model.gignn"
    link.symlink_to(target)

    with pytest.raises(ValidationError, match="symlink"):
        model.save(link)
    assert target.read_bytes() == b"sentinel"
    with pytest.raises(ValidationError, match="symlink"):
        GIGNNAutoClassifier.load(link)
