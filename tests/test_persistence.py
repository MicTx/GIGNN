import errno

import numpy as np
import pytest
import torch

import gignn.io.persistence as persistence
from gignn.exceptions import ValidationError
from gignn.io import load_checkpoint, save_checkpoint


class UnsupportedPayload:
    pass


class TensorSubclass(torch.Tensor):
    pass


def test_checkpoint_round_trips_nested_numpy_scalars(tmp_path):
    path = tmp_path / "numpy-scalars.pt"

    save_checkpoint(
        path,
        {
            "integer": np.int64(7),
            "floating": np.float32(1.25),
            "boolean": np.bool_(True),
            "nested": [np.int32(3), (np.float64(2.5),)],
            "tensor": torch.ones(2),
        },
    )
    loaded = load_checkpoint(path)

    assert type(loaded["integer"]) is int
    assert type(loaded["floating"]) is float
    assert type(loaded["boolean"]) is bool
    assert type(loaded["nested"][0]) is int
    assert type(loaded["nested"][1][0]) is float
    torch.testing.assert_close(loaded["tensor"], torch.ones(2))


def test_save_checkpoint_rejects_non_weights_only_payload_before_replace(tmp_path):
    path = tmp_path / "unsupported.pt"

    with pytest.raises(ValidationError, match="weights-only-safe"):
        save_checkpoint(path, {"unsupported": UnsupportedPayload()})

    assert not path.exists()


def test_save_checkpoint_rejects_tensor_subclasses_before_replace(tmp_path):
    path = tmp_path / "tensor-subclass.pt"
    value = torch.ones(1).as_subclass(TensorSubclass)

    with pytest.raises(ValidationError, match="weights-only-safe"):
        save_checkpoint(path, {"tensor": value})

    assert not path.exists()


def test_save_checkpoint_rejects_cyclic_payload_before_replace(tmp_path):
    path = tmp_path / "cyclic.pt"
    cycle = []
    cycle.append(cycle)

    with pytest.raises(ValidationError, match="cyclic|depth"):
        save_checkpoint(path, {"cycle": cycle})

    assert not path.exists()


def test_save_checkpoint_rejects_excessive_payload_depth(tmp_path):
    path = tmp_path / "deep.pt"
    value = None
    for _ in range(101):
        value = [value]

    with pytest.raises(ValidationError, match="depth"):
        save_checkpoint(path, {"deep": value})

    assert not path.exists()


@pytest.mark.parametrize(
    "scalar",
    [np.datetime64("2026-08-24"), np.complex64(1 + 2j)],
)
def test_save_checkpoint_rejects_numpy_scalars_without_safe_python_types(
    tmp_path, scalar
):
    path = tmp_path / "unsupported-numpy-scalar.pt"

    with pytest.raises(ValidationError, match="weights-only-safe"):
        save_checkpoint(path, {"unsupported": scalar})

    assert not path.exists()


def test_fsync_parent_directory_closes_descriptor(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(persistence.os, "open", lambda path, flags: 37)
    monkeypatch.setattr(
        persistence.os, "fsync", lambda descriptor: calls.append(("fsync", descriptor))
    )
    monkeypatch.setattr(
        persistence.os, "close", lambda descriptor: calls.append(("close", descriptor))
    )

    persistence._fsync_parent_directory(tmp_path)

    assert calls == [("fsync", 37), ("close", 37)]


def test_fsync_parent_directory_ignores_unsupported_filesystem(tmp_path, monkeypatch):
    def unsupported_open(path, flags):
        raise OSError(errno.EINVAL, "directory fsync unsupported")

    monkeypatch.setattr(persistence.os, "open", unsupported_open)

    persistence._fsync_parent_directory(tmp_path)


def test_fsync_parent_directory_ignores_windows_directory_access_denied(
    tmp_path, monkeypatch
):
    def unsupported_open(path, flags):
        raise OSError(errno.EACCES, "Windows directory descriptors unsupported")

    monkeypatch.setattr(persistence.os, "name", "nt")
    monkeypatch.setattr(persistence.os, "open", unsupported_open)

    persistence._fsync_parent_directory(tmp_path)


def test_fsync_parent_directory_propagates_unexpected_errors(tmp_path, monkeypatch):
    def failed_open(path, flags):
        raise OSError(errno.EIO, "filesystem I/O failure")

    monkeypatch.setattr(persistence.os, "open", failed_open)

    with pytest.raises(OSError, match="filesystem I/O failure"):
        persistence._fsync_parent_directory(tmp_path)


def test_save_checkpoint_reports_post_replace_fsync_failure(tmp_path, monkeypatch):
    path = tmp_path / "post-replace.pt"

    def failed_fsync(parent):
        raise OSError(errno.EIO, "directory durability unknown")

    monkeypatch.setattr(persistence, "_fsync_parent_directory", failed_fsync)

    with pytest.raises(OSError, match="directory durability unknown"):
        save_checkpoint(path, {"tensor": torch.ones(1)})

    assert path.is_file()
    torch.testing.assert_close(load_checkpoint(path)["tensor"], torch.ones(1))
