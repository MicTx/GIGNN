from __future__ import annotations

import errno
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch

from gignn.exceptions import ValidationError
from gignn.limits import MAX_CHECKPOINT_BYTES

MAX_SAFE_PAYLOAD_DEPTH = 100
MAX_SAFE_PAYLOAD_ITEMS = 100_000


def save_checkpoint(path: str | Path, payload: dict) -> None:
    checkpoint_path = Path(path).expanduser()
    if checkpoint_path.is_symlink():
        raise ValidationError("Checkpoint path must not be a symlink.")
    checkpoint_path = checkpoint_path.resolve()
    if checkpoint_path.exists() and checkpoint_path.is_dir():
        raise IsADirectoryError(checkpoint_path)
    normalized_payload = _weights_only_safe(payload, depth=0, active=set(), count=[0])
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    # Write to a temp file and atomically replace the target so a failed save
    # (disk full, unpicklable value, kill) never truncates the previous
    # checkpoint; mirrors auto_archive._write_archive.
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w+b",
            prefix=f".{checkpoint_path.name}.",
            suffix=".tmp",
            dir=checkpoint_path.parent,
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            torch.save(normalized_payload, temporary)
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_path, checkpoint_path)
        temporary_path = None
        _fsync_parent_directory(checkpoint_path.parent)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _weights_only_safe(
    value: Any,
    *,
    depth: int,
    active: set[int],
    count: list[int],
) -> Any:
    if depth > MAX_SAFE_PAYLOAD_DEPTH:
        raise ValidationError("Checkpoint payload exceeds safe nesting depth.")
    count[0] += 1
    if count[0] > MAX_SAFE_PAYLOAD_ITEMS:
        raise ValidationError("Checkpoint payload exceeds safe item limits.")
    if isinstance(value, np.generic):
        return _weights_only_safe(value.item(), depth=depth, active=active, count=count)
    if value is None or isinstance(value, (str, bytes, bool, int, float)):
        return value
    if type(value) is torch.Tensor:
        return value
    if isinstance(value, (dict, list, tuple)):
        identity = id(value)
        if identity in active:
            raise ValidationError("Checkpoint payload must not contain cyclic values.")
        active.add(identity)
        try:
            if isinstance(value, dict):
                normalized = {}
                for key, item in value.items():
                    if not isinstance(key, (str, bytes, bool, int, float)):
                        raise ValidationError(
                            "Checkpoint payload must use weights-only-safe dictionary keys."
                        )
                    normalized[key] = _weights_only_safe(
                        item,
                        depth=depth + 1,
                        active=active,
                        count=count,
                    )
                return normalized
            if isinstance(value, list):
                return [
                    _weights_only_safe(
                        item,
                        depth=depth + 1,
                        active=active,
                        count=count,
                    )
                    for item in value
                ]
            return tuple(
                _weights_only_safe(
                    item,
                    depth=depth + 1,
                    active=active,
                    count=count,
                )
                for item in value
            )
        finally:
            active.remove(identity)
    raise ValidationError(
        "Checkpoint payload must contain only weights-only-safe values."
    )


def _fsync_parent_directory(directory: Path) -> None:
    unsupported_errors = {errno.EINVAL, errno.ENOTSUP, errno.EBADF}
    if hasattr(errno, "EOPNOTSUPP"):
        unsupported_errors.add(errno.EOPNOTSUPP)
    try:
        descriptor = os.open(str(directory), os.O_RDONLY)
    except OSError as exc:
        if exc.errno in unsupported_errors or (
            os.name == "nt" and exc.errno in {errno.EACCES, errno.EPERM}
        ):
            return
        raise
    try:
        os.fsync(descriptor)
    except OSError as exc:
        if exc.errno not in unsupported_errors:
            raise
    finally:
        os.close(descriptor)


def load_checkpoint(path: str | Path) -> dict:
    checkpoint_path = Path(path).expanduser()
    if checkpoint_path.is_symlink():
        raise ValidationError("Checkpoint path must not be a symlink.")
    checkpoint_path = checkpoint_path.resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    if checkpoint_path.stat().st_size > MAX_CHECKPOINT_BYTES:
        raise ValidationError("Checkpoint file exceeds size limit.")
    return torch.load(checkpoint_path, map_location="cpu", weights_only=True)
