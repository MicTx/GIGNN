from __future__ import annotations

import hashlib
import io
import json
import math
import os
import re
import stat
import struct
import tempfile
import zipfile
from dataclasses import dataclass
import errno
from pathlib import Path
from typing import Any

import numpy as np
import torch

from gignn.exceptions import ValidationError
from gignn.limits import MAX_CHECKPOINT_BYTES, MAX_CHECKPOINT_TENSOR_ELEMENTS

ARCHIVE_FORMAT = "gignn-auto"
ARCHIVE_SCHEMA_VERSION = "1"

_METADATA_MEMBER = "metadata.json"
_ARRAY_REFERENCE_KEY = "__gignn_array__"
_ARRAY_ID_PATTERN = re.compile(r"array-[0-9]{4}")
_ARRAY_MEMBER_PATTERN = re.compile(r"arrays/array-[0-9]{4}\.npy")
_MAX_ARCHIVE_MEMBERS = 4_096
_MAX_METADATA_BYTES = 5 * 1024 * 1024
_MAX_PAYLOAD_DEPTH = 64
_MAX_PAYLOAD_ITEMS = 200_000
_MAX_ARRAY_DIMENSIONS = 8
_MAX_ARRAY_ELEMENTS = MAX_CHECKPOINT_TENSOR_ELEMENTS + 5_000_000
_ALLOWED_COMPRESSIONS = {zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED}


@dataclass(frozen=True)
class _ArrayEntry:
    name: str
    descriptor: dict[str, Any]
    content: bytes


def save_payload(path: str | Path, payload: dict[str, Any]) -> None:
    encoded, entries, _, _ = _encode_value(
        payload,
        entries=(),
        elements=0,
        depth=0,
        items=0,
    )
    if not isinstance(encoded, dict):
        raise ValidationError(
            "Automatic workflow checkpoint payload must be a dictionary."
        )
    metadata = {
        "format": ARCHIVE_FORMAT,
        "schema_version": ARCHIVE_SCHEMA_VERSION,
        "arrays": {entry.name: entry.descriptor for entry in entries},
        "payload": encoded,
    }
    _write_archive(path, metadata=metadata, entries=entries)


def load_payload(path: str | Path) -> dict[str, Any]:
    metadata, arrays = _read_archive(path)
    payload, used_arrays, _ = _decode_value(
        metadata["payload"],
        arrays,
        used_arrays=frozenset(),
        depth=0,
        items=0,
    )
    if used_arrays != frozenset(arrays):
        raise ValidationError(
            "Automatic workflow checkpoint contains unreferenced arrays."
        )
    if not isinstance(payload, dict):
        raise ValidationError("Automatic workflow checkpoint payload is invalid.")
    return payload


def _write_archive(
    path: str | Path,
    *,
    metadata: dict[str, Any],
    entries: tuple[_ArrayEntry, ...],
) -> None:
    checkpoint_path = _checkpoint_path(path)
    if checkpoint_path.is_symlink():
        raise ValidationError("Automatic workflow checkpoint must not be a symlink.")
    if checkpoint_path.exists() and checkpoint_path.is_dir():
        raise IsADirectoryError(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_bytes = _encode_metadata(metadata)
    if len(metadata_bytes) > _MAX_METADATA_BYTES:
        raise ValidationError(
            "Automatic workflow checkpoint metadata exceeds size limit."
        )

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
            with zipfile.ZipFile(
                temporary,
                mode="w",
                compression=zipfile.ZIP_DEFLATED,
                compresslevel=6,
            ) as archive:
                archive.writestr(_METADATA_MEMBER, metadata_bytes)
                for entry in entries:
                    archive.writestr(entry.descriptor["path"], entry.content)
            temporary.flush()
            size = os.fstat(temporary.fileno()).st_size
            if size > MAX_CHECKPOINT_BYTES:
                raise ValidationError(
                    "Automatic workflow checkpoint exceeds size limit."
                )
            os.fsync(temporary.fileno())
            _validate_replace_target(checkpoint_path)
            os.replace(temporary_path, checkpoint_path)
            temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _validate_replace_target(path: Path) -> None:
    try:
        mode = os.lstat(path).st_mode
    except FileNotFoundError:
        return
    if not stat.S_ISREG(mode):
        raise ValidationError(
            "Automatic workflow checkpoint target must be an ordinary file."
        )


def _read_archive(
    path: str | Path,
) -> tuple[dict[str, Any], dict[str, np.ndarray | torch.Tensor]]:
    checkpoint_path = _checkpoint_path(path)
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(checkpoint_path, flags)
    except FileNotFoundError:
        raise
    except OSError as exc:
        if exc.errno == errno.ELOOP:
            raise ValidationError(
                "Automatic workflow checkpoint must not be a symlink."
            ) from exc
        raise ValidationError("Automatic workflow checkpoint file is invalid.") from exc

    try:
        with os.fdopen(descriptor, "rb", closefd=True) as stream:
            file_stat = os.fstat(stream.fileno())
            if not stat.S_ISREG(file_stat.st_mode):
                raise ValidationError(
                    "Automatic workflow checkpoint must be an ordinary file."
                )
            if file_stat.st_size > MAX_CHECKPOINT_BYTES:
                raise ValidationError(
                    "Automatic workflow checkpoint exceeds size limit."
                )
            try:
                _validate_central_directory(stream, file_size=file_stat.st_size)
                with zipfile.ZipFile(stream) as archive:
                    infos = archive.infolist()
                    _validate_archive_members(infos)
                    metadata_info = infos[0]
                    if metadata_info.file_size > _MAX_METADATA_BYTES:
                        raise ValidationError(
                            "Automatic workflow checkpoint metadata exceeds size limit."
                        )
                    content = archive.read(metadata_info)
                    if len(content) > _MAX_METADATA_BYTES:
                        raise ValidationError(
                            "Automatic workflow checkpoint metadata exceeds size limit."
                        )
                    metadata = _decode_metadata(content)
                    descriptors = _validate_archive_metadata(metadata)
                    expected_members = {_METADATA_MEMBER}
                    expected_members.update(
                        descriptor["path"] for descriptor in descriptors.values()
                    )
                    if {info.filename for info in infos} != expected_members:
                        raise ValidationError(
                            "Automatic workflow checkpoint has an unexpected archive member."
                        )
                    arrays = {
                        name: _read_array(archive, descriptor)
                        for name, descriptor in descriptors.items()
                    }
            except ValidationError:
                raise
            except (
                OSError,
                EOFError,
                UnicodeDecodeError,
                json.JSONDecodeError,
                zipfile.BadZipFile,
                struct.error,
            ) as exc:
                raise ValidationError(
                    "Automatic workflow checkpoint archive is invalid."
                ) from exc
    except ValidationError:
        raise
    return metadata, arrays


def _validate_central_directory(stream: Any, *, file_size: int) -> None:
    if file_size < 22:
        raise ValidationError("Automatic workflow checkpoint archive is invalid.")
    tail_size = min(file_size, 65_557)
    stream.seek(file_size - tail_size)
    tail = stream.read(tail_size)
    marker = tail.rfind(b"PK\x05\x06")
    if marker < 0 or marker + 22 > len(tail):
        raise ValidationError("Automatic workflow checkpoint archive is invalid.")
    fields = struct.unpack_from("<4s4H2LH", tail, marker)
    disk_number, directory_disk, disk_entries, total_entries = fields[1:5]
    comment_length = fields[-1]
    if marker + 22 + comment_length != len(tail):
        raise ValidationError("Automatic workflow checkpoint archive is invalid.")
    if disk_number != 0 or directory_disk != 0 or disk_entries != total_entries:
        raise ValidationError(
            "Automatic workflow checkpoint multi-disk archives are unsupported."
        )
    if total_entries == 0xFFFF:
        raise ValidationError(
            "Automatic workflow checkpoint ZIP64 member counts are unsupported."
        )
    if total_entries == 0 or total_entries > _MAX_ARCHIVE_MEMBERS:
        raise ValidationError(
            "Automatic workflow checkpoint archive member count is invalid."
        )
    stream.seek(0)


def _checkpoint_path(path: str | Path) -> Path:
    return Path(path).expanduser().absolute()


def _encode_metadata(metadata: dict[str, Any]) -> bytes:
    try:
        return json.dumps(
            metadata,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValidationError(
            "Automatic workflow checkpoint metadata is not JSON serializable."
        ) from exc


def _decode_metadata(content: bytes) -> dict[str, Any]:
    def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        resolved: dict[str, Any] = {}
        for key, value in pairs:
            if key in resolved:
                raise ValidationError(
                    "Automatic workflow checkpoint metadata has duplicate keys."
                )
            resolved = {**resolved, key: value}
        return resolved

    def reject_constant(value: str) -> None:
        raise ValidationError(
            f"Automatic workflow checkpoint metadata contains {value}."
        )

    metadata = json.loads(
        content.decode("utf-8"),
        object_pairs_hook=reject_duplicate_keys,
        parse_constant=reject_constant,
    )
    if not isinstance(metadata, dict):
        raise ValidationError("Automatic workflow checkpoint metadata is invalid.")
    return metadata


def _validate_archive_members(infos: list[zipfile.ZipInfo]) -> None:
    if not infos or len(infos) > _MAX_ARCHIVE_MEMBERS:
        raise ValidationError(
            "Automatic workflow checkpoint archive member count is invalid."
        )
    names = [info.filename for info in infos]
    if len(names) != len(set(names)):
        raise ValidationError(
            "Automatic workflow checkpoint has duplicate archive members."
        )
    if names[0] != _METADATA_MEMBER:
        raise ValidationError(
            "Automatic workflow checkpoint metadata member is missing."
        )
    total_size = 0
    for info in infos:
        if info.is_dir() or info.flag_bits & 0x1:
            raise ValidationError(
                "Automatic workflow checkpoint archive member is invalid."
            )
        if info.compress_type not in _ALLOWED_COMPRESSIONS:
            raise ValidationError(
                "Automatic workflow checkpoint archive compression is invalid."
            )
        if (
            info.filename != _METADATA_MEMBER
            and _ARRAY_MEMBER_PATTERN.fullmatch(info.filename) is None
        ):
            raise ValidationError(
                "Automatic workflow checkpoint has an unexpected archive member."
            )
        total_size += info.file_size
        if total_size > MAX_CHECKPOINT_BYTES:
            raise ValidationError(
                "Automatic workflow checkpoint expanded size exceeds limit."
            )


def _validate_archive_metadata(metadata: dict[str, Any]) -> dict[str, dict[str, Any]]:
    if metadata.get("format") != ARCHIVE_FORMAT:
        raise ValidationError("Automatic workflow checkpoint format is invalid.")
    if metadata.get("schema_version") != ARCHIVE_SCHEMA_VERSION:
        raise ValidationError(
            "Automatic workflow checkpoint archive schema is incompatible."
        )
    if set(metadata) != {"format", "schema_version", "arrays", "payload"}:
        raise ValidationError(
            "Automatic workflow checkpoint metadata schema is invalid."
        )
    descriptors = metadata["arrays"]
    if not isinstance(descriptors, dict) or len(descriptors) > _MAX_ARCHIVE_MEMBERS - 1:
        raise ValidationError(
            "Automatic workflow checkpoint array metadata is invalid."
        )
    validated: dict[str, dict[str, Any]] = {}
    total_elements = 0
    for name, descriptor in descriptors.items():
        resolved = _validate_array_descriptor(name, descriptor)
        total_elements += resolved["elements"]
        if total_elements > _MAX_ARRAY_ELEMENTS:
            raise ValidationError(
                "Automatic workflow checkpoint arrays exceed element limit."
            )
        validated = {**validated, name: resolved}
    _validate_json_tree(metadata["payload"], depth=0, items=0)
    return validated


def _validate_array_descriptor(name: Any, descriptor: Any) -> dict[str, Any]:
    if not isinstance(name, str) or _ARRAY_ID_PATTERN.fullmatch(name) is None:
        raise ValidationError("Automatic workflow checkpoint array name is invalid.")
    if not isinstance(descriptor, dict) or set(descriptor) != {
        "path",
        "kind",
        "dtype",
        "shape",
        "elements",
        "sha256",
    }:
        raise ValidationError(
            "Automatic workflow checkpoint array descriptor is invalid."
        )
    if descriptor["path"] != f"arrays/{name}.npy":
        raise ValidationError("Automatic workflow checkpoint array path is invalid.")
    if descriptor["kind"] not in {"numpy", "torch"}:
        raise ValidationError("Automatic workflow checkpoint array kind is invalid.")
    dtype = _validated_dtype(descriptor["dtype"])
    shape = descriptor["shape"]
    if (
        not isinstance(shape, list)
        or len(shape) > _MAX_ARRAY_DIMENSIONS
        or not all(isinstance(value, int) and value >= 0 for value in shape)
    ):
        raise ValidationError("Automatic workflow checkpoint array shape is invalid.")
    elements = descriptor["elements"]
    if not isinstance(elements, int) or elements != math.prod(shape):
        raise ValidationError(
            "Automatic workflow checkpoint array element count is invalid."
        )
    digest = descriptor["sha256"]
    if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
        raise ValidationError("Automatic workflow checkpoint array digest is invalid.")
    return {**descriptor, "dtype": dtype.str, "shape": shape, "elements": elements}


def _read_array(
    archive: zipfile.ZipFile, descriptor: dict[str, Any]
) -> np.ndarray | torch.Tensor:
    content = archive.read(descriptor["path"])
    if hashlib.sha256(content).hexdigest() != descriptor["sha256"]:
        raise ValidationError("Automatic workflow checkpoint array digest mismatch.")
    _validate_npy_header(content, descriptor)
    try:
        loaded = np.load(io.BytesIO(content), allow_pickle=False)
    except (OSError, EOFError, ValueError) as exc:
        raise ValidationError(
            "Automatic workflow checkpoint array is invalid."
        ) from exc
    if not isinstance(loaded, np.ndarray):
        raise ValidationError("Automatic workflow checkpoint array is invalid.")
    if (
        loaded.dtype != np.dtype(descriptor["dtype"])
        or list(loaded.shape) != descriptor["shape"]
        or loaded.size != descriptor["elements"]
        or not _array_is_finite(loaded)
    ):
        raise ValidationError(
            "Automatic workflow checkpoint array metadata does not match content."
        )
    resolved = np.array(loaded, copy=True)
    if descriptor["kind"] == "numpy":
        return resolved
    try:
        return torch.from_numpy(resolved)
    except (TypeError, ValueError) as exc:
        raise ValidationError(
            "Automatic workflow checkpoint tensor dtype is unsupported."
        ) from exc


def _validate_npy_header(content: bytes, descriptor: dict[str, Any]) -> None:
    stream = io.BytesIO(content)
    try:
        major, minor = np.lib.format.read_magic(stream)
        if (major, minor) == (1, 0):
            shape, fortran_order, dtype = np.lib.format.read_array_header_1_0(stream)
        elif (major, minor) in {(2, 0), (3, 0)}:
            shape, fortran_order, dtype = np.lib.format.read_array_header_2_0(stream)
        else:
            raise ValueError("unsupported NPY version")
    except (EOFError, ValueError) as exc:
        raise ValidationError(
            "Automatic workflow checkpoint array header is invalid."
        ) from exc
    if (
        fortran_order
        or np.dtype(dtype) != np.dtype(descriptor["dtype"])
        or list(shape) != descriptor["shape"]
        or math.prod(shape) != descriptor["elements"]
    ):
        raise ValidationError(
            "Automatic workflow checkpoint array metadata does not match content."
        )
    if (
        len(content) - stream.tell()
        != descriptor["elements"] * np.dtype(dtype).itemsize
    ):
        raise ValidationError(
            "Automatic workflow checkpoint array byte length is invalid."
        )


def _encode_value(
    value: Any,
    *,
    entries: tuple[_ArrayEntry, ...],
    elements: int,
    depth: int,
    items: int,
) -> tuple[Any, tuple[_ArrayEntry, ...], int, int]:
    if depth > _MAX_PAYLOAD_DEPTH:
        raise ValidationError(
            "Automatic workflow checkpoint payload exceeds depth limit."
        )
    next_items = items + 1
    if next_items > _MAX_PAYLOAD_ITEMS:
        raise ValidationError(
            "Automatic workflow checkpoint payload exceeds item limit."
        )
    if isinstance(value, torch.Tensor):
        return _encode_array(
            value.detach().cpu().numpy(),
            kind="torch",
            entries=entries,
            elements=elements,
            items=next_items,
        )
    if isinstance(value, np.ndarray):
        return _encode_array(
            value,
            kind="numpy",
            entries=entries,
            elements=elements,
            items=next_items,
        )
    if isinstance(value, np.generic):
        return _encode_value(
            value.item(),
            entries=entries,
            elements=elements,
            depth=depth + 1,
            items=next_items,
        )
    if value is None or isinstance(value, (str, bool, int)):
        return value, entries, elements, next_items
    if isinstance(value, float):
        return (
            value if math.isfinite(value) else None,
            entries,
            elements,
            next_items,
        )
    if isinstance(value, (list, tuple)):
        return _encode_sequence(
            value,
            entries=entries,
            elements=elements,
            depth=depth,
            items=next_items,
        )
    if isinstance(value, dict):
        if not all(isinstance(key, str) for key in value):
            raise ValidationError(
                "Automatic workflow checkpoint dictionary keys must be strings."
            )
        return _encode_mapping(
            value,
            entries=entries,
            elements=elements,
            depth=depth,
            items=next_items,
        )
    raise ValidationError(
        "Automatic workflow checkpoint contains unsupported value "
        f"{type(value).__name__}."
    )


def _encode_sequence(
    values: list[Any] | tuple[Any, ...],
    *,
    entries: tuple[_ArrayEntry, ...],
    elements: int,
    depth: int,
    items: int,
) -> tuple[list[Any], tuple[_ArrayEntry, ...], int, int]:
    encoded: list[Any] = []
    current_entries = entries
    current_elements = elements
    current_items = items
    for value in values:
        item, current_entries, current_elements, current_items = _encode_value(
            value,
            entries=current_entries,
            elements=current_elements,
            depth=depth + 1,
            items=current_items,
        )
        encoded = [*encoded, item]
    return encoded, current_entries, current_elements, current_items


def _encode_mapping(
    values: dict[str, Any],
    *,
    entries: tuple[_ArrayEntry, ...],
    elements: int,
    depth: int,
    items: int,
) -> tuple[dict[str, Any], tuple[_ArrayEntry, ...], int, int]:
    encoded: dict[str, Any] = {}
    current_entries = entries
    current_elements = elements
    current_items = items
    for key, value in values.items():
        item, current_entries, current_elements, current_items = _encode_value(
            value,
            entries=current_entries,
            elements=current_elements,
            depth=depth + 1,
            items=current_items,
        )
        encoded = {**encoded, key: item}
    return encoded, current_entries, current_elements, current_items


def _encode_array(
    value: np.ndarray,
    *,
    kind: str,
    entries: tuple[_ArrayEntry, ...],
    elements: int,
    items: int,
) -> tuple[dict[str, str], tuple[_ArrayEntry, ...], int, int]:
    if len(entries) >= _MAX_ARCHIVE_MEMBERS - 1:
        raise ValidationError(
            "Automatic workflow checkpoint exceeds array count limit."
        )
    array = np.asarray(value)
    _validated_dtype(array.dtype.str)
    if array.ndim > _MAX_ARRAY_DIMENSIONS or not _array_is_finite(array):
        raise ValidationError("Automatic workflow checkpoint array values are invalid.")
    next_elements = elements + int(array.size)
    if next_elements > _MAX_ARRAY_ELEMENTS:
        raise ValidationError(
            "Automatic workflow checkpoint arrays exceed element limit."
        )
    contiguous = np.ascontiguousarray(array)
    stream = io.BytesIO()
    np.save(stream, contiguous, allow_pickle=False)
    content = stream.getvalue()
    name = f"array-{len(entries):04d}"
    descriptor = {
        "path": f"arrays/{name}.npy",
        "kind": kind,
        "dtype": contiguous.dtype.str,
        "shape": list(contiguous.shape),
        "elements": int(contiguous.size),
        "sha256": hashlib.sha256(content).hexdigest(),
    }
    entry = _ArrayEntry(name=name, descriptor=descriptor, content=content)
    return (
        {_ARRAY_REFERENCE_KEY: name},
        (*entries, entry),
        next_elements,
        items,
    )


def _decode_value(
    value: Any,
    arrays: dict[str, np.ndarray | torch.Tensor],
    *,
    used_arrays: frozenset[str],
    depth: int,
    items: int,
) -> tuple[Any, frozenset[str], int]:
    if depth > _MAX_PAYLOAD_DEPTH:
        raise ValidationError(
            "Automatic workflow checkpoint payload exceeds depth limit."
        )
    next_items = items + 1
    if next_items > _MAX_PAYLOAD_ITEMS:
        raise ValidationError(
            "Automatic workflow checkpoint payload exceeds item limit."
        )
    if isinstance(value, dict):
        if set(value) == {_ARRAY_REFERENCE_KEY}:
            name = value[_ARRAY_REFERENCE_KEY]
            if not isinstance(name, str) or name not in arrays or name in used_arrays:
                raise ValidationError(
                    "Automatic workflow checkpoint array reference is invalid."
                )
            return arrays[name], used_arrays | {name}, next_items
        if _ARRAY_REFERENCE_KEY in value:
            raise ValidationError(
                "Automatic workflow checkpoint array reference is invalid."
            )
        decoded: dict[str, Any] = {}
        current_used = used_arrays
        current_items = next_items
        for key, item in value.items():
            resolved, current_used, current_items = _decode_value(
                item,
                arrays,
                used_arrays=current_used,
                depth=depth + 1,
                items=current_items,
            )
            decoded = {**decoded, key: resolved}
        return decoded, current_used, current_items
    if isinstance(value, list):
        decoded_list: list[Any] = []
        current_used = used_arrays
        current_items = next_items
        for item in value:
            resolved, current_used, current_items = _decode_value(
                item,
                arrays,
                used_arrays=current_used,
                depth=depth + 1,
                items=current_items,
            )
            decoded_list = [*decoded_list, resolved]
        return decoded_list, current_used, current_items
    if value is None or isinstance(value, (str, bool, int)):
        return value, used_arrays, next_items
    if isinstance(value, float) and math.isfinite(value):
        return value, used_arrays, next_items
    raise ValidationError("Automatic workflow checkpoint payload is invalid.")


def _validate_json_tree(value: Any, *, depth: int, items: int) -> int:
    if depth > _MAX_PAYLOAD_DEPTH:
        raise ValidationError(
            "Automatic workflow checkpoint payload exceeds depth limit."
        )
    next_items = items + 1
    if next_items > _MAX_PAYLOAD_ITEMS:
        raise ValidationError(
            "Automatic workflow checkpoint payload exceeds item limit."
        )
    if isinstance(value, dict):
        if not all(isinstance(key, str) for key in value):
            raise ValidationError(
                "Automatic workflow checkpoint payload keys are invalid."
            )
        current_items = next_items
        for child in value.values():
            current_items = _validate_json_tree(
                child, depth=depth + 1, items=current_items
            )
        return current_items
    if isinstance(value, list):
        current_items = next_items
        for child in value:
            current_items = _validate_json_tree(
                child, depth=depth + 1, items=current_items
            )
        return current_items
    if value is None or isinstance(value, (str, bool, int)):
        return next_items
    if isinstance(value, float) and math.isfinite(value):
        return next_items
    raise ValidationError("Automatic workflow checkpoint payload is invalid.")


def _validated_dtype(value: Any) -> np.dtype[Any]:
    if not isinstance(value, str):
        raise ValidationError("Automatic workflow checkpoint array dtype is invalid.")
    try:
        dtype = np.dtype(value)
    except TypeError as exc:
        raise ValidationError(
            "Automatic workflow checkpoint array dtype is invalid."
        ) from exc
    if dtype.hasobject or dtype.kind not in {"b", "i", "u", "f"}:
        raise ValidationError(
            "Automatic workflow checkpoint array dtype is unsupported."
        )
    return dtype


def _array_is_finite(value: np.ndarray) -> bool:
    return value.dtype.kind != "f" or bool(np.isfinite(value).all())
