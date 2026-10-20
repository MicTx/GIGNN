#!/usr/bin/env python3
"""Generate the fixed Phase 2 synthetic spatial dataset for GIGNN experiments."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np
import pandas as pd

DATASET_NAME = "phase2_synthetic_lattice_v1"
CLASS_ORDER = ("low", "medium", "high")
GRID_SIZE = 9
FEATURE_COLUMNS = ("primary", "secondary", "noise")
SCHEMA_VERSION = "1"


def generate_phase2_dataset(*, seed: int = 20260714) -> pd.DataFrame:
    """Return a deterministic synthetic lattice with ordered labels and regions."""

    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    for index in range(GRID_SIZE * GRID_SIZE):
        row = index // GRID_SIZE
        column = index % GRID_SIZE
        region_row = row // 3
        region_column = column // 3
        region_score = float(region_row + region_column)
        primary = region_score + 0.15 * float(row) + 0.05 * float(column)
        secondary = 0.4 * region_score + 0.2 * float(column) + rng.normal(0.0, 0.05)
        noise = float(rng.normal(0.0, 1.0))
        if region_score <= 1.0:
            label = "low"
        elif region_score <= 3.0:
            label = "medium"
        else:
            label = "high"
        rows.append(
            {
                "node_id": f"unit-{index:03d}",
                "x": float(column),
                "y": float(row),
                "spatial_group": f"R{region_row}C{region_column}",
                "primary": float(primary),
                "secondary": float(secondary),
                "noise": noise,
                "label": label,
            }
        )
    frame = pd.DataFrame(rows)
    frame["label"] = pd.Categorical(
        frame["label"], categories=list(CLASS_ORDER), ordered=True
    )
    return frame


def dataset_schema() -> dict[str, object]:
    return {
        "schema_version": SCHEMA_VERSION,
        "dataset_name": DATASET_NAME,
        "license": "MIT",
        "source": "deterministic synthetic spatial lattice",
        "generator": "research/scripts/generate_phase2_dataset.py",
        "class_order": list(CLASS_ORDER),
        "spatial_unit_id": "node_id",
        "feature_columns": list(FEATURE_COLUMNS),
        "coordinate_columns": ["x", "y"],
        "group_column": "spatial_group",
        "target_column": "label",
        "row_count": GRID_SIZE * GRID_SIZE,
        "notes": [
            "No external remote-sensing or proprietary GIS data is used.",
            "Spatial groups are 3x3 blocks on a 9x9 lattice and must not cross partitions.",
            "Class order is explicit and fixed for ordinal encoding.",
        ],
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _package_version() -> str:
    try:
        return version("gignn")
    except PackageNotFoundError:
        return "0.1.0"


def write_phase2_dataset(
    output_dir: Path, *, seed: int = 20260714
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = generate_phase2_dataset(seed=seed)
    dataset_path = output_dir / "dataset.csv"
    schema_path = output_dir / "schema.json"
    manifest_path = output_dir / "manifest.json"

    frame_to_write = frame.copy()
    frame_to_write["label"] = frame_to_write["label"].astype(str)
    frame_to_write.to_csv(dataset_path, index=False)

    schema = dataset_schema()
    schema_path.write_text(
        json.dumps(schema, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    file_hashes = {
        "dataset.csv": _sha256_file(dataset_path),
        "schema.json": _sha256_file(schema_path),
    }
    manifest = {
        "dataset_name": DATASET_NAME,
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": datetime.now(UTC)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z"),
        "generator": "research/scripts/generate_phase2_dataset.py",
        "generator_parameters": {
            "seed": seed,
            "grid_size": GRID_SIZE,
            "class_order": list(CLASS_ORDER),
        },
        "license": "MIT",
        "package_version": _package_version(),
        "row_count": int(len(frame)),
        "class_counts": {
            label: int((frame["label"] == label).sum()) for label in CLASS_ORDER
        },
        "spatial_groups": sorted(frame["spatial_group"].astype(str).unique().tolist()),
        "files": file_hashes,
        "rebuild_command": (
            ".venv/bin/python research/scripts/generate_phase2_dataset.py "
            "--output-dir research/experiments"
        ),
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    manifest["files"]["manifest.json"] = _sha256_file(manifest_path)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("research/experiments"),
        help="Directory that receives dataset.csv, schema.json, and manifest.json",
    )
    parser.add_argument("--seed", type=int, default=20260714)
    args = parser.parse_args()
    manifest = write_phase2_dataset(args.output_dir, seed=args.seed)
    print(
        json.dumps(
            {"dataset_name": manifest["dataset_name"], "files": manifest["files"]},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
