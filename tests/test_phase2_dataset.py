from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

from research.scripts.generate_phase2_dataset import (
    CLASS_ORDER,
    generate_phase2_dataset,
    write_phase2_dataset,
)

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "research" / "experiments"
SCRIPT = ROOT / "research" / "scripts" / "generate_phase2_dataset.py"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_generate_phase2_dataset_is_deterministic_and_ordered():
    first = generate_phase2_dataset()
    second = generate_phase2_dataset()

    pd.testing.assert_frame_equal(first, second)
    assert list(first["label"].cat.categories) == list(CLASS_ORDER)
    assert first["node_id"].is_unique
    assert first["spatial_group"].nunique() == 9
    assert set(first["label"]) == set(CLASS_ORDER)
    assert len(first) == 81


def test_checked_in_phase2_dataset_matches_generator(tmp_path):
    write_phase2_dataset(tmp_path)
    for name in ("dataset.csv", "schema.json"):
        expected = EXPERIMENTS / name
        actual = tmp_path / name
        assert expected.is_file(), f"missing checked-in file: {name}"
        assert _sha256(expected) == _sha256(actual)

    schema = json.loads((EXPERIMENTS / "schema.json").read_text(encoding="utf-8"))
    manifest = json.loads((EXPERIMENTS / "manifest.json").read_text(encoding="utf-8"))
    assert schema["class_order"] == list(CLASS_ORDER)
    assert schema["license"] == "MIT"
    assert schema["spatial_unit_id"] == "node_id"
    assert manifest["files"]["dataset.csv"] == _sha256(EXPERIMENTS / "dataset.csv")
    assert manifest["files"]["schema.json"] == _sha256(EXPERIMENTS / "schema.json")


def test_phase2_dataset_rebuilds_in_clean_subprocess(tmp_path):
    output_dir = tmp_path / "rebuild"
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--output-dir", str(output_dir)],
        check=True,
        capture_output=True,
        text=True,
        cwd=ROOT,
    )
    assert "phase2_synthetic_lattice_v1" in result.stdout
    for name in ("dataset.csv", "schema.json"):
        assert _sha256(output_dir / name) == _sha256(EXPERIMENTS / name)
