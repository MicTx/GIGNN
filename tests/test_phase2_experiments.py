from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "research" / "experiments" / "runs"
SCRIPT = ROOT / "research" / "scripts" / "run_phase2_experiments.py"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_phase2_effect_gate_downgrades_when_ci_crosses_zero():
    decision = json.loads((RUNS / "effect_gate.json").read_text(encoding="utf-8"))
    summary = json.loads((RUNS / "summary.json").read_text(encoding="utf-8"))

    # Structural invariant is the only effect we can support on this dataset:
    # D keeps the training graph fixed under every holdout-node shift; C does not.
    assert decision["structure_invariant_passed"] is True
    assert decision["branch"] == "conservative"
    assert decision["paired_test_diff_ci_low"] <= 0.0
    assert summary["D_partition_independent"]["all_holdout_stability"] is True
    assert summary["C_build_then_filter"]["all_holdout_stability"] is False
    # Selector-breakpoint change audit: A leaks validation/test rows into
    # discretization, so its breakpoints differ from D on every seed.
    assert (
        summary["A_full_discretize_then_split"]["breakpoint_change_rate_vs_d"]["mean"]
        == 1.0
    )
    assert summary["C_build_then_filter"]["breakpoint_change_rate_vs_d"]["mean"] == 0.0


def test_phase2_experiments_reproduce_structural_hashes(tmp_path):
    output_dir = tmp_path / "runs"
    worktree = tmp_path / "worktree"
    worktree.mkdir()
    shutil.copytree(
        ROOT,
        worktree / "source",
        ignore=shutil.ignore_patterns(".git", ".venv", "__pycache__", ".coverage"),
        dirs_exist_ok=True,
    )
    isolated_root = worktree / "source"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(isolated_root)
    env["GIGNN_CODE_SHA"] = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
        cwd=ROOT,
    ).stdout.strip()
    result = subprocess.run(
        [
            sys.executable,
            str(isolated_root / "research" / "scripts" / "run_phase2_experiments.py"),
            "--output-dir",
            str(output_dir),
        ],
        check=True,
        capture_output=True,
        text=True,
        cwd=isolated_root,
        env=env,
    )
    assert "enhanced" in result.stdout or "conservative" in result.stdout

    baseline_runs = json.loads((RUNS / "runs.json").read_text(encoding="utf-8"))
    reproduced_runs = json.loads((output_dir / "runs.json").read_text(encoding="utf-8"))
    assert len(baseline_runs) == len(reproduced_runs)
    for left, right in zip(baseline_runs, reproduced_runs):
        assert left["configuration_id"] == right["configuration_id"]
        assert left["seed"] == right["seed"]
        # Structural fingerprints must match exactly.
        assert left["train_edge_hash"] == right["train_edge_hash"]
        assert (
            left["independent_train_edge_hash"] == right["independent_train_edge_hash"]
        )
        assert left["selector_fingerprint"] == right["selector_fingerprint"]
        assert left["topology_change_rate"] == right["topology_change_rate"]
        assert (
            left["train_graph_stable_under_all_holdout_shifts"]
            == right["train_graph_stable_under_all_holdout_shifts"]
        )
        # GNN training can vary slightly across processes; keep a fixed float tolerance.
        assert abs(left["validation_accuracy"] - right["validation_accuracy"]) < 0.35
        assert abs(left["test_accuracy"] - right["test_accuracy"]) < 0.35

    baseline_decision = json.loads(
        (RUNS / "effect_gate.json").read_text(encoding="utf-8")
    )
    reproduced_decision = json.loads(
        (output_dir / "effect_gate.json").read_text(encoding="utf-8")
    )
    assert reproduced_decision["structure_invariant_passed"] is True
    assert reproduced_decision["branch"] == baseline_decision["branch"]
