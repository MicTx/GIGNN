#!/usr/bin/env python3
"""Run auditable five-group Phase 2 GIGNN ablations."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score

from gignn.estimators import GIGNNClassifier
from gignn.geodetector import GeoDetectorSelector
from gignn.graph import (
    build_pyg_data,
    build_spatial_edges,
    calculate_edge_weights,
    calculate_gaussian_edge_attributes,
)
from gignn.models import GIGNN
from gignn.splitting import SplitIndices, resolve_split_indices
from research.scripts.phase2_metrics import (
    breakpoint_change_rate,
    build_partition_geometry_edges,
    build_then_filter_edges,
    edge_hash,
    factor_score_change,
    mean_variance_ci,
    paired_difference_ci,
    ranking_consistency,
    selector_fingerprint,
    topology_change_rate,
    validation_optimism_bias,
)

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA = ROOT / "research" / "experiments"
DEFAULT_CONFIG = DEFAULT_DATA / "ablation_configs.json"


class Point:
    def __init__(self, x: float, y: float) -> None:
        self.x = x
        self.y = y
        self.centroid = self


def load_dataset(path: Path) -> tuple[pd.DataFrame, pd.Series, list[Point], pd.Series]:
    frame = pd.read_csv(path / "dataset.csv")
    features = frame[["primary", "secondary", "noise"]].copy()
    labels = pd.Series(frame["label"].astype(str), name="label")
    geometry = [Point(float(x), float(y)) for x, y in zip(frame["x"], frame["y"])]
    return features, labels, geometry, frame["spatial_group"].astype(str)


def package_version() -> str:
    try:
        return version("gignn")
    except PackageNotFoundError:
        return "0.1.0"


def code_sha() -> str:
    provided = os.environ.get("GIGNN_CODE_SHA")
    if provided:
        return provided
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            cwd=ROOT,
        ).stdout.strip()
    except subprocess.CalledProcessError:
        return "unversioned-clean-copy"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _seed_everything(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def _coordinates(geometry: list[Point]) -> pd.DataFrame:
    return pd.DataFrame(
        {"x": [point.x for point in geometry], "y": [point.y for point in geometry]}
    )


def _selector(
    features: pd.DataFrame,
    labels: pd.Series,
    split: SplitIndices,
    shared: dict[str, Any],
    scope: str,
) -> tuple[pd.DataFrame, GeoDetectorSelector]:
    fit_indices = np.arange(len(features)) if scope == "full" else split.train
    ordinal_labels = pd.Series(
        pd.Categorical(
            labels.iloc[fit_indices].reset_index(drop=True),
            categories=list(shared["class_order"]),
            ordered=True,
        )
    )
    selector = GeoDetectorSelector(
        discretizer=shared["detector_grid"]["discretizer"][0],
        max_bins=int(shared["detector_grid"]["max_bins"][0]),
        strategy="factor",
    ).fit(
        features.iloc[fit_indices].reset_index(drop=True),
        ordinal_labels,
    )
    return selector.transform(features), selector


def _model_params(shared: dict[str, Any], seed: int) -> dict[str, Any]:
    grid = shared["model_grid"]
    return {
        "hidden_channels": tuple(grid["hidden_channels"][0]),
        "learning_rate": float(grid["learning_rate"][0]),
        "max_epochs": int(grid["max_epochs"][0]),
        "random_state": seed,
        "class_order": tuple(shared["class_order"]),
    }


def _partition_edges(
    coordinates: pd.DataFrame, indices: np.ndarray, mode: str, k: int
) -> np.ndarray:
    if mode == "partition_independent":
        return build_partition_geometry_edges(coordinates, indices, k=k)
    if mode == "build_then_filter":
        return build_then_filter_edges(coordinates, indices, k=k)
    raise ValueError(f"unsupported partition graph mode: {mode}")


def _partition_candidate(
    selected: pd.DataFrame,
    labels: pd.Series,
    split: SplitIndices,
    coordinates: pd.DataFrame,
    shared: dict[str, Any],
    mode: str,
    k: int,
    seed: int,
) -> dict[str, Any]:
    train_edges = _partition_edges(coordinates, split.train, mode, k)
    validation_edges = _partition_edges(coordinates, split.validation, mode, k)
    test_edges = _partition_edges(coordinates, split.test, mode, k)
    estimator = GIGNNClassifier(**_model_params(shared, seed)).fit(
        selected.iloc[split.train].reset_index(drop=True),
        labels.iloc[split.train].reset_index(drop=True),
        edge_index=train_edges,
    )
    validation_predictions = estimator.predict(
        selected.iloc[split.validation].reset_index(drop=True),
        edge_index=validation_edges,
    )
    test_predictions = estimator.predict(
        selected.iloc[split.test].reset_index(drop=True), edge_index=test_edges
    )
    return {
        "k": k,
        "validation_accuracy": float(
            accuracy_score(labels.iloc[split.validation], validation_predictions)
        ),
        "test_predictions": test_predictions,
        "train_edges": train_edges,
    }


def _full_mask_candidate(
    selected: pd.DataFrame,
    labels: pd.Series,
    split: SplitIndices,
    coordinates: pd.DataFrame,
    shared: dict[str, Any],
    k: int,
    seed: int,
) -> dict[str, Any]:
    geometry = [
        Point(float(row.x), float(row.y)) for row in coordinates.itertuples(index=False)
    ]
    edges = build_spatial_edges(
        geometry=geometry, num_nodes=len(selected), strategy="knn", k=k
    )
    attributes = calculate_gaussian_edge_attributes(selected, edges, sigma=2.0)
    data = build_pyg_data(
        selected,
        labels,
        edges,
        edge_attr=attributes,
        edge_weight=calculate_edge_weights(attributes, mode="mean"),
        class_order=tuple(shared["class_order"]),
    )
    params = _model_params(shared, seed)
    model = GIGNN(
        in_channels=data.num_features,
        hidden_channels=params["hidden_channels"],
        out_channels=len(shared["class_order"]),
        num_neighbors_list=(6, 12),
        random_state=seed,
    )
    device = torch.device("cpu")
    data = data.to(device)
    model = model.to(device)
    mask = torch.zeros(len(selected), dtype=torch.bool, device=device)
    mask[
        torch.as_tensor(
            np.asarray(split.train, dtype=np.int64).copy(), dtype=torch.long
        )
    ] = True
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=params["learning_rate"])
    model.train()
    for _ in range(params["max_epochs"]):
        optimizer.zero_grad()
        logits = model(data.x, data.edge_index, edge_weight=data.edge_weight)
        loss = criterion(logits[mask], data.y[mask])
        loss.backward()
        optimizer.step()
    model.eval()
    with torch.no_grad():
        predictions = (
            model(data.x, data.edge_index, edge_weight=data.edge_weight)
            .argmax(dim=1)
            .cpu()
            .numpy()
        )
    classes = np.asarray(shared["class_order"], dtype=object)
    return {
        "k": k,
        "validation_accuracy": float(
            accuracy_score(
                labels.iloc[split.validation], classes[predictions[split.validation]]
            )
        ),
        "test_predictions": classes[predictions[split.test]],
        "train_edges": edges,
    }


def _heldout_topology_stability(
    coordinates: pd.DataFrame,
    split: SplitIndices,
    mode: str,
    k: int,
    train_edges: np.ndarray,
) -> bool:
    heldout = np.concatenate([split.validation, split.test])
    for index in heldout:
        moved = coordinates.copy()
        moved.loc[int(index), "x"] += 50.0
        if mode == "full_graph_mask":
            points = [
                Point(float(row.x), float(row.y))
                for row in moved.itertuples(index=False)
            ]
            shifted = build_spatial_edges(
                geometry=points, num_nodes=len(moved), strategy="knn", k=k
            )
        else:
            shifted = _partition_edges(moved, split.train, mode, k)
        if edge_hash(train_edges) != edge_hash(shifted):
            return False
    return True


def _cross_region_accuracy(
    labels: pd.Series,
    test_predictions: np.ndarray,
    groups: pd.Series,
    test_indices: np.ndarray,
) -> float:
    group_scores = []
    for group in sorted(groups.iloc[test_indices].unique()):
        positions = np.flatnonzero(groups.iloc[test_indices].to_numpy() == group)
        indices = test_indices[positions]
        group_scores.append(
            float(accuracy_score(labels.iloc[indices], test_predictions[positions]))
        )
    return float(np.mean(group_scores))


def run_configuration(
    config: dict[str, Any],
    shared: dict[str, Any],
    *,
    features: pd.DataFrame,
    labels: pd.Series,
    geometry: list[Point],
    groups: pd.Series,
    seed: int,
) -> dict[str, Any]:
    _seed_everything(seed)
    split = resolve_split_indices(
        labels,
        validation_fraction=float(shared["validation_fraction"]),
        test_fraction=float(shared["test_fraction"]),
        random_state=seed,
        spatial_groups=groups if config["split_mode"] == "spatial_groups" else None,
    )
    coordinates = _coordinates(geometry)
    selected, selector = _selector(
        features, labels, split, shared, config["selector_scope"]
    )
    candidates = []
    for k in shared["graph_grid"]["k"]:
        if config["graph_mode"] == "full_graph_mask":
            candidate = _full_mask_candidate(
                selected, labels, split, coordinates, shared, int(k), seed
            )
        else:
            candidate = _partition_candidate(
                selected,
                labels,
                split,
                coordinates,
                shared,
                config["graph_mode"],
                int(k),
                seed,
            )
        candidates.append(candidate)
    candidates.sort(
        key=lambda candidate: (-candidate["validation_accuracy"], candidate["k"])
    )
    best = candidates[0]
    test_accuracy = float(
        accuracy_score(labels.iloc[split.test], best["test_predictions"])
    )
    independent = build_partition_geometry_edges(
        coordinates, split.train, k=int(best["k"])
    )
    bins = {
        feature: [float(value) for value in selector.discretizers_[feature].bins]
        for feature in selector.selected_features_
    }
    return {
        "configuration_id": config["id"],
        "seed": seed,
        "selector_scope": config["selector_scope"],
        "graph_mode": config["graph_mode"],
        "split_mode": config["split_mode"],
        "selector_fingerprint": selector_fingerprint(bins),
        "selector_bins": bins,
        "selected_features": selector.selected_features_,
        "factor_scores": {
            feature: float(score) for feature, score in selector.scores_["q"].items()
        },
        "candidate_ranking": [f"knn:k={candidate['k']}" for candidate in candidates],
        "selected_k": int(best["k"]),
        "train_edge_hash": edge_hash(best["train_edges"]),
        "independent_train_edge_hash": edge_hash(independent),
        "topology_change_rate": topology_change_rate(best["train_edges"], independent),
        "validation_accuracy": float(best["validation_accuracy"]),
        "test_accuracy": test_accuracy,
        "validation_optimism_bias": validation_optimism_bias(
            float(best["validation_accuracy"]), test_accuracy
        ),
        "cross_region_accuracy": _cross_region_accuracy(
            labels, best["test_predictions"], groups, split.test
        ),
        "train_size": int(len(split.train)),
        "validation_size": int(len(split.validation)),
        "test_size": int(len(split.test)),
        "train_graph_stable_under_all_holdout_shifts": _heldout_topology_stability(
            coordinates,
            split,
            config["graph_mode"],
            int(best["k"]),
            best["train_edges"],
        ),
    }


def summarize(runs: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for run in runs:
        grouped.setdefault(run["configuration_id"], []).append(run)

    result: dict[str, Any] = {}
    for identifier, values in grouped.items():
        result[identifier] = {
            name: mean_variance_ci([run[name] for run in values])
            for name in (
                "validation_accuracy",
                "test_accuracy",
                "validation_optimism_bias",
                "cross_region_accuracy",
                "topology_change_rate",
            )
        }
        result[identifier]["ranking_consistency"] = ranking_consistency(
            [run["candidate_ranking"] for run in values]
        )
        result[identifier]["all_holdout_stability"] = all(
            run["train_graph_stable_under_all_holdout_shifts"] for run in values
        )

    baseline_by_seed = {run["seed"]: run for run in grouped["D_partition_independent"]}
    for identifier, values in grouped.items():
        paired_test = [
            baseline_by_seed[run["seed"]]["test_accuracy"] - run["test_accuracy"]
            for run in values
            if run["seed"] in baseline_by_seed
        ]
        paired_bias = [
            baseline_by_seed[run["seed"]]["validation_optimism_bias"]
            - run["validation_optimism_bias"]
            for run in values
            if run["seed"] in baseline_by_seed
        ]
        zeros = [0.0] * len(paired_test)
        result[identifier]["paired_test_diff_vs_d"] = paired_difference_ci(
            paired_test, zeros
        )
        result[identifier]["paired_bias_diff_vs_d"] = paired_difference_ci(
            paired_bias, zeros
        )

        breakpoint_changes = []
        factor_changes = []
        for run in values:
            if run["seed"] not in baseline_by_seed:
                continue
            baseline = baseline_by_seed[run["seed"]]
            breakpoint_changes.append(
                breakpoint_change_rate(run["selector_bins"], baseline["selector_bins"])
            )
            factor_changes.append(
                factor_score_change(run["factor_scores"], baseline["factor_scores"])[
                    "mean_abs_change"
                ]
            )
        if breakpoint_changes:
            result[identifier]["breakpoint_change_rate_vs_d"] = mean_variance_ci(
                breakpoint_changes
            )
            result[identifier]["factor_score_mean_abs_change_vs_d"] = mean_variance_ci(
                factor_changes
            )
    return result


def decide_effect_gate(summary: dict[str, Any]) -> dict[str, Any]:
    enhanced, legacy = (
        summary["D_partition_independent"],
        summary["C_build_then_filter"],
    )
    structure_ok = (
        enhanced["all_holdout_stability"] and not legacy["all_holdout_stability"]
    )
    paired = summary["C_build_then_filter"]["paired_test_diff_vs_d"]
    # D must beat C on the independent test with a paired 95% CI entirely above zero.
    test_stable_improvement = paired["ci_low"] > 0.0
    bias_improved = (
        enhanced["validation_optimism_bias"]["mean"]
        < legacy["validation_optimism_bias"]["mean"]
    )
    ranking_improved = enhanced["ranking_consistency"] > legacy["ranking_consistency"]
    branch = "enhanced" if structure_ok and test_stable_improvement else "conservative"
    rationale = (
        "Retain enhanced branch: D beats C on the independent test with a paired 95% CI above zero "
        "and preserves the training graph under all holdout shifts."
        if branch == "enhanced"
        else "Use conservative branch: D does not beat C with a paired 95% CI above zero on the "
        "independent test, so only graph-isolation auditability is supported."
    )
    return {
        "structure_invariant_passed": structure_ok,
        "paired_test_diff_ci_low": paired["ci_low"],
        "paired_test_diff_mean": paired["mean"],
        "bias_improved": bias_improved,
        "ranking_improved": ranking_improved,
        "branch": branch,
        "rationale": rationale,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_DATA / "runs")
    args = parser.parse_args()
    features, labels, geometry, groups = load_dataset(args.data_dir)
    payload = json.loads(args.config.read_text(encoding="utf-8"))
    shared = payload["shared_budget"]
    runs = [
        run_configuration(
            config,
            shared,
            features=features,
            labels=labels,
            geometry=geometry,
            groups=groups,
            seed=int(seed),
        )
        for config in payload["configurations"]
        for seed in shared["random_seeds"]
    ]
    summary, decision = summarize(runs), None
    decision = decide_effect_gate(summary)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        name: args.output_dir / name
        for name in ("runs.json", "summary.json", "effect_gate.json")
    }
    paths["runs.json"].write_text(
        json.dumps(runs, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    paths["summary.json"].write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    paths["effect_gate.json"].write_text(
        json.dumps(decision, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    manifest = {
        "created_at_utc": datetime.now(UTC)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z"),
        "code_sha": code_sha(),
        "package_version": package_version(),
        "dataset_manifest_hash": sha256_file(args.data_dir / "manifest.json"),
        "dataset_hash": sha256_file(args.data_dir / "dataset.csv"),
        "config_hash": sha256_file(args.config),
        "runner_hash": sha256_file(Path(__file__)),
        "metrics_hash": sha256_file(Path(__file__).with_name("phase2_metrics.py")),
        "output_hashes": {name: sha256_file(path) for name, path in paths.items()},
        "seed_budget": shared["random_seeds"],
        "configuration_ids": [row["id"] for row in payload["configurations"]],
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"decision": decision, "manifest": manifest}, indent=2))


if __name__ == "__main__":
    main()
