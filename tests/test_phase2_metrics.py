from __future__ import annotations

import numpy as np
import pandas as pd

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


def test_edge_hash_and_topology_change_rate_are_order_invariant():
    left = np.array([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=np.int64)
    right = left[:, ::-1]
    assert edge_hash(left) == edge_hash(right)
    assert topology_change_rate(left, right) == 0.0
    altered = np.array([[0, 1], [1, 0]], dtype=np.int64)
    assert topology_change_rate(left, altered) > 0.0


def test_validation_bias_ranking_and_ci():
    assert abs(validation_optimism_bias(0.9, 0.7) - 0.2) < 1e-12
    assert ranking_consistency([["a", "b"], ["a", "b"], ["b", "a"]]) == 2 / 3
    summary = mean_variance_ci([0.5, 0.6, 0.7])
    assert summary["mean"] == 0.6
    assert summary["variance"] > 0.0
    assert summary["ci_low"] < summary["mean"] < summary["ci_high"]


def test_partition_independent_edges_ignore_holdout_geometry():
    coordinates = pd.DataFrame(
        {
            "x": [0.0, 1.0, 2.0, 100.0, 101.0, 102.0],
            "y": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        }
    )
    train = np.array([0, 1, 2], dtype=np.int64)
    independent = build_partition_geometry_edges(coordinates, train, k=1)
    moved = coordinates.copy()
    moved.loc[3:, "x"] = [0.1, 0.2, 0.3]
    independent_moved = build_partition_geometry_edges(moved, train, k=1)
    assert edge_hash(independent) == edge_hash(independent_moved)

    filtered = build_then_filter_edges(coordinates, train, k=1)
    filtered_moved = build_then_filter_edges(moved, train, k=1)
    assert edge_hash(filtered) != edge_hash(filtered_moved)
    assert selector_fingerprint({"primary": [0.0, 1.0]}) == selector_fingerprint(
        {"primary": [0.0, 1.0]}
    )


def test_breakpoint_factor_change_and_paired_ci():
    left_bins = {"a": [0.0, 1.0], "b": [0.0, 2.0, 4.0]}
    right_bins = {"a": [0.0, 1.0], "b": [0.0, 2.0]}
    assert breakpoint_change_rate(left_bins, left_bins) == 0.0
    assert breakpoint_change_rate(left_bins, right_bins) == 0.5

    factor = factor_score_change({"a": 0.4, "b": 0.2}, {"a": 0.3, "b": 0.2})
    assert abs(factor["per_feature"]["a"] - 0.1) < 1e-12
    assert factor["per_feature"]["b"] == 0.0
    assert abs(factor["mean_abs_change"] - 0.05) < 1e-12

    paired = paired_difference_ci([0.6, 0.5, 0.4], [0.4, 0.4, 0.4])
    assert paired["mean"] > 0.0
    assert paired["n_positive"] == 2.0
    assert paired["n_zero"] == 1.0
    assert paired["ci_low"] < paired["mean"] < paired["ci_high"]
