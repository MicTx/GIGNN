import math

import numpy as np
import pytest

import gignn.graph.builders as builders
from gignn.exceptions import ValidationError
from gignn.graph import (
    build_pyg_data,
    build_spatial_edges,
    calculate_edge_weights,
    calculate_gaussian_edge_attributes,
)


class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.centroid = self

    def touches(self, other):
        return abs(self.x - other.x) + abs(self.y - other.y) == 1

    def intersects(self, other):
        return self.touches(other)


def _brute_force_knn_edges(points, k):
    edges = []
    for source, point in enumerate(points):
        neighbors = sorted(
            (math.dist(point, other), target)
            for target, other in enumerate(points)
            if target != source
        )
        edges.extend((source, target) for _, target in neighbors[:k])
    return set(edges)


def test_user_edge_index_takes_priority(edge_index):
    geometry = [Point(0, 0), Point(100, 100)]

    built = build_spatial_edges(geometry=geometry, edge_index=edge_index, num_nodes=6)

    np.testing.assert_array_equal(built, edge_index)


def test_knn_builds_expected_edges():
    geometry = [Point(0, 0), Point(1, 0), Point(10, 0)]

    edge_index = build_spatial_edges(geometry=geometry, strategy="knn", k=1)

    assert edge_index.shape == (2, 3)
    assert set(map(tuple, edge_index.T)) == {(0, 1), (1, 0), (2, 1)}


def test_knn_does_not_query_all_nodes_when_k_is_small(monkeypatch):
    real_tree = builders.cKDTree
    query_sizes = []

    class TrackingTree:
        def __init__(self, points):
            self._tree = real_tree(points)

        def query(self, points, *, k):
            query_sizes.append(k)
            return self._tree.query(points, k=k)

        def query_ball_point(self, point, radius):
            return self._tree.query_ball_point(point, radius)

    monkeypatch.setattr(builders, "cKDTree", TrackingTree)
    geometry = [Point(index, index % 7) for index in range(64)]

    build_spatial_edges(geometry=geometry, strategy="knn", k=3)

    assert query_sizes == [4]


@pytest.mark.parametrize(
    ("points", "k"),
    [
        ([(0.0, 0.0), (1.0, 0.0), (3.0, 0.0), (7.0, 2.0)], 1),
        ([(0.0, 0.0), (0.0, 0.0), (1.0, 0.0), (-1.0, 0.0)], 1),
        ([(0.0, 0.0), (1.0, 0.0), (-1.0, 0.0), (0.0, 1.0)], 2),
        ([(0.0, 0.0), (1.0e150, 0.0), (-1.0e150, 0.0)], 5),
    ],
)
def test_knn_matches_brute_force_reference(points, k):
    geometry = [Point(x, y) for x, y in points]

    edge_index = build_spatial_edges(geometry=geometry, strategy="knn", k=k)

    assert set(map(tuple, edge_index.T)) == _brute_force_knn_edges(points, k)


@pytest.mark.parametrize("coordinate", [np.nan, np.inf, -np.inf])
def test_graph_builders_reject_non_finite_coordinates(coordinate):
    geometry = [Point(0.0, 0.0), Point(coordinate, 1.0)]

    with pytest.raises(ValidationError, match="finite"):
        build_spatial_edges(geometry=geometry, strategy="knn", k=1)


def test_distance_threshold_is_inclusive_and_bidirectional():
    geometry = [Point(0.0, 0.0), Point(3.0, 4.0), Point(20.0, 0.0)]

    edge_index = build_spatial_edges(
        geometry=geometry,
        strategy="distance",
        threshold=5.0,
    )

    assert set(map(tuple, edge_index.T)) == {(0, 1), (1, 0)}


def test_distance_threshold_builds_expected_edges():
    geometry = [Point(0, 0), Point(1, 0), Point(3, 0)]

    edge_index = build_spatial_edges(
        geometry=geometry, strategy="distance", threshold=1.5
    )

    assert set(map(tuple, edge_index.T)) == {(0, 1), (1, 0)}


def test_contiguity_uses_geometry_predicates():
    geometry = [Point(0, 0), Point(1, 0), Point(3, 0)]

    edge_index = build_spatial_edges(geometry=geometry, strategy="contiguity")

    assert set(map(tuple, edge_index.T)) == {(0, 1), (1, 0)}


def test_build_spatial_edges_requires_geometry_or_edge_index():
    with pytest.raises(ValidationError, match="geometry or edge_index"):
        build_spatial_edges()


def test_edge_index_without_num_nodes_or_geometry_raises_validation_error(edge_index):
    with pytest.raises(ValidationError, match="num_nodes or geometry"):
        build_spatial_edges(edge_index=edge_index)


def test_automatic_graph_construction_enforces_node_limit():
    geometry = [Point(i, 0) for i in range(3)]

    with pytest.raises(ValidationError, match="max_nodes"):
        build_spatial_edges(geometry=geometry, strategy="knn", k=1, max_nodes=2)


def test_automatic_graph_requires_geometry_to_match_num_nodes():
    geometry = [Point(i, 0) for i in range(2)]

    with pytest.raises(ValidationError, match="geometry length"):
        build_spatial_edges(geometry=geometry, num_nodes=3, strategy="knn", k=1)


def test_build_pyg_data_returns_data_object(indicator_frame, labels, edge_index):
    pytest.importorskip("torch")
    pytest.importorskip("torch_geometric")

    data = build_pyg_data(indicator_frame, labels, edge_index)

    assert data.x.shape == (6, 3)
    assert data.y.tolist() == [1, 2, 2, 0, 1, 0]
    assert data.edge_index.shape == (2, 6)
    assert data.label_encoder.classes_.tolist() == ["high", "low", "medium"]


def test_gaussian_edge_attributes_enter_pyg_data(indicator_frame, labels, edge_index):
    pytest.importorskip("torch")
    pytest.importorskip("torch_geometric")
    edge_attr = calculate_gaussian_edge_attributes(
        indicator_frame[["ndvi", "dem"]], edge_index, sigma=2.0
    )
    edge_weight = calculate_edge_weights(edge_attr, mode="mean")

    data = build_pyg_data(
        indicator_frame,
        labels,
        edge_index,
        edge_attr=edge_attr,
        edge_weight=edge_weight,
    )

    assert data.edge_attr.shape == (edge_index.shape[1], 2)
    assert data.edge_weight.shape == (edge_index.shape[1],)
    assert float(data.edge_weight.max()) <= 1.0
    assert float(data.edge_weight.min()) >= 0.0
