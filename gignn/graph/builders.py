from __future__ import annotations

import math
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from gignn.exceptions import ValidationError
from gignn.limits import MAX_AUTO_GRAPH_NODES, MAX_EDGES
from gignn.validation import validate_edge_index


def build_spatial_edges(
    geometry: Sequence[object] | None = None,
    *,
    edge_index: object | None = None,
    num_nodes: int | None = None,
    strategy: str = "knn",
    k: int = 8,
    threshold: float | None = None,
    max_nodes: int = MAX_AUTO_GRAPH_NODES,
    max_edges: int = MAX_EDGES,
) -> np.ndarray:
    if edge_index is not None:
        if num_nodes is None:
            if geometry is None:
                raise ValidationError(
                    "num_nodes or geometry must be provided with edge_index."
                )
            num_nodes = len(geometry)
        return validate_edge_index(edge_index, num_nodes=num_nodes, max_edges=max_edges)
    if geometry is None:
        raise ValidationError("Either geometry or edge_index must be provided.")

    if num_nodes is not None and len(geometry) != num_nodes:
        raise ValidationError("geometry length must match num_nodes.")
    if len(geometry) > max_nodes:
        raise ValidationError(
            "geometry exceeds max_nodes for automatic graph construction."
        )
    points = [_centroid_xy(item) for item in geometry]
    if strategy == "knn":
        return _knn_edges(points, k=k)
    if strategy == "distance":
        if threshold is None:
            raise ValidationError(
                "threshold must be provided for distance graph building."
            )
        return _distance_edges(points, threshold=threshold)
    if strategy == "contiguity":
        return _contiguity_edges(geometry)
    raise ValidationError(f"Unknown graph building strategy {strategy!r}.")


def build_adjacency_dict(edge_index: object, *, num_nodes: int) -> dict[int, list[int]]:
    edges = validate_edge_index(edge_index, num_nodes=num_nodes)
    adjacency = {index: [] for index in range(num_nodes)}
    for source, target in edges.T:
        source_int = int(source)
        target_int = int(target)
        adjacency[source_int].append(target_int)
        if source_int != target_int:
            adjacency[target_int].append(source_int)
    return {node: sorted(set(neighbors)) for node, neighbors in adjacency.items()}


def calculate_gaussian_edge_attributes(
    node_features: pd.DataFrame | np.ndarray,
    edge_index: object,
    *,
    sigma: float = 2.0,
) -> np.ndarray:
    """Per-feature Gaussian kernels ``exp(-(x_ik - x_jk)^2 / (2 sigma^2))``.

    Returns an ``(num_edges, num_features)`` matrix, the multi-dimensional
    spatial weights of the paper (Section 6.2). The default ``sigma=2`` follows
    Eq. (2) and assumes GeoDetector stratum codes (``1..L``) as features; on raw
    measurement scales the kernel collapses towards zero.
    """
    if sigma <= 0:
        raise ValidationError("sigma must be positive.")
    features = np.asarray(node_features, dtype=float)
    if features.ndim != 2 or features.shape[0] == 0 or features.shape[1] == 0:
        raise ValidationError("node_features must be a non-empty 2D array.")
    if not np.isfinite(features).all():
        raise ValidationError("node_features must contain only finite values.")
    edges = validate_edge_index(edge_index, num_nodes=features.shape[0])
    source = features[edges[0]]
    target = features[edges[1]]
    squared_diff = np.square(source - target)
    return np.exp(-squared_diff / (2 * sigma**2))


def calculate_edge_weights(
    edge_attributes: pd.DataFrame | np.ndarray,
    *,
    mode: str = "product",
) -> np.ndarray:
    """Reduce per-feature edge attributes to one scalar weight per edge.

    ``mode="product"`` (default) reproduces Eq. (2) of the paper exactly:
    ``prod_k exp(-(x_ik - x_jk)^2 / (2 sigma^2)) = exp(-||x_i - x_j||^2 /
    (2 sigma^2))``. ``mean``, ``min``, and ``max`` are non-paper alternatives.
    """
    attributes = np.asarray(edge_attributes, dtype=float)
    if attributes.ndim == 1:
        weights = attributes
    elif attributes.ndim == 2 and attributes.shape[0] > 0 and attributes.shape[1] > 0:
        if mode == "mean":
            weights = attributes.mean(axis=1)
        elif mode == "product":
            weights = attributes.prod(axis=1)
        elif mode == "min":
            weights = attributes.min(axis=1)
        elif mode == "max":
            weights = attributes.max(axis=1)
        else:
            raise ValidationError(f"Unknown edge weight mode {mode!r}.")
    else:
        raise ValidationError("edge_attributes must be a non-empty 1D or 2D array.")
    if not np.isfinite(weights).all():
        raise ValidationError("edge weights must contain only finite values.")
    return np.asarray(weights, dtype=float)


def filter_isolated_nodes(
    adjacency_dict: dict[int, list[int]],
) -> tuple[dict[int, list[int]], np.ndarray]:
    kept_nodes = [node for node, neighbors in adjacency_dict.items() if neighbors]
    mapping = {old: new for new, old in enumerate(kept_nodes)}
    filtered = {}
    for old in kept_nodes:
        filtered[mapping[old]] = [
            mapping[neighbor] for neighbor in adjacency_dict[old] if neighbor in mapping
        ]
    return filtered, np.asarray(kept_nodes, dtype=np.int64)


def _centroid_xy(geometry: object) -> tuple[float, float]:
    centroid = getattr(geometry, "centroid", geometry)
    try:
        return float(getattr(centroid, "x")), float(getattr(centroid, "y"))
    except (TypeError, ValueError, AttributeError) as exc:
        raise ValidationError(
            "geometry items must expose numeric x/y or centroid.x/centroid.y."
        ) from exc


def _knn_edges(points: Sequence[tuple[float, float]], k: int) -> np.ndarray:
    if k < 1:
        raise ValidationError("k must be at least 1.")
    points_array = np.asarray(points, dtype=float)
    n = len(points_array)
    if n == 0:
        raise ValidationError("Graph construction produced no edges.")
    if not np.isfinite(points_array).all():
        raise ValidationError("geometry coordinates must contain only finite values.")
    effective_k = min(k, n - 1)
    if effective_k < 1:
        return _edges_to_array([], n)
    tree = cKDTree(points_array)
    distances, _ = tree.query(points_array, k=effective_k + 1)
    edges = []
    for source, point in enumerate(points):
        boundary_distance = float(distances[source, effective_k])
        candidate_indices = tree.query_ball_point(
            points_array[source], np.nextafter(boundary_distance, np.inf)
        )
        neighbors = sorted(
            (_distance(point, points[target]), target)
            for target in candidate_indices
            if target != source
        )
        for _, target in neighbors[:effective_k]:
            edges.append((source, target))
    return _edges_to_array(edges, n)


def _distance_edges(
    points: Sequence[tuple[float, float]], threshold: float
) -> np.ndarray:
    if threshold <= 0:
        raise ValidationError("threshold must be positive.")
    points_array = np.asarray(points, dtype=float)
    n = len(points_array)
    if n == 0:
        raise ValidationError("Graph construction produced no edges.")
    if not np.isfinite(points_array).all():
        raise ValidationError("geometry coordinates must contain only finite values.")
    tree = cKDTree(points_array)
    pairs = tree.query_pairs(r=threshold, output_type="set")
    edges = []
    for source, target in pairs:
        edges.append((source, target))
        edges.append((target, source))
    return _edges_to_array(edges, n)


def _contiguity_edges(geometry: Sequence[object]) -> np.ndarray:
    edges = []
    for source, source_geometry in enumerate(geometry):
        for target, target_geometry in enumerate(geometry):
            if source == target:
                continue
            touches = getattr(source_geometry, "touches", None)
            intersects = getattr(source_geometry, "intersects", None)
            if callable(touches) and touches(target_geometry):
                edges.append((source, target))
            elif callable(intersects) and intersects(target_geometry):
                edges.append((source, target))
    return _edges_to_array(edges, len(geometry))


def _distance(left: tuple[float, float], right: tuple[float, float]) -> float:
    return math.dist(left, right)


def _edges_to_array(edges: Iterable[tuple[int, int]], num_nodes: int) -> np.ndarray:
    unique_edges = sorted(set(edges))
    if not unique_edges:
        raise ValidationError("Graph construction produced no edges.")
    return validate_edge_index(
        np.asarray(unique_edges, dtype=np.int64).T, num_nodes=num_nodes
    )
