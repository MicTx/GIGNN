from __future__ import annotations

import operator
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

INFERENCE_MODES = ("full", "sampled")


@dataclass(frozen=True)
class _NeighborTable:
    indptr: np.ndarray
    indices: np.ndarray
    weights: np.ndarray


@dataclass(frozen=True)
class _FullNeighborhood:
    source: torch.Tensor
    target: torch.Tensor
    weight: torch.Tensor | None
    count: torch.Tensor
    num_nodes: int


def multihop_sampling(
    src_nodes: Sequence[int] | np.ndarray,
    num_neighbors_list: Sequence[int],
    adjacency_dict: Mapping[int, Sequence[int]],
    *,
    random_state: int | None = None,
) -> list[np.ndarray]:
    """Sample fixed-size multi-hop neighborhoods for each source node.

    Follows the GraphSAGE sampler used in the reference notebook: for every
    node, ``num_neighbors_list[hop]`` neighbors are drawn uniformly without
    replacement when the node has at least that many distinct neighbors, with
    replacement when it has fewer, and the node itself is repeated when it has
    no neighbors (self-oversampling).

    Reproducibility contract:

    - ``random_state`` is an ``int``: draws come from
      ``np.random.default_rng(int)``, independent of NumPy's global RNG, so the
      same int always produces the same sampled neighborhoods.
    - ``random_state`` is ``None``: draws consume NumPy's global RNG, so
      ``np.random.seed`` (or ``set_random_seed``) replays the sampling sequence.
    """
    sample_nums = _validate_sample_nums(num_neighbors_list)
    nodes = np.asarray(src_nodes, dtype=np.int64)
    neighbor_lists = _neighbor_lists(adjacency_dict)
    known = [
        *neighbor_lists,
        *(v for values in neighbor_lists.values() for v in values),
    ]
    num_nodes = max([int(nodes.max()) if nodes.size else -1, *known], default=-1) + 1
    if (nodes.size and nodes.min() < 0) or any(value < 0 for value in known):
        raise ValueError("node ids must be non-negative integers.")
    table = _table_from_lists(neighbor_lists, num_nodes)
    rng = None if random_state is None else np.random.default_rng(random_state)
    sampled_nodes, _ = _multihop(table, nodes, sample_nums, rng)
    return sampled_nodes


class NeighborAggregator(nn.Module):
    """Aggregate neighbor messages ``A_k`` of Eq. (4) and apply ``W``.

    With edge weights ``w_uv`` (Eq. (2) Gaussian kernel), the ``mean``
    aggregator returns ``(1 / |N(v)|) * sum_u w_uv * h_u``: the weights scale
    each neighbor's message, so dissimilar neighbors contribute smaller
    increments. ``normalize_weights=True`` divides by ``sum_u w_uv`` instead
    (a kernel-weighted average, the legacy behavior kept for checkpoints
    saved before this option existed), which only re-balances
    neighbors relative to each other.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        *,
        use_bias: bool = False,
        aggr_method: str = "mean",
        normalize_weights: bool = False,
    ) -> None:
        super().__init__()
        if aggr_method not in {"mean", "sum", "max"}:
            raise ValueError("aggr_method must be one of 'mean', 'sum', or 'max'.")
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_bias = use_bias
        self.aggr_method = aggr_method
        self.normalize_weights = bool(normalize_weights)
        self.weight = nn.Parameter(torch.empty(input_dim, output_dim))
        if self.use_bias:
            self.bias = nn.Parameter(torch.empty(output_dim))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight)
        if self.use_bias:
            nn.init.zeros_(self.bias)

    def forward(
        self,
        neighbor_feature: torch.Tensor,
        neighbor_weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        weights = (
            None
            if neighbor_weight is None
            else neighbor_weight.to(neighbor_feature.device).unsqueeze(-1)
        )
        if self.aggr_method == "max":
            aggr_neighbor = _sampled_max(neighbor_feature, weights)
        else:
            weighted = (
                neighbor_feature if weights is None else neighbor_feature * weights
            )
            if self.aggr_method == "sum":
                aggr_neighbor = weighted.sum(dim=1)
            elif weights is not None and self.normalize_weights:
                aggr_neighbor = weighted.sum(dim=1) / weights.sum(dim=1).clamp_min(
                    1e-12
                )
            else:
                aggr_neighbor = weighted.mean(dim=1)
        return self._transform(aggr_neighbor)

    def forward_full(
        self,
        node_features: torch.Tensor,
        neighborhood: _FullNeighborhood,
        *,
        sample_num: int,
    ) -> torch.Tensor:
        """Aggregate over every neighbor; the expectation of the sampled path."""
        messages = node_features[neighborhood.source]
        if self.aggr_method == "max":
            return self._transform(_full_max(node_features, messages, neighborhood))
        if neighborhood.weight is not None:
            messages = messages * neighborhood.weight.unsqueeze(-1)
        summed = node_features.new_zeros(
            (neighborhood.num_nodes, node_features.size(1))
        ).index_add_(0, neighborhood.target, messages)
        if self.aggr_method == "sum":
            scale = sample_num / neighborhood.count
            return self._transform(summed * scale.unsqueeze(-1))
        if neighborhood.weight is not None and self.normalize_weights:
            denominator = (
                node_features.new_zeros(neighborhood.num_nodes)
                .index_add_(0, neighborhood.target, neighborhood.weight)
                .clamp_min(1e-12)
            )
        else:
            denominator = neighborhood.count
        return self._transform(summed / denominator.unsqueeze(-1))

    def _transform(self, aggr_neighbor: torch.Tensor) -> torch.Tensor:
        neighbor_hidden = torch.matmul(aggr_neighbor, self.weight)
        if self.use_bias:
            neighbor_hidden += self.bias
        return neighbor_hidden


class GCN(nn.Module):
    """GIGNN layer: ``ACT(W_self h_v (+|concat) W_neigh A({h_u}))``, Eq. (4)-(5)."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        *,
        activation=F.relu,
        aggr_neighbor_method: str = "mean",
        aggr_hidden_method: str = "sum",
        normalize_edge_weights: bool = False,
    ) -> None:
        super().__init__()
        if aggr_hidden_method not in {"sum", "concat"}:
            raise ValueError("aggr_hidden_method must be 'sum' or 'concat'.")
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = hidden_dim if aggr_hidden_method == "sum" else hidden_dim * 2
        self.aggr_neighbor_method = aggr_neighbor_method
        self.aggr_hidden_method = aggr_hidden_method
        self.activation = activation
        self.aggregator = NeighborAggregator(
            input_dim,
            hidden_dim,
            aggr_method=aggr_neighbor_method,
            normalize_weights=normalize_edge_weights,
        )
        self.weight = nn.Parameter(torch.empty(input_dim, hidden_dim))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight)

    def forward(
        self,
        src_node_features: torch.Tensor,
        neighbor_node_features: torch.Tensor,
        neighbor_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        neighbor_hidden = self.aggregator(neighbor_node_features, neighbor_weights)
        return self._combine(src_node_features, neighbor_hidden)

    def forward_full(
        self,
        node_features: torch.Tensor,
        neighborhood: _FullNeighborhood,
        *,
        sample_num: int,
    ) -> torch.Tensor:
        neighbor_hidden = self.aggregator.forward_full(
            node_features, neighborhood, sample_num=sample_num
        )
        return self._combine(node_features, neighbor_hidden)

    def _combine(
        self, src_node_features: torch.Tensor, neighbor_hidden: torch.Tensor
    ) -> torch.Tensor:
        self_hidden = torch.matmul(src_node_features, self.weight)
        if self.aggr_hidden_method == "sum":
            hidden = self_hidden + neighbor_hidden
        else:
            hidden = torch.cat([self_hidden, neighbor_hidden], dim=1)
        return self.activation(hidden) if self.activation else hidden


class GIGNN(nn.Module):
    """Geographically informed graph neural network (Ma, Zhang & Song, 2025).

    Training follows GraphSAGE mini-batch semantics: every training forward
    pass draws fresh fixed-size neighborhoods (``num_neighbors_list``), from a
    generator seeded by ``random_state`` when it is an int. ``reset_sampling``
    restarts that stream, so repeated fits are reproducible.

    Evaluation-mode forward passes use ``inference``:

    - ``"full"`` (default): deterministic aggregation over every neighbor,
      i.e. Eq. (4)-(5) with the complete neighborhood ``N(v)``; for the
      unnormalized ``mean`` and ``sum`` aggregators this equals the
      expectation of the sampled estimator used during training.
    - ``"sampled"``: notebook-style sampled inference; deterministic only when
      ``random_state`` is an int.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: tuple[int, ...],
        out_channels: int,
        *,
        num_neighbors_list: Sequence[int] = (6, 12),
        aggr_neighbor_method: str = "mean",
        aggr_hidden_method: str = "sum",
        dropout: float = 0.0,
        random_state: int | None = None,
        normalize_edge_weights: bool = False,
        inference: str = "full",
    ) -> None:
        super().__init__()
        self.input_dim = in_channels
        self.hidden_channels = tuple(hidden_channels)
        self.out_channels = out_channels
        self.num_neighbors_list = _validate_sample_nums(num_neighbors_list)
        layer_dims = (*self.hidden_channels, out_channels)
        if len(layer_dims) != len(self.num_neighbors_list):
            raise ValueError(
                "num_neighbors_list length must equal len(hidden_channels) + 1."
            )
        if inference not in INFERENCE_MODES:
            raise ValueError("inference must be 'full' or 'sampled'.")
        self.random_state = random_state
        self.normalize_edge_weights = bool(normalize_edge_weights)
        self.inference = inference
        self._sampling_rng: np.random.Generator | None = None
        self.dropout = nn.Dropout(dropout)
        self.gcn = nn.ModuleList()
        input_dim = in_channels
        for index, output_dim in enumerate(layer_dims):
            layer = GCN(
                input_dim,
                output_dim,
                activation=F.relu if index < len(layer_dims) - 1 else None,
                aggr_neighbor_method=aggr_neighbor_method,
                aggr_hidden_method=aggr_hidden_method,
                normalize_edge_weights=normalize_edge_weights,
            )
            self.gcn.append(layer)
            input_dim = layer.output_dim

    def reset_sampling(self) -> None:
        """Restart the training neighbor-sampling stream from ``random_state``."""
        self._sampling_rng = None

    def forward(
        self,
        x: torch.Tensor | list[torch.Tensor],
        edge_index: torch.Tensor | None = None,
        edge_weight: torch.Tensor | None = None,
        *,
        adjacency_dict: Mapping[int, Sequence[int]] | None = None,
        edge_weight_list: list[torch.Tensor] | None = None,
    ) -> torch.Tensor:
        if isinstance(x, list):
            return self._propagate_sampled(list(x), edge_weight_list)
        if edge_index is None:
            raise ValueError("edge_index is required when x is a tensor.")
        table = self._neighbor_table(x, edge_index, edge_weight, adjacency_dict)
        weighted = edge_weight is not None
        if not self.training and self.inference == "full":
            return self._propagate_full(x, table, weighted=weighted)
        rng = self._training_rng() if self.training else self._inference_rng()
        nodes, weights = _multihop(
            table, np.arange(x.size(0), dtype=np.int64), self.num_neighbors_list, rng
        )
        node_features_list = [
            x[torch.as_tensor(hop_nodes, dtype=torch.long, device=x.device)]
            for hop_nodes in nodes
        ]
        weight_list = (
            [
                torch.as_tensor(hop_weights, dtype=x.dtype, device=x.device)
                for hop_weights in weights
            ]
            if weighted
            else None
        )
        return self._propagate_sampled(node_features_list, weight_list)

    def _propagate_sampled(
        self,
        node_features_list: list[torch.Tensor],
        edge_weight_list: list[torch.Tensor] | None,
    ) -> torch.Tensor:
        hidden = list(node_features_list)
        for layer_index, gcn in enumerate(self.gcn):
            next_hidden = []
            for hop in range(len(self.num_neighbors_list) - layer_index):
                src_node_features = hidden[hop]
                src_node_num = src_node_features.size(0)
                neighbor_node_features = hidden[hop + 1].view(
                    src_node_num,
                    self.num_neighbors_list[hop],
                    -1,
                )
                neighbor_weights = None
                if edge_weight_list is not None:
                    neighbor_weights = edge_weight_list[hop].view(
                        src_node_num,
                        self.num_neighbors_list[hop],
                    )
                h = gcn(src_node_features, neighbor_node_features, neighbor_weights)
                if layer_index < len(self.gcn) - 1:
                    h = self.dropout(h)
                next_hidden.append(h)
            hidden = next_hidden
        return hidden[-1]

    def _propagate_full(
        self, x: torch.Tensor, table: _NeighborTable, *, weighted: bool
    ) -> torch.Tensor:
        neighborhood = _full_neighborhood(table, x, weighted=weighted)
        hidden = x
        last = len(self.gcn) - 1
        for layer_index, gcn in enumerate(self.gcn):
            hidden = gcn.forward_full(
                hidden,
                neighborhood,
                sample_num=self.num_neighbors_list[last - layer_index],
            )
            if layer_index < last:
                hidden = self.dropout(hidden)
        return hidden

    def _training_rng(self) -> np.random.Generator | None:
        if self.random_state is None:
            return None
        if self._sampling_rng is None:
            self._sampling_rng = np.random.default_rng(self.random_state)
        return self._sampling_rng

    def _inference_rng(self) -> np.random.Generator | None:
        if self.random_state is None:
            return None
        return np.random.default_rng(self.random_state)

    def _neighbor_table(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor | None,
        adjacency_dict: Mapping[int, Sequence[int]] | None,
    ) -> _NeighborTable:
        edges = np.asarray(_to_numpy(edge_index), dtype=np.int64)
        weights = None if edge_weight is None else _to_numpy(edge_weight)
        table = _table_from_edges(edges, x.size(0), weights)
        if adjacency_dict is None:
            return table
        return _table_from_adjacency(adjacency_dict, x.size(0), table)


def _validate_sample_nums(num_neighbors_list: Sequence[int]) -> tuple[int, ...]:
    values = []
    for sample_num in num_neighbors_list:
        if isinstance(sample_num, (bool, np.bool_)):
            raise ValueError("num_neighbors_list must contain positive integers.")
        try:
            value = operator.index(sample_num)
        except TypeError as exc:
            raise ValueError(
                "num_neighbors_list must contain positive integers."
            ) from exc
        if value <= 0:
            raise ValueError("num_neighbors_list must contain positive integers.")
        values.append(int(value))
    return tuple(values)


def _to_numpy(values: object) -> np.ndarray:
    if isinstance(values, torch.Tensor):
        return values.detach().cpu().numpy()
    return np.asarray(values)


def _neighbor_lists(
    adjacency_dict: Mapping[int, Sequence[int]],
) -> dict[int, list[int]]:
    return {
        int(node): sorted({int(value) for value in neighbors})
        for node, neighbors in adjacency_dict.items()
    }


def _table_from_lists(
    neighbor_lists: dict[int, list[int]], num_nodes: int
) -> _NeighborTable:
    counts = np.zeros(num_nodes, dtype=np.int64)
    for node, neighbors in neighbor_lists.items():
        counts[node] = len(neighbors)
    indptr = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
    indices = np.empty(int(indptr[-1]), dtype=np.int64)
    for node, neighbors in neighbor_lists.items():
        indices[indptr[node] : indptr[node + 1]] = neighbors
    return _NeighborTable(indptr, indices, np.ones(indices.size, dtype=float))


def _table_from_edges(
    edge_index: np.ndarray, num_nodes: int, edge_weight: np.ndarray | None
) -> _NeighborTable:
    """Symmetric, de-duplicated CSR neighbor table with per-edge weights.

    Edges are treated as undirected: an implied reverse edge inherits the
    weight of its explicit edge unless the reverse edge is listed explicitly.
    """
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError("edge_index must have shape (2, num_edges).")
    source, target = edge_index[0], edge_index[1]
    if source.size and (
        min(source.min(), target.min()) < 0
        or max(source.max(), target.max()) >= num_nodes
    ):
        raise ValueError("edge_index contains node indices out of range.")
    if edge_weight is None:
        weights = np.ones(source.size, dtype=float)
    else:
        weights = np.asarray(edge_weight, dtype=float).reshape(-1)
        if weights.size != source.size:
            raise ValueError(
                "edge_weight must contain one value per edge in edge_index."
            )
    mirrored = source != target
    rows = np.concatenate([source, target[mirrored]])
    cols = np.concatenate([target, source[mirrored]])
    values = np.concatenate([weights, weights[mirrored]])
    _, first = np.unique(rows * num_nodes + cols, return_index=True)
    rows, cols, values = rows[first], cols[first], values[first]
    counts = np.bincount(rows, minlength=num_nodes)
    indptr = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
    return _NeighborTable(indptr, cols.astype(np.int64), values)


def _table_from_adjacency(
    adjacency_dict: Mapping[int, Sequence[int]],
    num_nodes: int,
    edge_table: _NeighborTable,
) -> _NeighborTable:
    neighbor_lists = _neighbor_lists(adjacency_dict)
    ids = [*neighbor_lists, *(v for values in neighbor_lists.values() for v in values)]
    if any(value < 0 or value >= num_nodes for value in ids):
        raise ValueError("adjacency_dict contains node ids out of range.")
    table = _table_from_lists(neighbor_lists, num_nodes)
    rows = np.repeat(np.arange(num_nodes, dtype=np.int64), np.diff(table.indptr))
    weights = np.where(rows == table.indices, 1.0, 0.0)
    edge_rows = np.repeat(
        np.arange(num_nodes, dtype=np.int64), np.diff(edge_table.indptr)
    )
    edge_keys = edge_rows * num_nodes + edge_table.indices
    if edge_keys.size:
        query = rows * num_nodes + table.indices
        position = np.clip(np.searchsorted(edge_keys, query), 0, edge_keys.size - 1)
        found = edge_keys[position] == query
        weights = np.where(found, edge_table.weights[position], weights)
    return _NeighborTable(table.indptr, table.indices, weights)


def _uniform(rng: np.random.Generator | None, shape: tuple[int, int]) -> np.ndarray:
    return np.random.random_sample(shape) if rng is None else rng.random(shape)


def _sample_neighbors(
    table: _NeighborTable,
    nodes: np.ndarray,
    sample_num: int,
    rng: np.random.Generator | None,
) -> tuple[np.ndarray, np.ndarray]:
    start = table.indptr[nodes]
    degree = table.indptr[nodes + 1] - start
    sampled = np.repeat(nodes[:, None], sample_num, axis=1)
    weights = np.ones((nodes.size, sample_num), dtype=float)
    partial = np.flatnonzero((degree > 0) & (degree < sample_num))
    if partial.size:
        span = degree[partial, None]
        draws = _uniform(rng, (partial.size, sample_num))
        positions = start[partial, None] + np.minimum(
            (draws * span).astype(np.int64), span - 1
        )
        sampled[partial] = table.indices[positions]
        weights[partial] = table.weights[positions]
    complete = np.flatnonzero(degree >= sample_num)
    for span in np.unique(degree[complete]):
        group = complete[degree[complete] == span]
        keys = _uniform(rng, (group.size, int(span)))
        offsets = np.argsort(keys, axis=1, kind="stable")[:, :sample_num]
        positions = start[group, None] + offsets
        sampled[group] = table.indices[positions]
        weights[group] = table.weights[positions]
    return sampled, weights


def _multihop(
    table: _NeighborTable,
    src_nodes: np.ndarray,
    sample_nums: Sequence[int],
    rng: np.random.Generator | None,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    nodes = [np.asarray(src_nodes, dtype=np.int64)]
    weights = []
    for sample_num in sample_nums:
        sampled, sampled_weights = _sample_neighbors(table, nodes[-1], sample_num, rng)
        nodes.append(sampled.reshape(-1))
        weights.append(sampled_weights.reshape(-1))
    return nodes, weights


def _full_neighborhood(
    table: _NeighborTable, x: torch.Tensor, *, weighted: bool
) -> _FullNeighborhood:
    num_nodes = x.size(0)
    degree = np.diff(table.indptr)
    target = np.repeat(np.arange(num_nodes, dtype=np.int64), degree)
    source = table.indices
    weight = table.weights
    isolated = np.flatnonzero(degree == 0)
    if isolated.size:
        target = np.concatenate([target, isolated])
        source = np.concatenate([source, isolated])
        weight = np.concatenate([weight, np.ones(isolated.size)])
    count = np.bincount(target, minlength=num_nodes)
    return _FullNeighborhood(
        source=torch.as_tensor(source, dtype=torch.long, device=x.device),
        target=torch.as_tensor(target, dtype=torch.long, device=x.device),
        weight=(
            torch.as_tensor(weight, dtype=x.dtype, device=x.device)
            if weighted
            else None
        ),
        count=torch.as_tensor(count, dtype=x.dtype, device=x.device),
        num_nodes=num_nodes,
    )


def _sampled_max(
    neighbor_feature: torch.Tensor, weights: torch.Tensor | None
) -> torch.Tensor:
    if weights is None:
        return neighbor_feature.max(dim=1).values
    # Scaling features by weights before taking max flips the winner
    # whenever features can be negative; select the strongest-weighted
    # neighbor and aggregate its raw feature instead.
    selector = weights.squeeze(-1).argmax(dim=1)
    batch_index = torch.arange(neighbor_feature.size(0), device=neighbor_feature.device)
    return neighbor_feature[batch_index, selector]


def _full_max(
    node_features: torch.Tensor,
    messages: torch.Tensor,
    neighborhood: _FullNeighborhood,
) -> torch.Tensor:
    num_nodes = neighborhood.num_nodes
    if neighborhood.weight is None:
        initial = messages.new_full((num_nodes, messages.size(1)), float("-inf"))
        index = neighborhood.target.unsqueeze(-1).expand_as(messages)
        return initial.scatter_reduce(0, index, messages, reduce="amax")
    weight = neighborhood.weight
    best = weight.new_full((num_nodes,), float("-inf")).scatter_reduce(
        0, neighborhood.target, weight, reduce="amax"
    )
    positions = torch.arange(weight.numel(), device=weight.device)
    candidate = torch.where(
        weight == best[neighborhood.target],
        positions,
        torch.full_like(positions, weight.numel()),
    )
    chosen = torch.full(
        (num_nodes,), weight.numel(), dtype=torch.long, device=weight.device
    ).scatter_reduce(0, neighborhood.target, candidate, reduce="amin")
    return node_features[neighborhood.source[chosen]]
