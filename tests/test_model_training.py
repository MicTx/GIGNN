import numpy as np
import pytest
import torch

from gignn.exceptions import ValidationError
from gignn.graph import (
    build_pyg_data,
    calculate_edge_weights,
    calculate_gaussian_edge_attributes,
)
from gignn.models import (
    GIGNN,
    GraphSAGEClassifier,
    NeighborAggregator,
    multihop_sampling,
)
from gignn.training import TrainingConfig, train_node_classifier


def test_graphsage_forward_shape(indicator_frame, labels, edge_index):
    pytest.importorskip("torch")
    pytest.importorskip("torch_geometric")
    data = build_pyg_data(indicator_frame, labels, edge_index)
    model = GraphSAGEClassifier(
        in_channels=data.num_features,
        hidden_channels=(8,),
        out_channels=3,
    )

    logits = model(data.x, data.edge_index)

    assert logits.shape == (data.num_nodes, 3)


def test_train_node_classifier_runs_single_epoch(indicator_frame, labels, edge_index):
    pytest.importorskip("torch")
    pytest.importorskip("torch_geometric")
    data = build_pyg_data(indicator_frame, labels, edge_index)
    model = GraphSAGEClassifier(
        in_channels=data.num_features,
        hidden_channels=(8,),
        out_channels=3,
    )

    result = train_node_classifier(
        model,
        data,
        TrainingConfig(max_epochs=1, learning_rate=0.01, random_state=42),
    )

    assert len(result.losses) == 1
    assert result.losses[0] >= 0


def test_multihop_sampling_matches_notebook_neighbor_list_semantics():
    adjacency = {0: [1, 2], 1: [0], 2: [0]}

    sampled = multihop_sampling([0, 1], [2, 3], adjacency, random_state=42)

    assert [nodes.shape[0] for nodes in sampled] == [2, 4, 12]
    assert sorted(sampled[1][:2].tolist()) == [1, 2]
    assert sampled[1][2:].tolist() == [0, 0]


def test_multihop_sampling_draws_without_replacement_when_degree_suffices():
    adjacency = {0: [1, 2, 3, 4, 5]}

    for seed in range(20):
        sampled = multihop_sampling([0], [3], adjacency, random_state=seed)[1]
        assert len(set(sampled.tolist())) == 3
        assert set(sampled.tolist()) <= {1, 2, 3, 4, 5}


def test_multihop_sampling_repeats_isolated_node():
    sampled = multihop_sampling([0, 1], [3], {0: [], 1: [0]}, random_state=0)

    assert sampled[1].tolist() == [0, 0, 0, 0, 0, 0]


def test_multihop_sampling_none_uses_numpy_global_rng(monkeypatch):
    calls = []

    def global_random_sample(shape):
        calls.append(shape)
        return np.zeros(shape)

    monkeypatch.setattr(np.random, "random_sample", global_random_sample)

    def reject_default_rng(*args, **kwargs):
        pytest.fail("random_state=None must consume the numpy global RNG")

    monkeypatch.setattr(np.random, "default_rng", reject_default_rng)

    with_replacement = multihop_sampling(
        [0],
        [4],
        {0: [1, 2, 3]},
        random_state=None,
    )
    without_replacement = multihop_sampling(
        [0],
        [2],
        {0: [1, 2, 3]},
        random_state=None,
    )

    assert with_replacement[1].tolist() == [1, 1, 1, 1]
    assert without_replacement[1].tolist() == [1, 2]
    assert calls == [(1, 4), (1, 3)]


def test_eq2_product_edge_weight_equals_vector_gaussian_kernel():
    features = np.array([[1.0, 5.0, 2.0], [3.0, 4.0, 2.0], [5.0, 1.0, 1.0]])
    edges = np.array([[0, 1, 0], [1, 2, 2]])

    attributes = calculate_gaussian_edge_attributes(features, edges, sigma=2.0)
    weights = calculate_edge_weights(attributes)

    distances = np.sum((features[edges[0]] - features[edges[1]]) ** 2, axis=1)
    np.testing.assert_allclose(weights, np.exp(-distances / (2 * 2.0**2)))


def test_unnormalized_mean_limits_increment_of_dissimilar_neighbors():
    neighbor_feature = torch.tensor([[[1.0], [3.0]]])
    similar = torch.tensor([[1.0, 1.0]])
    dissimilar = torch.tensor([[0.5, 0.5]])
    paper = NeighborAggregator(1, 1)
    legacy = NeighborAggregator(1, 1, normalize_weights=True)
    for aggregator in (paper, legacy):
        with torch.no_grad():
            aggregator.weight.copy_(torch.ones(1, 1))

    assert paper(neighbor_feature, similar).item() == pytest.approx(2.0)
    assert paper(neighbor_feature, dissimilar).item() == pytest.approx(1.0)
    assert legacy(neighbor_feature, dissimilar).item() == pytest.approx(2.0)


def test_gignn_uniformly_scaled_weights_change_paper_output(indicator_frame):
    data = build_pyg_data(indicator_frame, [0, 1, 1, 2, 0, 2], _ring(6))
    weights = torch.ones(data.edge_index.size(1))
    torch.manual_seed(0)
    paper = GIGNN(3, (8,), 3, num_neighbors_list=(2, 2)).eval()
    torch.manual_seed(0)
    legacy = GIGNN(3, (8,), 3, num_neighbors_list=(2, 2), normalize_edge_weights=True)
    legacy.eval()

    with torch.no_grad():
        paper_full = paper(data.x, data.edge_index, edge_weight=weights)
        paper_scaled = paper(data.x, data.edge_index, edge_weight=weights * 0.01)
        legacy_full = legacy(data.x, data.edge_index, edge_weight=weights)
        legacy_scaled = legacy(data.x, data.edge_index, edge_weight=weights * 0.01)

    assert not torch.allclose(paper_full, paper_scaled)
    assert torch.allclose(legacy_full, legacy_scaled)


def test_gignn_full_inference_is_deterministic_without_seed(indicator_frame):
    data = build_pyg_data(indicator_frame, [0, 1, 1, 2, 0, 2], _ring(6))
    model = GIGNN(3, (8,), 3, num_neighbors_list=(1, 1), random_state=None).eval()
    weights = torch.rand(data.edge_index.size(1))

    with torch.no_grad():
        first = model(data.x, data.edge_index, edge_weight=weights)
        second = model(data.x, data.edge_index, edge_weight=weights)

    assert torch.equal(first, second)


@pytest.mark.parametrize("aggregation", ["mean", "sum", "max"])
@pytest.mark.parametrize("normalize", [False, True])
def test_gignn_full_inference_equals_sampled_when_degree_equals_sample_size(
    indicator_frame, aggregation, normalize
):
    data = build_pyg_data(indicator_frame, [0, 1, 1, 2, 0, 2], _ring(6))
    weights = torch.linspace(0.2, 1.0, data.edge_index.size(1))
    torch.manual_seed(0)
    model = GIGNN(
        3,
        (8,),
        3,
        num_neighbors_list=(2, 2),
        aggr_neighbor_method=aggregation,
        normalize_edge_weights=normalize,
        random_state=7,
    ).eval()

    with torch.no_grad():
        full = model(data.x, data.edge_index, edge_weight=weights)
        model.inference = "sampled"
        sampled = model(data.x, data.edge_index, edge_weight=weights)

    torch.testing.assert_close(full, sampled)


def test_training_resamples_each_epoch_from_one_seeded_stream(
    indicator_frame, labels, monkeypatch
):
    from gignn.models import gignn as gignn_module

    states = []
    original = gignn_module._multihop

    def spy(table, src_nodes, sample_nums, rng):
        states.append((id(rng), rng.bit_generator.state["state"]["state"]))
        return original(table, src_nodes, sample_nums, rng)

    monkeypatch.setattr(gignn_module, "_multihop", spy)
    data = build_pyg_data(indicator_frame, labels, _ring(6))
    config = TrainingConfig(max_epochs=3, learning_rate=0.01, random_state=42)

    for _ in range(2):
        model = GIGNN(3, (8,), 3, num_neighbors_list=(2, 2), random_state=42)
        train_node_classifier(model, data, config)

    first_fit, second_fit = states[:3], states[3:]
    assert len({rng_id for rng_id, _ in first_fit}) == 1
    assert len({state for _, state in first_fit}) == 3
    assert [state for _, state in first_fit] == [state for _, state in second_fit]


def test_train_node_classifier_ignores_labels_outside_mask(indicator_frame):
    mask = np.array([True, True, True, True, False, False])
    edges = _ring(6)
    first = build_pyg_data(indicator_frame, [0, 1, 1, 2, 0, 2], edges)
    second = build_pyg_data(
        indicator_frame, [0, 1, 1, 2, 2, 0], edges, class_order=[0, 1, 2]
    )
    config = TrainingConfig(max_epochs=3, learning_rate=0.01, random_state=1)
    models = []
    for data in (first, second):
        torch.manual_seed(1)
        model = GIGNN(3, (8,), 3, num_neighbors_list=(2, 2), random_state=1)
        train_node_classifier(model, data, config, train_mask=mask)
        models.append(model)

    for left, right in zip(models[0].parameters(), models[1].parameters()):
        torch.testing.assert_close(left, right)


def test_train_node_classifier_rejects_invalid_mask(indicator_frame, labels):
    data = build_pyg_data(indicator_frame, labels, _ring(6))
    model = GIGNN(3, (8,), 3, num_neighbors_list=(2, 2))

    with pytest.raises(ValidationError, match="train_mask"):
        train_node_classifier(
            model, data, TrainingConfig(max_epochs=1), train_mask=np.zeros(6, bool)
        )


def _ring(num_nodes):
    sources = np.arange(num_nodes)
    return np.vstack([sources, (sources + 1) % num_nodes])


def test_gignn_forward_uses_edge_weights(indicator_frame, edge_index):
    pytest.importorskip("torch")
    data = build_pyg_data(
        indicator_frame,
        ["low", "medium", "medium", "high", "low", "high"],
        edge_index,
        edge_weight=[1.0] * edge_index.shape[1],
    )
    model = GIGNN(
        in_channels=data.num_features,
        hidden_channels=(8,),
        out_channels=3,
        num_neighbors_list=(2, 2),
    )

    logits_with_edges = model(data.x, data.edge_index, edge_weight=data.edge_weight)
    logits_without_edges = model(
        data.x, data.edge_index, edge_weight=data.edge_weight * 0.0
    )

    assert logits_with_edges.shape == (data.num_nodes, 3)
    assert not logits_with_edges.allclose(logits_without_edges)


def test_train_node_classifier_runs_gignn_single_epoch(
    indicator_frame, labels, edge_index
):
    pytest.importorskip("torch")
    pytest.importorskip("torch_geometric")
    data = build_pyg_data(
        indicator_frame,
        labels,
        edge_index,
        edge_weight=[1.0] * edge_index.shape[1],
    )
    model = GIGNN(
        in_channels=data.num_features,
        hidden_channels=(8,),
        out_channels=3,
        num_neighbors_list=(2, 2),
    )

    result = train_node_classifier(
        model,
        data,
        TrainingConfig(max_epochs=1, learning_rate=0.01, random_state=42),
    )

    assert len(result.losses) == 1
    assert result.losses[0] >= 0
