from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from gignn.exceptions import ValidationError
from gignn.training.metrics import calculate_class_weights
from gignn.training.split import set_random_seed


@dataclass(frozen=True)
class TrainingConfig:
    max_epochs: int = 100
    learning_rate: float = 0.01
    weight_decay: float = 5e-4
    random_state: int | None = None
    device: str = "cpu"


@dataclass(frozen=True)
class TrainingResult:
    losses: list[float]


def train_node_classifier(
    model: nn.Module,
    data,
    config: TrainingConfig,
    *,
    train_mask: object | None = None,
) -> TrainingResult:
    """Train with class-weighted cross-entropy on the labeled nodes.

    ``train_mask`` selects the nodes whose labels enter the loss; every node
    still takes part in message passing (transductive node classification).
    Class weights are computed from the masked labels only.
    """
    if config.random_state is not None:
        set_random_seed(config.random_state)
    device = torch.device(config.device)
    model = model.to(device)
    data = data.to(device)
    mask = _training_mask(train_mask, int(data.num_nodes), device)
    targets = data.y[mask]
    class_weights = torch.tensor(
        calculate_class_weights(
            targets.cpu().numpy(),
            num_classes=int(data.class_count),
        ),
        dtype=torch.float32,
        device=device,
    )
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    reset_sampling = getattr(model, "reset_sampling", None)
    if callable(reset_sampling):
        reset_sampling()
    losses: list[float] = []

    model.train()
    for _ in range(config.max_epochs):
        optimizer.zero_grad()
        logits = model(
            data.x,
            data.edge_index,
            edge_weight=getattr(data, "edge_weight", None),
        )
        loss = criterion(logits[mask], targets)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu().item()))
    model.eval()
    return TrainingResult(losses=losses)


def _training_mask(
    train_mask: object | None, num_nodes: int, device: torch.device
) -> torch.Tensor:
    if train_mask is None:
        return torch.ones(num_nodes, dtype=torch.bool, device=device)
    mask = (
        train_mask.cpu().numpy() if isinstance(train_mask, torch.Tensor) else train_mask
    )
    mask = np.asarray(mask)
    if mask.dtype != bool or mask.shape != (num_nodes,):
        raise ValidationError(
            "train_mask must be a boolean array with one entry per node."
        )
    if not mask.any():
        raise ValidationError("train_mask must select at least one node.")
    return torch.as_tensor(mask, dtype=torch.bool, device=device)
