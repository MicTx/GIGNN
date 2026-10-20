from __future__ import annotations

import torch
from torch import nn
from torch_geometric.nn import SAGEConv


class GraphSAGEClassifier(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: tuple[int, ...],
        out_channels: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        channels = (in_channels, *hidden_channels, out_channels)
        self.convs = nn.ModuleList(
            SAGEConv(channels[index], channels[index + 1])
            for index in range(len(channels) - 1)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # SAGEConv has no edge-weighted variant; silently dropping the weights
        # would discard the Gaussian edge information callers expect to matter.
        if edge_weight is not None:
            raise NotImplementedError(
                "GraphSAGEClassifier does not support edge_weight; use GIGNN for "
                "weighted message passing."
            )
        hidden = x
        for conv in self.convs[:-1]:
            hidden = conv(hidden, edge_index).relu()
            hidden = self.dropout(hidden)
        return self.convs[-1](hidden, edge_index)
