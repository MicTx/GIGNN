from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from gignn.exceptions import ValidationError
from gignn.validation import (
    LabelEncoder,
    validate_class_order,
    validate_edge_index,
    validate_X_y,
)


def build_pyg_data(
    X: pd.DataFrame,
    y,
    edge_index,
    *,
    edge_attr: object | None = None,
    edge_weight: object | None = None,
    class_order: Sequence[object] | None = None,
):
    try:
        import torch
        from torch_geometric.data import Data
    except ModuleNotFoundError as exc:
        raise ValidationError(
            "build_pyg_data requires torch and torch_geometric to be installed."
        ) from exc

    X_validated, y_validated = validate_X_y(X, y)
    encoder = (
        LabelEncoder(classes_=validate_class_order(class_order))
        if class_order is not None
        else LabelEncoder().fit(y_validated)
    )
    encoded_y = encoder.transform(y_validated)
    edges = validate_edge_index(edge_index, num_nodes=len(X_validated))

    data_kwargs = {
        "x": torch.tensor(X_validated.to_numpy(dtype=float), dtype=torch.float32),
        "y": torch.tensor(encoded_y, dtype=torch.long),
        "edge_index": torch.tensor(edges, dtype=torch.long),
    }
    if edge_attr is not None:
        data_kwargs["edge_attr"] = torch.tensor(
            _validate_edge_attr(edge_attr, num_edges=edges.shape[1]),
            dtype=torch.float32,
        )
    if edge_weight is not None:
        data_kwargs["edge_weight"] = torch.tensor(
            _validate_edge_weight(edge_weight, num_edges=edges.shape[1]),
            dtype=torch.float32,
        )

    data = Data(
        **data_kwargs,
    )
    data.label_encoder = encoder
    data.class_count = len(encoder.classes_)
    return data


def _validate_edge_attr(edge_attr: object, *, num_edges: int) -> np.ndarray:
    attributes = np.asarray(edge_attr, dtype=float)
    if attributes.ndim == 1:
        attributes = attributes.reshape(-1, 1)
    if attributes.ndim != 2 or attributes.shape[0] != num_edges:
        raise ValidationError("edge_attr must have shape (num_edges, num_features).")
    if not np.isfinite(attributes).all():
        raise ValidationError("edge_attr must contain only finite values.")
    return attributes


def _validate_edge_weight(edge_weight: object, *, num_edges: int) -> np.ndarray:
    weights = np.asarray(edge_weight, dtype=float)
    if weights.ndim != 1 or weights.shape[0] != num_edges:
        raise ValidationError("edge_weight must have shape (num_edges,).")
    if not np.isfinite(weights).all():
        raise ValidationError("edge_weight must contain only finite values.")
    return weights
