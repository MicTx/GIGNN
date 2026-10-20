from __future__ import annotations

import numpy as np

from gignn.exceptions import ValidationError


def calculate_class_weights(
    encoded_labels, *, num_classes: int | None = None
) -> np.ndarray:
    labels = np.asarray(encoded_labels, dtype=np.int64)
    if labels.size == 0:
        raise ValidationError("encoded_labels must not be empty.")
    if labels.min() < 0:
        raise ValidationError("encoded_labels must be non-negative class indices.")
    resolved_class_count = (
        int(labels.max()) + 1 if num_classes is None else int(num_classes)
    )
    if resolved_class_count <= int(labels.max()):
        raise ValidationError("num_classes must include every encoded label.")
    counts = np.bincount(labels, minlength=resolved_class_count)
    present = counts > 0
    weights = np.zeros(resolved_class_count, dtype=float)
    weights[present] = labels.size / (present.sum() * counts[present].astype(float))
    return weights
