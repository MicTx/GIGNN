"""Tests for the paper-aligned evaluation protocol (§4.2, §5.2.1-5.2.2)."""

import numpy as np
import pandas as pd
import pytest
import torch

from gignn import GIGNNClassifier, cross_validate
from gignn.exceptions import EdgeWeightWarning, ValidationError
from gignn.generalization import MoranResult, morans_i

pytestmark = pytest.mark.filterwarnings("ignore::gignn.exceptions.EdgeWeightWarning")


def test_fit_train_mask_trains_transductively_on_masked_labels(
    indicator_frame, labels, edge_index
):
    pytest.importorskip("torch")
    pytest.importorskip("torch_geometric")
    partial_labels = labels.copy()
    partial_labels.iloc[4:] = None
    mask = np.asarray([True] * 4 + [False] * 2)
    model = GIGNNClassifier(hidden_channels=(8,), max_epochs=2, random_state=42)

    model.fit(indicator_frame, partial_labels, edge_index=edge_index, train_mask=mask)
    predictions = model.predict(indicator_frame, edge_index=edge_index)

    assert predictions.shape == (6,)


def test_fit_rejects_mask_covering_missing_labels(indicator_frame, labels, edge_index):
    partial_labels = labels.copy()
    partial_labels.iloc[0] = None
    mask = np.asarray([True] * 6)
    model = GIGNNClassifier(hidden_channels=(8,), max_epochs=1)

    with pytest.raises(ValidationError, match="inside train_mask"):
        model.fit(
            indicator_frame, partial_labels, edge_index=edge_index, train_mask=mask
        )


def test_fit_rejects_empty_or_wrong_mask(indicator_frame, labels, edge_index):
    model = GIGNNClassifier(hidden_channels=(8,), max_epochs=1)

    with pytest.raises(ValidationError, match="at least one"):
        model.fit(
            indicator_frame,
            labels,
            edge_index=edge_index,
            train_mask=np.zeros(6, dtype=bool),
        )
    with pytest.raises(ValidationError, match="boolean"):
        model.fit(
            indicator_frame,
            labels,
            edge_index=edge_index,
            train_mask=np.ones(6),
        )


def test_fit_warns_when_gaussian_edge_weights_collapse(edge_index):
    raw_scale = pd.DataFrame(
        {
            "income": [41_000.0, 95_000.0, 22_000.0, 130_000.0, 5_000.0, 88_000.0],
            "age": [100.0, 8_000.0, 55_000.0, 320.0, 40_000.0, 700.0],
        }
    )
    y = pd.Series([0, 1, 0, 1, 0, 1])
    model = GIGNNClassifier(hidden_channels=(8,), max_epochs=1)

    with pytest.warns(EdgeWeightWarning, match="degenerates"):
        model.fit(raw_scale, y, edge_index=edge_index)


def test_stratum_scale_features_do_not_warn(indicator_frame, labels, edge_index):
    import warnings

    strata = pd.DataFrame(
        {
            "x1": [1.0, 2.0, 3.0, 2.0, 1.0, 3.0],
            "x2": [1.0, 1.0, 2.0, 3.0, 2.0, 1.0],
        }
    )
    model = GIGNNClassifier(hidden_channels=(8,), max_epochs=1)

    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        model.fit(strata, pd.Series([0, 1, 2, 1, 0, 2]), edge_index=edge_index)
    assert not [w for w in record if issubclass(w.category, EdgeWeightWarning)]


def test_legacy_checkpoint_loads_with_legacy_inference_defaults(
    tmp_path, indicator_frame, labels, edge_index
):
    pytest.importorskip("torch_geometric")
    path = tmp_path / "legacy.pt"
    model = GIGNNClassifier(
        hidden_channels=(8,),
        edge_weight_mode="mean",
        normalize_edge_weights=True,
        inference="sampled",
        max_epochs=1,
        random_state=42,
    )
    model.fit(indicator_frame, labels, edge_index=edge_index)
    model.save(path)

    payload = torch.load(path, weights_only=False)
    payload["params"].pop("normalize_edge_weights")
    payload["params"].pop("inference")
    torch.save(payload, path)

    loaded = GIGNNClassifier.load(path)

    assert loaded.normalize_edge_weights is True
    assert loaded.inference == "sampled"
    np.testing.assert_array_equal(
        loaded.predict(indicator_frame, edge_index=edge_index),
        model.predict(indicator_frame, edge_index=edge_index),
    )


def test_cross_validate_pools_out_of_fold_predictions(
    indicator_frame, labels, edge_index
):
    pytest.importorskip("torch")
    pytest.importorskip("torch_geometric")

    def factory():
        return GIGNNClassifier(hidden_channels=(8,), max_epochs=2, random_state=42)

    report = cross_validate(
        factory,
        indicator_frame,
        labels,
        n_splits=2,
        random_state=0,
        edge_index=edge_index,
        moran_permutations=49,
    )

    assert report.fold_metrics["n_test"].sum() == len(labels)
    assert report.fold_metrics["accuracy"].between(0, 1).all()
    assert report.out_of_fold_predictions.shape == (len(labels),)
    assert isinstance(report.bias_moran, MoranResult)


def test_cross_validate_moran_matches_manual_bias(indicator_frame, edge_index):
    values = pd.Series([0, 1, 2, 0, 1, 2])

    report = cross_validate(
        _ShiftedPredictor,
        indicator_frame,
        values,
        n_splits=3,
        edge_index=edge_index,
        moran_permutations=0,
    )

    bias = np.asarray([(int(v) + 1) % 3 - int(v) for v in values], dtype=float)
    expected = morans_i(bias, edge_index, permutations=0)
    assert report.bias_moran == expected


def test_cross_validate_rejects_invalid_inputs(indicator_frame, labels, edge_index):
    with pytest.raises(ValidationError, match="n_splits"):
        cross_validate(lambda: None, indicator_frame, labels, n_splits=1)
    with pytest.raises(ValidationError, match="n_splits"):
        cross_validate(lambda: None, indicator_frame, labels, n_splits=7)
    bad_labels = labels.copy()
    bad_labels.iloc[0] = None
    with pytest.raises(ValidationError, match="complete labels"):
        cross_validate(lambda: None, indicator_frame, bad_labels, n_splits=2)


class _ShiftedPredictor:
    def fit(self, X, y, train_mask=None, geometry=None, edge_index=None):
        self.labels_ = pd.Series(y).to_numpy()
        return self

    def predict(self, X, geometry=None, edge_index=None):
        return (self.labels_.astype(int) + 1) % 3
