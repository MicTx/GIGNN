import numpy as np
import pandas as pd
import pytest

from gignn.exceptions import ValidationError
from gignn.validation import (
    LabelEncoder,
    validate_edge_index,
    validate_feature_table,
    validate_X_y,
)


def test_validate_x_y_rejects_length_mismatch(indicator_frame):
    y = pd.Series(["a", "b"])

    with pytest.raises(ValidationError, match="same number of rows"):
        validate_X_y(indicator_frame, y)


def test_validate_feature_table_rejects_non_finite_values(indicator_frame):
    bad = indicator_frame.copy()
    bad.loc[0, "ndvi"] = np.inf

    with pytest.raises(ValidationError, match="finite"):
        validate_feature_table(bad)


def test_validate_feature_table_does_not_mutate_input(indicator_frame):
    original = indicator_frame.copy(deep=True)

    validated = validate_feature_table(indicator_frame)

    pd.testing.assert_frame_equal(indicator_frame, original)
    assert validated is not indicator_frame


def test_validate_edge_index_rejects_too_many_edges(edge_index):
    with pytest.raises(ValidationError, match="max_edges"):
        validate_edge_index(edge_index, num_nodes=6, max_edges=1)


def test_validate_edge_index_rejects_excessive_list_before_array_conversion():
    huge_edges = [[0] * 1_000_001, [1] * 1_000_001]

    with pytest.raises(ValidationError, match="max_edges"):
        validate_edge_index(huge_edges, num_nodes=2)


def test_validate_edge_index_rejects_out_of_range_nodes(edge_index):
    bad = edge_index.copy()
    bad[1, 0] = 99

    with pytest.raises(ValidationError, match="out of range"):
        validate_edge_index(bad, num_nodes=6)


def test_validate_edge_index_accepts_transposed_edge_list(edge_index):
    validated = validate_edge_index(edge_index.T, num_nodes=6)

    assert validated.shape == (2, 6)
    assert validated.dtype == np.int64


def test_label_encoder_round_trips_string_labels(labels):
    encoder = LabelEncoder().fit(labels)

    encoded = encoder.transform(labels)
    decoded = encoder.inverse_transform(encoded)

    assert encoded.tolist() == [1, 2, 2, 0, 1, 0]
    assert decoded.tolist() == labels.tolist()


def test_label_encoder_round_trips_non_contiguous_integer_labels(non_contiguous_labels):
    encoder = LabelEncoder().fit(non_contiguous_labels)

    encoded = encoder.transform(non_contiguous_labels)
    decoded = encoder.inverse_transform(encoded)

    assert encoded.tolist() == [0, 1, 1, 2, 0, 2]
    assert decoded.tolist() == non_contiguous_labels.tolist()


def test_label_encoder_distinguishes_boolean_and_numeric_labels():
    encoder = LabelEncoder().fit([True, 1])

    assert encoder.classes_.tolist() == [True, 1]
    np.testing.assert_array_equal(encoder.transform([True, 1]), np.array([0, 1]))


def test_label_encoder_fit_transform_does_not_mutate_encoder(labels):
    encoder = LabelEncoder()

    encoded = encoder.fit_transform(labels)

    assert encoder.classes_ is None
    assert encoded.tolist() == [1, 2, 2, 0, 1, 0]
