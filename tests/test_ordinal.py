import numpy as np
import pandas as pd
import pytest

from gignn.exceptions import ValidationError
from gignn.ordinal import fit_ordinal_target
from gignn.validation import validate_class_order


def test_ordinal_target_uses_explicit_string_order():
    encoded, encoder = fit_ordinal_target(
        pd.Series(["medium", "low", "high", "medium"]),
        class_order=["low", "medium", "high"],
    )

    np.testing.assert_array_equal(encoded.to_numpy(), np.array([1, 0, 2, 1]))
    np.testing.assert_array_equal(
        encoder.inverse_transform(encoded),
        np.array(["medium", "low", "high", "medium"], dtype=object),
    )


def test_ordinal_target_rejects_duplicate_or_missing_order_entries():
    target = pd.Series(["low", "medium", "high"])

    with pytest.raises(ValidationError, match="duplicate"):
        fit_ordinal_target(
            target,
            class_order=["low", "medium", "medium", "high"],
        )
    with pytest.raises(ValidationError, match="every observed label"):
        fit_ordinal_target(target, class_order=["low", "high"])


def test_ordinal_target_distinguishes_boolean_and_numeric_labels():
    with pytest.raises(ValidationError, match="every observed label"):
        fit_ordinal_target(pd.Series([True, 1]), class_order=[True])

    encoded, encoder = fit_ordinal_target(pd.Series([True, 1]), class_order=[True, 1])
    np.testing.assert_array_equal(encoded.to_numpy(), np.array([0, 1]))
    assert encoder.classes_ == (True, 1)

    validated = validate_class_order([True, 1])
    assert validated.tolist() == [True, 1]


def test_numeric_ordinal_target_defaults_to_ascending_order():
    encoded, encoder = fit_ordinal_target(
        pd.Series([30, 10, 20, 30]),
        class_order=None,
    )

    assert encoder.classes_ == (10, 20, 30)
    np.testing.assert_array_equal(encoded.to_numpy(), np.array([2, 0, 1, 2]))


def test_ordinal_target_requires_order_for_numeric_strings():
    with pytest.raises(ValidationError, match="class_order"):
        fit_ordinal_target(pd.Series(["10", "2"]), class_order=None)


def test_ordinal_target_rejects_empty_missing_and_unhashable_labels():
    with pytest.raises(ValidationError, match="at least one label"):
        fit_ordinal_target(pd.Series([], dtype=object), class_order=[])
    with pytest.raises(ValidationError, match="missing labels"):
        fit_ordinal_target(pd.Series(["low", None]), class_order=["low"])
    with pytest.raises(ValidationError, match="hashable scalar"):
        fit_ordinal_target(pd.Series([["low"]], dtype=object), class_order=[["low"]])


def test_ordered_categorical_target_uses_declared_order():
    target = pd.Series(
        pd.Categorical(
            ["medium", "low", "high"],
            categories=["low", "medium", "high"],
            ordered=True,
        )
    )

    encoded, encoder = fit_ordinal_target(target, class_order=None)

    assert encoder.classes_ == ("low", "medium", "high")
    np.testing.assert_array_equal(encoded.to_numpy(), np.array([1, 0, 2]))


def test_ordered_categorical_target_rejects_unused_declared_category():
    target = pd.Series(
        pd.Categorical(
            ["low", "high"],
            categories=["low", "medium", "high"],
            ordered=True,
        )
    )

    with pytest.raises(ValidationError, match="every declared category"):
        fit_ordinal_target(target, class_order=None)


def test_ordinal_encoder_rejects_unknown_and_out_of_range_values():
    _, encoder = fit_ordinal_target(pd.Series([10, 20]), class_order=[10, 20])

    with pytest.raises(ValidationError, match="Unknown label"):
        encoder.transform([30])
    with pytest.raises(ValidationError, match="out of range"):
        encoder.inverse_transform([-1])
    np.testing.assert_array_equal(
        encoder.inverse_transform([]), np.asarray([], dtype=object)
    )


def test_ordinal_encoder_rejects_invalid_encoded_values():
    _, encoder = fit_ordinal_target(pd.Series([10, 20]), class_order=[10, 20])

    for values in ([0.9], ["0"], [True]):
        with pytest.raises(ValidationError, match="integer"):
            encoder.inverse_transform(values)
