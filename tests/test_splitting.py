import numpy as np
import pandas as pd
import pytest

from gignn.exceptions import ValidationError
from gignn.splitting import resolve_split_indices


def test_random_split_requires_three_rows():
    with pytest.raises(ValidationError, match="at least three rows"):
        resolve_split_indices(
            pd.Series([0, 1]),
            validation_fraction=0.2,
            test_fraction=0.2,
            random_state=42,
        )


def test_spatial_groups_are_assigned_as_indivisible_partitions():
    target = pd.Series([0, 1] * 6)
    groups = ["west"] * 3 + ["center"] * 3 + ["east"] * 3 + ["remote"] * 3

    first = resolve_split_indices(
        target,
        validation_fraction=0.25,
        test_fraction=0.25,
        random_state=42,
        spatial_groups=groups,
    )
    second = resolve_split_indices(
        target,
        validation_fraction=0.25,
        test_fraction=0.25,
        random_state=42,
        spatial_groups=groups,
    )

    np.testing.assert_array_equal(first.train, second.train)
    np.testing.assert_array_equal(first.validation, second.validation)
    np.testing.assert_array_equal(first.test, second.test)
    assignment = {}
    for split_name, indices in first.as_dict().items():
        for index in indices:
            group = groups[int(index)]
            assert assignment.setdefault(group, split_name) == split_name


def test_spatial_group_warning_uses_realized_row_fraction():
    groups = ["large"] * 80 + ["small-a"] * 5 + ["small-b"] * 5
    groups += ["small-c"] * 5 + ["small-d"] * 5
    target = pd.Series([0, 1] * 50)

    with pytest.warns(UserWarning) as caught:
        split = resolve_split_indices(
            target,
            validation_fraction=0.2,
            test_fraction=0.2,
            random_state=13,
            spatial_groups=groups,
        )

    messages = [str(item.message) for item in caught]
    assert any(
        "test_fraction=0.20" in message
        and "row_fraction=0.80" in message
        and "group_fraction=0.20" in message
        for message in messages
    )
    assert any(
        "validation_fraction=0.20" in message
        and "row_fraction=0.05" in message
        and "group_fraction=0.20" in message
        for message in messages
    )
    assert len(split.test) / len(target) == pytest.approx(0.8)


def test_random_split_is_reproducible_disjoint_and_complete():
    target = pd.Series([0, 1, 2] * 10)

    first = resolve_split_indices(
        target,
        validation_fraction=0.2,
        test_fraction=0.2,
        random_state=42,
    )
    second = resolve_split_indices(
        target,
        validation_fraction=0.2,
        test_fraction=0.2,
        random_state=42,
    )

    for name in ("train", "validation", "test"):
        np.testing.assert_array_equal(getattr(first, name), getattr(second, name))
    assert set(first.train).isdisjoint(first.validation)
    assert set(first.train).isdisjoint(first.test)
    assert set(first.validation).isdisjoint(first.test)
    assert sorted(np.concatenate([first.train, first.validation, first.test])) == list(
        range(len(target))
    )


def test_split_labels_and_spatial_groups_are_mutually_exclusive():
    target = pd.Series([0, 1] * 3)

    with pytest.raises(ValidationError, match="must not be provided together"):
        resolve_split_indices(
            target,
            validation_fraction=0.2,
            test_fraction=0.2,
            random_state=42,
            split_labels=[
                "train",
                "train",
                "validation",
                "validation",
                "test",
                "test",
            ],
            spatial_groups=["west", "west", "center", "center", "east", "east"],
        )


def test_split_fractions_are_validated_for_every_strategy():
    target = pd.Series([0, 1] * 3)
    for kwargs in (
        {
            "split_labels": [
                "train",
                "train",
                "validation",
                "validation",
                "test",
                "test",
            ]
        },
        {"spatial_groups": ["west", "west", "center", "center", "east", "east"]},
    ):
        with pytest.raises(ValidationError, match="fractions"):
            resolve_split_indices(
                target,
                validation_fraction=0.0,
                test_fraction=0.2,
                random_state=42,
                **kwargs,
            )


def test_split_indices_are_read_only():
    split = resolve_split_indices(
        pd.Series([0, 1] * 3),
        validation_fraction=0.2,
        test_fraction=0.2,
        random_state=42,
    )

    for values in (split.train, split.validation, split.test):
        assert values.flags.writeable is False
        with pytest.raises(ValueError, match="read-only"):
            values[0] = 99


def test_spatial_groups_require_three_groups():
    with pytest.raises(ValidationError, match="at least three groups"):
        resolve_split_indices(
            pd.Series([0, 1, 0, 1]),
            validation_fraction=0.2,
            test_fraction=0.2,
            random_state=42,
            spatial_groups=["west", "west", "east", "east"],
        )


def test_explicit_split_labels_must_cover_three_partitions():
    with pytest.raises(ValidationError, match="train, validation, and test"):
        resolve_split_indices(
            pd.Series([0, 1, 0]),
            validation_fraction=0.2,
            test_fraction=0.2,
            random_state=42,
            split_labels=["train", "validation", "validation"],
        )
