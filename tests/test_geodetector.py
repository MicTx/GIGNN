import math

import numpy as np
import pandas as pd
import pytest

from gignn.exceptions import ValidationError
from gignn.geodetector import (
    GeoDetectorSelector,
    factor_detector,
    interaction_detector,
    spatial_discretization,
)
from gignn.geodetector.discretization import (
    _discretize_values,
    _jenks_natural_breaks,
)
from gignn.geodetector.factor import _q_p_value, target_to_numeric
from gignn.geodetector.interaction import _classify_interaction
from gignn.geodetector.selector import SELECTION_STRATEGIES


def test_spatial_discretization_quantiles_does_not_mutate_input():
    frame = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0], "y": [0, 0, 1, 1]})
    original = frame.copy(deep=True)

    discretized, metadata = spatial_discretization(
        frame, ["x"], method="quantiles", num_intervals=2
    )

    pd.testing.assert_frame_equal(frame, original)
    assert discretized["x"].tolist() == [1, 1, 2, 2]
    assert metadata["x"].method == "quantiles"


def test_spatial_discretization_box_plot_uses_special_method_branch():
    frame = pd.DataFrame({"x": [1.0, 2.0, 3.0, 100.0]})

    discretized, metadata = spatial_discretization(
        frame, "x", method="box_plot", num_intervals=4
    )

    assert set(discretized["x"]).issubset(set(range(1, len(metadata["x"].bins) + 2)))
    assert metadata["x"].method == "box_plot"


def test_spatial_discretization_natural_breaks_uses_jenks_not_quantiles():
    frame = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 100.0, 101.0]})

    discretized, metadata = spatial_discretization(
        frame, "x", method="natural_breaks", num_intervals=2
    )

    assert discretized["x"].tolist() == [1, 1, 1, 1, 2, 2]
    assert metadata["x"].method == "natural_breaks"
    assert 4.0 <= metadata["x"].bins[0] < 100.0


def test_factor_detector_ranks_explanatory_variable_higher():
    frame = pd.DataFrame(
        {
            "Y": [1.0, 1.1, 9.0, 9.2, 1.2, 9.1],
            "strong": [0, 0, 1, 1, 0, 1],
            "weak": [0, 1, 0, 1, 0, 1],
        }
    )

    result = factor_detector(frame, y="Y", factors=["strong", "weak"], alpha=None)

    assert result.iloc[0]["factor"] == "strong"
    assert result.iloc[0]["q"] > result.iloc[1]["q"]


def test_interaction_detector_supports_ordered_categorical_targets():
    frame = pd.DataFrame(
        {
            "Y": pd.Categorical(
                ["low", "low", "medium", "medium", "high", "high"],
                categories=["low", "medium", "high"],
                ordered=True,
            ),
            "a": [0, 0, 1, 1, 2, 2],
            "b": [0, 1, 0, 1, 0, 1],
        }
    )

    result = interaction_detector(frame, y="Y", factors=["a", "b"])

    assert result.loc[0, "q_a"] > 0.9


def test_interaction_detector_uses_values_not_column_indices():
    frame = pd.DataFrame(
        {
            "Y": [0.0, 0.1, 5.0, 5.1, 10.0, 10.2],
            "a": [0, 0, 1, 1, 2, 2],
            "b": [0, 1, 0, 1, 0, 1],
        }
    )

    result = interaction_detector(frame, y="Y", factors=["a", "b"])

    assert result.loc[0, "factor_1"] == "a"
    assert result.loc[0, "factor_2"] == "b"
    assert result.loc[0, "q_interaction"] > result.loc[0, "q_b"]
    assert result.loc[0, "q_a"] > 0.9


def test_geodetector_selector_exposes_fitted_attributes():
    frame = pd.DataFrame(
        {
            "ndvi": [1.0, 1.1, 8.0, 8.1, 1.2, 8.2],
            "dem": [3.0, 4.0, 3.5, 4.5, 3.2, 4.2],
        }
    )
    y = pd.Series(["low", "low", "high", "high", "low", "high"])

    selector = GeoDetectorSelector(
        discretizer="quantiles", max_bins=2, top_k=1, strategy="factor"
    )
    transformed = selector.fit_transform(frame, y)

    assert selector.selected_features_ == ["ndvi"]
    assert "ndvi" in selector.scores_.index
    assert "ndvi" in selector.discretizers_
    assert transformed.columns.tolist() == ["ndvi"]
    assert transformed["ndvi"].tolist() == [1, 1, 2, 2, 1, 2]


def test_maximum_breaks_cuts_at_widest_gap_midpoints():
    labels, bins = _discretize_values(
        np.array([1.0, 2.0, 3.0, 10.0, 11.0, 30.0]), "maximum_breaks", 3
    )

    np.testing.assert_allclose(bins, [6.5, 20.5])
    assert labels.tolist() == [1, 1, 1, 2, 2, 3]


def test_maximum_breaks_equal_gaps_cut_earliest():
    _, bins = _discretize_values(np.array([0.0, 1.0, 2.0, 3.0]), "maximum_breaks", 3)

    np.testing.assert_allclose(bins, [0.5, 1.5])


def test_classify_interaction_exact_sum_is_independent():
    assert _classify_interaction(0.1, 0.2, 0.3) == "independent"


def test_classify_interaction_at_max_within_tolerance_is_weaken():
    assert _classify_interaction(0.2, 0.1, 0.2 + 1e-15) == "weaken"


def test_interaction_detector_p_value_nan_when_strata_are_singletons():
    frame = pd.DataFrame(
        {
            "Y": [1.0, 2.0, 3.0, 4.0],
            "a": [1, 1, 2, 2],
            "b": [1, 2, 1, 2],
        }
    )

    result = interaction_detector(frame, y="Y", factors=["a", "b"])

    assert "p_value_interaction" in result.columns
    assert math.isnan(result.loc[0, "p_value_interaction"])


def test_q_p_value_nan_when_n_le_group_count():
    assert math.isnan(_q_p_value(np.array([1.0, 2.0, 3.0]), np.array([1, 2, 3]), 1.0))


def test_q_p_value_zero_when_q_ge_1_and_n_gt_group_count():
    assert (
        _q_p_value(np.array([1.0, 1.0, 2.0, 2.0]), np.array([1, 1, 2, 2]), 1.0) == 0.0
    )


def test_q_p_value_one_for_single_group():
    assert _q_p_value(np.array([1.0, 2.0, 3.0]), np.array([1, 1, 1]), 0.0) == 1.0


def test_target_to_numeric_ordered_categorical_uses_order_not_alphabet():
    values = pd.Series(
        pd.Categorical(
            ["mid", "low", "high"],
            categories=["low", "mid", "high"],
            ordered=True,
        )
    )

    np.testing.assert_array_equal(target_to_numeric(values), [1.0, 0.0, 2.0])


def test_target_to_numeric_rejects_unordered_multiclass_strings():
    with pytest.raises(ValidationError, match="ordered categoricals"):
        target_to_numeric(pd.Series(["low", "mid", "high"]))


def test_target_to_numeric_allows_binary_strings():
    np.testing.assert_array_equal(
        target_to_numeric(pd.Series(["a", "b", "a"])), [0.0, 1.0, 0.0]
    )


def test_target_to_numeric_numeric_strings():
    np.testing.assert_array_equal(
        target_to_numeric(pd.Series(["1.5", "2.5"])), [1.5, 2.5]
    )


def test_selector_defaults_match_paper():
    selector = GeoDetectorSelector()

    assert selector.discretizer == "natural_breaks"
    assert selector.max_bins == 5
    assert selector.top_k is None
    assert selector.alpha is None
    assert selector.strategy == "interaction"
    assert selector.top_interactions == 10
    assert SELECTION_STRATEGIES == ("interaction", "factor")


def _interaction_frame():
    a = np.array([1.0] * 8 + [2.0] * 8)
    b = np.array([1.0, 2.0] * 8)
    c = np.array([1.0, 1.0, 2.0, 2.0] * 4)
    d = np.array([1.0] * 4 + [2.0] * 4 + [1.0] * 4 + [2.0] * 4)
    y = pd.Series(5.0 * a + 10.0 * (b == c) + 0.01 * np.arange(16))
    return pd.DataFrame({"a": a, "b": b, "c": c, "d": d}), y


def test_selector_interaction_strategy_ranks_by_network_degree():
    frame, y = _interaction_frame()

    selector = GeoDetectorSelector(strategy="interaction", top_interactions=1)
    selector.fit(frame, y)

    # "a" has the highest single-factor q but degree zero in the top
    # interaction network; factor ranking would return ["a", "d", "c", "b"].
    # "b" and "c" tie on degree and fall back to q order.
    assert selector.selected_features_ == ["c", "b", "a", "d"]
    assert selector.interaction_degree_.tolist() == [1, 1, 0, 0]
    assert selector.interaction_degree_.index.tolist() == ["c", "b", "a", "d"]
    assert selector.interaction_degree_.index.name == "factor"
    assert selector.interaction_degree_.name == "degree"
    assert selector.interaction_degree_.dtype == np.int64
    assert selector.interactions_["q_interaction"].is_monotonic_decreasing
    first = selector.interactions_.iloc[0]
    assert [first["factor_1"], first["factor_2"]] == ["c", "b"]


def test_selector_factor_strategy_keeps_q_ranking():
    frame, y = _interaction_frame()

    selector = GeoDetectorSelector(strategy="factor")
    selector.fit(frame, y)

    assert selector.selected_features_ == ["a", "d", "c", "b"]
    assert selector.interactions_ is None
    assert selector.interaction_degree_ is None


def test_selector_top_interactions_truncation_changes_degrees():
    frame, y = _interaction_frame()

    selector = GeoDetectorSelector(strategy="interaction", top_interactions=3)
    selector.fit(frame, y)

    assert selector.selected_features_ == ["a", "c", "d", "b"]
    assert selector.interaction_degree_.tolist() == [2, 2, 1, 1]


def test_selector_interaction_top_k_truncates_selected():
    frame, y = _interaction_frame()

    selector = GeoDetectorSelector(
        strategy="interaction", top_interactions=1, top_k=2
    ).fit(frame, y)

    assert selector.selected_features_ == ["c", "b"]
    assert selector.transform(frame).columns.tolist() == ["c", "b"]


def _alpha_frame():
    a = np.tile([1.0] * 8 + [2.0] * 8, 4)
    b = np.tile([1.0, 2.0] * 8, 4)
    c = np.tile([1.0, 1.0, 2.0, 2.0] * 4, 4)
    d = np.tile([1.0] * 4 + [2.0] * 4 + [1.0] * 4 + [2.0] * 4, 4)
    y = pd.Series(5.0 * a + 10.0 * (b == c) + 4.0 * d + 0.02 * np.arange(64))
    return pd.DataFrame({"a": a, "b": b, "c": c, "d": d}), y


def test_selector_alpha_filters_factors_before_network():
    frame, y = _alpha_frame()

    selector = GeoDetectorSelector(strategy="interaction", alpha=0.05)
    selector.fit(frame, y)

    # Only "a" and "d" survive alpha; "b" would dominate the unfiltered
    # interaction network but never enters it.
    assert selector.selected_features_ == ["a", "d"]
    members = set(selector.interactions_["factor_1"]) | set(
        selector.interactions_["factor_2"]
    )
    assert members == {"a", "d"}
    assert selector.interaction_degree_.tolist() == [1, 1]


def test_selector_interaction_fewer_than_two_candidates():
    frame, y = _alpha_frame()

    selector = GeoDetectorSelector(strategy="interaction", alpha=0.001)
    selector.fit(frame, y)

    assert selector.selected_features_ == ["a"]
    assert selector.interactions_.empty
    expected_columns = interaction_detector(
        frame.assign(y=y), y="y", factors=["a", "b"]
    ).columns.tolist()
    assert selector.interactions_.columns.tolist() == expected_columns
    assert selector.interaction_degree_.tolist() == [0]
    assert selector.interaction_degree_.index.tolist() == ["a"]


def test_selector_accepts_ordered_categorical_target():
    frame = pd.DataFrame(
        {
            "x": [1.0, 1.1, 8.0, 8.1, 1.2, 8.2],
            "z": [0.0, 0.2, 0.1, 0.3, 0.05, 0.25],
        }
    )
    y = pd.Series(
        pd.Categorical(
            ["low", "mid", "high", "low", "mid", "high"],
            categories=["low", "mid", "high"],
            ordered=True,
        )
    )

    selector = GeoDetectorSelector().fit(frame, y)

    assert selector.selected_features_


def test_selector_rejects_unordered_multiclass_string_target():
    frame = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]})
    y = pd.Series(["low", "mid", "high", "low", "mid", "high"])

    with pytest.raises(ValidationError, match="ordered categoricals"):
        GeoDetectorSelector().fit(frame, y)


def test_selector_rejects_target_length_mismatch():
    frame = pd.DataFrame({"x": [1.0, 2.0, 3.0]})

    with pytest.raises(ValidationError, match="same length"):
        GeoDetectorSelector().fit(frame, pd.Series([0.0, 1.0]))


@pytest.mark.parametrize(
    "overrides",
    [
        {"strategy": "bogus"},
        {"top_interactions": 0},
        {"top_interactions": True},
        {"top_k": 0},
        {"top_k": True},
    ],
)
def test_selector_rejects_invalid_params(overrides):
    frame = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
    with pytest.raises(ValidationError):
        GeoDetectorSelector(**overrides).fit(frame, pd.Series([0.0, 1.0, 0.0]))


def _reference_jenks_natural_breaks(values, num_intervals):
    sorted_values = np.sort(np.asarray(values, dtype=float))
    unique_values = np.unique(sorted_values)
    class_count = min(num_intervals, unique_values.size)
    if class_count <= 1:
        return np.asarray([], dtype=float)

    lower_class_limits = np.zeros(
        (sorted_values.size + 1, class_count + 1), dtype=np.int64
    )
    variance_combinations = np.full(
        (sorted_values.size + 1, class_count + 1), np.inf, dtype=float
    )

    for index in range(1, class_count + 1):
        lower_class_limits[1, index] = 1
        variance_combinations[1, index] = 0.0

    for sample_count in range(2, sorted_values.size + 1):
        value_sum = 0.0
        squared_sum = 0.0
        weight = 0
        variance = 0.0

        for offset in range(1, sample_count + 1):
            lower_index = sample_count - offset + 1
            value = sorted_values[lower_index - 1]
            value_sum += value
            squared_sum += value * value
            weight += 1
            variance = squared_sum - (value_sum * value_sum) / weight
            previous_index = lower_index - 1

            if previous_index == 0:
                continue
            for class_index in range(2, class_count + 1):
                candidate = (
                    variance + variance_combinations[previous_index, class_index - 1]
                )
                if candidate < variance_combinations[sample_count, class_index]:
                    lower_class_limits[sample_count, class_index] = lower_index
                    variance_combinations[sample_count, class_index] = candidate

        lower_class_limits[sample_count, 1] = 1
        variance_combinations[sample_count, 1] = variance

    class_maxima = [sorted_values[-1]]
    sample_count = sorted_values.size
    for class_index in range(class_count, 1, -1):
        lower_index = lower_class_limits[sample_count, class_index]
        class_maxima.append(sorted_values[lower_index - 2])
        sample_count = lower_index - 1
    class_maxima.append(sorted_values[0])
    class_maxima = sorted(class_maxima)

    cutpoints = []
    for class_maximum in class_maxima[1:-1]:
        greater_values = sorted_values[sorted_values > class_maximum]
        if greater_values.size:
            cutpoints.append((class_maximum + greater_values[0]) / 2.0)
        else:
            cutpoints.append(class_maximum)
    return np.asarray(cutpoints, dtype=float)


def test_jenks_natural_breaks_matches_scalar_dp_reference():
    rng = np.random.default_rng(20250619)
    arrays = []
    for _ in range(20):
        n = int(rng.integers(2, 301))
        arrays.append(rng.normal(size=n) * rng.uniform(0.5, 100.0))
    for _ in range(15):
        n = int(rng.integers(2, 301))
        arrays.append(rng.uniform(-1000.0, 1000.0, n))
    for _ in range(15):
        n = int(rng.integers(2, 301))
        arrays.append(rng.integers(0, 10, n).astype(float))
    for _ in range(5):
        arrays.append(np.full(int(rng.integers(2, 50)), rng.uniform()))
    for _ in range(5):
        arrays.append(rng.normal(size=int(rng.integers(2, 5))))

    for values in arrays:
        k = int(rng.integers(2, 10))
        expected = _reference_jenks_natural_breaks(values, k)
        actual = _jenks_natural_breaks(values, k)
        assert np.allclose(
            actual, expected, rtol=0, atol=1e-12
        ), f"k={k}: {actual!r} != {expected!r}"
