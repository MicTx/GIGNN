import numpy as np

from gignn.training import calculate_class_weights, set_random_seed


def test_calculate_class_weights_handles_non_contiguous_labels():
    encoded = np.array([0, 1, 1, 2, 2, 2])

    weights = calculate_class_weights(encoded)

    assert weights.shape == (3,)
    assert weights[0] > weights[1] > weights[2]


def test_set_random_seed_reproducibly_seeds_numpy():
    set_random_seed(123)
    first = np.random.random(3)
    set_random_seed(123)
    second = np.random.random(3)

    np.testing.assert_array_equal(first, second)
