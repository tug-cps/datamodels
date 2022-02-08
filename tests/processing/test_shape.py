import numpy as np

import pytest

from datamodels import processing

def test_prevent_zeros_scalar():
    data = 0
    corrected_data = processing.shape.prevent_zeros(data)

    assert corrected_data == 1

def test_prevent_zeros_all_zeros_array():
    data = np.zeros((4, ))
    corrected_data = processing.shape.prevent_zeros(data)

    assert np.all(np.isclose(corrected_data, np.ones_like(data)))

def test_prevent_zeros_one_zero_array():
    data = np.array([1., 1., 1., 0., 1.])
    corrected_data = processing.shape.prevent_zeros(data)

    assert np.all(np.isclose(corrected_data, np.ones_like(data)))

@pytest.mark.parametrize('lookback, lookahead', [
    (0, 0),
    (0, 1),
    (0, 2),
    (1, 1),
    (2, 1)
])
def test_get_windows(lookback, lookahead):
    samples = 5
    input_features = 4
    target_features = 1

    features = np.reshape(np.arange(samples * input_features), (samples, input_features))
    targets = np.reshape(np.arange(samples * target_features), (samples, target_features))

    x, y = processing.shape.get_windows(
        features, lookback, targets, lookahead, targets_as_sequence=True
    )

    expected_shape_x = (
        samples - lookback - lookahead,
        lookback + 1,
        input_features

    )
    expected_shape_y = (
        samples - lookback - lookahead,
        lookahead + 1,
        target_features
    )

    assert x.shape == expected_shape_x
    
    assert y.shape == expected_shape_y

    # assert that there are not duplicate entries
    _, counts = np.unique(x, return_counts=True, axis=0)
    assert np.all(counts == 1), f'not all samples are unique, x: {x}'

    _, counts = np.unique(y, return_counts=True, axis=0)
    assert np.all(counts == 1), f'not all samples are unique, y: {y}'

@pytest.mark.parametrize('lookback, lookahead', [
    (0, 0),
    (0, 1),
    (0, 2),
    (1, 1),
    (2, 1)
])
def test_get_windows_single_targets(lookback, lookahead):
    samples = 5
    input_features = 2
    target_features = 1

    features = np.reshape(np.arange(samples * input_features), (samples, input_features))
    targets = np.reshape(np.arange(samples * target_features), (samples, target_features))

    x, y = processing.shape.get_windows(
        features, lookback, targets, lookahead, targets_as_sequence=False
    )

    expected_shape_x = (
        samples - lookback - lookahead,
        lookback + 1,
        input_features
    )
    expected_shape_y = (
        samples - lookback - lookahead,
        1,
        target_features
    )

    assert x.shape == expected_shape_x
    
    assert y.shape == expected_shape_y

    # assert that there are not duplicate entries
    _, counts = np.unique(x, return_counts=True, axis=0)
    assert np.all(counts == 1), f'not all samples are unique, x: {x}'

    _, counts = np.unique(y, return_counts=True, axis=0)
    assert np.all(counts == 1), f'not all samples are unique, y: {y}'