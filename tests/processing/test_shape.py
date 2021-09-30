import numpy as np

import pytest

import datamodels.processing as processing

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

@pytest.mark.parametrize('lookback_horizon, prediction_horizon', [
    (0, 0),
    (0, 1),
    (1, 1),
    (2, 1)
])
def test_split_into_target_segments(lookback_horizon, prediction_horizon):
    samples = 5
    num_features = 4
    num_targets = 1

    features = np.reshape(np.arange(samples * num_features), (samples, num_features))
    targets = np.reshape(np.arange(samples * num_targets), (samples, num_targets))

    x, y = processing.shape.split_into_target_segments(
        features, targets, lookback_horizon, prediction_horizon
    )

    expected_shape_x = (
        samples - lookback_horizon - prediction_horizon,
        lookback_horizon + 1,
        num_features
    )
    expected_shape_y = (
        samples - lookback_horizon - prediction_horizon,
        num_targets
    )

    assert x.shape == expected_shape_x
    
    assert y.shape == expected_shape_y

    # assert that there are not duplicate entries
    _, counts = np.unique(x, return_counts=True, axis=0)
    assert np.all(counts == 1), f'not all samples are unique, x: {x}'

    _, counts = np.unique(y, return_counts=True, axis=0)
    assert np.all(counts == 1), f'not all samples are unique, y: {y}'

