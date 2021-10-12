import os
import numpy as np

from pathlib import Path

import pytest

from datamodels.processing import DataScaler, Standardizer, Normalizer, RobustStandardizer, IdentityScaler

"""
Data

"""
array = np.array([
    [1, -1, 5],
    [2, 0, 50],
    [3, 1, 500]
])

standardized_array = np.array([
    [-1.22474487, -1.22474487, -0.80538727],
    [0., 0., -0.60404045],
    [1.22474487, 1.22474487, 1.40942772]
])

robust_scaled_array = np.array([
    [-1., -1., -0.18181818],
    [0., 0., 0.],
    [1., 1., 1.81818182]])

normalized_array = np.array([
    [0, 0, 0],
    [0.5, 0.5, 0.09090909],
    [1, 1, 1]
])


"""
parameterized tests

"""
@pytest.mark.parametrize('scaler', [
    IdentityScaler(),
    Normalizer(),
    Standardizer(),
    RobustStandardizer()
])
def test_save_and_load_scaler(scaler):
    test_file_path = Path(__file__).parent.joinpath('testScaler.pickle')

    scaler.fit(np.ones((5,)))
    scaler.save(test_file_path)

    scaler_from_file = DataScaler.load(test_file_path)
    
    for attr in scaler_from_file.__dict__.values():
        assert attr is not None

    os.remove(test_file_path)


@pytest.mark.parametrize('data', [
    np.zeros(5),
    np.array([1, 2, 3, 4, 5]),
    array
])
def test_identity_scaler(data):
    scaler = IdentityScaler().fit(data)
    transformed_data = scaler.transform(data)

    assert np.all(np.isclose(transformed_data, data))


@pytest.mark.parametrize('data, expected', [
    (np.array([1, 2, 3, 4, 5]), np.array([-1.41421356, -0.70710678, 0., .70710678, 1.41421356])),
    (array, standardized_array)
])
def test_standardize(data, expected):
    scaler = Standardizer().fit(data)
    transformed_data = scaler.transform(data)

    assert np.all(np.isclose(transformed_data, expected))


@pytest.mark.parametrize('data, expected', [
    (np.array([1, 2, 3, 4, 5]), np.array([-1, -.5, 0., .5, 1])),
    (array, robust_scaled_array)
])
def test_scale_robust(data, expected):
    scaler = RobustStandardizer().fit(data)
    transformed_data = scaler.transform(data)

    assert np.all(np.isclose(transformed_data, expected))


@pytest.mark.parametrize('data, expected', [
    (np.array([1, 2, 3, 4, 5]), np.array([0, .25, .5, .75, 1])),
    (array, normalized_array)
])
def test_normalize(data, expected):
    scaler = Normalizer().fit(data)
    transformed_data = scaler.transform(data)

    assert np.all(np.isclose(transformed_data, expected))


@pytest.mark.parametrize('data, expected', [
    (np.zeros(5), np.zeros(5)),
    (np.zeros((2, 3)), np.zeros((2, 3)))
])
def test_standardize_with_all_zeros(data, expected):
    scaler = Standardizer().fit(data)

    assert np.all(np.isclose(scaler.transform(data), expected))


@pytest.mark.parametrize('data, expected', [
    (np.zeros(5), np.zeros(5)),
    (np.zeros((2, 3)), np.zeros((2, 3)))
])
def test_scale_robust_with_all_zeros(data, expected):
    scaler = RobustStandardizer().fit(data)

    assert np.all(np.isclose(scaler.transform(data), expected))


@pytest.mark.parametrize('data, expected', [
    (np.zeros(5), np.zeros(5)),
    (np.zeros((2, 3)), np.zeros((2, 3)))
])
def test_normalize_with_all_zeros(data, expected):
    scaler = Normalizer().fit(data)

    assert np.all(np.isclose(scaler.transform(data), expected))


@pytest.mark.parametrize('scaler', [
    Normalizer(), Standardizer(), RobustStandardizer()
])
def test_inverse_transform(scaler):
    data = np.array([1, 2, 3, 4, 5])
    scaler.fit(data)

    transformed_data = scaler.transform(data)
    inverse_transformation = scaler.inverse_transform(transformed_data)

    assert np.all(np.isclose(inverse_transformation, data))

    data = array
    scaler.fit(data)

    transformed_data = scaler.transform(data)
    inverse_transformation = scaler.inverse_transform(transformed_data)

    assert np.all(np.isclose(inverse_transformation, data))

@pytest.mark.parametrize('scaler', [
    Normalizer(), Standardizer(), RobustStandardizer()
])
def test_inverse_transform_all_zeros(scaler):
    data = np.zeros(5)
    scaler.fit(data)

    transformed_data = scaler.transform(data)
    inverse_transformation = scaler.inverse_transform(transformed_data)

    assert np.all(np.isclose(inverse_transformation, data))
