import numpy as np
import math

import pytest

from datamodels.validation import metrics

# test sanity check for inputs
@pytest.mark.parametrize('y_true, y_pred', [
    (np.ones((5,)), np.ones((10,))),
    (np.ones((10, 2)), np.ones((10, 2)))
])
def test_prevent_incorrect_dimensions(y_true, y_pred):
    with(pytest.raises(ValueError)):
        metrics.prevent_incorrect_dimensions(y_true, y_pred)

# actually test metrics
@pytest.mark.parametrize('y_true, y_pred, expected', [
    (np.arange(5), np.arange(5), 1),
    (np.arange(5)[..., np.newaxis], np.arange(5)[..., np.newaxis], 1)
])
def test_rsquared(y_true, y_pred, expected):
    assert math.isclose(metrics.rsquared(y_true, y_pred), expected)


@pytest.mark.parametrize('y_true, y_pred, expected', [
    (np.arange(5), np.arange(5), 0),
    (np.ones(5), np.zeros(5), 1)
])
def test_cvrmse(y_true, y_pred, expected):
    assert math.isclose(metrics.cvrmse(y_true, y_pred), expected)


@pytest.mark.parametrize('y_true, y_pred, expected', [
    (np.arange(5), np.arange(5), 0),
    (np.ones(5), np.zeros(5), 100)
])
def test_mape(y_true, y_pred, expected):    
    assert math.isclose(metrics.mape(y_true, y_pred), expected)