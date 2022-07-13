import numpy as np

import pytest

from datamodels.validation import metrics

# test sanity check for inputs
@pytest.mark.parametrize("y_true, y_pred", [(np.ones((5,)), np.ones((10,)))])
def test_prevent_incorrect_dimensions(y_true, y_pred):
    with (pytest.raises(ValueError)):
        metrics.prevent_incorrect_dimensions(y_true, y_pred)


# actually test metrics

a = np.random.random((5,))
b = np.random.random((5, 1))
c = np.random.random((5, 10))
d = np.random.random((5, 10, 1))
e = np.random.random((5, 10, 2))
ones = np.ones((5, 10, 2))
zeros = np.zeros((5, 10, 2))


@pytest.mark.parametrize(
    "y_true, y_pred, expected",
    [
        (a, a, np.ones((1,)).squeeze()),
        (b, b, np.ones((1,))),
        (c, c, np.ones((10,))),
        (d, d, np.ones((10, 1))),
        (e, e, np.ones((10, 2))),
    ],
)
def test_rsquared(y_true, y_pred, expected):
    rsq = metrics.rsquared(y_true, y_pred)

    assert rsq.shape == expected.shape
    assert np.all(np.isclose(rsq, expected))


@pytest.mark.parametrize(
    "y_true, y_pred, expected",
    [
        (a, a, np.zeros((1,)).squeeze()),
        (b, b, np.zeros((1,))),
        (c, c, np.zeros((10,))),
        (d, d, np.zeros((10, 1))),
        (e, e, np.zeros((10, 2))),
        (ones, zeros, np.ones((10, 2))),
    ],
)
def test_cvrmse(y_true, y_pred, expected):
    cvrmse = metrics.cvrmse(y_true, y_pred)

    assert cvrmse.shape == expected.shape
    assert np.all(np.isclose(cvrmse, expected))


@pytest.mark.parametrize(
    "y_true, y_pred, expected",
    [
        (a, a, np.zeros((1,)).squeeze()),
        (b, b, np.zeros((1,))),
        (c, c, np.zeros((10,))),
        (d, d, np.zeros((10, 1))),
        (e, e, np.zeros((10, 2))),
        (ones, zeros, 100 * np.ones((10, 2))),
    ],
)
def test_mape(y_true, y_pred, expected):
    mape = metrics.mape(y_true, y_pred)

    assert mape.shape == expected.shape
    assert np.all(np.isclose(mape, expected))
