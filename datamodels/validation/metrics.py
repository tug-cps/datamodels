import numpy as np

import copy

def prevent_zeros(value):
    """
    this ensures that the value does not contain zeros.

    it can be used to prevent division by zero.

    :param value: scalar or array-like
    :return: value without zeros
    """
    if np.isscalar(value):
        return value if value != 0 else 1.

    corrected_value = copy.deepcopy(value)
    corrected_value[corrected_value == 0] = 1.0
    return corrected_value

def cvrmse(y_true, y_pred):
    mse = np.square(np.subtract(y_true, y_pred)).mean()
    return np.sqrt(mse) / prevent_zeros(y_true.mean())


def mape(y_true, y_pred):
    return (100 * np.abs(y_true - y_pred) / prevent_zeros(y_true)).mean()


def all_metrics(y_true, y_pred):
    return {
        'CV-RMS': cvrmse(y_true, y_pred),
        'MAPE': mape(y_true, y_pred),
    }
