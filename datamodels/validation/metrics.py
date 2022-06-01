import copy
import numpy as np
from sklearn.metrics import r2_score
import RegscorePy


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


def prevent_incorrect_dimensions(y_true, y_pred):
    if not y_true.shape == y_pred.shape:
        raise ValueError('y_true and y_pred must have the same dimensions\n'
                         f'y_true was: {y_true.shape}, y_pred was: {y_pred.shape}')
    if y_true.ndim > 1 and y_true.shape[-1] != 1:
        raise ValueError('Cannot calculate this for multiple features, please make sure the arrays are either'
                         f'(samples, 1) or (samples, )\n'
                         f'y_true was: {y_true.shape}, y_pred was: {y_pred.shape}')


def rsquared(y_true, y_pred):
    prevent_incorrect_dimensions(y_true, y_pred)
    correlation_coefficients = np.corrcoef(y_true.flatten(), y_pred.flatten())
    return correlation_coefficients[0, 1] ** 2


def rsquared_sklearn(y_true, y_pred):
    prevent_incorrect_dimensions(y_true, y_pred)
    return r2_score(y_true,y_pred)


def rsquared_adj(y_true, y_pred, n_samples, n_predictors):
    prevent_incorrect_dimensions(y_true, y_pred)
    if n_samples == n_predictors + 1:
        raise ValueError('n_samples must not be equal n_predictors + 1.')
    return 1 - (1 - rsquared(y_true, y_pred)) * (n_samples - 1) / (n_samples - n_predictors - 1)


def rsquared_sklearn_adj(y_true, y_pred, n_samples, n_predictors):
    prevent_incorrect_dimensions(y_true, y_pred)
    if n_samples == n_predictors + 1:
        raise ValueError('n_samples must not be equal n_predictors + 1.')
    return 1 - (1 - rsquared_sklearn(y_true, y_pred)) * (n_samples - 1) / (n_samples - n_predictors - 1)


def rmse(y_true, y_pred):
    prevent_incorrect_dimensions(y_true, y_pred)
    return np.sqrt(np.square(np.subtract(y_true, y_pred)).mean())


def nrmse(y_true, y_pred):
    prevent_incorrect_dimensions(y_true, y_pred)
    return np.sqrt(np.square(np.subtract(y_true, y_pred)).mean()) / prevent_zeros(np.nanmax(y_true) - np.nanmin(y_true))


def mae(y_true, y_pred):
    prevent_incorrect_dimensions(y_true, y_pred)
    return np.abs(y_true - y_pred).mean()


def cvrmse(y_true, y_pred):
    prevent_incorrect_dimensions(y_true, y_pred)
    mse = np.square(np.subtract(y_true, y_pred)).mean()
    return np.sqrt(mse) / prevent_zeros(y_true.mean())


def mape(y_true, y_pred):
    prevent_incorrect_dimensions(y_true, y_pred)
    return (100 * np.abs(y_true - y_pred) / prevent_zeros(y_true)).mean()


def nmae(y_true, y_pred):
    prevent_incorrect_dimensions(y_true, y_pred)
    return np.abs(y_true - y_pred).mean() / prevent_zeros(np.nanmax(y_true) - np.nanmin(y_true))


def aic(y_true, y_pred, num_predictors):
    prevent_incorrect_dimensions(y_true, y_pred)
    return RegscorePy.aic.aic(y_true, y_pred, num_predictors)


def all_metrics(y_true, y_pred):
    return {
        'R2': rsquared(y_true, y_pred),
        'CV-RMS': cvrmse(y_true, y_pred),
        'MAPE': mape(y_true, y_pred),
        'R2_SKLEARN': rsquared_sklearn(y_true, y_pred),
        'MAE': mae(y_true, y_pred),
        'NMAE': nmae(y_true, y_pred),
        'RMS': rmse(y_true, y_pred),
        'NRMS': nrmse(y_true, y_pred)
    }