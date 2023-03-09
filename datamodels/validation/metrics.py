import copy
import numpy as np


def prevent_zeros(value):
    """

    This ensures that the value does not contain zeros.
    It can be used to prevent division by zero.
    
    This method replaces 0 by the smallest possible value on the current machine.
    THINK ABOUT WHETHER THIS IS WHAT YOU NEED!!

    Parameters
    ----------

    x : scalar or array_like
        the value where zeros should be replaced.

    Returns
    -------
    scalar or array_like
        the input with the zeroes replaced by a very small value.

    """
    epsilon = np.finfo(np.float64).eps
    corrected_value = np.maximum(np.abs(value), epsilon)
    return corrected_value


def prevent_incorrect_dimensions(y_true, y_pred):
    if not y_true.shape == y_pred.shape:
        raise ValueError(
            "y_true and y_pred must have the same dimensions\n"
            f"y_true was: {y_true.shape}, y_pred was: {y_pred.shape}"
        )


def rsquared(y_true, y_pred):
    """

    The coefficient of determination captures how much of the variation in one dependent variable
    can be predicted from a (set of) independent varaible(s).

    It captures the relationship between one or more independent to ONE dependent variable,
    i.e. if the model predicts multiple dependen variables, rsquare has to be calculated for
    each one of them separately.
    (This is also true if the model predicts one variable at multiple time steps)

    Returns
    -------
    sclar or array-like
        an array where the rsquared is calculated along the batch dimension.
        or a single rquare value if the input is (batch, ).

    """
    prevent_incorrect_dimensions(y_true, y_pred)

    t_mean = y_true.mean(axis=0)
    ss_res = np.square(y_true - y_pred).sum(axis=0)
    ss_tot = np.square(y_true - t_mean).sum(axis=0)

    return 1 - ss_res / prevent_zeros(ss_tot)


def adjusted_rsquared(y_true, y_pred, variables):
    """

    Extra input variables tend to increase the rsquared without actually improving the goodness of fit.
    The adjusted rsquared penalizes the statistic depending on the number of variables in the model.

    """
    n = y_true.shape[0]
    p = variables
    return 1 - (1 - rsquared(y_true, y_pred)) * (n - 1) / (n - p - 1)

def mae(y_true, y_pred):
    """
    Computes mse over batch dimension, i.e. it performs no reduction along
    the feature axis;

    Reduction policy depends on the context;
    i.e. it should be handled by the user

    e.g. call .mean(axis=0) on the return to get an average over the time dimension
         call .mean(axis=1) on the return to get the average over the features or
         call .mean() to get the average over both dimensions.

    """
    prevent_incorrect_dimensions(y_true, y_pred)
    return np.abs(y_true - y_pred).mean(axis=0)

def nmae(y_true, y_pred):
    """
    Computes normalized mse over batch dimension, i.e. it performs no reduction along
    the feature axis;
    
    note that it uses the range of actual values (i.e. max - min) for normalization

    Reduction policy depends on the context;
    i.e. it should be handled by the user

    e.g. call .mean(axis=0) on the return to get an average over the time dimension
         call .mean(axis=1) on the return to get the average over the features or
         call .mean() to get the average over both dimensions.

    """
    return mae(y_true, y_pred) / prevent_zeros(np.nanmax(y_true) - np.nanmin(y_true))

def mse(y_true, y_pred):
    """
    Computes mse over batch dimension, i.e. it performs no reduction along
    the feature axis;

    Reduction policy depends on the context;
    i.e. it should be handled by the user

    e.g. call .mean(axis=0) on the return to get an average over the time dimension
         call .mean(axis=1) on the return to get the average over the features or
         call .mean() to get the average over both dimensions.

    """
    prevent_incorrect_dimensions(y_true, y_pred)
    return np.square(y_true - y_pred).mean(axis=0)


def rmse(y_true, y_pred):
    """
    Computes mse over batch dimension, i.e. it performs no reduction along
    the feature axis;

    Reduction policy depends on the context;
    i.e. it should be handled by the user

    e.g. call .mean(axis=0) on the return to get an average over the time dimension
         call .mean(axis=1) on the return to get the average over the features or
         call .mean() to get the average over both dimensions.

    """
    return np.sqrt(mse(y_true, y_pred))

def nrmse(y_true, y_pred):
    """
    Computes normalized mse over batch dimension, i.e. it performs no reduction along
    the feature axis;
    
    note that it uses the range of actual values (i.e. max - min) for normalization

    Reduction policy depends on the context;
    i.e. it should be handled by the user

    e.g. call .mean(axis=0) on the return to get an average over the time dimension
         call .mean(axis=1) on the return to get the average over the features or
         call .mean() to get the average over both dimensions.

    """
    return rmse(y_true, y_pred) / prevent_zeros(np.nanmax(y_true) - np.nanmin(y_true))

def cvrmse(y_true, y_pred):
    """
    CV is the ration of std to mean (std / mean)

    This gives the CV of the root of the sum of squared errors;
    which is the RMSE scaled to the mean of the true distribution.
    (so it's more compareable between distributions that have different magnitudes)

    Reduction policy depends on the context;
    i.e. it should be handled by the user

    e.g. call .mean(axis=0) on the return to get an average over the time dimension
         call .mean(axis=1) on the return to get the average over the features or
         call .mean() to get the average over both dimensions.

    Returns
    -------
    scalar or array-like
        the cvrmse along the batch dimension,
        i.e. it returns an array for multiple variables

    """
    prevent_incorrect_dimensions(y_true, y_pred)
    axis = 0
    return rmse(y_true, y_pred) / prevent_zeros(y_true.mean(axis))


def mape(y_true, y_pred):
    """

    The absolute error between prediction and true value,
    scaled by the mean of the true value.
    (so it's more compareable between distributions that have different magnitudes)

    Reduction policy depends on the context;
    i.e. it should be handled by the user

    e.g. call .mean(axis=0) on the return to get an average over the time dimension
         call .mean(axis=1) on the return to get the average over the features or
         call .mean() to get the average over both dimensions.

    Returns
    -------
    scalar or array-like
        the mape along the batch dimension,
        i.e. it returns an array for multiple variables

    """
    prevent_incorrect_dimensions(y_true, y_pred)
    return (100 * np.abs(y_true - y_pred) / prevent_zeros(y_true)).mean(axis=0)
