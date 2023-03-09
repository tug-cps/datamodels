import numpy as np
import copy

from typing import Tuple


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


def split(data: np.ndarray, frac: float) -> Tuple[np.ndarray, np.ndarray]:
    if not 0 <= frac <= 1:
        raise ValueError('invalid fraction, must be between 0 and 1')

    index = int(frac * data.shape[0])

    if data.ndim == 1:
        first = data[:index]
        second = data[index:]
        return first, second

    if data.ndim > 1:
        first = data[:index, :]
        second = data[index:, :]
        return first, second


def get_windows(
        lookback: int,
        features: np.ndarray,
        lookahead: int,
        targets: np.ndarray=None,
        targets_as_sequence: bool=False
):
    """
    this generates feature and target windows of shapes that can be feed to a model.

    Parameters
    ----------  
    lookback : int
        feature window time axis; if 0, feature window is [(f_0 ... f_n)].
    features : array_like
        the array containing the input features.
    lookahead : int
        target window time axis; offset between t_0 and the end of target window.
        even if no targets are passed this is used to get the correct feature length.
        set to 0, if you don't need this check.  
    targets : array_like, optional
        the array containing the target features.
    targets_as_sequence: bool, optional
        whether target windows are a sequence between t_0 and the lookahead or just the value at t_0 + lookahead.

    Returns
    -------
    Tuple of np.ndarrays or single np.ndarray
        the feature windows, shape is (batch, lookback + 1, input features)
        [optional] the target features, shape is (batch, lookback + 1 or 1, target features)

    """
    if features.ndim != 2:
        raise RuntimeError(f'features must have shape (samples, input_features), '
                           f'but has {features.shape}')

    if targets is not None and targets.ndim != 2:
        raise RuntimeError(f'targets must have shape (samples, target_features), '
                           f'but has {targets.shape}')

    if targets is not None and features.shape[0] != targets.shape[0]:
        raise RuntimeError(f'features and targets must have the same length.\n'
                           f'features has: {features.shape}, targets has: {targets.shape}')

    samples = features.shape[0]
    start_index = lookback
    end_index = samples - lookahead

    if not start_index < end_index:
        raise RuntimeError(f'there are not enough samples in features and targets.\n'
                           f'samples: {samples}, lookback: {lookback}, lookahead {lookahead}\n'
                           f'so you need at least {lookback + lookahead + 1} sample(s).')

    feature_list = []
    target_list = []

    for i in range(start_index, end_index):
        feature_list.append(features[i - lookback: i + 1])
        if targets is not None:
            target_list.append(targets[i: i + lookahead + 1])

    x = np.array(feature_list)
    if targets is None: 
        return x
    
    y = np.array(target_list)
    if not targets_as_sequence:
        y = y[:, -1:, :]

    return x, y
