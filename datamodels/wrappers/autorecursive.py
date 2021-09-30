import numpy as np

from datamodels import Model


class AutoRecursive:
    """
    This is a wrapper for autorecursive prediction.
    It can be used for simulation-like, time series prediction

    It takes a trained model (M) and, given an intial set of features (y0), predicts 
    a variable number of steps into the future (y1 ... yn).

    Generally, the intial set of features matches the model output, just offset by one time step.
    i.e. M(yn) = yn+1.

    It is also possible to have a model that uses sets of features from different time steps as 
    input.
    i.e. M(yn-k, ... yn) = yn+1
    in this case the wrapper always replaces the most recent value along the time axis with 
    the prediction
    e.g. M(yn) 
    

    INPUTS:
    y0s: inital set of features, of shape (lookback + 1, state_variables)
    num_predictions, Integer, number of steps to predict
    inputs: optional set of input features that are not used recursively,
       in the shape of (n, lookback + 1, input features) or (n, 1, input features)

       must be at least as long as the number of predictions,
       if the time axis is 1 then in the set of y0s the lookback + 1 is flattened into a single input vector
       (1, [lookback + 1] * state_variables) so that the inputs can be of the same shape.

       otherwise the time axis must be of lookback + 1.

    """
    def __init__(self, model: Model):
        self.model = model

    def predict(self, y0s, num_predictions: int, inputs=None):
        if y0s.ndim != 2:
            raise ValueError(
                f'y0s must have two dimensons: (lookback + 1, features), but was {y0s.shape}.'
            )

        time_steps = y0s.shape[0]
        num_features = y0s.shape[1]

        if inputs is None:
            inputs = np.empty((num_predictions, time_steps, 0))

        if inputs.ndim != 3:
            raise ValueError(
                f'inputs must have three dimensons: (n > {num_predictions}, {time_steps} or 1, input features), but was {inputs.shape}.'
            )

        num_inputs = inputs.shape[0]
        time_steps_inputs = inputs.shape[1]

        if time_steps != time_steps_inputs and time_steps_inputs != 1:
            raise ValueError(
                'time axis of inputs has to either match the time axis of the y0s'
                f'or 1. i.e. (n > {num_predictions}, {time_steps}, input features)'
                f'or (n > {num_predictions}, 1, input features)'
                f'but was {input.shape}'
            )


        if num_inputs < num_predictions:
            raise ValueError(
                f'not enough inputs to generate desired number of predictons'
                f'(in: {num_inputs}, pred: {num_predictions}'
            )
    
        y = y0s
        for i in range(num_predictions):
            yc = y[i:]
            if time_steps_inputs == 1:
                yc = np.reshape(yc, (1, time_steps * num_features))
            x = np.concatenate((yc, inputs[i]), axis=-1)[np.newaxis, ...]
            yn = self.model.predict(x)
            y = np.concatenate((y, yn))

        return y[y0s.shape[0]:]
