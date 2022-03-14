import numpy as np

from datamodels import Model
from datamodels.wrappers import AutoRecursive


def test_predict_without_inputs():

    lookback = 0
    predictions = 10
    number_input_features = 1
    number_output_features = number_input_features

    class TestModel(Model):
        def __init__(self) -> None:
            super().__init__()
            self.x_shape = (lookback + 1, number_input_features)
            self.y_shape = (1, number_output_features)

        def predict_model(self, x):
            # predicts zeros, one time-step (batch, 1, input features = output features)
            return np.zeros((x.shape[0], 1, x.shape[2]))

    model = TestModel()

    x = np.ones((1, lookback + 1, number_input_features))
    y_true = np.zeros((predictions, number_output_features))

    wrapper = AutoRecursive(model)
    y_pred = wrapper.predict(x[0], predictions)

    assert np.all(np.isclose(y_pred, y_true))

def test_predict_with_inputs():

    lookback = 0
    predictions = 10
    number_input_features = 1
    number_inputs = 3
    number_output_features = 1

    class TestModel(Model):
        def __init__(self) -> None:
            super().__init__()
            self.x_shape = (lookback + 1, number_input_features)
            self.y_shape = (1, number_output_features)

        def predict_model(self, x):
            # predicts zeros, one time-step (batch, 1, output features)
            return np.zeros((x.shape[0], 1, number_output_features))

    model = TestModel()

    x = np.ones((1, lookback + 1, number_input_features))
    inputs = np.random.random((predictions, lookback + 1, number_inputs))
    y_true = np.zeros((predictions, number_output_features))

    wrapper = AutoRecursive(model)
    y_pred = wrapper.predict(x[0], predictions, inputs)

    assert np.all(np.isclose(y_pred, y_true))