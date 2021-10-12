import numpy as np

import pytest

from datamodels import (
    Model, 
    LinearRegression, 
    RandomForestRegression,
    SupportVectorRegression,
    XGBoost,
    NeuralNetwork,
    VanillaLSTM,
    ConvolutionNetwork,
    ConvolutionLSTM,
    EncoderDecoderLSTM
)


@pytest.mark.parametrize('model_class', [
    LinearRegression, 
    RandomForestRegression,
    SupportVectorRegression,
    XGBoost,
    NeuralNetwork,
    VanillaLSTM,
    ConvolutionNetwork,
    ConvolutionLSTM,
    EncoderDecoderLSTM
])
def test_save_and_load_model(model_class, tmpdir):
    test_model_path = tmpdir.join('testModel')

    model_name = 'testModel'
    x_train = np.random.random((5, 3, 1))
    y_train = np.random.random((5, 1))
    input_shape = x_train.shape[1:]

    model = model_class(name=model_name)
    model.train(x_train, y_train)
    model.save(test_model_path)

    model_from_file = Model.load(test_model_path)

    assert model_from_file.name == model_name
    assert model_from_file.input_shape == input_shape
    assert model_from_file.x_scaler is not None
    assert model_from_file.y_scaler is not None

    y_pred = model_from_file.predict(x_train)
    assert y_pred.shape == (5, 1)


