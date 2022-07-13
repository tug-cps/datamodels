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
    EncoderDecoderLSTM,
    GRU,
    RecurrentNetwork
)


@pytest.mark.parametrize('model_class', [
    LinearRegression, 
    RandomForestRegression,
    NeuralNetwork,
    VanillaLSTM,
    ConvolutionNetwork,
    ConvolutionLSTM,
    GRU,
    RecurrentNetwork
])
def test_predict_sequence_save_and_load(model_class, tmpdir):
    test_model_path = tmpdir.join('testModel')

    model_name = 'testModel'

    lookback = 5
    lookahead = 1

    number_input_features = 3
    number_target_features = 1

    x_train = np.random.random((15, lookback + 1, number_input_features))
    y_train = np.random.random((15, lookahead + 1, number_target_features))
    x_shape = (lookback + 1, number_input_features)
    y_shape = (lookahead + 1, number_target_features)

    model = model_class(name=model_name)
    model.train(x_train, y_train)
    model.save(test_model_path)

    model_from_file = Model.load(test_model_path)

    assert model_from_file.name == model_name
    assert model_from_file.x_shape == x_shape
    assert model_from_file.y_shape == y_shape
    assert model_from_file.x_scaler is not None
    assert model_from_file.y_scaler is not None

    x_test = np.random.random((5, lookback + 1, number_input_features))

    y_pred = model_from_file.predict(x_test)
    assert y_pred.shape == (5, lookahead + 1, number_target_features)

@pytest.mark.parametrize('model_class', [
    LinearRegression, 
    RandomForestRegression,
    SupportVectorRegression,
    XGBoost,
    NeuralNetwork,
    VanillaLSTM,
    ConvolutionNetwork,
    ConvolutionLSTM,
    EncoderDecoderLSTM,
    GRU,
    RecurrentNetwork
])
def test_predict_single_value_save_and_load(model_class, tmpdir):
    test_model_path = tmpdir.join('testModel')

    model_name = 'testModel'

    lookback = 5
    lookahead = 1

    number_input_features = 3
    number_target_features = 1

    x_train = np.random.random((15, lookback + 1, number_input_features))
    y_train = np.random.random((15, 1, number_target_features))
    
    x_shape = (lookback + 1, number_input_features)
    y_shape = (1, number_target_features)

    model = model_class(name=model_name)
    model.train(x_train, y_train)
    model.save(test_model_path)

    model_from_file = Model.load(test_model_path)

    assert model_from_file.name == model_name
    assert model_from_file.x_shape == x_shape
    assert model_from_file.y_shape == y_shape
    assert model_from_file.x_scaler is not None
    assert model_from_file.y_scaler is not None

    x_test = np.random.random((5, lookback + 1, number_input_features))

    y_pred = model_from_file.predict(x_test)
    assert y_pred.shape == (5, 1, number_target_features)