import json
from typing import Tuple

from tensorflow import keras
from tensorflow.keras import layers

from . import Model


def build_model(input_shape: Tuple, target_shape: Tuple) -> keras.Model:
    """
    DO NOT CHANGE this file if you want to use a different build function |
    create your own build function that matches this function's signature and pass it to the network's
    constructor.

    Parameters
    ----------
    input_shape : Tuple
        shape of one input sample, shape is (lookback + 1, input features).

    target_shape : Tuple
        shape of one output sample, shape is (lookahead + 1 or 1, target features).
    
    Returns
    -------
    keras.Model
        a keras model.

    """
    hidden_layer_size = 64
    return keras.Sequential(
        [
            layers.Input(shape=input_shape),
            layers.Flatten(),
            layers.Dense(units=hidden_layer_size, activation="relu"),
            layers.Dense(units=hidden_layer_size, activation="relu"),
            layers.Dense(units=target_shape[0]),
        ]
    )


def compile_model(model: keras.Model):
    """
    DO NOT CHANGE this file if you want to use a different compile function |
    create your own compile function that matches this function's signature and pass it to the network's
    constructor.
    
    Parameters
    ----------
    model : keras.Model
        a keras model.

    """
    optimizer = keras.optimizers.RMSprop()
    model.compile(loss="mse", optimizer=optimizer)


def train_model(model: keras.Model, x_train, y_train) -> keras.callbacks.History:
    """
    DO NOT CHANGE this file if you want to use a different train function |
    create your own train function that matches this function's signature and pass it to the network's
    constructor.

    Parameters
    ----------
    x_train : array_like
        a batch of feature windows, shape is (batch, lookback + 1, input features).

    y_train : array_like
        a batch of target windows, shape is (batch, lookahead + 1 or 1, target features).
    
    Returns
    -------
    keras.callbacks.History
        a keras history object with the training history.

    """

    return model.fit(x_train, y_train, epochs=10, batch_size=24, validation_split=0.2)


class NeuralNetwork(Model):
    """
    the NeuralNetwork class acts as a wrapper for networks.
    the neural network can be customized by passing a custom function for any of the three parameters in the constructor.
    if nothing else is passed to the constructor the class uses the implementations above to build, compile and train the model.

    DO NOT CHANGE the functions here if you want a different network, 
    instead implement any of the three functions and pass them to the constructor in the file where you use the network.

    """

    def __init__(
        self,
        build_function=build_model,
        compile_function=compile_model,
        train_function=train_model,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.build_function = build_function
        self.compile_function = compile_function
        self.train_function = train_function

        self.model = None
        self.history = None

    def train_model(self, x_train, y_train, **kwargs):
        self.model = self.build_function(self.x_shape, self.y_shape)

        self.compile_function(self.model)

        print(self.model.summary())

        self.history = self.train_function(self.model, x_train, y_train).history

    def predict_model(self, x):
        return self.model.predict(x)

    def save(self, path="data/models/NeuralNetwork"):
        super(NeuralNetwork, self).save(path)
        with open(f"{path}/architecture.json", "w") as file:
            json.dump(self.model.to_json(), file)

        self.model.save_weights(f"{path}/weights.h5")

    def load_model(self, path="data/models/NeuralNetwork"):
        super(NeuralNetwork, self).load_model(path)
        with open(f"{path}/architecture.json", "r") as file:
            self.model = keras.models.model_from_json(json.load(file))

        self.model.load_weights(f"{path}/weights.h5")
