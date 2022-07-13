from tensorflow import keras
from tensorflow.keras import layers

from . import NeuralNetwork


def build_model(input_shape: tuple, target_shape: tuple) -> keras.Model:
    return keras.Sequential(
        [
            layers.Input(shape=input_shape),
            layers.GRU(units=32),
            layers.Dense(units=target_shape[0])
        ]
    )


class GRU(NeuralNetwork):
    def __init__(self, build_function=build_model, **kwargs):
        super().__init__(build_function=build_function, **kwargs)
