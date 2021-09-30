from tensorflow import keras
from tensorflow.keras import layers
from datamodels import NeuralNetwork


def build_model(input_shape: tuple, target_shape: tuple) -> keras.Model:
    return keras.Sequential([
        layers.Input(input_shape),
        layers.Conv1D(filters=64, kernel_size=2,
                            strides=1, padding="same",
                            activation="relu",
                      ),
        layers.Conv1D(filters=64, kernel_size=2,
                      strides=1, padding="same",
                      activation="relu",
                      ),
        layers.LSTM(64, return_sequences=True, activation="tanh"),
        layers.Flatten(),
        layers.Dense(32),
        layers.Dense(target_shape[0]),
        keras.layers.Lambda(lambda x: x * 200)
    ])

class ConvolutionLSTM(NeuralNetwork):
    def __init__(self, build_function=build_model, **kwargs):
        super().__init__(build_function=build_function, **kwargs)