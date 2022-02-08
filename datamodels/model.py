import os
import pickle
import sys

import numpy as np

from abc import abstractmethod

from .processing import DataScaler, IdentityScaler
from .processing.shape import get_windows


class Model:
    
    @staticmethod
    def load(path="models/EXAMPLE"):
        """

        DO NOT OVERRIDE THIS METHOD | to implement override load_model() instead.

        this allows you to instantiate a subclass from file.

        """
        with open(f"{path}/params.pickle", "rb") as file:
            model_type = pickle.load(file)[0]

        parent_name = ".".join(__name__.split(".")[:-1])
        instance = getattr(sys.modules[parent_name], model_type)()
        instance.load_model(path)
        return instance

    def __init__(
        self,
        name="",
        x_scaler_class=IdentityScaler,
        y_scaler_class=IdentityScaler,
        **kwargs,
    ):
        self.model_type = self.__class__.__name__
        self.name = name
        self.x_shape = None
        self.y_shape = None

        self.x_scaler = x_scaler_class()
        self.y_scaler = y_scaler_class()

    def train(
        self,
        x_train,
        y_train,
        lookback: int,
        lookahead: int,
        predict_sequence: bool = False,
        shuffle_data: bool = True,
    ):
        """

        DO NOT OVERRIDE THIS METHOD | to implement override train_model() instead.

        Parameters
        ----------
        x_train : array_like
            the array containing the input features.
        y_train : array_like
            the array containing the target features.
        lookback : int
            feature window time axis; if 0, feature window is [(f_0 ... f_n)].
        lookahead : int
            target window time axis; offset between t_0 and the end of target window.
        predict_sequence : bool
            whether target window is a sequence or a sigle value,
            if False the model predicts the value that is t_0 + lookahead.
        shuffle_data : bool
            whether to shuffle the training data.

        """
        self.x_scaler.fit(x_train)
        self.y_scaler.fit(y_train)

        x = self.x_scaler.transform(x_train)
        y = self.y_scaler.transform(y_train)

        x_windows, y_windows = get_windows(
            x, lookback, y, lookahead, targets_as_sequence=predict_sequence
        )

        if shuffle_data:
            indices = np.arange(x_windows.shape[0])
            np.random.shuffle(indices)
            x_windows = x_windows[indices]
            y_windows = y_windows[indices]

        self.x_shape = x_windows.shape[1:]
        self.y_shape = y_windows.shape[1:]

        self.train_model(x_windows, y_windows)

    @abstractmethod
    def train_model(self, x_train, y_train):
        raise NotImplementedError(
            "you called the train function on the abstract model class."
        )

    def predict(self, x):
        """

        DO NOT OVERRIDE THIS METHOD | to implement override predict_model() instead.

        Parameters
        ----------

        x : array_like
            (batch of) feature window the model should predict for, shape is (batch, lookback, input features)

        Returns
        -------
        array_like
            the target windows
            shape (batch, lookahead or 1 [depending on whether the model was trained to predict sequences or not], target features)

        """

        if not x.ndim == 3:
            raise RuntimeError(
                f"x must be an array of shape (batch, input time axis, input features)\n"
                f"but is {x.shape}"
            )

        x = self.x_scaler.transform(x)
        y = self.predict_model(x)

        y = y.reshape(y.shape[0], *self.y_shape)
        y = self.y_scaler.inverse_transform(y)

        return y

    @abstractmethod
    def predict_model(self, x):
        raise NotImplementedError(
            "you called the train function on the abstract model class."
        )

    @abstractmethod
    def save(self, path="models/EXAMPLE"):
        if os.path.isdir(path):
            print(f"{path} already exists, overwriting ..")
        else:
            os.mkdir(path)
        with open(f"{path}/params.pickle", "wb") as file:
            pickle.dump(
                [self.model_type, self.name, self.x_shape, self.y_shape], file
            )

        self.x_scaler.save(f"{path}/x_scaler.pickle")
        self.y_scaler.save(f"{path}/y_scaler.pickle")

    @abstractmethod
    def load_model(self, path="models/EXAMPLE"):
        with open(f"{path}/params.pickle", "rb") as file:
            params = pickle.load(file)
            self.name = params[1]
            self.x_shape = params[2]
            self.y_shape = params[3]

        self.x_scaler = DataScaler.load(f"{path}/x_scaler.pickle")
        self.y_scaler = DataScaler.load(f"{path}/y_scaler.pickle")
