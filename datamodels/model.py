import os
import pickle
import sys

import numpy as np

from abc import abstractmethod

from .processing import DataScaler, IdentityScaler


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
        **kwargs,
    ):
        self.model_type = self.__class__.__name__
        self.name = name

        self.x_shape = None
        self.y_shape = None

        self.x_scaler = IdentityScaler()
        self.y_scaler = IdentityScaler()

    def set_x_scaler(self, x_scaler: DataScaler):
        if x_scaler.was_fitted():
            self.x_scaler = x_scaler
        else:
            raise RuntimeError(
                "you should only pass a scaler that was fit to the distribution.\n"
                "i.e call x_scaler.fit(distribution) before setting it."
            )

    def set_y_scaler(self, y_scaler: DataScaler):
        if y_scaler.was_fitted():
            self.y_scaler = y_scaler
        else:
            raise RuntimeError(
                "you should only pass a scaler that was fit to the distribution.\n"
                "i.e call y_scaler.fit(distribution) before setting it."
            )

    def train(
        self,
        x,
        y,
        shuffle_data: bool = True,
    ):
        """

        DO NOT OVERRIDE THIS METHOD | to implement override train_model() instead.

        Parameters
        ----------
        x : array_like
            the array containing the input features,
            shape is (batch, lookback + 1, input features)

        y : array_like
            the array containing the target features,
            shape is (batch, lookahead + 1 or 1, output features)

        shuffle_data : bool
            whether to shuffle the training data.

        """
        if not x.ndim == 3:
            raise RuntimeError(
                f"x must be an array of shape (batch, lookback + 1, input features)\n"
                f"but is {x.shape}"
            )

        if not y.ndim == 3:
            raise RuntimeError(
                f"y must be an array of shape (batch, lookahead + 1 or 1, output features)\n"
                f"but is {y.shape}"
            )

        self.x_shape = x.shape[1:]
        self.y_shape = y.shape[1:]

        x = self.x_scaler.transform(x)
        y = self.y_scaler.transform(y)

        if shuffle_data:
            indices = np.arange(x.shape[0])
            np.random.shuffle(indices)
            x = x[indices]
            y = y[indices]

        self.train_model(x, y)

    @abstractmethod
    def train_model(self, x, y):
        raise NotImplementedError(
            "you called the train function on the abstract model class."
        )

    def predict(self, x):
        """

        DO NOT OVERRIDE THIS METHOD | to implement override predict_model() instead.

        Parameters
        ----------

        x : array_like
            (batch of) feature window the model should predict for,
            shape is (batch, lookback + 1, input features)

        Returns
        -------
        array_like
            the target windows
            shape (batch, lookahead + 1 or 1 [depending on whether the model was trained to predict sequences or not], target features)

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
            pickle.dump([self.model_type, self.name, self.x_shape, self.y_shape, self.feature_names], file)

        self.x_scaler.save(f"{path}/x_scaler.pickle")
        self.y_scaler.save(f"{path}/y_scaler.pickle")

    @abstractmethod
    def load_model(self, path="models/EXAMPLE"):
        with open(f"{path}/params.pickle", "rb") as file:
            params = pickle.load(file)
            self.name = params[1]
            self.x_shape = params[2]
            self.y_shape = params[3]
            self.feature_names = params[4]

        self.x_scaler = DataScaler.load(f"{path}/x_scaler.pickle")
        self.y_scaler = DataScaler.load(f"{path}/y_scaler.pickle")

    def set_feature_names(self, feature_names):
        """ Set input feature names
            DO NOT OVERRIDE THIS METHOD
            to implement the set feature names function for different models override set_feature_names_model
        """
        self.feature_names = feature_names
        # If necessary, additional steps in feature name setting
        self.set_feature_names_model(feature_names)

    def set_feature_names_model(self, feature_names):
        """ Internal method - should be overridden by child classes """
        pass

    def get_feature_names_model(self):
        """
        Get feature names for model - override if necessary
        """
        return self.feature_names

    def get_estimator(self):
        """
        Get sklearn estimator from model
        @return estimator if exists, else None.
        """
        return getattr(self, "model", None)

    @classmethod
    def from_name(cls, model_type="LinearRegression", **kwargs):
        """
        Create object from name.
        @param model_type: model type (must be in same package)
        @return: object
        """
        return cls.cls_from_name(model_type)(**kwargs)

    @classmethod
    def cls_from_name(cls, model_type="LinearRegression"):
        """
        Get class type from name.
        @param model_type: Model type (must be in same package)
        @return: type
        """
        parent_name = '.'.join(__name__.split('.')[:-1])
        return getattr(sys.modules[parent_name], model_type, None)