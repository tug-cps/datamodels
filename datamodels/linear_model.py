from . import Model

import pickle
import numpy as np
import pandas as pd


class LinearModel(Model):

    def __init__(self, parameters=None, **kwargs):
        super().__init__(**kwargs)
        from sklearn.dummy import DummyRegressor
        self.model = DummyRegressor()
        self.feature_names = kwargs.pop('feature_names',None)

    def reshape_x(self, x):
        if x.ndim == 3:
            x = x.reshape(x.shape[0], -1)
        return x

    def reshape_y(self, y):
        if y.ndim == 1:
            y = y.ravel()
        if y.ndim == 3:
            y = y.reshape(y.shape[0], -1)
        return y

    def train_model(self, x_train, y_train, **kwargs):
        # Add feature names
        x_train = self.reshape_x(x_train)
        y_train = self.reshape_y(y_train)
        if self.feature_names is not None:
            x_train = pd.DataFrame(data=x_train, columns=self.feature_names)
        self.model.fit(x_train, y_train)

    def predict_model(self, x):
        # Add feature names
        x = self.reshape_x(x)
        if self.feature_names is not None:
            x = pd.DataFrame(data=x, columns=self.feature_names)
        result = self.model.predict(x)
        if result.ndim == 1:
            result = np.expand_dims(result, axis=-1)
        return result

    def save(self, path="data/models/DUMMY.txt"):
        super().save(path)
        with open(f'{path}/model.pickle', 'wb') as file:
            pickle.dump(self.model, file)

    def load_model(self, path="data/models/DUMMY.txt"):
        super().load_model(path)
        with open(f'{path}/model.pickle', 'rb') as file:
            self.model = pickle.load(file)

    def get_coef(self):
        return self.model.coef_