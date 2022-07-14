import numpy as np

from statsmodels.regression.linear_model import WLS
from sklearn.base import BaseEstimator

from . import LinearModel


class wls_wrapper(BaseEstimator):
    def __init__(self, **params):
        self.set_params(**params)

    def set_params(self, **params):
        self.l1_wt = params.pop("l1_wt", 0)
        self.method = params.pop("method", "elastic_net")
        self.alpha = params.pop("alpha", 1)
        return self

    def get_params(self):
        return {"l1_wt": self.l1_wt, "method": self.method, "alpha": self.alpha}

    def fit(self, x_train, y_train, **kwargs):
        self.estimator = WLS(y_train, x_train)
        self.coef_ = self.estimator.fit_regularized(
            method=self.method, alpha=self.alpha, L1_wt=self.l1_wt, **kwargs
        ).params

    def predict(self, x_test, **kwargs):
        if self.estimator is None:
            raise RuntimeError("The model is not trained yet.")
        return self.estimator.predict(self.coef_, exog=x_test)


class WeightedLS(LinearModel):
    def __init__(self, parameters=None, **kwargs):
        super().__init__(**kwargs)
        if parameters is None:
            parameters = {"l1_wt": 0, "method": "elastic_net", "alpha": 1}
        self.model = wls_wrapper(**parameters)

    def reshape_x(self, arr):
        if arr.shape[1] == arr.shape[1] == 1:
            arr = arr.ravel()
        else:
            arr = arr.reshape(arr.shape[0], arr.shape[1] * arr.shape[2])
        return arr

    def reshape_y(self, arr):
        if arr.shape[1] == arr.shape[1] == 1:
            arr = arr.ravel()
        else:
            raise RuntimeError("WLS cannot predict anything other than single values.")
        return arr

    def train_model(self, x, y, **kwargs):
        x = self.reshape_x(x)
        y = self.reshape_y(y)
        self.model.fit(x, y)