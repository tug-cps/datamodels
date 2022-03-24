import pickle
import numpy as np

from . import LinearModel

from statsmodels.regression.linear_model import WLS
from sklearn.base import BaseEstimator


class wls_wrapper(BaseEstimator):
    def __init__(self, **params):
        self.set_params(**params)

    def set_params(self, **params):
        self.l1_wt = params.pop('l1_wt', 0)
        self.method = params.pop('method', 'elastic_net')
        self.alpha = params.pop('alpha', 1)
        return self

    def get_params(self, deep=True):
        return {'l1_wt': self.l1_wt, 'method': self.method, 'alpha': self.alpha}

    def fit(self, x_train, y_train, **kwargs):
        y_train = y_train.reshape(y_train.shape[0])
        self.estimator = WLS(y_train, x_train)
        self.coeffs = self.estimator.fit_regularized(method=self.method, alpha=self.alpha, L1_wt=self.l1_wt, **kwargs).params

    def predict(self, x_test, **kwargs):
        if self.estimator is None:
            raise ValueError('The model is not trained yet.')
        return np.array(self.estimator.predict(self.coeffs, exog=x_test))


class WeightedLS(LinearModel):

    def __init__(self, parameters=None, **kwargs):
        super().__init__(**kwargs)
        if parameters is None:
            parameters = {'l1_wt':0, 'method':'elastic_net', 'alpha':1}
        self.model = wls_wrapper(**parameters)
