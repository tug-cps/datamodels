import pickle
import numpy as np

from . import Model


class WeightedLS(Model):

    def __init__(self, parameters=None, **kwargs):
        super().__init__(**kwargs)
        if parameters is None:
            parameters = {'l1_wt':0, 'method':'elastic_net', 'alpha':1}
        self.l1_wt = parameters.pop('l1_wt', 0)
        self.method = parameters.pop('method','elastic_net')
        self.alpha = parameters.pop('alpha', 1)
        self.model = None
        self.coeffs = None


    def reshape_data(self, x):
        if x.ndim == 3:
            x = x.reshape(x.shape[0], -1)
        return x

    def train_model(self, x_train, y_train, **kwargs):
        from statsmodels.regression.linear_model import WLS
        y_train = y_train.reshape(y_train.shape[0])
        self.model = WLS(y_train, x_train)
        self.coeffs = self.model.fit_regularized(method=self.method, alpha=self.alpha, L1_wt = self.l1_wt).params
        #self.coeffs = self.model.fit().params

    def predict_model(self, x):
        if self.model is None:
            raise ValueError('The model is not trained yet.')
        result = np.array(self.model.predict(self.coeffs, exog=x))
        if result.ndim == 1:
            result = np.expand_dims(result, axis=-1)
        return result

    def save(self, path="data/models/DUMMY.txt"):
        super(WeightedLS, self).save(path)
        with open(f'{path}/model.pickle', 'wb') as file:
            pickle.dump([self.l1_wt, self.method, self.alpha, self.model, self.coeffs], file)

    def load_model(self, path="data/models/DUMMY.txt"):
        super(WeightedLS, self).load_model(path)
        with open(f'{path}/model.pickle', 'rb') as file:
            [self.l1_wt, self.method, self.alpha, self.model, self.coeffs] = pickle.load(file)
