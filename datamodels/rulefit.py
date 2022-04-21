import pickle
import numpy as np

from . import Model

from rulefit.rulefit import RuleFit


class RuleFitRegression(Model):

    def __init__(self, parameters=None, **kwargs):
        super().__init__(**kwargs)
        parameters = {} if parameters is None else parameters
        self.model = RuleFit(**parameters)

    def reshape_data(self, x):
        if x.ndim == 3:
            x = x.reshape(x.shape[0], -1)
        return x

    def train_model(self, x_train, y_train, **kwargs):
        if y_train.ndim > 1:
            if y_train.shape[1] > 1:
                raise ValueError('The RuleFit currently only supports models with a single output feature.')
        y_train = y_train.ravel()
        self.model.fit(x_train,y_train, list(self.get_expanded_feature_names()))

    def predict_model(self, x):
        result = self.model.predict(x)
        if result.ndim == 1:
            result = np.expand_dims(result, axis=-1)
        return result

    def save(self, path="data/models/DUMMY.txt"):
        super().save(path)
        with open(f'{path}/model.pickle', 'wb') as file:
            pickle.dump([self.model], file)

    def load_model(self, path="data/models/DUMMY.txt"):
        super().load_model(path)
        with open(f'{path}/model.pickle', 'rb') as file:
            [self.model] = pickle.load(file)

    def get_rules(self):
        rules = self.model.get_rules()
        return rules[rules.coef != 0].sort_values("support", ascending=False)
