import pickle

from . import Model


class LassoRegression(Model):

    def __init__(self, parameters=None, **kwargs):
        super().__init__(**kwargs)

        if parameters is None:
            parameters = {'alpha':0.5}

        from sklearn.linear_model import Lasso
        self.model = Lasso(**parameters)

    def reshape_data(self, x):
        if x.ndim == 3:
            x = x.reshape(x.shape[0], -1)
        return x

    def train_model(self, x_train, y_train, **kwargs):
        self.model.fit(x_train, y_train)

    def predict_model(self, x):
        y = self.model.predict(x)
        return y.reshape(y.shape[0],1)

    def save(self, path="data/models/DUMMY.txt"):
        super(LassoRegression, self).save(path)
        with open(f'{path}/model.pickle', 'wb') as file:
            pickle.dump(self.model, file)

    def load_model(self, path="data/models/DUMMY.txt"):
        super(LassoRegression, self).load_model(path)
        with open(f'{path}/model.pickle', 'rb') as file:
            self.model = pickle.load(file)
