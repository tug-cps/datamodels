import pickle
from . import Model


class SymbolicRegression(Model):

    def __init__(self, parameters=None, **kwargs):
        super().__init__(**kwargs)

        if parameters is None:
            parameters = {'population_size':50, 'stopping_criteria':0.0001, 'metric':'rmse'}

        from gplearn.genetic import SymbolicRegressor
        self.model = SymbolicRegressor(**parameters)

    def reshape(self, x):
        if x.ndim == 3:
            x = x.reshape(x.shape[0], -1)
        return x

    def train_model(self, x_train, y_train, **kwargs):
        self.model.fit(self.reshape(x_train), y_train)

    def predict_model(self, x):
        y = self.model.predict(self.reshape(x))
        return y.reshape(y.shape[0],1)

    def save(self, path="data/models/DUMMY.txt"):
        super(SymbolicRegression, self).save(path)
        with open(f'{path}/model.pickle', 'wb') as file:
            pickle.dump(self.model, file)

    def load_model(self, path="data/models/DUMMY.txt"):
        super(SymbolicRegression, self).load_model(path)
        with open(f'{path}/model.pickle', 'rb') as file:
            self.model = pickle.load(file)

    def get_program(self):
        return self.model._program