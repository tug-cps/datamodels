import pickle

from . import Model


class SymbolicRegression(Model):

    def __init__(self, parameters=None, **kwargs):
        super().__init__(**kwargs)

        if parameters is None:
            parameters = {'population_size':50, 'stopping_criteria':0.0001, 'metric':'rmse'}

        from gplearn.genetic import SymbolicRegressor
        self.model = SymbolicRegressor(**parameters)

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
            raise RuntimeError("SymbolicRegression cannot predict anything other than single values.")
        return arr

    def train_model(self, x, y, **kwargs):
        x = self.reshape_x(x)
        y = self.reshape_y(y)
        self.model.fit(x, y)

    def predict_model(self, x):
        x = self.reshape_x(x)
        return self.model.predict(x)

    def save(self, path="data/models/SymbolicRegression"):
        super(SymbolicRegression, self).save(path)
        with open(f"{path}/model.pickle", "wb") as file:
            pickle.dump(self.model, file)

    def load_model(self, path="data/models/SymbolicRegression"):
        super(SymbolicRegression, self).load_model(path)
        with open(f"{path}/model.pickle", "rb") as file:
            self.model = pickle.load(file)