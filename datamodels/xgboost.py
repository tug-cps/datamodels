import pickle

from . import Model


class XGBoost(Model):
    def __init__(self, parameters=None, **kwargs):
        super().__init__(**kwargs)

        if parameters is None:
            parameters = {
                "n_estimators": 500,
                "max_depth": 4,
                "min_samples_split": 5,
                "learning_rate": 0.01,
                "loss": "ls",
            }

        from sklearn.ensemble import GradientBoostingRegressor

        self.model = GradientBoostingRegressor(**parameters)

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
            raise RuntimeError(
                "XGBoost cannot predict anything other than single values"
            )
        return arr

    def train_model(self, x_train, y_train, **kwargs):
        x_train = self.reshape_x(x_train)
        y_train = self.reshape_y(y_train)

        self.model.fit(x_train, y_train)

    def predict_model(self, x):
        x = self.reshape_x(x)
        return self.model.predict(x)

    def save(self, path="data/models/XGBoost"):
        super(XGBoost, self).save(path)
        with open(f"{path}/model.pickle", "wb") as file:
            pickle.dump(self.model, file)

    def load_model(self, path="data/models/XGBoost"):
        super(XGBoost, self).load_model(path)
        with open(f"{path}/model.pickle", "rb") as file:
            self.model = pickle.load(file)
