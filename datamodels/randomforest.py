import pickle

from . import Model


class RandomForestRegression(Model):
    def __init__(self, parameters=None, **kwargs):
        super().__init__(**kwargs)

        if parameters is None:
            parameters = {
                "bootstrap": True,
                "max_depth": None,
                "max_features": "auto",
                "min_samples_leaf": 1,
                "min_samples_split": 2,
                "n_estimators": 100,
            }

        from sklearn.ensemble import RandomForestRegressor

        self.model = RandomForestRegressor(**parameters)
    
    def get_coef(self):
        return self.model.feature_importances_

    def reshape(self, arr):
        if arr.shape[1] == arr.shape[1] == 1:
            arr = arr.ravel()
        else:
            arr = arr.reshape(arr.shape[0], arr.shape[1] * arr.shape[2])
        return arr

    def train_model(self, x, y, **kwargs):
        x = self.reshape(x)
        y = self.reshape(y)
        self.model.fit(x, y)

    def predict_model(self, x):
        x = self.reshape(x)
        return self.model.predict(x)

    def save(self, path="data/models/RandomForest"):
        super(RandomForestRegression, self).save(path)
        with open(f"{path}/model.pickle", "wb") as file:
            pickle.dump(self.model, file)

    def load_model(self, path="data/models/RandomForest"):
        super(RandomForestRegression, self).load_model(path)
        with open(f"{path}/model.pickle", "rb") as file:
            self.model = pickle.load(file)
