import pickle

from . import Model


class RuleFitRegression(Model):

    def __init__(self, parameters=None, **kwargs):
        super().__init__(**kwargs)

        if parameters is None:
            parameters = {}

        from rulefit.rulefit import RuleFit
        self.model = RuleFit(**parameters)

    def get_rules(self):
        rules = self.model.get_rules()
        return rules[rules.coef != 0].sort_values("support", ascending=False)

    def reshape_x(self, arr):
        if arr.shape[1] == arr.shape[2] == 1:
            arr = arr.ravel()
        else:
            arr = arr.reshape(arr.shape[0], arr.shape[1] * arr.shape[2])
        return arr

    def reshape_y(self, arr):
        if arr.shape[1] == arr.shape[2] == 1:
            arr = arr.ravel()
        else:
            raise RuntimeError("RuleFit cannot predict anything other than single values.")
        return arr

    def train_model(self, x, y, **kwargs):
        x = self.reshape_x(x)
        y = self.reshape_y(y)
        self.model.fit(x, y)

    def predict_model(self, x):
        x = self.reshape_x(x)
        return self.model.predict(x)

    def save(self, path="data/models/RuleFit"):
        super(RuleFitRegression, self).save(path)
        with open(f"{path}/model.pickle", "wb") as file:
            pickle.dump(self.model, file)

    def load_model(self, path="data/models/RuleFit"):
        super(RuleFitRegression, self).load_model(path)
        with open(f"{path}/model.pickle", "rb") as file:
            self.model = pickle.load(file)