from . import LinearModel


class RidgeRegression(LinearModel):

    def __init__(self, parameters=None, **kwargs):
        super().__init__(**kwargs)

        if parameters is None:
            parameters = {'alpha':0.5}

        from sklearn.linear_model import Ridge
        self.model = Ridge(**parameters)

