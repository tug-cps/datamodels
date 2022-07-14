from . import LinearModel


class LinearRegression(LinearModel):

    def __init__(self, parameters=None, **kwargs):
        super().__init__(**kwargs)

        if parameters is None:
            parameters = {"n_jobs": -1}

        from sklearn.linear_model import LinearRegression

        self.model = LinearRegression(**parameters)
