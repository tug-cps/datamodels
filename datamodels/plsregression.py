from . import LinearModel


class PLSRegression(LinearModel):

    def __init__(self, parameters=None, **kwargs):
        super().__init__(**kwargs)

        if parameters is None:
            parameters = {'n_components':2}

        from sklearn.cross_decomposition import PLSRegression
        self.model = PLSRegression(**parameters)
