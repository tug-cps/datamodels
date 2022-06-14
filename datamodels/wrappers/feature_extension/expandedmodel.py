from . import StoreInterface, ExpanderSet
from ... import Model


class ExpandedModel(StoreInterface):
    """
    Expansion Model - contains set of feature expanders
    """
    model: Model
    expanders: ExpanderSet
    feature_names = None

    def __init__(self, model: Model, expanders: ExpanderSet, feature_names=None):
        self.model = model
        self.expanders = expanders
        self.feature_names = feature_names

    def train(self, X, y, **fit_params):
        """
        Predict - prediction with feature expansion
        @param X: tensor of shape (n_samples, lookback_horizon + 1, input_features)
        @param y: tensor of shape (n_samples, input_features)
        """
        self.model.input_shape = X.shape[1:]
        X = self.model.reshape_data(X)
        X, y = self.model.scale(X, y)
        X = self.expanders.fit_transform(X)
        self.model.train_model(X, y)

    def reshape_data(self, X):
        return self.model.reshape_data(X)

    def scale(self, X, y):
        return self.model.scale(X, y)

    def fit(self, X, y, **fit_params):
        """
        Predict - prediction with feature expansion
        @param X: tensor of shape (n_samples, lookback_horizon + 1, input_features)
        @param y: tensor of shape (n_samples, input_features)
        """
        self.train(X, y, **fit_params)

    def predict(self, X):
        """
        Predict - prediction with feature expansion
        @param X: tensor of shape (n_samples, lookback_horizon + 1, input_features)
        @return: tensor of shape (n_samples, input_features)
        """
        if not X.ndim == 3:
            raise RuntimeError(f'x must be an array of shape (samples, lookback_horizon + 1, input_features)\n'
                               f'but is {X.shape}')
        X = self.model.reshape_data(X)
        X = self.model.x_scaler.transform(X)
        X = self.expanders.transform(X)
        y = self.model.predict_model(X)
        if not y.shape[0] == X.shape[0]:
            raise AssertionError(f'samples in prediction do not match samples in input\n'
                                 f'expected: {X.shape[0]}, but is {y.shape[0]}.')

        if not y.ndim == 2:
            raise AssertionError(f'predictions must be two dimensional (samples, target_features)\n'
                                 f'but are {y.shape}.')
        return self.model.y_scaler.inverse_transform(y)

    def get_expanded_feature_names(self):
        """
        Get expanded feature names - get feature names after all expansion steps
        """
        return self.expanders.get_feature_names(self.feature_names)

    def set_feature_names(self, feature_names=None):
        """
        Set feature names for model
        """
        self.feature_names = feature_names
        self.model.set_feature_names(self.get_expanded_feature_names())

    def get_estimator(self):
        return self.model.get_estimator()

    @property
    def name(self):
        return self.model.name

    @name.setter
    def name(self, name=""):
        self.model.name = name


