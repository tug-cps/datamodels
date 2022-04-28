import os
import numpy as np
from typing import List

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import SplineTransformer, PolynomialFeatures
from sklearn.base import TransformerMixin
from abc import abstractmethod
from .StoreInterface import StoreInterface


class FeatureExpansion(TransformerMixin, StoreInterface):
    """
    Feature Expansion
    Base class for feature expansion transformers.
    Implements scikit-learn's TransformerMixin interface, allows storing and loading from pickle
    """
    features_to_expand: List[bool] = None
    selected_features: List[bool] = None

    def set_feature_select(self, selected_features):
        """
        Set feature select for feature expander
        @param selected_features: Vector of booleans to select whether to use featuress
        """
        self.selected_features = selected_features

    def get_feature_names_out(self, feature_names=None):
        """
        Get feature names
        @param feature_names: Input feature names
        @return: Expanded feature names
        """
        feature_names = np.array(feature_names)
        feature_names_to_expand = feature_names[self.features_to_expand] if self.features_to_expand is not None else feature_names
        feature_names_basic = feature_names[np.bitwise_not(self.features_to_expand)] if self.features_to_expand is not None else []
        # Add: feature names basic features + feature names expanded features
        if len(feature_names_to_expand) > 0:
            feature_names_tr = np.hstack((feature_names_basic, np.array(self._get_feature_names(feature_names_to_expand))))
        else:
            feature_names_tr = feature_names_basic
        feature_names_tr = feature_names_tr[self.selected_features] if self.selected_features is not None else feature_names_tr
        return list(feature_names_tr)

    def fit_transform(self, X, y=None, **fit_params):
        """
        Fit transformer - Overrides TransformerMixin method.
        @param x: Input feature vector (n_samples, n_features)
        @param y: Target feature vector (n_samples)
        """
        return self.fit(X, y).transform(X)

    def fit(self, x=None, y=None):
        """
        Fit transformer to samples. Calls self._fit
        @param x: Input feature vector (n_samples, n_features) or (n_samples, lookback, n_features)
        @param y: Target feature vector (n_samples)
        """
        if x.ndim == 3:
            x = x.reshape((x.shape[0], x.shape[1] * x.shape[2]))
        x_to_expand = x[:,self.features_to_expand] if self.features_to_expand is not None else x
        if x_to_expand.shape[1] > 0:
            self._fit(x_to_expand, y)
        return self

    def transform(self, X):
        """
        Transform features. Calls self._transform
        @param x: Input feature vector (n_samples, n_features) or (n_samples, lookback, n_features)
        @return: Transformed sample vector (n_samples, n_features_expanded) or (n_samples, lookback, n_features_expanded)
        """
        # Reshape if necessary
        x_reshaped = X.reshape(X.shape[0], X.shape[1] * X.shape[2]) if X.ndim == 3 else X
        # Select features to expand
        if self.features_to_expand is not None:
            x_to_expand = x_reshaped[:, self.features_to_expand]
            x_basic = x_reshaped[:, np.bitwise_not(self.features_to_expand)]
            if x_to_expand.shape[1] > 0:
                x_expanded = self._transform(x_to_expand)
                # Add basic features to expanded features
                x_expanded = np.hstack((x_basic, x_expanded))
            else:
                x_expanded = x_basic
        else:
            x_expanded = self._transform(x_reshaped)
        # Select features if necessary
        x_expanded = x_expanded[:, self.selected_features] if self.selected_features is not None else x_expanded
        # Reshape to 3D if necessary
        x_expanded = x_expanded.reshape((X.shape[0], X.shape[1], int(x_expanded.shape[1] / X.shape[1]))) if X.ndim == 3 else x_expanded
        return x_expanded

    @classmethod
    def load_expanders(cls, path):
        """
        Load list of expander pickle files from directory.
        @param path: path to directory containing pickle files
        @return: sklearn pipeline of expanders
        """
        expanders = []
        for _, _, files in os.walk(path):
            for filename in files:
                if "expander" in filename:
                    expanders.append(cls.load_pkl(path, filename))
        return cls.create_pipeline(expanders)

    @classmethod
    def save_expanders(cls, path, expander_pipeline):
        """
        Load list of expander pickle files from directory.
        @param path: path to directory
        @param expander_pipeline: pipeline of expanders
        """
        for index, expander in enumerate(cls.get_list_expanders(expander_pipeline)):
            expander.save_pkl(path,f'expander_{index}.pickle')

    @staticmethod
    def create_pipeline(list_expanders):
        """
        Create sklearn pipeline from list of expanders
        @param list_expanders: list of expanders
        @return: expander_pipeline: pipeline of expanders
        """
        return make_pipeline(*list_expanders, 'passthrough')

    @staticmethod
    def get_list_expanders(expander_pipeline):
        """
        Create sklearn pipeline from list of expanders
        @param list_expanders: expander_pipeline: pipeline of expanders
        @return: list of expanders
        """
        return [expander for (name, expander) in expander_pipeline.steps[:-1]]

    @abstractmethod
    def _get_feature_names(self, feature_names=None):
        """
        Get feature names - Override this method.
        @param feature_names: Input feature names
        @return: Expanded feature names
        """
        raise NotImplementedError()

    @abstractmethod
    def _fit(self, X, y=None):
        """
        Fit transformer to samples - override this method.
        @param x: Input feature vector (n_samples, n_features)
        @param y: Target feature vector (n_samples)
        """
        raise NotImplementedError()

    @abstractmethod
    def _transform(self, X):
        """
        Transform features - Override this method.
        @param x: Input feature vector (n_samples, n_features)
        @return: Transformed sample vector (n_samples, n_features_expanded)
        """
        raise NotImplementedError()


class SplineInterpolator(FeatureExpansion):
    """
    Spline Interpolation
    Expands features by spline bases -
     see https://scikit-learn.org/dev/modules/generated/sklearn.preprocessing.SplineTransformer.html
    Implements scikit-learn's TransformerMixin interface.
    """
    def __init__(self, **kwargs):
        self.model = SplineTransformer(**kwargs)

    def _get_feature_names(self, feature_names=None):
        return self.model.get_feature_names_out(feature_names)

    def _fit(self, x=None, y=None):
        self.model = self.model.fit(x, y)

    def _transform(self, x=None):
        return self.model.transform(x)


class PolynomialExpansion(FeatureExpansion):
    """
    Polynomial Feature Expansion
    Expands features by polynomials of variable order -
    https://scikit-learn.org/dev/modules/generated/sklearn.preprocessing.PolynomialFeatures.html
    Implements scikit-learn's TransformerMixin interface.
    """
    def __init__(self, **kwargs):
        self.model = PolynomialFeatures(**kwargs)

    def _fit(self, x=None, y=None):
        self.model.fit(x, y)

    def _transform(self, x=None):
        return self.model.transform(x)

    def _get_feature_names(self, feature_names=None):
        return self.model.get_feature_names_out(feature_names)


class IdentityExpander(FeatureExpansion):
    """
    Identity Expander
    This class does not expand the features.
    Instantiate this class if you do not want to use any expansion.
    """
    def __init__(self):
        pass

    def _fit(self, x=None, y=None):
        pass

    def _transform(self, x=None):
        return x

    def _get_feature_names(self, feature_names=None):
        return feature_names


