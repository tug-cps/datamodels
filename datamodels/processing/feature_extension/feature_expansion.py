import numpy as np
from typing import List

from sklearn.base import TransformerMixin
from abc import abstractmethod
from .store_interface import StoreInterface


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

    ################################################## Internal methods - override these ###############################

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

