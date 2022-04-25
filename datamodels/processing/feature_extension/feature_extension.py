import os
import numpy as np
from typing import List

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import SplineTransformer, PolynomialFeatures
from sklearn.base import TransformerMixin
from abc import abstractmethod
from .StoreInterface import StoreInterface

class FeatureExpansion(TransformerMixin, StoreInterface):
    features_to_expand: List[bool] = None
    selected_features: List[bool] = None

    @abstractmethod
    def set_attrs(self, attrs):
            raise NotImplementedError()

    def set_feature_select(self, selected_features):
        self.selected_features = selected_features

    @abstractmethod
    def get_feature_names_model(self, feature_names=None):
        raise NotImplementedError()

    def get_feature_names_out(self, feature_names=None):
        feature_names = np.array(feature_names)
        feature_names_to_expand = feature_names[self.features_to_expand] if self.features_to_expand is not None else feature_names
        feature_names_basic = feature_names[np.bitwise_not(self.features_to_expand)] if self.features_to_expand is not None else []
        # Add: feature names basic features + feature names expanded features
        if len(feature_names_to_expand) > 0:
            feature_names_tr = np.hstack((feature_names_basic, np.array(self.get_feature_names_model(feature_names_to_expand))))
        else:
            feature_names_tr = feature_names_basic
        feature_names_tr = feature_names_tr[self.selected_features] if self.selected_features is not None else feature_names_tr
        return list(feature_names_tr)

    def fit(self, x=None, y=None):
        if x.ndim == 3:
            x = x.reshape((x.shape[0], x.shape[1] * x.shape[2]))
        x_to_expand = x[:,self.features_to_expand] if self.features_to_expand is not None else x
        if x_to_expand.shape[1] > 0:
            self.fit_transformer(x_to_expand, y)

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X, y)
        return self.transform(X)

    @abstractmethod
    def fit_transformer(self, x=None, y=None):
        raise NotImplementedError()

    @abstractmethod
    def transform_samples(self, x=None):
        raise NotImplementedError()

    def transform(self, x=None):
        # Reshape if necessary
        x_reshaped = x.reshape((x.shape[0], x.shape[1] * x.shape[2])) if x.ndim == 3 else x
        # Select features to expand
        if self.features_to_expand is not None:
            x_to_expand = x_reshaped[:, self.features_to_expand]
            x_basic = x_reshaped[:, np.bitwise_not(self.features_to_expand)]
            if x_to_expand.shape[1] > 0:
                x_expanded = self.transform_samples(x_to_expand)
                # Add basic features to expanded features
                x_expanded = np.hstack((x_basic, x_expanded))
            else:
                x_expanded = x_basic
        else:
            x_expanded = self.transform_samples(x_reshaped)
        # Select features if necessary
        x_expanded = x_expanded[:, self.selected_features] if self.selected_features is not None else x_expanded
        # Reshape to 3D if necessary
        x_expanded = x_expanded.reshape((x.shape[0], x.shape[1], int(x_expanded.shape[1] / x.shape[1]))) if x.ndim == 3 else x_expanded
        return x_expanded

    @abstractmethod
    def save(self, path='expander.pickle'):
        raise NotImplementedError()

    @staticmethod
    def load_expanders(path):
        expanders = []
        for root, dirs, files in os.walk(path):
            for filename in files:
                print(filename)
                if "expander" in filename:
                    expanders.append(FeatureExpansion.load_pkl(path, filename))
        return FeatureExpansion.create_pipeline(expanders)

    @staticmethod
    def save_expanders(path, expander_pipeline):
        for index, expander in enumerate(FeatureExpansion.get_list_expanders(expander_pipeline)):
            expander.save_pkl(path,f'expander_{index}.pickle')

    @staticmethod
    def create_pipeline(list_expanders):
        return make_pipeline(*list_expanders, 'passthrough')

    @staticmethod
    def get_list_expanders(expander_pipeline):
        return [expander for (name, expander) in expander_pipeline.steps[:-1]]


"""
Spline Interpolation 
Expands features by spline bases -
 see https://scikit-learn.org/dev/modules/generated/sklearn.preprocessing.SplineTransformer.html
Returns expanded features 
"""
class SplineInterpolator(FeatureExpansion):
    def __init__(self, **kwargs):
        self.model = SplineTransformer(**kwargs)

    def set_attrs(self, attrs):
        for i, name in enumerate(["selected_features", "model"]):
            setattr(self, name, attrs[i])

    def get_feature_names_model(self, feature_names=None):
        return self.model.get_feature_names_out(feature_names)

    def fit_transformer(self, x=None, y=None):
        self.model = self.model.fit(x, y)

    def transform_samples(self, x=None):
        return self.model.transform(x)



"""
Polynomial Feature Expansion 
Expands features by polynomials of variable order - 
https://scikit-learn.org/dev/modules/generated/sklearn.preprocessing.PolynomialFeatures.html
Returns expanded features and also the names
"""
class PolynomialExpansion(FeatureExpansion):
    model = None

    def __init__(self, **kwargs):
        self.model = PolynomialFeatures(**kwargs)

    def set_attrs(self, attrs):
        for i, name in enumerate(["selected_features", "model"]):
            setattr(self, name, attrs[i])

    def fit_transformer(self, x=None, y=None):
        self.model = self.model.fit(x,y)

    def transform_samples(self, x=None):
        return self.model.transform(x)

    def get_feature_names_model(self, feature_names=None):
        return self.model.get_feature_names_out(feature_names)


"""
Identity - if no expansion is used
"""
class IdentityExpander(FeatureExpansion):

    def __init__(self):
        pass

    def set_attrs(self, attrs):
        pass

    def fit_transformer(self, x=None, y=None):
        pass

    def transform_samples(self, x=None):
        return x

    def get_feature_names_model(self, feature_names=None):
        return feature_names


