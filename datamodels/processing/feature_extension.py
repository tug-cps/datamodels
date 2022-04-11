import os
import sys
import numpy as np
import pickle
from typing import List

from sklearn.preprocessing import SplineTransformer, PolynomialFeatures
from abc import abstractmethod

class FeatureExpansion:
    features_to_expand: List[bool] = None
    selected_features: List[bool] = None
    @staticmethod
    def load(path='expander.pickle'):
        with open(path, 'rb') as file:
            attrs = pickle.load(file)
            type = attrs[0]
            instance = getattr(sys.modules['datamodels.processing'], type)()
            instance.set_attrs(attrs[1:])

        return instance

    @abstractmethod
    def set_attrs(self, attrs):
            raise NotImplementedError()

    def set_feature_select(self, selected_features):
        self.selected_features = selected_features

    @abstractmethod
    def get_feature_names_model(self, feature_names=None):
        raise NotImplementedError()

    def get_feature_names(self, feature_names=None):
        feature_names_to_expand = np.array(feature_names)[self.features_to_expand] if self.features_to_expand is not None else feature_names
        feature_names_basic = list(np.array(feature_names)[np.bitwise_not(self.features_to_expand)]) if self.features_to_expand is not None else []
        # Add: feature names basic features + feature names expanded features
        feature_names_tr = feature_names_basic + list(self.get_feature_names_model(feature_names_to_expand))
        return list(np.array(feature_names_tr)[self.selected_features]) if self.selected_features is not None else feature_names_tr

    def fit(self, x=None, y=None):
        if x.ndim == 3:
            x = x.reshape((x.shape[0], x.shape[1] * x.shape[2]))
        x_to_expand = x[:,self.features_to_expand] if self.features_to_expand is not None else x
        self.fit_transformer(x_to_expand, y)

    def fit_transform(self, x=None, y=None):
        self.fit(x, y)
        return self.transform(x)

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
        x_to_expand = x_reshaped[:, self.features_to_expand] if self.features_to_expand is not None else x_reshaped
        # Transform samples
        x_expanded = self.transform_samples(x_to_expand)
        # Add basic features to expanded features if necessary
        x_expanded = np.hstack((x_reshaped[:, np.bitwise_not(self.features_to_expand)], x_expanded)) if self.features_to_expand is not None else x_expanded
        # Select features if necessary
        x_expanded = x_expanded[:, self.selected_features] if self.selected_features is not None else x_expanded
        # Reshape to 3D if necessary
        x_expanded = x_expanded.reshape((x.shape[0], x.shape[1], int(x_expanded.shape[1] / x.shape[1]))) if x.ndim == 3 else x_expanded
        return x_expanded

    @abstractmethod
    def save(self, path='expander.pickle'):
        if os.path.isfile(path):
            print(f'{path} already exists, overwriting ..')

    @staticmethod
    def load_expanders(path):
        expanders = []
        for root, dirs, files in os.walk(path):
            for filename in files:
                print(filename)
                if "expander" in filename:
                    expanders.append(FeatureExpansion.load(f'{path}/{filename}'))
        return expanders

    @staticmethod
    def save_expanders(path, expanders):
        for index, expander in enumerate(expanders):
            expander.save(f'{path}/expander_{index}.pickle')


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

    def save(self, path="expander.pickle"):
        with open(path, 'wb') as file:
            pickle.dump([
                self.__class__.__name__,
                self.selected_features,
                self.model], file)



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

    def save(self, path="expander.pickle"):
        with open(path, 'wb') as file:
            pickle.dump([
                self.__class__.__name__,
                self.selected_features, self.model], file)

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

    def save(self, path="expander.pickle"):
        with open(path, 'wb') as file:
            pickle.dump([self.__class__.__name__], file)

