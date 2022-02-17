import os
import sys
import numpy as np
import pickle
from typing import List

from sklearn.preprocessing import SplineTransformer, PolynomialFeatures
from abc import abstractmethod

class FeatureExpansion:
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

    def get_feature_names_model(self, feature_names=None):
        return feature_names

    def get_feature_names(self, feature_names=None):
        features = self.get_feature_names_model(feature_names)
        return [feature for feature, select in zip(features, self.selected_features) if select] if self.selected_features else features

    def fit(self, x=None, y=None):
        if x.ndim == 3:
            self.fit_transformer(x[:,0, :],y)
        else:
            self.fit_transformer(x, y)

    @abstractmethod
    def fit_transformer(self, x=None, y=None):
        raise NotImplementedError()

    def transform_samples(self, x=None):
        raise NotImplementedError()

    def transform(self, x=None):
        transformed_features = np.zeros(x.shape)
        if x.ndim == 3:
            lookback_states = x.shape[1]
            for i in range(lookback_states):
                transformed_features[:,i,:] = self.transform_samples(x[:,i,:])
            if self.selected_features is not None:
                transformed_features = transformed_features[:,:,self.selected_features]

        if x.ndim == 2:
            transformed_features = self.transform_samples(x)
            if self.selected_features is not None:
                transformed_features = transformed_features[:,self.selected_features]

        return transformed_features

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
    degree: int = 2
    n_knots: int = 4
    extrapolation: str = "periodic"
    model = None

    def __init__(self, degree=2, n_knots=4, extrapolation="periodic"):
        self.degree = degree
        self.n_knots = n_knots
        self.extrapolation = extrapolation
        self.model = SplineTransformer(n_knots=self.n_knots, degree=self.degree, extrapolation=self.extrapolation)

    def set_attrs(self, attrs):
        for i, name in enumerate(["degree", "n_knots", "extrapolation", "model"]):
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
                self.degree,
                self.n_knots,
                self.extrapolation,
                self.model], file)



"""
Polynomial Feature Expansion 
Expands features by polynomials of variable order - 
https://scikit-learn.org/dev/modules/generated/sklearn.preprocessing.PolynomialFeatures.html
Returns expanded features and also the names
"""
class PolynomialExpansion(FeatureExpansion):
    degree: int = 2
    include_bias = False
    model = None

    def __init__(self, degree=2, include_bias=False):
        self.degree = degree
        self.include_bias = include_bias
        self.model = PolynomialFeatures(degree=self.degree, include_bias=include_bias)

    def set_attrs(self, attrs):
        for i, name in enumerate(["degree", "include_bias", "model"]):
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
                self.degree,
                self.include_bias, self.model], file)

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

    def save(self, path="expander.pickle"):
        with open(path, 'wb') as file:
            pickle.dump([self.__class__.__name__], file)

