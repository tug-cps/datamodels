import os
import pickle
import sys

import numpy as np

from abc import abstractmethod

from . shape import prevent_zeros


class DataScaler:

    @staticmethod
    def load(path='scalers.pickle'):
        with open(path, 'rb') as file:
            attrs = pickle.load(file)
            scaler_type = attrs[0]
            
            parent_name = '.'.join(__name__.split('.')[:-1])
            instance = getattr(sys.modules[parent_name], scaler_type)()
            instance.set_attrs(attrs[1:])

        return instance

    @abstractmethod
    def set_attrs(self, attrs):
            raise NotImplementedError()  

    @abstractmethod
    def fit(self, distribution):
        """
        for statistical models it is often necessary to scale datasets.

        to properly evaluate statistical models it is important that the validation data does not bias
        the training process, i.e. information about the validation/test data must not be used during the
        training phase. dataset metrics such as min, max, std, etc. are information about the data which is why
        validation data is typically scales with metrics from the training set.

        :param distribution: the distribution that supplies the metrics for scaling
        """
        raise NotImplementedError()

    @abstractmethod
    def transform(self, data):
        raise NotImplementedError()

    @abstractmethod
    def inverse_transform(self, data):
        raise NotImplementedError()

    @abstractmethod
    def save(self, path='scaler.pickle'):
        if os.path.isfile(path):
            print(f'{path} already exists, overwriting ..')

    @classmethod
    def from_name(cls, scaler_type="IdentityScaler", **kwargs):
        return cls.cls_from_name(scaler_type)(**kwargs)

    @classmethod
    def cls_from_name(cls, scaler_type="IdentityScaler"):
        parent_name = '.'.join(__name__.split('.')[:-1])
        return getattr(sys.modules[parent_name], scaler_type, None)


class IdentityScaler(DataScaler):

    def fit(self, distribution):
        return self

    def transform(self, data):
        return data

    def inverse_transform(self, data):
        return data

    def save(self, path='scaler.pickle'):
        super(IdentityScaler, self).save(path)
        with open(path, 'wb') as file:
            pickle.dump([
                self.__class__.__name__, 
            ], file)
    
    def set_attrs(self, attrs):
        pass


class Normalizer(DataScaler):

    def __init__(self):
        self.min = None
        self.max = None
        self.scale = None

    def fit(self, distribution):
        self.min = np.nanmin(distribution, axis=0)
        self.max = np.nanmax(distribution, axis=0)
        self.scale = prevent_zeros(self.max - self.min)
        return self

    def transform(self, data):
        """
        data is scaled such that it is 0 <= data <= 1.
        in a feature vector this puts all features to the same scale.

        this is useful for data that is not Gaussian distributed and might help with convergence.
        features are more consistent, however it can be disadvantageous if the scale between features is
        important.

        :param data: array-like, pd.DataFrame, numpy-array
        :return: data, normalized between 0 and 1
        """
        if self.min is None or self.max is None or self.scale is None:
            raise ValueError(
                "parameters not set, cannot transform data, you must call .fit(distribution) first."
            )

        return (data - self.min) / self.scale

    def inverse_transform(self, data):
        if self.min is None or self.max is None or self.scale is None:
            raise ValueError(
                "parameters not set, cannot transform data, you must call .fit(distribution) first."
            )
        return data * self.scale + self.min
    
    def save(self, path='scaler.pickle'):
        super(Normalizer, self).save(path)
        with open(path, 'wb') as file:
            pickle.dump([
                self.__class__.__name__,
                self.min,
                self.max,
                self.scale
            ], file)

    def set_attrs(self, attrs):
        self.min = attrs[0]
        self.max = attrs[1]
        self.scale = attrs[2]


class Standardizer(DataScaler):

    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, distribution):
        self.mean = np.nanmean(distribution, axis=0)
        self.std = prevent_zeros(np.nanstd(distribution, axis=0))
        return self

    def transform(self, data):
        """
        data is scaled to 0 mean and 1 standard deviation.
        mostly helpful when data follows a Gaussian distribution. for PCA features have to be centered around the
        mean.

        :param data: array-like, pd.DataFrame, numpy-array
        :return: data, scaled such that mean=0 and std=1
        """
        if self.mean is None or self.std is None:
            raise ValueError(
                "parameters not set, cannot transform data, you must call .fit(distribution) first."
            )

        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        if self.mean is None or self.std is None:
            raise ValueError(
                "parameters not set, cannot transform data, you must call .fit(distribution) first."
            )
        return data * self.std + self.mean

    def save(self, path='scaler.pickle'):
        super(Standardizer, self).save(path)
        with open(path, 'wb') as file:
            pickle.dump([
                self.__class__.__name__,
                self.mean,
                self.std,
            ], file)

    def set_attrs(self, attrs):
        self.mean = attrs[0]
        self.std = attrs[1]


class RobustStandardizer(DataScaler):

    def __init__(self):
        self.median = None
        self.scale = None

    def fit(self, distribution):
        self.median = np.nanmedian(distribution, axis=0)
        q25 = np.nanquantile(distribution, .25, axis=0)
        q75 = np.nanquantile(distribution, .75, axis=0)
        self.scale = prevent_zeros(q75 - q25)
        return self

    def transform(self, data):
        """
        mean and variance are often influenced by outliers. using median and interquanitle range instead often
        improves standardization results.

        :param data: array-like, pd.DataFrame, numpy-array
        :return: standardized data, scaled by range between 1st and 3rd quantile
        """
        if self.median is None or self.scale is None:
            raise ValueError(
                "parameters not set, cannot transform data, you must call .fit(distribution) first."
            )
        return (data - self.median) / self.scale

    def inverse_transform(self, data):
        if self.median is None or self.scale is None:
            raise ValueError(
                "parameters not set, cannot transform data, you must call .fit(distribution) first."
            )
        return data * self.scale + self.median

    def save(self, path='scaler.pickle'):
        super(RobustStandardizer, self).save(path)
        with open(path, 'wb') as file:
            pickle.dump([
                self.__class__.__name__,
                self.median,
                self.scale,
            ], file)

    def set_attrs(self, attrs):
        self.median = attrs[0]
        self.scale = attrs[1]