import os
import pickle
import sys

from abc import abstractmethod

from . processing import DataScaler, IdentityScaler
from . processing.feature_extension import FeatureExpansion, IdentityExpander

class Model:

    @staticmethod
    def load(path='models/EXAMPLE'):
        """
        DO NOT OVERRIDE THIS METHOD

        this allows you to instantiate a subclass from file,
        it reads the class name from the pickle file
        to implement loading functionality that is specific to the subclass
        override load_model(...)

        """
        with open(f'{path}/params.pickle', 'rb') as file:
            model_type = pickle.load(file)[0]

        parent_name = '.'.join(__name__.split('.')[:-1])
        instance = getattr(sys.modules[parent_name], model_type)()        
        instance.load_model(path)
        return instance

    def __init__(self, name='', x_scaler_class=IdentityScaler, y_scaler_class=IdentityScaler, expanders=None, **kwargs):
        self.model_type = self.__class__.__name__
        self.name = name
        self.input_shape = None

        self.expanders = expanders if expanders is not None else [IdentityExpander()]
        self.x_scaler = x_scaler_class()
        self.y_scaler = y_scaler_class()

    def reshape_data(self, X):
        """
        Override this method to perform data transformation on the matrix of feature vectors
        BEFORE feature scaling.

        :param X: matrix of feature vectors, typically 2 or 3 dimensional
        :return: array-like, transformation of X.
        """
        return X

    def scale(self, x_train, y_train):
        self.x_scaler.fit(x_train)
        self.y_scaler.fit(y_train)

        return self.x_scaler.transform(x_train), self.y_scaler.transform(y_train)


    def train(self, x_train, y_train):
        """
        DO NOT OVERRIDE THIS METHOD

        override train_model with the actual implementation.

        this is the wrapper that ensures that the input data is
        _ reshaped (if necessary)
        _ preprocessed, i.e. the data_scaler is fitted to the distribution and training data transformed
        _ the train_model method (i.e. the actual method that fits the data model) is called AFTER these steps.

        Inputs:
            x_train: tensor of shape (samples, lookback_horizon + 1, input_features)
            y_train: tensor of shape (samples, target_features)
        """

        if not x_train.shape[0] == y_train.shape[0]:
            raise RuntimeError(f'number of samples in inputs and targets must match.\n'
                               f'X: {x_train.shape}, y: {y_train}')

        if not x_train.ndim == 3:
            raise RuntimeError(f'x_train must be an array of shape (samples, lookback_horizon + 1, num_features)\n'
                               f'but is {x_train.shape}')

        if not y_train.ndim == 2:
            raise RuntimeError(f'y_train must be an array of shape (samples, target_features)\n'
                               f'but is {y_train.shape}')

        self.input_shape = x_train.shape[1:]
        x_train = self.reshape_data(x_train)
        x_train, y_train = self.scale(x_train, y_train)

        """
            Feature Expansion: polynomial or spline expansion - use all expanders
        """
        for expander in self.expanders:
            expander.fit(x_train, y_train)
            x_train = expander.transform(x_train)

        self.train_model(x_train, y_train)

    @abstractmethod
    def train_model(self, x_train, y_train):
        raise NotImplementedError('you called the train function on the abstract model class.')

    def predict(self, x):
        """
        DO NOT OVERRIDE THIS METHOD

        this ensures that the input is scaled correctly (i.e. with the distribution characteristics fitted during
        training) BEFORE the model is called to predict the labels and the predictions are transformed to the original
        magnitude (i.e. inverse transformed).

        to implement the prediction function override predict_model below.

        Inputs:
        - x: tensor of shape (samples, lookback_horizon + 1, input_features)

        Returns:
        
        - Tensor of shape (samples, target_features)

        """
        if not x.ndim == 3:
            raise RuntimeError(f'x must be an array of shape (samples, lookback_horizon + 1, input_features)\n'
                               f'but is {x.shape}')

        x = self.reshape_data(x)
        x = self.x_scaler.transform(x)

        ''' Expand features '''
        for expander in self.expanders:
            x = expander.transform(x)

        y = self.predict_model(x)

        if not y.shape[0] == x.shape[0]:
            raise AssertionError(f'samples in prediction do not match samples in input\n'
                                 f'expected: {x.shape[0]}, but is {y.shape[0]}.')

        if not y.ndim == 2:
            raise AssertionError(f'predictions must be two dimensional (samples, target_features)\n'
                                 f'but are {y.shape}.')

        return self.y_scaler.inverse_transform(y)

    @abstractmethod
    def predict_model(self, x):
        raise NotImplementedError('you called the train function on the abstract model class.')

    @abstractmethod
    def save(self, path='models/EXAMPLE'):
        if os.path.isdir(path):
            print(f'{path} already exists, overwriting ..')
        else:
            os.mkdir(path)
        with open(f'{path}/params.pickle', 'wb') as file:
            pickle.dump([self.model_type, 
                self.name, 
                self.input_shape, 
            ], file)

        self.x_scaler.save(f'{path}/x_scaler.pickle')
        self.y_scaler.save(f'{path}/y_scaler.pickle')
        FeatureExpansion.save_expanders(path, self.expanders)


    @abstractmethod
    def load_model(self, path='models/EXAMPLE'):
        with open(f'{path}/params.pickle', 'rb') as file:
            params = pickle.load(file)
            self.name = params[1]
            self.input_shape = params[2]
        
        self.x_scaler = DataScaler.load(f'{path}/x_scaler.pickle')
        self.y_scaler = DataScaler.load(f'{path}/y_scaler.pickle')
        self.expanders = FeatureExpansion.load_expanders(path)
