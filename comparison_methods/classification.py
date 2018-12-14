'''
Logistic Regression
KMeans
Random Forest Classifier
'''

import numpy as np
import os
import copy
import math
import logging

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans

import sys
sys.path.append('..')
import constants
from util import MLClass


class SKLearnBase(MLClass):
    '''Base class for the classification
    '''
    def __init__(self, name, func, kwargs):
        '''
        -----------
        args
        -----------
        name (str)
            - Type of classification
        func (callable function)
            - Constructor function to initialize the classifier
        kwargs (dict)
            - arguments that are used to specify the instance
        '''
        MLClass.__init__(self)
        self.name = name
        self.kwargs = kwargs
        self.clf = func(**kwargs)
        self.n_clusters = None

    def fit(self, X, y = None):
        '''Train the Random Forrest Classifier

        Set n_clusters
        ----------
        args
        ----------
        X (array(n_samples, n_features))
            - Covariates
            - Instead of using a wrapper.DataWrapper, you could manually
              specify the X and y
        y (array(n_samples, {1,-1}), Optional)
            - Labels
            - Do not have to pass in y for KMeans
        '''
        if y is None or X is None:
            raise CLFError('No X or y')

        if len(y.shape) >= 2:
            self.n_clusters = y.shape[1]
        else:
            self.n_clusters = len(np.unique(y))
        if y is None:
            # For KMeans
            self.clf.fit(X)
        else:
            self.clf.fit(X,y)
        self.trained = True

    def predict(self,X):
        if not self.trained:
            raise CLFError('Must first train the model before you predict')
        if type(src) is DataWrapper:
            X = X.X
        # If `src` is not a DataWrapper, then it is an array and we can pass it right in
        return self.clf.predict(X)

class RandomForestWrapper(SKLearnBase):
    '''Wrapper for sklearn's random forest classifier
    '''
    def __init__(self, **kwargs):
        SKLearnBase.__init__(
            self,
            name = 'RandomForest',
            func = RandomForestClassifier,
            kwargs = kwargs)

class KMeansWrapper(SKLearnBase):
    '''Wrapper for sklearn's KMeans clustering algorithm
    You can you the `fit` method in `SKLearnBase` because it
    automatically ignores the `y`.
    '''
    def __init__(self, **kwargs):
        '''
        n_clusters (int)
            - Number of clusters
        n_init (int)
            - Number of times to run KMeans with different centroid seeds
        '''
        if kwargs['n_clusters'] == None:
            kwargs['n_clusters'] = constants.DEFAULT_CLF_N_CLUSTERS
        if kwargs['n_init'] == None:
            kwargs['n_init'] = constants.DEFAULT_CLF_N_INIT
        SKLearnBase.__init__(
            self,
            name = 'KMeans',
            func = KMeans,
            kwargs = kwargs)

class LogisticRegressionWrapper(SKLearnBase):
    ''' Wrapper for sklearn's logistic regression classifier
    '''
    def __init__(self, **kwargs):
        '''penalty (str)
            - either 'l1' or 'l2'
        solver (str)
            - Algorithm to use in the optimization
            - `newton-cg`, `lbfgs`, `liblinear`, `sag`, `saga`
        C (int)
            - Inverse regularization strength

        Class weight is always constants.DEFAULT_CLF_CLASS_WEIGHT
        '''
        if kwargs['penalty'] == None:
            kwargs['penalty'] = constants.DEFAULT_CLF_PENALTY
        if kwargs['solver'] == None:
            kwargs['solver'] = constants.DEFAULT_CLF_SOLVER
        if kwargs['C'] == None:
            kwargs['C'] = constants.DEFAULT_CLF_C
        kwargs['class_weight'] = constants.DEFAULT_CLF_CLASS_WEIGHT
        SKLearnBase.__init__(
            self,
            name = 'Logistic Regression',
            func = LogisticRegression,
            kwargs = kwargs)

class NNEnsemble:
    '''Ensemble of Neural networks.

    A given neural network is passed in and it makes `size` copies of
    the network. When it trains, it will train each network on a random,
    disjoint, subset of the data. For the prediction, it run the example
    through each of the networks and then it will do some sort of consensus
    between them and that is the prediction.
    '''

    def __init__(self, model, size, consensus = None):
        '''
        model (Keras model)
            - Neural network model
        size (int)
            - How many networks to make in the ensemble
        consensus (str, Optional)
            - Type of consensus method to use
            - Consensus types:
                  * `mode`
                      - Returns the most frequent prediction
        '''
        def _mode(arr):
            rs = False
            if len(arr.shape) == 3:
                rs = True
                arr = np.argmax(arr, axis = 2)
            arr = np.mean(arr, axis = 0)
            arr = np.array(np.rint(arr), dtype = int)
            if rs:
                arr_ = np.zeros(shape=(len(arr),2))
                arr_[np.arange(len(arr),arr)] = 1
                arr_ = arr
            return arr

        if consensus is None:
            consensus = 'mode'

        self.models = []
        self.size = size
        for i in range(self.size):
            self.models.append(copy.deepcopy(model))

        if consensus == 'mode':
            self.consensus = _mode
        self.trained = False

    def fit(self, X, y, epochs, batch_size = None):
        if batch_size is None:
            batch_size = 50

        idxs = np.arange(X.shape[0])
        np.random.shuffle(idxs)
        set_size = math.floor(X.shape[0]/self.size)

        base_idx = 0
        for i in range(self.size):
            subX = X[idxs[base_idx:base_idx+set_size]]
            suby = y[idxs[base_idx:base_idx+set_size]]
            logging.info('model {}/{}'.format(i+1,self.size))
            self.models[i].fit(subX,suby,epochs=epochs,batch_size=batch_size)
            base_idx += set_size

        self.trained = True

    def predict(self,X):
        '''Predict using the consensus method
        '''
        if not self.trained:
            raise CLFError('NNEnsemble must be trained before it can predict')
        out = []
        for i in range(self.size):
            out.append(self.models[i].predict(X))
        out = self.consensus(np.array(out))
        return ret

    # def save(self, loc):

class CLFError(Exception):
    pass
