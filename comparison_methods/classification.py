'''
Logistic Regression
KMeans
Random Forest Classifier
'''

import numpy as np
import os

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
import tensorflow as tf

import sys
sys.path.append('..')
import constants
from util import IOClass


class BaseClassifier(MLClass):
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
        IOClass.__init__(self)
        self.name = name
        self.kwargs = kwargs
        self.clf = func(**kwargs)
        self.n_categories = None

    def fit(self, dw = None, X = None, y = None):
        '''Train the Random Forrest Classifier

        Set n_clusters
        ----------
        args
        ----------
        dw (DataWrapper, Optional)
            - Use `dw.X` as X
            - Use `dw.y_true` as y
        X (array(n_samples, n_features), Optional)
            - Covariates
            - Instead of using a wrapper.DataWrapper, you could manually
              specify the X and y
        y (array(n_samples, {1,-1}), Optional)
            - Labels
            - If X is specified, y also must be specified.
        '''
        if dw is None:
            if (X is not None) and (y is not None):
                self.n_clusters = y.shape[1]
                self.clf.fit(X,y)
            else:
                raise CLFError('If `dw` is not specified, then both `X` and `y` must be specified.')
        else:
            self.clf.fit(dw.X, dw.y_true)
            self.n_clusters = dw.y_true.shape[1]
        self.trained = True

    def predict(self,src):
        if not self.trained:
            raise CLFError('Must first train the model before you predict')
        if type(src) is DataWrapper:
            src = src.X
        # If `src` is not a DataWrapper, then it is an array and we can pass it right in
        return self.clf.predict(src)

class RandomForestWrapper(BaseClassifier):
    '''Wrapper for sklearn's random forest classifier
    '''
    def __init__(self, **kwargs):
        BaseClassifier.__init__(
            self,
            name = 'RandomForest',
            func = RandomForestClassifier,
            kwargs = kwargs)

class KMeansWrapper(BaseClassifier):
    '''Wrapper for sklearn's KMeans clustering algorithm
    You can you the `fit` method in `BaseClassifier` because it
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
        BaseClassifier.__init__(
            self,
            name = 'KMeans',
            func = KMeans,
            kwargs = kwargs)

class LogisticRegressionWrapper(BaseClassifier):
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
        BaseClassifier.__init__(
            self,
            name = 'Logistic Regression',
            func = LogisticRegression,
            kwargs = kwargs)

class CLFError(Exception):
    pass
