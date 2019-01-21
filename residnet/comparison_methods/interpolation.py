'''
Bilinear
Nearest Neighbor
Bicubic
Inverse Distance Weighting
Multiple Linear Regression (MLR)
Classification preprocess neural network interpolation

`ll` refers to the lower left corner of the subgrid
`lr` refers to the lower right corner of the subgrid
`ul` refers to the upper left corner of the subgrid
`ur` refers to the upper right corner of the subgrid


TODO
    - Fix checking if a parameter for functions for NN-Prep is a function
      or an object
    - Make ClfInterpWrapper.predict faster

'''
import numpy as np
import math
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import logging
import inspect

import sys
sys.path.append('..')
from residnet.constants import DEFAULT_INTERP_IDW_P, DEFAULT_DP_RES_IN
from residnet.data_processing import transforms
from residnet.util import MLClass, get_idxs_of_val

def bilinear(corners, loc):
    '''Compute a bilinear interpolation at location `loc` (ai,aj).
    Assume corners are in order [ll,lr,ul,ur]
    Ignore kwargs
    '''
    (ai,aj) = loc
    return ((corners[0]*(1-ai)*(1-aj))+ \
        (corners[1]*ai*(1-aj))+ \
        (corners[2]*(1-ai)*aj)+ \
        (corners[3]*ai*aj))

def nearest_neighbor(corners, loc):
    '''Computes a nearest neighbor interpolation at the location `loc` (x, y).
    Assumes corners are in the order [ll,lr,ul,ur].
    '''
    (ai,aj) = loc
    if ai < 0.5:
        if aj < 0.5:
            return corners[0] #ll
        else:
            return corners[2] #ul
    else:
        if aj < 0.5:
            return corners[1] #lr
        else:
            return corners[3] #ur

def bicubic(corners, loc):
    '''Computes a bicubic interpolation on the input grid `corners`.
    Assumes structure:

              position                   index number

        a1     a2    a3    a4          12    13    14    15

        b1     b2    b3    b4          8     9     10    11
                 o
        c1     c2    c3    c4          4     5     6     7

        d1     d2    d3    d4          0     1     2     3

        Where `o` is an example interpolation location. Interpolation locations
        will only in the domain within c2, c3, b2, b3, which correspond to the standard
        corners ll, lr, ul, ur, respectfully. We are defining these points to be the
        boundary of the grid we are interpolating in.

        Additionally, assume that the distance between each of the points are equal
        and have a relative distance of 1.

    ##################
    1 dimension
    ##################

    If we define a function f(x) such that the derivative is known at x = 0, x = 1,
    we can interpolate on the interval x in [0,1] using a third degree polynomial.

    Define polynomial and its derivative:

        f(x) = (a * x^3) + (b * x^2) + (c * x) + d
        f'(x) = (3 * a * x^2) + (2 * b * x) + c

    such that the interpolated and first derivative function are continuous.
    Given above definition, we can write:

        f(0) = d
        f(1) = a + b + c + d
        f'(0) = c
        f'(1) = 3a + 2b + c

    We can rewrite this as:

        a = 2*f(0) - 2* f(1) + f'(0) + f'(1)
        b = -3*f(0) + 3*f(1) - 2*f'(0) - f'(1)
        c = f'(0)
        d = f(0)

    For example, we are interpolating on a line where [a1     a2     a3     a4] are the
    values at positions [-1     0     1     2]. With the imposed boundary conditions:

        f(0) = a2
        f(1) = a3
        f'(0) = (a3 - a1)/2
        f'(1) = (a4 - a2)/2

    We can rewrite a1, a2, a3, and a4:

        a = (-0.5 * a1) + (1.5 * a2) - (1.5 * a3) + (0.5 * a4)
        b = (a1)        - (2.5 * a2) + (2 * a3)   - (0.5 * a4)
        c = (-0.5 * a1) + (0.5 * a3)
        d = a2

    Thus our 1 dimensional interpolation formula becomes:

        f(a1, a2, a3, a4, x) = ((-0.5 * a1) + (1.5 * a2) - (1.5 * a3) + (0.5 * a4)) * x^3 +
                               ((a1) - (2.5 * a2) + (2 * a3) - (0.5 * a4)) * x^2 +
                               ((-0.5 * a1) + (0.5 * a3)) * x + a2


    ##################
    2 dimensions
    ##################

    Assuming we are only interpolating in the area [0,1] x [0,1], we first interpolate each of
    the 4 rows and then we interpolate in the column. Assuming our position is (x, y), the bicubic
    interpolation is:

        g(x,y) = f(f(d1, d2, d3, d4, x), f(c1, c2, c3, c4, x), f(b1, b2, b3, b4, x), f(a1, a2, a3, a4, x), y)

    ------------
    args
    ------------
    corners (numpy array)
        - length of 16
        - corresponds to the surrounding grid
    loc (2 - tuple)
        - (x, y) location
    '''
    (x,y) = loc
    return _cubic(_cubic(corners[0],  corners[1],  corners[2],  corners[3], x),
        _cubic(corners[4],  corners[5],  corners[6],  corners[7], x),
        _cubic(corners[8],  corners[9],  corners[10], corners[11], x),
        _cubic(corners[12], corners[13], corners[14], corners[15], x), y)

def _cubic(a1,a2,a3,a4,x):
    '''1 dimensional cubic interpolation.
    Description in bicubic function description
    '''
    return ((-0.5 * a1) + (1.5 * a2) - (1.5 * a3) + (0.5 * a4)) * x**3 + \
        ((a1) - (2.5 * a2) + (2 * a3) - (0.5 * a4)) * x**2 + \
        ((-0.5 * a1) + (0.5 * a3)) * x + a2

def idw(corners, loc, p = None):
    '''Compute the Inverse Distance Weighting interpolation.

    The auxiliary function calculates the inverse distance betwen the point
    being interpolated and each of the 4 corners
    '''
    if p == None:
        p = DEFAULT_INTERP_IDW_P
    (ai,aj) = loc
    ret = 0

    # Set if it is a corner
    if ai == 0:
        if aj == 0:
            return corners[0] #ll
        elif aj == 1:
            return corners[2] #ul
    elif ai == 1:
        if aj == 0:
            return corners[1] #lr
        elif aj == 1:
            return corners[3] #ur
    distances = _inverse_distance((ai,aj), p)
    return np.sum(distances * corners) / np.sum(distances)

def _inverse_distance(xi, p):
    '''Calculates the inverse distance to each of the corners (ll,lr,ul,ur)

    `xi` is where you want to interpolate to
    `p` is the power parameter
    '''
    # Corner locations
    xs = [[0.,0.], [1.,0.], [0.,1.], [1.,1.]]
    ret = []
    # Get location
    (ai,aj) = xi
    for i in range(4):
        [x,y] = xs[i]
        ret.append(1 / ((math.sqrt((ai-x)**2 + (aj-y)**2)) ** p))
    return ret

class MLR(MLClass):
    '''Multiple Linear Regression

    Basically wraps sklearn's linear regression package so that a single
    linear regression model is fitted for every pixel for the output.
    '''

    def __init__(self, **kwargs):
        MLClass.__init__(self)
        self.kwargs = kwargs
        self.kwargs['copy_X'] = True
        self.clfs = []
        self.output_len = None

    def fit(self,X,y):
        ''' Fits the model.
        X: input data
        y: labels
            - can be more than a 1 dimensional element
        '''
        y = np.array(y)
        if len(y.shape) == 1:
            self.output_len = 1
        else:
            self.output_len = y.shape[1]
        for i in range(self.output_len):
            logging.info('training {}/{}'.format(i,self.output_len))
            if self.output_len == 1:
                y_ = y
            else:
                y_ = y[:,i]
            self.clfs.append(LinearRegression(**self.kwargs).fit(X,y_))
        self.trained = True

    def predict(self, X):
        ''' Runs the model with the input data X
        '''
        if not self.trained:
            raise InterpError('MLR: Cannot predict without first training')
        X = np.array(X)
        if len(X.shape) == 1:
             # Only one thing to predict
            y = np.zeros(len(self.clfs))
            for i in range(len(y)):
                y[i] = self.clfs[i].predict(X)
        # Many things to predict
        y = np.zeros(
            shape = (X.shape[0], len(self.clfs)))
        for i in range(self.output_len):
            logging.info('{}/{}'.format(i,self.output_len))
            y[:,i] = self.clfs[i].predict(X)
        return y

    def score(X, y):
        '''
        Return a list of scores from each of the classifiers
        '''
        scores = []
        for i in range(len(y)):
            scores.append(self.clfs[i].score(X,y[:,i]))
        return scores

class ClfInterpWrapper(MLClass):
    ''' Classification Preprocess Interpolation

    This class encpasulates a classification method that preprocesses the data
    before it is fed to a single or multiple different interpolation schemes.

    The goal of this structure is to divide up data between different interpolation
    methods and then combine it effectively.
    '''

    class _NonMLInterpolationWrapper:
        '''
        Dummy class that wraps a non-machine learning method (ex: bilinear)
        so that the code can be smooth in InterpolationWrapper
        '''
        def __init__(self,func,res):
            '''
            func (callable function from comparison_methods.interpolation)
                - THIS DOES NOT INCLUDE MLR
            res (int)
                - resolution
            '''
            self.func = func
            self.ml = False
            self.res = res

        def fit(self, **kwargs):
            '''Do nothing because there is no training involved
            '''
            return

        def predict(self, src):
            if len(src.shape) == 1:
                return transforms.interpolate_grid(
                    input_grid = src, res = self.res, interp_func = self.func)

            ret = np.zeros(shape=(src.shape[0], self.res ** 2))
            for i in range(src.shape[0]):
                ret[i,:] = transforms.interpolate_grid(
                input_grid = src[i], res = self.res, interp_func = self.func)

    def __init__(self,clf,regs,res=None):
        '''
        -----------
        args
        -----------
        clf (comparison_methods.classification.BaseClassifier)
            - This is an INSTANCE of a class that is inhereited from the base
              class BaseClassifier.
        regs (dict: key -> comparison_method)
            - The key has to be the outputs of the classifier associated with
              that interpolation method

              Example:
                    Let my classifier be a binary classifier.
                    If the output of the classifier if 1, I want to use bilinear
                    interpolation.
                    If the output of the classifier is 0, I want to use MLR.

                    Then my regs argument will be:
                    reg = {0: bilinear, 1: MLR()}
                        - Note that MLR IS INSTANTIATED.
        res (int, Optional)
            - Resolution of the output interpolation
        '''
        if res is None:
            res = DEFAULT_DP_RES_IN

        self.clf = clf
        self.regs = {}
        self.res = res
        # self.clf_trained = False
        self.regs_trained = False
        self.out_len = None

        for key,val in regs.items():
            # If the value is a class, we know it is a ML class and we do nothing
            # If it is a function, then the inspect class will return
            # false, so we will have to wrap it

            # if inspect.isclass(val):
            #     logging.info('{} is a ml-interpolation method'.format(key))
            #     self.regs[key] = val
            # else:
            #     logging.info('{} is a non-ml interpolation method'.format(key))
            #     self.regs[key] = ClfInterpWrapper._NonMLInterpolationWrapper(func=val, res = res)
            self.regs[key] = val


    def fit_clf(self, X, y):
        '''Trains the classifier

        X (np.ndarray)
            - Covariates.
        y (np.ndarray)
            - Observations
        '''
        self.clf.fit(X = X, y = y)

    def fit_interp(self, X, y, epochs = None, batch_size = None):
        '''Trains the regressors.

        The first thing we need to do is divide up the appropriate data for
        each regressor. And then we train each one of them.

        X (np.ndarray)
            - Covariates.
        y (np.ndarray)
        '''
        # if not self.clf.trained:
        #     raise InterpError('Classifier must be trained before regression can be trained')
        if epochs is None:
            epochs = 10
        if batch_size is None:
            batch_size = 50

        self.out_len = y.shape[1]
        pred = self.clf.predict(X)
        ret = np.argmax(pred, axis = 1)
        # get the indices for each regression
        for key,val in self.regs.items():
            ind = get_idxs_of_val(arr = ret, val = key)
            self.regs[key].fit(
                X[ind],
                y[ind],
                epochs = epochs,
        		batch_size = batch_size)
        self.regs_trained = True

    def predict(self, X):
        '''
        For each example in X, use the classifier to see which regressor it should go to
        and then use that regressor to make the prediction
        '''
        if not self.regs_trained:
            raise InterpError('Regressors must be trained before prediction')

        if len(X.shape) == 1:
            X = X[np.newaxis, ...]
        out = np.zeros(shape=(X.shape[0], self.out_len))


        for i in range(X.shape[0]):
            if i % 50000 == 0:
                logging.info('{}/{}'.format(i, X.shape[0]))
            clf_pred = self.clf.predict(X[i:i+1,:])
            clf_pred = np.argmax(clf_pred)
            out[i,:] = self.regs[clf_pred].predict(X[i:i+1,:])
        return out

class InterpError(Exception):
    pass
