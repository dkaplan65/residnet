''' Interpolation and classification metrics
classification
    * Confusion Matrix
        - accuracy
        - precision
        - recall
        - specificity
        - etc.

regression
    * Error
        - MSE
            * Mean square error
        - RMSE
            * Root mean square error
        - logMSE
            * log mean square error
        - MAE
            * Mean absolute error
        - bias
    * Deviation
        - Variance
        - Standard Deviation

These functions are meant to be computed over a single dimension arrays.
If you have a 2 dimension array that you want it to do the computation over,
use the command:
    np.apply_along_axis(func, axis, arr)

Future versions:
    Different kinds of means besides arithmetic:
        harmonic
        geometric
'''
import numpy as np
from sklearn.metrics import confusion_matrix as cm
from .transforms import collapse_one_hot

#############
# Interpolation functions
# If arr2 is None, then assume that `arr` is already the error
#############
def logMSE(arr, arr2=None):
    return np.log(MSE(arr,arr2))

def RMSE(arr, arr2=None):
    return np.sqrt(MSE(arr, arr2))

def MSE(arr, arr2=None):
    return np.mean(SE(arr, arr2))

def SE(arr, arr2=None):
    '''Square Error
    '''
    if arr2 is None:
        return np.square(arr)
    else:
        return np.square(arr - arr2)

def AE(arr, arr2 = None):
    '''Absolute Error
    '''
    if arr2 is None:
        return np.absolute(arr)
    else:
        return np.absolute(arr - arr2)

def Error(arr, arr2 = None):
    '''Plain error
    '''
    if arr2 is None:
        return arr
    else:
        return arr - arr2

def MAE(arr, arr2=None):
    '''Mean Absolute Error
    '''
    if arr2 is None:
        return np.mean(np.absolute(arr))
    else:
        return np.mean(np.absolute(arr - arr2))

def bias(arr, arr2=None):
    if arr2 is None:
        return np.sum(arr)
    else:
        return np.sum(arr - arr2)

def variance(arr):
    return np.var(arr)

def std(arr):
    return np.std(arr)


#############
# Classification functions
#############

def confusion_matrix(y_true, y_pred):
    '''Returns in this order:
        tn, fp, fn, tp
    '''
    if len(y_true.shape) == 2:
        y_true = collapse_one_hot(arr = y_true)
    if len(y_pred.shape) == 2:
        y_pred = collapse_one_hot(arr = y_pred)
    return cm(y_true, y_pred).ravel()

def precision(y_true = None, y_pred = None, cm = None):
    '''Precision.
    If `cm` (confusion matrix) is specified, use that instead.
    '''
    if confusion_matrix is not None:
        tn, fp, fn, tp = cm
    else:
        raise Exception('If `cm` is not specified, both y_true'
            ' and y_pred must be specified')
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred)
    return tp/(tp + fp)

def accuracy(y_true = None, y_pred = None, cm = None):
    '''Accuracy
    If `cm` (confusion matrix) is specified, use that instead.
    '''
    if confusion_matrix is not None:
        tn, fp, fn, tp = cm
    else:
        if y_true is None or y_pred is None:
            raise Exception('If `cm` is not specified, both y_true'
                ' and y_pred must be specified')
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred)
    return (tp+tn)/(tp+fp+fn+tn)

def recall(y_true = None, y_pred = None, cm = None):
    '''Reccall.
    If `cm` (confusion matrix) is specified, use that instead.
    '''
    if confusion_matrix is not None:
        tn, fp, fn, tp = cm
    else:
        raise Exception('If `cm` is not specified, both y_true'
            ' and y_pred must be specified')
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred)
    return tp/(tp+fn)

def F1(y_true = None, y_pred = None, cm = None):
    ''' F1
    If `cm` (confusion matrix) is specified, use that instead.
    '''
    if confusion_matrix is not None:
        tn, fp, fn, tp = cm
    else:
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred)
    return 2*prec*rec/(prec+rec)

def specificity(y_true = None, y_pred = None, cm = None):
    ''' Specificity.
    If `cm` (confusion matrix) is specified, use that instead.
    '''
    if confusion_matrix is not None:
        tn, fp, fn, tp = cm
    else:
        raise Exception('If `cm` is not specified, both y_true'
            ' and y_pred must be specified')
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred)
    return tn/(tn+fp)

def sensitivity(y_true = None, y_pred = None, cm = None):
    '''Alias for recall
    If `cm` (confusion matrix) is specified, use that instead.
    '''
    if confusion_matrix is not None:
        tn, fp, fn, tp = cm
    else:
        raise Exception('If `cm` is not specified, both y_true'
            ' and y_pred must be specified')
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred)
    return recall(y_pred, y_true)
