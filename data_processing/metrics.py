'''
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

'''
import numpy as np
from sklearn.metrics import confusion_matrix as cm

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

def confusion_matrix(y_pred, y_truth):
    '''Returns in this order:
        tn, fp, fn, tp
    '''
    return cm(y_truth, y_pred).ravel()

def precision(y_pred, y_truth):
    '''Use confusion matrix
    '''
    tn, fp, fn, tp = confusion_matrix(y_pred, y_truth)
    return tp/(tp + fp)

def accuracy(y_pred, y_truth):
    tn, fp, fn, tp = confusion_matrix(y_pred, y_truth)

    return (tp+tn)/(tp+fp+fn+tn)

def recall(y_pred, y_truth):
    tn, fp, fn, tp = confusion_matrix(y_pred, y_truth)

    return tp/(tp+fn)

def F1(y_pred, y_truth):
    prec = precision(y_pred, y_truth)
    rec = recall(y_pred, y_truth)

    return 2*prec*rec/(prec+rec)

def specificity(y_pred, y_truth):
    tn, fp, fn, tp = confusion_matrix(y_pred, y_truth)

    return tn/(tn+fp)

def sensitivity(y_pred, y_truth):
    '''Alias for recall
    '''
    return recall(y_pred, y_truth)
