'''
Author: David Kaplan
Advisor: Stephen Penny
'''
import numpy as np
import logging
import datetime
import matplotlib.pyplot as plt
import os
import pickle

from .constants import DEFAULT_UTIL_PICKLE_PROTOCOL


def saveobj(obj,filename, protocol=None):
    if protocol is None:
        protocol = DEFAULT_UTIL_PICKLE_PROTOCOL
    check_savepath_valid(filename)
    with open(filename, 'wb') as output:  # Overwrites any existing file.
        pickle.dump(obj, output, protocol)

def loadobj(filename):
    check_savepath_valid(filename)
    with open(filename, 'rb') as input_file:
        e = pickle.load(input_file)
    return e

def check_savepath_valid(filepath, create_filepath = True):
    '''Return a boolean to check if the filepath is valid.
    If create_filepath is True, create the filepath indicated.
    '''
    directory = os.path.dirname(filepath)
    if not os.path.exists(directory):
        if create_filepath:
            os.makedirs(directory)
            return True
        else:
            return False
    return True

def str_date(year,day):
    '''Converts an int of the year and day and converts it to a string of
    year, month, and day

    Assumes year, day are floats or ints.

    Example:
        input
            year: 2005.
            day: 66.
        output
            year: '2005'
            month: 'March'
            day: '07'
    '''
    year = int(year)
    day = int(day)
    a = datetime.datetime(year, 1, 1) + datetime.timedelta(day - 1)
    month = a.strftime('%B')
    day = a.strftime('%d')
    year = a.strftime('%Y')
    return year, month, day

def get_idxs_of_val(arr, val):
    '''Returns the idxs in arr where `val` appears.
    This works for both arrays and regular values
    ex1:
        input
            arr: np.array([[0,1],[0,1],[1,0],[1,0]])
            val: [0,1]
        output:
            np.array([0,1])

    ex2:
        input:
            arr: np.array([0,1,1,0,1,0])
            val: 0
        output:
            np.array([0,3,5])

    '''
    def _eq_arr(a1):
        return np.array_equal(a1,val)
    if type(val) == list or type(val) == np.ndarray:
        return np.where(np.apply_along_axis(_eq_arr, 1, arr))[0]
    else:
        return np.where(arr == val)[0]

class IOClass:
    '''Base class
    Defines saving, loading with pickle.
    If a class is very large in size (data_processing.DataPreprocessing),
    do not use this class.
    '''
    def __init__(self):
        pass

    def save(self, filename):
        '''If filename is not None, override the internal save location.
        '''

        saveobj(self,filename)

    @classmethod
    def load(cls, filename):
        '''Defines a simple load function with pickle.
        '''
        return loadobj(filename)

class MLClass(IOClass):

    def __init__(self):
        IOClass.__init__(self)
        self.trained = False
        self.ml = True

class IOError(Exception):
    pass
