'''
Author: David Kaplan
Advisor: Stephen Penny
'''
import numpy as np
import logging
import datetime
import matplotlib.pyplot as plt
import os
import sys
import pickle
import json

from .constants import DEFAULT_UTIL_PICKLE_PROTOCOL, MAX_BYTES


def saveobj(obj, filename, protocol=None):
    '''pickle backend
    '''
    if protocol is None:
        protocol = DEFAULT_UTIL_PICKLE_PROTOCOL
    check_savepath_valid(filename)
    bytes_out = pickle.dumps(obj)
    with open(filename, 'wb') as f_out:
        for idx in range(0, len(bytes_out), MAX_BYTES):
            f_out.write(bytes_out[idx: idx + MAX_BYTES])

def loadobj(filename):
    '''pickle backend
    '''
    check_savepath_valid(filename)
    bytes_in = bytearray(0)
    input_size = os.path.getsize(filename)
    with open(filename, 'rb') as f_in:
        for _ in range(0, input_size, MAX_BYTES):
            bytes_in += f_in.read(MAX_BYTES)
    e = pickle.loads(bytes_in)
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

def gen_corner_idxs(res):
    '''Generate corner indices in the order [ll,lr,ul,ur] based on the resolution
    `res`
    '''
    return np.array([0, res*(res-1), res-1, res**2-1])


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
