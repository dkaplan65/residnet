'''
Author: David Kaplan
Advisor: Stephen Penny
'''
from keras.models import Sequential
from keras.layers import Dense, Dropout
import keras

import numpy as np
import matplotlib.pyplot as plt
import logging
import copy

from data_processing import wrappers, transforms, metrics
from comparison_methods import classification, interpolation
import util
import constants
import visualization
logging.basicConfig(format = constants.LOGGING_FORMAT, level = logging.INFO)

def ex1():
    '''Load raw data and parse from scratch. Save.
    '''
    prep_data = wrappers.DataPreprocessing(
        name = 'sample',
        years = ['2005'],
        denorm_local = False,
        num_days = 4)
    prep_data.parse_data()
    prep_data.save()

def ex2():
    d = wrappers.DataPreprocessing.load('output/datapreprocessing/sample_denormLocalFalse_res6')
    f1 = d.subgrids['temp'][0]
    f2 = d.subgrids['temo'][1]
    a1 = f1 - f2
    print(a1)

ex2()
