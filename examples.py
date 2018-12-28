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
    '''Load raw HYCOM data and parse from scratch. Save.
    '''
    prep_data = wrappers.DataPreprocessing(
        name = 'sample',
        years = ['2005'],
        denorm_local = False,
        num_days = 2)
    prep_data.parse_data()
    prep_data.save()

def ex2():
    '''Normalization testing
    '''
    d = wrappers.DataPreprocessing.load('output/datapreprocessing/sample_denormLocalFalse_res6')
    f1 = d.subgrids['temp'][0]
    print('before normalization')
    print(f1)
    print(d.norm_data['temp'][0])

    d.normalize()
    f2 = d.subgrids['temp'][0]
    print('after normalization')
    print(f2)

    d.denormalize()
    f3 = d.subgrids['temp'][0]
    print('after denormalization')
    print(f3)

    print(f1-f3)

def ex3():
    d = wrappers.DataPreprocessing.load('output/datapreprocessing/sample_denormLocalFalse_res6')
    d.normalize()
    idxs = {'training': 0.8, 'testing': 0.2}

    idxs = d.split_data_idxs('split', randomize = True, split_dict = idxs)
    training = d.make_array(idxs = idxs['training'])
    # testing = d.make_array(idxs = idxs['testing'])

    print('training')
    print(training.X[0])
    print(training.y_true[0])

ex2()
