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

from residnet.data_processing import wrappers, transforms, metrics
from residnet.comparison_methods import classification, interpolation
from residnet import constants, visualization
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
    d = wrappers.DataPreprocessing.load('output/datapreprocessing/sample_denormLocalFalse_res6')
    idxs = {'training': 0.9, 'validation': 0.1}

    idxs = d.split_data_idxs('split', randomize = True, split_dict = idxs)
    training = d.make_array(idxs = idxs['training'])
    training.normalize()
    # testing = d.make_array(idxs = idxs['testing'])

    print('training')
    print(training.X[0])
    print(training.y_true[0])

ex2()
