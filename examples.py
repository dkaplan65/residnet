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
from residnet import constants, visualization, util
logging.basicConfig(format = constants.LOGGING_FORMAT, level = logging.INFO)

def main():
    ex2()

def ex1():
    '''Load raw HYCOM data and parse from scratch. Save.
    Here, we are only going to parase 2 days worth of data from 2005.
    '''
    prep_data = wrappers.DataPreprocessing(
        name = 'sample',
        years = ['2005'],
        denorm_local = False,
        num_days = 2)
    prep_data.parse_data()
    prep_data.save()

def ex2():
    '''
    Load preprocessed data and create a training datawrapper.
    '''
    # Load the preprocessing data from `ex1`
    d = wrappers.DataPreprocessing.load('output/datapreprocessing/sample_denormLocalFalse_res6')

    # Create an index dictionary. This should be of the form:
    # key -> float, where sum(values) <= 1.0. The values represet what
    # proportion of data to put into the 'key'th set
    idxs = {'training': 0.9, 'validation': 0.1}

    # 'split' means that we are splitting the data based on a probability,
    # split_dict indicates the proportions to split them to, and randomize
    # being true means that we are going to shuffle the indices beforehand.
    idxs = d.split_data_idxs('split', randomize = True, split_dict = idxs)

    # Make the training data based on the indices calcualted in the previous step
    # Only include the corners of 'temp' in the input and the truth for temp
    # as the output
    training = d.make_array(idxs = idxs['training'], input_keys = ['temp'],
        output_key = 'temp')

    # Normalize both the input and output data
    training.normalize()

    # Save the DataWrapper.
    training.save(filename = 'output/datawrapper/sample_training_data.pkl')

def ex3():
    '''Get the bilinear interpolation error on the training data.
    '''
    # Load the training data we produced in `ex2`
    training = wrappers.DataWrapper.load('output/datawrapper/sample_training_data.pkl')

    # denormalize the data so we can do non-ML interpolation with it
    training.denormalize()

    # Since the object is self consistent, if we denormalize again, nothing will
    # happend (because it is already denormalized). It is designed to fail gracefully.
    training.denormalize()

    # Produce the interpolation error using bilinear for each subgrid
    # Do a deep copy of the data before sending it in so that there is no
    # corruption.
    bilin_error = transforms.InterpolationErrorRegression(
		src = copy.deepcopy(training),
		func = interpolation.bilinear,
		cost = metrics.Error,
		output_size = src.res ** 2)

    # The output was set as `y_true` of `bilin_error`
    # Get the mean RMSE, RMSE std, and bias

    RMSE = np.apply_along_axis(
        transforms.RMSE,
        axis = 1,
        arr = bilin_error.y_true)
    print('Bilinear interpolation results:')
    print('\tbias: {}'.format(np.sum(bilin_error.y_true)))
    print('\tmean RMSE: {}'.format(np.mean(RMSE)))
    print('\tstd RMSE: {}'.format(np.std(RMSE)))

def ex4():
    '''Produce a 2 element, one-hot array of subgrids that are greater than
    or less than the bilinear interpolation error of the subgrid.
    '''
    # Load the training data we produced in `ex2` and denormalize
    training = wrappers.DataWrapper.load('output/datawrapper/sample_training_data.pkl')
    training.denormalize()

    # This function encapsulates the entire process
    # threshold tells the cutoff between the classes. If the error is greater than the
    # threshold (0.1), then it is one category. If it is less than the threshold then it
    # is a different category.
    out = InterpolationErrorClassification(
        src = training,
        func = interpolation.bilinear,
        cost = RMSE,
        threshold = 0.1):

    # `out` is now a `wrapper.DataWrapper` object where the input is the same as
    # `training` and the output is a one-hot array based on the bilinear interpolation
    # error. We can feed this object to a classification method for trainig, etc.

def ex5():
    '''This examples produces a picture of the first day. On the left it is
    going to be the base temperature data and on the right it is going to be
    the RMSE error of a bilinear interpolation.
    '''
    training = wrappers.DataWrapper.load('output/datawrapper/sample_training_data.pkl')
    training.denormalize()

    fig = plt.figure()
    ax1 = fig.add_subplot(1,2,1)

    # Produce the 2D map of the first day
    basemap = transforms.mapify(
        loc_to_idx = training.loc_to_idx,
        arr = training.y_true,
        year = 2005,
        day = 1,
        res = training.res,
        classification = False)

    ax1 = visualization.map(
        ax = ax1,
        arr = basemap,
        title = 'Original data')

    # Make the bilinear error data
    bilin_error = transforms.InterpolationErrorRegression(
		src = copy.deepcopy(training),
		func = interpolation.bilinear,
		cost = metrics.Error,
		output_size = src.res ** 2)

    # Produce the 2D map of the first day
    basemap = transforms.mapify(
        loc_to_idx = training.loc_to_idx,
        arr = training.y_true,
        year = 2005,
        day = 1,
        res = training.res,
        classification = False)

    ax2 = visualization.map(
        ax = fig.add_subplot(1,2,2),
        arr = basemap,
        title = 'Bilinear error')

    plt.show()






if __name__ == '__main__': main()
