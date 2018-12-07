
from data_processing.wrappers import DataPreprocessing
from data_processing import transforms
from data_processing import metrics
import datasets
from comparison_methods import interpolation as interp
import util

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import logging


def example_read_in_raw_data():
    # Parse raw data into netcdfSet object
    raw_data = datasets.HYCOM(
        filepath2d = 'input_data/hycom/2D_vars/',
        filepath3d = 'input_data/hycom/3D_vars/',
        years = ['2005','2006'])

def example_make_data_from_scratch_default():

    # Get the defualt settings.
    # Defaults described in _Settings constructor
    test = DataPreprocessing(name = 'test_data', num_days = 5)
    test.parse_data()
    test.normalize()
    test.save()

    # Make the data split percentages
    # This is for a random split at these proportions

    # split_dict = {'training': 0.7, 'validation': 0.1, 'testing': 0.2}
    # idxs = test.split_data_idxs('split', split_dict = split_dict, randomize = False)
    #
    # for key,val in idxs.items():
    #     print('{}: {}'.format(key,val.shape))

    # Makes the training array.
    # Default input keys are all the keys
    # Default output key is temperature
    # Output is of type InputTruthWrapper

    # training_data = test.make_array(idxs = idxs['training'])
    # print(training_data.shape)
    # print(training_data.input_array[0,:])
    # print(training_data.y_true[0,:])
    # validation_data = test.make_array(idxs = idxs['validation'])
    # testing_data = test.make_array(idxs = idxs['testing'])

def run_classification_example():

    # Make new data
    # test = DataPreprocessing(num_days = 2)
    # test.parse_data()
    test = DataPreprocessing().load('data/dataobj/data_denormLocalFalse_res6')
    test.normalize()
    test.save()

    # Get datasets
    split_dict = {'training': 0.9, 'testing': 0.1}
    idxs = test.split_data_idxs('split', split_dict = split_dict, randomize = True)
    training_data = test.make_array(idxs = idxs['training'])

    print('input',training_data.input_array[0:2,:])
    print('truth', training_data.y_true[0:2,:])

    training_data = itwt.BilinearMSERegression(training_data, True, True)

    print('input',training_data.input_array[0:2,:])
    print('truth', training_data.y_true[0:2,:])
    print(np.max(np.squeeze(training_data.y_true)))
    print(np.min(np.squeeze(training_data.y_true)))
    print(np.mean(np.squeeze(training_data.y_true)))
    print(np.std(np.squeeze(training_data.y_true)))

    print(training_data.y_true.shape)

    fig = vis.pca(training_data.input_array, training_data.y_true, False)


    plt.figure()
    y = np.squeeze(training_data.y_true)

    plt.plot(np.arange(len(y)),sorted(y))


    plt.show()

def make_data_example():

    # Make new data
    # test = DataPreprocessing(num_days = 2)
    # test.parse_data()
    test = DataPreprocessing().load('data/dataobj/data_denormLocalFalse_res6')
    test.normalize()
    test.save()

    # Get datasets
    split_dict = {'training': 0.9, 'testing': 0.1}
    idxs = test.split_data_idxs('split', split_dict = split_dict, randomize = True)
    training_data = test.make_array(idxs = idxs['testing'])

    print(training_data.loc_to_idx)

def nn_test():
    '''
    Tests the nn module
    '''
    # Load data
    test = DataPreprocessing().load('data/dataobj/full_denormLocalFalse_res6')
    test.normalize()

    split_dict = {'training': 0.7, 'validation': 0.1, 'testing': 0.2}
    idxs = test.split_data_idxs('split', split_dict = split_dict, randomize = True)
    training_data = test.make_array(idxs = idxs['training'])
    validation_data = test.make_array(idxs = idxs['validation'])

    print(len(training_data))
    print(len(validation_data))
    test = default_network()

    # Train
    test.train(training_data = training_data, num_epochs = 2,
               validation_data = validation_data)

    test.visualize_training()

def bicubicTest():
    # test = DataPreprocessing(name = 'test_data', num_days = 5)
    # test.parse_data()
    # test.normalize()
    # test.save()

    test = DataPreprocessing.load('output/datapreprocessing/testing_data_denormLocal{}_res6.pkl')
    test.denormalize()

    split_dict = {'training': 0.99, 'validation': 0.01}
    idxs = test.split_data_idxs('split', split_dict = split_dict, randomize = False)
    training_data = test.make_array(input_keys = ['temp'], idxs = idxs['training'])

    training_data = transforms.makeBicubicArrays(training_data)
    
    a = transforms.InterpolationErrorRegression(src = training_data,
                                     func = interp.bicubic,
                                     cost = transforms.MSE,
                                     set_to_y_true = False)
    print(np.mean(a))
    print(np.std(a))

    a = transforms.InterpolationErrorRegression(src = training_data,
                                     func = interp.idw,
                                     cost = metrics.MSE,
                                     set_to_y_true = False)

    print(np.mean(a))
    print(np.std(a))

    a = transforms.InterpolationErrorRegression(src = training_data,
                                     func = interp.nearest_neighbor,
                                     cost = transforms.MSE,
                                     set_to_y_true = False)

    print(np.mean(a))
    print(np.std(a))

def datatest():

    test = DataPreprocessing(name = 'test_data', num_days = 5)
    test.parse_data()
    test.normalize()
    test.save()

    # test = DataPreprocessing().load('data/dataobj/test_data_denormLocalFalse_res6/')
    test.denormalize()

    split_dict = {'training': 0.99, 'validation': 0.01}
    idxs = test.split_data_idxs('split', split_dict = split_dict, randomize = False)
    training_data = test.make_array(input_keys = ['temp'], idxs = idxs['training'])

    a = util.Data(training_data)

    b = a.matrixify(arr = a.base.y_true, year = 2005, day = 0)

    plt.imshow(b)
    plt.show()



datatest()
