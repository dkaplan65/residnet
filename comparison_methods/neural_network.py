'''Defines a wrapper class for tensorflow.

Classification or Interpolation networks are defined in
`classification.py` and `interpolation.py`, respectfully.

Class implemented so that you caould be run in an `Eager`-ish
fashion.

TODO:
    SAVE AND LOAD METHODS

Next iterations:
    - Add convolution layer functionality
'''

import tensorflow as tf
import numpy as np
import time
import os
import logging
import matplotlib.pyplot as plt

import sys
sys.path.append('..')
from util MLClass
import constants
import data_processing.metrics

class TFWrapper(MLClass):
    '''Tensorflow wrapper class

    Provides functionality for visualization of the training and validation
    performance during training and validation

    Order to how to build the NN
        1) initialize
        2) add layers
        4) train

    Then you can run and visualize training performance.
    Default parameters are specified in constants.
    '''

    def __init__(self, base_save_loc = None, classification = None,
        optimizer = None, cost_func = None, name = None):
        ''' Constructor. Defaults defined in constants
        -----------
        args
        -----------
        base_save_loc (str)
            - Base folder to save the neural network
            - If the path does not exist then create it
        classification (bool, Optional)
            - If True, sets the NN as a classification nn
            - If False, sets the NN as a regression nn
        optimizer (tf callable function, Optional)
            - The optimization function
        cost_func (str, Optional)
            - Type of cost function for regression nn
            - If `self.classification` == True, this does not matter
            - valid inputs:
                'MSE', 'RMSE', 'MAE'
        '''
        if base_save_loc is None:
            base_save_loc = constants.DEFAULT_NN_BASE_SAVE_LOC
        if classification is None:
            classification = constants.DEFAULT_NN_TYPE_CLASSIFICATION
        if optimizer is None:
            optimizer = constants.DEFAULT_NN_OPTIMIZER
        if cost_func is None:
            cost_func = constants.DEFAULT_NN_COST_FUNC
        if name is None:
            name = constants.DEFAULT_NN_NAME

        MLClass.__init__(self)
        self.base_save_loc = base_save_loc
        self.classification = classification
        self.optimizer = optimizer
        self.cost_func = cost_func
        self.built = False # set to True once `finished_building` is called
        self.layers = {}
        self.num_layers = 0
        self.layer_sizes = []
        self.name = name

        # Make the save directory if necessary
        if not os.path.exists(self.base_save_loc):
            os.makedirs(self.base_save_loc)

    def __str__(self):
        s = ''
        for key,val in self.__dict__.items():
            s += '{}: {}\n'.format(key,str(val))
        return s

    def add_dense_layer(self, size, activation = None, use_bias = None,
        initializer = None):
        '''Adds a dense layer to the end of the neural network
        y = sigma(w*x + b), where
            `y` is the scalar output
            `x` is the vector input
            `w` is the weight matrix
            `b` is the bias of the activation
            `sigma` is the activation function
        ------------
        args
        ------------
        size (int)
            - How many neurons in the layer
        activation (tf callable function, Optional)
            - Activation function for the neurons
        use_bias (bool, Optional)
            - If True,
                y = sigma(w*x + b)
            - If False,
                y = sigma(w*x)
        initializer (tf callable function, Optional)
            - Function to set the values of the
              `w` and `b` matrix at initialization
        '''

        if activation == None:
            activation = constants.DEFAULT_NN_DENSE_ACTIVATION
        if use_bias == None:
            use_bias = constants.DEFAULT_NN_DENSE_USE_BIAS
        if initializer == None:
            initializer = constants.DEFAULT_NN_INITIALIZER

        # If this is the first layer, set as input_size
        if self.num_layers == 0:
            self.input_size = size
            self.x = tf.placeholder('float', [None, size], name = 'x')
            self.prev_layer = self.x

        # Make the layer
        self.layers[self.num_layers] = tf.layers.dense(
            inputs = self.prev_layer,
            units = size,
            activation = activation,
            use_bias = use_bias,
            kernel_initializer = initializer(),
            bias_initializer = initializer(),
            name = 'layer_{}'.format(int(self.num_layers)))

        self.prev_layer = self.layers[self.num_layers]
        self.num_layers += 1
        self.layer_sizes.append(size)
        # Iteratively set the output size
        self.output_size = size

    def finished_building(self):
        '''Renames last layer as prediction, sets y placeholder,
        cost, and other attributes depending if it is a classification
        nn or not
        '''
        if self.built:
            return
        self.saver = tf.train.Saver()

        # Set output layer
        self.prediction = self.layers.pop(self.num_layers - 1)
        self.y = tf.placeholder('float', [None,self.output_size], 'y')
        # `self.penalizer` acts as an alias so during training the code is
        # more readible, shorter, and prettier
        # Set cost functions
        # Make both a cost for tensorflow and regular
        self._set_cost()
        if self.classification:
            self.cost = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits_v2(
                    labels = self.y, logits = self.prediction), name = 'cost')
            self.correct_prediction = tf.equal(
                tf.argmax(self.y,1),
                tf.argmax(self.prediction,1),
                name = 'correct_prediction')
            self.accuracy = tf.reduce_mean(
                tf.cast(self.correct_prediction, tf.float32),
                name = 'accuracy')
            self.penalizer = self.accuracy

        else:
            if self.cost_func == 'MSE':
                self.cost = tf.reduce_mean(
                    tf.square(self.y - self.prediction),
                    name = 'cost')
            elif self.cost_func == 'RMSE':
                self.cost = tf.sqrt(
                    tf.reduce_mean(
                        tf.square(self.y - self.prediction)),
                        name = 'cost')
            elif self.cost_func == 'MAE':
                self.cost = tf.reduce_mean(
                    tf.abs(self.y - self.prediction),
                    name = 'cost')
            else:
                raise NNError('Cost function ({}) not recognized'.format(cost))
            self.penalizer = self.cost

        # Set optimizer
        self.optimizer = self.optimizer().minimize(self.cost)
        self.built = True

    def _set_cost(self):
        '''Creates a cost that you can use during validation.
        ***Only use internally***
        '''
        def _acc(x,y):
            return float(np.argmax(x) == np.argmax(y))
        def _mse(x,y):
            return np.mean(np.square(x-y))
        def _rmse(x,y):
            return np.sqrt(_mse(x,y))
        def _mae(x,y):
            return np.mean(np.absolute(x-y))

        if self.classification:
            self.cost_ = _acc
        elif self.cost_func == 'MSE':
            self.cost_ = _mse
        elif self.cost_func == 'RMSE':
            self.cost_ = _rmse
        elif self.cost_func == 'MAE':
            self.cost_ = _mae
        else:
            raise NNError('NN._set_cost: typ `{}` does not exist'.format(typ))

    def fit(self, training_data, num_epochs, batch_size = None,
        validation_data = None, val_freq = None, save_periodically = None,
        verbose = None):
        '''Trains the neural network with the training data.
        If a validation set is specified, it will run validation every
        `val_freq` iterations.

        Calls `finished_building()` in the beginning to finish building the network.
        ----------
        args
        ----------
        training_data (DataWrapper)
            - Training data
        num_epochs (int)
            - Number of iterations to run the training data
        batch_size (int, Optional)
            - Size of the batches for training
            - Default number is 50
        validation_data (DataWrapper, Optional)
            - Validation data to check the progress of the training
            - If nothing is specified, then do not run valiation while training
        val_freq (int, Optional)
            - How often to run the training data (in epochs)
            - If validation_data is `None`, this has no effect
        save_periodically (int, Optional)
            - If provided, saves the network every `save_periodically` iterations
            - Default is to not save
        verbose (bool, Optional)
            - If True, prints out everything during logging.debug
            - If False, prints out much less
        '''
        if not self.built:
            self.finished_building()
        logging.info(str(self))

        if batch_size == None:
            batch_size = constants.DEFAULT_NN_BATCH_SIZE
        if val_freq == None:
            val_freq = constants.DEFAULT_NN_VAL_FREQ
        if save_periodically == None:
            save_periodically = constants.DEFAULT_NN_SAVE_PERIODICALLY
        if verbose == None:
            verbose = constants.DEFAULT_NN_VERBOSE
        if val_freq is not None and validation_data is None:
            raise NNError('If `val_freq` is not None, `validation_data` must be specified')

        # Initialize batching and global variables
        logging.info('Starting training.')
        training_data.initialize_batching(batch_size)
        # neural network save loc
        tf_save_loc = self.base_save_loc + 'nn/'
        num_training = len(training_data)
        iters_per_epoch = int(num_training/batch_size)
        self.training_performance = _TrainingPerformance(num_epochs,val_freq)

        with tf.Session() as sess:
            # Initialize the graph
            sess.run(tf.global_variables_initializer())
            for epoch in range(num_epochs):
                # reset time, epoch loss, and batching for each epoch
                last_time = time.time()
                epoch_loss = np.zeros(iters_per_epoch)
                training_data.restart_batching()

                # Run training for the epoch
                for iter in range(iters_per_epoch):
                    # Get data
                    Xs, ys = training_data.get_batch()
                    _,c = sess.run([self.optimizer,self.penalizer],
                        feed_dict = {self.x: Xs, self.y: ys})
                    if verbose:
                        logging.debug('epoch {}/{}, batch {}/{}, cost: {}'.format(
                            epoch,num_epochs,iter,iters_per_epoch,c))
                    elif iter % 2000 == 0:
                        logging.debug('epoch {}/{}, batch {}/{}, cost: {}'.format(
                            epoch,num_epochs,iter,iters_per_epoch,c))
                    epoch_loss[iter] = c

                # Log required information
                logging.info('\nepoch {}/{}, time: {}'.format(
                    epoch,num_epochs,time.time() - last_time))
                if self.classification:
                    logging.info('accuracy: mean - {:.3f}, std - {:.3f}'.format(
                        100 * np.mean(epoch_loss), 100 * np.std(epoch_loss)))
                else:
                    logging.info('cost: mean - {:.5f}, std - {:.5f}'.format(
                        np.mean(epoch_loss), np.std(epoch_loss)))

                # save if necessary and record training loss
                if (epoch+1) % save_periodically == 0:
                    self.saver.save(sess, tf_save_loc, global_step = epoch)
                self.training_performance.add_training_error(
                    np.mean(epoch_loss), np.std(epoch_loss))

                # run validation if necessary
                if (not validation_data is None) and epoch % val_freq == 0:
                    val_loss = np.zeros(len(validation_data))
                    y_true = validation_data.y_true
                    # Run validation
                    for i in range(len(validation_data)):
                        if i % 5000 == 0:
                            logging.debug('val {}/{} complete'.format(i,len(validation_data)))
                        Xs = np.reshape(validation_data.X[i,:],
                            [1, self.input_size])
                        # run nn to get prediction
                        y_pred = sess.run(self.prediction, feed_dict = {self.x: Xs})
                        val_loss[i] = self.cost_(y_pred,y_true[i,:])
                    # record performance
                    m_ = np.mean(val_loss)
                    s_ = np.std(val_loss)
                    self.training_performance.add_validation_error(m_, s_)
                    logging.info('val: mean: {}, std: {}'.format(m_,s_))
        self.trained = True
        return self

    def predict(self, src):
        '''Runs the trained nn on the input data
        Returns a multidimensional array
        For easier analysis, wrap the output in a `data_processing.DataWrapper`
        class. This is not done by defualt.
        No post processing or normalization is done
        '''
        if not self.trained:
            raise NNError('NN.run: nn not trained, cannot run')
        if len(src.shape) == 1:
            out = np.zeros(shape = (1,self.output_size))
            src_ = src[np.newaxis, ...]
        else:
            out = np.zeros(shape = (src.shape[0],self.output_size))
            src_ = src

        with tf.Session() as sess:
            for i in range(out.shape[0]):
                # Keep the shape of the array in (1,n)
                d = src_[i:i+1,:]
                out[i,:] = sess.run(self.prediction,
                                    feed_dict = {self.x: d})
        return np.squeeze(out)

    def visualize_training(self):
        if not self.trained:
            raise NNError('Cannot call visualize, has not been trained')
        return self.training_performance.visualize()

    def save(self, filename = None):
        pass
        # '''Saves the network
        # '''
        # if filename is None:
        #     if self.base_save_loc[-1] != '/':
        #         self.base_save_loc += '/'
        #     filename = self.base_save_loc + self.name
        # d = self.__dict__
        # saveobj(d,filename)

    @classmethod
    def load(cls, filename):
        # a = cls()
        # d = loadobj(filename)
        # a.__dict__ = d
        # return a
        pass


class _TrainingPerformance:
    '''Wrapper class for the training and validation performance
    during training
    '''
    def __init__(self, num_epochs, val_freq):
        '''Parameters
        ----------------
        num_epochs (int)
            - Number of epochs that are goign to be run
        val_freq (int)
            - At what epochs validation is run
        '''
        self.num_epochs = num_epochs
        self.val_freq = val_freq
        # Second dimension
        #   first element is mean
        #   second element is std
        self.train_acc = []
        self.val_acc = []


    def add_training_error(self,mean,std):
        if type(self.train_acc) != list:
            self.train_acc = self.train_acc.tolist()
        self.train_acc.append([mean,std])

    def add_validation_error(self,mean,std):
        if type(self.val_acc) != list:
            self.val_acc = self.val_acc.tolist()
        self.val_acc.append([mean,std])

    def visualize(self):
        '''Plots the training and visualization performance (mean and std) over
        the epochs
        '''
        self.train_acc = np.array(self.train_acc)
        self.val_acc = np.array(self.val_acc)

        train_x = np.arange(self.train_acc.shape[0])
        val_x = np.arange(self.val_acc.shape[0]) * self.val_freq
        fig, ax1 = plt.subplots()

        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Mean Error')
        lns1 = ax1.plot(train_x[1:], self.train_acc[1:,0], color = 'black', label = 'mean training')
        lns2 = ax1.plot(val_x[1:], self.val_acc[1:,0], color = 'blue', label = 'mean validation')

        ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
        ax2.set_ylabel('Standard Deviation')  # we already handled the x-label with ax1
        lns3 = ax1.plot(train_x[1:], self.train_acc[1:,1], color = 'red', label = 'std training')
        lns4 = ax1.plot(val_x[1:], self.val_acc[1:,1], color = 'green', label = 'std validation')

        # add a legend
        lns = lns1+lns2+lns3+lns4
        labs = [l.get_label() for l in lns]
        ax1.legend(lns, labs, loc=0)

        fig.tight_layout()  # otherwise the right y-label is slightly clipped
        return fig

def experiment_interpolation_network(name):
    '''
    This is the standard neural network used during the experiments
    for interpolation
    '''
    test = TFWrapper(base_save_loc = 'output/neural_network/',
        classification = False, name = name)
    # input size
    test.add_dense_layer(size = 20,
        activation = tf.nn.relu)
    # hidden layers
    test.add_dense_layer(size = 100,
        activation = tf.nn.relu)
    test.add_dense_layer(size = 100,
        activation = tf.nn.relu)
    # output layer
    test.add_dense_layer(size = 36,
        activation = tf.nn.relu)
    return test

def experiment_classification_network(name):
    '''
    This is the standard neural network used during the experiments
    for interpolation
    '''
    test = TFWrapper(base_save_loc = 'output/neural_network/',
        classification = True, name = name)
    # input size
    test.add_dense_layer(size = 20,
        activation = tf.nn.relu)
    # hidden layers
    test.add_dense_layer(size = 100,
        activation = tf.nn.relu)
    test.add_dense_layer(size = 100,
        activation = tf.nn.relu)
    # output layer
    test.add_dense_layer(size = 2,
        activation = tf.nn.relu)
    return test

class NNError(Exception):
    pass
