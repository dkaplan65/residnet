'''
Author: David Kaplan
Advisor: Stephen Penny

Runs full experiments from scratch to finish.

If no parameters are passed into the experiment functions, then it
uses default locations and data.
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
default_denorm_local = False

def main():
	generate_preprocess_data.all_data()
	# experiment2(denorm_local = False)
	# experiment4(denorm_local = True)

def experiment0():
	'''Experiment 0:

	These are the non machine-learning comparison methods that we are going to compare the
	neural network to. This experiments runs these methods. Later functions
	will then do the statistical analysis (RMSE, variance, bias) of the errors
		- bilinear
		- bicubic
		- IDW (p == 2)
		- nearest neighbour
	Save the results in wrappers.DataWrapper class.

	-----------
	output
	-----------
	A data_processing.DataWrapper object for each of the interpolation methods,
	where the DataWrapper.y_true attribute is the error of the interpolation method.
	Saved in the location specified above.
	'''
	# Set default parameters if necessary
	# It doesn't matter whether we do denorm local or global
	logging.info('Entering experiment 0')
	logging.info('Load data')
	data = wrappers.DataPreprocessing.load('output/datapreprocessing/testing_data_denormLocalFalse_res6/')
	logging.info('Denormalize')
	data.denormalize()
	logging.info('Make array')
	src = data.make_array(input_keys = ['temp'], output_key = 'temp')
	logging.info('denormalize')
	src.denormalize()

	nn_save_loc = 'output/datawrapper/nearest_neighbor_test_error.pkl'
	bilinear_save_loc = 'output/datawrapper/bilinear_test_error.pkl'
	bicubic_save_loc = 'output/datawrapper/bicubic_test_error.pkl'
	idw_save_loc = 'output/datawrapper/idw_test_error.pkl'

	logging.info('Start nearest neighbour')
	nn = transforms.InterpolationErrorRegression(
		src = copy.deepcopy(src),
		func = interpolation.nearest_neighbor,
		cost = metrics.Error,
		output_size = src.res ** 2)
	nn.save(nn_save_loc)

	logging.info('Start bilinear')
	bilinear = transforms.InterpolationErrorRegression(
		src = copy.deepcopy(src),
		func = interpolation.bilinear,
		cost = metrics.Error,
		output_size = src.res ** 2)
	bilinear.save(bilinear_save_loc)

	logging.info('Start inverse distance weighting')
	idw = transforms.InterpolationErrorRegression(
		src = copy.deepcopy(src),
		func = interpolation.idw,
		cost = metrics.Error,
		output_size = src.res ** 2)
	idw.save(idw_save_loc)

	logging.info('Start bicubic')
	bicubic = transforms.InterpolationErrorRegression(
		src = transforms.makeBicubicArrays(copy.deepcopy(src)),
		func = interpolation.nearest_neighbor,
		cost = metrics.Error,
		output_size = src.res ** 2)
	bicubic.save(bicubic_save_loc)

def experiment1(denorm_local = None):
	'''Experiment 1:
	Trains MLR on the training data and then runs it on the testing data.
	Later functions will do the statistical analysis.
	Save the results in a wrapper.DataWrapper class
	'''
	if denorm_local is None:
		denorm_local = default_denorm_local

	logging.info('denorm_local: {}'.format(denorm_local))
	mlr_error_save_loc = 'output/datawrapper/mlr_test_error_denormLocal{}.pkl'.format(denorm_local)
	mlr_model_save_loc = 'output/models/mlr_denormLocal{}.pkl'.format(denorm_local)

	# Make the data
	logging.info('loading training data')
	training_data = wrappers.DataPreprocessing.load(
		'output/datapreprocessing/training_data_denormLocal{}_res6/'.format(denorm_local))
	training_data.normalize()
	logging.info('make training data')
	train_src = training_data.make_array(output_key = 'temp')

	logging.info('loading testing data')
	testing_data = wrappers.DataPreprocessing.load(
		'output/datapreprocessing/testing_data_denormLocal{}_res6/'.format(denorm_local))
	testing_data.normalize()
	logging.info('make testing data')
	test_src = testing_data.make_array(output_key = 'temp')

	# Train the model and then predict the testing data
	logging.info('Run MLR')
	mlr = interpolation.MLR()
	mlr.fit(
		X = train_src.X,
		y = train_src.y_true)
	y_pred = mlr.predict(X = test_src.X)
	mlr.save(filename=mlr_model_save_loc)

	# Set y_true to the denormalized difference between MLR and truth
	y_pred = transforms.Denormalize_arr(
		arr = y_pred, norm_data = test_src.norm_data)

	test_src.denormalize(denorm_y = True)
	test_src.y_true -= y_pred
	test_src.save(mlr_error_save_loc)

def experiment2(denorm_local = None):
	'''Trains NN_LoRes on the training data and then runs it on the testing data
	'''
	if denorm_local is None:
		denorm_local = default_denorm_local
	nn_error_save_loc = 'output/datawrapper/nn_lores_test_error_denormLocal{}.pkl'.format(denorm_local)
	nn_model_save_loc = 'output/models/nn_lores.h5'

	logging.info('denorm local {}'.format(denorm_local))

	# Make the (training, validation, testing)
	training_data = wrappers.DataPreprocessing.load('output/datapreprocessing/training_data_denormLocal{}_res6/'.format(
		denorm_local))
	training_data.normalize()
	# Split the training_data into both training and validation sets
	idxs = training_data.split_data_idxs(
		division_format = 'split',
		split_dict = {'training': 0.85, 'validation': 0.15},
		randomize = True)
	train_src = training_data.make_array(output_key = 'temp', idxs = idxs['training'])
	val_src = training_data.make_array(output_key = 'temp', idxs = idxs['validation'])

	# Make the testing data
	testing_data = wrappers.DataPreprocessing.load('output/datapreprocessing/testing_data_denormLocal{}_res6/'.format(
		denorm_local))
	testing_data.normalize()
	test_src = testing_data.make_array(output_key = 'temp')
	print('norm test_src.y_true[0,:]',test_src.y_true[0,:])

	# Make, train, and save the training performance.
	model = standard_regression_network()
	model.fit(
		train_src.X,
		train_src.y_true,
		validation_data = (val_src.X, val_src.y_true),
		epochs = 2,
		batch_size = 50)
	model.save(nn_model_save_loc)

	y_pred = model.predict(test_src.X)
	print('norm y_pred[0,:]',y_pred[0,:])
	# Denormalized prediction
	y_pred = transforms.Denormalize_arr(
		arr = y_pred,
		norm_data = test_src.norm_data)
	print('denorm y_pred[0,:]',y_pred[0,:])

	test_src.denormalize(denorm_y = True)
	print('denorm test_src.y_true[0,:]',test_src.y_true[0,:])

	test_src.y_true -= y_pred

	print(metrics.RMSE(test_src.y_true))

	map = transforms.mapify(
		loc_to_idx = test_src.loc_to_idx,
		arr = test_src.y_true,
		year = '2008',
		day = 25,
		res = test_src.res,
		classification = False)

	plt.imshow(map)
	plt.show()

	test_src.save(nn_error_save_loc)

def experiment3(denorm_local = None, threshold = None):
	'''Trains the classification preprocess nn (NN-RMSE) with threshold `threshold`
	'''
	if denorm_local is None:
		denorm_local = default_denorm_local
	if threshold is None:
		# Defaults to the mean bilinear mean RMSE
		threshold = 0.103

	nn_error_save_loc = 'output/datawrapper/nn_rmse_test_error.pkl'
	nn_model_save_loc = 'output/models/nn_rmse.h5'

	d = load_datawrappers(denorm_local = denorm_local, threshold = threshold,
		load_reg_train = False, load_clf_train = True)
	train_src = d['clf_train_src']
	test_src = d['test_src']

	# Make, train, and save the training performance.
	model = standard_classification_network()

	model.fit(
		train_src.X,
		train_src.y_true,
		# validation_data = (val_src.X, val_src.y_true),
		epochs = 5,
		batch_size = 50)

	model.save(nn_model_save_loc)

	y_pred = model.predict(test_src.X)

	print('y_pred type', type(y_pred))
	print('y_pred.shape:',y_pred.shape)

	prec = metrics.precision(y_true = test_src.y_true, y_pred = y_pred)
	acc = metrics.accuracy(y_true = test_src.y_true, y_pred = y_pred)
	f1 = metrics.F1(y_true = test_src.y_true, y_pred = y_pred)

	print('precision: {}'.format(prec))
	print('accuracy: {}'.format(acc))
	print('F1: {}'.format(f1))

	# Calculate error
	error = np.zeros(y_pred.shape[0])
	for i in range(len(y_pred)):
		if not np.array_equal(y_pred[i,:], test_src.y_true[i,:]):
			error[i] = 1

	map = transforms.mapify(
		loc_to_idx = test_src.loc_to_idx,
		arr = error,
		year = '2008',
		day = 25,
		res = test_src.res,
		classification = True)

	plt.imshow(map)
	plt.show()

def experiment4(denorm_local = None, threshold = None):
	'''Classification with Logistic regression
	'''
	if denorm_local is None:
		denorm_local = default_denorm_local
	if threshold is None:
		# Defaults to the mean bilinear mean RMSE
		threshold = 0.103

	nn_error_save_loc = 'output/datawrapper/nn_rmse_test_error.pkl'
	nn_model_save_loc = 'output/models/nn_rmse.h5'

	d = load_datawrappers(denorm_local = denorm_local, threshold = threshold,
		load_reg_train = False, load_clf_train = True)
	train_src = d['clf_train_src']
	test_src = d['test_src']

	test_src.y_true = np.argmax(test_src.y_true, axis = 1)

	model = classification.LogisticRegression(penalty = 'l2')
	model.fit(X = train_src.X, y = train_src.y_true)
	y_pred = model.predict(X = test_src.X)

	prec = metrics.precision(y_true = test_src.y_true, y_pred = y_pred)
	acc = metrics.accuracy(y_true = test_src.y_true, y_pred = y_pred)
	f1 = metrics.F1(y_true = test_src.y_true, y_pred = y_pred)

	print('precision: {}'.format(prec))
	print('accuracy: {}'.format(acc))
	print('F1: {}'.format(f1))

def experiment5(denorm_local = None, threshold = None):
	'''Trains the classification preprocess nn (NN-ENSM) with threshold `threshold`
	'''
	if denorm_local is None:
		denorm_local = default_denorm_local
	if threshold is None:
		# Defaults to the mean bilinear mean RMSE
		threshold = 0.103

	# nn_error_save_loc = 'output/datawrapper/nn_rmse_test_error.pkl'
	# nn_model_save_loc = 'output/models/nn_rmse.h5'

	d = load_datawrappers(denorm_local = denorm_local, threshold = threshold,
		load_reg_train = False, load_clf_train = True)
	train_src = d['clf_train_src']
	test_src = d['test_src']

	# Make, train, and save the training performance.
	model = standard_classification_network()

	nn = classification.NNEnsemble(model = model, size = 9)
	nn.fit(X = train_src.X, y = train_src.y_true, epochs = 5)
	y_pred = nn.predict(test_src.X)

	print('y_pred.shape:',y_pred.shape)
	print('y_true.shape:',test_src.y_true.shape)

	prec = metrics.precision(y_true = test_src.y_true, y_pred = y_pred)
	acc = metrics.accuracy(y_true = test_src.y_true, y_pred = y_pred)
	f1 = metrics.F1(y_true = test_src.y_true, y_pred = y_pred)

	print('precision: {}'.format(prec))
	print('accuracy: {}'.format(acc))
	print('F1: {}'.format(f1))

def experiment6(denorm_local = None, threshold = None):
	'''Train and test NN-Prep
	'''
	if denorm_local is None:
		denorm_local = default_denorm_local
	if threshold is None:
		# Defaults to the mean bilinear mean RMSE
		threshold = 0.103

	# nn_error_save_loc = 'output/datawrapper/nn_rmse_test_error.pkl'
	# nn_model_save_loc = 'output/models/nn_rmse.h5'

	d = load_datawrappers(denorm_local = denorm_local, threshold = threshold,
		load_reg_train = False, load_clf_train = True)



####################
# Auxiliary methods
####################
class generate_preprocess_data:
	'''Static method wrapper for generating data to use in experiments.
	'''
	@classmethod
	def training(cls, denorm_local = None):
		'''Generate data used for training and validation.

		if save_loc is None, it will generate the data from scratch.
		If save_loc is not None, it will load it from that location
		'''
		logging.info('Generating training data')
		if denorm_local is None:
			denorm_local = default_denorm_local
		return cls._generate_data(years = ['2005','2006'], denorm_local = denorm_local,
			name = 'training_data')

	@classmethod
	def testing(cls, denorm_local = None):
		'''Generates data for testing. This is only year 2008.
		'''
		logging.info('Generating testing data')
		if denorm_local is None:
			denorm_local = default_denorm_local
		return cls._generate_data(years = ['2008'], denorm_local = denorm_local,
			name = 'testing_data')

	@classmethod
	def classifier_training(cls, denorm_local = None):
		'''Generates the data that is used to train the classifiers (year
		2007).
		'''
		logging.info('Generating classifier training data')
		if denorm_local is None:
			denorm_local = default_denorm_local
		return cls._generate_data(years = ['2007'], denorm_local = denorm_local,
			name = 'clf_training_data')

	@classmethod
	def all_data(cls):
		'''Generates all the data used in the entire study.
		'''
		cls.training(denorm_local = False)
		cls.training(denorm_local = True)
		cls.testing(denorm_local = False)
		cls.testing(denorm_local = True)
		cls.classifier_training(denorm_local = False)
		cls.classifier_training(denorm_local = True)

	@staticmethod
	def _generate_data(years, denorm_local, name):
		'''Core function to generate data
		'''
		prep_data = wrappers.DataPreprocessing(
			name = name,
			years = years,
			denorm_local = denorm_local)
		prep_data.parse_data()
		prep_data.normalize()
		prep_data.save()
		return prep_data

def table1a_analysis(
	src_bilinear,
	src_nearest_neighbor,
	src_bicubic,
	src_idw,
	src_global_mlr,
	src_local_mlr,
	src_global_nn_lores,
	src_local_nn_lores,
	src_global_nn_prep,
	src_local_nn_prep,
	src_global_kmeans,
	src_local_kmeans):
	'''Computes statistics for table 1a in the paper
	Each one of the inputs is a wrapper.DataWrapper object that is
	the residual of respective interpolation.
	'''
	pass

def table1b_analysis(
	src_bilinear,
	src_nearest_neighbor,
	src_bicubic,
	src_idw,
	src_global_mlr,
	src_local_mlr,
	src_global_nn_lores,
	src_local_nn_lores,
	src_global_nn_prep,
	src_local_nn_prep,
	src_global_kmeans,
	src_local_kmeans):
	'''Computes statistics for table 1b in the paper
	Each one of the inputs is a wrapper.DataWrapper object that is
	the residual of respective interpolation.
	'''
	pass

def standard_classification_network():
	model = Sequential()
	model.add(Dense(units=100, activation='relu',input_dim=20))
	model.add(Dense(units=100, activation='relu'))
	model.add(Dropout(0.3))
	model.add(Dense(units=2,activation='softmax'))
	model.compile(
		loss = 'categorical_crossentropy',
		optimizer = 'adam',
		metrics=['accuracy'])
	return model

def standard_regression_network():
	model = Sequential()
	model.add(Dense(units=100, activation='relu',input_dim=20))
	model.add(Dense(units=100, activation='relu'))
	model.add(Dropout(0.3))
	model.add(Dense(units=36,activation='relu'))
	model.compile(
		loss = 'mse',
		optimizer = 'adam',
		metrics=['mse'])
	return model

def load_datawrappers(denorm_local, load_reg_train, load_clf_train,
	threshold = None):
	'''Loads baseline datawrappers for training
	'''
	if load_reg_train:
		logging.info('Loading regression training data')
		training_data = wrappers.DataPreprocessing.load(
			'output/datapreprocessing/training_data_denormLocal{}_res6/'.format(
			denorm_local))
		training_data.normalize()
		train_src = training_data.make_array(output_key = 'temp')
		if threshold is not None:
			train_src.denormalize(denorm_y = True)
			train_src = transforms.InterpolationErrorClassification(
			    src = train_src,
			    func = interpolation.bilinear,
			    cost = metrics.RMSE,
			    threshold = threshold,
				use_corners = True)
	else:
		train_src = None

	if load_clf_train:
		logging.info('Loading classification training data')
		clf_training_data = wrappers.DataPreprocessing.load(
			'output/datapreprocessing/clf_training_data_denormLocal{}_res6/'.format(
			denorm_local))
		clf_training_data.normalize()
		clf_train_src = clf_training_data.make_array(output_key = 'temp')
		if threshold is not None:
			clf_train_src.denormalize(denorm_y = True)
			clf_train_src = transforms.InterpolationErrorClassification(
			    src = clf_train_src,
			    func = interpolation.bilinear,
			    cost = metrics.RMSE,
			    threshold = threshold,
				use_corners = True)
	else:
		clf_train_src = None

	# Make the testing data
	logging.info('Loading testing data')
	testing_data = wrappers.DataPreprocessing.load(
		'output/datapreprocessing/testing_data_denormLocal{}_res6/'.format(
		denorm_local))
	testing_data.normalize()
	# test_src = testing_data.make_array(output_key = 'temp')
	test_src = testing_data.make_array(output_key = 'temp')
	if threshold is not None:
		test_src.denormalize(denorm_y = True)
		test_src = transforms.InterpolationErrorClassification(
			src = test_src,
			func = interpolation.bilinear,
			cost = metrics.RMSE,
			threshold = threshold,
			use_corners = True)

	return {'train_src':train_src,
		'test_src': test_src,
		'clf_train_src': clf_train_src}


if __name__ == '__main__':
	main()
