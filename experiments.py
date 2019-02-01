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
import matplotlib as mpl
import logging
import copy
import collections

from residnet.data_processing import wrappers, transforms, metrics
from residnet.comparison_methods import classification, interpolation
from residnet import util, constants, visualization

logging.basicConfig(format = constants.LOGGING_FORMAT, level = logging.INFO)
default_denorm_local = False

def main():
	table2()
	generate_preprocess_data.all(res=12)
	generate_preprocess_data.all(res=4)


def table2():
	f = open('table2_results.txt', 'w')
	n_clusters = [2,3,4,5]
	rmse_cutoffs = [[0.11],[0.07,0.15],[0.07,0.1,0.15],[0.07,0.1,0.15,0.2]]

	for i in range(4):
		f.write('\n\n\n\n\nNumber of clusters: {}\n'.format(n_clusters[i]))
		f.write('RMSE cutoffs: {}\n'.format(rmse_cutoffs[i]))
		for denorm_local in [True,False]:
			f.write('\ndenorm_local? {}\n'.format(denorm_local))
			f = experiment3(f=f, denorm_local=denorm_local, threshold=rmse_cutoffs[i],
				n_clusters=n_clusters[i])
			dw = experiment5(denorm_local=denorm_local, n_clusters= n_clusters[i])
			f.write('experiment 5, NN-Prep\n:')
			f = performance(f=f,dw=dw)

			dw = experiment6(denorm_local=denorm_local, n_clusters= n_clusters[i])
			f.write('experiment 5, NN-KMeans\n:')
			f = performance(f=f,dw=dw)

	f.close()

def table3(resolution):
	pass


def table1(threshold):
	_table1a(threshold)
	_table1b(threshold)

def _table1a(threshold):
	f = open('table1a_results.txt', 'w')
	lst = [True,False]
	f.write('experiment 0\n')
	f = experiment0(f=f)

	for ele in lst:
		f.write('\n\ndenorm_local?: {}\n'.format(ele))

		f.write('experiment 1: MLR:\n')
		f = performance(f=f, dw=experiment1(denorm_local=ele))

		f.write('experiment 2: NN_LoRes:\n')
		f = performance(f=f, dw=experiment2(denorm_local=ele))

		f.write('experiment 3: NN_RMSE:\n')
		f = experiment3(f=f,denorm_local=ele,threshold=threshold)

		f.write('experiment 4: NN_l1LR:\n')
		f = experiment4(f=f,denorm_local=ele,penalty='l1',threshold=threshold)

		f.write('experiment 4.5: NN_l2LR:\n')
		f = experiment4(f=f,denorm_local=ele,penalty='l2',threshold=threshold)

		f.write('experiment 5: NN-prep:\n')
		f = performance(f=f, dw=experiment5(denorm_local=ele,threshold=threshold))

		f.write('experiment 6: NN-KMeans:\n')
		f = performance(f=f, dw=experiment6(denorm_local=ele))

	f.close()

def _table1b(threshold):
	f = open('table1b_results.txt', 'w')
	lst = [True,False]

	for ele in lst:
		f.write('\n\ndenorm_local?: {}\n'.format(ele))

		#load only the high error idxs of the test set
		nn_rmse_load_path = 'output/models/nn_rmse_denormLocal{}.h5'.format(ele)
		model = keras.models.load_model(nn_rmse_load_path)
		testing_data = wrappers.DataPreprocessing.load(
			'output/datapreprocessing/testing_data_denorm' \
			'Local{}_res6/'.format(ele))
		test_src = testing_data.make_array(output_key = 'temp')
		pred = model.predict(test_src.X)
		pred = transforms.collapse_one_hot(pred)
		test_idxs = np.where(pred==np.argmax(constants.ONE_HOT_GREATER_THAN))[0]

		print('n start', len(pred))
		print('n idxs',len(test_idxs))

		f.write('\nexperiment 0')
		f = experiment0(f=f,test_idxs=test_idxs)

		f.write('\nexperiment 1: MLR:')
		f = performance(f=f, dw=experiment1(denorm_local=ele,test_idxs=test_idxs))

		f.write('\nexperiment 2: NN_LoRes:')
		f = performance(f=f, dw=experiment2(denorm_local=ele,test_idxs=test_idxs))

		f.write('\nexperiment 3: NN_RMSE:')
		f = experiment3(f=f,denorm_local=ele,test_idxs=test_idxs,threshold=threshold)

		f.write('\nexperiment 4: NN_l1LR:')
		f = experiment4(f=f,denorm_local=ele,penalty='l1',test_idxs=test_idxs,threshold=threshold)

		f.write('\nexperiment 4: NN_l2LR:')
		f = experiment4(f=f,denorm_local=ele,penalty='l2',test_idxs=test_idxs,threshold=threshold)

		f.write('\nexperiment 5: NN-prep:')
		f = performance(f=f, dw=experiment5(denorm_local=ele,test_idxs=test_idxs,threshold=threshold))

		f.write('\nexperiment 6: NN-KMeans:')
		f = performance(f=f, dw=experiment6(denorm_local=ele,test_idxs=test_idxs))

	f.close()

def performance(f,dw):
	'''Writes the mean RMSE, std RMSE, and bias of the datawrapper
	'''
	rmse = metrics.RMSE(dw.y_true)
	f.write('\tmean RMSE: {}\n'.format(np.mean(rmse)))
	f.write('\tstd RMSE: {}\n'.format(np.std(rmse)))
	f.write('\tbias: {} per sample\n'.format(metrics.bias(dw.y_true)/len(dw)))

	return f

def experiment0(f,test_idxs=None):
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
	data = wrappers.DataPreprocessing.load('output/datapreprocessing/testing_data_'
		'denormLocalFalse_res6/')
	logging.info('Make array')
	src = data.make_array(input_keys = ['temp'], output_key = 'temp', norm_input = False,
		idxs=test_idxs)

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
	f.write('\nnn performance')
	f = performance(f=f,dw=nn)
	nn.save(nn_save_loc)

	logging.info('Start bilinear')
	bilinear = transforms.InterpolationErrorRegression(
		src = copy.deepcopy(src),
		func = interpolation.bilinear,
		cost = metrics.Error,
		output_size = src.res ** 2)
	f.write('\nbilinear performance')
	f = performance(f=f,dw=bilinear)
	bilinear.save(bilinear_save_loc)

	logging.info('Start inverse distance weighting')
	idw = transforms.InterpolationErrorRegression(
		src = copy.deepcopy(src),
		func = interpolation.idw,
		cost = metrics.Error,
		output_size = src.res ** 2)
	f.write('\nidw performance')
	f = performance(f=f,dw=idw)
	idw.save(idw_save_loc)

	# logging.info('Start bicubic')
	# bicubic = transforms.InterpolationErrorRegression(
	# 	src = transforms.makeBicubicArrays(copy.deepcopy(src)),
	# 	func = interpolation.bicubic,
	# 	cost = metrics.Error,
	# 	output_size = src.res ** 2)
	# bicubic.save(bicubic_save_loc)
	return f

def experiment1(denorm_local,test_idxs=None):
	'''Experiment 1:
	Trains MLR on the training data and then runs it on the testing data.
	Later functions will do the statistical analysis.
	Save the results in a wrapper.DataWrapper class
	'''
	if denorm_local is None:
		denorm_local = default_denorm_local

	logging.info('denorm_local: {}'.format(denorm_local))
	mlr_error_save_loc = 'output/datawrapper/mlr_denormLocal{}' \
		'_test_error.pkl'.format(denorm_local)
	mlr_model_save_loc = 'output/models/mlr_denorm' \
		'Local{}.pkl'.format(denorm_local)

	# Make the data
	logging.info('loading training data')
	training_data = wrappers.DataPreprocessing.load(
		'output/datapreprocessing/training_data_denorm' \
		'Local{}_res6/'.format(denorm_local))
	logging.info('make training data')
	train_src = training_data.make_array(output_key = 'temp')
	train_src.normalize()

	logging.info('loading testing data')
	testing_data = wrappers.DataPreprocessing.load(
		'output/datapreprocessing/testing_data_denorm' \
		'Local{}_res6/'.format(denorm_local))
	logging.info('make testing data')
	test_src = testing_data.make_array(output_key = 'temp',idxs=test_idxs)
	test_src.normalize()

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
		arr = y_pred, norm_data = test_src.norm_data, res = test_src.res,
		output = True)

	test_src.denormalize(output = True)
	test_src.y_true -= y_pred
	test_src.save(mlr_error_save_loc)
	return test_src

def experiment2(denorm_local,test_idxs=None):
	'''Trains NN_LoRes on the training data and then runs it on the testing data
	'''
	if denorm_local is None:
		denorm_local = default_denorm_local
	nn_error_save_loc = 'output/datawrapper/nn_lores_denormLocal{}_test_error.pkl' \
		''.format(denorm_local)
	nn_model_save_loc = 'output/models/nn_lores_denormLocal{}.h5'.format(denorm_local)
	logging.info('denorm local {}'.format(denorm_local))

	# Make the training and validation data
	logging.info('loading training data')
	training_data = wrappers.DataPreprocessing.load(
		'output/datapreprocessing/training_data_denorm' \
		'Local{}_res6/'.format(denorm_local))
	# Split the training_data into both training and validation sets
	idxs = training_data.split_data_idxs(division_format = 'split',
		split_dict = {'training': 0.85, 'validation': 0.15}, randomize = True)
	train_src = training_data.make_array(output_key = 'temp', idxs = idxs['training'])
	train_src.normalize(output=True)
	val_src = training_data.make_array(output_key = 'temp', idxs = idxs['validation'])
	val_src.normalize(output=True)

	logging.info('loading testing data')
	testing_data = wrappers.DataPreprocessing.load(
		'output/datapreprocessing/testing_data_denorm' \
		'Local{}_res6/'.format(denorm_local))
	logging.info('make testing data')
	test_src = testing_data.make_array(output_key = 'temp',idxs=test_idxs)
	test_src.normalize(output = True)

	# Make, train, and save the training performance.
	model = standard_regression_network()
	model.fit(
		train_src.X,
		train_src.y_true,
		validation_data = (val_src.X, val_src.y_true),
		epochs = 10,
		batch_size = 50)
	model.save(nn_model_save_loc)

	y_pred = model.predict(test_src.X)
	# print('norm y_pred[0,:]',y_pred[0,:])
	# Denormalized prediction
	y_pred = transforms.Denormalize_arr(
		arr = y_pred, norm_data = test_src.norm_data, res = test_src.res,
		output = True)
	# print('denorm y_pred[0,:]',y_pred[0,:])

	test_src.denormalize(output = True)
	# print('denorm test_src.y_true[0,:]',test_src.y_true[0,:])

	test_src.y_true -= y_pred
	test_src.save(nn_error_save_loc)
	return test_src

def experiment3(f,denorm_local, threshold, n_clusters, test_idxs=None):
	'''Trains the classification nn (NN-RMSE) with threshold `threshold`
	'''

	nn_error_save_loc = 'output/datawrapper/nn_rmse_denormLocal{}_test_error.pkl'.format(denorm_local)
	nn_model_save_loc = 'output/models/nn_rmse_denormLocal{}.h5'.format(denorm_local)

	# Make the training and validation data
	logging.info('loading training data')
	training_data = wrappers.DataPreprocessing.load(
		'output/datapreprocessing/clf_training_data_denorm' \
		'Local{}_res6/'.format(denorm_local))
	# Split the training_data into both training and validation sets
	idxs = training_data.split_data_idxs(division_format = 'split',
		split_dict = {'training': 0.85, 'validation': 0.15}, randomize = True)
	train_src = training_data.make_array(output_key = 'temp', idxs = idxs['training'])
	val_src = training_data.make_array(output_key = 'temp', idxs = idxs['validation'])

	logging.info('loading testing data')
	testing_data = wrappers.DataPreprocessing.load(
		'output/datapreprocessing/testing_data_denorm' \
		'Local{}_res6/'.format(denorm_local))
	logging.info('make testing data')
	test_src = testing_data.make_array(output_key = 'temp',idxs=test_idxs)

	train_src = transforms.InterpolationErrorClassification(
	    src = train_src,
	    func = interpolation.bilinear,
	    cost = metrics.RMSE,
	    threshold = threshold,
	    use_corners = True)

	val_src = transforms.InterpolationErrorClassification(
	    src = val_src,
	    func = interpolation.bilinear,
	    cost = metrics.RMSE,
	    threshold = threshold,
	    use_corners = True)

	test_src = transforms.InterpolationErrorClassification(
	    src = test_src,
	    func = interpolation.bilinear,
	    cost = metrics.RMSE,
	    threshold = threshold,
	    use_corners = True)


	# Make, train, and save the training performance.
	model = standard_classification_network(n_clusters=n_clusters)

	model.fit(
		train_src.X,
		train_src.y_true,
		validation_data = (val_src.X, val_src.y_true),
		epochs = 10,
		batch_size = 50)

	model.save(nn_model_save_loc)

	y_pred = model.predict(test_src.X)

	prec = metrics.precision(y_true = test_src.y_true, y_pred = y_pred)
	acc = metrics.accuracy(y_true = test_src.y_true, y_pred = y_pred)
	f1 = metrics.F1(y_true = test_src.y_true, y_pred = y_pred)

	f.write('\n\tprecision: {}'.format(prec))
	f.write('\n\taccuracy: {}'.format(acc))
	f.write('\n\tF1: {}'.format(f1))

	# a = collections.Counter(transforms.collapse_one_hot(y_pred))
	# f.write('\n\tn0: {}'.format( a[0]))
	# f.write('\n\tn1: {}'.format( a[1]))
	return f

def experiment4(f,denorm_local, threshold,penalty='l1',test_idxs=None):
	'''Classification with Logistic regression
	'''

	# Make the training and validation data
	logging.info('loading training data')
	training_data = wrappers.DataPreprocessing.load(
		'output/datapreprocessing/clf_training_data_denorm' \
		'Local{}_res6/'.format(denorm_local))
	train_src = training_data.make_array(output_key = 'temp')

	logging.info('loading testing data')
	testing_data = wrappers.DataPreprocessing.load(
		'output/datapreprocessing/testing_data_denorm' \
		'Local{}_res6/'.format(denorm_local))
	logging.info('make testing data')
	test_src = testing_data.make_array(output_key = 'temp',idxs=test_idxs)

	train_src = transforms.InterpolationErrorClassification(
	    src = train_src,
	    func = interpolation.bilinear,
	    cost = metrics.RMSE,
	    threshold = threshold,
	    use_corners = True)

	test_src = transforms.InterpolationErrorClassification(
	    src = test_src,
	    func = interpolation.bilinear,
	    cost = metrics.RMSE,
	    threshold = threshold,
	    use_corners = True)

	test_src.y_true = np.argmax(test_src.y_true, axis = 1)
	train_src.y_true = np.argmax(train_src.y_true, axis = 1)

	model = classification.LogisticRegression(penalty = penalty)
	model.fit(X = train_src.X, y = train_src.y_true)
	y_pred = model.predict(X = test_src.X)

	prec = metrics.precision(y_true = test_src.y_true, y_pred = y_pred)
	acc = metrics.accuracy(y_true = test_src.y_true, y_pred = y_pred)
	f1 = metrics.F1(y_true = test_src.y_true, y_pred = y_pred)

	f.write('\n\tprecision: {}'.format(prec))
	f.write('\n\taccuracy: {}'.format(acc))
	f.write('\n\tF1: {}'.format(f1))

	# a = collections.Counter(transforms.collapse_one_hot(y_pred))
	# f.write('\n\tn0: {}'.format( a[0]))
	# f.write('\n\tn1: {}'.format( a[1]))

	return f

def experiment5(denorm_local, n_clusters, test_idxs=None):
	'''Train and test NN-Prep
	'''

	nn_error_save_loc = 'output/datawrapper/nn_prep_' \
		'denormLocal{}_test_error.pkl'.format(denorm_local)

	# Load interpolation training and testing data
	logging.info('loading training data')
	training_data = wrappers.DataPreprocessing.load(
		'output/datapreprocessing/training_data_denorm' \
		'Local{}_res6/'.format(denorm_local))
	train_src = training_data.make_array(output_key = 'temp')
	train_src.normalize(output=True)

	logging.info('loading testing data')
	testing_data = wrappers.DataPreprocessing.load(
		'output/datapreprocessing/testing_data_denorm' \
		'Local{}_res6/'.format(denorm_local))
	logging.info('make testing data')
	test_src = testing_data.make_array(output_key = 'temp',idxs=test_idxs)
	test_src.normalize(output = True)

	# Load classification network and make NN-Prep
	nn_clf_model_save_loc = 'output/models/nn_rmse_denormLocal{}.h5'.format(denorm_local)
	regs = {}
	for i in range(n_clusters):
		regs[int(i)] = standard_regression_network()
	a = interpolation.ClfInterpWrapper(
		clf = keras.models.load_model(nn_clf_model_save_loc),
		regs = regs,
		res = 6)

	# Train
	a.fit_interp(X=train_src.X, y=train_src.y_true)
	y_pred = a.predict(test_src.X)

	# Denormalize
	# print('norm y_pred[0,:]',y_pred[0,:])
	y_pred = transforms.Denormalize_arr(
		arr = y_pred, norm_data = test_src.norm_data, res = test_src.res,
		output = True)
	# print('denorm y_pred[0,:]',y_pred[0,:])
	test_src.denormalize(output = True)

	test_src.y_true -= y_pred
	test_src.save(nn_error_save_loc)
	return test_src

def experiment6(denorm_local, n_clusters,test_idxs=None):
	'''Train and test NN-KMeans
	'''

	nn_error_save_loc = 'output/datawrapper/nn_kmeans_' \
		'denormLocal{}_test_error.pkl'.format(denorm_local)

	# Load interpolation training and testing data
	logging.info('loading training data')
	training_data = wrappers.DataPreprocessing.load(
		'output/datapreprocessing/training_data_denorm' \
		'Local{}_res6/'.format(denorm_local))
	train_src = training_data.make_array(output_key = 'temp')
	train_src.normalize(output=True)

	logging.info('loading testing data')
	testing_data = wrappers.DataPreprocessing.load(
		'output/datapreprocessing/testing_data_denorm' \
		'Local{}_res6/'.format(denorm_local))
	logging.info('make testing data')
	test_src = testing_data.make_array(output_key = 'temp',idxs=test_idxs)
	test_src.normalize(output = True)

	clf_training_data = wrappers.DataPreprocessing.load(
		'output/datapreprocessing/clf_training_data_denorm' \
		'Local{}_res6/'.format(denorm_local))
	clf_train_src = clf_training_data.make_array(output_key = 'temp')

	kmeans = classification.KMeansWrapper(n_clusters=n_clusters)
	kmeans.fit(X=clf_train_src.X)

	# Load classification network and make NN-Prep
	regs = {}
	for i in range(n_clusters):
		regs[int(i)] = standard_regression_network()

	a = interpolation.ClfInterpWrapper(
		clf = kmeans,
		regs = regs,
		res = 6)

	# Train
	a.fit_interp(X=train_src.X, y=train_src.y_true)
	y_pred = a.predict(test_src.X)

	# Denormalize
	# print('norm y_pred[0,:]',y_pred[0,:])
	y_pred = transforms.Denormalize_arr(
		arr = y_pred, norm_data = test_src.norm_data, res = test_src.res,
		output = True)
	# print('denorm y_pred[0,:]',y_pred[0,:])
	test_src.denormalize(output = True)

	test_src.y_true -= y_pred
	test_src.save(nn_error_save_loc)
	return test_src

####################
# Auxiliary methods
####################
class generate_preprocess_data:
	'''Static method wrapper for generating data to use in experiments.
	'''
	@classmethod
	def training(cls, denorm_local = None, res = None):
		'''Generate data used for training and validation.

		if save_loc is None, it will generate the data from scratch.
		If save_loc is not None, it will load it from that location
		'''
		logging.info('Generating training data')
		if denorm_local is None:
			denorm_local = default_denorm_local
		if res is None:
			res = 6
		return cls._generate_data(years = ['2005','2006'], denorm_local = denorm_local,
			name = 'training_data', res_in=res)

	@classmethod
	def testing(cls, denorm_local = None, res=None):
		'''Generates data for testing. This is only year 2008.
		'''
		logging.info('Generating testing data')
		if denorm_local is None:
			denorm_local = default_denorm_local
		if res is None:
			res = 6
		return cls._generate_data(years = ['2008'], denorm_local = denorm_local,
			name = 'testing_data', res_in=res)

	@classmethod
	def classifier_training(cls, denorm_local = None, res=None):
		'''Generates the data that is used to train the classifiers (year
		2007).
		'''
		logging.info('Generating classifier training data')
		if denorm_local is None:
			denorm_local = default_denorm_local
		if res is None:
			res = 6
		return cls._generate_data(years = ['2007'], denorm_local = denorm_local,
			name = 'clf_training_data', res_in=res)

	@classmethod
	def all(cls,res=None):
		'''Generates all the data used in the entire study.
		'''
		if res is None:
			res = 6
		cls.training(denorm_local = False, res=res)
		cls.training(denorm_local = True, res=res)
		cls.testing(denorm_local = False, res=res)
		cls.testing(denorm_local = True, res=res)
		cls.classifier_training(denorm_local = False, res=res)
		cls.classifier_training(denorm_local = True, res=res)

	@staticmethod
	def _generate_data(years, denorm_local, name, res_in):
		'''Core function to generate data
		'''
		prep_data = wrappers.DataPreprocessing(
			name = name,
			years = years,
			denorm_local = denorm_local,
			res_in=res_in)
		prep_data.parse_data()
		prep_data.save()
		return prep_data

def standard_classification_network(n_clusters):
	model = Sequential()
	model.add(Dense(units=100, activation='relu',input_dim=20))
	model.add(Dense(units=100, activation='relu'))
	model.add(Dropout(0.3))
	model.add(Dense(units=n_clusters,activation='softmax'))
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
	model.add(Dense(units=36,activation='tanh'))
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
		train_src = training_data.make_array(output_key = 'temp')
		if threshold is not None:
			train_src.denormalize(output = True)
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
		clf_train_src = clf_training_data.make_array(output_key = 'temp')
		if threshold is not None:
			clf_train_src.denormalize(output = True)
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
	# test_src = testing_data.make_array(output_key = 'temp')
	test_src = testing_data.make_array(output_key = 'temp')
	if threshold is not None:
		test_src.denormalize(output = True)
		test_src = transforms.InterpolationErrorClassification(
			src = test_src,
			func = interpolation.bilinear,
			cost = metrics.RMSE,
			threshold = threshold,
			use_corners = True)

	return {'train_src':train_src,
		'test_src': test_src,
		'clf_train_src': clf_train_src}

if __name__ == '__main__': main()
