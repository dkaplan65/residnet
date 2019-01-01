# Description of modules

The purpose of this document is to describe the purpose of the modules and classes used within the code. For a more in depth description, look at the code in the module and in the module `../exmaples.py`.

## Basic workflow
The basic workflow of how to use this package is as follows:  
1. Read the raw data (`datasets`)
2. Preprocess the raw data into subgrids (`data_processing/wrappers.DataPreprocessing`)
3. Aggregate the subgrids into training and validation sets (`data_processing/wrappers.DataWrapper`)
  * Optional: Transform the data into the appropriate format (`data_processing.transforms`)
4. Build and train classification or interpolation models on these sets (`comparison_methods/*`)
5. Asses the performance of these models (`data_processing/metrics`)

See below for further descriptions of the modules/packages

## constants.py
Constants used within every module so they are consistent.

## datasets.py
The classes in this module read in raw data like netCDF files. The class `HYCOM` reads in netCDF files produced specifically by [HYCOM](https://www.hycom.org). You can make your own class that reads in data from a different source. The key attributes that the dataset class must have are:  
* data (dict: str -> subclass)  
  * The key is the year as a string (ex: '2008')
  * subclass is a container class another data dictionary that maps 'key' (ex: 'temp', 'ssh', etc.) to an actual data array that contains the data for that 'key' at the specific 'year'.

## util.py
Defines utility functions and base classes that handle IO.

## visualization.py
Defines functions that graph results. Descriptions of the functions can be found in the functions themselves. If you want to create some functions for visualization, implement them in this module.

## data_processing/*
This package in general holds modules that have to do with performance analysis (`metrics.py`), transforms (`transforms.py`), data preprocessing (`wrappers.DataPreprocessing`), and data aggregation/containers (`wrappers.DataWrapper`).

## data_processing/metrics.py

This module has methods for assessing performance of both classification and interpolation methods.

#### Interpolation
* `Error` (plain difference)
* `SE` (squared error)
* `AE` (absolute error)
* `MAE` (mean absolute error)
* `MSE` (mean square error)
* `RMSE` (root mean square error)
* `logMSE` (log mean square error)
* `bias`
* `variance`
* `std` (standard deviation)

#### Classification
* `confusion_matrix`
* `precision`
* `accuracy`
* `recall`
* `F1`
* `specificity`
* `sensitivity`

## data_processing/wrappers.py

This is one of the core modules because it holds two key classes: `DataPreprocessing` and `DataWrapper`.

#### DataPreprocessing

The purpose of `DataPreprocessing` is to convert raw input data (most likely from netCDF files) into preprocessed, subgrids with metadata that can be combined into training data, testing data, etc. There are two major functions; `parse_data` and `make_array`.

###### Load and parse the raw data with the function `parse_data`  
This function loads the raw data and parses it with a single function `parse_data`. This only parses the data into an internal representation.

###### Optional:
After you have parsed the data, call the function `split_data_idxs` to divide the data into different categories like 'training', 'testing', etc. This function assigns each subgrid (with respect to its index in the master array in `DataPreprocessing`) to a category that is specified in `split_data_idxs`.   

For example,  
```python
idxs = dataprerocessing.split_data_idxs(
    division_format = 'split',
    d = {'training': 0.8, 'testing': 0.2},
    randomize = True)  
```  
splits the data into two categories: _training_ and _testing_ and assigns 80% of the subgrids randomly to _training_ and 20% of subgrids to _testing_. Once you do this, you can feed these indices into `make_array`.

###### Create a useful dataset with `make_array`
The next step is to transform this into a dataset that you can use for training or for feeding to interpolation methods. You can do this with the function `make_array`. `make_array` creates a `DataWrapper` object that encapsulates the data. Description of `DataWrapper` below. For example,

```python
$ datawrapper = datapreprocessing.make_array(
    idxs = idxs['testing'])
```

creates a `DataWrapper` that encapsulates the _testing_ data and this is ready to be fed into an interpolation. The output of this function is a `DataWrapper` where the input data are corner/s data and the output is the truth high resolution subgrid scale data as described in the root README.md. To transform this data into classification data, refer to `data_processing.transforms`. Description about transforms below.

#### DataWrapper

The purpose of `DataWrapper` is to be a container for data made by `DataPreprocessing.make_array`. It keeps all of the data related to normalization, location, input and truth output data, and other metadata together and provides functionality for saving. `DataPreprocessing.make_array` automatically returns a `DataWrapper` object. You use this object to pass around data for interpolation, classification, etc.

## data_processing/transforms.py

This module provides functionality for transforming the data in `DataWrappers` into different types of arrays. One of the most common transforms you'll do is to transform an interpolation truth data into classification data. This can be accomplished by calling the method `transforms.InterpolationErrorClassification`, which creates a one-hot array based on the error of the output given an interpolation method and input data. This module also provides functions that are used for de/normalization, transforming output data into map images, and transforming classification output data to and from one-hot encoding.

## comparison_methods/*

This package holds modules for both classification and interpolation modules.

## comparison_methods/interpolation.py

This module provides functions and classes that are used for interpolation. Some interpolation methods are:

* Functions
  * `bilinear`
  * `bicubic`
  * `nearest_neighbor`
  * `idw`  (Inverse-Distance Weighting)
* Classes
  * `MLR` (Multiple Linear Regression)
  * `ClfInterpWrapper` (Classification Interpolation Wrapper)
    * This class is what was used for _NN-Prep_ in the paper. It divides the input data into different sets based on a classification method. Each set is then applied to a different interpolation method.

## comparison_methods/classification.py

This module provides classes for classification. Many of these classes are wrappers for sklearn objects. This is the list of classification methods:

* `RandomForestWrapper` (Random Forest)
* `KMeansWrapper` (KMeans)
* `LogisticRegressionWrapper` (Logistic Regression)
