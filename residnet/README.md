# Description of modules

The purpose of this document is to describe the purpose of the modules and classes used within the code.

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

TODO

## data_processing/transforms.py

TODO

## comparison_methods/*

TODO

## comparison_methods/interpolation.py

TODO

## comparison_methods/classification.py
