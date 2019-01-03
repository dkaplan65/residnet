# Residnet (RESIDual neural NETwork)

A supervised interpolation correction scheme for geophysical models by learning sub-pixel, nonlinear behavior.  

This repository holds the code that was used to generate the results for the paper "Optimization of Geophysical Interpolation Interpolation using Neural Networks" by David Kaplan and Stephen Penny. Reference at the bottom.

## Background

Simple interpolation techniques (bilinear, nearest neighbor, inverse distance weighting) are used in data assimilation to convert from model space to observation space and are known to introduce errors because of their inability to resolve dynamics at the subgrid scale level - a phenomenon particularly noticeable along fronts and eddies. In data assimilation applications these regions are the most sensitive to noise, and such errors can lead to false ‘corrections’ in the analysis. For example, the image below shows the RMSE of a bilinear interpolation.  

![Residual_Error](https://github.com/dkaplan65/residnet/blob/master/residual_error.png)

The result is a reduction in accuracy for any forecasts that use this analysis as initial conditions. The best way to eliminate this error is by increasing the resolution of the model. However, the computational resources needed for this are prohibitive, so an alternate, more computationally feasible approach has to be implemented.

We propose a two step interpolation correction scheme for geophysical forecast models that uses feed-forward networks (FFN) to predict and correct errors by a bilinear interpolation due to unresolved subgrid-scale dynamics.

For full details on the method, see reference below.

## Installation
Current version only tested with Python 3.6.5.  

The recommended way to install the program is to clone the repository into an encapsulated `conda` environment. You can also do this within a global environment without using `conda`.

#### With Anaconda
`$ conda create -n residnet python=3.6.5`  
`$ conda activate residnet` or `$ source activate residnet`  
`$ git clone https://github.com/dkaplan65/residnet.git`  
`$ pip install sklearn scipy scikit-learn matplotlib keras netcdf4=1.3.1 numpy`  

#### Supported platforms
This version is only tested on Mac and Linux systems.

## Obtaining Data
The data used in the paper is simulated, high-resolution data from
[HYCOM](https://www.hycom.org), specifically the data from [Gulf of Mexico Reanalysis GOMl0.04 Experiment 20.1](https://www.hycom.org/data/goml0pt04/expt-20pt1) from years 2005-2008. You can download the data for free directly from the experiment webpage.

## Quick Start

See `examples.py`

For descriptions about the code, look at README.md in `residnet/`

## Extension
The code was built to be flexible in terms of data source, correction methods, process control, and application domains. If you want to use this code as a base for your project, cite the reference below.

## Reference
"Optimization of Geophysical Interpolation using Neural Networks", David Kaplan and Stephen Penny (in preparation).

## Correspondence
For questions, email dkaplan4 [at] terpmail [dot] umd [dot] edu.

## License Information
Copyright 2018 David Kaplan and Dr. Stephen Penny. Released under the GNU General Public License version 3.0, see LICENSE.txt.

## Future Features
* Zip saved files
* Parallelize code where it makes sense
  * de/normalization, preprocessing, transforms
