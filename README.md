# Residnet

A supervised interpolation correction scheme for geophysical models that learns sub-pixel, nonlinear behavior.  

This repository holds the code that was used to generate the results for the paper "Optimization of Geophysical Interpolation Interpolation using Neural Networks" by David Kaplan and Stephen Penny. Reference at the bottom.

## Background
TODO - talk about the background of the problem, why it is necessary (look at introduction of paper)

'''
This data is then subsampled to create the low resolution maps that are representative of the resolution of typical global geophysical forecasting models ([NCEP](https://www.ncep.noaa.gov)). We then train the neural network on the low resolution information to try to make it match the information in the high resolution data. Full details
'''

## Installation
Current version only tested with Python 3.6.5.  

The recommended way to install the program is to use an encapsulated _conda_ environment. However, you can do this within a global environment without using _conda_.

#### With Anaconda
`$ conda create -n residnet python=3.6.5`  
`$ conda activate residnet` or `$ source activate residnet`  
`$ git clone https://github.com/dkaplan65/residnet.git`  
`$ pip install sklearn scipy scikit-learn matplotlib keras netcdf4=1.3.1 numpy`  

#### Supported platforms
This version is only tested on Mac and Linux systems.

## Obtaining Data
The data used in the paper is simulated, high-resolution data from
[HYCOM](https://www.hycom.org), specifically the data from [Gulf of Mexico Reanalysis GOMl0.04 Experiment 20.1](https://www.hycom.org/data/goml0pt04/expt-20pt1) from years 2005-2008. You can download the data for free directly from the experiment webpage for free.

## Quick Start

TODO

There are also some walkthrough examples in _examples.py_.

## Extension
The code was built to be flexible in data source, correction methods, and application domains.

## References
"Optimization of Geophysical Interpolation using Neural Networks", David Kaplan and Stephen Penny (in preparation).

## Correspondence
For questions, email dkaplan4 [at] terpmail [dot] umd [dot] edu

## License Information
Copyright 2018 David Kaplan and Dr. Stephen Penny. Released under the GNU General Public License version 3.0, see LICENSE.txt.
