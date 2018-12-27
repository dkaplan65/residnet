# Residnet

A supervised interpolation correction scheme for geophysical models that learns sub-pixel, nonlinear behavior.


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

## Quick Start

TODO


There are also some walkthrough examples in _examples.py_.

## References
"Optimization of Geophysical Interpolation using Neural Networks", David Kaplan and Stephen Penny (in preparation).

## License Information
Copyright 2018 David Kaplan and Dr. Stephen Penny. Released under the GNU General Public License version 3.0, see LICENSE.txt.
