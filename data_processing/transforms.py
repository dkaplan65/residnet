'''
Author: David Kaplan
Advisor: Stephen Penny

Utilities for transforming data matrices.
Divided between core and aggregation/wrapper functions.

#########################################################
A description about normalization and denormalization:
#########################################################
    Normalization is different if the array is an input (feeds into an algorithm)
    or an output (ground truth).

    If the array is an input array
        normalized = (orig - avg)/range, where
            orig = denormalized input array
            avg = mean(orig)
            range = max(abs(orig - avg))

    If the array is an output array
        normalized = (orig - bilin)/range, where
            orig = denormalized output array
            bilin = bilinear interpolation from the corners of orig
            avg = max(abs(orig-mean(corners(orig))))

    Denormalization is just this in reverse.

    Why do normalization this way?
        Normalization is only necessary if we are feeding in the data into a ML
        algorithm. The goal of the ML algorithm is to predict subgrid scale nonlinear
        behavior - so the output should be the residual of a bilinear interpolation. We
        then divide by the average of the corner values to get the data in the range of
        (approximately) [-1,1]. We divide by the average of the corners only because that
        is the only data we have available in a real setting.

        Why is normalization different between input and output? The input to the ML
        algorithm is just the corners of the subgrid, which is the only data we "know".
        If we were to subtract the corners of a bilinear interpolation from the corners
        like we did for the output array then the values at the corners will be zero,
        which is not useful... So instead of subtracting the corner values of a bilinear
        interpolation we subtract the average of the corners. We then do the same thing
        of dividing by the absolute maximum of this difference, which gets the values
        in the range [-1,1].

'''


import copy
import logging
import numpy as np

import sys
sys.path.append('..')
import constants
from comparison_methods import interpolation

##################
# Aggregation and Wrapper Functions
##################
def Denormalize_arr(arr, norm_data, res, output):
    '''Normalizes a list of values indicated in arr

    arr (np.ndarray)
        - array to be denormalized

    norm_data (np.ndarray)
        - norm[:,[ll,lr,ul,ur,avg,norm]]
            * ll - denormalized lower left corner of the grid
            * lr - denormalized lower right corner of the grid
            * ul - denormalized upper left corner of the grid
            * ur - denormalized upper right corner of the grid
            * avg - average value of the residual
            * norm - range of the residual
    res (int)
        - Resolution
    output (bool)
        - Set to True if you want to denormalize an output array
        - Set to False if you want to denormalize an input array
    '''
    if len(arr.shape) == 1:
        return denormalize(arr,avg,norm,output)

    num_samples = arr.shape[0]
    ret = np.zeros(shape=arr.shape)
    for i in range(num_samples):
        ret[i,:] = denormalize(arr[i,:], norm_data[i], res, output)
    return ret

def Normalize_arr(arr, norm_data, res, output):
    '''Normalizes a list of values indicated in arr

    The first dimension indexes the different samples and the second dimension is the
    sample.

    '''
    if len(arr.shape) == 1:
        return normalize(arr,avg,norm,output)

    num_samples = arr.shape[0]
    ret = np.zeros(shape=arr.shape)
    for i in range(num_samples):
        ret[i,:] = normalize(arr[i,:], norm_data[i], res, output)
    return ret

def InterpolationErrorRegression(
    src,
    func,
    cost,
    output_size = None,
    use_corners = False):
    '''Returns the DENORMALIZED interpolation error for each sample.

    IF YOU DO NOT WANT THE ORIGINAL `src` OBJECT CHANGED, deep copy the original before.
    ---------
    args
    ---------
    src (dataProcessing.wrappers.InputDataWrapper)
        - Contains the information necessary

    func (callable function)
        - Interpolation function

    cost (callable function, returns a scalar)
        - Error function. Returns a scalar

    output_size (int or None)
        - Specific size of the output
        - If None, defaults to the size for interpoation (1)

    use_corners (bool or None)
        - If True, uses corner indecies of y_true instead of src.X
        - This is useful if you have the input (src.X) from multiple different sources
          (e.g. temperature and salinity)
    ---------
    output
    ---------
    (dataProcessing.wrappers.InputDataWrapper)
        - Replaces the truth value for src with the interpolation error array

    Example:
        For bilinear MSE interpolation, run:
            out = InterpolationErrorRegression(
                src=src,
                func=comparison_methods.interpolation.bilinear,
                cost=metrics.MSE,
                output_size = src.res ** 2)
    '''
    return _InterpolationError(src = src,
                               func = func,
                               cost = cost,
                               threshold = None,
                               output_size = output_size,
                               use_corners = use_corners)

def InterpolationErrorClassification(
    src,
    func,
    cost,
    threshold,
    use_corners = False):
    '''Returns a 2-dim one hot vector indicating the the sample's MSE is above or below
    the DENORMALIZED threshold indicated by `threshold`.

    Denormalized threshold meaning the error that you would get if you took the
    error of the interpolation when the corners are NOT normalized.

    ---------
    args
    ---------
    src (dataProcessing.wrappers.InputDataWrapper or np.ndarray)
        - Contains the information necessary
        - If src is a np.ndarray, `res` must also be provided

    cost (callable function, returns a scalar)
        - Error function. Returns a scalar

    threshold (float)
        - The threshold value to compare the output to
        - If val >= threshold, value is constants.ONE_HOT_GREATER_THAN
        - else, constants.ONE_HOT_LESS_THAN

    use_corners (bool or None)
        - If True, uses corner indecies of y_true instead of src.X
        - This is useful if you have the input (src.X) from multiple different sources
          (e.g. temperature and salinity)
    ---------
    output
    ---------
    (dataProcessing.wrappers.InputDataWrapper)
        - Replaces the truth value for src with the interpolation one-hot error array

    Example:
        For bilinear MSE interpolation where you want to cutoff to be 0.12, run:
            out = InterpolationErrorRegression(
                src=src,
                func=comparison_methods.interpolation.bilinear,
                cost=metrics.MSE,
                threshold=0.12,
                output_size = 2)
    '''
    return _InterpolationError(src = src,
                               func = func,
                               cost = cost,
                               threshold = threshold,
                               use_corners = use_corners)

##################
# Core Functions
##################
def denormalize(arr,norm_data,res,output):
    '''
    Denormalize a single array

    -----------
    args
    -----------
    arr (np.ndarray)
        * Array to be denormalized
    - norm_data[:,[ll,lr,ul,ur,avg,norm]]
        * ll - denormalized lower left corner of the grid
        * lr - denormalized lower right corner of the grid
        * ul - denormalized upper left corner of the grid
        * ur - denormalized upper right corner of the grid
        * avg - average value of the residual
        * norm - range of the residual
    - res (int)
        * Resolution
    - output (bool)
        * True if it is an output array (description at top)
        * False if it is an input array
    '''
    if output:
        avg = interpolate_grid(
            input_grid = norm_data[0:4],
            res = res,
            interp_func = interpolation.bilinear)
    else:
        avg = norm_data[-2]
    # return (arr * norm_data[-1] + norm_data[-2]) + bil
    return (arr * norm_data[-1]) + avg

def normalize(arr, norm_data, res):
    '''
    Normalize a single array

    -----------
    args
    -----------
    arr (np.ndarray)
        * Array to be denormalized
    - norm_data[:,[ll,lr,ul,ur,avg,norm]]
        * ll - denormalized lower left corner of the grid
        * lr - denormalized lower right corner of the grid
        * ul - denormalized upper left corner of the grid
        * ur - denormalized upper right corner of the grid
        * avg - average value of the residual
        * norm - range of the residual
    - res (int)
        * Resolution
    - output (bool)
        * True if it is an output array (description at top)
        * False if it is an input array
    '''
    if output:
        avg = interpolate_grid(
            input_grid = norm_data[0:4],
            res = res,
            interp_func = interpolation.bilinear)
    else:
        avg = norm_data[-2]
    # return (arr - bil - norm_data[-2])/norm_data[-1]
    return (arr - avg)/norm_data[-1]

def _InterpolationError(
    src,
    func,
    cost,
    threshold,
    output_size = None,
    use_corners = False):
    '''Description of the function can be seen in `InterpoaltionErrorRegression` or
    `InterpoaltionErrorClassification`.
    '''

    if threshold is None:
        # Just the cost
        if output_size is None:
            out = np.zeros(
                shape=(len(src),constants.DEFAULT_TRANS_SIZE_INTERP))
        else:
            out = np.zeros(shape=(len(src),output_size))
    else:
        # Two categories for the classification (greater or less than)
        out = np.zeros(shape=(len(src), constants.DEFAULT_TRANS_SIZE_CLASS))

    X = None
    if use_corners:
        X = src.y_true[:,[0, src.res - 1, src.res * (src.res - 1), src.res ** 2 - 1]]
    else:
        X = src.X

    # Get error and set to output
    num_samples = len(src)
    for i in range(num_samples):
        if i % 50000 == 0:
            logging.info('{}/{}'.format(i,num_samples))
        interpolated_grid = interpolate_grid(X[i,:], src.res, func)
        error = cost(interpolated_grid - src.y_true[i,:])

        if threshold == None:
            out[i,:] = error

        else:
            if error >= threshold:
                out[i,:] = constants.ONE_HOT_GREATER_THAN.copy()
            else:
                out[i,:] = constants.ONE_HOT_LESS_THAN.copy()

    src.y_true = out
    return src

def makeBicubicArrays(src):
    '''Transforms the src `InputDataWrapper` input array to have a 4x4 grid for each subgrid
    instead of the regular 2x2 grid (corners). This is necessary for the bicubic interpolation.

    Finds the adjacent edges using the loc_to_idx dictionary and assigns the specific
    corners.

    If a subgrid does not have all of the necessary surrounding subgrids (for example it is
    a boundary case), it will delete the subgrid from the src file.

    Assumes that the input array in src are just the corners of the desired grid.
        i.e. src.input_keys[0] == src.output_key
    If this is not the condition then it throws an error.

    Additionally, you will not be able to normalize the data again after this is called
    so everythiing is denormalized before
    '''
    if (not src.input_keys[0] == src.output_key) or len(src.input_keys) > 1:
        raise ValueError('makeBucubicArrays: src.input_keys is not the same as src.output_key')

    src = copy.deepcopy(src)

    num_samples = src.X.shape[0]
    res = src.res

    # Keeps track of the indecies to delete at the end
    idx_to_delete = []

    # generate a new loc_to_idx dictionary
    nl2i = {}

    # initialize new array
    ret = np.zeros(shape = (num_samples, 16))

    z = 0
    for i in range(src.X.shape[0]):
        # get the location
        # check to see if the surrounding region is valid.
        # If it is valid, then set the appropriate values in the ret array
        # If it is not valid, add the index to be deleted at the end.
        [year,t,ymin,ymax,xmin,xmax] = src.locations[i,:]

        # make the 4 quadrants locations
        llquad = (year, t, ymin - res, ymin, xmin - res, xmin)
        lrquad = (year, t, ymin - res, ymin, xmax, xmax + res)
        ulquad = (year, t, ymax, ymax + res, xmin - res, xmin)
        urquad = (year, t, ymax, ymax + res, xmax, xmax + res)

        # check if the quadrants are there
        if not (llquad in src.loc_to_idx and lrquad in src.loc_to_idx and \
                ulquad in src.loc_to_idx and urquad in src.loc_to_idx):
            idx_to_delete.append(i)

        # Get the input data at the specific index
        else:
            ll = src.X[src.loc_to_idx[llquad]]
            lr = src.X[src.loc_to_idx[lrquad]]
            ul = src.X[src.loc_to_idx[ulquad]]
            ur = src.X[src.loc_to_idx[urquad]]

            # piece together data into the right order
            ret[z,:] = np.array([ll[0], ll[1], lr[0], lr[1],ll[2],ll[3],lr[2],lr[3],
                                 ul[0], ul[1], ur[0], ur[1],ul[2],ul[3],ur[2],ur[3]])

            # set the loc_to_idx dictionary
            nl2i[(year,t,ymin,ymax,xmin,xmax)] = z
            z += 1

    # trim ret array, set src.X to ret
    ret = ret[0:z,:]
    src.X = ret

    # Delete the necessary indecies from the rest of the arrays in the data structure
    src.y_true = np.delete(src.y_true, idx_to_delete, axis = 0)
    src.norm_data = np.delete(src.norm_data, idx_to_delete, axis = 0)
    src.locations = np.delete(src.locations, idx_to_delete, axis = 0)
    src.loc_to_idx = nl2i

    return src

def interpolate_grid(input_grid, res, interp_func, **kwargs):
    '''Applys the interpolation functions below on the desired input_grid. The output
    is a 2 dimensional grid of size `res` x `res`.

    Assumes the input grid has length of 4 (because of the 4 corners) if
    interp_func is either `bilinear`, `nearest_neighbor`, or `idw`. It assumes
    the input grid has length of 16 if the interp_func is `bicubic`.

    Returns the flattened array.

    Example:
        Compute bilinear interpolation on a 6x6 grid with on the corners `corners`:
            interpolate_grid(corners,6,interpolation.bilinear)
    ---------------
    args
    ---------------
    input_grid (numpy array)
        - 1 dimensional array
        - input grid for the interpolation (either length 4 or 16)
    res (int)
        - Size to make the output grid.
    interp_func (function)
        - Interpolation function to apply to each of the locations on the return grid.
        - must have the parameters (input_grid, location), where location is a tupple
          of (x, y) location, and have a scalar as an output.
            * x is the proportion in the x direction
            * y is the proportion in the y direction
    kwargs (dict)
        - Additional arguments to pass to the interpolation function
        - Example:
            * idw could have an extra argument `p`
    '''
    ret = np.zeros(shape=(res,res))
    for x in range(res):
        for y in range(res):
            ret[x,y] = interp_func(input_grid, (x/(res-1),y/(res-1)), **kwargs)
    return ret.flatten()

def collapse_one_hot(arr):
    '''Transforms a one-hot array of matricies into a single dimensional binary array
    1: greater than
    0: less than
    '''
    ret = np.zeros(arr.shape[0])
    for i in range(arr.shape[0]):
        ret[i] = np.argmax(arr[i,:])
    return ret

def one_hotify(arr):
    '''Transforms a binary array to a one hot array
    Uses
    '''

    ret = np.zeros(shape=(len(arr),2))
    for i in range(len(arr)):
        ret[i,int(arr[i])] = 1
    return ret

def mapify(loc_to_idx,arr,year,day,res,classification):
    '''Converts the list of subgrids into a map where each subgrid is in the right
    location

    --------------
    args
    --------------
    loc_to_idx (dict)
        - maps location 6-tuple (year,day,ymin,ymax,xmin,xmax) to the index
          where that subgrid appears in `arr`
    arr (2-dim ndarray)
        - Holds the data to map
        - The first dimension idexes the subgrid
        - The second dimension is a flattened array of the subgrid
    year, day (float)
        - The year and day to index to find the days
    res (int)
        - Resolution that the grid was subsampled at
        - Indicates the way to reshape the flattened subgrid
    '''

    # get max x and max y
    maxx = constants.LON_LEN
    maxy = constants.LAT_LEN
    ret = np.zeros(shape=(maxy,maxx)) * np.nan
    logging.debug('Map size: {}'.format(ret.shape))

    if classification and len(arr.shape) == 2:
        arr = np.argmax(arr, axis = 1)

    # For each index, check if it is in the
    for x in range(maxx):
        for y in range(maxy):
            x_max = x + res
            y_max = y + res

            loc = (float(year),
                   float(day),
                   float(y),
                   float(y_max),
                   float(x),
                   float(x_max))
            if loc in loc_to_idx:
                logging.debug('{} in dict'.format(loc))
                idx = loc_to_idx[loc]
                if classification:
                    ret[int(y):int(y_max), int(x):int(x_max)] = arr[idx]
                else:
                    ret[int(y):int(y_max), int(x):int(x_max)] = np.reshape(arr[idx,:], [res, res])
    # The returned array is upside down, flip the right way
    return np.flipud(ret)
