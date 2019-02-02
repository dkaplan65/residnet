'''
Author: David Kaplan
Advisor: Stephen Penny

Since the amount of memory that a typical `DataPreprocessing` object is >> 4Gb,
we need to make a custom saving method using netCDF files and separate the
attributes into a different object.
'''
import numpy as np
import time
import math
import os
import pickle
import logging
from netCDF4 import Dataset

from .transforms import Denormalize_arr as Denormalize
from .transforms import Normalize_arr as Normalize

import sys
sys.path.append('..')
from residnet.util import *
from residnet.constants import *
from residnet.datasets import HYCOM

'''
Future Versions:
    - Zip files for saving
'''


class Settings:
    '''
    Acts as a container of the attributes for the DataPreprocessing class.
    Necessary to separate them because the saving method is more complicated.
    '''
    def __init__(self, name = None, res_in = None, savepath = None,
        filepath2d = None, filepath3d = None, years = None, dataset = None,
        denorm_local = None, keys = None, num_days = None):
        '''Once core arrays (subgrids, locations, norm_data) are built, do not
        manipulate them as it fill make `loc_to_idx` invalid
        -------------
        Args
        -------------
        savepath (str)
            - location to store the object
        res_in (int)
            - Number of pixels per subgrid
        dataset (str)
            - What dataset to load from
            - example, 'hycom'
        filepath2d (str)
            - Location where the HYCOM netcdf 2D data is stored
        filepath3d (str)
            - Location where the HYCOM netcdf 3D data is stored
        years (list[str]), ex: ['2006', '2007']
            - A list of years to parse to get the data
        denorm_local (bool)
            - If True, normalizes a subgrid by the absolute max value of the
              difference in the corners of the subgrid.
            - If False, normalizes a subgrid by the absolute max value over the
              entire dataset.
        keys (list[str]), ex: ['ssh','temp', etc.]
            - A list of the keys to parse from the data
        num_days (int or None)
            - Number of days to parse when loading in the data
        settings (Settings or None)
            - A settings object that overwrites the previous attributes
        '''

        # Set to default if it was not set during initialization
        if name == None:
            name = DEFAULT_DP_NAME
        if res_in == None:
            res_in = DEFAULT_DP_RES_IN
        if filepath2d == None:
            filepath2d = DEFAULT_DP_FILEPATH2D
        if filepath3d == None:
            filepath3d = DEFAULT_DP_FILEPATH3D
        if dataset == None:
            dataset = DEFAULT_DP_DATASET
        if years == None:
            years = DEFAULT_DP_YEARS
        if denorm_local == None:
            denorm_local = DEFAULT_DP_DENORM_LOCAL
        if keys == None:
            keys = DEFAULT_DP_KEYS
        if num_days == None:
            num_days = DEFAULT_DP_NUM_DAYS
        if savepath == None:
            savepath = 'output/datapreprocessing/{}_denormLocal{}_res{}/'.format(name,denorm_local,res_in)

        self.savepath = savepath
        self.res_in = res_in
        self.filepath2d = filepath2d
        self.filepath3d = filepath3d
        self.dataset = dataset
        self.years = years
        self.denorm_local = denorm_local
        self.keys = keys
        self.num_days = num_days

        # Non-passed in attributes
        self.parsed = False # If the data has been parsed or not
        self.num_samples = np.nan # Total number of samples parsed

        # Make sure paths are in the right format
        if self.filepath2d[-1] != '/':
            self.filepath2d += '/'
        if self.filepath3d[-1] != '/':
            self.filepath3d += '/'
        if self.savepath[-1] != '/':
            self.savepath += '/'

        check_savepath_valid(self.savepath)

    def __str__(self):
        s = 'Settings Object:\n'
        s += '\tsavepath: {}\n'.format(self.savepath)
        s += '\tResolution: {}\n'.format(self.res_in)
        s += '\tfilepath2d: {}\n'.format(self.filepath2d)
        s += '\tfilepath3d: {}\n'.format(self.filepath3d)
        s += '\tyears: {}\n'.format(self.years)
        s += '\tkeys: {}\n'.format(self.keys)
        s += '\tnumber of days: {}\n'.format(self.num_days)
        s += '\tnumber of samples: {}\n'.format(self.num_samples)
        s += '\tdenorm_local? {}\n'.format(self.denorm_local)
        s += '\tparsed? {}\n'.format(self.parsed)

        return s

class DataPreprocessing:
    '''Preprocesses raw data into subgrids that we can do interpolation and
    and regression over.
    '''

    def __init__(self, *args, **kwargs):

        if 'settings' not in kwargs:
            self.settings = Settings(*args, **kwargs)
        else:
            if not isinstance(kwargs['settings'], Settings):
                raise DataPreProcessingError('settings must be a `Settings` object')
            self.settings = kwargs['settings']
        self.subgrids = {}
        self.locations = {} #key -> [year, day, lat_min, lat_max, lon_min, lon_max]
        self.norm_data = {} #key -> [avg, range]

    def __len__(self):
        return self.settings.num_samples

    def __str__(self):
        s = str(self.settings)
        s += 'num samples: {}\n'.format(self.settings.num_samples)
        return s

    def _valid(self,arr):
        '''Determines if the array is valid.
        checks if there are any nans or any weird numbers out of range
        '''
        return np.all(np.abs(np.ma.filled(arr.astype(int),99999999)) < 1000)

    def parse_data(self):
        '''Converts data from arrays loaded from netcdf's to arrays of individual subgrids.

        First, it reads the raw data from netcdf.
        Second, it parses the data into the internal data structure.
        Lastly, it sets normalization factors
        '''
        if self.settings.parsed:
            logging.info('Already parsed')
            return
        logging.debug('Reading in raw data.')

        # Load the dataset
        if self.settings.dataset == 'hycom':
            raw_data = HYCOM(filepath2d = self.settings.filepath2d,
                filepath3d = self.settings.filepath3d,
                years = self.settings.years, keys = self.settings.keys)
        else:
            raise DataProcessingError('DataPreprocessing.parse: dataset `{}` not valid'.format(
                self.settings.dataset))

        logging.debug('Initializing variables.')
        res_in = self.settings.res_in
        corner_idxs = gen_corner_idxs(res_in)
        # placeholder for current index
        z = 0
        # Loose upper bound of how many subgrids will be parsed in total
        # Will trim at the end
        num_subgrids = len(self.settings.years) * 366 * DEFAULT_DP_SPD[self.settings.res_in]
        # Initialize the arrays so that we are not constantly appending arrays,
        # instead we are just setting values
        for ele in self.settings.keys:
            self.subgrids[ele]  = np.zeros(shape=(num_subgrids, res_in * res_in))
            # [year, day, min_y, max_y, min_x, max_x]
            self.locations[ele] = np.zeros(shape=(num_subgrids, 6))
            # [ll, lr, ul, ur, resid_avg, resid_range]
            self.norm_data[ele] = np.zeros(shape=(num_subgrids, 6))

        # Iterate over every position and parse the raw data into the internal data structure
        # Throw an exception once the total number of days has been read.
        n_accepted = 0
        last_time = time.time()
        try:
            num_days = 0
            # Iterate over year
            for year in raw_data.years:
                logging.info('\n\nYear {}'.format(year))
                # Iterate over day
                for t in range(raw_data[year].num_days):
                    time_ = time.time() - last_time
                    logging.info('{}/{}, z {}, n_accpeted: {}, {:.5}s'.format(
                        t+1, raw_data[year].num_days,z,n_accepted,time_))
                    n_accepted = 0
                    last_time = time.time()
                    if num_days >= self.settings.num_days:
                        raise MaxDays('Maximum number of days parsed.')
                    num_days += 1
                    #Iterate over latitudes
                    for y_ in range(math.floor(len(raw_data.lat)/res_in)):
                        y = res_in * y_
                        # Iterate over longitudes
                        for x_ in range(math.floor(len(raw_data.lon)/res_in)):
                            x = res_in * x_
                            # Check if the subgrid is valid.
                            if self._valid(raw_data[year]['temp'][t,y : y + res_in,x : x + res_in].flatten()):
                                # Append flattened arrays for each key
                                n_accepted += 1
                                for key in self.settings.keys:
                                    # Get corners, calculate statistics, set values
                                    self.locations[key][z,:] = [year, t, y, y + res_in, x, x + res_in]
                                    self.subgrids[key][z,:] = raw_data.data[year][key][
                                        t, y : y + res_in, x : x + res_in].flatten()
                                    corners = self.subgrids[key][z, corner_idxs]
                                    avg = np.mean(corners)
                                    norm = np.max(np.absolute(self.subgrids[key][z,corner_idxs] - avg))
                                    self.norm_data[key][z,:] = np.array(np.append(corners, [avg, norm]))

                                    # log
                                    # logging.debug('subgrids:\n{}\n'.format(self.subgrids[key][z,:]))
                                    # logging.debug('locations:\n{}\n'.format(self.locations[key][z,:]))
                                    # logging.debug('corner_vals:\n{}\n'.format(corners))
                                    # logging.debug('norm_data:\n{}\n'.format(self.norm_data[key][z,:]))
                                z += 1
                            # else:
                            #     print('\n\ny: {}, x: {}'.format(x,y))
                            #     print('not accepted')
                            #     print('print(arr:\n{}'.format(raw_data[year]['temp'][t,
                            #         y : y + res_in,x : x + res_in].reshape(6,6)))
        except MaxDays:
            logging.debug('Total number of days read: num_days: {}, year {}, day: {}'.format(num_days,year,t))
        # Trim excess from arrays
        for key in self.settings.keys:
            self.subgrids[key]  = self.subgrids[key][0:z, :]
            self.locations[key] = self.locations[key][0:z, :]
            self.norm_data[key] = self.norm_data[key][0:z, :]

        # Setting the normalization constant to be the absolute maximum difference
        # Between the average value and the corners.
        # For each key, get the max norm, set it, and then normalize subgrids if necessary
        if not self.settings.denorm_local:
            logging.info('Denorm global')
            for key in self.settings.keys:
                logging.info('key: {}'.format(key))
                max_range = np.max(self.norm_data[key][:,-1])
                logging.info('max_range: {}'.format(max_range))
                self.norm_data[key][:,-1] = max_range

        # Set global variables
        self.settings.parsed = True
        self.settings.num_samples = z
        logging.info('Total number of samples: {}'.format(z))

    def split_data_idxs(self,division_format, **kwargs):
        '''Creates index arrays for testing, validation, and testing sets.
        `division_format` indicates the type of division that is done.
        kwargs is auxiliary information needed for the division

        Valid division formats: necessary in kwargs
            - `k-fold`: `k` (int), `randomize` (bool)
                * Creates `k` even sets of indices
                * If randomize is True, it will also randomize the base set
            - `year-fold`: None
                * Makes each year a separate set
            - `split`: `split_dict` (dict key -> floats), `randomize` (bool)
                * Splits the data into the proportion indicated by the values of the dictionary.
                * All the floats must be positive, greater than zero.
                * If `randomize` is True, it also shuffles the indices
                # Example split:
                    - split_dict = {'training': 0.7, 'validation': 0.1, 'testing': 0.2}
                    - 70% to training, 10% to validation, 20% to testing

        Output:
            d (dict)
                - key -> idx arrays
                - if the division format is `year-fold`, the keys are the years (strings)
                - Otherwise they are just ints corresponding to each fold

        '''
        logging.debug('division_format: {}'.format(division_format))
        logging.debug('kwargs: {}'.format(kwargs))
        valid_formats = ['k-fold','year-fold','split']
        if not division_format in valid_formats:
            raise DataProcessingError('`{}` is an invalid split data format'.format(division_format))
        idxs = np.arange(self.settings.num_samples)
        d = {} # return dictionary

        if division_format == 'year-fold':
            # Get a random key to look at
            key_ = self.settings.keys[0]
            for i in range(len(self.locations[key_])):
                y_ = self.locations[key_][i,0]
                if y_ not in d:
                    d[y_] = []
                d[y_].append(i)
            return d

        elif division_format == 'k-fold':
            # Check to see if kwargs has k
            if len(kwargs) != 2 or 'k' not in kwargs or 'randomize' not in kwargs:
                raise DataProcessingError('Improper arguments for k-fold. Arguments: {}'.format(kwargs))
            # This is just a special case for random, where the
            # length of the array is k long and they are all equal
            k = kwargs['k']
            randomize = kwargs['randomize']
            split = {}
            for i in range(k):
                split[i] = 1/k

        elif division_format == 'split':
            if len(kwargs) != 2 or 'split_dict' not in kwargs or 'randomize' not in kwargs:
                raise DataProcessingError('Improper arguments for split. Arguments: {}'.format(kwargs))
            split = kwargs['split_dict']
            randomize = kwargs['randomize']

            if (np.sum(list(split.values())) > 1) or (not np.all(i > 0 for i in list(split.values()))):
                raise DataProcessingError('Improper arguments for Random. Improper values for lst')

        if randomize:
            logging.debug('Shuffling base array indices')
            np.random.shuffle(idxs)

        ss = {}

        # Get the start and end index for each subset
        # Set the rest of the examples to the last set
        prev_idx = 0
        keys = list(split.keys())
        for i in range(len(keys)):
            key = keys[i]
            if i - 1 == len(keys):
                end = len(keys)
            else:
                end = int(prev_idx + np.floor(self.settings.num_samples * split[key]))
            ss[key] = (prev_idx, end)
            logging.debug('{} start idx: {}, end idx: {}'.format(
                        key,prev_idx,end))
            prev_idx = end
        for key,val in ss.items():
            start = ss[key][0]
            end = ss[key][1]
            d[key] = idxs[start: end]

        return d

    def make_array(self,idxs = None,input_keys = None,
        output_key = None, norm_input = True):
        '''Concatenates the corners together to form the input.
        Returns the truth of the input and the denormalization information
        as well.

        Only makes the array for the designated indices passed in (idx).
        If `idxs` are not set, make everything.
        Creates a dictionary that maps the locations of the subgrids to the
        place where it is put in the return datastructure.

        idxs (int, list(int), None)
            - if None, make everything
            - if type(int), make everything up to index `idxs`
            - if it is a list of ints, use these as the indices to draw them out
        input_keys (list(str), Optional)
            - List of input keys to put in the input
            - If nothing is passed in, everything is made
        output_key (str)
            - Output truth.
            - If nothing is passed in, 'temp' is used as the output
        norm_input (bool)
            - If True, it normalizes the corners of the input
            - If False, no normalization of the input is done
            - Default is True
        '''
        if not self.settings.parsed:
            raise DataProcessingError('make_array: Data is not parsed yet.')

        if idxs is None:
            # Make everything
            idxs = np.arange(self.settings.num_samples)

        if type(idxs) == int:
            idxs = np.arange(idxs)

        # Defaults to all the keys
        if input_keys == None:
            input_keys = self.settings.keys
        if output_key == None:
            output_key = DEFAULT_DP_OUTPUT_KEY

        # (str,int,int,int,int,int) -> int
        loc_to_idx = {}
        logging.debug('num samples: {}, input_keys: {}, output_key: {}'.format(
                    len(idxs), input_keys, output_key))
        num_samples = len(idxs)
        corner_idxs = gen_corner_idxs(self.settings.res_in)

        # Multiplied by 4 because there are 4 corners
        X = np.zeros(shape = (num_samples, len(input_keys) * 4))

        output_array = np.zeros(shape = (num_samples, self.settings.res_in ** 2))
        norm_data = np.zeros(shape = (num_samples, NORM_LENGTH))
        locations = np.zeros(shape = (num_samples, LOCATIONS_LENGTH))

        for i in range(num_samples):
            if i % 50000 == 0:
                logging.info('{}/{}'.format(i,num_samples))
            idx = idxs[i]
            arr = np.array([])
            for key in input_keys:
                temp_arr = np.array(self.subgrids[key][idx,corner_idxs])
                if norm_input:
                    temp_arr = (temp_arr - self.norm_data[key][idx,-2])/ \
                        self.norm_data[key][idx,-1]
                arr = np.append(arr, temp_arr)
            X[i,:] = arr.copy()
            output_array[i,:] = self.subgrids[output_key][idx,:]
            norm_data[i,:] = self.norm_data[output_key][idx,:]
            locations[i,:] = self.locations[output_key][idx,:]
            key_ = tuple(locations[i,:].tolist())
            loc_to_idx[key_] = i

        return DataWrapper(
            X = X,
            input_keys = input_keys,
            y_true = output_array,
            output_key = output_key,
            norm_data = norm_data,
            locations = locations,
            loc_to_idx = loc_to_idx,
            res = self.settings.res_in)

    def save(self, savepath = None):
        '''Save the object. Use netcdf files because the object is potentially
        very large (too large for pickle).
        Store arrays in netcdf files, pickle Settings object.
        -----------
        Args
        -----------
        savepath (None or str)
            - If savepath != None, overwrite self.savepath
        '''
        if savepath != None:
            self.settings.savepath = savepath
            check_savepath_valid(self.settings.savepath)
            if self.settings.savepath[-1] != '/':
                self.settings.savepath += '/'

        # Make all paths that youre going to save into
        basepath = self.settings.savepath
        subgridspath  = basepath + 'subgrids.nc'
        locationspath = basepath + 'locations.nc'
        norm_datapath = basepath + 'norm_data.nc'
        settingspath  = basepath + 'settings.pkl'

        check_savepath_valid(subgridspath)
        check_savepath_valid(locationspath)
        check_savepath_valid(norm_datapath)
        check_savepath_valid(settingspath)

        # Save the data - pickle the settings object
        self._save_data_obj_netcdf(subgridspath, self.subgrids)
        self._save_data_obj_netcdf(locationspath, self.locations)
        self._save_data_obj_netcdf(norm_datapath, self.norm_data)

        saveobj(self.settings, settingspath)

    @classmethod
    def load(cls,loadpath):
        '''Load object from the passed in load path
        '''
        if not os.path.exists(loadpath):
            raise DataProcessingError('Data.load: {} is not a valid path'.format(loadpath))
        if loadpath[-1] != '/':
            loadpath += '/'

        # Check if the necessary files exist
        subgridspath  = loadpath + 'subgrids.nc'
        locationspath = loadpath + 'locations.nc'
        norm_datapath = loadpath + 'norm_data.nc'
        settingspath  = loadpath + 'settings.pkl'

        logging.debug('Loading settings')
        ret = cls(settings = loadobj(settingspath))
        logging.debug('Loading subgrids')
        ret.subgrids  = ret._load_data_obj_netcdf(subgridspath)
        logging.debug('Loading locations')
        ret.locations = ret._load_data_obj_netcdf(locationspath)
        logging.debug('Loading norm_data')
        ret.norm_data = ret._load_data_obj_netcdf(norm_datapath)
        return ret

    def _load_data_obj_netcdf(self,path):
        f = Dataset(path, 'r', DEFAULT_DP_NETCDF_FORMAT)
        # Get dimensions
        num_samples = len(f.dimensions['num_samples'])
        _dim = len(f.dimensions['_dim'])
        d = {}
        for key in self.settings.keys:
            d[key] = f.variables[key][:].copy()
        f.close()
        return d

    def _save_data_obj_netcdf(self, path, d):
        '''Save data in netcdf
        `d` is a ictionary of multidimensional arrays of the same size
        '''
        data = Dataset(path, 'w', format = DEFAULT_DP_NETCDF_FORMAT)
        (num_samples, _dim) = d[self.settings.keys[0]].shape

        # Create global variables
        data.raw_data_source = 'HYCOM GOMl0.04 experiment 20.1, Naval Research Laboratory'
        data.save_loc = path[0:-2]
        data.geospatial_lat_min = '18.0916 degrees'
        data.geospatial_lat_max = '31.9606 degrees'
        data.geospatial_lon_min = '-98 degrees'
        data.geospatial_lon_max = '-76.4 degrees'

        # Create dimensions
        data.createDimension('num_samples', num_samples)
        data.createDimension('_dim', _dim)

        # Create variables
        save_dict = {}
        for key in self.settings.keys:
            save_dict[key] = data.createVariable(key, np.float32, ('num_samples', '_dim'))

        # Set variables
        for key in self.settings.keys:
            save_dict[key][:] = d[key]

        # Close file
        data.close()

################
# Array Wrapper Classes
################
class DataWrapper(IOClass):
    '''Wrapper class to hold a set of input and truth arrays.
    Includes metadata about the arrays.

    Provides functions for manipulating the arrays and for retreiving batches
    of data (Used for training in comparison_methods.neural_network.TFWrapper)
    '''

    def __init__(self, X, input_keys, y_true, output_key,
        norm_data, locations, loc_to_idx, res):
        '''
        -----------
        args
        -----------
        X (numpy array)
            - The array used as the input to the desired function. This function could be an
              interpolation function or some other kind.
            - In this implementation we can assume that this are the corners of a subgrid.
            - The first dimension is how you index each of the samples, and the second dimension
              are the actual samples.
        input_keys (list or str)
            - The keys that the input corresponds to.
        y_true (numpy array)
            - The desired output of each of the samples.
            - The length of the first dimension of the `X` and `y_true` are identical.
        output_key (str)
            - The key to which the output corresponds to
        norm_data (numpy array)
            - The normalization data used to normalize and denormalize the input or output arrays
        locations (numpy array)
            - Indicates the location spatially and temporally where this subgrid belongs
        loc_to_idx (dict [6-tuple -> int])
            - maps the location of a subgrid as a tuple to an integer
        normalized (bool)
            - If true, it means that the output is normalized
        res_in (int)
            - Resolution that the corners are derived from the base.
        '''
        IOClass.__init__(self)
        self.input_keys = input_keys
        self.output_key = output_key
        self.X = X
        self.y_true = y_true
        self.norm_data = norm_data
        self.locations = locations
        self.loc_to_idx = loc_to_idx
        self.y_normalized = False
        self.X_normalized = False
        self.res = res
        self.shape = {'X': self.X.shape, 'y_true': self.y_true.shape}
        self.input_size = self.X.shape[1]
        self.output_size = self.y_true.shape[1]

        # Set to True once batching initialized
        self.batching_initialized = False

    def __len__(self):
        return self.X.shape[0]

    def transform_X(self, func):
        '''
        DOES NOT SET THE INPUT ARRAY TO WHAT IS RETURNED. MUST DO THAT MANUALLY.
        '''
        return self._transform(func, self.X)

    def transform_y_true(self, func):
        '''
        DOES NOT SET THE OUTPUT ARRAY TO WHAT IS RETURNED. MUST DO THAT MANUALLY.
        '''
        return self._transform(func, self.y_true)

    def denormalize(self, output = None):
        '''
        output (bool or None)
            - If True, denormalize y_true
            - If False, denormalize X
            - If None, denormalize both
        '''
        if (output is None or output is True) and self.y_normalized:
            self.y_true =  Denormalize(
                arr = self.y_true, norm_data = self.norm_data, res = self.res,
                output = True)
            self.y_normalized = False

        if (output is None or output is False) and self.X_normalized:
            self.X =  Denormalize(
                arr = self.X, norm_data = self.norm_data, res = self.res,
                output = False)
            self.X_normalized = False
        return self

    def normalize(self, output = None):
        '''
        output (bool or None)
            - If True, denormalize y_true
            - If False, denormalize X
            - If None, denormalize both
        '''
        if (output is None or output is True) and not self.y_normalized:
            self.y_true =  Normalize(
                arr = self.y_true, norm_data = self.norm_data, res = self.res,
                output = True)
            self.y_normalized = True

        if (output is None or output is False) and not self.X_normalized:
            self.X =  Normalize(
                arr = self.X, norm_data = self.norm_data, res = self.res,
                output = False)
            self.X_normalized = True
        return self

    def _transform(self, func, arr):
        '''
        Applys an arbitrary transform to the input array for each sample.

        This function is called from `transform_X` and `transform_y_true`.
        ---------
        args
        ---------
        func (function with output)
            - A function that takes in a flattened vector and outputs a value,
              which is then set as the output vector.
            - Assumes the dimension of the output is always the same.
        array (numpy array)
            - Array that the transformation is being done on
        '''
        return np.apply_along_axis(func,1,arr)

    def delete_idxs(self, idxs):
        '''OPPOSITE OF `keep_idxs`

        Deletes the indices that are specified in `idxs`.

        This is useful when you want to divide up a set of data into subsets
        based on some preprocessing like clustering.

        idxs (list(ints))
        '''
        if np.max(idxs) > self.X.shape[0] - 1:
            logging.critical('max_idx: {}, len: {}'.format(np.max(idxs), self.X.shape[0] - 1))
            raise DataProcessingError('InputDataWrapper.delete_idxs: Max idx greater than length')

        # Invert the idxs and call keep_idxs
        idxs = list(set(np.arange(self.X.shape[0]) - set(idxs)))
        self.keep_idxs(idxs)
        return self

    def keep_idxs(self, idxs):
        '''OPPOSITE OF `delete_idxs`

        Keeps only the indices that are specified in `idxs`.

        This is useful when you want to divide up a set of data into subsets
        based on some preprocessing like clustering.

        idxs (list(ints))
        '''
        if np.max(idxs) > self.X.shape[0] - 1:
            logging.critical('max_idx: {}, len: {}'.format(np.max(idxs), self.X.shape[0] - 1))
            raise DataProcessingError('InputDataWrapper.keep_idxs: Max idx greater than length')

        self.X = self.X[idxs,:]
        self.y_true = self.y_true[idxs,:]
        self.norm_data = self.norm_data[idxs,:]
        self.locations = self.locations[idxs,:]
        self.shape = {'X': self.X.shape, 'y_true': self.y_true.shape}

        # redo `self.loc_to_idx`
        self.loc_to_idx = {}
        for i in range(self.locations.shape[0]):
            self.loc_to_idx[tuple(self.locations[i,:])] = i
        return self

    ##############
    # Batch training functionality
    ##############
    def initialize_batching(self,batch_size):
        '''
        Set up function to allow batch retrieval of data
        We cannot shuffle the original dataset because it would
        mess up the `loc_to_idx` dictionary. Instead, we shuffle a
        list of indices that we use to index the main arrays.

        All functions are applied to an array of indices.
            - shuffle
                * shuffle the index array
            - get_batch
                * get the next set of data based on the index array
        '''
        self.batching_initialized = True
        self.batch_size = batch_size
        self.base_idx = 0
        self.idxs = np.arange(self.X.shape[0])
        np.random.shuffle(self.idxs)
        return self

    def shuffle(self):
        if not self.batching_initialized:
            raise DataProcessingError('Batching not initialized. Call `initialize_batching`')
        np.random.shuffle(self.idxs)
        return self

    def get_batch(self):
        '''Get the next batch of data
        '''
        if not self.batching_initialized:
            raise DataProcessingError('Batching not initialized. Call `initialize_batching`')
        Xs = self.X[self.idxs[self.base_idx: self.base_idx + self.batch_size], :]
        ys = self.y_true[self.idxs[self.base_idx: self.base_idx + self.batch_size], :]
        self.base_idx += self.batch_size
        return Xs,ys

    def restart_batching(self):
        if not self.batching_initialized:
            raise DataProcessingError('Batching not initialized. Call `initialize_batching`')
        self.base_idx = 0
        self.shuffle()
        return self

class DataProcessingError(Exception):
    pass

class MaxDays(Exception):
    pass
