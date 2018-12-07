'''
Loads default datasets like HYCOM
'''
import logging
import numpy as np
from netCDF4 import Dataset

import constants
import util

def load_hycom(years, keys = None, dtype = None):
	filepath2d = constants.DEFAULT_DP_FILEPATH2D
	filepath3d = constants.DEFAULT_DP_FILEPATH3D
	dtype = constants.DEFAULT_NC_DTYPE
	keys = constants.DEFAULT_NC_KEYS

	return HYCOM(filepath2d=filepath2d,filepath3d=filepath3d,
		years=years,read_imm = True,dtype = dtype,keys=keys)


################
# Classes for raw netCDF data IO.
################

class HYCOM:
    '''A wrapper class to deal with a set of `_HYCOMYear`s.
    Assumes the data has directory format:
        `path/to/data/2D_vars/__year__.nc`
        `path/to/data/3D_vars/__year__.nc`

        example:
            `data/2D_vars/2006.nc`
    '''

    def __init__(self,filepath2d,filepath3d,years,
        read_imm = True,dtype = None,keys=None):
        '''
        Args
        --------------
        filepath_2D (str)
            - Filepath to the folder that has the netcdf files
        filepath_3D (str)
            - Filepath to the folder that has the netcdf files
        savepath (str)
            - Base file location where to save this object.
            - Since object is potentially very big, save as netCDFs
        years (list[str])
            - A list of strings for the different years of data it
              should read
        read_imm (bool)
            - If True, calls read_files from the constructor
        dtype (str)
            - Datatype to read the netCDF files.
            - This is typically `.nc*`
        keys (list of str)
            - Names of the variables to read in
        '''
        self.filepath2d = filepath2d
        self.filepath3d = filepath3d
        self.dtype = dtype
        self.keys = keys
        self.years = years
        self.data = {} # year -> _HYCOMYear. Does not distinguish between 2D and 3D vars
        logging.debug('Creating Object.')
        if read_imm:
            logging.debug('Reading immediately.')
            self.read_files()

    def __getitem__(self,key):
        return self.data[key]

    def read_files(self):
        '''
        Reads in the set of data specified by year and filepath_2D/3D
        Creates dictionaries mapping from year
        '''
        for year in self.years:
            logging.info('Starting to read year {}'.format(year))
            self.data[year] = _HYCOMYear(self.filepath2d,self.filepath3d,year,
                dtype=self.dtype,keys_=self.keys)
        # Set attributes - assume these are same for each year
        year = self.years[0]
        self.lat = self.data[year].lat
        self.lon = self.data[year].lon
        self.total_days = 0
        for _,year in self.data.items():
            self.total_days += year.num_days
        logging.debug('lon: {}, lat: {}, total days: {}'.format(
                len(self.lon), len(self.lat), self.total_days))

class _HYCOMYear:
    '''Wrapper class encompassing with 2D and 3D netcdf data
    from the same year.
    '''
    def __init__(self,filepath2d,filepath3d,year,
        read_imm = True,keys_ = None,dtype = None):
        
        logging.debug('Object Created')
        if keys_ == None:
            keys_ = constants.DEFAULT_NC_KEYS
        if dtype == None:
            dtype = constants.DEFAULT_NC_DTYPE
        self.filepath2d = filepath2d
        self.filepath3d = filepath3d
        self.year = year
        self.keys = keys_
        self.data = {}
        self.dtype = dtype
        logging.debug('\n\tfilepath2d: {}\n\tfilepath3d: {}\n\tyear: {}\n\tkeys: {}'.format(
                self.filepath2d, self.filepath3d,self.year,self.keys))
        if read_imm:
            logging.debug('Reading immediately')
            self.read_data()

    def __getitem__(self,key):
        return self.data[key]

    def read_data(self):
        # Read in data
        logging.info('Reading from 2D data folder')
        t2d = self.read_from_folder(self.filepath2d,['ssh'])
        logging.info('Reading from 3D data folder')
        t3d = self.read_from_folder(self.filepath3d,['temperature','u','v','salinity'])

        # Set data
        self.lat = t2d['lat'] # Assume these are identical for 2d and 3d data
        self.lon = t2d['lon'] # Assume these are identical for 2d and 3d data
        self.num_days = t2d['num_days']
        self.data['ssh']  = t2d['ssh']
        self.data['temp'] = t3d['temperature']
        self.data['u']    = t3d['u']
        self.data['v']    = t3d['v']
        self.data['sal']  = t3d['salinity']
        logging.debug('\n\tlat: {}\n\tlon: {}\n\tnum_days: {}'.format(
                len(self.lat),len(self.lon),self.num_days))

    def read_from_folder(self,folder,keys):
        loc = folder + self.year + self.dtype
        logging.debug('Reading from location: {}'.format(loc))
        f = Dataset(loc)
        data = {}
        logging.debug('Reading from folder `{}`, keys: `{}`'.format(folder,keys))

        # Get attributes - Longitude and Latitude
        data['lat'] = f.variables['Latitude'][:].copy()
        data['lon'] = f.variables['Longitude'][:].copy()
        data['num_days'] = len(f.variables['MT'])

        for key in keys:
            data[key] = np.squeeze(f.variables[key][:].copy())
            logging.debug('key: {}, shape: {}'.format(key, data[key].shape))
        logging.debug('Exiting normally')
        f.close()
        return data

