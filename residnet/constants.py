'''
Author: David Kaplan
Advisor: Stephen Penny

Default and reference values
'''
from numpy import array
from pickle import HIGHEST_PROTOCOL


# Global variables
ONE_HOT_LESS_THAN = array([0,1])
ONE_HOT_GREATER_THAN = array([1,0])
LAT_LEN = 385
LON_LEN = 541
LOGGING_FORMAT = '%(levelname)s: %(module)s.%(funcName)s: %(message)s'
NORM_LENGTH = 6 # 4 corners, avg, local range
LOCATIONS_LENGTH = 6 # year, t, y_min, y_max, x_min, x_max

# Default values for util
DEFAULT_UTIL_PICKLE_PROTOCOL = HIGHEST_PROTOCOL
MAX_BYTES = 2**31 - 1

# Default values for data_processing.wrappers.DataPreprocessing
DEFAULT_DP_NAME = 'data'
DEFAULT_DP_RES_IN = 6
DEFAULT_DP_FILEPATH2D = 'input_data/hycom/2D_vars/'
DEFAULT_DP_FILEPATH3D = 'input_data/hycom/3D_vars/'
DEFAULT_DP_DATASET = 'hycom'
DEFAULT_DP_YEARS = ['2005', '2006']
DEFAULT_DP_DENORM_LOCAL = False
DEFAULT_DP_KEYS = ['temp','ssh','u','v','sal']
DEFAULT_DP_NUM_DAYS = float('inf')
DEFAULT_DP_OUTPUT_KEY = 'temp'
DEFAULT_DP_NETCDF_FORMAT = 'NETCDF4_CLASSIC'
# Set the number of subgrids per day. Dependent on `DEFAULT_DP_RES_IN`
DEFAULT_DP_SPD = {
	50: 35, #2 degree
	25: 200, #1 degree
	12: 950, #1/2 degree
	6: 3900, #1/4 degree
	5: 10000, #1/5 degree
	4: 12000, #1/8 degree
	3: 16300} #1/12 degree

# Default values for datasets
DEFAULT_NC_KEYS = ['temp','sal','u','v','ssh']
DEFAULT_NC_DTYPE = '.nc'

# Default values for comparison_methods.interpolation
DEFAULT_INTERP_IDW_P = 2

# Default values for comparison_methods.classification
DEFAULT_CLF_N_CLUSTERS = 2
DEFAULT_CLF_N_INIT = 8
DEFAULT_CLF_PENALTY = 'l1'
DEFAULT_CLF_SOLVER = 'liblinear'
DEFAULT_CLF_C = 1.
DEFAULT_CLF_CLASS_WEIGHT = 'balanced'

# Default visualization parameters
DEFAULT_VIS_PLOT_REL_PRIM_COLOR = 'black'
DEFAULT_VIS_PLOT_LINEWIDTH = 1
DEFAULT_VIS_PRIMARY_LABEL = 'Bilinear'
