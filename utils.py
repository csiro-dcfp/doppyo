"""
    General support functions for the pyLatte package
    Author: Dougie Squire
    Date created: 04/04/2018
    Python Version: 3.6
"""

__all__ = ['categorize','compute_pdf', 'compute_cdf', 'compute_rank', 'compute_histogram',
           'calc_integral', 'calc_difference', 'calc_division', 'load_climatology', 'anomalize', 
           'trunc_time', 'infer_freq', 'month_delta', 'year_delta', 'leadtime_to_datetime', 
           'datetime_to_leadtime', 'repeat_data', 'calc_boxavg_latlon', 'stack_by_init_date', 
           'prune', 'get_nearest_point', 'make_lon_positive', 'get_bin_edges', 'timer', 
           'find_other_dims']

# ===================================================================================================
# Packages
# ===================================================================================================
from datetime import timedelta
import numpy as np
import pandas as pd
import xarray as xr
import time
import collections
import itertools
from scipy.interpolate import interp1d
from scipy import ndimage

# ===================================================================================================
# Probability tools
# ===================================================================================================
def categorize(da, bin_edges):
    """ 
    Returns the indices of the bins to which each value in input array belongs 
    Output indices are such that bin_edges[i-1] <= x < bin_edges[i]
    """

    return xr.apply_ufunc(np.digitize, da, bin_edges,
                          input_core_dims=[[],[]],
                          dask='allowed',
                          output_dtypes=[int]).rename('categorized')


# ===================================================================================================
def compute_pdf(da, bin_edges, over_dims):
    """ Returns the probability distribution function along the specified dimensions"""
    
    hist = compute_histogram(da, bin_edges, over_dims)
    
    return hist / calc_integral(hist, over_dim='bins', method='rect')


# ===================================================================================================
def compute_cdf(da, bin_edges, over_dims):
    """ Returns the cumulative probability distribution function along the specified dimensions"""
    
    pdf = compute_pdf(da, bin_edges, over_dims)
    
    return calc_integral(pdf, over_dim='bins', method='rect', cumulative=True)


# ===================================================================================================
def rank_gufunc(x):
    ''' Returns ranked data along specified dimension '''
    
    import bottleneck
    ranks = bottleneck.rankdata(x,axis=-1)
    ranks = ranks[...,0]
    
    return ranks


def compute_rank(da_1, da_2, over_dim): 
    ''' Feeds forecast and observation data to ufunc that ranks data along specified dimension'''
    
    # Add 'ensemble' coord to obs if one does not exist -----
    if over_dim not in da_2.coords:
        da_2_pass = da_2.copy()
        da_2_pass.coords[over_dim] = -1
        da_2_pass = da_2_pass.expand_dims(over_dim)
    else:
        da_2_pass = da_2.copy()

    combined = xr.concat([da_2_pass, da_1], dim=over_dim)
    
    return xr.apply_ufunc(rank_gufunc, combined,
                          input_core_dims=[[over_dim]],
                          dask='allowed',
                          output_dtypes=[int]).rename('rank')


# ===================================================================================================
def unstack_and_count(da, dims):
    """ Unstacks provided xarray object and returns the total number of elements along dims """
    
    unstacked = da.unstack(da.dims[0])
    
    return ((0 * unstacked) + 1).sum(dim = dims)


def compute_histogram(da, bin_edges, over_dims):
    """ Returns the histogram of data over the specified dimensions """
    
    # To use groupby_bins, da must have a name -----
    da = da.rename('data') 
    
    hist = da.groupby_bins(da,bins=bin_edges) \
             .apply(unstack_and_count, dims=over_dims) \
             .fillna(0) \
             .rename({'data_bins' : 'bins'})
    hist['bins'] = (bin_edges[0:-1]+bin_edges[1:])/2
    
    return hist.astype(int)


# ===================================================================================================
# Operational tools
# ===================================================================================================
def calc_integral(da, over_dim, method='trapz', cumulative=False):
    """ Returns trapezoidal/rectangular integration along specified dimension """
    
    x = da[over_dim]
    if method == 'trapz':
        dx = x - x.shift(**{over_dim:1})
        dx = dx.fillna(0.0)

        if cumulative:
            integral = ((da.shift(**{over_dim:1}) + da) * dx / 2.0) \
                       .fillna(0.0) \
                       .cumsum(over_dim)
        else:
            integral = ((da.shift(**{over_dim:1}) + da) * dx / 2.0) \
                       .fillna(0.0) \
                       .sum(over_dim)
    elif method == 'rect':
        dx1 = x - x.shift(**{over_dim:1})
        dx2 = -(x - x.shift(**{over_dim:-1}))
        dx = dx1.combine_first(dx2)
        
        if cumulative:
            integral = (da * dx).cumsum(over_dim) 
        else:
            integral = (da * dx).sum(over_dim) 
    else:
        raise ValueError(f'{method} is not a recognised integration method')
        
    return integral
    

# ===================================================================================================
def calc_difference(data_1, data_2):
    """ Returns the difference of two fields """
    
    return data_1 - data_2


# ===================================================================================================
def calc_division(data_1, data_2):
    """ Returns the division of two fields """
    
    return data_1 / data_2


# ===================================================================================================
# Climatology tools
# ===================================================================================================
def load_mean_climatology(clim, variable, freq, **kwargs):
    """ 
    Returns pre-saved climatology at desired frequency (greater than daily)
    """
    
    data_path = '/OSM/CBR/OA_DCFP/data/intermediate_products/pylatte_climatologies/'
    
    # Load specified dataset -----
    if clim == 'jra_1958-2016':
        data_loc = data_path + 'jra.55.isobaric.1958010100_2016123118.clim.nc'
        ds = xr.open_dataset(data_loc, **kwargs)
        
        if variable not in ds.data_vars:
            raise ValueError(f'"{variable}" is not (yet) available in {clim}')
            
    elif clim == 'cafe_fcst_v1_atmos_2003-2021':
        data_loc = data_path + 'cafe.fcst.v1.atmos.2003010112_2021063012.clim.nc'
        ds = xr.open_dataset(data_loc, **kwargs)
        
        if variable not in ds.data_vars:
            raise ValueError(f'"{variable}" is not (yet) available in {clim}')
            
    elif clim == 'cafe_fcst_v1_ocean_2003-2021':
        data_loc = data_path + 'cafe.fcst.v1.ocean.2003010112_2021063012.clim.nc'
        ds = xr.open_dataset(data_loc, **kwargs)
        
        if variable not in ds.data_vars:
            raise ValueError(f'"{variable}" is not (yet) available in {clim}')
    
    elif clim == 'HadISST_1870-2018':
        data_loc = data_path + 'HadISST.1870011612_2018021612.clim.nc'
        ds = xr.open_dataset(data_loc, **kwargs)
        
        if variable not in ds.data_vars:
            raise ValueError(f'"{variable}" is not (yet) available in {clim}')
            
    else:
        raise ValueError(f'"{clim}" is not an available climatology. Available options are "jra_1958-2016", "cafe_fcst_v1_atmos_2003-2021", "cafe_fcst_v1_ocean_2003-2021", "HadISST_1870-2018"')
        
    if variable == 'precip':
        clim = ds[variable].resample(time=freq).sum(dim='time')
    else:
        clim = ds[variable].resample(time=freq).mean(dim='time')
        
    return clim


# ===================================================================================================
def anomalize(data, clim):
    """ Receives raw and climatology data at matched frequencies and returns the anomaly """
    
    data_use = data.copy(deep=True)
    clim_use = clim.copy(deep=True)
    
    # If only climatological time instance is given, assume this is annual average -----
    if len(clim_use.time) > 1:
        data_freq = infer_freq(data_use.time.values)
        clim_freq = infer_freq(clim_use.time.values)
    else:
        data_freq = 'A'
        clim_freq = 'A'

    if data_freq != clim_freq:
        raise ValueError('"data" and "clim" must have matched frequencies')
    
    if 'D' in data_freq:
        time_keep = data_use.time

        # Contruct month-day arrays -----
        data_mon = np.array([str(i).zfill(2) + '-' for i in data_use.time.dt.month.values])
        data_day = np.array([str(i).zfill(2)  for i in data_use.time.dt.day.values])
        data_use['time'] = np.core.defchararray.add(data_mon, data_day)
        clim_mon = np.array([str(i).zfill(2) + '-' for i in clim_use.time.dt.month.values])
        clim_day = np.array([str(i).zfill(2)  for i in clim_use.time.dt.day.values])
        clim_use['time'] = np.core.defchararray.add(clim_mon, clim_day)

        anom = data_use.groupby('time') - clim_use.groupby('time',squeeze=False).mean(dim='time')
        anom['time'] = time_keep
    elif 'M' in data_freq:
        freq = 'time.month'
        anom = data_use.groupby(freq) - clim_use.groupby(freq,squeeze=False).mean(dim='time')
    elif ('A' in data_freq) | ('Y' in data_freq):
        freq = 'time.year'
        anom = data_use - prune(clim_use.groupby(freq,squeeze=False).mean(dim='time').squeeze())
        
    return prune(anom)


# ===================================================================================================
# IO tools
# ===================================================================================================
def trunc_time(time, freq):
    """ 
    Truncates values in provided time array to provided frequency. E.g. 2018-01-15T12:00 with 
    freq = 'M' becomes 2018-01-01. 
    """
    
    return time.astype('<M8[' + freq + ']')


# ===================================================================================================
def infer_freq(time):
    """ 
    Returns most likely frequency of provided time array 
    """
    
    return pd.infer_freq(time)


# ===================================================================================================
def month_delta(date_in, delta, trunc_to_start=False):
    """ Increments provided datetime64 array by delta months """
    
    date_mod = pd.Timestamp(date_in)
    
    m, y = (date_mod.month + delta) % 12, date_mod.year + ((date_mod.month) + delta - 1) // 12
    
    if not m: m = 12
    
    d = min(date_mod.day, [31,
        29 if y % 4 == 0 and not y % 400 == 0 else 28,31,30,31,30,31,31,30,31,30,31][m - 1])
    
    if trunc_to_start:
        date_out = trunc_time(np.datetime64(date_mod.replace(day=d,month=m, year=y)),'M')
    else:
        date_out = np.datetime64(date_mod.replace(day=d,month=m, year=y))
    
    return date_out


# ===================================================================================================
def year_delta(date_in, delta, trunc_to_start=False):
    """ Increments provided datetime64 array by delta years """
    
    date_mod = month_delta(date_in, 12 * delta)
    
    if trunc_to_start:
        date_out = trunc_time(date_mod,'Y')
    else: date_out = date_mod
        
    return date_out


# ===================================================================================================
def leadtime_to_datetime(data_in, lead_time_name='lead_time', init_date_name='init_date'):
    """ Converts time information from lead time/initial date dimension pair to single datetime dimension """
    
    try:
        init_date = data_in[init_date_name].values[0]
    except IndexError:
        init_date = data_in[init_date_name].values
        
    lead_times = list(map(int, data_in[lead_time_name].values))
    freq = data_in[lead_time_name].attrs['units']
    
    # Split frequency into numbers and strings -----
    incr_string = ''.join([i for i in freq if i.isdigit()])
    freq_incr = [int(incr_string) if incr_string else 1][0]
    freq_type = ''.join([i for i in freq if not i.isdigit()])

    # Deal with special cases of monthly and yearly frequencies -----
    if 'M' in freq_type:
        datetimes = np.array([month_delta(init_date, freq_incr * ix) for ix in lead_times])
    elif ('A' in freq_type) | ('Y' in freq_type):
        datetimes = np.array([year_delta(init_date, freq_incr * ix) for ix in lead_times])
    else:
        datetimes = (pd.date_range(init_date, periods=len(lead_times), freq=freq)).values
    
    data_out = data_in.drop(init_date_name)
    data_out = data_out.rename({lead_time_name : 'time'})
    data_out['time'] = datetimes
    
    return prune(data_out)


# ===================================================================================================
def datetime_to_leadtime(data_in):
    """ Converts time information from single datetime dimension to init_date/lead_time dimension pair """
    
    init_date = data_in.time.values[0]
    lead_times = range(len(data_in.time))
    freq = infer_freq(data_in.time.values)

    data_out = data_in.rename({'time' : 'lead_time'})
    data_out['lead_time'] = lead_times
    data_out['lead_time'].attrs['units'] = freq

    data_out.coords['init_date'] = init_date
    
    return data_out


# ===================================================================================================
def repeat_data(data, repeat_dim, repeat_dim_value=0):
    """ 
    Returns object the same sizes as data, but with data at repeat_dim = repeat_dim_value repeated across all 
    other entries in repeat_dim
    """

    repeat_data = data.loc[{repeat_dim : repeat_dim_value}].drop(repeat_dim).squeeze()
    
    return (0 * data).groupby(repeat_dim,squeeze=True).apply(calc_difference, data_2=-repeat_data)


# ===================================================================================================
def calc_boxavg_latlon(da, box):
    '''
        Returns the average of a given quantity over a provide lat-lon region
        Note: longitudinal coordinates must be positive easterly
    '''
    
    # Adjust longitudes to be positive -----
    da = make_lon_positive(da) 

    # Extract desired region -----
    da = da.where(da['lat']>box[0],drop=True) \
            .where(da['lat']<box[1],drop=True) \
            .where(da['lon']>box[2],drop=True) \
            .where(da['lon']<box[3],drop=True)

    # Average over extracted region -----
    da = da.mean(dim=('lat','lon'))
    
    return da


# ===================================================================================================
def stack_by_init_date(data_in, init_dates, N_lead_steps, init_date_name='init_date', lead_time_name='lead_time'):
    """ 
    Splits provided data array into n chunks beginning at time=init_date[n] and spanning 
    N_lead_steps time increments
    Input Dataset/DataArray must span full range of times required for this operation
    """

    init_list = []
    for init_date in init_dates:
        start_index = np.where(data_in.time == np.datetime64(init_date))[0].item()
        end_index = min([start_index + N_lead_steps, len(data_in.time)])
        init_list.append(
                      datetime_to_leadtime(
                          data_in.isel(time=range(start_index, end_index))))
    
    data_out = xr.concat(init_list, dim=init_date_name)
    
    return data_out


# ===================================================================================================
def prune(data,squeeze=False):
    """ Removes all coordinates that are not dimensions from an xarray object """
    
    codims = list(set(data.coords)-set(data.dims))

    for codim in codims:
        if codim in data.coords:
            data = data.drop(codim)
        
    if squeeze:
        data = data.squeeze()
    
    return data


# ===================================================================================================
def get_nearest_point(da, lat, lon):
    """ Returns the nearest grid point to the specified lat/lon location """

    return da.sel(lat=lat,lon=lon,method='nearest')


# ===================================================================================================
def make_lon_positive(da):
    ''' Adjusts longitudes to be positive '''
    
    da['lon'] = np.where(da['lon'] < 0, da['lon'] + 360, da['lon']) 
    da = da.sortby('lon')
    
    return da


# ===================================================================================================
def get_bin_edges(bins):
    ''' Returns bin edges of provided bins '''
    
    dbin = np.diff(bins)/2
    bin_edges = np.concatenate(([bins[0]-dbin[0]], 
                                 bins[:-1]+dbin, 
                                 [bins[-1]+dbin[-1]]))
    
    return bin_edges


# ===================================================================================================
class timer(object):
    '''
        File name: timer

        Description: class for timing code snippets 

        Author: Dougie Squire
        Date created: 21/03/2018
        Python Version: 3.5

        Usage:
            with timer():
                # do something
    '''
    
    def __init__(self, name=None):
        self.name = name

    def __enter__(self):
        self.tstart = time.time()

    def __exit__(self, type, value, traceback):
        if self.name:
            print('   f{self.name}')
        print(f'   Elapsed: {time.time() - self.tstart} sec')


# ===================================================================================================
# xarray processing tools
# ===================================================================================================
def find_other_dims(da, dims_exclude):
    """ Returns all dimensions in dataset excluding dim_exclude """
    
    dims = da.dims
    
    if dims_exclude == None:
        other_dims = dims
    else:
        if isinstance(dims, str):
            dims = [dims]
        if isinstance(dims_exclude, str):
            dims_exclude = [dims_exclude]

        other_dims = list(set(dims).difference(set(dims_exclude)))
        if len(other_dims) > 1:
            other_dims = tuple(other_dims)
        if other_dims == []:
            other_dims = None
        
    return other_dims