"""
    General support functions for the pyLatte package
    Author: Dougie Squire
    Date created: 04/04/2018
    Python Version: 3.6
"""

__all__ = ['categorize','compute_pdf', 'compute_cdf', 'compute_rank', 'compute_histogram',
           'calc_integral', 'calc_difference', 'calc_division', 'load_climatology', 'anomalize',
           'leadtime_to_datetime', 'datetime_to_leadtime', 'repeat_data', 'calc_boxavg_latlon',
           'stack_by_init_date', 'prune', 'get_nearest_point', 'make_lon_positive', 'get_bin_edges', 
           'get_lead_times', 'timer', 'find_other_dims']

# ===================================================================================================
# Packages
# ===================================================================================================
from datetime import timedelta
import numpy as np
import pandas as pd
import xarray as xr
import time


# ===================================================================================================
# Probability tools
# ===================================================================================================
def categorize(da,bin_edges):
    """ 
    Returns the indices of the bins to which each value in input array belongs 
    Output indices are such that bin_edges[i-1] <= x < bin_edges[i]
    """
    
    return da.to_dataset(name='categorized').apply(np.digitize,bins=bin_edges)['categorized']


# ===================================================================================================
def compute_pdf(da, bin_edges, dim):
    """ Returns the probability distribution function along the specified dimensions"""
    
    hist = compute_histogram(da, bin_edges, dim)
    
    return hist / calc_integral(hist, dim='bins', method='rect')


# ===================================================================================================
def compute_cdf(da, bin_edges, dim):
    """ Returns the cumulative probability distribution function along the specified dimensions"""
    
    pdf = compute_pdf(da, bin_edges, dim)
    
    return calc_integral(pdf, dim='bins', method='rect', cumulative=True)


# ===================================================================================================
def rank_gufunc(x):
    ''' Returns ranked data along specified dimension '''
    
    import bottleneck
    ranks = bottleneck.rankdata(x,axis=-1)
    ranks = ranks[...,0]
    
    return ranks


def compute_rank(fcst, obsv, dim): 
    ''' Feeds forecast and observation data to ufunc that ranks data along specified dimension'''
    
    # Add 'ensemble' coord to obs if one does not exist -----
    if dim not in obsv.coords:
        obsv_pass = obsv.copy()
        obsv_pass.coords[dim] = -1
        obsv_pass = obsv_pass.expand_dims(dim)
    else:
        obsv_pass = obsv.copy()

    combined = xr.concat([obsv_pass, fcst], dim=dim)
    
    return xr.apply_ufunc(rank_gufunc, combined,
                          input_core_dims=[[dim]],
                          dask='allowed',
                          output_dtypes=[int]).rename('rank')


# ===================================================================================================
def make_histogram(da, bin_edges):
    """ Constructs histogram data along specified dimension """
    
    bins = (bin_edges[0:-1]+bin_edges[1:])/2
    
    return xr.DataArray(np.histogram(da.values, bins=bin_edges)[0], coords=[bins], dims='bins').rename('histogram')


def compute_histogram(da, bin_edges, dims):
    """ Returns the histogram of data along the specified dimensions"""
    
    other_dims = find_other_dims(da,dims)
    if other_dims == None:
        hist = da.to_dataset(name='histogram').apply(make_histogram,bin_edges=bin_edges)['histogram']
    else:
        hist = da.stack(stacked=other_dims) \
                 .groupby('stacked') \
                 .apply(make_histogram,bin_edges=bin_edges) \
                 .unstack('stacked')
    return hist.rename('histogram')


# ===================================================================================================
# Operational tools
# ===================================================================================================
def calc_integral(da, dim, method='trapz', cumulative=False):
    """ Returns trapezoidal/rectangular integration along specified dimension """
    
    x = da[dim]
    if method == 'trapz':
        dx = x - x.shift(**{dim:1})
        dx = dx.fillna(0.0)

        if cumulative:
            integral = ((da.shift(**{dim:1}) + da) * dx / 2.0) \
                       .fillna(0.0) \
                       .cumsum(dim)
        else:
            integral = ((da.shift(**{dim:1}) + da) * dx / 2.0) \
                       .fillna(0.0) \
                       .sum(dim)
    elif method == 'rect':
        dx1 = x - x.shift(**{dim:1})
        dx2 = -(x - x.shift(**{dim:-1}))
        dx = dx1.combine_first(dx2)
        
        if cumulative:
            integral = (da * dx).cumsum(dim) 
        else:
            integral = (da * dx).sum(dim) 
    else:
        raise ValueError(f'{method} is not a recognised integration method')
    return integral
    

# ===================================================================================================
def calc_difference(fcst,obsv):
    """ Returns the difference of two fields """
    
    return fcst - obsv


# ===================================================================================================
def calc_division(fcst,obsv):
    """ Returns the division of two fields """
    
    return fcst / obsv


# ===================================================================================================
# Climatology tools
# ===================================================================================================
def load_climatology(clim, variable, freq):
    """ 
    Returns pre-saved climatology at desired frequency 
    Current frequency options are: daily ('D'); monthly ('M'); yearly ('Y')
    """
    
    # Load specified dataset -----
    if clim == 'jra_1958-2016':
        data_loc = '/OSM/CBR/OA_DCFP/data/intermediate_products/pylatte_climatologies/jra.55.isobaric.1958010100_2016123118.clim.nc'

        ds = xr.open_mfdataset(data_loc, autoclose=True)
        
        if variable not in ds.data_vars:
            raise ValueError(f'"{variable}" is not (yet) available in {clim}')
            
    else:
        raise ValueError(f'"{clim}" is not an available climatology. Available options are "jra_1958-2016"')
    
        
    if variable == 'precip':
        clim = ds[variable].resample(freq=freq, dim='time', how='sum')
    else:
        clim = ds[variable].resample(freq=freq, dim='time', how='mean')
        
    clim['time'] = clim['time'].values.astype('<M8[' + freq + ']')

    return clim


# ===================================================================================================
def anomalize(data,clim):
    """ Receives raw and climatology data at matched frequencies and returns the anomaly """
    
    data_freq = pd.infer_freq(data.time.values)
    clim_freq = pd.infer_freq(clim.time.values)
    
    if data_freq != clim_freq:
        raise ValueError('"data" and "clim" must have matched frequencies')
    
    if (data_freq == 'M') | (data_freq == 'MS'):
        freq = 'time.month'
    elif data_freq == 'D':
        freq = 'time.day'
        
    anom = data.groupby(freq) - clim.groupby(freq).mean()
    
    return prune(anom)

# ===================================================================================================
# IO tools
# ===================================================================================================
def leadtime_to_datetime(data):
    """ Converts time information from init_date/lead_time dimension pair to single datetime dimension """
    
    init_date = data.init_date.values
    lead_time= list(map(int,data.lead_time.values))
    freq = data.lead_time.attrs['units']

    if freq == 'MS':
        freq = 'M'

    datetime = np.array([init_date.astype('<M8[' + freq + ']') + 
                         np.timedelta64(ix, freq) for ix in lead_time]) \
                         .astype('<M8[' + freq + ']')
    
    data = data.rename({'lead_time' : 'time'})
    data['time'] = datetime
    
    return prune(data)


# ===================================================================================================
def datetime_to_leadtime(data):
    """ Converts time information from single datetime dimension to init_date/lead_time dimension pair """
    
    init_date = data.time.values[0]
    lead_times = range(len(data.time))
    freq = pd.infer_freq(data.time.values)

    data = data.rename({'time' : 'lead_time'})
    data['lead_time'] = lead_times
    data['lead_time'].attrs['units'] = freq

    data.coords['init_date'] = init_date
    
    return prune(data)


# ===================================================================================================
def repeat_data(data, repeat_dim, repeat_dim_value=0):
    """ 
    Returns object the same sizes as data, but with data at repeat_dim = repeat_dim_value repeated across all 
    other entries in repeat_dim
    """

    repeat_data = data.loc[{repeat_dim : repeat_dim_value}].drop(repeat_dim).squeeze()
    
    return (0 * data).groupby(repeat_dim,squeeze=True).apply(calc_difference, obsv=-repeat_data)



# ===================================================================================================
def calc_boxavg_latlon(da,box):
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

    # Initialize xarray object for first initialization date -----
    start_index = np.where(data_in.time == np.datetime64(init_dates[0]))[0].item()
    data_out = data_in.isel(time=range(start_index, start_index + N_lead_steps))
    data_out.coords[init_date_name] = init_dates[0]
    data_out = data_out.expand_dims(init_date_name)
    data_out = data_out.rename({'time' : lead_time_name})
    data_out[lead_time_name] = range(N_lead_steps)
    data_out[lead_time_name].attrs['units'] = pd.infer_freq(data_in.time.values)
    
    # Loop over remaining initialization dates -----
    for init_date in init_dates[1:]:
        start_index = np.where(data_in.time == np.datetime64(init_date))[0].item()
        data_temp = data_in.isel(time=range(start_index, start_index + N_lead_steps))

        # Concatenate along initialization date dimension/coordinate -----
        data_temp = data_temp.rename({'time' : lead_time_name})
        data_temp[lead_time_name] = range(N_lead_steps)
        data_temp.coords[init_date_name] = init_date
        data_out = xr.concat([data_out, data_temp],init_date_name) 
    
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
    ''' Returns bin edges '''
    
    dbin = np.diff(bins)/2
    bin_edges = np.concatenate(([bins[0]-dbin[0]], 
                                 bins[:-1]+dbin, 
                                 [bins[-1]+dbin[-1]]))
    
    return bin_edges


# ===================================================================================================
def get_lead_times(FCST_LENGTH, resample_freq):
    """ 
    Returns range() of the minimum number of time increments at resample_freq 
    over the period of years, FCST_LENGTH
    """

    no_leap = 2001
    n_incr = len(pd.date_range('1/1/' + str(no_leap),
                               '12/1/' + str(no_leap+FCST_LENGTH-1),
                               freq=resample_freq)) # number of lead_time increments
    
    return range(n_incr+1)


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


