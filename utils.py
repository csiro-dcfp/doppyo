"""
    General support functions for the pyLatte package
    Author: Dougie Squire
    Date created: 04/04/2018
    Python Version: 3.6
"""

__all__ = ['categorize','compute_pdf', 'compute_cdf', 'compute_rank', 'compute_histogram',
           'calc_integral', 'calc_difference', 'calc_division', 'calc_boxavg_latlon',
           'make_lon_positive', 'find_other_dims']

# ===================================================================================================
# Packages
# ===================================================================================================
import numpy as np
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
    
    return da.to_dataset('categorized').apply(np.digitize,bins=bin_edges)['categorized']


# ===================================================================================================
def compute_pdf(da, bin_edges, dim):
    """ Returns the probability distribution function along the specified dimensions"""
    
    hist = compute_histogram(da, bin_edges, dim)
    
    return hist / calc_integral(hist, dim='bins')


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
        hist = da.to_dataset('histogram').apply(make_histogram,bin_edges=bin_edges)['histogram']
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
# IO tools
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
def make_lon_positive(da):
    ''' Adjusts longitudes to be positive '''
    
    da['lon'] = np.where(da['lon'] < 0, da['lon'] + 360, da['lon']) 
    da = da.sortby('lon')
    
    return da


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
