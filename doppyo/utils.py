"""
    General support functions for the doppyo package
    Author: Dougie Squire (some ocean focused additions & edits Thomas Moore)
    Date created: 04/04/2018
    Python Version: 3.6
"""

__all__ = ['timer', 'constant', 'skewness', 'kurtosis', 'digitize', 'pdf', 'cdf', 
           'histogram', 'differentiate_wrt', 'xy_from_lonlat', 'integrate', 'calc_difference', 
           'calc_division', 'calc_average', 'calc_fft', 'calc_ifft', 'fftfilt', 'stack_times', 'normal_mbias_correct', 
           'normal_msbias_correct', 'conditional_bias_correct', 'load_climatology', 'anomalize', 
           'trunc_time', 'month_delta', 'year_delta', 'leadtime_to_datetime', 'datetime_to_leadtime', 
           'repeat_data', 'calc_boxavg_latlon', 'stack_by_init_date', 'prune', 'get_nearest_point', 'get_bin_edges', 
           'is_datetime', 'find_other_dims', 'get_lon_name', 'get_lat_name', 'get_level_name',
           'get_pres_name', 'cftime_to_datetime64', 'get_depth_name', 'size_GB']

# ===================================================================================================
# Packages
# ===================================================================================================
from datetime import timedelta
import numpy as np
import pandas as pd
import xarray as xr
from xarray.core.duck_array_ops import dask_array_type
import time
import collections
import itertools
from scipy.interpolate import interp1d
from scipy import ndimage
import dask.array
import copy
import warnings

import cartopy
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from cartopy.util import add_cyclic_point
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

# Load doppyo packages -----
from doppyo import skill

# ===================================================================================================
# Classes
# ===================================================================================================
class timer(object):
    """
        Reports time taken to complete code snippets.
        Author: Dougie Squire
        Date: 14/02/2018

        Examples
        --------
        >>> with doppyo.utils.timer():
        >>>     x = 1 + 1
        Elapsed: 4.5299530029296875e-06 sec
    """
    
    def __init__(self, name=None):
        self.name = name

    def __enter__(self):
        self.tstart = time.time()

    def __exit__(self, type, value, traceback):
        if self.name:
            print('   f{self.name}')
        print(f'   Elapsed: {time.time() - self.tstart} sec')
        

# ===================================================================================================
class constants(object):
    """ 
        Returns commonly used constants.
        Author: Dougie Squire
        Date: 14/02/2018
    
        Examples
        --------
        >>> pi = doppyo.utils.constants().pi
    """
    
    def _constant(f):
        """ Decorator to make constants unmodifiable """

        def fset(self, value):
            raise TypeError('Cannot overwrite constant values')
        def fget(self):
            return f()
        return property(fget, fset)
    
    @_constant
    def R_d():
        return 287.04 # gas constant of dry air [J / (kg * degK)]
    
    @_constant
    def R_v():
        return 461.50 # gas constant of water vapor [J / (kg * degK)]
    
    @_constant
    def C_vd():
        return 719.0 # heat capacity of dry air at constant volume [J / (kg * degK)]
    
    @_constant
    def C_pd():
        return 1005.7 # 'heat capacity of dry air at constant pressure [J / (kg * degK)]
    
    @_constant
    def C_vv():
        return 1410.0 # heat capacity of water vapor at constant volume [J / (kg * degK)]
    
    @_constant
    def C_pv():
        return 1870.0 # heat capacity of water vapor at constant pressure [J / (kg * degK)]
    
    @_constant
    def C_l():
        return 4190.0 # heat capacity of liquid water [J / (kg * degK)] 
    
    @_constant
    def g():
        return 9.81 # gravitational acceleration [m / s^2]
    
    @_constant
    def R_earth():
        return 6.371e6 # radius of the earth ['m']
    
    @_constant
    def Omega():
        return 7.2921e-5 # earth rotation rate [rad/s]
    
    @_constant
    def pi():
        return 2*np.arccos(0) # pi
    
    @_constant
    def Ce():
        return 0.3098 # Eady constant


# ===================================================================================================
# Probability tools
# ===================================================================================================
def skewness(da, dim):
    """
        Returns the skewness along dimension dim
        Author: Dougie Squire
        Date: 20/08/2018

        Parameters
        ----------
        da : xarray DataArray
            Array containing values for which to compute skewness
        dim : str or sequence of str
            Dimension(s) over which to compute the skewness

        Returns
        -------
        skewness : xarray DataArray
            New DataArray object with skewness applied to its data and the indicated dimension(s) removed

        Examples
        --------
        >>> arr = xr.DataArray(np.arange(6).reshape(2, 3), 
        ...                    coords=[('x', ['a', 'b']), ('y', [0, 1, 2])])
        >>> arr
        <xarray.DataArray (x: 2, y: 3)>
        array([[0, 1, 2],
               [3, 4, 5]])
        Coordinates:
          * x        (x) <U1 'a' 'b'
          * y        (y) int64 0 1 2
        >>> doppyo.utils.skewness(arr, 'x')
        <xarray.DataArray (y: 3)>
        array([0., 0., 0.])
        Coordinates:
          * y        (y) int64 0 1 2
    """
    
    daf = da - da.mean(dim)
    return ((daf ** 3).mean(dim) / ((daf ** 2).mean(dim) ** (3/2))).rename('skewness')


# ===================================================================================================
def kurtosis(da, dim):
    """
        Returns the kurtosis along dimension dim
        Author: Dougie Squire
        Date: 20/08/2018

        Parameters
        ----------
        da : xarray DataArray
            Array containing values for which to compute kurtosis
        dim : str or sequence of str
            Dimension(s) over which to compute the kurtosis

        Returns
        -------
        kurtosis : xarray DataArray
            New DataArray object with kurtosis applied to its data and the indicated dimension(s) removed

        Examples
        --------
        >>> arr = xr.DataArray(np.arange(6).reshape(2, 3), 
        ...                    coords=[('x', ['a', 'b']), ('y', [0, 1, 2])])
        >>> arr
        <xarray.DataArray (x: 2, y: 3)>
        array([[0, 1, 2],
               [3, 4, 5]])
        Coordinates:
          * x        (x) <U1 'a' 'b'
          * y        (y) int64 0 1 2
        >>> doppyo.utils.kurtosis(arr, 'x')
        <xarray.DataArray (y: 3)>
        array([1., 1., 1.])
        Coordinates:
          * y        (y) int64 0 1 2
    """
    
    daf = da - da.mean(dim)
    return ((daf ** 4).mean(dim) / ((daf ** 2).mean(dim) ** (2))).rename('kurtosis')


# ===================================================================================================
def digitize(da, bin_edges):
    """
        Returns the indices of the bins to which each value in input array belongs.
        Author: Dougie Squire
        Date: 31/10/2018
        
        Parameters
        ----------
        da : xarray DataArray
            Array containing values to digitize
        dim : array_like
            Array of bin edges. Output indices, i, are such that bin_edges[i-1] <= x < bin_edges[i]

        Returns
        -------
        digitized : xarray DataArray
            New DataArray object of indices

        Examples
        --------
        >>> da = xr.DataArray(np.random.normal(size=(20,40)), coords=[('x', np.arange(20)), 
        ...                                                           ('y', np.arange(40))])
        >>> bins=np.linspace(-2,2,10)
        >>> bin_edges = doppyo.utils.get_bin_edges(bins)
        >>> doppyo.utils.digitize(da, bin_edges)
        <xarray.DataArray 'digitized' (x: 20, y: 40)>
        array([[ 7,  6,  4, ...,  5,  6,  7],
               [ 5, 11,  2, ...,  7,  6,  0],
               [ 9,  3,  2, ...,  6,  5,  6],
               ...,
               [11, 10,  8, ...,  6,  5,  2],
               [ 3, 10,  3, ...,  8,  7,  7],
               [ 5,  4,  9, ...,  5,  5,  7]])
        Coordinates:
          * x        (x) int64 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19
          * y        (y) int64 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 ...
    """
    
    data = da.data
    if isinstance(data, dask_array_type):
        return xr.DataArray(dask.array.digitize(data, bins=bin_edges), da.coords).rename('digitized')
    else:
        return xr.DataArray(np.digitize(data, bins=bin_edges), da.coords).rename('digitized')


# ===================================================================================================
def pdf(da, bin_edges, over_dims):
    """ 
        Returns the probability distribution function along the specified dimensions
        Author: Dougie Squire
        Date: 01/10/2018
        
        Parameters
        ----------
        da : xarray DataArray
            Array containing values used to compute the pdf
        bin_edges : array_like
            The bin edges, including the rightmost edge
        over_dims : str or sequence of str
            Dimension(s) over which to compute the pdf
            
        Returns
        -------
        pdf : xarray DataArray
            New DataArray object containing pdf
        
        Examples
        --------
        >>> da = xr.DataArray(np.random.normal(size=(100,100)), coords=[('x', np.arange(100)), ('y', np.arange(100))])
        >>> bins=np.linspace(-2,2,10)
        >>> bin_edges = doppyo.utils.get_bin_edges(bins)
        >>> doppyo.utils.pdf(da, bin_edges=bin_edges, over_dims='x')
        <xarray.DataArray (bins: 10, y: 100)>
        array([[0.069588, 0.046392, 0.046875, ..., 0.090909, 0.070312, 0.090909],
               [0.208763, 0.255155, 0.140625, ..., 0.113636, 0.117187, 0.113636],
               [0.278351, 0.115979, 0.304688, ..., 0.25    , 0.234375, 0.227273],
               ...,
               [0.115979, 0.255155, 0.46875 , ..., 0.25    , 0.210937, 0.136364],
               [0.046392, 0.139175, 0.117188, ..., 0.090909, 0.1875  , 0.136364],
               [0.046392, 0.069588, 0.046875, ..., 0.022727, 0.070312, 0.068182]])
        Coordinates:
          * bins     (bins) float64 -2.0 -1.556 -1.111 -0.6667 -0.2222 0.2222 0.6667 ...
          * y        (y) int64 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 ...
          
        Limitations
        -----------
        This function uses doppyo.utils.histogram() which uses xr.groupby_bins when over_dims is a subset 
        of da.dims and is therefore not parallelized in these cases. There are efforts underway to parallelize 
        groupby operations in xarray, see https://github.com/pydata/xarray/issues/585
    """
    
    hist = histogram(da, bin_edges, over_dims)
    
    return (hist / integrate(hist, over_dim='bins', method='rect')).rename('pdf')


# ===================================================================================================
def cdf(da, bin_edges, over_dims):
    """ 
        Returns the cumulative probability distribution function along the specified dimensions
        Author: Dougie Squire
        Date: 01/10/2018
        
        Parameters
        ----------
        da : xarray DataArray
            Array containing values used to compute the cdf
        bin_edges : array_like
            The bin edges, including the rightmost edge
        over_dims : str or sequence of str
            Dimension(s) over which to compute the cdf
            
        Returns
        -------
        cdf : xarray DataArray
            New DataArray object containing cdf
        
        Examples
        --------
        >>> da = xr.DataArray(np.random.normal(size=(100,100)), coords=[('x', np.arange(100)), ('y', np.arange(100))])
        >>> bins=np.linspace(-2,2,10)
        >>> bin_edges = doppyo.utils.get_bin_edges(bins)
        >>> doppyo.utils.cdf(da, bin_edges=bin_edges, over_dims='x')
        <xarray.DataArray (bins: 10, y: 100)>
        array([[0.050505, 0.      , 0.030612, ..., 0.020202, 0.010204, 0.020619],
               [0.121212, 0.085106, 0.081633, ..., 0.080808, 0.081633, 0.061856],
               [0.232323, 0.138298, 0.142857, ..., 0.171717, 0.183673, 0.195876],
               ...,
               [0.939394, 0.925532, 0.908163, ..., 0.909091, 0.94898 , 0.907216],
               [0.979798, 0.968085, 0.969388, ..., 0.959596, 0.979592, 0.989691],
               [1.      , 1.      , 1.      , ..., 1.      , 1.      , 1.      ]])
        Coordinates:
          * bins     (bins) float64 -2.0 -1.556 -1.111 -0.6667 -0.2222 0.2222 0.6667 ...
          * y        (y) int64 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 ...
          
        Limitations
        -----------
        This function uses doppyo.utils.histogram() which uses xr.groupby_bins when over_dims is a subset 
        of da.dims and is therefore not parallelized in these cases. There are efforts underway to parallelize 
        groupby operations in xarray, see https://github.com/pydata/xarray/issues/585
    """
    
    return integrate(pdf(da, bin_edges, over_dims), over_dim='bins', method='rect', cumulative=True).rename('cdf')


# ===================================================================================================
def histogram(da, bin_edges, over_dims):
    """ 
        Returns the histogram over the specified dimensions
        Author: Dougie Squire
        Date: 01/10/2018
        
        Parameters
        ----------
        da : xarray DataArray
            Array containing values used to compute the histogram
        bin_edges : array_like
            The bin edges, including the rightmost edge
        over_dims : str or sequence of str
            Dimension(s) over which to compute the histogram
            
        Returns
        -------
        histogram : xarray DataArray
            New DataArray object containing the histogram
            
        Examples
        --------
        >>> da = xr.DataArray(np.random.normal(size=(100,100)), 
        ...                   coords=[('x', np.arange(100)), ('y', np.arange(100))])
        >>> bins=np.linspace(-2,2,10)
        >>> bin_edges = doppyo.utils.get_bin_edges(bins)
        >>> doppyo.utils.histogram(da, bin_edges=bin_edges, over_dims='x')
        <xarray.DataArray 'data' (bins: 10, y: 100)>
        array([[ 3.,  1.,  6., ...,  2.,  4.,  3.],
               [ 2., 12.,  4., ...,  7.,  3.,  7.],
               [ 9.,  9., 11., ..., 19., 13.,  6.],
               ...,
               [13.,  9.,  4., ...,  6.,  6., 11.],
               [ 3.,  6.,  3., ...,  3.,  7.,  4.],
               [ 2.,  0.,  1., ...,  3.,  3.,  4.]])
        Coordinates:
          * bins     (bins) float64 -2.0 -1.556 -1.111 -0.6667 -0.2222 0.2222 0.6667 ...
          * y        (y) int64 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 ...
  
        Limitations
        -----------
        This function uses xr.groupby_bins when over_dims is a subset of da.dims and is therefore not 
        parallelized in these cases. There are efforts underway to parallelize groupby operations in 
        xarray, see https://github.com/pydata/xarray/issues/585
        
        See also
        --------
        numpy.histogram()
        dask.array.histogram()
    """
    
    def _unstack_and_count(da, dims):
        """ Unstacks provided xarray object and returns the total number of elements along dims """
        try:
            unstacked = da.unstack(da.dims[0])
        except ValueError:
            unstacked = da

        if dims is None:
            return unstacked.count(keep_attrs=True)
        else:
            return ((0 * unstacked) + 1).sum(dim=dims, skipna=True) # da.count has no skipna option in 0.10.8

    bins = (bin_edges[0:-1]+bin_edges[1:]) / 2
    
    # If histogram is computed over all dimensions, use dask/np.histogram
    if set(da.dims) == set(over_dims):
        data = da.data
        if isinstance(data, dask_array_type):
            hist, _ = dask.array.histogram(da.data, bins=bin_edges)
            return xr.DataArray(hist, coords=[bins], dims=['bins']).rename('data')
        else:
            hist, _ = np.histogram(da.data, bins=bin_edges)
            return xr.DataArray(hist, coords=[bins], dims=['bins']).rename('data')
    else:
        # To use groupby_bins, da must have a name -----
        da = da.rename('histogram') 

        hist = da.groupby_bins(da, bins=bin_edges, squeeze=False) \
                 .apply(_unstack_and_count, dims=over_dims) \
                 .fillna(0) \
                 .rename({'histogram_bins' : 'bins'})
        hist['bins'] = (bin_edges[0:-1]+bin_edges[1:]) / 2
    
    # Add nans where data did not fall in any bin -----
    return hist.astype(int).where(hist.sum('bins') != 0).rename('histogram')


# ===================================================================================================
# Operational tools
# ===================================================================================================
def differentiate_wrt(da, dim, x):
    """ 
        Returns the gradient along dim using x to compute differences. This function is required
        because the current implementation of xr.differentiate (0.10.9) can only differentiate with 
        respect to a 1D coordinate. It is common to want to differentiate with respect to something 
        that changes as a function of multiple dimensions (e.g. the zonal distance between regularly 
        spaced lat/lon points varies as a function of lat and lon). Uses second order accurate central 
        differencing in the interior points and first order accurate one-sided (forward or backwards) 
        differencing at the boundaries.
        
        Parameters
        ----------
        da : xarray DataArray
            Array containing values to differentiate
        dim : str
            The dimension to be used to compute the gradient
        x : xarray DataArray
            Array containing values to differentiate with respect to. Must have the same dimensions
            as da
            
        Returns
        -------
        differentiated : xarray DataArray
            New DataArray object containing the differentiate data
            
        Examples
        --------
        >>> A = xr.DataArray(np.random.normal(size=(180,360)), coords=[('lat', np.arange(-90,90,1)), ('lon', np.arange(0,360,1))])
        >>> x, y = doppyo.utils.xy_from_lonlat(A['lon'], A['lat'])
        >>> differentiate_wrt(A, dim='lon', x=x)
        <xarray.DataArray 'differentiated' (lat: 180, lon: 360)>
        array([[ 3.674336e+10, -9.015981e+10, -2.150203e+11, ..., -6.471076e+10,
                -6.057067e+09,  1.253664e+11],
               [-4.133347e-07, -3.932972e-04, -5.982892e-04, ...,  2.972605e-04,
                 9.456351e-04,  1.907131e-03],
               [-6.596434e-04,  6.147016e-06,  2.370071e-04, ..., -8.578490e-06,
                 9.281731e-06,  2.211755e-04],
               ...,
               [-6.467389e-05,  6.315746e-05,  1.713705e-04, ...,  9.742767e-05,
                 1.043358e-04,  1.066228e-04],
               [ 1.542484e-04,  2.802838e-04,  5.511727e-05, ...,  1.665500e-04,
                -6.087167e-06, -3.060961e-04],
               [-5.991109e-04,  2.085148e-04,  4.525132e-04, ..., -9.346556e-05,
                -7.977593e-05,  3.411080e-05]])
        Coordinates:
          * lat      (lat) int64 -90 -89 -88 -87 -86 -85 -84 ... 83 84 85 86 87 88 89
          * lon      (lon) int64 0 1 2 3 4 5 6 7 8 ... 352 353 354 355 356 357 358 359

        See also
        --------
        xarray.DataArray.differentiate()
        numpy.gradient()
    """ 
    
    if not (set(da.dims) == set(x.dims)):
        raise ValueError('da and x must have the same dimensions')
        
    # Replace dimension values if specified -----
    da_n = da.copy()
        
    centre_chunk = range(len(x[dim])-2)
    
    f_hd = da_n.shift(**{dim:-2})
    f = da_n.shift(**{dim:-1})
    f_hs = da_n
    hs = x.shift(**{dim:-1}) - x
    hd = x.shift(**{dim:-2}) - x.shift(**{dim:-1})
    c = (hs ** 2 * f_hd + (hd ** 2 - hs ** 2) * f - hd ** 2 * f_hs) / \
        (hs * hd * (hd + hs)).isel(**{dim : centre_chunk})
    c[dim] = x[dim][1:-1]

    l = (da_n.shift(**{dim:-1}) - da_n).isel(**{dim : 0}) / \
        (x.shift(**{dim:-1}) - x).isel(**{dim : 0})

    r = (-da_n.shift(**{dim:1}) + da_n).isel(**{dim : -1}) / \
        (-x.shift(**{dim:1}) + x).isel(**{dim : -1})
    
    diff = xr.concat([l, c, r], dim=dim)
    diff[dim] = da[dim]
    
    return diff.rename('differentiated')


# ===================================================================================================
def xy_from_lonlat(lon, lat):
    ''' 
        Returns x/y in m from grid points that are in a longitude/latitude format.
        
        Parameters
        ----------
        lon : xarray DataArray
            Array containing longitudes stored relative to longitude dimension/coordinate
        lat : xarray DataArray
            Array containing latitudes stored relative to latitude dimension/coordinate
            
        Returns
        -------
        x : xarray DataArray
            Array containing zonal distance in m
        y : xarray DataArray
            Array containing meridional distance in m
            
        Examples
        --------
        >>> lat = xr.DataArray(np.arange(-90,90,90), dims=['lat'])
        >>> lon = xr.DataArray(np.arange(0,360,90), dims=['lon'])
        >>> doppyo.utils.xy_from_lonlat(lon=lon, lat=lat)
        (<xarray.DataArray (lat: 2, lon: 4)>
         array([[0.000000e+00, 6.127853e-10, 1.225571e-09, 1.838356e-09],
                [0.000000e+00, 1.000754e+07, 2.001509e+07, 3.002263e+07]])
         Dimensions without coordinates: lat, lon, <xarray.DataArray (lat: 2, lon: 4)>
         array([[-10007543.39801, -10007543.39801, -10007543.39801, -10007543.39801],
                [        0.     ,         0.     ,         0.     ,         0.     ]])
         Dimensions without coordinates: lat, lon)
    '''
    
    degtorad = constants().pi / 180
    
    y = (2 * constants().pi * constants().R_earth * lat / 360)
    x = 2 * constants().pi * constants().R_earth * xr.ufuncs.cos(lat * degtorad) * lon / 360
    y = y * (0 * x + 1)
    
    return x, y


# ===================================================================================================
def integrate(da, over_dim, x=None, dx=None, method='trapz', cumulative=False):
    """ 
        Returns trapezoidal/rectangular integration along specified dimension 
    
        Parameters
        ----------
        da : xarray DataArray
            Array containing values to integrate
        over_dim : str
            Dimension to integrate
        x : xarray DataArray, optional
            Values to use for integrand. Must contain dimensions over_dim. If None, x is determined
            from the coords associated with over_dim
        dx : value, optional
            Integrand spacing used to compute the integral. If None, dx is determined from x
        method : str, optional
            Method of performing integral. Options are 'trapz' for trapezoidal integration, or 'rect'
            for rectangular integration
        cumulative : bool, optional
            If True, return the cumulative integral    
            
        Returns
        -------
        integral : xarray DataArray
            Array containing the integral along the specified dimension
            
        Examples
        --------
        >>> A = xr.DataArray(np.random.normal(size=(3,2)), coords=[('x', np.arange(3)), 
        ...                                                        ('y', np.arange(2))])
        >>> doppyo.utils.integrate(A, over_dim='x')
        <xarray.DataArray 'integral' (y: 2)>
        array([-0.20331 , -0.781251])
        Coordinates:
          * y        (y) int64 0 1
    """

    if x is None:
        x = da[over_dim]
    
    if len(x) == 1:
        if dx is None:
            raise ValueError('Must provide dx for integration along dimension with length 1')
        integral = (da * dx).drop(over_dim).squeeze()
    elif method == 'trapz':
        if dx is None:
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
        if dx is None:
            dx1 = x - x.shift(**{over_dim:1})
            dx2 = -(x - x.shift(**{over_dim:-1}))
            dx = dx1.combine_first(dx2)

        if cumulative:
            integral = (da * dx).cumsum(over_dim, skipna=False) 
        else:
            integral = (da * dx).sum(over_dim, skipna=False) 
    else:
        raise ValueError(f'{method} is not a recognised integration method')
    
    return integral.where(da.sum(over_dim, skipna=False).notnull()).rename('integral')
    

# ===================================================================================================
def calc_difference(data_1, data_2):
    """ Returns the difference of two fields """
    
    return data_1 - data_2


# ===================================================================================================
def calc_division(data_1, data_2):
    """ Returns the division of two fields """
    
    return data_1 / data_2


# ===================================================================================================
def calc_average(da, dim=None, weights=None):
    """
        Returns the weighted average

        Shape of weights must be broadcastable to shape of da
    """

    if weights is None:
        return da.mean(dim)
    else:
        weights = (0 * da + 1) * weights
        return (da * weights).sum(dim) / weights.sum(dim)


# ===================================================================================================
def calc_fft(da, dim, nfft=None, dx=None, twosided=False, shift=True):
    """
        Returns the sequentual ffts of the provided array along the specified dimensions

        da : xarray.DataArray
            Array from which compute the fft
        dim : str or sequence
            Dimensions along which to compute the fft
        nfft : float or sequence, optional
            Number of points in each dimensions=to use in the transformation. If None, the full length
            of each dimension is used.
        dx : float or sequence, optional
            Define the spacing of the dimensions. If None, the spacing is computed directly from the 
            coordinates associated with the dimensions.
        twosided : bool, optional
            When the DFT is computed for purely real input, the output is Hermitian-symmetric, 
            meaning the negative frequency terms are just the complex conjugates of the corresponding 
            positive-frequency terms, and the negative-frequency terms are therefore redundant.
            If True, force the fft to include negative and positive frequencies, even if the input 
            data is real.
        shift : bool, optional
            If True, the frequency axes are shifted to center the 0 frequency, otherwise negative 
            frequencies follow positive frequencies as in numpy.fft.ftt

        A real fft is performed over the first dimension, which is faster. The transforms over the 
        remaining dimensions are then computed with the classic fft.

        If the input array is complex, one must set twosided = True
        
        If dx is a time array, frequencies are computed in Hz
    """

    if isinstance(dim, str):
        dim = [dim]   
    if nfft is not None and not isinstance(nfft, (list,)):
        nfft = [nfft]
    if dx is not None and not isinstance(dx, (list,)):
        dx = [dx]

    # Build nfft and dx into dictionaries -----
    nfft_n = dict()
    for i, di in enumerate(dim):
        try:
            nfft_n[di] = nfft[i]
        except TypeError:
            nfft_n[di] = len(da[di])
    dx_n = dict()
    for i, di in enumerate(dim):
        try:
            dx_n[di] = dx[i]
        except TypeError:
            diff = da[di].diff(di)
            if is_datetime(da[di].values):
                # Drop differences on leap days so that still works with 'noleap' calendars -----
                diff = diff.where(((diff[di].dt.month != 3) | (diff[di].dt.day != 1)), drop=True)
                
            if np.all(diff == diff[0]):
                if is_datetime(da[di].values):
                    dx_n[di] = diff.values[0] / np.timedelta64(1, 's')
                else:
                    dx_n[di] = diff.values[0]
            else:
                raise ValueError(f'Coordinate {di} must be regularly spaced to compute fft')    
    
    # Initialise fft data, dimensions and coordinates -----
    fft_array = da.data
    fft_coords = dict()
    fft_dims = tuple()
    for di in da.dims:
        if di not in dim:
            fft_dims += (di,)
            fft_coords[di] = da[di].values
        else:
            fft_dims += ('f_' + di,)

    # Loop over dimensions and perform fft -----
    # Auto-rechunk -----
    # chunks = copy.copy(fft_array.chunks)
    first = True
    for di in dim:
        if di in da.dims:
            axis_num = da.get_axis_num(di)

            if first and not twosided:
                # The first FFT is performed on real numbers: the use of rfft is faster -----
                fft_coords['f_' + di] = np.fft.rfftfreq(nfft_n[di], dx_n[di])
                if isinstance(fft_array, dask_array_type):
                    fft_array = dask.array.fft.rfft(fft_array, n=nfft_n[di], axis=axis_num)
                else:
                    fft_array = np.fft.rfft(fft_array, n=nfft_n[di], axis=axis_num)
                # Auto-rechunk -----
                # fft_array = dask.array.fft.rfft(fft_array.rechunk({axis_num: nfft_n[di]}),
                #                                 n=nfft_n[di],
                #                                 axis=axis_num).rechunk({axis_num: chunks[axis_num][0]})
            else:
                # The successive FFTs are performed on complex numbers: need to use classic fft -----
                fft_coords['f_' + di] = np.fft.fftfreq(nfft_n[di], dx_n[di])
                if isinstance(fft_array, dask_array_type):
                    fft_array = dask.array.fft.fft(fft_array, n=nfft_n[di], axis=axis_num)
                    if shift:
                        fft_coords['f_' + di] = np.fft.fftshift(fft_coords['f_' + di])
                        fft_array = dask.array.fft.fftshift(fft_array, axes=axis_num)
                else:
                    fft_array = np.fft.fft(fft_array, n=nfft_n[di], axis=axis_num)
                    if shift:
                        fft_coords['f_' + di] = np.fft.fftshift(fft_coords['f_' + di])
                        fft_array = np.fft.fftshift(fft_array, axes=axis_num)
                # Auto-rechunk -----
                # fft_array = dask.array.fft.fft(fft_array.rechunk({axis_num: nfft_n[di]}),
                #                                n=nfft_n[di],
                #                                axis=axis_num).rechunk({axis_num: chunks[axis_num][0]})

            first = False

        else:
            raise ValueError(f'Cannot find dimension {di} in DataArray')

    return xr.DataArray(fft_array, coords=fft_coords, dims=fft_dims, name='fft')


# ===================================================================================================
def calc_ifft(da, dim, nifft=None, shifted=True):
    """
        Returns the sequentual iffts of the provided array along the specified dimensions. 
        
        Note, it is not possible to reconstruct the dimension along which the fft was performed (r_dim) 
        from knowledge only of the fft "frequencies" (f_dim). For example, time cannot be reconstructed 
        from frequency. Here, r_dim is defined relative to 0 in steps of dx as determined from f_dim. It
        may be necessary for the user to use the original (pre-fft) dimension to redefine r_dim after the
        ifft is performed.

        da : xarray.DataArray
            Array from which compute the ifft
        dim : str or sequence
            Dimensions along which to compute the ifft
        nifft : float or sequence, optional
            Number of points in each dimensions to use in the transformation. If None, the full length
            of each dimension is used.
        shifted : bool, optional
            If True, assumes that the frequency axes are shifted to center the 0 frequency, otherwise 
            assumes negative frequencies follow positive frequencies as in numpy.fft.ftt
    """

    if isinstance(dim, str):
        dim = [dim]   
    if nifft is not None and not isinstance(nifft, (list,)):
        nifft = [nifft]

    # Build nifft into a dictionary -----
    nifft_n = dict()
    for i, di in enumerate(dim):
        try:
            nifft_n[di] = nifft[i]
        except TypeError:
            nifft_n[di] = len(da[di])
    
    # Initialise ifft data, dimensions and coordinates -----
    ifft_array = da.data
    ifft_coords = dict()
    ifft_dims = tuple()
    for di in da.dims:
        if di not in dim:
            ifft_dims += (di,)
            ifft_coords[di] = da[di].values
        else:
            
            if di[0:2] == 'f_':
                ifft_dims += (di[2:],)
            else:
                ifft_dims += ('r_' + di,)

    # Loop over dimensions and perform ifft -----
    for di in dim:
        if di in da.dims:
            axis_num = da.get_axis_num(di)
                
            nfft = len(da[di])
            
            if isinstance(ifft_array, dask_array_type):
                if shifted:
                    dx = 1 / np.fft.ifftshift(da[di]).values[1] / nfft
                    ifft_array = dask.array.fft.ifftshift(ifft_array, axes=axis_num)
                else:
                    dx = 1 / da[di].values[1] / nfft
                ifft_array = dask.array.fft.ifft(ifft_array, n=nifft_n[di], axis=axis_num)
            else:
                if shifted:
                    dx = 1 / np.fft.ifftshift(da[di])[1] / nfft
                    ifft_array = np.fft.ifftshift(ifft_array, axes=axis_num)
                else:
                    dx = 1 / da[di].values[1] / nfft
                ifft_array = np.fft.ifft(ifft_array, n=nifft_n[di], axis=axis_num)
                
            if di[0:2] == 'f_':
                ifft_coords[di[2:]] = dx * np.linspace(0, nifft_n[di]-1, nifft_n[di])
            else:
                ifft_coords['r_' + di] = dx * np.linspace(0, nifft_n[di]-1, nifft_n[di])

        else:
            raise ValueError(f'Cannot find dimension {di} in DataArray')

    return xr.DataArray(ifft_array, coords=ifft_coords, dims=ifft_dims, name='ifft')


# ===================================================================================================
def fftfilt(da, dim, method, dx, x_cut):
    """
        Spectrally filters da along dimension dim.
        
        da : xarray.DataArray
            Array to filter
        dim : str
            Dimensions along which to filter
        method : str
            'low pass', 'high pass' or 'band pass'
        dx : float
            Define the spacing of the dimension.
        xc : float or array (if method = 'band pass')
            Define the cut-off value(s), e.g. xc = 5*dx
    """

    if not isinstance(dx, (list,)):
        dx = [dx]
    if not isinstance(x_cut, (list,)):
        x_cut = [x_cut]

    if ((method == 'low pass') | (method == 'high pass')) & (len(x_cut) != 1):
        raise ValueError('Only one cut-off value can be specified for "low pass" or "high pass"')
    if (method == 'band pass') & (len(x_cut) != 2):
        raise ValueError('Two cut-off values must be specified for "band pass"')

    freq_cut = 1 / np.array(x_cut)

    dafft = calc_fft(da, dim=dim, dx=dx, twosided=True, shift=False)

    if method == 'low pass':
        danull = dafft.where(abs(dafft['f_'+dim]) <= freq_cut, other=0)
    elif method == 'high pass':
        danull = dafft.where(abs(dafft['f_'+dim]) >= freq_cut, other=0)
    elif method == 'band pass':
        danull = dafft.where((abs(dafft['f_'+dim]) >= np.min(freq_cut)) &
                             (abs(dafft['f_'+dim]) <= np.max(freq_cut)), other=0)
    else:
        raise ValueError('Unrecognised filter method. Choose from "low pass" or "high pass" or "band pass"')

    dafilt = calc_ifft(danull, dim='f_'+dim, shifted=False).real
    dafilt[dim] = da[dim]

    return dafilt


# ===================================================================================================
def stack_times(da):
    da_list = []
    for init_date in da.init_date.values:
        da_list.append(leadtime_to_datetime(da.sel(init_date=init_date)))
    return xr.concat(da_list, dim='time')


# ===================================================================================================
lambda_anomalize = lambda data, clim: datetime_to_leadtime(
                                           anomalize(
                                               leadtime_to_datetime(data),clim))

rescale = lambda da, scale : datetime_to_leadtime(
                                      scale_per_month(
                                          leadtime_to_datetime(da), scale))

def groupby_lead_and_mean(da, over_dims):
    return da.unstack('stacked_init_date_lead_time').groupby('lead_time').mean(over_dims, skipna=True)

def groupby_lead_and_std(da, over_dims):
    return da.unstack('stacked_init_date_lead_time').groupby('lead_time').std(over_dims, skipna=True)

def unstack_and_shift_per_month(da, shift):
    da_us = da.unstack('stacked_init_date_lead_time')
    the_month = np.ndarray.flatten(da_us.month.values)
    the_month = int(np.unique(the_month[~np.isnan(the_month)]))
    return da_us - shift.sel(month=the_month)

def unstack_and_scale_per_month(da, scale):
    da_us = da.unstack('stacked_init_date_lead_time')
    the_month = np.ndarray.flatten(da_us.month.values)
    the_month = int(np.unique(the_month[~np.isnan(the_month)]))
    return da_us * scale.sel(month=the_month)

def scale_per_month(da, scale):
    return da.groupby('time.month') * scale

def normal_mbias_correct(da_biased, da_target, da_target_clim=False):
    """
        Adjusts, per month and lead time, the mean and standard deviation of da_biased to match that of da_target
        
        If da_target_clim is provided, returns both the corrected full field and the anomalies. Otherwise, returns
        only the anomalies
    """
    
    month = (da_biased.init_date.dt.month + da_biased.lead_time) % 12
    month = month.where(month != 0, 12)

    # Correct the mean -----
    da_biased.coords['month'] = month
    try:
        da_biased_mean = da_biased.groupby('month').apply(groupby_lead_and_mean, over_dims=['init_date','ensemble'])
    except ValueError:
        da_biased_mean = da_biased.groupby('month').apply(groupby_lead_and_mean, over_dims='init_date')
    
    if da_target_clim is not False:
        da_target_mean = da_target.groupby('time.month').mean('time')
        
        da_meancorr = da_biased.groupby('month').apply(unstack_and_shift_per_month, \
                                                       shift=(da_biased_mean - da_target_mean)) \
                                      .mean('month', skipna=True)
        da_meancorr['lead_time'] = da_biased['lead_time']
        da_meancorr.coords['month'] = month

        # Compute the corrected anomalies -----
        da_anom_meancorr = da_meancorr.groupby('init_date').apply(lambda_anomalize, clim=da_target_clim)
        da_anom_meancorr.coords['month'] = month
    else:
        da_anom_meancorr = da_biased.groupby('month').apply(unstack_and_shift_per_month, \
                                                            shift=(da_biased_mean)) \
                                      .mean('month', skipna=True)
        da_anom_meancorr['lead_time'] = da_anom_meancorr['lead_time']
        da_anom_meancorr.coords['month'] = month
    
    if da_target_clim is not False:
        da_meancorrr = da_anom_meancorr.groupby('init_date').apply(lambda_anomalize, clim=-da_target_clim)
        return da_meancorr.drop('month'), da_anom_meancorr.drop('month')
    else:
        return da_anom_meancorr.drop('month')
    
def normal_msbias_correct(da_biased, da_target, da_target_clim=False):
    """
        Adjusts, per month and lead time, the mean and standard deviation of da_biased to match that of da_target
        
        If da_target_clim is provided, returns both the corrected full field and the anomalies. Otherwise, returns
        only the anomalies
    """
    
    month = (da_biased.init_date.dt.month + da_biased.lead_time) % 12
    month = month.where(month != 0, 12)

    # Correct the mean -----
    da_biased.coords['month'] = month
    try:
        da_biased_mean = da_biased.groupby('month').apply(groupby_lead_and_mean, over_dims=['init_date','ensemble'])
    except ValueError:
        da_biased_mean = da_biased.groupby('month').apply(groupby_lead_and_mean, over_dims='init_date')
    
    if da_target_clim is not False:
        da_target_mean = da_target.groupby('time.month').mean('time')
        
        da_meancorr = da_biased.groupby('month').apply(unstack_and_shift_per_month, \
                                                       shift=(da_biased_mean - da_target_mean)) \
                                      .mean('month', skipna=True)
        da_meancorr['lead_time'] = da_biased['lead_time']
        da_meancorr.coords['month'] = month

        # Compute the corrected anomalies -----
        da_anom_meancorr = da_meancorr.groupby('init_date').apply(lambda_anomalize, clim=da_target_clim)
        da_anom_meancorr.coords['month'] = month
    else:
        da_anom_meancorr = da_biased.groupby('month').apply(unstack_and_shift_per_month, \
                                                            shift=(da_biased_mean)) \
                                      .mean('month', skipna=True)
        da_anom_meancorr['lead_time'] = da_anom_meancorr['lead_time']
        da_anom_meancorr.coords['month'] = month
    
    # Correct the standard deviation -----
    try:
        da_biased_std_tmp = da_anom_meancorr.groupby('month').apply(groupby_lead_and_std, over_dims=['init_date','ensemble'])
    except ValueError:
        da_biased_std_tmp = da_anom_meancorr.groupby('month').apply(groupby_lead_and_std, over_dims='init_date')
    try:
        da_target_std = da_target.sel(lat=da_biased.lat, lon=da_biased.lon).groupby('time.month').std('time')
    except:
        da_target_std = da_target.groupby('time.month').std('time')
        
    da_anom_stdcorr_tmp = da_anom_meancorr.groupby('month').apply(unstack_and_scale_per_month, \
                                                                  scale=(da_target_std / da_biased_std_tmp)) \
                                              .mean('month', skipna=True)
    da_anom_stdcorr_tmp['lead_time'] = da_biased['lead_time']
    da_anom_stdcorr_tmp.coords['month'] = month
    
    # This will "squeeze" each pdf at each lead time appropriately. However, the total variance across all leads for 
    # a given month will now be incorrect. Thus, we now rescale as a function of month only
    try:
        da_biased_std = stack_times(da_anom_stdcorr_tmp).groupby('time.month').std(['time','ensemble'])
    except ValueError:
        da_biased_std = stack_times(da_anom_stdcorr_tmp).groupby('time.month').std('time')
    da_anom_stdcorr = da_anom_stdcorr_tmp.groupby('init_date').apply(rescale, scale=(da_target_std / da_biased_std))
    
    if da_target_clim is not False:
        da_stdcorr = da_anom_stdcorr.groupby('init_date').apply(lambda_anomalize, clim=-da_target_clim)
        return da_stdcorr.drop('month'), da_anom_stdcorr.drop('month')
    else:
        return da_anom_stdcorr.drop('month')

    
# ===================================================================================================
def conditional_bias_correct(da_cmp, da_ref, over_dims):
    """
        Return conditional bias corrected data using the approach of Goddard et al. 2013
    """

    cc = skill.compute_Pearson_corrcoef(da_cmp.mean('ensemble'), da_ref, over_dims=over_dims, subtract_local_mean=False)
    correct_cond_bias = (da_ref.std(over_dims) / da_cmp.mean('ensemble').std(over_dims)) * cc
    
    return da_cmp * correct_cond_bias

    
# ===================================================================================================
# Climatology tools
# ===================================================================================================
def load_mean_climatology(clim, freq, variable=None, chunks=None, **kwargs):
    """ 
    Returns pre-saved climatology at desired frequency (daily or longer).
    
    Currently available options are: "jra_1958-2016", "cafe_f1_atmos_2003-2017", "cafe_f1_ocean_2003-2017", 
    "cafe_c2_atmos_400-499", "cafe_c2_atmos_500-549", cafe_c2_ocean_400-499", "cafe_c2_ocean_500-549", "HadISST_1870-2018", "REMSS_2002-2018".
    """
    
    data_path = '/OSM/CBR/OA_DCFP/data/intermediate_products/doppyo/mean_climatologies/'
    
    # Load specified dataset -----
    if clim == 'jra_1958-2016':
        data_loc = data_path + 'jra.isobaric.1958010100_2016123118.clim.nc'
        ds = xr.open_dataset(data_loc, chunks=chunks, **kwargs)
            
    elif clim == 'cafe_f1_atmos_2003-2017':
        data_loc = data_path + 'cafe.f1.atmos.2003010112_2017123112.clim.nc'
        ds = xr.open_dataset(data_loc, chunks=chunks, **kwargs)
            
    elif clim == 'cafe_f1_ocean_2003-2017':
        data_loc = data_path + 'cafe.f1.ocean.2003010112_2017123112.clim.nc'
        ds = xr.open_dataset(data_loc, chunks=chunks, **kwargs)
            
    elif clim == 'cafe_c2_atmos_400-499':
        data_loc = data_path + 'cafe.c2.atmos.400_499.clim.nc'
        ds = xr.open_dataset(data_loc, chunks=chunks, **kwargs)
    
    elif clim == 'cafe_c2_atmos_500-549':
        data_loc = data_path + 'cafe.c2.atmos.500_549.clim.nc'
        ds = xr.open_dataset(data_loc, chunks=chunks, **kwargs)
            
    elif clim == 'cafe_c2_ocean_400-499':
        data_loc = data_path + 'cafe.c2.ocean.400_499.clim.nc'
        ds = xr.open_dataset(data_loc, chunks=chunks, **kwargs)
        
    elif clim == 'cafe_c2_ocean_500-549':
        data_loc = data_path + 'cafe.c2.ocean.500_549.clim.nc'
        ds = xr.open_dataset(data_loc, chunks=chunks, **kwargs)
    
    elif clim == 'HadISST_1870-2018':
        data_loc = data_path + 'hadisst.1870011612_2018021612.clim.nc'
        ds = xr.open_dataset(data_loc, chunks=chunks, **kwargs)
    
    elif clim == 'REMSS_2002-2018':
        data_loc = data_path + 'remss.2002060112_2018041812.clim.nc'
        ds = xr.open_dataset(data_loc, chunks=chunks, **kwargs)
            
    else:
        raise ValueError(f'"{clim}" is not an available climatology. Available options are "jra_1958-2016", "cafe_f1_atmos_2003-2017", "cafe_f1_ocean_2003-2017", "cafe_c2_atmos_400-499", "cafe_c2_atmos_500-549", "cafe_c2_ocean_400-499", "cafe_c2_ocean_500-549", "HadISST_1870-2018","REMSS_2002-2018"')
        
    if variable is not None:
        try:
            ds = ds[variable]
        except KeyError:
            raise ValueError(f'"{variable}" is not a variable in "{clim}"')
    
    # Resample if required -----    
    load_freq = pd.infer_freq(ds['time'].values)
    if load_freq != freq:
        if variable == 'precip':
            ds = ds.resample(time=freq).sum(dim='time')
        else:
            ds = ds.resample(time=freq).mean(dim='time')
        ds = ds.chunk(chunks=chunks)

    return ds


# ===================================================================================================
def anomalize(data, clim):
    """ 
        Receives raw and climatology data at matched frequencies and returns the anomaly 
    """
    
    data_use = data.copy(deep=True)
    clim_use = clim.copy(deep=True)
    
    # If clim is saved on a time dimension, deal with accordingly ----- 
    if 'time' in clim_use.dims:
        # Find frequency (assume this is annual average if only one time value exists) -----
        if len(clim_use.time) > 1:
            clim_freq = pd.infer_freq(clim_use.time.values[:3])
        else:
            clim_freq = 'A'
            
        # Build daily, monthly or annual climatologies -----
        if 'D' in clim_freq:
            # Contruct month-day array (to deal with leap years) -----
            clim_mon = np.array([str(i).zfill(2) + '-' for i in clim_use.time.dt.month.values])
            clim_day = np.array([str(i).zfill(2)  for i in clim_use.time.dt.day.values])
            clim_use['time'] = np.core.defchararray.add(clim_mon, clim_day)
            
            clim_use = clim_use.groupby('time', squeeze=False).mean(dim='time')
            deal_with_leap = True
        elif 'M' in clim_freq:
            clim_use = clim_use.groupby('time.month', squeeze=False).mean(dim='time')
        elif ('A' in clim_freq) | ('Y' in clim_freq):
            clim_use = prune(clim_use.groupby('time.year', squeeze=False).mean(dim='time').squeeze())
    elif 'dayofyear' in clim_use.dims:
        clim_freq = 'D'
        deal_with_leap = False
    elif 'month' in clim_use.dims:
        clim_freq = 'M'
    else:
        warnings.warn('Unable to determine frequency of climatology DataArray, assuming annual average')
        clim_freq = 'A'
    
    # Subtract the climatology from the full field -----
    if ('D' in clim_freq) and (deal_with_leap is True):
        time_keep = data_use.time

        # Contruct month-day arrays -----
        data_mon = np.array([str(i).zfill(2) + '-' for i in data_use.time.dt.month.values])
        data_day = np.array([str(i).zfill(2)  for i in data_use.time.dt.day.values])
        data_use['time'] = np.core.defchararray.add(data_mon, data_day)

        anom = data_use.groupby('time') - clim_use
        anom['time'] = time_keep
    elif ('D' in clim_freq) and (deal_with_leap is False):
        anom = data_use.groupby('time.dayofyear') - clim_use
    elif 'M' in clim_freq:
        anom = data_use.groupby('time.month') - clim_use
    elif ('A' in clim_freq) | ('Y' in clim_freq):
        anom = data_use - clim_use
        
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
    
    return np.datetime64(date_out,'ns')


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
    try:
        freq = pd.infer_freq(data_in.time.values)
        
        # If pandas tries to assign start time to frequency (e.g. QS-OCT), remove this -----
        if '-' in freq:
            freq = freq[:freq.find('-')]
        
        # Split frequency into numbers and strings -----
        incr_string = ''.join([i for i in freq if i.isdigit()])
        freq_incr = [int(incr_string) if incr_string else 1][0]
        freq_type = ''.join([i for i in freq if not i.isdigit()])
    
        # Specify all lengths great than 1 month in months -----
        if 'QS' in freq_type:
            freq = str(3*freq_incr) + 'MS'
        elif 'Q' in freq_type:
            freq = str(3*freq_incr) + 'M'
        elif ('YS' in freq_type) | ('AS' in freq_type):
            freq = str(12*freq_incr) + 'MS'
        elif ('Y' in freq_type) | ('A' in freq_type):
            freq = str(12*freq_incr) + 'M'
            
    except ValueError:
        dt = (data_in.time.values[1] - data_in.time.values[0]) / np.timedelta64(1, 's')
        month = data_in.time.dt.month[0]
        if dt == 60*60*24:
            freq = 'D'
        elif ((month == 1) | (month == 3) | (month == 5) | (month == 7) | (month == 8) | (month == 10) | 
               (month == 12)) & (dt == 31*60*60*24):
            freq = 'MS'
        elif ((month == 4) | (month == 6) | (month == 9) | (month == 11)) & (dt == 30*60*60*24):
            freq = 'MS'
        elif (month == 2) & ((dt == 28*60*60*24) | (dt == 29*60*60*24)):  
            freq = 'MS'
        elif (dt == 365*60*60*24) | (dt == 366*60*60*24):
            freq = 'A'
        else:
            freq = 'NA'

    data_out = data_in.rename({'time' : 'lead_time'})
    data_out['lead_time'] = lead_times
    data_out['lead_time'].attrs['units'] = freq

    data_out.coords['init_date'] = init_date
    
    return data_out


# ===================================================================================================
def repeat_data(data, repeat_dim, index_to_repeat=0):
    """ 
    Returns object the same sizes as data, but with data at index repeat_dim = index_to_repeat repeated 
    across all other entries in repeat_dim
    """

    repeat_data = data.loc[{repeat_dim : index_to_repeat}].drop(repeat_dim).squeeze()
    
    return (0 * data).groupby(repeat_dim,squeeze=True).apply(calc_difference, data_2=-repeat_data)


# ===================================================================================================
def get_latlon_region(da, box):
    '''
        Returns da over a provided lat-lon region, where box = [lat_min, lat_max, lon_min, lon_max]
    '''

    # Account for datasets with negative longitudes -----
    if np.any(da['lon'] < 0):
        lons = da['lon'].values
        lons_pos = np.where(lons < 0, lons+360, lons)
        idx = np.where((lons_pos >= box[2]) & (lons_pos <= box[3]))[0]
        if np.all(np.diff(idx)[0] == np.diff(idx)):
            return da.sel(lat=slice(box[0],box[1])).isel(lon=slice(idx[0],idx[-1]))
        else:
            return da.sel(lat=slice(box[0],box[1])).isel(lon=idx)
    else:
        return da.sel(lat=slice(box[0],box[1]), lon=slice(box[2],box[3]))

# ===================================================================================================
def calc_boxavg_latlon(da, box):
    '''
        Returns the average of a given quantity over a provide lat-lon region, where box = [lat_min,
        lat_max, lon_min, lon_max]
    '''
    
    return get_latlon_region(da, box).mean(dim=['lat', 'lon'])


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
def get_bin_edges(bins):
    ''' Returns bin edges of provided bins '''
    
    dbin = np.diff(bins)/2
    bin_edges = np.concatenate(([bins[0]-dbin[0]], 
                                 bins[:-1]+dbin, 
                                 [bins[-1]+dbin[-1]]))
    
    return bin_edges


# ===================================================================================================
def is_datetime(value):
    """ Return True or False depending on whether input is datetime64 or not """
    
    return pd.api.types.is_datetime64_dtype(value)


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

# ===================================================================================================
def get_lon_name(da):
    """ Returns name of longitude coordinate in da """
    
    if 'lon' in da.dims:
        return 'lon'
    elif 'lon_2' in da.dims:
        return 'lon_2'
    elif 'xt_ocean' in da.dims:
        return 'xt_ocean'
    else:
        raise KeyError('Unable to determine longitude dimension')
        pass


# ===================================================================================================
def get_lat_name(da):
    """ Returns name of latitude coordinate in da """
    
    if 'lat' in da.dims:
        return 'lat'
    elif 'lat_2' in da.dims:
        return 'lat_2'
    elif 'yt_ocean' in da.dims:
        return 'yt_ocean'
    else:
        raise KeyError('Unable to determine latitude dimension')
        pass

    
# ===================================================================================================
def get_depth_name(da):
    """ Returns name of ocean depth coordinate in da """
    
    if 'depth' in da.dims:
        return 'depth'
    elif 'depth_coord' in da.dims:
        return 'depth_coord'
    elif 'st_ocean' in da.dims:
        return 'st_ocean'
    else:
        raise KeyError('Unable to determine depth dimension')
        pass
    
# ===================================================================================================
def get_level_name(da):
    """ Returns name of level coordinate in da """
    
    if 'level' in da.dims:
        return 'level'
    else:
        raise KeyError('Unable to determine level dimension')
        pass
    
    
# ===================================================================================================
def get_plevel_name(da):
    """ Returns name of pressure level coordinate in da """
    
    if 'level' in da.dims:
        return 'level'
    else:
        raise KeyError('Unable to determine pressure level dimension')
        pass
    
    
# ===================================================================================================
def get_pres_name(da):
    """ Returns name of pressure coordinate in da """
    
    if 'pfull' in da.dims:
        return 'pfull'
    elif 'phalf' in da.dims:
        return 'phalf'
    else:
        raise KeyError('Unable to determine pressure dimension')
        pass


# ===================================================================================================
def cftime_to_datetime64(time,shift_year=0):
    """ 
    Convert cftime object to datetime64 object
    
    Assumes times are sequential
    """

    if (time.values[0].timetuple()[0]+shift_year < 1678) | (time.values[-1].timetuple()[0]+shift_year > 2261):
        raise ValueError('Cannot create datetime64 object for years outside on 1678-2262')
        
    return np.array([np.datetime64(time.values[i].replace(year=time.values[i].timetuple()[0]+shift_year) \
                                                 .strftime(), 'ns') \
                                                 for i in range(len(time))])


# ===================================================================================================
# visualization tools
# ===================================================================================================
def plot_fields(data, title, headings, vmin, vmax, cmin=None, cmax=None, ncol=2, mult_row=1, 
                mult_col=1, mult_cshift=1, mult_cbar=1, contour=False, cmap='viridis', fontsize=12, invert=False):
    """ Plots tiles of figures """
    
    matplotlib.rc('font', family='sans-serif')
    matplotlib.rc('font', serif='Helvetica') 
    matplotlib.rc('text', usetex='false') 
    matplotlib.rcParams.update({'font.size': fontsize})

    nrow = int(np.ceil(len(data)/ncol));

    fig = plt.figure(figsize=(11*mult_col, nrow*4*mult_row))
        
    count = 1
    for idx,dat in enumerate(data):
        if ('lat' in dat.dims) and ('lon' in dat.dims):
            trans = cartopy.crs.PlateCarree()
            ax = plt.subplot(nrow, ncol, count, projection=cartopy.crs.PlateCarree(central_longitude=180))
            extent = [dat.lon.min(), dat.lon.max(), 
                      dat.lat.min(), dat.lat.max()]

            if contour is True:
                if cmin is not None:
                    ax.coastlines(color='gray')
                    im = ax.contourf(dat.lon, dat.lat, dat, np.linspace(vmin,vmax,12), origin='lower', transform=trans, 
                                  vmin=vmin, vmax=vmax, cmap=cmap, extend='both')
                    ax.contour(dat.lon, dat.lat, dat, np.linspace(cmin,cmax,12), origin='lower', transform=trans,
                              colors='w', linewidths=2)
                    ax.contour(dat.lon, dat.lat, dat, np.linspace(cmin,cmax,12), origin='lower', transform=trans,
                              colors='k', linewidths=1)
                else:
                    ax.coastlines(color='black')
                    im = ax.contourf(dat.lon, dat.lat, dat, np.linspace(vmin,vmax,20), origin='lower', transform=trans, 
                                  vmin=vmin, vmax=vmax, cmap=cmap, extend='both')
            else:
                im = ax.imshow(dat, origin='lower', extent=extent, transform=trans, vmin=vmin, vmax=vmax, cmap=cmap)

            gl = ax.gridlines(crs=cartopy.crs.PlateCarree(), draw_labels=True)
            gl.xlines = False
            gl.ylines = False
            gl.xlabels_top = False
            if count % ncol == 0:
                gl.ylabels_left = False
            elif (count+ncol-1) % ncol == 0: 
                gl.ylabels_right = False
            else:
                gl.ylabels_left = False
                gl.ylabels_right = False
            gl.xlocator = mticker.FixedLocator([-90, 0, 90, 180])
            gl.ylocator = mticker.FixedLocator([-90, -60, 0, 60, 90])
            gl.xformatter = LONGITUDE_FORMATTER
            gl.yformatter = LATITUDE_FORMATTER
            ax.set_title(headings[idx], fontsize=fontsize)
        else:
            ax = plt.subplot(nrow, ncol, count)
            if 'lat' in dat.dims:
                x_plt = dat['lat']
                y_plt = dat[find_other_dims(dat,'lat')[0]]
                # if dat.get_axis_num('lat') > 0:
                #     dat = dat.transpose()
            elif 'lon' in dat.dims:
                x_plt = dat['lon']
                y_plt = dat[find_other_dims(dat,'lon')[0]]
                # if dat.get_axis_num('lon') > 0:
                #     dat = dat.transpose()
            else: 
                x_plt = dat[dat.dims[1]]
                y_plt = dat[dat.dims[0]]
                
            extent = [x_plt.min(), x_plt.max(), 
                      y_plt.min(), y_plt.max()]
            
            if contour is True:
                if cmin is not None:
                    im = ax.contourf(x_plt, y_plt, dat, levels=np.linspace(vmin,vmax,12), vmin=vmin, 
                                     vmax=vmax, cmap=cmap, extend='both')
                    ax.contour(x_plt, y_plt, dat, levels=np.linspace(cmin,cmax,12), colors='w', linewidths=2)
                    ax.contour(x_plt, y_plt, dat, levels=np.linspace(cmin,cmax,12), colors='k', linewidths=1)
                else:
                    im = ax.contourf(x_plt, y_plt, dat, levels=np.linspace(vmin,vmax,20), vmin=vmin, 
                                     vmax=vmax, cmap=cmap, extend='both')
            else:
                im = ax.imshow(dat, origin='lower', extent=extent, vmin=vmin, vmax=vmax, cmap=cmap)
                
            if count % ncol == 0:
                ax.yaxis.tick_right()
            elif (count+ncol-1) % ncol == 0: 
                ax.set_ylabel(y_plt.dims[0], fontsize=fontsize)
            else:
                ax.set_yticks([])
            if idx / ncol >= nrow - 1:
                ax.set_xlabel(x_plt.dims[0], fontsize=fontsize)
            ax.set_title(headings[idx], fontsize=fontsize)
            
            if invert:
                ax.invert_yaxis()

        count += 1

    plt.tight_layout()
    fig.subplots_adjust(bottom=mult_cshift*0.16)
    cbar_ax = fig.add_axes([0.15, 0.13, 0.7, mult_cbar*0.020])
    cbar = fig.colorbar(im, cax=cbar_ax, orientation='horizontal', extend='both');
    cbar_ax.set_xlabel(title, rotation=0, labelpad=15, fontsize=fontsize);
    cbar.set_ticks(np.linspace(vmin,vmax,5))
    
    
# ===================================================================================================
def size_GB(xr_object):
    """
    How many GB (or GiB) is your xarray object?
    
    // Requires an xarray object
        
    // Returns:
    * equivalent GB (GBytes) - 10^9 conversion
    * equivalent GiB (GiBytes) - 2^ 30 conversion
        
    < Thomas Moore - thomas.moore@csiro.au - 10102018 >
    """ 
    bytes = xr_object.nbytes
    Ten2the9 = 10**9
    Two2the30 = 2**30
    GBytes = bytes / Ten2the9
    GiBytes = bytes / Two2the30
    
    #print out results
    print(xr_object.name, "is", GBytes, "GB", 'which is', GiBytes,"GiB")
    
    
    return GBytes,GiBytes
