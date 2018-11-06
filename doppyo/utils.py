"""
    General support functions for the doppyo package
    Authors: Dougie Squire & Thomas Moore
    Date created: 04/04/2018
    Python Version: 3.6
"""

__all__ = ['timer', 'constants', 'skewness', 'kurtosis', 'digitize', 'pdf', 'cdf', 'histogram', 
           'get_bin_edges', 'differentiate_wrt', 'xy_from_lonlat', 'integrate', 'add', 'subtract', 
           'multiply','divide', 'average', 'fft', 'ifft', 'fftfilt', 'isosurface',
           'load_mean_climatology', 'anomalize', 'trunc_time', 'leadtime_to_datetime', 
           'datetime_to_leadtime', 'repeat_datapoint', 'get_latlon_region', 'latlon_average', 
           'stack_by_init_date', 'concat_times', 'prune', 'get_other_dims', 'cftime_to_datetime64', 
           'get_time_name', 'get_lon_name', 'get_lat_name', 'get_depth_name', 'get_level_name', 
           'get_plevel_name', '_is_datetime']

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
        >>> bins = np.linspace(-2,2,10)
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
        >>> bins = np.linspace(-2,2,10)
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
        >>> bins = np.linspace(-2,2,10)
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
        >>> bins = np.linspace(-2,2,10)
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

    if over_dims is None:
        over_dims = []
    bins = (bin_edges[0:-1]+bin_edges[1:]) / 2
    
    # Replace nans with a value not in any bin (np.histogram has difficulty with nans) -----
    replace_val = 1000 * max(bin_edges) 
    da = da.copy().fillna(replace_val)
    
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
        
        group = da.groupby_bins(da, bins=bin_edges, squeeze=False)
        
        if list(group) == []:
            raise ValueError('Input array must contain at least one element that falls in a bin')
        else:
            hist =  group.apply(_unstack_and_count, dims=over_dims) \
                         .fillna(0) \
                         .rename({'histogram_bins' : 'bins'})
            hist['bins'] = (bin_edges[0:-1]+bin_edges[1:]) / 2
    
    # Add nans where data did not fall in any bin -----
    return hist.astype(int).where(hist.sum('bins') != 0).rename('histogram')


# ===================================================================================================
def get_bin_edges(bins):
    """ 
        Returns bin edges of provided bins 
        Author: Dougie Squire
        Date: 06/03/2018
        
        Parameters
        ----------
        bins : array_like
            One-dimensional array of bin values to compute bin edges
        
        Returns
        -------
        edges : array_like
            Array of bin edges where the first and last edge are computed using the spacing between
            the first-and-second and second-last-and-last bins, respectively. This array is one
            element larger than the input array
            
        Examples
        --------
        >>> bins = np.linspace(-2,2,10)
        >>> bin_edges = doppyo.utils.get_bin_edges(bins)
        array([-2.5, -1.5, -0.5,  0.5,  1.5,  2.5])
    """
    
    dbin = np.diff(bins)/2
    bin_edges = np.concatenate(([bins[0]-dbin[0]], 
                                 bins[:-1]+dbin, 
                                 [bins[-1]+dbin[-1]]))
    
    return bin_edges


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
        Author: Dougie Squire
        Date: 02/11/2018
        
        Parameters
        ----------
        da : xarray DataArray
            Array containing values to differentiate
        dim : str
            The dimension to be used to compute the gradient
        x : xarray DataArray
            Array containing values to differentiate with respect to. Must be broadcastable da
            
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
    """
        Returns x/y in m from grid points that are in a longitude/latitude format.
        Author: Dougie Squire
        Date: 01/11/2018
        
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
    """
    
    degtorad = constants().pi / 180
    
    y = (2 * constants().pi * constants().R_earth * lat / 360)
    x = 2 * constants().pi * constants().R_earth * xr.ufuncs.cos(lat * degtorad) * lon / 360
    y = y * (0 * x + 1)
    
    return x, y


# ===================================================================================================
def integrate(da, over_dim, x=None, dx=None, method='trapz', cumulative=False):
    """ 
        Returns trapezoidal/rectangular integration along specified dimension 
        Author: Dougie Squire
        Date: 16/08/2018
        
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
def add(data_1, data_2):
    """ 
        Returns the addition of two arrays, data_1 + data_2. Useful for xr.apply type operations
        Author: Dougie Squire
        Date: 27/06/2018

        Parameters
        ----------
        data_1 : array_like
            The first array
        data_2 : array_like 
            The second array

        Returns
        -------
        addition : array_like
            The addition of data_1 and data_2

        Examples
        --------
        >>> A = xr.DataArray(np.random.normal(size=(3,2)), coords=[('x', np.arange(3)), ('y', np.arange(2))])
        >>> B = xr.DataArray(np.random.normal(size=(3,2)), coords=[('x', np.arange(3)), ('y', np.arange(2))])
        >>> doppyo.utils.add(A,B)
        <xarray.DataArray (x: 3, y: 2)>
        array([[-0.333176,  0.344428],
               [ 0.629463,  0.515872],
               [ 1.121926,  0.567797]])
        Coordinates:
          * x        (x) int64 0 1 2
          * y        (y) int64 0 1
    """
    
    return data_1 - data_2

# ===================================================================================================
def subtract(data_1, data_2):
    """ 
        Returns the difference of two arrays, data_1 - data_2. Useful for xr.apply type operations
        Author: Dougie Squire
        Date: 27/06/2018

        Parameters
        ----------
        data_1 : array_like
            The first array
        data_2 : array_like 
            The second array

        Returns
        -------
        subtraction : array_like
            The difference between data_1 and data_2

        Examples
        --------
        >>> A = xr.DataArray(np.random.normal(size=(3,2)), coords=[('x', np.arange(3)), ('y', np.arange(2))])
        >>> B = xr.DataArray(np.random.normal(size=(3,2)), coords=[('x', np.arange(3)), ('y', np.arange(2))])
        >>> doppyo.utils.subtract(A,B)
        <xarray.DataArray (x: 3, y: 2)>
        array([[-0.265376,  1.331496],
               [ 1.065077, -1.278974],
               [ 3.691209, -1.928883]])
        Coordinates:
          * x        (x) int64 0 1 2
          * y        (y) int64 0 1
    """
    
    return data_1 - data_2


# ===================================================================================================
def multiply(data_1, data_2):
    """ 
        Returns the multiplication of two fields, data_1 * data_2. Useful for xr.apply type operations
        Author: Dougie Squire
        Date: 27/06/2018
        
        Parameters
        ----------
        data_1 : array_like
            The first array
        data_2 : array_like 
            The second array
            
        Returns
        -------
        multiplication : array_like
            The multiplication of data_1 and data_2
            
        Examples
        --------
        >>> A = xr.DataArray(np.random.normal(size=(3,2)), coords=[('x', np.arange(3)), ('y', np.arange(2))])
        >>> B = xr.DataArray(np.random.normal(size=(3,2)), coords=[('x', np.arange(3)), ('y', np.arange(2))])
        >>> doppyo.utils.multiply(A,B)
        <xarray.DataArray (x: 3, y: 2)>
        array([[-0.219773,  0.235889],
               [-0.529542, -1.30342 ],
               [-1.048924,  0.20482 ]])
        Coordinates:
          * x        (x) int64 0 1 2
          * y        (y) int64 0 1
    """
    
    return data_1 * data_2


# ===================================================================================================
def divide(data_1, data_2):
    """ 
        Returns the division of two fields, data_1 / data_2. Useful for xr.apply type operations
        Author: Dougie Squire
        Date: 27/06/2018
        
        Parameters
        ----------
        data_1 : array_like
            The first array
        data_2 : array_like 
            The second array
            
        Returns
        -------
        division : array_like
            The division of data_1 by data_2
            
        Examples
        --------
        >>> A = xr.DataArray(np.random.normal(size=(3,2)), coords=[('x', np.arange(3)), ('y', np.arange(2))])
        >>> B = xr.DataArray(np.random.normal(size=(3,2)), coords=[('x', np.arange(3)), ('y', np.arange(2))])
        >>> doppyo.utils.divide(A,B)
        <xarray.DataArray (x: 3, y: 2)>
        array([[-0.310139,  0.071369],
               [-0.647227, -0.427525],
               [-0.179623,  1.229811]])
        Coordinates:
          * x        (x) int64 0 1 2
          * y        (y) int64 0 1
    """
    
    return data_1 / data_2


# ===================================================================================================
def average(da, dim=None, weights=None):
    """
        Returns the weighted average
        Author: Dougie Squire
        Date: 06/08/2018

        Parameters
        ----------
        da : xarray DataArray
            Array to be averaged
        dim : str or sequence of str, optional
            Dimension(s) over which to compute weighted average. If None, average is computed over all
            dimensions
        weights : xarray DataArray, optional
            Weights to apply during averaging. Shape of weights must be broadcastable to shape of da.
            If None, unity weighting is applied
            
        Returns
        -------
        weighted : xarray DataArray
            Weighted average of input array along specified dimensions
            
        Examples
        --------
        >>> A = xr.DataArray(np.random.normal(size=(4,4)), coords=[('lat', np.arange(-90,90,45)), 
        ...                                                        ('lon', np.arange(0,360,90))])
        >>> degtorad = doppyo.utils.constants().pi / 180
        >>> cos_lat = xr.ufuncs.cos(A['lat'] * degtorad) 
        >>> doppyo.utils.average(A, dim='lat', weights=cos_lat)
        <xarray.DataArray (lon: 4)>
        array([-0.473632, -0.241208, -0.954826,  0.498559])
        Coordinates:
          * lon      (lon) int64 0 90 180 270
    """

    if weights is None:
        return da.mean(dim)
    else:
        weights = (0 * da + 1) * weights
        return (da * weights).sum(dim) / weights.sum(dim)


# ===================================================================================================
def fft(da, dim, nfft=None, dx=None, twosided=False, shift=True):
    """
        Returns the sequentual ffts of the provided array along the specified dimensions
        Author: Dougie Squire
        Date: 06/08/2018
        
        Parameters
        ----------
        da : xarray.DataArray
            Array from which compute the fft
        dim : str or sequence
            Dimensions along which to compute the fft
        nfft : float or sequence, optional
            Number of points in each dimensions to use in the transformation. If None, the full length
            of each dimension is used.
        dx : float or sequence, optional
            Define the spacing of the dimensions. If None, the spacing is computed directly from the 
            coordinates associated with the dimensions. If dx is a time array, frequencies are computed 
            in Hz
        twosided : bool, optional
            When the DFT is computed for purely real input, the output is Hermitian-symmetric, 
            meaning the negative frequency terms are just the complex conjugates of the corresponding 
            positive-frequency terms, and the negative-frequency terms are therefore redundant.
            If True, force the fft to include negative and positive frequencies, even if the input 
            data is real. If the input array is complex, one must set twosided=True
        shift : bool, optional
            If True, the frequency axes are shifted to center the 0 frequency, otherwise negative 
            frequencies follow positive frequencies as in numpy.fft.ftt

        Returns
        -------
        fft : xarray DataArray
            Array containing the sequentual ffts of the provided array along the specified dimensions
        
        Examples
        --------
        >>> A = xr.DataArray(np.random.normal(size=(4,4)), 
        ...                  coords=[('lat', np.arange(-90,90,45)), 
        ...                          ('time', pd.date_range(start='1/1/2018', periods=4, freq='D'))])
        >>> doppyo.utils.fft(A, dim='time', twosided=True, shift=True)
        <xarray.DataArray 'fft' (lat: 4, f_time: 4)>
        array([[ 2.996572+0.j      , -2.833156-0.676355j, -0.038218+0.j      ,
                -2.833156+0.676355j],
               [-0.66788 +0.j      ,  0.551732-3.406326j,  2.003329+0.j      ,
                 0.551732+3.406326j],
               [ 2.032978+0.j      ,  0.657454+1.703941j,  2.085695+0.j      ,
                 0.657454-1.703941j],
               [ 0.462405+0.j      , -0.815011+2.357146j, -1.257371+0.j      ,
                -0.815011-2.357146j]])
        Coordinates:
          * lat      (lat) int64 -90 -45 0 45
          * f_time   (f_time) float64 -5.787e-06 -2.894e-06 0.0 2.894e-06
  
        See also
        --------
        dask.array.fft
        numpy.fft
        
        Notes
        -----
        A real fft is performed over the first dimension, which is faster. The transforms over the 
        remaining dimensions are then computed with the classic fft.
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
            if _is_datetime(da[di].values):
                # Drop differences on leap days so that still works with 'noleap' calendars -----
                diff = diff.where(((diff[di].dt.month != 3) | (diff[di].dt.day != 1)), drop=True)
                
            if np.all(diff == diff[0]):
                if _is_datetime(da[di].values):
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
def ifft(da, dim, nifft=None, shifted=True):
    """
        Returns the sequentual iffts of the provided array along the specified dimensions. Note, it is 
        not possible to reconstruct the  dimension along which the fft was performed (r_dim) from 
        knowledge only of the fft "frequencies" (f_dim). For example, time cannot be reconstructed from 
        frequency. Here, r_dim is defined relative to 0 in steps of dx as determined from f_dim. It may 
        be necessary for the user to use the original (pre-fft) dimension to redefine r_dim after the
        ifft is performed (see the Examples s ection of this docstring).
        Author: Dougie Squire
        Date: 06/08/2018
        
        Parameters
        ----------
        da : xarray.DataArray
            Array from which compute the ifft
        dim : str or sequence
            Dimensions along which to compute the ifft
        nifft : float or sequence, optional
            Number of points in each dimensions to use in the transformation. If None, the full length
            of each dimension is used.
        shifted : bool, optional
            If True, assumes that the input dimensions are shifted to center the 0 frequency, otherwise 
            assumes negative frequencies follow positive frequencies as in numpy.fft.ftt
            
        Returns
        -------
        ifft : xarray DataArray
            Array containing the sequentual iffts of the provided array along the specified dimensions
            
        Examples
        --------
        >>> A = xr.DataArray(np.random.normal(size=(4,4)), 
        ...                  coords=[('lat', np.arange(-90,90,45)), 
        ...                  ('time', pd.date_range(start='1/1/2018', periods=4, freq='D'))])
        >>> A_fft = doppyo.utils.fft(A, dim=['time', 'lat'], twosided=True, shift=False)
        >>> A_new = doppyo.utils.ifft(A_fft, dim=['f_lat', 'f_time'], shifted=False).real
        >>> print(A_new)
        <xarray.DataArray 'ifft' (lat: 4, time: 4)>
        array([[-0.821396, -0.321925, -0.183761,  1.020338],
               [ 0.147125,  0.17867 ,  0.343659,  1.487173],
               [-1.53012 ,  1.586665, -0.097846,  1.535701],
               [ 0.663949, -0.9256  ,  0.086642,  0.586463]])
        Coordinates:
          * lat      (lat) float64 0.0 45.0 90.0 135.0
          * time     (time) float64 0.0 8.64e+04 1.728e+05 2.592e+05
        >>> A_new['lat'] = A['lat']
        >>> A_new['time'] = A['time']
        >>> print(A_new)
        <xarray.DataArray 'ifft' (lat: 4, time: 4)>
        array([[-0.821396, -0.321925, -0.183761,  1.020338],
               [ 0.147125,  0.17867 ,  0.343659,  1.487173],
               [-1.53012 ,  1.586665, -0.097846,  1.535701],
               [ 0.663949, -0.9256  ,  0.086642,  0.586463]])
        Coordinates:
          * lat      (lat) int64 -90 -45 0 45
          * time     (time) datetime64[ns] 2018-01-01 2018-01-02 2018-01-03 2018-01-04
          
        See also
        --------
        dask.array.fft
        numpy.fft
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
        Spectrally filters the provided array along dimension dim.
        Author: Dougie Squire
        Date: 15/09/2018
        
        Parameters
        ----------
        da : xarray.DataArray
            Array to filter
        dim : str
            Dimension along which to filter
        method : str
            Filter method to use. Options are 'low pass', 'high pass' or 'band pass'
        dx : value
            Define the spacing of the dimension.
        xc : value or array_like (if method = 'band pass')
            Define the filter cut-off value(s), e.g. x_cut = 5*dx
            
        Returns
        -------
        filtered : xarray.DataArray
            Filtered array
            
        Examples
        --------
        >>> A = xr.DataArray(np.random.normal(size=(100)), 
        ...                  coords=[('time', pd.date_range(start='1/1/2018', periods=100, freq='D'))])
        >>> A_filt = doppyo.utils.fftfilt(A, dim='time', method='low pass', dx=1, x_cut=10)
        >>> print(A_filt)
        <xarray.DataArray 'filtered' (time: 1000)>
        array([ 0.120893,  0.059256, -0.085101, ..., -0.351555, -0.112201,  0.061701])
        Coordinates:
          * time     (time) datetime64[ns] 2018-01-01 2018-01-02 ... 2020-09-26
        >>> A.plot()
        >>> A_filt.plot()
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

    dafft = fft(da, dim=dim, dx=dx, twosided=True, shift=False)

    if method == 'low pass':
        danull = dafft.where(abs(dafft['f_'+dim]) <= freq_cut, other=0)
    elif method == 'high pass':
        danull = dafft.where(abs(dafft['f_'+dim]) >= freq_cut, other=0)
    elif method == 'band pass':
        danull = dafft.where((abs(dafft['f_'+dim]) >= np.min(freq_cut)) &
                             (abs(dafft['f_'+dim]) <= np.max(freq_cut)), other=0)
    else:
        raise ValueError('Unrecognised filter method. Choose from "low pass" or "high pass" or "band pass"')

    dafilt = ifft(danull, dim='f_'+dim, shifted=False).real
    dafilt[dim] = da[dim]

    return dafilt.rename('filtered')


# ===================================================================================================
def isosurface(da, coord, target):
    """
        Returns the values of a coordinate in the input array where the input array values equals
        a prescribed target. E.g. returns the depth of the 20 degC isotherm. Returns nans for all
        points in input array where isosurface is not defined. If 
        Author: Thomas Moore
        Date: 02/10/2018
        
        Parameters
        ----------
        da : xarray DataArray
            Array of values to be isosurfaced
        coord : str
            Name of coordinate to contruct isosurface about
        target : value
            Isosurface value
            
        Returns
        -------
        isosurface : xarray DataArray
            Values of coord where da is closest to target. If multiple occurences of target occur 
            along coord, only the maximum value of coord is returned
            
        Examples
        --------
        >>> A = xr.DataArray(np.random.normal(size=(5,5)), 
        ...                  coords=[('x', np.arange(5)), ('y', np.arange(5))])
        >>> isosurface(A, coord='x', target=0)
        >>> doppyo.utils.isosurface(A, coord='x', target=0)
        <xarray.DataArray 'isosurface' (y: 5)>
        array([ 4.,  1., nan,  3.,  4.])
        Coordinates:
          * y        (y) int64 0 1 2 3 4
  
        Limitations
        -----------
        If multiple occurences of target occur along coord, only the maximum value of coord is
        returned
        
        To do
        -----
        The current version includes no interpolation between grid spacing. This should be added as
        an option in the future
    """
    
    # Find isosurface -----
    mask = da > target
    da_mask = mask * da[coord]
    isosurface = da_mask.max(coord)

    return isosurface.where(da.max(dim=coord) > target).rename('isosurface')


# ===================================================================================================
# Climatology tools
# ===================================================================================================
def load_mean_climatology(clim, freq, variable=None, time_name=None, **kwargs):
    """ 
        Returns pre-saved climatology at desired frequency.
        Author: Dougie Squire
        Date: 04/03/2018
        
        Parameters
        ----------
        clim : str
            Name of climatology to load. Currently available options are: "jra_1958-2016", 
            "cafe_f1_atmos_2003-2017", "cafe_f1_ocean_2003-2017", "cafe_c2_atmos_400-499", 
            "cafe_c2_atmos_500-549", cafe_c2_ocean_400-499", "cafe_c2_ocean_500-549", 
            "HadISST_1870-2018", "REMSS_2002-2018"
        freq : str
            Desired frequency of climatology (daily or longer) e.g. 'D', 'M'
        variable : str, optional
            Variable to load. If None, all variables are returned
        time_name : str, optional
            Name of the time dimension. If None, doppyo will attempt to determine time_name 
            automatically
        **kwargs : dict
            Additional arguments to pass to load command
        
        Returns
        -------
        climatology : xarray DataArray
            Requested climatology
        
        Examples
        --------
        >>> doppyo.utils.load_mean_climatology(clim='cafe_c2_atmos_500-549', freq='D', variable='u')
        <xarray.DataArray 'u' (time: 366, level: 37, lat: 90, lon: 144)>
        [175504320 values with dtype=float32]
        Coordinates:
          * lon      (lon) float64 1.25 3.75 6.25 8.75 11.25 ... 351.2 353.8 356.2 358.8
          * lat      (lat) float64 -89.49 -87.98 -85.96 -83.93 ... 85.96 87.98 89.49
          * level    (level) float32 1.0 2.0 3.0 5.0 7.0 ... 925.0 950.0 975.0 1000.0
          * time     (time) datetime64[ns] 2016-01-01T12:00:00 ... 2016-12-31T12:00:00
        Attributes:
            long_name:      zonal wind
            units:          m/sec
            valid_range:    [-32767  32767]
            packing:        4
            cell_methods:   time: mean
            time_avg_info:  average_T1,average_T2,average_DT
        
        Limitations
        -----------
        Can only be run from a system connected to Bowen cloud storage
    """
    
    data_path = '/OSM/CBR/OA_DCFP/data/intermediate_products/doppyo/mean_climatologies/'
    
    # Load specified dataset -----
    if clim == 'jra_1958-2016':
        data_loc = data_path + 'jra.isobaric.1958010100_2016123118.clim.nc'
    elif clim == 'cafe_f1_atmos_2003-2017':
        data_loc = data_path + 'cafe.f1.atmos.2003010112_2017123112.clim.nc'
    elif clim == 'cafe_f1_ocean_2003-2017':
        data_loc = data_path + 'cafe.f1.ocean.2003010112_2017123112.clim.nc'  
    elif clim == 'cafe_c2_atmos_400-499':
        data_loc = data_path + 'cafe.c2.atmos.400_499.clim.nc'
    elif clim == 'cafe_c2_atmos_500-549':
        data_loc = data_path + 'cafe.c2.atmos.500_549.clim.nc'    
    elif clim == 'cafe_c2_ocean_400-499':
        data_loc = data_path + 'cafe.c2.ocean.400_499.clim.nc'
    elif clim == 'cafe_c2_ocean_500-549':
        data_loc = data_path + 'cafe.c2.ocean.500_549.clim.nc'
    elif clim == 'HadISST_1870-2018':
        data_loc = data_path + 'hadisst.1870011612_2018021612.clim.nc'
    elif clim == 'REMSS_2002-2018':
        data_loc = data_path + 'remss.2002060112_2018041812.clim.nc' 
    else:
        raise ValueError(f'"{clim}" is not an available climatology. Available options are "jra_1958-2016", "cafe_f1_atmos_2003-2017", "cafe_f1_ocean_2003-2017", "cafe_c2_atmos_400-499", "cafe_c2_atmos_500-549", "cafe_c2_ocean_400-499", "cafe_c2_ocean_500-549", "HadISST_1870-2018","REMSS_2002-2018"')
        
    ds = xr.open_dataset(data_loc, **kwargs)
    
    if time_name is None:
        time_name = get_time_name(ds)
        
    if variable is not None:
        try:
            ds = ds[variable]
        except KeyError:
            raise ValueError(f'"{variable}" is not a variable in "{clim}"')
    
    # Resample if required -----    
    load_freq = pd.infer_freq(ds[time_name].values)
    if load_freq != freq:
        if variable == 'precip':
            ds = ds.resample({time_name : freq}).sum(dim=time_name)
        else:
            ds = ds.resample({time_name : freq}).mean(dim=time_name)

    return ds


# ===================================================================================================
def anomalize(data, clim, time_name=None):
    """ 
        Returns anomalies of data about clim
        Author: Dougie Squire
        Date: 04/03/2018
        
        Parameters
        ----------
        data : xarray DataArray
            Array to compute anomalies from
        clim : xarray DataArray
            Array to compute anomalies about
        time_name : str, optional
            Name of the time dimension. If None, doppyo will attempt to determine time_name 
            automatically
            
        Returns
        -------
        anomalies : xarray DataArray
            Array containing anomalies of data about clim
            
        Examples
        --------
        >>> A = xr.DataArray(np.random.normal(size=(1000)), 
        ...                  coords=[('time', pd.date_range(start='1/1/2000', periods=1000, freq='D'))])
        >>> A_clim = A.groupby('time.month').mean('time')
        >>> doppyo.utils.anomalize(A, A_clim)
        <xarray.DataArray (time: 1000)>
        array([-3.050884, -0.361403, -0.893451, ...,  0.685141,  0.477916, -1.175434])
        Coordinates:
          * time     (time) datetime64[ns] 2000-01-01 2000-01-02 ... 2002-09-26
          
        Limitations
        -----------
        Cannot anomalize about multiple day/month/year climatologies, e.g. 5-day averages 
    """
    
    data_use = data.copy(deep=True)
    clim_use = clim.copy(deep=True)
    if time_name is None:
        time_name = get_time_name(data_use)
    
    def _contains_int(string):
        """ Checks if string contains an integer """
        return any(char.isdigit() for char in string)
    
    # If clim is saved on a time dimension, deal with accordingly ----- 
    if time_name in clim_use.dims:
        # Find frequency (assume this is annual average if only one time value exists) -----
        if len(clim_use[time_name]) > 1:
            clim_freq = pd.infer_freq(clim_use[time_name].values[:3])
            if _contains_int(clim_freq):
                raise ValueError('Cannot anomalize about multiple day/month/year climatologies')
        else:
            clim_freq = 'A'
            
        # Build daily, monthly or annual climatologies -----
        if 'D' in clim_freq:
            # Contruct month-day array (to deal with leap years) -----
            clim_mon = np.array([str(i).zfill(2) + '-' for i in clim_use[time_name].dt.month.values])
            clim_day = np.array([str(i).zfill(2)  for i in clim_use[time_name].dt.day.values])
            clim_use[time_name] = np.core.defchararray.add(clim_mon, clim_day)
            
            clim_use = clim_use.groupby(time_name, squeeze=False).mean(dim=time_name)
            deal_with_leap = True
        elif 'M' in clim_freq:
            clim_use = clim_use.groupby(time_name+'.month', squeeze=False).mean(dim=time_name)
        elif ('A' in clim_freq) | ('Y' in clim_freq):
            clim_use = prune(clim_use.groupby(time_name+'.year', squeeze=False).mean(dim=time_name).squeeze())
    elif 'dayofyear' in clim_use.dims:
        clim_freq = 'D'
        deal_with_leap = False
    elif 'month' in clim_use.dims:
        clim_freq = 'M'
    elif 'season' in clim_use.dims:
        clim_freq = 'seas'
    elif 'year' in clim_use.dims:
        clim_freq = 'A'
    else:
        warnings.warn('Unable to determine frequency of climatology DataArray, assuming annual average')
        clim_freq = 'A'
    
    # Subtract the climatology from the full field -----
    if ('D' in clim_freq) and (deal_with_leap is True):
        time_keep = data_use[time_name]

        # Contruct month-day arrays -----
        data_mon = np.array([str(i).zfill(2) + '-' for i in data_use[time_name].dt.month.values])
        data_day = np.array([str(i).zfill(2)  for i in data_use[time_name].dt.day.values])
        data_use[time_name] = np.core.defchararray.add(data_mon, data_day)

        anom = data_use.groupby(time_name) - clim_use
        anom[time_name] = time_keep
    elif ('D' in clim_freq) and (deal_with_leap is False):
        anom = data_use.groupby(time_name+'.dayofyear') - clim_use
    elif 'M' in clim_freq:
        anom = data_use.groupby(time_name+'.month') - clim_use
    elif 'seas' in clim_freq:
        anom = data_use.groupby(time_name+'.season') - clim_use
    elif ('A' in clim_freq) | ('Y' in clim_freq):
        anom = data_use.groupby(time_name+'.year') - clim_use
        
    return prune(anom)


# ===================================================================================================
# IO tools
# ===================================================================================================
def trunc_time(da, freq, time_name=None):
    """ 
        Truncates values in provided array to provided frequency 
        Author: Dougie Squire
        Date: 04/04/2018
        
        Parameters
        ----------
        da : xarray DataArray
            Array containing time coordinate to be truncated
        freq : str
            Truncation frequency. Options are 's', 'm', 'h', D', 'M', 'Y'
        time_name : str, optional
            Name of the time dimension. If None, doppyo will attempt to determine time_name 
            automatically
            
        Returns
        -------
        truncated : xarray DataArray
            time-truncated array
        
        Examples
        --------
        >>> A = xr.DataArray(np.random.normal(size=(10)), 
        ...                  coords=[('time', pd.date_range(start='1/1/2000', 
        ...                           periods=10, freq='M').shift(5,'D'))])
        >>> doppyo.utils.trunc_time(A, freq='M') 
        <xarray.DataArray (time: 10)>
        array([-0.197528,  1.022739, -0.50139 ,  0.128189, -0.886135,  0.570657,
               -0.336125, -0.499281,  1.143722,  1.987681])
        Coordinates:
          * time     (time) datetime64[ns] 2000-02-01 2000-03-01 ... 2000-11-01
    """
    
    if time_name is None:
        time_name = get_time_name(da)
        
    da = da.copy()
    da[time_name] = da[time_name].astype('<M8[' + freq + ']')
    
    return da


# ===================================================================================================
def leadtime_to_datetime(da, init_date_name='init_date', lead_time_name='lead_time', time_name='time'):
    """ 
        Converts time information from initial date / lead time dimension pair to single datetime 
        dimension (i.e. timeseries) 
        Author: Dougie Squire
        Date: 04/04/2018
        
        Parameters
        ----------
        da : xarray DataArray
            Array in initial date / lead time format to convert to datetime format
        init_date_name : str, optional
            Name of initial date dimension
        lead_time_name : str, optional
            Name of lead time dimension
        time_name : str, optional
            Name of time dimension to create
            
        Returns
        -------
        converted : xarray DataArray
            Array converted to datetime format
            
        Examples
        --------
        >>> A = xr.DataArray(np.random.normal(size=(10)), 
        ...                  coords=[('time', pd.date_range(start='1/1/2000', periods=10, freq='M'))])
        >>> B = doppyo.utils.datetime_to_leadtime(A)
        >>> doppyo.utils.leadtime_to_datetime(B)
        <xarray.DataArray (time: 10)>
        array([-0.158172,  1.319148,  0.648378,  0.577859,  0.371392, -1.380317,
                0.126416,  1.184546,  0.107898,  1.304755])
        Coordinates:
          * time     (time) datetime64[ns] 2000-01-31 2000-02-29 ... 2000-10-31
    """
    
    try:
        init_date = da[init_date_name].values[0]
    except IndexError:
        init_date = da[init_date_name].values
        
    lead_times = list(map(int, da[lead_time_name].values))
    freq = da[lead_time_name].attrs['units']
     
    datetimes = (pd.date_range(init_date, periods=len(lead_times), freq=freq)).values
    
    da_out = da.drop(init_date_name)
    da_out = da_out.rename({lead_time_name : time_name})
    da_out[time_name] = datetimes
    
    return prune(da_out)


# ===================================================================================================
def datetime_to_leadtime(da, init_date_name='init_date', lead_time_name='lead_time', time_name='time'):
    """ 
        Converts time information from single datetime dimension (i.e. timeseries) to initial date / 
        lead time dimension pair
        Author: Dougie Squire
        Date: 04/04/2018
        
        Parameters
        ----------
        da : xarray DataArray
            Array to in datetime format to convert to initial date / lead time format
        init_date_name : str, optional
            Name of initial date dimension to create
        lead_time_name : str, optional
            Name of lead time dimension to create
        time_name : str, optional
            Name of time dimension
            
        Returns
        -------
        converted : xarray DataArray
            Array converted to initial date / lead time format
            
        Examples
        --------
        >>> A = xr.DataArray(np.random.normal(size=(10)), 
        ...                  coords=[('time', pd.date_range(start='1/1/2000', periods=10, freq='M'))])
        >>> doppyo.utils.datetime_to_leadtime(A)
        <xarray.DataArray (lead_time: 10)>
        array([ 0.450976, -1.671764,  0.681519,  0.836319, -0.005434,  0.144954,
                0.719887,  0.344615,  0.461055,  0.736307])
        Coordinates:
          * lead_time  (lead_time) int64 0 1 2 3 4 5 6 7 8 9
            init_date  datetime64[ns] 2000-01-31

        Limitations
        -----------
        Only compatible with time coordinates that have frequencies that can be determined by pandas.infer_freq().
        This means that ambiguous frequencies, such as month-centred monthly frequencies must be preprocessed for
        compatibility (see doppyo.utils.trunc_freq())
    """
    
    init_date = da[time_name].values[0]
    lead_times = range(len(da[time_name]))

    freq = pd.infer_freq(da[time_name].values)
    
    if freq is None:
        raise ValueError('Unable to determine frequency of time coordinate. If using monthly data that is not stored relative to the start or end of each month, first truncate the time coordinate to the start of each month using doppyo.utils.trunc_time(da, freq="M")')
    
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

    da_out = da.rename({time_name : lead_time_name})
    da_out[lead_time_name] = lead_times
    da_out[lead_time_name].attrs['units'] = freq

    da_out.coords[init_date_name] = init_date
    
    return da_out


# ===================================================================================================
def repeat_datapoint(da, coord, coord_val):
    """ 
        Returns array with data at coord = coord_val repeated across all other elements in coord. 
        This is useful for generating persistence forecasts
        Author: Dougie Squire
        Date: 02/06/2018
        
        Parameters
        ----------
        da : xarray DataArray
            Array containing darta to repeat
        coord : str
            Coordinate in da over which to repeat the data at coord = coord_val
        coord_val : value
            The value of coord giving the data to be repeated
        
        Returns
        -------
        repeated : xarray DataArray
            Array with data at coord=coord_val repeated across all other elements in coord
            
        Examples
        --------
        >>> A = xr.DataArray(np.random.normal(size=(3,2)), 
        ...                  coords=[('x', np.arange(3)),('y', np.arange(2))])
        >>> doppyo.utils.repeat_datapoint(A, 'x', 2)
        <xarray.DataArray (x: 3, y: 2)>
        array([[-1.805652,  0.526434],
               [-1.805652,  0.526434],
               [-1.805652,  0.526434]])
        Coordinates:
          * x        (x) int64 0 1 2
          * y        (y) int64 0 1
    """

    repeat_data = da.sel({coord : coord_val}, drop=True)
    
    return (0 * da) + repeat_da


# ===================================================================================================
def get_latlon_region(da, box):
    """
        Returns an array containing those elements of the input array that fall within the provided
        lat-lon box
        Author: Dougie Squire
        Date: 04/04/2018
        
        Parameters
        ----------
        da : xarray DataArray
            Array to extract lat-lon box from
        box : array_like
            Edges of lat-lon box in the format [lat_min, lat_max, lon_min, lon_max]
            
        Returns
        -------
        reduced : xarray DataArray
            Array containing those elements of the input array that fall within the box
            
        Examples
        --------
        >>> A = xr.DataArray(np.random.normal(size=(180,360)), 
        ...                  coords=[('lat', np.arange(-90,90,1)),('lon', np.arange(-280,80,1))])
        >>> doppyo.utils.get_latlon_region(A, [-10, 10, 70, 90])
        <xarray.DataArray (lat: 21, lon: 21)>
        array([[ 0.854745,  1.53709 ,  0.491165, ..., -0.675664,  1.572102, -0.931492],
               [ 0.570822,  0.60621 , -0.125524, ..., -1.731507,  0.853652,  0.845369],
               [-0.061811,  0.758512,  1.215573, ..., -1.275482,  2.668203,  0.791314],
               ...,
               [-0.263597,  0.102755, -2.775252, ..., -0.736136,  0.944762,  0.005952],
               [ 0.009949,  0.409897, -0.138621, ...,  1.054246,  1.30817 , -0.539534],
               [ 1.281245, -0.792166, -1.736007, ...,  0.474207, -0.781518,  0.738593]])
        Coordinates:
          * lat      (lat) int64 -10 -9 -8 -7 -6 -5 -4 -3 -2 -1 0 1 2 3 4 5 6 7 8 9 10
          * lon      (lon) int64 -280 -279 -278 -277 -276 -275 ... 74 75 76 77 78 79
    """

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
def latlon_average(da, box):
    '''
        Returns the average of the input array over a provide lat-lon box, 
        
        Parameters
        ----------
        da : xarray DataArray
            Array to average lat-lon box from
        box : array_like
            Edges of lat-lon box in the format [lat_min, lat_max, lon_min, lon_max]
            
        Returns
        -------
        reduced : xarray DataArray
            Array containing those elements of the input array that fall within the box
            
        Examples
        --------
        >>> A = xr.DataArray(np.random.normal(size=(180,360)), 
        ...                  coords=[('lat', np.arange(-90,90,1)),('lon', np.arange(-280,80,1))])
        >>> doppyo.utils.latlon_average(A, [-10, 10, 70, 90])
        <xarray.DataArray ()>
        array(-0.056776)
    '''
    
    return get_latlon_region(da, box).mean(dim=['lat', 'lon'])


# ===================================================================================================
def stack_by_init_date(da, init_dates, N_lead_steps, init_date_name='init_date', 
                       lead_time_name='lead_time', time_name='time'):
    """ 
        Stacks provided timeseries array in an inital date / lead time format. Note this process
        replicates data and can substantially increase memory usage. Lead time frequency will match
        frequency of input data. Returns nans if requested times lie outside of the available range
        
        Parameters
        ----------
        da : xarray DataArray
            Timeseries array to be stacked
        init_dates : array_like of datetime objects
            Initial dates to stack onto
        N_lead_steps : value
            Number of lead time steps
        init_date_name : str, optional
            Name of initial date dimension
        lead_time_name : str, optional
            Name of lead time dimension
        time_name : str, optional
            Name of time dimension
            
        Returns
        -------
        stacked : xarray DataArray
            Stacked xarray in inital date / lead time format
            
        Examples
        --------
        >>> A = xr.DataArray(np.random.normal(size=(3)), 
        ...                  coords=[('time', pd.date_range(start='2000-01-01', periods=3, freq='MS'))])
        >>> init_dates = pd.date_range(start='1999-11-01', periods=3, freq='MS')
        >>> doppyo.utils.stack_by_init_date(A, init_dates=init_dates, N_lead_steps=3)
        <xarray.DataArray (init_date: 3, lead_time: 3)>
        array([[      nan,       nan,       nan],
               [      nan,       nan,       nan],
               [ 0.509276, -3.046124, -0.665343]])
        Coordinates:
          * lead_time  (lead_time) int64 0 1 2
          * init_date  (init_date) datetime64[ns] 1999-11-01 1999-12-01 2000-01-01
    """

    init_list = []
    for init_date in init_dates:
        start_index = np.where(da[time_name] == np.datetime64(init_date))[0]
        
        # If init_date falls outside time bounds, fill with nans -----
        if start_index.size == 0:
            da_nan = np.nan * da.isel({time_name:range(0, N_lead_steps)})
            da_nan[time_name] = pd.date_range(init_date, periods=N_lead_steps, freq=pd.infer_freq(da[time_name].values))
            init_list.append(datetime_to_leadtime(da_nan))
        else:
            start_index = start_index.item()
            end_index = min([start_index + N_lead_steps, len(da[time_name])])
            init_list.append(datetime_to_leadtime(da.isel({time_name:range(start_index, end_index)})))
    
    return xr.concat(init_list, dim=init_date_name)


# ===================================================================================================
def concat_times(da, init_date_name='init_date', lead_time_name='lead_time', time_name='time'):
    """
        Unstack and concatenate all init_date/lead_time rows into single time dimension
        Author: Dougie Squire
        Date: 22/04/2018
        
        Parameters
        ----------
        da : xarray DataArray
            Array to be unstacked and concatenated
        init_date_name : str, optional
            Name of initial date dimension
        lead_time_name : str, optional
            Name of lead time dimension
        time_name : str, optional
            Name of time dimension
        
        Returns
        -------
        concatenated : xarray DataArray
            Unstacked and concatenated array
        
        Examples
        --------
        >>> A = xr.DataArray(np.random.normal(size=(3,3)), 
        ...                  coords=[('init_date', 
        ...                           pd.date_range(start='1/1/2018', periods=3, freq='M')), 
        ...                          ('lead_time', np.arange(3))])
        >>> A['lead_time'].attrs['units'] = 'M'
        >>> doppyo.utils.concat_times(A)
        <xarray.DataArray (time: 9)>
        array([-1.65746 ,  0.57727 ,  0.010619, -0.008245,  0.119201, -0.445606,
               -0.546745,  0.157267, -1.616096])
        Coordinates:
          * time     (time) datetime64[ns] 2018-01-31 2018-02-28 ... 2018-05-31
    """
    
    da_list = []
    for init_date in da[init_date_name].values:
        da_list.append(leadtime_to_datetime(da.sel({init_date_name :init_date}), 
                                            init_date_name=init_date_name, 
                                            lead_time_name=lead_time_name))
    return xr.concat(da_list, dim=time_name)


# ===================================================================================================
def prune(da, squeeze=False):
    """ 
        Removes all coordinates that are not dimensions
        Author: Dougie Squire
        Date: 22/04/2018
        
        Parameters
        ----------
        da : xarray DataArray
            Array to prune
        squeeze : bool, optional
            If True, squeeze the array (i.e. remove 1D dimensions) prior to pruning
            
        Returns
        -------
        pruned : xarray DataArray
            The pruned array
            
        Examples
        --------
        >>> A = xr.DataArray(np.random.normal(size=(3,1)), 
        ...                  coords=[('x', np.arange(3)),('y', np.arange(1))]).expand_dims('z')
        >>> A.coords['w'] = 1
        >>> doppyo.utils.prune(A, squeeze=True)
        <xarray.DataArray (x: 3)>
        array([-1.323662,  1.464171,  0.480917])
        Coordinates:
          * x        (x) int64 0 1 2
    """
    
    if squeeze:
        da = da.squeeze()
        
    codims = list(set(da.coords)-set(da.dims))

    for codim in codims:
        if codim in da.coords:
            da = da.drop(codim)

    return da


# ===================================================================================================
# xarray processing tools
# ===================================================================================================
def get_other_dims(da, dims_exclude):
    """ 
        Returns all dimensions in provided dataset excluding dim_exclude 
        Author: Dougie Squire
        Date: 22/04/2018
        
        Parameters
        ----------
        da : xarray DataArray
            Array to retreive dimensions from
        dims_exclude : str or sequence of str
            Dimensions to exclude
        
        Returns
        -------
        dims : str or sequence of str
            Dimensions of input array, excluding dims_exclude
            
        Examples
        --------
        >>> A = xr.DataArray(np.random.normal(size=(3,2)), coords=[('x', np.arange(3)), 
        ...                                                        ('y', np.arange(2))])
        >>> doppyo.utils.get_other_dims(A, 'y')
        'x'
    """
    
    dims = da.dims
    
    if dims_exclude == None:
        other_dims = dims
    else:
        if isinstance(dims, str):
            dims = [dims]
        if isinstance(dims_exclude, str):
            dims_exclude = [dims_exclude]

        other_dims = tuple(set(dims).difference(set(dims_exclude)))
        if len(other_dims) == 0:
            return None
        elif len(other_dims) == 1:
            return other_dims[0]
        else:
            return other_dims


# ===================================================================================================
def cftime_to_datetime64(time, shift_year=0):
    """ 
        Convert cftime object to datetime64 object, allowing for `NOLEAP` calendar configuration
        Author: Dougie Squire
        Date: 04/09/2018
        
        Parameters
        ----------
        time : cftime or array_like of cftime
            Times to be converted to datetime64
        shift_year: values
            Number of years to shift times by. cftime objects are generated by xarray when times fall
            outside of the range 1678-2261. Shifting years to within this range enables conversion to
            datetime64 within an xarray object
            
        Returns
        --------
        converted : numpy datetime64 or array_like of numpy datetime64
            Input times converted from cftime to numpy datetime64
            
        Examples
        --------
        >>> A = xr.DataArray(np.random.normal(size=(12)), 
        ...                  coords=[('time', np.array([cftime.datetime(0, m, 1) for m in np.arange(1,13)]))])
        >>> A['time'] = doppyo.utils.cftime_to_datetime64(A['time'], shift_year=2000)
        >>> A
        <xarray.DataArray (time: 12)>
        array([ 0.391673, -1.317681,  1.51771 , -0.195475,  0.525342,  0.390625,
                1.426725, -0.261821,  1.021318,  1.205761, -0.907714,  1.009402])
        Coordinates:
          * time     (time) datetime64[ns] 2000-01-01 2000-02-01 ... 2000-12-01

        Limitations
        -----------
        Times must be sequential and monotonic
    """

    if (time.values[0].timetuple()[0]+shift_year < 1678) | (time.values[-1].timetuple()[0]+shift_year > 2261):
        raise ValueError('Cannot create datetime64 object for years outside on 1678-2262')
        
    return np.array([np.datetime64(time.values[i].replace(year=time.values[i].timetuple()[0]+shift_year) \
                                                 .strftime(), 'ns') \
                                                 for i in range(len(time))])


# ===================================================================================================
def get_time_name(da):
    """ 
        Returns name of time dimension in input array
        Author: Dougie Squire
        Date: 03/03/2018
        
        Parameters
        ----------
        da : xarray DataArray
            Array with coordinate corresponding to time
        
        Returns
        -------
        name : str
            Name of dimension corresponding to time
        
        Examples
        --------
        >>> A = xr.DataArray(np.random.normal(size=(2,2,2,2,2)), 
        ...                  coords=[('lat', np.arange(2)), ('lon', np.arange(2)), 
        ...                          ('depth', np.arange(2)), ('time', np.arange(2))])
        >>> doppyo.utils.get_time_name(A)
        'time'
    """
    
    if 'time' in da.dims:
        return 'time'
    else:
        raise KeyError('Unable to determine longitude dimension')
        pass
    

# ===================================================================================================
def get_lon_name(da):
    """ 
        Returns name of longitude dimension in input array
        Author: Dougie Squire
        Date: 03/03/2018
        
        Parameters
        ----------
        da : xarray DataArray
            Array with coordinate corresponding to longitude
        
        Returns
        -------
        name : str
            Name of dimension corresponding to longitude
        
        Examples
        --------
        >>> A = xr.DataArray(np.random.normal(size=(2,2,2,2,2)), 
        ...                  coords=[('lat', np.arange(2)), ('lon', np.arange(2)), 
        ...                          ('depth', np.arange(2)), ('level', np.arange(2))])
        >>> doppyo.utils.get_lon_name(A)
        'lon'
    """
    
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
    """ 
        Returns name of latitude dimension in input array
        Author: Dougie Squire
        Date: 03/03/2018
        
        Parameters
        ----------
        da : xarray DataArray
            Array with coordinate corresponding to latitude
        
        Returns
        -------
        name : str
            Name of dimension corresponding to latitude
        
        Examples
        --------
        >>> A = xr.DataArray(np.random.normal(size=(2,2,2,2,2)), 
        ...                  coords=[('lat', np.arange(2)), ('lon', np.arange(2)), 
        ...                          ('depth', np.arange(2)), ('level', np.arange(2))])
        >>> doppyo.utils.get_lat_name(A)
        'lat'
    """
    
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
    """ 
        Returns name of depth dimension in input array
        Author: Thomas Moore
        Date: 31/10/2018
        
        Parameters
        ----------
        da : xarray DataArray
            Array with coordinate corresponding to depth
        
        Returns
        -------
        name : str
            Name of dimension corresponding to depth
        
        Examples
        --------
        >>> A = xr.DataArray(np.random.normal(size=(2,2,2,2,2)), 
        ...                  coords=[('lat', np.arange(2)), ('lon', np.arange(2)), 
        ...                          ('depth', np.arange(2)), ('level', np.arange(2))])
        >>> doppyo.utils.get_depth_name(A)
        'depth'
    """
    
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
    """ 
        Returns name of atmospheric level dimension in input array
        Author: Dougie Squire
        Date: 03/03/2018
        
        Parameters
        ----------
        da : xarray DataArray
            Array with coordinate corresponding to atmospheric level
        
        Returns
        -------
        name : str
            Name of dimension corresponding to atmospheric level
        
        Examples
        --------
        >>> A = xr.DataArray(np.random.normal(size=(2,2,2,2,2)), 
        ...                  coords=[('lat', np.arange(2)), ('lon', np.arange(2)), 
        ...                          ('depth', np.arange(2)), ('level', np.arange(2))])
        >>> doppyo.utils.get_level_name(A)
        'level'
    """
    
    if 'level' in da.dims:
        return 'level'
    else:
        raise KeyError('Unable to determine level dimension')
        pass
    
    
# ===================================================================================================
def get_plevel_name(da):
    """ 
        Returns name of pressure level dimension in input array
        Author: Dougie Squire
        Date: 03/03/2018
        
        Parameters
        ----------
        da : xarray DataArray
            Array with coordinate corresponding to pressure level
        
        Returns
        -------
        name : str
            Name of dimension corresponding to pressure level
        
        Examples
        --------
        >>> A = xr.DataArray(np.random.normal(size=(2,2,2,2,2)), 
        ...                  coords=[('lat', np.arange(2)), ('lon', np.arange(2)), 
        ...                          ('depth', np.arange(2)), ('level', np.arange(2))])
        >>> doppyo.utils.get_plevel_name(A)
        'level'
    """
    
    if 'level' in da.dims:
        return 'level'
    else:
        raise KeyError('Unable to determine pressure level dimension')
        pass

    
# ===================================================================================================
# General tools    
# ===================================================================================================
def _is_datetime(object):
    """ 
        Return True or False depending on whether input is datetime64 or not 
        Author: Dougie Squire
        Date: 19/15/2018
        
        Parameters
        ----------
        object : value
            Object to query
            
        Returns
        -------
        isdatetime : bool
            True or False depending on whether input is datetime64 or not
            
        Examples
        --------
        >>> A = np.datetime64('2000-01-01')
        >>> doppyo.utils._is_datetime(A)
        True
    """
    
    return pd.api.types.is_datetime64_dtype(object)


# ===================================================================================================
def _equal_coords(da_1, da_2):
    """ 
        Returns True if coordinates of da_1 and da_2 are equal (or flipped) 
        Author: Dougie Squire
        Date: 19/15/2018
        
        Parameters
        ----------
        da_1 : xarray DataArray
            First array to compare coordinates
        da_2 : xarray DataArray
            Second array to compare coordinates
            
        Returns
        -------
        equal : bool
            True if coordinates of da_1 and da_2 are equal (or flipped), False otherwise
            
        Examples
        --------
        >>> A = xr.DataArray(np.random.normal(size=(3,3)), coords=[('x', np.arange(3)), 
        ...                                                        ('y', np.arange(3))])
        >>> B = xr.DataArray(np.random.normal(size=(3,3)), coords=[('x', np.arange(3)), 
        ...                                                        ('y', np.arange(3))])
        >>> doppyo.utils._equal_coords(A,B)
        True
    """
    
    da1_coords = da_1.coords.to_dataset()
    da2_coords = da_2.coords.to_dataset()
    
    if da1_coords.equals(da2_coords):
        return True
    elif list(set(da_1.coords) - set(da_2.coords)) != []:
        return False
    else:
        # Check if coordinates are the same but reversed -----
        bool_list = [(da1_coords[coord].equals(da2_coords[coord])) | \
                     (da1_coords[coord].equals(da2_coords[coord] \
                                       .sel({coord:slice(None, None, -1)}))) \
                     for coord in da1_coords.coords]
        return np.all(bool_list)




