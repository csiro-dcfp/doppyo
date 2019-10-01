"""
    Collection of old doppyo functions and useful tidbits for internal dcfp use
    Authors: Dougie Squire and Thomas Moore
    Date created: 01/10/2018
    Python Version: 3.6
"""

# ===================================================================================================
# Packages
# ===================================================================================================
import numpy as np
import pandas as pd
import xarray as xr
import dask

import cartopy
from collections import Sequence
from itertools import chain, count
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from cartopy.util import add_cyclic_point
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

# Load doppyo packages -----
from doppyo import utils

# ===================================================================================================
def rank_gufunc(x):
    ''' Returns ranked data along specified dimension '''
    
    import bottleneck
    ranks = bottleneck.nanrankdata(x,axis=-1)
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

    # Only keep and combine instances that appear in both dataarrays (excluding the ensemble dim) -----
    aligned = xr.align(da_2_pass, da_1, join='inner', exclude=over_dim)
    combined = xr.concat(aligned, dim=over_dim)
    
    return xr.apply_ufunc(rank_gufunc, combined,
                          input_core_dims=[[over_dim]],
                          dask='allowed',
                          output_dtypes=[int]).rename('rank')


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
def unstack_and_count(da, dims):
    """ Unstacks provided xarray object and returns the total number of elements along dims """
    
    try:
        unstacked = da.unstack(da.dims[0])
    except ValueError:
        unstacked = da

    if dims is None:
        return ((0 * unstacked) + 1)
    else:
        return ((0 * unstacked) + 1).sum(dim=dims, skipna=True)


def compute_histogram(da, bin_edges, over_dims):
    """ Returns the histogram of data over the specified dimensions """

    # To use groupby_bins, da must have a name -----
    da = da.rename('data') 
    
    hist = da.groupby_bins(da, bins=bin_edges, squeeze=False) \
             .apply(unstack_and_count, dims=over_dims) \
             .fillna(0) \
             .rename({'data_bins' : 'bins'})
    hist['bins'] = (bin_edges[0:-1]+bin_edges[1:])/2
    
    # Add nans where data did not fall in any bin -----
    return hist.astype(int).where(hist.sum('bins') != 0)


# ===================================================================================================
def calc_gradient(da, dim, x=None):
    """
        Returns the gradient computed using second order accurate central differences in the 
        interior points and either first order accurate one-sided (forward or backwards) 
        differences at the boundaries

        See https://docs.scipy.org/doc/numpy-1.14.0/reference/generated/numpy.gradient.html
    """ 
    
    # Replace dimension values if specified -----
    da_n = da.copy()
    if x is None:
        x = da_n[dim]
        
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
    
    grad = xr.concat([l, c, r], dim=dim)
    grad[dim] = da[dim]
    
    return grad


# ===================================================================================================
def bias_correct_ms(da_biased, da_target, da_target_clim=None, init_date_name='init_date', 
                          lead_time_name='lead_time'):
    """
        Adjusts, per month and lead time, the mean and standard deviation of da_biased to match that 
        of da_target.
        Author: Dougie Squire
        Date: 01/09/2018
        
        Parameters
        ----------
        da_biased : xarray DataArray
            Array containing values to be corrected. The time information of this array is anticipated 
            in a lead_time/inital_date format
        da_target : xarray DataArray
            Array containing values to use for the correction.
        da_target_clim : xarray DataArray, optional
            Array containing a climatology of da_target. If da_target_clim is provided, this function 
            returns both the corrected full field and the anomalies. Otherwise, returns only the 
            anomalies
        init_date_name : str, optional
            Name of initial date dimension
        lead_time_name : str, optional
            Name of lead time dimension
            
        Returns
        -------
        corrected : xarray DataArray
            Bias corrected array
            
        Examples
        --------
        >>> biased = xr.DataArray(np.random.normal(size=(48,6)), 
        ...                       coords=[('init_date', pd.date_range(start='1/1/2018', periods=48, freq='M')), 
        ...                               ('lead_time', np.arange(6))])
        >>> biased['lead_time'].attrs['units'] = 'M'
        >>> target = xr.DataArray(np.random.normal(size=(48)), 
        ...                       coords=[('time', pd.date_range(start='1/1/2000', periods=48, freq='M'))])
        >>> doppyo.utils.bias_correct_ms(biased, target)
        <xarray.DataArray (init_date: 48, lead_time: 6)>
        array([[ 9.336394e-02,  1.133997e-01, -5.851293e-01, -4.908594e-02,
                 7.952765e-01,  5.325052e-01],
               [-1.131123e+00,  1.603380e-01, -1.626906e+00, -1.811439e+00,
                -1.653359e-01, -1.871170e-01],
               [ 6.515435e-01, -1.064662e+00,  2.249610e+00,  6.881682e-01,
                -1.831233e-01, -1.159470e+00],
               ...,
               [-2.096226e+00,  3.143062e-04,  3.603787e-01, -1.515535e+00,
                 5.421578e-02, -6.446119e-01],
               [-8.186274e-01, -9.858171e-01,  1.933307e+00,  5.227265e-02,
                 5.443201e-01, -7.059492e-01],
               [ 2.253396e-02,  2.238470e+00,  1.138728e-01, -3.617103e-01,
                 1.678223e+00, -2.413158e+00]])
        Coordinates:
          * lead_time  (lead_time) int64 0 1 2 3 4 5
          * init_date  (init_date) datetime64[ns] 2018-01-31 2018-02-28 ... 2021-12-31
          
        Notes
        -----------
        Many years of initial dates (in da_biased) and times (in da_target) must exist for the mean and standard 
        deviation to be computed reliably
    """
    
    def _groupby_lead_and_mean(da, over_dims, init_date_name, lead_time_name):
        """ Groups provided array by lead time and computes mean """
        
        return da.unstack('stacked_' + init_date_name + '_' + lead_time_name).groupby(lead_time_name).mean(over_dims, skipna=True)

    def _groupby_lead_and_std(da, over_dims, init_date_name, lead_time_name):
        """ Groups provided array by lead time and computes standard deviation """
        
        return da.unstack('stacked_' + init_date_name + '_' + lead_time_name).groupby(lead_time_name).std(over_dims, skipna=True)

    def _unstack_and_shift_per_month(da, shift, init_date_name, lead_time_name):
        """ Unstacks and adjusts input array by a constant shift as a function of month """
        
        da_us = da.unstack('stacked_' + init_date_name + '_' + lead_time_name)
        the_month = np.ndarray.flatten(da_us.month.values)
        the_month = int(np.unique(the_month[~np.isnan(the_month)]))
        
        return da_us - shift.sel(month=the_month)

    def _unstack_and_scale_per_month(da, scale, init_date_name, lead_time_name):
        """ Unstacks and scales input array by a constant value as a function of month """
        
        da_us = da.unstack('stacked_' + init_date_name + '_' + lead_time_name)
        the_month = np.ndarray.flatten(da_us.month.values)
        the_month = int(np.unique(the_month[~np.isnan(the_month)]))
        
        return da_us * scale.sel(month=the_month)

    def _scale_per_month(da, scale):
        """ Scales input array by a constant value as a function of month """
        
        return da.groupby('time.month') * scale
    
    _anomalize = lambda data, clim: datetime_to_leadtime(
                                        anomalize(
                                            leadtime_to_datetime(data),clim))

    _rescale = lambda da, scale : datetime_to_leadtime(
                                          _scale_per_month(
                                              leadtime_to_datetime(da), scale))

    da_biased = da_biased.copy()
    da_target = da_target.copy()
    month = (da_biased[init_date_name].dt.month + da_biased[lead_time_name]) % 12
    month = month.where(month != 0, 12)

    # Correct the mean -----
    da_biased.coords['month'] = month
    try:
        da_biased_mean = da_biased.groupby('month').apply(_groupby_lead_and_mean, over_dims=[init_date_name,'ensemble'],
                                                         init_date_name=init_date_name, lead_time_name=lead_time_name)
    except ValueError:
        da_biased_mean = da_biased.groupby('month').apply(_groupby_lead_and_mean, over_dims=init_date_name,
                                                         init_date_name=init_date_name, lead_time_name=lead_time_name)
    
    if da_target_clim is not None:
        da_target_mean = da_target.groupby('time.month').mean('time')
        
        da_meancorr = da_biased.groupby('month').apply(_unstack_and_shift_per_month, shift=(da_biased_mean - da_target_mean),
                                                       init_date_name=init_date_name, lead_time_name=lead_time_name) \
                                      .mean('month', skipna=True)
        da_meancorr[lead_time_name] = da_biased[lead_time_name]
        da_meancorr.coords['month'] = month

        # Compute the corrected anomalies -----
        da_anom_meancorr = da_meancorr.groupby(init_date_name).apply(_anomalize, clim=da_target_clim)
        da_anom_meancorr.coords['month'] = month
    else:
        da_anom_meancorr = da_biased.groupby('month').apply(_unstack_and_shift_per_month, shift=(da_biased_mean),
                                                            init_date_name=init_date_name, lead_time_name=lead_time_name) \
                                      .mean('month', skipna=True)
        da_anom_meancorr[lead_time_name] = da_anom_meancorr[lead_time_name]
        da_anom_meancorr.coords['month'] = month
    
    # Correct the standard deviation -----
    try:
        da_biased_std_tmp = da_anom_meancorr.groupby('month').apply(_groupby_lead_and_std, over_dims=[init_date_name,'ensemble'],
                                                                    init_date_name=init_date_name, lead_time_name=lead_time_name)
    except ValueError:
        da_biased_std_tmp = da_anom_meancorr.groupby('month').apply(_groupby_lead_and_std, over_dims=init_date_name,
                                                                    init_date_name=init_date_name, lead_time_name=lead_time_name)
    try:
        da_target_std = da_target.sel(lat=da_biased.lat, lon=da_biased.lon).groupby('time.month').std('time')
    except:
        da_target_std = da_target.groupby('time.month').std('time')
        
    da_anom_stdcorr_tmp = da_anom_meancorr.groupby('month').apply(_unstack_and_scale_per_month, 
                                                                  scale=(da_target_std / da_biased_std_tmp),
                                                                  init_date_name=init_date_name, 
                                                                  lead_time_name=lead_time_name) \
                                              .mean('month', skipna=True)
    da_anom_stdcorr_tmp[lead_time_name] = da_biased[lead_time_name]
    da_anom_stdcorr_tmp.coords['month'] = month
    
    # This will "squeeze" each pdf at each lead time appropriately. However, the total variance across all leads for 
    # a given month will now be incorrect. Thus, we now rescale as a function of month only
    try:
        da_biased_std = concat_times(da_anom_stdcorr_tmp).groupby('time.month').std(['time','ensemble'])
    except ValueError:
        da_biased_std = concat_times(da_anom_stdcorr_tmp).groupby('time.month').std('time')
    da_anom_stdcorr = da_anom_stdcorr_tmp.groupby(init_date_name).apply(_rescale, scale=(da_target_std / da_biased_std))
    
    if da_target_clim is not None:
        da_stdcorr = da_anom_stdcorr.groupby(init_date_name).apply(_anomalize, clim=-da_target_clim)
        return da_stdcorr.drop('month'), da_anom_stdcorr.drop('month')
    else:
        return da_anom_stdcorr.drop('month')


# ===================================================================================================
def bias_correct_m(da_biased, da_target, da_target_clim=None, init_date_name='init_date', 
                          lead_time_name='lead_time'):
    """
        Adjusts, per month and lead time, the mean of da_biased to match that of da_target
        Author: Dougie Squire
        Date: 01/09/2018
        
        Parameters
        ----------
        da_biased : xarray DataArray
            Array containing values to be corrected. The time information of this array is anticipated 
            in a lead_time/inital_date format
        da_target : xarray DataArray
            Array containing values to use for the correction.
        da_target_clim : xarray DataArray, optional
            Array containing a climatology of da_target. If da_target_clim is provided, this function 
            returns both the corrected full field and the anomalies. Otherwise, returns only the 
            anomalies
        init_date_name : str, optional
            Name of initial date dimension
        lead_time_name : str, optional
            Name of lead time dimension
            
        Returns
        -------
        corrected : xarray DataArray
            Bias corrected array
            
        Examples
        --------
        >>> biased = xr.DataArray(np.random.normal(size=(48,6)), 
        ...                       coords=[('init_date', pd.date_range(start='1/1/2018', periods=48, freq='M')), 
        ...                               ('lead_time', np.arange(6))])
        >>> biased['lead_time'].attrs['units'] = 'M'
        >>> target = xr.DataArray(np.random.normal(size=(48)), 
        ...                       coords=[('time', pd.date_range(start='1/1/2000', periods=48, freq='M'))])
        >>> doppyo.utils.bias_correct_m(biased, target)
        <xarray.DataArray (init_date: 48, lead_time: 6)>
        array([[ 0.541226,  0.693622, -0.367322,  0.820282,  0.111487,  0.078355],
               [-0.299829,  0.164297, -0.976883,  0.463365, -0.26428 , -0.536119],
               [ 0.078832, -0.260615, -0.235059, -0.349185,  0.567183, -1.543395],
               ...,
               [ 0.335494, -1.121158,  1.313004,  0.604279,  0.135053,  0.031851],
               [ 0.33103 ,  0.876521, -0.980873,  0.640328,  1.053691,  0.166768],
               [ 1.207329,  0.021916,  0.210883, -0.189922,  0.075786,  0.047616]])
        Coordinates:
          * init_date  (init_date) datetime64[ns] 2018-01-31 2018-02-28 ... 2021-12-31
          * lead_time  (lead_time) int64 0 1 2 3 4 5
          
        Notes
        -----------
        Many years of initial dates (in da_biased) and times (in da_target) must exist for the mean to be 
        computed reliably
    """

    def _groupby_lead_and_mean(da, over_dims, init_date_name, lead_time_name):
        """ Groups provided array by lead time and computes mean """
        
        return da.unstack('stacked_' + init_date_name + '_' + lead_time_name).groupby(lead_time_name).mean(over_dims, skipna=True)
    
    def _unstack_and_shift_per_month(da, shift, init_date_name, lead_time_name):
        """ Unstacks and adjusts input array by a constant shift as a function of month """
        
        da_us = da.unstack('stacked_' + init_date_name + '_' + lead_time_name)
        the_month = np.ndarray.flatten(da_us.month.values)
        the_month = int(np.unique(the_month[~np.isnan(the_month)]))
        
        return da_us - shift.sel(month=the_month)
    
    _anomalize = lambda data, clim: datetime_to_leadtime(
                                        anomalize(
                                            leadtime_to_datetime(data),clim))
    
    da_biased = da_biased.copy()
    da_target = da_target.copy()
    
    month = (da_biased[init_date_name].dt.month + da_biased[lead_time_name]) % 12
    month = month.where(month != 0, 12)

    # Correct the mean -----
    da_biased.coords['month'] = month
    try:
        da_biased_mean = da_biased.groupby('month').apply(_groupby_lead_and_mean, over_dims=[init_date_name,'ensemble'],
                                                         init_date_name=init_date_name, lead_time_name=lead_time_name)
    except ValueError:
        da_biased_mean = da_biased.groupby('month').apply(_groupby_lead_and_mean, over_dims=init_date_name,
                                                         init_date_name=init_date_name, lead_time_name=lead_time_name)
    
    if da_target_clim is not None:
        da_target_mean = da_target.groupby('time.month').mean('time')
        
        da_meancorr = da_biased.groupby('month').apply(_unstack_and_shift_per_month, shift=(da_biased_mean - da_target_mean),
                                                       init_date_name=init_date_name, lead_time_name=lead_time_name) \
                                      .mean('month', skipna=True)
        da_meancorr[lead_time_name] = da_biased[lead_time_name]
        da_meancorr.coords['month'] = month

        # Compute the corrected anomalies -----
        da_anom_meancorr = da_meancorr.groupby(init_date_name).apply(_anomalize, clim=da_target_clim)
        da_anom_meancorr.coords['month'] = month
    else:
        da_anom_meancorr = da_biased.groupby('month').apply(_unstack_and_shift_per_month, shift=(da_biased_mean),
                                                            init_date_name=init_date_name, lead_time_name=lead_time_name) \
                                      .mean('month', skipna=True)
        da_anom_meancorr[lead_time_name] = da_anom_meancorr[lead_time_name]
        da_anom_meancorr.coords['month'] = month
    
    if da_target_clim is not None:
        da_meancorrr = da_anom_meancorr.groupby(init_date_name).apply(_anomalize, clim=-da_target_clim)
        return da_meancorr.drop('month'), da_anom_meancorr.drop('month')
    else:
        return da_anom_meancorr.drop('month')

    
# ===================================================================================================
def conditional_bias_correct(da_cmp, da_ref, over_dims):
    """
        Return conditional bias corrected data using the approach of Goddard et al. 2013
        
        
    """

    cc = skill.compute_Pearson_corrcoef(da_cmp.mean('ensemble'), da_ref, over_dims=over_dims, subtract_local_mean=False)
    correct_cond_bias = (da_ref.std(over_dims) / da_cmp.mean('ensemble').std(over_dims)) * cc
    
    return da_cmp * correct_cond_bias


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
def leadtime_to_datetime(data_in, init_date_name='init_date', lead_time_name='lead_time'):
    """ Converts time information from lead time/initial date dimension pair to single datetime dimension """
    
    try:
        init_date = data_in[init_date_name].values[0]
    except IndexError:
        init_date = data_in[init_date_name].values
        
    lead_times = list(map(int, data_in[lead_time_name].values))
    freq = data_in[lead_time_name].attrs['units']
    
    # # Split frequency into numbers and strings -----
    # incr_string = ''.join([i for i in freq if i.isdigit()])
    # freq_incr = [int(incr_string) if incr_string else 1][0]
    # freq_type = ''.join([i for i in freq if not i.isdigit()])

    # Deal with special cases of monthly and yearly frequencies -----
    # if 'M' in freq_type:
    #     datetimes = np.array([month_delta(init_date, freq_incr * ix) for ix in lead_times])
    # elif ('A' in freq_type) | ('Y' in freq_type):
    #     datetimes = np.array([year_delta(init_date, freq_incr * ix) for ix in lead_times])
    # else:
    #     datetimes = (pd.date_range(init_date, periods=len(lead_times), freq=freq)).values  
    datetimes = (pd.date_range(init_date, periods=len(lead_times), freq=freq)).values
    
    data_out = data_in.drop(init_date_name)
    data_out = data_out.rename({lead_time_name : 'time'})
    data_out['time'] = datetimes
    
    return prune(data_out)


# ===================================================================================================
def get_nearest_point(da, lat, lon):
    """ Returns the nearest grid point to the specified lat/lon location """

    return da.sel(lat=lat,lon=lon,method='nearest')


# ===================================================================================================
# visualization tools
# ===================================================================================================
def plot_fields(data, title=None, headings=None, ncol=2, ncontour=None, vlims=None, clims=None, squeeze_row=1, 
                squeeze_col=1, squeeze_cbar=1, shift_cbar=1, cmaps='viridis', fontsize=12, invert=False):
    """ Plots tiles of figures """
    
    def _depth(seq):
        for level in count():
            if not seq:
                return level
            seq = list(chain.from_iterable(s for s in seq if isinstance(s, Sequence)))

    matplotlib.rc('font', family='sans-serif')
    matplotlib.rc('font', serif='Helvetica') 
    matplotlib.rc('text', usetex='false') 
    matplotlib.rcParams.update({'font.size': fontsize})

    nrow = int(np.ceil(len(data)/ncol));

    fig = plt.figure(figsize=(11*squeeze_col, nrow*4*squeeze_row))
    
    if not isinstance(data,list):
        data = [data]
    
    if (clims is not None) & (np.shape(vlims) != np.shape(clims)):
        raise ValueError('The input clims must be equal in size to vlims')
    
    # Check if vlims are given per figure or for all figures -----
    one_cbar = False
    if vlims is None:
        vlims = [[None, None]] * len(data)
    if _depth(vlims) == 1:
        one_cbar = True
        
    over_count = 1
    for idx,dat in enumerate(data):
        if one_cbar:
            cmap = cmaps
            vmin, vmax = vlims
            if clims is not None:
                cmin, cmax = clims
        else:
            if isinstance(cmaps, list):
                cmap = cmaps[idx]
            else:
                cmap = cmaps
            vmin, vmax = vlims[idx]
            if clims is not None:
                cmin, cmax = clims[idx]
        
        if ('lat' in dat.dims) and ('lon' in dat.dims):
            trans = cartopy.crs.PlateCarree()
            ax = plt.subplot(nrow, ncol, over_count, projection=cartopy.crs.PlateCarree(central_longitude=180))
            extent = [dat.lon.min()+1e-6, dat.lon.max(), 
                      dat.lat.min(), dat.lat.max()]

            if ncontour is not None:
                if clims is not None:
                    ax.coastlines(color='gray')
                    im = ax.contourf(dat.lon, dat.lat, dat, levels=np.linspace(vmin,vmax,ncontour), origin='lower', transform=trans, 
                                     vmin=vmin, vmax=vmax, cmap=cmap)
                    ax.contour(dat.lon, dat.lat, dat, levels=np.linspace(cmin,cmax,ncontour), origin='lower', transform=trans,
                               vmin=vmin, vmax=vmax, colors='w', linewidths=2)
                    ax.contour(dat.lon, dat.lat, dat, levels=np.linspace(cmin,cmax,ncontour), origin='lower', transform=trans,
                               vmin=vmin, vmax=vmax, colors='k', linewidths=1)
                else:
                    ax.coastlines(color='black')
                    im = ax.contourf(dat.lon, dat.lat, dat, origin='lower', transform=trans, vmin=vmin, vmax=vmax, 
                                     cmap=cmap)
            else:
                ax.coastlines(color='black')
                im = ax.imshow(dat, origin='lower', extent=extent, transform=trans, vmin=vmin, vmax=vmax, cmap=cmap)

            gl = ax.gridlines(crs=cartopy.crs.PlateCarree(), draw_labels=True)
            gl.xlines = False
            gl.ylines = False
            gl.xlabels_top = False
            if over_count % ncol == 0:
                gl.ylabels_left = False
            elif (over_count+ncol-1) % ncol == 0: 
                gl.ylabels_right = False
            else:
                gl.ylabels_left = False
                gl.ylabels_right = False
            gl.xlocator = mticker.FixedLocator([-90, 0, 90, 180])
            gl.ylocator = mticker.FixedLocator([-90, -60, 0, 60, 90])
            gl.xformatter = LONGITUDE_FORMATTER
            gl.yformatter = LATITUDE_FORMATTER
            ax.set_extent(extent)
            
            if not one_cbar:
                cbar = plt.colorbar(im, ax=ax, orientation="horizontal", aspect=30/squeeze_cbar, pad=shift_cbar*0.1)
                tick_locator = mticker.MaxNLocator(nbins=6)
                cbar.locator = tick_locator
                cbar.update_ticks()
                if headings is not None:
                    cbar.set_label(headings[idx], labelpad=5, fontsize=fontsize);
            elif headings is not None:
                ax.set_title(headings[idx], fontsize=fontsize)
        else:
            if len(dat.dims) == 1:
                ax = plt.subplot(nrow, ncol, over_count)
                if len(dat) == 1:
                    x_plt = 0
                    y_plt = dat

                    if 'asp' not in locals():
                        asp = 1

                    ax.bar(x_plt, y_plt, width=asp*y_plt*1)
                    ax.set_xlim(-asp*y_plt,asp*y_plt)
                    ax.set_xticks([])
                    ax.set_aspect(asp)
                    if headings is not None:
                        ax.set_title(headings[idx], fontsize=fontsize)
                else:
                    x_plt = dat[dat.dims[0]]
                    y_plt = dat
                    
                    if 'asp' not in locals():
                        asp = 1
                        
                    ax.plot(x_plt, y_plt)
                    if headings is not None:
                        ax.set_title(headings[idx], fontsize=fontsize)
                        
                    if over_count % ncol == 0:
                        ax.yaxis.tick_right()
                    elif (over_count+ncol-1) % ncol == 0: 
                        ax.set_ylabel(y_plt.name, fontsize=fontsize)
                    else:
                        ax.set_yticks([])
                    if idx / ncol >= nrow - 1:
                        ax.set_xlabel(x_plt.name, fontsize=fontsize)
                    
            else:
                ax = plt.subplot(nrow, ncol, over_count)
                if 'lat' in dat.dims:
                    x_plt = dat['lat']
                    y_plt = dat[utils.get_other_dims(dat,'lat')[0]]
                    # if dat.get_axis_num('lat') > 0:
                    #     dat = dat.transpose()
                elif 'lon' in dat.dims:
                    x_plt = dat['lon']
                    y_plt = dat[utils.get_other_dims(dat,'lon')[0]]
                    # if dat.get_axis_num('lon') > 0:
                    #     dat = dat.transpose()
                else: 
                    x_plt = dat[dat.dims[1]]
                    y_plt = dat[dat.dims[0]]

                extent = [x_plt.min(), x_plt.max(), 
                          y_plt.min(), y_plt.max()]

                if ncontour is not None:
                    if clims is not None:
                        im = ax.contourf(x_plt, y_plt, dat, levels=np.linspace(vmin,vmax,ncontour), vmin=vmin, vmax=vmax, 
                                         cmap=cmap)
                        ax.contour(x_plt, y_plt, dat, levels=np.linspace(cmin,cmax,ncontour), colors='w', linewidths=2)
                        ax.contour(x_plt, y_plt, dat, levels=np.linspace(cmin,cmax,ncontour), colors='k', linewidths=1)
                    else:
                        im = ax.contourf(x_plt, y_plt, dat, vmin=vmin, vmax=vmax, cmap=cmap)
                else:
                    im = ax.imshow(dat, origin='lower', extent=extent, vmin=vmin, vmax=vmax, cmap=cmap)

                if not one_cbar:
                    cbar = plt.colorbar(im, ax=ax, orientation="horizontal", aspect=30/squeeze_cbar, pad=shift_cbar*0.1)
                    tick_locator = mticker.MaxNLocator(nbins=6)
                    cbar.locator = tick_locator
                    cbar.update_ticks()
                    if headings is not None:
                        cbar.set_label(headings[idx], labelpad=5, fontsize=fontsize);
                elif headings is not None:
                    ax.set_title(headings[idx], fontsize=fontsize)
                    
                if over_count % ncol == 0:
                    ax.yaxis.tick_right()
                elif (over_count+ncol-1) % ncol == 0: 
                    ax.set_ylabel(y_plt.dims[0], fontsize=fontsize)
                else:
                    ax.set_yticks([])
                if idx / ncol >= nrow - 1:
                    ax.set_xlabel(x_plt.dims[0], fontsize=fontsize)
            
            if invert:
                ax.invert_yaxis()

        over_count += 1
        
        if idx == 0:
            asp = np.diff(ax.get_xlim())[0] / np.diff(ax.get_ylim())[0]
            
    plt.tight_layout()
        
    if one_cbar:
        vmin, vmax = vlims
        fig.subplots_adjust(bottom=shift_cbar*0.16)
        cbar_ax = fig.add_axes([0.15, 0.13, 0.7, squeeze_cbar*0.020])
        cbar = fig.colorbar(im, cax=cbar_ax, orientation='horizontal');
        cbar_ax.set_xlabel(title, rotation=0, labelpad=15, fontsize=fontsize);
        cbar.set_ticks(np.linspace(vmin,vmax,5))
    elif title is not None:
        fig.suptitle(title, y=1)
        
    
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


# ===================================================================================================
def get_pres_name(da):
    """ 
        Returns name of pressure dimension in input array
        Author: Dougie Squire
        Date: 03/03/2018
        
        Parameters
        ----------
        da : xarray DataArray
            Array with coordinate corresponding to pressure
        
        Returns
        -------
        name : str
            Name of dimension corresponding to pressure
        
        Examples
        --------
        >>> A = xr.DataArray(np.random.normal(size=(2,2,2,2,2)), 
        ...                  coords=[('lat', np.arange(2)), ('lon', np.arange(2)), 
        ...                          ('depth', np.arange(2)), ('level', np.arange(2)), 
        ...                          ('pfull', np.arange(2))])
        >>> doppyo.utils.get_pres_name(A)
        'pfull'
    """
    
    if 'pfull' in da.dims:
        return 'pfull'
    elif 'phalf' in da.dims:
        return 'phalf'
    else:
        raise KeyError('Unable to determine pressure dimension')
        pass
    
    
# ===================================================================================================    
def did_event(da, event):
    """ 
        Returns array containing True/False where event occurs/does not occur 
        
        Notes
        -----
        See http://www.cawcr.gov.au/projects/verification/
    """
    
    eval_expr = event.replace(">", "da >").replace("<", "da <").replace("==", "da ==") \
                     .replace("=", "da ==").replace('&&', '&').replace('||', '|') \
                     .replace("and", "&").replace("or", "|")
    eval_expr = '(' + eval_expr + ').rename("event_logical")'
    
    return eval(eval_expr)


# ===================================================================================================
def compute_likelihood(da_logical, dim='ensemble'):
    """ 
        Returns array of likelihoods computed along dim from logical event data 
        
        Notes
        -----
        See http://www.cawcr.gov.au/projects/verification/
    """
    
    if dim == None:
        likelihood = da_logical
    else:
        likelihood = da_logical.mean(dim=dim).rename('likelihood')
    return likelihood


# ===================================================================================================
def atmos_energy_cycle(temp, u, v, omega, gh, terms=None, vgradz=False, spectral=False, n_wavenumbers=20,
                       integrate=True, loop_triple_terms=False, lat_name=None, lon_name=None, 
                       plevel_name=None):
    """
        Returns all terms in the Lorenz energy cycle. Follows formulae and notation used in `Marques 
        et al. 2011 Global diagnostic energetics of five state-of-the-art climate models. Climate 
        Dynamics`. Note that this decomposition is in the space domain. A space-time decomposition 
        can also be carried out (though not in Fourier space, but this is not implemented here (see 
        `Oort. 1964 On Estimates of the atmospheric energy cycle. Monthly Weather Review`).

        Parameters
        ----------
        temp : xarray DataArray
            Array containing fields of temperature with at least coordinates latitude, longitude 
            and level (following standard naming - see Limitations)
        u : xarray DataArray
            Array containing fields of zonal velocity with at least coordinates latitude, longitude 
            and level (following standard naming - see Limitations)
        v : xarray DataArray
            Array containing fields of meridional velocity with at least coordinates latitude, longitude 
            and level (following standard naming - see Limitations)
        omega : xarray DataArray
            Array containing fields of vertical velocity (pressure coordinates) with at least coordinates 
            latitude, longitude and level (following standard naming - see Limitations)
        gh : xarray DataArray
            Array containing fields of geopotential height with at least coordinates latitude, longitude 
            and level (following standard naming - see Limitations)
        terms : str or sequence of str
            List of terms to compute. If None, returns all terms. Available options are:
            Pz; total available potential energy in the zonally averaged temperature distribution
            Kz; total kinetic energy in zonally averaged motion
            Pe; total eddy available potential energy [= sum_n Pn (n > 0 only) for spectral=True] (Note that 
            for spectral=True, an additional term, Sn, quantifying the rate of transfer of available potential 
            energy to eddies of wavenumber n from eddies of all other wavenumbers is also returned)
            Ke; total eddy kinetic energy [= sum_n Kn (n > 0 only) for spectral=True] (Note that for 
            spectral=True, an additional term, Ln, quantifying the rate of transfer of kinetic energy to eddies 
            of wavenumber n from eddies of all other wavenumbers is also returned)
            Cz; rate of conversion of zonal available potential energy to zonal kinetic energy
            Ca; rate of transfer of total available potential energy in the zonally averaged temperature 
            distribution (Pz) to total eddy available potential energy (Pe) [= sum_n Rn (n > 0 only) for 
            spectral=True]
            Ce; rate of transfer of total eddy available potential energy (Pe) to total eddy kinetic energy 
            (Ke) [= sum_n Cn (n > 0 only) for spectral=True]
            Ck; rate of transfer of total eddy kinetic energy (Ke) to total kinetic energy in zonally 
            averaged motion (Kz) [= sum_n Mn (n > 0 only) for spectral=True]
            Gz; rate of generation of zonal available potential energy due to the zonally averaged heating (Pz).
            Note that this term is computed as a residual (Cz + Ca) and cannot be returned in spectral space. 
            If Gz is requested with spectral=True, Gz is returned in real-space only
            Ge; rate of generation of eddy available potential energy (Pe). Note that this term is computed as 
            a residual (Ce - Ca) and cannot be returned in spectral space. If Ge is requested with spectral=True, 
            Ge is returned in real-space only
            Dz; rate of viscous dissipation of zonal kinetic energy (Kz). Note that this term is computed as a 
            residual (Cz - Ck) and cannot be returned in spectral space. If Dz is requested with spectral=True, Dz 
            is returned in real-space only
            De; rate of dissipation of eddy kinetic energy (Ke). Note that this term is computed as a residual 
            (Ce - Ck) and cannot be returned in spectral space. If De is requested with spectral=True, De is 
            returned in real-space only
        vgradz : bool, optional
            If True, uses `v-grad-z` approach for computing terms relating to conversion
            of potential energy to kinetic energy. Otherwise, defaults to using the 
            `omaga-alpha` approach (see reference above for details)
        spectral : bool, optional
            If True, computes all terms as a function of wavenumber on longitudinal bands. To use this 
            option, longitudes must be regularly spaced. Note that Ge and De are computed as residuals and
            cannot be computed in spectral space
        n_wavenumbers : int, optional
            Number of wavenumbers to retain either side of wavenumber=0. Obviously only does anything if 
            spectral=True
        integrate : bool, optional
            If True, computes and returns the integral of each term over the mass of the 
            atmosphere. Otherwise, only the integrands are returned.

        Returns
        -------
        atmos_energy_cycle : xarray Dataset
            
            
        Limitations
        -----------
        All input array coordinates must follow standard naming (see doppyo.utils.get_lat_name(), 
        doppyo.utils.get_lon_name(), etc)
        Pressure levels must be provided in units of hPa
            
        Notes
        -----
        The following notation is used below (stackable, e.g. *_ZT indicates the time average of the zonal 
        average):
        *_A -> area average over an isobaric surface
        *_a -> departure from area average
        *_Z -> zonal average
        *_z -> departure from zonal average
        *_T -> time average
        *_t -> departure from time average
        Additionally, capital variables indicate Fourier transforms:
        F(u) = U
        F(v) = V
        F(omega) = O
        F(gh) = A
        F(temp) = B
    """
    
    def _flip_n(da):
        """ Flips data along wavenumber coordinate """

        daf = da.copy()
        daf['n'] = -daf['n']

        return daf.sortby(daf['n'])


    def _truncate(F, n_truncate, dim):
        """ 
            Converts spatial frequency dim to wavenumber, n, and truncates all wavenumbers greater than 
            n_truncate 
        """
        F[dim] = 360 * F[dim]
        F = F.rename({dim : 'n'})
        F = F.where(abs(F.n) <= n_truncate, drop=True)
        return F, _flip_n(F)


    def _triple_terms(A, B, C):
        """ 
            Calculate triple term summation of the form \int_{m=-inf}^{inf} A(m) * B(n) * C(n - m)
        """

        # Use rolling operator to build shifted terms -----
        Am = A.rename({'n' : 'm'})
        Cnm = C.rolling(n=len(C.n), center=True).construct('m', fill_value=0)
        Cnm['m'] = -C['n'].values

        # Drop m = 0 and n < 0 -----
        Am = Am.where(Am['m'] != 0, drop=True) 
        Cnm = Cnm.where(Cnm['m'] != 0, drop=True)

        return (B * (Am * Cnm)).sum(dim='m', skipna=False)


    def _triple_terms_loop(A, B, C):
        """ 
            Calculate triple term summation of the form \int_{m=-inf}^{inf} A(m) * B(n) * C(n - m)
        """

        # Loop over all m's and perform rolling sum -----
        ms = A['n'].where(A['n'] != 0, drop=True).values
        ABC = A.copy() * 0
        for m in ms:
            Am = A.sel(n=m)
            Cnm = C.shift(n=int(m)).fillna(0)
            ABC = ABC + (Am * B * Cnm)

        return ABC
    
    if terms is None:
        terms = ['Pz', 'Kz', 'Pe', 'Ke', 'Cz', 'Ca', 'Ce', 'Ck', 'Gz', 'Ge', 'Dz', 'De']
    if isinstance(terms, str):
        terms = [terms]
    
    # Initialize some things -----
    if lat_name is None:
        lat_name = utils.get_lat_name(temp)
    if lon_name is None:
        lon_name = utils.get_lon_name(temp)
    if plevel_name is None:
        plevel_name = utils.get_plevel_name(temp)
    
    degtorad = utils.constants().pi / 180
    tan_lat = xr.ufuncs.tan(temp[lat_name] * degtorad)
    cos_lat = xr.ufuncs.cos(temp[lat_name] * degtorad) 
    
    # Determine the stability parameter using Saltzman's approach -----
    kappa = utils.constants().R_d / utils.constants().C_pd
    p_kap = (1000 / temp[plevel_name]) ** kappa
    theta_A = utils.average(temp * p_kap, [lat_name, lon_name], weights=cos_lat)
    dtheta_Adp = utils.differentiate_wrt(theta_A, dim=plevel_name, x=(theta_A[plevel_name] * 100))
    gamma = - p_kap * (utils.constants().R_d) / ((temp[plevel_name] * 100) * utils.constants().C_pd) / dtheta_Adp # [1/K]
    energies = gamma.rename('gamma').to_dataset()
    
    # Compute zonal terms
    # ========================
    
    if ('Pz' in terms):
    # Compute the total available potential energy in the zonally averaged temperature
    # distribution, Pz [also commonly called Az] -----
        temp_A = utils.average(temp, [lat_name, lon_name], weights=cos_lat)
        temp_Z = temp.mean(dim=lon_name)
        temp_Za = temp_Z - temp_A
        Pz_int = gamma * utils.constants().C_pd / 2 * temp_Za ** 2  # [J/kg]
        energies['Pz_int'] = Pz_int
        if integrate:
            Pz = _int_over_atmos(Pz_int, lat_name, lon_name, plevel_name, lon_dim=temp[lon_name]) # [J/m^2]
            energies['Pz'] = Pz
    
    if ('Kz' in terms):
    # Compute the total kinetic energy in zonally averaged motion, Kz [also commonly 
    # called Kz] -----
        u_Z = u.mean(dim=lon_name)
        v_Z = v.mean(dim=lon_name)
        Kz_int = 0.5 * (u_Z ** 2 + v_Z ** 2) # [J/kg]
        energies['Kz_int'] = Kz_int
        if integrate:
            Kz = _int_over_atmos(Kz_int, lat_name, lon_name, plevel_name, lon_dim=u[lon_name]) # [J/m^2]
            energies['Kz'] = Kz
    
    if ('Cz' in terms):
    # Compute the rate of conversion of zonal available potential energy (Pz) to zonal kinetic
    # energy (Kz), Cz [also commonly called Cz] -----
        if vgradz:
            if 'v_Z' not in locals():
                v_Z = v.mean(dim=lon_name)
            gh_Z = gh.mean(dim=lon_name)
            dghdlat = utils.differentiate_wrt(gh_Z, dim=lat_name, x=(gh_Z[lat_name] * degtorad))
            Cz_int = - (utils.constants().g / utils.constants().R_earth) * v_Z * dghdlat # [W/kg]
            energies['Cz_int'] = Cz_int
            if integrate:
                Cz = _int_over_atmos(Cz_int, lat_name, lon_name, plevel_name, lon_dim=gh[lon_name]) # [W/m^2]
                energies['Cz'] = Cz
        else:
            if 'temp_Za' not in locals():
                temp_A = utils.average(temp, [lat_name, lon_name], weights=cos_lat)
                temp_Z = temp.mean(dim=lon_name)
                temp_Za = temp_Z - temp_A
            omega_A = utils.average(omega, [lat_name, lon_name], weights=cos_lat)
            omega_Z = omega.mean(dim=lon_name)
            omega_Za = omega_Z - omega_A
            Cz_int = - (utils.constants().R_d / (temp[plevel_name] * 100)) * omega_Za * temp_Za # [W/kg]
            energies['Cz_int'] = Cz_int
            if integrate:
                Cz = _int_over_atmos(Cz_int, lat_name, lon_name, plevel_name, lon_dim=omega[lon_name]) # [W/m^2]
                energies['Cz'] = Cz
    
    # Compute eddy terms in Fourier space if spectral=True
    # ==========================================================
    if spectral:
        
        if ('Pe' in terms):
        # Compute the total available potential energy eddies of wavenumber n, Pn -----
            Bp, Bn = _truncate(utils.fft(temp, dim=lon_name, nfft=len(temp[lon_name]), twosided=True, shift=True) / 
                              len(temp[lon_name]), n_truncate=n_wavenumbers, dim='f_'+lon_name)

            Pn_int = (gamma * utils.constants().C_pd * abs(Bp) ** 2)
            energies['Pn_int'] = Pn_int
            if integrate:
                Pn = _int_over_atmos(Pn_int, lat_name, lon_name, plevel_name, lon_dim=temp[lon_name]) # [J/m^2]
                energies['Pn'] = Pn

        # Compute the rate of transfer of available potential energy to eddies of 
        # wavenumber n from eddies of all other wavenumbers, Sn -----
            Up, Un = _truncate(utils.fft(u, dim=lon_name, nfft=len(u[lon_name]), twosided=True, shift=True) /
                               len(u[lon_name]), n_truncate=n_wavenumbers, dim='f_'+lon_name)
            Vp, Vn = _truncate(utils.fft(v, dim=lon_name, nfft=len(v[lon_name]), twosided=True, shift=True) /
                               len(v[lon_name]), n_truncate=n_wavenumbers, dim='f_'+lon_name)
            Op, On = _truncate(utils.fft(omega, dim=lon_name, nfft=len(omega[lon_name]), twosided=True, shift=True) /
                               len(omega[lon_name]), n_truncate=n_wavenumbers, dim='f_'+lon_name)
                
            dBpdlat = utils.differentiate_wrt(Bp, dim=lat_name, x=(Bp[lat_name] * degtorad))
            dBndlat = utils.differentiate_wrt(Bn, dim=lat_name, x=(Bn[lat_name] * degtorad))
            dBpdp = utils.differentiate_wrt(Bp, dim=plevel_name, x=(Bp[plevel_name] * 100))
            dBndp = utils.differentiate_wrt(Bn, dim=plevel_name, x=(Bn[plevel_name] * 100))

            if loop_triple_terms:
                BpBnUp = _triple_terms_loop(Bp, Bn, Up)
                BpBpUn = _triple_terms_loop(Bp, Bp, Un)
                BpglBnVp = _triple_terms_loop(Bp, dBndlat, Vp)
                BpglBpVn = _triple_terms_loop(Bp, dBpdlat, Vn)
                BpgpBnOp = _triple_terms_loop(Bp, dBndp, Op)
                BpgpBpOn = _triple_terms_loop(Bp, dBpdp, On)
                BpBnOp = _triple_terms_loop(Bp, Bn, Op)
                BpBpOn = _triple_terms_loop(Bp, Bp, On)
            else:
                BpBnUp = _triple_terms(Bp, Bn, Up)
                BpBpUn = _triple_terms(Bp, Bp, Un)
                BpglBnVp = _triple_terms(Bp, dBndlat, Vp)
                BpglBpVn = _triple_terms(Bp, dBpdlat, Vn)
                BpgpBnOp = _triple_terms(Bp, dBndp, Op)
                BpgpBpOn = _triple_terms(Bp, dBpdp, On)
                BpBnOp = _triple_terms(Bp, Bn, Op)
                BpBpOn = _triple_terms(Bp, Bp, On)

            Sn_int = -gamma * utils.constants().C_pd * (1j * Bp['n']) / \
                         (utils.constants().R_earth * xr.ufuncs.cos(Bp[lat_name] * degtorad)) * \
                         (BpBnUp + BpBpUn) + \
                     gamma * utils.constants().C_pd / utils.constants().R_earth * \
                         (BpglBnVp + BpglBpVn) + \
                     gamma * utils.constants().C_pd * (BpgpBnOp + BpgpBpOn) + \
                     gamma * utils.constants().R_d / Bp[plevel_name] * \
                         (BpBnOp + BpBpOn)
            energies['Sn_int'] = Sn_int
            if integrate:
                Sn = abs(_int_over_atmos(Sn_int, lat_name, lon_name, plevel_name, lon_dim=temp[lon_name])) # [W/m^2]
                energies['Sn'] = Sn
                
        if ('Ke' in terms):
        # Compute the total kinetic energy in eddies of wavenumber n, Kn -----
            if 'U' not in locals():
                Up, Un = _truncate(utils.fft(u, dim=lon_name, nfft=len(u[lon_name]), twosided=True, shift=True) /
                                   len(u[lon_name]), n_truncate=n_wavenumbers, dim='f_'+lon_name)
            if 'V' not in locals():
                Vp, Vn = _truncate(utils.fft(v, dim=lon_name, nfft=len(v[lon_name]), twosided=True, shift=True) / 
                                   len(v[lon_name]), n_truncate=n_wavenumbers, dim='f_'+lon_name)

            Kn_int = abs(Up) ** 2 + abs(Vp) ** 2
            energies['Kn_int'] = Kn_int
            if integrate:
                Kn = _int_over_atmos(Kn_int, lat_name, lon_name, plevel_name, lon_dim=u[lon_name]) # [J/m^2]
                energies['Kn'] = Kn

        # Compute the rate of transfer of kinetic energy to eddies of wavenumber n from 
        # eddies of all other wavenumbers, Ln -----
            if 'O' not in locals():
                Op, On = _truncate(utils.fft(omega, dim=lon_name, nfft=len(omega[lon_name]), twosided=True, shift=True) / 
                                   len(omega[lon_name]), n_truncate=n_wavenumbers, dim='f_'+lon_name)
                
            dUpdp = utils.differentiate_wrt(Up, dim=plevel_name, x=(Up[plevel_name] * 100))
            dVpdp = utils.differentiate_wrt(Vp, dim=plevel_name, x=(Vp[plevel_name] * 100))
            dOpdp = utils.differentiate_wrt(Op, dim=plevel_name, x=(Op[plevel_name] * 100))
            dOndp = utils.differentiate_wrt(On, dim=plevel_name, x=(On[plevel_name] * 100))
            dVpcdl = utils.differentiate_wrt(Vp * cos_lat, dim=lat_name, x=(Vp[lat_name] * degtorad))
            dVncdl = utils.differentiate_wrt(Vn * cos_lat, dim=lat_name, x=(Vn[lat_name] * degtorad))
            dUpdl = utils.differentiate_wrt(Up, dim=lat_name, x=(Up[lat_name] * degtorad))
            dVpdl = utils.differentiate_wrt(Vp, dim=lat_name, x=(Vp[lat_name] * degtorad))

            if loop_triple_terms:
                UpUnUp = _triple_terms_loop(Up, Un, Up)
                UpUpUn = _triple_terms_loop(Up, Up, Un)
                VpVnUp = _triple_terms_loop(Vp, Vn, Up)
                VpVpUn = _triple_terms_loop(Vp, Vp, Un)
                VpUnUp = _triple_terms_loop(Vp, Un, Up)
                VpUpUn = _triple_terms_loop(Vp, Up, Un)
                UpVnUp = _triple_terms_loop(Up, Vn, Up)
                UpVpUn = _triple_terms_loop(Up, Vp, Un)
                gpUpUngpOp = _triple_terms_loop(dUpdp, Un, dOpdp)
                gpUpUpgpOn = _triple_terms_loop(dUpdp, Up, dOndp)
                gpVpVngpOp = _triple_terms_loop(dVpdp, Vn, dOpdp)
                gpVpVpgpOn = _triple_terms_loop(dVpdp, Vp, dOndp)
                glUpUnglVpc = _triple_terms_loop(dUpdl, Un, dVpcdl)
                glUpUpglVnc = _triple_terms_loop(dUpdl, Up, dVncdl)
                glVpVnglVpc = _triple_terms_loop(dVpdl, Vn, dVpcdl)
                glVpVpglVnc = _triple_terms_loop(dVpdl, Vp, dVncdl)
            else:
                UpUnUp = _triple_terms(Up, Un, Up)
                UpUpUn = _triple_terms(Up, Up, Un)
                VpVnUp = _triple_terms(Vp, Vn, Up)
                VpVpUn = _triple_terms(Vp, Vp, Un)
                VpUnUp = _triple_terms(Vp, Un, Up)
                VpUpUn = _triple_terms(Vp, Up, Un)
                UpVnUp = _triple_terms(Up, Vn, Up)
                UpVpUn = _triple_terms(Up, Vp, Un)
                gpUpUngpOp = _triple_terms(dUpdp, Un, dOpdp)
                gpUpUpgpOn = _triple_terms(dUpdp, Up, dOndp)
                gpVpVngpOp = _triple_terms(dVpdp, Vn, dOpdp)
                gpVpVpgpOn = _triple_terms(dVpdp, Vp, dOndp)
                glUpUnglVpc = _triple_terms(dUpdl, Un, dVpcdl)
                glUpUpglVnc = _triple_terms(dUpdl, Up, dVncdl)
                glVpVnglVpc = _triple_terms(dVpdl, Vn, dVpcdl)
                glVpVpglVnc = _triple_terms(dVpdl, Vp, dVncdl)

            Ln_int = -(1j * Up['n']) / (utils.constants().R_earth * cos_lat) * \
                         (UpUnUp - UpUpUn) + \
                     (1j * Vp['n']) / (utils.constants().R_earth * cos_lat) * \
                         (VpVnUp - VpVpUn) - \
                     tan_lat / utils.constants().R_earth * \
                         (VpUnUp + VpUpUn) + \
                     tan_lat / utils.constants().R_earth * \
                         (UpVnUp + UpVpUn) + \
                     (gpUpUngpOp + gpUpUpgpOn) + \
                     (gpVpVngpOp + gpVpVpgpOn) + \
                     1 / (utils.constants().R_earth * cos_lat) * \
                         (glUpUnglVpc + glUpUpglVnc + glVpVnglVpc + glVpVpglVnc)
            energies['Ln_int'] = Ln_int
            if integrate:
                Ln = abs(_int_over_atmos(Ln_int, lat_name, lon_name, plevel_name, lon_dim=u[lon_name])) # [W/m^2]
                energies['Ln'] = Ln
        
        if ('Ca' in terms):
        # Compute the rate of transfer of zonal available potential energy to eddy 
        # available potential energy in wavenumber n, Rn -----
            if 'temp_Z' not in locals():
                temp_Z = temp.mean(dim=lon_name)
            if 'V' not in locals():
                Vp, Vn = _truncate(utils.fft(v, dim=lon_name, nfft=len(v[lon_name]), twosided=True, shift=True) / 
                                   len(v[lon_name]), n_truncate=n_wavenumbers, dim='f_'+lon_name)
            if 'B' not in locals():
                Bp, Bn = _truncate(utils.fft(temp, dim=lon_name, nfft=len(temp[lon_name]), twosided=True, shift=True) / 
                                   len(temp[lon_name]), n_truncate=n_wavenumbers, dim='f_'+lon_name)
            if 'O' not in locals():
                Op, On = _truncate(utils.fft(omega, dim=lon_name, nfft=len(omega[lon_name]), twosided=True, shift=True) / 
                                   len(omega[lon_name]), n_truncate=n_wavenumbers, dim='f_'+lon_name)

            dtemp_Zdlat = utils.differentiate_wrt(temp_Z, dim=lat_name, x=(temp_Z[lat_name] * degtorad))
            theta = temp * p_kap
            theta_Z = theta.mean(dim=lon_name)
            theta_Za = theta_Z - theta_A
            dtheta_Zadp = utils.differentiate_wrt(theta_Za, dim=plevel_name, x=(theta_Za[plevel_name] * 100))
            Rn_int = gamma * utils.constants().C_pd * ((dtemp_Zdlat / utils.constants().R_earth) * (Vp * Bn + Vn * Bp) + 
                                                       (p_kap * dtheta_Zadp) * (Op * Bn + On * Bp)) # [W/kg]
            energies['Rn_int'] = Rn_int
            if integrate:
                Rn = abs(_int_over_atmos(Rn_int, lat_name, lon_name, plevel_name, lon_dim=temp[lon_name])) # [W/m^2]
                energies['Rn'] = Rn

        if ('Ce' in terms):
        # Compute the rate of conversion of available potential energy of wavenumber n 
        # to eddy kinetic energy of wavenumber n, Cn -----
            if vgradz:
                if 'U' not in locals():
                    Up, Un = _truncate(utils.fft(u, dim=lon_name, nfft=len(u[lon_name]), twosided=True, shift=True) / 
                                       len(u[lon_name]), n_truncate=n_wavenumbers, dim='f_'+lon_name)
                if 'V' not in locals():
                    Vp, Vn = _truncate(utils.fft(v, dim=lon_name, nfft=len(v[lon_name]), twosided=True, shift=True) / 
                                       len(v[lon_name]), n_truncate=n_wavenumbers, dim='f_'+lon_name)
                Ap, An = _truncate(utils.fft(gh, dim=lon_name, nfft=len(gh[lon_name]), twosided=True, shift=True) / 
                                   len(gh[lon_name]), n_truncate=n_wavenumbers, dim='f_'+lon_name)

                dApdlat = utils.differentiate_wrt(Ap, dim=lat_name, x=(Ap[lat_name] * degtorad))
                dAndlat = utils.differentiate_wrt(An, dim=lat_name, x=(An[lat_name] * degtorad))

                Cn_int = (((-1j * utils.constants().g * Up['n']) / \
                           (utils.constants().R_earth * xr.ufuncs.cos(Up[lat_name] * degtorad))) * \
                                (Ap * Un - An * Up)) - \
                         ((utils.constants().g / utils.constants().R_earth) * \
                                (dApdlat * Vn + dAndlat * Vp)) # [W/kg]
                energies['Cn_int'] = Cn_int
                if integrate:
                    Cn = abs(_int_over_atmos(Cn_int, lat_name, lon_name, plevel_name, lon_dim=u[lon_name])) # [W/m^2]
                    energies['Cn'] = Cn
            else:
                if 'O' not in locals():
                    Op, On = _truncate(utils.fft(omega, dim=lon_name, nfft=len(omega[lon_name]), twosided=True, shift=True) / 
                                       len(omega[lon_name]), n_truncate=n_wavenumbers, dim='f_'+lon_name)
                if 'B' not in locals():
                    Bp, Bn = _truncate(utils.fft(temp, dim=lon_name, nfft=len(temp[lon_name]), twosided=True, shift=True) / 
                                       len(temp[lon_name]), n_truncate=n_wavenumbers, dim='f_'+lon_name)
                Cn_int = - (utils.constants().R_d / (omega[plevel_name] * 100)) * (Op * Bn + On * Bp) # [W/kg]
                energies['Cn_int'] = Cn_int
                if integrate:
                    Cn = abs(_int_over_atmos(Cn_int, lat_name, lon_name, plevel_name, lon_dim=temp[lon_name])) # [W/m^2]
                    energies['Cn'] = Cn
    
        if ('Ck' in terms):
        # Compute the rate of transfer of kinetic energy to the zonally averaged flow 
        # from eddies of wavenumber n, Mn -----
            if 'v_Z' not in locals():
                v_Z = v.mean(dim=lon_name)
            if 'u_Z' not in locals():
                u_Z = u.mean(dim=lon_name)
            if 'U' not in locals():
                Up, Un = _truncate(utils.fft(u, dim=lon_name, nfft=len(u[lon_name]), twosided=True, shift=True) / 
                                   len(u[lon_name]), n_truncate=n_wavenumbers, dim='f_'+lon_name)
            if 'V' not in locals():
                Vp, Vn = _truncate(utils.fft(v, dim=lon_name, nfft=len(v[lon_name]), twosided=True, shift=True) / 
                                   len(v[lon_name]), n_truncate=n_wavenumbers, dim='f_'+lon_name)
            if 'O' not in locals():
                Op, On = _truncate(utils.fft(omega, dim=lon_name, nfft=len(omega[lon_name]), twosided=True, shift=True) / 
                                   len(omega[lon_name]), n_truncate=n_wavenumbers, dim='f_'+lon_name)
            dv_Zdlat = utils.differentiate_wrt(v_Z, dim=lat_name, x=(v[lat_name] * degtorad))
            du_Zndlat = utils.differentiate_wrt(u_Z / xr.ufuncs.cos(u[lat_name] * degtorad), 
                                            dim=lat_name, x=(u[lat_name] * degtorad))
            dv_Zdp = utils.differentiate_wrt(v_Z, dim=plevel_name, x=(v[plevel_name] * 100))
            du_Zdp = utils.differentiate_wrt(u_Z, dim=plevel_name, x=(u[plevel_name] * 100))

            Mn_int = (-2 * Up * Un * v_Z * tan_lat / utils.constants().R_earth) + \
                     (2 * Vp * Vn * dv_Zdlat / utils.constants().R_earth + (Vp * On + Vn * Op) * dv_Zdp) + \
                     ((Up * On + Un * Op) * du_Zdp) + \
                     ((Up * Vn + Un * Vp) * xr.ufuncs.cos(u[lat_name] * degtorad) / \
                         utils.constants().R_earth * du_Zndlat) # [W/kg]
            energies['Mn_int'] = Mn_int
            if integrate:
                Mn = abs(_int_over_atmos(Mn_int, lat_name, lon_name, plevel_name, lon_dim=u[lon_name])) # [W/m^2]
                energies['Mn'] = Mn
        
    else:
        
        if ('Pe' in terms):
        # Compute the total eddy available potential energy, Pe [also commonly called 
        # Ae] -----
            if 'temp_Z' not in locals():
                temp_Z = temp.mean(dim=lon_name)
            temp_z = temp - temp_Z
            Pe_int = gamma * utils.constants().C_pd / 2 * (temp_z ** 2).mean(dim=lon_name)  # [J/kg]
            energies['Pe_int'] = Pe_int
            if integrate:
                Pe = _int_over_atmos(Pe_int, lat_name, lon_name, plevel_name, lon_dim=temp[lon_name]) # [J/m^2]
                energies['Pe'] = Pe
        
        if ('Ke' in terms):
        # Compute the total eddy kinetic energy, Ke -----
            if 'u_Z' not in locals():
                    u_Z = u.mean(dim=lon_name)
            if 'v_Z' not in locals():
                    v_Z = v.mean(dim=lon_name)
            u_z = u - u_Z
            v_z = v - v_Z
            Ke_int = 0.5 * (u_z ** 2 + v_z ** 2).mean(dim=lon_name) # [J/kg]
            energies['Ke_int'] = Ke_int
            if integrate:
                Ke = _int_over_atmos(Ke_int, lat_name, lon_name, plevel_name, lon_dim=u[lon_name]) # [J/m^2]
                energies['Ke'] = Ke
                
        if ('Ca' in terms):
        # Compute the rate of transfer of total available potential energy in the zonally 
        # averaged temperature distribution (Pz) to total eddy available potential energy 
        # (Pe), Ca -----
            if 'v_Z' not in locals():
                v_Z = v.mean(dim=lon_name)
            if 'temp_Z' not in locals():
                temp_Z = temp.mean(dim=lon_name)
            if 'omega_Z' not in locals():
                omega_Z = omega.mean(dim=lon_name)
            if 'theta_Z' not in locals():
                theta = temp * p_kap
                theta_Z = theta.mean(dim=lon_name)
            if 'dtemp_Zdlat' not in locals():
                dtemp_Zdlat = utils.differentiate_wrt(temp_Z, dim=lat_name, x=(temp_Z[lat_name] * degtorad))
            v_z = v - v_Z
            temp_z = temp - temp_Z
            omega_z = omega - omega_Z
            oT_Z = (omega_z * temp_z).mean(dim=lon_name)
            oT_A = utils.average(omega_z * temp_z, [lat_name, lon_name], weights=cos_lat)
            oT_Za = oT_Z - oT_A
            theta_Za = theta_Z - theta_A
            dtheta_Zadp = utils.differentiate_wrt(theta_Za, dim=plevel_name, x=(theta_Za[plevel_name] * 100))
            Ca_int = - gamma * utils.constants().C_pd * \
                           (((v_z * temp_z).mean(dim=lon_name) * dtemp_Zdlat / utils.constants().R_earth) + \
                            (p_kap * oT_Za * dtheta_Zadp)) # [W/kg]
            energies['Ca_int'] = Ca_int
            if integrate:
                Ca = _int_over_atmos(Ca_int, lat_name, lon_name, plevel_name, lon_dim=v[lon_name]) # [W/m^2]
                energies['Ca'] = Ca
            
        if ('Ce' in terms):
        # Compute the rate of transfer of total eddy available potential energy (Pe) to 
        # total eddy kinetic energy (Ke), Ce -----
            if 'temp_Z' not in locals():
                temp_Z = temp.mean(dim=lon_name)
            if 'omega_Z' not in locals():
                omega_Z = omega.mean(dim=lon_name)
            temp_z = temp - temp_Z
            omega_z = omega - omega_Z
            Ce_int = - (utils.constants().R_d / (temp[plevel_name] * 100)) * \
                           (omega_z * temp_z).mean(dim=lon_name) # [W/kg]  
            energies['Ce_int'] = Ce_int
            if integrate:
                Ce = _int_over_atmos(Ce_int, lat_name, lon_name, plevel_name, lon_dim=temp[lon_name]) # [W/m^2]
                energies['Ce'] = Ce
        
        if ('Ck' in terms):
        # Compute the rate of transfer of total eddy kinetic energy (Ke) to total kinetic 
        # energy in zonally averaged motion (Kz), Ck -----
            if 'u_Z' not in locals():
                    u_Z = u.mean(dim=lon_name)
            if 'v_Z' not in locals():
                    v_Z = v.mean(dim=lon_name)
            if 'omega_Z' not in locals():
                omega_Z = omega.mean(dim=lon_name)
            u_z = u - u_Z
            v_z = v - v_Z
            omega_z = omega - omega_Z
            du_Zndlat = utils.differentiate_wrt(u_Z / cos_lat, dim=lat_name, x=(u_Z[lat_name] * degtorad))
            dv_Zdlat = utils.differentiate_wrt(v_Z, dim=lat_name, x=(v_Z[lat_name] * degtorad))
            du_Zdp = utils.differentiate_wrt(u_Z, dim=plevel_name, x=(u_Z[plevel_name] * 100))
            dv_Zdp = utils.differentiate_wrt(v_Z, dim=plevel_name, x=(v_Z[plevel_name] * 100))
            Ck_int = (u_z * v_z).mean(dim=lon_name)  * cos_lat * du_Zndlat / utils.constants().R_earth + \
                     (u_z * omega_z).mean(dim=lon_name) * du_Zdp + \
                     (v_z ** 2).mean(dim=lon_name) * dv_Zdlat / utils.constants().R_earth + \
                     (v_z * omega_z).mean(dim=lon_name) * dv_Zdp - \
                     (u_z ** 2).mean(dim=lon_name) * v_Z * tan_lat / utils.constants().R_earth
            energies['Ck_int'] = Ck_int
            if integrate:
                Ck = _int_over_atmos(Ck_int, lat_name, lon_name, plevel_name, lon_dim=temp[lon_name]) # [W/m^2]
                energies['Ck'] = Ck
                
    if ('Gz' in terms):
    # Compute the rate of generation of zonal available potential energy due to the zonally
    # averaged heating, Gz -----
        if ('Cz' not in terms) | ('Ca' not in terms):
            raise ValueError('The rate of generation of zonal available potential energy, Gz, is computed from the sum of Cz and Ca. Please add these to the list, terms=[<terms>].')
        if spectral:
            warnings.warn('Rate of generation of zonal available potential energy is computed from the sum of Cz and Ca and cannot be computed in Fourier space. Returning Gz in real-space.')
            Ca_int = Rn_int.where(Rn_int.n > 0, drop=True).sum(dim='n').real # sum Rn to get Ca
        Gz_int = Cz_int + Ca_int
        energies['Gz_int'] = Gz_int
        if integrate:
            Gz = _int_over_atmos(Gz_int, lat_name, lon_name, plevel_name, lon_dim=temp[lon_name]) # [W/m^2]
            energies['Gz'] = Gz

    if ('Ge' in terms):
    # Compute the rate of generation of eddy available potential energy (Ae), Ge -----
        if ('Ce' not in terms) | ('Ca' not in terms):
            raise ValueError('The rate of generation of eddy available potential energy, Ge, is computed from the residual of Ce and Ca. Please add these to the list, terms=[<terms>].')
        if spectral:
            warnings.warn('The rate of generation of eddy available potential energy is computed from the residual of Ce and Ca and cannot be computed in Fourier space. Returning Ge in real-space.')
            Ce_int = Cn_int.where(Cn_int.n > 0, drop=True).sum(dim='n').real # sum Cn to get Ce
            if 'Ca_int' not in locals():
                Ca_int = Rn_int.where(Rn_int.n > 0, drop=True).sum(dim='n').real # sum Rn to get Ca
        Ge_int = Ce_int - Ca_int
        energies['Ge_int'] = Ge_int
        if integrate:
            Ge = _int_over_atmos(Ge_int, lat_name, lon_name, plevel_name, lon_dim=temp[lon_name]) # [W/m^2]
            energies['Ge'] = Ge
    
    if ('Dz' in terms):
    # Compute the rate of viscous dissipation of zonal kinetic energy, Dz -----
        if ('Cz' not in terms) | ('Ck' not in terms):
            raise ValueError('The rate of viscous dissipation of zonal kinetic energy, Dz, is computed from the residual of Cz and Ck. Please add these to the list, terms=[<terms>].')
        if spectral:   
            warnings.warn('The rate of viscous dissipation of zonal kinetic energy, Dz, is computed from the residual of Cz and Ck and cannot be computed in Fourier space. Returning De in real-space.')
            Ck_int = Mn_int.where(Mn_int.n > 0, drop=True).sum(dim='n').real # sum Mn to get Ck
        Dz_int = Cz_int - Ck_int
        energies['Dz_int'] = Dz_int
        if integrate:
            Dz = _int_over_atmos(Dz_int, lat_name, lon_name, plevel_name, lon_dim=temp[lon_name]) # [W/m^2]
            energies['Dz'] = Dz

    if ('De' in terms):
    # Compute the rate of dissipation of eddy kinetic energy (Ke), De -----
        if ('Ce' not in terms) | ('Ck' not in terms):
            raise ValueError('The rate of viscous dissipation of eddy kinetic energy, De, is computed from the residual of Ce and Ck. Please add these to the list, terms=[<terms>].')
        if spectral:
            warnings.warn('The rate of viscous dissipation of eddy kinetic energy, De, is computed from the residual of Ce and Ck and cannot be computed in Fourier space. Returning De in real-space.')
            if 'Ce_int' not in locals():
                Ce_int = Cn_int.where(Cn_int.n > 0, drop=True).sum(dim='n').real # sum Cn to get Ce
            if 'Ck_int' not in locals():
                Ck_int = Mn_int.where(Mn_int.n > 0, drop=True).sum(dim='n').real # sum Mn to get Ck
        De_int = Ce_int - Ck_int
        energies['De_int'] = De_int
        if integrate:
            De = _int_over_atmos(De_int, lat_name, lon_name, plevel_name, lon_dim=temp[lon_name]) # [W/m^2]
            energies['De'] = De
    
    return energies


# ===================================================================================================
def auto_merge(paths, preprocess=None, parallel=True, **kwargs):
    """
    Automatically merge a split xarray Dataset. This is designed to behave like
    `xarray.open_mfdataset`, except it supports concatenation along multiple
    dimensions.
    Parameters
    ----------
    datasets : str or list of str or list of xarray.Dataset
        Either a glob expression or list of paths as you would pass to
        xarray.open_mfdataset, or a list of xarray datasets. If a list of
        datasets is passed, you should make sure that they are represented
        as dask arrays to avoid reading the whole dataset into memory.
    Returns
    -------
    xarray.Dataset
        The merged dataset.
    """
    
    if parallel:
        # wrap the open_dataset, getattr, and preprocess with delayed
        open_ = dask.delayed(xr.open_dataset)
        getattr_ = dask.delayed(getattr)
        if preprocess is not None:
            preprocess = dask.delayed(preprocess)
    else:
        open_ = open_dataset
        getattr_ = getattr

    datasets = [open_(p, **kwargs) for p in paths]
    file_objs = [getattr_(ds, '_file_obj') for ds in datasets]
    if preprocess is not None:
        datasets = [preprocess(ds) for ds in datasets]

    if parallel:
        # calling compute here will return the datasets/file_objs lists,
        # the underlying datasets will still be stored as dask arrays
        datasets, file_objs = dask.compute(datasets, file_objs)

    def _combine_along_last_dim(datasets):
        merged = []

        # Determine the dimension along which the dataset is split
        split_dims = [d for d in datasets[0].dims if
                      len(np.unique([ds[d].values[0] for ds in datasets])) > 1]

        # Concatenate along one of the split dimensions
        concat_dim = split_dims[-1]

        # Group along the remaining dimensions and concatenate within each
        # group.
        sorted_ds = sorted(datasets, key=lambda ds: tuple(ds[d].values[0]
                                                          for d in split_dims))
        for _, group in itertools.groupby(
                sorted_ds,
                key=lambda ds: tuple(ds[d].values[0] for d in split_dims[:-1])
                ):
            merged.append(xr.auto_combine(group, concat_dim=concat_dim))

        return merged

    merged = datasets
    while len(merged) > 1:
        merged = _combine_along_last_dim(merged)

    return merged[0]


# ===================================================================================================
import dask
def _drop_offending_variables(ds):
    drop_vars = ['average_T1','average_T2','average_DT','time_bounds', 'geolat_t', 'geolat_c', 'geolon_t', 'geolon_c']
    for drop_var in drop_vars:
        if (drop_var in ds.data_vars) | (drop_var in ds.coords):
            ds = ds.drop(drop_var)
    return ds

def _load_ncfile(row, variables, chunks, resample_time_like, convert_time_to_lead, clip_time_at, time_dim='time', **kwargs):
    """ 
        Lazily load row[1] and add coordinates stored in dictionary in row[0] 
        A number of unecessary variables are automatically droppped while loading as these caused issues in concatenation
    """
    
    coords = row[0]
    path = row[-1]
    
    # Lazily load the dataset -----
    dataset = xr.open_mfdataset(path, chunks, preprocess=_drop_offending_variables, **kwargs)[variables] 
        
    # Add new coordinates -----
    for key, value in zip(coords.keys(), coords.values()):
        dataset.coords[key] = value
        
    # Resample time dimension -----
    if resample_time_like is not None:
        resample_freq = resample_time_like[0]
        resample_method = resample_time_like[1]
        if resample_method == 'mean':
            dataset = dataset.resample({time_dim: resample_freq}).mean(time_dim)
        elif resample_method == 'sum':
            dataset = dataset.resample({time_dim: resample_freq}).sum(time_dim)
        else:
            raise ValueError('Unrecognised resample method. Method options are "mean" and "sum", provided via resample_time_like=[freq, method]')
    
    # Clip time dimension -----
    if (clip_time_at is not None) and (clip_time_at < len(dataset[time_dim])):
        dataset = dataset.isel({time_dim : range(clip_time_at)})
    
    # Convert "time" dimension to a "time since date" (lead_time) dimension -----
    if convert_time_to_lead:
        if xr.coding.times.contains_cftime_datetimes(dataset[time_dim]):
            freq = pd.infer_freq(xr.coding.times.cftime_to_nptime(dataset[time_dim].values[:3]))
        else:
            freq = pd.infer_freq(dataset[time_dim].values[:3])
        dataset[time_dim] = np.arange(len(dataset[time_dim]))
        dataset = dataset.rename({time_dim : 'lead_time'})
        dataset.lead_time.attrs = {'units' : freq}

    if isinstance(variables, list):
        return dataset
    else:
        return dataset.to_dataset()

def load_and_concat(rows, variables, chunks=None, resample_time_like=None, convert_time_to_lead=False, clip_time_at=None, 
                    time_dim='time', **kwargs):
    """ 
        Lazily load all paths in rows in parallel and concatenate into single object 
        The loading functions below expect a list of tuples with the following format:
            ```
            paths = [({dictionary of coordinate names, and their values, to expand and concatentate along}, [list of files corresponding to coordinates in dictionary])]
            ```
        For example,
            ```
            paths = [({'ensemble' : 1}, ['path/to/file_1_containing_ensemble_1', 'path/to/file_2_containing_ensemble_1']), 
                     ({'ensemble' : 2}, ['path/to/file_1_containing_ensemble_2', 'path/to/file_2_containing_ensemble_2'])]
            ```
    """
    
    open_ = dask.delayed(_load_ncfile)
    datasets = [open_(row, variables=variables, chunks=chunks, resample_time_like=resample_time_like,
                      convert_time_to_lead=convert_time_to_lead, clip_time_at=clip_time_at, time_dim=time_dim, **kwargs) for row in rows]
    datasets = dask.compute(datasets)[0]

    # Get list of new_dims, dropping those which only have a single element -----
    all_dims = [row[0] for row in rows]
    new_dims = []
    for key in all_dims[0]:
        if len(np.unique([val[key] for val in all_dims])) > 1:
            new_dims.append(key)
    
    if (chunks is not None) & (convert_time_to_lead is True):
        chunks['lead_time'] = chunks.pop(time_dim)
    if len(new_dims) == 0:
        return xr.auto_combine(datasets)[variables]
    elif len(new_dims) == 1:
        return xr.concat(datasets, dim=new_dims[0])[variables]
    else:
        return xr.concat(datasets, dim='stack') \
                 .set_index(stack=new_dims).unstack('stack')[variables]


# ===================================================================================================   
def interpolate_lonlat(da, lon_des, lat_des):
    """
        Interpolate to specified latitude and longitude values, wrapping edges longitudinally and flipping edges latitudinally 
    """

    lat_name = utils.get_lat_name(da)
    lon_name = utils.get_lon_name(da)
    dims = da.dims

    # Wrap longitudes at egdes-----
    da_minlon = da.isel({lon_name : 0})
    minlon = da_minlon[lon_name]
    da_maxlon = da.isel({lon_name : -1})
    maxlon = da_maxlon[lon_name]
    da_minlon[lon_name] = minlon + 360
    da_maxlon[lon_name] = maxlon - 360
    da_lonwrap = xr.concat([da_maxlon, da, da_minlon], dim=lon_name)

    # Flip latitudes at edges -----
    da_minlat = da_lonwrap.isel({lat_name : 0})
    minlat = da_minlat[lat_name] 
    da_maxlat = da_lonwrap.isel({lat_name : -1})
    maxlat = da_maxlat[lat_name]
    da_minlat[lat_name] = maxlat - 180
    da_maxlat[lat_name] = minlat + 180
    da_wrap = xr.concat([da_minlat, da_lonwrap, da_maxlat], dim=lat_name).chunk({lon_name:-1, lat_name:-1})

    return da_wrap.interp({lon_name : lon_des, lat_name : lat_des}).transpose(*dims)


 # ===================================================================================================   
def global_average(da):
    """
        Returns the area weighted global average of da
    """
    def _get_area_weights(lon, lat):
        """
            Get area weights, wrapping edges longitudinally and flipping edges latitudinally 
        """

        lat_name = utils.get_lat_name(lat)
        lon_name = utils.get_lon_name(lon)

        dlon = lon.diff(lon_name) / 2
        minlon = lon.isel({lon_name:0})-dlon.isel({lon_name:0})
        minlon[lon_name] = minlon.values
        maxlon = lon.isel({lon_name:-1})+dlon.isel({lon_name:-1})
        maxlon[lon_name] = maxlon.values
        midlon = lon.isel({lon_name:slice(0,-1)})+dlon.values
        midlon[lon_name] = midlon.values
        lonb = xr.concat([minlon, midlon, maxlon], dim=lon_name)

        dlat = lat.diff(lat_name) / 2
        minlat = lat.isel({lat_name:0})-dlat.isel({lat_name:0})
        minlat[lat_name] = minlat.values
        maxlat = lat.isel({lat_name:-1})+dlat.isel({lat_name:-1})
        maxlat[lat_name] = maxlat.values
        midlat = lat.isel({lat_name:slice(0,-1)})+dlat.values
        midlat[lat_name] = midlat.values
        latb = xr.concat([minlat, midlat, maxlat], dim=lat_name)

        xb, yb = utils.xy_from_lonlat(lonb,latb)
        dxb = abs(xb.diff(lon_name))
        dyb = abs(yb.diff(lat_name))
        area = dxb * dyb
        area[lon_name] = lon.values
        area[lat_name] = lat.values

        return area
    
    lat_name = utils.get_lat_name(da)
    lon_name = utils.get_lon_name(da)
    
    return utils.average(da, dim=[lat_name, lon_name], 
                         weights=_get_area_weights(da[lon_name], 
                                                   da[lat_name]))
