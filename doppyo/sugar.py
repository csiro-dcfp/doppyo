"""
    Collection of old doppyo functions and useful tidbits for internal dcfp use
    Authors: Dougie Squire and Thomas Moore
    Date created: 01/10/2018
    Python Version: 3.6
"""

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