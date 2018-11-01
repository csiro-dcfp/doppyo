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