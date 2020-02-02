"""
    doppyo functions for assessing one dataset relative to another (usually model output to observation)
    
    In the following we refer to the datasets being assessed as comparison data (da_cmp) and reference
    data (da_ref). We seek to assess the skill of the former relative to the latter. Usually, da_cmp
    and da_ref comprise model output (e.g. forecasts) and observations, respectively.

    API
    ===
"""

__all__ = ['rank_histogram', 'rps', 'reliability', 'roc', 'discrimination', 'Brier_score', 
           'contingency', '_sum_contingency', 'accuracy_score', 'Heidke_score', 'Peirce_score', 
           'Gerrity_score', 'bias_score', 'hit_rate', 'false_alarm_ratio', 'false_alarm_rate', 
           'success_ratio', 'threat_score', 'equit_threat_score', 'odds_ratio', 
           'odds_ratio_skill_score', 'mean_additive_bias', 'mean_multiplicative_bias', 
           'mean_absolute_error', 'mean_squared_error', 'rms_error', 'Pearson_corrcoeff', 
           'sign_test']

# ===================================================================================================
# Packages
# ===================================================================================================
import numpy as np
import xarray as xr
import itertools
import bottleneck
from xarray.core.duck_array_ops import dask_array_type

# Load doppyo packages -----
from doppyo import utils


# ===================================================================================================
# Methods for probabilistic comparisons
# ===================================================================================================
def rank_histogram(da_cmp, da_ref, over_dims, norm=True, ensemble_dim='ensemble'):
    """ 
        Returns the rank histogram along the specified dimensions
        
        | Authors: Dougie Squire
        | Date: 01/11/2018
        
        Parameters
        ----------
        da_cmp : xarray DataArray
            Array containing data to be compared to reference dataset (usually forecasts). This data \
                    is used to rank the reference data. Must include an ensemble dimension
        da_ref : xarray DataArray
            Array containing reference data (usually observations). This data is ranked within the \
                    comparison data. Dimensions should match those of da_cmp
        over_dims : str or sequence of str
            The dimension(s) over which to compute the histogram of ranks
        norm : bool, optional
            If True, rank histograms are normalised by their enclosed area
        ensemble_dim : str, optional
            The name of the ensemble dimension in da_cmp
            
        Returns
        -------
        rank_histogram : xarray DataArray
            New DataArray object containing the rank histograms
            
        Examples
        --------
        >>> da_cmp = xr.DataArray(np.random.normal(size=(3,3,3)), 
        ...                       coords=[('x', np.arange(3)), ('y', np.arange(3)), 
        ...                               ('ensemble', np.arange(3))])
        >>> da_ref = xr.DataArray(np.random.normal(size=(3,3)), coords=[('x', np.arange(3)), 
        ...                                                             ('y', np.arange(3))])
        >>> doppyo.skill.rank_histogram(da_cmp, da_ref, over_dims='x')
        <xarray.DataArray 'rank_histogram' (bins: 4, y: 3)>
        array([[1.      , 0.333333, 0.333333],
               [0.      , 0.333333, 0.333333],
               [0.      , 0.      , 0.333333],
               [0.      , 0.333333, 0.      ]])
        Coordinates:
          * bins     (bins) float64 1.0 2.0 3.0 4.0
          * y        (y) int64 0 1 2

        Notes
        -----
        See http://www.cawcr.gov.au/projects/verification/
    """
    
    def _rank_first(x):
        """ Returns the rank of the first element along the last axes """

        ranks = bottleneck.nanrankdata(x,axis=-1)

        return ranks[...,0]
    
    if over_dims is None:
        raise ValueError('Cannot compute rank histogram with no independent dimensions')
       
    # Stack da_cmp and da_ref along ensemble dimension -----
    if ensemble_dim not in da_ref.coords:
        da_2 = da_ref.copy()
        da_2.coords[ensemble_dim] = -1
        da_2 = da_2.expand_dims(ensemble_dim)
    else:
        raise ValueError('da_ref cannot contain an ensemble dimension')

    # Only keep and combine instances that appear in both dataarrays (excluding the ensemble dim) -----
    aligned = xr.align(da_2, da_cmp, join='inner', exclude=ensemble_dim)
    combined = xr.concat(aligned, dim=ensemble_dim)

    # Rank the data -----
    if isinstance(combined.data, dask_array_type):
        combined = combined.chunk(chunks={ensemble_dim: -1})
    da_ranked = xr.apply_ufunc(_rank_first, combined,
                               input_core_dims=[[ensemble_dim]],
                               dask='parallelized',
                               output_dtypes=[int]).rename('rank')
    
    # Initialise bins -----
    bins = range(1, len(da_cmp[ensemble_dim])+2)
    bin_edges = utils.get_bin_edges(bins)
    
    if norm:
        return utils.pdf(da_ranked, bin_edges, over_dims=over_dims).rename('rank_histogram')
    else:
        return utils.histogram(da_ranked, bin_edges, over_dims=over_dims).rename('rank_histogram')


# ===================================================================================================
def rps(da_cmp, da_ref, bins, over_dims=None, ensemble_dim='ensemble'):
    """ 
        Returns the ranked probability score
        
        | Author: Dougie Squire
        | Date: 10/05/2018
        
        Parameters
        ----------
        da_cmp : xarray DataArray
            Array containing data to be compared to reference dataset (usually forecasts)
        da_ref : xarray DataArray
            Array containing reference data (usually observations)
        bins : array_like
            Bins to compute the ranked probability score over
        over_dims : str or sequence of str, optional
            Dimensions over which to average the ranked probability score
        ensemble_dim : str, optional
            Name of ensemble dimension
            
        Returns
        -------
        rps : xarray DataArray
            Array containing ranked probability score 
            
        Examples
        --------
        >>> da_cmp = xr.DataArray(np.random.normal(size=(3,3,3)), 
        ...                       coords=[('x', np.arange(3)), ('y', np.arange(3)), 
        ...                               ('ensemble', np.arange(3))])
        >>> da_ref = xr.DataArray(np.random.normal(size=(3,3)), coords=[('x', np.arange(3)), 
        ...                                                             ('y', np.arange(3))])
        >>> bins = np.linspace(-2,2,10)
        >>> doppyo.skill.rps(da_cmp, da_ref, bins=bins, over_dims='x')
        <xarray.DataArray 'rps' (y: 3)>
        array([0.36214 , 0.806584, 0.263374])
        Coordinates:
          * y        (y) int64 0 1 2
        
        Notes
        -----
        See http://www.cawcr.gov.au/projects/verification/
    """

    if over_dims is None:
        over_dims = []
    
    # Initialise bins -----
    bin_edges = utils.get_bin_edges(bins)

    # Compute cumulative density functions -----
    cdf_cmp = utils.cdf(da_cmp, bin_edges=bin_edges, over_dims=ensemble_dim)
    cdf_ref = utils.cdf(da_ref, bin_edges=bin_edges, over_dims=None)
    
    return utils.integrate((cdf_cmp - cdf_ref) ** 2, over_dim='bins') \
                .mean(dim=over_dims, skipna=True).rename('rps')


# ===================================================================================================
def reliability(cmp_likelihood, ref_logical, over_dims, probability_bins=np.linspace(0,1,5), 
                nans_as_zeros=True):
    """ 
        Computes the relative frequency of an event for a range of probability threshold bins \
                given the comparison likelihood and reference logical event data 
        
        | Author: Dougie Squire
        | Date: 10/05/2018
        
        Parameters
        ----------
        cmp_likelihood : xarray DataArray
            Array containing likelihoods of the event from the comparison data (e.g. cmp_likelihood = \
                    (da_cmp > 1).mean(dim='ensemble'))
        ref_logical : xarray DataArray
            Array containing logical (True/False) outcomes of the event from the reference data (e.g.\
                    ref_logical = (da_ref > 1))
        over_dims : str or sequence of str
            Dimensions over which to compute the reliability
        probability_bins : array_like, optional
            Probability threshold bins. Defaults to 5 equally spaced bins between 0 and 1
        nans_as_zeros : bool, optional
            Replace output nans (resulting fron bins with no data) with zeros
            
        Returns
        -------
        reliability : xarray DataSet
            | Dataset containing the following variables:
            | relative_freq; the relative frequency of occurence for each probability threshold bin
            | cmp_number; the number of instances that the comparison data fall within each probability \
                    threshold bin
            | ref_occur; the number of instances that the reference data is True when the comparison data \
                    falls within each probability threshold bin
        
        Examples
        --------
        >>> da_cmp = xr.DataArray(np.random.normal(size=(3,3,3)), 
        ...                       coords=[('x', np.arange(3)), ('y', np.arange(3)), 
        ...                               ('ensemble', np.arange(3))])
        >>> da_ref = xr.DataArray(np.random.normal(size=(3,3)), 
        ...                       coords=[('x', np.arange(3)), ('y', np.arange(3))])
        >>> cmp_likelihood = (da_cmp > 0.1).mean('ensemble')
        >>> ref_logical = da_ref > 0.1
        >>> doppyo.skill.reliability(cmp_likelihood, ref_logical, over_dims='x')
        <xarray.Dataset>
        Dimensions:          (probability_bin: 5, y: 3)
        Coordinates:
          * y                (y) int64 0 1 2
          * probability_bin  (probability_bin) float64 0.0 0.25 0.5 0.75 1.0
        Data variables:
            relative_freq    (probability_bin, y) float64 0.0 0.5 0.0 ... 1.0 0.0 0.0
            cmp_number       (probability_bin, y) int64 0 2 1 2 0 1 0 0 0 0 1 1 1 0 0
            ref_occur        (probability_bin, y) int64 0 1 0 1 0 0 0 0 0 0 0 1 1 0 0
        
        Notes
        -----
        See http://www.cawcr.gov.au/projects/verification/

        To do
        
        - Currently using a for-loop to process each probability bin separately. Is it possible \
                to remove this loop?
    """
    
    if over_dims is None:
        over_dims = []
        
    ref_binary = ref_logical.copy()*1
        
    # Check that comparison data is likelihoods and reference data is binary -----
    if ((cmp_likelihood > 1).any().item()) | ((cmp_likelihood < 0).any().item()):
        raise ValueError('Input "cmp_likelihood" must represent likelihoods and must have values between 0 and 1')
    if not ((ref_logical == 0) | (ref_logical == 1)).all().item():
        raise ValueError('Input "ref_logical" must represent logical (True/False) outcomes')
    
    # Initialise probability bins -----
    probability_bin_edges = utils.get_bin_edges(probability_bins)

    # Loop over probability bins -----
    cmp_number_list = []
    ref_occur_list = []
    for idx in range(len(probability_bin_edges)-1):
        # Logical of comparisons that fall within probability bin -----
        cmp_in_bin = (cmp_likelihood >= probability_bin_edges[idx]) & \
                     (cmp_likelihood < probability_bin_edges[idx+1])
        
        # Number of comparisons that fall within probability bin -----
        cmp_number_list.append((1 * cmp_in_bin).sum(dim=over_dims, skipna=True))  
        
        # Number of reference occurences where comparison likelihood is within probability bin -----
        ref_occur_list.append((1 * ((cmp_in_bin == True) & (ref_logical == True))) \
                      .sum(dim=over_dims, skipna=True))
    
    # Concatenate lists -----
    cmp_number = xr.concat(cmp_number_list, dim='probability_bin')
    cmp_number['probability_bin'] = probability_bins       

    ref_occur = xr.concat(ref_occur_list, dim='probability_bin')
    ref_occur['probability_bin'] = probability_bins  

    # Reference relative frequency -----
    relative_freq = ref_occur / cmp_number

    # Replace nans with zeros -----
    if nans_as_zeros:
        relative_freq = relative_freq.fillna(0)

    # Package in dataset -----
    reliability = relative_freq.to_dataset(name='relative_freq')
    reliability.relative_freq.attrs['name'] = 'relative frequency'
    reliability['cmp_number'] = cmp_number
    reliability.cmp_number.attrs['name'] = 'number of comparisons in bin'
    reliability['ref_occur'] = ref_occur
    reliability.ref_occur.attrs['name'] = 'number of reference occurences when comparisons in bin'

    return reliability


# ===================================================================================================
def roc(cmp_likelihood, ref_logical, over_dims, probability_bins=np.linspace(0,1,5)):
    """ 
        Computes the relative operating characteristic of an event for a range of probability \
                threshold bins given the comparison likelihood and reference logical event data 
        
        | Author: Dougie Squire
        | Date: 10/05/2018
        
        Parameters
        ----------
        cmp_likelihood : xarray DataArray
            Array containing likelihoods of the event from the comparison data (e.g. cmp_likelihood = \
                    (da_cmp > 1).mean(dim='ensemble'))
        ref_logical : xarray DataArray
            Array containing logical (True/False) outcomes of the event from the reference data (e.g.\
                    ref_logical = (da_ref > 1))
        over_dims : str or sequence of str
            Dimensions over which to compute the relative operating characteristic
        probability_bins : array_like, optional
            Probability threshold bins. Defaults to 5 equally spaced bins between 0 and 1
            
        Returns
        -------
        roc : xarray DataSet
            | Dataset containing the following variables:
            | hit_rate; the hit rate in each probability bin
            | false_alarm_rate; the false alarm rate in each probability bin
            | area; the area under the roc curve (false alarm rate vs hit rate)
        
        Examples
        --------
        >>> da_cmp = xr.DataArray(np.random.normal(size=(3,3,3)), 
        ...                       coords=[('x', np.arange(3)), ('y', np.arange(3)), 
        ...                               ('ensemble', np.arange(3))])
        >>> da_ref = xr.DataArray(np.random.normal(size=(3,3)), 
        ...                       coords=[('x', np.arange(3)), ('y', np.arange(3))])
        >>> cmp_likelihood = (da_cmp > 0.1).mean('ensemble')
        >>> ref_logical = da_ref > 0.1
        >>> doppyo.skill.roc(cmp_likelihood, ref_logical, over_dims='x')
        <xarray.Dataset>
        Dimensions:           (probability_bin: 5, y: 3)
        Coordinates:
          * y                 (y) int64 0 1 2
          * probability_bin   (probability_bin) float64 0.0 0.25 0.5 0.75 1.0
        Data variables:
            hit_rate          (probability_bin, y) float64 1.0 1.0 1.0 ... nan 0.0 0.0
            false_alarm_rate  (probability_bin, y) float64 1.0 1.0 1.0 ... 0.0 0.0 0.0
            area              (y) float64 0.0 0.0 0.0
        
        Notes
        -----
        See http://www.cawcr.gov.au/projects/verification/

        To do
        
        - Currently using a for-loop to process each probability bin separately. Is it possible \
                to remove this loop?
    """
    
    if over_dims is None:
        over_dims = []
    
    ref_binary = ref_logical * 1

    # Initialise probability bins -----
    dprob = np.diff(probability_bins)/2
    probability_bin_edges = probability_bins[:-1]+dprob
    if np.any(probability_bin_edges >= 1.0):
            raise ValueError('No element of probability_bins can exceed 1.0')

    # Fill first probability bin with ones -----
    all_ones = 0 * ref_binary.mean(dim=over_dims) + 1
    hit_rate_list = [all_ones]
    false_alarm_rate_list = [all_ones]
    
    # Loop over probability bins -----
    for idx,probability_bin_edge in enumerate(probability_bin_edges):
            
        # Compute contingency table for current probability -----
        category_edges = [-np.inf, probability_bin_edge, np.inf]
        contingency = compute_contingency_table(cmp_likelihood, ref_binary, 
                                                category_edges, category_edges, over_dims=over_dims)
        
        # Add hit rate and false alarm rate to lists -----
        hit_rate_list.append(compute_hit_rate(contingency,yes_category=2))
        false_alarm_rate_list.append(compute_false_alarm_rate(contingency,yes_category=2))
    
    # Concatenate lists -----
    hit_rate = xr.concat(hit_rate_list, dim='probability_bin')
    hit_rate['probability_bin'] = probability_bins
    false_alarm_rate = xr.concat(false_alarm_rate_list, dim='probability_bin')
    false_alarm_rate['probability_bin'] = probability_bins
    
    # Calculate area under curve -----
    dx = false_alarm_rate - false_alarm_rate.shift(**{'probability_bin':1})
    dx = dx.fillna(0.0)
    area = abs(((hit_rate.shift(**{'probability_bin':1}) + hit_rate) * dx / 2.0) \
                 .fillna(0.0).sum(dim='probability_bin'))
    
    # Package in dataset -----
    roc = hit_rate.to_dataset(name='hit_rate')
    roc.hit_rate.attrs['name'] = 'hit rate'
    roc['false_alarm_rate'] = false_alarm_rate
    roc.false_alarm_rate.attrs['name'] = 'false alarm rate'
    roc['area'] = area
    roc.area.attrs['name'] = 'area under roc'

    return roc


# ===================================================================================================
def discrimination(cmp_likelihood, ref_logical, over_dims, probability_bins=np.linspace(0,1,5)):
    """ 
        Returns the discrimination diagram of an event; the histogram of comparison likelihood when \
                references indicate the event has occurred and has not occurred
        
        | Author: Dougie Squire
        | Date: 10/05/2018
        
        Parameters
        ----------
        cmp_likelihood : xarray DataArray
            Array containing likelihoods of the event from the comparison data (e.g. cmp_likelihood = \
                    (da_cmp > 1).mean(dim='ensemble'))
        ref_logical : xarray DataArray
            Array containing logical (True/False) outcomes of the event from the reference data (e.g.\
                    ref_logical = (da_ref > 1))
        over_dims : str or sequence of str
            Dimensions over which to compute the discrimantion histograms
        probability_bins : array_like, optional
            Probability threshold bins. Defaults to 5 equally spaced bins between 0 and 1
            
        Returns
        -------
        discrimination : xarray DataSet
            | Dataset containing the following variables:
            | hist_event; histogram of comparison likelihoods when reference data indicates that the \
                    event has occurred
            | hist_no_event; histogram of comparison likelihoods when reference data indicates that the \
                    event has not occurred
        
        Examples
        --------
        >>> da_cmp = xr.DataArray(np.random.normal(size=(3,3,3)), 
        ...                       coords=[('x', np.arange(3)), ('y', np.arange(3)), 
        ...                               ('ensemble', np.arange(3))])
        >>> da_ref = xr.DataArray(np.random.normal(size=(3,3)), 
        ...                       coords=[('x', np.arange(3)), ('y', np.arange(3))])
        >>> cmp_likelihood = (da_cmp > 0.1).mean('ensemble')
        >>> ref_logical = da_ref > 0.1
        >>> doppyo.skill.discrimination(cmp_likelihood, ref_logical, over_dims='x')
        <xarray.Dataset>
        Dimensions:        (bins: 5, y: 3)
        Coordinates:
          * bins           (bins) float64 0.0 0.25 0.5 0.75 1.0
          * y              (y) int64 0 1 2
        Data variables:
            hist_event     (bins, y) float64 0.0 0.0 nan 0.5 1.0 ... 0.0 nan 0.0 0.0 nan
            hist_no_event  (bins, y) float64 0.0 0.0 0.0 1.0 ... 0.3333 0.0 0.0 0.3333
        
        Notes
        -----
        See http://www.cawcr.gov.au/projects/verification/

        To do
        
        - Currently using a for-loop to process each probability bin separately. Is it possible \
                to remove this loop?
    """
    
    # Initialise probability bins -----
    probability_bin_edges = utils.get_bin_edges(probability_bins)

    # Compute histogram of comparison likelihoods when reference is True/False -----
    hist_event = utils.histogram(cmp_likelihood.where(ref_logical == True), 
                                 probability_bin_edges, over_dims=over_dims) / \
                                 (ref_logical == True).sum(dim=over_dims)
    hist_no_event = utils.histogram(cmp_likelihood.where(ref_logical == False), 
                                    probability_bin_edges, over_dims=over_dims) / \
                                    (ref_logical == False).sum(dim=over_dims)
    
    # Package in dataset -----
    discrimination = hist_event.to_dataset(name='hist_event')
    discrimination.hist_event.attrs['name'] = 'histogram of comparison likelihood, yes event'
    discrimination['hist_no_event'] = hist_no_event
    discrimination.hist_no_event.attrs['name'] = 'histogram of comparison likelihood, no event'

    return discrimination


# ===================================================================================================
def Brier_score(cmp_likelihood, ref_logical, over_dims, probability_bins=None):
    """ 
        Computes the Brier score(s) of an event given the comparison likelihood and reference logical \
                event data. When comparison probability bins are also provided, this function also computes \
                the reliability, resolution and uncertainty components of the Brier score, where Brier = \
                reliability - resolution + uncertainty
        
        | Author: Dougie Squire
        | Date: 10/05/2018
        
        Parameters
        ----------
        cmp_likelihood : xarray DataArray
            Array containing likelihoods of the event from the comparison data (e.g. cmp_likelihood = \
                    (da_cmp > 1).mean(dim='ensemble'))
        ref_logical : xarray DataArray
            Array containing logical (True/False) outcomes of the event from the reference data (e.g.\
                    ref_logical = (da_ref > 1))
        over_dims : str or sequence of str
            Dimensions over which to compute the Brier score
        probability_bins : array_like, optional
            Probability threshold bins. If specified, this function also computes the reliability, \
                    resolution and uncertainty components of the Brier score. Defaults to None
            
        Returns
        -------
        Brier : xarray DataArray or xarray DataSet
            If probability_bins = None, returns a DataArray containing Brier scores. Otherwise returns \
                    a DataSet containing the reliability, resolution and uncertainty components of the Brier \
                    score, where Brier = reliability - resolution + uncertainty
        
        Examples
        --------
        >>> da_cmp = xr.DataArray(np.random.normal(size=(3,3,3)), 
        ...                       coords=[('x', np.arange(3)), ('y', np.arange(3)), 
        ...                               ('ensemble', np.arange(3))])
        >>> da_ref = xr.DataArray(np.random.normal(size=(3,3)), 
        ...                       coords=[('x', np.arange(3)), ('y', np.arange(3))])
        >>> cmp_likelihood = (da_cmp > 0.1).mean('ensemble')
        >>> ref_logical = da_ref > 0.1
        >>> doppyo.skill.Brier_score(cmp_likelihood, ref_logical, over_dims='x')
        <xarray.DataArray (y: 3)>
        array([0.148148, 0.444444, 0.222222])
        Coordinates:
          * y        (y) int64 0 1 2
        
        Notes
        -----
        See http://www.cawcr.gov.au/projects/verification/

        To do
        
        - Currently using a for-loop to process each probability bin separately. Is it possible \
                to remove this loop?
    """
    
    if over_dims is None:
        over_dims = []   
    N = (0*cmp_likelihood + 1).sum(dim=over_dims, skipna=True)

    ref_binary = ref_logical.copy()*1
    
    # Calculate total Brier score -----
    Brier = (1 / N) * ((cmp_likelihood - ref_binary) ** 2).sum(dim=over_dims, skipna=True) \
                                                          .rename('Brier_score')
        
    # Calculate components
    if probability_bins is not None:

        # Initialise probability bins -----
        probability_bin_edges = utils.get_bin_edges(probability_bins)

        # Initialise mean_cmp_likelihood array -----
        mean_cmp_likelihood = cmp_likelihood.copy(deep=True)
        
        # Loop over probability bins -----
        mean_probability_bin_list = []
        cmp_number_list = []
        ref_occur_list = []
        for idx in range(len(probability_bin_edges)-1):
            # Logical of comparisons that fall within probability bin -----
            cmp_in_bin = (cmp_likelihood >= probability_bin_edges[idx]) & \
                         (cmp_likelihood < probability_bin_edges[idx+1])
            
            # Replace likelihood with mean likelihood (so that Brier components add to total) -----
            mean_cmp_likelihood = mean_cmp_likelihood.where(~cmp_in_bin).fillna( \
                                                     mean_cmp_likelihood.where(cmp_in_bin)
                                                                        .mean(dim=over_dims, skipna=True))

            # Mean comparison probability within current probability bin -----
            mean_probability_bin_list.append(cmp_likelihood.where(cmp_in_bin,np.nan) \
                                                    .mean(dim=over_dims, skipna=True)) 

            # Number of comparisons that fall within probability bin -----
            cmp_number_list.append(cmp_in_bin.sum(dim=over_dims, skipna=True))

            # Number of reference occurences where comparison likelihood is within probability bin -----
            ref_occur_list.append(((cmp_in_bin == True) & (ref_logical == True)) \
                                    .sum(dim=over_dims)) 

        # Concatenate lists -----
        mean_probability_bin = xr.concat(mean_probability_bin_list, dim='probability_bin')
        mean_probability_bin['probability_bin'] = probability_bins
        cmp_number = xr.concat(cmp_number_list, dim='probability_bin')
        cmp_number['probability_bin'] = probability_bins
        ref_occur = xr.concat(ref_occur_list, dim='probability_bin')
        ref_occur['probability_bin'] = probability_bins

        # Compute Brier components -----
        base_rate = ref_occur / cmp_number
        Brier_reliability = (1 / N) * (cmp_number*(mean_probability_bin - base_rate) ** 2) \
                                       .sum(dim='probability_bin', skipna=True)
            
        sample_clim = ref_binary.mean(dim=over_dims, skipna=True)
        Brier_resolution = (1 / N) * (cmp_number*(base_rate - sample_clim) ** 2) \
                                      .sum(dim='probability_bin', skipna=True)
        Brier_uncertainty = sample_clim * (1 - sample_clim)
        
        # When a binned approach is used, compute total Brier using binned probabilities -----
        # (This way Brier_total = Brier_reliability - Brier_resolution + Brier_uncertainty)
        Brier_total = (1 / N) * ((mean_cmp_likelihood - ref_binary) ** 2) \
                      .sum(dim=over_dims, skipna=True)
        
        # Package in dataset -----
        Brier = Brier_total.to_dataset(name='Brier_total')
        Brier.Brier_total.attrs['name'] = 'total Brier score'
        Brier['Brier_reliability'] = Brier_reliability
        Brier.Brier_reliability.attrs['name'] = 'reliability component of Brier score'
        Brier['Brier_resolution'] = Brier_resolution
        Brier.Brier_resolution.attrs['name'] = 'resolution component of Brier score'
        Brier['Brier_uncertainty'] = Brier_uncertainty
        Brier.Brier_uncertainty.attrs['name'] = 'uncertainty component of Brier score'
        
    return Brier


# ===================================================================================================
# Methods for categorized comparisons
# ===================================================================================================
def contingency(da_cmp, da_ref, category_edges_cmp, category_edges_ref, over_dims):
    """ 
        Return the contingency table between da_cmp and da_ref for given categories
        
        | Author: Dougie Squire
        | Date: 12/05/2018
        
        Parameters
        ----------
        da_cmp : xarray DataArray
            Array containing data to be compared to reference dataset (usually forecasts)
        da_ref : xarray DataArray
            Array containing reference data (usually observations)
        category_edges_cmp : array_like
            Bin edges for categorising da_cmp
        category_edges_ref : array_like
            Bin edges for categorising da_ref
        over_dims : str or sequence of str, optional
            Dimensions over which to compute the contingency table
            
        Returns
        -------
        contingency : xarray DataArray
            Contingency table of input data
            
        Examples
        --------
        >>> da_cmp = xr.DataArray(np.random.normal(size=(3,3)), 
        ...                       coords=[('x', np.arange(3)), ('y', np.arange(3))])
        >>> da_ref = xr.DataArray(np.random.normal(size=(3,3)), 
        ...                       coords=[('x', np.arange(3)), ('y', np.arange(3))])
        >>> category_edges_cmp = np.linspace(-2,2,5)
        >>> category_edges_ref = np.linspace(-2,2,5)
        doppyo.skill.contingency(da_cmp, da_ref, category_edges_cmp, 
        ...                      category_edges_ref, over_dims=['x','y'])
        <xarray.DataArray 'contingency' (comparison_category: 4, reference_category: 4)>
        array([[0, 1, 0, 1],
               [1, 0, 1, 0],
               [0, 2, 1, 0],
               [0, 0, 0, 0]])
        Coordinates:
          * comparison_category  (comparison_category) int64 1 2 3 4
          * reference_category   (reference_category) int64 1 2 3 4
        
        Notes
        -----
        See http://www.cawcr.gov.au/projects/verification/
    """
    
    if over_dims is None:
        over_dims = []
    
    cmp_edges = [(category_edges_cmp[i], category_edges_cmp[i+1],i+1) 
                 for i in range(len(category_edges_cmp)-1)]
    ref_edges = [(category_edges_ref[i], category_edges_ref[i+1],i+1) 
                 for i in range(len(category_edges_ref)-1)]
    da_list = []
    for category in itertools.product(cmp_edges, ref_edges):
        da_temp = (((da_cmp >= category[0][0]) & (da_cmp < category[0][1])) & \
                   ((da_ref >= category[1][0]) & (da_ref < category[1][1]))).sum(dim=over_dims)
        da_temp.coords['comparison_category'] = category[0][2]
        da_temp.coords['reference_category'] = category[1][2]
        da_list.append(da_temp)
    
    if len(da_list) == 1:
        return da_list[0].rename('contingency')
    else:
        return xr.concat(da_list, dim='stack') \
                 .set_index(stack=['comparison_category', 'reference_category']) \
                 .unstack('stack').rename('contingency') 

    
# ===================================================================================================
def _sum_contingency(contingency, category='total'):
    """ 
        Returns sums of specified categories in contingency table 
        
        | Author: Dougie Squire
        | Date: 12/05/2018
        
        Parameters
        ----------
        contingency : xarray DataArray
            A contingency table of the form output from doppyo.skill.contingency
        category : str, optional
            Contingency table category to sum. Options are 'total', 'reference' and 'comparison'
            
        Returns
        -------
        summed : 
            Sum of all counts in specified category
            
        Examples
        --------
        >>> da_cmp = xr.DataArray(np.random.normal(size=(3,3)), 
        ...                       coords=[('x', np.arange(3)), ('y', np.arange(3))])
        >>> da_ref = xr.DataArray(np.random.normal(size=(3,3)), 
        ...                       coords=[('x', np.arange(3)), ('y', np.arange(3))])
        >>> category_edges_cmp = np.linspace(-2,2,3)
        >>> category_edges_ref = np.linspace(-2,2,3)
        >>> contingency = doppyo.skill.contingency(da_cmp, da_ref, category_edges_cmp, 
        ...                                        category_edges_ref, over_dims='y')
        >>> doppyo.skill._sum_contingency(contingency, category='reference')
        <xarray.DataArray 'contingency' (x: 3, category: 2)>
        array([[0, 3],
               [2, 1],
               [2, 1]])
        Coordinates:
          * x         (x) int64 0 1 2
          * category  (category) int64 1 2
    """
    
    if category == 'total':
        N = contingency.sum(dim=('reference_category','comparison_category'), skipna=True)
    elif category == 'reference':
        N = contingency.sum(dim='comparison_category', skipna=True) \
                       .rename({'reference_category' : 'category'})
    elif category == 'comparison':
        N = contingency.sum(dim='reference_category', skipna=True) \
                       .rename({'comparison_category' : 'category'})    
    else: raise ValueError(f'"{category}" is not a recognised category')
        
    return N


# ===================================================================================================
def accuracy_score(contingency):
    """ 
        Returns the accuracy score given a contingency table
        
        | Author: Dougie Squire
        | Date: 12/05/2018
        
        Parameters
        ----------
        contingency : xarray DataArray
            A contingency table of the form output from doppyo.skill.contingency
            
        Returns
        -------
        accuracy_score : xarray DataArray
            An array containing the accuracy scores
            
        Examples
        --------
        >>> da_cmp = xr.DataArray(np.random.normal(size=(3,3)), 
        ...                       coords=[('x', np.arange(3)), ('y', np.arange(3))])
        >>> da_ref = xr.DataArray(np.random.normal(size=(3,3)), 
        ...                       coords=[('x', np.arange(3)), ('y', np.arange(3))])
        >>> category_edges_cmp = np.linspace(-2,2,5)
        >>> category_edges_ref = np.linspace(-2,2,5)
        >>> contingency = doppyo.skill.contingency(da_cmp, da_ref, category_edges_cmp, 
        ...                                        category_edges_ref, over_dims='y')
        >>> doppyo.skill.accuracy_score(contingency)
        <xarray.DataArray 'accuracy_score' (x: 3)>
        array([0.      , 0.333333, 0.333333])
        Coordinates:
          * x        (x) int64 0 1 2

        Notes
        -----
        See http://www.cawcr.gov.au/projects/verification/
    """
    
    hits = contingency.where(contingency.reference_category == contingency.comparison_category) \
           .sum(dim=('reference_category','comparison_category'), skipna=True)
    N = _sum_contingency(contingency, 'total')
    
    return (hits / N).rename('accuracy_score')


# ===================================================================================================
def Heidke_score(contingency):
    """ 
        Returns the Heidke skill score given a contingency table 
        
        | Author: Dougie Squire
        | Date: 12/05/2018
        
        Parameters
        ----------
        contingency : xarray DataArray
            A contingency table of the form output from doppyo.skill.contingency
            
        Returns
        -------
        Heidke_score : xarray DataArray
            An array containing the Heidke scores
            
        Examples
        --------
        >>> da_cmp = xr.DataArray(np.random.normal(size=(3,3)), 
        ...                       coords=[('x', np.arange(3)), ('y', np.arange(3))])
        >>> da_ref = xr.DataArray(np.random.normal(size=(3,3)), 
        ...                       coords=[('x', np.arange(3)), ('y', np.arange(3))])
        >>> category_edges_cmp = np.linspace(-2,2,5)
        >>> category_edges_ref = np.linspace(-2,2,5)
        >>> contingency = doppyo.skill.contingency(da_cmp, da_ref, category_edges_cmp, 
        ...                                        category_edges_ref, over_dims='y')
        >>> doppyo.skill.Heidke_score(contingency)
        <xarray.DataArray 'Heidke_score' (x: 3)>
        array([-0.285714,  0.      ,  0.142857])
        Coordinates:
          * x        (x) int64 0 1 22

        Notes
        -----
        See http://www.cawcr.gov.au/projects/verification/
    """
    
    numer_1 = contingency.where(contingency.reference_category == contingency.comparison_category) \
              .sum(dim=('reference_category','comparison_category'), skipna=True) / \
              _sum_contingency(contingency, 'total')
    numer_2 = (_sum_contingency(contingency, 'reference') * \
               _sum_contingency(contingency, 'comparison')).sum(dim='category', skipna=True) / \
              (_sum_contingency(contingency, 'total')**2)
    denom = 1 - numer_2

    return ((numer_1 - numer_2) / denom).rename('Heidke_score')


# ===================================================================================================
def Peirce_score(contingency):
    """ 
        Returns the Peirce score (also called Hanssen and Kuipers discriminant) given a contingency \
                table 
        
        | Author: Dougie Squire
        | Date: 12/05/2018
        
        Parameters
        ----------
        contingency : xarray DataArray
            A contingency table of the form output from doppyo.skill.contingency
            
        Returns
        -------
        Peirce_score : xarray DataArray
            An array containing the Peirce scores
            
        Examples
        --------
        >>> da_cmp = xr.DataArray(np.random.normal(size=(3,3)), 
        ...                       coords=[('x', np.arange(3)), ('y', np.arange(3))])
        >>> da_ref = xr.DataArray(np.random.normal(size=(3,3)), 
        ...                       coords=[('x', np.arange(3)), ('y', np.arange(3))])
        >>> category_edges_cmp = np.linspace(-2,2,5)
        >>> category_edges_ref = np.linspace(-2,2,5)
        >>> contingency = doppyo.skill.contingency(da_cmp, da_ref, category_edges_cmp, 
        ...                                        category_edges_ref, over_dims='y')
        >>> doppyo.skill.Peirce_score(contingency)
        <xarray.DataArray 'Peirce_score' (x: 3)>
        array([-0.25,  0.  , -0.5 ])
        Coordinates:
          * x        (x) int64 0 1 2

        Notes
        -----
        See http://www.cawcr.gov.au/projects/verification/
    """
    
    numer_1 = contingency.where(contingency.reference_category == contingency.comparison_category) \
              .sum(dim=('reference_category','comparison_category'), skipna=True) / \
              _sum_contingency(contingency, 'total')

    numer_2 = (_sum_contingency(contingency, 'reference') * \
               _sum_contingency(contingency, 'comparison')).sum(dim='category', skipna=True) / \
              (_sum_contingency(contingency, 'total')**2)

    denom = 1 - (_sum_contingency(contingency, 'reference')**2).sum(dim='category', skipna=True) / \
                (_sum_contingency(contingency, 'total')**2)

    return ((numer_1 - numer_2) / denom).rename('Peirce_score')


# ===================================================================================================
def Gerrity_score(contingency):
    """ 
        Returns Gerrity equitable score given a contingency table 
        
        | Author: Dougie Squire
        | Date: 12/05/2018
        
        Parameters
        ----------
        contingency : xarray DataArray
            A contingency table of the form output from doppyo.skill.contingency
            
        Returns
        -------
        Gerrity_score : xarray DataArray
            An array containing the Gerrity scores
            
        Examples
        --------
        >>> da_cmp = xr.DataArray(np.random.normal(size=(3,3)), 
        ...                       coords=[('x', np.arange(3)), ('y', np.arange(3))])
        >>> da_ref = xr.DataArray(np.random.normal(size=(3,3)), 
        ...                       coords=[('x', np.arange(3)), ('y', np.arange(3))])
        >>> category_edges_cmp = np.linspace(-2,2,5)
        >>> category_edges_ref = np.linspace(-2,2,5)
        >>> contingency = doppyo.skill.contingency(da_cmp, da_ref, category_edges_cmp, 
        ...                                        category_edges_ref, over_dims='y')
        >>> doppyo.skill.Gerrity_score(contingency)
        <xarray.DataArray 'Gerrity_score' (x: 3)>
        array([-2.777778e-01,  0.000000e+00, -5.551115e-17])
        Coordinates:
          * x        (x) int64 0 1 2

        Notes
        -----
        See http://www.cawcr.gov.au/projects/verification/

        To do
        
        - Currently computes the Gerrity scoring matrix using nested for-loops. Is it possible \
                to remove these?
    """
    
    def _Gerrity_S(a):
        """ Returns Gerrity scoring matrix, S """

        categories = a.category.values
        K = len(categories)

        # Loop over reference categories
        ref_list = []
        for ref_category in categories:

            # Loop over comparison categories
            cmp_list = []
            for cmp_category in categories:

                i = ref_category
                j = cmp_category

                if i == j:
                    cmp_list.append((1 / (K - 1)) * ((1 / a.sel(category=range(1,i))).sum(dim='category', skipna=True) + \
                                                    a.sel(category=range(j,K)).sum(dim='category', skipna=True)))
                elif i > j:
                    cmp_list.append((1 / (K - 1)) * ((1 / a.sel(category=range(1,j))).sum(dim='category', skipna=True) - \
                                                    (i - j) + a.sel(category=range(i,K)).sum(dim='category', skipna=True)))
                else:
                    cmp_list.append((1 / (K - 1)) * ((1 / a.sel(category=range(1,i))).sum(dim='category', skipna=True) - \
                                                    (j - i) + a.sel(category=range(j,K)).sum(dim='category', skipna=True)))

            # Concatenate comparison categories -----
            cmp = xr.concat(cmp_list, dim='comparison_category')
            cmp['comparison_category'] = categories

            # Add to reference list -----
            ref_list.append(cmp)

        # Concatenate reference categories -----
        S = xr.concat(ref_list, dim='reference_category')
        S['reference_category'] = categories

        return S
    
    # Compute 'a' -----
    sum_p = (_sum_contingency(contingency, 'reference') / \
             _sum_contingency(contingency, 'total'))
    a = ((1 - sum_p.cumsum('category', skipna=True)) / sum_p.cumsum('category', skipna=True))
    
    # Compute 'S' -----
    S = _Gerrity_S(a)
    
    return ((contingency * S).sum(dim=('reference_category','comparison_category'), skipna=True) / \
            _sum_contingency(contingency, 'total')).rename('Gerrity_score')


# ===================================================================================================
# Methods for dichotomously categorized comparisons
# ===================================================================================================
def bias_score(contingency, yes_category=2):
    """ 
        Returns the bias score given dichotomous contingency data 
        
        | Author: Dougie Squire
        | Date: 12/05/2018
        
        Parameters
        ----------
        contingency : xarray DataArray
            A 2 category contingency table of the form output from doppyo.skill.contingency
        yes_category : value, optional
            The coordinate value of the category corresponding to 'yes'
            
        Returns
        -------
        bias_score : xarray DataArray
            An array containing the bias scores
            
        Examples
        --------
        >>> da_cmp = xr.DataArray(np.random.normal(size=(3,3)), 
        ...                       coords=[('x', np.arange(3)), ('y', np.arange(3))])
        >>> da_ref = xr.DataArray(np.random.normal(size=(3,3)), 
        ...                       coords=[('x', np.arange(3)), ('y', np.arange(3))])
        >>> category_edges_cmp = np.linspace(-2,2,3)
        >>> category_edges_ref = np.linspace(-2,2,3)
        >>> contingency = doppyo.skill.contingency(da_cmp, da_ref, category_edges_cmp, 
        ...                                        category_edges_ref, over_dims='y')
        >>> doppyo.skill.bias_score(contingency)
        <xarray.DataArray 'bias_score' (x: 3)>
        array([0.5     , 0.333333, 1.      ])
        Coordinates:
          * x        (x) int64 0 1 2
        
        Notes
        -----
        See http://www.cawcr.gov.au/projects/verification/
    """
    
    no_category = abs(yes_category - 2) + 1
    
    if len(contingency.comparison_category) > 2:
        raise ValueError('Bias score is defined for dichotomous contingency data only')
    
    hits = contingency.sel(comparison_category=yes_category, 
                           reference_category=yes_category, drop=True)
    false_alarms = contingency.sel(comparison_category=yes_category, 
                                   reference_category=no_category, drop=True)
    misses = contingency.sel(comparison_category=no_category, 
                             reference_category=yes_category, drop=True)

    return ((hits + false_alarms) / (hits + misses)).rename('bias_score')


# ===================================================================================================
def hit_rate(contingency, yes_category=2):
    """ 
        Returns the hit rate (probability of detection) given dichotomous contingency data 
        
        | Author: Dougie Squire
        | Date: 12/05/2018
        
        Parameters
        ----------
        contingency : xarray DataArray
            A 2 category contingency table of the form output from doppyo.skill.contingency
        yes_category : value, optional
            The coordinate value of the category corresponding to 'yes'
            
        Returns
        -------
        hit_rate : xarray DataArray
            An array containing the hit rates
            
        Examples
        --------
        >>> da_cmp = xr.DataArray(np.random.normal(size=(3,3)), 
        ...                       coords=[('x', np.arange(3)), ('y', np.arange(3))])
        >>> da_ref = xr.DataArray(np.random.normal(size=(3,3)), 
        ...                       coords=[('x', np.arange(3)), ('y', np.arange(3))])
        >>> category_edges_cmp = np.linspace(-2,2,3)
        >>> category_edges_ref = np.linspace(-2,2,3)
        >>> contingency = doppyo.skill.contingency(da_cmp, da_ref, category_edges_cmp, 
        ...                                        category_edges_ref, over_dims='y')
        >>> doppyo.skill.hit_rate(contingency)
        <xarray.DataArray 'hit_rate' (x: 3)>
        array([ 0., nan,  1.])
        Coordinates:
          * x        (x) int64 0 1 2
          
        Notes
        -----
        See http://www.cawcr.gov.au/projects/verification/
    """
    
    no_category = abs(yes_category - 2) + 1
    
    if len(contingency.comparison_category) > 2:
        raise ValueError('Hit rate is defined for dichotomous contingency data only')
    
    hits = contingency.sel(comparison_category=yes_category, 
                           reference_category=yes_category, drop=True)
    misses = contingency.sel(comparison_category=no_category, 
                             reference_category=yes_category, drop=True)
    
    return (hits / (hits + misses)).rename('hit_rate')


# ===================================================================================================
def false_alarm_ratio(contingency, yes_category=2):
    """ 
        Returns the false alarm ratio given dichotomous contingency data 
        
        | Author: Dougie Squire
        | Date: 12/05/2018
        
        Parameters
        ----------
        contingency : xarray DataArray
            A 2 category contingency table of the form output from doppyo.skill.contingency
        yes_category : value, optional
            The coordinate value of the category corresponding to 'yes'
            
        Returns
        -------
        false_alarm_ratio : xarray DataArray
            An array containing the false alarm ratios
            
        Examples
        --------
        >>> da_cmp = xr.DataArray(np.random.normal(size=(3,3)), 
        ...                       coords=[('x', np.arange(3)), ('y', np.arange(3))])
        >>> da_ref = xr.DataArray(np.random.normal(size=(3,3)), 
        ...                       coords=[('x', np.arange(3)), ('y', np.arange(3))])
        >>> category_edges_cmp = np.linspace(-2,2,3)
        >>> category_edges_ref = np.linspace(-2,2,3)
        >>> contingency = doppyo.skill.contingency(da_cmp, da_ref, category_edges_cmp, 
        ...                                        category_edges_ref, over_dims='y')
        >>> doppyo.skill.false_alarm_ratio(contingency)
        <xarray.DataArray 'false_alarm_ratio' (x: 3)>
        array([nan, nan,  0.])
        Coordinates:
          * x        (x) int64 0 1 2
          
        Notes
        -----
        See http://www.cawcr.gov.au/projects/verification/
    """
    
    no_category = abs(yes_category - 2) + 1
    
    if len(contingency.comparison_category) > 2:
        raise ValueError('False alarm ratio is defined for dichotomous contingency data only')
    
    hits = contingency.sel(comparison_category=yes_category, 
                           reference_category=yes_category, drop=True)
    false_alarms = contingency.sel(comparison_category=yes_category, 
                                   reference_category=no_category, drop=True)

    return (false_alarms / (hits + false_alarms)).rename('false_alarm_ratio')


# ===================================================================================================
def false_alarm_rate(contingency, yes_category=2):
    """ 
        Returns the false alarm rate given dichotomous contingency data 
        
        | Author: Dougie Squire
        | Date: 12/05/2018
        
        Parameters
        ----------
        contingency : xarray DataArray
            A 2 category contingency table of the form output from doppyo.skill.contingency
        yes_category : value, optional
            The coordinate value of the category corresponding to 'yes'
            
        Returns
        -------
        false_alarm_rate : xarray DataArray
            An array containing the false alarm rates
            
        Examples
        --------
        >>> da_cmp = xr.DataArray(np.random.normal(size=(3,3)), 
        ...                       coords=[('x', np.arange(3)), ('y', np.arange(3))])
        >>> da_ref = xr.DataArray(np.random.normal(size=(3,3)), 
        ...                       coords=[('x', np.arange(3)), ('y', np.arange(3))])
        >>> category_edges_cmp = np.linspace(-2,2,3)
        >>> category_edges_ref = np.linspace(-2,2,3)
        >>> contingency = doppyo.skill.contingency(da_cmp, da_ref, category_edges_cmp, 
        ...                                        category_edges_ref, over_dims='y')
        >>> doppyo.skill.false_alarm_rate(contingency)
        <xarray.DataArray 'false_alarm_rate' (x: 3)>
        array([ 0.,  0., nan])
        Coordinates:
          * x        (x) int64 0 1 2
          
        Notes
        -----
        See http://www.cawcr.gov.au/projects/verification/
    """
    
    no_category = abs(yes_category - 2) + 1
    
    if len(contingency.comparison_category) > 2:
        raise ValueError('False alarm rate is defined for dichotomous contingency data only')
        
    false_alarms = contingency.sel(comparison_category=yes_category, 
                                   reference_category=no_category, drop=True)
    correct_negs = contingency.sel(comparison_category=no_category, 
                                   reference_category=no_category, drop=True)

    return (false_alarms / (correct_negs + false_alarms)).rename('false_alarm_rate')


# ===================================================================================================
def success_ratio(contingency, yes_category=2):
    """ 
        Returns the success ratio given dichotomous contingency data 
        
        | Author: Dougie Squire
        | Date: 12/05/2018
        
        Parameters
        ----------
        contingency : xarray DataArray
            A 2 category contingency table of the form output from doppyo.skill.contingency
        success_ratio : value, optional
            The coordinate value of the category corresponding to 'yes'
            
        Returns
        -------
        success_ratio : xarray DataArray
            An array containing the success ratios
            
        Examples
        --------
        >>> da_cmp = xr.DataArray(np.random.normal(size=(3,3)), 
        ...                       coords=[('x', np.arange(3)), ('y', np.arange(3))])
        >>> da_ref = xr.DataArray(np.random.normal(size=(3,3)), 
        ...                       coords=[('x', np.arange(3)), ('y', np.arange(3))])
        >>> category_edges_cmp = np.linspace(-2,2,3)
        >>> category_edges_ref = np.linspace(-2,2,3)
        >>> contingency = doppyo.skill.contingency(da_cmp, da_ref, category_edges_cmp, 
        ...                                        category_edges_ref, over_dims='y')
        >>> doppyo.skill.success_ratio(contingency)
        <xarray.DataArray 'success_ratio' (x: 3)>
        array([nan, nan,  1.])
        Coordinates:
          * x        (x) int64 0 1 2
          
        Notes
        -----
        See http://www.cawcr.gov.au/projects/verification/
    """
    
    no_category = abs(yes_category - 2) + 1
    
    if len(contingency.comparison_category) > 2:
        raise ValueError('Success ratio is defined for dichotomous contingency data only')
        
    hits = contingency.sel(comparison_category=yes_category, 
                           reference_category=yes_category, drop=True)
    false_alarms = contingency.sel(comparison_category=yes_category, 
                                   reference_category=no_category, drop=True)
    
    return (hits / (hits + false_alarms)).rename('success_ratio')


# ===================================================================================================
def threat_score(contingency, yes_category=2):
    """ 
        Returns the threat score given dichotomous contingency data 
        
        | Author: Dougie Squire
        | Date: 12/05/2018
        
        Parameters
        ----------
        contingency : xarray DataArray
            A 2 category contingency table of the form output from doppyo.skill.contingency
        success_ratio : value, optional
            The coordinate value of the category corresponding to 'yes'
            
        Returns
        -------
        threat_score : xarray DataArray
            An array containing the threat scores
            
        Examples
        --------
        >>> da_cmp = xr.DataArray(np.random.normal(size=(3,3)), 
        ...                       coords=[('x', np.arange(3)), ('y', np.arange(3))])
        >>> da_ref = xr.DataArray(np.random.normal(size=(3,3)), 
        ...                       coords=[('x', np.arange(3)), ('y', np.arange(3))])
        >>> category_edges_cmp = np.linspace(-2,2,3)
        >>> category_edges_ref = np.linspace(-2,2,3)
        >>> contingency = doppyo.skill.contingency(da_cmp, da_ref, category_edges_cmp, 
        ...                                        category_edges_ref, over_dims='y')
        >>> doppyo.skill.threat_score(contingency)
        <xarray.DataArray 'threat_score' (x: 3)>
        array([0. , 0. , 0.5])
        Coordinates:
          * x        (x) int64 0 1 2
          
        Notes
        -----
        See http://www.cawcr.gov.au/projects/verification/
    """
    
    no_category = abs(yes_category - 2) + 1
    
    if len(contingency.comparison_category) > 2:
        raise ValueError('Threat score is defined for dichotomous contingency data only')
        
    hits = contingency.sel(comparison_category=yes_category, 
                           reference_category=yes_category, drop=True)
    misses = contingency.sel(comparison_category=no_category, 
                             reference_category=yes_category, drop=True)
    false_alarms = contingency.sel(comparison_category=yes_category, 
                                   reference_category=no_category, drop=True)
    
    return (hits / (hits + misses + false_alarms)).rename('threat_score')


# ===================================================================================================
def equit_threat_score(contingency, yes_category=2):
    """ 
        Returns the equitable threat score given dichotomous contingency data 
        
        | Author: Dougie Squire
        | Date: 12/05/2018
        
        Parameters
        ----------
        contingency : xarray DataArray
            A 2 category contingency table of the form output from doppyo.skill.contingency
        success_ratio : value, optional
            The coordinate value of the category corresponding to 'yes'
            
        Returns
        -------
        equit_threat_score : xarray DataArray
            An array containing the equitable threat scores
            
        Examples
        --------
        >>> da_cmp = xr.DataArray(np.random.normal(size=(3,3)), 
        ...                       coords=[('x', np.arange(3)), ('y', np.arange(3))])
        >>> da_ref = xr.DataArray(np.random.normal(size=(3,3)), 
        ...                       coords=[('x', np.arange(3)), ('y', np.arange(3))])
        >>> category_edges_cmp = np.linspace(-2,2,3)
        >>> category_edges_ref = np.linspace(-2,2,3)
        >>> contingency = doppyo.skill.contingency(da_cmp, da_ref, category_edges_cmp, 
        ...                                        category_edges_ref, over_dims='y')
        >>> doppyo.skill.equit_threat_score(contingency)
        <xarray.DataArray 'equit_threat_score' (x: 3)>
        array([0., 0., 0.])
        Coordinates:
          * x        (x) int64 0 1 2
          
        Notes
        -----
        See http://www.cawcr.gov.au/projects/verification/
    """
    
    no_category = abs(yes_category - 2) + 1
    
    if len(contingency.comparison_category) > 2:
        raise ValueError('Equitable threat score is defined for dichotomous contingency data only')
        
    hits = contingency.sel(comparison_category=yes_category, 
                           reference_category=yes_category, drop=True)
    misses = contingency.sel(comparison_category=no_category, 
                             reference_category=yes_category, drop=True)
    false_alarms = contingency.sel(comparison_category=yes_category, 
                                   reference_category=no_category, drop=True)
    hits_random = ((hits + misses) * (hits + false_alarms)) / _sum_contingency(contingency, 'total')
    
    return ((hits - hits_random) / (hits + misses + false_alarms + hits_random)).rename('equit_threat_score')


# ===================================================================================================
def odds_ratio(contingency, yes_category=2):
    """ 
        Returns the odds ratio given dichotomous contingency data 
        
        | Author: Dougie Squire
        | Date: 12/05/2018
        
        Parameters
        ----------
        contingency : xarray DataArray
            A 2 category contingency table of the form output from doppyo.skill.contingency
        success_ratio : value, optional
            The coordinate value of the category corresponding to 'yes'
            
        Returns
        -------
        odds_ratio : xarray DataArray
            An array containing the equitable odds ratios
            
        Examples
        --------
        >>> da_cmp = xr.DataArray(np.random.normal(size=(3,3)), 
        ...                       coords=[('x', np.arange(3)), ('y', np.arange(3))])
        >>> da_ref = xr.DataArray(np.random.normal(size=(3,3)), 
        ...                       coords=[('x', np.arange(3)), ('y', np.arange(3))])
        >>> category_edges_cmp = np.linspace(-2,2,3)
        >>> category_edges_ref = np.linspace(-2,2,3)
        >>> contingency = doppyo.skill.contingency(da_cmp, da_ref, category_edges_cmp, 
        ...                                        category_edges_ref, over_dims='y')
        >>> doppyo.skill.odds_ratio(contingency)
        <xarray.DataArray 'odds_ratio' (x: 3)>
        array([ 0.,  0., nan])
        Coordinates:
          * x        (x) int64 0 1 2
          
        Notes
        -----
        See http://www.cawcr.gov.au/projects/verification/
    """

    no_category = abs(yes_category - 2) + 1
    
    if len(contingency.comparison_category) > 2:
        raise ValueError('Odds ratio is defined for dichotomous contingency data only')
        
    hits = contingency.sel(comparison_category=yes_category, 
                           reference_category=yes_category, drop=True)
    correct_negs = contingency.sel(comparison_category=no_category, 
                                   reference_category=no_category, drop=True)
    misses = contingency.sel(comparison_category=no_category, 
                             reference_category=yes_category, drop=True)
    false_alarms = contingency.sel(comparison_category=yes_category, 
                                   reference_category=no_category, drop=True)
    
    return ((hits * correct_negs) / (misses * false_alarms)).rename('odds_ratio')


# ===================================================================================================
def odds_ratio_skill_score(contingency, yes_category=2):
    """ 
        Returns the odds ratio skill score given dichotomous contingency data 
        
        | Author: Dougie Squire
        | Date: 12/05/2018
        
        Parameters
        ----------
        contingency : xarray DataArray
            A 2 category contingency table of the form output from doppyo.skill.contingency
        success_ratio : value, optional
            The coordinate value of the category corresponding to 'yes'
            
        Returns
        -------
        odds_ratio_skill_score : xarray DataArray
            An array containing the equitable odds ratio skill scores
            
        Examples
        --------
        >>> da_cmp = xr.DataArray(np.random.normal(size=(3,3)), 
        ...                       coords=[('x', np.arange(3)), ('y', np.arange(3))])
        >>> da_ref = xr.DataArray(np.random.normal(size=(3,3)), 
        ...                       coords=[('x', np.arange(3)), ('y', np.arange(3))])
        >>> category_edges_cmp = np.linspace(-2,2,3)
        >>> category_edges_ref = np.linspace(-2,2,3)
        >>> contingency = doppyo.skill.contingency(da_cmp, da_ref, category_edges_cmp, 
        ...                                        category_edges_ref, over_dims='y')
        >>> doppyo.skill.odds_ratio_skill_score(contingency)
        <xarray.DataArray 'odds_ratio_skill' (x: 3)>
        array([-1., -1., nan])
        Coordinates:
          * x        (x) int64 0 1 2

        Notes
        -----
        See http://www.cawcr.gov.au/projects/verification/
    """
    
    no_category = abs(yes_category - 2) + 1
    
    if len(contingency.comparison_category) > 2:
        raise ValueError('Odd ratio skill score is defined for dichotomous contingency data only')
        
    hits = contingency.sel(comparison_category=yes_category, 
                           reference_category=yes_category, drop=True)
    correct_negs = contingency.sel(comparison_category=no_category, 
                                   reference_category=no_category, drop=True)
    misses = contingency.sel(comparison_category=no_category, 
                             reference_category=yes_category, drop=True)
    false_alarms = contingency.sel(comparison_category=yes_category, 
                                   reference_category=no_category, drop=True)
    
    return ((hits * correct_negs - misses * false_alarms) / \
            (hits * correct_negs + misses * false_alarms)).rename('odds_ratio_skill')


# ===================================================================================================
# Methods for continuous variables
# ===================================================================================================
def mean_additive_bias(da_cmp, da_ref, over_dims):
    """ 
        Returns the additive bias between comparison and reference datasets
        
        | Author: Dougie Squire
        | Date: 28/04/2018
        
        Parameters
        ----------
        da_cmp : xarray DataArray
            Array containing data to be compared to reference dataset (usually forecasts)
        da_ref : xarray DataArray
            Array containing reference data (usually observations)
        over_dims : str or sequence of str, optional
            Dimensions over which to compute the mean additive bias
            
        Returns
        -------
        mean_additive_bias : xarray DataArray
            Array containing the mean additive biases
            
        Examples
        --------
        >>> da_cmp = xr.DataArray(np.random.normal(size=(3,3)), 
        ...                       coords=[('x', np.arange(3)), ('y', np.arange(3))])
        >>> da_ref = xr.DataArray(np.random.normal(size=(3,3)), 
        ...                       coords=[('x', np.arange(3)), ('y', np.arange(3))])
        >>> doppyo.skill.mean_additive_bias(da_cmp, da_ref, over_dims='x')
        <xarray.DataArray 'mean_additive_bias' (y: 3)>
        array([0.328462, 0.172263, 0.402438])
        Coordinates:
          * y        (y) int64 0 1 2
        
        Notes
        -----
        See http://www.cawcr.gov.au/projects/verification/
    """
    
    if isinstance(over_dims, str):
        over_dims = [over_dims]
        
    if over_dims == None:
        over_dims = []

    return (da_cmp - da_ref).mean(dim=over_dims, skipna=True) \
                            .rename('mean_additive_bias')


# ===================================================================================================
def mean_multiplicative_bias(da_cmp, da_ref, over_dims):
    """ 
        Returns the multiplicative bias between comparison and reference datasets 
        
        | Author: Dougie Squire
        | Date: 28/04/2018
        
        Parameters
        ----------
        da_cmp : xarray DataArray
            Array containing data to be compared to reference dataset (usually forecasts)
        da_ref : xarray DataArray
            Array containing reference data (usually observations)
        over_dims : str or sequence of str, optional
            Dimensions over which to compute the mean multiplicative bias
            
        Returns
        -------
        mean_multiplicative_bias : xarray DataArray
            Array containing the mean multiplicative biases
            
        Examples
        --------
        >>> da_cmp = xr.DataArray(np.random.normal(size=(3,3)), 
        ...                       coords=[('x', np.arange(3)), ('y', np.arange(3))])
        >>> da_ref = xr.DataArray(np.random.normal(size=(3,3)), 
        ...                       coords=[('x', np.arange(3)), ('y', np.arange(3))])
        >>> doppyo.skill.mean_multiplicative_bias(da_cmp, da_ref, over_dims='x')
        <xarray.DataArray 'mean_multiplicative_bias' (y: 3)>
        array([ 2.108882,  4.356835, -0.83234 ])
        Coordinates:
          * y        (y) int64 0 1 2
          
        Notes
        -----
        See http://www.cawcr.gov.au/projects/verification/
    """
    
    if isinstance(over_dims, str):
        over_dims = [over_dims]
        
    if over_dims == None:
        over_dims = []  

    return (da_cmp / da_ref).mean(dim=over_dims, skipna=True) \
                            .rename('mean_multiplicative_bias')

    
# ===================================================================================================
def mean_absolute_error(da_cmp, da_ref, over_dims):
    """ 
        Returns the mean absolute error between comparison and reference datasets 
        
        | Author: Dougie Squire
        | Date: 28/04/2018
        
        Parameters
        ----------
        da_cmp : xarray DataArray
            Array containing data to be compared to reference dataset (usually forecasts)
        da_ref : xarray DataArray
            Array containing reference data (usually observations)
        over_dims : str or sequence of str, optional
            Dimensions over which to compute the mean absolute error
            
        Returns
        -------
        mean_absolute_error : xarray DataArray
            Array containing the mean absolute error
            
        Examples
        --------
        >>> da_cmp = xr.DataArray(np.random.normal(size=(3,3)), 
        ...                       coords=[('x', np.arange(3)), ('y', np.arange(3))])
        >>> da_ref = xr.DataArray(np.random.normal(size=(3,3)), 
        ...                       coords=[('x', np.arange(3)), ('y', np.arange(3))])
        >>> doppyo.skill.mean_absolute_error(da_cmp, da_ref, over_dims='x')
        <xarray.DataArray 'mean_absolute_error' (y: 3)>
        array([1.030629, 1.265555, 0.770711])
        Coordinates:
          * y        (y) int64 0 1 2
          
        Notes
        -----
        See http://www.cawcr.gov.au/projects/verification/
    """
    
    if isinstance(over_dims, str):
        over_dims = [over_dims]

    if over_dims == None:
        over_dims = []  
    
    return abs(da_cmp - da_ref).mean(dim=over_dims, skipna=True) \
                               .rename('mean_absolute_error')


# ===================================================================================================
def mean_squared_error(da_cmp, da_ref, over_dims):
    """ 
        Returns the mean sqaured error between comparison and reference datasets 
        
        | Author: Dougie Squire
        | Date: 28/04/2018
        
        Parameters
        ----------
        da_cmp : xarray DataArray
            Array containing data to be compared to reference dataset (usually forecasts)
        da_ref : xarray DataArray
            Array containing reference data (usually observations)
        over_dims : str or sequence of str, optional
            Dimensions over which to compute the mean squared error
            
        Returns
        -------
        mean_squared_error : xarray DataArray
            Array containing the mean squared error
            
        Examples
        --------
        >>> da_cmp = xr.DataArray(np.random.normal(size=(3,3)), 
        ...                       coords=[('x', np.arange(3)), ('y', np.arange(3))])
        >>> da_ref = xr.DataArray(np.random.normal(size=(3,3)), 
        ...                       coords=[('x', np.arange(3)), ('y', np.arange(3))])
        >>> doppyo.skill.mean_squared_error(da_cmp, da_ref, over_dims='x')
        <xarray.DataArray 'mean_squared_error' (y: 3)>
        array([1.257412, 1.725008, 0.721863])
        Coordinates:
          * y        (y) int64 0 1 2
          
        Notes
        -----
        See http://www.cawcr.gov.au/projects/verification/
    """
    
    if isinstance(over_dims, str):
        over_dims = [over_dims]

    if over_dims == None:
        over_dims = []  
        
    return xr.ufuncs.square(da_cmp - da_ref).mean(dim=over_dims,skipna=True) \
                                            .rename('mean_squared_error')


# ===================================================================================================
def rms_error(da_cmp, da_ref, over_dims):
    """ 
        Returns the root mean sqaured error between comparison and reference datasets 
        
        | Author: Dougie Squire
        | Date: 28/04/2018
        
        Parameters
        ----------
        da_cmp : xarray DataArray
            Array containing data to be compared to reference dataset (usually forecasts)
        da_ref : xarray DataArray
            Array containing reference data (usually observations)
        over_dims : str or sequence of str, optional
            Dimensions over which to compute the root mean squared error
            
        Returns
        -------
        rms_error : xarray DataArray
            Array containing the root mean squared error
            
        Examples
        --------
        >>> da_cmp = xr.DataArray(np.random.normal(size=(3,3)), 
        ...                       coords=[('x', np.arange(3)), ('y', np.arange(3))])
        >>> da_ref = xr.DataArray(np.random.normal(size=(3,3)), 
        ...                       coords=[('x', np.arange(3)), ('y', np.arange(3))])
        >>> doppyo.skill.rms_error(da_cmp, da_ref, over_dims='x')
        <xarray.DataArray 'root_mean_squared_error' (y: 3)>
        array([1.964753, 1.426566, 1.20612 ])
        Coordinates:
          * y        (y) int64 0 1 2
        
        Notes
        -----
        See http://www.cawcr.gov.au/projects/verification/
    """
    
    return xr.ufuncs.sqrt(mean_squared_error(da_cmp, da_ref, over_dims=over_dims)) \
                    .rename('root_mean_squared_error')


# ===================================================================================================
def Pearson_corrcoeff(da_cmp, da_ref, over_dims, subtract_local_mean=True):
    """ 
        Returns the Pearson correlation coefficients over the specified dimensions. 
        
        | Author: Dougie Squire
        | Date: 28/04/2018
        
        Parameters
        ----------
        da_cmp : xarray DataArray
            Array containing data to be compared to reference dataset (usually forecasts)
        da_ref : xarray DataArray
            Array containing reference data (usually observations)
        over_dims : str or sequence of str, optional
            Dimensions over which to compute the correlation coefficients
        subtract_local_mean : bool, optional
            If True, this function will subtract the mean computed over over_dims. Otherwise, no mean\
                    field is removed prior to computing the correlation
            
        Returns
        -------
        Pearson_corrcoeff : xarray DataArray
            Array containing the Pearson correlation coefficients
            
        Examples
        --------
        >>> da_cmp = xr.DataArray(np.random.normal(size=(100,3)), 
        ...                       coords=[('x', np.arange(100)), ('y', np.arange(3))])
        >>> da_ref = xr.DataArray(np.random.normal(size=(100,3)), 
        ...                       coords=[('x', np.arange(100)), ('y', np.arange(3))])
        >>> doppyo.skill.Pearson_corrcoeff(da_cmp, da_ref, over_dims='x')
        <xarray.DataArray 'Pearson_corrcoeff' (y: 3)>
        array([-0.040584, -0.037983, -0.020941])
        Coordinates:
          * y        (y) int64 0 1 2
          
        Notes
        -----
        | If any dimensions in over_dims do not exist in either da_cmp or da_ref, the correlation is \
                computed over all dimensions in over_dims that appear in both da_cmp and da_ref, and then \
                averaged over any remaining dimensions in over_dims
        | See http://www.cawcr.gov.au/projects/verification/
    """
    
    if over_dims is None:
        raise ValueError('Pearson correlation cannot be computed over 0 dimensions') 
    elif isinstance(over_dims, str):
        over_dims = [over_dims]
    
    # Find over_dims that appear in both da_cmp and da_ref, and those that don't -----
    over_dims_in_cmp = [over_dim for over_dim in over_dims if over_dim in da_cmp.dims]
    over_dims_in_ref = [over_dim for over_dim in over_dims if over_dim in da_ref.dims]
    intersection_dims = list(set(over_dims_in_cmp).intersection(set(over_dims_in_ref)))
    difference_dims = list(set(over_dims_in_cmp).difference(set(over_dims_in_ref)))

    if subtract_local_mean:
        cov = ((da_cmp - da_cmp.mean(intersection_dims)) * 
               (da_ref - da_ref.mean(intersection_dims))).mean(intersection_dims)
        norm = da_cmp.std(intersection_dims) * da_ref.std(intersection_dims)
    else:
        cov = (da_cmp * da_ref).mean(intersection_dims)
        norm = ((da_cmp ** 2).mean(intersection_dims) ** 0.5) * \
                ((da_ref ** 2).mean(intersection_dims) ** 0.5)

    return (cov / norm).mean(difference_dims)


# ===================================================================================================
def sign_test(da_cmp1, da_cmp2, da_ref, time_dim='init_date', categorical=False):
    """
        Returns the Delsole and Tippett sign test over the given time period
        
        | Author: Dougie Squire
        | Date: 26/03/2019
        
        Parameters
        ----------
        da_cmp1 : xarray DataArray
            Array containing data to be compared to da_cmp1
        da_cmp2 : xarray DataArray
            Array containing data to be compared to da_cmp2
        da_ref : xarray DataArray
            Array containing data to use as reference
        time_dim : str, optional
            Name of dimension over which to compute the random walk
        categorical : bool, optional
            If True, the winning forecast is only rewarded a point if it exactly equals the observations
            
        Returns
        -------
        sign_test : xarray DataArray
            Array containing the results of the sign test
        confidence : xarray DataArray
            Array containing 95% confidence bounds
            
        Examples
        --------
        >>> x = xr.DataArray(np.random.normal(size=(3,3)), 
        ...                  coords=[('t', np.arange(3)), ('x', np.arange(3))])
        >>> y = xr.DataArray(np.random.normal(size=(3,3)), 
        ...                 coords=[('t', np.arange(3)), ('x', np.arange(3))])
        >>> o = xr.DataArray(np.random.normal(size=(3,3)), 
        ...                  coords=[('t', np.arange(3)), ('x', np.arange(3))])
        >>> walk, confidence = sign_test(x, y, o, time_dim='t')
        >>> walk
        <xarray.DataArray (t: 3, x: 3)>
        array([[-1, -1, -1],
               [ 0,  0, -2],
               [-1, -1, -3]])
        Coordinates:
          * t        (t) int64 0 1 2
          * x        (x) int64 0 1 2
          
        Notes
        -----
        See Delsole and Tippett 2016 `Forecast Comparison Based on Random Walks`
    """
    
    if categorical:
        cmp1_diff = -1*(da_cmp1 == da_ref)
        cmp2_diff = -1*(da_cmp2 == da_ref)
    else:
        cmp1_diff = abs(da_cmp1 - da_ref)
        cmp2_diff = abs(da_cmp2 - da_ref)

    sign_test = (1 * (cmp1_diff < cmp2_diff) - 1 * (cmp2_diff < cmp1_diff)).cumsum(time_dim)
    
    # Estimate 95% confidence interval -----
    notnan = 1*(cmp1_diff.notnull() & cmp2_diff.notnull())
    N = notnan.cumsum(time_dim)
    # z_alpha is the value at which the standardized cumulative Gaussian distributed exceeds alpha
    confidence = 1.95996496 * xr.ufuncs.sqrt(N) 
    
    return sign_test.rename('sign_test'), confidence.rename('confidence')


    
