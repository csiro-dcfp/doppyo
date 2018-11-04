"""
    doppyo functions for assessing one data-set relative to another (usually model output to observation)
    Author: Dougie Squire
    Date created: 04/04/2018
    Python Version: 3.6
"""

__all__ = ['rank_histogram', 'compute_rps', 'compute_reliability', 'compute_roc', 
           'compute_discrimination', 'compute_Brier_score', 'compute_contingency_table', 
           'compute_accuracy_score', 'compute_Heidke_score', 'compute_Peirce_score', 
           'compute_Gerrity_score', 'compute_bias_score', 'compute_hit_rate', 
           'compute_false_alarm_ratio', 'compute_false_alarm_rate', 'compute_success_ratio', 
           'compute_threat_score', 'compute_equit_threat_score', 'compute_odds_ratio',
           'compute_odds_ratio_skill', 'compute_mean_additive_bias', 
           'compute_mean_multiplicative_bias', 'compute_mean_absolute_error', 
           'compute_mean_squared_error', 'compute_rms_error', 'did_event', 'compute_likelihood',
           'sum_contingency']

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
        Author: Dougie Squire
        Date: 01/11/2018
        
        Parameters
        ----------
        da_cmp : xarray DataArray
            Comparison data. This data is used to rank the reference data. Must include an ensemble 
            dimension
        da_ref : xarray DataArray
            Reference data. This data is ranked within the comparison data. Dimensions should match
            those of da_cmp
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
        >>> da_cmp = xr.DataArray(np.random.normal(size=(100,100,20)), 
        ...                       coords=[('x', np.arange(100)), ('y', np.arange(100)), ('e', np.arange(20))])
        >>> da_ref = xr.DataArray(np.random.normal(size=(100,100)), 
        ...                       coords=[('x', np.arange(100)), ('y', np.arange(100))])
        >>> doppyo.skill.rank_histogram(da_cmp, da_ref, over_dims='x', ensemble_dim='e')
        <xarray.DataArray (bins: 21, y: 100)>
        array([[0.04, 0.05, 0.04, ..., 0.06, 0.06, 0.02],
               [0.04, 0.06, 0.05, ..., 0.04, 0.03, 0.05],
               [0.06, 0.03, 0.04, ..., 0.01, 0.05, 0.01],
               ...,
               [0.05, 0.02, 0.06, ..., 0.05, 0.03, 0.08],
               [0.08, 0.07, 0.04, ..., 0.04, 0.03, 0.04],
               [0.07, 0.04, 0.05, ..., 0.08, 0.03, 0.02]])
        Coordinates:
          * bins     (bins) float64 1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0 9.0 10.0 11.0 ...
          * y        (y) int64 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 ...
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
def compute_rps(da_cmp, da_ref, bins, over_dims=None, ensemble_dim='ensemble'):
    """ Returns the ranked probability score """

    if over_dims is None:
        over_dims = []
    
    # Initialise bins -----
    bin_edges = utils.get_bin_edges(bins)

    # Compute cumulative density functions -----
    cdf_cmp = utils.compute_cdf(da_cmp, bin_edges=bin_edges, over_dims=ensemble_dim)
    cdf_ref = utils.compute_cdf(da_ref, bin_edges=bin_edges, over_dims=None)
    
    return utils.integrate((cdf_cmp - cdf_ref) ** 2, over_dim='bins') \
                .mean(dim=over_dims, skipna=True)


# ===================================================================================================
def compute_reliability(cmp_likelihood, ref_logical, cmp_prob, over_dims, nans_as_zeros=True):
    """ 
    Computes the relative frequency of an event given the comparison likelihood and reference 
    logical event data 
    """
    
    if over_dims is None:
        over_dims = []
        
    ref_binary = ref_logical.copy()*1
    
    # Initialise probability bins -----
    cmp_prob_edges = utils.get_bin_edges(cmp_prob)
    
    # Loop over probability bins -----
    cmp_number_list = []
    ref_occur_list = []
    for idx in range(len(cmp_prob_edges)-1):
        # Logical of comparisons that fall within probability bin -----
        cmp_in_bin = (cmp_likelihood >= cmp_prob_edges[idx]) & \
                     (cmp_likelihood < cmp_prob_edges[idx+1])
        
        # Number of comparisons that fall within probability bin -----
        cmp_number_list.append((1 * cmp_in_bin).sum(dim=over_dims, skipna=True))  
        
        # Number of reference occurences where comparison likelihood is within probability bin -----
        ref_occur_list.append((1 * ((cmp_in_bin == True) & (ref_logical == True))) \
                      .sum(dim=over_dims, skipna=True))
        
    # Concatenate lists -----
    cmp_number = xr.concat(cmp_number_list, dim='comparison_probability')
    cmp_number['comparison_probability'] = cmp_prob       

    ref_occur = xr.concat(ref_occur_list, dim='comparison_probability')
    ref_occur['comparison_probability'] = cmp_prob  

    # Reference relative frequency -----
    relative_freq = ref_occur / cmp_number

    # Replace nans with zeros -----
    if nans_as_zeros:
        relative_freq = relative_freq.fillna(0)

    # Package in dataset -----
    reliability = relative_freq.to_dataset(name='relative_freq')
    reliability.relative_freq.attrs['name'] = 'relative frequency'
    reliability['cmp_number'] = cmp_number
    reliability.cmp_number.attrs['name'] = 'number of comparisons'
    reliability['ref_occur'] = ref_occur
    reliability.ref_occur.attrs['name'] = 'number of reference occurences'

    return reliability


# ===================================================================================================
def compute_roc(cmp_likelihood, ref_logical, cmp_prob, over_dims):
    """ 
    Computes the relative operating characteristic of an event given the comparison likelihood and 
    reference logical event data 
    """
    
    if over_dims is None:
        over_dims = []
    
    ref_binary = ref_logical * 1

    # Initialise probability bins -----
    dprob = np.diff(cmp_prob)/2
    cmp_prob_edges = cmp_prob[:-1]+dprob

    # Fill first probability bin with ones -----
    all_ones = 0 * ref_binary.mean(dim=over_dims) + 1
    hit_rate_list = [all_ones]
    false_alarm_rate_list = [all_ones]
    
    # Loop over probability bins -----
    for idx,cmp_prob_edge in enumerate(cmp_prob_edges):

        if cmp_prob_edge >= 1.0:
            raise ValueError('cmp_prob cannot exceed 1.0')
            
        # Compute contingency table for current probability -----
        category_edges = [-np.inf, cmp_prob_edge, np.inf]
        contingency = compute_contingency_table(cmp_likelihood, ref_binary, 
                                                category_edges, over_dims=over_dims)
        
        # Add hit rate and false alarm rate to lists -----
        hit_rate_list.append(compute_hit_rate(contingency,yes_category=2))
        false_alarm_rate_list.append(compute_false_alarm_rate(contingency,yes_category=2))
    
    # Concatenate lists -----
    hit_rate = xr.concat(hit_rate_list, dim='comparison_probability')
    hit_rate['comparison_probability'] = cmp_prob
    false_alarm_rate = xr.concat(false_alarm_rate_list, dim='comparison_probability')
    false_alarm_rate['comparison_probability'] = cmp_prob
    
    # Calculate area under curve -----
    dx = false_alarm_rate - false_alarm_rate.shift(**{'comparison_probability':1})
    dx = dx.fillna(0.0)
    area = abs(((hit_rate.shift(**{'comparison_probability':1}) + hit_rate) * dx / 2.0) \
                 .fillna(0.0).sum(dim='comparison_probability'))
    
    # Package in dataset -----
    roc = hit_rate.to_dataset(name='hit_rate')
    roc.hit_rate.attrs['name'] = 'hit rate'
    roc['false_alarm_rate'] = false_alarm_rate
    roc.false_alarm_rate.attrs['name'] = 'false alarm rate'
    roc['area'] = area
    roc.area.attrs['name'] = 'area under roc'

    return roc


# ===================================================================================================
def compute_discrimination(cmp_likelihood, ref_logical, cmp_prob, over_dims):
    """ 
    Returns the histogram of comparison likelihood when references indicate the event has occurred 
    and has not occurred
    """
    
    # Initialise probability bins -----
    cmp_prob_edges = utils.get_bin_edges(cmp_prob)

    # Compute histogram of comparison likelihoods when reference is True/False -----
    replace_val = 1000 * max(cmp_prob_edges) # Replace nans with a value not in any bin
    hist_event = utils.histogram(cmp_likelihood.where(ref_logical == True).fillna(replace_val), 
                                         cmp_prob_edges, over_dims=over_dims) \
                                         / (ref_logical == True).sum(dim=over_dims)
    hist_no_event = utils.histogram(cmp_likelihood.where(ref_logical == False).fillna(replace_val), 
                                            cmp_prob_edges, over_dims=over_dims) \
                                            / (ref_logical == False).sum(dim=over_dims)
    
    # Package in dataset -----
    discrimination = hist_event.to_dataset(name='hist_event')
    discrimination.hist_event.attrs['name'] = 'histogram of comparison likelihood, yes event'
    discrimination['hist_no_event'] = hist_no_event
    discrimination.hist_no_event.attrs['name'] = 'histogram of comparison likelihood, no event'

    return discrimination


# ===================================================================================================
def compute_Brier_score(cmp_likelihood, ref_logical, over_dims, cmp_prob=None):
    """ 
    Computes the Brier score(s) of an event given the comparison likelihood and reference logical 
    event data. When comparison probability bins are also provided, also computes the reliability, 
    resolution and uncertainty components of the Brier score
    """
    
    if over_dims is None:
        over_dims = []   
    N = (0*cmp_likelihood + 1).sum(dim=over_dims, skipna=True)

    ref_binary = ref_logical.copy()*1
    
    # Calculate total Brier score -----
    Brier = (1 / N) * ((cmp_likelihood - ref_binary) ** 2).sum(dim=over_dims, skipna=True) \
                                                          .rename('Brier_score')
        
    # Calculate components
    if cmp_prob is not None:

        # Initialise probability bins -----
        cmp_prob_edges = utils.get_bin_edges(cmp_prob)

        # Initialise mean_cmp_likelihood array -----
        mean_cmp_likelihood = cmp_likelihood.copy(deep=True)
        
        # Loop over probability bins -----
        mean_cmp_prob_list = []
        cmp_number_list = []
        ref_occur_list = []
        for idx in range(len(cmp_prob_edges)-1):
            # Logical of comparisons that fall within probability bin -----
            cmp_in_bin = (cmp_likelihood >= cmp_prob_edges[idx]) & \
                         (cmp_likelihood < cmp_prob_edges[idx+1])
            
            # Replace likelihood with mean likelihood (so that Brier components add to total) -----
            mean_cmp_likelihood = mean_cmp_likelihood.where(~cmp_in_bin).fillna( \
                                                     mean_cmp_likelihood.where(cmp_in_bin)
                                                                        .mean(dim=over_dims, skipna=True))

            # Mean comparison probability within current probability bin -----
            mean_cmp_prob_list.append(cmp_likelihood.where(cmp_in_bin,np.nan) \
                                                    .mean(dim=over_dims, skipna=True)) 

            # Number of comparisons that fall within probability bin -----
            cmp_number_list.append(cmp_in_bin.sum(dim=over_dims, skipna=True))

            # Number of reference occurences where comparison likelihood is within probability bin -----
            ref_occur_list.append(((cmp_in_bin == True) & (ref_logical == True)) \
                                    .sum(dim=over_dims)) 

        # Concatenate lists -----
        mean_cmp_prob = xr.concat(mean_cmp_prob_list, dim='comparison_probability')
        mean_cmp_prob['comparison_probability'] = cmp_prob
        cmp_number = xr.concat(cmp_number_list, dim='comparison_probability')
        cmp_number['comparison_probability'] = cmp_prob
        ref_occur = xr.concat(ref_occur_list, dim='comparison_probability')
        ref_occur['comparison_probability'] = cmp_prob

        # Compute Brier components -----
        base_rate = ref_occur / cmp_number
        Brier_reliability = (1 / N) * (cmp_number*(mean_cmp_prob - base_rate) ** 2) \
                                       .sum(dim='comparison_probability', skipna=True)
            
        sample_clim = ref_binary.mean(dim=over_dims, skipna=True)
        Brier_resolution = (1 / N) * (cmp_number*(base_rate - sample_clim) ** 2) \
                                      .sum(dim='comparison_probability', skipna=True)
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
def compute_contingency_table(da_cmp, da_ref, category_edges_cmp, category_edges_ref, over_dims):
    """ Return contingency table for given categories """
    
    if over_dims is None:
        over_dims = []
    
    cmp_edges = [(category_edges_cmp[i], category_edges_cmp[i+1],i+1) for i in range(len(category_edges_cmp)-1)]
    ref_edges = [(category_edges_ref[i], category_edges_ref[i+1],i+1) for i in range(len(category_edges_ref)-1)]
    da_list = []
    for category in itertools.product(cmp_edges, ref_edges):
        da_temp = (((da_cmp >= category[0][0]) & (da_cmp < category[0][1])) & \
                   ((da_ref >= category[1][0]) & (da_ref < category[1][1]))).sum(dim=over_dims)
        da_temp.coords['comparison_category'] = category[0][2]
        da_temp.coords['reference_category'] = category[1][2]
        da_list.append(da_temp)
    
    if len(da_list) == 1:
        return da_list[0]
    else:
        return xr.concat(da_list, dim='stack').set_index(stack=['comparison_category', 'reference_category']).unstack('stack')   

# ===================================================================================================
def compute_accuracy_score(contingency):
    """ Returns the accuracy score given a contingency table """
    
    hits = contingency.where(contingency.reference_category == contingency.comparison_category) \
           .sum(dim=('reference_category','comparison_category'), skipna=True)
    N = sum_contingency(contingency, 'total')
    
    return (hits / N).rename('accuracy_score')


# ===================================================================================================
def compute_Heidke_score(contingency):
    """ Returns the Heidke skill score given a contingency table """
    
    numer_1 = contingency.where(contingency.reference_category == contingency.comparison_category) \
              .sum(dim=('reference_category','comparison_category'), skipna=True) / \
              sum_contingency(contingency, 'total')
    numer_2 = (sum_contingency(contingency, 'reference') * \
               sum_contingency(contingency, 'comparison')).sum(dim='category', skipna=True) / \
              (sum_contingency(contingency, 'total')**2)
    denom = 1 - numer_2

    return ((numer_1 - numer_2) / denom).rename('Heidke_score')


# ===================================================================================================
def compute_Peirce_score(contingency):
    """ Returns the Hanssen and Kuipers discriminant given a contingency table """
    
    numer_1 = contingency.where(contingency.reference_category == contingency.comparison_category) \
              .sum(dim=('reference_category','comparison_category'), skipna=True) / \
              sum_contingency(contingency, 'total')

    numer_2 = (sum_contingency(contingency, 'reference') * \
               sum_contingency(contingency, 'comparison')).sum(dim='category', skipna=True) / \
              (sum_contingency(contingency, 'total')**2)

    denom = 1 - (sum_contingency(contingency, 'reference')**2).sum(dim='category', skipna=True) / \
                (sum_contingency(contingency, 'total')**2)

    return ((numer_1 - numer_2) / denom).rename('Peirce_score')

# ===================================================================================================
def calc_Gerrity_S(a):
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


def compute_Gerrity_score(contingency):
    """ Returns Gerrity equitable score given a contingency table """
    
    # Compute 'a' -----
    sum_p = (sum_contingency(contingency, 'reference') / \
             sum_contingency(contingency, 'total'))
    a = ((1 - sum_p.cumsum('category', skipna=True)) / sum_p.cumsum('category', skipna=True))
    
    # Compute 'S' -----
    S = calc_Gerrity_S(a)
    
    return ((contingency * S).sum(dim=('reference_category','comparison_category'), skipna=True) / \
           sum_contingency(contingency, 'total')).rename('Gerrity_score')


# ===================================================================================================
# Methods for dichotomously categorized comparisons
# ===================================================================================================
def compute_bias_score(contingency, yes_category=2):
    """ Returns the bias score given dichotomous contingency data """
    
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
def compute_hit_rate(contingency, yes_category=2):
    """ Returns the probability of detection given dichotomous contingency data """
    
    no_category = abs(yes_category - 2) + 1
    
    if len(contingency.comparison_category) > 2:
        raise ValueError('Bias score is defined for dichotomous contingency data only')
    
    hits = contingency.sel(comparison_category=yes_category, 
                           reference_category=yes_category, drop=True)
    misses = contingency.sel(comparison_category=no_category, 
                             reference_category=yes_category, drop=True)
    
    return (hits / (hits + misses)).rename('hit_rate')


# ===================================================================================================
def compute_false_alarm_ratio(contingency, yes_category=2):
    """ Returns the false alarm ratio given dichotomous contingency data """
    
    no_category = abs(yes_category - 2) + 1
    
    if len(contingency.comparison_category) > 2:
        raise ValueError('Bias score is defined for dichotomous contingency data only')
    
    hits = contingency.sel(comparison_category=yes_category, 
                           reference_category=yes_category, drop=True)
    false_alarms = contingency.sel(comparison_category=yes_category, 
                                   reference_category=no_category, drop=True)

    return (false_alarms / (hits + false_alarms)).rename('false_alarm_ratio')


# ===================================================================================================
def compute_false_alarm_rate(contingency, yes_category=2):
    """ Returns the false alarm rate given dichotomous contingency data """
    
    no_category = abs(yes_category - 2) + 1
    
    if len(contingency.comparison_category) > 2:
        raise ValueError('Bias score is defined for dichotomous contingency data only')
        
    false_alarms = contingency.sel(comparison_category=yes_category, 
                                   reference_category=no_category, drop=True)
    correct_negs = contingency.sel(comparison_category=no_category, 
                                   reference_category=no_category, drop=True)

    return (false_alarms / (correct_negs + false_alarms)).rename('false_alarm_rate')


# ===================================================================================================
def compute_success_ratio(contingency, yes_category=2):
    """ Returns the success ratio given dichotomous contingency data """
    
    no_category = abs(yes_category - 2) + 1
    
    if len(contingency.comparison_category) > 2:
        raise ValueError('Bias score is defined for dichotomous contingency data only')
        
    hits = contingency.sel(comparison_category=yes_category, 
                           reference_category=yes_category, drop=True)
    false_alarms = contingency.sel(comparison_category=yes_category, 
                                   reference_category=no_category, drop=True)
    
    return (hits / (hits + false_alarms)).rename('success_ratio')


# ===================================================================================================
def compute_threat_score(contingency, yes_category=2):
    """ Returns the threat score given dichotomous contingency data """
    
    no_category = abs(yes_category - 2) + 1
    
    if len(contingency.comparison_category) > 2:
        raise ValueError('Bias score is defined for dichotomous contingency data only')
        
    hits = contingency.sel(comparison_category=yes_category, 
                           reference_category=yes_category, drop=True)
    misses = contingency.sel(comparison_category=no_category, 
                             reference_category=yes_category, drop=True)
    false_alarms = contingency.sel(comparison_category=yes_category, 
                                   reference_category=no_category, drop=True)
    
    return (hits / (hits + misses + false_alarms)).rename('threat_score')


# ===================================================================================================
def compute_equit_threat_score(contingency, yes_category=2):
    """ Returns the equitable threat score given dichotomous contingency data """
    
    no_category = abs(yes_category - 2) + 1
    
    if len(contingency.comparison_category) > 2:
        raise ValueError('Bias score is defined for dichotomous contingency data only')
        
    hits = contingency.sel(comparison_category=yes_category, 
                           reference_category=yes_category, drop=True)
    misses = contingency.sel(comparison_category=no_category, 
                             reference_category=yes_category, drop=True)
    false_alarms = contingency.sel(comparison_category=yes_category, 
                                   reference_category=no_category, drop=True)
    hits_random = ((hits + misses) * (hits + false_alarms)) / sum_contingency(contingency, 'total')
    
    return ((hits - hits_random) / (hits + misses + false_alarms + hits_random)).rename('equit_threat_score')


# ===================================================================================================
def compute_odds_ratio(contingency, yes_category=2):
    """ Returns the odds ratio given dichotomous contingency data """

    no_category = abs(yes_category - 2) + 1
    
    if len(contingency.comparison_category) > 2:
        raise ValueError('Bias score is defined for dichotomous contingency data only')
        
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
def compute_odds_ratio_skill(contingency, yes_category=2):
    """ Returns the odds ratio skill score given dichotomous contingency data """
    
    no_category = abs(yes_category - 2) + 1
    
    if len(contingency.comparison_category) > 2:
        raise ValueError('Bias score is defined for dichotomous contingency data only')
        
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
def compute_mean_additive_bias(da_cmp, da_ref, over_dims):
    """ Returns the additive bias between comparison and reference datasets """
    
    if isinstance(over_dims, str):
        over_dims = [over_dims]
        
    if over_dims == None:
        over_dims = []

    mean_additive_bias = da_cmp.to_dataset(name='mean_additive_bias') \
                             .apply(utils.subtract, data_2=da_ref) \
                             .mean(dim=over_dims, skipna=True) \
                             ['mean_additive_bias']

    return mean_additive_bias


# ===================================================================================================
def compute_mean_multiplicative_bias(da_cmp, da_ref, over_dims):
    """ Returns the multiplicative bias between comparison and reference datasets """
    
    if isinstance(over_dims, str):
        over_dims = [over_dims]
        
    if over_dims == None:
        over_dims = []  

    mean_multiplicative_bias = (da_cmp / da_ref).mean(dim=over_dims, skipna=True) \
                               .rename('mean_multiplicative_bias')

    return mean_multiplicative_bias

    
# ===================================================================================================
def compute_mean_absolute_error(da_cmp, da_ref, over_dims):
    """ Returns the mean absolute error between comparison and reference datasets """
    
    if isinstance(over_dims, str):
        over_dims = [over_dims]

    if over_dims == None:
        over_dims = []  
    
    mean_absolute_error = ((da_cmp.to_dataset(name='mean_absolute_error') \
                                  .apply(utils.subtract, data_2=da_ref) \
                                  ** 2) ** 0.5) \
                                  .mean(dim=over_dims, skipna=True)['mean_absolute_error']
    
    return mean_absolute_error


# ===================================================================================================
def compute_mean_squared_error(da_cmp, da_ref, over_dims):
    """ Returns the mean sqaured error between comparison and reference datasets """
    
    if isinstance(over_dims, str):
        over_dims = [over_dims]

    if over_dims == None:
        over_dims = []  
        
    mean_squared_error = (da_cmp.to_dataset(name='mean_squared_error') \
                                .apply(utils.subtract, data_2=da_ref) \
                                ** 2) \
                                .mean(dim=over_dims,skipna=True)['mean_squared_error']
                    
    return mean_squared_error


# ===================================================================================================
def compute_rms_error(da_cmp, da_ref, over_dims):
    """ Returns the mean sqaured error between comparison and reference datasets """
    
    return ((compute_mean_squared_error(da_cmp, da_ref, over_dims=over_dims)) ** 0.5) \
           .rename('root_mean_squared_error')


# ===================================================================================================
def compute_Pearson_corrcoef(da_cmp, da_ref, over_dims, subtract_local_mean=True):
    """ 
    Returns the Pearson correlation over the specified dimensions. 
    
    If any dimensions in over_dims do not exist in either da_cmp or da_ref, the correlation is computed
    over all dimensions in over_dims that appear in both da_cmp and da_ref, and then averaged over any
    remaining dimensions in over_dims
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

    return (cov / norm).mean(difference_dims).rename('Pearson_corrcoef')


# ===================================================================================================
# General verification functions
# ===================================================================================================
def did_event(da, event):
    """ Returns array containing True/False where event occurs/does not occur """
    
    eval_expr = event.replace(">", "da >").replace("<", "da <").replace("==", "da ==") \
                     .replace("=", "da ==").replace('&&', '&').replace('||', '|') \
                     .replace("and", "&").replace("or", "|")
    eval_expr = '(' + eval_expr + ').rename("event_logical")'
    
    return eval(eval_expr)


# ===================================================================================================
def compute_likelihood(da_logical, dim='ensemble'):
    """ Returns array of likelihoods computed along dim from logical event data """
    
    if dim == None:
        likelihood = da_logical
    else:
        likelihood = da_logical.mean(dim=dim).rename('likelihood')
    return likelihood


# ===================================================================================================
def sum_contingency(contingency, category='total'):
    """ Returns sums of specified categories in contingency table """
    
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



    