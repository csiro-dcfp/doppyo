"""
    pyLatte functions for assessing one data-set relative to another (usually model output to observation)
    Author: Dougie Squire
    Date created: 04/04/2018
    Python Version: 3.6
"""

__all__ = ['compute_rank_histogram', 'compute_rps', 'compute_reliability', 'compute_roc', 
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

# Load cafepy packages -----
from pylatte import utils


# ===================================================================================================
# Methods for probabilistic comparisons
# ===================================================================================================
def compute_rank_histogram(da_cmp, da_ref, indep_dims, ensemble_dim='ensemble'):
    """ Returns rank histogram """
    
    if indep_dims == None:
        raise ValueError('Cannot compute rank histogram with no independent dimensions')
        
    # Rank the data -----
    da_ranked = utils.compute_rank(da_cmp, da_ref, dim=ensemble_dim)

    # Initialise bins -----
    bins = range(1,len(da_cmp[ensemble_dim])+2)
    bin_edges = utils.get_bin_edges(bins)
    
    return utils.compute_histogram(da_ranked, bin_edges, dims=indep_dims)


# ===================================================================================================
def compute_rps(da_cmp, da_ref, bins, indep_dims, ensemble_dim):
    """ Returns the (continuous) ranked probability score """

    # Initialise bins -----
    bin_edges = utils.get_bin_edges(bins)

    # Compute cumulative density functions -----
    cdf_cmp = utils.compute_cdf(da_cmp, bin_edges=bin_edges, dim=ensemble_dim)
    cdf_ref = utils.compute_cdf(da_ref, bin_edges=bin_edges, dim=None)
    
    if indep_dims == None:
        rps = utils.calc_integral((cdf_cmp - cdf_ref) ** 2, dim='bins')
    else:
        rps = utils.calc_integral((cdf_cmp - cdf_ref) ** 2, dim='bins').mean(dim=indep_dims)
    
    return rps


# ===================================================================================================
def compute_reliability(cmp_likelihood, ref_logical, cmp_prob, indep_dims, nans_as_zeros=True):
    """ 
    Computes the relative frequency of an event given the comparison likelihood and reference 
    logical event data 
    """
    
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
        if indep_dims is None:
            cmp_number_list.append(cmp_in_bin*1)
        else:
            cmp_number_list.append(cmp_in_bin.sum(dim=indep_dims))  
        
        # Number of reference occurences where comparison likelihood is within probability bin -----
        if indep_dims is None:
            ref_occur_list.append(((cmp_in_bin == True) & (ref_logical == True))*1)
        else:
            ref_occur_list.append(((cmp_in_bin == True) & (ref_logical == True)).sum(dim=indep_dims))
        
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
def compute_roc(cmp_likelihood, ref_logical, cmp_prob, indep_dims):
    """ 
    Computes the relative operating characteristic of an event given the comparison likelihood and 
    reference logical event data 
    """
    
    ref_binary = ref_logical * 1

    # Initialise probability bins -----
    dprob = np.diff(cmp_prob)/2
    cmp_prob_edges = cmp_prob[:-1]+dprob

    # Fill first probability bins with ones -----
    if indep_dims == None:
        all_ones = 0 * ref_binary + 1
    else:
        all_ones = 0 * ref_binary.mean(dim=indep_dims) + 1
    hit_rate_list = [all_ones]
    false_alarm_rate_list = [all_ones]
    
    # Loop over probability bins -----
    for idx,cmp_prob_edge in enumerate(cmp_prob_edges):

        if cmp_prob_edge >= 1.0:
            raise ValueError('cmp_prob cannot exceed 1.0')
            
        # Compute contingency table for current probability -----
        category_edges = [-np.inf, cmp_prob_edge, np.inf]
        contingency = compute_contingency_table(cmp_likelihood, ref_binary, 
                                                category_edges, indep_dims=indep_dims)
        
        # Add hit rate and false alarm rate to lists -----
        hit_rate_list.append(compute_hit_rate(contingency,yes_category=2))
        false_alarm_rate_list.append(compute_false_alarm_rate(contingency,yes_category=2))
        
    # Concatenate lists -----
    hit_rate = xr.concat(hit_rate_list, dim='comparison_probability')
    hit_rate['comparison_probability'] = cmp_prob
    false_alarm_rate = xr.concat(false_alarm_rate_list, dim='comparison_probability')
    false_alarm_rate['comparison_probability'] = cmp_prob
    
    # Package in dataset -----
    roc = hit_rate.to_dataset(name='hit_rate')
    roc.hit_rate.attrs['name'] = 'hit rate'
    roc['false_alarm_rate'] = false_alarm_rate
    roc.false_alarm_rate.attrs['name'] = 'false alarm rate'

    return roc


# ===================================================================================================
def compute_discrimination(cmp_likelihood, ref_logical, cmp_prob, indep_dims):
    """ 
    Returns the histogram of comparison likelihood when references indicate the event has occurred 
    and has not occurred
    """
    
    # Initialise probability bins -----
    cmp_prob_edges = utils.get_bin_edges(cmp_prob)

    # Compute histogram of comparison likelihoods when reference is True/False -----
    replace_val = 1000 * max(cmp_prob_edges) # Replace nans with a value not in any bin
    hist_event = utils.compute_histogram(cmp_likelihood.where(ref_logical == True).fillna(replace_val), 
                                       cmp_prob_edges, dims=indep_dims)
    hist_no_event = utils.compute_histogram(cmp_likelihood.where(ref_logical == False).fillna(replace_val), 
                                           cmp_prob_edges, dims=indep_dims)
    
    # Package in dataset -----
    discrimination = hist_event.to_dataset(name='hist_event')
    discrimination.hist_event.attrs['name'] = 'histogram of comparison likelihood, yes event'
    discrimination['hist_no_event'] = hist_no_event
    discrimination.hist_no_event.attrs['name'] = 'histogram of comparison likelihood, no event'

    return discrimination


# ===================================================================================================
def compute_Brier_score(cmp_likelihood, ref_logical, indep_dims, cmp_prob=None):
    """ 
    Computes the Brier score(s) of an event given the comparison likelihood and reference logical 
    event data. When comparison probability bins are also provided, also computes the reliability, 
    resolution and uncertainty components of the Brier score
    """
    
    if isinstance(indep_dims, str):
        indep_dims = [indep_dims]
    
    ref_binary = ref_logical.copy()*1

    # Calculate total Brier score -----
    N = 1
    if indep_dims == None:
        Brier = (1 / N) * ((cmp_likelihood - ref_binary) ** 2).sum(dim=indep_dims).rename('Brier_score')
    else: 
        N = [N * len(cmp_likelihood[indep_dim]) for indep_dim in indep_dims][0]
        Brier = (1 / N) * ((cmp_likelihood - ref_binary) ** 2).rename('Brier_score')
        
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
            
            if indep_dims == None:
                # Mean comparison probability within current probability bin -----
                mean_cmp_prob_list.append(cmp_likelihood.where(cmp_in_bin,np.nan))
                
                # Number of comparisons that fall within probability bin -----
                cmp_number_list.append(cmp_in_bin)
                
                # Number of reference occurences where comparison likelihood is within probability bin -----
                ref_occur_list.append(((cmp_in_bin == True) & (ref_logical == True)))
            else:
                # Replace likelihood with mean likelihood (so that Brier components add to total) -----
                mean_cmp_likelihood = mean_cmp_likelihood.where(~cmp_in_bin).fillna( \
                                                         mean_cmp_likelihood.where(cmp_in_bin)
                                                                            .mean(dim=indep_dims))
                
                # Mean comparison probability within current probability bin -----
                mean_cmp_prob_list.append(cmp_likelihood.where(cmp_in_bin,np.nan) \
                                                        .mean(dim=indep_dims)) 
                
                # Number of comparisons that fall within probability bin -----
                cmp_number_list.append(cmp_in_bin.sum(dim=indep_dims))
                
                # Number of reference occurences where comparison likelihood is within probability bin -----
                ref_occur_list.append(((cmp_in_bin == True) & (ref_logical == True)) \
                                        .sum(dim=indep_dims)) 

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
                                       .sum(dim='comparison_probability',skipna=True)
        if indep_dims == None:
            sample_clim = ref_binary
        else:
            sample_clim = ref_binary.mean(dim=indep_dims)
        Brier_resolution = (1 / N) * (cmp_number*(base_rate - sample_clim) ** 2) \
                                      .sum(dim='comparison_probability',skipna=True)
        Brier_uncertainty = sample_clim * (1 - sample_clim)
        
        # When a binned approach is used, compute total Brier using binned probabilities -----
        # (This way Brier_total = Brier_reliability - Brier_resolution + Brier_uncertainty)
        if indep_dims == None:
            Brier_total = (1 / N) * ((mean_cmp_likelihood - ref_binary) ** 2)
        else:
            Brier_total = (1 / N) * ((mean_cmp_likelihood - ref_binary) ** 2).sum(dim=indep_dims)
        
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
def calc_contingency(da_cmp, da_ref, category_edges, indep_dims):
    """ Returns contingency table from categorized comparison and reference arrays """
    
    categories = range(1,len(category_edges))
    
    # Loop over comparison categories -----
    cmp_list = []
    for cmp_category in categories:
        
        # Loop over reference categories -----
        ref_list = []
        for ref_category in categories:
            
            # Add to reference list -----
            if indep_dims == None:
                ref_list.append(((da_cmp == cmp_category) & (da_ref == ref_category)))
            else:
                ref_list.append(((da_cmp == cmp_category) & (da_ref == ref_category)).sum(dim=indep_dims))
        
        # Concatenate reference categories -----
        ref = xr.concat(ref_list, dim='reference_category')
        ref['reference_category'] = categories
        
        # Add to comparison list -----
        cmp_list.append(ref)
        
    # Concatenate comparison categories -----
    contingency = xr.concat(cmp_list, dim='comparison_category')
    contingency['comparison_category'] = categories
    
    return contingency


def compute_contingency_table(da_cmp, da_ref, category_edges, indep_dims, ensemble_dim=None):
    """ Return contingency table for given categories """
    
    cmp_categorized = utils.categorize(da_cmp, category_edges)
    ref_categorized = utils.categorize(da_ref, category_edges)
    
    if ensemble_dim == None:
        contingency = cmp_categorized.to_dataset(name='contingency') \
                                     .apply(calc_contingency, da_ref=ref_categorized, \
                                            category_edges=category_edges, \
                                            indep_dims=indep_dims)['contingency']
    else:
        contingency = cmp_categorized.groupby(ensemble_dim) \
                                     .apply(calc_contingency, da_ref=ref_categorized, \
                                            category_edges=category_edges, 
                                            indep_dims=indep_dims) \
                                     .sum(dim=ensemble_dim) \
                                     .rename('contingency')

    return contingency

# ===================================================================================================
def compute_accuracy_score(contingency):
    """ Returns the accuracy score given a contingency table """
    
    hits = contingency.where(contingency.reference_category == contingency.comparison_category) \
           .sum(dim=('reference_category','comparison_category'))
    N = sum_contingency(contingency, 'total')
    
    return (hits / N).rename('accuracy_score')


# ===================================================================================================
def compute_Heidke_score(contingency):
    """ Returns the Heidke skill score given a contingency table """
    
    numer_1 = contingency.where(contingency.reference_category == contingency.comparison_category) \
              .sum(dim=('reference_category','comparison_category')) / \
              sum_contingency(contingency, 'total')
    numer_2 = (sum_contingency(contingency, 'reference') * \
               sum_contingency(contingency, 'comparison')).sum('category') / \
              (sum_contingency(contingency, 'total')**2)
    denom = 1 - numer_2

    return ((numer_1 - numer_2) / denom).rename('Heidke_score')


# ===================================================================================================
def compute_Peirce_score(contingency):
    """ Returns the Hanssen and Kuipers discriminant given a contingency table """
    
    numer_1 = contingency.where(contingency.reference_category == contingency.comparison_category) \
              .sum(dim=('reference_category','comparison_category')) / \
              sum_contingency(contingency, 'total')

    numer_2 = (sum_contingency(contingency, 'reference') * \
               sum_contingency(contingency, 'comparison')).sum('category') / \
              (sum_contingency(contingency, 'total')**2)

    denom = 1 - (sum_contingency(contingency, 'reference')**2).sum('category') / \
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
                cmp_list.append((1 / (K - 1)) * ((1 / a.sel(category=range(1,i))).sum(dim='category') + \
                                                a.sel(category=range(j,K)).sum(dim='category')))
            elif i > j:
                cmp_list.append((1 / (K - 1)) * ((1 / a.sel(category=range(1,j))).sum(dim='category') - \
                                                (i - j) + a.sel(category=range(i,K)).sum(dim='category')))
            else:
                cmp_list.append((1 / (K - 1)) * ((1 / a.sel(category=range(1,i))).sum(dim='category') - \
                                                (j - i) + a.sel(category=range(j,K)).sum(dim='category')))
    
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
    a = ((1 - sum_p.cumsum('category')) / sum_p.cumsum('category'))
    
    # Compute 'S' -----
    S = calc_Gerrity_S(a)
    
    return ((contingency * S).sum(dim=('reference_category','comparison_category')) / \
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
def compute_mean_additive_bias(da_cmp, da_ref, indep_dims, ensemble_dim=None):
    """ Returns the additive bias between comparison and reference datasets """
    
    if isinstance(indep_dims, str):
        indep_dims = [indep_dims]

    if (ensemble_dim == None) & (indep_dims == None):
        mean_additive_bias = da_cmp.to_dataset(name='mean_additive_bias') \
                                 .apply(utils.calc_difference, data_2=da_ref) \
                                 ['mean_additive_bias']
    elif ensemble_dim == None:
        mean_additive_bias = da_cmp.to_dataset(name='mean_additive_bias') \
                                 .apply(utils.calc_difference, data_2=da_ref) \
                                 .mean(dim=indep_dims) \
                                 ['mean_additive_bias']
    elif indep_dims == None:
        mean_additive_bias = da_cmp.groupby(ensemble_dim) \
                                 .apply(utils.calc_difference, data_2=da_ref) \
                                 .mean(dim=ensemble_dim) \
                                 .rename('mean_additive_bias')
    else:
        cmp_mean_dims = tuple(indep_dims) + tuple([ensemble_dim])
        mean_additive_bias = da_cmp.groupby(ensemble_dim) \
                                 .apply(utils.calc_difference, data_2=da_ref) \
                                 .mean(dim=cmp_mean_dims) \
                                 .rename('mean_additive_bias')
    return mean_additive_bias


# ===================================================================================================
def compute_mean_multiplicative_bias(da_cmp, da_ref, indep_dims, ensemble_dim=None):
    """ Returns the multiplicative bias between comparison and reference datasets """
    
    if isinstance(indep_dims, str):
        indep_dims = [indep_dims]
        
    if (ensemble_dim == None) & (indep_dims == None):
        mean_multiplicative_bias = (da_cmp / da_ref).rename('mean_multiplicative_bias')
    elif ensemble_dim == None:
        mean_multiplicative_bias = (da_cmp.mean(dim=indep_dims) / da_ref.mean(dim=indep_dims)) \
                                   .rename('mean_multiplicative_bias')
    elif indep_dims == None:
        mean_multiplicative_bias = (da_cmp.mean(dim=ensemble_dim) / da_ref) \
                                   .rename('mean_multiplicative_bias')
    else:
        cmp_mean_dims = tuple(indep_dims) + tuple([ensemble_dim])
        mean_multiplicative_bias = (da_cmp.mean(dim=cmp_mean_dims) / da_ref.mean(dim=indep_dims)) \
                                   .rename('mean_multiplicative_bias')

    return mean_multiplicative_bias

    
# ===================================================================================================
def compute_mean_absolute_error(da_cmp, da_ref, indep_dims, ensemble_dim=None):
    """ Returns the mean absolute error between comparison and reference datasets """
    
    if isinstance(indep_dims, str):
        indep_dims = [indep_dims]

    if (ensemble_dim == None) & (indep_dims == None):
        mean_absolute_error = ((da_cmp.to_dataset(name='mean_absolute_error') \
                                    .apply(utils.calc_difference, data_2=da_ref) \
                                    ** 2) ** 0.5)['mean_absolute_error']
    elif ensemble_dim == None:
        mean_absolute_error = ((da_cmp.to_dataset(name='mean_absolute_error') \
                                    .apply(utils.calc_difference, data_2=da_ref) \
                                    ** 2) ** 0.5) \
                                    .mean(dim=indep_dims)['mean_absolute_error']
    elif indep_dims == None:
        mean_absolute_error = ((da_cmp.groupby(ensemble_dim) \
                                    .apply(utils.calc_difference, data_2=da_ref) \
                                    ** 2) ** 0.5) \
                                    .mean(dim=ensemble_dim) \
                                    .rename('mean_absolute_error')
    else:
        cmp_mean_dims = tuple(indep_dims) + tuple([ensemble_dim])
        mean_absolute_error = ((da_cmp.groupby(ensemble_dim) \
                                    .apply(utils.calc_difference, data_2=da_ref) \
                                    ** 2) ** 0.5) \
                                    .mean(dim=cmp_mean_dims) \
                                    .rename('mean_absolute_error')
    
    return mean_absolute_error


# ===================================================================================================
def compute_mean_squared_error(da_cmp, da_ref, indep_dims, ensemble_dim=None):
    """ Returns the mean sqaured error between comparison and reference datasets """
    
    if isinstance(indep_dims, str):
        indep_dims = [indep_dims]

    if (ensemble_dim == None) & (indep_dims == None):
        mean_squared_error = (da_cmp.to_dataset(name='mean_squared_error') \
                                  .apply(utils.calc_difference, data_2=da_ref) \
                                  ** 2)['mean_squared_error']
    elif ensemble_dim == None:
        mean_squared_error = (da_cmp.to_dataset(name='mean_squared_error') \
                                  .apply(utils.calc_difference, data_2=da_ref) \
                                  ** 2) \
                                  .mean(dim=indep_dims)['mean_squared_error']
    elif indep_dims == None:
        mean_squared_error = (da_cmp.groupby(ensemble_dim) \
                                  .apply(utils.calc_difference, data_2=da_ref) \
                                  ** 2) \
                                  .mean(dim=ensemble_dim) \
                                  .rename('mean_squared_error')
    else:
        cmp_mean_dims = tuple(indep_dims) + tuple([ensemble_dim])
        mean_squared_error = (da_cmp.groupby(ensemble_dim) \
                                  .apply(utils.calc_difference, data_2=da_ref) \
                                  ** 2) \
                                  .mean(dim=cmp_mean_dims) \
                                  .rename('mean_squared_error')
                    
    return mean_squared_error


# ===================================================================================================
def compute_rms_error(da_cmp, da_ref, indep_dims, ensemble_dim=None):
    """ Returns the mean sqaured error between comparison and reference datasets """
    
    return ((compute_mean_squared_error(da_cmp, da_ref, indep_dims=indep_dims, ensemble_dim=ensemble_dim)) ** 0.5) \
           .rename('root_mean_squared_error')

    
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
def compute_likelihood(da_logical, ensemble_dim='ensemble'):
    """ Returns array of likelihoods computed along ensemble_dim from logical event data """
    
    if ensemble_dim == None:
        likelihood = da_logical
    else:
        likelihood = da_logical.mean(dim=ensemble_dim).rename('likelihood')
    return likelihood


# ===================================================================================================
def sum_contingency(contingency, category='total'):
    """ Returns sums of specified categories in contingency table """
    
    if category == 'total':
        N = contingency.sum(dim=('reference_category','comparison_category'))
    elif category == 'reference':
        N = contingency.sum(dim='comparison_category').rename({'reference_category' : 'category'})
    elif category == 'comparison':
        N = contingency.sum(dim='reference_category').rename({'comparison_category' : 'category'})    
    else: raise ValueError(f'"{category}" is not a recognised category')
        
    return N



    