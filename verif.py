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
from pyLatte import utils


# ===================================================================================================
# Methods for probabilistic forecasts
# ===================================================================================================
def compute_rank_histogram(fcst, obsv, indep_dims, ensemble_dim='ensemble'):
    """ Returns rank histogram """
    
    # Rank the data -----
    da_ranked = utils.compute_rank(fcst, obsv, dim=ensemble_dim)

    # Initialise bins -----
    bins = range(1,len(fcst[ensemble_dim])+2)
    bin_edges = utils.get_bin_edges(bins)
    
    return utils.compute_histogram(da_ranked, bin_edges, dims=indep_dims)


# ===================================================================================================
def compute_rps(fcst, obsv, bins, indep_dims, ensemble_dim):
    """ Returns the (continuous) ranked probability score """

    if isinstance(indep_dims, str):
            indep_dims = [indep_dims]
    fcst_hist_dims = tuple(indep_dims) + tuple([ensemble_dim])
    obsv_hist_dims = indep_dims

    # Initialise bins -----
    bin_edges = utils.get_bin_edges(bins)

    # Compute cumulative density functions -----
    cdf_fcst = utils.compute_cdf(fcst, bin_edges=bin_edges, dim=ensemble_dim)
    cdf_obsv = utils.compute_cdf(obsv, bin_edges=bin_edges, dim=None)
    
    return utils.calc_integral((cdf_fcst - cdf_obsv) ** 2, dim='bins').mean(dim=indep_dims)


# ===================================================================================================
def compute_reliability(fcst_likelihood, obsv_logical, fcst_prob, indep_dims, nans_as_zeros=True):
    """ 
    Computes the relative frequency of an event given the forecast likelihood and observation 
    logical event data 
    """
    
    obsv_binary = obsv_logical.copy()*1
    
    # Initialise probability bins -----
    fcst_prob_edges = utils.get_bin_edges(fcst_prob)
    
    # Logical of forecasts that fall within probability bin -----
    fcst_in_bin = (fcst_likelihood >= fcst_prob_edges[0]) & \
                  (fcst_likelihood < fcst_prob_edges[1])
    
    # Number of forecasts that fall within probability bin -----
    if indep_dims is None:
        fcst_number = fcst_in_bin*1
    else:
        fcst_number = fcst_in_bin.sum(dim=indep_dims)
    fcst_number.coords['forecast_probability'] = fcst_prob[0]
    fcst_number = fcst_number.expand_dims('forecast_probability')
    
    # Number of observed occurences where forecast likelihood is within probability bin -----
    if indep_dims is None:
        obsved_occur = ((fcst_in_bin == True) & (obsv_logical == True))*1
    else:
        obsved_occur = ((fcst_in_bin == True) & (obsv_logical == True)).sum(dim=indep_dims)
    obsved_occur.coords['forecast_probability'] = fcst_prob[0]
    obsved_occur = obsved_occur.expand_dims('forecast_probability')

    # Loop over probability bins -----
    for idx in range(1,len(fcst_prob_edges)-1):
        # Logical of forecasts that fall within probability bin -----
        fcst_in_bin = (fcst_likelihood >= fcst_prob_edges[idx]) & \
                      (fcst_likelihood < fcst_prob_edges[idx+1])
        fcst_in_bin.coords['forecast_probability'] = fcst_prob[idx]
        
        # Number of forecasts that fall within probability bin -----
        if indep_dims is None:
            fcst_number = xr.concat([fcst_number, fcst_in_bin*1], dim='forecast_probability')
        else:
            fcst_number = xr.concat([fcst_number, fcst_in_bin.sum(dim=indep_dims)], 
                                dim='forecast_probability')
        
        # Number of observed occurences where forecast likelihood is within probability bin -----
        if indep_dims is None:
            obsved_occur = xr.concat([obsved_occur, ((fcst_in_bin == True) \
                                                 & (obsv_logical == True))*1], \
                                     dim='forecast_probability')
        else:
            obsved_occur = xr.concat([obsved_occur, ((fcst_in_bin == True) \
                                                 & (obsv_logical == True)) \
                                     .sum(dim=indep_dims)],dim='forecast_probability')

    # Observed relative frequency -----
    relative_freq = obsved_occur / fcst_number
    
    # Replace nans with zeros -----
    if nans_as_zeros:
        relative_freq = relative_freq.fillna(0)
        
    # Package in dataset -----
    reliability = relative_freq.to_dataset(name='relative_freq')
    reliability.relative_freq.attrs['name'] = 'relative frequency'
    reliability['fcst_number'] = fcst_number
    reliability.fcst_number.attrs['name'] = 'number of forecasts'
    reliability['obsved_occur'] = obsved_occur
    reliability.obsved_occur.attrs['name'] = 'number of observed occurences'
    
    return reliability


# ===================================================================================================
def compute_roc(fcst_likelihood, obsv_logical, fcst_prob, indep_dims):
    
    obsv_binary = obsv_logical * 1

    # Initialise probability bins -----
    dprob = np.diff(fcst_prob)/2
    fcst_prob_edges = fcst_prob[:-1]+dprob

    # Initialise first (0.0) probability bin -----
    hit_rate = 0 * obsv_binary.sum(dim=indep_dims) + 1
    hit_rate.coords['forecast_probability'] = 0.0
    hit_rate = hit_rate.expand_dims('forecast_probability')
    false_alarm_rate = 0 * obsv_binary.sum(dim=indep_dims) + 1
    false_alarm_rate.coords['forecast_probability'] = 0.0
    false_alarm_rate = false_alarm_rate.expand_dims('forecast_probability')

    # Loop over other probability bins -----
    for idx,fcst_prob_edge in enumerate(fcst_prob_edges):

        if fcst_prob_edge >= 1.0:
            raise ValueError('fcst_probabilities cannot exceed 1.0')
            
        category_edges = [-1000, fcst_prob_edge, 1000]
        contingency = compute_contingency_table(fcst_likelihood, obsv_binary, 
                                                          category_edges, indep_dims=indep_dims)
        hit_rate_temp = compute_hit_rate(contingency,yes_category=2)
        hit_rate_temp.coords['forecast_probability'] = fcst_prob[idx+1]
        hit_rate = xr.concat([hit_rate, hit_rate_temp],dim='forecast_probability')
        false_alarm_rate_temp = compute_false_alarm_rate(contingency,yes_category=2)
        false_alarm_rate_temp.coords['forecast_probability'] = fcst_prob[idx+1]
        false_alarm_rate = xr.concat([false_alarm_rate, false_alarm_rate_temp],
                                     dim='forecast_probability')

    # Package in dataset -----
    roc = hit_rate.to_dataset(name='hit_rate')
    roc.hit_rate.attrs['name'] = 'hit rate'
    roc['false_alarm_rate'] = false_alarm_rate
    roc.false_alarm_rate.attrs['name'] = 'false alarm rate'

    return roc


# ===================================================================================================
def compute_discrimination(fcst_likelihood, obsv_logical, fcst_prob, indep_dims):
    """ 
    Returns the histogram of forecast likelihood when observations indicate the event has occurred 
    and has not occurred
    """
    
    # Initialise probability bins -----
    fcst_prob_edges = utils.get_bin_edges(fcst_prob)

    # Compute histogram of forecast likelihoods when observation is True/False -----
    replace_val = 100*max(fcst_prob_edges) # Replace nans with a value not in any bin
    hist_obsved = utils.compute_histogram(fcst_likelihood.where(obsv_logical == True) \
                                                             .fillna(replace_val), 
                                              fcst_prob_edges, dims=indep_dims)
    hist_not_obsved = utils.compute_histogram(fcst_likelihood.where(obsv_logical == False) \
                                                                 .fillna(replace_val), 
                                                  fcst_prob_edges, dims=indep_dims)
    
    # Package in dataset -----
    discrimination = hist_obsved.to_dataset(name='hist_obsved')
    discrimination.hist_obsved.attrs['name'] = 'histogram of forecast likelihood, event observed'
    discrimination['hist_not_obsved'] = hist_not_obsved
    discrimination.hist_not_obsved.attrs['name'] = 'histogram of forecast likelihood, event not observed'

    return discrimination


# ===================================================================================================
def compute_Brier_score(fcst_likelihood, obsv_logical, indep_dims, fcst_prob=None):
    """ 
    Computes the Brier score(s) of an event given the forecast likelihood and observation logical 
    event data. When forecast probability bins are also provided, also computes the reliability, 
    resolution and uncertainty components of the Brier score
    """
    
    if isinstance(indep_dims, str):
        indep_dims = [indep_dims]
    
    obsv_binary = obsv_logical.copy()*1

    # Calculate total Brier score -----
    N = 1
    for indep_dim in indep_dims:
        N = N*len(fcst_likelihood[indep_dim])
    Brier = (1/N)*((fcst_likelihood - obsv_binary)**2).sum(dim=indep_dims).rename('Brier_score')

    # Calculate components
    if fcst_prob is not None:

        # Initialise probability bins -----
        fcst_prob_edges = utils.get_bin_edges(fcst_prob)

        # Initialise mean_fcst_prob array -----
        mean_fcst_likelihood = fcst_likelihood.copy()
        
        # Logical of forecasts that fall within first probability bin -----
        fcst_in_bin = (fcst_likelihood >= fcst_prob_edges[0]) & \
                      (fcst_likelihood < fcst_prob_edges[1])

        # Compute mean forecast probability -----
        mean_fcst_likelihood = mean_fcst_likelihood.where(~fcst_in_bin) \
                               .fillna(mean_fcst_likelihood.where(fcst_in_bin).mean(dim=indep_dims))
        
        # Mean forecast probability within first probability bin -----
        mean_fcst_prob = fcst_likelihood.where(fcst_in_bin,np.nan).mean(dim=indep_dims)
        mean_fcst_prob.coords['forecast_probability'] = fcst_prob[0]
        mean_fcst_prob = mean_fcst_prob.expand_dims('forecast_probability')

        # Number of forecasts that fall within probability bin -----
        fcst_number = fcst_in_bin.sum(dim=indep_dims)
        fcst_number.coords['forecast_probability'] = fcst_prob[0]
        fcst_number = fcst_number.expand_dims('forecast_probability')

        # Number of observed occurences where forecast likelihood is within probability bin -----
        obsved_occur = ((fcst_in_bin == True) & (obsv_logical == True)).sum(dim=indep_dims)
        obsved_occur.coords['forecast_probability'] = fcst_prob[0]
        obsved_occur = obsved_occur.expand_dims('forecast_probability')

        # Loop over probability bins -----
        for idx in range(1,len(fcst_prob_edges)-1):
            # Logical of forecasts that fall within probability bin -----
            del fcst_in_bin
            fcst_in_bin = (fcst_likelihood >= fcst_prob_edges[idx]) & \
                          (fcst_likelihood < fcst_prob_edges[idx+1])
            mean_fcst_likelihood = mean_fcst_likelihood.where(~fcst_in_bin) \
                                   .fillna(mean_fcst_likelihood.where(fcst_in_bin).mean(dim=indep_dims))
            fcst_in_bin.coords['forecast_probability'] = fcst_prob[idx]
            
            # Mean forecast probability within current probability bin -----
            mean_fcst_prob_temp = fcst_likelihood.where(fcst_in_bin,np.nan).mean(dim=indep_dims)
            mean_fcst_prob_temp.coords['forecast_probability'] = fcst_prob[idx]
            mean_fcst_prob = xr.concat([mean_fcst_prob, mean_fcst_prob_temp],
                                       dim='forecast_probability')

            # Number of forecasts that fall within probability bin -----
            fcst_number = xr.concat([fcst_number, fcst_in_bin.sum(dim=indep_dims)], 
                                    dim='forecast_probability')

            # Number of observed occurences where forecast likelihood is within probability bin -----
            obsved_occur = xr.concat([obsved_occur, ((fcst_in_bin == True) \
                                                     & (obsv_logical == True))
                                      .sum(dim=indep_dims)],dim='forecast_probability')

        # Compute Brier components -----
        base_rate = obsved_occur / fcst_number
        Brier_reliability = (1 / N) * (fcst_number*(mean_fcst_prob - base_rate) ** 2) \
                                       .sum(dim='forecast_probability',skipna=True)
        sample_clim = obsv_binary.mean(dim=indep_dims)
        Brier_resolution = (1 / N) * (fcst_number*(base_rate - sample_clim) ** 2) \
                                      .sum(dim='forecast_probability',skipna=True)
        Brier_uncertainty = sample_clim * (1 - sample_clim)
        
        # When a binned approach is used, compute total Brier using binned probabilities -----
        # (This way Brier_total = Brier_reliability - Brier_resolution + Brier_uncertainty)
        Brier_total = (1 / N) * ((mean_fcst_likelihood - obsv_binary) ** 2).sum(dim=indep_dims)

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
# Methods for categorized forecasts
# ===================================================================================================
def calc_contingency(fcst, obsv, category_edges, indep_dims):
    """ Returns contingency table from categorized forecast and observation arrays """
    
    categories = range(1,len(category_edges))
    
    # Initialize first row -----
    fcst_row = ((fcst == categories[0]) & (obsv == categories[0])).sum(dim=indep_dims)
    fcst_row['observed_category'] = categories[0]
    fcst_row['forecast_category'] = categories[0]
    for obsv_category in categories[1:]:
        fcst_row_temp = ((fcst == categories[0]) & (obsv == obsv_category)).sum(dim=indep_dims)
        fcst_row_temp['observed_category'] = obsv_category
        fcst_row_temp['forecast_category'] = categories[0]
        fcst_row = xr.concat([fcst_row, fcst_row_temp],'observed_category')
    contingency = fcst_row
    
    # Compute other rows -----
    for fcst_category in categories[1:]:
        fcst_row = ((fcst == fcst_category) & (obsv == categories[0])).sum(dim=indep_dims)
        fcst_row['observed_category'] = categories[0]
        fcst_row['forecast_category'] = fcst_category
        for obsv_category in categories[1:]:
            fcst_row_temp = ((fcst == fcst_category) & (obsv == obsv_category)).sum(dim=indep_dims)
            fcst_row_temp['observed_category'] = obsv_category
            fcst_row_temp['forecast_category'] = fcst_category
            fcst_row = xr.concat([fcst_row, fcst_row_temp], 'observed_category')
        contingency = xr.concat([contingency, fcst_row], 'forecast_category')

    return contingency


def compute_contingency_table(da_fcst, da_obsv, category_edges, indep_dims, ensemble_dim=None):
    """ Return contingency table for given categories """
    
    fcst_categorized = utils.categorize(da_fcst, category_edges)
    obsv_categorized = utils.categorize(da_obsv, category_edges)
    
    if ensemble_dim == None:
        contingency = fcst_categorized.to_dataset(name='contingency') \
                                      .apply(calc_contingency, obsv=obsv_categorized, \
                                             category_edges=category_edges, \
                                             indep_dims=indep_dims)['contingency']
    else:
        contingency = fcst_categorized.groupby(ensemble_dim) \
                                      .apply(calc_contingency, obsv=obsv_categorized, \
                                             category_edges=category_edges, indep_dims=indep_dims) \
                                      .sum(dim=ensemble_dim) \
                                      .rename('contingency')

    return contingency


# ===================================================================================================
def compute_accuracy_score(contingency):
    """ Returns the accuracy score given a contingency table """
    
    hits = contingency.where(contingency.observed_category == contingency.forecast_category) \
           .sum(dim=('observed_category','forecast_category'))
    N = sum_contingency(contingency, 'total')
    
    return (hits / N).rename('accuracy_score')


# ===================================================================================================
def compute_Heidke_score(contingency):
    """ Returns the Heidke skill score given a contingency table """
    
    numer_1 = contingency.where(contingency.observed_category == contingency.forecast_category) \
              .sum(dim=('observed_category','forecast_category')) / \
              sum_contingency(contingency, 'total')
    numer_2 = (sum_contingency(contingency, 'observed') * \
               sum_contingency(contingency, 'forecast')).sum('category') / \
              (sum_contingency(contingency, 'total')**2)
    denom = 1 - numer_2

    return ((numer_1 - numer_2) / denom).rename('Heidke_score')


# ===================================================================================================
def compute_Peirce_score(contingency):
    """ Returns the Hanssen and Kuipers discriminant given a contingency table """
    
    numer_1 = contingency.where(contingency.observed_category == contingency.forecast_category) \
              .sum(dim=('observed_category','forecast_category')) / \
              sum_contingency(contingency, 'total')

    numer_2 = (sum_contingency(contingency, 'observed') * \
               sum_contingency(contingency, 'forecast')).sum('category') / \
              (sum_contingency(contingency, 'total')**2)

    denom = 1 - (sum_contingency(contingency, 'observed')**2).sum('category') / \
                (sum_contingency(contingency, 'total')**2)

    return ((numer_1 - numer_2) / denom).rename('Peirce_score')

# ===================================================================================================
def calc_Gerrity_S(a):
    """ Returns Gerrity scoring matrix, S """
    
    categories = a.category.values
    K = len(categories)

    # Initialize first row -----
    i = categories[0]
    j = categories[0]
    obsv_row = (1 / (K - 1)) * ((1 / a.sel(category=range(1,i))).sum(dim='category') + \
                        a.sel(category=range(j,K)).sum(dim='category'))
    obsv_row['observed_category'] = i
    obsv_row['forecast_category'] = j
    for fcst_category in categories[1:]:
        j = fcst_category
        obsv_row_temp = (1 / (K - 1)) * ((1 / a.sel(category=range(1,i))).sum(dim='category') - \
                                        (j - i) + a.sel(category=range(j,K)).sum(dim='category'))    
        obsv_row_temp['observed_category'] = i
        obsv_row_temp['forecast_category'] = j
        obsv_row = xr.concat([obsv_row, obsv_row_temp],'forecast_category')
    S = obsv_row

    # Compute other rows -----
    for obsv_category in categories[1:]:
        i = obsv_category
        j = categories[0]
        if i > j:
            obsv_row = (1 / (K - 1)) * ((1 / a.sel(category=range(1,j))).sum(dim='category') - \
                                       (i - j) + a.sel(category=range(i,K)).sum(dim='category'))
        else:
            obsv_row = (1 / (K - 1)) * ((1 / a.sel(category=range(1,i))).sum(dim='category') - \
                                       (j - i) + a.sel(category=range(j,K)).sum(dim='category'))
        obsv_row['observed_category'] = i
        obsv_row['forecast_category'] = j
        obsv_row
        for fcst_category in categories[1:]:
            j = fcst_category
            if obsv_category == fcst_category:
                obsv_row_temp = (1 / (K - 1)) * ((1 / a.sel(category=range(1,i))).sum(dim='category') + \
                                                a.sel(category=range(j,K)).sum(dim='category'))
            elif i > j:
                obsv_row_temp = (1 / (K - 1)) * ((1 / a.sel(category=range(1,j))).sum(dim='category') - \
                                                (i - j) + a.sel(category=range(i,K)).sum(dim='category'))
            else:
                obsv_row_temp = (1 / (K - 1)) * ((1 / a.sel(category=range(1,i))).sum(dim='category') - \
                                                (j - i) + a.sel(category=range(j,K)).sum(dim='category'))
            obsv_row_temp['observed_category'] = obsv_category
            obsv_row_temp['forecast_category'] = fcst_category
            obsv_row = xr.concat([obsv_row, obsv_row_temp], 'forecast_category')
        S = xr.concat([S, obsv_row], 'observed_category')
        
    return S


def compute_Gerrity_score(contingency):
    """ Returns Gerrity equitable score given a contingency table """
    
    # Compute 'a' -----
    sum_p = (sum_contingency(contingency, 'observed') / \
             sum_contingency(contingency, 'total'))
    a = ((1 - sum_p.cumsum('category')) / sum_p.cumsum('category'))
    
    # Compute 'S' -----
    S = calc_Gerrity_S(a)
    
    return ((contingency * S).sum(dim=('observed_category','forecast_category')) / \
           sum_contingency(contingency, 'total')).rename('Gerrity_score')


# ===================================================================================================
# Methods for dichotomously categorized forecasts
# ===================================================================================================
def compute_bias_score(contingency, yes_category=2):
    """ Returns the bias score given dichotomous contingency data """
    
    no_category = abs(yes_category - 2) + 1
    
    if len(contingency.forecast_category) > 2:
        raise ValueError('Bias score is defined for dichotomous contingency data only')
    
    hits = contingency.sel(forecast_category=yes_category, 
                           observed_category=yes_category, drop=True)
    false_alarms = contingency.sel(forecast_category=yes_category, 
                                   observed_category=no_category, drop=True)
    misses = contingency.sel(forecast_category=no_category, 
                             observed_category=yes_category, drop=True)

    return ((hits + false_alarms) / (hits + misses)).rename('bias_score')


# ===================================================================================================
def compute_hit_rate(contingency, yes_category=2):
    """ Returns the probability of detection given dichotomous contingency data """
    
    no_category = abs(yes_category - 2) + 1
    
    if len(contingency.forecast_category) > 2:
        raise ValueError('Bias score is defined for dichotomous contingency data only')
    
    hits = contingency.sel(forecast_category=yes_category, 
                           observed_category=yes_category, drop=True)
    misses = contingency.sel(forecast_category=no_category, 
                             observed_category=yes_category, drop=True)
    
    return (hits / (hits + misses)).rename('hit_rate')


# ===================================================================================================
def compute_false_alarm_ratio(contingency, yes_category=2):
    """ Returns the false alarm ratio given dichotomous contingency data """
    
    no_category = abs(yes_category - 2) + 1
    
    if len(contingency.forecast_category) > 2:
        raise ValueError('Bias score is defined for dichotomous contingency data only')
    
    hits = contingency.sel(forecast_category=yes_category, 
                           observed_category=yes_category, drop=True)
    false_alarms = contingency.sel(forecast_category=yes_category, 
                                   observed_category=no_category, drop=True)

    return (false_alarms / (hits + false_alarms)).rename('false_alarm_ratio')


# ===================================================================================================
def compute_false_alarm_rate(contingency, yes_category=2):
    """ Returns the false alarm rate given dichotomous contingency data """
    
    no_category = abs(yes_category - 2) + 1
    
    if len(contingency.forecast_category) > 2:
        raise ValueError('Bias score is defined for dichotomous contingency data only')
        
    false_alarms = contingency.sel(forecast_category=yes_category, 
                                   observed_category=no_category, drop=True)
    correct_negs = contingency.sel(forecast_category=no_category, 
                                   observed_category=no_category, drop=True)

    return (false_alarms / (correct_negs + false_alarms)).rename('false_alarm_rate')


# ===================================================================================================
def compute_success_ratio(contingency, yes_category=2):
    """ Returns the success ratio given dichotomous contingency data """
    
    no_category = abs(yes_category - 2) + 1
    
    if len(contingency.forecast_category) > 2:
        raise ValueError('Bias score is defined for dichotomous contingency data only')
        
    hits = contingency.sel(forecast_category=yes_category, 
                           observed_category=yes_category, drop=True)
    false_alarms = contingency.sel(forecast_category=yes_category, 
                                   observed_category=no_category, drop=True)
    
    return (hits / (hits + false_alarms)).rename('success_ratio')


# ===================================================================================================
def compute_threat_score(contingency, yes_category=2):
    """ Returns the threat score given dichotomous contingency data """
    
    no_category = abs(yes_category - 2) + 1
    
    if len(contingency.forecast_category) > 2:
        raise ValueError('Bias score is defined for dichotomous contingency data only')
        
    hits = contingency.sel(forecast_category=yes_category, 
                           observed_category=yes_category, drop=True)
    misses = contingency.sel(forecast_category=no_category, 
                             observed_category=yes_category, drop=True)
    false_alarms = contingency.sel(forecast_category=yes_category, 
                                   observed_category=no_category, drop=True)
    
    return (hits / (hits + misses + false_alarms)).rename('threat_score')


# ===================================================================================================
def compute_equit_threat_score(contingency, yes_category=2):
    """ Returns the equitable threat score given dichotomous contingency data """
    
    no_category = abs(yes_category - 2) + 1
    
    if len(contingency.forecast_category) > 2:
        raise ValueError('Bias score is defined for dichotomous contingency data only')
        
    hits = contingency.sel(forecast_category=yes_category, 
                           observed_category=yes_category, drop=True)
    misses = contingency.sel(forecast_category=no_category, 
                             observed_category=yes_category, drop=True)
    false_alarms = contingency.sel(forecast_category=yes_category, 
                                   observed_category=no_category, drop=True)
    hits_random = ((hits + misses) * (hits + false_alarms)) / sum_contingency(contingency, 'total')
    
    return ((hits - hits_random) / (hits + misses + false_alarms + hits_random)).rename('equit_threat_score')


# ===================================================================================================
def compute_odds_ratio(contingency, yes_category=2):
    """ Returns the odds ratio given dichotomous contingency data """

    no_category = abs(yes_category - 2) + 1
    
    if len(contingency.forecast_category) > 2:
        raise ValueError('Bias score is defined for dichotomous contingency data only')
        
    hits = contingency.sel(forecast_category=yes_category, 
                           observed_category=yes_category, drop=True)
    correct_negs = contingency.sel(forecast_category=no_category, 
                                   observed_category=no_category, drop=True)
    misses = contingency.sel(forecast_category=no_category, 
                             observed_category=yes_category, drop=True)
    false_alarms = contingency.sel(forecast_category=yes_category, 
                                   observed_category=no_category, drop=True)
    
    return ((hits * correct_negs) / (misses * false_alarms)).rename('odds_ratio')


# ===================================================================================================
def compute_odds_ratio_skill(contingency, yes_category=2):
    """ Returns the odds ratio skill score given dichotomous contingency data """
    
    no_category = abs(yes_category - 2) + 1
    
    if len(contingency.forecast_category) > 2:
        raise ValueError('Bias score is defined for dichotomous contingency data only')
        
    hits = contingency.sel(forecast_category=yes_category, 
                           observed_category=yes_category, drop=True)
    correct_negs = contingency.sel(forecast_category=no_category, 
                                   observed_category=no_category, drop=True)
    misses = contingency.sel(forecast_category=no_category, 
                             observed_category=yes_category, drop=True)
    false_alarms = contingency.sel(forecast_category=yes_category, 
                                   observed_category=no_category, drop=True)
    
    return ((hits * correct_negs - misses * false_alarms) / \
            (hits * correct_negs + misses * false_alarms)).rename('odds_ratio_skill')


# ===================================================================================================
# Methods for continuous variables
# ===================================================================================================
def compute_mean_additive_bias(fcst, obsv, indep_dims, ensemble_dim=None):
    """ Returns the additive bias between forecast and observation datasets """
    
    if isinstance(indep_dims, str):
        indep_dims = [indep_dims]

    if ensemble_dim == None:
        mean_additive_bias = fcst.to_dataset(name='mean_additive_bias') \
                                 .apply(utils.calc_difference, obsv=obsv) \
                                 .mean(dim=indep_dims)['mean_additive_bias']
    else:
        fcst_mean_dims = tuple(indep_dims) + tuple([ensemble_dim])
        mean_additive_bias = fcst.groupby(ensemble_dim) \
                                 .apply(utils.calc_difference, obsv=obsv) \
                                 .mean(dim=fcst_mean_dims) \
                                 .rename('mean_additive_bias')
    return mean_additive_bias


# ===================================================================================================
def compute_mean_multiplicative_bias(fcst, obsv, indep_dims, ensemble_dim=None):
    """ Returns the multiplicative bias between forecast and observation datasets """
    
    if isinstance(indep_dims, str):
        indep_dims = [indep_dims]

    if ensemble_dim == None:
        fcst_mean_dims = indep_dims
    else:
        fcst_mean_dims = tuple(indep_dims) + tuple([ensemble_dim])

    return (fcst.mean(dim=fcst_mean_dims) / obsv.mean(dim=indep_dims)) \
           .rename('mean_multiplicative_bias')

    
# ===================================================================================================
def compute_mean_absolute_error(fcst, obsv, indep_dims, ensemble_dim=None):
    """ Returns the mean absolute error between forecast and observation datasets """
    
    if isinstance(indep_dims, str):
        indep_dims = [indep_dims]

    if ensemble_dim == None:
        mean_absolute_error = ((fcst.to_dataset(name='mean_absolute_error') \
                                    .apply(utils.calc_difference, obsv=obsv) \
                                    ** 2) ** 0.5) \
                                    .mean(dim=indep_dims)['mean_absolute_error']
    else:
        fcst_mean_dims = tuple(indep_dims) + tuple([ensemble_dim])
        mean_absolute_error = ((fcst.groupby(ensemble_dim) \
                                    .apply(utils.calc_difference, obsv=obsv) \
                                    ** 2) ** 0.5) \
                                    .mean(dim=fcst_mean_dims) \
                                    .rename('mean_absolute_error')
    
    return mean_absolute_error


# ===================================================================================================
def compute_mean_squared_error(fcst, obsv, indep_dims, ensemble_dim=None):
    """ Returns the mean sqaured error between forecast and observation datasets """
    
    if isinstance(indep_dims, str):
        indep_dims = [indep_dims]

    if ensemble_dim == None:
        mean_squared_error = (fcst.to_dataset(name='mean_squared_error') \
                                  .apply(utils.calc_difference, obsv=obsv) \
                                  ** 2) \
                                  .mean(dim=indep_dims)['mean_squared_error']
    else:
        fcst_mean_dims = tuple(indep_dims) + tuple([ensemble_dim])
        mean_squared_error = (fcst.groupby(ensemble_dim) \
                                  .apply(utils.calc_difference, obsv=obsv) \
                                  ** 2) \
                                  .mean(dim=fcst_mean_dims) \
                                  .rename('mean_squared_error')

    return mean_squared_error


# ===================================================================================================
def compute_rms_error(fcst, obsv, indep_dims, ensemble_dim=None):
    """ Returns the mean sqaured error between forecast and observation datasets """
    
    return ((compute_mean_squared_error(fcst, obsv, indep_dims=indep_dims, ensemble_dim=ensemble_dim)) ** 0.5) \
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
    
    return da_logical.mean(dim=ensemble_dim).rename('likelihood')


# ===================================================================================================
def sum_contingency(contingency, category='total'):
    """ Returns sums of specified categories in contingency table """
    
    if category == 'total':
        N = contingency.sum(dim=('observed_category','forecast_category'))
    elif category == 'observed':
        N = contingency.sum(dim='forecast_category').rename({'observed_category' : 'category'})
    elif category == 'forecast':
        N = contingency.sum(dim='observed_category').rename({'forecast_category' : 'category'})    
    else: raise ValueError(f'"{category}" is not a recognised category')
        
    return N



    