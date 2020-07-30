from doppyo.skill import Heidke_score, accuracy_score, Heidke_score, bias_score, Peirce_score,Gerrity_score, rank_histogram, Brier_score, contingency,threat_score,odds_ratio_skill_score, hit_rate,false_alarm_rate, false_alarm_ratio,discrimination, success_ratio, equit_threat_score,odds_ratio, rps, mean_multiplicative_bias,Pearson_corrcoeff, roc, reliability, mean_absolute_error,mean_squared_error,rms_error
import numpy as np
import xarray as xr
import pytest

# missing

over_dims=[['x','y'],'x','y',None]

skill_ct_many_cat=[accuracy_score,Heidke_score, Peirce_score,Gerrity_score]

skill_ct_2_cat=[false_alarm_ratio,
equit_threat_score,odds_ratio,odds_ratio_skill_score, success_ratio,false_alarm_rate,threat_score, hit_rate,bias_score]

distance_metrics = [mean_multiplicative_bias,rms_error, Pearson_corrcoeff,mean_absolute_error, mean_squared_error]

probabilistic_metrics = [rank_histogram, Brier_score, rps, roc, discrimination, reliability]



@pytest.fixture
def da_cmp():
    return xr.DataArray(np.random.normal(size=(3,3)),coords=[('x', np.arange(3)), ('y', np.arange(3))])

@pytest.fixture
def ds_cmp(da_cmp):
    return da_cmp.to_dataset(name='var')

@pytest.fixture
def da_cmp_ensemble():
    return xr.DataArray(np.random.normal(size=(3,3,3)),coords=[('x', np.arange(3)), ('y', np.arange(3)), ('ensemble', np.arange(3))])

@pytest.fixture
def ds_cmp_ensemble(da_cmp_ensemble):
    return da_cmp_ensemble.to_dataset(name='var')

@pytest.fixture
def da_ref():
    return xr.DataArray(np.random.normal(size=(3,3)),coords=[('x', np.arange(3)), ('y', np.arange(3))])

@pytest.fixture
def ds_ref(da_ref):
    return da_ref.to_dataset(name='var')


@pytest.fixture
def category_4_edges_ref():
    return np.linspace(-2,2,5)

@pytest.fixture
def category_4_edges_cmp():
    return np.linspace(-2,2,5)

@pytest.fixture
def category_2_edges_ref():
    return np.linspace(-2,2,3)

@pytest.fixture
def category_2_edges_cmp():
    return np.linspace(-2,2,3)

@pytest.fixture
def bins():
    return np.linspace(-2,2,10)

@pytest.fixture
def probability_bin_edges():
    return np.linspace(0,1,5)

@pytest.mark.parametrize('over_dims', over_dims)
@pytest.mark.parametrize('skill', skill_ct_many_cat)
def test_4_contingency_skill_da(da_cmp, da_ref, category_4_edges_cmp, category_4_edges_ref, skill, over_dims):
    ct = contingency(da_cmp, da_ref, category_4_edges_cmp, category_4_edges_ref, over_dims)
    skill(ct)

@pytest.mark.parametrize('over_dims', over_dims)
@pytest.mark.parametrize('skill', skill_ct_many_cat)
def test_4_contingency_skill_ds(ds_cmp, ds_ref, category_4_edges_cmp, category_4_edges_ref, skill, over_dims):
    ct = contingency(ds_cmp, ds_ref, category_4_edges_cmp, category_4_edges_ref, over_dims)
    skill(ct)


@pytest.mark.parametrize('over_dims', over_dims)
@pytest.mark.parametrize('skill', skill_ct_2_cat)
def test_2_contingency_skill_da(da_cmp, da_ref, category_2_edges_cmp, category_2_edges_ref, skill, over_dims):
    ct = contingency(da_cmp, da_ref, category_2_edges_cmp, category_2_edges_ref, over_dims)
    skill(ct)

@pytest.mark.parametrize('over_dims', over_dims)
@pytest.mark.parametrize('skill', skill_ct_many_cat)
def test_2_contingency_skill_ds(ds_cmp, ds_ref, category_2_edges_cmp, category_2_edges_ref, skill, over_dims):
    ct = contingency(ds_cmp, ds_ref, category_2_edges_cmp, category_2_edges_ref, over_dims)
    skill(ct)


@pytest.mark.parametrize('over_dims', ['x','y',['x','y']])
@pytest.mark.parametrize('skill', probabilistic_metrics)
def test_probabilistic_skill_da(da_cmp_ensemble, da_ref, skill, bins, probability_bin_edges, over_dims):
    skill_name = skill.__name__
    if skill_name in ['Brier_score','roc','discrimination','reliability']:
        da_cmp_ensemble = (da_cmp_ensemble > .1).mean('ensemble')
        da_ref = da_ref > 0.1
    if skill_name in ['rps']:
        args = bins
    if skill_name in ['roc','discrimination']:
        args = probability_bin_edges

    if skill_name in ['rps']:
        skill(da_cmp_ensemble, da_ref, args, over_dims)
    elif skill_name in ['roc','discrimination']:
        skill(da_cmp_ensemble, da_ref, over_dims, args)
    else:
        skill(da_cmp_ensemble, da_ref, over_dims)

@pytest.mark.parametrize('over_dims', ['x','y',['x','y'],None])
@pytest.mark.parametrize('skill', probabilistic_metrics)
def test_probabilistic_skill_ds(skill, ds_cmp_ensemble, ds_ref, bins, probability_bin_edges, over_dims):
    skill_name = skill.__name__
    if skill_name in ['Brier_score', 'roc','discrimination','reliability']:
        ds_cmp_ensemble = (ds_cmp_ensemble > .1).mean('ensemble')
        ds_ref = ds_ref > 0.1
    if skill_name in ['rps']:
        args = bins
    elif skill_name in ['roc','discrimination']:
        args = probability_bin_edges

    if skill_name in ['rps']:
        skill(ds_cmp_ensemble, ds_ref, args, over_dims)
    elif skill_name in ['roc','discrimination']:
        skill(ds_cmp_ensemble, ds_ref, over_dims, args)
    elif skill_name in ['Brier_score']:
        skill(ds_cmp_ensemble, ds_ref, over_dims)


@pytest.mark.parametrize('over_dims', [['x','y'],'x','y'])
@pytest.mark.parametrize('skill', distance_metrics)
def test_distance_metrics_da(da_cmp, da_ref, skill, over_dims):
    skill(da_cmp, da_ref, over_dims)


@pytest.mark.parametrize('over_dims', [['x','y'],'x','y'])
@pytest.mark.parametrize('skill', distance_metrics)
def test_distance_metrics_ds(ds_cmp, ds_ref, skill, over_dims):
    skill(ds_cmp, ds_ref, over_dims)
