"""
    Tests for functions in pyLatte utils module
    Author: Dougie Squire
    Date created: 05/04/2018
    Python Version: 3.6
"""

# ===================================================================================================
# Packages
# ===================================================================================================
from pylatte import utils
import numpy as np
import xarray as xr


# ===================================================================================================
# Probability tools
# ===================================================================================================
def test_categorize():
    tile = np.linspace(1,100,100)
    da_data = np.tile(tile,(100,1))
    da = xr.DataArray(da_data, coords =[tile, tile], dims=['dim1','dim2'])

    bin_edges = utils.get_bin_edges(tile)
    assert np.all(utils.categorize(da,bin_edges) == da)
    
# ===================================================================================================
def test_compute_pdf():
    tile = np.linspace(1,100,100)
    da_data = np.tile(tile,(100,1))
    da = xr.DataArray(da_data, coords =[tile, tile], dims=['dim1','dim2'])

    bin_edges = utils.get_bin_edges(tile)

    pdf1 = utils.compute_pdf(da, bin_edges, dim='dim1')
    assert np.all(pdf1.values == np.eye(100))

    pdf2 = utils.compute_pdf(da, bin_edges, dim='dim2')
    assert np.all(pdf2.values == np.ones(100)/100)
    
# ===================================================================================================
def test_compute_cdf():
    tile = np.linspace(1,100,100)
    da_data = np.tile(tile,(100,1))
    da = xr.DataArray(da_data, coords =[tile, tile], dims=['dim1','dim2'])

    bin_edges = utils.get_bin_edges(tile)

    cdf1 = utils.compute_cdf(da, bin_edges, dim='dim1')
    assert np.all(cdf1 == np.tril(np.ones((100,100)),k=0))

