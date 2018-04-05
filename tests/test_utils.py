"""
    Tests for functions in pyLatte utils module
    Author: Dougie Squire
    Date created: 05/04/2018
    Python Version: 3.6
"""

# ===================================================================================================
# Packages
# ===================================================================================================
from pyLatte import utils
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