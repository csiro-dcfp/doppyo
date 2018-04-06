"""
    pyLatte functions for computing indices
    Author: Dougie Squire
    Date created: 04/04/2018
    Python Version: 3.6
"""

__all__ = ['compute_nino3', 'compute_nino34', 'compute_nino4', 'compute_emi', 'compute_dmi']


# ===================================================================================================
# Packages
# ===================================================================================================
from pylatte import utils


# ===================================================================================================
def compute_nino3(da_sstanom,convert2monthly=False):
    ''' Returns nino3 index '''  
    
    # Convert to monthly average sst if more frequent -----
    if convert2monthly:
        da_sstanom = da_sstanom.resample(freq='1MS', dim='time', how='mean')
    
    box = (-5.0,5.0,360.0-150.0,360.0-90.0) # (lat_min,lat_max,lon_min,lon_max)
    
    da_nino3 = utils.calc_boxavg_latlon(da_sstanom,box)
    
    return da_nino3


# ===================================================================================================
def compute_nino34(da_sstanom,convert2monthly=False):
    ''' Returns nino3.4 index '''  
    
    # Convert to monthly average sst if more frequent -----
    if convert2monthly:
        da_sstanom = da_sstanom.resample(freq='1MS', dim='time', how='mean')
    
    box = (-5.0,5.0,360.0-170.0,360.0-120.0) # (lat_min,lat_max,lon_min,lon_max)
    
    da_nino34 = utils.calc_boxavg_latlon(da_sstanom,box)
    
    return da_nino34


# ===================================================================================================
def compute_nino4(da_sstanom,convert2monthly=False):
    ''' Returns nino4 index '''  
    
    # Convert to monthly average sst if more frequent -----
    if convert2monthly:
        da_sstanom = da_sstanom.resample(freq='1MS', dim='time', how='mean')
    
    box = (-5.0,5.0,360.0-160.0,360.0-150.0) # (lat_min,lat_max,lon_min,lon_max)
    
    da_nino4 = utils.calc_boxavg_latlon(da_sstanom,box)
    
    return da_nino4


# ===================================================================================================
def compute_emi(da_sstanom,convert2monthly=False):
    ''' Returns EMI index ''' 
    
    # Convert to monthly average sst if more frequent -----
    if convert2monthly:
        da_sstanom = da_sstanom.resample(freq='1MS', dim='time', how='mean')
    
    boxA = (-10.0,10.0,360.0-165.0,360.0-140.0) # (lat_min,lat_max,lon_min,lon_max)
    boxB = (-15.0,5.0,360.0-110.0,360.0-70.0) # (lat_min,lat_max,lon_min,lon_max)
    boxC = (-10.0,20.0,125.0,145.0) # (lat_min,lat_max,lon_min,lon_max)
    
    da_sstA = utils.calc_boxavg_latlon(da_sstanom,boxA)
    da_sstB = utils.calc_boxavg_latlon(da_sstanom,boxB)
    da_sstC = utils.calc_boxavg_latlon(da_sstanom,boxC)
    da_emi = da_sstA - 0.5*da_sstB - 0.5*da_sstC
    
    return da_emi

# ===================================================================================================
def compute_dmi(da_sstanom,convert2monthly=False):
    ''' Returns DMI index ''' 
    
    # Convert to monthly average sst if more frequent -----
    if convert2monthly:
        da_sstanom = da_sstanom.resample(freq='1MS', dim='time', how='mean')
    
    boxW = (-10.0,10.0,50.0,70.0) # (lat_min,lat_max,lon_min,lon_max)
    boxE = (-10.0,0.0,90.0,110.0) # (lat_min,lat_max,lon_min,lon_max)
    
    da_W = utils.calc_boxavg_latlon(da_sstanom,boxW)
    da_E = utils.calc_boxavg_latlon(da_sstanom,boxE)
    da_dmi = da_W - da_E
    
    return da_dmi

# ===================================================================================================