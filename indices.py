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
def compute_nino3(da_sst_anom):
    ''' Returns nino3 index '''  
    
    box = (-5.0,5.0,360.0-150.0,360.0-90.0) # (lat_min,lat_max,lon_min,lon_max)
    
    da_nino3 = utils.calc_boxavg_latlon(da_sst_anom,box)
    
    return da_nino3


# ===================================================================================================
def compute_nino34(da_sst_anom):
    ''' Returns nino3.4 index '''  
    
    box = (-5.0,5.0,360.0-170.0,360.0-120.0) # (lat_min,lat_max,lon_min,lon_max)
    
    da_nino34 = utils.calc_boxavg_latlon(da_sst_anom,box)
    
    return da_nino34


# ===================================================================================================
def compute_nino4(da_sst_anom):
    ''' Returns nino4 index '''  
    
    box = (-5.0,5.0,360.0-160.0,360.0-150.0) # (lat_min,lat_max,lon_min,lon_max)
    
    da_nino4 = utils.calc_boxavg_latlon(da_sst_anom,box)
    
    return da_nino4


# ===================================================================================================
def compute_emi(da_sst_anom):
    ''' Returns EMI index ''' 
    
    boxA = (-10.0,10.0,360.0-165.0,360.0-140.0) # (lat_min,lat_max,lon_min,lon_max)
    boxB = (-15.0,5.0,360.0-110.0,360.0-70.0) # (lat_min,lat_max,lon_min,lon_max)
    boxC = (-10.0,20.0,125.0,145.0) # (lat_min,lat_max,lon_min,lon_max)
    
    da_sstA = utils.calc_boxavg_latlon(da_sst_anom,boxA)
    da_sstB = utils.calc_boxavg_latlon(da_sst_anom,boxB)
    da_sstC = utils.calc_boxavg_latlon(da_sst_anom,boxC)
    da_emi = da_sstA - 0.5*da_sstB - 0.5*da_sstC
    
    return da_emi


# ===================================================================================================
def compute_dmi(da_sst_anom):
    ''' Returns DMI index ''' 
    
    boxW = (-10.0,10.0,50.0,70.0) # (lat_min,lat_max,lon_min,lon_max)
    boxE = (-10.0,0.0,90.0,110.0) # (lat_min,lat_max,lon_min,lon_max)
    
    da_W = utils.calc_boxavg_latlon(da_sst_anom,boxW)
    da_E = utils.calc_boxavg_latlon(da_sst_anom,boxE)
    da_dmi = da_W - da_E
    
    return da_dmi


# ===================================================================================================
def compute_soi(da_slp_anom):
    ''' Returns SOI index as defined by NOAA '''  
    
    lat_Tahiti = 17.6509
    lon_Tahiti = 149.4260

    lat_Darwin = 12.4634
    lon_Darwin = 130.8456

    da_Tahiti_anom = utils.get_nearest_point(da_slp_anom, lat_Tahiti, lon_Tahiti)
    da_Tahiti_std = da_Tahiti_anom.std(dim='lead_time')
    da_Tahiti_stdzd = da_Tahiti_anom / da_Tahiti_std

    da_Darwin_anom = utils.get_nearest_point(da_slp_anom, lat_Darwin, lon_Darwin)
    da_Darwin_std = da_Darwin_anom.std(dim='lead_time')
    da_Darwin_stdzd = da_Darwin_anom / da_Darwin_std

    MSD = (da_Tahiti_stdzd - da_Darwin_stdzd).std(dim='lead_time')

    return (da_Tahiti_stdzd - da_Darwin_stdzd) / MSD


# ===================================================================================================