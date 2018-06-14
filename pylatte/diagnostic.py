"""
    pyLatte functions for computing various climate diagnostics
    Author: Dougie Squire
    Date created: 04/04/2018
    Python Version: 3.6
"""

__all__ = ['compute_velocitypotential', 'compute_streamfunction', 'compute_rws', 'compute_divergent', 
           'compute_waf', 'compute_BruntVaisala', 'compute_ks2', 'compute_Eady', 'compute_nino3', 
           'compute_nino34', 'compute_nino4', 'compute_emi', 'compute_dmi']

# ===================================================================================================
# Packages
# ===================================================================================================
import xarray as xr
import windspharm as wsh

# Load pyLatte packages -----
from pylatte import utils

# ===================================================================================================
# Flow field diagnostics
# ===================================================================================================
def compute_velocitypotential(u, v):
    """ 
        Returns the velocity potential given fields of u and v.
        
        u and v must have at least latitude and longitude dimensions with standard naming
    """
    
    lat_name = utils.get_lat_name(u)
    lon_name = utils.get_lon_name(u)

    # The windspharm package is iffy with arrays larger than 3D. Stack accordingly -----
    if not u.coords.to_dataset().equals(v.coords.to_dataset()):
        raise ValueError('u and v coordinates do not match')
    stacked = False
    stack_dims = utils.find_other_dims(u, [lat_name, lon_name])
    if stack_dims is not None:
        u_stack = u.stack(stacked=stack_dims)
        v_stack = v.stack(stacked=stack_dims)
        stacked = True
    else:
        u_stack = u
        v_stack = v
        
    # Create a VectorWind instance -----
    w = wsh.xarray.VectorWind(u_stack, v_stack)

    # Compute the velocity potential -----
    phi = w.velocitypotential()
    
    # Unstack the stacked dimensions -----
    if stacked:
        phi = phi.unstack('stacked')
        
    return phi


# ===================================================================================================
def compute_streamfunction(u, v):
    """ 
        Returns the streamfunction given fields of u and v.
        
        u and v must have at least latitude and longitude dimensions with standard naming
    """

    lat_name = utils.get_lat_name(u)
    lon_name = utils.get_lon_name(u)

    # The windspharm package is iffy with arrays larger than 3D. Stack accordingly -----
    if not u.coords.to_dataset().equals(v.coords.to_dataset()):
        raise ValueError('u and v coordinates do not match')
    stacked = False
    stack_dims = utils.find_other_dims(u, [lat_name, lon_name])
    if stack_dims is not None:
        u_stack = u.stack(stacked=stack_dims)
        v_stack = v.stack(stacked=stack_dims)
        stacked = True
    else:
        u_stack = u
        v_stack = v
        
    # Create a VectorWind instance -----
    w = wsh.xarray.VectorWind(u_stack, v_stack)

    # Compute the streamfunction -----
    psi = w.streamfunction()
    
    # Unstack the stacked dimensions -----
    if stacked:
        psi = psi.unstack('stacked')
        
    return psi


# ===================================================================================================
def compute_rws(u, v):
    """ 
        Returns the Rossby wave source given fields of u and v.
        
        u and v must have at least latitude and longitude dimensions with standard naming
    """

    lat_name = utils.get_lat_name(u)
    lon_name = utils.get_lon_name(u)

    # The windspharm package is iffy with arrays larger than 3D. Stack accordingly -----
    if not u.coords.to_dataset().equals(v.coords.to_dataset()):
        raise ValueError('u and v coordinates do not match')
    stacked = False
    stack_dims = utils.find_other_dims(u, [lat_name, lon_name])
    if stack_dims is not None:
        u_stack = u.stack(stacked=stack_dims)
        v_stack = v.stack(stacked=stack_dims)
        stacked = True
    else:
        u_stack = u
        v_stack = v
        
    # Create a VectorWind instance -----
    w = wsh.xarray.VectorWind(u_stack, v_stack)

    # Compute components of Rossby wave source -----
    eta = w.absolutevorticity() # absolute vorticity
    div = w.divergence() # divergence
    uchi, vchi = w.irrotationalcomponent() # irrotational (divergent) wind components
    etax, etay = w.gradient(eta) # gradients of absolute vorticity

    # Combine the components to form the Rossby wave source term -----
    rws = 1e11 * (-eta * div - (uchi * etax + vchi * etay)).rename('rws')
    rws.attrs['units'] = '1e-11/s^2'
    rws.attrs['long_name'] = 'Rossby wave source'

    # Unstack the stacked dimensions -----
    if stacked:
        rws = rws.unstack('stacked')
    
    return rws


# ===================================================================================================
def compute_divergent(u, v):
    """ 
        Returns the irrotational (divergent) component of u and v.
        
        u and v must have at least latitude and longitude dimensions with standard naming
    """

    lat_name = utils.get_lat_name(u)
    lon_name = utils.get_lon_name(u)

    # The windspharm package is iffy with arrays larger than 3D. Stack accordingly -----
    if not u.coords.to_dataset().equals(v.coords.to_dataset()):
        raise ValueError('u and v coordinates do not match')
    stacked = False
    stack_dims = utils.find_other_dims(u, [lat_name, lon_name])
    if stack_dims is not None:
        u_stack = u.stack(stacked=stack_dims)
        v_stack = v.stack(stacked=stack_dims)
        stacked = True
    else:
        u_stack = u
        v_stack = v
            
    # Create a VectorWind instance -----
    w = wsh.xarray.VectorWind(u_stack, v_stack)

    # Compute the irrotational components -----
    uchi, vchi = w.irrotationalcomponent()
    
    # Unstack the stacked dimensions -----
    if stacked:
        uchi = uchi.unstack('stacked')
        vchi = vchi.unstack('stacked')
        
    return uchi, vchi


# ===================================================================================================
def compute_waf(psi_anom, u, v, p_lev=None):
    """ 
        Returns the stationary component of the wave activity flux, following,
            Takaya and Nakamura, 2001, A Formulation of a Phase-Independent Wave-Activity Flux 
            for Stationary and Migratory Quasigeostrophic Eddies on a Zonally Varying Basic Flow,
        using zonal and meridional velocity fields on one or more isobaric surface(s).
        
        psi_anom, u and v must have at least latitude and longitude dimensions with standard naming
        Pressure level(s) [hPa] are extracted from the psi_anom/u/v coordinate, or, in the absence 
        of such a coordinate, are provided by p_lev, The pressure coordinate, when supplied, must 
        use standard naming conventions, 'pfull' or 'phalf'
    """
    
    lat_name = utils.get_lat_name(u)
    lon_name = utils.get_lon_name(u)
    pres_name = utils.get_pres_name(u)

    # The windspharm package is iffy with arrays larger than 3D. Stack accordingly -----
    if not (u.coords.to_dataset().equals(v.coords.to_dataset()) & \
            u.coords.to_dataset().equals(psi_anom.coords.to_dataset())):
        raise ValueError('psi_anom, u and v coordinates must match')
    stacked = False
    stack_dims = utils.find_other_dims(u, [lat_name, lon_name])
    if stack_dims is not None:
        u_stack = u.stack(stacked=stack_dims)
        v_stack = v.stack(stacked=stack_dims)
        psi_stack = psi_anom.stack(stacked=stack_dims)
        stacked = True
        try:
            pres = u_stack['stacked'][pres_name] / 1000 # Takaya and Nakmura p.610
        except KeyError:
            if p_lev is None:
                raise TypeError('Cannot determine pressure level(s) of provided data. This should' +
                                'be stored as a coordinate, "pfull" or "phalf" in the provided' +
                                'objects. Alternatively, for single level computations, the' + 
                                'pressure can be provided using the p_lev argument')
            else:
                pres = p_lev / 1000 # Takaya and Nakmura p.610
    else:
        u_stack = u
        v_stack = v
        psi_stack = psi_anom
        if p_lev is None:
            raise TypeError('Cannot determine pressure level(s) of provided data. This should' +
                            'be stored as a coordinate, "pfull" or "phalf" in the provided' +
                            'objects. Alternatively, for single level computations, the' + 
                            'pressure can be provided using the p_lev argument')
        else:
            pres = p_lev / 1000 # Takaya and Nakmura p.610

    # Create a VectorWind instance (use gradient function) -----
    w = wsh.xarray.VectorWind(u_stack, v_stack)
    
    # Compute the various streamfunction gradients required -----
    psi_x, psi_y = w.gradient(psi_stack)
    psi_xx, psi_xy = w.gradient(psi_x)
    psi_yx, psi_yy = w.gradient(psi_y)
    
    # Compute the wave activity flux -----
    vel = (u_stack * u_stack + v_stack * v_stack) ** 0.5
    uwaf = (0.5 * pres * (u_stack * (psi_x * psi_x - psi_stack * psi_xx) + 
                          v_stack * (psi_x * psi_y - 0.5 * psi_stack * (psi_xy + psi_yx))) / vel).rename('uwaf')
    uwaf.attrs['units'] = 'm^2/s^2'
    uwaf.attrs['long_name'] = 'Zonal Rossby wave activity flux'
    vwaf = (0.5 * pres * (v_stack * (psi_y * psi_y - psi_stack * psi_yy) + 
                          u_stack * (psi_x * psi_y - 0.5 * psi_stack * (psi_xy + psi_yx))) / vel).rename('vwaf')
    vwaf.attrs['units'] = 'm^2/s^2'
    vwaf.attrs['long_name'] = 'Meridional Rossby wave activity flux'
    
    # Unstack the stacked dimensions -----
    if stacked:
        uwaf = uwaf.unstack('stacked')
        vwaf = vwaf.unstack('stacked')
    
    return uwaf, vwaf


# ===================================================================================================
def compute_BruntVaisala(temp):
    """
        Returns the Brunt Väisälä frequency
        
        temp must be saved on pressure levels
    """

    R = utils.constants().R_d
    Cp = utils.constants().C_pd
    g = utils.constants().g

    p_name = utils.get_pres_name(temp)
    dTdp = utils.calc_gradient(temp, p_name)
    pdR = temp[p_name] / R

    nsq = ((-dTdp * pdR + (temp / Cp)) / (temp / g) ** 2).rename('nsq')
    nsq.attrs['long_name']='Brunt-Vaisala frequency squared'
    nsq.attrs['units']='s^-2'
    
    return nsq


# ===================================================================================================
def compute_ks2(u, v, u_clim):
    """ 
        Returns the square of the stationary Rossby wave number, Ks**2
        
        u, v and u_clim must have at least latitude and longitude dimensions with standard naming
    """

    lat_name = utils.get_lat_name(u)
    lon_name = utils.get_lon_name(u)

    # The windspharm package is iffy with arrays larger than 3D. Stack accordingly -----
    if not u.coords.to_dataset().equals(v.coords.to_dataset()):
        raise ValueError('u and v coordinates do not match')
    stacked = False
    stack_dims = utils.find_other_dims(u, [lat_name, lon_name])
    if stack_dims is not None:
        u_stack = u.stack(stacked=stack_dims)
        uc_stack = u_clim.stack(stacked=stack_dims)
        v_stack = v.stack(stacked=stack_dims)
        stacked = True
    else:
        u_stack = u
        uc_stack = u_clim
        v_stack = v

    # Create a VectorWind instance -----
    w = wsh.xarray.VectorWind(u_stack, v_stack)

    # Compute the absolute vorticity gradient -----
    etau, etav = w.gradient(w.absolutevorticity())
    
    # Compute the stationary wave number -----
    ks2 = (xr.ufuncs.cos(etav[lat_name] / 180 * utils.constants().pi)**2 * \
                    (etav * utils.constants().R_earth ** 2) / uc_stack).rename('ks2')
    
    # Unstack the stacked dimensions -----
    if stacked:
        ks2 = ks2.unstack('stacked')
    ks2.attrs['units'] = 'real number'
    ks2.attrs['long_name'] = 'Square of Rossby stationary wavenumber'
        
    return ks2


# ===================================================================================================
def compute_Eady(u, v, gh, nsq):
    """ 
        Returns the square of the Eady growth rate
        
        u, v, gh and nsq must have at least latitude and longitude dimensions with 
        standard naming
        Data must be saved on pressure levels
    """
    
    lat_name = utils.get_lat_name(u)
    lon_name = utils.get_lon_name(u)
    
    f = utils.constants().O * xr.ufuncs.sin(gh[lat_name] / 180 * utils.constants().pi)
    eady2 = ((utils.constants().Ce * f) * (utils.calc_gradient((u ** 2 + v ** 2) ** 0.5, 'pfull') / \
            utils.calc_gradient(gh, 'pfull'))) ** 2 / nsq
    eady2.attrs['units'] = 's^-2'
    eady2.attrs['long_name'] = 'Square of Eady growth rate'
    
    return eady2


# ===================================================================================================
# Indices
# ===================================================================================================
def compute_nino3(da_sst_anom):
    ''' Returns nino3 index '''  
    
    box = (-5.0,5.0,360.0-150.0,360.0-90.0) # (lat_min,lat_max,lon_min,lon_max)
    
    return utils.calc_boxavg_latlon(da_sst_anom,box)


# ===================================================================================================
def compute_nino34(da_sst_anom):
    ''' Returns nino3.4 index '''  
    
    box = (-5.0,5.0,360.0-170.0,360.0-120.0) # (lat_min,lat_max,lon_min,lon_max) 
    
    return utils.calc_boxavg_latlon(da_sst_anom,box)


# ===================================================================================================
def compute_nino4(da_sst_anom):
    ''' Returns nino4 index '''  
    
    box = (-5.0,5.0,360.0-160.0,360.0-150.0) # (lat_min,lat_max,lon_min,lon_max)
    
    return utils.calc_boxavg_latlon(da_sst_anom,box)


# ===================================================================================================
def compute_emi(da_sst_anom):
    ''' Returns EMI index ''' 
    
    boxA = (-10.0,10.0,360.0-165.0,360.0-140.0) # (lat_min,lat_max,lon_min,lon_max)
    boxB = (-15.0,5.0,360.0-110.0,360.0-70.0) # (lat_min,lat_max,lon_min,lon_max)
    boxC = (-10.0,20.0,125.0,145.0) # (lat_min,lat_max,lon_min,lon_max)
    
    da_sstA = utils.calc_boxavg_latlon(da_sst_anom,boxA)
    da_sstB = utils.calc_boxavg_latlon(da_sst_anom,boxB)
    da_sstC = utils.calc_boxavg_latlon(da_sst_anom,boxC)
    
    return da_sstA - 0.5*da_sstB - 0.5*da_sstC


# ===================================================================================================
def compute_dmi(da_sst_anom):
    ''' Returns DMI index ''' 
    
    boxW = (-10.0,10.0,50.0,70.0) # (lat_min,lat_max,lon_min,lon_max)
    boxE = (-10.0,0.0,90.0,110.0) # (lat_min,lat_max,lon_min,lon_max)
    
    da_W = utils.calc_boxavg_latlon(da_sst_anom,boxW)
    da_E = utils.calc_boxavg_latlon(da_sst_anom,boxE)
    
    return da_W - da_E


# ===================================================================================================
def compute_soi(da_slp_anom, std_dim, n_rolling=None):
    ''' Returns SOI index as defined by NOAA '''  
    
    if std_dim is None:
        raise ValueError('The dimension over which to compute the standard deviations must be specified')
    
    lat_Tahiti = 17.6509
    lon_Tahiti = 149.4260

    lat_Darwin = 12.4634
    lon_Darwin = 130.8456

    da_Tahiti_anom = utils.get_nearest_point(da_slp_anom, lat_Tahiti, lon_Tahiti)
    da_Tahiti_std = da_Tahiti_anom.std(dim=std_dim)
    da_Tahiti_stdzd = da_Tahiti_anom / da_Tahiti_std

    da_Darwin_anom = utils.get_nearest_point(da_slp_anom, lat_Darwin, lon_Darwin)
    da_Darwin_std = da_Darwin_anom.std(dim=std_dim)
    da_Darwin_stdzd = da_Darwin_anom / da_Darwin_std

    MSD = (da_Tahiti_stdzd - da_Darwin_stdzd).std(dim=std_dim)
        
    return (da_Tahiti_stdzd - da_Darwin_stdzd) / MSD


# ===================================================================================================
