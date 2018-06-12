"""
    pyLatte functions for computing various climate diagnostics
    Author: Dougie Squire
    Date created: 04/04/2018
    Python Version: 3.6
"""

__all__ = ['compute_velocitypotential', 'compute_streamfunction', 'compute_rws', 
           'compute_irrotationalcomponent']

# ===================================================================================================
# Packages
# ===================================================================================================
import xarray as xr
import windspharm as wsh

# Load cafepy packages -----
from pylatte import utils

# ===================================================================================================
# Flow field diagnostics
# ===================================================================================================
def compute_velocitypotential(U, V):
    """ 
        Returns the velocity potential given fields of U and V.
        U and V must have dimensions 'lat' and 'lon', at least 
    """

    # The windspharm package anticipates 2D or 3D arrays. Stack accordingly -----
    if not U.coords.to_dataset().equals(V.coords.to_dataset()):
        raise ValueError('U and V coordinates do not match')
    stacked = False
    stack_dims = utils.find_other_dims(U, ['lat','lon'])
    if stack_dims is not None:
        U_stack = U.stack(stacked=stack_dims)
        V_stack = V.stack(stacked=stack_dims)
        stacked = True
    else:
        U_stack = U
        V_stack = V
        
    # Create a VectorWind instance -----
    w = wsh.xarray.VectorWind(U_stack, V_stack)

    # Compute the velocity potential -----
    phi = w.velocitypotential()
    
    # Unstack the stacked dimensions -----
    if stacked:
        phi = phi.unstack('stacked')
        
    return phi


# ===================================================================================================
def compute_streamfunction(U, V):
    """ 
        Returns the streamfunction given fields of U and V.
        U and V must have dimensions 'lat' and 'lon', at least 
    """

    # The windspharm package anticipates 2D or 3D arrays. Stack accordingly -----
    if not U.coords.to_dataset().equals(V.coords.to_dataset()):
        raise ValueError('U and V coordinates do not match')
    stacked = False
    stack_dims = utils.find_other_dims(U, ['lat','lon'])
    if stack_dims is not None:
        U_stack = U.stack(stacked=stack_dims)
        V_stack = V.stack(stacked=stack_dims)
        stacked = True
    else:
        U_stack = U
        V_stack = V
        
    # Create a VectorWind instance -----
    w = wsh.xarray.VectorWind(U_stack, V_stack)

    # Compute the streamfunction -----
    psi = w.streamfunction()
    
    # Unstack the stacked dimensions -----
    if stacked:
        psi = psi.unstack('stacked')
        
    return psi


# ===================================================================================================
def compute_rws(U, V):
    """ 
        Returns the Rossby wave source given fields of U and V.
        U and V must have dimensions 'lat' and 'lon', at least 
    """

    # The windspharm package anticipates 2D or 3D arrays. Stack accordingly -----
    if not U.coords.to_dataset().equals(V.coords.to_dataset()):
        raise ValueError('U and V coordinates do not match')
    stacked = False
    stack_dims = utils.find_other_dims(U, ['lat','lon'])
    if stack_dims is not None:
        U_stack = U.stack(stacked=stack_dims)
        V_stack = V.stack(stacked=stack_dims)
        stacked = True
    else:
        U_stack = U
        V_stack = V
        
    # Create a VectorWind instance -----
    w = wsh.xarray.VectorWind(U_stack, V_stack)

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
def compute_irrotationalcomponent(U, V):
    """ 
        Returns the irrotational (divergent) component of U and V.
        U and V must have dimensions 'lat' and 'lon', at least 
    """

    # The windspharm package anticipates 2D or 3D arrays. Stack accordingly -----
    if not U.coords.to_dataset().equals(V.coords.to_dataset()):
        raise ValueError('U and V coordinates do not match')
    stacked = False
    stack_dims = utils.find_other_dims(U, ['lat','lon'])
    if stack_dims is not None:
        U_stack = U.stack(stacked=stack_dims)
        V_stack = V.stack(stacked=stack_dims)
        stacked = True
    else:
        U_stack = U
        V_stack = V
            
    # Create a VectorWind instance -----
    w = wsh.xarray.VectorWind(U_stack, V_stack)

    # Compute the irrotational components -----
    Uchi, Vchi = w.irrotationalcomponent()
    
    # Unstack the stacked dimensions -----
    if stacked:
        Uchi = Uchi.unstack('stacked')
        Vchi = Vchi.unstack('stacked')
        
    return Uchi, Vchi


# ===================================================================================================
def compute_waf(psi_anom, U, V, p_lev=None):
    """ 
        Returns the stationary component of the wave activity flux, following,
            Takaya and Nakamura, 2001, A Formulation of a Phase-Independent Wave-Activity Flux 
            for Stationary and Migratory Quasigeostrophic Eddies on a Zonally Varying Basic Flow,
        using zonal and meridional velocity fields on one or more isobaric surface(s).
        
        psi_anom, U and V must have matching dimensions and include 'lat' and 'lon', at least
        Pressure level(s) [hPa] are extracted from the psi_anom/U/V coordinate, or, in the absence 
        of such a coordinate, are provided by p_lev, The pressure coordinate, when supplied, must 
        use standard naming conventions, 'pfull' or 'phalf'
    """
    
    # The windspharm package anticipates 2D or 3D arrays. Stack accordingly -----
    if not (U.coords.to_dataset().equals(V.coords.to_dataset()) & \
            U.coords.to_dataset().equals(psi_anom.coords.to_dataset())):
        raise ValueError('psi_anom, U and V coordinates must match')
    stacked = False
    stack_dims = utils.find_other_dims(U, ['lat','lon'])
    if stack_dims is not None:
        U_stack = U.stack(stacked=stack_dims)
        V_stack = V.stack(stacked=stack_dims)
        psi_stack = psi_anom.stack(stacked=stack_dims)
        stacked = True
        try:
            pres = U_stack['stacked']['pfull'] / 1000 # Takaya and Nakmura p.610
        except KeyError:
            try:
                pres = U_stack['stacked']['phalf'] / 1000 # Takaya and Nakmura p.610
            except KeyError:
                if p_lev is None:
                    raise TypeError('Cannot determine pressure level(s) of provided data. This should' +
                                    'be stored as a coordinate, "pfull" or "phalf" in the provided' +
                                    'objects. Alternatively, for single level computations, the' + 
                                    'pressure can be provided using the p_lev argument')
                else:
                    pres = p_lev / 1000 # Takaya and Nakmura p.610
    else:
        U_stack = U
        V_stack = V
        psi_stack = psi_anom
        if p_lev is None:
            raise TypeError('Cannot determine pressure level(s) of provided data. This should' +
                            'be stored as a coordinate, "pfull" or "phalf" in the provided' +
                            'objects. Alternatively, for single level computations, the' + 
                            'pressure can be provided using the p_lev argument')
        else:
            pres = p_lev / 1000 # Takaya and Nakmura p.610

    # Create a VectorWind instance (use gradient function) -----
    w = wsh.xarray.VectorWind(U_stack, V_stack)
    
    # Compute the various streamfunction gradients required -----
    psi_x, psi_y = w.gradient(psi_stack)
    psi_xx, psi_xy = w.gradient(psi_x)
    psi_yx, psi_yy = w.gradient(psi_y)
    
    # Compute the wave activity flux -----
    vel = (U_stack * U_stack + V_stack * V_stack) ** 0.5
    uwaf = (0.5 * pres * (U_stack * (psi_x * psi_x - psi_stack * psi_xx) + 
                          V_stack * (psi_x * psi_y - 0.5 * psi_stack * (psi_xy + psi_yx))) / vel).rename('uwaf')
    uwaf.attrs['units'] = 'm^2/s^2'
    uwaf.attrs['long_name'] = 'Zonal Rossby wave activity flux'
    vwaf = (0.5 * pres * (V_stack * (psi_y * psi_y - psi_stack * psi_yy) + 
                          U_stack * (psi_x * psi_y - 0.5 * psi_stack * (psi_xy + psi_yx))) / vel).rename('vwaf')
    vwaf.attrs['units'] = 'm^2/s^2'
    vwaf.attrs['long_name'] = 'Meridional Rossby wave activity flux'
    
    # Unstack the stacked dimensions -----
    if stacked:
        uwaf = uwaf.unstack('stacked')
        vwaf = vwaf.unstack('stacked')
    
    return uwaf, vwaf
