"""
    doppyo functions for computing various ocean, atmosphere, & climate diagnostics

    API
    ===
"""

__all__ = ['velocity_potential', 'stream_function', 'Rossby_wave_source', 'divergent', 'wave_activity_flux', 
           'Brunt_Vaisala', 'Rossby_wave_number', 'Eady_growth_rate', 'thermal_wind', 'eofs', 
           'mean_merid_mass_streamfunction', 'atmos_energy_cycle', 'isotherm_depth', 'pwelch', 
           'inband_variance', 'nino3', 'nino34', 'nino4', 'emi', 'dmi', 'soi', '_int_over_atmos']

# ===================================================================================================
# Packages
# ===================================================================================================
import collections
import warnings
import numpy as np
import xarray as xr
import windspharm as wsh
from scipy.sparse import linalg

# Load doppyo packages -----
from doppyo import utils


# ===================================================================================================
# Flow field diagnostics
# ===================================================================================================
def velocity_potential(u, v, lat_name=None, lon_name=None):
    """ 
        Returns the velocity potential given fields of u and v
            
        | Author: Dougie Squire
        | Date: 11/07/2018
            
        Parameters
        ----------
        u : xarray DataArray
            Array containing fields of zonal velocity with at least coordinates latitude and longitude \
                    (following standard naming - see Notes)
        v : xarray DataArray
            Array containing fields of meridional velocity with at least coordinates latitude and \
                    longitude (following standard naming - see Notes)
        lat_name : str, optional
            Name of latitude coordinate. If None, doppyo will attempt to determine lat_name \
                    automatically
        lon_name : str, optional
            Name of longitude coordinate. If None, doppyo will attempt to determine lon_name \
                    automatically
                
        Returns
        -------
        velocity_potential : xarray DataArray
            Array containing values of velocity potential
                
        Examples
        --------
        >>> u = xr.DataArray(np.random.normal(size=(6,4)), 
        ...                  coords=[('lat', np.arange(-75,76,30)), ('lon', np.arange(45,316,90))])
        >>> v = xr.DataArray(np.random.normal(size=(6,4)), 
        ...                  coords=[('lat', np.arange(-75,76,30)), ('lon', np.arange(45,316,90))])
        >>> doppyo.diagnostic.velocity_potential(u, v)
        <xarray.DataArray 'velocity_potential' (lat: 6, lon: 4)>
        array([[  431486.75 ,   431486.75 ,   431486.75 ,   431486.75 ],
               [ -240990.94 , -3553409.   ,  -970673.56 ,  2341744.5  ],
               [ 3338223.5  ,  1497203.9  , -1723363.2  ,   117656.31 ],
               [ 1009613.5  ,  1571693.6  ,   326689.3  ,  -235390.69 ],
               [ -931064.8  ,  -124736.375, -2516887.8  , -3323216.   ],
               [-1526244.   , -1526244.   , -1526244.   , -1526244.   ]], dtype=float32)
        Coordinates:
          * lat      (lat) int64 75 45 15 -15 -45 -75
          * lon      (lon) int64 45 135 225 315
        Attributes:
            units:          m**2 s**-1
            standard_name:  atmosphere_horizontal_velocity_potential
            long_name:      velocity potential
    
        Notes
        -----------
        | All input array coordinates must follow standard naming (see ``doppyo.utils.get_lat_name()``, \
                ``doppyo.utils.get_lon_name()``, etc)
        | This function utilises the windspharm package, which is a wrapper on pyspharm, which is a \
                wrapper on SPHEREPACK. These packages require that the latitudinal and longitudinal grid \
                is regular or Gaussian.
        | These calculations are not yet dask-compatible.
    
        To Do

        - Make dask-compatible by either developing the windspharm package, or using a kernel approach
    """
    
    if lat_name is None:
        lat_name = utils.get_lat_name(u)
    if lon_name is None:
        lon_name = utils.get_lon_name(u)

    if not utils._equal_coords(u, v):
        raise ValueError('u and v coordinates must match')

    # Create a VectorWind instance -----
    w = wsh.xarray.VectorWind(u, v)

    # Compute the velocity potential -----
    return w.velocitypotential().rename('phi')


# ===================================================================================================
def stream_function(u, v, lat_name=None, lon_name=None):
    """ 
        Returns the stream function given fields of u and v
        
        | Author: Dougie Squire
        | Date: 11/07/2018
        
        Parameters
        ----------
        u : xarray DataArray
            Array containing fields of zonal velocity with at least coordinates latitude and longitude \
                    (following standard naming - see Notes)
        v : xarray DataArray
            Array containing fields of meridional velocity with at least coordinates latitude and \
                    longitude (following standard naming - see Notes)
        lat_name : str, optional
            Name of latitude coordinate. If None, doppyo will attempt to determine lat_name \
                    automatically
        lon_name : str, optional
            Name of longitude coordinate. If None, doppyo will attempt to determine lon_name \
                    automatically
            
        Returns
        -------
        stream_function : xarray DataArray
            Array containing values of stream function
            
        Examples
        --------
        >>> u = xr.DataArray(np.random.normal(size=(6,4)), 
        ...                  coords=[('lat', np.arange(-75,76,30)), ('lon', np.arange(45,316,90))])
        >>> v = xr.DataArray(np.random.normal(size=(6,4)), 
        ...                  coords=[('lat', np.arange(-75,76,30)), ('lon', np.arange(45,316,90))])
        >>> doppyo.diagnostic.stream_function(u, v)
        <xarray.DataArray 'psi' (lat: 6, lon: 4)>
        array([[ -690643.6 ,  -690643.6 ,  -690643.6 ,  -690643.6 ],
               [-2041977.8 , -1060127.  , -3052305.8 , -4034156.5 ],
               [ 4112389.2 ,  4630193.5 , -5212595.5 , -5730399.5 ],
               [  528500.75,  4670647.5 ,  2589393.  , -1552753.9 ],
               [-2686391.2 ,  -707369.25,  4204334.  ,  2225311.5 ],
               [ 1703481.9 ,  1703481.9 ,  1703481.9 ,  1703481.9 ]], dtype=float32)
        Coordinates:
          * lat      (lat) int64 75 45 15 -15 -45 -75
          * lon      (lon) int64 45 135 225 315
        Attributes:
            units:          m**2 s**-1
            standard_name:  atmosphere_horizontal_streamfunction
            long_name:      streamfunction
        
        Notes
        -----------
        | All input array coordinates must follow standard naming (see ``doppyo.utils.get_lat_name()``, \
                ``doppyo.utils.get_lon_name()``, etc)
        | This function utilises the windspharm package, which is a wrapper on pyspharm, which is a \
                wrapper on SPHEREPACK. These packages require that the latitudinal and longitudinal grid \
                is regular or Gaussian.
        | These calculations are not yet dask-compatible.
        
        To Do
        
        - Make dask-compatible by either developing the windspharm package, or using a kernel approach
    """

    if lat_name is None:
        lat_name = utils.get_lat_name(u)
    if lon_name is None:
        lon_name = utils.get_lon_name(u)

    if not utils._equal_coords(u, v):
        raise ValueError('u and v coordinates must match')
        
    # Create a VectorWind instance -----
    w = wsh.xarray.VectorWind(u, v)

    # Compute the streamfunction -----
    return w.stream_function().rename('psi')


# ===================================================================================================
def Rossby_wave_source(u, v, lat_name=None, lon_name=None):
    """ 
        Returns the Rossby wave source given fields of u and v
        
        | Author: Dougie Squire
        | Date: 11/07/2018
        
        Parameters
        ----------
        u : xarray DataArray
            Array containing fields of zonal velocity with at least coordinates latitude and longitude \
                    (following standard naming - see Notes)
        v : xarray DataArray
            Array containing fields of meridional velocity with at least coordinates latitude and \
                    longitude (following standard naming - see Notes)
        lat_name : str, optional
            Name of latitude coordinate. If None, doppyo will attempt to determine lat_name \
                    automatically
        lon_name : str, optional
            Name of longitude coordinate. If None, doppyo will attempt to determine lon_name \
                    automatically
            
        Returns
        -------
        Rossby_wave_source : xarray DataArray
            Array containing values of Rossby wave source
            
        Examples
        --------
        >>> u = xr.DataArray(np.random.normal(size=(6,4)), 
        ...                  coords=[('lat', np.arange(-75,76,30)), ('lon', np.arange(45,316,90))])
        >>> v = xr.DataArray(np.random.normal(size=(6,4)), 
        ...                  coords=[('lat', np.arange(-75,76,30)), ('lon', np.arange(45,316,90))])
        >>> doppyo.diagnostic.Rossby_wave_source(u, v)
        <xarray.DataArray 'rws' (lat: 6, lon: 4)>
        array([[ 4.382918,  4.382918,  4.382918,  4.382918],
               [-2.226769,  5.020311, -2.600087, -9.838818],
               [ 2.1693  , -2.133569,  0.498156,  4.818402],
               [-1.404836,  0.192032,  0.112654, -1.494616],
               [-0.103261,  4.518184,  0.648616, -4.05276 ],
               [ 4.070806,  4.070806,  4.070806,  4.070806]])
        Coordinates:
          * lat      (lat) int64 75 45 15 -15 -45 -75
          * lon      (lon) int64 45 135 225 315
        Attributes:
            units:      1e-11/s^2
            long_name:  Rossby wave source
        
        Notes
        -----------
        | All input array coordinates must follow standard naming (see ``doppyo.utils.get_lat_name()``, \
                ``doppyo.utils.get_lon_name()``, etc)
        | This function utilises the windspharm package, which is a wrapper on pyspharm, which is a \
                wrapper on SPHEREPACK. These packages require that the latitudinal and longitudinal grid \
                is regular or Gaussian.
        | These calculations are not yet dask-compatible.
        
        To Do
        
        - Make dask-compatible by either developing the windspharm package, or using a kernel approach
    """

    if lat_name is None:
        lat_name = utils.get_lat_name(u)
    if lon_name is None:
        lon_name = utils.get_lon_name(u)

    if not utils._equal_coords(u, v):
        raise ValueError('u and v coordinates must match')
        
    # Create a VectorWind instance -----
    w = wsh.xarray.VectorWind(u, v)

    # Compute components of Rossby wave source -----
    eta = w.absolutevorticity() # absolute vorticity
    div = w.divergence() # divergence
    uchi, vchi = w.irrotationalcomponent() # irrotational (divergent) wind components
    etax, etay = w.gradient(eta) # gradients of absolute vorticity

    # Combine the components to form the Rossby wave source term -----
    rws = 1e11 * (-eta * div - (uchi * etax + vchi * etay)).rename('rws')
    rws.attrs['units'] = '1e-11 / s**2'
    rws.attrs['long_name'] = 'Rossby wave source'
    
    return rws


# ===================================================================================================
def divergent(u, v, lat_name=None, lon_name=None):
    """ 
        Returns the irrotational (divergent) component of u and v
        
        | Author: Dougie Squire
        | Date: 11/07/2018
        
        Parameters
        ----------
        u : xarray DataArray
            Array containing fields of zonal velocity with at least coordinates latitude and longitude \
                    (following standard naming - see Notes)
        v : xarray DataArray
            Array containing fields of meridional velocity with at least coordinates latitude and \
                    longitude (following standard naming - see Notes)
        lat_name : str, optional
            Name of latitude coordinate. If None, doppyo will attempt to determine lat_name \
                    automatically
        lon_name : str, optional
            Name of longitude coordinate. If None, doppyo will attempt to determine lon_name \
                    automatically
            
        Returns
        -------
        divergent : xarray Dataset
            | Dataset containing the following variables:
            | u_chi; array containing the irrotational component of u
            | v_chi; array containing the irrotational component of v
            
        Examples
        --------
        >>> u = xr.DataArray(np.random.normal(size=(6,4)), 
        ...                  coords=[('lat', np.arange(-75,76,30)), ('lon', np.arange(45,316,90))])
        >>> v = xr.DataArray(np.random.normal(size=(6,4)), 
        ...                  coords=[('lat', np.arange(-75,76,30)), ('lon', np.arange(45,316,90))])
        >>> doppyo.diagnostic.divergent(u, v)
        <xarray.Dataset>
        Dimensions:  (lat: 6, lon: 4)
        Coordinates:
          * lat      (lat) int64 75 45 15 -15 -45 -75
          * lon      (lon) int64 45 135 225 315
        Data variables:
            u_chi     (lat, lon) float32 0.5355302 -0.45865965 ... -0.7270669 -0.64930713
            v_chi     (lat, lon) float32 -0.45865965 -0.5355302 ... 0.64930713 -0.7270669
        
        Notes
        -----------
        | All input array coordinates must follow standard naming (see ``doppyo.utils.get_lat_name()``, \
                ``doppyo.utils.get_lon_name()``, etc)
        | This function utilises the windspharm package, which is a wrapper on pyspharm, which is a \
                wrapper on SPHEREPACK. These packages require that the latitudinal and longitudinal grid \
                is regular or Gaussian.
        | These calculations are not yet dask-compatible.
        
        To Do
        
        - Make dask-compatible by either developing the windspharm package, or using a kernel approach
    """

    if lat_name is None:
        lat_name = utils.get_lat_name(u)
    if lon_name is None:
        lon_name = utils.get_lon_name(u)

    if not utils._equal_coords(u, v):
        raise ValueError('u and v coordinates must match')
            
    # Create a VectorWind instance -----
    w = wsh.xarray.VectorWind(u, v)

    # Compute the irrotational components -----
    u_chi, v_chi = w.irrotationalcomponent()
    
    # Combine into dataset -----
    div = u_chi.rename('u_chi').to_dataset()
    div['v_chi'] = v_chi
    
    return div


# ===================================================================================================
def wave_activity_flux(psi_anom, u, v, plevel=None, lat_name=None, lon_name=None):
    """ 
        Returns the stationary component of the wave activity flux, following Takaya and Nakamura, \
                (2001) using zonal and meridional velocity fields on one or more isobaric surface(s)
        
        | Author: Dougie Squire
        | Date: 11/07/2018
        
        Parameters
        ----------
        psi_anom : xarray DataArray
            Array containing fields of stream function anomalies with at least coordinates latitude \
                    and longitude (following standard naming - see Notes)
        u : xarray DataArray
            Array containing fields of zonal velocity with at least coordinates latitude and longitude \
                    (following standard naming - see Notes)
        v : xarray DataArray
            Array containing fields of meridional velocity with at least coordinates latitude and \
                    longitude (following standard naming - see Notes)
        plevel : value, optional
            Value of the pressure level corresponding to the provided arrays. If None, pressure \
                    level(s) are extracted from the psi_anom/u/v coordinate. Pressure levels must be provided \
                    in units of hPa
        lat_name : str, optional
            Name of latitude coordinate. If None, doppyo will attempt to determine lat_name \
                    automatically
        lon_name : str, optional
            Name of longitude coordinate. If None, doppyo will attempt to determine lon_name \
                    automatically
            
        Returns
        -------
        wave_activity_flux : xarray Dataset
            | Dataset containing the following variables:
            | u_waf; array containing the zonal component of the wave activity flux
            | v_waf; array containing the meridonal component of the wave activity flux
            
        Examples
        --------
        >>> u = xr.DataArray(np.random.normal(size=(6,4,2,24)), 
        ...                  coords=[('lat', np.arange(-75,76,30)), ('lon', np.arange(45,316,90)), 
        ...                          ('level', [100,500]), 
        ...                          ('time', pd.date_range('2000-01-01',periods=24,freq='M'))])
        >>> v = xr.DataArray(np.random.normal(size=(6,4,2,24)), 
        ...                  coords=[('lat', np.arange(-75,76,30)), ('lon', np.arange(45,316,90)), 
        ...                          ('level', [100,500]), 
        ...                          ('time', pd.date_range('2000-01-01',periods=24,freq='M'))])
        >>> u_clim = u.groupby('time.month').mean(dim='time')
        >>> v_clim = v.groupby('time.month').mean(dim='time')
        >>> u_anom = doppyo.utils.anomalize(u, u_clim)
        >>> v_anom = doppyo.utils.anomalize(v, v_clim)
        >>> psi_anom = doppyo.diagnostic.stream_function(u_anom, v_anom)
        >>> doppyo.diagnostic.wave_activity_flux(psi_anom, u, v)
        <xarray.Dataset>
        Dimensions:  (lat: 6, level: 2, lon: 4, time: 24)
        Coordinates:
          * level    (level) int64 100 500
          * lat      (lat) int64 -75 -45 -15 15 45 75
          * lon      (lon) int64 45 135 225 315
          * time     (time) datetime64[ns] 2000-01-31 2000-02-29 ... 2001-12-31
        Data variables:
            u_waf     (level, lat, lon, time) float64 0.003852 0.0001439 ... -0.06913
            v_waf     (level, lat, lon, time) float64 0.01495 3.032e-05 ... 0.02944
        
        Notes
        -----------
        | All input array coordinates must follow standard naming (see ``doppyo.utils.get_lat_name()``, \
                ``doppyo.utils.get_lon_name()``, etc)
        | Pressure levels must be provided in units of hPa
        | This function utilises the windspharm package, which is a wrapper on pyspharm, which is a \
                wrapper on SPHEREPACK. These packages require that the latitudinal and longitudinal grid \
                is regular or Gaussian.
        | These calculations are not yet dask-compatible
        
        To Do
        
        - Make dask-compatible by either developing the windspharm package, or using a kernel approach
    """
    
    if lat_name is None:
        lat_name = utils.get_lat_name(u)
    if lon_name is None:
        lon_name = utils.get_lon_name(u)
        
    if not (utils._equal_coords(u, v) & utils._equal_coords(u, psi_anom)):
        raise ValueError('psi_anom, u and v coordinates must match')
    
    # Get plev from dimension if it exists -----
    try:
        plev_name = utils.get_level_name(u)
        plev = u[plev_name] / 1000 # Takaya and Nakmura p.610
    except KeyError:
        if plevel is None:
            raise TypeError('Cannot determine pressure level(s) of provided data. This should' +
                            'be stored as a coordinate, "level" or "plev" in the provided' +
                            'objects. Alternatively, for single level computations, the' + 
                            'pressure can be provided using the plevel argument')
        else:
            plev = plevel / 1000 # Takaya and Nakmura p.610
    
    # Create a VectorWind instance (use gradient function) -----
    w = wsh.xarray.VectorWind(u, v)
    
    # Compute the various stream function gradients required -----
    psi_x, psi_y = w.gradient(psi_anom)
    psi_xx, psi_xy = w.gradient(psi_x)
    psi_yx, psi_yy = w.gradient(psi_y)
    
    # Compute the wave activity flux -----
    vel = (u * u + v * v) ** 0.5
    u_waf = (0.5 * plev * (u * (psi_x * psi_x - psi_anom * psi_xx) + 
                          v * (psi_x * psi_y - 0.5 * psi_anom * (psi_xy + psi_yx))) / vel).rename('u_waf')
    u_waf.attrs['units'] = 'm^2/s^2'
    u_waf.attrs['long_name'] = 'Zonal Rossby wave activity flux'
    v_waf = (0.5 * plev * (v * (psi_y * psi_y - psi_anom * psi_yy) + 
                          u * (psi_x * psi_y - 0.5 * psi_anom * (psi_xy + psi_yx))) / vel).rename('v_waf')
    v_waf.attrs['units'] = 'm^2/s^2'
    v_waf.attrs['long_name'] = 'Meridional Rossby wave activity flux'
        
    # Combine into dataset -----
    waf = u_waf.to_dataset()
    waf['v_waf'] = v_waf
    
    return waf


# ===================================================================================================
def Brunt_Vaisala(temp, plevel_name=None):
    """
        Returns the Brunt Väisälä frequency
        
        | Author: Dougie Squire
        | Date: 15/07/2018
        
        Parameters
        ----------
        temp : xarray DataArray
            Array containing fields of temperature with at least coordinates latitude, longitude and \
                    pressure level (following standard naming - see Notes)
        plevel_name : str, optional
            Name of pressure level coordinate. If None, doppyo will attempt to determine plevel_name \
                    automatically
            
        Returns
        -------
        nsq : xarray DataArray
            Array with same dimensions as input arrays containing the Brunt Väisälä frequency
        
        Examples
        --------
        >>> temp = xr.DataArray(np.random.normal(size=(4,4,2)), 
        ...                     coords=[('lat', np.arange(-90,90,45)), ('lon', np.arange(0,360,90)), 
        ...                             ('level', [100,200])])
        >>> doppyo.diagnostic.Brunt_Vaisala(temp)
        <xarray.DataArray 'nsq' (level: 2, lon: 4, lat: 4)>
        array([[[-2.928266e-01, -2.709919e+01,  2.826585e-02,  6.083374e-01],
                [ 3.260879e-01,  1.933501e-01, -9.033669e+00, -1.468327e+00],
                [-1.957892e+00,  2.408426e-01,  5.597183e-01, -2.548981e+01],
                [-3.234550e-01, -1.907664e+00,  2.506510e-01, -7.385499e-01]],
        ...
               [[-1.136451e-01, -1.796130e+00, -1.095550e-02,  5.748574e+00],
                [ 4.407484e+02,  4.736099e-01, -5.086917e-01, -6.610682e-01],
                [-2.458302e+00,  6.864762e+00,  2.633289e+00, -4.246873e-01],
                [-1.839424e+01, -1.194455e+00,  5.659980e+02, -2.567729e+00]]])
        Coordinates:
          * lat      (lat) int64 -90 -45 0 45
          * lon      (lon) int64 0 90 180 270
          * level    (level) int64 100 200
        Attributes:
            long_name:  Brunt-Vaisala frequency squared
            units:      s^-2

        Notes
        -----------
        | All input array coordinates must follow standard naming (see ``doppyo.utils.get_lat_name()``, \
                ``doppyo.utils.get_lon_name()``, etc)
        | Pressure levels must be provided in units of hPa
        
        To do
        
        - Add switch for atmosphere/ocean input
    """

    R = utils.constants().R_d
    Cp = utils.constants().C_pd
    g = utils.constants().g

    if plevel_name is None:
        plevel_name = utils.get_plevel_name(temp)
    
    dTdp = temp.differentiate(coord=plevel_name)
    pdR = temp[plevel_name] / R

    nsq = ((-dTdp * pdR + (temp / Cp)) / (temp / g) ** 2).rename('nsq')
    nsq.attrs['long_name']='Brunt-Vaisala frequency squared'
    nsq.attrs['units']='s^-2'
    
    return nsq.rename('nsq')


# ===================================================================================================
def Rossby_wave_number(u, v, u_clim, lat_name=None, lon_name=None):
    """ 
        Returns the square of the stationary Rossby wave number, Ks**2
        
        | Author: Dougie Squire
        | Date: 11/07/2018
        
        Parameters
        ----------
        u : xarray DataArray
            Array containing fields of zonal velocity with at least coordinates latitude and longitude \
                    (following standard naming - see Notes)
        v : xarray DataArray
            Array containing fields of meridional velocity with at least coordinates latitude and \
                    longitude (following standard naming - see Notes)
        u_clim : xarray DataArray
            Array containing climatological fields of zonal velocity with at least coordinates latitude \
                    and longitude (following standard naming - see Notes)
        lat_name : str, optional
            Name of latitude coordinate. If None, doppyo will attempt to determine lat_name \
                    automatically
        lon_name : str, optional
            Name of longitude coordinate. If None, doppyo will attempt to determine lon_name \
                    automatically
            
        Returns
        -------
        Rossby_wave_number : xarray DataArray
            Array containing the square of the Rossby wave source
            
        Examples
        --------
        >>> u = xr.DataArray(np.random.normal(size=(6,4,24)), 
        ...                  coords=[('lat', np.arange(-75,76,30)), ('lon', np.arange(45,316,90)), 
        ...                          ('time', pd.date_range('2000-01-01',periods=24,freq='M'))])
        >>> v = xr.DataArray(np.random.normal(size=(6,4,24)), 
        ...                  coords=[('lat', np.arange(-75,76,30)), ('lon', np.arange(45,316,90)), 
        ...                          ('time', pd.date_range('2000-01-01',periods=24,freq='M'))])
        >>> u_clim = u.groupby('time.month').mean(dim='time')
        >>> u_clim = doppyo.utils.anomalize(0*u, -u_clim)
        >>> doppyo.diagnostic.Rossby_wave_number(u, v, u_clim)
        <xarray.DataArray 'ks2' (lat: 6, lon: 4, time: 24)>
        array([[[ 8.077277e-01,  1.885835e-01, ...,  6.383953e-01, -4.686696e-01],
                [-3.756420e-01,  1.210226e+00, ..., -2.055076e+00, -2.291500e+00],
                [ 8.786361e-01,  4.181778e-01, ..., -2.071749e+00,  4.018699e-01],
                [ 8.218020e-01,  5.197270e+00, ...,  5.181735e+00,  7.112056e-01]],
        ...
               [[-5.323813e+02, -2.894449e+02, ..., -5.063012e+03, -3.921559e+02],
                [ 3.167388e+02, -5.406136e+02, ..., -1.987485e+03, -2.692395e+02],
                [ 2.916992e+03,  2.318578e+02, ...,  8.611478e+02,  8.559919e+02],
                [-4.380459e+02, -5.035198e+02, ..., -1.844072e+03, -2.856807e+02]],
        ...,
               [[ 3.832781e+02, -1.272144e+03, ...,  3.900539e+02, -5.402686e+02],
                [-2.494814e+02, -2.041985e+02, ...,  3.426493e+02, -5.557717e+02],
                [-6.290198e+03,  1.606871e+03, ...,  2.894713e+03,  3.284330e+02],
                [-3.325505e+02, -2.406172e+02, ..., -3.270787e+03, -1.040641e+03]],
        ...
               [[ 1.401437e+00,  6.053096e-01, ...,  1.725558e-01, -7.287578e+01],
                [-8.905873e-01,  1.469694e-01, ...,  1.308367e+00, -7.136195e-01],
                [ 4.318194e+01, -1.850361e-01, ..., -2.447798e-01, -4.454747e-01],
                [ 1.247740e+00,  9.826164e-02, ...,  2.808380e+00,  1.254609e+00]]])
        Coordinates:
          * lat      (lat) int64 75 45 15 -15 -45 -75
          * lon      (lon) int64 45 135 225 315
          * time     (time) datetime64[ns] 2000-01-31 2000-02-29 ... 2001-12-31
        Attributes:
            units:      real number
            long_name:  Square of Rossby stationary wavenumber

        Notes
        -----------
        | The input u_clim must have the same dimensions as u and v. One can project a mean climatology, \
                A_clim, over the time dimension in A using ``doppyo.utils.anomalize(0*A, -A_clim)``
        | All input array coordinates must follow standard naming (see ``doppyo.utils.get_lat_name()``, \
                ``doppyo.utils.get_lon_name()``, etc)
        | This function utilises the windspharm package, which is a wrapper on pyspharm, which is a \
                wrapper on SPHEREPACK. These packages require that the latitudinal and longitudinal grid \
                is regular or Gaussian.
        | These calculations are not yet dask-compatible.
        
        To Do
        
        - Make dask-compatible by either developing the windspharm package, or using a kernel approach
        
    """

    if lat_name is None:
        lat_name = utils.get_lat_name(u)
    if lon_name is None:
        lon_name = utils.get_lon_name(u)

    if not (utils._equal_coords(u, v) & utils._equal_coords(u, u_clim)):
        raise ValueError('u, v and u_clim coordinates must match')

    # Create a VectorWind instance -----
    w = wsh.xarray.VectorWind(u, v)

    # Compute the absolute vorticity gradient -----
    etau, etav = w.gradient(w.absolutevorticity())
    
    # Compute the stationary wave number -----
    ks2 = (xr.ufuncs.cos(etav[lat_name] / 180 * utils.constants().pi)**2 * \
                    (etav * utils.constants().R_earth ** 2) / u_clim).rename('ks2')
    ks2.attrs['units'] = 'real number'
    ks2.attrs['long_name'] = 'Square of Rossby stationary wavenumber'
        
    return ks2


# ===================================================================================================
def Eady_growth_rate(u, v, gh, nsq, lat_name=None, lon_name=None, level_name=None):
    """ 
        Returns the square of the Eady growth rate
        
        | Author: Dougie Squire
        | Date: 15/07/2018
        
        Parameters
        ----------
        u : xarray DataArray
            Array containing fields of zonal velocity with at least coordinates latitude, longitude and \
                    level (following standard naming - see Notes)
        v : xarray DataArray
            Array containing fields of meridional velocity with at least coordinates latitude, longitude \
                    and level (following standard naming - see Notes)
        gh : xarray DataArray
            Array containing fields of geopotential height with at least coordinates latitude, longitude \
                    and level (following standard naming - see Notes)
        nsq : xarray DataArray
            Array containing fields of Brunt Väisälä frequency with at least coordinates latitude, \
                    longitude and level (following standard naming - see Notes)
        lat_name : str, optional
            Name of latitude coordinate. If None, doppyo will attempt to determine lat_name \
                    automatically
        lon_name : str, optional
            Name of longitude coordinate. If None, doppyo will attempt to determine lon_name \
                    automatically
        level_name : str, optional
            Name of level coordinate. If None, doppyo will attempt to determine level_name
            automatically
            
        Returns
        -------
        Eady^2 : xarray DataArray
            Array containing the square of the Eady growth rate
        
        Examples
        --------
        >>> u = xr.DataArray(np.random.normal(size=(6,4,2)), 
        ...                  coords=[('lat', np.arange(-75,76,30)), ('lon', np.arange(45,316,90)), 
        ...                          ('level', [200, 500])])
        >>> v = xr.DataArray(np.random.normal(size=(6,4,2)), 
        ...                  coords=[('lat', np.arange(-75,76,30)), ('lon', np.arange(45,316,90)), 
        ...                          ('level', [200, 500])])
        >>> temp = xr.DataArray(np.random.normal(size=(6,4,2)), 
        ...                     coords=[('lat', np.arange(-75,76,30)), ('lon', np.arange(45,316,90)), 
        ...                             ('level', [200, 500])])
        >>> gh = xr.DataArray(np.random.normal(size=(6,4,2)), 
        ...                   coords=[('lat', np.arange(-75,76,30)), ('lon', np.arange(45,316,90)), 
        ...                           ('level', [200, 500])])
        >>> nsq = doppyo.diagnostic.Brunt_Vaisala(temp)
        >>> doppyo.diagnostic.Eady_growth_rate(u, v, gh, nsq)
        <xarray.DataArray 'Eady^2' (level: 2, lon: 4, lat: 6)>
        array([[[-5.371897e-08,  1.338133e-11, -7.254014e-13, -8.196598e-12,
                  2.062633e-09, -7.200158e-12],
                [ 9.906932e-10, -7.349832e-09, -2.558847e-12, -1.695842e-09,
                  4.986779e-09, -3.090147e-09],
                [ 3.948602e-07,  1.397756e-09,  1.508010e-10,  1.481968e-10,
                  5.627093e-11,  7.463454e-10],
                [ 4.326971e-09, -2.528522e-09, -1.243954e-13, -3.138463e-11,
                 -6.801250e-09, -6.286382e-10]],
        ...
               [[-8.580527e-10,  7.040065e-12, -3.760004e-13, -1.213131e-12,
                  2.437557e-11, -6.522981e-11],
                [ 6.119671e-09, -1.644123e-09, -5.124997e-11,  1.725101e-08,
                  2.574158e-08, -3.101566e-10],
                [ 1.601742e-06,  1.994867e-11,  3.341006e-11,  1.641253e-11,
                  5.601919e-08,  5.527214e-11],
                [ 4.700271e-09, -1.422149e-11, -1.302035e-12, -2.153002e-11,
                 -4.607096e-10, -3.813686e-09]]])
        Coordinates:
          * lat      (lat) int64 -75 -45 -15 15 45 75
          * lon      (lon) int64 45 135 225 315
          * level    (level) int64 200 500
        Attributes:
            units:      s^-2
            long_name:  Square of Eady growth rate
    
        Notes
        -----------
        All input array coordinates must follow standard naming (see ``doppyo.utils.get_lat_name()``, \
                ``doppyo.utils.get_lon_name()``, etc)
    """
    
    degtorad = utils.constants().pi / 180
    if lat_name is None:
        lat_name = utils.get_lat_name(u)
    if lon_name is None:
        lon_name = utils.get_lon_name(u)
    if level_name is None:
        level_name = utils.get_level_name(u)
    
    f = 2 * utils.constants().Omega * xr.ufuncs.sin(gh[lat_name] * degtorad)
    eady2 = ((utils.constants().Ce * f) * (xr.ufuncs.sqrt(u ** 2 + v ** 2).differentiate(coord=level_name) / \
            gh.differentiate(coord=level_name))) ** 2 / nsq
    eady2.attrs['units'] = 's^-2'
    eady2.attrs['long_name'] = 'Square of Eady growth rate'
    
    return eady2.rename('Eady^2')


# ===================================================================================================
def thermal_wind(gh, plevel_lower, plevel_upper, lat_name=None, lon_name=None, plevel_name=None):
    """ 
        Returns the thermal wind, (u_tw, v_tw) = 1/f x k x grad(thickness), where f = 2*Omega*sin(lat)
        
        | Author: Dougie Squire
        | Date: 15/07/2018
        
        Parameters
        ----------
        gh : xarray DataArray
            Array containing fields of geopotential height with at least coordinates latitude, longitude \
                    and level (following standard naming - see Notes)
        plevel_lower : value
            Value of lower pressure level used to compute termal wind. Must exist in level coordinate of \
                    gh
        plevel_upper : value
            Value of upper pressure level used to compute termal wind. Must exist in level coordinate of \
                    gh
        lat_name : str, optional
            Name of latitude coordinate. If None, doppyo will attempt to determine lat_name \
                    automatically
        lon_name : str, optional
            Name of longitude coordinate. If None, doppyo will attempt to determine lon_name \
                    automatically
        plevel_name : str, optional
            Name of pressure level coordinate. If None, doppyo will attempt to determine plevel_name \
                    automatically
            
        Returns
        -------
        thermal_wind : xarray Dataset
            | Dataset containing the following variables:
            | u_tw; array containing the zonal component of the thermal wind
            | v_tw; array containing the meridonal component of the thermal wind
        
        Examples
        --------
        >>> gh = xr.DataArray(np.random.normal(size=(3,4,4)), 
        ...                   coords=[('level', [400, 500, 600]), ('lat', np.arange(-90,90,45)), 
        ...                   ('lon', np.arange(0,360,90))])
        >>> doppyo.diagnostic.thermal_wind(gh, plevel_lower=400, plevel_upper=600)
        <xarray.Dataset>
        Dimensions:  (lat: 4, lon: 4)
        Coordinates:
            level    float64 500.0
          * lon      (lon) int64 0 90 180 270
          * lat      (lat) int64 -90 -45 0 45
        Data variables:
            u_tw     (lon, lat) float64 0.003727 0.0006837 inf ... inf -0.0001238
            v_tw     (lat, lon) float64 4.515e+12 -1.443e+12 ... -0.000569 -0.0002777

        Notes
        -----------
        | All input array coordinates must follow standard naming (see ``doppyo.utils.get_lat_name()``, \
                ``doppyo.utils.get_lon_name()``, etc)
        | Pressure levels must be provided in units of hPa
    """
    
    degtorad = utils.constants().pi / 180
    if lat_name is None:
        lat_name = utils.get_lat_name(gh)
    if lon_name is None:
        lon_name = utils.get_lon_name(gh)
    if plevel_name is None:
        plevel_name = utils.get_plevel_name(gh)
    
    # Compute the thickness -----
    upper = gh.sel({plevel_name : plevel_lower})
    upper[plevel_name] = (plevel_lower + plevel_upper) / 2
    lower = gh.sel({plevel_name : plevel_upper})
    lower[plevel_name] = (plevel_lower + plevel_upper) / 2
    thickness = upper - lower
    
    # Compute the gradient -----
    x, y = utils.xy_from_lonlat(gh[lon_name], gh[lat_name])
    u_tmp = utils.differentiate_wrt(thickness, dim=lon_name, x=x)
    v_tmp = utils.differentiate_wrt(thickness, dim=lat_name, x=y)
    
    # Or use windspharm -----
    # w = wsh.xarray.VectorWind(thickness, thickness)
    # u_tmp, v_tmp = w.gradient(thickness)
    
    # k x (u_tw,v_tw) -> (-v_tw, u_tw) -----
    u_tw = -v_tmp / (2 * utils.constants().Omega * xr.ufuncs.sin(thickness[lat_name] * degtorad))
    v_tw = u_tmp / (2 * utils.constants().Omega * xr.ufuncs.sin(thickness[lat_name] * degtorad))
    
    # Combine into dataset -----
    tw = u_tw.to_dataset(name='u_tw')
    tw['v_tw'] = v_tw
    
    return tw


# ===================================================================================================
def eofs(da, sample_dim='time', weight=None, n_modes=20):
    """
        Returns the empirical orthogonal functions (EOFs), and associated principle component \
                timeseries (PCs), and explained variances of provided array. Follows notation used in \
                "Bjornsson H. and Venegas S. A. 1997 A Manual for EOF and SVD analyses of Climatic Data", \
                whereby, (phi, sqrt_lambdas, EOFs) = svd(data) and PCs = phi * sqrt_lambdas
        
        | Author: Dougie Squire
        | Date: 19/18/2018
        
        Parameters
        ----------
        da : xarray DataArray or sequence of xarray DataArrays
            Array to use to compute EOFs. When input array is a list of xarray objects, returns the \
                    joint EOFs associated with each object. In this case, all xarray objects in da must have \
                    sample_dim dimensions of equal length.
        sample_dim : str, optional
            EOFs sample dimension
        weight : xarray DataArray or sequence of xarray DataArrays, optional
            Weighting to apply prior to svd. If weight=None, cos(lat)^2 weighting are used. If weight \
                    is specified, it must be the same length as da with each element broadcastable onto each \
                    element of da
        n_modes : values, optional
            Number of EOF modes to return
            
        Returns
        -------
        eofs : xarray Dataset
            | Dataset containing the following variables:
            | EOFs; array containing the empirical orthogonal functions
            | PCs; array containing the associated principle component timeseries
            | lambdas; array containing the eigenvalues of the covariance of the input data
            | explained_var; array containing the fraction of the total variance explained by each EOF \
                    mode
            
        Examples
        --------
        >>> A = xr.DataArray(np.random.normal(size=(6,4,40)), 
        ...                  coords=[('lat', np.arange(-75,76,30)), ('lon', np.arange(45,316,90)), 
        ...                          ('time', pd.date_range('2000-01-01', periods=40, freq='M'))])
        >>> doppyo.diagnostic.eofs(A)
        <xarray.Dataset>
        Dimensions:        (lat: 6, lon: 4, mode: 20, time: 40)
        Coordinates:
          * time           (time) datetime64[ns] 2000-01-31 2000-02-29 ... 2003-04-30
          * mode           (mode) int64 1 2 3 4 5 6 7 8 9 ... 12 13 14 15 16 17 18 19 20
          * lat            (lat) int64 -75 -45 -15 15 45 75
          * lon            (lon) int64 45 135 225 315
        Data variables:
            EOFs           (mode, lat, lon) float64 -0.05723 -0.01997 ... 0.08166
            PCs            (time, mode) float64 1.183 -1.107 -0.5385 ... -0.08552 0.1951
            lambdas        (mode) float64 87.76 80.37 68.5 58.14 ... 8.269 6.279 4.74
            explained_var  (mode) float64 0.1348 0.1234 0.1052 ... 0.009644 0.00728
            
        Notes
        -----------
        This function is a wrapper on scipy.sparse.linalg.svds which is a naive implementation \
                using ARPACK. Thus, the approach implemented here is non-lazy and could incur large \
                increases in memory usage.
    """
    
    if isinstance(da, xr.core.dataarray.DataArray):
        da = [da]
    if isinstance(weight, xr.core.dataarray.DataArray):
        weight = [weight]

    # Apply weights -----
    if weight is None:
        degtorad = utils.constants().pi / 180
        weight = [xr.ufuncs.cos(da[idx][utils.get_lat_name(da[idx])] * degtorad) ** 0.5
                  for idx in range(len(da))]
    
    if len(weight) != len(da):
        raise ValueError('da and weight must be of equal length')
    da = [weight[idx].fillna(0) * da[idx] for idx in range(len(da))]
    
    # Stack along everything but the sample dimension -----
    sensor_dims = [utils.get_other_dims(d, sample_dim) for d in da]
    da = [d.stack(sensor_dim=sensor_dims[idx])
           .transpose(*[sample_dim, 'sensor_dim'])  for idx, d in enumerate(da)]
    sensor_segs = np.cumsum([0] + [len(d.sensor_dim) for d in da])
    
    # Load and concatenate each object in da -----
    try: 
        data = np.concatenate(da, axis=1)
    except ValueError:
        raise ValueError('sample_dim must be equal length for all data in da')
    
    # First dimension must be sample dimension -----
    phi, sqrt_lambdas, eofs = linalg.svds(data, k=n_modes)
    pcs = phi * sqrt_lambdas
    lambdas = sqrt_lambdas ** 2
    
    # n_modes largest modes are ordered from smallest to largest -----
    pcs = np.flip(pcs, axis=1)
    lambdas = np.flip(lambdas, axis=0)
    eofs = np.flip(eofs, axis=0)

    # Compute the sum of the lambdas -----
    sum_of_lambdas = np.trace(np.dot(data,data.T))

    # Restructure back into xarray object -----
    dims_eof = ['mode', 'sensor_dim']
    dims_pc = [sample_dim, 'mode']
    dims_lambda = ['mode']
    EOF = []
    for idx in range(len(sensor_segs)-1):
        data_vars = {'EOFs' : (tuple(dims_eof), eofs[:, sensor_segs[idx]:sensor_segs[idx+1]]),
                     'PCs' : (tuple(dims_pc), pcs),
                     'lambdas' : (tuple(dims_lambda), lambdas),
                     'explained_var' : (tuple(dims_lambda), lambdas / sum_of_lambdas)}
        coords = dict(da[idx].coords.items())
        coords['mode'] = np.arange(1, n_modes+1)
        EOF.append(xr.Dataset(data_vars,coords).unstack('sensor_dim'))
    
    if len(EOF) == 1:
        return EOF[0]
    else:
        return EOF
    

# ===================================================================================================
def mean_merid_mass_streamfunction(v, lat_name=None, lon_name=None, plevel_name=None):
    """
        Returns the mean meridional mass stream function averaged over all provided longitudes
        
        | Author: Dougie Squire
        | Date: 15/07/2018
        
        Parameters
        ----------
        v : xarray DataArray
            Array containing fields of meridional velocity with at least coordinates latitude, longitude \
                    and level (following standard naming - see Notes)
        lat_name : str, optional
            Name of latitude coordinate. If None, doppyo will attempt to determine lat_name \
                    automatically
        lon_name : str, optional
            Name of longitude coordinate. If None, doppyo will attempt to determine lon_name \
                    automatically
        plevel_name : str, optional
            Name of pressure level coordinate. If None, doppyo will attempt to determine plevel_name\
                    automatically
            
        Returns
        -------
        mmms : xarray DataArray
            New DataArray object containing the mean meridional mass stream function
        
        Examples
        --------
        >>> v = xr.DataArray(np.random.normal(size=(2,4,4)), 
        ...                   coords=[('level', [400, 600]), ('lat', np.arange(-90,90,45)), 
        ...                   ('lon', np.arange(0,360,90))])
        >>> doppyo.diagnostic.mean_merid_mass_streamfunction(v)
        <xarray.DataArray 'mmms' (lat: 4, level: 2)>
        array([[ 0.000000e+00, -1.336316e-07],
               [ 0.000000e+00, -1.447547e+10],
               [ 0.000000e+00, -3.208457e+09],
               [ 0.000000e+00, -2.562681e+10]])
        Coordinates:
          * lat      (lat) int64 -90 -45 0 45
          * level    (level) int64 400 600

        Notes
        -----------
        | All input array coordinates must follow standard naming (see ``doppyo.utils.get_lat_name()``, \
                ``doppyo.utils.get_lon_name()``, etc)
        | Pressure levels must be provided in units of hPa
    """
    
    degtorad = utils.constants().pi / 180

    if lat_name is None:
        lat_name = utils.get_lat_name(v)
    if lon_name is None:
        lon_name = utils.get_lon_name(v)
    if plevel_name is None:
        plevel_name = utils.get_plevel_name(v)
    cos_lat = xr.ufuncs.cos(v[lat_name] * degtorad) 

    v_Z = v.mean(dim=lon_name)
    
    return (2 * utils.constants().pi * utils.constants().R_earth * cos_lat * \
                utils.integrate(v_Z, over_dim=plevel_name, x=(v_Z[plevel_name] * 100), cumulative=True) \
                / utils.constants().g).rename('mmms')


# ===================================================================================================
def atmos_energy_cycle(temp, u, v, omega, gh, terms=None, vgradz=False, spectral=False, n_wavenumbers=20,
                       integrate=True, lat_name=None, lon_name=None, plevel_name=None):
    """
        Returns all terms in the Lorenz energy cycle. Follows formulae and notation used in `Marques \
                et al. 2011 Global diagnostic energetics of five state-of-the-art climate models. Climate \
                Dynamics`. Note that this decomposition is in the space domain. A space-time decomposition \
                can also be carried out (though not in Fourier space, but this is not implemented here (see \
                `Oort. 1964 On Estimates of the atmospheric energy cycle. Monthly Weather Review`).
        
        | Author: Dougie Squire
        | Date: 15/07/2018
        
        Parameters
        ----------
        temp : xarray DataArray
            Array containing fields of temperature with at least coordinates latitude, longitude \
                    and level (following standard naming - see Notes)
        u : xarray DataArray
            Array containing fields of zonal velocity with at least coordinates latitude, longitude \
                    and level (following standard naming - see Notes)
        v : xarray DataArray
            Array containing fields of meridional velocity with at least coordinates latitude, longitude \
                    and level (following standard naming - see Notes)
        omega : xarray DataArray
            Array containing fields of vertical velocity (pressure coordinates) with at least coordinates \
                    latitude, longitude and level (following standard naming - see Notes)
        gh : xarray DataArray
            Array containing fields of geopotential height with at least coordinates latitude, longitude \
                    and level (following standard naming - see Notes)
        terms : str or sequence of str
            | List of terms to compute. If None, returns all terms. Available options are:
            | **Pz**; total available potential energy in the zonally averaged temperature distribution
            | **Kz**; total kinetic energy in zonally averaged motion
            | **Pe**; total eddy available potential energy [= sum_n Pn (n > 0 only) for spectral=True] (Note that \
                    for spectral=True, an additional term, Sn, quantifying the rate of transfer of available potential \
                    energy to eddies of wavenumber n from eddies of all other wavenumbers is also returned)
            | **Ke**; total eddy kinetic energy [= sum_n Kn (n > 0 only) for spectral=True] (Note that for \
                    spectral=True, an additional term, Ln, quantifying the rate of transfer of kinetic energy to eddies \
                    of wavenumber n from eddies of all other wavenumbers is also returned)
            | **Cz**; rate of conversion of zonal available potential energy to zonal kinetic energy
            | **Ca**; rate of transfer of total available potential energy in the zonally averaged temperature \
                    distribution (Pz) to total eddy available potential energy (Pe) [= sum_n Rn (n > 0 only) for \
                    spectral=True]
            | **Ce**; rate of transfer of total eddy available potential energy (Pe) to total eddy kinetic energy \
                    (Ke) [= sum_n Cn (n > 0 only) for spectral=True]
            | **Ck**; rate of transfer of total eddy kinetic energy (Ke) to total kinetic energy in zonally \
                    averaged motion (Kz) [= sum_n Mn (n > 0 only) for spectral=True]
            | **Gz**; rate of generation of zonal available potential energy due to the zonally averaged heating (Pz). \
                    Note that this term is computed as a residual (Cz + Ca) and cannot be returned in spectral space. \
                    If Gz is requested with spectral=True, Gz is returned in real-space only
            | **Ge**; rate of generation of eddy available potential energy (Pe). Note that this term is computed as \
                    a residual (Ce - Ca) and cannot be returned in spectral space. If Ge is requested with spectral=True, \
                    Ge is returned in real-space only
            | **Dz**; rate of viscous dissipation of zonal kinetic energy (Kz). Note that this term is computed as a \
                    residual (Cz - Ck) and cannot be returned in spectral space. If Dz is requested with spectral=True, Dz \
                    is returned in real-space only
            | **De**; rate of dissipation of eddy kinetic energy (Ke). Note that this term is computed as a residual \
                    (Ce - Ck) and cannot be returned in spectral space. If De is requested with spectral=True, De is \
                    returned in real-space only
        vgradz : bool, optional
            If True, uses `v-grad-z` approach for computing terms relating to conversion of potential energy to \
                    kinetic energy. Otherwise, defaults to using the `omega-alpha` approach (see reference above for details)
        spectral : bool, optional
            If True, computes all terms as a function of wavenumber on longitudinal bands. To use this \
                    option, longitudes must be regularly spaced. Note that Ge and De are computed as residuals and \
                    cannot be computed in spectral space
        n_wavenumbers : int, optional
            Number of wavenumbers to retain either side of wavenumber=0. Obviously only does anything if \
                    spectral=True
        integrate : bool, optional
            If True, computes and returns the integral of each term over the mass of the atmosphere. Otherwise, \
                    only the integrands are returned
        lat_name : str, optional
            Name of latitude coordinate. If None, doppyo will attempt to determine lat_name \
                    automatically
        lon_name : str, optional
            Name of longitude coordinate. If None, doppyo will attempt to determine lon_name \
                    automatically
        plevel_name : str, optional
            Name of pressure level coordinate. If None, doppyo will attempt to determine plevel_name\
                    automatically

        Returns
        -------
        atmos_energy_cycle : xarray Dataset
            Dataset containing the requested variables plus gamma, the stability parameter. If integrate=True, \
                    both the integrand (<term>_int) and the integral over the mass of the atmosphere (<term>) are \
                    returned for each requested term. Otherwise, only the integrands are returned.
        
        Examples
        --------
        >>> temp = xr.DataArray(np.random.normal(size=(90,90,9,5)), 
        ...                     coords=[('lat', np.arange(-90,90,2)), ('lon', np.arange(0,360,4)), 
        ...                             ('level', np.arange(100,1000,100)), 
        ...                             ('time', pd.date_range('2000-01-01', periods=5, freq='M'))])
        >>> u = xr.DataArray(np.random.normal(size=(90,90,9,5)), 
        ...                  coords=[('lat', np.arange(-90,90,2)), ('lon', np.arange(0,360,4)), 
        ...                          ('level', np.arange(100,1000,100)), 
        ...                          ('time', pd.date_range('2000-01-01', periods=5, freq='M'))])
        >>> v = xr.DataArray(np.random.normal(size=(90,90,9,5)), 
        ...                  coords=[('lat', np.arange(-90,90,2)), ('lon', np.arange(0,360,4)), 
        ...                          ('level', np.arange(100,1000,100)), 
        ...                          ('time', pd.date_range('2000-01-01', periods=5, freq='M'))])
        >>> omega = xr.DataArray(np.random.normal(size=(90,90,9,5)), 
        ...                      coords=[('lat', np.arange(-90,90,2)), ('lon', np.arange(0,360,4)), 
        ...                              ('level', np.arange(100,1000,100)), 
        ...                              ('time', pd.date_range('2000-01-01', periods=5, freq='M'))])
        >>> gh = xr.DataArray(np.random.normal(size=(90,90,9,5)), 
        ...                   coords=[('lat', np.arange(-90,90,2)), ('lon', np.arange(0,360,4)), 
        ...                           ('level', np.arange(100,1000,100)), 
        ...                           ('time', pd.date_range('2000-01-01', periods=5, freq='M'))])
        >>> doppyo.diagnostic.atmos_energy_cycle(temp, u, v, omega, gh, spectral=True)
        <xarray.Dataset>
        Dimensions:  (lat: 90, level: 9, n: 41, time: 5)
        Coordinates:
          * level    (level) int64 100 200 300 400 500 600 700 800 900
          * time     (time) datetime64[ns] 2000-01-31 2000-02-29 ... 2000-05-31
          * lat      (lat) int64 -90 -88 -86 -84 -82 -80 -78 ... 76 78 80 82 84 86 88
          * n        (n) float64 -20.0 -19.0 -18.0 -17.0 -16.0 ... 17.0 18.0 19.0 20.0
        Data variables:
            gamma    (level, time) float64 8.993 120.3 68.1 ... -6.874 1.083 -1.383
            Pz_int   (level, time, lat) float64 83.76 64.07 67.67 ... -8.283 -0.7205
            Pz       (time) float64 -9.73e+04 8.225e+05 -1.892e+04 -4.197e+06 -9.113e+05
            Kz_int   (lat, level, time) float64 0.03326 0.01417 ... 0.01454 0.005276
            Kz       (time) float64 88.08 93.48 97.19 91.19 85.38
            Cz_int   (level, lat, time) float64 0.0001505 -6.762e-05 ... -3.222e-06
            Cz       (time) float64 0.01983 0.02128 -0.04917 -0.04136 -0.04431
            Pn_int   (level, time, lat, n) float64 109.5 163.6 132.4 ... -0.5592 -31.31
            Pn       (time, n) float64 -1.496e+05 -1.48e+05 ... -1.712e+06 -1.534e+06
            Sn_int   (level, time, n, lat) complex128 (-1.635e+12+1.365e+12j) ... (1.546e-04-3.464e-03j)
            Sn       (time, n) float64 54.46 42.84 39.39 10.27 ... 43.73 55.43 37.28
            Kn_int   (lat, n, level, time) float64 0.02795 0.02618 ... 0.0314 0.005119
            Kn       (n, time) float64 184.5 179.1 183.1 186.4 ... 183.1 186.4 176.6
            Ln_int   (n, lat, level, time) complex128 (-1.401e+09+4.623e+08j) ... (2.400e-06+9.272e-07j)
            Ln       (n, time) float64 7.325e-05 0.0001285 ... 8.57e-05 0.0001433
            Rn_int   (level, time, lat, n) complex128 (5.631e-03-1.433e-17j) ... (-3.295e-04+8.862e-19j)
            Rn       (time, n) float64 0.3357 0.5211 0.00877 ... 3.811 0.04257 3.209
            Cn_int   (level, lat, n, time) complex128 (-1.694e-04+4.232e-19j) ... (-1.836e-04+4.979e-19j)
            Cn       (n, time) float64 0.09795 0.04268 0.01845 ... 0.04054 0.03553
            Mn_int   (lat, n, level, time) complex128 (-1.376e+06+2.933e-09j) ... (5.344e-07-1.670e-21j)
            Mn       (n, time) float64 1.526e+06 8.963e+05 ... 4.038e+06 8.648e+05
            Gz_int   (level, lat, time) float64 -0.01321 -0.2201 ... -6.633e-05
            Gz       (time) float64 1.33 -10.62 27.5 -31.06 -10.44
            Ge_int   (level, lat, time) float64 0.01375 0.2213 ... 0.0007708 2.494e-05
            Ge       (time) float64 -1.011 10.69 -27.39 31.14 10.57
            Dz_int   (level, lat, time) float64 7.444e+07 -6.406e+07 ... -3.544e-06
            Dz       (time) float64 1.009e+07 6.951e+06 1.85e+07 1.491e+07 1.306e+07
            De_int   (level, lat, time) float64 7.444e+07 -6.406e+07 ... -3.849e-05
            De       (time) float64 1.009e+07 6.951e+06 1.85e+07 1.491e+07 1.306e+07
            
        Notes
        ===========
        | The following notation is used below (stackable, e.g. \*_ZT indicates the time average of the zonal average):
        | \*_A -> area average over an isobaric surface
        | \*_a -> departure from area average
        | \*_Z -> zonal average
        | \*_z -> departure from zonal average
        | \*_T -> time average
        | \*_t -> departure from time average
        | Additionally, capital variables indicate Fourier transforms:
        | F(u) = U
        | F(v) = V
        | F(omega) = O
        | F(gh) = A
        | F(temp) = B
        
        | All input array coordinates must follow standard naming (see ``doppyo.utils.get_lat_name()``, \
                ``doppyo.utils.get_lon_name()``, etc).
        | Pressure levels must be provided in units of hPa
        | The terms Sn and Ln, which are computed when Pe and Ke are requested with spectral=True, \
                rely on "triple terms" that are very intensive and can take a significant amount of time \
                and memory to compute (see _triple_terms() below). Often (i.e. for arrays of sufficient \
                size to be of interest) requesting these terms yeilds a MemoryError--if working in memory\
                --or a KilledWorkerError--if working out of memory
        
        To do
        
        - Arrays that are sufficiently large to be interesting currently max out the available memory when \
                Sn or Ln are requested. I need to implement a less hungry method for computing the "triple terms" \
                (see _triple_terms() below)
    """
    
    def _flip_n(da):
        """ Flips data along wavenumber coordinate """

        daf = da.copy()
        daf['n'] = -daf['n']

        return daf.sortby(daf['n'])


    def _truncate(F, n_truncate, dim):
        """ 
            Converts spatial frequency dim to wavenumber, n, and truncates all wavenumbers greater than 
            n_truncate 
        """
        F[dim] = 360 * F[dim]
        F = F.rename({dim : 'n'})
        F = F.where(abs(F.n) <= n_truncate, drop=True)
        return F, _flip_n(F)


    def _triple_terms(A, B, C):
        """ 
            Calculate triple term summation of the form \int_{m=-inf}^{inf} A(m) * B(n) * C(n - m)
        """

        # Use rolling operator to build shifted terms -----
        Am = A.rename({'n' : 'm'})
        Cnm = C.rolling(n=len(C.n), center=True).construct('m', fill_value=0)
        Cnm['m'] = -C['n'].values

        # Drop m = 0 and n < 0 -----
        Am = Am.where(Am['m'] != 0, drop=True) 
        Cnm = Cnm.where(Cnm['m'] != 0, drop=True)

        return (B * (Am * Cnm)).sum(dim='m', skipna=False)

    
    if terms is None:
        terms = ['Pz', 'Kz', 'Pe', 'Ke', 'Cz', 'Ca', 'Ce', 'Ck', 'Gz', 'Ge', 'Dz', 'De']
    if isinstance(terms, str):
        terms = [terms]
    
    # Initialize some things -----
    if lat_name is None:
        lat_name = utils.get_lat_name(temp)
    if lon_name is None:
        lon_name = utils.get_lon_name(temp)
    if plevel_name is None:
        plevel_name = utils.get_plevel_name(temp)
    
    degtorad = utils.constants().pi / 180
    tan_lat = xr.ufuncs.tan(temp[lat_name] * degtorad)
    cos_lat = xr.ufuncs.cos(temp[lat_name] * degtorad) 
    
    # Determine the stability parameter using Saltzman's approach -----
    kappa = utils.constants().R_d / utils.constants().C_pd
    p_kap = (1000 / temp[plevel_name]) ** kappa
    theta_A = utils.average(temp * p_kap, [lat_name, lon_name], weights=cos_lat)
    dtheta_Adp = utils.differentiate_wrt(theta_A, dim=plevel_name, x=(theta_A[plevel_name] * 100))
    gamma = - p_kap * (utils.constants().R_d) / ((temp[plevel_name] * 100) * utils.constants().C_pd) / dtheta_Adp # [1/K]
    energies = gamma.rename('gamma').to_dataset()
    
    # Compute zonal terms
    # ========================
    
    if ('Pz' in terms):
    # Compute the total available potential energy in the zonally averaged temperature
    # distribution, Pz [also commonly called Az] -----
        temp_A = utils.average(temp, [lat_name, lon_name], weights=cos_lat)
        temp_Z = temp.mean(dim=lon_name)
        temp_Za = temp_Z - temp_A
        Pz_int = gamma * utils.constants().C_pd / 2 * temp_Za ** 2  # [J/kg]
        energies['Pz_int'] = Pz_int
        if integrate:
            Pz = _int_over_atmos(Pz_int, lat_name, lon_name, plevel_name, lon_dim=temp[lon_name]) # [J/m^2]
            energies['Pz'] = Pz
    
    if ('Kz' in terms):
    # Compute the total kinetic energy in zonally averaged motion, Kz [also commonly 
    # called Kz] -----
        u_Z = u.mean(dim=lon_name)
        v_Z = v.mean(dim=lon_name)
        Kz_int = 0.5 * (u_Z ** 2 + v_Z ** 2) # [J/kg]
        energies['Kz_int'] = Kz_int
        if integrate:
            Kz = _int_over_atmos(Kz_int, lat_name, lon_name, plevel_name, lon_dim=u[lon_name]) # [J/m^2]
            energies['Kz'] = Kz
    
    if ('Cz' in terms):
    # Compute the rate of conversion of zonal available potential energy (Pz) to zonal kinetic
    # energy (Kz), Cz [also commonly called Cz] -----
        if vgradz:
            if 'v_Z' not in locals():
                v_Z = v.mean(dim=lon_name)
            gh_Z = gh.mean(dim=lon_name)
            dghdlat = utils.differentiate_wrt(gh_Z, dim=lat_name, x=(gh_Z[lat_name] * degtorad))
            Cz_int = - (utils.constants().g / utils.constants().R_earth) * v_Z * dghdlat # [W/kg]
            energies['Cz_int'] = Cz_int
            if integrate:
                Cz = _int_over_atmos(Cz_int, lat_name, lon_name, plevel_name, lon_dim=gh[lon_name]) # [W/m^2]
                energies['Cz'] = Cz
        else:
            if 'temp_Za' not in locals():
                temp_A = utils.average(temp, [lat_name, lon_name], weights=cos_lat)
                temp_Z = temp.mean(dim=lon_name)
                temp_Za = temp_Z - temp_A
            omega_A = utils.average(omega, [lat_name, lon_name], weights=cos_lat)
            omega_Z = omega.mean(dim=lon_name)
            omega_Za = omega_Z - omega_A
            Cz_int = - (utils.constants().R_d / (temp[plevel_name] * 100)) * omega_Za * temp_Za # [W/kg]
            energies['Cz_int'] = Cz_int
            if integrate:
                Cz = _int_over_atmos(Cz_int, lat_name, lon_name, plevel_name, lon_dim=omega[lon_name]) # [W/m^2]
                energies['Cz'] = Cz
    
    # Compute eddy terms in Fourier space if spectral=True
    # ==========================================================
    if spectral:
        
        if ('Pe' in terms):
        # Compute the total available potential energy eddies of wavenumber n, Pn -----
            Bp, Bn = _truncate(utils.fft(temp, dim=lon_name, nfft=len(temp[lon_name]), twosided=True, shift=True) / 
                              len(temp[lon_name]), n_truncate=n_wavenumbers, dim='f_'+lon_name)

            Pn_int = (gamma * utils.constants().C_pd * abs(Bp) ** 2)
            energies['Pn_int'] = Pn_int
            if integrate:
                Pn = _int_over_atmos(Pn_int, lat_name, lon_name, plevel_name, lon_dim=temp[lon_name]) # [J/m^2]
                energies['Pn'] = Pn

        # Compute the rate of transfer of available potential energy to eddies of 
        # wavenumber n from eddies of all other wavenumbers, Sn -----
            Up, Un = _truncate(utils.fft(u, dim=lon_name, nfft=len(u[lon_name]), twosided=True, shift=True) /
                               len(u[lon_name]), n_truncate=n_wavenumbers, dim='f_'+lon_name)
            Vp, Vn = _truncate(utils.fft(v, dim=lon_name, nfft=len(v[lon_name]), twosided=True, shift=True) /
                               len(v[lon_name]), n_truncate=n_wavenumbers, dim='f_'+lon_name)
            Op, On = _truncate(utils.fft(omega, dim=lon_name, nfft=len(omega[lon_name]), twosided=True, shift=True) /
                               len(omega[lon_name]), n_truncate=n_wavenumbers, dim='f_'+lon_name)
                
            dBpdlat = utils.differentiate_wrt(Bp, dim=lat_name, x=(Bp[lat_name] * degtorad))
            dBndlat = utils.differentiate_wrt(Bn, dim=lat_name, x=(Bn[lat_name] * degtorad))
            dBpdp = utils.differentiate_wrt(Bp, dim=plevel_name, x=(Bp[plevel_name] * 100))
            dBndp = utils.differentiate_wrt(Bn, dim=plevel_name, x=(Bn[plevel_name] * 100))

            BpBnUp = _triple_terms(Bp, Bn, Up)
            BpBpUn = _triple_terms(Bp, Bp, Un)
            BpglBnVp = _triple_terms(Bp, dBndlat, Vp)
            BpglBpVn = _triple_terms(Bp, dBpdlat, Vn)
            BpgpBnOp = _triple_terms(Bp, dBndp, Op)
            BpgpBpOn = _triple_terms(Bp, dBpdp, On)
            BpBnOp = _triple_terms(Bp, Bn, Op)
            BpBpOn = _triple_terms(Bp, Bp, On)

            Sn_int = -gamma * utils.constants().C_pd * (1j * Bp['n']) / \
                         (utils.constants().R_earth * xr.ufuncs.cos(Bp[lat_name] * degtorad)) * \
                         (BpBnUp + BpBpUn) + \
                     gamma * utils.constants().C_pd / utils.constants().R_earth * \
                         (BpglBnVp + BpglBpVn) + \
                     gamma * utils.constants().C_pd * (BpgpBnOp + BpgpBpOn) + \
                     gamma * utils.constants().R_d / Bp[plevel_name] * \
                         (BpBnOp + BpBpOn)
            energies['Sn_int'] = Sn_int
            if integrate:
                Sn = abs(_int_over_atmos(Sn_int, lat_name, lon_name, plevel_name, lon_dim=temp[lon_name])) # [W/m^2]
                energies['Sn'] = Sn
                
        if ('Ke' in terms):
        # Compute the total kinetic energy in eddies of wavenumber n, Kn -----
            if 'U' not in locals():
                Up, Un = _truncate(utils.fft(u, dim=lon_name, nfft=len(u[lon_name]), twosided=True, shift=True) /
                                   len(u[lon_name]), n_truncate=n_wavenumbers, dim='f_'+lon_name)
            if 'V' not in locals():
                Vp, Vn = _truncate(utils.fft(v, dim=lon_name, nfft=len(v[lon_name]), twosided=True, shift=True) / 
                                   len(v[lon_name]), n_truncate=n_wavenumbers, dim='f_'+lon_name)

            Kn_int = abs(Up) ** 2 + abs(Vp) ** 2
            energies['Kn_int'] = Kn_int
            if integrate:
                Kn = _int_over_atmos(Kn_int, lat_name, lon_name, plevel_name, lon_dim=u[lon_name]) # [J/m^2]
                energies['Kn'] = Kn

        # Compute the rate of transfer of kinetic energy to eddies of wavenumber n from 
        # eddies of all other wavenumbers, Ln -----
            if 'O' not in locals():
                Op, On = _truncate(utils.fft(omega, dim=lon_name, nfft=len(omega[lon_name]), twosided=True, shift=True) / 
                                   len(omega[lon_name]), n_truncate=n_wavenumbers, dim='f_'+lon_name)
                
            dUpdp = utils.differentiate_wrt(Up, dim=plevel_name, x=(Up[plevel_name] * 100))
            dVpdp = utils.differentiate_wrt(Vp, dim=plevel_name, x=(Vp[plevel_name] * 100))
            dOpdp = utils.differentiate_wrt(Op, dim=plevel_name, x=(Op[plevel_name] * 100))
            dOndp = utils.differentiate_wrt(On, dim=plevel_name, x=(On[plevel_name] * 100))
            dVpcdl = utils.differentiate_wrt(Vp * cos_lat, dim=lat_name, x=(Vp[lat_name] * degtorad))
            dVncdl = utils.differentiate_wrt(Vn * cos_lat, dim=lat_name, x=(Vn[lat_name] * degtorad))
            dUpdl = utils.differentiate_wrt(Up, dim=lat_name, x=(Up[lat_name] * degtorad))
            dVpdl = utils.differentiate_wrt(Vp, dim=lat_name, x=(Vp[lat_name] * degtorad))

            UpUnUp = _triple_terms(Up, Un, Up)
            UpUpUn = _triple_terms(Up, Up, Un)
            VpVnUp = _triple_terms(Vp, Vn, Up)
            VpVpUn = _triple_terms(Vp, Vp, Un)
            VpUnUp = _triple_terms(Vp, Un, Up)
            VpUpUn = _triple_terms(Vp, Up, Un)
            UpVnUp = _triple_terms(Up, Vn, Up)
            UpVpUn = _triple_terms(Up, Vp, Un)
            gpUpUngpOp = _triple_terms(dUpdp, Un, dOpdp)
            gpUpUpgpOn = _triple_terms(dUpdp, Up, dOndp)
            gpVpVngpOp = _triple_terms(dVpdp, Vn, dOpdp)
            gpVpVpgpOn = _triple_terms(dVpdp, Vp, dOndp)
            glUpUnglVpc = _triple_terms(dUpdl, Un, dVpcdl)
            glUpUpglVnc = _triple_terms(dUpdl, Up, dVncdl)
            glVpVnglVpc = _triple_terms(dVpdl, Vn, dVpcdl)
            glVpVpglVnc = _triple_terms(dVpdl, Vp, dVncdl)

            Ln_int = -(1j * Up['n']) / (utils.constants().R_earth * cos_lat) * \
                         (UpUnUp - UpUpUn) + \
                     (1j * Vp['n']) / (utils.constants().R_earth * cos_lat) * \
                         (VpVnUp - VpVpUn) - \
                     tan_lat / utils.constants().R_earth * \
                         (VpUnUp + VpUpUn) + \
                     tan_lat / utils.constants().R_earth * \
                         (UpVnUp + UpVpUn) + \
                     (gpUpUngpOp + gpUpUpgpOn) + \
                     (gpVpVngpOp + gpVpVpgpOn) + \
                     1 / (utils.constants().R_earth * cos_lat) * \
                         (glUpUnglVpc + glUpUpglVnc + glVpVnglVpc + glVpVpglVnc)
            energies['Ln_int'] = Ln_int
            if integrate:
                Ln = abs(_int_over_atmos(Ln_int, lat_name, lon_name, plevel_name, lon_dim=u[lon_name])) # [W/m^2]
                energies['Ln'] = Ln
        
        if ('Ca' in terms):
        # Compute the rate of transfer of zonal available potential energy to eddy 
        # available potential energy in wavenumber n, Rn -----
            if 'temp_Z' not in locals():
                temp_Z = temp.mean(dim=lon_name)
            if 'V' not in locals():
                Vp, Vn = _truncate(utils.fft(v, dim=lon_name, nfft=len(v[lon_name]), twosided=True, shift=True) / 
                                   len(v[lon_name]), n_truncate=n_wavenumbers, dim='f_'+lon_name)
            if 'B' not in locals():
                Bp, Bn = _truncate(utils.fft(temp, dim=lon_name, nfft=len(temp[lon_name]), twosided=True, shift=True) / 
                                   len(temp[lon_name]), n_truncate=n_wavenumbers, dim='f_'+lon_name)
            if 'O' not in locals():
                Op, On = _truncate(utils.fft(omega, dim=lon_name, nfft=len(omega[lon_name]), twosided=True, shift=True) / 
                                   len(omega[lon_name]), n_truncate=n_wavenumbers, dim='f_'+lon_name)

            dtemp_Zdlat = utils.differentiate_wrt(temp_Z, dim=lat_name, x=(temp_Z[lat_name] * degtorad))
            theta = temp * p_kap
            theta_Z = theta.mean(dim=lon_name)
            theta_Za = theta_Z - theta_A
            dtheta_Zadp = utils.differentiate_wrt(theta_Za, dim=plevel_name, x=(theta_Za[plevel_name] * 100))
            Rn_int = gamma * utils.constants().C_pd * ((dtemp_Zdlat / utils.constants().R_earth) * (Vp * Bn + Vn * Bp) + 
                                                       (p_kap * dtheta_Zadp) * (Op * Bn + On * Bp)) # [W/kg]
            energies['Rn_int'] = Rn_int
            if integrate:
                Rn = abs(_int_over_atmos(Rn_int, lat_name, lon_name, plevel_name, lon_dim=temp[lon_name])) # [W/m^2]
                energies['Rn'] = Rn

        if ('Ce' in terms):
        # Compute the rate of conversion of available potential energy of wavenumber n 
        # to eddy kinetic energy of wavenumber n, Cn -----
            if vgradz:
                if 'U' not in locals():
                    Up, Un = _truncate(utils.fft(u, dim=lon_name, nfft=len(u[lon_name]), twosided=True, shift=True) / 
                                       len(u[lon_name]), n_truncate=n_wavenumbers, dim='f_'+lon_name)
                if 'V' not in locals():
                    Vp, Vn = _truncate(utils.fft(v, dim=lon_name, nfft=len(v[lon_name]), twosided=True, shift=True) / 
                                       len(v[lon_name]), n_truncate=n_wavenumbers, dim='f_'+lon_name)
                Ap, An = _truncate(utils.fft(gh, dim=lon_name, nfft=len(gh[lon_name]), twosided=True, shift=True) / 
                                   len(gh[lon_name]), n_truncate=n_wavenumbers, dim='f_'+lon_name)

                dApdlat = utils.differentiate_wrt(Ap, dim=lat_name, x=(Ap[lat_name] * degtorad))
                dAndlat = utils.differentiate_wrt(An, dim=lat_name, x=(An[lat_name] * degtorad))

                Cn_int = (((-1j * utils.constants().g * Up['n']) / \
                           (utils.constants().R_earth * xr.ufuncs.cos(Up[lat_name] * degtorad))) * \
                                (Ap * Un - An * Up)) - \
                         ((utils.constants().g / utils.constants().R_earth) * \
                                (dApdlat * Vn + dAndlat * Vp)) # [W/kg]
                energies['Cn_int'] = Cn_int
                if integrate:
                    Cn = abs(_int_over_atmos(Cn_int, lat_name, lon_name, plevel_name, lon_dim=u[lon_name])) # [W/m^2]
                    energies['Cn'] = Cn
            else:
                if 'O' not in locals():
                    Op, On = _truncate(utils.fft(omega, dim=lon_name, nfft=len(omega[lon_name]), twosided=True, shift=True) / 
                                       len(omega[lon_name]), n_truncate=n_wavenumbers, dim='f_'+lon_name)
                if 'B' not in locals():
                    Bp, Bn = _truncate(utils.fft(temp, dim=lon_name, nfft=len(temp[lon_name]), twosided=True, shift=True) / 
                                       len(temp[lon_name]), n_truncate=n_wavenumbers, dim='f_'+lon_name)
                Cn_int = - (utils.constants().R_d / (omega[plevel_name] * 100)) * (Op * Bn + On * Bp) # [W/kg]
                energies['Cn_int'] = Cn_int
                if integrate:
                    Cn = abs(_int_over_atmos(Cn_int, lat_name, lon_name, plevel_name, lon_dim=temp[lon_name])) # [W/m^2]
                    energies['Cn'] = Cn
    
        if ('Ck' in terms):
        # Compute the rate of transfer of kinetic energy to the zonally averaged flow 
        # from eddies of wavenumber n, Mn -----
            if 'v_Z' not in locals():
                v_Z = v.mean(dim=lon_name)
            if 'u_Z' not in locals():
                u_Z = u.mean(dim=lon_name)
            if 'U' not in locals():
                Up, Un = _truncate(utils.fft(u, dim=lon_name, nfft=len(u[lon_name]), twosided=True, shift=True) / 
                                   len(u[lon_name]), n_truncate=n_wavenumbers, dim='f_'+lon_name)
            if 'V' not in locals():
                Vp, Vn = _truncate(utils.fft(v, dim=lon_name, nfft=len(v[lon_name]), twosided=True, shift=True) / 
                                   len(v[lon_name]), n_truncate=n_wavenumbers, dim='f_'+lon_name)
            if 'O' not in locals():
                Op, On = _truncate(utils.fft(omega, dim=lon_name, nfft=len(omega[lon_name]), twosided=True, shift=True) / 
                                   len(omega[lon_name]), n_truncate=n_wavenumbers, dim='f_'+lon_name)
            dv_Zdlat = utils.differentiate_wrt(v_Z, dim=lat_name, x=(v[lat_name] * degtorad))
            du_Zndlat = utils.differentiate_wrt(u_Z / xr.ufuncs.cos(u[lat_name] * degtorad), 
                                            dim=lat_name, x=(u[lat_name] * degtorad))
            dv_Zdp = utils.differentiate_wrt(v_Z, dim=plevel_name, x=(v[plevel_name] * 100))
            du_Zdp = utils.differentiate_wrt(u_Z, dim=plevel_name, x=(u[plevel_name] * 100))

            Mn_int = (-2 * Up * Un * v_Z * tan_lat / utils.constants().R_earth) + \
                     (2 * Vp * Vn * dv_Zdlat / utils.constants().R_earth + (Vp * On + Vn * Op) * dv_Zdp) + \
                     ((Up * On + Un * Op) * du_Zdp) + \
                     ((Up * Vn + Un * Vp) * xr.ufuncs.cos(u[lat_name] * degtorad) / \
                         utils.constants().R_earth * du_Zndlat) # [W/kg]
            energies['Mn_int'] = Mn_int
            if integrate:
                Mn = abs(_int_over_atmos(Mn_int, lat_name, lon_name, plevel_name, lon_dim=u[lon_name])) # [W/m^2]
                energies['Mn'] = Mn
        
    else:
        
        if ('Pe' in terms):
        # Compute the total eddy available potential energy, Pe [also commonly called 
        # Ae] -----
            if 'temp_Z' not in locals():
                temp_Z = temp.mean(dim=lon_name)
            temp_z = temp - temp_Z
            Pe_int = gamma * utils.constants().C_pd / 2 * (temp_z ** 2).mean(dim=lon_name)  # [J/kg]
            energies['Pe_int'] = Pe_int
            if integrate:
                Pe = _int_over_atmos(Pe_int, lat_name, lon_name, plevel_name, lon_dim=temp[lon_name]) # [J/m^2]
                energies['Pe'] = Pe
        
        if ('Ke' in terms):
        # Compute the total eddy kinetic energy, Ke -----
            if 'u_Z' not in locals():
                    u_Z = u.mean(dim=lon_name)
            if 'v_Z' not in locals():
                    v_Z = v.mean(dim=lon_name)
            u_z = u - u_Z
            v_z = v - v_Z
            Ke_int = 0.5 * (u_z ** 2 + v_z ** 2).mean(dim=lon_name) # [J/kg]
            energies['Ke_int'] = Ke_int
            if integrate:
                Ke = _int_over_atmos(Ke_int, lat_name, lon_name, plevel_name, lon_dim=u[lon_name]) # [J/m^2]
                energies['Ke'] = Ke
                
        if ('Ca' in terms):
        # Compute the rate of transfer of total available potential energy in the zonally 
        # averaged temperature distribution (Pz) to total eddy available potential energy 
        # (Pe), Ca -----
            if 'v_Z' not in locals():
                v_Z = v.mean(dim=lon_name)
            if 'temp_Z' not in locals():
                temp_Z = temp.mean(dim=lon_name)
            if 'omega_Z' not in locals():
                omega_Z = omega.mean(dim=lon_name)
            if 'theta_Z' not in locals():
                theta = temp * p_kap
                theta_Z = theta.mean(dim=lon_name)
            if 'dtemp_Zdlat' not in locals():
                dtemp_Zdlat = utils.differentiate_wrt(temp_Z, dim=lat_name, x=(temp_Z[lat_name] * degtorad))
            v_z = v - v_Z
            temp_z = temp - temp_Z
            omega_z = omega - omega_Z
            oT_Z = (omega_z * temp_z).mean(dim=lon_name)
            oT_A = utils.average(omega_z * temp_z, [lat_name, lon_name], weights=cos_lat)
            oT_Za = oT_Z - oT_A
            theta_Za = theta_Z - theta_A
            dtheta_Zadp = utils.differentiate_wrt(theta_Za, dim=plevel_name, x=(theta_Za[plevel_name] * 100))
            Ca_int = - gamma * utils.constants().C_pd * \
                           (((v_z * temp_z).mean(dim=lon_name) * dtemp_Zdlat / utils.constants().R_earth) + \
                            (p_kap * oT_Za * dtheta_Zadp)) # [W/kg]
            energies['Ca_int'] = Ca_int
            if integrate:
                Ca = _int_over_atmos(Ca_int, lat_name, lon_name, plevel_name, lon_dim=v[lon_name]) # [W/m^2]
                energies['Ca'] = Ca
            
        if ('Ce' in terms):
        # Compute the rate of transfer of total eddy available potential energy (Pe) to 
        # total eddy kinetic energy (Ke), Ce -----
            if 'temp_Z' not in locals():
                temp_Z = temp.mean(dim=lon_name)
            if 'omega_Z' not in locals():
                omega_Z = omega.mean(dim=lon_name)
            temp_z = temp - temp_Z
            omega_z = omega - omega_Z
            Ce_int = - (utils.constants().R_d / (temp[plevel_name] * 100)) * \
                           (omega_z * temp_z).mean(dim=lon_name) # [W/kg]  
            energies['Ce_int'] = Ce_int
            if integrate:
                Ce = _int_over_atmos(Ce_int, lat_name, lon_name, plevel_name, lon_dim=temp[lon_name]) # [W/m^2]
                energies['Ce'] = Ce
        
        if ('Ck' in terms):
        # Compute the rate of transfer of total eddy kinetic energy (Ke) to total kinetic 
        # energy in zonally averaged motion (Kz), Ck -----
            if 'u_Z' not in locals():
                    u_Z = u.mean(dim=lon_name)
            if 'v_Z' not in locals():
                    v_Z = v.mean(dim=lon_name)
            if 'omega_Z' not in locals():
                omega_Z = omega.mean(dim=lon_name)
            u_z = u - u_Z
            v_z = v - v_Z
            omega_z = omega - omega_Z
            du_Zndlat = utils.differentiate_wrt(u_Z / cos_lat, dim=lat_name, x=(u_Z[lat_name] * degtorad))
            dv_Zdlat = utils.differentiate_wrt(v_Z, dim=lat_name, x=(v_Z[lat_name] * degtorad))
            du_Zdp = utils.differentiate_wrt(u_Z, dim=plevel_name, x=(u_Z[plevel_name] * 100))
            dv_Zdp = utils.differentiate_wrt(v_Z, dim=plevel_name, x=(v_Z[plevel_name] * 100))
            Ck_int = (u_z * v_z).mean(dim=lon_name)  * cos_lat * du_Zndlat / utils.constants().R_earth + \
                     (u_z * omega_z).mean(dim=lon_name) * du_Zdp + \
                     (v_z ** 2).mean(dim=lon_name) * dv_Zdlat / utils.constants().R_earth + \
                     (v_z * omega_z).mean(dim=lon_name) * dv_Zdp - \
                     (u_z ** 2).mean(dim=lon_name) * v_Z * tan_lat / utils.constants().R_earth
            energies['Ck_int'] = Ck_int
            if integrate:
                Ck = _int_over_atmos(Ck_int, lat_name, lon_name, plevel_name, lon_dim=temp[lon_name]) # [W/m^2]
                energies['Ck'] = Ck
                
    if ('Gz' in terms):
    # Compute the rate of generation of zonal available potential energy due to the zonally
    # averaged heating, Gz -----
        if ('Cz' not in terms) | ('Ca' not in terms):
            raise ValueError('The rate of generation of zonal available potential energy, Gz, is computed from the sum of Cz and Ca. Please add these to the list, terms=[<terms>].')
        if spectral:
            warnings.warn('Rate of generation of zonal available potential energy is computed from the sum of Cz and Ca and cannot be computed in Fourier space. Returning Gz in real-space.')
            Ca_int = Rn_int.where(Rn_int.n > 0, drop=True).sum(dim='n').real # sum Rn to get Ca
        Gz_int = Cz_int + Ca_int
        energies['Gz_int'] = Gz_int
        if integrate:
            Gz = _int_over_atmos(Gz_int, lat_name, lon_name, plevel_name, lon_dim=temp[lon_name]) # [W/m^2]
            energies['Gz'] = Gz

    if ('Ge' in terms):
    # Compute the rate of generation of eddy available potential energy (Ae), Ge -----
        if ('Ce' not in terms) | ('Ca' not in terms):
            raise ValueError('The rate of generation of eddy available potential energy, Ge, is computed from the residual of Ce and Ca. Please add these to the list, terms=[<terms>].')
        if spectral:
            warnings.warn('The rate of generation of eddy available potential energy is computed from the residual of Ce and Ca and cannot be computed in Fourier space. Returning Ge in real-space.')
            Ce_int = Cn_int.where(Cn_int.n > 0, drop=True).sum(dim='n').real # sum Cn to get Ce
            if 'Ca_int' not in locals():
                Ca_int = Rn_int.where(Rn_int.n > 0, drop=True).sum(dim='n').real # sum Rn to get Ca
        Ge_int = Ce_int - Ca_int
        energies['Ge_int'] = Ge_int
        if integrate:
            Ge = _int_over_atmos(Ge_int, lat_name, lon_name, plevel_name, lon_dim=temp[lon_name]) # [W/m^2]
            energies['Ge'] = Ge
    
    if ('Dz' in terms):
    # Compute the rate of viscous dissipation of zonal kinetic energy, Dz -----
        if ('Cz' not in terms) | ('Ck' not in terms):
            raise ValueError('The rate of viscous dissipation of zonal kinetic energy, Dz, is computed from the residual of Cz and Ck. Please add these to the list, terms=[<terms>].')
        if spectral:   
            warnings.warn('The rate of viscous dissipation of zonal kinetic energy, Dz, is computed from the residual of Cz and Ck and cannot be computed in Fourier space. Returning De in real-space.')
            Ck_int = Mn_int.where(Mn_int.n > 0, drop=True).sum(dim='n').real # sum Mn to get Ck
        Dz_int = Cz_int - Ck_int
        energies['Dz_int'] = Dz_int
        if integrate:
            Dz = _int_over_atmos(Dz_int, lat_name, lon_name, plevel_name, lon_dim=temp[lon_name]) # [W/m^2]
            energies['Dz'] = Dz

    if ('De' in terms):
    # Compute the rate of dissipation of eddy kinetic energy (Ke), De -----
        if ('Ce' not in terms) | ('Ck' not in terms):
            raise ValueError('The rate of viscous dissipation of eddy kinetic energy, De, is computed from the residual of Ce and Ck. Please add these to the list, terms=[<terms>].')
        if spectral:
            warnings.warn('The rate of viscous dissipation of eddy kinetic energy, De, is computed from the residual of Ce and Ck and cannot be computed in Fourier space. Returning De in real-space.')
            if 'Ce_int' not in locals():
                Ce_int = Cn_int.where(Cn_int.n > 0, drop=True).sum(dim='n').real # sum Cn to get Ce
            if 'Ck_int' not in locals():
                Ck_int = Mn_int.where(Mn_int.n > 0, drop=True).sum(dim='n').real # sum Mn to get Ck
        De_int = Ce_int - Ck_int
        energies['De_int'] = De_int
        if integrate:
            De = _int_over_atmos(De_int, lat_name, lon_name, plevel_name, lon_dim=temp[lon_name]) # [W/m^2]
            energies['De'] = De
    
    return energies
        

# ===================================================================================================
# Ocean diagnostics
# ===================================================================================================
def isotherm_depth(temp, target_temp=20, depth_name=None):
    """ 
        Returns the depth of an isotherm given a target temperature. If no temperatures in the column
        exceed the target temperature, a nan is returned at that point
        
        | Author: Thomas Moore
        | Date: 02/10/2018
        
        Parameters
        ----------
        temp : xarray DataArray
            Array containing values of temperature with at least a depth dimension
        target_temp : value, optional
            Value of temperature used to compute isotherm depth. Default value is 20 degC
        depth_name : str, optional
            Name of depth coordinate. If None, doppyo will attempt to determine depth_name \
                    automatically
            
        Returns
        -------
        isotherm_depth : xarray DataArray
            Array containing the depth of the requested isotherm

        Examples
        --------
        >>> temp = xr.DataArray(20 + np.random.normal(scale=5, size=(4,4,10)), 
        ...                     coords=[('lat', np.arange(-90,90,45)), ('lon', np.arange(0,360,90)), 
        ...                             ('depth', np.arange(2000,0,-200))])
        >>> doppyo.diagnostic.isotherm_depth(temp)
        <xarray.DataArray 'isosurface' (lat: 4, lon: 4)>
        array([[ 400., 1600., 2000.,  800.],
               [1800., 2000., 1800., 2000.],
               [2000., 2000., 2000., 1600.],
               [1400., 2000., 2000., 2000.]])
        Coordinates:
          * lat      (lat) int64 -90 -45 0 45
          * lon      (lon) int64 0 90 180 270
        
        Notes
        -----------
        | All input array coordinates must follow standard naming (see ``doppyo.utils.get_lat_name()``, \
                ``doppyo.utils.get_lon_name()``, etc)
        | If multiple occurences of target occur along the depth coordinate, only the maximum value of \
                coord is returned
        | The current version includes no interpolation between grid spacing. This should be added as \
                an option in the future
    """

    if depth_name is None:
        depth_name = utils.get_depth_name(temp)

    return utils.isosurface(temp, coord=depth_name, target=target_temp).rename('isotherm_depth')


# ===================================================================================================
# General diagnostics
# ===================================================================================================
def pwelch(da1, da2, dim, nwindow, overlap=50, dx=None, hanning=False):
    """
        Compute the cross/power spectral density along a dimension using welch's method. Note that \
                the spectral density is always computed relative to a "frequency" f = 1/dx (see Notes for \
                details)
        
        | Author: Dougie Squire
        | Date: 20/07/2018
        
        Parameters
        ----------
        da1 : xarray DataArray
            First array of data to use in spectral density calculation. For power spectral density, \
                    da1 = da2
        da2 : xarray DataArray
            Second array of data to use in spectral density calculation. For power spectral density, \
                    da1 = da2
        dim : str
            Dimension to compute spectral density along
        nwindow : value
            Length of the signal segments for pwelch calculation
        overlap : value, optional
            Percentage overlap of the signal segments for pwelch calculation
        dx : value, optional
            Spacing along the dimension dim. If None, dx is determined from the coordinate dim. For \
                    consistency between spatial and temporal dim, spectra is computed relative to a "frequency", \
                    f = 1/dx, where dx is the spacing along dim, e.g.:
            
            - for temporal dim, dx is computed in seconds. Thus, f = 1/seconds = Hz
            - for spatial dim in meters, f = 1/meters = k/(2*pi) 
            - for spatial dim in degrees, f = 1/degrees = k/360
            
            If converting the "frequency" to wavenumber, for example, one must also adjust the spectra \
                    magnitude so that the integral remains equal to the variance, e.g. for spatial spectra,\
                    k = f*(2*pi)  ->  phi_new = phi_old/(2*pi)
        hanning : bool, optional
            If True, a Hanning window weighting is applied prior to the fft
            
        Returns
        -------
        spectra : xarray DataArray
            Array containing the power spectral density of the input array(s)
            
        Examples
        --------
        >>> u = xr.DataArray(np.random.normal(size=(500)), 
        ...                  coords=[('time', pd.date_range('2000-01-01', periods=500, freq='D'))])
        >>> spectra = doppyo.diagnostic.pwelch(u, u, dim='time', nwindow=20)
        >>> seconds_to_days = 60*60*24
        >>> spectra['f_time'] = seconds_to_days*spectra['f_time'] # Change freq from Hz to 1/days
        >>> spectra = spectra/seconds_to_days
        >>> print(spectra)
        <xarray.DataArray 'spectra' (f_time: 11)>
        array([1.381912, 1.89882 , 1.641378, 1.939686, 2.218824, 1.941639, 2.142277,
               1.689319, 1.983302, 2.225709, 2.732023])
        Coordinates:
          * f_time   (f_time) float64 0.0 0.05 0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.45 0.5
    """

    # Force nwindow to be even -----
    if nwindow % 2 != 0:
        nwindow = nwindow - 1
        
    if not da1.coords.to_dataset().equals(da2.coords.to_dataset()):
            raise ValueError('da1 and da2 coordinates do not match')

    # Determine dx if not provided -----
    if dx is None:
        diff = da1[dim].diff(dim)
        if utils._is_datetime(da1[dim].values):
            # Drop differences on leap days so that still works with 'noleap' calendars -----
            diff = diff.where((diff[dim].dt.month != 3) & (diff[dim].dt.day != 1), drop=True)
        if np.all(diff == diff[0]):
            if utils._is_datetime(da1[dim].values):
                dx = diff.values[0] / np.timedelta64(1, 's')
            else:
                dx = diff.values[0]
        else:
            raise ValueError(f'Coordinate {dim} must be regularly spaced to compute fft')

    # Use rolling operator to break into overlapping windows -----
    stride = int(((100-overlap) / 100) * nwindow)
    da1_windowed = da1.rolling(**{dim:nwindow}, center=True).construct('fft_dim', stride=stride)
    da2_windowed = da2.rolling(**{dim:nwindow}, center=True).construct('fft_dim', stride=stride)
    
    # Only keep completely filled windows -----
    if nwindow == len(da1[dim]):
        da1_windowed = da1
        da1_windowed = da1_windowed.expand_dims('n')
        da2_windowed = da2
        da2_windowed = da2_windowed.expand_dims('n')
    else:
        da1_windowed = da1_windowed.isel({dim : range(max([int(np.floor(nwindow / stride / 2)), 1]),
                                                      len(da1_windowed[dim]) - 
                                                          max([int(np.floor(nwindow / stride / 2)), 1]))}) \
                                   .rename({dim : 'n'})
        da2_windowed = da2_windowed.isel({dim : range(max([int(np.floor(nwindow / stride / 2)), 1]),
                                                      len(da2_windowed[dim]) - 
                                                          max([int(np.floor(nwindow / stride / 2)), 1]))}) \
                                   .rename({dim : 'n'})
        da1_windowed['fft_dim'] = da1[dim][:len(da1_windowed['fft_dim'])].values
        da1_windowed = da1_windowed.rename({'fft_dim' : dim})
        da2_windowed['fft_dim'] = da2[dim][:len(da2_windowed['fft_dim'])].values
        da2_windowed = da2_windowed.rename({'fft_dim' : dim})

    # Apply weight to windows if specified -----
    if hanning:
        hwindow = xr.DataArray(np.hanning(nwindow), coords={dim : da1_windowed[dim]}, dims=[dim])
        weight_numer = (da1_windowed * da2_windowed).mean(dim)
        da1_windowed = hwindow * da1_windowed
        da2_windowed = hwindow * da2_windowed
        weight_denom = (da1_windowed * da2_windowed).mean(dim)
        weight = weight_numer / weight_denom # Account for effect of Hanning window on energy (8/3 in theory)
    else:
        weight = 1

    # Compute the spectral density -----
    da1_fft = utils.fft(da1_windowed, dim=dim, dx=dx)
    da2_fftc = xr.ufuncs.conj(utils.fft(da2_windowed, dim=dim, dx=dx))

    return (weight * 2 * dx * (da1_fft * da2_fftc).mean('n') / nwindow).real.rename('spectra')


# ===================================================================================================
def inband_variance(da, dim, bounds, nwindow, overlap=50):
    """ 
        Compute the in-band variance along a specified dimension. 
        
        | Author: Dougie Squire
        | Date: 20/07/2018
        
        Parameters
        ----------
        da : xarray DataArray
            Array with which to compute in-band variance
        dim : str
            Dimension along which to compute in-band variance
        bounds : sequence
            Frequency bounds for in-band variance calculation. Note that for consistency between \
                    spatial and temporal dim, all spectra are computed relative to a "frequency", f = 1/dx, \
                    where dx is the spacing along dim, e.g.:
            
            - for temporal dim, dx is computed in seconds. Thus, f = 1/seconds = Hz
            - for spatial dim in meters, f = 1/meters = k/(2*pi) 
            - for spatial dim in degrees, f = 1/degrees = k/360
            
            Thus, bounds must be provided in a way consistent with this, e.g.:
            
            - for temporal dim, bounds = 1 / (60*60*24*[d1, d2, d3]), where d# are numbers of days
            - for spatial dim, bounds = 1 / [l1, l2, l3], where l# are numbers of meters, degrees, etc
        nwindow : value
            Length of the signal segments for pwelch calculation
        overlap : value, optional
            Percentage overlap of the signal segments for pwelch calculation
        
        Returns
        -------
        inband_var : xarray DataArray
            Array containing the in-band variances of the input array
            
        Examples
        --------
        >>> A = xr.DataArray(np.random.normal(size=(500)), 
        ...                  coords=[('time', pd.date_range('2000-01-01', periods=500, freq='D'))])
        >>> doppyo.diagnostic.inband_variance(A, dim='time', 
        ...                                   bounds=1/(60*60*24*np.array([2, 5, 10])), nwindow=20)
        <xarray.DataArray 'in-band' (f_time_bins: 2)>
        array([0.106615, 0.492033])
        Coordinates:
          * f_time_bins  (f_time_bins) object [1.16e-06, 2.31e-06) [2.31e-06, 5.79e-06)
    """
    
    bounds = np.sort(bounds)
    spectra = pwelch(da, da, dim=dim, nwindow=nwindow, overlap=overlap)
    dx = spectra['f_'+dim].diff('f_'+dim).values[0]
    bands = spectra.groupby_bins('f_'+dim, bounds, right=False)
    
    return bands.apply(utils.integrate, over_dim='f_'+dim, dx=dx).rename('inband_var')


# ===================================================================================================
# Indices
# ===================================================================================================
def nino3(sst_anom):
    """ 
        Returns Nino-3 index 
        
        | Author: Dougie Squire
        | Date: 10/04/2018
        
        Parameters
        ----------
        sst_anom : xarray DataArray
            Array containing sea surface temperature anomalies
            
        Returns
        -------
        nino3 : xarray DataArray
            Average of the provided sst anomalies over the nino-3 box
            
        Examples
        --------
        >>> sst = xr.DataArray(np.random.normal(size=(90,90,24)), 
        ...                    coords=[('lat', np.arange(-90,90,2)), ('lon', np.arange(0,360,4)), 
        ...                            ('time', pd.date_range('2000-01-01', periods=24, freq='M'))])
        >>> sst_clim = sst.groupby('time.month').mean(dim='time')
        >>> sst_anom = doppyo.utils.anomalize(sst, sst_clim)
        >>> doppyo.diagnostic.nino3(sst_anom)
        <xarray.DataArray 'nino3' (time: 24)>
        array([-0.137317, -0.094808, -0.01091 , -0.04653 ,  0.030562, -0.065515,
               -0.109851,  0.118016,  0.092496, -0.030821, -0.011724, -0.002773,
                0.137317,  0.094808,  0.01091 ,  0.04653 , -0.030562,  0.065515,
                0.109851, -0.118016, -0.092496,  0.030821,  0.011724,  0.002773])
        Coordinates:
          * time     (time) datetime64[ns] 2000-01-31 2000-02-29 ... 2001-12-31
    """ 
    
    box = [-5.0,5.0,210.0,270.0] # [lat_min,lat_max,lon_min,lon_max]
        
    return utils.latlon_average(sst_anom, box).rename('nino3')


# ===================================================================================================
def nino34(sst_anom):
    """ 
        Returns Nino-3.4 index 
        
        | Author: Dougie Squire
        | Date: 10/04/2018
        
        Parameters
        ----------
        sst_anom : xarray DataArray
            Array containing sea surface temperature anomalies
            
        Returns
        -------
        nino34 : xarray DataArray
            Average of the provided sst anomalies over the nino-3.4 box
            
        Examples
        --------
        >>> sst = xr.DataArray(np.random.normal(size=(90,90,24)), 
        ...                    coords=[('lat', np.arange(-90,90,2)), ('lon', np.arange(0,360,4)), 
        ...                            ('time', pd.date_range('2000-01-01', periods=24, freq='M'))])
        >>> sst_clim = sst.groupby('time.month').mean(dim='time')
        >>> sst_anom = doppyo.utils.anomalize(sst, sst_clim)
        >>> doppyo.diagnostic.nino34(sst_anom)
        <xarray.DataArray 'nino34' (time: 24)>
        array([-0.052202,  0.00467 ,  0.121013,  0.007983, -0.070645,  0.051945,
               -0.045485,  0.065569, -0.018723, -0.053734,  0.10527 , -0.113451,
                0.052202, -0.00467 , -0.121013, -0.007983,  0.070645, -0.051945,
                0.045485, -0.065569,  0.018723,  0.053734, -0.10527 ,  0.113451])
        Coordinates:
          * time     (time) datetime64[ns] 2000-01-31 2000-02-29 ... 2001-12-31
    """ 
    
    box = [-5.0,5.0,190.0,240.0] # [lat_min,lat_max,lon_min,lon_max]
        
    return utils.latlon_average(sst_anom, box).rename('nino34')


# ===================================================================================================
def nino4(sst_anom):
    """ 
        Returns Nino-4 index 
        
        | Author: Dougie Squire
        | Date: 10/04/2018
        
        Parameters
        ----------
        sst_anom : xarray DataArray
            Array containing sea surface temperature anomalies
            
        Returns
        -------
        nino4 : xarray DataArray
            Average of the provided sst anomalies over the nino-4 box
            
        Examples
        --------
        >>> sst = xr.DataArray(np.random.normal(size=(90,90,24)), 
        ...                    coords=[('lat', np.arange(-90,90,2)), ('lon', np.arange(0,360,4)), 
        ...                            ('time', pd.date_range('2000-01-01', periods=24, freq='M'))])
        >>> sst_clim = sst.groupby('time.month').mean(dim='time')
        >>> sst_anom = doppyo.utils.anomalize(sst, sst_clim)
        >>> doppyo.diagnostic.nino4(sst_anom)
        <xarray.DataArray 'nino4' (time: 24)>
        array([ 0.017431, -0.086129,  0.106992, -0.097994,  0.109215, -0.120221,
                0.042459, -0.189595,  0.005097,  0.034218,  0.019478,  0.054122,
               -0.017431,  0.086129, -0.106992,  0.097994, -0.109215,  0.120221,
               -0.042459,  0.189595, -0.005097, -0.034218, -0.019478, -0.054122])
        Coordinates:
          * time     (time) datetime64[ns] 2000-01-31 2000-02-29 ... 2001-12-31
    """   
    
    box = [-5.0,5.0,160.0,210.0] # [lat_min,lat_max,lon_min,lon_max]
        
    return utils.latlon_average(sst_anom, box).rename('nino4')


# ===================================================================================================
def emi(sst_anom):
    """ 
        Returns El Nino Modoki index
        
        | Author: Dougie Squire
        | Date: 10/04/2018
        
        Parameters
        ----------
        sst_anom : xarray DataArray
            Array containing sea surface temperature anomalies
            
        Returns
        -------
        emi : xarray DataArray
            Array containing the El Nino Modoki index
            
        Examples
        --------
        >>> sst = xr.DataArray(np.random.normal(size=(90,90,24)), 
        ...                    coords=[('lat', np.arange(-90,90,2)), ('lon', np.arange(0,360,4)), 
        ...                            ('time', pd.date_range('2000-01-01', periods=24, freq='M'))])
        >>> sst_clim = sst.groupby('time.month').mean(dim='time')
        >>> sst_anom = doppyo.utils.anomalize(sst, sst_clim)
        >>> doppyo.diagnostic.emi(sst_anom)
        <xarray.DataArray 'emi' (time: 24)>
        array([-0.046743,  0.181795,  0.020386, -0.215317, -0.209294,  0.109291,
                0.202055, -0.021001, -0.013106,  0.094376, -0.000516, -0.021762,
                0.046743, -0.181795, -0.020386,  0.215317,  0.209294, -0.109291,
               -0.202055,  0.021001,  0.013106, -0.094376,  0.000516,  0.021762])
        Coordinates:
          * time     (time) datetime64[ns] 2000-01-31 2000-02-29 ... 2001-12-31
    """
    
    boxA = [-10.0,10.0,360.0-165.0,360.0-140.0] # [lat_min,lat_max,lon_min,lon_max]
    boxB = [-15.0,5.0,360.0-110.0,360.0-70.0] # [lat_min,lat_max,lon_min,lon_max]
    boxC = [-10.0,20.0,125.0,145.0] # [lat_min,lat_max,lon_min,lon_max]
        
    da_sstA = utils.latlon_average(sst_anom, boxA)
    da_sstB = utils.latlon_average(sst_anom, boxB)
    da_sstC = utils.latlon_average(sst_anom, boxC)
    
    return (da_sstA - 0.5*da_sstB - 0.5*da_sstC).rename('emi')


# ===================================================================================================
def dmi(sst_anom):
    """ 
        Returns dipole mode index 
        
        | Author: Dougie Squire
        | Date: 10/04/2018
        
        Parameters
        ----------
        sst_anom : xarray DataArray
            Array containing sea surface temperature anomalies
            
        Returns
        -------
        dmi : xarray DataArray
            Array containing the dipole mode index
            
        Examples
        --------
        >>> sst = xr.DataArray(np.random.normal(size=(90,90,24)), 
        ...                    coords=[('lat', np.arange(-90,90,2)), ('lon', np.arange(0,360,4)), 
        ...                            ('time', pd.date_range('2000-01-01', periods=24, freq='M'))])
        >>> sst_clim = sst.groupby('time.month').mean(dim='time')
        >>> sst_anom = doppyo.utils.anomalize(sst, sst_clim)
        >>> doppyo.diagnostic.dmi(sst_anom)
        <xarray.DataArray 'dmi' (time: 24)>
        array([-0.225498,  0.220686,  0.032038,  0.019634,  0.00511 , -0.202789,
               -0.014349, -0.293248,  0.020925,  0.114059,  0.066389,  0.238707,
                0.225498, -0.220686, -0.032038, -0.019634, -0.00511 ,  0.202789,
                0.014349,  0.293248, -0.020925, -0.114059, -0.066389, -0.238707])
        Coordinates:
          * time     (time) datetime64[ns] 2000-01-31 2000-02-29 ... 2001-12-31
    """
    
    boxW = [-10.0,10.0,50.0,70.0] # [lat_min,lat_max,lon_min,lon_max]
    boxE = [-10.0,0.0,90.0,110.0] # [lat_min,lat_max,lon_min,lon_max]
        
    da_W = utils.latlon_average(sst_anom, boxW)
    da_E = utils.latlon_average(sst_anom, boxE)
    
    return (da_W - da_E).rename('dmi')


# ===================================================================================================
def soi(slp_anom, lat_name=None, lon_name=None, time_name=None):
    """
        Returns southern oscillation index as defined by NOAA (see, for example, \
                https://www.esrl.noaa.gov/psd/gcos_wgsp/Timeseries/SOI/)
        
        | Author: Dougie Squire
        | Date: 10/04/2018
        
        Parameters
        ----------
        slp_anom : xarray DataArray
            Array containing sea level pressure anomalies
        lat_name : str, optional
            Name of the latitude dimension. If None, doppyo will attempt to determine lat_name \
                    automatically
        lon_name : str, optional
            Name of the longitude dimension. If None, doppyo will attempt to determine lon_name \
                    automatically
        time_name : str, optional
            Name of the time dimension. If None, doppyo will attempt to determine time_name \
                    automatically
            
        Returns
        -------
        soi : xarray DataArray
            Array containing the southern oscillation index
            
        Examples
        --------
        >>> slp = xr.DataArray(np.random.normal(size=(90,90,24)), 
        ...                    coords=[('lat', np.arange(-90,90,2)), ('lon', np.arange(0,360,4)), 
        ...                    ('time', pd.date_range('2000-01-01', periods=24, freq='M'))])
        >>> slp_clim = slp.groupby('time.month').mean(dim='time')
        >>> slp_anom = doppyo.utils.anomalize(slp, slp_clim)
        >>> doppyo.diagnostic.soi(slp_anom)
        <xarray.DataArray 'soi' (time: 24)>
        array([ 0.355277,  0.38263 ,  0.563005, -1.256122, -1.252341,  0.202942,
                0.691819,  0.412523, -1.368695,  0.421943,  2.349053,  0.069382,
               -0.355277, -0.38263 , -0.563005,  1.256122,  1.252341, -0.202942,
               -0.691819, -0.412523,  1.368695, -0.421943, -2.349053, -0.069382])
        Coordinates:
          * time     (time) datetime64[ns] 2000-01-31 2000-02-29 ... 2001-12-31
    """
    
    if lat_name is None:
        lat_name = utils.get_lat_name(slp_anom)
    if lon_name is None:
        lon_name = utils.get_lon_name(slp_anom)
    if time_name is None:
        time_name = utils.get_time_name(slp_anom)
    
    lat_Tahiti = 17.6509
    lon_Tahiti = 149.4260

    lat_Darwin = 12.4634
    lon_Darwin = 130.8456

    da_Tahiti_anom = slp_anom.sel({lat_name : lat_Tahiti, lon_name : lon_Tahiti}, method='nearest')
    da_Tahiti_std = da_Tahiti_anom.std(dim=time_name)
    da_Tahiti_stdzd = da_Tahiti_anom / da_Tahiti_std

    da_Darwin_anom = slp_anom.sel({lat_name : lat_Darwin, lon_name : lon_Darwin}, method='nearest')
    da_Darwin_std = da_Darwin_anom.std(dim=time_name)
    da_Darwin_stdzd = da_Darwin_anom / da_Darwin_std

    MSD = (da_Tahiti_stdzd - da_Darwin_stdzd).std(dim=time_name)
        
    return ((da_Tahiti_stdzd - da_Darwin_stdzd) / MSD).rename('soi')


# ===================================================================================================
def sam(slp_anom, lat_name=None, lon_name=None, time_name=None):
    """
        Returns southern annular mode index as defined by Gong, D. and Wang, S., 1999. Definition \
            of Antarctic oscillation index. Geophysical research letters, 26(4), pp.459-462.
        
        | Author: Dougie Squire
        | Date: 21/06/2019
        
        Parameters
        ----------
        slp_anom : xarray DataArray
            Array containing sea level pressure anomalies
        lat_name : str, optional
            Name of the latitude dimension. If None, doppyo will attempt to determine lat_name \
                    automatically
        lon_name : str, optional
            Name of the longitude dimension. If None, doppyo will attempt to determine lon_name \
                    automatically
        time_name : str, optional
            Name of the time dimension. If None, doppyo will attempt to determine time_name \
                    automatically
            
        Returns
        -------
        sam : xarray DataArray
            Array containing the Gong and Wang (1999) southern annular mode index
            
        Examples
        --------
        >>> slp = xr.DataArray(np.random.normal(size=(90,90,24)), 
        ...                    coords=[('lat', np.arange(-90,90,2)), ('lon', np.arange(0,360,4)), 
        ...                    ('time', pd.date_range('2000-01-01', periods=24, freq='M'))])
        >>> slp_clim = slp.groupby('time.month').mean(dim='time')
        >>> slp_anom = doppyo.utils.anomalize(slp, slp_clim)
        >>> doppyo.diagnostic.soi(slp_anom)
        <xarray.DataArray 'soi' (time: 24)>
        array([ 0.355277,  0.38263 ,  0.563005, -1.256122, -1.252341,  0.202942,
                0.691819,  0.412523, -1.368695,  0.421943,  2.349053,  0.069382,
               -0.355277, -0.38263 , -0.563005,  1.256122,  1.252341, -0.202942,
               -0.691819, -0.412523,  1.368695, -0.421943, -2.349053, -0.069382])
        Coordinates:
          * time     (time) datetime64[ns] 2000-01-31 2000-02-29 ... 2001-12-31
    """
    
    if lat_name is None:
        lat_name = utils.get_lat_name(slp_anom)
    if lon_name is None:
        lon_name = utils.get_lon_name(slp_anom)
    if time_name is None:
        time_name = utils.get_time_name(slp_anom)
    
    slp_40 = slp_anom.interp({lat_name:-40}).mean(lon_name)
    slp_40_stdzd = slp_40 / slp_40.std(dim=time_name)
    
    slp_65 = slp_anom.interp({lat_name:-65}).mean(lon_name)
    slp_65_stdzd = slp_65 / slp_65.std(dim=time_name)
    
    return (slp_40_stdzd - slp_65_stdzd).rename('sam')
    

# ===================================================================================================
# General functions
# ===================================================================================================
def _int_over_atmos(da, lat_name, lon_name, plevel_name, lon_dim=None):
    """ 
        Returns integral of da over the mass of the atmosphere 
        
        | Author: Dougie Squire
        | Date: 15/07/2018
        
        Parameters
        ----------
        da : xarray DataArray
            Array to integrate over the mass of the atmosphere
        lat_name : str
            Name of latitude dimension
        lon_name : str
            Name of longitude dimension
        plevel_name : str
            Name of pressure level dimension
        lon_dim : xarray DataArray, optional
            Value of longitude to use in the integration. Must be broadcastable onto da. If not \
                    provided, the coordinate associated with the longitudinal dimension of da will be used\
                    (if it exists). This option is made available because it is often desirable to integrate\
                    a zonally averaged quantity, which has no longitudinal dimension
            
        Returns
        -------
        mass_integral : xarray DataArray
            Array containing the integral of the input array over the mass of the atmosphere
            
        Examples
        --------
        >>> A = xr.DataArray(np.random.normal(size=(90,90,9)),
        ...                  coords=[('lat', np.arange(-90,90,2)), ('lon', np.arange(0,360,4)),
        ...                          ('level', np.arange(100,1000,100))])
        >>> doppyo.diagnostic._int_over_atmos(A, lat_name='lat', lon_name='lon', plevel_name='level')
        <xarray.DataArray 'mass_integral' ()>
        array(21.194873)
    """
    
    degtorad = utils.constants().pi / 180
    
    if lon_dim is None:
        lat = da[lat_name]
        lon = da[lon_name]
        da = da.sortby([lat, lon])
    else:
        lat = da[lat_name]
        lon = lon_dim
        da = da.sortby(lat)
        
    c = 2 * utils.constants().pi * utils.constants().R_earth
    lat_m = c / 360
    lon_m = c * np.cos(da[lat_name] * degtorad) / 360

    da_z = utils.integrate(da, over_dim=plevel_name, x=(da[plevel_name] * 100) / utils.constants().g)
    if lon_dim is None:
        da_zx = utils.integrate(da_z, over_dim=lon_name, x=da[lon_name] * lon_m)
    else:
        lon_extent = lon_dim * lon_m
        da_zx = (lon_extent.max(lon_name) - lon_extent.min(lon_name)) * da_z
    da_zxy = utils.integrate(da_zx, over_dim=lat_name, x=da[lat_name] * lat_m)
    
    return (da_zxy / (4 * utils.constants().pi * utils.constants().R_earth ** 2)).rename('mass_integral')
