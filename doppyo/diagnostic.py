"""
    doppyo functions for computing various climate diagnostics
    Author: Dougie Squire
    Date created: 04/04/2018
    Python Version: 3.6
"""

__all__ = ['compute_velocitypotential', 'compute_streamfunction', 'compute_rws', 'compute_divergent', 
           'compute_waf', 'compute_BruntVaisala', 'compute_ks2', 'compute_Eady', 'compute_thermal_wind',
           'compute_eofs', 'compute_mmms', 'compute_atmos_energy_cycle', 'pwelch', 'compute_inband_variance', 
           'compute_nino3', 'compute_nino34', 'compute_nino4', 'compute_emi', 'compute_dmi']

# ===================================================================================================
# Packages
# ===================================================================================================
import collections
import numpy as np
import xarray as xr
import windspharm as wsh
from scipy.sparse import linalg

# Load doppyo packages -----
from doppyo import utils

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

    if not u.coords.to_dataset().equals(v.coords.to_dataset()):
        raise ValueError('u and v coordinates do not match')

    # Create a VectorWind instance -----
    w = wsh.xarray.VectorWind(u, v)

    # Compute the velocity potential -----
    phi = w.velocitypotential()
        
    return phi


# ===================================================================================================
def compute_streamfunction(u, v):
    """ 
        Returns the streamfunction given fields of u and v.
        
        u and v must have at least latitude and longitude dimensions with standard naming
    """

    lat_name = utils.get_lat_name(u)
    lon_name = utils.get_lon_name(u)

    if not u.coords.to_dataset().equals(v.coords.to_dataset()):
        raise ValueError('u and v coordinates do not match')
        
    # Create a VectorWind instance -----
    w = wsh.xarray.VectorWind(u, v)

    # Compute the streamfunction -----
    psi = w.streamfunction()
        
    return psi


# ===================================================================================================
def compute_rws(u, v):
    """ 
        Returns the Rossby wave source given fields of u and v.
        
        u and v must have at least latitude and longitude dimensions with standard naming
    """

    lat_name = utils.get_lat_name(u)
    lon_name = utils.get_lon_name(u)

    if not u.coords.to_dataset().equals(v.coords.to_dataset()):
        raise ValueError('u and v coordinates do not match')
        
    # Create a VectorWind instance -----
    w = wsh.xarray.VectorWind(u, v)

    # Compute components of Rossby wave source -----
    eta = w.absolutevorticity() # absolute vorticity
    div = w.divergence() # divergence
    uchi, vchi = w.irrotationalcomponent() # irrotational (divergent) wind components
    etax, etay = w.gradient(eta) # gradients of absolute vorticity

    # Combine the components to form the Rossby wave source term -----
    rws = 1e11 * (-eta * div - (uchi * etax + vchi * etay)).rename('rws')
    rws.attrs['units'] = '1e-11/s^2'
    rws.attrs['long_name'] = 'Rossby wave source'
    
    return rws


# ===================================================================================================
def compute_divergent(u, v):
    """ 
        Returns the irrotational (divergent) component of u and v.
        
        u and v must have at least latitude and longitude dimensions with standard naming
    """

    lat_name = utils.get_lat_name(u)
    lon_name = utils.get_lon_name(u)

    if not u.coords.to_dataset().equals(v.coords.to_dataset()):
        raise ValueError('u and v coordinates do not match')
            
    # Create a VectorWind instance -----
    w = wsh.xarray.VectorWind(u, v)

    # Compute the irrotational components -----
    uchi, vchi = w.irrotationalcomponent()
    
    # Combine into dataset -----
    div = uchi.rename('uchi').to_dataset()
    div['vchi'] = vchi
    
    return div


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
        use standard naming conventions, 'level' or 'plev'
    """
    
    lat_name = utils.get_lat_name(u)
    lon_name = utils.get_lon_name(u)

    if not (u.coords.to_dataset().equals(v.coords.to_dataset()) & \
            u.coords.to_dataset().equals(psi_anom.coords.to_dataset())):
        raise ValueError('psi_anom, u and v coordinates must match')
    
    # Get plev from dimension if it exists -----
    try:
        plev_name = utils.get_level_name(u)
        plev = u[plev_name] / 1000 # Takaya and Nakmura p.610
    except KeyError:
        if p_lev is None:
            raise TypeError('Cannot determine pressure level(s) of provided data. This should' +
                            'be stored as a coordinate, "level" or "plev" in the provided' +
                            'objects. Alternatively, for single level computations, the' + 
                            'pressure can be provided using the p_lev argument')
        else:
            plev = p_lev / 1000 # Takaya and Nakmura p.610
    
    # Create a VectorWind instance (use gradient function) -----
    w = wsh.xarray.VectorWind(u, v)
    
    # Compute the various streamfunction gradients required -----
    psi_x, psi_y = w.gradient(psi_anom)
    psi_xx, psi_xy = w.gradient(psi_x)
    psi_yx, psi_yy = w.gradient(psi_y)
    
    # Compute the wave activity flux -----
    vel = (u * u + v * v) ** 0.5
    uwaf = (0.5 * plev * (u * (psi_x * psi_x - psi_anom * psi_xx) + 
                          v * (psi_x * psi_y - 0.5 * psi_anom * (psi_xy + psi_yx))) / vel).rename('uwaf')
    uwaf.attrs['units'] = 'm^2/s^2'
    uwaf.attrs['long_name'] = 'Zonal Rossby wave activity flux'
    vwaf = (0.5 * plev * (v * (psi_y * psi_y - psi_anom * psi_yy) + 
                          u * (psi_x * psi_y - 0.5 * psi_anom * (psi_xy + psi_yx))) / vel).rename('vwaf')
    vwaf.attrs['units'] = 'm^2/s^2'
    vwaf.attrs['long_name'] = 'Meridional Rossby wave activity flux'
        
    # Combine into dataset -----
    waf = uwaf.to_dataset()
    waf['vwaf'] = vwaf
    
    return waf


# ===================================================================================================
def compute_BruntVaisala(temp):
    """
        Returns the Brunt Väisälä frequency
        
        temp must be saved on pressure levels
    """

    R = utils.constants().R_d
    Cp = utils.constants().C_pd
    g = utils.constants().g

    plev_name = utils.get_level_name(temp)
    dTdp = utils.calc_gradient(temp, plev_name)
    pdR = temp[plev_name] / R

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

    if not u.coords.to_dataset().equals(v.coords.to_dataset()):
        raise ValueError('u and v coordinates do not match')

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
def compute_Eady(u, v, gh, nsq):
    """ 
        Returns the square of the Eady growth rate
        
        u, v, gh and nsq must have at least latitude and longitude dimensions with 
        standard naming
        Data must be saved on pressure levels
    """
    
    degtorad = utils.constants().pi / 180
    lat_name = utils.get_lat_name(u)
    lon_name = utils.get_lon_name(u)
    plev_name = utils.get_level_name(u)
    
    f = 2 * utils.constants().Omega * xr.ufuncs.sin(gh[lat_name] * degtorad)
    eady2 = ((utils.constants().Ce * f) * (utils.calc_gradient((u ** 2 + v ** 2) ** 0.5, dim=plev_name) / \
            utils.calc_gradient(gh, dim=plev_name))) ** 2 / nsq
    eady2.attrs['units'] = 's^-2'
    eady2.attrs['long_name'] = 'Square of Eady growth rate'
    
    return eady2


# ===================================================================================================
def compute_thermal_wind(gh, plev_lower, plev_upper):
    """ 
        Returns the thermal wind, (u_tw, v_tw) = 1/f x k x grad(thickness), where f = 2*Omega*sin(lat)
        
        *_lower and *_upper refer to the value of pressure, not height
        
        gh must have at least latitude and longitude dimensions and two pressure levels with 
        standard naming. plev_lower and plev_upper must be available levels in the pressure
        level dimension.
    """
    
    degtorad = utils.constants().pi / 180
    lat_name = utils.get_lat_name(gh)
    plev_name = utils.get_level_name(gh)
    
    # Compute the thickness -----
    upper = gh.sel({plev_name : plev_lower})
    upper[plev_name] = (plev_lower + plev_upper) / 2
    lower = gh.sel({plev_name : plev_upper})
    lower[plev_name] = (plev_lower + plev_upper) / 2
    thickness = upper - lower
    
    # Compute the gradient -----
    # utils.calc_gradient(thickness, dim=lat_name, x=(thickness[lat_name] * degtorad))
    w = wsh.xarray.VectorWind(thickness, thickness)
    
    # Compute the thickness gradient -----
    u_tmp, v_tmp = w.gradient(thickness)
    
    # k x (u_tw,v_tw) -> (-v_tw, u_tw) -----
    u_tw = -v_tmp / (2 * utils.constants().Omega * xr.ufuncs.sin(thickness[lat_name] * degtorad))
    v_tw = u_tmp / (2 * utils.constants().Omega * xr.ufuncs.sin(thickness[lat_name] * degtorad))
    
    # Combine into dataset -----
    tw = u_tw.to_dataset('u_tw')
    tw['v_tw'] = v_tw
    
    return tw


# ===================================================================================================
def compute_eofs(da, sample_dim='time', weight=None, n_modes=20):
    """
        Returns the empirical orthogonal functions (EOFs), and associated principle component 
        timeseries (PCs) and explained variances of da. When da is a list of xarray objects, 
        returns the joint EOFs associated with each object. In this case, all xarray objects 
        in da must have sample_dim dimensions of equal length.
        
        All dimensions other than sample_dim are treated as sensor dimensions.
        weight=None uses cos(lat)^2 weighting. If weight is specified, it must be the same 
        length as da with each element broadcastable onto each element of da
        
        Follows notation used in "Bjornsson H. and Venegas S. A. A Manual for EOF and SVD 
        analyses of Climatic Data"

        Note that the approach implemented here is non-lazy. I'm am not sure if there is a 
        good way to do this in a lazy way.
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
    sensor_dims = [utils.find_other_dims(d, sample_dim) for d in da]
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
def compute_mmms(v):
    """
        Returns the mean meridional mass streamfunction averaged over all provided longitudes
        
        Pressures must be in hPa
    """
    
    degtorad = utils.constants().pi / 180

    lat = utils.get_lat_name(v)
    lon = utils.get_lon_name(v)
    plev = utils.get_level_name(v)
    cos_lat = xr.ufuncs.cos(v[lat] * degtorad) 

    v_Z = v.mean(dim=lon)
    
    return (2 * utils.constants().pi * utils.constants().R_earth * cos_lat * \
                utils.calc_integral(v_Z, over_dim=plev, x=(v_Z[plev] * 100), cumulative=True) \
                / utils.constants().g)


# ===================================================================================================
def int_over_atmos(da, lat_n, lon_n, plev_n, lon_dim=None):
    """ 
        Returns integral of da over the mass of the atmosphere 
        
        If a longitudinal dimension does not exist, this must be provided as lon_dim
    """
    
    degtorad = utils.constants().pi / 180
    
    if lon_dim is None:
        lat = da[lat_n]
        lon = da[lon_n]
        da = da.sortby([lat, lon])
    else:
        lat = da[lat_n]
        lon = lon_dim
        da = da.sortby(lat)
        
    c = 2 * utils.constants().pi * utils.constants().R_earth
    lat_m = c / 360
    lon_m = c * np.cos(da[lat_n] * degtorad) / 360

    da_z = utils.calc_integral(da, over_dim=plev_n, x=(da[plev_n] * 100) / utils.constants().g)
    if lon_dim is None:
        da_zx = utils.calc_integral(da_z, over_dim=lon_n, x=da[lon_n] * lon_m)
    else:
        lon_extent = lon_dim * lon_m
        da_zx = (lon_extent.max(lon_n) - lon_extent.min(lon_n)) * da_z
    da_zxy = utils.calc_integral(da_zx, over_dim=lat_n, x=da[lat_n] * lat_m)
    
    return da_zxy / (4 * utils.constants().pi * utils.constants().R_earth ** 2)


def flip_n(da):
    """ Flips data along wavenumber coordinate """
    
    daf = da.copy()
    daf['n'] = -daf['n']
    
    return daf.sortby(daf['n'])


def truncate(F, n_truncate, dim):
    """ 
        Converts spatial frequency dim to wavenumber, n, and truncates all wavenumbers greater than 
        n_truncate 
    """
    F[dim] = 360 * F[dim]
    F = F.rename({dim : 'n'})
    F = F.where(abs(F.n) <= n_truncate, drop=True)
    return F, flip_n(F)
    
    
def triple_terms(A, B, C):
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


def triple_terms_loop(A, B, C):
    """ 
        Calculate triple term summation of the form \int_{m=-inf}^{inf} A(m) * B(n) * C(n - m)
    """

    # Loop over all m's and perform rolling sum -----
    ms = A['n'].where(A['n'] != 0, drop=True).values
    ABC = A.copy() * 0
    for m in ms:
        Am = A.sel(n=m)
        Cnm = C.shift(n=int(m)).fillna(0)
        ABC = ABC + (Am * B * Cnm)

    return ABC


def compute_atmos_energy_cycle(temp, u, v, omega, gh, terms=None, vgradz=False, spectral=False, n_wavenumbers=20,
                               integrate=True, loop_triple_terms=False):
    """
        Returns all terms in the Lorenz energy cycle. Follows formulae and notation used in 
        `Marques et al. 2011 Global diagnostic energetics of five state-of-the-art climate 
        models. Climate Dynamics`. Note that this decomposition is in the space domain. A
        space-time decomposition can also be carried out (though not in Fourier space, but 
        this is not implemented here (see `Oort. 1964 On Estimates of the atmospheric energy 
        cycle. Monthly Weather Review`).

        Inputs:
            terms : str or sequence
                list of terms to compute. If None, returns all terms. Available options are:
                    Pz : total available potential energy in the zonally averaged temperature
                         distribution
                    Kz : total kinetic energy in zonally averaged motion
                    Pe : total eddy available potential energy [= sum_n Pn for spectral=True]
                         (Note that for spectral=True, an additional term, Sn, quantifying the
                         rate of transfer of available potential energy to eddies of wavenumber 
                         n from eddies of all other wavenumbers is also returned)
                    Ke : total eddy kinetic energy [= sum_n Kn for spectral=True]
                         (Note that for spectral=True, an additional term, Ln, quantifying the
                         rate of transfer of kinetic energy to eddies of wavenumber n from eddies 
                         of all other wavenumbers is also returned)
                    Cz : rate of conversion of zonal available potential energy to zonal kinetic 
                         energy
                    Ca : rate of transfer of total available potential energy in the zonally 
                         averaged temperature distribution (Pz) to total eddy available potential 
                         energy (Pe) [= sum_n Rn for spectral=True]
                    Ce : rate of transfer of total eddy available potential energy (Pe) to total 
                         eddy kinetic energy (Ke) [= sum_n Cn for spectral=True]
                    Ck : rate of transfer of total eddy kinetic energy (Ke) to total kinetic 
                         energy in zonally averaged motion (Kz) [= sum_n Mn for spectral=True]
            vgradz : bool, optional
                if True, uses `v-grad-z` approach for computing terms relating to conversion
                of potential energy to kinetic energy. Otherwise, defaults to using the 
                `omaga-alpha` approach (see reference above for details)
            spectral : bool, optional
                if True, computes all terms as a function of wavenumber on longitudinal bands
            n_wavenumbers : int, optional
                number of wavenumbers to retain either side of wavenumber=0. Obviously only does
                anything if spectral=True
            integrate : bool, optional
                if True, computes and returns the integral of each term over the mass of the 
                atmosphere. Otherwise, only the integrands are returned.

        Restrictions:
            Pressures must be in hPa
            lat and lon are in degrees
            For spectral=True, lon must be regularly spaced

        Notation: (stackable, e.g. *_ZT indicates the time average of the zonal average)
            *_A -> area average over an isobaric surface
            *_a -> departure from area average
            *_Z -> zonal average
            *_z -> departure from zonal average
            *_T -> time average
            *_t -> departure from time average
            Capital variables indicate Fourier transforms:
                F(u) = U
                F(v) = V
                F(omega) = O
                F(gh) = A
                F(temp) = B
    """
    
    if isinstance(terms,str):
        terms = [terms]
    
    # Initialize some things -----
    lat = utils.get_lat_name(temp)
    lon = utils.get_lon_name(temp)
    # THE FOLLOWING LINE IS CURRENTLY INCORRECT - TEMPORARILY USING HYBRID LEVELS AS PRESSURES LEVELS UNTIL ALL
    # REQUIRED VARIABLES ARE AVAILABLE ON ISOBARIC LEVELS
    plev = utils.get_pres_name(temp)
    # SWITCH TO:
    # plev = utils.get_level_name(temp)
    
    degtorad = utils.constants().pi / 180
    tan_lat = xr.ufuncs.tan(temp[lat] * degtorad)
    cos_lat = xr.ufuncs.cos(temp[lat] * degtorad) 
    
    # Determine the stability parameter using Saltzman's approach -----
    kappa = utils.constants().R_d / utils.constants().C_pd
    p_kap = (1000 / temp[plev]) ** kappa
    theta_A = utils.calc_average(temp * p_kap, [lat, lon], weights=cos_lat)
    dtheta_Adp = utils.calc_gradient(theta_A, dim=plev, x=(theta_A[plev] * 100))
    gamma = - p_kap * (utils.constants().R_d) / ((temp[plev] * 100) * utils.constants().C_pd) / dtheta_Adp # [1/K]
    energies = gamma.rename('gamma').to_dataset()
    
    # Compute zonal terms
    # ========================
    
    if ('Pz' in terms) | (terms is None):
    # Compute the total available potential energy in the zonally averaged temperature
    # distribution, Pz [also commonly called Az] -----
        temp_A = utils.calc_average(temp, [lat, lon], weights=cos_lat)
        temp_Z = temp.mean(dim=lon)
        temp_Za = temp_Z - temp_A
        Pz_int = gamma * utils.constants().C_pd / 2 * temp_Za ** 2  # [J/kg]
        energies['Pz_int'] = Pz_int
        if integrate:
            Pz = int_over_atmos(Pz_int, lat, lon, plev, lon_dim=temp[lon]) # [J/m^2]
            energies['Pz'] = Pz
    
    if ('Kz' in terms) | (terms is None):
    # Compute the total kinetic energy in zonally averaged motion, Kz [also commonly 
    # called Kz] -----
        u_Z = u.mean(dim=lon)
        v_Z = v.mean(dim=lon)
        Kz_int = 0.5 * (u_Z ** 2 + v_Z ** 2) # [J/kg]
        energies['Kz_int'] = Kz_int
        if integrate:
            Kz = int_over_atmos(Kz_int, lat, lon, plev, lon_dim=u[lon]) # [J/m^2]
            energies['Kz'] = Kz
    
    if ('Cz' in terms) | (terms is None):
    # Compute the rate of conversion of zonal available potential energy (Pz) to zonal kinetic
    # energy (Kz), Cz [also commonly called Cz] -----
        if vgradz:
            if 'v_Z' not in locals():
                v_Z = v.mean(dim=lon)
            gh_Z = gh.mean(dim=lon)
            dghdlat = utils.calc_gradient(gh_Z, dim=lat, x=(gh_Z[lat] * degtorad))
            Cz_int = - (utils.constants().g / utils.constants().R_earth) * v_Z * dghdlat # [W/kg]
            energies['Cz_int'] = Cz_int
            if integrate:
                Cz = int_over_atmos(Cz_int, lat, lon, plev, lon_dim=gh[lon]) # [W/m^2]
                energies['Cz'] = Cz
        else:
            if 'temp_Za' not in locals():
                temp_A = utils.calc_average(temp, [lat, lon], weights=cos_lat)
                temp_Z = temp.mean(dim=lon)
                temp_Za = temp_Z - temp_A
            omega_A = utils.calc_average(omega, [lat, lon], weights=cos_lat)
            omega_Z = omega.mean(dim=lon)
            omega_Za = omega_Z - omega_A
            Cz_int = - (utils.constants().R_d / (temp[plev] * 100)) * omega_Za * temp_Za # [W/kg]
            energies['Cz_int'] = Cz_int
            if integrate:
                Cz = int_over_atmos(Cz_int, lat, lon, plev, lon_dim=omega[lon]) # [W/m^2]
                energies['Cz'] = Cz
    
    # Compute eddy terms in Fourier space if spectral=True
    # ==========================================================
    if spectral:
        
        if ('Pe' in terms) | (terms is None):
        # Compute the total available potential energy eddies of wavenumber n, Pn -----
            Bp, Bn = truncate(utils.calc_fft(temp, dim=lon, nfft=len(temp[lon]), twosided=True, shift=True) / 
                              len(temp[lon]), n_truncate=n_wavenumbers, dim='f_'+lon)

            Pn_int = (gamma * utils.constants().C_pd * abs(Bp) ** 2)
            energies['Pn_int'] = Pn_int
            if integrate:
                Pn = int_over_atmos(Pn_int, lat, lon, plev, lon_dim=temp[lon]) # [J/m^2]
                energies['Pn'] = Pn

        # Compute the rate of transfer of available potential energy to eddies of 
        # wavenumber n from eddies of all other wavenumbers, Sn -----
            Up, Un = truncate(utils.calc_fft(u, dim=lon, nfft=len(u[lon]), twosided=True, shift=True) /
                              len(u[lon]), n_truncate=n_wavenumbers, dim='f_'+lon)
            Vp, Vn = truncate(utils.calc_fft(v, dim=lon, nfft=len(v[lon]), twosided=True, shift=True) /
                              len(v[lon]), n_truncate=n_wavenumbers, dim='f_'+lon)
            Op, On = truncate(utils.calc_fft(omega, dim=lon, nfft=len(omega[lon]), twosided=True, shift=True) /
                              len(omega[lon]), n_truncate=n_wavenumbers, dim='f_'+lon)
                
            dBpdlat = utils.calc_gradient(Bp, dim=lat, x=(Bp[lat] * degtorad))
            dBndlat = utils.calc_gradient(Bn, dim=lat, x=(Bn[lat] * degtorad))
            dBpdp = utils.calc_gradient(Bp, dim=plev, x=(Bp[plev] * 100))
            dBndp = utils.calc_gradient(Bn, dim=plev, x=(Bn[plev] * 100))

            if loop_triple_terms:
                BpBnUp = triple_terms_loop(Bp, Bn, Up)
                BpBpUn = triple_terms_loop(Bp, Bp, Un)
                BpglBnVp = triple_terms_loop(Bp, dBndlat, Vp)
                BpglBpVn = triple_terms_loop(Bp, dBpdlat, Vn)
                BpgpBnOp = triple_terms_loop(Bp, dBndp, Op)
                BpgpBpOn = triple_terms_loop(Bp, dBpdp, On)
                BpBnOp = triple_terms_loop(Bp, Bn, Op)
                BpBpOn = triple_terms_loop(Bp, Bp, On)
            else:
                BpBnUp = triple_terms(Bp, Bn, Up)
                BpBpUn = triple_terms(Bp, Bp, Un)
                BpglBnVp = triple_terms(Bp, dBndlat, Vp)
                BpglBpVn = triple_terms(Bp, dBpdlat, Vn)
                BpgpBnOp = triple_terms(Bp, dBndp, Op)
                BpgpBpOn = triple_terms(Bp, dBpdp, On)
                BpBnOp = triple_terms(Bp, Bn, Op)
                BpBpOn = triple_terms(Bp, Bp, On)

            Sn_int = -gamma * utils.constants().C_pd * (1j * Bp['n']) / \
                         (utils.constants().R_earth * xr.ufuncs.cos(Bp[lat] * degtorad)) * \
                         (BpBnUp + BpBpUn) + \
                     gamma * utils.constants().C_pd / utils.constants().R_earth * \
                         (BpglBnVp + BpglBpVn) + \
                     gamma * utils.constants().C_pd * (BpgpBnOp + BpgpBpOn) + \
                     gamma * utils.constants().R_d / Bp[plev] * \
                         (BpBnOp + BpBpOn)
            energies['Sn_int'] = Sn_int
            if integrate:
                Sn = abs(int_over_atmos(Sn_int, lat, lon, plev, lon_dim=temp[lon])) # [W/m^2]
                energies['Sn'] = Sn
                
        if ('Ke' in terms) | (terms is None):
        # Compute the total kinetic energy in eddies of wavenumber n, Kn -----
            if 'U' not in locals():
                Up, Un = truncate(utils.calc_fft(u, dim=lon, nfft=len(u[lon]), twosided=True, shift=True) /
                                  len(u[lon]), n_truncate=n_wavenumbers, dim='f_'+lon)
            if 'V' not in locals():
                Vp, Vn = truncate(utils.calc_fft(v, dim=lon, nfft=len(v[lon]), twosided=True, shift=True) / 
                                  len(v[lon]), n_truncate=n_wavenumbers, dim='f_'+lon)

            Kn_int = abs(Up) ** 2 + abs(Vp) ** 2
            energies['Kn_int'] = Kn_int
            if integrate:
                Kn = int_over_atmos(Kn_int, lat, lon, plev, lon_dim=u[lon]) # [J/m^2]
                energies['Kn'] = Kn

        # Compute the rate of transfer of kinetic energy to eddies of wavenumber n from 
        # eddies of all other wavenumbers, Ln -----
            if 'O' not in locals():
                Op, On = truncate(utils.calc_fft(omega, dim=lon, nfft=len(omega[lon]), twosided=True, shift=True) / 
                                  len(omega[lon]), n_truncate=n_wavenumbers, dim='f_'+lon)
                
            dUpdp = utils.calc_gradient(Up, dim=plev, x=(Up[plev] * 100))
            dVpdp = utils.calc_gradient(Vp, dim=plev, x=(Vp[plev] * 100))
            dOpdp = utils.calc_gradient(Op, dim=plev, x=(Op[plev] * 100))
            dOndp = utils.calc_gradient(On, dim=plev, x=(On[plev] * 100))
            dVpcdl = utils.calc_gradient(Vp * cos_lat, dim=lat, x=(Vp[lat] * degtorad))
            dVncdl = utils.calc_gradient(Vn * cos_lat, dim=lat, x=(Vn[lat] * degtorad))
            dUpdl = utils.calc_gradient(Up, dim=lat, x=(Up[lat] * degtorad))
            dVpdl = utils.calc_gradient(Vp, dim=lat, x=(Vp[lat] * degtorad))

            if loop_triple_terms:
                UpUnUp = triple_terms_loop(Up, Un, Up)
                UpUpUn = triple_terms_loop(Up, Up, Un)
                VpVnUp = triple_terms_loop(Vp, Vn, Up)
                VpVpUn = triple_terms_loop(Vp, Vp, Un)
                VpUnUp = triple_terms_loop(Vp, Un, Up)
                VpUpUn = triple_terms_loop(Vp, Up, Un)
                UpVnUp = triple_terms_loop(Up, Vn, Up)
                UpVpUn = triple_terms_loop(Up, Vp, Un)
                gpUpUngpOp = triple_terms_loop(dUpdp, Un, dOpdp)
                gpUpUpgpOn = triple_terms_loop(dUpdp, Up, dOndp)
                gpVpVngpOp = triple_terms_loop(dVpdp, Vn, dOpdp)
                gpVpVpgpOn = triple_terms_loop(dVpdp, Vp, dOndp)
                glUpUnglVpc = triple_terms_loop(dUpdl, Un, dVpcdl)
                glUpUpglVnc = triple_terms_loop(dUpdl, Up, dVncdl)
                glVpVnglVpc = triple_terms_loop(dVpdl, Vn, dVpcdl)
                glVpVpglVnc = triple_terms_loop(dVpdl, Vp, dVncdl)
            else:
                UpUnUp = triple_terms(Up, Un, Up)
                UpUpUn = triple_terms(Up, Up, Un)
                VpVnUp = triple_terms(Vp, Vn, Up)
                VpVpUn = triple_terms(Vp, Vp, Un)
                VpUnUp = triple_terms(Vp, Un, Up)
                VpUpUn = triple_terms(Vp, Up, Un)
                UpVnUp = triple_terms(Up, Vn, Up)
                UpVpUn = triple_terms(Up, Vp, Un)
                gpUpUngpOp = triple_terms(dUpdp, Un, dOpdp)
                gpUpUpgpOn = triple_terms(dUpdp, Up, dOndp)
                gpVpVngpOp = triple_terms(dVpdp, Vn, dOpdp)
                gpVpVpgpOn = triple_terms(dVpdp, Vp, dOndp)
                glUpUnglVpc = triple_terms(dUpdl, Un, dVpcdl)
                glUpUpglVnc = triple_terms(dUpdl, Up, dVncdl)
                glVpVnglVpc = triple_terms(dVpdl, Vn, dVpcdl)
                glVpVpglVnc = triple_terms(dVpdl, Vp, dVncdl)

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
                Ln = abs(int_over_atmos(Ln_int, lat, lon, plev, lon_dim=u[lon])) # [W/m^2]
                energies['Ln'] = Ln
        
        if ('Ca' in terms) | (terms is None):
        # Compute the rate of transfer of zonal available potential energy to eddy 
        # available potential energy in wavenumber n, Rn -----
            if 'temp_Z' not in locals():
                temp_Z = temp.mean(dim=lon)
            if 'V' not in locals():
                Vp, Vn = truncate(utils.calc_fft(v, dim=lon, nfft=len(v[lon]), twosided=True, shift=True) / 
                                  len(v[lon]), n_truncate=n_wavenumbers, dim='f_'+lon)
            if 'B' not in locals():
                Bp, Bn = truncate(utils.calc_fft(temp, dim=lon, nfft=len(temp[lon]), twosided=True, shift=True) / 
                                  len(temp[lon]), n_truncate=n_wavenumbers, dim='f_'+lon)
            if 'O' not in locals():
                Op, On = truncate(utils.calc_fft(omega, dim=lon, nfft=len(omega[lon]), twosided=True, shift=True) / 
                                  len(omega[lon]), n_truncate=n_wavenumbers, dim='f_'+lon)

            dtemp_Zdlat = utils.calc_gradient(temp_Z, dim=lat, x=(temp_Z[lat] * degtorad))
            theta = temp * p_kap
            theta_Z = theta.mean(dim=lon)
            theta_Za = theta_Z - theta_A
            dtheta_Zadp = utils.calc_gradient(theta_Za, dim=plev, x=(theta_Za[plev] * 100))
            Rn_int = gamma * utils.constants().C_pd * ((dtemp_Zdlat / utils.constants().R_earth) * (Vp * Bn + Vn * Bp) + 
                                                       (p_kap * dtheta_Zadp) * (Op * Bn + On * Bp)) # [W/kg]
            energies['Rn_int'] = Rn_int
            if integrate:
                Rn = abs(int_over_atmos(Rn_int, lat, lon, plev, lon_dim=temp[lon])) # [W/m^2]
                energies['Rn'] = Rn

        if ('Ce' in terms) | (terms is None):
        # Compute the rate of conversion of available potential energy of wavenumber n 
        # to eddy kinetic energy of wavenumber n, Cn -----
            if vgradz:
                if 'U' not in locals():
                    Up, Un = truncate(utils.calc_fft(u, dim=lon, nfft=len(u[lon]), twosided=True, shift=True) / 
                                      len(u[lon]), n_truncate=n_wavenumbers, dim='f_'+lon)
                if 'V' not in locals():
                    Vp, Vn = truncate(utils.calc_fft(v, dim=lon, nfft=len(v[lon]), twosided=True, shift=True) / 
                                      len(v[lon]), n_truncate=n_wavenumbers, dim='f_'+lon)
                Ap, An = truncate(utils.calc_fft(gh, dim=lon, nfft=len(gh[lon]), twosided=True, shift=True) / 
                                  len(gh[lon]), n_truncate=n_wavenumbers, dim='f_'+lon)

                dApdlat = utils.calc_gradient(Ap, dim=lat, x=(Ap[lat] * degtorad))
                dAndlat = utils.calc_gradient(An, dim=lat, x=(An[lat] * degtorad))

                Cn_int = (((-1j * utils.constants().g * Up['n']) / \
                           (utils.constants().R_earth * xr.ufuncs.cos(Up[lat] * degtorad))) * \
                                (Ap * Un - An * Up)) - \
                         ((utils.constants().g / utils.constants().R_earth) * \
                                (dApdlat * Vn + dAndlat * Vp)) # [W/kg]
                energies['Cn_int'] = Cn_int
                if integrate:
                    Cn = abs(int_over_atmos(Cn_int, lat, lon, plev, lon_dim=u[lon])) # [W/m^2]
                    energies['Cn'] = Cn
            else:
                if 'O' not in locals():
                    Op, On = truncate(utils.calc_fft(omega, dim=lon, nfft=len(omega[lon]), twosided=True, shift=True) / 
                                      len(omega[lon]), n_truncate=n_wavenumbers, dim='f_'+lon)
                if 'B' not in locals():
                    Bp, Bn = truncate(utils.calc_fft(temp, dim=lon, nfft=len(temp[lon]), twosided=True, shift=True) / 
                                      len(temp[lon]), n_truncate=n_wavenumbers, dim='f_'+lon)
                Cn_int = - (utils.constants().R_d / (omega[plev] * 100)) * (Op * Bn + On * Bp) # [W/kg]
                energies['Cn_int'] = Cn_int
                if integrate:
                    Cn = abs(int_over_atmos(Cn_int, lat, lon, plev, lon_dim=temp[lon])) # [W/m^2]
                    energies['Cn'] = Cn
    
        if ('Ck' in terms) | (terms is None):
        # Compute the rate of transfer of kinetic energy to the zonally averaged flow 
        # from eddies of wavenumber n, Mn -----
            if 'v_Z' not in locals():
                v_Z = v.mean(dim=lon)
            if 'u_Z' not in locals():
                u_Z = u.mean(dim=lon)
            if 'U' not in locals():
                Up, Un = truncate(utils.calc_fft(u, dim=lon, nfft=len(u[lon]), twosided=True, shift=True) / 
                                  len(u[lon]), n_truncate=n_wavenumbers, dim='f_'+lon)
            if 'V' not in locals():
                Vp, Vn = truncate(utils.calc_fft(v, dim=lon, nfft=len(v[lon]), twosided=True, shift=True) / 
                                  len(v[lon]), n_truncate=n_wavenumbers, dim='f_'+lon)
            if 'O' not in locals():
                Op, On = truncate(utils.calc_fft(omega, dim=lon, nfft=len(omega[lon]), twosided=True, shift=True) / 
                                      len(omega[lon]), n_truncate=n_wavenumbers, dim='f_'+lon)
            dv_Zdlat = utils.calc_gradient(v_Z, dim=lat, x=(v[lat] * degtorad))
            du_Zndlat = utils.calc_gradient(u_Z / xr.ufuncs.cos(u[lat] * degtorad), 
                                            dim=lat, x=(u[lat] * degtorad))
            dv_Zdp = utils.calc_gradient(v_Z, dim=plev, x=(v[plev] * 100))
            du_Zdp = utils.calc_gradient(u_Z, dim=plev, x=(u[plev] * 100))

            Mn_int = (-2 * Up * Un * v_Z * tan_lat / utils.constants().R_earth) + \
                     (2 * Vp * Vn * dv_Zdlat / utils.constants().R_earth + (Vp * On + Vn * Op) * dv_Zdp) + \
                     ((Up * On + Un * Op) * du_Zdp) + \
                     ((Up * Vn + Un * Vp) * xr.ufuncs.cos(u[lat] * degtorad) / \
                         utils.constants().R_earth * du_Zndlat) # [W/kg]
            energies['Mn_int'] = Mn_int
            if integrate:
                Mn = abs(int_over_atmos(Mn_int, lat, lon, plev, lon_dim=u[lon])) # [W/m^2]
                energies['Mn'] = Mn
        
        if ('Ge' in terms) | (terms is None):
        # Compute the rate of generation of eddy available potential energy of wavenumber 
        # n due to nonadiabatic heating, Gn -----
            raise ValueError('Rate of generation of eddy available potential energy is computed as ' +
                             'a residual and cannot be computed in Fourier space. Please set spectral' +
                             '=False')
        
        if ('De' in terms) | (terms is None):
        # Compute the rate of dissipation of the kinetic energy of eddies of wavenumber n, 
        # Dn -----
            raise ValueError('Rate of viscous dissipation of eddy kinetic energy is computed as a ' +
                             'residual and cannot be computed in Fourier space. Please set spectral' +
                             '=False')
        
    else:
        
        if ('Pe' in terms) | (terms is None):
        # Compute the total eddy available potential energy, Pe [also commonly called 
        # Ae] -----
            if 'temp_Z' not in locals():
                temp_Z = temp.mean(dim=lon)
            temp_z = temp - temp_Z
            Pe_int = gamma * utils.constants().C_pd / 2 * (temp_z ** 2).mean(dim=lon)  # [J/kg]
            energies['Pe_int'] = Pe_int
            if integrate:
                Pe = int_over_atmos(Pe_int, lat, lon, plev, lon_dim=temp[lon]) # [J/m^2]
                energies['Pe'] = Pe
        
        if ('Ke' in terms) | (terms is None):
        # Compute the total eddy kinetic energy, Ke -----
            if 'u_Z' not in locals():
                    u_Z = u.mean(dim=lon)
            if 'v_Z' not in locals():
                    v_Z = v.mean(dim=lon)
            u_z = u - u_Z
            v_z = v - v_Z
            Ke_int = 0.5 * (u_z ** 2 + v_z ** 2).mean(dim=lon) # [J/kg]
            energies['Ke_int'] = Ke_int
            if integrate:
                Ke = int_over_atmos(Ke_int, lat, lon, plev, lon_dim=u[lon]) # [J/m^2]
                energies['Ke'] = Ke
                
        if ('Ca' in terms) | (terms is None):
        # Compute the rate of transfer of total available potential energy in the zonally 
        # averaged temperature distribution (Pz) to total eddy available potential energy 
        # (Pe), Ca -----
            if 'v_Z' not in locals():
                v_Z = v.mean(dim=lon)
            if 'temp_Z' not in locals():
                temp_Z = temp.mean(dim=lon)
            if 'omega_Z' not in locals():
                omega_Z = omega.mean(dim=lon)
            if 'theta_Z' not in locals():
                theta = temp * p_kap
                theta_Z = theta.mean(dim=lon)
            if 'dtemp_Zdlat' not in locals():
                dtemp_Zdlat = utils.calc_gradient(temp_Z, dim=lat, x=(temp_Z[lat] * degtorad))
            v_z = v - v_Z
            temp_z = temp - temp_Z
            omega_z = omega - omega_Z
            oT_Z = (omega_z * temp_z).mean(dim=lon)
            oT_A = utils.calc_average(omega_z * temp_z, [lat, lon], weights=cos_lat)
            oT_Za = oT_Z - oT_A
            theta_Za = theta_Z - theta_A
            dtheta_Zadp = utils.calc_gradient(theta_Za, dim=plev, x=(theta_Za[plev] * 100))
            Ca_int = - gamma * utils.constants().C_pd * \
                           (((v_z * temp_z).mean(dim=lon) * dtemp_Zdlat / utils.constants().R_earth) + \
                            (p_kap * oT_Za * dtheta_Zadp)) # [W/kg]
            energies['Ca_int'] = Ca_int
            if integrate:
                Ca = int_over_atmos(Ca_int, lat, lon, plev, lon_dim=v[lon]) # [W/m^2]
                energies['Ca'] = Ca
            
        if ('Ce' in terms) | (terms is None):
        # Compute the rate of transfer of total eddy available potential energy (Pe) to 
        # total eddy kinetic energy (Ke), Ce -----
            if 'temp_Z' not in locals():
                temp_Z = temp.mean(dim=lon)
            if 'omega_Z' not in locals():
                omega_Z = omega.mean(dim=lon)
            temp_z = temp - temp_Z
            omega_z = omega - omega_Z
            Ce_int = - (utils.constants().R_d / (temp[plev] * 100)) * \
                           (omega_z * temp_z).mean(dim=lon) # [W/kg]  
            energies['Ce_int'] = Ce_int
            if integrate:
                Ce = int_over_atmos(Ce_int, lat, lon, plev, lon_dim=temp[lon]) # [W/m^2]
                energies['Ce'] = Ce
        
        if ('Ck' in terms) | (terms is None):
        # Compute the rate of transfer of total eddy kinetic energy (Ke) to total kinetic 
        # energy in zonally averaged motion (Kz), Ck -----
            if 'u_Z' not in locals():
                    u_Z = u.mean(dim=lon)
            if 'v_Z' not in locals():
                    v_Z = v.mean(dim=lon)
            if 'omega_Z' not in locals():
                omega_Z = omega.mean(dim=lon)
            u_z = u - u_Z
            v_z = v - v_Z
            omega_z = omega - omega_Z
            du_Zndlat = utils.calc_gradient(u_Z / cos_lat, dim=lat, x=(u_Z[lat] * degtorad))
            dv_Zdlat = utils.calc_gradient(v_Z, dim=lat, x=(v_Z[lat] * degtorad))
            du_Zdp = utils.calc_gradient(u_Z, dim=plev, x=(u_Z[plev] * 100))
            dv_Zdp = utils.calc_gradient(v_Z, dim=plev, x=(v_Z[plev] * 100))
            Ck_int = (u_z * v_z).mean(dim=lon)  * cos_lat * du_Zndlat / utils.constants().R_earth + \
                     (u_z * omega_z).mean(dim=lon) * du_Zdp + \
                     (v_z ** 2).mean(dim=lon) * dv_Zdlat / utils.constants().R_earth + \
                     (v_z * omega_z).mean(dim=lon) * dv_Zdp - \
                     (u_z ** 2).mean(dim=lon) * v_Z * tan_lat / utils.constants().R_earth
            energies['Ck_int'] = Ck_int
            if integrate:
                Ck = int_over_atmos(Ck_int, lat, lon, plev, lon_dim=temp[lon]) # [W/m^2]
                energies['Ck'] = Ck
                
    if ('Gz' in terms) | (terms is None):
    # Compute the rate of generation of zonal available potential energy due to the zonally
    # averaged heating, Gz -----
        if ('Cz' not in terms) | ('Ca' not in terms):
            raise ValueError('The rate of generation of zonal available potential energy, Gz, is ' +
                             'computed from the residual of Cz and Ca. Please add these to the list, ' +
                             'terms=[<terms>].')
        Gz_int = Cz_int + Ca_int
        energies['Gz_int'] = Gz_int
        if integrate:
            Gz = int_over_atmos(Gz_int, lat, lon, plev, lon_dim=temp[lon]) # [W/m^2]
            energies['Gz'] = Gz

    if ('Ge' in terms) | (terms is None):
    # Compute the rate of generation of eddy available potential energy (Ae), Ge -----
        if ('Ce' not in terms) | ('Ca' not in terms):
            raise ValueError('The rate of generation of eddy available potential energy, Ge, is ' +
                             'computed from the residual of Ce and Ca. Please add these to the list, ' +
                             'terms=[<terms>].')
        Ge_int = Ce_int - Ca_int
        energies['Ge_int'] = Ge_int
        if integrate:
            Ge = int_over_atmos(Ge_int, lat, lon, plev, lon_dim=temp[lon]) # [W/m^2]
            energies['Ge'] = Ge
    
    if ('Dz' in terms) | (terms is None):
    # Compute the rate of viscous dissipation of zonal kinetic energy, Dz -----
        if ('Cz' not in terms) | ('Ck' not in terms):
            raise ValueError('The rate of viscous dissipation of zonal kinetic energy, Dz, is ' +
                             'computed from the residual of Cz and Ck. Please add these to the ' +
                             'list, terms=[<terms>].')
        Dz_int = Cz_int - Ck_int
        energies['Dz_int'] = Dz_int
        if integrate:
            Dz = int_over_atmos(Dz_int, lat, lon, plev, lon_dim=temp[lon]) # [W/m^2]
            energies['Dz'] = Dz

    if ('De' in terms) | (terms is None):
    # Compute the rate of dissipation of eddy kinetic energy (Ke), De -----
        if ('Ce' not in terms) | ('Ck' not in terms):
            raise ValueError('The rate of viscous dissipation of eddy kinetic energy, De, is ' +
                             'computed from the residual of Ce and Ck. Please add these to the ' +
                             'list, terms=[<terms>].')
        De_int = Ce_int - Ck_int
        energies['De_int'] = De_int
        if integrate:
            De = int_over_atmos(De_int, lat, lon, plev, lon_dim=temp[lon]) # [W/m^2]
            energies['De'] = De
    
    return energies
        

# ===================================================================================================
def pwelch(da1, da2, dim, nwindow, overlap=50, dx=None, hanning=False):
    """
        Compute the (cross) power spectral density along dimension dim using welch's method

        nwindow is the length of the signal segments, and overlap is the overlap [%] of the segments

        For consistency between spatial and temporal dim, spectra is computed relative to
        a "frequency", f = 1/dx, where dx is the spacing along dim, e.g.:
            - for temporal dim, dx is computed in seconds. Thus, f = 1/seconds = Hz
            - for spatial dim in meters, f = 1/meters = k/(2*pi) 
            - for spatial dim in degrees, f = 1/degrees = k/360
        If converting the "frequency" to wavenumber, for example, one must also adjust the spectra 
        magnitude so that the integral remains equal to the variance, e.g. for spatial spectra:
            k = f*(2*pi)  ->  phi_new = phi_old/(2*pi)
    """

    # Force nwindow to be even -----
    if nwindow % 2 != 0:
        nwindow = nwindow - 1
        
    if not da1.coords.to_dataset().equals(da2.coords.to_dataset()):
            raise ValueError('da1 and da2 coordinates do not match')

    # Determine dx if not provided -----
    if dx is None:
        diff = da1[dim].diff(dim)
        if utils.is_datetime(da1[dim].values):
            # Drop differences on leap days so that still works with 'noleap' calendars -----
            diff = diff.where((diff[dim].dt.month != 3) & (diff[dim].dt.day != 1), drop=True)
        if np.all(diff == diff[0]):
            if utils.is_datetime(da1[dim].values):
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
    da1_fft = utils.calc_fft(da1_windowed, dim=dim)
    da2_fftc = xr.ufuncs.conj(utils.calc_fft(da2_windowed, dim=dim))

    return (weight * 2 * dx * (da1_fft * da2_fftc).mean('n') / nwindow).real


# ===================================================================================================
def compute_inband_variance(da, dim, bounds, nwindow, overlap=50):
    """ 
        Compute the in-band variance along dimension dim.
        
        For consistency between spatial and temporal dim, spectra is computed relative to
        a "frequency", f = 1/dx, where dx is the spacing along dim, e.g.:
            - for temporal dim, dx is computed in seconds. Thus, f = 1/seconds = Hz
            - for spatial dim in meters, f = 1/meters = k/(2*pi) 
            - for spatial dim in degrees, f = 1/degrees = k/360
        bounds must be provided in a way consistent with this, e.g.:
            - for temporal dim, bounds = 1 / (60*60*24*[d1, d2, d3]), where d# are numbers 
              of days
            - for spatial dim, bounds = 1 / [l1, l2, l3], where l# are numbers of meters, 
              degrees, etc
    """
    
    bounds = np.sort(bounds)
    spectra = pwelch(da, da, dim=dim, nwindow=nwindow, overlap=overlap)
    dx = spectra['f_'+dim].diff('f_'+dim).values[0]
    bands = spectra.groupby_bins('f_'+dim, bounds, right=False)
    
    return bands.apply(utils.calc_integral, over_dim='f_'+dim, dx=dx)


# ===================================================================================================
# Indices
# ===================================================================================================
def compute_nino3(da_sst_anom):
    ''' Returns nino3 index '''  
    
    box = [-5.0,5.0,210.0,270.0] # [lat_min,lat_max,lon_min,lon_max]
        
    return utils.calc_boxavg_latlon(da_sst_anom,box)


# ===================================================================================================
def compute_nino34(da_sst_anom):
    ''' Returns nino3.4 index '''  
    
    box = [-5.0,5.0,190.0,240.0] # [lat_min,lat_max,lon_min,lon_max]
        
    return utils.calc_boxavg_latlon(da_sst_anom,box)


# ===================================================================================================
def compute_nino4(da_sst_anom):
    ''' Returns nino4 index '''  
    
    box = [-5.0,5.0,160.0,210.0] # [lat_min,lat_max,lon_min,lon_max]
        
    return utils.calc_boxavg_latlon(da_sst_anom,box)


# ===================================================================================================
def compute_emi(da_sst_anom):
    ''' Returns EMI index ''' 
    
    boxA = [-10.0,10.0,360.0-165.0,360.0-140.0] # [lat_min,lat_max,lon_min,lon_max]
    boxB = [-15.0,5.0,360.0-110.0,360.0-70.0] # [lat_min,lat_max,lon_min,lon_max]
    boxC = [-10.0,20.0,125.0,145.0] # [lat_min,lat_max,lon_min,lon_max]
        
    da_sstA = utils.calc_boxavg_latlon(da_sst_anom,boxA)
    da_sstB = utils.calc_boxavg_latlon(da_sst_anom,boxB)
    da_sstC = utils.calc_boxavg_latlon(da_sst_anom,boxC)
    
    return da_sstA - 0.5*da_sstB - 0.5*da_sstC


# ===================================================================================================
def compute_dmi(da_sst_anom):
    ''' Returns DMI index ''' 
    
    boxW = [-10.0,10.0,50.0,70.0] # [lat_min,lat_max,lon_min,lon_max]
    boxE = [-10.0,0.0,90.0,110.0] # [lat_min,lat_max,lon_min,lon_max]
        
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
