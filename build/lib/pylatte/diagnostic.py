"""
    pyLatte functions for computing various climate diagnostics
    Author: Dougie Squire
    Date created: 04/04/2018
    Python Version: 3.6
"""

__all__ = ['compute_velocitypotential', 'compute_streamfunction', 'compute_rws', 'compute_divergent', 
           'compute_waf', 'compute_BruntVaisala', 'compute_ks2', 'compute_Eady', 'compute_atmos_energy_cycle',
           'compute_nino3', 'compute_nino34', 'compute_nino4', 'compute_emi', 'compute_dmi']

# ===================================================================================================
# Packages
# ===================================================================================================
import numpy as np
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
def int_over_atmos(da, lat_n, lon_n, pres_n, lon_dim=None):
    """ Returns integral of da over the mass of the atmosphere """
    
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

    da_z = utils.calc_integral(da, over_dim=pres_n, x=(da[pres_n] * 100) / utils.constants().g)
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
    B = B.where(B['n'] > 0, drop=True)
    Cnm = Cnm.where((Cnm['m'] != 0) & (Cnm['n'] > 0), drop=True)

    return (B * (Am * Cnm)).sum(dim='m', skipna=False)


def compute_atmos_energy_cycle(temp, u, v, omega, gh, terms=None, vgradz=False, spectral=False, integrate=True):
    """
        Returns all terms in the Lorenz energy cycle. Follows formulae and notation used in 
        `Marques et al. 2011 Global diagnostic energetics of five state-of-the-art climate 
        models. Climate Dynamics`. 

        Inputs:
            terms : list of terms to compute. If None, returns all terms. Available options are:
                        P0 : total available potential energy in the zonally averaged temperature
                             distribution
                        K0 : total kinetic energy in zonally averaged motion
                        Pe : total eddy available potential energy [= sum_n Pn for spectral=True]
                             (Note that for spectral=True, an additional term, Sn, quantifying the
                             rate of transfer of available potential energy to eddies of wavenumber 
                             n from eddies of all other wavenumbers is also returned)
                        Ke : total eddy kinetic energy [= sum_n Kn for spectral=True]
                             (Note that for spectral=True, an additional term, Ln, quantifying the
                             rate of transfer of kinetic energy to eddies of wavenumber n from eddies 
                             of all other wavenumbers is also returned)
                        C0 : rate of conversion of zonal available potential energy to zonal kinetic 
                             energy
                        Ca : rate of transfer of total available potential energy in the zonally 
                             averaged temperature distribution (P0) to total eddy available potential 
                             energy (Pe) [= sum_n Rn for spectral=True]
                        Ce : rate of transfer of total eddy available potential energy (Pe) to total 
                             eddy kinetic energy (Ke) [= sum_n Cn for spectral=True]
                        Ck : rate of transfer of total eddy kinetic energy (Ke) to total kinetic 
                             energy in zonally averaged motion (K0) [= sum_n Mn for spectral=True]
                        (Note that for spectral=True, two additional terms are )
            vgradz : if True, uses `v-grad-z` approach for computing terms relating to conversion
                of potential energy to kinetic energy. Otherwise, defaults to using the 
                `omaga-alpha` approach (see reference above for details)
            spectral : if True, computes all terms as a function of wavenumber on longitudinal bands
            integrate : if True, computes and returns the integral of each term over the mass of the 
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
    pres = utils.get_pres_name(temp)
    
    degtorad = utils.constants().pi / 180
    tan_lat = xr.ufuncs.tan(temp[lat] * degtorad)
    cos_lat = xr.ufuncs.cos(temp[lat] * degtorad) 
    
    # Determine the stability parameter using Saltzman's approach -----
    kappa = utils.constants().R_d / utils.constants().C_pd
    p_kap = (1 / (temp[pres] * 100)) ** kappa
    theta_A = utils.calc_average(temp * p_kap, [lat, lon], weights=cos_lat)
    dtheta_Adp = utils.calc_gradient(theta_A, dim=pres, x=(theta_A[pres] * 100))
    gamma = - p_kap * (utils.constants().R_d) / ((temp[pres] * 100) * utils.constants().C_pd) / dtheta_Adp # [1/K]
    energies = gamma.rename('gamma').to_dataset()
    
    # Compute zonal terms
    # ========================
    
    if ('P0' in terms) | (terms is None):
    # Compute the total available potential energy in the zonally averaged temperature
    # distribution, P0 [also commonly called Az] -----
        temp_A = utils.calc_average(temp, [lat, lon], weights=cos_lat)
        temp_Z = temp.mean(dim=lon)
        temp_Za = temp_Z - temp_A
        P0_int = gamma * utils.constants().C_pd / 2 * temp_Za ** 2  # [J/kg]
        energies['P0_int'] = P0_int
        if integrate:
            P0 = int_over_atmos(P0_int, lat, lon, pres, lon_dim=temp[lon]) # [J/m^2]
            energies['P0'] = P0
    
    if ('K0' in terms) | (terms is None):
    # Compute the total kinetic energy in zonally averaged motion, K0 [also commonly 
    # called Kz] -----
        u_Z = u.mean(dim=lon)
        v_Z = v.mean(dim=lon)
        K0_int = 0.5 * (u_Z ** 2 + v_Z ** 2) # [J/kg]
        energies['K0_int'] = K0_int
        if integrate:
            K0 = int_over_atmos(K0_int, lat, lon, pres, lon_dim=u[lon]) # [J/m^2]
            energies['K0'] = K0

    
    if ('C0' in terms) | (terms is None):
    # Compute the rate of conversion of zonal available potential energy to zonal kinetic
    # energy, C0 [also commonly called Cz] -----
        if vgradz:
            if 'v_Z' not in locals():
                v_Z = v.mean(dim=lon)
            gh_Z = gh.mean(dim=lon)
            dghdlat = utils.calc_gradient(gh_Z, dim=lat, x=(gh_Z[lat] * degtorad))
            C0_int = - (utils.constants().g / utils.constants().R_earth) * v_Z * dghdlat # [W/kg]
            energies['C0_int'] = C0_int
            if integrate:
                C0 = int_over_atmos(C0_int, lat, lon, pres, lon_dim=gh[lon]) # [W/m^2]
                energies['C0'] = C0
        else:
            if 'temp_Za' not in locals():
                temp_A = utils.calc_average(temp, [lat, lon], weights=cos_lat)
                temp_Z = temp.mean(dim=lon)
                temp_Za = temp_Z - temp_A
            omega_A = utils.calc_average(omega, [lat, lon], weights=cos_lat)
            omega_Z = omega.mean(dim=lon)
            omega_Za = omega_Z - omega_A
            C0_int = - (utils.constants().R_d / (omega[pres] * 100)) * omega_Za * temp_Za # [W/kg]
            energies['C0_int'] = C0_int
            if integrate:
                C0 = int_over_atmos(C0_int, lat, lon, pres, lon_dim=omega[lon]) # [W/m^2]
                energies['C0'] = C0
    
    # Compute the rate of generation of zonal available potential energy due to the zonally
    # averaged heating, G0 [also commonly called Gz] -----
    # NOT YET IMPLEMENTED - TO BE CALCULATED AS RESIDUALS OF THE OTHER TERMS
    
    # Compute the rate of viscous dissipation of zonal kinetic energy, D0 [also commonly 
    # called Dz] -----
    # NOT YET IMPLEMENTED - TO BE CALCULATED AS RESIDUALS OF THE OTHER TERMS
    
    # Compute eddy terms in Fourier space if spectral=True
    # ==========================================================
    if spectral:
        
        if ('Pe' in terms) | (terms is None):
        # Compute the total available potential energy eddies of wavenumber n, Pn -----
            B = utils.calc_fft(temp, dim=lon, nfft=len(temp[lon]), twosided=True, shift=True) / len(temp[lon])
            B['f_' + lon] = 360 * B['f_' + lon]
            B = B.rename({'f_' + lon : 'n'})
            Bp = B
            Bn = flip_n(B)

            Pn_int = (gamma * utils.constants().C_pd * abs(Bp) ** 2)
            energies['Pn_int'] = Pn_int
            if integrate:
                Pn = int_over_atmos(Pn_int, lat, lon, pres, lon_dim=temp[lon]) # [J/m^2]
                energies['Pn'] = Pn

        # Compute the rate of transfer of available potential energy to eddies of 
        # wavenumber n from eddies of all other wavenumbers, Sn -----
            U = utils.calc_fft(u, dim=lon, nfft=len(u[lon]), twosided=True, shift=True) / len(u[lon])
            U['f_' + lon] = 360 * U['f_' + lon]
            U = U.rename({'f_' + lon : 'n'})
            Up = U
            Un = flip_n(U)
            V = utils.calc_fft(v, dim=lon, nfft=len(v[lon]), twosided=True, shift=True) / len(v[lon])
            V['f_' + lon] = 360 * V['f_' + lon]
            V = V.rename({'f_' + lon : 'n'})
            Vp = V
            Vn = flip_n(V)
            O = utils.calc_fft(omega, dim=lon, nfft=len(omega[lon]), twosided=True, shift=True) / len(omega[lon])
            O['f_' + lon] = 360 * O['f_' + lon]
            O = O.rename({'f_' + lon : 'n'})
            Op = O
            On = flip_n(O)
                
            dBpdlat = utils.calc_gradient(Bp, dim=lat, x=(Bp[lat] * degtorad))
            dBndlat = utils.calc_gradient(Bn, dim=lat, x=(Bn[lat] * degtorad))
            dBpdp = utils.calc_gradient(Bp, dim=pres, x=(Bp[pres] * 100))
            dBndp = utils.calc_gradient(Bn, dim=pres, x=(Bn[pres] * 100))

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
                     gamma * utils.constants().R_d / Bp[pres] * \
                         (BpBnOp + BpBpOn)
            energies['Sn_int'] = Sn_int
            if integrate:
                Sn = abs(int_over_atmos(Sn_int, lat, lon, pres, lon_dim=temp[lon])) # [W/m^2]
                energies['Sn'] = Sn
                
        if ('Ke' in terms) | (terms is None):
        # Compute the total kinetic energy in eddies of wavenumber n, Kn -----
            if 'U' not in locals():
                U = utils.calc_fft(u, dim=lon, nfft=len(u[lon]), twosided=True, shift=True) / len(u[lon])
                U['f_' + lon] = 360 * U['f_' + lon]
                U = U.rename({'f_' + lon : 'n'})
                Up = U
                Un = flip_n(U)
            if 'V' not in locals():
                V = utils.calc_fft(v, dim=lon, nfft=len(v[lon]), twosided=True, shift=True) / len(v[lon])
                V['f_' + lon] = 360 * V['f_' + lon]
                V = V.rename({'f_' + lon : 'n'})
                Vp = V
                Vn = flip_n(V)

            Kn_int = abs(Up) ** 2 + abs(Vp) ** 2
            energies['Kn_int'] = Kn_int
            if integrate:
                Kn = int_over_atmos(Kn_int, lat, lon, pres, lon_dim=u[lon]) # [J/m^2]
                energies['Kn'] = Kn

        # Compute the rate of transfer of kinetic energy to eddies of wavenumber n from 
        # eddies of all other wavenumbers, Ln -----
            if 'O' not in locals():
                O = utils.calc_fft(omega, dim=lon, nfft=len(omega[lon]), twosided=True, shift=True) / len(omega[lon])
                O['f_' + lon] = 360 * O['f_' + lon]
                O = O.rename({'f_' + lon : 'n'})
                Op = O
                On = flip_n(O)
                
            dUpdp = utils.calc_gradient(Up, dim=pres, x=(Up[pres] * 100))
            dVpdp = utils.calc_gradient(Vp, dim=pres, x=(Vp[pres] * 100))
            dOpdp = utils.calc_gradient(Op, dim=pres, x=(Op[pres] * 100))
            dOndp = utils.calc_gradient(On, dim=pres, x=(On[pres] * 100))
            dVpcdl = utils.calc_gradient(Vp * cos_lat, dim=lat, x=(Vp[lat] * degtorad))
            dVncdl = utils.calc_gradient(Vn * cos_lat, dim=lat, x=(Vn[lat] * degtorad))
            dUpdl = utils.calc_gradient(Up, dim=lat, x=(Up[lat] * degtorad))
            dVpdl = utils.calc_gradient(Vp, dim=lat, x=(Vp[lat] * degtorad))

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
                Ln = abs(int_over_atmos(Ln_int, lat, lon, pres, lon_dim=u[lon])) # [W/m^2]
                energies['Ln'] = Ln
        
        if ('Ca' in terms) | (terms is None):
        # Compute the rate of transfer of zonal available potential energy to eddy 
        # available potential energy in wavenumber n, Rn -----
            if 'temp_Z' not in locals():
                temp_Z = temp.mean(dim=lon)
            if 'V' not in locals():
                V = utils.calc_fft(v, dim=lon, nfft=len(v[lon]), twosided=True, shift=True) / len(v[lon])
                V['f_' + lon] = 360 * V['f_' + lon]
                V = V.rename({'f_' + lon : 'n'})
                Vp = V
                Vn = flip_n(V)
            if 'B' not in locals():
                B = utils.calc_fft(temp, dim=lon, nfft=len(temp[lon]), twosided=True, shift=True) / len(temp[lon])
                B['f_' + lon] = 360 * B['f_' + lon]
                B = B.rename({'f_' + lon : 'n'})
                Bp = B
                Bn = flip_n(B)
            if 'O' not in locals():
                O = utils.calc_fft(omega, dim=lon, nfft=len(omega[lon]), twosided=True, shift=True) / len(omega[lon])
                O['f_' + lon] = 360 * O['f_' + lon]
                O = O.rename({'f_' + lon : 'n'})
                Op = O
                On = flip_n(O)

            dtemp_Zdlat = utils.calc_gradient(temp_Z, dim=lat, x=(temp_Z[lat] * degtorad))
            theta_Z = omega.mean(dim=lon)
            theta_Za = theta_Z - theta_A
            dtheta_Zadp = utils.calc_gradient(theta_Za, dim=pres, x=(theta_Za[pres] * 100))
            Rn_int = gamma * utils.constants().C_pd * ((dtemp_Zdlat / utils.constants().R_earth) * (Vp * Bn + Vn * Bp) + 
                                                       (p_kap * dtheta_Zadp) * (Op * Bn + On * Bp)) # [W/kg]
            energies['Rn_int'] = Rn_int
            if integrate:
                Rn = abs(int_over_atmos(Rn_int, lat, lon, pres, lon_dim=temp[lon])) # [W/m^2]
                energies['Rn'] = Rn

        if ('Ce' in terms) | (terms is None):
        # Compute the rate of conversion of available potential energy of wavenumber n 
        # to eddy kinetic energy of wavenumber n, Cn -----
            if vgradz:
                if 'U' not in locals():
                    U = utils.calc_fft(u, dim=lon, nfft=len(u[lon]), twosided=True, shift=True) / len(u[lon])
                    U['f_' + lon] = 360 * U['f_' + lon]
                    U = U.rename({'f_' + lon : 'n'})
                    Up = U
                    Un = flip_n(U)
                if 'V' not in locals():
                    V = utils.calc_fft(v, dim=lon, nfft=len(v[lon]), twosided=True, shift=True) / len(v[lon])
                    V['f_' + lon] = 360 * V['f_' + lon]
                    V = V.rename({'f_' + lon : 'n'})
                    Vp = V
                    Vn = flip_n(V)
                A = utils.calc_fft(gh, dim=lon, nfft=len(gh[lon]), twosided=True, shift=True) / len(gh[lon])
                A['f_' + lon] = 360 * A['f_' + lon]
                A = A.rename({'f_' + lon : 'n'})
                Ap = A
                An = flip_n(A)

                dApdlat = utils.calc_gradient(Ap, dim=lat, x=(Ap[lat] * degtorad))
                dAndlat = utils.calc_gradient(An, dim=lat, x=(An[lat] * degtorad))

                Cn_int = (((-1j * utils.constants().g * Up['n']) / \
                           (utils.constants().R_earth * xr.ufuncs.cos(Up[lat] * degtorad))) * \
                                (Ap * Un - An * Up)) - \
                         ((utils.constants().g / utils.constants().R_earth) * \
                                (dApdlat * Vn + dAndlat * Vp)) # [W/kg]
                energies['Cn_int'] = Cn_int
                if integrate:
                    Cn = abs(int_over_atmos(Cn_int, lat, lon, pres, lon_dim=u[lon])) # [W/m^2]
                    energies['Cn'] = Cn
            else:
                if 'O' not in locals():
                    O = utils.calc_fft(omega, dim=lon, nfft=len(omega[lon]), twosided=True, shift=True) / len(omega[lon])
                    O['f_' + lon] = 360 * O['f_' + lon]
                    O = O.rename({'f_' + lon : 'n'})
                    Op = O
                    On = flip_n(O)
                if 'B' not in locals():
                    B = utils.calc_fft(temp, dim=lon, nfft=len(temp[lon]), twosided=True, shift=True) / len(temp[lon])
                    B['f_' + lon] = 360 * B['f_' + lon]
                    B = B.rename({'f_' + lon : 'n'})
                    Bp = B
                    Bn = flip_n(B)
                Cn_int = - (utils.constants().R_d / (omega[pres] * 100)) * (Op * Bn + On * Bp) # [W/kg]
                energies['Cn_int'] = Cn_int
                if integrate:
                    Cn = abs(int_over_atmos(Cn_int, lat, lon, pres, lon_dim=temp[lon])) # [W/m^2]
                    energies['Cn'] = Cn
    
        if ('Ck' in terms) | (terms is None):
        # Compute the rate of transfer of kinetic energy to the zonally averaged flow 
        # from eddies of wavenumber n, Mn -----
            if 'v_Z' not in locals():
                v_Z = v.mean(dim=lon)
            if 'u_Z' not in locals():
                u_Z = u.mean(dim=lon)
            if 'U' not in locals():
                U = utils.calc_fft(u, dim=lon, nfft=len(u[lon]), twosided=True, shift=True) / len(u[lon])
                U['f_' + lon] = 360 * U['f_' + lon]
                U = U.rename({'f_' + lon : 'n'})
                Up = U
                Un = flip_n(U)
            if 'V' not in locals():
                V = utils.calc_fft(v, dim=lon, nfft=len(v[lon]), twosided=True, shift=True) / len(v[lon])
                V['f_' + lon] = 360 * V['f_' + lon]
                V = V.rename({'f_' + lon : 'n'})
                Vp = V
                Vn = flip_n(V)
            if 'O' not in locals():
                O = utils.calc_fft(omega, dim=lon, nfft=len(omega[lon]), twosided=True, shift=True) / len(omega[lon])
                O['f_' + lon] = 360 * O['f_' + lon]
                O = O.rename({'f_' + lon : 'n'})
                Op = O
                On = flip_n(O)
            dv_Zdlat = utils.calc_gradient(v_Z, dim=lat, x=(v[lat] * degtorad))
            du_Zndlat = utils.calc_gradient(u_Z / xr.ufuncs.cos(u[lat] * degtorad), 
                                            dim=lat, x=(u[lat] * degtorad))
            dv_Zdp = utils.calc_gradient(v_Z, dim=pres, x=(v[pres] * 100))
            du_Zdp = utils.calc_gradient(u_Z, dim=pres, x=(u[pres] * 100))

            Mn_int = (-2 * Up * Un * v_Z * tan_lat / utils.constants().R_earth) + \
                     (2 * Vp * Vn * dv_Zdlat / utils.constants().R_earth + (Vp * On + Vn * Op) * dv_Zdp) + \
                     ((Up * On + Un * Op) * du_Zdp) + \
                     ((Up * Vn + Un * Vp) * xr.ufuncs.cos(u[lat] * degtorad) / \
                         utils.constants().R_earth * du_Zndlat) # [W/kg]
            energies['Mn_int'] = Mn_int
            if integrate:
                Mn = abs(int_over_atmos(Mn_int, lat, lon, pres, lon_dim=u[lon])) # [W/m^2]
                energies['Mn'] = Mn
        
        # Compute the rate of generation of zonal available potential energy of wavenumber 
        # n due to nonadiabatic heating, Gn -----
        # NOT YET IMPLEMENTED - TO BE CALCULATED AS RESIDUALS OF THE OTHER TERMS
        
        # Compute the rate of dissipation of the kinetic energy of eddies of wavenumber n, 
        # Dn -----
        # NOT YET IMPLEMENTED - TO BE CALCULATED AS RESIDUALS OF THE OTHER TERMS
        
    else:
        
        if ('Pe' in terms) | (terms is None):
        # Compute the total eddy available potential energy, Pe [also commonly called 
        # Ae] -----
            if 'temp_Z' not in locals():
                temp_Z = temp.mean(dim=lon)
            temp_z = temp - temp_Z
            Pe_int = gamma * utils.constants().C_pd / 2 * temp_z ** 2  # [J/kg]
            energies['Pe_int'] = Pe_int
            if integrate:
                Pe = int_over_atmos(Pe_int, lat, lon, pres) # [J/m^2]
                energies['Pe'] = Pe
        
        if ('Ke' in terms) | (terms is None):
        # Compute the total eddy kinetic energy, Ke -----
            # NOT YET IMPLEMENTED
            pass
            
        if ('Ca' in terms) | (terms is None):
        # Compute the rate of transfer of total available potential energy in the zonally 
        # averaged temperature distribution (P0) to total eddy available potential energy 
        # (Pe), Ca -----
            # NOT YET IMPLEMENTED
            pass
        
        if ('Ce' in terms) | (terms is None):
        # Compute the rate of transfer of total eddy available potential energy (Pe) to 
        # total eddy kinetic energy (Ke), Ce -----
            # NOT YET IMPLEMENTED
            pass
        
        if ('Ck' in terms) | (terms is None):
        # Compute the rate of transfer of total eddy kinetic energy (Ke) to total kinetic 
        # energy in zonally averaged motion (K0), Ck -----
            # NOT YET IMPLEMENTED
            pass
        
        # Compute the rate of generation of eddy available potential energy (Ae), Ge -----
        # NOT YET IMPLEMENTED - TO BE CALCULATED AS RESIDUALS OF THE OTHER TERMS
        
        # Compute the rate of dissipation of eddy kinetic energy (Ke), De -----
        # NOT YET IMPLEMENTED - TO BE CALCULATED AS RESIDUALS OF THE OTHER TERMS
        
    return energies
        

# ===================================================================================================
# Indices
# ===================================================================================================
def compute_nino3(da_sst_anom):
    ''' Returns nino3 index '''  
    
    box = [-5.0,5.0,360.0-150.0,360.0-90.0] # [lat_min,lat_max,lon_min,lon_max]
    
    # Account for datasets with negative longitudes -----
    if np.any(da_sst_anom['lon'] < 0):
        box[2] = box[2] - 360
        box[3] = box[3] - 360
        
    return calc_boxavg_latlon(da_sst_anom,box)


# ===================================================================================================
def compute_nino34(da_sst_anom):
    ''' Returns nino3.4 index '''  
    
    box = [-5.0,5.0,360.0-170.0,360.0-120.0] # [lat_min,lat_max,lon_min,lon_max]
    
    # Account for datasets with negative longitudes -----
    if np.any(da_sst_anom['lon'] < 0):
        box[2] = box[2] - 360
        box[3] = box[3] - 360
        
    return calc_boxavg_latlon(da_sst_anom,box)


# ===================================================================================================
def compute_nino4(da_sst_anom):
    ''' Returns nino4 index '''  
    
    box = [-5.0,5.0,360.0-160.0,360.0-150.0] # [lat_min,lat_max,lon_min,lon_max]
    
    # Account for datasets with negative longitudes -----
    if np.any(da_sst_anom['lon'] < 0):
        box[2] = box[2] - 360
        box[3] = box[3] - 360
        
    return calc_boxavg_latlon(da_sst_anom,box)


# ===================================================================================================
def compute_emi(da_sst_anom):
    ''' Returns EMI index ''' 
    
    boxA = [-10.0,10.0,360.0-165.0,360.0-140.0] # [lat_min,lat_max,lon_min,lon_max]
    boxB = [-15.0,5.0,360.0-110.0,360.0-70.0] # [lat_min,lat_max,lon_min,lon_max]
    boxC = [-10.0,20.0,125.0,145.0] # [lat_min,lat_max,lon_min,lon_max]
    
    # Account for datasets with negative longitudes -----
    if np.any(da_sst_anom['lon'] < 0):
        boxA[2] = boxA[2] - 360
        boxA[3] = boxA[3] - 360
        boxB[2] = boxB[2] - 360
        boxB[3] = boxB[3] - 360
        boxC[2] = boxC[2] - 360
        boxC[3] = boxC[3] - 360
        
    da_sstA = utils.calc_boxavg_latlon(da_sst_anom,boxA)
    da_sstB = utils.calc_boxavg_latlon(da_sst_anom,boxB)
    da_sstC = utils.calc_boxavg_latlon(da_sst_anom,boxC)
    
    return da_sstA - 0.5*da_sstB - 0.5*da_sstC


# ===================================================================================================
def compute_dmi(da_sst_anom):
    ''' Returns DMI index ''' 
    
    boxW = [-10.0,10.0,50.0,70.0] # [lat_min,lat_max,lon_min,lon_max]
    boxE = [-10.0,0.0,90.0,110.0] # [lat_min,lat_max,lon_min,lon_max]
    
    if np.any(da_sst_anom['lon'] < 0):
        boxW[2] = boxW[2] - 360
        boxW[3] = boxW[3] - 360
        boxE[2] = boxE[2] - 360
        boxE[3] = boxE[3] - 360
        
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
