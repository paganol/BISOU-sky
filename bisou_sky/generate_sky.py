import warnings
from typing import List, Union, Optional, Dict
from importlib.resources import files, as_file

import numpy as np
import healpy as hp
import pysm3
import pysm3.units as u
from scipy import interpolate
from astropy.constants import h, c, k_B

# -----------------------------------------------------------------------------
# CONSTANTS
# -----------------------------------------------------------------------------

# CMB
# http://dx.doi.org/10.1088/0004-637X/707/2/916
T_CMB_K = 2.72549

# CIB Reference Frequency
NU0_CIB = 3000.0  # GHz

# Solar Dipole
DIPOLE_DIRECTION = hp.ang2vec(np.deg2rad(90.0 - 48.253), np.deg2rad(264.021))
BETA_SUN = 369816.0 / c.value

# -----------------------------------------------------------------------------
# HELPER FUNCTIONS
# -----------------------------------------------------------------------------

def planck(nu, tcmb):
    """
    Calculates the blackbody spectral radiance B_nu(T).
    Returns: Spectral radiance in MJy/sr
    """
    # x = h*nu / k*T
    x = h.value / k_B.value / tcmb * 1e9 * nu
    return 2.0 * h.value * nu ** 3 * 1e27 / c.value / c.value / (np.exp(x) - 1) * 1e20


def CIB(nu, acib=None, betacib=None, tcib=None, model='fixsen'):
    """
    Calculates CIB emission for various models.
    Defaults for A, Beta, and T are handled explicitly inside each model block.
    """
    model = model.lower()

    # Helper: Dimensionless frequency x = h*nu / k*T
    def get_x(freq_ghz, temp):
        return h.value * (freq_ghz * 1e9) / (k_B.value * temp)

    if model == 'fixsen':
        # Defaults for Fixsen
        _acib = 1.3e-05 if acib is None else acib
        _betacib = 0.64 if betacib is None else betacib
        _tcib = 18.5 if tcib is None else tcib
        
        # Formula: A * (nu/nu0)^beta * B_nu(T)
        return _acib * (nu / NU0_CIB)**_betacib * planck(nu, _tcib)

    elif model == 'abitbol':
        # Defaults for Abitbol
        _acib = 0.346 if acib is None else acib
        _betacib = 0.86 if betacib is None else betacib
        _tcib = 18.8 if tcib is None else tcib
        
        # Formula: A * x^beta * x^3 / (e^x - 1)
        x = get_x(nu, _tcib)
        return _acib * x**_betacib * x**3 / (np.exp(x) - 1)

    elif model == 'gispert':
        # Defaults for Gispert
        _acib = 8.8e-05 if acib is None else acib
        _betacib = 1.4 if betacib is None else betacib
        _tcib = 13.6 if tcib is None else tcib
        
        # Formula: A * (nu/nu0)^beta * B_nu(T)
        return _acib * (nu / NU0_CIB)**_betacib * planck(nu, _tcib)
    
    else:
        raise ValueError(f"Unknown CIB model: {model}")


def monopole_and_dipole_CMB(nu, tcmb, beta_sun, dipole_direction, nside):
    npix = hp.nside2npix(nside)
    beta_dot_n = beta_sun * np.dot(dipole_direction, hp.pix2vec(nside, np.arange(npix)))
    gamma = 1 / np.sqrt(1 - beta_sun ** 2)
    return planck(nu, tcmb / gamma / (1 - beta_dot_n))


def monopole_and_dipole_CIB(nu, beta_sun, dipole_direction, nside, 
                            acib=None, betacib=None, tcib=None, model='fixsen'):
    """
    Computes boosted CIB monopole and dipole.
    """
    npix = hp.nside2npix(nside)
    beta_dot_n = beta_sun * np.dot(dipole_direction, hp.pix2vec(nside, np.arange(npix)))
    gamma = 1 / np.sqrt(1 - beta_sun ** 2)
    
    # Doppler shifted Frequency
    nu_boosted = nu * gamma * (1 - beta_dot_n)
    
    # Calculate Rest Frame CIB at Boosted Frequency
    cib_val = CIB(nu_boosted, acib=acib, betacib=betacib, tcib=tcib, model=model)
    
    # Apply intensity boost: I / (gamma * (1 - beta*n))^3
    return cib_val / (gamma * (1 - beta_dot_n)) ** 3


def extragalactic_CO(nu, aco, template: Optional[str] = None):
    if not template:
        ref = files("bisou_sky").joinpath("data/extragalactic_co_template.txt")
        with as_file(ref) as template_path:
            freqs, signal = np.loadtxt(template_path, unpack=True)
    else:
        freqs, signal = np.loadtxt(template, unpack=True)

    f = interpolate.interp1d(
        np.log(freqs), 
        np.log(signal),
        kind="linear", 
        bounds_error=False, 
        fill_value="extrapolate"
    )
    return aco * np.exp(f(np.log(nu)))


def deltaI_y_distortions(nu, y, tcmb):
    x = h.value / k_B.value / tcmb * 1e9 * nu
    I0 = 2.0 * h.value / c.value / c.value * (k_B.value * tcmb / h.value) ** 3 * 1e20
    fac = x ** 4 * np.exp(x) / (np.exp(x) - 1) ** 2
    return I0 * fac * (x / np.tanh(0.5 * x) - 4) * y


def deltaI_y_distortions_relativistic_corrections(nu, y, tesz, tcmb):
    x = h.value / k_B.value / tcmb * 1e9 * nu
    I0 = 2.0 * h.value / c.value / c.value * (k_B.value * tcmb / h.value) ** 3 * 1e20
    thetae = tesz / 510.998950

    xtil = x / np.tanh(0.5 * x)
    stil = x / np.sinh(0.5 * x)

    Y1 = -10.0 + 23.5 * xtil - 8.4 * xtil**2 + 0.7 * xtil**3 + stil**2 * (-4.2 + 1.4 * xtil)

    Y2 = (-7.5 + 127.875 * xtil - 173.6 * xtil**2 + 65.8 * xtil**3 
          - 8.8 * xtil**4 + 11.0 / 30.0 * xtil**5 
          + stil**2 * (-86.8 + 131.6 * xtil - 48.4 * xtil**2 + 143.0 / 30.0 * xtil**3) 
          + stil**4 * (-8.8 + 187.0 / 60.0 * xtil))

    Y3 = (7.5 + 313.125 * xtil - 1419.6 * xtil**2 + 1425.3 * xtil**3 
          - 18594.0 / 35.0 * xtil**4 + 12059.0 / 140.0 * xtil**5 
          - 128.0 / 21.0 * xtil**6 + 16.0 / 105.0 * xtil**7
          + stil**2 * (-709.8 + 2850.6 * xtil - 102267.0 / 35.0 * xtil**2 + 156767.0 / 140.0 * xtil**3 
          - 1216.0 / 7.0 * xtil**4 + 64.0 / 7.0 * xtil**5)  
          + stil**4 * (-18594.0 / 35.0 + 205003.0 / 280.0 * xtil - 1920.0 / 7.0 * xtil**2 
          + 1024.0 / 35.0 * xtil**3) + stil**6 * (-544.0 / 21.0 + 992.0 / 105.0 * xtil))

    Y4 = (-4.21875 + 237.3046875 * xtil - 6239.1 * xtil**2 + 15368.175 * xtil**3 
          - 12438.9 * xtil**4 + 4446.2875 * xtil**5 
          - 16568.0 / 21.0 * xtil**6 + 7516.0 / 105.0 * xtil**7
          - 22.0 / 7.0 * xtil**8
          + 11.0 / 210.0 * xtil**9
          + stil**2 * (-3119.55 + 30736.35 * xtil - 68413.95 * xtil**2 + 57801.7375 * xtil**3 
          - 157396.0 / 7.0 * xtil**4 + 30064.0 / 7.0 * xtil**5
          - 2717.0 / 7.0 * xtil**6 + 2761.0 / 210.0 * xtil**7)
          + stil**4 * (-12438.9 + 37793.44375 * xtil - 248520.0 / 7.0 * xtil**2 
          + 481024.0 / 35.0 * xtil**3 - 15972.0 / 7.0 * xtil**4
          + 18689.0 / 140.0 * xtil**5) + stil**6 * (
          - 70414.0 / 21.0 + 465992.0 / 105.0 * xtil - 11792.0 / 7.0 * xtil**2 + 19778.0 / 105.0 * xtil**3)
          + stil**8 * (-682.0 / 7.0 + 7601.0 / 210.0 * xtil))

    delta_I_over_I = x**4 * np.exp(x) / (np.exp(x) - 1.0)**2 * (
        Y1 * thetae + Y2 * thetae**2
        + Y3 * thetae**3 
        + Y4 * thetae**4
    ) * y

    return I0 * delta_I_over_I


def deltaI_mu_distortions(nu, mu, tcmb):
    x = h.value / k_B.value / tcmb * 1e9 * nu
    I0 = 2.0 * h.value / c.value / c.value * (k_B.value * tcmb / h.value) ** 3 * 1e20
    fac = x ** 4 * np.exp(x) / (np.exp(x) - 1) ** 2
    return I0 * fac * (1 / 2.1923 - 1 / x) * mu


# -----------------------------------------------------------------------------
# MAIN GENERATOR
# -----------------------------------------------------------------------------

def get_sky(
    freqs,
    nside,
    models: Optional[List[str]] = None,
    fwhm_deg: float = 1.0,
    add_cmb_monopole_and_dipole: bool = True,
    add_cib_monopole_and_dipole: bool = True,
    cib_model: str = 'fixsen',
    A_cib: Optional[float] = None,
    cib_params: Optional[Dict[str, float]] = None,
    y_distortions: Optional[float] = 1e-6,
    t_e_sz: Optional[float] = 1.24,
    mu_distortions: Optional[float] = 1e-8,
    A_eg_CO: Optional[float] = 1.0,
    maps_coord: str = "C",
):
    """
    Args:
    - ``freqs``: array of frequencies of dimension n_freq
    - ``nside``: healpix resolution of the maps generated
    - ``models``: foreground to use in pysm3 jargon (default: ["c1", "d2", "s1", "a1", "f1"])
    - ``fwhm_deg``: Gaussian smoothing in deg (default: 1 degree)
    - ``add_cmb_monopole_and_dipole``: add CMB monopole and dipole
    - ``add_cib_monopole_and_dipole``: add CIB monopole and dipole
    - ``cib_model``: CIB model name ('fixsen', 'abitbol', 'gispert')
    - ``A_cib``: CIB amplitude override (default: None, uses model default)
    - ``cib_params``: Dictionary with 'beta' and 'T' to override CIB model defaults
    - ``y_distortions``: add y-type distortions with amplitude y_distortions
    - ``t_e_sz``: electron temperature t_e_sz for relativistic corrections in keV
    - ``mu_distortions``: add mu-type distortions with amplitude mu_distortions
    - ``A_eg_CO``: add extragalactic CO signal with amplitude A_eg_CO
    - ``maps_coord``: coordinates of the output maps. Default celestial coordinates
    """
    
    # Safe default handling for models list
    if models is None:
        models = ["c1", "d2", "s1", "a1", "f1"]
    
    # Handle CIB override params
    user_beta_cib = None
    user_t_cib = None
    if cib_params:
        user_beta_cib = cib_params.get('beta')
        user_t_cib = cib_params.get('T')

    if models:
        sky = pysm3.Sky(nside=nside, preset_strings=models, output_unit="MJy/sr")

    npix = hp.nside2npix(nside)
    m = np.zeros((len(freqs), npix))

    if maps_coord != "G":
        r = hp.Rotator(coord=["G", maps_coord])

    for ifreq, freq in enumerate(freqs):
        if models:
            m[ifreq] += sky.get_emission(freq * u.GHz)[0].value
        
        if add_cmb_monopole_and_dipole:
            m[ifreq] += monopole_and_dipole_CMB(
                freq, T_CMB_K, BETA_SUN, DIPOLE_DIRECTION, nside,
            )
            
        if add_cib_monopole_and_dipole:
            m[ifreq] += monopole_and_dipole_CIB(
                freq,
                BETA_SUN,
                DIPOLE_DIRECTION,
                nside,
                acib=A_cib,      # None unless overridden by user
                betacib=user_beta_cib,
                tcib=user_t_cib,
                model=cib_model
            )

        if fwhm_deg > 0.0:
            m[ifreq] = hp.smoothing(
                m[ifreq],
                fwhm=np.deg2rad(fwhm_deg),
                use_pixel_weights=True if nside > 16 else False,
            )

        if maps_coord != "G":
            m[ifreq] = r.rotate_map_pixel(m[ifreq])

    if A_eg_CO:
        m += extragalactic_CO(freqs, A_eg_CO)[:, np.newaxis]

    if y_distortions:
        m += deltaI_y_distortions(
            freqs, y_distortions, T_CMB_K,
        )[:, np.newaxis]
        if t_e_sz:
            m += deltaI_y_distortions_relativistic_corrections(
                freqs, y_distortions, t_e_sz, T_CMB_K
            )[:, np.newaxis]

    if mu_distortions:
        m += deltaI_mu_distortions(
            freqs, mu_distortions, T_CMB_K,
        )[:, np.newaxis]

    return m