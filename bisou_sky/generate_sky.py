import warnings
from typing import List, Union, Optional, Dict
from importlib.resources import files, as_file

import numpy as np
import healpy as hp
import pysm3
import pysm3.units as u
from scipy import interpolate
from astropy.constants import h, c, k_B

# Added for ZodiPy support
import astropy.units as au
from astropy.time import Time
from astropy.coordinates import SkyCoord
try:
    import zodipy
except ImportError:
    zodipy = None

# -----------------------------------------------------------------------------
# CONSTANTS
# -----------------------------------------------------------------------------

# CMB Temperature (Fixsen 2009)
# http://dx.doi.org/10.1088/0004-637X/707/2/916
T_CMB_K = 2.72549

# CIB Reference Frequency
NU0_CIB = 3000.0  # GHz

# Solar Dipole Parameters
# Direction and velocity of the Sun relative to the CMB rest frame
DIPOLE_DIRECTION = hp.ang2vec(np.deg2rad(90.0 - 48.253), np.deg2rad(264.021))
BETA_SUN = 369816.0 / c.value

# Zodiacal Light Defaults
ZODI_DEFAULT_TIME = "2040-01-01"

# -----------------------------------------------------------------------------
# HELPER FUNCTIONS
# -----------------------------------------------------------------------------

def planck(nu: float, tcmb: float) -> float:
    """
    Calculates the blackbody spectral radiance B_nu(T).

    Parameters
    ----------
    nu : float or np.ndarray
        Frequency in GHz.
    tcmb : float
        Blackbody temperature in Kelvin.

    Returns
    -------
    float or np.ndarray
        Spectral radiance in MJy/sr.
    """
    # x = h*nu / k*T
    # Constants converted to compatible units for GHz input
    x = h.value / k_B.value / tcmb * 1e9 * nu
    return 2.0 * h.value * nu ** 3 * 1e27 / c.value / c.value / (np.exp(x) - 1) * 1e20


def CIB(nu: float, acib: float = None, betacib: float = None, tcib: float = None, model: str = 'fixsen') -> float:
    """
    Calculates Cosmic Infrared Background (CIB) emission for various parametric models.

    Parameters
    ----------
    nu : float or np.ndarray
        Frequency in GHz.
    acib : float, optional
        Amplitude of the CIB. Defaults depend on the selected model.
    betacib : float, optional
        Spectral index beta. Defaults depend on the selected model.
    tcib : float, optional
        CIB Temperature in Kelvin. Defaults depend on the selected model.
    model : str, optional
        The CIB model to use. Options are 'fixsen', 'abitbol', 'gispert'.
        Default is 'fixsen'.

    Returns
    -------
    float or np.ndarray
        CIB intensity in MJy/sr.

    Raises
    ------
    ValueError
        If an unknown model string is provided.
    """
    model = model.lower()

    def get_x(freq_ghz, temp):
        return h.value * (freq_ghz * 1e9) / (k_B.value * temp)

    if model == 'fixsen':
        # Fixsen et al. (2009)
        _acib = 1.3e-05 if acib is None else acib
        _betacib = 0.64 if betacib is None else betacib
        _tcib = 18.5 if tcib is None else tcib
        return _acib * (nu / NU0_CIB)**_betacib * planck(nu, _tcib)

    elif model == 'abitbol':
        # Abitbol et al. (2017)
        _acib = 0.346 if acib is None else acib
        _betacib = 0.86 if betacib is None else betacib
        _tcib = 18.8 if tcib is None else tcib
        x = get_x(nu, _tcib)
        return _acib * x**_betacib * x**3 / (np.exp(x) - 1)

    elif model == 'gispert':
        # Gispert et al. (2000)
        _acib = 8.8e-05 if acib is None else acib
        _betacib = 1.4 if betacib is None else betacib
        _tcib = 13.6 if tcib is None else tcib
        return _acib * (nu / NU0_CIB)**_betacib * planck(nu, _tcib)
    
    else:
        raise ValueError(f"Unknown CIB model: {model}")


def monopole_and_dipole_CMB(nu: float, tcmb: float, beta_sun: float, dipole_direction: np.ndarray, nside: int) -> np.ndarray:
    """
    Calculates the CMB monopole and dipole induced by the solar velocity (Doppler boosting).

    Parameters
    ----------
    nu : float
        Frequency in GHz.
    tcmb : float
        CMB temperature in Kelvin.
    beta_sun : float
        Solar velocity v/c.
    dipole_direction : np.ndarray
        Vector defining the direction of the solar dipole.
    nside : int
        HEALPix Nside parameter.

    Returns
    -------
    np.ndarray
        HEALPix map of the CMB monopole and dipole in MJy/sr.
    """
    npix = hp.nside2npix(nside)
    beta_dot_n = beta_sun * np.dot(dipole_direction, hp.pix2vec(nside, np.arange(npix)))
    gamma = 1 / np.sqrt(1 - beta_sun ** 2)
    # The temperature is Doppler shifted: T' = T / (gamma * (1 - beta * n))
    return planck(nu, tcmb / gamma / (1 - beta_dot_n))


def monopole_and_dipole_CIB(nu: float, beta_sun: float, dipole_direction: np.ndarray, nside: int, 
                            acib: float = None, betacib: float = None, tcib: float = None, model: str = 'fixsen') -> np.ndarray:
    """
    Calculates the CIB monopole and dipole induced by the solar velocity (Doppler boosting and aberration).

    Parameters
    ----------
    nu : float
        Frequency in GHz.
    beta_sun : float
        Solar velocity v/c.
    dipole_direction : np.ndarray
        Vector defining the direction of the solar dipole.
    nside : int
        HEALPix Nside parameter.
    acib, betacib, tcib : float, optional
        CIB model parameters (Amplitude, Spectral Index, Temperature).
    model : str, optional
        CIB model name.

    Returns
    -------
    np.ndarray
        HEALPix map of the CIB monopole and dipole in MJy/sr.
    """
    npix = hp.nside2npix(nside)
    beta_dot_n = beta_sun * np.dot(dipole_direction, hp.pix2vec(nside, np.arange(npix)))
    gamma = 1 / np.sqrt(1 - beta_sun ** 2)
    
    # Doppler shifted Frequency in the rest frame
    nu_boosted = nu * gamma * (1 - beta_dot_n)
    
    # Calculate Rest Frame CIB intensity at the Boosted Frequency
    cib_val = CIB(nu_boosted, acib=acib, betacib=betacib, tcib=tcib, model=model)
    
    # Apply intensity boost factor: I_obs / I_rest = 1 / (gamma * (1 - beta*n))^3
    return cib_val / (gamma * (1 - beta_dot_n)) ** 3


def zodiacal_emission(
    nu: float, 
    nside: int, 
    time: Union[str, Time] = ZODI_DEFAULT_TIME, 
    observer: str = "semb-l2", 
    model_name: str = "planck18", 
    zodi_coords: Optional[SkyCoord] = None
) -> np.ndarray:
    """
    Calculates Zodiacal light emission using the ZodiPy library.
    Updated to initialize ZodiPy Model with frequency directly.
    """
    if zodipy is None:
        raise ImportError("The 'zodipy' package is required to generate zodiacal emission.")

    # Generate coordinates if not provided (slower if called in a loop)
    if zodi_coords is None:
        npix = hp.nside2npix(nside)
        if isinstance(time, str):
            time = Time(time)
        
        theta, phi = hp.pix2ang(nside, np.arange(npix))
        lon = np.degrees(phi) * au.deg
        lat = (90.0 - np.degrees(theta)) * au.deg
        
        zodi_coords = SkyCoord(
            l=lon, 
            b=lat, 
            frame="galactic", 
            obstime=time
        )

    # Instantiate model with the frequency (x) and model name explicitly.
    # The frequency must be an astropy Quantity (e.g., 100 * au.GHz).
    model = zodipy.Model(nu * au.GHz, name=model_name, extrapolate=True)
    
    # Evaluate the model. 
    # Note: Frequency is already defined in the model object, so we only pass coordinates and observer.
    emission = model.evaluate(
        zodi_coords, 
        obspos=observer
    )
    
    return emission.to(au.MJy / au.sr).value


def extragalactic_CO(nu: float, aco: float, template: Optional[str] = None) -> float:
    """
    Calculates Extragalactic CO emission based on a template.

    Parameters
    ----------
    nu : float
        Frequency in GHz.
    aco : float
        Amplitude scaling factor for the CO signal.
    template : str, optional
        Path to the CO template file. If None, uses the internal 'bisou_sky' data.

    Returns
    -------
    float
        CO emission intensity.
    """
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


def deltaI_y_distortions(nu: float, y: float, tcmb: float) -> float:
    """
    Calculates the thermal Sunyaev-Zel'dovich (y-distortion) effect intensity.

    Parameters
    ----------
    nu : float or np.ndarray
        Frequency in GHz.
    y : float
        Compton y-parameter (amplitude of the distortion).
    tcmb : float
        CMB temperature in Kelvin.

    Returns
    -------
    float or np.ndarray
        Intensity change delta_I in MJy/sr.
    """
    x = h.value / k_B.value / tcmb * 1e9 * nu
    I0 = 2.0 * h.value / c.value / c.value * (k_B.value * tcmb / h.value) ** 3 * 1e20
    fac = x ** 4 * np.exp(x) / (np.exp(x) - 1) ** 2
    return I0 * fac * (x / np.tanh(0.5 * x) - 4) * y


def deltaI_y_distortions_relativistic_corrections(nu: float, y: float, tesz: float, tcmb: float) -> float:
    """
    Calculates relativistic corrections to the thermal SZ effect using the Itoh et al. expansion.

    Parameters
    ----------
    nu : float or np.ndarray
        Frequency in GHz.
    y : float
        Compton y-parameter.
    tesz : float
        Electron temperature in keV.
    tcmb : float
        CMB temperature in Kelvin.

    Returns
    -------
    float or np.ndarray
        Relativistic correction intensity delta_I in MJy/sr.
    """
    x = h.value / k_B.value / tcmb * 1e9 * nu
    I0 = 2.0 * h.value / c.value / c.value * (k_B.value * tcmb / h.value) ** 3 * 1e20
    thetae = tesz / 510.998950 # Electron temperature in units of electron rest mass energy

    xtil = x / np.tanh(0.5 * x)
    stil = x / np.sinh(0.5 * x)

    # Coefficients for the analytical approximation (Itoh et al.)
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
        Y1 * thetae + Y2 * thetae**2 + Y3 * thetae**3 + Y4 * thetae**4
    ) * y
    return I0 * delta_I_over_I


def deltaI_mu_distortions(nu: float, mu: float, tcmb: float) -> float:
    """
    Calculates the spectral distortion due to chemical potential (mu-type) distortions.

    Parameters
    ----------
    nu : float or np.ndarray
        Frequency in GHz.
    mu : float
        Chemical potential mu parameter.
    tcmb : float
        CMB temperature in Kelvin.

    Returns
    -------
    float or np.ndarray
        Intensity change delta_I in MJy/sr.
    """
    x = h.value / k_B.value / tcmb * 1e9 * nu
    I0 = 2.0 * h.value / c.value / c.value * (k_B.value * tcmb / h.value) ** 3 * 1e20
    fac = x ** 4 * np.exp(x) / (np.exp(x) - 1) ** 2
    return I0 * fac * (1 / 2.1923 - 1 / x) * mu


# -----------------------------------------------------------------------------
# MAIN GENERATOR
# -----------------------------------------------------------------------------

def get_sky(
    freqs: np.ndarray,
    nside: int,
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
    add_zodi: bool = False,
    zodi_model_name: str = "dirbe",
    zodi_time: Union[str, Time] = ZODI_DEFAULT_TIME,
    zodi_observer: str = "semb-l2",
    maps_coord: str = "C",
) -> np.ndarray:
    """
    Generates a multi-frequency sky map containing various cosmological and foreground components.

    The components are generated in the following order:
    1. Galactic Foregrounds (via PySM3)
    2. CMB Monopole & Dipole (Doppler boosted)
    3. CIB Monopole & Dipole (Doppler boosted)
    4. Zodiacal Emission (via ZodiPy)
    5. Smoothing (Gaussian beam)
    6. Coordinate Rotation (if requested)
    7. Isotropic backgrounds (Extragalactic CO, y-distortion, mu-distortion)

    Parameters
    ----------
    freqs : np.ndarray
        Array of frequencies in GHz.
    nside : int
        HEALPix Nside resolution.
    models : list of str, optional
        PySM3 model strings (e.g., ["c1", "d2"]). Defaults to ["c1", "d2", "s1", "a1", "f1"].
    fwhm_deg : float, optional
        Gaussian beam smoothing FWHM in degrees. Default is 1.0.
    add_cmb_monopole_and_dipole : bool, optional
        Whether to add the Solar dipole-induced CMB signal. Default is True.
    add_cib_monopole_and_dipole : bool, optional
        Whether to add the Solar dipole-induced CIB signal. Default is True.
    cib_model : str, optional
        CIB model to use ('fixsen', 'abitbol', 'gispert'). Default is 'fixsen'.
    A_cib : float, optional
        Override for CIB amplitude. If None, uses model default.
    cib_params : dict, optional
        Dictionary to override 'beta' and 'T' for CIB. e.g., {'beta': 1.2, 'T': 20.0}.
    y_distortions : float, optional
        Amplitude of thermal SZ y-distortion. Default is 1e-6.
    t_e_sz : float, optional
        Electron temperature in keV for relativistic SZ corrections. Default is 1.24 keV.
    mu_distortions : float, optional
        Amplitude of spectral mu-distortion. Default is 1e-8.
    A_eg_CO : float, optional
        Amplitude of extragalactic CO signal. Default is 1.0.
    add_zodi : bool, optional
        Whether to add Zodiacal emission using ZodiPy. Default is False.
    zodi_model_name : str, optional
        ZodiPy model name (e.g., 'dirbe', 'planck18'). Default is 'dirbe'.
    zodi_time : str or astropy.time.Time, optional
        Date of observation for Zodi calculation. Default is ZODI_DEFAULT_TIME.
    zodi_observer : str, optional
        Observer location for Zodi (e.g., 'semb-l2', 'earth'). Default is 'semb-l2'.
    maps_coord : str, optional
        Output coordinate system ('G' for Galactic, 'C' for Celestial/Equatorial). 
        PySM3 generates in Galactic; this rotates the result. Default is 'C'.

    Returns
    -------
    np.ndarray
        Array of HEALPix maps with shape (len(freqs), npix) in units of MJy/sr.
    """
    if models is None:
        models = ["c1", "d2", "s1", "a1", "f1"]
    
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

    # Optimization: Pre-compute coordinates for Zodi to avoid rebuilding them per frequency
    zodi_coords = None
    if add_zodi and zodipy is not None:
        if isinstance(zodi_time, str):
            zodi_time = Time(zodi_time)
        
        theta, phi = hp.pix2ang(nside, np.arange(npix))
        lon = np.degrees(phi) * au.deg
        lat = (90.0 - np.degrees(theta)) * au.deg
        
        zodi_coords = SkyCoord(
            l=lon, 
            b=lat, 
            frame="galactic", 
            obstime=zodi_time
        )

    for ifreq, freq in enumerate(freqs):
        # 1. Galactic Foregrounds
        if models:
            m[ifreq] += sky.get_emission(freq * u.GHz)[0].value
        
        # 2. CMB Dipole
        if add_cmb_monopole_and_dipole:
            m[ifreq] += monopole_and_dipole_CMB(
                freq, T_CMB_K, BETA_SUN, DIPOLE_DIRECTION, nside,
            )
            
        # 3. CIB Dipole
        if add_cib_monopole_and_dipole:
            m[ifreq] += monopole_and_dipole_CIB(
                freq,
                BETA_SUN,
                DIPOLE_DIRECTION,
                nside,
                acib=A_cib,
                betacib=user_beta_cib,
                tcib=user_t_cib,
                model=cib_model
            )

        # 4. Zodiacal Light
        if add_zodi:
            m[ifreq] += zodiacal_emission(
                freq,
                nside,
                time=zodi_time,
                observer=zodi_observer,
                model_name=zodi_model_name,
                zodi_coords=zodi_coords
            )

        # 5. Smoothing
        if fwhm_deg > 0.0:
            m[ifreq] = hp.smoothing(
                m[ifreq],
                fwhm=np.deg2rad(fwhm_deg),
                use_pixel_weights=True if nside > 16 else False,
            )

        # 6. Rotation
    if maps_coord != "G":
        m = r.rotate_map_pixel(m)

    # 7. Monopole/Isotropic additions (Added after rotation as they are isotropic)
    if A_eg_CO:
        m += extragalactic_CO(freqs, A_eg_CO)[:, np.newaxis]

    if y_distortions:
        m += deltaI_y_distortions(freqs, y_distortions, T_CMB_K)[:, np.newaxis]
        if t_e_sz:
            m += deltaI_y_distortions_relativistic_corrections(
                freqs, y_distortions, t_e_sz, T_CMB_K
            )[:, np.newaxis]

    if mu_distortions:
        m += deltaI_mu_distortions(freqs, mu_distortions, T_CMB_K)[:, np.newaxis]

    return m