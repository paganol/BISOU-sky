import numpy as np
import healpy as hp
from typing import Union, List
import pysm3.units as u
import pysm3
from astropy.constants import h, c, k_B

import warnings

warnings.filterwarnings("ignore")

DIPOLE_DIRECTION = hp.ang2vec(np.deg2rad(90.0 - 48.253), np.deg2rad(264.021))
BETA_SUN = 369816.0 / c.value

# http://dx.doi.org/10.1088/0004-637X/707/2/916
T_CMB_K = 2.72549

# http://iopscience.iop.org/article/10.1086/306383/pdf
A_CIB = 1.3e-5
SPECTRAL_INDEX_CIB = 0.64
T_CIB_K = 18.5
NU0_CIB = 3000.0


def planck(nu, tcmb):
    x = h.value / k_B.value / tcmb * 1e9 * nu
    return 2.0 * h.value * nu ** 3 * 1e27 / c.value / c.value / (np.exp(x) - 1) * 1e20


def monopole_and_dipole_CMB(nu, tcmb, beta_sun, dipole_direction, nside):
    npix = hp.nside2npix(nside)
    beta_dot_n = beta_sun * np.dot(dipole_direction, hp.pix2vec(nside, np.arange(npix)))
    gamma = 1 / np.sqrt(1 - beta_sun ** 2)
    return planck(nu, tcmb / gamma / (1 - beta_dot_n))


def CIB(nu, acib, betacib, tcib):
    return acib * (nu / NU0_CIB) ** betacib * planck(nu, tcib)


def monopole_and_dipole_CIB(nu, acib, betacib, tcib, beta_sun, dipole_direction, nside):
    npix = hp.nside2npix(nside)
    beta_dot_n = beta_sun * np.dot(dipole_direction, hp.pix2vec(nside, np.arange(npix)))
    gamma = 1 / np.sqrt(1 - beta_sun ** 2)
    boostedCIB = (
        CIB(nu * gamma * (1 - beta_dot_n), acib, betacib, tcib)
        / (gamma * (1 - beta_dot_n)) ** 3
    )
    return boostedCIB


def deltaI_y_distortions(nu, y, tcmb):
    x = h.value / k_B.value / tcmb * 1e9 * nu
    I0 = 2.0 * h.value / c.value / c.value * (k_B.value * tcmb / h.value)**3 * 1e20
    fac = x**4 * np.exp(x) / (np.exp(x) - 1)**2
    return I0 * fac * (x / np.tanh(0.5 * x) - 4) * y 


def deltaI_y_distortions_relativistic_corrections(nu, y, tesz, tcmb):
    #eq. 2.25 of Naoki Itoh, Yasuharu Kohyama, and Satoshi Nozawa
    #Relativistic Corrections to the Sunyaev-Zeldovich Effect for Clusters of Galaxies
    #https://iopscience.iop.org/article/10.1086/305876
    #not sure about 2.25
    #The main y term (Y0 = xtil - 4.0) in omitted here 
    x = h.value / k_B.value / tcmb * 1e9 * nu

    I0 = 2.0 * h.value / c.value / c.value * (k_B.value * tcmb / h.value)**3 * 1e20

    thetae = tesz / 510.998950 

    xtil = x / np.tanh(0.5 * x)
    stil = x / np.sinh(0.5 * x)

    Y1 = -10.0 + 23.5 * xtil - 8.4 * xtil * xtil + 0.7 * xtil * xtil * xtil + stil * stil * ( -4.2 + 1.4 * xtil)

    Y2 = (-7.5 + 127.875 * xtil - 173.6 * xtil * xtil + 65.8 * xtil * xtil * xtil 
        - 8.8 * xtil * xtil * xtil * xtil + 11.0 / 30.0 * xtil * xtil * xtil * xtil * xtil 
        + stil * stil * (-86.8 + 131.6 * xtil - 48.4 * xtil * xtil + 143.0 / 30.0 * xtil * xtil * xtil) 
        + stil * stil * stil * stil *( -8.8 + 187.0 / 60.0 * xtil ))

    Y3 = (7.5 + 313.125 * xtil - 1419.6 * xtil * xtil + 1425.3 * xtil * xtil * xtil 
        - 18594.0 / 35.0 * xtil * xtil * xtil * xtil + 12059.0 / 140.0 * xtil * xtil * xtil * xtil * xtil 
        - 128.0 / 21.0 * xtil * xtil * xtil * xtil * xtil * xtil + 16.0 / 105.0 * xtil * xtil * xtil * xtil * xtil * xtil * xtil
        + stil * stil * ( -709.8 + 2850.6 * xtil - 102267.0 / 35.0 * xtil * xtil + 156767.0 / 140.0 * xtil * xtil * xtil 
        - 1216.0 / 7.0 * xtil * xtil * xtil * xtil + 64.0 / 7.0 * xtil * xtil * xtil * xtil * xtil)  
        + stil * stil * stil * stil * ( - 18594.0 / 35.0 + 205003.0 / 280.0 * xtil - 1920.0 / 7.0 * xtil * xtil 
        + 1024.0 / 35.0 * xtil * xtil * xtil) + stil * stil * stil * stil * stil * stil * (- 544.0 / 21.0 + 992.0 / 105.0 * xtil))

    Y4 = (-4.21875 + 237.3046875 * xtil - 6239.1 * xtil * xtil + 15368.175 * xtil * xtil * xtil 
        - 12438.9 * xtil * xtil * xtil * xtil + 4446.2875 * xtil * xtil * xtil * xtil * xtil 
        - 16568.0 / 21.0 * xtil * xtil * xtil * xtil * xtil * xtil + 7516.0 / 105.0 * xtil * xtil * xtil * xtil * xtil * xtil * xtil
        - 22.0 / 7.0 * xtil * xtil * xtil * xtil * xtil * xtil * xtil * xtil
        + 11.0 / 210.0 * xtil * xtil * xtil * xtil * xtil * xtil * xtil * xtil * xtil
        + stil * stil * ( -3119.55 + 30736.35 * xtil - 68413.95 * xtil * xtil + 57801.7375 * xtil * xtil * xtil 
        - 157396.0 / 7.0 * xtil * xtil * xtil * xtil + 30064.0 / 7.0 * xtil * xtil * xtil * xtil * xtil
        - 2717.0 / 7.0 * xtil * xtil * xtil * xtil * xtil * xtil + 2761.0 / 210.0 * xtil * xtil * xtil * xtil * xtil * xtil * xtil)
        + stil * stil * stil * stil * ( - 12438.9 + 37793.44375 * xtil - 248520.0 / 7.0 * xtil * xtil 
        + 481024.0 / 35.0 * xtil * xtil * xtil - 15972.0 / 7.0 * xtil * xtil * xtil * xtil
        + 18689.0 / 140.0 * xtil * xtil * xtil * xtil * xtil) + stil * stil * stil * stil * stil * stil * (
        - 70414.0 / 21.0 + 465992.0 / 105.0 * xtil - 11792.0 / 7.0 * xtil * xtil + 19778.0 / 105.0 * xtil * xtil * xtil)
        + stil * stil * stil * stil * stil * stil * stil * stil * (- 682.0 / 7.0 + 7601.0 / 210.0 * xtil))

    delta_I_over_I =  x * x * x * x * np.exp(x) / ( np.exp(x) - 1.0 ) / ( np.exp(x) - 1.0 ) * (
        + Y1 * thetae + Y2 * thetae * thetae
        + Y3 * thetae * thetae * thetae 
        + Y4 * thetae * thetae * thetae * thetae) * y

    return I0 * delta_I_over_I


def deltaI_mu_distortions(nu, mu, tcmb):
    x = h.value / k_B.value / tcmb * 1e9 * nu
    I0 = 2.0 * h.value / c.value / c.value * (k_B.value * tcmb / h.value)**3 * 1e20
    fac = x**4 * np.exp(x) / (np.exp(x) - 1)**2
    return I0 * fac * (1 / 2.1923 - 1 / x) * mu


def get_sky(
    freqs,
    nside,
    models: Union[List, None]=["c1", "d2", "s1", "a1", "f1"],
    fwhm_deg=1.0,
    add_cmb_monopole_and_dipole=True,
    add_cib_monopole_and_dipole=True,
    y_distortions: Union[float, None]=1e-6,
    t_e_sz: Union[float, None]=1.24,
    mu_distortions: Union[float, None]=1e-8,
    maps_in_ecliptic=False,
):

    """

    Args:

    - ``freqs``: array of frequencies of dimension n_freq

    - ``nside``: healpix resolution of the maps generated

    - ``models``: foreground to use in pysm3 jargon

    - ``fwhm_deg``: Gaussian smoothing in deg (default: 1 degree)

    - ``add_cmb_monopole_and_dipole``: add CMB monopole and dipole

    - ``add_cib_monopole_and_dipole``: add CIB monopole and dipole

    - ``y_distortions``: add y-type distortions with amplitude y_distortions

    - ``t_e_sz``: electron temperature t_e_sz for relativistic corrections in keV

    - ``mu_distortions``: add mu-type distortions with amplitude mu_distortions

    - ``maps_in_ecliptic``: maps in eclipitc coordinates

    It returns a 2d array (n_freq,npix) containing the skies for each freqs

    """

    if models:
        sky = pysm3.Sky(nside=nside, preset_strings=models, output_unit="MJy/sr")

    npix = hp.nside2npix(nside)
    m = np.zeros((len(freqs), npix))

    if maps_in_ecliptic:
        r = hp.Rotator(coord=["G", "E"])

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
                A_CIB,
                SPECTRAL_INDEX_CIB,
                T_CIB_K,
                BETA_SUN,
                DIPOLE_DIRECTION,
                nside,
            )

        m[ifreq] = hp.smoothing(m[ifreq], fwhm=np.deg2rad(fwhm_deg))

        if maps_in_ecliptic:
            m[ifreq] = r.rotate_map_pixel(m[ifreq])

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
