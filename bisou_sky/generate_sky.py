import numpy as np
import healpy as hp
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


def get_sky(
    freqs,
    models,
    nside,
    fwhm_deg=1.0,
    add_cmb_monopole_and_dipole=True,
    add_cib_monopole_and_dipole=True,
    rotate_to_ecliptic=False,
):
    sky = pysm3.Sky(nside=nside, preset_strings=models, output_unit="MJy/sr")
    npix = hp.nside2npix(nside)
    m = np.zeros((len(freqs), npix))
    if rotate_to_ecliptic:
        r = hp.Rotator(coord=["G", "E"])
    for ifreq, freq in enumerate(freqs):
        m[ifreq] = sky.get_emission(freq * u.GHz)[0]
        m[ifreq] = hp.smoothing(m[ifreq], fwhm=np.deg2rad(fwhm_deg))
        if add_cmb_monopole_and_dipole:
            m[ifreq] += monopole_and_dipole_CMB(
                freq, T_CMB_K, BETA_SUN, DIPOLE_DIRECTION, nside
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
        if rotate_to_ecliptic:
            m[ifreq] = r.rotate_map_pixel(m[ifreq])
    return m
