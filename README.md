# BISOU sky
Generates sky for an array of frequencies using [pysm3](https://pysm3.readthedocs.io).

It includes:
* monopole, dipole, and higher order terms of CMB and CIB
* foreground emissions as provided by pysm3
* zodiacal light emission (via ZodiPy)
* extragalactic CO signal
* y distortions plus relativistic corrections
* mu distortions

## Installation


```

git clone [https://github.com/paganol/BISOU-sky.git](https://github.com/paganol/BISOU-sky.git)
cd BISOU-sky
pip install -e .
pip install zodipy  # Required for Zodiacal light emission

```

## Usage

```python
import numpy as np
from bisou_sky import get_sky

nside = 128
models = ["c1", "d2", "s1", "a1", "f1"]
frequencies = np.arange(60, 2000, 15)

# Standard usage
m = get_sky(frequencies, nside, models)

# Usage with Zodiacal emission
m_zodi = get_sky(
    frequencies, 
    nside, 
    models, 
    add_zodi=True, 
    zodi_time="2040-01-01"
)

m.shape # (130, 196608)

```

## Parameters

```
- ``freqs``: array of frequencies of dimension n_freq

- ``models``: foreground to use in pysm3 jargon

- ``nside``: healpix resolution of the maps generated

- ``fwhm_deg``: Gaussian smoothing in deg (default: 1 degree)

- ``add_cmb_monopole_and_dipole``: add CMB monopole and dipole

- ``add_cib_monopole_and_dipole``: add CIB monopole and dipole

- ``cib_model``: CIB model name ('fixsen', 'abitbol', 'gispert')

- ``A_cib``: CIB amplitude override (default: None, uses model default)

- ``cib_params``: Dictionary with 'beta' and 'T' to override CIB model defaults

- ``y_distortions``: add y-type distortions with amplitude y_distortions

- ``t_e_sz``: electron temperature t_e_sz [in keV] for relativistic corrections 

- ``mu_distortions``: add mu-type distortions with amplitude mu_distortions

- ``A_eg_CO``: add extragalactic CO signal with amplitude A_eg_CO

- ``add_zodi``: add zodiacal light emission using ZodiPy (default: False)

- ``zodi_model_name``: ZodiPy model name (e.g., 'dirbe', 'planck18'). Default 'dirbe'

- ``zodi_time``: Date of observation for Zodi calculation (default: "2040-01-01")

- ``zodi_observer``: Observer location for Zodi (e.g., 'semb-l2', 'earth'). Default is 'semb-l2'.

- ``maps_coord``: coordinates of the output maps. Default celestial coordinates 

It returns a 2d array (n_freq,npix) containing the skies for each freqs
