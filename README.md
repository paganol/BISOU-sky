# BISOU sky
Generates sky for an array of frequencies using [pysm3](https://pysm3.readthedocs.io).

It includes:
* monopole, dipole, and higher order terms of CMB and CIB
* foreground emissions as provided by pysm3
* extragalactic CO signal
* y distortions plus relativistic corrections
* mu distortions

## Installation

```
git clone https://github.com/paganol/BISOU-sky.git
cd BISOU-sky
pip install -e .
```

## Usage

```python
import numpy as np
from bisou_sky import get_sky

nside = 128
models = ["c1", "d2", "s1", "a1", "f1"]
frequencies = np.arange(60, 2000, 15)

m = get_sky(frequencies,nside,models)
m.shape # (130, 196608)
```


## Parameters

    - ``freqs``: array of frequencies of dimension n_freq

    - ``models``: foreground to use in pysm3 jargon

    - ``nside``: healpix resolution of the maps generated

    - ``fwhm_deg``: Gaussian smoothing in deg (default: 1 degree)

    - ``add_cmb_monopole_and_dipole``: add CMB monopole and dipole

    - ``add_cib_monopole_and_dipole``: add CIB monopole and dipole

    - ``y_distortions``: add y-type distortions with amplitude y_distortions

    - ``t_e_sz``: electron temperature t_e_sz [in keV] for relativistic corrections 

    - ``mu_distortions``: add mu-type distortions with amplitude mu_distortions

    - ``A_eg_CO``: add extragalactic CO signal with amplitude A_eg_CO

    - ``maps_coord``: coordinates of the output maps. Default celestial coordinates 

    It returns a 2d array (n_freq,npix) containing the skies for each freqs

