# BISOU sky
Generates sky for an array of frequencies using [pysm3](https://pysm3.readthedocs.io).

It includes monopole, dipole, and higher order terms of CMB and CIB


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

m = get_sky(frequencies,models,nside)
m.shape # (130, 196608)
```
