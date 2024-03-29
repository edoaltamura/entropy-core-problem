![Entropy cores](https://img.shields.io/badge/ArXiv-2210.09978-red)
![Entropy cores](https://img.shields.io/badge/ADS-2022arXiv221009978A-important)
![MIT License](https://img.shields.io/github/license/edoaltamura/entropy-core-problem)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/3)
![DiRAC](https://img.shields.io/badge/DiRAC-COSMA7-green)
![Stargazers](https://img.shields.io/github/stars/edoaltamura/entropy-core-problem)
![Issues](https://img.shields.io/github/issues/edoaltamura/entropy-core-problem)

<p align="center">
     <picture>
       <source media="(prefers-color-scheme: light)" srcset="img/logo_light.jpg">
       <img alt="Logo" src="img/logo_dark.jpg" width="200"/>
     </picture>
</p>


# The entropy core problem
#### Data products from the study by [Altamura et al. (2022)](https://ui.adsabs.harvard.edu/abs/2022arXiv221009978A/abstract)

![Profiles](img/profiles_observations.png)

The data are supplied in `hdf5` format. They can be inspectes with `h5dump`, `HDFView` and can be read programmatically using the supplied file `load_data.py`.

`load_data.py` organises the data in categories through the following classes:
- `RefModelExtendedSample`
- `PropertiesReducedSample`
- `ProfilesReducedSample`

For each class, the data structure is the same as in the `hdf5` files and can be navigated using class attributes. The datasets already include the units from the `unyt` module.

This is an example code to plot the profile for the group (`VR2915`) at high resolution (`+1res` in the `hdf5` file or `plus_1res` in the class instance attributes), run with the reference EAGLE-like model (`Ref`).
```python
from load_data import ProfilesReducedSample

profiles = ProfilesReducedSample()
entropy_profile = profiles.data.VR2915_plus_1res.Ref.entropy_profile

# The entropy profile is in dimensionless units, normalised to the self-similar scaling $K_{500}$ 
print(entropy_profile)

# The radial distance of the shells used to bin the particles is scaled by $r_{500}$
radial_bin_centres = profiles.data.VR2915_plus_1res.Ref.radial_bin_centres

from matplotlib import pyplot as plt

# To display the dimensionless profile, use matplotlib
plt.plot(radial_bin_centres, entropy_profile)
plt.xlabel(r"$r/r_{500}$")
plt.ylabel(r"$K/K_{500}$")
plt.show()
```

## Data structure: `RefModelExtendedSample`
```text
RefModelExtendedSample()
|
|-- resolution_minus_8res
|    |-- entropy_profile
|    |-- gas_fraction
|    |-- m500
|    |-- radial_bin_centres
|    |-- star_fraction
|    |-- VR_numbers
|
|-- resolution_plus_1res
     |--  ...
```

## Data structure: `PropertiesReducedSample`
```text
PropertiesReducedSample()
|
|-- redshift_0
|    |-- VR18_minus_1res
|    |    |-- AGNdT8
|    |    |    |-- entropy_core
|    |    |    |-- fbary            {Baryon fraction == fgas + fstar}
|    |    |    |-- fgas
|    |    |    |-- fstar
|    |    |    |-- m500
|    |    |    |-- mbh              {Mass of the SMBH}
|    |    |    |-- mgas             {Mass of the hot gas inside r500}
|    |    |    |-- mstar_100kpc     {Stellar mass of the BCG}
|    |    |    |-- ssfr_100kpc      {1Gyr-averaged specific star-formation rate in the BCG}
|    |    |
|    |    |-- AGNdT9 [same fields]
|    |    |-- Bipolar [...]
|    |    |-- Isotropic [...]
|    |    |-- Random [...]
|    |    |-- Ref [...]
|    |    |-- alpha0 [...]
|    |    |-- noAGN [...]
|    |    |-- noMetalCooling [...]
|    |    |-- noSN [...]
|    |
|    |-- VR18_plus_8res
|    |    |--  ...
|    |
|    |-- VR2915_plus_1res
|    |    |--  ...
|    |
|    |-- VR2915_minus_8res
|         |--  ...
|
|
|-- redshift_1
     |--  ...
```

## Data structure: `ProfilesReducedSample`
```text
ProfilesReducedSample()
|
|-- VR18_minus_1res
|    |-- AGNdT8
|    |    |-- density_profile
|    |    |-- entropy_profile
|    |    |-- temperature_profile
|    |    |-- radial_bin_centers
|    |
|    |-- AGNdT9 [same fields]
|    |-- Bipolar [...]
|    |-- Isotropic [...]
|    |-- Random [...]
|    |-- Ref [...]
|    |-- alpha0 [...]
|    |-- noAGN [...]
|    |-- noMetalCooling [...]
|    |-- noSN [...]
|
|-- VR18_plus_8res
|    |--  ...
|
|-- VR2915_plus_1res
|    |--  ...
|
|-- VR2915_minus_8res
     |--  ...
```
## Raw simulation data for the simulated cluster
The raw data for the simulated cluster at $z=0$ run with the _Ref_ model is publicly accessible and can be downloaded from Zenodo at [this link](https://doi.org/10.5281/zenodo.8410619).

If you use these raw data files for your work, please consider citing the MNRAS paper, as well as the Zenodo dataset with this BibTeX handle:
```text
@dataset{edoardo_altamura_2023_8410619,
  author       = {Edoardo Altamura},
  title        = {{Simulated galaxy cluster data at $z=0$ demonstrating 
                   the entropy core problem with the SWIFT-EAGLE
                   galaxy formation model}},
  month        = oct,
  year         = 2023,
  note         = {{Main snapshot data: snap\_2640.hdf5 Main catalogue 
                   data: snap\_2640.properties}},
  publisher    = {Zenodo},
  version      = {1.0},
  doi          = {10.5281/zenodo.8410619},
  url          = {https://doi.org/10.5281/zenodo.8410619}
}
```
