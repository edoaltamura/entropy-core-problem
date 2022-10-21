# Entropy core problem
## Data products from the study by [Altamura et al. (2022)](https://ui.adsabs.harvard.edu/abs/2022arXiv221009978A/abstract)
The data are supplied in `hdf5` format. They can be inspectes with `h5dump`, `HDFView` and can be read programmatically using the supplied file `load_data.py`. The `load_data.py` organises the data in categories through the following classes:
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
