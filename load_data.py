""" 
Author: Edoardo Altamura (October 2022)
Contact: edoardo.altamura@manchester.ac.uk

Tools to access data presented in the paper:
    Altamura et al. (2022)
    https://ui.adsabs.harvard.edu/abs/2022arXiv221009978A/abstract
    `EAGLE-like simulation models do not solve the entropy core problem in groups 
    and clusters of galaxies`
    
Includes two generic functions to retrieve data from hdf5 files and specific
classes to load data used to generate the figures.
"""

import h5py
import numpy as np
from itertools import product
from unyt import Solar_Mass, Gyr, dimensionless


def load_dict_from_hdf5(filename: str) -> dict:
    """
    Given the path to an HDF5 file, returns a dictionary with the same structure as
    the HDF5 file and the data loaded in memory.

    Args:
        filename (str): Path of the HDF5 file

    Returns:
        dict: Outputs the dictionary with the same structure as the HDF5 file
    """
    with h5py.File(filename, "r") as h5file:
        return _recursively_load_dict_contents_from_group(h5file, "/")


def _recursively_load_dict_contents_from_group(h5file: h5py.File, path: str) -> dict:
    """
    Auxiliary function to `load_dict_from_hdf5`, used for recursively load the 
    fields from the HDF5 file into the dictionary.

    Args:
        h5file (h5py.File): Data handle to the h5py.File
        path (str): Path of the HDF5 file

    Returns:
        dict: Partially loaded dictionary, updated recursively
    """
    ans = {}
    for key, item in h5file[path].items():
        if isinstance(item, h5py._hl.dataset.Dataset):
            ans[key] = item[...]
        elif isinstance(item, h5py._hl.group.Group):
            ans[key] = _recursively_load_dict_contents_from_group(
                h5file, path + key + "/"
            )
    return ans


class DataContainer(object):

    redshift_names = ["redshift_0", "redshift_1"]
    halo_names = [
        "VR18_+1res",
        "VR18_-8res",
        "VR2915_+1res",
        "VR2915_-8res",
    ]

    model_names = [
        "AGNdT8",
        "AGNdT9",
        "Bipolar",
        "Isotropic",
        "Random",
        "Ref",
        "alpha0",
        "noAGN",
        "noMetalCooling",
        "noSN",
    ]

    def __init__(self) -> None:
        pass

    def bind_dict_to_object(self, dictionary: dict):
        """
        Auxiliary function to parse the data structure from a dictionary to the attribute
        structure of the current class instance.
        Example:
            Instead of accessing a dataset as object.data['group_name']['dataset_name'],
            the same dataset will be accessible as object.data.group_name.dataset_name.
            The + sign is converted into _plus_ and - sign into _minus_ to protect the
            python operator bindings when using this mode.

        Args:
            dictionary (dict): The dictionary containing data to bind to the instance
                of the class
        """
        if isinstance(dictionary, list):
            dictionary = [self.bind_dict_to_object(x) for x in dictionary]
        if not isinstance(dictionary, dict):
            return dictionary

        class AuxiliaryClass(object):
            pass

        obj = AuxiliaryClass()
        for key in dictionary:
            k_name = key.lower() if key == "True" else key

            k_name = k_name.replace("+", "_plus_").replace("-", "_minus_")
            k_name = k_name.replace("__", "_")

            setattr(
                obj, k_name, self.bind_dict_to_object(dictionary[key]),
            )

        return obj


class RefModelExtendedSample(DataContainer):
    def __init__(
        self,
        filepath: str = "./ref_model_entended_sample.hdf5",
        bind_data_to_class: bool = True,
    ) -> None:

        data = load_dict_from_hdf5(filepath)

        for resolution_key in data.keys():
            data[resolution_key]["entropy_profile"] *= dimensionless
            data[resolution_key]["gas_fraction"] *= dimensionless
            data[resolution_key]["m500"] *= Solar_Mass
            data[resolution_key]["radial_bin_centres"] *= dimensionless
            data[resolution_key]["star_fraction"] *= dimensionless

        if bind_data_to_class:
            self.data = self.bind_dict_to_object(data)
        else:
            self.data = data


class PropertiesReducedSample(DataContainer):
    def __init__(
        self,
        filepath: str = "./properties_reduced_sample.hdf5",
        bind_data_to_class: bool = True,
    ) -> None:

        data = load_dict_from_hdf5(filepath)

        iterator_fields = product(
            self.redshift_names, self.halo_names, self.model_names
        )

        for redshift_name, halo_name, model_name in iterator_fields:

            # Skip incomplete datasets, e.g. VR18_+1res for alternative models
            try:
                _ = data[redshift_name][halo_name][model_name]
            except KeyError:
                continue

            nested_group = data[redshift_name][halo_name][model_name]
            for dataset_name in nested_group.keys():

                # Do not convert strings, e.g. filepaths
                if nested_group[dataset_name].dtype != float:
                    continue

                # Convert from np.ndarray(float) to simple float
                data[redshift_name][halo_name][model_name][dataset_name] = float(
                    nested_group[dataset_name],
                )

            data[redshift_name][halo_name][model_name]["entropy_core"] *= dimensionless
            data[redshift_name][halo_name][model_name]["fbary"] *= dimensionless
            data[redshift_name][halo_name][model_name]["fgas"] *= dimensionless
            data[redshift_name][halo_name][model_name]["fstar"] *= dimensionless
            data[redshift_name][halo_name][model_name]["m500"] *= Solar_Mass
            data[redshift_name][halo_name][model_name]["mbh"] *= Solar_Mass
            data[redshift_name][halo_name][model_name]["mgas"] *= Solar_Mass
            data[redshift_name][halo_name][model_name]["mstar_100kpc"] *= Solar_Mass
            data[redshift_name][halo_name][model_name]["ssfr_100kpc"] /= Gyr

        if bind_data_to_class:
            self.data = self.bind_dict_to_object(data)
        else:
            self.data = data


class ProfilesReducedSample(DataContainer):
    def __init__(
        self,
        filepath: str = "./profiles_reduced_sample.hdf5",
        bind_data_to_class: bool = True,
    ) -> None:

        data = load_dict_from_hdf5(filepath)

        for halo_model in data.keys():
            for dataset_name in data[halo_model]:
                data[halo_model][dataset_name] *= dimensionless

        # Split dictionary by object name
        for halo_name in self.halo_names:
            data[halo_name] = dict()

            for key in data.keys():
                preable_halo_name = f"{halo_name:s}_"

                if key.startswith(preable_halo_name):
                    data[halo_name][key.lstrip(preable_halo_name)] = data[key]

        if bind_data_to_class:
            self.data = self.bind_dict_to_object(data)
        else:
            self.data = data


if __name__ == "__main__":
    RefModelExtendedSample()
    PropertiesReducedSample()
    ProfilesReducedSample()
