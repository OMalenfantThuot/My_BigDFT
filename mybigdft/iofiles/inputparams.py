r"""
File containing the InputParams, Posinp, Logfile and Atom classes.
"""

from __future__ import print_function
import warnings
from copy import deepcopy
from collections import MutableMapping
import oyaml as yaml
try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:  # pragma: no cover
    from yaml import Loader, Dumper
from mybigdft.globals import INPUT_VARIABLES
from .posinp import Posinp


__all__ = ["check", "clean", "InputParams"]


def check(params):
    """
    Check that the keys of `params` correspond to BigDFT parameters.

    Parameters
    ----------
    params : dict
        Trial input parameters.

    Raises
    ------
    KeyError
        If a key or a sub-key is not a BigDFT parameter.
    """
    for key, value in params.items():
        if key not in INPUT_VARIABLES:
            raise KeyError("Unknown key '{}'".format(key))
        if value is not None:
            for subkey in value.keys():
                if subkey not in INPUT_VARIABLES[key].keys():
                    raise KeyError(
                        "Unknown key '{}' in '{}'".format(subkey, key))


def clean(params):
    """
    Parameters
    ----------
    params : dict
        Trial BigDFT input parameters.

    Returns
    -------
    dict
        Input parameters whose values are not their default, after
        checking that all the keys in `params` correspond to actual
        BigDFT parameters.
    """
    # Check the validity of the given input parameters
    check(params)
    # Return the cleaned input parameters
    real_params = deepcopy(params)
    for key, value in params.items():
        # The key might be empty (e.g.: logfile with many documents)
        if value is None:
            del real_params[key]
            continue
        # Delete the child keys whose values are default
        for child_key, child_value in value.items():
            default_value = INPUT_VARIABLES[key][child_key].get("default")
            if child_value == default_value:
                del real_params[key][child_key]
        # Delete the key if it is empty
        if real_params[key] == {}:
            del real_params[key]
    # Remove the cumbersome geopt key if ncount_cluster_x is the only
    # key (it happens when the input parameters are read from a Logfile)
    dummy_value = {'ncount_cluster_x': 1}
    if "geopt" in real_params and real_params["geopt"] == dummy_value:
        del real_params["geopt"]
    return real_params


class InputParams(MutableMapping):
    r"""
    Class allowing to initialize, read, write and interact with the
    input parameters of a BigDFT calculation.
    """

    def __init__(self, params=None):
        r"""
        Input parameters are initialized from a yaml dictionary.

        Parameters
        ----------
        data : dict
            yaml dictionary of the input parameters.


        >>> InputParams({'dft': {'hgrids': [0.35]*3}})
        {'dft': {'hgrids': [0.35, 0.35, 0.35]}}

        Default values are cleaned from the input parameters:

        >>> InputParams({'dft': {'hgrids': [0.45]*3}})
        {}

        The input parameters can be empty:

        >>> InputParams()
        {}

        Initializing with unknown parameters raises a ``KeyError``:

        >>> InputParams({'dfpt': {'hgrids': [0.35]*3}})
        Traceback (most recent call last):
        ...
        KeyError: "Unknown key 'dfpt'"
        >>> InputParams({'dft': {'hgrid': [0.35]*3}})
        Traceback (most recent call last):
        ...
        KeyError: "Unknown key 'hgrid' in 'dft'"
        """
        if params is None:
            params = {}
        if "posinp" in params:
            self.posinp = Posinp.from_dict(params.pop("posinp"))
        else:
            self.posinp = None
        self._params = clean(params)

    @classmethod
    def from_file(cls, filename):
        r"""
        Initialize an InputParams instance from a BigDFT input file.

        Parameters
        ----------
        filename : str
            Name of the file to read.

        Returns
        -------
        InputParams
            InputParams instance initialized from the file.
        """
        with open(filename, "r") as stream:
            return cls.from_string(stream)

    @classmethod
    def from_string(cls, string):
        r"""
        Initialize an InputParams instance from a string.

        Parameters
        ----------
        string : str
            Input parameters dictionary as a string.

        Returns
        -------
        InputParams
            InputParams instance initialized from the string.


        >>> InputParams.from_string("{'dft': {'rmult': [6, 8]}}")
        {'dft': {'rmult': [6, 8]}}
        """
        params = yaml.load(string, Loader=Loader)
        return cls(params=params)

    @property
    def params(self):
        """
        Returns
        -------
        dict
            Input parameters.
        """
        return self._params

    @property
    def posinp(self):
        r"""
        Returns
        -------
        Posinp or None
            Initial positions contained in the input parameters
        """
        return self._posinp

    @posinp.setter
    def posinp(self, new):
        if new is not None and not isinstance(new, Posinp):
            raise ValueError(
                "Update the posinp attribute with a Posinp instance")
        else:
            self._posinp = new

    def __getitem__(self, key):
        return self.params[key]

    def __setitem__(self, key, value):
        r"""
        Set the input parameters after making sure that they are valid.

        Parameters
        ----------
        key : str
            Input parameters key.
        value : dict
            Value of the given key of input parameters.

        Warns
        -----
        UserWarning
            If the proposed update does not modify the input parameters.
        """
        if key == "posinp":
            # Set the new posinp
            self.posinp = Posinp.from_dict(value)
        else:
            # Check that the key and its value are valid.
            params = {key: value}
            cleaned_params = clean(params)
            # Set the input parameters with cleaned parameters
            if cleaned_params == {}:
                try:
                    # Update with default params
                    del self.params[key]
                except KeyError:
                    warnings.warn("Nothing to update.", UserWarning)
            else:
                # Update with cleaned params
                self.params[key] = cleaned_params[key]

    def __delitem__(self, key):
        del self.params[key]

    def __iter__(self):
        return iter(self.params)

    def __len__(self):
        return len(self.params)

    def __repr__(self):
        return repr(self.params)

    def write(self, filename):
        """
        Write the input parameters on disk.

        Parameters
        ----------
        filename : str
            Name of the input file.
        """
        with open(filename, "w") as stream:
            self._params = clean(self.params)  # Make sure it is valid
            yaml.dump(self.params, stream=stream, Dumper=Dumper)
