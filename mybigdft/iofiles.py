r"""
File containing the Input and Posinp classes.
"""

from __future__ import print_function
import os
import warnings
from copy import deepcopy
from collections import Sequence, Mapping, MutableMapping
import yaml
import numpy as np


__all__ = ["BIGDFT_SOURCES", "inp_vars", "profiles", "check", "clean",
           "InputParams", "Logfile", "Posinp"]


# Read the definition of the input variables from the BigDFT sources
BIGDFT_SOURCES = os.environ["BIGDFT_SOURCES"]
inp_vars_file = os.path.join(BIGDFT_SOURCES,
                             "src/input_variables_definition.yaml")
with open(inp_vars_file, "r") as f:
    source = yaml.load_all(f)
    inp_vars = next(source)
    profiles = next(source)
# Add the posinp key (as it is not in input_variables_definition.yaml)
inp_vars["posinp"] = {"units": {"default": "atomic"},
                      "cell": {"default": []},
                      "positions": {"default": []},
                      "properties": {"default": {"format": "xyz",
                                                 "source": "posinp.xyz"}}}


def check(params):
    """
    Function checking that all the keys of params correspond to BigDFT
    parameters.

    :param params: Trial input parameters.
    :type params: dict
    :raises: KeyError
    """
    for key, value in params.items():
        if key not in inp_vars:
            raise KeyError("Unknown key '{}'".format(key))
        for subkey in value.keys():
            if subkey not in inp_vars[key].keys():
                raise KeyError("Unknown key '{}' in '{}'".format(subkey, key))


def clean(params):
    """
    Function returning the parameters whose values are not their default,
    after checking that all the keys in params correspond to actual
    BigDFT parameters.

    :param params: Trial input parameters.
    :type params: dict
    :returns: Input parameters whose values are not their default, after
        checking that all the keys in params correspond to actual BigDFT
        parameters.
    :rtype: dict
    """
    # Check the validity of the given input parameters
    check(params)
    # Return the cleaned input parameters
    real_params = deepcopy(params)
    for key, value in params.items():
        # Delete the child keys whose values are default
        for child_key, child_value in value.items():
            if child_value == inp_vars[key][child_key].get("default"):
                del real_params[key][child_key]
        # Delete the key if it is empty
        if real_params[key] == {}:
            del real_params[key]
    return real_params


class InputParams(MutableMapping):
    r"""
    Class allowing to initialize, read, write and interact with an input
    file of a BigDFT calculation.
    """

    def __init__(self, params=None):
        r"""
        An input file is created from a yaml dictionary.

        :param data: yaml dictionary of the input file.
        :type data: dict
        """
        if params is None:
            params = {}
        self._params = clean(params)

    @classmethod
    def from_file(cls, filename):
        r"""
        Initialize an InputParams instance from a BigDFT input file.

        :param filename: Name of the file to read.
        :type filename: str
        :returns: InputParams instance initialized from the file.
        :rtype: InputParams
        """
        with open(filename, "r") as f:
            params = yaml.safe_load(f)
        return cls(params=params)

    @classmethod
    def from_Logfile(cls, filename):
        """
        Initialize an InputParams instance from a BigDFT logfile.

        :param filename: Name of the file to read.
        :type filename: str
        :returns: InputParams instance initialized from the logfile.
        :rtype: InputParams
        """
        log = Logfile.from_file(filename)
        params = {key: log.log[key] for key in inp_vars}
        return cls(params=params)

    @classmethod
    def from_string(cls, string):
        r"""
        Initialize an InputParams instance from a string.

        :param string: InputParams parameters as a string.
        :type string: str
        :returns: InputParams instance initialized from the string.
        :rtype: InputParams
        """
        params = yaml.safe_load(string)
        return cls(params=params)

    @property
    def params(self):
        """
        :returns: Input parameters.
        :rtype: dict
        """
        return self._params

    def __getitem__(self, key):
        return self.params[key]

    def __setitem__(self, key, value):
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

        :param filename: Name of the input file.
        :type filename: str
        """
        with open(filename, "w") as f:
            self._params = clean(self.params)  # Make sure it is valid
            yaml.safe_dump(self.params, stream=f)


class Logfile(Mapping):
    r"""
    Class allowing to initialize, read, write and interact with an
    output file of a BigDFT calculation.
    """

    def __init__(self, log):
        r"""
        :param log: Logfile.
        :type log: dict
        """
        self._log = log
        self._posinp = self._extract_posinp()

    @classmethod
    def from_file(cls, filename):
        r"""
        Initialize the Logfile from a file on disk.

        :param filename: Name of the logfile.
        :type filename: str
        :returns: Logfile
        """
        with open(filename, "r") as f:
            log = yaml.safe_load(f)
        return cls(log)

    @property
    def log(self):
        r"""
        :returns: Logfile.
        :rtype: dict
        """
        return self._log

    def __getitem__(self, key):
        return self.log[key]

    def __iter__(self):
        return iter(self.log)

    def __len__(self):
        return len(self.log)

    def __repr__(self):
        return repr(self.log)

    def write(self, filename):
        r"""
        Write the logfile on disk.

        :param filename: Name of the logfile.
        :type filename: str
        """
        with open(filename, "w") as f:
            yaml.safe_dump(self.log, stream=f)

    @property
    def posinp(self):
        r"""
        :returns: Posinp used to during the calculation.
        :rtype: Posinp
        """
        return self._posinp

    def _extract_posinp(self):
        r"""
        Extract the posinp from the information contained in the
        logfile.

        :returns: Posinp used to during the calculation.
        :rtype: Posinp
        """
        # Initialize some variables from the logfile
        log_pos = self["posinp"]
        atoms = log_pos["positions"]
        n_at = len(atoms)
        units = log_pos["units"].lower()
        cell = log_pos["cell"]
        BC = self["Atomic System Properties"]["Boundary Conditions"].lower()
        if BC == "surface":
            cell = self["Atomic System Properties"]["Box Sizes (AU)"]
            if units not in ["reduced", "atomic", "bohr"]:
                raise NotImplementedError(
                    "Need to convert from atomic to {}".format(units))
        # Prepare the data in ordrer to initialize a Posinp instance
        posinp = [[n_at, units], [BC] + cell]
        for atom in atoms:
            [(atom_type, position)] = atom.items()
            posinp.append(Atom(atom_type, position))
        return Posinp(posinp)


class Posinp(Sequence):
    r"""
    Class allowing to initialize, read, write and interact with the
    input geometry of a BigDFT calculation (in the form of an xyz file).
    """

    def __init__(self, posinp):
        r"""
        The posinp is created from a list whose elements correspond to
        the various lines of an xyz file:
        * the first element of of the list is a list made of the number
          of atoms and the units (given by a string)
        * the second element is made of another list, made of the
          boundary conditions (given by a string) and possibly three
          distances defining the cell size along each space coordinate
          (x, y and z).
        * all the other elements must be Atom instances (defining the
          type (as a string) and the position of each atom).

        :param posinp: xyz file written as a list.
        :type posinp: list
        """
        # Set the attributes associated to the first line of the xyz
        # file, namely the number of atoms and the units used to define
        # the atomic positions
        first_line = posinp.pop(0)
        self._n_at = first_line[0]
        self._units = first_line[1].lower()
        # Set the attributes associated to the second line of the xyz
        # file, namely the boundary conditions and the size of the cell
        second_line = posinp.pop(0)
        self._BC = second_line[0].lower()
        if self.BC == "free" and self.units == "reduced":
            raise ValueError(
                "Reduced coordinates are not allowed with isolated BC.")
        if self.BC != "free":
            self._cell = [float(coord) for coord in second_line[1:4]]
        else:
            self._cell = None
        # Set the attributes associated to all the other lines of the
        # xyz file, namely the atoms
        assert all([isinstance(atom, Atom) for atom in posinp])
        if self.n_at != len(posinp):
            raise ValueError("The number of atoms do not correspond to the "
                             "number of positions.")
        self._atoms = posinp

    @classmethod
    def _from_stream(cls, stream):
        r"""
        Initialize the input positions from a stream that mimics an xyz
        file.

        :returns: Posinp read from the stream.
        :rtype: Posinp
        """
        for i, line in enumerate(stream):
            if i == 0:
                # Read the first line, containing the number of atoms
                # and the units of the coordinates of each atom
                posinp = []
                content = line.split()
                n_at = int(content[0])
                units = content[1]
                posinp.append([n_at, units])
            elif i == 1:
                # Read the second line,
                # containing the boundary conditions
                content = line.split()
                BC = content[:1]
                if content[0] != "free":
                    cell = [float(c) for c in content[1:4]]
                    BC += cell
                posinp.append(BC)
            else:
                # Read the atom (type and position)
                content = line.split()
                atom_type = content[0]
                position = [float(c) for c in content[1:4]]
                posinp.append(Atom(atom_type, position))
        return cls(posinp)

    @classmethod
    def from_string(cls, posinp):
        r"""
        :param posinp: Content of an xyz file.
        :type posinp: str
        :returns: Posinp read from the string.
        :rtype: Posinp
        """
        posinp = posinp.split("\n")
        return cls._from_stream(posinp)

    @classmethod
    def from_file(cls, filename):
        r"""
        :param filename: Name of the input positions file on disk.
        :type filename: str
        :returns: Posinp read from a file.
        :rtype: Posinp
        """
        with open(filename, "r") as f:
            return cls._from_stream(f)

    @classmethod
    def from_Logfile(cls, logname):
        r"""
        :param filename: Name of the logfile on disk.
        :type filename: str
        :returns: Posinp read from a logfile.
        :rtype: Posinp
        """
        return Logfile.from_file(logname).posinp

    @property
    def n_at(self):
        r"""
        :returns: Number of atoms.
        :rtype: int
        """
        return self._n_at

    @property
    def units(self):
        r"""
        :returns: Units used to represent the atomic positions.
        :rtype: str
        """
        return self._units

    @property
    def BC(self):
        r"""
        :returns: Boundary conditions.
        :rtype: str
        """
        return self._BC

    @property
    def cell(self):
        r"""
        :returns: Cell size.
        :rtype: list of three float or None
        """
        return self._cell

    @property
    def atoms(self):
        r"""
        :returns: Atoms of the system (atomic type and positions).
        :rtype: list
        """
        return self._atoms

    def __getitem__(self, index):
        return self.atoms[index]

    def __len__(self):
        return len(self.atoms)

    def __eq__(self, other):
        return str(self) == str(other)

    def __str__(self):
        r"""
        Convert the Posinp to a string.

        :returns: The Posinp instance as a string.
        :rtype: str
        """
        # Create the first two lines of the posinp file
        pos_str = "{}   {}\n".format(self.n_at, self.units)
        pos_str += self.BC
        if self.cell is not None:
            pos_str += "   {}   {}   {}\n".format(*self.cell)
        else:
            pos_str += "\n"
        # Add all the other lines, representing the atoms
        pos_str += "".join([str(atom) for atom in self])
        return pos_str

    def write(self, filename):
        r"""
        Write the Posinp on disk.

        :param filename: Name of the input positions file.
        :type filename: str
        """
        with open(filename, "w") as f:
            f.write(str(self))

    def translate_atom(self, i_at, vector):
        r"""
        :param i_at: Index of the atom.
        :type i_at: int
        :param vector: Translation vector to apply.
        :type vector: list or numpy.array of length 3
        :returns: A new posinp where the i_at atom was translated by
            vector.
        :rtype: Posinp

        .. Warning::

            You have to make sure that the units of the vector match
            those used by the posinp.
        """
        new_posinp = deepcopy(self)
        new_posinp.atoms[i_at] = self[i_at].translate(vector)
        return new_posinp


class Atom(object):
    r"""
    Class allowing to represent an atom by its type and position.
    """

    def __init__(self, atom_type, position):
        r"""
        :param atom_type: Type of the atom.
        :type atom_type: str
        :param position: Position of the atom.
        :type position: list or numpy.array of length 3
        """
        # TODO: Check that the atom type exists
        assert len(position) == 3
        self._atom_type = atom_type
        self._position = np.array(position)

    @property
    def atom_type(self):
        r"""
        :returns: Type of the atom.
        :rtype: str
        """
        return self._atom_type

    @property
    def position(self):
        r"""
        :returns: Position of the atom.
        :rtype: list or numpy.array of length 3
        """
        return self._position

    def translate(self, vector):
        r"""
        :returns: Atom translated according to the given vector.
        :rtype: Atom
        :param vector: Translation vector to apply.
        :type vector: list or numpy.array of length 3
        """
        assert len(vector) == 3
        new_atom = deepcopy(self)
        new_atom._position = self.position + np.array(vector)
        return new_atom

    def __str__(self):
        r"""
        :returns: String representation of the atom, mainly used to
            create the string representation of a Posinp instance.
        :rtype: str
        """
        return "{t}  {: .15}  {: .15}  {: .15}\n"\
               .format(t=self.atom_type, *self.position)

    def __repr__(self):
        return "Atom('{}', {})".format(self.atom_type, list(self.position))

    def __eq__(self, other):
        return (np.array_equal(self.position, other.position)
                and self.atom_type == other.atom_type)
