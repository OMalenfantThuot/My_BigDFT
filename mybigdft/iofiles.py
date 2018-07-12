r"""
File containing the Input and Posinp classes.
"""

from __future__ import print_function
import warnings
from copy import deepcopy
from collections import Sequence, Mapping, MutableMapping
import yaml
import numpy as np
from .globals import inp_vars


__all__ = ["check", "clean", "InputParams", "Logfile", "Posinp", "Atom"]


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


PATHS = "paths"
PRINT = "print"
GLOBAL = "global"
FLOAT_SCALAR = "scalar"
ATTRIBUTES = {
    "n_at": {PATHS: [["Atomic System Properties", "Number of atoms"]],
             PRINT: "Number of Atoms", GLOBAL: True},
    "boundary_conditions": {PATHS: [["Atomic System Properties",
                                     "Boundary Conditions"]],
                            PRINT: "Boundary Conditions", GLOBAL: True},
    "cell": {PATHS: [["Atomic System Properties", "Box Sizes (AU)"]],
             PRINT: "Cell size", GLOBAL: True},
    "energy": {PATHS: [["Last Iteration", "FKS"], ["Last Iteration", "EKS"],
                       ["Energy (Hartree)"]],
               PRINT: "Energy", GLOBAL: False},
    "fermi_level": {PATHS: [["Ground State Optimization", -1,
                             "Fermi Energy"],
                            ["Ground State Optimization", -1,
                             "Hamiltonian Optimization", -1,
                             "Subspace Optimization", "Fermi Energy"]],
                    PRINT: True, GLOBAL: False},
    "astruct": {PATHS: [["Atomic structure"]]},
    "evals": {PATHS: [["Complete list of energy eigenvalues"],
                      ["Ground State Optimization", -1, "Orbitals"],
                      ["Ground State Optimization", -1,
                       "Hamiltonian Optimization", -1,
                       "Subspace Optimization", "Orbitals"]]},
    "kpts": {PATHS: [["K points"]], PRINT: False, GLOBAL: True},
    "gnrm_cv": {PATHS: [["dft", "gnrm_cv"]],
                PRINT: "Convergence criterion on Wfn. Residue",
                GLOBAL: True},
    "kpt_mesh": {PATHS: [["kpt", "ngkpt"]], PRINT: True, GLOBAL: True},
    "forcemax": {PATHS: [["Geometry", "FORCES norm(Ha/Bohr)", "maxval"],
                         ["Clean forces norm (Ha/Bohr)", "maxval"]],
                 PRINT: "Max val of Forces"},
    "pressure": {PATHS: [["Pressure", "GPa"]], PRINT: True},
    "dipole": {PATHS: [["Electric Dipole Moment (AU)", "P vector"]]},
    "forces": {PATHS: [["Atomic Forces (Ha/Bohr)"]]},
    "forcemax_cv": {PATHS: [["geopt", "forcemax"]],
                    PRINT: "Convergence criterion on forces",
                    GLOBAL: True, FLOAT_SCALAR: True},
    "force_fluct": {PATHS: [["Geometry", "FORCES norm(Ha/Bohr)", "fluct"]],
                    PRINT: "Threshold fluctuation of Forces"},
    "magnetization": {PATHS: [["Ground State Optimization", -1,
                               "Total magnetization"],
                              ["Ground State Optimization", -1,
                               "Hamiltonian Optimization", -1,
                               "Subspace Optimization",
                               "Total magnetization"]],
                      PRINT: "Total magnetization of the system"},
    "support_functions": {PATHS: [["Gross support functions moments",
                                   "Multipole coefficients", "values"]]},
    "electrostatic_multipoles": {PATHS: [["Multipole coefficients",
                                          "values"]]},
    "sdos": {PATHS: [["SDos files"]], GLOBAL: True},
    "symmetry": {PATHS: [["Atomic System Properties", "Space group"]],
                 PRINT: "Symmetry group", GLOBAL: True},
    "walltime": {PATHS: [["Walltime since initialization"]],
                 PRINT: "Walltime since initialization"}
}


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
        self._set_builtin_attributes()
        self._clean_attributes()
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

    def _set_builtin_attributes(self):
        r"""
        Set all the base attributes of a BigDFT Logfile.

        They are defined by the ATTRIBUTES dictionary, whose keys are
        the base name of each attribute, the values being the
        description of the attribute as another dictionary.

        Once retrieved from the logfile, the attributes are set under
        their base name preceded by an underscore (e.g., the number of
        atoms read thanks to the 'n_at' key of  ATTRIBUTES is finally
        stored as the attribute '_n_at' of the Logfile instance).
        This extra underscore is meant to prevent the user from updating
        the value of the attribute.
        """
        for name, description in ATTRIBUTES.items():
            # Loop over the various paths (or logfile levels) where the
            # value might be stored.
            for path in description[PATHS]:
                # Loop over the different levels of the logfile to
                # retrieve the value
                value = self  # Always start from the bare logfile
                for key in path:
                    try:
                        value = value.get(key)  # value can be a dict
                    except AttributeError:
                        try:
                            value = value[key]  # value can be a list
                        except TypeError:
                            # This path leads to a dead-end: set a
                            # default value before moving to the next
                            # possible path.
                            value = None
                            continue
                # No need to look for other paths if a value is found
                if value is not None:
                    break
            setattr(self, "_"+name, value)

    def _clean_attributes(self):
        r"""
        Clean the value of the built-in attributes.
        """
        self._boundary_conditions = self._boundary_conditions.lower()

    def __getattr__(self, name):
        r"""
        Make the base attributes look for their private counterpart
        (whose name has an initial underscore) that actually stores the
        value of the attribute.

        All other attributes behave as they should by default.
        """
        if name in ATTRIBUTES:
            return getattr(self, "_"+name)
        else:
            super(Logfile, self).__setattr__(name)

    def __setattr__(self, name, value):
        r"""
        Make the base attributes behave like properties while keeping
        the default behaviour for the other ones.
        """
        if name in ATTRIBUTES:
            raise AttributeError("can't set attribute")
        else:
            super(Logfile, self).__setattr__(name, value)

    def __dir__(self):  # pragma: no cover
        r"""
        The base attributes are not found when doing dir() on a
        Logfile instance, but their counterpart with a preceding
        underscore is. What is done here is a removal of the
        underscored names, replaced by the bare names (in order
        to avoid name repetition).

        The bare attributes still behave as properties, while their
        value might be updated via the underscored attribute.
        """
        hidden_attributes = list(ATTRIBUTES.keys())
        base_dir = super(Logfile, self).__dir__()
        for name in hidden_attributes:
            base_dir.remove("_"+name)
        return base_dir + hidden_attributes

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
        BC = self.boundary_conditions
        if BC not in ["free", "surface"]:
            cell = log_pos["cell"]
        else:
            cell = []
        if BC == "surface":
            if units not in ["reduced", "atomic", "bohr"]:
                raise NotImplementedError(
                    "Need to convert cell size from atomic to {}"
                    .format(units))
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
