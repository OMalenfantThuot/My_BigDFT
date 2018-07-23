r"""
File containing the InputParams, Posinp, Logfile and Atom classes.
"""

from __future__ import print_function
import warnings
from copy import deepcopy
from collections import Sequence, Mapping, MutableMapping
import oyaml as yaml
try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper
import numpy as np
from .globals import input_variables


__all__ = ["check", "clean", "InputParams", "Logfile", "Posinp", "Atom"]


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
        if key not in input_variables:
            raise KeyError("Unknown key '{}'".format(key))
        for subkey in value.keys():
            if subkey not in input_variables[key].keys():
                raise KeyError("Unknown key '{}' in '{}'".format(subkey, key))


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
        # Delete the child keys whose values are default
        for child_key, child_value in value.items():
            default_value = input_variables[key][child_key].get("default")
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
            self._posinp = Posinp.from_dict(params.pop("posinp"))
        else:
            self._posinp = None
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
        with open(filename, "r") as f:
            return cls.from_string(f)

    @classmethod
    def from_Logfile(cls, logfile):
        """
        Initialize an InputParams instance from a BigDFT
        :class:`Logfile`.

        Parameters
        ----------
        logfile : Logfile
            Logfile of a BigDFT calculation.

        Returns
        -------
        InputParams
            InputParams instance initialized from the logfile.


        >>> inp = InputParams({
        ...     'posinp': {'units': 'angstroem',
        ...     'positions':
        ...         [{'N': [2.9763078243490115e-23, 6.872205952043537e-23,
        ...                 0.01071619987487793]},
        ...          {'N': [-1.1043449194501671e-23, -4.873421744830746e-23,
        ...                1.104273796081543]}],
        ...     'properties': {'format': 'xyz', 'source': 'N2.xyz'}}
        ... })
        >>> log = Logfile.from_file("tests/log.yaml")
        >>> inp == InputParams.from_Logfile(log)
        True
        """
        params = {key: logfile[key] for key in input_variables}
        return cls(params=params)

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
        params = yaml.load(string, Loader=CLoader)
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
        with open(filename, "w") as f:
            self._params = clean(self.params)  # Make sure it is valid
            yaml.dump(self.params, stream=f, Dumper=CDumper)


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
    "atom_types": {PATHS: [["Atomic System Properties", "Types of atoms"]],
                   PRINT: "List of the atomic types present in the posinp"},
    "walltime": {PATHS: [["Walltime since initialization"]],
                 PRINT: "Walltime since initialization"},
    "WARNINGS": {PATHS: [["WARNINGS"]],
                 PRINT: "Warnings raised during the BigDFT run"},
}


class Logfile(Mapping):
    r"""
    Class allowing to initialize, read, write and interact with an
    output file of a BigDFT calculation.
    """

    def __init__(self, log):
        r"""
        Parameters
        ----------
        log : dict
            Output of the BigDFT code as a yaml dictionary.
        """
        self._log = log
        self._set_builtin_attributes()
        self._clean_attributes()
        self._posinp = Posinp.from_dict(log['posinp'])
        self._check_warnings()

    def _set_builtin_attributes(self):
        r"""
        Set all the base attributes of a BigDFT Logfile.

        They are defined by the ATTRIBUTES dictionary, whose keys are
        the base name of each attribute, the values being the
        description of the attribute as another dictionary.

        Once retrieved from the logfile, the attributes are set under
        their base name preceded by an underscore (e.g., the number of
        atoms read thanks to the `n_at` key of ATTRIBUTES is finally
        stored as the attribute `_n_at` of the Logfile instance).
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
        # Make the forces as a numpy array of shape (n_at, 3)
        if self.forces is not None:
            new_forces = np.array([])
            for force in self.forces:
                new_forces = np.append(new_forces, list(force.values())[0])
            new_forces = new_forces.reshape((self.n_at, 3))
            self._forces = new_forces

    def _check_warnings(self):
        r"""
        Warns
        -----
        UserWarning
            If there are some warnings in the Logfile or if the XC of
            the pseudo-potentials do not match those of the input
            parameters.
        """
        if self.WARNINGS is not None:
            for warning in self.WARNINGS:
                warnings.warn(warning, UserWarning)
        self._check_psppar()

    def _check_psppar(self):
        r"""
        Warns
        -----
        UserWarning
            If the XC of the potential is different from the XC of the
            input parameters.
        """
        for atom_type in self.atom_types:
            psp_ixc = self["psppar.{}".format(atom_type)]["Pseudopotential XC"]
            inp_ixc = self["dft"]["ixc"]
            if psp_ixc != inp_ixc:
                warnings.warn("The XC of pseudo potentials ({}) is different "
                              "from the input XC ({}) for the '{}' atoms"
                              .format(psp_ixc, inp_ixc, atom_type),
                              UserWarning)

    @classmethod
    def from_file(cls, filename):
        r"""
        Initialize the Logfile from a file on disk.

        Parameters
        ----------
        filename : str
            Name of the logfile.

        Returns
        -------
        Logfile
            Logfile initialized from a file on disk.


        >>> log = Logfile.from_file("tests/log.yaml")
        >>> print(log.posinp)
        2   angstroem
        free
        N   2.97630782434901e-23   6.87220595204354e-23   0.0107161998748779
        N  -1.10434491945017e-23  -4.87342174483075e-23   1.10427379608154
        <BLANKLINE>
        >>> log.energy
        -19.884659235401838
        """
        with open(filename, "r") as f:
            return cls.from_stream(f)

    @classmethod
    def from_stream(cls, stream):
        r"""
        Initialize the Logfile from a stream.

        Parameters
        ----------
        stream
            Logfile as a stream.

        Returns
        -------
        Logfile
            Logfile initialized from a stream.
        """
        log = yaml.load(stream, Loader=CLoader)
        return cls(log)

    @property
    def log(self):
        r"""
        Returns
        -------
        Logfile
            Yaml dictionary of the output of the BigDFT code.
        """
        return self._log

    def __getattr__(self, name):
        r"""
        Make the base attributes look for their private counterpart
        (whose name has an initial underscore) that actually stores the
        value of the attribute.

        All other attributes behave as they should by default.

        Parameters
        ----------
        name : str
            Name of the attribute.

        Returns
        -------
            Value of the required attribute.
        """
        if name in ATTRIBUTES:
            return getattr(self, "_"+name)
        else:
            super(Logfile, self).__getattr__(name)

    def __setattr__(self, name, value):
        r"""
        Make the base attributes behave like properties while keeping
        the default behaviour for the other ones.

        Parameters
        ----------
        name : str
            Name of the attribute.
        value
            Value of the attribute.

        Raises
        ------
        AttributeError
            If one tries to update one of the base attribute.
        """
        if name in ATTRIBUTES:
            raise AttributeError("can't set attribute")
        else:
            super(Logfile, self).__setattr__(name, value)

    def __dir__(self):
        r"""
        The base attributes are not found when doing `dir()` on a
        `Logfile` instance, but their counterpart with a preceding
        underscore is. What is done here is a removal of the underscored
        names, replaced by the bare names (in order to avoid name
        repetition).

        The bare attributes still behave as properties, while their
        value might be updated via the underscored attribute.
        """
        hidden_attributes = list(ATTRIBUTES.keys())
        try:  # pragma: no cover
            base_dir = super(Logfile, self).__dir__()  # Python3
        except AttributeError:  # pragma: no cover
            base_dir = dir(super(Logfile, self))  # Python2
            # Add the missing stuff
            base_dir += ["write", "log", "from_file", "from_stream",
                         "posinp", "values", "keys", "get", "items",
                         "_check_psppar", "_check_warnings",
                         "_clean_attributes", "_set_builtin_attributes"]
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

        Parameters
        ----------
        filename : str
            Name of the logfile.
        """
        with open(filename, "w") as f:
            yaml.dump(self.log, stream=f, Dumper=CDumper)

    @property
    def posinp(self):
        r"""
        Returns
        -------
        Posinp
            Posinp used during the calculation.
        """
        return self._posinp


class Posinp(Sequence):
    r"""
    Class allowing to initialize, read, write and interact with the
    input geometry of a BigDFT calculation in the form of an xyz file.

    Such a file is made of a few lines, containing all the necessary
    information to specify a given system of interest:

    * the first line contains the number of atoms :math:`n_{at}` and the
      units for the coordinates (and possibly the cell size),
    * the second line contains the boundary conditions used and possibly
      the simulation cell size (for periodic or surface boundary
      conditions),
    * the subsequent :math:`n_{at}` lines are used to define each atom
      of the system: first its type, then its position given by three
      coordinates (for :math:`x`, :math:`y` and :math:`z`).
    """

    def __init__(self, atoms, units, BC, cell=None):
        r"""
        Parameters
        ----------
        atoms : list
            List of :class:`Atom` instances.
        units : str
            Units of the coordinate system.
        BC : str
            Boundary conditions.
        cell : Sequence of length 3 or None
            Size of the simulation domain in the three space
            coordinates.


        >>> posinp = Posinp([Atom('N', [0, 0, 0]), Atom('N', [0, 0, 1.1])],
        ...                 'angstroem', 'free')
        >>> len(posinp)
        2
        >>> posinp.BC
        'free'
        >>> posinp.units
        'angstroem'
        >>> for atom in posinp:
        ...     repr(atom)
        "Atom('N', [0.0, 0.0, 0.0])"
        "Atom('N', [0.0, 0.0, 1.1])"
        """
        # Check initial values
        BC = BC.lower()
        units = units.lower()
        if cell is None:
            assert BC == "free"
        else:
            assert len(cell) == 3
            cell = [abs(float(coord)) if coord != ".inf" else coord
                    for coord in cell]
        if BC == "periodic":
            assert ".inf" not in cell
        elif BC == "free":
            assert units != "reduced"
        # Set the attributes
        self._atoms = atoms
        self._units = units
        self._cell = cell
        self._BC = BC

    @classmethod
    def _from_stream(cls, stream):
        r"""
        Initialize the input positions from a stream that mimics an xyz
        file.

        Returns
        -------
        Posinp
            Posinp read from a stream.
        """
        atoms = []
        for i, line in enumerate(stream):
            if i == 0:
                # Read the first line, containing the number of atoms
                # and the units of the coordinates of each atom
                content = line.split()
                n_at = int(content[0])
                units = content[1]
            elif i == 1:
                # Read the second line, containing the boundary
                # conditions and possibly the cell size.
                content = line.split()
                BC = content[0].lower()
                if BC != "free":
                    cell = content[1:4]
                else:
                    cell = None
            else:
                # Read the atom (type and position)
                content = line.split()
                atom_type = content[0]
                position = content[1:4]
                atoms.append(Atom(atom_type, position))
        assert n_at == len(atoms)
        return cls(atoms, units, BC, cell=cell)

    @classmethod
    def from_string(cls, posinp):
        r"""
        Initialize the input positions from a string.

        Parameters
        ----------
        posinp : str
            Content of an xyz file as a string.

        Returns
        -------
        Posinp
            Posinp read from the string.
        """
        posinp = posinp.split("\n")
        return cls._from_stream(posinp)

    @classmethod
    def from_file(cls, filename):
        r"""
        Initialize the input positions from a file on disk.

        Parameters
        ----------
        filename : str
            Name of the input positions file on disk.

        Returns
        -------
        Posinp
            Posinp read from a file on disk.


        >>> posinp = Posinp.from_file("tests/surface.xyz")
        >>> posinp.cell
        [8.07007483423, '.inf', 4.65925987792]
        >>> print(posinp)
        4   reduced
        surface   8.07007483423   .inf   4.65925987792
        C   0.08333333333   0.5   0.25
        C   0.41666666666   0.5   0.25
        C   0.58333333333   0.5   0.75
        C   0.91666666666   0.5   0.75
        <BLANKLINE>
        """
        with open(filename, "r") as f:
            return cls._from_stream(f)

    @classmethod
    def from_dict(cls, posinp):
        r"""
        Initialize the input positions from a dictionary.

        Parameters
        ----------
        posinp : dict
            Posinp as a dictionary coming from an InputParams or
            Logfile instance.

        Returns
        -------
        Posinp
            Posinp initialized from an dictionary.
        """
        # Read data from the dictionary
        atoms = []  # atomic positions
        for atom in posinp["positions"]:
            [(atom_type, position)] = atom.items()
            atoms.append(Atom(atom_type, position))
        units = posinp["units"]  # Units of the coordinates
        cell = posinp.get("cell")  # Simulation cell size
        # Infer the boundary conditions from the value of cell
        if cell is None:
            BC = "free"
        else:
            if cell[1] == ".inf":
                BC = "surface"
            else:
                BC = "periodic"
        return cls(atoms, units, BC, cell=cell)

    @property
    def units(self):
        r"""
        Returns
        -------
        str
            Units used to represent the atomic positions.
        """
        return self._units

    @property
    def BC(self):
        r"""
        Returns
        -------
        str
            Boundary conditions.
        """
        return self._BC

    @property
    def cell(self):
        r"""
        Returns
        -------
        list of three float or None
            Cell size.
        """
        return self._cell

    @property
    def atoms(self):
        r"""
        Returns
        -------
        list
            Atoms of the system (atomic type and positions).
        """
        return self._atoms

    def __getitem__(self, index):
        r"""
        The items of a Posinp instance actually are the atoms (so as to
        behave like an immutable list of atoms).

        Parameters
        ----------
        index : int
            Index of a given atom

        Returns
        -------
        Atom
            The required atom.
        """
        return self.atoms[index]

    def __len__(self):
        return len(self.atoms)

    def __eq__(self, other):
        r"""
        Parameters
        ----------
        other : object
            Any other object.

        Returns
        -------
        bool
            `True` if both initial positions have the same number of
            atoms, the same units and boundary conditions and the same
            atoms (whatever the order of the atoms in the initial list
            of atoms).
        """
        try:
            same_base = self.BC == other.BC and len(self) == len(other) and \
                self.units == other.units and self.cell == other._cell
            if same_base:
                same_atoms = all([atom in other.atoms for atom in self.atoms])
                return same_base and same_atoms
            else:
                return False
        except AttributeError:
            return False

    def __ne__(self, other):
        # This is only for the python2 version to work
        return not self.__eq__(other)

    def __str__(self):
        r"""
        Convert the Posinp to a string in the xyz format.

        Returns
        -------
        str
            The Posinp instance as a string.
        """
        # Create the first two lines of the posinp file
        pos_str = "{}   {}\n".format(len(self), self.units)
        pos_str += self.BC
        if self.cell is not None:
            pos_str += "   {}   {}   {}\n".format(*self.cell)
        else:
            pos_str += "\n"
        # Add all the other lines, representing the atoms
        pos_str += "".join([str(atom) for atom in self])
        return pos_str

    def __repr__(self):
        r"""
        Returns
        -------
            The string representation of a Posinp instance.
        """
        r = "Posinp({0.atoms}, {0.units}, {0.BC}, cell={0.cell})".format(self)
        return r

    def write(self, filename):
        r"""
        Write the Posinp on disk.

        Parameters
        ----------
        filename : str
            Name of the input positions file.
        """
        with open(filename, "w") as f:
            f.write(str(self))

    def translate_atom(self, i_at, vector):
        r"""
        Translate the `i_at` atom in the three space coordinates
        according to the value of `vector`.

        Parameters
        ----------
        i_at : int
            Index of the atom.
        vector : list or numpy.array of length 3
            Translation vector to apply.

        Returns
        -------
        Posinp
            A new posinp where the `i_at` atom was translated by
            `vector`.


        .. Warning::

            You have to make sure that the units of the vector match
            those used by the posinp.


        >>> posinp = Posinp([Atom('N', [0, 0, 0]), Atom('N', [0, 0, 1.1])],
        ...                 'angstroem', 'free')
        >>> new_posinp = posinp.translate_atom(1, [0.0, 0.0, 0.05])
        >>> print(new_posinp)
        2   angstroem
        free
        N   0.0   0.0   0.0
        N   0.0   0.0   1.15
        <BLANKLINE>
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
        Parameters
        ----------
        atom_type : str
            Type of the atom.
        position : list or numpy.array of length 3
            Position of the atom.


        >>> a = Atom('C', [0, 0, 0])
        >>> a.type
        'C'
        >>> a.position
        array([0., 0., 0.])
        """
        # TODO: Check that the atom type exists
        assert len(position) == 3
        self._type = atom_type
        self._position = np.array(position, dtype=float)

    @property
    def type(self):
        r"""
        Returns
        -------
        str
            Type of the atom.
        """
        return self._type

    @property
    def position(self):
        r"""
        Returns
        -------
        list or numpy.array of length 3
            Position of the atom.
        """
        return self._position

    def translate(self, vector):
        r"""
        Translate the coordinates of the atom by the values of the
        vector.

        Returns
        -------
        Atom
            Atom translated according to the given vector.

        Parameters
        ----------
        vector : list or numpy.array of length 3
            Translation vector to apply.


        >>> Atom('C', [0, 0, 0]).translate([0.5, 0.5, 0.5])
        Atom('C', [0.5, 0.5, 0.5])
        """
        assert len(vector) == 3
        new_atom = deepcopy(self)
        new_atom._position = self.position + np.array(vector)
        return new_atom

    def __str__(self):
        r"""
        Returns
        -------
        str
            String representation of the atom, mainly used to create the
            string representation of a Posinp instance.
        """
        return "{t}  {: .15}  {: .15}  {: .15}\n"\
               .format(t=self.type, *self.position)

    def __repr__(self):
        r"""
        Returns
        -------
        str
            General string representation of an Atom instance.
        """
        return "Atom('{}', {})".format(self.type, list(self.position))

    def __eq__(self, other):
        r"""
        Two atoms are the same if they are located on the same position
        and have the same type.

        Parameters
        ----------
        other
            Other object.

        Returns
        -------
        bool
            True if both atoms have the same type and position.


        >>> a = Atom('C', [0., 0., 0.])
        >>> a == 1
        False
        >>> a == Atom('N', [0., 0., 0.])
        False
        >>> a == Atom('C', [1., 0., 0.])
        False
        """
        try:
            return (np.allclose(self.position, other.position)
                    and self.type == other.type)
        except AttributeError:
            return False

    # def __ne__(self, other):
    #     # This is only for the python2 version to work correctly
    #     return not self.__eq__(other)
