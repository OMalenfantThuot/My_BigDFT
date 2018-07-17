r"""
File containing the Input and Posinp classes.
"""

from __future__ import print_function
import warnings
from copy import deepcopy
from collections import Sequence, Mapping, MutableMapping
import oyaml as yaml
import numpy as np
from .globals import inp_vars


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
        if key not in inp_vars:
            raise KeyError("Unknown key '{}'".format(key))
        for subkey in value.keys():
            if subkey not in inp_vars[key].keys():
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
            if (child_value == inp_vars[key][child_key].get("default")) or (
                    child_key == "ncount_cluster_x" and
                    params["geopt"].get("method") is None):
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


        >>> inp = {
        ... 'posinp': {'units': 'angstroem', 'positions':
        ... [{'N': [2.9763078243490115e-23, 6.872205952043537e-23,
        ...         0.01071619987487793]},
        ...  {'N': [-1.1043449194501671e-23, -4.873421744830746e-23,
        ...         1.104273796081543]}],
        ...  'properties': {'format': 'xyz', 'source': 'N2.xyz'}}}
        >>> log = Logfile.from_file("tests/log.yaml")
        >>> inp == InputParams.from_Logfile(log)
        True
        """
        params = {key: logfile[key] for key in inp_vars}
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
        params = yaml.safe_load(string)
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
        self._posinp = self._extract_posinp()
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
                              "than the input XC ({}) for the '{}' atoms"
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
        log = yaml.safe_load(stream)
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

    def __dir__(self):  # pragma: no cover
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

        Parameters
        ----------
        filename : str
            Name of the logfile.
        """
        with open(filename, "w") as f:
            yaml.safe_dump(self.log, stream=f)

    @property
    def posinp(self):
        r"""
        Returns
        -------
        Posinp
            Posinp used during the calculation.
        """
        return self._posinp

    def _extract_posinp(self):
        r"""
        Extract the posinp from the information contained in the
        logfile.

        Returns
        -------
        Posinp
            Posinp used during the calculation.
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
          of atoms and the units (given by a string),
        * the second element is made of another list, made of the
          boundary conditions (given by a string) and possibly three
          distances defining the cell size along each space coordinate
          (:math:`x`, :math:`y` and :math:`z`).
        * all the other elements must be Atom instances (defining the
          type (as a string) and the position of each atom).

        Parameters
        ----------
        posinp : list
            xyz file stored as a list.


        >>> posinp = Posinp([[2, 'angstroem'], ['free'],
        ... Atom('N', [0, 0, 0]), Atom('N', [0, 0, 1.1])])
        >>> posinp.n_at
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
            self._cell = [float(coord) if coord != ".inf" else coord
                          for coord in second_line[1:4]]
        else:
            self._cell = None
        # Set the attributes associated to all the other lines of the
        # xyz file, namely the atoms
        if self.n_at != len(posinp):
            raise ValueError("The number of atoms do not correspond to the "
                             "number of positions.")
        self._atoms = posinp

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
                    cell = [float(c) if c != ".inf" else c
                            for c in content[1:4]]
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
    def from_InputParams(cls, inputparams):
        r"""
        Initialize the input positions from BigDFT input parameters.

        Parameters
        ----------
        inputparams : InputParams
            Input parameters of a BigDFT calculation.

        Returns
        -------
        Posinp
            Posinp initialized from an InputParams instance.
        """
        ref_pos = inputparams["posinp"]
        # Set the values converning the first line, e.g.:
        # - the number of atoms
        # - the units
        n_at = len(ref_pos["positions"])
        units = ref_pos["units"]
        posinp = [[n_at, units]]
        # Set the values concerning the second line, e.g.:
        # - the boundary condition (BC)
        # - the size of the cell (optional)
        cell = ref_pos.get("cell")
        if cell is None:
            BC = "free"
            second_line = [BC]
        else:
            if cell[1] == ".inf":
                BC = "surface"
            else:
                BC = "periodic"
            second_line = [BC]
            second_line += cell
        posinp.append(second_line)
        # Set the values for the the atoms
        for atom in ref_pos["positions"]:
            [(atom_type, position)] = atom.items()
            posinp.append(Atom(atom_type, position))
        return cls(posinp)

    @property
    def n_at(self):
        r"""
        Returns
        -------
        int
            Number of atoms.
        """
        return self._n_at

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
        other : Posinp
            Other initial positions to be compared.

        Returns
        -------
        bool
            True if both initial positions have the same string
            representation.
        """
        return str(self) == str(other)

    def __ne__(self, other):
        # This is only for the python2 version to work
        return not self.__eq__(other)

    def __str__(self):
        r"""
        Convert the Posinp to a string.

        Returns
        -------
        str
            The Posinp instance as a string.
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

    def __repr__(self):
        r"""
        Returns
        -------
            The string representation of a Posinp instance.
        """
        msg = "[[{}, {}], [{}".format(self.n_at, self.units, self.BC)
        if self.cell is not None:
            msg += ", {}".format(self.cell)
        msg += "], "
        msg += ", ".join((repr(atom) for atom in self))
        msg += "]"
        return msg

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


        >>> posinp = Posinp([[2, 'angstroem'], ['free'],
        ... Atom('N', [0, 0, 0]), Atom('N', [0, 0, 1.1])])
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
        >>> a.atom_type
        'C'
        >>> a.position
        array([0., 0., 0.])
        """
        # TODO: Check that the atom type exists
        assert len(position) == 3
        self._atom_type = atom_type
        self._position = np.array(position, dtype=float)

    @property
    def atom_type(self):
        r"""
        Returns
        -------
        str
            Type of the atom.
        """
        return self._atom_type

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
               .format(t=self.atom_type, *self.position)

    def __repr__(self):
        r"""
        Returns
        -------
        str
            General string representation of an Atom instance.
        """
        return "Atom('{}', {})".format(self.atom_type, list(self.position))

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
            return (np.array_equal(self.position, other.position)
                    and self.atom_type == other.atom_type)
        except AttributeError:
            return False
