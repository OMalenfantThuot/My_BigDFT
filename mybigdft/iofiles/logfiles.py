r"""
File containing the InputParams, Posinp, Logfile and Atom classes.
"""

from __future__ import print_function
import warnings
from collections import Sequence, Mapping
import oyaml as yaml
try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:  # pragma: no cover
    from yaml import Loader, Dumper
import numpy as np
from mybigdft.globals import INPUT_VARIABLES
from .inputparams import InputParams, clean
from .posinp import Posinp


__all__ = ["Logfile"]


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
        params = {key: log.get(key) for key in INPUT_VARIABLES}
        params = clean(params)
        self._inputparams = InputParams(params=params)
        self._posinp = self.inputparams.posinp
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
        if self._boundary_conditions is not None:
            self._boundary_conditions = self._boundary_conditions.lower()
        # Make the forces as a numpy array of shape (n_at, 3)
        if self.forces is not None:
            new_forces = np.array([])
            for force in self.forces:
                new_forces = np.append(new_forces, list(force.values())[0])
            n_at = len(self.forces)
            new_forces = new_forces.reshape((n_at, 3))
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
                if isinstance(warning, dict):  # pragma: no cover
                    assert len(warning) == 1
                    key, value = list(warning.items())[0]
                    warning = "{}: {}".format(key, value)
                elif not isinstance(warning, str):  # pragma: no cover
                    print("MyBigDFT: weird error message found")
                    warning = str(warning)
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
        if self.atom_types is not None:
            for atom_type in self.atom_types:
                psp = "psppar.{}".format(atom_type)
                psp_ixc = self[psp]["Pseudopotential XC"]
                inp_ixc = self["dft"]["ixc"]
                if psp_ixc != inp_ixc:
                    warnings.warn(
                        "The XC of pseudo potentials ({}) is different from "
                        "the input XC ({}) for the '{}' atoms"
                        .format(psp_ixc, inp_ixc, atom_type), UserWarning)

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
        Logfile or GeoptLogfile or MultipleLogfile
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
        with open(filename, "r") as stream:
            return cls.from_stream(stream)

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
        Logfile or GeoptLogfile or MultipleLogfile
            Logfile initialized from a stream.
        """
        # The logfile might contain multiple documents
        docs = yaml.load_all(stream, Loader=Loader)
        logs = [cls(doc) for doc in docs]
        if len(logs) == 1:
            # If only one document, return a Logfile instance
            return logs[0]
        else:
            warnings.warn(
                "More than one document found in the logfile!", UserWarning)
            if logs[0].inputparams["geopt"] is not None:
                # If the logfile corresponds to a geopt calculation,
                # return a GeoptLogfile instance
                return GeoptLogfile(logs)
            else:  # pragma: no cover
                # In other cases, just return a MultipleLogfile instance
                return MultipleLogfile(logs)

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
            return super(Logfile, self).__getattr__(name)

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
        with open(filename, "w") as stream:
            yaml.dump(self.log, stream=stream, Dumper=Dumper)

    @property
    def posinp(self):
        r"""
        Returns
        -------
        Posinp
            Posinp used during the calculation.
        """
        return self._posinp

    @property
    def inputparams(self):
        r"""
        Returns
        -------
        InputParams
            Input parameters used during the calculation.
        """
        return self._inputparams


class MultipleLogfile(Sequence):
    r"""
    Class allowing to initialize, read, write and interact with an
    output file of a BigDFT calculation containing multiple documents.
    """

    def __init__(self, logs):
        r"""
        Parameters
        ----------
        logs : list
            List of the various documents contained in the logfile of a
            BigDFT calculation.
        """
        self._logs = logs

    @property
    def logs(self):
        r"""
        Returns
        -------
        list
            List of the documents read from a single output of a BigDFT
            calculation.
        """
        return self._logs

    def __getitem__(self, index):
        return self.logs[index]

    def __len__(self):
        return len(self.logs)

    def write(self, filename):
        r"""
        Write the logfile on disk.

        Parameters
        ----------
        filename : str
            Name of the logfile.
        """
        logs = [log.log for log in self.logs]
        with open(filename, "w") as stream:
            yaml.dump_all(
                logs, stream=stream, Dumper=Dumper, explicit_start=True)


class GeoptLogfile(MultipleLogfile):
    r"""
    Class allowing to initialize, read, write and interact with an
    output file of a geometry optimization calculation.
    """

    def __init__(self, logs):
        r"""
        Parameters
        ----------
        logs : list
            List of the various documents contained in the logfile of a
            geometry optimization calculation.
        """
        super(GeoptLogfile, self).__init__(logs)
        # Update the input parameters and positions of the documents
        for log in self.logs[1:]:
            log._inputparams = self.inputparams
            log._posinp = Posinp.from_dict(log['Atomic structure'])
        self._posinps = [log.posinp for log in self.logs]

    @property
    def inputparams(self):
        r"""
        Returns
        -------
        InputParams
            Input parameters used for each step of the geometry
            optimization procedure.
        """
        return self.logs[0].inputparams

    @property
    def posinps(self):
        r"""
        Returns
        -------
        list
            List of the input positions for each step of the geometry
            optimization procedure.
        """
        return self._posinps
