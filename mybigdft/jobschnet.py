"""
The :class: `JobSchnet` class is the base class defining a SchnetPack calculation.
"""

from __future__ import print_function, absolute_import
import os
import torch
import numpy as np
from mybigdft import Posinp
from schnetpack.utils.script_utils.predict import predict


class Jobschnet(object):
    r"""
    This class defines a SchnetPack calculation similarly
    to the Job class for BigDFT.
    """

    def __init__(
        self, name="", posinp=None, require_forces=False, run_dir=None, skip=False
    ):
        r"""
        Parameters
        ----------
        name : str
            Name of the job. Will be used to name the created files.
        posinp : Posinp
            Base atomic positions for the job
        require_forces : bool
            If `True`, the forces need to be evaluated also.
        run_dir : str or None
            Folder where to run calculations (default to current directory)
        skip : bool
            If `True`, the calculation will be skipped (not used presently).
        """
        # Verify there are initial positions
        if posinp is None:
            raise ValueError("A JobSchnet instance has no initial positions.")
        elif not isinstance(posinp, list):
            posinp = [posinp]
        for pos in posinp:
            if not isinstance(pos, Posinp):
                raise TypeError(
                    "Atomic Positions should be given in a list of mybigdft.Posinp instances."
                )

        # Set the base attributes
        self._posinp = posinp
        self._number_of_structures = len(self._posinp)
        self._require_forces = require_forces
        self._name = str(name)
        self._skip = bool(skip)
        self._logfile = Logfileschnet(self._posinp)

        self._set_directories(run_dir)
        self._set_filenames()

        if self._require_forces:
            self.create_additional_structures()

    @property
    def name(self):
        r"""
        Returns
        -------
        str
            Base name of the calculation used to set the names of
            files and directories.
        """
        return self._name

    @property
    def posinp(self):
        r"""
        Returns
        -------
        Posinp
            Initial positions of the calculation
        """
        return self._posinp

    @property
    def number_of_structures(self):
        r"""
        Returns
        -------
        int
            Number of different structures when the job is declared
        """
        return self._number_of_structures

    @property
    def require_forces(self):
        r"""
        Returns
        -------
        bool
            If `True`, the forces of the structure must be evaluated.
        """
        return self._require_forces

    @property
    def skip(self):
        r"""
        Returns
        -------
        bool
            If `True`, the calculation will be skipped.
        """
        return self._skip

    @property
    def logfile(self):
        r"""
        Returns
        -------
        Logfileschnet
            Object empty or containing the results of the calculation.
        """
        return self._logfile

    @property
    def init_dir(self):
        r"""
        Returns
        -------
        str
            Absolute path to the initial directory of the calculation
        """
        return self._init_dir

    @property
    def run_dir(self):
        r"""
        Returns
        -------
        str
            Absolute path to the directory where the calculation is run.
        """
        return self._run_dir

    @property
    def posinp_name(self):
        r"""
        Returns
        -------
        str
            Name of base posinp file
        """
        return self._posinp_name

    @property
    def outfile_name(self):
        r"""
        Returns
        -------
        str
            Name of the output file
        """
        return self._outfile_name

    def _set_directories(self, run_dir):
        self._init_dir = os.getcwd()
        if run_dir is None:
            self._run_dir = self.init_dir
        else:
            basename = os.path.commonprefix([self.init_dir, run_dir])
            if basename == "":
                self._run_dir = os.path.join([self.init_dir, run_dir])
            else:
                self._init_dir = basename
                new_run_dir = os.path.relpath(run_dir, start=basename)
                self._run_dir = os.path.join(self.init_dir, new_run_dir)

    def _set_filenames(self):
        if self.name != "":
            self._posinp_name = self.name + ".xyz"
            self._outfile_name = self.name + ".out"
        else:
            self._posinp_name = "posinp.xyz"
            self._outfile_name = "outfile.out"

    def __enter__(self):
        r"""
        When entering the context manager:

        * create the directory where the calculations must be run,
        * go to that directory.
        """
        if self.run_dir not in [".", ""]:
            if not os.path.exists(self.run_dir):
                os.makedirs(self.run_dir)
            os.chdir(self.run_dir)
        print(os.getcwd())
        return self

    def __exit__(self, *args):
        r"""
        When leaving the context manager, go back to the initial
        directory.
        """
        os.chdir(self.init_dir)

    def create_additional_structures(self, deriv_length=0.001):
        r"""
        Creates the additional structures needed to do a numeric
        derivation of the energy to calculate the forces.
        """
        all_structs = []
        for str_idx, struct in enumerate(self._posinp):
            all_structs.append(struct)
            for dim in [
                np.array([1, 0, 0]),
                np.array([-1, 0, 0]),
                np.array([0, 1, 0]),
                np.array([0, -1, 0]),
                np.array([0, 0, 1]),
                np.array([0, 0, -1]),
            ]:
                all_structs.extend(
                    [
                        struct.translate_atom(atom_idx, deriv_length * dim)
                        for atom_idx in range(len(struct))
                    ]
                )
        self._posinp = all_structs

    def run(
        self,
        model_dir=None,
        device="cpu",
        write_to_disk=False,
        batch_size=128,
        overwrite=False,
    ):
        r"""
        Parameters
        ----------
        model_dir: str
            Absolute path to the SchnetPack model to use in calculation
        device : str
            Either 'cpu' or 'cuda' to run on cpu or gpu
        write_to_disk : bool
            If `True`, an outfile will be written after the calculation.
        batch_size : int
            Size of the mini-batches used in predictions
        overwrite : bool
            If `True`, all .db files are removed from the run_dir before 
            the calculation is done.
        """
        # Verify model_dir
        if model_dir is None:
            raise ValueError("This job needs a path to a stored model.")
        if not isinstance(model_dir, str):
            raise TypeError("The path to the stored model must be a string.")

        # Verify device
        device = str(device)
        if device.startswith("cuda"):
            if not torch.cuda.is_available():
                raise Warning("CUDA was asked for, but is not available.")

        # Verify batch_size
        if not isinstance(batch_size, int):
            try:
                batch_size == int(batch_size)
            except:
                raise TypeError("The mini-batches sizes are not defined correctly.")

        # Run the actual calculation
        predictions = predict(
            modelpath=model_dir,
            posinp=self._posinp,
            name=self._name,
            device=device,
            disk_out=write_to_disk,
            batch_size=batch_size,
            overwrite=overwrite,
            return_values=True,
        )

        # calculate the forces
        if self._require_forces:
            idx = 0
            pass

class Logfileschnet(object):
    # Container class to emulate the Logfile object used in BigDFT calculations
    def __init__(self, posinp):

        self._posinp=posinp
        self._n_at = []
        self._atom_types = []
        self._boundary_conditions = []
        self._cell = []
        
        for struct in posinp:
            self._n_at.append(len(struct))
            self._atom_types.append(set([atom.type for atom in struct]))
            self._boundary_conditions.append(struct.boundary_conditions)
            self._cell.append(struct.cell)
        
        self._energies=None
        self._forces=None
        self._dipole=None

    def update_results(self, predictions):
        r"""
        Method to store Jobschnet results in the Logfileschnet container

        Parameters
        ----------
        predictions : dict
            Dictionary containing the predictions returned by the
            Schnetpack calculations
        """

        available_properties = list(predictions.keys())

        if any(x in available_properties for x in ["energy", "energy_U0"]):
            try:
                self._energies = predictions["energy_U0"]
            except KeyError:
                self._energies = predictions["energy"]
