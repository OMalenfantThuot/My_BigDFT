"""
The :class: `JobSchnet` class is the base class defining a SchnetPack calculation.
"""

from __future__ import print_function, absolute_import
import os
import torch
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
        """
        # Verify there are initial positions
        if posinp is None:
            raise ValueError("A JobSchnet instance has no initial positions.")
        elif not isinstance(posinp, list):
            posinp = [posinp]
        for pos in posinp:
            if not isinstance(pos, mybigdft.Posinp):
                raise TypeError(
                    "Atomic Positions should be given in a list of mybigdft.Posinp instances."
                )

        # Set the base attributes
        self._posinp = posinp
        self._require_forces = require_forces
        self._name = str(name)
        self._skip = bool(skip)

        self._set_directories(run_dir)
        self._set_filenames()

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
            device=device,
            disk_out=write_to_disk,
            batch_size=batch_size,
            overwrite=overwrite,
            return_values=True,
        )
