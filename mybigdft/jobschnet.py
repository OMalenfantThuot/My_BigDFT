"""
The :class: `JobSchnet` class is the base class defining a SchnetPack calculation.
"""

from __future__ import print_function, absolute_import
import os


class JobSchnet(object):
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

        # Set the base attributes
        self._posinp = posinp
        self._require_forces = require_forces
        self._name = str(name)
        self._skip = bool(skip)

        self._set_directories(run_dir)
        self._set_filename_attributes()
        self._set_cmd_attributes

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

    def _set_directories(self, run_dir):
        self.init_dir = os.getcwd()
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
