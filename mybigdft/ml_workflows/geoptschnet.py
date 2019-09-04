r"""
The :class:`Geoptschnet` allows to find a relaxed structure using
a SchnetPack trained model
"""

import os
import numpy as np
from copy import deepcopy
from mybigdft import Posinp, Jobschnet


class Geoptschnet:
    r"""
    This class allows to relax the input geometry of a given system in
    order to find the structure that minimizes the forces. The final
    result obtained depends on the trained SchnetPack model used.
    """

    def __init__(
        self,
        posinp,
        forcemax=5e-3,
        step_size=0.002,
        max_iter=200,
        write_to_disk=False,
        out_name="",
    ):
        r"""
        Parameters
        ----------
        posinp : mybigdft.Posinp
            Starting configuration to relax
        forcemax : float
            Stopping criterion on the forces (in eV/Angstrom)
        step_size : float
            Step size for each relaxation step
        max_iter : int
            Maximum number of iterations
        write_to_disk : bool
            If `True`, the final positions will be written on disk.
        out_name : str
            Name of the output file. Default is "final_posinp"
        """
        if posinp is None:
            raise ValueError("No initial positions were provided.")
        self._posinp = posinp
        self._forcemax = forcemax
        self._step_size = step_size
        self._max_iter = max_iter
        self._final_posinp = None

        self._write_to_disk = write_to_disk
        if out_name == "":
            self._out_name = "final_posinp"

    @property
    def posinp(self):
        r"""
        Returns
        -------
        Posinp
            Initial posinp of the geometry optimization procedure
        """
        return self._posinp

    @property
    def final_posinp(self):
        r"""
        Returns
        -------
        Posinp or None
            Final posinp of the geometry optimization or None if
            the the optimization has not been completed
        """
        return self._final_posinp

    @property
    def forcemax(self):
        r"""
        Returns
        -------
        float
            Stopping criterion on the forces (in eV/Angstrom)
        """
        return self._forcemax

    @property
    def step_size(self):
        r"""
        Returns
        -------
        float
            Step size for each relaxation step
        """
        return self._step_size

    @property
    def max_iter(self):
        r"""
        Returns
        -------
        int
            Maximum number of iterations
        """
        return self._max_iter

    @property
    def write_to_disk(self):
        r"""
        Returns
        -------
        bool
            If `True`, the obtained structure will be written on disk.
        """
        return self._write_to_disk

    @property
    def out_name(self):
        r"""
        Returns
        -------
        str
            Name of the output position file, only relevant if
            write_to_disk is `True`.
        """
        return self._out_name

    def run(self, model_dir=None, device="cpu", batch_size=128):
        r"""
        Parameters
        ----------
        model_dir : str
            Absolute path to the SchnetPack model to use in calculation
        device : str
            Either 'cpu' or 'cuda' to run on cpu or gpu
        batch_size : int
            Size of the mini-batches used in predictions
        """

        try:
            model_dir = os.environ["MODELDIR"] + model_dir
        except KeyError:
            pass
        temp_posinp = deepcopy(self._posinp)

        for i in range(1, self._max_iter + 1):
            job = Jobschnet(posinp=temp_posinp)
            job.run(
                model_dir=model_dir, forces=True, device=device, batch_size=batch_size
            )
            for j in range(job.logfile.n_at[0]):
                temp_posinp = temp_posinp.translate_atom(
                    j, self._step_size * job.logfile.forces[0][j]
                )
            if np.max(np.abs(job.logfile.forces[0])) < self._forcemax:
                print("Geometry optimization stopped at iteration {}.".format(i))
                print("Max remaining force is {:6.4f}.".format(np.max(np.abs(job.logfile.forces[0]))))
                self._final_posinp = temp_posinp
                break
            if i == self._max_iter:
                print(
                    "Geometry optimization was not succesful at iteration {}.".format(i)
                )

        if self._write_to_disk:
            self._final_posinp.write(self._out_name + ".xyz")
