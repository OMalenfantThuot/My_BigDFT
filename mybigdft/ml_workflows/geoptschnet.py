r"""
The :class:`Geoptschnet` allows to find a relaxed structure using
a SchnetPack trained model
"""

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
        forcemax=0.01,
        step_size=0.003,
        max_iter=300,
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
        self.posinp = posinp
        self.forcemax = forcemax
        self.step_size = step_size
        self.max_iter = max_iter
        self.final_posinp = None

        self._write_to_disk = write_to_disk
        if self.write_to_disk:
            if out_name == "":
                self.out_name = "final_posinp"
            else:
                self.out_name = out_name
        else:
            self.out_name = None

    @property
    def posinp(self):
        r"""
        Returns
        -------
        Posinp
            Initial posinp of the geometry optimization procedure
        """
        return self._posinp

    @posinp.setter
    def posinp(self, posinp):
        self._posinp = posinp

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

    @final_posinp.setter
    def final_posinp(self, final_posinp):
        self._final_posinp = final_posinp

    @property
    def forcemax(self):
        r"""
        Returns
        -------
        float
            Stopping criterion on the forces (in eV/Angstrom)
        """
        return self._forcemax

    @forcemax.setter
    def forcemax(self, forcemax):
        self._forcemax = forcemax

    @property
    def step_size(self):
        r"""
        Returns
        -------
        float
            Step size for each relaxation step
        """
        return self._step_size

    @step_size.setter
    def step_size(self, step_size):
        self._step_size = step_size

    @property
    def max_iter(self):
        r"""
        Returns
        -------
        int
            Maximum number of iterations
        """
        return self._max_iter

    @max_iter.setter
    def max_iter(self, max_iter):
        self._max_iter = max_iter

    @property
    def write_to_disk(self):
        r"""
        Returns
        -------
        bool
            If `True`, the obtained structure will be written on disk.
        """
        return self._write_to_disk

    @write_to_disk.setter
    def write_to_disk(self, write_to_disk):
        self._write_to_disk = write_to_disk

    @property
    def out_name(self):
        r"""
        Returns
        -------
        str or None
            Name of the output position file, None if write_to_disk is `False`.
        """
        return self._out_name

    @out_name.setter
    def out_name(self, out_name):
        self._out_name = out_name

    def run(self, model_dir=None, device="cpu", batch_size=128, recenter=False):
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

        temp_posinp = deepcopy(self.posinp)

        for i in range(1, self.max_iter + 1):
            job = Jobschnet(posinp=temp_posinp)
            job.run(
                model_dir=model_dir, forces=True, device=device, batch_size=batch_size
            )
            for j in range(job.logfile.n_at[0]):
                temp_posinp = temp_posinp.translate_atom(
                    j, self.step_size * job.logfile.forces[0][j]
                )
            if np.max(np.abs(job.logfile.forces[0])) < self.forcemax:
                print("Geometry optimization stopped at iteration {}.".format(i))
                break
            if i % 100 == 0:
                self.step_size = self.step_size * 0.9
            if i == self.max_iter:
                print(
                    "Geometry optimization was not succesful at iteration {}.".format(i)
                )
        print(
            "Max remaining force is {:6.4f}.".format(
                np.max(np.abs(job.logfile.forces[0]))
            )
        )
        self.final_posinp = temp_posinp
        if recenter:
            self.final_posinp = self.final_posinp.to_centroid()

        if self.write_to_disk:
            self.final_posinp.write(self.out_name + ".xyz")
