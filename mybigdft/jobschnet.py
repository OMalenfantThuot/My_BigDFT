r"""
The :class:`Jobschnet` class is the base class defining a SchnetPack calculation.
"""

from __future__ import print_function, absolute_import
import os
import torch
import numpy as np
from copy import deepcopy
from mybigdft import Posinp
from schnetpack.utils.script_utils.predict import predict


class Jobschnet(object):
    r"""
    This class defines a SchnetPack calculation similarly
    to the Job class for BigDFT.
    """

    def __init__(self, name="", posinp=None):
        r"""
        Parameters
        ----------
        name : str
            Name of the job. Will be used to name the created files.
        posinp : Posinp
            Base atomic positions for the job
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
        self.posinp = posinp
        self.number_of_structures = len(self.posinp)
        self.name = name
        self.logfile = Logfileschnet(self.posinp)

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

    @name.setter
    def name(self, name):
        self._name = str(name)

    @property
    def posinp(self):
        r"""
        Returns
        -------
        Posinp
            Initial positions of the calculation
        """
        return self._posinp

    @posinp.setter
    def posinp(self, posinp):
        self._posinp = posinp

    @property
    def number_of_structures(self):
        r"""
        Returns
        -------
        int
            Number of different structures when the job is declared
        """
        return self._number_of_structures

    @number_of_structures.setter
    def number_of_structures(self, number_of_structures):
        self._number_of_structures = int(number_of_structures)

    @property
    def logfile(self):
        r"""
        Returns
        -------
        Logfileschnet
            Object empty or containing the results of the calculation.
        """
        return self._logfile

    @logfile.setter
    def logfile(self, logfile):
        self._logfile = logfile

    @property
    def outfile_name(self):
        r"""
        Returns
        -------
        str
            Name of the output file
        """
        return self._outfile_name

    @outfile_name.setter
    def outfile_name(self, outfile_name):
        self._outfile_name = outfile_name

    def _set_filenames(self):
        if self.name != "":
            self.outfile_name = self.name + ".out"
        else:
            self.outfile_name = "outfile.out"

    def run(
        self,
        model_dir=None,
        forces=False,
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
        forces : int or bool
            Order of the force calculations (0, 1 or 2)                                   
            If 0 (or `False`), forces are not evaluated.
            If 1, forces are evaluated on first-order (6N calculations)
            If 2 (or `True`), forces are evaluated on second-order (12N calculations)
        device : str
            Either 'cpu' or 'cuda' to run on cpu or gpu
        write_to_disk : bool
            If `True`, an outfile will be written after the calculation.
        batch_size : int
            Size of the mini-batches used in predictions
        overwrite : bool
            If `True`, all .db files are removed from the directory before 
            the calculation is done.
        """
        # Verify model_dir
        if model_dir is None:
            raise ValueError("This job needs a path to a stored model.")
        if not isinstance(model_dir, str):
            raise TypeError("The path to the stored model must be a string.")
        try:
            model_dir = os.environ["MODELDIR"] + model_dir
        except KeyError:
            pass

        # Forces verification and preparation
        if isinstance(forces, bool):
            forces = 2 if forces else 0
        if forces in [0, 1, 2]:
            if forces == 1:
                self._create_additional_structures(order=1)
            if forces == 2:
                self._create_additional_structures(order=2)
        else:
            raise ValueError(
                "Parameter `forces` should be a bool or a int between 0 and 2."
            )

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
        raw_predictions = predict(
            modelpath=model_dir,
            posinp=self.posinp,
            name=self.name,
            device=device,
            disk_out=False,
            batch_size=batch_size,
            overwrite=overwrite,
            return_values=True,
        )

        # Determine available properties
        if "energy_U0" in list(raw_predictions.keys()):
            raw_predictions["energy"] = raw_predictions.pop("energy_U0")
        available_properties = list(raw_predictions.keys())
        available_properties.remove("idx")

        # Format the predictions
        predictions = {}
        if not forces:
            for prop in available_properties:
                predictions[prop] = list(raw_predictions[prop])
        else:
            # Calculate the forces
            pred_idx = 0
            predictions["energy"], predictions["forces"] = [], []
            for struct_idx in range(self.number_of_structures):
                predictions["energy"].append(raw_predictions["energy"][pred_idx][0])
                pred_idx += 1
                if forces == 1:
                    predictions["forces"].append(
                        self._calculate_forces(
                            raw_predictions["energy"][
                                pred_idx : pred_idx
                                + 6 * len(self._init_posinp[struct_idx])
                            ],
                            order=forces,
                        )
                    )
                    pred_idx += 6 * len(self._init_posinp[struct_idx])
                elif forces == 2:
                    predictions["forces"].append(
                        self._calculate_forces(
                            raw_predictions["energy"][
                                pred_idx : pred_idx
                                + 12 * len(self._init_posinp[struct_idx])
                            ],
                            order=forces,
                        )
                    )
                    pred_idx += 12 * len(self._init_posinp[struct_idx])

        self.logfile._update_results(predictions)

        # Reset self._posinp for more calculations
        try:
            self.posinp = deepcopy(self._init_posinp)
        except:
            pass

        if write_to_disk:
            # To improve?
            with open(self.outfile_name, "w") as out:
                for idx, struct in enumerate(self.posinp):
                    out.write("Structure {}\n".format(idx))
                    out.write("-------------------\n")
                    out.write("Energy : {}\n".format(self.logfile.energy[idx]))
                    out.write("Forces : \n")
                    np.savetxt(out, self.logfile.forces[idx])
                    out.write("\n")

    def _create_additional_structures(self, order, deriv_length=0.02):
        r"""
        Creates the additional structures needed to do a numeric
        derivation of the energy to calculate the forces.
        """
        if order not in [1, 2]:
            raise ValueError("Order of the forces calculation should be 1 or 2.")
        self._init_posinp = deepcopy(self.posinp)
        self._deriv_length = deriv_length
        all_structs = []
        # First order forces calculation
        if order == 1:
            for str_idx, struct in enumerate(self.posinp):
                all_structs.append(struct)
                for factor in [1, -1]:
                    for dim in [
                        np.array([1, 0, 0]),
                        np.array([0, 1, 0]),
                        np.array([0, 0, 1]),
                    ]:
                        all_structs.extend(
                            [
                                struct.translate_atom(
                                    atom_idx, deriv_length * factor * dim
                                )
                                for atom_idx in range(len(struct))
                            ]
                        )
            self.posinp = all_structs
        # Second order forces calculations
        elif order == 2:
            for str_idx, struct in enumerate(self.posinp):
                all_structs.append(struct)
                for factor in [2, 1, -1, -2]:
                    for dim in [
                        np.array([1, 0, 0]),
                        np.array([0, 1, 0]),
                        np.array([0, 0, 1]),
                    ]:
                        all_structs.extend(
                            [
                                struct.translate_atom(
                                    atom_idx, deriv_length * factor * dim
                                )
                                for atom_idx in range(len(struct))
                            ]
                        )
            self.posinp = all_structs

    def _calculate_forces(self, predictions, order):
        r"""
        Method to calculate forces from the displaced atomic positions

        Parameters
        ----------
        predictions : 1D numpy array (size 6*n_at)
             Contains the predictions obtained from the neural network

        Returns
        -------
        forces : 2D numpy array (size (n_at, 3))
            Forces for each structure
        """
        if order == 1:
            nat = int(len(predictions) / 6)
            forces = np.zeros((nat, 3))
            for i in range(3):
                ener1, ener2 = (
                    predictions[np.arange(i * nat, (i + 1) * nat, 1)],
                    predictions[np.arange((i + 3) * nat, (i + 4) * nat, 1)],
                )
                forces[:, i] = -(ener1 - ener2).reshape(nat) / (2 * self._deriv_length)
        elif order == 2:
            nat = int(len(predictions) / 12)
            forces = np.zeros((nat, 3))
            for i in range(3):
                ener1, ener2, ener3, ener4 = (
                    predictions[np.arange(i * nat, (i + 1) * nat, 1)],
                    predictions[np.arange((i + 3) * nat, (i + 4) * nat, 1)],
                    predictions[np.arange((i + 6) * nat, (i + 7) * nat, 1)],
                    predictions[np.arange((i + 9) * nat, (i + 10) * nat, 1)],
                )
                forces[:, i] = -(
                    (-ener1 + 8 * ener2 - 8 * ener3 + ener4).reshape(nat)
                    / (12 * self._deriv_length)
                )
        else:
            raise ValueError("Order of the forces calculation should be 1 or 2.")
        return forces


class Logfileschnet(object):
    r"""
    Container class to emulate the Logfile object used in BigDFT calculations
    """

    def __init__(self, posinp):

        self.posinp = posinp
        self.n_at = []
        self.atom_types = []
        self.boundary_conditions = []
        self.cell = []

        for struct in self.posinp:
            self.n_at.append(len(struct))
            self.atom_types.append(set([atom.type for atom in struct]))
            self.boundary_conditions.append(struct.boundary_conditions)
            self.cell.append(struct.cell)

        self.energy = None
        self.forces = None
        self.dipole = None

    @property
    def posinp(self):
        r"""
        Returns
        -------
        list of Posinps
            List containing the base Posinp objects for the predictions.
        """
        return self._posinp

    @posinp.setter
    def posinp(self, posinp):
        self._posinp = posinp

    @property
    def n_at(self):
        r"""
        Returns
        -------
        list of ints
            List containing the number of atoms of each structure.
        """
        return self._n_at

    @n_at.setter
    def n_at(self, n_at):
        self._n_at = n_at

    @property
    def atom_types(self):
        r"""
        Returns
        -------
        list of sets
            List containing sets of the elements present in each structure.
        """
        return self._atom_types

    @atom_types.setter
    def atom_types(self, atom_types):
        self._atom_types = atom_types

    @property
    def boundary_conditions(self):
        r"""
        Returns
        -------
        list of strings
            List containing the boundary conditions, either `free`,
            `surface` or `periodic`, of each structure.
        """
        return self._boundary_conditions

    @boundary_conditions.setter
    def boundary_conditions(self, boundary_conditions):
        self._boundary_conditions = boundary_conditions

    @property
    def cell(self):
        r"""
        Returns
        -------
        list of lists of floats
            List containing cell dimensions of each structure,
            None for free boundary conditions.
        """
        return self._cell

    @cell.setter
    def cell(self, cell):
        self._cell = cell

    @property
    def energy(self):
        r"""
        Returns
        -------
        list of floats or None
            List containing the energy value for each structure.
            If None, the energies have not been calculated yet.
        """
        return self._energy

    @energy.setter
    def energy(self, energy):
        self._energy = energy

    @property
    def forces(self):
        r"""
        Returns
        -------
        list of arrays of floats or None
            List containing the forces values for each structure.
            If None, the forces have not been calculated yet.
        """
        return self._forces

    @forces.setter
    def forces(self, forces):
        self._forces = forces

    @property
    def dipole(self):
        r"""
        Returns
        -------
        list of arrays of floats or None
            List containing the dipole values for each structure.
            If None, the dipoles have not been calculated yet.
        """
        return self._dipole

    @dipole.setter
    def dipole(self, dipole):
        self._dipole = dipole

    def _update_results(self, predictions):
        r"""
        Method to store Jobschnet results in the Logfileschnet container.
        Useful for the workflows.

        Parameters
        ----------
        predictions : dict
            Dictionary containing the predictions returned by the
            Schnetpack calculations
        """

        available_properties = list(predictions.keys())

        if "energy" in available_properties:
            self.energy = predictions["energy"]
        if "forces" in available_properties:
            self.forces = predictions["forces"]
        if "dipole" in available_properties:
            self.dipole = predictions["dipole"]
