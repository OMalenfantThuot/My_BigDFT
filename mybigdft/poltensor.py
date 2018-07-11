r"""
File containing the class allowing allowing to compute the
polarizability tensor of a molecule with BigDFT.
"""

from __future__ import print_function, absolute_import
import os
from copy import deepcopy
import numpy as np
from .mybigdft import BigDFTCalc, COORDS, SIGNS
from .iofiles import Logfile


class PolTensorCalc(BigDFTCalc):
    r"""
    This class allows to run all the calculations required to get the
    polarizability tensor for a given geometry of a system.
    """

    def __init__(self, input_yaml, posinp, ef_amplitudes=[1.E-4]*3,
                 prefix=None, run_folder=None, ref_calc=None):
        r"""
        Method initializing a polarizability tensor calculation. It
        inherits from the BigDFT class. To get the polarizability
        tensor, electric fields in the directions x, y and z are
        applied. The elements of the tensor are defined as the
        coefficients between the change of the dipole (in one
        direction) and the amplitude of the perturbative electric
        field applied (in another direction).

        The amplitudes of the electric fields along the three space
        coordinates are therefore important parameters, specified by
        ef_amplitudes.

        :param input_yaml: Reference input file for all the
                           calculations.
        :type input_yaml: inputfile.Input
        :param posinp: Reference geometry for all the calculations.
        :type posinp: posinp.Posinp
        :param ef_amplitudes: Amplitudes of the electric field to
                              apply along each direction.
        :type ef_amplitudes: list of length 3
        :param prefix: Prefix of the BigDFT calculation (optional,
                       default value set to None).
        :type prefix: str
        :param run_folder: Folder where to run the calculation
        :type run_folder: str
        :param ref_calc: Other BigDFT calculation, taken as reference.
        :type ref_calc: bigdft.BigDFTCalc
        """
        # Initialize the instance via the BigDFT class
        super(PolTensorCalc, self).__init__(input_yaml, posinp, prefix=prefix,
            run_folder=run_folder, ref_calc=ref_calc)  # noqa
        # Initialize the electric field amplitudes
        self.ef_amplitudes = ef_amplitudes
        # Initialize the dictionary of paths to all the logfiles.
        self.logfiles_paths = {}
        for coord in COORDS:
            self.logfiles_paths[coord] = []

    def run(self, nmpi=1, nomp=1, force_run=False, dry_run=False):
        r"""
        Method running all the calculations and post-processing them
        to obtain the polarizability tensor. You may force the
        calculation by setting force_run to True.

        :param nmpi: Number of MPI tasks
        :type nmpi: int
        :param nomp: Number of OpenMP tasks.
        :type nomp: int
        :param force_run: States if the calculation has to be run,
                          even though a logfile already exists.
                          (Optional, default value set to False)
        :type force_run: boolean
        :param dry_run: If True, all folders and input files are written
            on disk, but are not run.
        :type dry_run: bool
        """
        # Loop over the directions of the electric field
        for i, coord in enumerate(COORDS):
            # Add a new key to the dictionary of logfiles
            norm = self.ef_amplitudes[i]
            if norm is not None:
                # Loop over the sign of the electric field amplitude
                for sign in SIGNS:
                    # Set the run folder of the calculation
                    ef_dir = "along_{}{}_{}".format(coord, sign, norm)
                    run_folder = os.path.join(self.run_folder, ef_dir)
                    # Set the electric field in the input file
                    ef = [0.0] * 3
                    ef[i] = SIGNS[sign] * norm
                    new_input = deepcopy(self.input_yaml)
                    new_input['dft']['elecfield'] = ef
                    # Run the calculation
                    bdft = BigDFTCalc(new_input, self.posinp,
                                      prefix=self.prefix,
                                      run_folder=run_folder,
                                      ref_calc=self.ref_calc)
                    bdft.run(nmpi=nmpi, nomp=nomp, force_run=force_run,
                             dry_run=dry_run)
                    # Append the path to the logfile to the dictionary
                    self.logfiles_paths[coord].append(bdft.logfile_path)
        # Use all the logfiles to compute the polarizability tensor
        self.pol_tensor = self.find_polarizability_tensor()

    def find_polarizability_tensor(self):
        r"""
        Function returning the polarizability tensor. It corresponds
        to the response of the system (here, the modification of its
        dipole) when an electric field is applied.

        The dipole and the electric field being vectors, the
        polarizability is represented by a tensor. Its elements
        :math:`alpha_{i, j} = d D_i / d E_j` represent the
        proportionality coefficient between the dipole :math:`D` in the
        direction :math:`i` when an electric field of amplitude
        :math:`E` is applied in the direction :math:`j`. (:math:`i` and
        :math:`j` represent one of the :math:`x`, :math:`y` or :math:`z`
        axis).

        :returns: Polarizability tensor
        :rtype: 2D np.array of dimension 3*3
        """
        pol_tensor = []
        # Loop over the electric field directions
        for i, coord in enumerate(COORDS):
            dipoles = []
            if self.ef_amplitudes[i] is not None:
                # Loop over the electric field calculations in that
                # direction to get their dipoles
                for j in range(2):
                    fname = self.logfiles_paths[coord][j]
                    log = Logfile.from_file(fname)
                    dipoles.append(np.array(
                        log.log['Electric Dipole Moment (AU)']['P vector']))
                # Get the delta of the dipoles
                delta_dipoles = dipoles[0] - dipoles[1]
                # Find the delta of the electric field (which is twice the
                # value of the positive electric field along the direction
                # considered)
                delta_efield = 2 * self.ef_amplitudes[i]
                # Append the new line of the polarizability tensor,
                # defined as the the ratio of the delta of the dipoles and
                # the delta of the electric fields
                pol_tensor.append(delta_dipoles / delta_efield)
            else:
                pol_tensor.append(np.zeros(3))
        # Return the polarizability tensor as a numpy array
        return np.array(pol_tensor)
