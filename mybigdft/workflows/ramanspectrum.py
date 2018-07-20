r"""
File containing the class allowing to compute the Raman Spectrum of a
molecular system with BigDFT.
"""

from __future__ import print_function, absolute_import
import warnings
# from copy import deepcopy
import os
from collections import Sequence, namedtuple
import numpy as np
# from mybigdft import Logfile, Job
from mybigdft import Job
from mybigdft.job import COORDS, SIGNS
from .workflow import AbstractWorkflow
from .poltensor import PolTensor


# Mass of the different types of atoms in atomic mass units
# TODO: Add more types of atoms
#       (found in $SRC_DIR/bigdft/src/orbitals/eleconf-inc.f90)
ATOMS_MASS = {"H": 1.00794, "He": 4.002602, "Li": 6.941, "Be": 9.012182,
              "B": 10.811, "C": 12.011, "N": 14.00674, "O": 15.9994,
              "F": 18.9984032, "Ne": 20.1797}
# Conversion from atomic to electronic mass unit
AMU_TO_EMU = 1.660538782e-27 / 9.10938215e-31
# Conversion from electronic to atomic mass unit
EMU_TO_AMU = 1. / AMU_TO_EMU
# Conversion factor from bohr to angstroem
B_TO_ANG = 0.529177249
# Conversion factor from angstroem to bohr
ANG_TO_B = 1. / B_TO_ANG
# Conversion factor from Hartree to cm^-1
HA_TO_CMM1 = 219474.6313705


class PhononEnergies(AbstractWorkflow):
    r"""
    This class allows to run all the calculations enabling the
    computation of the phonon energies of a given system.

    To get the phonon energies of the system, one needs to find the
    eigenvalues of the dynamical matrix, that is closely related to the
    Hessian. To build these matrices, one must find the derivatives of
    the forces when each coordinate of each atom is translated by a
    small amount around the equilibrium positions. To get a better
    precision on the derivative, each coodinate is translated
    positively and negatively, so that the number of BigDFT calculations
    amounts to :math:`2*3*n_{at} = 6 n_{at}`, where :math:`n_{at}` is
    the number of atoms (3 for the coordinates (:math:`x`, :math:`y` and
    :math:`z`), 2 for the number of calculations per coordinates).
    """

    def __init__(self, ground_state, translation_amplitudes=[0.45/64]*3):
        r"""
        From a ground state calculation, which must correspond to the
        equilibrium calculation geometry, the :math:`6 n_{at}` jobs
        necessary for the calculation of the phonon energies are
        prepared.

        The distance of the displacement in each direction is controlled
        by `translation_amplitudes` (one amplitude per space coordinate
        must be given).

        The phonon energies calculations are computed while post-
        processing the results of all the calculations (after running
        the run method). They correspond to the energies of the Raman
        spectrum. If interested in getting the intensity and
        depolarization ratio of each phonon (or normal mode), see
        :class:`RamanSpectrum`.

        Parameters
        ----------
        translation_amplitudes: Sequence of length 3
            Amplitudes of the translations to be applied to each atom
            along each of the three space coordinates.
        """
        # Check the translation amplitudes
        if not isinstance(translation_amplitudes, Sequence) or \
                len(translation_amplitudes) != 3:
            raise ValueError("You must provide three electric field "
                             "amplitudes, one for each space coordinate.")
        translation_amplitudes = [amp if amp is not None else 0.0
                                  for amp in translation_amplitudes]
        if 0.0 in translation_amplitudes:
            raise NotImplementedError()
        # Initialize the attributes that are specific to this workflow
        self._ground_state = ground_state
        self._translation_amplitudes = translation_amplitudes
        # Initialize some attributes containing relevant data
        self._displacements = self._set_displacements()
        # The phonon energies are not yet computed
        self._phonon_energies = None
        # Initialize the queue of jobs for this workflow
        queue = self._initialize_queue()
        super(PolTensor, self).__init__(queue=queue)

    @property
    def ground_state(self):
        r"""
        Returns
        -------
        Job
            Job of the ground state of the system under consideration.
        """
        return self._ground_state

    @property
    def translation_amplitudes(self):
        r"""
        Returns
        -------
        Sequence of length 3
            Amplitudes of the translations to be applied to each atom
            along each of the three space coordinates.
        """
        return self._translation_amplitudes

    @property
    def phonon_energies(self):
        r"""
        Returns
        -------
        numpy.array or None
            Phonon energies of the system.
        """
        self._phonon_energies

    @property
    def displacements(self):
        r"""
        Returns
        -------
        dict of length 6
            Displacements each atom of the system must undergo. There
            are six of them (two per space coordinate) in order to be
            able to compute the Hessian matrix by using the central
            difference scheme.
        """
        return self._displacements

    def _initialize_queue(self):
        r"""
        Initialize the queue of jobs to be run in order to compute the
        phonon energies.
        """
        queue = []
        gs = self.ground_state
        # # Add the ground state job to the queue after updating the run
        # # directory if needed
        # queue.append(gs)
        # Add the jobs where each atom is displaced along each space
        # coordinate
        for i_at in range(len(gs.posinp)):
            for key, disp in self.displacements.items():
                # Prepare the new job by translating an atom
                run_dir = os.path.join("atom{:04d}".format(i_at), key)
                new_posinp = gs.posinp.translate_atom(i_at, disp.vector)
                job = Job(inputparams=gs.inputparams, posinp=new_posinp,
                          name=gs.name, run_dir=run_dir, skip=gs.skip,
                          ref_job=gs)
                # Add attributes to the job to facilitate post-processing
                job.moved_atom = i_at
                job.displacement = disp
                self.queue.append(job)
        return queue

    def _set_displacements(self):
        r"""
        Set the six displacements each atom must undergo from the
        amplitudes of displacement in each direction.
        """
        displacements = {}
        for i, coord in enumerate(COORDS):
            for sign in SIGNS:
                key = coord + sign
                amplitude = SIGNS[sign] * self.translation_amplitudes[i]
                displacements[key] = Displacement(i, amplitude, sign)
        return displacements

    def run(self, nmpi=1, nomp=1, force_run=False, dry_run=False):
        r"""
        Run the calculations allowing to compute the phonon energies,
        which are computed at the post-processing level.

        Parameters
        ----------
        nmpi : int
            Number of MPI tasks.
        nomp : int
            Number of OpenMP tasks.
        force_run : bool
            If `True`, the calculations are run even though a logfile
            already exists.
        dry_run : bool
            If `True`, the input files are written on disk, but the
            bigdft-tool command is run instead of the bigdft one.
        """
        if self.phonon_energies is None:
            super(PhononEnergies, self).run(
                nmpi=nmpi, nomp=nomp, force_run=force_run, dry_run=dry_run)
        else:
            warning_msg = "Calculations already performed; set the argument "\
                          "'force_run' to True to re-run them."
            warnings.warn(warning_msg, UserWarning)

    def post_proc(self):
        r"""
        Run the post-processing of the calculations, here:

        * compute the dynamical matrix,
        * solve it to get the phonons (normal modes) and their energies.
        """
        # - Set the dynamical matrix
        self.dyn_mat = self._compute_dyn_mat()
        # - Find the energies as eigenvalues of the dynamical matrix
        self.energies = {}
        self.energies['Ha'], self.normal_modes = self._solve_dyn_mat()
        self.energies['cm^-1'] = self.energies['Ha'] * HA_TO_CMM1

    def _compute_dyn_mat(self):
        r"""
        Compute the dynamical matrix of the system. It is very similar
        to the Hessian matrix: its elements are only corrected by a
        weight :math:`w,` which is the inverse of the square-root of the
        product of the atomic masses of the atoms involved in the
        Hessian matrix element.

        The masses are counted in electronic mass units (which is the
        atomic unit of mass, that is different from the atomic mass
        unit).

        Returns
        -------
        2D square numpy array of dimension :math:`3 n_{at}`
            Dynamical matrix.
        """
        # Numpy does the ratio of arrays intelligently: by making masses
        # an array of the same size as the Hessian, there is nothing but
        # the ratio of both arrays to perform to get the dynamical
        # matrix.
        h = self._compute_hessian()
        masses = self._compute_masses()
        return h / masses

    def _compute_masses(self):
        r"""
        Compute the masses array used to define the dynamical matrix.
        The masses are counted in electronic mass units (which is the
        atomic unit of mass, that is different from the atomic mass
        unit).

        Returns
        -------
        2D square numpy array of dimension :math:`3 n_{at}`
            Masses matrix.
        """
        # Get the atoms of the system from the reference posinp
        posinp = self.ground_state.posinp
        atom_types = [atom.keys()[0] for atom in posinp]
        # Build the masses matrix (the loops over range(3) are here
        # to ensure that masses has the same dimension as the Hessian)
        masses = [[ATOMS_MASS[atom1] * ATOMS_MASS[atom2]
                   for atom2 in atom_types for j in range(3)]
                  for atom1 in atom_types for i in range(3)]
        # Return the masses as a numpy array, converted in electronic
        # mass units
        return np.sqrt(masses) * AMU_TO_EMU

    def _compute_hessian(self):
        r"""
        Compute the Hessian of the system. Its dimension is
        :math:`3 n_{at}`, where :math:`n_{at}` is the number of atoms of
        the system.

        Returns
        -------
        2D square numpy array of dimension :math:`3 n_{at}`
            Hessian matrix.
        """
        # Initialization of variables
        gs = self.ground_state
        n_at = len(gs.posinp)
        h = np.array([])  # Hessian matrix
        # First loop over all atoms
        for i_at in range(n_at):
            for job1, job2 in zip(*[iter(self.queue)]*2):
                # The Hessian is made of the delta of the forces
                # with respect to the delta of the move distances.
                new_line = []
                # 1- Find the delta displacement. It is twice the
                #    distance of the positive move along the direction
                #    of the displacement.
                amp1 = job1.displacement.amplitude
                amp2 = job2.displacement.amplitude
                assert amp1 == - amp2
                delta_x = amp1 - amp2
                # # Make sure there is no div. by 0
                # amplitude = self.amplitudes[i_coord]
                # if amplitude is None or amplitude == 0.0:
                #     amplitude = 1.0
                # # The Hessian is then built line by line:
                # # 1- Find the delta displacement. It is twice the
                # #    distance of the positive move along the direction
                # #    of the displacement.
                # delta_x = 2 * amplitude
                # 2- Find the delta forces for each atom and update
                #    the new line of the Hessian.
                forces1 = job1.logfile.forces.flatten()
                forces2 = job2.logfile.forces.flatten()
                new_line = (forces1 - forces2) / delta_x
                h = np.append(h, new_line)
                # for j_at in range(n_at):
                #     delta_forces = forces1[j_at] - forces2[j_at]
                #     new_line += list(delta_forces/delta_x)
                # # The new line of the Hessian is now complete
                # h.append(new_line)
        # Return the Hessian matrix as a numpy array
        h = h.reshape(3*n_at, 3*n_at)
        # h = np.array(h)
        return (h + h.T) / 2.

    def _solve_dyn_mat(self):
        r"""
        Solve the dynamical matrix to get the phonon energies (converted
        in Hartree) and the eigenvectors.

        Returns
        -------
        tuple
            Tuple made of the eigenvalues (as an array) and the
            eigenvectors (as a matrix).
        """
        eigs, vecs = np.linalg.eig(self.dyn_mat)
        eigs = np.array([np.sqrt(-e) if e < 0 else np.sqrt(e) for e in eigs])
        return eigs, vecs


class Displacement(namedtuple('Displacement', ['i_coord', 'amplitude'])):
    r"""
    """

    __slots__ = ()

    @property
    def vector(self):
        vector = [0.0] * 3
        vector[self.i_coord] = self.amplitude
        # if amplitude is not None and amplitude != 0.:
        #     vector = [0.0] * 3
        #     vector[i] = SIGNS[sign] * amplitude
        # else:
        #     raise NotImplementedError()
        return vector

    def __str__(self):
        return('Displacement: {}'.format(self.vector))


class RamanSpectrum(AbstractWorkflow):
    r"""
    """

    pass
