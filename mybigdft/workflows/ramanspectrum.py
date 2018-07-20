r"""
File containing the class allowing to compute the Raman Spectrum of a
molecular system with BigDFT.
"""

from __future__ import print_function, absolute_import
import warnings
import os
from collections import Sequence, namedtuple, OrderedDict
import numpy as np
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
        # The displacements define the 6 translation vectors each atom
        # must undergo
        self._displacements = self._init_displacements()
        # The phonon energies are not yet computed
        self._energies = None
        self._dyn_mat = None
        self._normal_modes = None
        # Initialize the queue of jobs for this workflow
        queue = self._initialize_queue()
        super(PhononEnergies, self).__init__(queue=queue)

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
    def energies(self):
        r"""
        Returns
        -------
        numpy.array or None
            Phonon energies of the system.
        """
        return self._energies

    @property
    def dyn_mat(self):
        r"""
        Returns
        -------
        numpy.array or None
            Dynamical matrix deduced from the calculations.
        """
        return self._dyn_mat

    @property
    def normal_modes(self):
        r"""
        Returns
        -------
        numpy.array or None
            Normal modes of the system found as eigenvectors of the
            dynamical matrix.
        """
        return self._normal_modes

    @property
    def displacements(self):
        r"""
        Returns
        -------
        OrderedDict of length 6
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
                run_dir = os.path.join(
                    gs.run_dir, "atom{:04d}".format(i_at), key)
                new_posinp = gs.posinp.translate_atom(i_at, disp.vector)
                job = Job(inputparams=gs.inputparams, posinp=new_posinp,
                          name=gs.name, run_dir=run_dir, skip=gs.skip,
                          ref_job=gs)
                # Add attributes to the job to facilitate post-processing
                job.moved_atom = i_at
                job.displacement = disp
                queue.append(job)
        return queue

    def _init_displacements(self):
        r"""
        Set the six displacements each atom must undergo from the
        amplitudes of displacement in each direction.
        """
        displacements = OrderedDict()
        for i, coord in enumerate(COORDS):
            for sign in SIGNS:
                key = coord + sign
                amplitude = SIGNS[sign] * self.translation_amplitudes[i]
                if self.ground_state.posinp.units == 'angstroem':
                    amplitude *= B_TO_ANG
                displacements[key] = Displacement(i, amplitude)
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
        if self.energies is None:
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
        self._dyn_mat = self._compute_dyn_mat()
        # - Find the energies as eigenvalues of the dynamical matrix
        self._energies = {}
        self._energies['Ha'], self._normal_modes = self._solve_dyn_mat()
        self._energies['cm^-1'] = self.energies['Ha'] * HA_TO_CMM1

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
        atom_types = [atom.type for atom in posinp]
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
        for job1, job2 in zip(*[iter(self.queue)]*2):
            # Get the value of the delta of move amplitudes
            amp1 = job1.displacement.amplitude
            amp2 = job2.displacement.amplitude
            assert amp1 == - amp2
            delta_x = amp1 - amp2
            if self.ground_state.posinp.units == 'angstroem':
                delta_x *= ANG_TO_B
            # Get the value of the delta of forces
            forces1 = job1.logfile.forces.flatten()
            forces2 = job2.logfile.forces.flatten()
            # Set a new line of the Hessian matrix
            new_line = (forces1 - forces2) / delta_x
            h = np.append(h, new_line)
        # Return the Hessian matrix as a symmetric numpy array
        h = h.reshape(3*n_at, 3*n_at)
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
        return vector

    # def __str__(self):
    #     return('Displacement: {}'.format(self.vector))


class RamanSpectrum(AbstractWorkflow):
    r"""
    """

    def __init__(self, phonons, ef_amplitudes=[1e-4]*3):
        r"""
        Parameters
        ----------
        phonons : PhononEnergies
            Phonon energies workflow.
        ef_amplitudes : list or numpy array of length 3
            Amplitude of the electric field to be applied in the three
            directions of space (:math:`x`, :math:`y`, :math:`z`).
        """
        # Check the electric field amplitudes
        if not isinstance(ef_amplitudes, Sequence) or len(ef_amplitudes) != 3:
            raise ValueError("You must provide three electric field "
                             "amplitudes, one for each space coordinate.")
        # Initialize the attributes that are specific to this workflow
        self._phonons = phonons
        self._ef_amplitudes = ef_amplitudes
        # The phonon intensities are not yet computed
        self._intensities = None
        self._depol_ratios = None
        # Initialize the poltensor workflows, no need of a queue
        self._poltensor_workflows = self._init_poltensor_workflows()
        super(RamanSpectrum, self).__init__(queue=[])

    @property
    def phonons(self):
        r"""
        Returns
        -------
        PhononEnergies
            Workflow allowing to compute the phonon energies of the
            system under consideration.
        """
        return self._phonons

    @property
    def ef_amplitudes(self):
        r"""
        Returns
        -------
        list or numpy array of length 3
            Amplitude of the electric field to be applied in the three
            directions of space (:math:`x`, :math:`y`, :math:`z`).
        """
        return self._ef_amplitudes

    @property
    def energies(self):
        r"""
        Returns
        -------
        numpy.array or None
            Phonon energies of the system.
        """
        return self.phonons.energies

    @property
    def intensities(self):
        r"""
        Returns
        -------
        list or None
            Intensities of the phonons.
        """
        return self._intensities

    @property
    def depolarization_ratios(self):
        r"""
        Returns
        -------
        list or None
            Depolarization ratios of the phonons.
        """
        return self._depolarization_ratios

    @property
    def poltensor_workflows(self):
        r"""
        Returns
        -------
        list
            Polarizability tensor workflows to be performed in order to
            compute the Raman intensities.
        """
        return self._poltensor_workflows

    def _init_poltensor_workflows(self):
        r"""
        Returns
        -------
        list
            Polarizability tensor workflows to be performed in order to
            compute the Raman intensities.
        """
        return [PolTensor(job, ef_amplitudes=self.ef_amplitudes)
                for job in self.phonons.queue]

    def run(self, nmpi=1, nomp=1, force_run=False, dry_run=False):
        r"""
        Run the calculations allowing to compute the phonon energies and
        the related intensities in order to be able to plot the Raman
        spectrum of the system under consideration.

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
        if self.intensities is None:
            self.phonons.run(
                nmpi=nmpi, nomp=nomp, force_run=force_run, dry_run=dry_run)
            for pt in self.poltensor_workflows:
                pt.run(
                    nmpi=nmpi, nomp=nomp, force_run=force_run, dry_run=dry_run)
            super(RamanSpectrum, self).run(
                nmpi=nmpi, nomp=nomp, force_run=force_run, dry_run=dry_run)
        else:
            warning_msg = "Calculations already performed; set the argument "\
                          "'force_run' to True to re-run them."
            warnings.warn(warning_msg, UserWarning)

    def post_proc(self):
        r"""
        Compute the intensities of the Raman spectrum set its value (you can
        access its value via the attribute :attr:`poltensor`).
        """
        # - Set the derivatives of the polarizability tensors
        #   along each displacement directions
        deriv_pol_tensors = self._compute_deriv_pol_tensors()
        # - Set the mean polarizability derivatives (alpha), the
        #   anisotropies of the polarizability tensor derivative
        #   (beta_sq), the intensity and the depolarization ratio
        #   for each normal mode:
        self._alphas = []
        self._betas_sq = []
        self._intensities = []
        self._depolarization_ratios = []
        # - Loop over the normal modes
        for pt_flat in deriv_pol_tensors.dot(self.phonons.normal_modes).T:
            # Reshape the derivative of the polarizability tensor
            # along the current normal mode
            pt = pt_flat.reshape(3, 3)
            # Find the principal values of polarizability
            alphas = np.linalg.eigvals(pt)
            # Mean polarizability derivative
            alpha = np.sum(alphas) / 3.
            self._alphas.append(alpha)
            # Anisotropy of the polarizability tensor derivative
            beta_sq = ((alphas[0]-alphas[1])**2 +
                       (alphas[1]-alphas[2])**2 +
                       (alphas[2]-alphas[0])**2) / 2.
            self._betas_sq.append(beta_sq)
            # # Mean polarizability derivative
            # alpha = 1./3. * pt.trace()
            # self._alphas.append(alpha)
            # # Anisotropy of the polarizability tensor derivative
            # beta_sq = 1./2. * ((pt[0][0]-pt[1][1])**2 +
            #                    (pt[0][0]-pt[2][2])**2 +
            #                    (pt[1][1]-pt[2][2])**2 +
            #                    6.*(pt[0][1]**2+pt[0][2]**2+pt[1][2]**2))
            # self._betas_sq.append(beta_sq)
            # From the two previous quantities, it is possible to
            # compute the intensity (converted from atomic units
            # to Ang^4.amu^-1) and the depolarization ratio
            # of the normal mode.
            conversion = B_TO_ANG**4 / EMU_TO_AMU
            self._intensities.append((45*alpha**2 + 7*beta_sq) * conversion)
            self._depolarization_ratios.append(
                3*beta_sq / (45*alpha**2 + 4*beta_sq))

    def _compute_deriv_pol_tensors(self):
        r"""
        Compute the derivative of the polarizability tensor along all
        the atom displacements.

        All the elements of the derivative of the polarizability tensor
        along one displacement direction are represented by a line of
        the returned array. There are :math:`3 n_at` such lines (because
        there are 3 displacements per atom). This representation allows
        for a simpler evaluation of these derivatives along the normal
        modes.

        Note that each element is also weighted by the inverse of the
        square root of the atom that is moved.

        Returns
        -------
        2D np.array of shape :math:`(3 n_{at}, 9)`
            Derivatives of the polarizability tensor.
        """
        deriv_pts = np.array([])
        for pt1, pt2 in zip(*[iter(self.poltensor_workflows)]*2):
            # Get the value of the delta of move amplitudes
            gs1 = pt1.ground_state
            gs2 = pt2.ground_state
            amp = gs1.displacement.amplitude
            assert amp == - gs2.displacement.amplitude
            delta_x = 2 * amp
            if gs1.posinp.units == 'angstroem':
                delta_x *= ANG_TO_B
            # Get the value of the delta of poltensors
            i_at = gs1.moved_atom
            assert i_at == gs2.moved_atom
            m = ATOMS_MASS[gs1.posinp[i_at].type]
            delta_pol_tensor = pt1.poltensor - pt2.poltensor
            # Compute the derivative of the polarizability tensor
            deriv = delta_pol_tensor / delta_x / np.sqrt(m*AMU_TO_EMU)
            deriv_pts = np.append(deriv_pts, deriv.flatten())
        # Return the transpose of this array
        deriv_pts = deriv_pts.reshape(3*len(gs1.posinp), 9)
        return deriv_pts.T
        # # Return the Hessian matrix as a symmetric numpy array
        # h = h.reshape(3*n_at, 3*n_at)
        # # Loop over the atoms
        # for i_at, atom_dir in enumerate(self.atom_dirs):
        #     # Loop over the displacement directions
        #     for i, coord in enumerate(COORDS):
        #         # Get the delta of the polarizability tensors
        #         # corresponding to the two atomic displacements along
        #         # the current direction
        #         m = ATOMS_MASS[self.posinp.atoms[i_at]['Type']]
        #        pts = [self.pol_tensors[atom_dir][coord][j] for j in range(2)]
        #         delta_pol_tensor = (pts[0] - pts[1]) / np.sqrt(m*AMU_TO_EMU)
        #         # Norm of the atom displacement in that direction (in
        #         # bohr)
        #         delta_x = 2 * self.displacements[i]
        #         # Compute the derivative of each component of the
        #         # polarizability tensor for a given direction
        #         deriv = delta_pol_tensor / delta_x
        #         # The polarizability tensor is flatten to make it a
        #         # line of the derivative of the polarizability tensor
        #         # array.
        #         deriv_pol_tensors.append(deriv.flatten())
        # # Convert the array into a numpy array and transpose it before
        # # returning it
        # deriv_pol_tensors = np.array(deriv_pol_tensors).T
        # return deriv_pol_tensors
