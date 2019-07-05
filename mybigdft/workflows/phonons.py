r"""
The :class:`Phonons` class allows to compute the normal modes of a
system.
"""

from __future__ import print_function, absolute_import
import os
from collections import Sequence, namedtuple, OrderedDict
import numpy as np
from mybigdft import Job
from mybigdft.globals import (
    COORDS,
    SIGNS,
    AMU_TO_EMU,
    HA_TO_CMM1,
    B_TO_ANG,
    ANG_TO_B,
    DEFAULT_PARAMETERS,
)
from .workflow import AbstractWorkflow


class Phonons(AbstractWorkflow):
    r"""
    This class allows to run all the calculations enabling the
    computation of the phonon energies of a given system.

    To get the phonon energies of the system, one needs to find the
    eigenvalues of the dynamical matrix, that is closely related to the
    Hessian. To build these matrices, one must find the derivatives of
    the forces when each coordinate of each atom is translated by a
    small amount around the equilibrium positions. This derivative must
    be performed numerically. For a first order evaluation of the
    derivative, :math:`3 n_{at} + 1` DFT calculations must be performed,
    where :math:`n_{at}` is the number of atoms of the system, the 3
    factors comes from the translations along each space coordinate,
    while the extra calculation corresponds to the ground state.
    However, this might not be precise enough because you want the
    ground state forces to be equal to 0, or at least negligible with
    respect to the forces of the out of equilibrium positions. This can
    be difficult to obtain. It is therefore advised to evaluate that
    derivative with a second order scheme, where each coodinate is
    translated positively and negatively, so that the number of BigDFT
    calculations amounts to :math:`2*3*n_{at} = 6 n_{at}` (no
    :math:`+ 1` here, because there is no need to compute the ground
    state anymore).
    """

    POST_PROCESSING_ATTRIBUTES = ["dyn_mat", "energies", "normal_modes"]

    def __init__(self, ground_state, translation_amplitudes=None, order=2):
        r"""
        From a ground state calculation, which must correspond to the
        equilibrium calculation geometry, the :math:`3 n_{at}+1` or
        :math:`6 n_{at}` jobs necessary for the calculation of the
        phonon energies are prepared (depending on the order of the
        calculation).

        The distance of the displacement in each direction is controlled
        by `translation_amplitudes` (one amplitude per space coordinate
        must be given).

        The phonon energies calculations are computed while post-
        processing the results of all the calculations (after running
        the run method). They correspond to the energies of the Raman
        spectrum. If interested in getting the Raman intensity and
        depolarization ratio of each phonon (or normal mode), see
        :class:`~mybigdft.workflows.ramanspectrum.RamanSpectrum`. If
        interested in getting the infrared intensity of each phonon, see
        :class:`~mybigdft.workflows.infreredspectrum.InfraredSpectrum`.

        Parameters
        ----------
        ground_state : Job
            Job of the ground state of the system under consideration.
        translation_amplitudes: Sequence of length 3
            Amplitudes of the translations to be applied to each atom
            along each of the three space coordinates (in atomic units).
        order : int
            Order of the numerical differentiation used to compute the
            dynamical matrix. If second order (resp. first), then six
            (resp. three) calculations per atom are to be performed.
        """
        # Set default translation amplitudes
        if translation_amplitudes is None:
            translation_amplitudes = [0.45 / 64] * 3
        # Check the desired order
        order = int(order)
        if order not in [1, 2]:
            raise NotImplementedError("Only first and second order available")
        # Check the translation amplitudes
        if (
            not isinstance(translation_amplitudes, Sequence)
            or len(translation_amplitudes) != 3
        ):
            raise ValueError(
                "You must provide three electric field "
                "amplitudes, one for each space coordinate."
            )
        translation_amplitudes = [
            amp if amp is not None else 0.0 for amp in translation_amplitudes
        ]
        if 0.0 in translation_amplitudes:
            raise NotImplementedError()
        # Initialize the attributes that are specific to this workflow
        self._ground_state = ground_state
        self._translation_amplitudes = translation_amplitudes
        self._order = order
        # The displacements define the 3 or 6 translation vectors each
        # atom must undergo
        self._displacements = self._init_displacements()
        # Initialize the queue of jobs for this workflow
        queue = self._initialize_queue()
        super(Phonons, self).__init__(queue=queue)

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
    def order(self):
        r"""
        Returns
        -------
        order : int
            Order of the numerical differentiation used to compute the
            dynamical matrix. If second order (resp. first), then six
            (resp. three) calculations per atom are to be performed.
        """
        return self._order

    @property
    def energies(self):
        r"""
        Returns
        -------
        numpy.array or None
            Phonon energies of the system (units: cm^-1).
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
        OrderedDict of length 3 or 6
            Displacements each atom of the system must undergo before
            computing the dynamical matrix as post-processing. There are
            three or six of them (one or two per space coordinate,
            depending on the order of the numerical derivative
            procedure) if the forward or central difference scheme is
            used, respectively.
        """
        return self._displacements

    def _initialize_queue(self):
        r"""
        Initialize the queue of jobs to be run in order to compute the
        phonon energies.
        """
        queue = []
        gs = self.ground_state
        # Add the ground state job to the queue if needed
        if self.order == 1:
            queue.append(gs)
        # Add the jobs where each atom is displaced along each space
        # coordinate
        for i_at in range(len(gs.posinp)):
            for key, disp in self.displacements.items():
                # Prepare the new job by translating an atom
                run_dir = os.path.join(gs.run_dir, "atom{:04d}".format(i_at), key)
                new_posinp = gs.posinp.translate_atom(i_at, disp.vector)
                # Set the correct reference data directory
                default = DEFAULT_PARAMETERS["output"]["orbitals"]
                write_orbitals = (
                    "output" in gs.inputparams and gs.inputparams["output"] != default
                )
                if self.order == 1 and write_orbitals:
                    ref_data_dir = gs.data_dir  # pragma: no cover
                else:
                    ref_data_dir = gs.ref_data_dir
                job = Job(
                    inputparams=gs.inputparams,
                    posinp=new_posinp,
                    name=gs.name,
                    run_dir=run_dir,
                    skip=gs.skip,
                    ref_data_dir=ref_data_dir,
                    pseudos=gs.pseudos,
                )
                # Add attributes to the job to facilitate post-processing
                job.moved_atom = i_at
                job.displacement = disp
                queue.append(job)
        return queue

    def _init_displacements(self):
        r"""
        Set the displacements each atom must undergo from the amplitudes
        of displacement in each direction.
        """
        displacements = OrderedDict()
        if self.order == 1:
            signs = {"+": 1.0}  # One displacement per coordinate
        elif self.order == 2:
            signs = SIGNS  # Two displacements per coordinate
        for i, coord in enumerate(COORDS):
            for sign in signs:
                key = coord + sign
                amplitude = signs[sign] * self.translation_amplitudes[i]
                if self.ground_state.posinp.units == "angstroem":
                    amplitude *= B_TO_ANG
                displacements[key] = Displacement(i, amplitude)
        return displacements

    def post_proc(self):
        r"""
        Run the post-processing of the calculations, here:

        * compute the dynamical matrix,
        * solve it to get the phonons (normal modes) and their energies.
        """
        # - Set the dynamical matrix
        self._dyn_mat = self._compute_dyn_mat()
        # - Find the energies as eigenvalues of the dynamical matrix
        self._energies, self._normal_modes = self._solve_dyn_mat()
        self._energies *= HA_TO_CMM1  # Convert from Hartree to cm^-1

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
        hessian = self._compute_hessian()
        masses = self._compute_masses()
        return hessian / masses

    def _compute_masses(self):
        r"""
        Compute the masses array used to define the dynamical matrix.
        The masses are counted in electronic mass units (which is the
        atomic unit of mass, that is different from the atomic mass
        unit).

        Returns
        -------
        2D square numpy array of dimension :math:`3 n_{at}`
            Masses matrix converted to electronic mass units.
        """
        posinp = self.ground_state.posinp
        to_mesh = [atom.mass for atom in posinp for _ in range(3)]
        m_i, m_j = np.meshgrid(to_mesh, to_mesh)
        return np.sqrt(m_i * m_j) * AMU_TO_EMU

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
        gs = self.ground_state
        pos = gs.posinp
        n_at = len(pos)
        hessian = np.zeros((3 * n_at, 3 * n_at))
        if self.order == 1:
            # Compute the first order matrix elements
            gs_forces = gs.logfile.forces.flatten()
            for i, job in enumerate(self.queue[1:]):
                # Get the value of the delta of move amplitudes
                delta_x = job.displacement.amplitude
                # Get the value of the delta of forces
                forces = job.logfile.forces.flatten()
                # Set a new line of the Hessian matrix
                hessian[i] = (forces - gs_forces) / delta_x
        elif self.order == 2:
            # Compute the second order matrix elements
            # for i, job1, job2 in enue(zip(self.queue[::2], self.queue[1::2]))
            for i, (job1, job2) in enumerate(zip(*[iter(self.queue)] * 2)):
                # Get the value of the delta of move amplitudes
                amp1 = job1.displacement.amplitude
                amp2 = job2.displacement.amplitude
                assert amp1 == -amp2
                delta_x = amp1 - amp2
                # Get the value of the delta of forces
                forces1 = job1.logfile.forces.flatten()
                forces2 = job2.logfile.forces.flatten()
                # Set a new line of the Hessian matrix
                hessian[i] = (forces1 - forces2) / delta_x
        # Convert to atomic units if needed
        if pos.units == "angstroem":
            hessian /= ANG_TO_B
        # Return the Hessian matrix as a symmetric numpy array
        return -(hessian + hessian.T) / 2.0

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
        # eigs actually gives the square of the expected eigenvalues.
        # Given they can be negative, enforce a positive value with
        # np.where() before taking the signed square-root
        eigs = np.sign(eigs) * np.sqrt(np.where(eigs < 0, -eigs, eigs))
        return eigs, vecs


class Displacement(namedtuple("Displacement", ["i_coord", "amplitude"])):
    r"""
    This class defines an atomic displacement from the coordinate index
    and the amplitude of the displacement in that direction.
    """

    __slots__ = ()

    @property
    def vector(self):
        r"""
        Returns
        -------
        list
            Displacement vector.
        """
        vector = [0.0] * 3
        vector[self.i_coord] = self.amplitude
        return vector
