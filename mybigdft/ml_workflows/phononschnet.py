r"""
The :class:`Phononschnet` class allows to compute the normal vibration
modes of a system using SchetPack trained models.
"""

from __future__ import absolute_import
import numpy as np
from mybigdft import Jobschnet, Posinp
from mybigdft.ml_workflows import Geoptschnet
from copy import deepcopy
from mybigdft.globals import ANG_TO_B, B_TO_ANG, EV_TO_HA, HA_TO_CMM1, AMU_TO_EMU


class Phononschnet:
    r"""
    This class allows to run all the calculations enabling the
    computation of the phonon energies of a given system, using
    machine learning models obtained with the SchnetPack package.

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

    def __init__(self, init_state, relax=True, translation_amplitudes=None, order=3):
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
        inti_state : mybigdft.Posinp
            Initial positions of the system under consideration.
        translation_amplitudes: Sequence of length 3
            Amplitudes of the translations to be applied to each atom
            along each of the three space coordinates (in angstroms).
        order : int
            Order of the numerical differentiation used to compute the
            dynamical matrix. If second order (resp. first), then six
            (resp. three) calculations per atom are to be performed.
        """
        self.init_state = init_state
        self.relax = relax
        self.translation_amplitudes = translation_amplitudes
        self.order = order
        self.dyn_mat = None
        self.energies = None
        self.normal_modes = None
        self.ground_state = None

    @property
    def init_state(self):
        r"""
        Returns
        -------
        init_state : mybigdft.Posinp
            Initial positions of the system for which phonon properties
            will be calculated.
        """
        return self._init_state

    @init_state.setter
    def init_state(self, init_state):
        if isinstance(init_state, Posinp):
            self._init_state = init_state
        else:
            raise TypeError(
                "Initial positions should be given in a mybigdft.Posinp instance."
            )

    @property
    def ground_state(self):
        r"""
        Returns
        -------
        ground_state : mybigdft.Posinp
            Relaxed positions of the system for which phonon properties
            will be calculated. If `relax`is `False`, will be the same
            as init_state.
        """
        return self._ground_state

    @ground_state.setter
    def ground_state(self, ground_state):
        self._ground_state = ground_state

    @property
    def translation_amplitudes(self):
        r"""
        Returns
        -------
        translation_amplitudes : list of 3 floats
            Displacements of atoms in all three dimensions to calculate
            the phonon properties
        """
        return self._translation_amplitudes

    @translation_amplitudes.setter
    def translation_amplitudes(self, translation_amplitudes):
        if translation_amplitudes == None:
            self._translation_amplitudes = [0.015] * 3
        else:
            assert (
                len(translation_amplitudes) == 3
            ), "Translation amplitudes should be given in a list of length 3"
            self._translation_amplitudes = translation_amplitudes

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

    @order.setter
    def order(self, order):
        order = int(order)
        if order not in [1, 2, 3]:
            raise NotImplementedError("Only first and second order available.")
        self._order = order

    @property
    def relax(self):
        r"""
        Returns
        -------
        relax : bool
            If `True`, which is default, the initial positions are relaxed
            before the phonon properties are calculated. Recommended,
            especially if more than one model is used.
        """
        return self._relax

    @relax.setter
    def relax(self, relax):
        relax = bool(relax)
        self._relax = relax

    @property
    def energies(self):
        r"""
        Returns
        -------
        numpy.array or None
            Phonon energies of the system (units: cm^-1).
        """
        return self._energies

    @energies.setter
    def energies(self, energies):
        self._energies = energies

    @property
    def dyn_mat(self):
        r"""
        Returns
        -------
        numpy.array or None
            Dynamical matrix deduced from the calculations.
        """
        return self._dyn_mat

    @dyn_mat.setter
    def dyn_mat(self, dyn_mat):
        self._dyn_mat = dyn_mat

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

    @normal_modes.setter
    def normal_modes(self, normal_modes):
        self._normal_modes = normal_modes

    def run(self, model_dir, device="cpu", batch_size=128, recenter=False, **kwargs):
        r"""
        Parameters
        ----------
        model_dir : str
            Path to the model used for calculations relative to `$MODELDIR`
            (absolute path if not defined).
        device : str
            Either "cpu" or "cuda" to run on cpu or gpu.
        **kwargs : 
            Optional arguments for the geometry optimization.
            Only useful if the relaxation is unstable.
        """
        if self.relax:
            geopt = Geoptschnet(posinp=self.init_state, write_to_disk=False, **kwargs)
            geopt.run(
                model_dir=model_dir,
                device=device,
                batch_size=batch_size,
                recenter=recenter,
            )
            self.ground_state = deepcopy(geopt.final_posinp)
        else:
            self.ground_state = deepcopy(self.init_state)
        job = Jobschnet(posinp=self._create_displacements())
        job.run(
            model_dir=model_dir,
            forces=2,
            device=device,
            write_to_disk=False,
            batch_size=batch_size,
            overwrite=False,
        )
        self._post_proc(job)

    def _create_displacements(self):
        r"""
        Set the displacements each atom must undergo from the amplitudes
        of displacement in each direction.
        """
        structs = []
        # First order phonon calculation
        if self.order == 1:
            structs.append(deepcopy(self.ground_state))
            for i, atom in enumerate(self.ground_state):
                for j, dim in zip(
                    range(3),
                    [np.array([1, 0, 0]), np.array([0, 1, 0]), np.array([0, 0, 1])],
                ):
                    structs.append(
                        self.ground_state.translate_atom(
                            i, self.translation_amplitudes[j] * dim
                        )
                    )
        # Second order phonon calculation
        elif self.order == 2:
            for i, atom in enumerate(self.ground_state):
                for j, dim in zip(
                    [0, 1, 2],
                    [np.array([1, 0, 0]), np.array([0, 1, 0]), np.array([0, 0, 1])],
                ):
                    for factor in [1, -1]:
                        structs.append(
                            self.ground_state.translate_atom(
                                i, self.translation_amplitudes[j] * dim * factor
                            )
                        )
        # Third order phonon calculation
        elif self.order == 3:
            for i in range(len(self.ground_state)):
                for j, dim in zip(
                    [0, 1, 2],
                    [np.array([1, 0, 0]), np.array([0, 1, 0]), np.array([0, 0, 1])],
                ):
                    for factor in [2, 1, -1, -2]:
                        structs.append(
                            self.ground_state.translate_atom(
                                i, self.translation_amplitudes[j] * dim * factor
                            )
                        )
        return structs

    def _post_proc(self, job):
        r"""
        """
        self.dyn_mat = self._compute_dyn_mat(job)
        self.energies, self.normal_modes = self._solve_dyn_mat()
        self.energies *= HA_TO_CMM1

    def _compute_dyn_mat(self, job):
        r"""
        """
        hessian = self._compute_hessian(job)
        masses = self._compute_masses()
        return hessian / masses

    def _compute_masses(self):
        r"""
        """
        to_mesh = [atom.mass for atom in self.ground_state for _ in range(3)]
        m_i, m_j = np.meshgrid(to_mesh, to_mesh)
        return np.sqrt(m_i * m_j) * AMU_TO_EMU

    def _compute_hessian(self, job):
        r"""
        """
        pos = self.ground_state
        n_at = len(pos)
        hessian = np.zeros((3 * n_at, 3 * n_at))
        forces = np.array(job.logfile.forces) * EV_TO_HA * B_TO_ANG
        if self.order == 1:
            for i in range(3 * n_at):
                hessian[i, :] = (forces[i + 1].flatten() - forces[0].flatten()) / (
                    self.translation_amplitudes[i % 3] * ANG_TO_B
                )
        elif self.order == 2:
            for i in range(3 * n_at):
                hessian[i, :] = (
                    forces[2 * i].flatten() - forces[2 * i + 1].flatten()
                ) / (2 * self.translation_amplitudes[i % 3] * ANG_TO_B)
        elif self.order == 3:
            for i in range(3 * n_at):
                hessian[i, :] = (
                    -forces[2 * i].flatten()
                    + 8 * forces[2 * i + 1].flatten()
                    - 8 * forces[2 * i + 2].flatten()
                    + forces[2 * i + 3].flatten()
                ) / (12 * self.translation_amplitudes[i % 3] * ANG_TO_B)
        return -(hessian + hessian.T) / 2.0

    def _solve_dyn_mat(self):
        r"""
        """
        eigs, vecs = np.linalg.eig(self.dyn_mat)
        eigs = np.sign(eigs) * np.sqrt(np.where(eigs < 0, -eigs, eigs))
        return eigs, vecs
