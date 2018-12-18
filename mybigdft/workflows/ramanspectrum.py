r"""
The :class:`RamanSpectrum` class allows to compute the normal modes
and their respective Raman intensities, allowing one to study the Raman
spectrum.
"""

from __future__ import print_function, absolute_import
import numpy as np
from mybigdft.globals import AMU_TO_EMU, EMU_TO_AMU, B_TO_ANG, ANG_TO_B
from .workflow import AbstractWorkflow
from .poltensor import PolTensor


class RamanSpectrum(AbstractWorkflow):
    r"""
    This class allows to run all the calculations enabling the
    computation of the Raman spectrum of a given system, that is its
    normal modes of vibration (or phonons) associated to a given energy
    and a Raman intensity. The so-called depolarization ratios of each
    phonon mode are also be computed.

    One therefore needs to compute the phonons first. This is done by
    solving the dynamical matrix (the eigenvalues giving the phonon
    energies and the eigenvectors the normal modes). This matrix is
    computed at the expense of :math:`6 n_{at}` BigDFT calculations,
    where each atom is in turns translated by a small amount around its
    equilibrium positions. You may want to refer to the
    :class:`~mybigdft.workflows.phonons.Phonons` class for more details.

    To get the Raman intensities (or activities) of the spectrum, one
    must compute the derivative of the polarizability tensor along the
    normal modes. To that end, one must compute the polarizability
    tensor at each of the positons used to get the vibrational energies,
    and this means applying an external electric field along each
    coordinate. One calculation per space coordinate lead to 3 extra
    calculations, meaning that :math:`18 n_{at}` additional BigDFT
    standard calculations are required to obtain a Raman spectrum
    intensities, leading to :math:`24 n_{at}` calculations in total.
    """

    POST_PROCESSING_ATTRIBUTES = ["intensities", "depolarization_ratios"]

    def __init__(self, phonons, ef_amplitudes=None, order=1):
        r"""
        From a phonon calculation, one is able to compute the Raman
        spectrum of a given system by only specifying the electric field
        amplitudes used to compute the polarizability tensor at each
        out-of-equilibrium positions used to compute the phonons.

        Parameters
        ----------
        phonons : Phonons
            Phonon energies workflow.
        ef_amplitudes : list or numpy array of length 3
            Amplitude of the electric field to be applied in the three
            directions of space (:math:`x`, :math:`y`, :math:`z`).
        order : int
            Order of the numerical differentiation used to compute the
            polarizability tensors that are then used to compute the
            Raman intensities. If second (resp. first) order, then six
            (resp. three) calculations per atom are to be performed.
        """
        # Initialize the attributes that are specific to this workflow
        self._phonons = phonons
        # Some other quantities are not yet computed
        self._alphas = None  # mean polarizability derivatives
        self._betas_sq = None  # anisotropies of pol. tensor deriv.
        # Initialize the poltensor workflows to run
        self._poltensor_workflows = [
            PolTensor(job, ef_amplitudes=ef_amplitudes, order=order)
            for job in self.phonons.queue
        ]
        super(RamanSpectrum, self).__init__(queue=[])

    @property
    def phonons(self):
        r"""
        Returns
        -------
        Phonons
            Workflow allowing to compute the phonon energies of the
            system under consideration.
        """
        return self._phonons

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

    def _run(self, nmpi, nomp, force_run, dry_run, restart_if_incomplete):
        r"""
        Run the calculations allowing to compute the phonon energies and
        the related Raman intensities in order to be able to plot the
        Raman spectrum of the system under consideration.

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
        restart_if_incomplete : bool
            If `True`, the job is restarted if the existing logfile is
            incomplete.
        """
        self.phonons.run(
            nmpi=nmpi, nomp=nomp, force_run=force_run, dry_run=dry_run,
            restart_if_incomplete=restart_if_incomplete)
        for pt in self.poltensor_workflows:
            pt.run(
                nmpi=nmpi, nomp=nomp, force_run=force_run, dry_run=dry_run,
                restart_if_incomplete=restart_if_incomplete)
        super(RamanSpectrum, self)._run(
            nmpi, nomp, force_run, dry_run,
            restart_if_incomplete=restart_if_incomplete)

    def post_proc(self):
        r"""
        Compute the Raman intensities and depolarization ratio of each
        normal mode and set their values (you can access their values
        via the attributes :attr:`intensities` and
        :attr:`depolarization_ratios`, respectively).
        """
        # - Set the derivatives of the polarizability tensors
        #   along each displacement directions
        deriv_pol_tensors = self._compute_deriv_pol_tensors()
        # - Loop over the normal modes to get the the mean
        #   polarizability derivative (alphas) and of the anisotropy of
        #   the polarizability tensor derivative (betas_sq)
        alphas = []
        betas_sq = []
        for pt in deriv_pol_tensors.dot(self.phonons.normal_modes).T:
            alphas.append(pt.trace() / 3.)
            evals = np.linalg.eigvals(pt)
            beta_sq = ((evals[0]-evals[1])**2 +
                       (evals[1]-evals[2])**2 +
                       (evals[2]-evals[0])**2) / 2.
            betas_sq.append(beta_sq)
            # beta_sq = 1./2. * ((pt[0][0]-pt[1][1])**2 +
            #                    (pt[0][0]-pt[2][2])**2 +
            #                    (pt[1][1]-pt[2][2])**2 +
            #                    6.*(pt[0][1]**2+pt[0][2]**2+pt[1][2]**2))
            # betas_sq.append(beta_sq)
        self._alphas = np.array(alphas)
        self._betas_sq = np.array(betas_sq)
        # From the two previous quantities, it is possible to
        # compute the intensity (converted from atomic units
        # to Ang^4.amu^-1) and the depolarization ratio
        # of the normal mode.
        conversion = B_TO_ANG**4 / EMU_TO_AMU
        self._intensities = \
            (45*self._alphas**2 + 7*self._betas_sq) * conversion
        self._depolarization_ratios = \
            3*self._betas_sq / (45*self._alphas**2 + 4*self._betas_sq)

    def _compute_deriv_pol_tensors(self):
        r"""
        Compute the derivative of the polarizability tensor along all
        the atomic displacements.

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
        n_at = len(self.phonons.ground_state.posinp)
        deriv_pts = np.zeros((3*n_at, 3, 3))
        pt_wfs = self.poltensor_workflows
        if self.phonons.order == 1:
            ref_pt = pt_wfs.pop(0)
            pol_tensors = (pt_wfs, [ref_pt]*len(pt_wfs))
        elif self.phonons.order == 2:
            pol_tensors = (pt_wfs[::2], pt_wfs[1::2])
        else:
            raise NotImplementedError
        for i, (pt1, pt2) in enumerate(zip(*pol_tensors)):
            # Get the value of the delta of move amplitudes
            gs1 = pt1.ground_state
            amp = gs1.displacement.amplitude
            if self.phonons.order == 1:
                delta_x = amp
            elif self.phonons.order == 2:
                delta_x = 2 * amp
            if gs1.posinp.units == 'angstroem':
                delta_x *= ANG_TO_B
            # Get the value of the delta of poltensors
            i_at = gs1.moved_atom
            delta_pol_tensor = pt1.pol_tensor - pt2.pol_tensor
            # Compute the derivative of the polarizability tensor
            mass = gs1.posinp[i_at].mass * AMU_TO_EMU
            deriv_pts[i] = delta_pol_tensor / delta_x / np.sqrt(mass)
        return deriv_pts.T
