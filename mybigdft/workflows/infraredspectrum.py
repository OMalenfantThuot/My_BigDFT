r"""
The :class:`InfraredSpectrum` class allows to compute the normal modes
and their respective infrared intensities, allowing one to study the
infrared spectrum.
"""

import numpy as np
from mybigdft.globals import ANG_TO_B, B_TO_ANG, AU_TO_DEBYE
from .workflow import AbstractWorkflow


class InfraredSpectrum(AbstractWorkflow):
    r"""
    This class allows to run all the calculations enabling the
    computation of the infrared spectrum of a given system, that is its
    normal modes of vibration (or phonons) associated to a given energy
    and an infrared intensity.

    One therefore needs to compute the phonons first. This is done by
    solving the dynamical matrix (the eigenvalues giving the phonon
    energies and the eigenvectors the normal modes). This matrix is
    computed at the expense of :math:`6 n_{at}` BigDFT calculations,
    where each atom is in turns translated by a small amount around its
    equilibrium positions. You may want to refer to the
    :class:`~mybigdft.workflows.phonons.Phonons` class for more details.

    To get the intensities of the infrared spectrum, one must compute
    the derivative of the dipole moment along the normal modes. All the
    necessary dipole moments to use are readily found in the logfiles of
    the BigDFT calculations performed to compute the phonons: no extra
    calculation is required.
    """

    POST_PROCESSING_ATTRIBUTES = ["intensities", "Z", "Zbvs"]

    def __init__(self, phonons):
        r"""
        From a phonon calculation, one is able to compute the infrared
        spectrum of a given system by only measuring the dipole moment
        at each out-of-equilibrium positions used to compute the
        phonons.

        Parameters
        ----------
        phonons : Phonons
            Phonon energies workflow.
        """
        self._phonons = phonons
        super(InfraredSpectrum, self).__init__(queue=[])

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
            Phonon energies of the system (units: cm^-1).
        """
        return self.phonons.energies

    @property
    def intensities(self):
        r"""
        Returns
        -------
        list or None
            Infrared intensities of the phonons (units:
            (D/Ang)^2.amu^-1).
        """
        return self._intensities

    @property
    def Z(self):
        r"""
        Returns
        -------
        numpy array of dimension :math:`3 * 3 n_{at}`:
            Matrix measuring the derivative of the dipole moment with
            respect to atomic displacements (units: (D/A).amu^-1/2).
        """
        return self._Z

    @property
    def Zbvs(self):
        r"""
        Returns
        -------
        list or None
            Intensities of the phonons.
        """
        return self._Zbvs

    def _run(self, nmpi, nomp, force_run, dry_run, restart_if_incomplete,
             timeout):
        r"""
        Run the calculations allowing to compute the phonon energies and
        the related infrared intensities in order to be able to plot the
        infrared spectrum of the system under consideration.

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
        timeout : float or int or None
            Number of minutes after which each job must be stopped.
        """
        self.phonons.run(
            nmpi=nmpi, nomp=nomp, force_run=force_run, dry_run=dry_run,
            restart_if_incomplete=restart_if_incomplete, timeout=timeout)
        super(InfraredSpectrum, self)._run(
            nmpi, nomp, force_run, dry_run, restart_if_incomplete, timeout)

    def post_proc(self):
        r"""
        Compute the infrared intensities, the Z matrix and the Z
        matrices for each normal mode and set their values (you can
        access their values via the attributes :attr:`intensities`,
        :attr:`Z` and :attr:`Zbvs`, respectively).
        """
        self._Z = self._compute_Z_matrix()
        nms = self.phonons.normal_modes.T
        self._Zbvs = [self.Z*nm for nm in nms]
        self._intensities = np.array([np.sum(np.sum(Zbv, axis=1)**2)
                                      for Zbv in self.Zbvs])

    def _compute_Z_matrix(self):
        r"""
        Returns
        -------
        numpy array of dimension :math:`3 * 3 n_{at}`:
            Matrix measuring the derivative of the dipole moment with
            respect to atomic displacements (units: (D/A).amu^-1/2).
        """
        posinp = self.phonons.ground_state.posinp
        n_at = len(posinp)
        # Compute the derivatives of the dipole with respect to the atomic
        # displacement
        ZT = np.zeros((3*n_at, 3))  # transpose of the wanted Z matrix
        for i, (job1, job2) in enumerate(zip(*[iter(self.phonons.queue)]*2)):
            # Compute the delta dipole (in atomic units)
            d1 = np.array(job1.logfile.dipole)
            d2 = np.array(job2.logfile.dipole)
            delta_dipoles = d1 - d2
            # Compute the delta displacement (in atomic units or bohr)
            amp1 = job1.displacement.amplitude
            amp2 = job2.displacement.amplitude
            delta_u = amp1-amp2
            if job1.posinp.units == "angstroem":
                delta_u *= ANG_TO_B
            # Set the new line of the transpose of the Z matrix
            ZT[i] = (delta_dipoles*AU_TO_DEBYE) / (delta_u*B_TO_ANG)
        # Normalize by the square root of the masses of the displaced atom
        mass_norm = np.array([[mass]*9 for mass in posinp.masses])
        mass_norm = mass_norm.reshape((3*n_at, 3))
        return (ZT / np.sqrt(mass_norm)).T
