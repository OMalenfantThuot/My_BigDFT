r"""
The :class:`VibPolTensor` class allows to compute the vibrational
polarizability tensor (actually only its mean value for the moment).
"""

import numpy as np
from mybigdft.globals import ANG_TO_B
from .workflow import AbstractWorkflow


class VibPolTensor(AbstractWorkflow):
    r"""
    This class allows to run all the calculations enabling the
    computation of the vibrational polarizability tensor of a given
    system.

    One therefore needs to compute the infrared spectrum first. This is
    done by solving the dynamical matrix (the eigenvalues giving the
    phonon energies and the eigenvectors the normal modes). This matrix
    is computed at the expense of :math:`6 n_{at}` BigDFT calculations,
    where each atom is in turns translated by a small amount around its
    equilibrium positions. You may want to refer to the
    :class:`~mybigdft.workflows.phonons.Phonons` class for more details.

    To get the intensities of the infrared spectrum, one must compute
    the derivative of the dipole moment along the normal modes. All the
    necessary dipole moments to use are readily found in the logfiles of
    the BigDFT calculations performed to compute the phonons: no extra
    calculation is required.

    Finally, given these intensities and energies, one can compute the
    mean value of the polarizability tensor.
    """

    POST_PROCESSING_ATTRIBUTES = ["mean_polarizability"]

    def __init__(self, infrared, e_cut=200):
        r"""
        From an infrared spectrum calculation, one is able to compute
        the mean vibrational polarizability. One only needs to use the
        normal modes that are physically relevant, and not all the
        numerical normal modes: the latter contain some artificial modes
        that should have zero energy and zero intensity. They only add
        noise to the calculation of the vibrational polarizabily tensor,
        hence the need for a cut-off energy `e_cut` to use only the
        normal modes with an energy higher than that user-defined value.

        Parameters
        ----------
        infrared : InfraredSpectrum
            Infrared spectrum workflow.
        e_cut : float
            If phonons have a lower energy than `e_cut`, they are not
            considered in the computation of the vibrational
            polarizability tensor (units: cm^-1).
        """
        self._infrared = infrared
        self._e_cut = e_cut
        super(VibPolTensor, self).__init__(queue=[])

    @property
    def infrared(self):
        r"""
        Returns
        -------
        InfraredSpectrum
            Infrared spectrum workflow.
        """
        return self._infrared

    @property
    def e_cut(self):
        r"""
        Returns
        -------
        float
            Cut-off energy considered in the computation of the
            vibrational polarizability tensor (units: cm^-1).
        """
        return self._e_cut

    @e_cut.setter
    def e_cut(self, new_value):
        r"""
        Set a new cut-off energy and compute a new value for the mean
        vibrational polarizability if necessary (*i.e.*, if the
        calculations are already performed).

        Parameters
        ----------
        new_value: float
            Cut-off energy for the phonon modes to be taken into account
            when computing the vibrational polarizability (units:
            cm^-1).
        """
        if new_value != self.e_cut:
            self._e_cut = new_value
            if self.is_completed:
                self.post_proc()

    @property
    def mean_polarizability(self):
        r"""
        Returns
        -------
        float
            Mean vibrational polarizability (units: atomic).
        """
        return self._mean_polarizability

    def _run(self, nmpi, nomp, force_run, dry_run, restart_if_incomplete,
             timeout):
        r"""
        Run the calculations allowing to compute the phonon energies and
        the related infrared intensities in order to be able to compute
        the mean vibrational polarizability of the system under
        consideration.

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
        self.infrared.run(
            nmpi=nmpi, nomp=nomp, force_run=force_run, dry_run=dry_run,
            restart_if_incomplete=restart_if_incomplete, timeout=timeout)
        super(VibPolTensor, self)._run(
            nmpi, nomp, force_run, dry_run, restart_if_incomplete, timeout)

    def post_proc(self):
        r"""
        Compute and set the mean vibrational polarizability of the
        considered system (units: atomic).
        """
        # Filter energies and normal modes according to e_cut
        e = self.infrared.phonons.energies["cm^-1"]
        e_f = e[np.where(e > self.e_cut)]
        # Convert intensities to km.mol^-1 and filter them
        intensities = self.infrared.intensities * 42.255
        i_f = intensities[np.where(e > self.e_cut)]
        conversion = 1.4891465E-37/1.1126501E-40*ANG_TO_B**3
        self._mean_polarizability = np.sum(i_f/e_f**2) * conversion  # atomic
        # self.mean_polarizability = np.sum(i_f/e_f**2) * 1.4891465E-37  # SI
