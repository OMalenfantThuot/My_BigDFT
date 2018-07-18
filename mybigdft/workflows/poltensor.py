from __future__ import print_function
import warnings
from copy import deepcopy
from collections import Sequence
import numpy as np
from mybigdft import Job
from mybigdft.job import COORDS
from .workflow import Workflow


class PolTensor(Workflow):
    r"""
    This Workflow allows to compute the polarizability tensor of a given
    system.

    The polarizability tensor represents the response of the charges of
    a system (its dipole) to the application of an external electric
    field.

    To compute this polarizability tensor, some BigDFT calculations are
    performed, where the system is subject to an external electric
    field along each direction of space (:math:`x`, :math:`y` and
    :math:`z`). The elements of the polarizability tensor are then
    defined by the ratio of the delta of the dipole in one direction and
    the delta of the electric field amplitudes:

    .. math::

        \alpha_ij = \frac{\Delta D_i}{\Delta E_j}

    where :math:`i, j \in \{x, y, z\}`, :math:`\Delta D_i` is the
    variation of the dipole along the :math:`i` direction and
    :math:`\Delta E_j` is the variation of the electric field amplitude
    alog the :math:`j` direction
    """

    def __init__(self, ground_state, ef_amplitudes=[1e-4]*3, run_dir=None,
                 skip=False):
        r"""
        A PolTensor workflow is initialized by the job of the ground-
        state of the system and three electric field amplitudes.

        Parameters
        ----------
        ground_state : Job
            Job used to compute the ground state of the system under
            consideration.
        ef_amplitudes : list or numpy array of length 3
            Amplitude of the electric field to be applied in the three
            directions of space (:math:`x`, :math:`y`, :math:`z`).
        """
        # Check the ground state has no electric field
        if 'dft' in ground_state.inputparams:
            efield = ground_state.inputparams['dft'].get('elecfield')
            if efield is not None:
                warnings.warn("The ground state input parameters define an "
                              "electric field", UserWarning)
        # Check the electric field amplitudes
        if not isinstance(ef_amplitudes, Sequence) or len(ef_amplitudes) != 3:
            raise ValueError("You must provide three electric field "
                             "amplitudes, one for each space coordinate.")
        # Initialize the attributes that are specific to this workflow
        self._ground_state = ground_state
        self._ef_amplitudes = ef_amplitudes
        self._poltensor = None  # The pol. tensor is not yet computed
        # Initialize the queue of jobs for this workflow
        queue = self._initialize_queue(run_dir, skip)
        super(PolTensor, self).__init__(queue=queue)

    @property
    def ground_state(self):
        r"""
        Returns
        -------
        Job
            Job used to compute the ground state of the system under
            consideration.
        """
        return self._ground_state

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
    def poltensor(self):
        r"""
        """
        return self._poltensor

    def _initialize_queue(self, run_dir, skip):
        r"""
        """
        queue = []
        # Add the ground state job to the queue after updating the run
        # directory
        gs = self.ground_state
        if run_dir is not None:
            gs._set_directory_attributes(run_dir)
        queue.append(gs)
        # Add the electric field calculations along each coordinate
        efields = np.eye(3) * self.ef_amplitudes
        for i, efield in enumerate(efields):
            name = (gs.name + "_EF_along_{}".format(COORDS[i])).lstrip("_")
            inp = deepcopy(gs.inputparams)
            if 'dft' in inp:
                inp['dft']['elecfield'] = efield.tolist()
            else:
                inp['dft'] = {'elecfield': efield.tolist()}
            job = Job(name=name, inputparams=inp, posinp=gs.posinp,
                      run_dir=run_dir, skip=skip, ref_job=gs)
            queue.append(job)
        return queue

    def run(self, nmpi=1, nomp=1, force_run=False, dry_run=False):
        r"""
        Run the calculations allowing to compute the polarizability
        tensor if the latter was not already computed.
        """
        if self.poltensor is None:
            super(PolTensor, self).run(
                nmpi=nmpi, nomp=nomp, force_run=force_run, dry_run=dry_run)
        else:
            warning_msg = "Calculations already performed; set the argument "\
                          "'force_run' to True to re-run them."
            warnings.warn(warning_msg, UserWarning)

    def post_proc(self):
        r"""
        Method computing the polarisability tensor.
        """
        poltensor = np.zeros((3, 3))
        # Ground state dipole
        d0 = np.array(self.ground_state.logfile.dipole)
        for i, job in enumerate(self.queue[1:]):
            # Get the amplitude of the electric field applied
            ef = np.array(job.logfile['dft']['elecfield'])
            ef_amplitude = np.max(np.abs(ef))
            # Dipole after application of the electric field
            d1 = np.array(job.logfile.dipole)
            # Update the polarizability tensor
            poltensor[i] = (d1 - d0) / ef_amplitude
        self._poltensor = poltensor
