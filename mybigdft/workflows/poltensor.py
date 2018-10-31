r"""
This module defines the :class:`PolTensor` workflow.
"""
from __future__ import print_function
import warnings
import os
from copy import deepcopy
from collections import Sequence, namedtuple, OrderedDict
import numpy as np
from mybigdft import Job
from mybigdft.globals import COORDS, SIGNS
from .workflow import AbstractWorkflow


class PolTensor(AbstractWorkflow):
    r"""
    This workflow allows to compute the (electronic) polarizability
    tensor of a given system.

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

        \alpha_{ij} = \frac{\Delta D_i}{\Delta E_j}

    where :math:`i, j \in \{x, y, z\}`, :math:`\Delta D_i` is the
    variation of the dipole along the :math:`i` direction and
    :math:`\Delta E_j` is the variation of the electric field amplitude
    along the :math:`j` direction.
    """

    POST_PROCESSING_ATTRIBUTES = ["pol_tensor", "mean_polarizability"]

    def __init__(self, ground_state, ef_amplitudes=None, order=1):
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
        order : int
            Order of the numerical differentiation used to compute the
            polarizability tensor. If second order (resp. first), then
            six (resp. three) calculations per atom are to be performed.
        """
        # Set a default value to ef_amplitudes
        if ef_amplitudes is None:
            ef_amplitudes = [1e-4]*3
        # Check the value of the order
        order = int(order)
        if order not in [1, 2]:
            raise NotImplementedError("Only first and second order available")
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
        self._order = order
        # Depending on the desired order, there are 3 or 6 electric
        # fields to be applied on the system
        self._efields = self._init_efields()
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
    def order(self):
        r"""
        Returns
        -------
        order : int
            Order of the numerical differentiation used to compute the
            polarizability tensor. If second order (resp. first), then
            six (resp. three) calculations per atom are to be performed.
        """
        return self._order

    @property
    def pol_tensor(self):
        r"""
        Returns
        -------
        numpy.array
            Polarizability tensor of the system.
        """
        return self._pol_tensor

    @property
    def mean_polarizability(self):
        r"""
        Returns
        -------
        float
            Mean (electronic) polarizability of the system.
        """
        return self._mean_polarizability

    @property
    def efields(self):
        r"""
        Returns
        -------
        OrderedDict of length 3 or 6
            Electric fields the system must undergo before computing the
            polarizability tensor as post-processing. There are three or
            six of them (one or two per space coordinate, depending on
            the order of the numerical derivative procedure) if the
            forward or central difference scheme is used, respectively.
        """
        return self._efields

    def _initialize_queue(self):
        r"""
        Initialize the queue of calculations to be performed in order to
        compute the polarizability tensor.
        """
        queue = []
        gs = self.ground_state
        # Add the ground state job to the queue after updating the run
        # directory if needed
        if self.order == 1:
            queue.append(gs)
        # Add a job for each electric field calculation (one along each
        # space coordinate)
        for key, efield in self.efields.items():
            inp = deepcopy(gs.inputparams)
            if 'dft' in inp:
                inp['dft']['elecfield'] = efield.vector
            else:
                inp['dft'] = {'elecfield': efield.vector}
            run_dir = os.path.join(gs.run_dir, "EF_along_{}".format(key))
            job = Job(name=gs.name, inputparams=inp, posinp=gs.posinp,
                      run_dir=run_dir, skip=gs.skip, ref_data_dir=gs.data_dir)
            job.efield = efield
            queue.append(job)
        return queue

    def _init_efields(self):
        r"""
        Set the electric fields each atom must undergo from the
        electric field amplitudes in each direction.
        """
        efields = OrderedDict()
        if self.order == 1:
            signs = {"+": 1.}  # One electric field per coordinate
        elif self.order == 2:
            signs = SIGNS  # Two electric fields per coordinate
        for i, coord in enumerate(COORDS):
            for sign in signs:
                key = coord + sign
                amplitude = signs[sign] * self.ef_amplitudes[i]
                efields[key] = ElectricField(i, amplitude)
        return efields

    def post_proc(self):
        r"""
        Compute the polarisability tensor and set its value (you can
        access its value via the attribute :attr:`pol_tensor`).
        """
        pol_tensor = np.zeros((3, 3))
        if self.order == 1:
            d0 = np.array(self.ground_state.logfile.dipole)
            for i, job in enumerate(self.queue[1:]):
                delta_ef = job.efield.amplitude
                d1 = np.array(job.logfile.dipole)
                # Update the polarizability tensor
                pol_tensor[:, i] = (d1 - d0) / delta_ef
        elif self.order == 2:
            # Compute the second order tensor elements
            # for i, job1, job2 in enue(zip(self.queue[::2], self.queue[1::2]))
            for i, (job1, job2) in enumerate(zip(*[iter(self.queue)]*2)):
                # Get the delta of electric field amplitude
                amp1 = job1.efield.amplitude
                amp2 = job2.efield.amplitude
                assert amp1 == - amp2
                delta_ef = amp1 - amp2
                # Get the delta of dipoles
                d1 = np.array(job1.logfile.dipole)
                d2 = np.array(job2.logfile.dipole)
                # Update the polarizability tensor
                pol_tensor[:, i] = (d1 - d2) / delta_ef
        # Set some attributes
        self._pol_tensor = pol_tensor  # atomic units
        self._mean_polarizability = pol_tensor.trace()/3  # atomic units


class ElectricField(namedtuple('ElectricField', ['i_coord', 'amplitude'])):
    r"""
    This class defines an electric field from the coordinate index and
    the amplitude of the electric field in that direction.
    """

    __slots__ = ()

    @property
    def vector(self):
        r"""
        Returns
        -------
        list
            ElectricField vector.
        """
        vector = [0.0] * 3
        vector[self.i_coord] = self.amplitude
        return vector
