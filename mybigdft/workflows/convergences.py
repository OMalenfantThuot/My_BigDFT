r"""
Here are defined the various classes allowing to perform the convergence
of a BigDFT calculation with respect to some input parameters,
including:

* HgridsConvergence for the grid steps of the wavelet grid in each
  direction,
* RmultConvergence for the grid extension of the wavelet grid(s),
"""

import os
import warnings
from copy import deepcopy
import numpy as np
from mybigdft import Job
from mybigdft.globals import EV_TO_HA
from mybigdft.workflows.workflow import AbstractWorkflow


__all__ = ["HgridsConvergence", "RmultConvergence"]


class HgridsConvergence(AbstractWorkflow):
    r"""
    BigDFT uses wavelets on a grid to represent the wavefunctions and
    density of the system. One key parameter of a BigDFT calculation
    therefore is the grid spacing (noted hgrids) between two grid points
    (which must be defined in the three directions of space).

    This class allows to run all the necessary calculations to determine
    the largest hgrids which must be used so that the energy error
    compared to the reference calculation (with the lowest hgrid
    considered) lie within the required precision per atom.
    """

    POST_PROCESSING_ATTRIBUTES = ["converged"]

    def __init__(self, base_job, minimum, maximum, n_jobs,
                 precision_per_atom=0.01*EV_TO_HA):
        r"""
        One must provide a base `Job` instance defining the system
        considered and possibly some extra input parameters. Then, one
        must give the minimum and maximum hgrids to be considered as
        well as the number of jobs to be run in order to bridge the gap
        between both values. Finally, a precision per atom (in Ha) is
        required (default to 0.3675 mHa (= 10 meV) par atom).

        Parameters
        ----------
        base_job : Job
            Template for all the other jobs of this workflow.
        minimum : float or list of 3 floats
            Minimal hgrid considered.
        maximum : float or list of 3 floats
            Maximal hgrid considered.
        n_jobs : int
            Number of jobs to be run.
        precision_per_atom : float
            Precision per atom for a job to be considered as converged.
        """
        self._base_job = base_job
        self._precision_per_atom = precision_per_atom
        queue = self._initialize_queue(minimum, maximum, n_jobs)
        super(HgridsConvergence, self).__init__(queue)

    @property
    def base_job(self):
        r"""
        Returns
        -------
        Job
            Template for all the other jobs of this workflow.
        """
        return self._base_job

    @property
    def precision_per_atom(self):
        r"""
        Returns
        -------
        float
            Precision per atom for a job to be considered as converged.
        """
        return self._precision_per_atom

    @property
    def converged(self):
        r"""
        Returns
        -------
        Job
            Job with the highest hgrids and an energy that can be
            considered as converged (given a precision per atom).
        """
        return self._converged

    def _initialize_queue(self, minimum, maximum, n_jobs):
        r"""
        Initialize the `n_jobs` jobs to be run in order to perform the
        hgrids convergence.

        Parameters
        ----------
        minimum : float or list of 3 floats
            Minimal hgrid considered.
        maximum : float or list of 3 floats
            Maximal hgrid considered.
        n_jobs : int
            Number of jobs to be run.

        Returns
        -------
        list
            Queue of jobs to be run.

        Raises
        ------
        ValueError
            If minimum and maximum hgrids are inverted.
        """
        # Make sure the hgrids are a list of length 3
        if not isinstance(minimum, list):
            minimum = [float(minimum)]*3
        if not isinstance(maximum, list):
            maximum = [float(maximum)]*3
        for hgrids in [minimum, maximum]:
            assert len(hgrids) == 3, "{} not of length 3".format(hgrids)
        # Make sure the minimal hgrids are smaller than the maximal ones
        if any([mini >= maxi for mini, maxi in zip(minimum, maximum)]):
            raise ValueError("Minimum hgrid larger than or equal to maximum "
                             "hgrid ({} >= {})".format(minimum, maximum))
        # Define the hgrids to be used during this workflow
        minimum = np.array(minimum, dtype=float)
        maximum = np.array(maximum, dtype=float)
        delta = np.array(minimum, dtype=float) - np.array(maximum, dtype=float)
        hgrids = [(minimum - i*delta/(n_jobs-1)).tolist()
                  for i in range(n_jobs)]
        # Set the queue of jobs according to the hgrids defined
        pos = self.base_job.posinp
        name = self.base_job.name
        base_run_dir = self.base_job.run_dir
        queue = []
        for hgrid in hgrids:
            # The input parameters and the run directory of the base job
            # are updated given the value of the considered hgrid
            new_inp = deepcopy(self.base_job.inputparams)
            if "dft" not in new_inp:
                new_inp["dft"] = {"hgrids": hgrid}
            else:
                new_inp["dft"]["hgrids"] = hgrid
            new_run_dir = os.path.join(base_run_dir, "{}_{}_{}".format(*hgrid))
            job = Job(posinp=pos, inputparams=new_inp, name=name,
                      run_dir=new_run_dir)
            job.hgrids = hgrid
            queue.append(job)
        return queue

    def post_proc(self):
        r"""
        Define which job can be considered as converged with respect to
        the hgrids input parameter. It is defined by the job with the
        highest hgrids whose energy lie within the required precision
        per atom.

        Warns
        -----
        UserWarning
            If the job with the lowest hgrids can be considered as
            converged.
        """
        min_index = np.argmin([job.logfile.energy for job in self.queue])
        if min_index != 0:  # pragma: no cover
            raise ValueError("The job with minimal energy does not correspond "
                             "to the job with minimal hgrid")
        # Check which job can be considered as converged, given the
        # required precision per atom
        ref_job = self.queue[0]
        ref_job.is_converged = True
        min_en = ref_job.logfile.energy
        n_at = len(ref_job.posinp)
        index = 0  # index of the converged job
        for i, job in enumerate(self.queue[1:]):
            en = job.logfile.energy
            # Compare the energy of the other job to the reference one
            if 0 <= (en-min_en) / n_at <= self.precision_per_atom:
                # This job can be considered as converged
                index = i+1
                job.is_converged = True
            else:
                if i == 1:
                    warnings.warn("You might want to use smaller hgrids to "
                                  "make sure the convergence is achieved.",
                                  UserWarning)
                # The job cannot be considered as converged, and so are
                # those using larger hgrids
                for j in self.queue[i+1:]:
                    j.is_converged = False
                break
        # Set the value of the converged attribute
        self._converged = self.queue[index]


class RmultConvergence(AbstractWorkflow):
    r"""
    BigDFT uses wavelets on a grid to represent the wavefunctions and
    density of the system. One key parameter of a BigDFT calculation
    therefore is the grid extension (noted rmult). There are actually
    two grids centered on the atoms of the system: a coarse grid, with a
    shorter extension and one fine grid, with a longer extension.

    This class allows to run all the necessary calculations to determine
    the smallest rmult which must be used so that the energy error
    compared to the reference calculation (with the largest grid
    extension considered) lie within the required precision per atom.
    """

    POST_PROCESSING_ATTRIBUTES = ["converged"]

    def __init__(self, base_job, minimum, maximum, n_jobs,
                 precision_per_atom=0.01*EV_TO_HA):
        r"""
        One must provide a base `Job` instance defining the system
        considered and possibly some extra input parameters. Then, one
        must give the minimum and maximum rmult to be considered as well
        as the number of jobs to be run in order to bridge the gap
        between both values. Finally, a precision per atom (in Ha) is
        required (default to 0.3675 mHa (= 10 meV) par atom).

        Parameters
        ----------
        base_job : Job
            Template for all the other jobs of this workflow.
        minimum : list of 2 floats
            Minimal rmult considered.
        maximum : list of 2 floats
            Maximal rmult considered.
        n_jobs : int
            Number of jobs to be run.
        precision_per_atom : float
            Precision per atom for a job to be considered as converged.
        """
        self._base_job = base_job
        self._precision_per_atom = precision_per_atom
        queue = self._initialize_queue(minimum, maximum, n_jobs)
        super(RmultConvergence, self).__init__(queue)

    @property
    def base_job(self):
        r"""
        Returns
        -------
        Job
            Template for all the other jobs of this workflow.
        """
        return self._base_job

    @property
    def precision_per_atom(self):
        r"""
        Returns
        -------
        float
            Precision per atom for a job to be considered as converged.
        """
        return self._precision_per_atom

    @property
    def converged(self):
        r"""
        Returns
        -------
        Job
            Job with the highest hgrids and an energy that can be
            considered as converged (given a precision per atom).
        """
        return self._converged

    def _initialize_queue(self, minimum, maximum, n_jobs):
        r"""
        Initialize the `n_jobs` jobs to be run in order to perform the
        hgrids convergence.

        Parameters
        ----------
        minimum : float or list of 3 floats
            Minimal hgrid considered.
        maximum : float or list of 3 floats
            Maximal hgrid considered.
        n_jobs : int
            Number of jobs to be run.

        Returns
        -------
        list
            Queue of jobs to be run.

        Raises
        ------
        ValueError
            If minimum and maximum hgrids are inverted.
        """
        # Make sure the hgrids are a list of length 3
        for rmult in [minimum, maximum]:
            assert len(rmult) == 2, "{} not of length 2".format(rmult)
        # Make sure the minimal hgrids are smaller than the maximal ones
        if any([mini > maxi for mini, maxi in zip(minimum, maximum)]):
            raise ValueError("Minimum rmult larger than maximum "
                             "rmult ({} > {})".format(minimum, maximum))
        # Define the hgrids to be used during this workflow
        minimum = np.array(minimum, dtype=float)
        maximum = np.array(maximum, dtype=float)
        delta = np.array(minimum, dtype=float) - np.array(maximum, dtype=float)
        rmults = [(minimum - i*delta/(n_jobs-1)).tolist()
                  for i in range(n_jobs)]
        # Set the queue of jobs according to the rmults defined
        pos = self.base_job.posinp
        name = self.base_job.name
        base_run_dir = self.base_job.run_dir
        queue = []
        for rmult in rmults:
            # The input parameters and the run directory of the base job
            # are updated given the value of the considered rmult
            new_inp = deepcopy(self.base_job.inputparams)
            if "dft" not in new_inp:
                new_inp["dft"] = {"rmult": rmult}
            else:
                new_inp["dft"]["rmult"] = rmult
            new_run_dir = os.path.join(base_run_dir, "{}_{}".format(*rmult))
            job = Job(posinp=pos, inputparams=new_inp, name=name,
                      run_dir=new_run_dir)
            job.rmult = rmult
            queue.append(job)
        return queue

    def post_proc(self):
        r"""
        Define which job can be considered as converged with respect to
        the hgrids input parameter. It is defined by the job with the
        highest hgrids whose energy lie within the required precision
        per atom.

        Warns
        -----
        UserWarning
            If the job with the lowest hgrids can be considered as
            converged.
        """
        n_jobs = len(self.queue)
        min_index = np.argmin([job.logfile.energy for job in self.queue])
        if min_index != n_jobs-1:  # pragma: no cover
            warnings.warn("The job with minimal energy does not correspond "
                          "to the job with maximal rmult", UserWarning)
        # Check which job can be considered as converged, given the
        # required precision per atom
        ref_job = self.queue[-1]
        min_en = ref_job.logfile.energy
        n_at = len(ref_job.posinp)
        index = n_jobs - 1  # index of the converged job
        for i, job in enumerate(self.queue):
            en = job.logfile.energy
            # Compare the energy of the other job to the reference one
            if 0 <= (en-min_en) / n_at <= self.precision_per_atom:
                # This job can be considered as converged, and so are
                # the next ones
                index = i
                for j in self.queue[i:]:
                    j.is_converged = True
                break
            else:
                job.is_converged = False
                if i == n_jobs-1:
                    warnings.warn("You might want to use larger rmult to "
                                  "make sure the convergence is achieved.",
                                  UserWarning)
        # Set the value of the converged attribute
        self._converged = self.queue[index]
