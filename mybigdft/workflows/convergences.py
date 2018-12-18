r"""
Here are defined the various classes allowing to perform the convergence
of a BigDFT calculation with respect to some input parameters,
including:

* AbstractConvergence, which is the base class for all the other classes
  here,
* HgridsConvergence for the grid steps of the wavelet grid in each
  direction,
* RmultConvergence for the grid extension of the wavelet grid(s).
"""

import os
import sys
import warnings
import abc
from copy import deepcopy
import numpy as np
from mybigdft import Job
from mybigdft.globals import EV_TO_HA
from mybigdft.workflows.workflow import AbstractWorkflow

if sys.version_info >= (3, 4):  # pragma: no cover
    ABC = abc.ABC
else:  # pragma: no cover
    ABC = abc.ABCMeta(str('ABC'), (), {})


__all__ = ["HgridsConvergence", "RmultConvergence"]


class AbstractConvergence(AbstractWorkflow, ABC):
    r"""
    This is the base class allowing to run a convergence procedure.
    Starting from a reference value for the considered input parameter,
    a queue of jobs is defined, each job with a value of that input
    parameter that should lead to results of decreasing quality.

    When the worklfow is run, the jobs are run by decreasing quality.
    After each job is run, its energy is compared to the reference
    energy and must lie in a given precision window, defined by an
    energy per atom to be defined as converged. Once a job is not in the
    acceptable range, this means that lower quality jobs won't be
    neither, so the workflow is stopped, even if more jobs are in the
    queue.
    """

    POST_PROCESSING_ATTRIBUTES = ["converged"]

    def __init__(self, base_job, reference, delta, n_jobs=10,
                 precision_per_atom=0.01*EV_TO_HA):
        r"""
        One must provide a base `Job` instance defining the system
        considered and possibly non-default input parameters. Then, one
        must give the reference value of the input
        parameter for which the convergence is considered. A delta value
        of this input parameter is then used to initialize a given
        number of jobs (`n_jobs`) of decreasing quality. Finally, a
        precision per atom (in Ha) is required (default to 0.3675 mHa
        (= 10 meV) par atom).

        Parameters
        ----------
        base_job : Job
            Template for all the other jobs of this workflow.
        reference
            Reference input parameter, giving high quality results.
        delta
            Variation of of the input parameter between two runs.
        n_jobs : int
            Maximal number of jobs to be run.
        precision_per_atom : float
            Precision per atom for a job to be considered as converged,
            the reference energy being that of the job using the
            reference input parameter (units: Ha).
        """
        reference, delta = self._clean_initial_parameters(reference, delta)
        # Set all the important attributes
        self._base_job = base_job
        self._precision_per_atom = precision_per_atom
        queue = self._initialize_queue(reference, delta, n_jobs)
        super(AbstractConvergence, self).__init__(queue=queue)

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

    @abc.abstractmethod
    def summary(self, first_column_label, first_column_values):
        r"""
        Print a summary of the convergence workflow.

        This method must be called at the end of each child method
        :meth:`summary`.

        Parameters
        ----------
        first_column_label: str
            Label of the first column.
        first_column_values: list of str
            String representation of the value of the varied parameter
            for each job that ran.
        """
        shift = 2
        unformatted = ["{:.2e}", "{}"]
        other_labels = ["precision_per_atom (Ha)", "is_converged"]
        other_lengths = [len(label) + shift for label in other_labels]
        # Print the desired precision per atom
        print("Requested precision per atom: {:.2e} (Ha)"
              .format(self.precision_per_atom))
        if first_column_values != []:
            # Print the header of the table
            first_column_length = len(first_column_values[0]) + shift
            first_column = first_column_label.center(first_column_length)
            other_columns = ""
            for label in other_labels:
                other_columns += label.center(len(label)+shift)
            header = first_column + other_columns
            print("-"*len(header))
            print(header)
            print("-"*len(header))
            # Print each line of the table
            for i, value in enumerate(first_column_values):
                first_column = value.center(first_column_length)
                job = self.queue[i]
                values = [job.precision_per_atom, job.is_converged]
                other_columns = ""
                for i, value in enumerate(values):
                    other_columns += unformatted[i]\
                                     .format(value).center(other_lengths[i])
                print(first_column + other_columns)

    @staticmethod
    @abc.abstractmethod
    def _clean_initial_parameters(reference, delta):
        r"""
        Clean the value of the initial parameters.

        Parameters
        ----------
        reference
            Reference input parameter, giving high quality results.
        delta
            Variation of of the input parameter between two runs.

        Returns
        -------
        tuple of length 2
            Reference value and delta value of the varied parameter.
        """
        raise NotImplementedError

    def _initialize_queue(self, reference, delta, n_jobs):
        r"""
        Initialize the jobs to be run in order to perform the hgrids
        convergence.

        Parameters
        ----------
        reference
            Reference input parameter, giving high quality results.
        delta
            Variation of of the input parameter between two runs.
        n_jobs : int
            Maximal number of jobs to be run.

        Returns
        -------
        list
            Queue of jobs to be run.
        """
        # Define the parameters to be used during this workflow
        param_variations = self._initialize_param_variations(
            reference, delta, n_jobs)
        # Set the queue of jobs according to the hgrids defined
        pos = self.base_job.posinp
        name = self.base_job.name
        queue = []
        for param in param_variations:
            # The input parameters and the run directory of the base job
            # are updated given the value of the parameter
            new_inp = self._new_inputparams(param)
            new_run_dir = self._new_run_dir(param)
            job = Job(posinp=pos, inputparams=new_inp, name=name,
                      run_dir=new_run_dir)
            job.param = param
            queue.append(job)
        return queue

    @staticmethod
    @abc.abstractmethod
    def _initialize_param_variations(reference, delta, n_jobs):
        r"""
        Initialize the value of the input parameter that varies
        for each job to be created.

        Parameters
        ----------
        reference
            Reference input parameter, giving high quality results.
        delta
            Variation of of the input parameter between two runs.
        n_jobs : int
            Maximal number of jobs to be run.

        Returns
        -------
        list
            Values of the input parameter for all the jobs.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def _new_inputparams(self, param):
        r"""
        Define the input parameters of the new job, according to the
        value of the varied parameter.

        Parameters
        ----------
        param
            Value of the varied parameter

        Returns
        -------
        InputParams
            Value of the new input parameters.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def _new_run_dir(self, param):
        r"""
        Define the directory where the new job must run according
        to the value of the varied parameter.

        Parameters
        ----------
        param
            Value of the varied parameter

        Returns
        -------
        str
            Value of the new run_dir
        """
        raise NotImplementedError

    def _run(self, nmpi, nomp, force_run, dry_run, restart_if_incomplete):
        r"""
        This method runs the jobs until the hgrids are too high to
        stop giving results in the desired precision range.

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

        Warns
        ------
        UserWarning
            If the job with minimal energy does not correspond to the
            job with minimal hgrids.
        """
        # Run the first job of the queue and get the reference energy
        with self.queue[0] as ref_job:
            ref_job.run(nmpi=nmpi, nomp=nomp, force_run=force_run,
                        dry_run=dry_run,
                        restart_if_incomplete=restart_if_incomplete)
        ref_job.is_converged = True
        ref_job.precision_per_atom = 0.0
        self._converged = True
        min_en = ref_job.logfile.energy
        n_at = len(ref_job.posinp)
        # Run the jobs until the energy of a given run is above the
        # requested precision
        for i, job in enumerate(self.queue[1:]):
            if not self.queue[i].is_converged:
                # If the previous job is not converged, then neither is
                # this one.
                job.is_converged = False
            else:
                with job as j:
                    j.run(nmpi=nmpi, nomp=nomp, force_run=force_run,
                          dry_run=dry_run,
                          restart_if_incomplete=restart_if_incomplete)
                # Warn a UserWarning if the current job gives a lower
                # energy than the reference one
                en = job.logfile.energy
                if en <= min_en:
                    warnings.warn(self._too_low_energy_msg, UserWarning)
                # Assess if the job is converged or not
                job.precision_per_atom = (en-min_en) / n_at
                job.is_converged = \
                    job.precision_per_atom <= self.precision_per_atom
                if job.is_converged:
                    self._converged = job
        if not dry_run:
            self.post_proc()
            assert self.is_completed, ("You must define all post-processing "
                                       "attributes in post_proc.")

    @property
    @abc.abstractmethod
    def _too_low_energy_msg(self):
        r"""
        Define the message telling the user what steps he mighy consider
        to remove the UserWarning in :method:`_run`.

        Raises
        ------
        NotImplementedError
        """
        raise NotImplementedError

    def post_proc(self):
        r"""
        Warns
        -----
        UserWarning
            If only the reference job can be considered as converged or
            if all the jobs are below the requested precision.
        """
        # Warn the user if he is too conservative
        if self.queue[-1].is_converged:
            warnings.warn(self._all_converged_msg, UserWarning)
        # Warn the user if he must consider using parameters giving
        # better results
        if not self.queue[1].is_converged:
            warnings.warn(self._only_reference_converged_msg, UserWarning)

    @property
    @abc.abstractmethod
    def _all_converged_msg(self):
        r"""
        Define the message warning the user when he is being too
        conservative with the current set of varied input parameters.

        Raises
        ------
        NotImplementedError
        """
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def _only_reference_converged_msg(self):
        r"""
        Define the message warning the user that he should probably
        increase the quality of the calculations.

        Raises
        ------
        NotImplementedError
        """
        raise NotImplementedError


class HgridsConvergence(AbstractConvergence):
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

    def summary(self):
        r"""
        Print a summary of the results of the hgrids convergence
        workflow.
        """
        first_column_values = ["[{:.2f}, {:.2f}, {:.2f}]".format(*job.param)
                               for job in self.queue if job.is_completed]
        super(HgridsConvergence, self).summary("hgrids",
                                               first_column_values)

    @staticmethod
    def _clean_initial_parameters(reference, delta):
        r"""
        Clean the value of the initial parameters.

        Returns
        -------
        tuple made of two numpy arrays of length 3
            Reference value and delta value of hgrids.
        """
        # Make sure all the received hgrids are lists of length 3
        if not isinstance(delta, list):
            delta = [float(delta)]*3
        if not isinstance(reference, list):
            reference = [float(reference)]*3
        for hgrids in [reference, delta]:
            assert len(hgrids) == 3, "{} not of length 3".format(hgrids)
        # Make sure the signs in delta are positive
        delta = np.array([abs(value) for value in delta], dtype=float)
        return np.array(reference, dtype=float), delta

    @staticmethod
    def _initialize_param_variations(reference, delta, n_jobs):
        r"""
        Initialize the value of hgrids for each job to be created.

        Parameters
        ----------
        reference
            Reference input parameter, giving high quality results.
        delta
            Variation of of the input parameter between two runs.
        n_jobs : int
            Maximal number of jobs to be run.

        Returns
        -------
        list
            Values of hgrids for all the jobs.
        """
        return [(reference + i*delta).round(decimals=2).tolist()
                for i in range(n_jobs)]

    def _new_inputparams(self, param):
        r"""
        Define the input parameters of the new job, according to the
        value of hgrids.

        Parameters
        ----------
        param
            Value of hgrids

        Returns
        -------
        InputParams
            Value of the new input parameters.
        """
        new_inp = deepcopy(self.base_job.inputparams)
        if "dft" not in new_inp:
            new_inp["dft"] = {"hgrids": param}
        else:
            new_inp["dft"]["hgrids"] = param
        return new_inp

    def _new_run_dir(self, param):
        r"""
        Define the directory where the new job must run according
        to the value of hgrids.

        Parameters
        ----------
        param
            Value of hgrids

        Returns
        -------
        str
            Value of the new run_dir
        """
        base_run_dir = self.base_job.run_dir
        return os.path.join(base_run_dir,
                            "{:.2f}_{:.2f}_{:.2f}".format(*param))

    @property
    def _too_low_energy_msg(self):  # pragma: no cover
        r"""
        Returns
        -------
        str
            Message telling the user what steps he mighy consider
            to remove the UserWarning he is going to receive because
            a lower quality calculation gave a lower energy than the
            reference calculation.
        """
        return ("The job with minimal energy does not correspond to the job "
                "with minimal hgrids. Consider increasing rmult or the "
                "reference hgrids.")

    @property
    def _all_converged_msg(self):  # pragma: no cover
        r"""
        Returns
        -------
        str
            Message warning the user he is being too conservative with
            this set of varied input parameters.
        """
        return ("You may want to test higher hgrids as there is still room "
                "to increase hgrids while staying below the requested "
                "precision per atom.")

    @property
    def _only_reference_converged_msg(self):  # pragma: no cover
        r"""
        Returns
        -------
        str
            Message warning the user he should probably increase the
            quality of the calculations.
        """
        return ("You may want to consider smaller values of hgrids to make "
                "sure convergence was achieved.")


class RmultConvergence(AbstractConvergence):
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

    def summary(self):
        r"""
        Print a summary of the results of the rmult convergence
        workflow.
        """
        first_column_values = ["[{:.1f}, {:.1f}]".format(*job.param)
                               for job in self.queue if job.is_completed]
        super(RmultConvergence, self).summary("rmult",
                                              first_column_values)

    @staticmethod
    def _clean_initial_parameters(reference, delta):
        r"""
        Clean the value of the initial parameters.

        Returns
        -------
        tuple made of two numpy arrays of length 2
            Reference value and delta value of rmult.
        """
        # Make sure the rmults are lists of length 2
        for rmult in [reference, delta]:
            assert len(rmult) == 2, "{} not of length 2".format(rmult)
        # Make sure the signs in delta are positive
        delta = np.array([-abs(value) for value in delta])
        return np.array(reference, dtype=float), delta

    @staticmethod
    def _initialize_param_variations(reference, delta, n_jobs):
        r"""
        Initialize the value of rmult for each job to be created.

        Parameters
        ----------
        reference
            Reference input parameter, giving high quality results.
        delta
            Variation of of the input parameter between two runs.
        n_jobs : int
            Maximal number of jobs to be run.

        Returns
        -------
        list
            Values of rmults for all the jobs.
        """
        return [(reference + i*delta).round(decimals=1).tolist()
                for i in range(n_jobs)]

    def _new_inputparams(self, param):
        r"""
        Define the input parameters of the new job, according to the
        value of rmult.

        Parameters
        ----------
        param
            Value of rmult

        Returns
        -------
        InputParams
            Value of the new input parameters.
        """
        new_inp = deepcopy(self.base_job.inputparams)
        if "dft" not in new_inp:
            new_inp["dft"] = {"rmult": param}
        else:
            new_inp["dft"]["rmult"] = param
        return new_inp

    def _new_run_dir(self, param):
        r"""
        Define the directory where the new job must run according
        to the value of rmult.

        Parameters
        ----------
        param
            Value of rmult

        Returns
        -------
        str
            Value of the new run_dir
        """
        base_run_dir = self.base_job.run_dir
        return os.path.join(base_run_dir, "{:.1f}_{:.1f}".format(*param))

    @property
    def _too_low_energy_msg(self):
        r"""
        Returns
        -------
        str
            Message telling the user what steps he mighy consider
            to remove the UserWarning he is going to receive because
            a lower quality calculation gave a lower energy than the
            reference calculation.
        """
        return ("The job with minimal energy does not correspond to the job "
                "with maximal rmult. Consider decreasing hgrids or the "
                "reference rmult.")

    @property
    def _all_converged_msg(self):
        r"""
        Returns
        -------
        str
            Message warning the user he is being too conservative with
            this set of varied input parameters.
        """
        return ("You may want to test lower rmult as there is still room to "
                "decrease rmult while staying below the requested precision "
                "per atom.")

    @property
    def _only_reference_converged_msg(self):
        r"""
        Returns
        -------
        str
            Message warning the user he should probably increase the
            quality of the calculations.
        """
        return ("You may want to consider larger values of rmult to make sure "
                "convergence was achieved.")
