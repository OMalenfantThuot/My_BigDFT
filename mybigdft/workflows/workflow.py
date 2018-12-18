r"""
The workflows of the MyBigDFT package are meant to ease the calculation
of some interesting quantities requiring to launch multiple BigDFT
calculations by providing an API that focuses on the main parameters of
such nested calculations.

Here are defined an :class:`AbstractWorkflow` class (meant to be the
base class of all the other workflow classes implemented in the
workflows module) and a :class:`Workflow` class, which represents the
simplest way of implementing such a child class (intended to be used
when one wants to create a toy implementation of a new workflow).
"""

from __future__ import print_function, unicode_literals
import sys
import warnings
import abc

if sys.version_info >= (3, 4):  # pragma: no cover
    ABC = abc.ABC
else:  # pragma: no cover
    ABC = abc.ABCMeta(str('ABC'), (), {})


class AbstractWorkflow(ABC):
    r"""
    This abstract class is the base class of all the workflows of this
    module. It defines the queue of jobs as a list of
    :class:`~mybigdft.job.Job` instances, that are run sequentially when
    the :meth:`run` method is used.
    """

    POST_PROCESSING_ATTRIBUTES = []

    def __init__(self, queue):
        r"""
        Parameters
        ----------
        queue : list
            List of all the jobs to run.
        """
        self._initialize_post_processing_attributes()
        self._queue = queue

    def _initialize_post_processing_attributes(self):
        r"""
        Set all the completion attributes to their default value.
        """
        for attr in self.POST_PROCESSING_ATTRIBUTES:
            setattr(self, "_"+attr, None)

    @property
    def queue(self):
        r"""
        Returns
        -------
        list
            All the jobs of the workflow.
        """
        return self._queue

    @property
    def logfiles(self):
        r"""
        Returns
        -------
        dict
            A dictionary of all the logfiles of the workflow, with the
            name of the associated job as key.
        """
        return {job.name: job.logfile for job in self.queue}

    def run(self, nmpi=1, nomp=1, force_run=False, dry_run=False,
            restart_if_incomplete=False):
        r"""
        Run all the calculations if the post-processing was not already
        performed.

        .. Warning::

            If `force_run` or `dry_run` is set to `True`, then any
            previous value of the post-processing attributes is deleted
            and set back to their default value, so that the
            post-processing is not considered as being performed.


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
        -----
        UserWarning
            If the post-processing was already completed.
        """
        if force_run or dry_run:
            self._initialize_post_processing_attributes()
        if not self.is_completed:
            self._run(nmpi, nomp, force_run, dry_run, restart_if_incomplete)
        else:
            warning_msg = "Calculations already performed; set the argument "\
                          "'force_run' to True to re-run them."
            warnings.warn(warning_msg, UserWarning)
        if any([not job.is_completed for job in self.queue]):
            warnings.warn("Some jobs of the workflow were not run.",
                          UserWarning)

    @property
    def is_completed(self):
        r"""
        Returns
        -------
        bool
            `True` if all the post-processing attributes are no longer
            set to their default value.
        """
        return all([getattr(self, attr) is not None
                    for attr in self.POST_PROCESSING_ATTRIBUTES])

    def _run(self, nmpi, nomp, force_run, dry_run, restart_if_incomplete):
        r"""
        This method runs all the jobs in the queue sequentially before
        running the post_proc method if not in `dry_run` mode.

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
        for job in self.queue:
            with job as j:
                j.run(
                    nmpi=nmpi, nomp=nomp, force_run=force_run, dry_run=dry_run,
                    restart_if_incomplete=restart_if_incomplete)
        if not dry_run:
            self.post_proc()
            assert self.is_completed, ("You must define all post-processing "
                                       "attributes in post_proc.")

    @abc.abstractmethod
    def post_proc(self):
        r"""
        This should be an abstract method used to post-process the
        output of the calculations and get some meaningful results out of
        them.
        """
        raise NotImplementedError


class Workflow(AbstractWorkflow):
    r"""
    This is a usable workflow that one can play with, but without
    post-processing. This means you can add jobs to the queue and run
    them as usual, but you must then code the post-processing yourself,
    in a separate function taking the workflow as parameter.

    This latter scheme can even be part of the development cycle of a
    new workflow that could be later added to the MyBigDFT project: you
    first define roughly the jobs to be added to the queue for your
    workflow, and then develop a post-processing function taking your
    workflow as argument. When you are happy with the result, it is
    then easy to re-use most of the code to create a workflow class
    deriving from the :class:`AbstractWorkflow` class.

    To do that, you only have to make sure that you override the
    `__init__` method to create the queue of jobs, then define which are
    the post-processing arguments and create properties to access them,
    and finally override the `post_proc` method to make sure these
    attributes are properly initialized.
    """

    POST_PROCESSING_ATTRIBUTES = ["completed"]

    def __init__(self, queue=None):
        r"""
        Parameters
        ----------
        queue : list
            List of all the jobs to run.


        The queue can be empty:

        >>> wf = Workflow()
        >>> wf.queue
        []
        >>> wf.logfiles
        {}
        """
        if queue is None:
            queue = []
        super(Workflow, self).__init__(queue)

    @property
    def completed(self):
        r"""
        Returns
        -------
        bool
            `True` if the post_proc method was run successfully.
        """
        return self._completed

    def post_proc(self):
        r"""
        Set the post-processing attribute ``completed`` to `True`
        """
        self._completed = True
