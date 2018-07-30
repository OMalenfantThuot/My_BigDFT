r"""
The workflows of the MyBigDFT package are meant to ease the calculation
of some interesting quantities requiring to launch multiple BigDFT
calculations by providing an API that focuses on the main parameters of
such nested calculations.

Here are defined an AbstractWorkflow class (meant to be the base class
of all the other workflow classes implemented in the workflows module)
and a Workflow class, which represents the simplest way of implementing
such a child class (intended to be used when one wants to create a toy
implementation of a new workflow).
"""

from __future__ import print_function, unicode_literals
import sys
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

    def __init__(self, queue):
        r"""
        Parameters
        ----------
        queue : list
            List of all the jobs to run.
        """
        self._queue = queue

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
        return {job.name: job.logfiles for job in self.queue}

    def run(self, nmpi=1, nomp=1, force_run=False, dry_run=False):
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
        """
        for job in self.queue:
            with job as j:
                j.run(nmpi=nmpi, nomp=nomp,
                      force_run=force_run, dry_run=dry_run)
        if not dry_run:
            self.post_proc()

    @abc.abstractmethod
    def post_proc(self):  # pragma: no cover
        r"""
        This should be an abstract method used to post-process the
        output of the calculations and get some meaningful results out of
        them.
        """
        pass


class Workflow(AbstractWorkflow):
    r"""
    This is a usable workflow that one can play with, but without post-
    processing. This means you can add jobs to the queue and run them as
    usual, but you must then code the post-processing yourself.
    """

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

    def post_proc(self):  # pragma: no cover
        r"""
        Nothing is done here.
        """
        pass
