from __future__ import print_function


class Workflow(object):
    r"""
    This (due to be an) abstract class is the base class of all the
    workflows of this module. It
    defines the queue of jobs as a list of :class:`~mybigdft.job.Job`
    instances, that are run sequentially when the :meth:`run` method is
    used.
    """

    def __init__(self, queue=None):
        r"""
        Parameters
        ----------
        queue : list
            List of all the jobs to run.
        """
        if queue is None:
            queue = []
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

    def post_proc(self):
        r"""
        This should be an abstract method used to post-process the
        output of the calculations and get some meaningful results out of
        them.
        """
        pass
