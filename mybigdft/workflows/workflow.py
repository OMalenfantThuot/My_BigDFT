from __future__ import print_function
from mybigdft import Job


class Workflow(object):

    def __init__(self, queue=None):
        if queue is None:
            self.queue = []
        elif isinstance(queue, list) and all([isinstance(job, Job)
                                              for job in queue]):
            self.queue = queue
        else:
            raise ValueError("The queue must be a list of Job instances.")

    @property
    def results(self):
        return {job.name: job.result for job in self.queue}

    def run(self, nmpi=1, nomp=1, force_run=False, dry_run=False):
        for job in self.queue:
            with job as j:
                j.run(nmpi=nmpi, nomp=nomp,
                      force_run=force_run, dry_run=dry_run)
        if not dry_run:
            self.post_proc()

    def post_proc(self):
        pass
