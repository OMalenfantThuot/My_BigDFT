from __future__ import print_function


class Workflow(object):

    def __init__(self, queue=None):
        if queue is None:
            queue = []
        self.queue = queue

    @property
    def results(self):
        return {job.name: job for job in self.queue}

    def run(self, nmpi=1, nomp=1, force_run=False, dry_run=False):
        for job in self.queue:
            with job as j:
                j.run(nmpi=nmpi, nomp=nomp,
                      force_run=force_run, dry_run=dry_run)
        if not dry_run:
            self.post_proc()
