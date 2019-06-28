r"""
The :class:`Geopt` allows to find the relaxed structure that minimizes
the forces of a given input system.
"""

import os
from copy import deepcopy
from mybigdft import Posinp
from .workflow import AbstractWorkflow


class Geopt(AbstractWorkflow):
    r"""
    This class allows to relax the input geometry of a given system in
    order to find the structure that minimizes the forces. It is meant
    to ease the creation of this type of calculation by automatically
    setting the main input parameters under the "geopt" key.
    """

    POST_PROCESSING_ATTRIBUTES = ["final_posinp"]

    def __init__(self, base_job, method="SQNM", forcemax=1e-6, frac_fluct=0.0,
                 betax=1, ncount_cluster_x=200, **kwargs):
        r"""
        The base job has a given set of input parameters, and default
        values for the main parameters of the "geopt" key are
        automatically updated. The extra arguments of the "geopt"
        input parameters key can also be passed as keyword arguments.

        Parameters
        ----------
        base_job : Job
            Base job for which a geometry optimization procedure is
            desired.
        method : str
            Name of the minimization method.
        forcemax : float
            Convergence criterion.
        frac_fluct : float
            Convergence criterion.
        betax : float
            Step size.
        ncount_cluster_x : int
            Maximum number of force evaluations to be used.
        kwargs
            Other valid input parameters under the "geopt" key.
        """
        # The queue will contain only one job, a copy of the base one
        # with updated "geopt" input parameters
        base_job = deepcopy(base_job)
        attributes = {
            "method": method, "forcemax": forcemax, "frac_fluct": frac_fluct,
            "betax": betax, "ncount_cluster_x": ncount_cluster_x}
        for key, val in kwargs.items():
            attributes[key] = val
        base_job.inputparams["geopt"] = attributes
        queue = [base_job]
        super(Geopt, self).__init__(queue=queue)

    @property
    def final_posinp(self):
        r"""
        Returns
        -------
        Posinp
            Final posinp of the geometry optimization procedure.
        """
        return self._final_posinp

    def post_proc(self):
        r"""
        Read the final posinp from a file.
        """
        job = self.queue[0]
        directory = os.path.join(job.init_dir, job.run_dir)
        filename = "final_" + job.posinp_name
        final_posinp_path = os.path.join(directory, filename)
        self._final_posinp = Posinp.from_file(final_posinp_path)
