r"""
The :class:`Dissociation` class allows to perform all the jobs
necessary to obtain the dissociation curve between two fragments (sub-
systems).
"""

import os
from copy import deepcopy
import numpy as np
from mybigdft import Job
from mybigdft.workflows.workflow import AbstractWorkflow


class Dissociation(AbstractWorkflow):
    r"""
    This workflow allows to get all the necessary data in order to plot
    a dissociation curve of two fragments (or sub-systems). This curve
    represent how the energy of the whole system is modified when the
    distance between the two fragments varies.

    The fragments will be separated from each other in the :math:`y`
    direction according to a set of distances.
    """

    POST_PROCESSING_ATTRIBUTES = ["energies", "minimum"]

    def __init__(
        self, fragment1, fragment2, distances, inputparams=None, name="", run_dir=None
    ):
        r"""
        A queue of :class:`~mybigdft.job.Job` instances is initialized,
        one per distance between the fragments. For each job, the system
        is made of both fragments, where each atom of the second
        fragment is translated along the :math:`y` direction by the
        value of the distance.

        .. Warning::

            If both fragments do not have the same boundary conditions
            or cell, it is the ones of the first fragment that are used.


        .. Note::

            The :math:`y` axis was chosen so that surface boundary
            conditions could be used as well (for instance, the first
            fragment can be a surface while the second is an atom or
            molecule).

        Parameters
        ----------
        inputparams : InputParams
        fragment1 : Posinp
            Posinp of the first fragment.
        fragment2 : Posinp
            Posinp of the second fragment.
        distances : list or numpy.array
            Distances between both fragments.
        name : str
            Name to be used for each job.

        Raises
        ------
        ValueError
            If one fragment defines periodic boundary conditions.
        """
        # Check that the fragments are not periodic
        for frag in [fragment1, fragment2]:
            if frag.boundary_conditions == "periodic":
                raise ValueError(
                    "Cannot compute a dissociation curve with periodic "
                    "boundary conditions:\n{}".format(frag)
                )
        # Make sure both fragments use the same units (could actually be
        # implemented properly in the __add__ method of posinp)
        if fragment1.units != fragment2.units:
            raise NotImplementedError(
                "Unit conversion of positions needed"
            )  # pragma: no cover
        # Set the base attributes that are specific to this workflow
        self.fragment1 = fragment1
        self.fragment2 = fragment2
        self.distances = distances
        # Define a fake job from the given arguments in order to
        # initialize properly the other base atributes
        job = Job(name=name, inputparams=inputparams, posinp=fragment1, run_dir=run_dir)
        self.inputparams = job.inputparams
        self.name = job.name
        self.run_dir = job.run_dir
        # Initialize the queue of jobs for this workflow
        queue = self._initialize_queue()
        super(Dissociation, self).__init__(queue=queue)

    @property
    def minimum(self):
        r"""
        Returns
        -------
        Job
            Job that gave the lowest energy.
        """
        return self._minimum

    @property
    def energies(self):
        r"""
        Returns
        -------
        list
            Total energy of each job in the queue.
        """
        return self._energies

    def _initialize_queue(self):
        r"""
        A job per distance is created. Each job uses the same input
        parameters while the whole system considered is made of both
        fragments, the first one being unchanged, the second one being
        translated by the given distance in the :math:`y` direction. The
        name used to initialize the workflow is used for each job, while
        a specific run_dir per job is defined.
        """
        queue = []
        for y_0 in self.distances:
            # Define a specific run directory for each job
            run_dir = os.path.join(self.run_dir, "y_{}".format(y_0))
            # Set the positions of the whole system, where the second
            # fragment is translated along the y direction
            new_frag2 = self.fragment2.translate([0, y_0, 0])
            pos = deepcopy(self.fragment1)
            pos._atoms += new_frag2.atoms
            # Add a new job to the queue
            job = Job(
                name=self.name,
                inputparams=self.inputparams,
                posinp=pos,
                run_dir=run_dir,
            )
            job.distance = y_0  # We add the distance attribute
            queue.append(job)
        return queue

    def post_proc(self):
        r"""
        Find the energy of each job and the job that gave the minimum
        energy.
        """
        self._energies = [job.logfile.energy for job in self.queue]
        index = np.argmin(self.energies)
        self._minimum = self.queue[index]
