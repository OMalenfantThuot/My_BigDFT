from __future__ import absolute_import
import numpy as np
from mybigdft import Atom, Posinp, Job
from mybigdft.workflows import PolTensor

pos = Posinp([[2, 'angstroem'], ['free'],
              Atom('N', [0, 0, 0]), Atom('N', [0, 0, 1.095])])


class TestPolTensor:

    def test_run(self):
        gs = Job(posinp=pos, name='N2')
        pt = PolTensor(gs, run_dir='tests/pol_tensor_N2')
        pt.run()
        expected = [
            [1.05558000e+01, -2.00000000e-04, -2.00000000e-04],
            [-2.00000000e-04, 1.05558000e+01, -2.00000000e-04],
            [0.00000000e+00, 0.00000000e+00, 1.50535000e+01]
        ]
        np.testing.assert_almost_equal(pt.poltensor, expected)
