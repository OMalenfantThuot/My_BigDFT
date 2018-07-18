from __future__ import absolute_import
import os
import pytest
import numpy as np
from mybigdft import Atom, Posinp, Job, InputParams, Workflow
from mybigdft.workflows import PolTensor

pos = Posinp([[2, 'angstroem'], ['free'],
              Atom('N', [0, 0, 0]), Atom('N', [0, 0, 1.095])])


class TestWorkflow:

    wf = Workflow()

    @pytest.mark.parametrize("value, expected", [
        (wf.queue, []), (wf.results, {}),
    ])
    def test_init(self, value, expected):
        assert value == expected


class TestPolTensor:

    # Run a pol. tensor calculation
    gs = Job(posinp=pos, name='N2')
    pt = PolTensor(gs, run_dir='tests/pol_tensor_N2')
    pt.run()
    # Another pol. tensor which is not run
    inp2 = InputParams({'dft': {'hgrids': [0.55]*3}})
    gs2 = Job(inputparams=inp2, posinp=pos, name='N2', run_dir='N2_pol_tensor')
    pt2 = PolTensor(gs2, run_dir='tests/pol_tensor_N2')

    @pytest.mark.parametrize("value, expected", [
        (pt2.ground_state, gs2),
        (pt2.poltensor, None),
        (pt2.ef_amplitudes, [1.e-4]*3),
        (os.path.basename(pt2.ground_state.run_dir), "pol_tensor_N2"),
    ])
    def test_init(self, value, expected):
        assert value == expected

    def test_init_warns_UserWarning(self):
        test_inp = InputParams({'dft': {'elecfield': [0.1]*3}})
        job = Job(inputparams=test_inp, posinp=pos)
        with pytest.warns(UserWarning):
            PolTensor(job)

    @pytest.mark.parametrize("to_evaluate", [
        'PolTensor(self.gs, ef_amplitudes=0.0)',
        'PolTensor(self.gs, ef_amplitudes=[0.0])',
        'PolTensor(self.gs, ef_amplitudes=[0.1]*4)',
    ])
    def test_init_ef_amplitudes_raises_ValueError(self, to_evaluate):
        with pytest.raises(ValueError):
            eval(to_evaluate)

    def test_run(self):
        expected = [
            [1.05558000e+01, -2.00000000e-04, -2.00000000e-04],
            [-2.00000000e-04, 1.05558000e+01, -2.00000000e-04],
            [0.00000000e+00, 0.00000000e+00, 1.50535000e+01]
        ]
        np.testing.assert_almost_equal(self.pt.poltensor, expected)

    def test_run_multiple_times_warns_UserWarning(self):
        with pytest.warns(UserWarning):
            self.pt.run()
