from __future__ import absolute_import
import os
import pytest
import numpy as np
from mybigdft import Atom, Posinp, Job, InputParams
from mybigdft.workflows import PolTensor, PhononEnergies


pos = Posinp(
    [Atom('N', [0, 0, 0]), Atom('N', [0, 0, 1.095])], 'angstroem', 'free')


class TestPolTensor:

    # Polarizibility tensor workflow which is not run
    inp = InputParams({'dft': {'hgrids': [0.55]*3}})
    gs = Job(inputparams=inp, posinp=pos, name='N2', run_dir='pol_tensor_N2')
    pt = PolTensor(gs)

    @pytest.mark.parametrize("value, expected", [
        (pt.ground_state, gs),
        (pt.poltensor, None),
        (pt.ef_amplitudes, [1.e-4]*3),
        (os.path.basename(pt.ground_state.run_dir), "pol_tensor_N2"),
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
        # Run a pol. tensor calculation
        gs2 = Job(posinp=pos, name='N2', run_dir='tests/pol_tensor_N2')
        pt2 = PolTensor(gs2)
        pt2.run()
        # Test the computed polarizability tensor
        expected = [
            [1.05558000e+01, -2.00000000e-04, -2.00000000e-04],
            [-2.00000000e-04, 1.05558000e+01, -2.00000000e-04],
            [0.00000000e+00, 0.00000000e+00, 1.50535000e+01]
        ]
        np.testing.assert_almost_equal(pt2.poltensor, expected)
        # Test that running the workflow again warns a UserWarning
        with pytest.warns(UserWarning):
            pt2.run()


class TestPhononEnergies:

    gs = Job(posinp=pos, name='N2', run_dir='tests/phonon_N2')

    def test_run(self):
        N2_ref = """2  angstroem
free
N    2.976307744763e-23    6.872205902435e-23    1.071620018790e-02
N   -1.104344885754e-23   -4.873421785298e-23    1.104273795769e+00"""
        pos = Posinp.from_string(N2_ref)
        gs = Job(posinp=pos, name='N2', run_dir='tests/phonon_N2')
        ph = PhononEnergies(gs)
        ph.run(nmpi=2, nomp=2)
        expected = [2.38632021e+03, 2.01915188e+01, 2.01915074e+01,
                    2.76508259e-03, 7.63586555e-06, 1.53974962e-06]
        np.testing.assert_almost_equal(ph.energies['cm^-1'], expected,
            decimal=6)
        # Test that running the workflow again warns a UserWarning
        with pytest.warns(UserWarning):
            ph.run()

    @pytest.mark.parametrize("to_evaluate", [
        "PhononEnergies(self.gs, translation_amplitudes=1)",
        "PhononEnergies(self.gs, translation_amplitudes=[3]*2)",
        "PhononEnergies(self.gs, translation_amplitudes=[3]*4)",
    ])
    def test_init_raises_ValueError(self, to_evaluate):
        with pytest.raises(ValueError):
            eval(to_evaluate)
