from __future__ import absolute_import
import os
import pytest
import numpy as np
from mybigdft import Atom, Posinp, Job, InputParams
from mybigdft.workflows import PolTensor, Phonons, RamanSpectrum


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


class TestPhonons:

    gs = Job(posinp=pos, name='N2', run_dir='tests/phonons_N2')

    @pytest.mark.parametrize("to_evaluate", [
        "Phonons(self.gs, translation_amplitudes=1)",
        "Phonons(self.gs, translation_amplitudes=[3]*2)",
        "Phonons(self.gs, translation_amplitudes=[3]*4)",
    ])
    def test_init_raises_ValueError(self, to_evaluate):
        with pytest.raises(ValueError):
            eval(to_evaluate)


class TestRamanSpectrum:

    gs = Job(posinp=pos, name='N2', run_dir='tests/phonons_N2')
    ph = Phonons(gs)

    def test_run(self):
        N2_ref = """\
2   angstroem
free
N   3.571946174   3.571946174   3.620526682
N   3.571946174   3.571946174   4.71401439"""
        ref_pos = Posinp.from_string(N2_ref)
        gs = Job(posinp=ref_pos, name='N2', run_dir='tests/phonons_N2')
        phonons = Phonons(gs)
        raman = RamanSpectrum(phonons)
        raman.run(nmpi=2, nomp=2)
        # Test the only physically relevant phonon energy
        np.testing.assert_almost_equal(
            max(raman.energies['cm^-1']), 2386.9463343478246, decimal=6)
        # Test the only physically relevant intensity
        np.testing.assert_almost_equal(
            max(raman.intensities), 22.564457304830206)
        # Test the only physically relevant depolarization ratio
        i = np.argmax(raman.energies['cm^-1'])
        np.testing.assert_almost_equal(
            raman.depolarization_ratios[i], 0.09412173797731693)
        # Test that running the workflow again warns a UserWarning
        with pytest.warns(UserWarning):
            phonons.run()
        with pytest.warns(UserWarning):
            raman.run()

    @pytest.mark.parametrize("to_evaluate", [
        "RamanSpectrum(self.ph, ef_amplitudes=1)",
        "RamanSpectrum(self.ph, ef_amplitudes=[3]*2)",
        "RamanSpectrum(self.ph, ef_amplitudes=[3]*4)",
    ])
    def test_init_raises_ValueError(self, to_evaluate):
        with pytest.raises(ValueError):
            eval(to_evaluate)
