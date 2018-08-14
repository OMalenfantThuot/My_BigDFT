from __future__ import absolute_import
import pytest
import numpy as np
from mybigdft import Atom, Posinp, Job, InputParams
from mybigdft.workflows import HgridsConvergence, RmultConvergence
from mybigdft.globals import EV_TO_HA


class TestHgridsConvergence:

    def test_init(self):
        atoms = [Atom('N', [0, 0, 0]), Atom('N', [0, 0, 1.1])]
        pos = Posinp(atoms, units="angstroem", boundary_conditions="free")
        base = Job(posinp=pos, name="N2",
                   run_dir="tests/hgrids_convergence_N2")
        hgc = HgridsConvergence(base, 0.36, 0.5, 8,
                                precision_per_atom=0.01*EV_TO_HA)
        hgrids = [job.hgrids for job in hgc.queue]
        expected = [[0.36+i*0.02]*3 for i in range(8)]
        np.testing.assert_array_almost_equal(hgrids, expected)

    @pytest.mark.filterwarnings("ignore::UserWarning")
    def test_run(self):
        atoms = [Atom('N', [0, 0, 0]), Atom('N', [0, 0, 1.1])]
        pos = Posinp(atoms, units="angstroem", boundary_conditions="free")
        inp = InputParams({"dft": {"rmult": [5, 7]}})
        base = Job(inputparams=inp, posinp=pos, name="N2",
                   run_dir="tests/hgrids_convergence_N2")
        hgc = HgridsConvergence(base, 0.36, 0.5, 8,
                                precision_per_atom=0.01*EV_TO_HA)
        assert not hgc.is_completed
        hgc.run(nmpi=6, nomp=3)
        assert hgc.is_completed
        # The correct maximum hgrids is found
        assert hgc.converged.hgrids == [0.38]*3
        # The output energies are correct
        expected_energies = [
            -19.888197, -19.887715, -19.887196, -19.886415, -19.885138,
            -19.883615, -19.881536, -19.878895
        ]
        energies = [job.logfile.energy for job in hgc.queue]
        np.testing.assert_array_almost_equal(energies, expected_energies)
        # Test that running the workflow again warns a UserWarning
        with pytest.warns(UserWarning):
            hgc.run()

    def test_init_raises_ValueError(self):
        atoms = [Atom('N', [0, 0, 0]), Atom('N', [0, 0, 1.1])]
        pos = Posinp(atoms, units="angstroem", boundary_conditions="free")
        base = Job(posinp=pos, name="N2",
                   run_dir="tests/hgrids_convergence_N2")
        with pytest.raises(ValueError):
            HgridsConvergence(base, [0.5, 0.4, 0.1], [0.3]*3, 8)

    def test_init_raises_AssertionError(self):
        atoms = [Atom('N', [0, 0, 0]), Atom('N', [0, 0, 1.1])]
        pos = Posinp(atoms, units="angstroem", boundary_conditions="free")
        base = Job(posinp=pos, name="N2",
                   run_dir="tests/hgrids_convergence_N2")
        with pytest.raises(AssertionError):
            HgridsConvergence(base, [0.5, 0.4], [0.3]*3, 8)


class TestRmultConvergence:

    def test_init(self):
        atoms = [Atom('N', [0, 0, 0]), Atom('N', [0, 0, 1.1])]
        pos = Posinp(atoms, units="angstroem", boundary_conditions="free")
        base = Job(posinp=pos, name="N2",
                   run_dir="tests/rmult_convergence_N2")
        rmc = RmultConvergence(base, [5, 7], [6, 8], 3,
                               precision_per_atom=0.01*EV_TO_HA)
        rmults = [job.rmult for job in rmc.queue]
        expected = [[5.0, 7.0], [5.5, 7.5], [6.0, 8.0]]
        np.testing.assert_array_almost_equal(rmults, expected)

    @pytest.mark.filterwarnings("ignore::UserWarning")
    def test_run(self):
        atoms = [Atom('N', [0, 0, 0]), Atom('N', [0, 0, 1.1])]
        pos = Posinp(atoms, units="angstroem", boundary_conditions="free")
        inp = InputParams({"dft": {"hgrids": 0.4}})
        base = Job(inputparams=inp, posinp=pos, name="N2",
                   run_dir="tests/rmult_convergence_N2")
        rmc = RmultConvergence(base, [4, 6], [7, 9], 4,
                               precision_per_atom=0.001*EV_TO_HA)
        assert not rmc.is_completed
        rmc.run(nmpi=6, nomp=3)
        assert rmc.is_completed
        # The correct minimum rmult is found
        assert rmc.converged.rmult == [6., 8.]
        # The output energies are correct
        expected_energies = [-19.883572, -19.887196, -19.887484, -19.887511]
        energies = [job.logfile.energy for job in rmc.queue]
        np.testing.assert_array_almost_equal(energies, expected_energies)
        # Test that running the workflow again warns a UserWarning
        with pytest.warns(UserWarning):
            rmc.run()

    def test_init_raises_ValueError(self):
        atoms = [Atom('N', [0, 0, 0]), Atom('N', [0, 0, 1.1])]
        pos = Posinp(atoms, units="angstroem", boundary_conditions="free")
        base = Job(posinp=pos, name="N2",
                   run_dir="tests/rmult_convergence_N2")
        with pytest.raises(ValueError):
            RmultConvergence(base, [5, 10], [6, 7], 3)

    def test_init_raises_AssertionError(self):
        atoms = [Atom('N', [0, 0, 0]), Atom('N', [0, 0, 1.1])]
        pos = Posinp(atoms, units="angstroem", boundary_conditions="free")
        base = Job(posinp=pos, name="N2",
                   run_dir="tests/rmult_convergence_N2")
        with pytest.raises(AssertionError):
            RmultConvergence(base, [5, 8], [10]*3, 8)
