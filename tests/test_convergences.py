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
        hgc = HgridsConvergence(base, 0.36, 0.02, n_jobs=8,
                                precision_per_atom=0.01*EV_TO_HA)
        hgrids = [job.param for job in hgc.queue]
        expected = [[0.36+i*0.02]*3 for i in range(8)]
        np.testing.assert_array_almost_equal(hgrids, expected)

    @pytest.mark.filterwarnings("ignore::UserWarning")
    def test_run(self):
        atoms = [Atom('N', [0, 0, 0]), Atom('N', [0, 0, 1.1])]
        pos = Posinp(atoms, units="angstroem", boundary_conditions="free")
        inp = InputParams({"dft": {"rmult": [5, 7]}})
        base = Job(inputparams=inp, posinp=pos, name="N2",
                   run_dir="tests/hgrids_convergence_N2")
        hgc = HgridsConvergence(base, 0.36, 0.02, n_jobs=8,
                                precision_per_atom=0.01*EV_TO_HA)
        assert not hgc.is_completed
        hgc.run(nmpi=6, nomp=3)
        assert hgc.is_completed
        # The correct maximum hgrids is found
        assert hgc.converged.param == [0.38]*3
        # The output energies are correct
        expected_energies = [-19.888197, -19.887715, -19.887196, 0, 0, 0, 0, 0]
        energies = [job.logfile.energy if job.is_completed else 0
                    for job in hgc.queue]
        np.testing.assert_array_almost_equal(energies, expected_energies)
        # Print the summary of the workflow
        hgc.summary()
        # Test that running the workflow again warns a UserWarning
        with pytest.warns(UserWarning):
            hgc.run()


class TestRmultConvergence:

    def test_init(self):
        atoms = [Atom('N', [0, 0, 0]), Atom('N', [0, 0, 1.1])]
        pos = Posinp(atoms, units="angstroem", boundary_conditions="free")
        base = Job(posinp=pos, name="N2",
                   run_dir="tests/rmult_convergence_N2")
        rmc = RmultConvergence(base, [6, 8], [0.5, 0.5], n_jobs=3,
                               precision_per_atom=0.01*EV_TO_HA)
        rmults = [job.param for job in rmc.queue]
        expected = [[6.0, 8.0], [5.5, 7.5], [5.0, 7.0]]
        np.testing.assert_array_almost_equal(rmults, expected)

    @pytest.mark.filterwarnings("ignore::UserWarning")
    def test_run(self):
        atoms = [Atom('N', [0, 0, 0]), Atom('N', [0, 0, 1.1])]
        pos = Posinp(atoms, units="angstroem", boundary_conditions="free")
        inp = InputParams({"dft": {"rmult": [5, 7], "hgrids": 0.55}})
        base = Job(posinp=pos, inputparams=inp, name="N2",
                   run_dir="tests/rmult_convergence_N2")
        rmc = RmultConvergence(base, [8, 11], [-1, -1], n_jobs=4,
                               precision_per_atom=0.01*EV_TO_HA)
        assert not rmc.is_completed
        rmc.run(nmpi=6, nomp=3)
        assert rmc.is_completed
        # The correct minimum rmult is found
        assert rmc.converged.param == [6., 9.]
        # The output energies are correct
        expected_energies = [-19.871104, -19.871032, -19.870892, -19.869907]
        energies = [job.logfile.energy for job in rmc.queue]
        np.testing.assert_array_almost_equal(energies, expected_energies)
        # Print the summary of the workflow
        rmc.summary()
        # Test that running the workflow again warns a UserWarning
        with pytest.warns(UserWarning):
            rmc.run()

    def test_run_warns_UserWarning_too_low_energy(self):
        atoms = [Atom('N', [0, 0, 0]), Atom('N', [0, 0, 1.1])]
        pos = Posinp(atoms, units="angstroem", boundary_conditions="free")
        inp = InputParams({"dft": {"rmult": [5, 7], "hgrids": 0.55}})
        base = Job(posinp=pos, inputparams=inp, name="N2",
                   run_dir="tests/rmult_convergence_N2")
        rmc = RmultConvergence(base, [9, 12], [-1, -1], n_jobs=2,
                               precision_per_atom=0.01*EV_TO_HA)
        with pytest.warns(UserWarning):
            rmc.run(nmpi=6, nomp=3)

    def test_run_warns_UserWarning_all_converged(self):
        atoms = [Atom('N', [0, 0, 0]), Atom('N', [0, 0, 1.1])]
        pos = Posinp(atoms, units="angstroem", boundary_conditions="free")
        inp = InputParams({"dft": {"rmult": [5, 7], "hgrids": 0.55}})
        base = Job(posinp=pos, inputparams=inp, name="N2",
                   run_dir="tests/rmult_convergence_N2")
        rmc = RmultConvergence(base, [8, 11], [-1, -1], n_jobs=3,
                               precision_per_atom=0.01*EV_TO_HA)
        with pytest.warns(UserWarning):
            rmc.run(nmpi=6, nomp=3)

    def test_run_warns_UserWarning_only_reference_converged(self):
        atoms = [Atom('N', [0, 0, 0]), Atom('N', [0, 0, 1.1])]
        pos = Posinp(atoms, units="angstroem", boundary_conditions="free")
        inp = InputParams({"dft": {"rmult": [5, 7], "hgrids": 0.55}})
        base = Job(posinp=pos, inputparams=inp, name="N2",
                   run_dir="tests/rmult_convergence_N2")
        rmc = RmultConvergence(base, [5, 8], [-1, -1], n_jobs=2,
                               precision_per_atom=0.01*EV_TO_HA)
        with pytest.warns(UserWarning):
            rmc.run(nmpi=6, nomp=3)
