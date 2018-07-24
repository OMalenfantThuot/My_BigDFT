from __future__ import absolute_import
import os
import pytest
import numpy as np
from mybigdft import InputParams, Posinp, Logfile, Job


class TestJob:

    # Extract the input and posinp from an N2 calculation of bad quality
    logname = os.path.join("tests", "log-warnings.yaml")
    log = Logfile.from_file(logname)
    inp = log.inputparams
    pos = log.posinp
    job = Job(inputparams=inp, posinp=pos)
    job_with_name = Job(inputparams=inp, posinp=pos, name="test")

    @pytest.mark.parametrize("attr, expected", [
        ("inputparams", inp), ("posinp", pos),
        ("input_name", "input.yaml"), ("posinp_name", "posinp.xyz"),
        ("logfile", None), ("logfile_name", "log.yaml"),
        ("data_dir", "data"), ("ref_job", None), ("run_dir", "MyBigDFT")
    ])
    def test_init(self, attr, expected):
        if "_dir" in attr:
            value = getattr(self.job, attr)
            value = os.path.basename(os.path.normpath(value))
            assert value == expected
        else:
            assert getattr(self.job, attr) == expected

    @pytest.mark.parametrize("attr, expected", [
        ("inputparams", inp), ("posinp", pos),
        ("input_name", "test.yaml"), ("posinp_name", "test.xyz"),
        ("logfile", None), ("logfile_name", "log-test.yaml"),
        ("data_dir", "data-test"), ("ref_job", None),
        ("run_dir", "MyBigDFT"),
    ])
    def test_init_with_name(self, attr, expected):
        if "_dir" in attr:
            value = getattr(self.job_with_name, attr)
            value = os.path.basename(os.path.normpath(value))
            assert value == expected
        else:
            assert getattr(self.job_with_name, attr) == expected

    def test_init_with_posinp_only(self):
        assert Job(posinp=self.pos).inputparams == {}

    def test_init_with_skip(self):
        j = Job(inputparams=self.inp, skip=True)
        cmd = j.bigdft_cmd
        to_str = "{} "*len(cmd)
        cmd = to_str.format(*cmd)
        assert "-s Yes" in cmd

    def test_init_without_posinp_raises_ValueError(self):
        with pytest.raises(ValueError, match="provide initial positions"):
            Job()

    def test_init_with_different_posinp_in_inputparams_raises_ValueError(self):
        with pytest.raises(ValueError, match="do not define the same posinp."):
            Job(inputparams=self.inp,
                posinp=Posinp.from_file("tests/surface.xyz"))

    @pytest.mark.parametrize("attr", [
        "inputparams", "posinp", "logfile", "ref_job",
        "input_name", "posinp_name", "logfile_name",
        "bigdft_cmd", "bigdft_tool_cmd",
        "init_dir", "run_dir", "data_dir",
    ])
    def test_cannot_set_attributes(self, attr):
        with pytest.raises(AttributeError):
            setattr(self.job, attr, 1)

    @pytest.mark.filterwarnings("ignore::UserWarning")
    def test_run(self):
        with Job(inputparams=self.inp, run_dir="tests",
                 name='warnings') as job:
            job.run()
        assert np.allclose([job.logfile.energy], [-191.74377352940274])

    @pytest.mark.filterwarnings("ignore::UserWarning")
    def test_run_with_force_run(self):
        with Job(inputparams=self.inp, run_dir="tests", name="runtest") as job:
            job.run(force_run=True, nmpi=2, nomp=4)
        assert np.allclose([job.logfile.energy], [-191.74377352940274])

    @pytest.mark.filterwarnings("ignore::UserWarning")
    def test_run_with_force_run_and_inp_and_pos(self):
        with Job(inputparams=self.inp, posinp=self.pos,
                 run_dir="tests", name="runtest") as job:
            job.run(force_run=True)
        assert np.allclose([job.logfile.energy], [-191.74377352940274])

    def test_clean(self):
        with Job(inputparams=self.inp, name="dry_run", run_dir="tests") as job:
            job.write_input_files()
            job.clean(logfiles_dir=True, data_dir=True)
            assert not os.path.exists(job.posinp_name)
            assert not os.path.exists(job.input_name)
            assert not os.path.exists(job.logfile_name)
            assert not os.path.exists("logfiles")
            assert not os.path.exists("data-dry_run")

    @pytest.mark.filterwarnings("ignore::UserWarning")
    def test_run_with_dry_run(self):
        with Job(inputparams=self.inp, name="dry_run", run_dir="tests") as job:
            # Run the calculation
            job.clean()
            job.run(dry_run=True, nmpi=2, nomp=4)
            # There must be input and output files afterwards
            new_inp = InputParams.from_file(job.input_name)
            assert new_inp == self.inp
            new_pos = Posinp.from_file(job.posinp_name)
            assert new_pos == self.pos
            bigdft_tool_log = Logfile.from_file(job.logfile_name)
            assert bigdft_tool_log.energy is None

    @pytest.mark.filterwarnings("ignore::UserWarning")
    def test_run_with_dry_run_with_posinp(self):
        with Job(inputparams=self.inp, posinp=self.pos, name="dry_run",
                 run_dir="tests") as job:
            job.clean()
            job.run(dry_run=True, nmpi=2, nomp=4)
            # Make sure that input, posinp and output files are created
            new_inp = InputParams.from_file(job.input_name)
            assert new_inp == self.inp
            new_pos = Posinp.from_file(job.posinp_name)
            assert new_pos == self.pos
            bigdft_tool_log = Logfile.from_file(job.logfile_name)
            assert bigdft_tool_log.energy is None

    def test__check_logfile_posinp(self):
        pos_name = os.path.join("tests", "surface.xyz")
        pos = Posinp.from_file(pos_name)
        with pytest.raises(UserWarning):
            with Job(posinp=pos, run_dir="tests") as job:
                job.run()

    @pytest.mark.filterwarnings("ignore::UserWarning")
    def test__check_logfile_inputparams(self):
        with pytest.raises(UserWarning):
            with Job(inputparams=InputParams(), posinp=self.pos,
                     name="warnings", run_dir="tests") as job:
                job.run()

    def test_posinp_with_inf(self):
        inp = InputParams({"posinp": {
            "units": "angstroem",
            "cell": [40, ".inf", 40],
            "positions": [
                {'N': [2.97630782434901e-23, 6.87220595204354e-23,
                       0.0107161998748779]},
                {'N': [-1.10434491945017e-23, -4.87342174483075e-23,
                       1.10427379608154]},
            ]
        }})
        with Job(inputparams=inp, name="test") as job:
            job.run(nmpi=6, nomp=3, dry_run=True)
            job.clean()
        assert job.logfile.boundary_conditions == 'surface'

    def test_run_raises_RuntimeError(self):
        # Error because two ".inf" in cell
        inp = InputParams({"posinp": {
            "units": "angstroem",
            "cell": [40, ".inf", ".inf"],
            "positions": [
                {'N': [2.97630782434901e-23, 6.87220595204354e-23,
                       0.0107161998748779]},
                {'N': [-1.10434491945017e-23, -4.87342174483075e-23,
                       1.10427379608154]},
            ]
        }})
        with pytest.raises(RuntimeError):
            with Job(inputparams=inp, run_dir="tests/dummy") as job:
                job.run(force_run=True)

    def test_dry_run_raises_RuntimeError(self):
        # Error because two ".inf" in cell
        inp = InputParams({"posinp": {
            "units": "angstroem",
            "cell": [40, ".inf", ".inf"],
            "positions": [
                {'N': [2.97630782434901e-23, 6.87220595204354e-23,
                       0.0107161998748779]},
                {'N': [-1.10434491945017e-23, -4.87342174483075e-23,
                       1.10427379608154]},
            ]
        }})
        with pytest.raises(RuntimeError):
            with Job(inputparams=inp, run_dir="tests/dummy") as job:
                job.run(dry_run=True)
