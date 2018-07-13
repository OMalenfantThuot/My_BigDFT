from __future__ import absolute_import
import os
import pytest
import numpy as np
from mybigdft.job import Job
from mybigdft.iofiles import InputParams, Posinp, Logfile

# Extract the input and posinp from an N2 calculation of bad quality
logname = os.path.join("tests", "log.yaml")
inp = InputParams.from_Logfile(logname)
pos = Posinp.from_Logfile(logname)
job = Job(inputparams=inp, posinp=pos)
job_with_name = Job(inputparams=inp, posinp=pos, name="test")


class TestJob:

    @pytest.mark.parametrize("attr, expected",
        [("inputparams", inp), ("posinp", pos),
         ("logfile", None), ("logfile_name", "log.yaml"),
         ("input_name", "input.yaml"), ("posinp_name", "posinp.xyz"),
         ("data_dir", "data"), ("ref_job", None), ("run_dir", "MyBigDFT"),
        ])
    def test_init(self, attr, expected):
        if "_dir" in attr:
            value = getattr(job, attr)
            value = os.path.basename(os.path.normpath(value))
            assert value == expected
        else:
            assert getattr(job, attr) == expected

    @pytest.mark.parametrize("attr, expected",
        [("inputparams", inp), ("posinp", pos),
         ("logfile", None), ("logfile_name", "log-test.yaml"),
         ("input_name", "test.yaml"), ("posinp_name", "test.xyz"),
         ("data_dir", "data-test"), ("ref_job", None),
         ("run_dir", "MyBigDFT"),
        ])
    def test_init_with_name(self, attr, expected):
        if "_dir" in attr:
            value = getattr(job_with_name, attr)
            value = os.path.basename(os.path.normpath(value))
            assert value == expected
        else:
            assert getattr(job_with_name, attr) == expected

    def test_init_with_posinp_only(self):
        assert Job(posinp=pos).inputparams == {}

    def test_init_with_skip(self):
        j = Job(inputparams=inp, skip=True)
        cmd = j.bigdft_cmd
        to_str = "{} "*len(cmd)
        cmd = to_str.format(*cmd)
        assert "-s Yes" in cmd

    def test_init_without_posinp_raises_ValueError(self):
        with pytest.raises(ValueError):
            Job()

    def test_run(self):
        with Job(inputparams=inp, run_dir="tests") as job:
            job.run()
        assert np.allclose([job.logfile.energy], [-191.74377352940274])

    def test_run_with_force_run(self):
        with Job(inputparams=inp, run_dir="tests", name="runtest") as job:
            job.run(force_run=True, nmpi=2, nomp=4)
        assert np.allclose([job.logfile.energy], [-191.74377352940274])

    def test_run_with_force_run_and_inp_and_pos(self):
        with Job(inputparams=inp, posinp=pos,
                 run_dir="tests", name="runtest") as job:
            job.run(force_run=True)
        assert np.allclose([job.logfile.energy], [-191.74377352940274])

    def test_clean(self):
        with Job(inputparams=inp, name="dry_run", run_dir="tests") as job:
            job._write_input_files(nmpi=1, dry_run=False)
            job.clean()
            assert not os.path.exists(job.posinp_name)
            assert not os.path.exists(job.input_name)
            assert not os.path.exists(job.logfile_name)

    def test_run_with_dry_run(self):
        with Job(inputparams=inp, name="dry_run", run_dir="tests") as job:
            # Run the calculation
            job.clean()
            job.run(dry_run=True, nmpi=2, nomp=4)
            # Assert that there is no posinp file afterwards
            assert not os.path.exists(job.posinp_name)
            # There must be input and output files afterwards
            new_inp = InputParams.from_file(job.input_name)
            assert new_inp == inp
            bigdft_tool_log = Logfile.from_file(job.logfile_name)
            assert bigdft_tool_log.energy is None

    def test_run_with_dry_run_with_posinp(self):
        with Job(inputparams=inp, posinp=pos, name="dry_run",
                 run_dir="tests") as job:
            job.clean()
            job.run(dry_run=True, nmpi=2, nomp=4)
            # Make sure that input, posinp and output files are created
            new_inp = InputParams.from_file(job.input_name)
            assert new_inp == inp
            new_pos = Posinp.from_file(job.posinp_name)
            assert new_pos == pos
            bigdft_tool_log = Logfile.from_file(job.logfile_name)
            assert bigdft_tool_log.energy is None
