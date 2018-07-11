from __future__ import absolute_import
import os
import pytest
from mybigdft.iofiles import InputParams, Posinp, Logfile, Atom

tests_fol = "tests"
logname = os.path.join(tests_fol, "log.yaml")


class TestInputParams:

    filename = os.path.join(tests_fol, "test.yaml")

    @pytest.mark.parametrize("params, expected",
        [({"dft": {"hgrids": 0.35}}, {"dft": {"hgrids": 0.35}}),
         ({"dft": {"hgrids": [0.45]*3, "rmult": [5.0, 8.0]}}, {}), ])
    def test_init(self, params, expected):
        inp = InputParams(params=params)
        assert inp == expected

    @pytest.mark.parametrize("to_evaluate",
        ["InputParams({'dfpt': {'hgrids': 0.35}})",
         "InputParams({'dft': {'hgrid': 0.35}})", ])
    def test_init_raises_KeyError(self, to_evaluate):
        with pytest.raises(KeyError):
            eval(to_evaluate)

    def test_set(self):
        inp = InputParams()
        inp["dft"] = {"hgrids": 0.45}
        assert inp == {"dft": {"hgrids": 0.45}}

    def test_set_with_default_values(self):
        inp = InputParams({"dft": {"hgrids": 0.45}})
        inp["dft"] = {"hgrids": [0.45]*3}
        assert inp == {}

    def test_set_warns_UserWarning(self):
        inp = InputParams()
        with pytest.warns(UserWarning):
            inp["posinp"] = {}
        assert inp == {}

    @pytest.mark.parametrize("key, value",
        [("dfpt", {"hgrids": 0.45}),
         ("dft", {"hgrid": 0.45}), ])
    def test_set_raises_KeyError(self, key, value):
        inp = InputParams()
        with pytest.raises(KeyError):
            inp[key] = value

    @pytest.mark.parametrize("k1, k2, value",
        [("dfpt", "hgrids", 0.45),
         ("dpt", "hgrid", 0.45), ])
    def test_set_two_keys_raises_KeyError(self, k1, k2, value):
        inp = InputParams()
        with pytest.raises(KeyError):
            inp[k1][k2] = value

    def test_del(self):
        inp = InputParams({"dft": {"hgrids": 0.45}})
        del inp["dft"]
        assert inp == {}

    def test_write(self):
        inp = InputParams()
        inp.write(self.filename)
        assert inp == {}

    def test_len(self):
        assert len(InputParams()) == 0

    def test_from_file(self):
        inp = InputParams.from_file(self.filename)
        assert inp == {}

    def test_from_Logfile(self):
        inp = InputParams.from_Logfile(logname)
        inp["posinp"]["cell"][1] = 0
        expected = {
'dft': {'hgrids': 0.35, 'ixc': -101130, 'rmult': [6, 8]},
'kpt': {'method': 'mpgrid', 'ngkpt': [4, 1, 4]},
'geopt': {'ncount_cluster_x': 1},
'posinp': {'units': 'reduced',
           'cell': [7.99354093169, 0, 4.61507300869],
           'positions': [{'C': [0.08333333333, 0.5, 0.25]},
                         {'C': [0.41666666666, 0.5, 0.25]},
                         {'C': [0.5833333333299999, 0.5, 0.7499999999999999]},
                         {'C': [0.91666666666, 0.5, 0.7499999999999999]}]}
}
        assert inp == expected

    def test_from_string(self):
        inp_str = "{'dft': {'hgrids': 0.3, 'rmult': [5, 7]}}"
        inp = InputParams.from_string(inp_str)
        assert inp == {'dft': {'hgrids': 0.3, 'rmult': [5, 7]}}


class TestLogfile:

    log = Logfile.from_file(logname)

    @pytest.mark.parametrize("key, value",
        [("Energy (Hartree)", -2.63322126878162308E+01),
         ("Number of MPI tasks", 6),
         ("OpenMP parallelization", True),
         ("Maximal OpenMP threads per MPI task", 3),
         ("Force Norm (Hartree/Bohr)", 1.71711365963343639E-02),
         ("Walltime since initialization", 148.76689)])
    def test_from_file(self, key, value):
        assert self.log[key] == value

    def test_cannot_set(self):
        with pytest.raises(TypeError):
            self.log['Walltime since initialization'] = 0

    def test_len(self):
        assert len(self.log) == 97

    def test_write(self):
        fname = os.path.join(tests_fol, "log-test.yaml")
        self.log.write(fname)
        log2 = Logfile.from_file(fname)
        assert log2 == self.log


class TestPosinp:

    surface_filename = os.path.join(tests_fol, "surface.xyz")
    pos = Posinp.from_file(surface_filename)
    free_filename = os.path.join(tests_fol, "free.xyz")
    free_pos = Posinp.from_file(free_filename)
    log_pos = Posinp.from_Logfile(logname)
    string = """\
4   atomic
free
C    0.6661284109   0.000000000   1.153768252
C    3.330642055    0.000000000   1.153768252
C    4.662898877    0.000000000   3.461304757
C    7.327412521    0.000000000   3.461304757"""
    str_pos = Posinp.from_string(string)

    @pytest.mark.parametrize("value, expected",
        [(pos.n_at, 4), (pos.units, "reduced"), (len(pos), 4),
         (pos.BC, "surface"),
         (pos.cell, [8.07007483423, 1.0, 4.65925987792]),
         (pos[0], Atom("C", [0.08333333333, 0.5, 0.25]))])
    def test_from_file(self, value, expected):
        assert value == expected

    @pytest.mark.parametrize("value, expected",
        [(log_pos.n_at, 4), (log_pos.units, "reduced"), (len(log_pos), 4),
         (log_pos.BC, "surface"),
         (log_pos.cell, [7.99354, 0.0, 4.61507]),
         (log_pos[0], Atom("C", [0.08333333333, 0.5, 0.25]))])
    def test_from_Logfile(self, value, expected):
        assert value == expected

    def test_from_string(self):
        assert self.str_pos == self.free_pos

    def test_write(self):
        fname = os.path.join(tests_fol, "test.xyz")
        self.pos.write(fname)
        assert self.pos == Posinp.from_file(fname)
        os.remove(fname)

    def test_free_BC_has_no_cell(self):
        assert self.free_pos.cell is None

    def test_translate_atom(self):
        new_pos = self.pos.translate_atom(0, [0.5, 0, 0])
        assert new_pos != self.pos
        assert new_pos[0] == Atom("C", [0.58333333333, 0.5, 0.25])

    @pytest.mark.parametrize("fname",
        ["free_reduced.xyz", "missing_atom.xyz", "additional_atom.xyz"])
    def test_init_raises_ValueError(self, fname):
        with pytest.raises(ValueError):
            Posinp.from_file(os.path.join(tests_fol, fname))
