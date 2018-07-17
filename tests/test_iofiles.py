from __future__ import absolute_import
import os
import pytest
from mybigdft import InputParams, Posinp, Logfile, Atom

tests_fol = "tests"
# Result of an N2 calculation of very bad quality
logname = os.path.join(tests_fol, "log-warnings.yaml")
log = Logfile.from_file(logname)


class TestInputParams:

    filename = os.path.join(tests_fol, "test.yaml")

    @pytest.mark.parametrize("params, expected", [
        ({"dft": {"hgrids": 0.35}}, {"dft": {"hgrids": 0.35}}),
        ({"dft": {"hgrids": [0.45]*3, "rmult": [5.0, 8.0]}}, {}),
    ])
    def test_init(self, params, expected):
        inp = InputParams(params=params)
        assert inp == expected

    @pytest.mark.parametrize("to_evaluate", [
        "InputParams({'dfpt': {'hgrids': 0.35}})",
        "InputParams({'dft': {'hgrid': 0.35}})",
    ])
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

    @pytest.mark.parametrize("key, value", [
        ("dfpt", {"hgrids": 0.45}),
        ("dft", {"hgrid": 0.45}),
    ])
    def test_set_raises_KeyError(self, key, value):
        inp = InputParams()
        with pytest.raises(KeyError):
            inp[key] = value

    @pytest.mark.parametrize("k1, k2, value", [
        ("dfpt", "hgrids", 0.45),
        ("dpt", "hgrid", 0.45),
    ])
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

    @pytest.mark.filterwarnings("ignore::UserWarning")
    def test_from_Logfile(self):
        inp = InputParams.from_Logfile(log)
        expected = {
            'dft': {'rmult': [2, 4], 'hgrids': 2.5, 'itermax': 1,
                    'disablesym': True},
            'posinp': {
                'units': 'angstroem',
                'positions':
                    [{'N': [2.9763078243490115e-23, 6.872205952043537e-23,
                            0.01071619987487793]},
                     {'N': [-1.1043449194501671e-23, -4.873421744830746e-23,
                            1.104273796081543]}],
                'properties': {'format': 'xyz', 'source': 'N2.xyz'}
            }
        }
        assert inp == expected

    def test_from_string(self):
        inp_str = "{'dft': {'hgrids': 0.3, 'rmult': [5, 7]}}"
        inp = InputParams.from_string(inp_str)
        assert inp == {'dft': {'hgrids': 0.3, 'rmult': [5, 7]}}


class TestLogfile:

    # Logfile of an N2 calculation of bad quality
    log = Logfile.from_file(logname)

    @pytest.mark.parametrize("key, value", [
        ("Energy (Hartree)", -191.74377352940274),
        ("Number of MPI tasks", 2),
        ("OpenMP parallelization", True),
        ("Maximal OpenMP threads per MPI task", 16),
        ("Force Norm (Hartree/Bohr)", 448.63530538041755),
        ("Walltime since initialization", 1.835567),
    ])
    def test_from_file(self, key, value):
        assert self.log[key] == value

    @pytest.mark.parametrize("attr, value", [
        ("_walltime", 1.835567),
        ("_energy", -191.74377352940274),
        ("_n_at", 2),
        ("_boundary_conditions", "free"),
        ("_dipole", [0.69649, 0.69649, -2.4954]),
        ("_sdos", None),
        ("_magnetization", None),
        ("_pressure", None),
        ("_atom_types", ['N']),
    ])
    def test_attributes(self, attr, value):
        # Two asserts as one can get both attributes (say, _n_at and n_at)
        assert getattr(self.log, attr) == value
        assert getattr(self.log, attr[1:]) == value

    @pytest.mark.parametrize("name", [
        "n_at", "forces", "walltime"
    ])
    def test_set_base_attributes_raises_AttributeError(self, name):
        with pytest.raises(AttributeError):
            setattr(self.log, name, getattr(self.log, name))

    def test_cannot_set_values_of_log_attr(self):
        with pytest.raises(TypeError):
            self.log['Walltime since initialization'] = 0

    def test_len(self):
        assert len(self.log) == 90

    @pytest.mark.filterwarnings("ignore::UserWarning")
    def test_write(self):
        fname = os.path.join(tests_fol, "log-test.yaml")
        self.log.write(fname)
        log2 = Logfile.from_file(fname)
        assert log2 == self.log

    def test_init_wrong_ixc_warns_UserWarning(self):
        self.log["dft"]["ixc"] = -101130
        with pytest.warns(UserWarning):
            self.log._check_psppar()
        self.log["dft"]["ixc"] = 1

    def test_init_run_warnings_warns_UserWarnings(self):
        with pytest.warns(UserWarning):
            Logfile.from_file("tests/log-warnings.yaml")


class TestPosinp:

    # Posinp with surface boundary conditions
    surface_filename = os.path.join(tests_fol, "surface.xyz")
    pos = Posinp.from_file(surface_filename)
    # Posinp with free boundary conditions
    free_filename = os.path.join(tests_fol, "free.xyz")
    free_pos = Posinp.from_file(free_filename)
    # Posinp read from a string
    string = """\
4   atomic
free
C    0.6661284109   0.000000000   1.153768252
C    3.330642055    0.000000000   1.153768252
C    4.662898877    0.000000000   3.461304757
C    7.327412521    0.000000000   3.461304757"""
    str_pos = Posinp.from_string(string)
    # Posinp read from an N2 calculation of bad quality
    log_pos = Logfile.from_file(logname).posinp

    @pytest.mark.parametrize("value, expected", [
        (pos.n_at, 4), (pos.units, "reduced"), (len(pos), 4),
        (pos.BC, "surface"),
        (pos.cell, [8.07007483423, 1.0, 4.65925987792]),
        (pos[0], Atom('C', [0.08333333333, 0.5, 0.25])),
    ])
    def test_from_file(self, value, expected):
        assert value == expected

    @pytest.mark.parametrize("value, expected", [
        (log_pos.n_at, 2), (log_pos.units, "angstroem"), (len(log_pos), 2),
        (log_pos.BC, "free"),
        (log_pos.cell, None),
        (log_pos[0], Atom('N', [2.9763078243490115e-23, 6.872205952043537e-23, 0.01071619987487793])),  # noqa
    ])
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

    @pytest.mark.parametrize("fname", [
        "free_reduced.xyz", "missing_atom.xyz", "additional_atom.xyz",
    ])
    def test_init_raises_ValueError(self, fname):
        with pytest.raises(ValueError):
            Posinp.from_file(os.path.join(tests_fol, fname))
