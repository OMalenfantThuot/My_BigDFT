"""
The :class:`Job` class is the base class defining a BigDFT calculation.
"""

from __future__ import print_function, absolute_import
import os
import shutil
import subprocess
from threading import Timer
from copy import deepcopy
from mybigdft.iofiles import InputParams, Logfile
from mybigdft.iofiles.logfiles import GeoptLogfile
from mybigdft.iofiles.inputparams import clean
from .globals import BIGDFT_PATH, BIGDFT_TOOL_PATH, DEFAULT_PARAMETERS


class Job(object):
    r"""
    This class is meant to define a BigDFT calculation. :meth:`run` is
    its main method and it must be used in a context manager to ensure
    that the calculation is run the desired directory.
    """

    def __init__(
        self,
        name="",
        inputparams=None,
        posinp=None,
        run_dir=None,
        ref_data_dir=None,
        skip=False,
        pseudos=False,
    ):
        r"""
        You may pass input parameters and/or initial geometry (posinp).
        Make sure to at least provide initial positions, either via the
        posinp or the input parameters.

        You may give a `name` for the calculation, used to name the
        input and output files written on disk (default naming
        conventions are used if not). You can also specify the directory
        where to run the calculation with `run_dir`.

        A reference calculation may be given in order to copy its data
        directory to the present calculation (main use: restart from the
        wavefunctions of the reference calculation).

        Parameters
        ----------
        inputparams : InputParams or None
            BigDFT input parameters.
        posinp : Posinp or None
            BigDFT initial geometry file.
        name : str
            Prefix of the BigDFT calculation (used to define the input
            and output file names).
        run_dir : str or None
            Folder where to run the calculation (default to current
            directory).
        ref_data_dir : str
            Path to the data directory of a reference BigDFT
            calculation.
        skip : bool
            If `True`, the calculation will be skipped. (Note: Might not
            be useful now, since we check for the existence of the
            logfile before running, which might be the actual check of
            the skip option of BigDFT.)
        pseudos : bool
            If `True`, the pseudopotential files stored in $PSEUDODIR
            will be used to complete the job.

        Raises
        ------
        ValueError
            If no initial positions are given in the posinp or the input
            parameters.


        A Job instance can be initialized by using a posinp only:

        >>> from mybigdft import Posinp, Atom
        >>> pos = Posinp(
        ...     [Atom('N', [2.9763078243490115e-23, 6.872205952043537e-23,
        ...                 0.01071619987487793]),
        ...      Atom('N', [-1.1043449194501671e-23, -4.873421744830746e-23,
        ...                 1.104273796081543])], "angstroem", "free"
        ... )
        >>> job = Job(posinp=pos, run_dir="tests")

        Default values are therefore used for the input parameters:

        >>> job.inputparams
        {}

        Input and output file names are defined from the `name` passed
        as argument. Here, no name is passed, so that default names are
        used:

        >>> job.input_name
        'input.yaml'
        >>> job.posinp_name
        'posinp.xyz'
        >>> job.logfile_name
        'log.yaml'

        The directories are defined from the `run_dir` argument:

        >>> import os
        >>> os.getcwd() == job.init_dir
        True
        >>> os.path.basename(job.init_dir) != 'tests'
        True
        >>> os.path.basename(job.run_dir)
        'tests'

        There is no logfile associated to the job yet as it was not run:

        >>> job.logfile == {}
        True

        To run the job, do it from a context manager:

        >>> with job as j:
        ...     j.run()
        ...
        /.../tests
        Logfile log.yaml already exists!
        <BLANKLINE>

        A logfile being found, it is read and not computed again:

        >>> job.logfile == {}
        False
        """
        # Check the input parameters of the calculation
        if inputparams is None:
            inputparams = InputParams()
        if posinp is None:
            posinp = inputparams.posinp
        if inputparams.posinp is None and posinp is None:
            raise ValueError("Please provide initial positions.")
        elif inputparams.posinp is not None and posinp != inputparams.posinp:
            raise ValueError("inputparams and posinp do not define the same posinp.")

        # Set the base attributes
        inputparams.params = clean(inputparams.params)
        self.inputparams = inputparams
        self.posinp = posinp
        self.logfile = Logfile()
        self.ref_data_dir = ref_data_dir
        self.name = name
        self.skip = skip
        self.is_completed = False
        self.pseudos = pseudos
        if self.pseudos:
            try:
                self.inputparams["dft"]["ixc"] = -101130
            except KeyError:
                self.inputparams["dft"] = {"ixc": -101130}

        # Derive the rest of the attributes from the other arguments
        self._set_directory_attributes(run_dir)
        self._set_filename_attributes()
        self._set_cmd_attributes()

    @property
    def name(self):
        r"""
        Returns
        -------
        str
            Base name of the calculation used to set the names of
            files and directories as well as the commands.
        """
        return self._name

    @name.setter
    def name(self, name):
        self._name = str(name)

    @property
    def inputparams(self):
        r"""
        Returns
        -------
        InputParams
            Input parameters of the calculation.
        """
        return self._inputparams

    @inputparams.setter
    def inputparams(self, inputparams):
        self._inputparams = deepcopy(inputparams)

    @property
    def posinp(self):
        r"""
        Returns
        -------
        Posinp or None
            Initial positions of the calculation.
        """
        return self._posinp

    @posinp.setter
    def posinp(self, posinp):
        self._posinp = posinp

    @property
    def logfile(self):
        r"""
        Returns
        -------
        Logfile or None
            Logfile of the calculation (output of the bigdft or
            bigdft-tool executable).
        """
        return self._logfile

    @logfile.setter
    def logfile(self, logfile):
        self._logfile = logfile

    @property
    def ref_data_dir(self):
        r"""
        Returns
        -------
        str
            Reference directory where some relevant data (such as
            wavefunctions) is stored.
        """
        return self._ref_data_dir

    @ref_data_dir.setter
    def ref_data_dir(self, ref_data_dir):
        self._ref_data_dir = ref_data_dir

    @property
    def pseudos(self):
        r"""
        Returns
        -------
        bool
            if `True`, the calculation uses the pseudopotential files
            in $PSEUDODIR (environment variable).
        """
        return self._pseudos

    @pseudos.setter
    def pseudos(self, pseudos):
        self._pseudos = pseudos

    @property
    def skip(self):
        r"""
        Returns
        -------
        bool
            If `True`, the calculation will be skipped. (Note: Might not
            be useful now, since we check for the existence of the
            logfile before running, which might be the actual check of
            the skip option of BigDFT.)
        """
        return self._skip

    @skip.setter
    def skip(self, skip):
        self._skip = bool(skip)

    @property
    def init_dir(self):
        r"""
        Returns
        -------
        str
            Absolute path to the initial directory of the calculation
            (can differ from :meth:`~mybigdft.job.Job.run_dir`).
        """
        return self._init_dir

    @init_dir.setter
    def init_dir(self, init_dir):
        self._init_dir = init_dir

    @property
    def run_dir(self):
        r"""
        Returns
        -------
        str
            Absolute path to the directory where the calculation is run.
        """
        return self._run_dir

    @run_dir.setter
    def run_dir(self, run_dir):
        self._run_dir = run_dir

    @property
    def data_dir(self):
        r"""
        Returns
        -------
        str
            Absolute path to the data directory of the calculation.
        """
        return self._data_dir

    @data_dir.setter
    def data_dir(self, data_dir):
        self._data_dir = data_dir

    @property
    def bigdft_tool_cmd(self):
        r"""
        Returns
        -------
        list
            Base command to run the bigdft-tool executable.
        """
        return self._bigdft_tool_cmd
    
    @bigdft_tool_cmd.setter
    def bigdft_tool_cmd(self, bigdft_tool_cmd):
        self._bigdft_tool_cmd = bigdft_tool_cmd

    @property
    def bigdft_cmd(self):
        r"""
        Returns
        -------
        list
            Base command to run the bigdft executable.
        """
        return self._bigdft_cmd

    @bigdft_cmd.setter
    def bigdft_cmd(self, bigdft_cmd):
        self._bigdft_cmd = bigdft_cmd

    @property
    def input_name(self):
        r"""
        Returns
        -------
        str
            Name of the input parameters file.
        """
        return self._input_name

    @input_name.setter
    def input_name(self, input_name):
        self._input_name = input_name

    @property
    def posinp_name(self):
        r"""
        Returns
        -------
        str
            Name of the input position file.
        """
        return self._posinp_name

    @posinp_name.setter
    def posinp_name(self, posinp_name):
        self._posinp_name = posinp_name

    @property
    def logfile_name(self):
        r"""
        Returns
        -------
        str
            Name of the logfile.
        """
        return self._logfile_name

    @logfile_name.setter
    def logfile_name(self, logfile_name):
        self._logfile_name = logfile_name

    @property
    def is_completed(self):
        r"""
        Returns
        -------
        bool
            `True` if the job has already run successfully.
        """
        return self._is_completed

    @is_completed.setter
    def is_completed(self, is_completed):
        self._is_completed = is_completed

    def _set_directory_attributes(self, run_dir):
        r"""
        Set the attributes regarding the directories used to run the
        calculation and to store data.

        Parameters
        ----------
        run_dir : str or None
            Folder where to run the calculation.
        """
        self._set_init_and_run_directories(run_dir)
        self._set_data_directory()

    def _set_init_and_run_directories(self, run_dir):
        r"""
        Set the attributes regarding the directories used to run the
        calculation.

        Parameters
        ----------
        run_dir : str or None
            Folder where to run the calculation.
        """
        # Set the initial directory
        self.init_dir = os.getcwd()
        # Set the directory where the calculation will be run
        if run_dir is None:
            self.run_dir = self.init_dir
        else:
            # A run directory was given, find the common prefix with the
            # current working directory
            basename = os.path.commonprefix([self.init_dir, run_dir])
            if basename == "":
                # If there is no common prefix, then the run directory
                # is already well defined, and the absolute directory is
                # the concatenation of the current working directory and
                # the run directory
                self.run_dir = os.path.join(self.init_dir, run_dir)
            else:
                # Else, find the relative path with the common prefix to
                # define run_dir, and use run_dir to define the
                # absolute directory. The initial directory is changed to the
                # common prefix.
                self.init_dir = basename
                new_run_dir = os.path.relpath(run_dir, start=basename)
                self.run_dir = os.path.join(self.init_dir, new_run_dir)
                # print("run_dir switched from {} to {}"
                #       .format(run_dir, new_run_dir))

    def _set_data_directory(self):
        r"""
        Set the attributes regarding the directories used to store data.
        """
        # Set the data directory
        data_dir = "data"  # base name for the BigDFT data directory
        if self.name != "":
            data_dir += "-" + self.name
        self.data_dir = os.path.join(self.run_dir, data_dir)

    def _set_cmd_attributes(self):
        r"""
        Set the base commands to run bigdft or bigdft-tool.
        """
        # The base bigdft-tool command is always the same
        self.bigdft_tool_cmd = [BIGDFT_TOOL_PATH]
        if self.name:
            self.bigdft_tool_cmd += ["--name", self.name]
        # The base bigdft command depends on name and on skip
        skip_option = []
        if self.skip:
            skip_option += ["-s", "Yes"]
        if self.name != "":
            self.bigdft_cmd = [BIGDFT_PATH, self.name] + skip_option
        else:
            self.bigdft_cmd = [BIGDFT_PATH] + skip_option

    def _set_filename_attributes(self):
        r"""
        Set the attributes regarding the name of the input and output
        files.
        """
        if self.name != "":
            self.input_name = self.name + ".yaml"  # input file name
            self.posinp_name = self.name + ".xyz"  # posinp file name
            self.logfile_name = "log-" + self.input_name  # output file name
        else:
            self.input_name = "input.yaml"  # input file name
            self.posinp_name = "posinp.xyz"  # posinp file name
            self.logfile_name = "log.yaml"  # output file name

    def __enter__(self):
        r"""
        When entering the context manager:

        * create the directory where the calculations must be run,
        * go to that directory.
        """
        if self.run_dir not in [".", ""]:
            if not os.path.exists(self.run_dir):
                os.makedirs(self.run_dir)
            os.chdir(self.run_dir)
        print(os.getcwd())
        return self

    def __exit__(self, *args):
        r"""
        When leaving the context manager, go back to the initial
        directory.
        """
        os.chdir(self.init_dir)

    def run(
        self,
        nmpi=1,
        nomp=1,
        force_run=False,
        dry_run=False,
        restart_if_incomplete=False,
        timeout=None,
    ):
        r"""
        Run the BigDFT calculation if it was not already performed.
        The number of MPI and OpenMP tasks may be specified.

        You may force the calculation to run even though it was
        previously successful (*e.g.*, a logfile already exists) by
        setting `force_run` to `True`.

        If `dry_run` is set to `True`, then bigdft-tool is run instead
        of the BigDFT executable.

        If `restart_if_incomplete` is set to `True`, the previously
        existing logfile is removed and the calculation restarts.

        Parameters
        ----------
        nmpi : int
            Number of MPI tasks.
        nomp : int
            Number of OpenMP tasks.
        force_run : bool
            If `True`, the calculation is run even though a logfile
            already exists.
        dry_run : bool
            If `True`, the input files are written on disk, but the
            bigdft-tool command is run instead of the bigdft one.
        restart_if_incomplete : bool
            If `True`, the job is restarted if the existing logfile is
            incomplete.
        timeout : float or int or None
            Number of minutes after which the job must be stopped.
        """
        # Copy the data directory of a reference calculation
        if self.ref_data_dir is not None:
            # Copy the data directory only when bigdft has to run
            if force_run or not os.path.exists(self.logfile_name):
                self._copy_reference_data_dir()
            # Always update the input file, so that it reads the
            # reference wavefunctions in the data directory
            if os.path.exists(self.data_dir):
                self._read_wavefunctions_from_data_dir()

        if dry_run or force_run or not os.path.exists(self.logfile_name):
            # Run bigdft (if dry_run is False) or bigdft-tool (if
            # dry_run is True)
            self._set_environment(nomp)
            self.write_input_files()
            command = self._get_command(nmpi, dry_run)
            output_msg = self._launch_calculation(command, timeout)
            if dry_run:
                self._write_bigdft_tool_output(output_msg)
            else:
                output_msg = output_msg.decode("unicode_escape")
                print(output_msg)
            try:
                self.logfile = Logfile.from_file(self.logfile_name)
            except ValueError as e:
                if str(e) == "The logfile is incomplete!":
                    raise RuntimeError("Timeout exceded ({} minutes)".format(timeout))
            if os.path.exists(self.data_dir):
                self._clean_data_dir()
        else:
            # The logfile already exists: the initial positions and the
            # initial parameters used to perform that calculation must
            # correspond to the ones used to initialize the current job.
            print("Logfile {} already exists!\n".format(self.logfile_name))
            try:
                self.logfile = Logfile.from_file(self.logfile_name)
            except ValueError as e:
                incomplete_log = str(e) == "The logfile is incomplete!"
                if incomplete_log and restart_if_incomplete:
                    # Remove the logfile and restart the calculation
                    print("The logfile was incomplete, restart calculation")
                    os.remove(self.logfile_name)
                    self.run(
                        nmpi=nmpi,
                        nomp=nomp,
                        force_run=force_run,
                        dry_run=dry_run,
                        restart_if_incomplete=False,
                        timeout=timeout,
                    )
                else:
                    raise e
            else:
                self._check_logfile_posinp()
                self._check_logfile_inputparams()
        self.is_completed = True

    def _copy_reference_data_dir(self):
        r"""
        Copy the reference data directory to the current calculation
        directory so as to restart the new calculation from the result
        of the reference calculation.
        """
        if os.path.exists(self.ref_data_dir):
            if os.path.basename(self.data_dir) in os.listdir(os.curdir):
                # Remove the previously existing data directory before
                # copying the reference data directory (otherwise,
                # shutil.copytree raises an error).
                shutil.rmtree(self.data_dir)
            shutil.copytree(self.ref_data_dir, self.data_dir)
            print("Data directory copied from {}.".format(self.ref_data_dir))
        else:
            print("Data directory {} not found.".format(self.ref_data_dir))

    def _read_wavefunctions_from_data_dir(self):
        r"""
        Set the input parameters to read the wavefunctions from the data
        directory if they exist.
        """
        # Check that there are wavefunction files
        wf_files = [f for f in os.listdir(self.data_dir) if "wavefunction" in f]
        if wf_files:
            # If there are wavefunction files, add the
            # option to read them from files.
            try:
                self.inputparams["dft"]["inputpsiid"] = 2
            except KeyError:
                self.inputparams["dft"] = {"inputpsiid": 2}
        else:
            # Else, delete the option from the input file, if
            # it is equal to 2 (might be better than completely
            # removing inputpsiid ?).
            try:
                if self.inputparams["dft"]["inputpsiid"] == 2:
                    del self.inputparams["dft"]["inputpsiid"]
            except KeyError:
                pass

    @staticmethod
    def _set_environment(nomp):
        r"""
        Set the number of OpenMP threads.

        Parameters
        ----------
        nomp : int
            Number of OpenMP tasks.
        """
        nomp = int(nomp)  # Make sure you get an integer
        if nomp > 1:
            os.environ["OMP_NUM_THREADS"] = str(nomp)

    def _get_command(self, nmpi, dry_run):
        r"""
        Returns
        -------
        command : list
            The command to run bigdft if `dry_run` is set to `False`,
            else the command to run bigdft-tool.

        Parameters
        ----------
        nmpi : int
            Number of MPI tasks.
        dry_run : bool
            If `True`, the input files are written on disk, but the
            bigdft-tool command is run instead of the bigdft one.
        """
        nmpi = int(nmpi)  # Make sure you get an integer
        mpi_option = []
        if dry_run:
            if nmpi > 1:
                mpi_option = ["-n", str(nmpi)]
            command = self.bigdft_tool_cmd + mpi_option
        else:
            if nmpi > 1:
                mpi_option = ["mpirun", "-np", str(nmpi)]
            command = mpi_option + self.bigdft_cmd
        return command

    def write_input_files(self):
        r"""
        Write the input files on disk (there might be no posinp to write,
        since the initial positions can be defined in the input
        parameters).
        """
        self.inputparams.write(self.input_name)
        if self.posinp is not None:
            self.posinp.write(self.posinp_name)
        if self.pseudos:
            elements = set([atom.type for atom in self.posinp])
            for element in elements:
                shutil.copyfile(
                    os.environ["PSEUDODIR"] + "psppar." + element, "psppar." + element
                )

    @staticmethod
    def _launch_calculation(command, timeout):
        r"""
        Launch the command to run the bigdft or bigdft-tool command.

        Parameters
        ----------
        command : list
            The command to run bigdft or bigdft-tool.

        Raises
        ------
        RuntimeError
            If the calculation ended with an error message.
        """
        # Print the command in a human readable way
        to_str = "{} " * len(command)
        command_msg = to_str.format(*command) + "..."
        print(command_msg)
        # Run the calculation for at most timeout minutes
        run = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if timeout is None:
            # 60 years timeout should be enough...
            timeout = 60 * 365 * 24 * 60
        timer = Timer(timeout * 60, run.kill)
        try:
            timer.start()
            out, err = run.communicate()
            error_msg = err.decode("unicode_escape")
        finally:
            timer.cancel()
        # Raise an error if the calculation ended badly, else return the
        # decoded output message
        if error_msg != "":
            raise RuntimeError(
                "The calculation ended with the following error message:{}".format(
                    error_msg
                )
            )
        return out

    def _write_bigdft_tool_output(self, output_msg):
        r"""
        Write the output of the bigdft-tool command on disk.

        Parameters
        ----------
        output_msg : str
            Output of the bigdft-tool command as a Logfile.
        """
        log = Logfile.from_stream(output_msg)
        log.write(self.logfile_name)

    def _clean_data_dir(self):
        r"""
        Clean the data directory, namely delete the wavefunctions in
        the data folder if it was not requested to output them, and
        delete the output files of a geopt calculation if a geopt
        was not performed.
        """
        # Delete the wavefunction files in the data directory and
        # replace them by empty files if needed.
        inp = self.inputparams
        default = DEFAULT_PARAMETERS["output"]["orbitals"]
        write_orbitals = (
            "output" in inp
            and "orbitals" in inp["output"]
            and inp["output"]["orbitals"] != default
        )
        if "output" not in inp or not write_orbitals:
            wf_files = [
                os.path.join(self.data_dir, filename)
                for filename in os.listdir(self.data_dir)
                if filename.startswith("wavefunction")
            ]
            for wf_file in wf_files:
                os.remove(wf_file)
                # Equivalent to touch wf_file in bash
                with open(wf_file, "a"):
                    os.utime(wf_file, None)
        # Delete geopt data if no geopt was required
        if "geopt" not in inp:
            # Delete the posout files
            posout_files = [
                os.path.join(self.data_dir, filename)
                for filename in os.listdir(self.data_dir)
                if filename.startswith("posout")
            ]
            for posout_file in posout_files:
                os.remove(posout_file)
            # Delete the geopt.mon file
            try:
                os.remove(os.path.join(self.data_dir, "geopt.mon"))
            except OSError:
                pass

    def _check_logfile_posinp(self):
        r"""
        Check that the posinp used in the logfile corresponds to the one
        used to initialize the job.

        Raises
        ------
        UserWarning
            If the initial geometry of the job does not correspond to
            the one of the Logfile previously read from the disk.
        """
        if isinstance(self.logfile, GeoptLogfile):
            log_pos = self.logfile.posinps[0]
        else:
            log_pos = self.logfile.posinp
        if log_pos != self.posinp:
            raise UserWarning(
                "The initial geometry of this job do not correspond to the "
                "one used in the Logfile:\n"
                "Logfile posinp:\n{}Actual posinp:\n{}".format(log_pos, self.posinp)
            )

    def _check_logfile_inputparams(self):
        r"""
        Check that the input parameters used in the logfile correspond
        to the ones used to initialize the job.

        Raises
        ------
        UserWarning
            If the input parameters of the job does not correspond to
            the one used in the Logfile previously read from the disk.
        """
        log_inp = self.logfile.inputparams
        base_inp = self.inputparams
        # Clean the disablesym key:
        disablesym_in_log_inp = "dft" in log_inp and "disablesym" in log_inp["dft"]
        disablesym_not_in_log_inp = (
            "dft" in log_inp and "disablesym" not in log_inp["dft"]
        )
        disablesym_in_base_inp = "dft" in base_inp and "disablesym" in base_inp["dft"]
        disablesym_not_in_base_inp = (
            "dft" in base_inp and "disablesym" not in base_inp["dft"]
        )
        # - if present only in the log_inp
        if disablesym_in_log_inp and disablesym_not_in_base_inp:
            del log_inp["dft"]["disablesym"]
            log_inp._params = clean(log_inp.params)
        # - if present only in the base_inp
        if disablesym_not_in_log_inp and disablesym_in_base_inp:
            del base_inp["dft"]["disablesym"]
            base_inp._params = clean(log_inp.params)
        if base_inp != log_inp:
            raise UserWarning(
                "The input parameters of this job do not correspond to the "
                "ones used in the Logfile:\n"
                "Logfile input parameters:\n{}\nActual input parameters:\n{}".format(
                    log_inp, base_inp
                )
            )

    def clean(self, data_dir=False, logfiles_dir=False):
        r"""
        Delete all input and output files on disk as well as some
        directories if required.

        Parameters
        ----------
        data_dir : bool
            If `True`, removes the data directory that might exist.
        logfiles : bool
            If `True`, removes the logfiles directory that might exist.


        .. Warning::

            The directories are forced to be removed when the above-
            mentioned options are set to `True`: use with caution.
        """
        # Delete the input and output files
        filenames = [
            self.logfile_name,
            self.input_name,
            self.posinp_name,
            "forces_" + self.posinp_name,
            "forces.xyz",
            "time.yaml",
            "input_minimal.yaml",
            self.name + "_minimal.yaml",
        ]
        for filename in filenames:
            try:
                os.remove(filename)
            except OSError:
                pass
        # Delete the required directories
        directories = []
        if data_dir:
            directories += ["data", "data-" + self.name]
        if logfiles_dir:
            directories += ["logfiles"]
        for directory in directories:
            shutil.rmtree(directory, ignore_errors=True)
