"""
File containing the base class defining a BigDFT calculation.
"""

from __future__ import print_function, absolute_import
import os
import shutil
import subprocess
from .globals import BIGDFT_PATH, BIGDFT_TOOL_PATH
from .iofiles import InputParams, Logfile, clean, GeoptLogfile

# Space coordinates
COORDS = ["x", "y", "z"]
# Dictionary to convert the string of the signs to floats
SIGNS = {"+": 1., "-": -1.}


class Job(object):
    r"""
    This class is meant to define a BigDFT calculation. :meth:`run` is
    its main method and it must be used in a context manager.
    """

    def __init__(self, name="", inputparams=None, posinp=None, run_dir=None,
                 ref_job=None, skip=False):
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
        ref_job : Job or None
            Other BigDFT calculation, taken as reference.
        skip : bool
            If `True`, the calculation will be skipped. (Note: Might not
            be useful now, since we check for the existence of the
            logfile before running, which might be the actual check of
            the skip option of BigDFT.)

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

        >>> job.logfile is None
        True

        To run the job, do it from a context manager:

        >>> with job as j:
        ...     j.run()
        ...
        /.../tests
        Logfile log.yaml already exists!
        <BLANKLINE>

        A logfile being found, it is read and not computed again:

        >>> job.logfile is None
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
            raise ValueError(
                "inputparams and posinp do not define the same posinp.")

        # Set the base attributes
        self._inputparams = inputparams
        self._posinp = posinp
        self._logfile = None
        self._ref_job = ref_job
        self._name = str(name)
        self._skip = bool(skip)

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

    @property
    def inputparams(self):
        r"""
        Returns
        -------
        InputParams
            Input parameters of the calculation.
        """
        return self._inputparams

    @property
    def posinp(self):
        r"""
        Returns
        -------
        Posinp or None
            Initial positions of the calculation.
        """
        return self._posinp

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

    @property
    def ref_job(self):
        r"""
        Returns
        -------
        Job
            Job of the reference calculation.
        """
        return self._ref_job

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

    @property
    def run_dir(self):
        r"""
        Returns
        -------
        str
            Absolute path to the directory where the calculation is run.
        """
        return self._run_dir

    @property
    def data_dir(self):
        r"""
        Returns
        -------
        str
            Absolute path to the data directory of the calculation.
        """
        return self._data_dir

    @property
    def bigdft_tool_cmd(self):
        r"""
        Returns
        -------
        list
            Base command to run the bigdft-tool executable.
        """
        return self._bigdft_tool_cmd

    @property
    def bigdft_cmd(self):
        r"""
        Returns
        -------
        list
            Base command to run the bigdft executable.
        """
        return self._bigdft_cmd

    @property
    def input_name(self):
        r"""
        Returns
        -------
        str
            Name of the input parameters file.
        """
        return self._input_name

    @property
    def posinp_name(self):
        r"""
        Returns
        -------
        str
            Name of the input position file.
        """
        return self._posinp_name

    @property
    def logfile_name(self):
        r"""
        Returns
        -------
        str
            Name of the logfile.
        """
        return self._logfile_name

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
        self._init_dir = os.getcwd()
        # Set the directory where the calculation will be run
        if run_dir is None:
            self._run_dir = self.init_dir
        else:
            # A run directory was given, find the common prefix with the
            # current working directory
            basename = os.path.commonprefix([self.init_dir, run_dir])
            if basename == '':
                # If there is no common prefix, then the run directory
                # is already well defined, and the absolute directory is
                # the concatenation of the current working directory and
                # the run directory
                self._run_dir = os.path.join(self.init_dir, run_dir)
            else:
                # Else, find the relative path with the common prefix to
                # define run_dir, and use run_dir to define the
                # absolute directory. The initial directory is changed to the
                # common prefix.
                self._init_dir = basename
                new_run_dir = os.path.relpath(run_dir, start=basename)
                self._run_dir = os.path.join(self.init_dir, new_run_dir)
                # print("run_dir switched from {} to {}"
                #       .format(run_dir, new_run_dir))

    def _set_data_directory(self):
        r"""
        Set the attributes regarding the directories used to store data.
        """
        # Set the data directory
        data_dir = "data"  # base name for the BigDFT data directory
        if self.name != "":
            data_dir += '-'+self.name
        self._data_dir = os.path.join(self.run_dir, data_dir)

    def _set_cmd_attributes(self):
        r"""
        Set the base commands to run bigdft or bigdft-tool.
        """
        # The base bigdft-tool command is always the same
        self._bigdft_tool_cmd = [BIGDFT_TOOL_PATH, "--name", self.name]
        # The base bigdft command depends on name and on skip
        skip_option = []
        if self.skip:
            skip_option += ["-s", "Yes"]
        if self.name != "":
            self._bigdft_cmd = [BIGDFT_PATH, self.name] + skip_option
        else:
            self._bigdft_cmd = [BIGDFT_PATH] + skip_option

    def _set_filename_attributes(self):
        r"""
        Set the attributes regarding the name of the input and output
        files.
        """
        if self.name != "":
            self._input_name = self.name+".yaml"  # input file name
            self._posinp_name = self.name+".xyz"  # posinp file name
            self._logfile_name = "log-"+self.input_name  # output file name
        else:
            self._input_name = "input.yaml"  # input file name
            self._posinp_name = "posinp.xyz"  # posinp file name
            self._logfile_name = "log.yaml"  # output file name

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

    def run(self, nmpi=1, nomp=1, force_run=False, dry_run=False):
        r"""
        Run the BigDFT calculation if it was not already performed.

        You may force the calculation by setting force_run to `True`.

        If `dry_run` is set to `True`, then bigdft-tool is run instead.

        The number of MPI and OpenMP tasks may also be specified.

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
        """
        # Copy the data directory of a reference calculation
        if self.ref_job is not None:
            self._copy_reference_data_dir()

        # Update the input file, so that it reads the reference
        # wavefunctions in the data directory
        if os.path.exists(self.data_dir):
            self._read_wavefunctions_from_data_dir()

        if dry_run or force_run or not os.path.exists(self.logfile_name):
            # Run bigdft (if dry_run is False) or bigdft-tool (if
            # dry_run is True)
            self._set_environment(nomp)
            self.write_input_files()
            command = self._get_command(nmpi, dry_run)
            output_msg = self._launch_calculation(command)
            if dry_run:
                self._write_bigdft_tool_output(output_msg)
            else:
                print(output_msg)
            self._logfile = Logfile.from_file(self.logfile_name)
        else:
            # The logfile already exists: the initial positions and the
            # initial parameters used to perform that calculation must
            # correspond to the ones used to initialize the current job.
            print("Logfile {} already exists!\n".format(self.logfile_name))
            self._logfile = Logfile.from_file(self.logfile_name)
            self._check_logfile_posinp()
            self._check_logfile_inputparams()

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
                "Logfile posinp:\n{}Actual posinp:\n{}"
                .format(log_pos, self.posinp))

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
        # Clean the disablesym key, if present only in the log_inp
        if 'dft' in log_inp and 'disablesym' in log_inp['dft']:
            if 'dft' in base_inp and 'disablesym' not in base_inp['dft']:
                del log_inp['dft']['disablesym']
                log_inp._params = clean(log_inp.params)
        if base_inp != log_inp:
            raise UserWarning(
                "The input parameters of this job do not correspond to the "
                "ones used in the Logfile.", UserWarning)

    def _copy_reference_data_dir(self):
        r"""
        Copy the reference data directory to the current calculation
        directory so as to restart the new calculation from the result
        of the reference calculation.
        """
        ref = self.ref_job
        if os.path.exists(ref.data_dir):
            if os.path.basename(self.data_dir) in os.listdir(os.curdir):
                # Remove the previously existing data directory before
                # copying the reference data directory (otherwise,
                # shutil.copytree raises an error).
                shutil.rmtree(self.data_dir)
            shutil.copytree(ref.data_dir, self.data_dir)
        else:
            print("Data directory not found for reference calculation.")

    def _read_wavefunctions_from_data_dir(self):
        r"""
        Set the input parameters to read the wavefunctions from the data
        directory if they exist.
        """
        # Check that there are wavefunction files
        wf_files = [f for f in os.listdir(self.data_dir)
                    if 'wavefunction' in f]
        if wf_files:
            # If there are wavefunction files, add the
            # option to read them from files.
            try:
                self.inputparams['dft']['inputpsiid'] = 2
            except KeyError:
                self.inputparams['dft'] = {'inputpsiid': 2}
        else:
            # Else, delete the option from the input file, if
            # it is equal to 2 (might be better than completely
            # removing inputpsiid ?).
            try:
                if self.inputparams['dft']['inputpsiid'] == 2:
                    del self.inputparams['dft']['inputpsiid']
            except Exception:
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
                mpi_option = ['-n', str(nmpi)]
            command = self.bigdft_tool_cmd + mpi_option
        else:
            if nmpi > 1:
                mpi_option = ['mpirun', '-np', str(nmpi)]
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

    @staticmethod
    def _launch_calculation(command):
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
        to_str = "{} "*len(command)
        command_msg = to_str.format(*command)+"..."
        print(command_msg)
        # Run the calculation
        run = subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        # Raise an error if the calculation ended badly, else return the
        # decoded output message
        out, err = run.communicate()
        error_msg = err.decode('unicode_escape')
        if error_msg != '':
            raise RuntimeError(
                "The calculation ended with the following error message:{}"
                .format(error_msg))
        output_msg = out.decode('unicode_escape')
        return output_msg

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
        filenames = [self.logfile_name, self.input_name, self.posinp_name,
                     "forces_"+self.posinp_name, "forces.xyz",
                     "input_minimal.yaml", "time.yaml"]
        for filename in filenames:
            try:
                os.remove(filename)
            except OSError:
                pass
        # Delete the required directories
        directories = []
        if data_dir:
            directories += ["data", "data-"+self.name]
        if logfiles_dir:
            directories += ["logfiles"]
        for directory in directories:
            shutil.rmtree(directory, ignore_errors=True)
