"""
File containing the base class defining a BigDFT calculation.
"""

from __future__ import print_function
import os
import shutil
import subprocess
from .globals import bigdft_path, bigdft_tool_path
from .iofiles import Logfile

# Name for the BigDFT data directory
DATA = "data"
# Space coordinates
COORDS = ["x", "y", "z"]
# Dictionary to convert the string of the signs to floats
SIGNS = {"+": 1., "-": -1.}


class Job(object):

    def __init__(self, name="", inputparams=None, posinp=None, run_dir=None,
                 ref_calc=None, skip=False):
        r"""
        You may pass input parameters and/or initial geometry (posinp).
        Make sure to at list provide initial positions, either via the
        posinp or the input parameters.

        You may give a prefix to name the output files and specify the
        directory where to run the calculation.

        A reference calculation may be given in order to copy its data
        directory to the present calculation (main use: restart from the
        wavefunctions of the reference calculation).

        :param inputparams: BigDFT input file.
        :type inputparams: ~mybigdft.iofiles.InputParams
        :param posinp: BigDFT initial geometry file.
        :type posinp: ~mybigdft.iofiles.Posinp
        :param name: Prefix of the BigDFT calculation (optional,
            default value set to None).
        :type name: str
        :param run_dir: Folder where to run the calculation
            (optional, default value set to None).
        :type run_dir: str
        :param ref_calc: Other BigDFT calculation, taken as reference
            (optional, default value set to None)
        :type ref_calc: ~mybigdft.job.Job
        :param skip: If True, the calculation will be skipped. (Note:
            Might not be useful now, since we check for the existence
            of the logfile before running, which might be the actual
            check of the skip option of BigDFT)
        :type skip: bool
        """
        # Check the input parameters of the calculation
        if (inputparams is None and posinp is None) or \
           (posinp is None and "posinp" not in inputparams):
            raise ValueError("Please provide initial positions.")
        elif inputparams is None:
            inputparams = {}

        # Set the initial directory
        self.init_dir = os.getcwd()
        # Set the directory where the calculation will be run
        if run_dir is None:
            self.run_dir = os.path.curdir
            self.abs_dir = self.init_dir
        else:
            # A run directory was given, find the common prefix with the
            # current working directory
            basename = os.path.commonprefix([self.init_dir, run_dir])
            if basename == '':
                # If there is no common prefix, then the run directory
                # is already well defined, and the absolute directory is
                # the concatenation of the current working directory and
                # the run directory
                self.run_dir = run_dir
                self.abs_dir = os.path.join(self.init_dir, run_dir)
            else:
                # Else, find the relative path with the common prefix to
                # define run_dir, and use run_dir to define the
                # absolute directory. The initial directory is changed to the
                # common prefix.
                self.init_dir = basename
                self.run_dir = os.path.relpath(run_dir, start=basename)
                print("Switch run_dir from {} to {}"
                      .format(run_dir, self.run_dir))
                self.abs_dir = run_dir

        # Set the attributes
        self.inputparams = inputparams
        self.name = name
        self.posinp = posinp
        self.logfile = None
        self.ref_calc = ref_calc

        # Initialize some file and directory names and also BigDFT commands
        skip_option = []
        if skip:
            skip_option += ["-s", "Yes"]
        if name != "":
            self.input_name = name+".yaml"  # input file name
            self.logfile_name = "log-"+self.input_name  # output file name
            self.posinp_name = name+".xyz"  # posinp file name
            self.bigdft_cmd = [bigdft_path, name] + skip_option
            self.bigdft_tool_cmd = [bigdft_tool_path]
            self.data_dir = DATA+'-'+name  # data directory
        else:
            self.input_name = "input.yaml"  # input file name
            self.logfile_name = "log.yaml"  # output file name
            self.posinp_name = "posinp.xyz"  # posinp file name
            self.bigdft_cmd = [bigdft_path] + skip_option
            self.bigdft_tool_cmd = [bigdft_tool_path]
            self.data_dir = DATA  # Data directory

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
        When leaving the context manager:

        * Read the logfile (if not a dry run),
        * Go back to the initial directory.
        """
        self.logfile = Logfile.from_file(self.logfile_name)
        os.chdir(self.init_dir)

    def run(self, nmpi=1, nomp=1, force_run=False, dry_run=False):
        r"""
        Run the BigDFT calculation if it was not already performed.

        You may force the calculation by setting force_run to True.

        If dry_run is set to True, then bigdft-tool is run instead.

        The number of MPI and OpenMP tasks may also be specified.

        :param nmpi: Number of MPI tasks.
        :type nmpi: int
        :param nomp: Number of OpenMP tasks.
        :type nomp: int
        :param force_run: States if the calculation has to be run,
            even though a logfile already exists.
        :type force_run: bool
        :param dry_run: If True, the input files are written on disk,
            but the bigdft-tool command is run instead of the bigdft
            one.
        :type dry_run: bool
        """
        # Copy the data directory of a reference calculation
        if self.ref_calc is not None:
            self._copy_reference_data_dir()

        # Update the input file, so that it reads the reference
        # wavefunctions in the data directory
        if os.path.exists(self.data_dir):
            self._read_wavefunctions_from_data_dir()

        # Run bigdft or bigdft-tool when asked or when the logfile does
        # not exist (meaning that the calculation was not already done)
        if dry_run or force_run or not os.path.exists(self.logfile_name):
            self._set_environment(nomp)
            self._write_input_files(nmpi, dry_run)
            command = self._get_command(nmpi, dry_run)
            output_msg = self._launch_calculation(command)
            if dry_run:
                self._after_dry_run(output_msg)
            else:
                print(output_msg)
        else:
            print("Logfile {} already exists!\n".format(self.logfile_name))

    def _copy_reference_data_dir(self):
        r"""
        Copy the reference data directory to the current calculation
        directory so as to restart the new calculation from the result
        of the reference calculation.
        """
        # Find and copy the path to the reference data directory
        ref = self.ref_calc
        ref_data_path = os.path.join(ref.abs_dir, ref.data_dir)
        if os.path.exists(ref_data_path):
            # Remove the previously existing data directory before
            # copying the reference data directory (otherwise,
            # shutil.copytree raises an error).
            if self.data_dir in os.listdir("."):
                shutil.rmtree(self.data_dir)
            shutil.copytree(ref_data_path, self.data_dir)
        elif not os.path.exists(ref_data_path):
            print("Data directory not found for reference calculation.")

    def _read_wavefunctions_from_data_dir(self):
        r"""
        Set the input parameters to read the wavefunctions from the data
        directory if they exist.
        """
        # Check that there are wavefunction files
        wf_files = [f for f in os.listdir(self.data_dir)
                    if 'wavefunction' in f]
        if len(wf_files) > 0:
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

    def _set_environment(self, nomp):
        r"""
        Set the number of OpenMP threads.

        :param nomp: Number of OpenMP tasks.
        :type nomp: int
        """
        nomp = int(nomp)  # Make sure you get an integer
        if nomp > 1:
            os.environ["OMP_NUM_THREADS"] = str(nomp)

    def _get_command(self, nmpi, dry_run):
        r"""
        :returns: The command to run bigdft if dry_run is set to False,
            else the command to run bigdft-tool.
        :rtype: list
        :param nmpi: Number of MPI tasks.
        :type nmpi: int
        :param dry_run: If True, the input files are written on disk,
            but the bigdft-tool command is run instead of the bigdft
            one.
        :type dry_run: bool
        """
        nmpi = int(nmpi)  # Make sure you get an integer
        mpi_command = []
        if dry_run:
            if nmpi > 1:
                mpi_command = ['-n', str(nmpi)]
            command = self.bigdft_tool_cmd + mpi_command
        else:
            if nmpi > 1:
                mpi_command = ['mpirun', '-np', str(nmpi)]
            command = mpi_command + self.bigdft_cmd
        return command

    def _write_input_files(self, nmpi, dry_run):
        r"""
        Write the input files on disk (there might be no posinp to write,
        since the input positions can be defined in the input).

        :param nmpi: Number of MPI tasks (default value set to 1).
        :type nmpi: int
        :param dry_run: If True, the input files are written on disk,
            but the bigdft-tool command is run instead of the bigdft
            one.
        :type dry_run: bool
        """
        # Set the name of the input files to write on disk
        if dry_run:
            # Use default names to create dummy files. They will be
            # deleted after the bigdft-tool command is run.
            inpname = "input.yaml"
            posname = "posinp.xyz"
            self._dummy_files = [inpname, posname]
        else:
            inpname = self.input_name
            posname = self.posinp_name

        # Write the input files (if needed)
        self.inputparams.write(inpname)
        if self.posinp is not None:
            self.posinp.write(posname)
        elif dry_run:
            # There is no dummy posinp file to write on disk, so it will
            # not have to be deleted after bigdft-tool is run
            self._dummy_files.pop()

    def _launch_calculation(self, command):
        r"""
        Launch the command to run the bigdft or bigdft-tool command.

        :param command: The command to run bigdft or bigdft-tool.
        :rtype: list
        """
        # Print the command in a human readable way
        to_str = "{} "*len(command)
        command_msg = to_str.format(*command)+"..."
        print(command_msg)
        # Run the calculation
        run = subprocess.Popen(command,
                               stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE)
        # Return a clean output message
        out, err = run.communicate()
        output_msg = out.decode('unicode_escape')
        return output_msg

    def _after_dry_run(self, output_msg):
        r"""
        Perform the following action:

        * Write the output of the bigdft-tool command in a Logfile,
        * Delete the dummy input files on disk and the associated
          temporary attribute.

        :param output_msg: Output of the bigdft-tool command.
        :type output_msg: str
        """
        log = Logfile.from_stream(output_msg)
        log.write(self.logfile_name)

        for filename in self._dummy_files:
            os.remove(filename)
        del self._dummy_files
