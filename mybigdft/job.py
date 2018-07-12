"""
File containing the base class defining a BigDFT calculation.
"""

from __future__ import print_function
import os
import shutil
import subprocess
from .globals import bigdft_path, bigdft_tool_path
from .iofiles import Logfile

# Name for the BigDFT data folder
DATA = "data"
# Space coordinates
COORDS = ["x", "y", "z"]
# Dictionary to convert the string of the signs to floats
SIGNS = {"+": 1., "-": -1.}


class Job(object):

    def __init__(self, name="", inputparams=None, posinp=None, run_folder=None,
                 ref_calc=None, skip=False):
        r"""
        You may pass input parameters and/or initial geometry (posinp).
        Make sure to at list provide initial positions, either via the
        posinp or the input parameters.

        You may give a prefix to name the output files and specify the
        folder where to run the calculation.

        A reference calculation may be given in order to copy its data
        folder to the present calculation (main use: restart from the
        wavefunctions of the reference calculation).

        :param inputparams: BigDFT input file.
        :type inputparams: mybigdft.iofiles.InputParams
        :param posinp: BigDFT initial geometry file.
        :type posinp: mybigdft.iofiles.Posinp
        :param name: Prefix of the BigDFT calculation (optional,
            default value set to None).
        :type name: str
        :param run_folder: Folder where to run the calculation
            (optional, default value set to None).
        :type run_folder: str
        :param ref_calc: Other BigDFT calculation, taken as reference
            (optional, default value set to None)
        :type ref_calc: bigdft.BigDFTCalc
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

        # Set the initial folder
        self.init_folder = os.getcwd()

        # Set the folder where the calculation will be run
        if run_folder is None:
            self.run_folder = os.path.curdir
            self.abs_folder = self.init_folder
        else:
            # A run folder was given, find the common prefix with the
            # current working directory
            basename = os.path.commonprefix([self.init_folder, run_folder])
            if basename == '':
                # If there is no common prefix, then the run folder is
                # already well defined, and the absolute folder is the
                # the concatenation of the current working directory and
                # the run folder
                self.run_folder = run_folder
                self.abs_folder = os.path.join(self.init_folder, run_folder)
            else:
                # Else, find the relative path with the common prefix to
                # define run_folder, and use run_folder to define the
                # absolute folder. The initial folder is changed to the
                # common prefix.
                self.init_folder = basename
                self.run_folder = os.path.relpath(run_folder, basename)
                self.abs_folder = run_folder

        # Set the attributes
        self.inputparams = inputparams
        self.name = name
        self.posinp = posinp
        self.logfile = None

        # Initialize some file and directory names and also BigDFT commands
        skip_option = []
        if skip:
            skip_option += ["-s", "Yes"]
        if name != "":
            self.input_name = name+".yaml"  # input file name
            self.logfile_name = "log-"+self.input_name  # output file name
            self.posinp_name = name+".xyz"  # posinp file name
            self.bigdft_cmd = [bigdft_path, name] + skip_option
            self.bigdft_tool_cmd = [bigdft_tool_path, name]
            self.data_dir = DATA+'-'+name  # data folder
        else:
            self.input_name = "input.yaml"  # input file name
            self.logfile_name = "log.yaml"  # output file name
            self.posinp_name = "posinp.xyz"  # posinp file name
            self.bigdft_cmd = [bigdft_path] + skip_option
            self.bigdft_tool_cmd = [bigdft_tool_path]
            self.data_dir = DATA  # Data folder

        # Set the reference calculation
        self.ref_calc = ref_calc

    def __enter__(self):
        # Create a folder where to run the calculation if required
        if self.run_folder not in [".", ""]:
            if not os.path.exists(self.run_folder):
                os.makedirs(self.run_folder)
            os.chdir(self.run_folder)
        print(os.getcwd())
        return self

    def __exit__(self, *args):
        if not self._dry_run:
            # Read the result of the calculation
            self.logfile = Logfile.from_file(self.logfile_name)
        # Go back to the initial folder
        os.chdir(self.init_folder)

    def run(self, nmpi=1, nomp=1, force_run=False, dry_run=False):
        r"""
        Method running the BigDFT calculation if it was not already
        performed.

        You may force the calculation by setting force_run to True.

        The number of MPI and OpenMP tasks may also be specified
        (both default values to 1).

        :param nmpi: Number of MPI tasks (default value set to 1).
        :type nmpi: int
        :param nomp: Number of OpenMP tasks (default value set to 1).
        :type nomp: int
        :param force_run: States if the calculation has to be run,
            even though a logfile already exists. (Optional, default
            value set to False)
        :type force_run: boolean
        :param dry_run: If True, all folders and input files are written
            on disk, but are not run.
        :type dry_run: bool
        """
        self._dry_run = dry_run

        # Copy the data folder of a reference calculation
        ref = self.ref_calc
        if ref is not None:
            # Find and copy the path to the reference data folder
            ref_data_path = os.path.join(ref.abs_folder, ref.data_dir)
            if os.path.exists(ref_data_path):
                # Remove the previously existing data folder
                # before copying the reference data folder
                # (otherwise, shutil.copytree raises an error).
                if self.data_dir in os.listdir("."):
                    shutil.rmtree(self.data_dir)
                shutil.copytree(ref_data_path, self.data_dir)
            elif not os.path.exists(ref_data_path):
                print("Data folder not found for reference calculation.")

        # Update the input file, so that it reads the reference
        # wavefunctions in the data folder
        if os.path.exists(self.data_dir):
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

        # Check that the calculation was not already done
        if dry_run or force_run or not os.path.exists(self.logfile_name):

            # Set the environment for OpenMP
            if int(nomp) > 1:
                os.environ["OMP_NUM_THREADS"] = str(nomp)

            # Set the actual command and file names
            mpi_command = []
            if dry_run:
                if int(nmpi) > 1:
                    mpi_command = ['-n', str(nmpi)]
                command = self.bigdft_tool_cmd + mpi_command
                inpname = "input.yaml"
                posname = "posinp.xyz"
                self._dummy_files = [inpname, posname]
            else:
                if int(nmpi) > 1:
                    mpi_command = ['mpirun', '-np', str(nmpi)]
                command = mpi_command + self.bigdft_cmd
                inpname = self.input_name
                posname = self.posinp_name

            # Write the input files (if needed)
            self.inputparams.write(inpname)
            if self.posinp is not None:
                self.posinp.write(posname)
            elif dry_run:
                # Remove the posinp from the files to be cleaned
                self._dummy_files.pop()

            # Launch the calculation
            to_str = "{} "*len(command)
            command_msg = to_str.format(*command)+"..."
            print(command_msg)
            run = subprocess.Popen(command,
                                   stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE)
            out, err = run.communicate()
            output_msg = out.decode('unicode_escape')
            print(output_msg)

            # Clean after dry run
            if dry_run:
                for filename in self._dummy_files:
                    os.remove(filename)
                del self._dummy_files
        else:
            print("Logfile {} already exists!\n".format(self.logfile_name))
