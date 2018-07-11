"""
File containing the base class defining a BigDFT calculation.
"""

from __future__ import print_function
import os
import shutil
import subprocess


# Define the path to the BigDFT executable
bigdft = os.path.join(os.environ["BIGDFT_ROOT"], "bigdft")
bigdft_tool = os.path.join(os.environ["BIGDFT_ROOT"], "bigdft-tool")
# Global name for the BigDFT data folder
DATA = "data"
# Space coordinates
COORDS = ["x", "y", "z"]
# Dictionary to convert the string of the signs to floats
SIGNS = {"+": 1., "-": -1.}


class BigDFTCalc(object):
    r"""
    This class represents an usual BigDFT calculation. There are two
    main methods:

    * one to initialize the calculation,

    * the other to actually run the calculation.
    """

    def __init__(self, input_yaml, posinp, prefix=None, run_folder=None,
                 ref_calc=None):
        r"""
        Method initializing a BigDFT calculation. It requires a yaml
        input file and an initial geometry posinp.

        You may give a prefix to name the output files and specify the
        folder where to run the calculation.

        A reference calculation may be given in order to copy its data
        folder to the present calculation (main use: restart from the
        wavefunctions of the reference calculation).

        :param input_yaml: BigDFT input file.
        :type input_yaml: inputfile.Input
        :param posinp: BigDFT initial geometry file.
        :type posinp: posinp.Posinp
        :param prefix: Prefix of the BigDFT calculation (optional,
            default value set to None).
        :type prefix: str
        :param run_folder: Folder where to run the calculation
            (optional, default value set to None).
        :type run_folder: str
        :param ref_calc: Other BigDFT calculation, taken as reference
            (optional, default value set to None)
        :type ref_calc: bigdft.BigDFTCalc
        """
        # Set the initial folder
        self.init_folder = os.getcwd()

        # Set the folder where the calculation will be run
        if run_folder is None:
            self.run_folder = "."
            self.abs_run_folder = self.init_folder
        else:
            # A run folder was given, find the common prefix with the
            # current working directory
            basename = os.path.commonprefix([self.init_folder, run_folder])
            # If there is no common prefix, then the run folder is
            # already well defined, and the absolute folder is the
            # the concatenation of the current working directory and
            # the run folder
            if basename == '':
                self.run_folder = run_folder
                self.abs_folder = os.path.join(self.init_folder, run_folder)
            # Else, find the relative path with the common prefix to
            # define run_folder, and use run_folder to define the
            # absolute folder. The initial folder is changed to the
            # common prefix.
            else:
                self.init_folder = basename
                self.run_folder = os.path.relpath(run_folder, basename)
                self.abs_folder = run_folder

        # Set the input files
        self.input_yaml = input_yaml
        self.posinp = posinp

        # Set the prefix of the calculation
        self.prefix = prefix

        # Initialize some names and the BigDFT command
        if prefix is not None:
            self.input_name = prefix+".yaml"  # input file name
            self.output_name = "log-"+self.input_name  # output file name
            self.posinp_name = prefix+".xyz"  # posinp file name
            self.command = [bigdft, prefix]  # BigDFT command
            self.data_dir = DATA+'-'+prefix   # data folder
        else:
            self.input_name = "input.yaml"  # input file name
            self.output_name = "log.yaml"  # output file name
            self.posinp_name = "posinp.xyz"  # posinp file name
            self.command = [bigdft]  # BigDFT command
            self.data_dir = DATA  # Data folder

        # Set the reference calculation
        self.ref_calc = ref_calc

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
                          even though a logfile already exists.
                          (Optional, default value set to False)
        :type force_run: boolean
        :param dry_run: If True, all folders and input files are written
            on disk, but are not run.
        :type dry_run: bool
        """
        # Update the BigDFT command with mpi, if necessary
        if int(nmpi) > 1:
            self.command = ['mpirun', '-np', str(nmpi)] + self.command

        # Set the environment for OpenMP
        if int(nomp) > 1:
            os.environ["OMP_NUM_THREADS"] = str(nomp)

        # Create a folder where to run the calculation if required
        if self.run_folder not in [".", ""]:
            if not os.path.exists(self.run_folder):
                os.makedirs(self.run_folder)
            os.chdir(self.run_folder)
        print(os.getcwd())

        try:
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
                        self.input_yaml['dft']['inputpsiid'] = 2
                    except KeyError:
                        self.input_yaml['dft'] = {'inputpsiid': 2}
                else:
                    # Else, delete the option from the input file, if
                    # it exists.
                    try:
                        del self.input_yaml['dft']['inputpsiid']
                    except Exception:
                        pass

            # Check that the calculation was not already done
            if (not os.path.exists(self.output_name)) or force_run:
                # Write the input file
                self.input_yaml.write(self.input_name)
                # Write the posinp file
                self.posinp.write(self.posinp_name)
                # Launch the calculation
                if not dry_run:
                    print(self.command, "...")
                    run = subprocess.Popen(self.command,
                                           stdout=subprocess.PIPE,
                                           stderr=subprocess.PIPE)
                    out, err = run.communicate()
                    print(out)
            else:
                print("Logfile {} already exists!\n".format(self.output_name))
        finally:
            # Set the path to the logfile
            self.logfile_path = os.path.join(self.abs_folder, self.output_name)
            # Go back to the initial folder
            os.chdir(self.init_folder)
