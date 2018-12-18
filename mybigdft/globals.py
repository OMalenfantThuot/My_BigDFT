r"""
Define some global parameters such as:

- the BigDFT input parameters and some profiles (defining a set of basic
input parameters that can be used in input files for brevity) initalized
from a BigDFT source file,
- the path to the bigdft and the bigdft-tool executables.
"""
import os
import warnings
import yaml


__all__ = ["INPUT_PARAMETERS_DEFINITIONS", "DEFAULT_PARAMETERS", "PROFILES",
           "BIGDFT_PATH", "BIGDFT_TOOL_PATH",
           "ATOMS_MASS", "AMU_TO_EMU", "EMU_TO_AMU", "HA_TO_CMM1", "ANG_TO_B",
           "B_TO_ANG", "HA_TO_EV", "EV_TO_HA", "AU_TO_DEBYE", "DEBYE_TO_AU",
           "COORDS", "SIGNS"]


# Read the definition of the input parameters from the BigDFT sources
try:
    # Path to the BigDFT sources
    BIGDFT_SOURCES = os.environ["BIGDFT_SOURCES"]
    # Read the input parameters and available profiles
    input_parameters_file = os.path.join(
        BIGDFT_SOURCES, "src/input_variables_definition.yaml")
    with open(input_parameters_file, "r") as f:
        source = yaml.load_all(f)
        INPUT_PARAMETERS_DEFINITIONS = next(source)
        PROFILES = next(source)
    # Add the CheSS input parameters
    input_parameters_file = os.path.join(
        BIGDFT_SOURCES,
        "../chess/src/chess_input_variables_definition.yaml")
    with open(input_parameters_file, "r") as f:
        chess_parameters_definition = yaml.load(f)
    INPUT_PARAMETERS_DEFINITIONS["chess"] = chess_parameters_definition
    # Path to the BigDFT and BigDFT-tool executables
    BIGDFT_ROOT = os.environ["BIGDFT_ROOT"]
    BIGDFT_PATH = os.path.join(BIGDFT_ROOT, "bigdft")
    BIGDFT_TOOL_PATH = os.path.join(BIGDFT_ROOT, "bigdft-tool")
except KeyError:  # pragma: no cover
    # This allows the building of the docs when BigDFT sources are not
    # found.
    warnings.warn("BigDFT sources and/or root not found. If BigDFT is "
                  "installed, run 'source bigdftvars.sh' in the same folder "
                  "as the bigdft executable.",
                  RuntimeWarning)
    INPUT_PARAMETERS_DEFINITIONS = {}
    chess_parameters_definition = {}
    PROFILES = {}
    BIGDFT_PATH = "bigdft"
    BIGDFT_TOOL_PATH = "bigdft-tool"

# Add the posinp key (as it is not in input_parameters_definition.yaml)
INPUT_PARAMETERS_DEFINITIONS["posinp"] = {
    "units": {"default": "atomic"},
    "cell": {"default": None},
    "positions": {"default": None},
    "properties": {"default": {"format": "xyz", "source": "posinp.xyz"}}
}

# Define a dictionary containing the default value of all the input
# parameters
DEFAULT_PARAMETERS = {
    key: {subkey: subval.get("default")
          for subkey, subval in val.items() if subkey != "DESCRIPTION"}
    for key, val in INPUT_PARAMETERS_DEFINITIONS.items() if key != "chess"
}
DEFAULT_PARAMETERS["chess"] = {
    key: {subkey: subval.get("default")
          for subkey, subval in val.items() if subkey != "DESCRIPTION"}
    for key, val in chess_parameters_definition.items()
}

# Mass of the different types of atoms in atomic mass units
# TODO: Add more types of atoms
#       (found in $SRC_DIR/bigdft/src/orbitals/eleconf-inc.f90)
ATOMS_MASS = {"H": 1.00794, "He": 4.002602, "Li": 6.941, "Be": 9.012182,
              "B": 10.811, "C": 12.011, "N": 14.00674, "O": 15.9994,
              "F": 18.9984032, "Ne": 20.1797, "Na": 22.989768, "Mg": 24.3050,
              "Al": 26.981539, "Si": 28.0855, "P": 30.973762, "S": 32.066,
              "Cl": 35.4527, "Ar": 39.948}

# Space coordinates
COORDS = ["x", "y", "z"]
# Dictionary to convert the string of the signs to floats
SIGNS = {"+": 1., "-": -1.}


####
# Conversion factors
####

# Conversion from atomic to electronic mass unit
AMU_TO_EMU = 1.660538782e-27 / 9.10938215e-31
# Conversion from electronic to atomic mass unit
EMU_TO_AMU = 1. / AMU_TO_EMU
# Conversion factor from bohr to angstroem
B_TO_ANG = 0.529177249
# Conversion factor from angstroem to bohr
ANG_TO_B = 1. / B_TO_ANG
# Conversion factor from Hartree to cm^-1
HA_TO_CMM1 = 219474.6313705
# Conversion factor from Hartree to electron-Volt
HA_TO_EV = 27.21138602
# Conversion factor from electron-Volt to Hartree
EV_TO_HA = 1 / HA_TO_EV
# Conversion factor from Debye to atomic units of dipole moment
DEBYE_TO_AU = 0.393430307
# Conversion factor from atomic units of dipole moment to Debye
AU_TO_DEBYE = 1 / DEBYE_TO_AU
