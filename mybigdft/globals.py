r"""
Define some global variables such as:

- the BigDFT input variables and some profiles (defining a set of basic
input parameters that can be used in input files for brevity) initalized
from a BigDFT source file,
- the path to the bigdft and the bigdft-tool executables.
"""
import os
import warnings
import yaml


__all__ = ["INPUT_VARIABLES", "PROFILES", "BIGDFT_PATH", "BIGDFT_TOOL_PATH"]


# Read the definition of the input variables from the BigDFT sources
try:
    # Path to the BigDFT sources
    BIGDFT_SOURCES = os.environ["BIGDFT_SOURCES"]
    # Read the input variables and available profiles
    input_variables_file = os.path.join(BIGDFT_SOURCES,
                                        "src/input_variables_definition.yaml")
    with open(input_variables_file, "r") as f:
        source = yaml.load_all(f)
        INPUT_VARIABLES = next(source)
        PROFILES = next(source)
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
    INPUT_VARIABLES = {}
    PROFILES = {}
    BIGDFT_PATH = "bigdft"
    BIGDFT_TOOL_PATH = "bigdft-tool"
# Add the posinp key (as it is not in input_variables_definition.yaml)
INPUT_VARIABLES["posinp"] = {
    "units": {"default": "atomic"},
    "cell": {"default": None},
    "positions": {"default": None},
    "properties": {"default": {"format": "xyz", "source": "posinp.xyz"}}
}
