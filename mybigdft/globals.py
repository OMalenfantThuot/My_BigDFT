import os
import warnings
import yaml


__all__ = ["inp_vars", "profiles", "bigdft_path", "bigdft_tool_path"]


# Read the definition of the input variables from the BigDFT sources
try:
    # Path to the BigDFT sources
    BIGDFT_SOURCES = os.environ["BIGDFT_SOURCES"]
    # Read the input variables and available profiles
    inp_vars_file = os.path.join(BIGDFT_SOURCES,
                                 "src/input_variables_definition.yaml")
    with open(inp_vars_file, "r") as f:
        source = yaml.load_all(f)
        inp_vars = next(source)
        profiles = next(source)
    # Path to the BigDFT and BigDFT-tool executables
    BIGDFT_ROOT = os.environ["BIGDFT_ROOT"]
    bigdft_path = os.path.join(BIGDFT_ROOT, "bigdft")
    bigdft_tool_path = os.path.join(BIGDFT_ROOT, "bigdft-tool")
except KeyError:  # pragma: no cover
    # This allows the building of the docs when BigDFT sources are not
    # found.
    warnings.warn("BigDFT sources and/or root not found. If BigDFT is "
                  "installed, run 'source bigdftvars.sh' in the same folder "
                  "as the bigdft executable.",
                  RuntimeWarning)
    inp_vars = {}
    profiles = {}
    bigdft_path = "bigdft"
    bigdft_tool_path = "bigdft-tool"
# Add the posinp key (as it is not in input_variables_definition.yaml)
inp_vars["posinp"] = {"units": {"default": "atomic"},
                      "cell": {"default": []},
                      "positions": {"default": []},
                      "properties": {"default": {"format": "xyz",
                                                 "source": "posinp.xyz"}}}
