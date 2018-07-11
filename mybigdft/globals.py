import os
import warnings
import yaml

# Read the definition of the input variables from the BigDFT sources
try:
    BIGDFT_SOURCES = os.environ["BIGDFT_SOURCES"]
    inp_vars_file = os.path.join(BIGDFT_SOURCES,
                                 "src/input_variables_definition.yaml")
except KeyError:
    warnings.warn("BigDFT sources not found. Default files will be used "
                  "instead (might lead to unwanted errors)", RuntimeWarning)
    curdir = os.path.normpath(os.path.join(os.path.abspath(__file__), ".."))
    BIGDFT_SOURCES = os.path.join(curdir,
                                  "src/input_variables_definition.yaml")
with open(inp_vars_file, "r") as f:
    source = yaml.load_all(f)
    inp_vars = next(source)
    profiles = next(source)
# Add the posinp key (as it is not in input_variables_definition.yaml)
inp_vars["posinp"] = {"units": {"default": "atomic"},
                      "cell": {"default": []},
                      "positions": {"default": []},
                      "properties": {"default": {"format": "xyz",
                                                 "source": "posinp.xyz"}}}
