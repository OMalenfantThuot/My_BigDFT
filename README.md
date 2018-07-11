# MyBigDFT

MyBigDFT provides a python wrapper over the BigDFT code.
It can be considered as a sandbox for the PyBigDFT package.
It might be used instead of the PyBigDFT package,
even though both packages do not provide the same API and functionalities.

It is currently supported for the following python versions: 2.7, 3.6, 3.7.

## Installation

BigDFT 1.8.2 must be installed in a given build directory.
Then, go to $BUILDDIR/install/bin and run 'source bigdftvars.sh'.
End by running 'unset PYTHONPATH'.
This gives the environment variables necessary for MyBigDFT to run correctly.

You can now git clone this package and run 'pip install .'

## Tests

To be able to run all tests, make sure to install the package by running the
'pip install [-e] .[test]' as it might install extra packages required for the
tests to run (use the -e option if you wish to edit the source files).

You can then simply run the command 'pytest' in the main directory to run all
the tests.

If all the python versions supported by MyBigDFT are installed, you can even
run the 'tox' command to launch the tests for each of these versions.
