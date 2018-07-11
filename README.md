# MyBigDFT

MyBigDFT provides a python wrapper over the BigDFT code.
It can be considered as a sandbox for the PyBigDFT package.
It might be used instead of the PyBigDFT package,
even though both packages do not provide the same API and functionalities.

## Installation

BigDFT 1.8.2 must be installed in a given builddir.
Then, go to that builddir and run source bigdftvars.sh and finally unset PYTHONPATH
This gives the environment variables necessary for MyBigDFT.

You can now git clone this package and run pip install .

## Tests

run pytest in the main directory
