# MyBigDFT

MyBigDFT provides a python wrapper over the BigDFT code.
It can be considered as a sandbox for the PyBigDFT package.
It might be used instead of the PyBigDFT package,
even though both packages do not provide the same API and functionalities.

It is currently supported for the following python versions: 2.7, 3.6, 3.7.

Credit to [mmoriniere](https://gitlab.com/mmoriniere) for the original [code](https://gitlab.com/mmoriniere/MyBigDFT).

## Installation

BigDFT 1.8.2 must be installed in a given build directory (noted BUILDDIR).
Then, run the following commands (where BUILDDIR is the appropriate build
directory):
- git clone this package
- cd BUILDDIR/install/bin
- source bigdftvars.sh'
- unset PYTHONPATH
- cd -
- cd MyBigDFT
- pip install .

This will copy the MyBigDFT sources, then set the environment variables
needed by MyBigDFT before installing it.


### Extra steps you might need to follow

* Make sure that the BIGDFT\_ROOT environment variables does not lead to
multiple directories. If so, choose only one of them (where BUILDDIR is the path
to the actual BigDFT build directory you want to use):
- export BIGDFT\_ROOT='BUILDDIR/install/bin'

* If your default python version is python3, modify the first line of 
the bigdft-tool executable so that it forces the use of python2 (*i.e.*,
set it to '#!/usr/bin/env python2'). This executable will be found in the
same directory as the bigdft one (*i.e.* BUILDDIR/install/bin)


## Documentation

The documentation of MyBigDFT can be found
[here](https://mmoriniere.gitlab.io/MyBigDFT/index.html).

It can also be built locally by running the following commands (where
MYBIGDFT\_SOURCES is the directory where this README.md file is located):
- cd MYBIGDFT\_SOURCES
- pip install -e .[doc]
- cd doc
- make html

The -e option is optional (use it if you wish to edit some source files
locally), while the [doc] directive might install extra packages required for
the doc to build correctly.


## Tests

To be able to run all tests, make sure to install the package by running the
following commands (where MYBIGDFT\_SOURCES is the directory where this
README.md file is located):
- cd MYBIGDFT\_SOURCES
- pip install -e .[test]
- pytest 

Again, the -e option is optional, use it if you wish to edit the source files.
The [test] directive might install extra packages required for the tests to run.
The pytest command should run all the tests (including the doctests).


### Notes

* If all the python versions supported by MyBigDFT are installed, you can even
run the 'tox' command to launch the tests for each of these versions instead of
'pytest'.
