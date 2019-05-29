# -*- coding: utf-8 -*
"""setup for MyBigDFT."""

from setuptools import setup, find_packages
from codecs import open
from os import path


# Get the long description from the README file
here = path.abspath(path.dirname(__file__))
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()


def readme():
    """Function returning the README."""
    return long_description


setup(
    name="mybigdft",
    packages=find_packages(),
    )
