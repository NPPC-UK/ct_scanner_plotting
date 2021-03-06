#!/usr/bin/env python

from __future__ import absolute_import
from __future__ import print_function

import io
from glob import glob
from os.path import basename
from os.path import dirname
from os.path import join
from os.path import splitext

from setuptools import find_packages
from setuptools import setup


def read(*names, **kwargs):
    return io.open(
        join(dirname(__file__), *names),
        encoding=kwargs.get("encoding", "utf8"),
    ).read()


setup(
    name="ct_plotting",
    version="0.1.0",
    license="GPLv3",
    description=(
        "Scripts and libraries for plotting data from a micro" " ct scanner"
    ),
    author="Maximilian Friedersdorff",
    author_email="maf54@aber.ac.uk",
    packages=find_packages("src"),
    package_dir={"": "src"},
    py_modules=[splitext(basename(path))[0] for path in glob("src/*.py")],
    include_package_data=True,
    zip_safe=False,
    classifiers=[
        # complete classifier list:
        # http://pypi.python.org/pypi?%3Aaction=list_classifiers
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: Unix",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
    ],
    install_requires=["numpy", "matplotlib", "pre-commit"],
    tests_require=["pytest"],
    entry_points={"console_scripts": ["plot = ct_plotting.plot_things:main"]},
)
