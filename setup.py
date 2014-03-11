#!/usr/bin/env python

import sys

# check Python version
if sys.version_info < (2, 6):
    sys.exit("bad only supports python 2.6 and above")

try:
    import ROOT
except ImportError:
    sys.exit("ROOT cannot be imported. Is ROOT installed with PyROOT enabled?")

ROOT.PyConfig.IgnoreCommandLineOptions = True

# check that we have at least the minimum required version of ROOT
if ROOT.gROOT.GetVersionInt() < 52800:
    sys.exit("rootpy requires at least ROOT 5.28/00; "
             "You have ROOT {0}.".format(ROOT.gROOT.GetVersion()))

import os

try:
    from setuptools import setup
    from setuptools.command.test import test as TestCommand
except ImportError as ex:
    sys.exit("bad requires that setuptools 0.7 is installed")


class PyTest(TestCommand):
    def finalize_options(self):
        TestCommand.finalize_options(self)
        self.test_args = []
        self.test_suite = True

    def run_tests(self):
        #import here, cause outside the eggs aren't loaded
        import pytest
        errno = pytest.main(self.test_args)
        sys.exit(errno)


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
    name="bad",
    version="0.0.0",
    author="Ruggero Turra",
    author_email="ruggero.turra@cern.ch",
    description="An application to create performance plots using ROOT",
    keywords="ROOT, python, plots, performance",
    long_description=read("README.md"),
    url="https://github.com/wiso/bad",
    tests_require=['pytest'],
    cmdclass = {'test': PyTest},
    )

extra_require = {
    'test': [
        'nose',
        'unittest2',
        'flake8',
        ],
    'development': [
        'zest.releaser',
        'check-manifest',
        ],
}
