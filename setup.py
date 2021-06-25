#!/usr/bin/env python

import os
import sys
import shutil
from setuptools.command.test import test as TestCommand
from setuptools import find_packages

def remove_dir(dirpath):
	if os.path.exists(dirpath) and os.path.isdir(dirpath):
		  shutil.rmtree(dirpath)

try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

requires = [] #during runtime
tests_require=['pytest>=3.0'] #for testing

PACKAGE_PATH = os.path.abspath(os.path.join(__file__, os.pardir))

setup(
    name='tfcook',
    version='0.1.0',
    description='Python imputation package',
    author='Alireza Vafaei Sadr',
    url='https://github.com/vafaei-ar/tfcook.git',
    packages=find_packages(PACKAGE_PATH, "tfcook"),
    package_dir={'tfcook': 'tfcook'},
    include_package_data=True,
    install_requires=requires,
    license='GPLv3+',
    zip_safe=False,
    keywords='tfcook',
    classifiers=[
        'Development Status :: 2 - Alpha',
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT",
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
    ]
)

remove_dir('build')
remove_dir('tfcook.egg-info')
remove_dir('dist')
