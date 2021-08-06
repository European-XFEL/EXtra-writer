#!/usr/bin/env python

# Distributed under the terms of the BSD 3-Clause License.
# The full license is in the file LICENSE, distributed with this software.
#
# Author: Egor Sobolev <egor.sobolev@xfel.eu>
# Copyright (c) 2021, European X-Ray Free-Electron Laser Facility GmbH.
# All rights reserved.

from setuptools import setup, find_packages
import pathlib

here = pathlib.Path(__file__).parent.resolve()

exec((here / 'src/extra_writer/version.py').read_text(encoding='utf-8'))

long_description = (here / 'README.md').read_text(encoding='utf-8')

setup(
    name="EXtra-writer",
    version=__version__,
    author="Egor Sobolev",
    author_email="egor.sobolev@xfel.eu",
    maintainer="Egor Sobolev",
    url="https://github.com/European-XFEL/EXtra-writer",
    description="Tools for writing data in EuXFEL format",
    long_description=long_description,
    long_description_content_type='text/markdown',
    license="BSD-3-Clause",
    install_requires=[
        'h5py>=2.7.1',
        'numpy',
    ],
    extras_require={
        'docs': [
            'sphinx',
            'nbsphinx',
            'ipython',  # For nbsphinx syntax highlighting
            'sphinxcontrib_github_alt',
        ],
        'test': [
            'nbval',
            'pytest',
            'pytest-cov',
            'testpath',
        ],
    },
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python :: 3',
        'Topic :: Scientific/Engineering :: Information Analysis',
        'Topic :: Scientific/Engineering :: Physics',
    ],
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.6",
)
