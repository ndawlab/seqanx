#! /usr/bin/env python
#
# Copyright (c) 2019 Daw Lab
# https://dawlab.princeton.edu/

import os, sys
from setuptools import setup, find_packages
path = os.path.abspath(os.path.dirname(__file__))

## Metadata
DISTNAME = 'sisyphus'
MAINTAINER = 'Sam Zorowitz'
MAINTAINER_EMAIL = 'zorowitz@princeton.edu'
DESCRIPTION = 'Code associated with manuscript'
URL = 'https://dawlab.princeton.edu/'
LICENSE = 'MIT'
DOWNLOAD_URL = 'https://github.com/szorowi1/SecretFunTimes'

with open(os.path.join(path, 'README.rst'), encoding='utf-8') as readme_file:
    README = readme_file.read()

with open(os.path.join(path, 'requirements.txt')) as requirements_file:
    # Parse requirements.txt, ignoring any commented-out lines.
    requirements = [line for line in requirements_file.read().splitlines()
                    if not line.startswith('#')]
    
VERSION = None
with open(os.path.join('sisyphus', '__init__.py'), 'r') as fid:
    for line in (line.strip() for line in fid):
        if line.startswith('__version__'):
            VERSION = line.split('=')[1].strip().strip('\'')
            break
if VERSION is None:
    raise RuntimeError('Could not determine version')

setup(name=DISTNAME,
      maintainer=MAINTAINER,
      maintainer_email=MAINTAINER_EMAIL,
      description=DESCRIPTION,
      url=URL,
      version=VERSION,
      download_url=DOWNLOAD_URL,
      long_description=README,
      packages=find_packages(exclude=['docs', 'tests']),
      install_requires=requirements,
      license=LICENSE
)