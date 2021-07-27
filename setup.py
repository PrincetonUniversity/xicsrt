# -*- coding: utf-8 -*-
"""
.. Authors:
   Novimir Pablant <npablant@pppl.gov>

To prepare an update:
  1. Update _version.py
  2. git tag -a v0.4.0 -m "Version 0.4.0"
  3. git push --tags
  4. Make everything is merged with 'master' and 'stable'.
  5. Push to both bitbucket and github.

To submit an update to pypi:
  1. python setup.py sdist bdist_wheel
  2. python -m twine upload --repository testpypi dist/*
  3. python -m twine upload dist/*
"""

import setuptools

with open('README.md') as ff:
    long_description = ff.read()

exec(open('xicsrt/_version.py').read())

classifiers = [
    "Development Status :: 4 - Beta",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3",
    "Operating System :: OS Independent",
    ]

params ={
    'name': 'xicsrt',
    'version': __version__,
    'author': 'Novimir Antoniuk Pablant',
    'author_email': 'npablant@pppl.gov',
    'maintainer': 'Novimir Antoniuk Pablant',
    'maintainer_email': 'npablant@pppl.gov',
    'description': 'A photon based raytracing application written in Python.',
    'long_description': long_description,
    'long_description_content_type': 'text/markdown',
    'url': 'https://github.com/PrincetonUniversity/xicsrt',
    'license': 'MIT',
    'packages': setuptools.find_packages(exclude=['xicsrt_contrib*']),
    'entry_points': {'console_scripts':['xicsrt=xicsrt.__main__:run']},
    'classifiers': classifiers,
    'install_requires': ['numpy', 'scipy', 'pillow', 'h5py'],
    'python_requires': '>=3.8',
    }
    
setuptools.setup(**params)
