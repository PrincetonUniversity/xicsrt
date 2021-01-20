XICSRT: Photon based raytracing in Python
=========================================

Documentation: https://xicsrt.readthedocs.org  
Git Repository: https://bitbucket.org/amicitas/xicsrt  
Git Mirror: https://github.com/PrincetonUniversity/xicsrt

Purpose
-------
XICSRT is a general purpose, photon based, raytracing code intended
for both optical and x-ray raytracing.

Installation
------------

XICSRT can be simply installed using `pip`

    pip install xicsrt

Alternatively it is possible to install from source using `setuptools`

    python setup.py install

Usage
-----

XICSRT is run by supplying a config dictionary to xicsrt.raytrace(config). 
The easiest way to run XICSRT is through a [Jupyter Notebook].

To learn how format the input, and interpret the output, see the examples
provided in the [documentation].


[![Image][idocs] ][docs]

[docs]: https://xicsrt.readthedocs.io/en/latest/?badge=latest
[idocs]: https://readthedocs.org/projects/docs/badge/?version=latest
[documentation]: https://xicsrt.readthedocs.org
[Jupyter Notebook]: https://jupyter.org/
