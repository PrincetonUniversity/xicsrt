XICSRT: Photon based raytracing in Python
=========================================

Documentation: https://xicsrt.readthedocs.org  
Git Repository: https://bitbucket.org/amicitas/xicsrt  
Git Mirror: https://github.com/PrincetonUniversity/xicsrt

Purpose
-------
XICSRT is a general purpose, photon based, scientific raytracing code intended
for both optical and x-ray raytracing. 

XICSRT includes handling for x-ray Bragg reflections from crystals which allows 
modeling of x-ray spectrometers and other x-ray systems. Care has been taken to 
allow for modeling of emission sources in real units and accurate preservation 
of photon statistics throughout. The XICSRT code has similar functionality to 
the well known [SHADOW] raytracing code, though the intention is to be a 
complementary tool rather than a replacement.  These two projects have somewhat 
different goals, and therefore different strengths.

Current development is focused on x-ray raytracing for fusion science and 
high energy density physics (HEDP) research, in particular X-Ray Imaging Crystal 
Spectrometers for Wendelstein 7-X (W7-X), ITER and the National Ignition 
Facility (NIF).

Installation
------------

XICSRT can be simply installed using `pip`

    pip install xicsrt

Alternatively it is possible to install from source using `setuptools`

    python setup.py install

Usage
-----

XICSRT is run by supplying a config dictionary to ```xicsrt.raytrace(config)```. 
The easiest way to run XICSRT is through a [Jupyter Notebook]. A command line
interface is also available.

To learn how format the input, and interpret the output, see the examples
provided in the [documentation].


[![Image][idocs] ][docs]

[docs]: https://xicsrt.readthedocs.io/en/latest/?badge=latest
[idocs]: https://readthedocs.org/projects/docs/badge/?version=latest
[documentation]: https://xicsrt.readthedocs.org
[Jupyter Notebook]: https://jupyter.org/
[SHADOW]: https://github.com/oasys-kit/shadow3