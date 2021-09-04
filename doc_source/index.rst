XICSRT: Photon based raytracing in Python
=========================================

XICSRT is a general purpose, photon based, raytracing code intended
for both optical and x-ray raytracing.

**The best way to get started with XICSRT is with the** :any:`examples`.

| **Documentation:** `xicsrt.readthedocs.org`_
| **Git Repository:** `bitbucket.org/amicitas/xicsrt`_
| **Git Mirror:** `github.com/PrincetonUniversity/xicsrt`_

XICSRT provides a simple, extensible, optical and x-ray raytracing capability
in Python. Input is a single python dictionary (which can be saved to a `json`
file), output is a python dictionary (which can be saved to a `hdf5` file).

For interactive use XICSRT can run within a `jupyter`_ notebook. Simple examples
for 2D and 3D plotting using the `matplotlib`_ and `plotly`_ libraries are
included. A command line interface to XICSRT is also available.

XICSRT has been written with a primary goal of simplicity and ease of
extensibility, rather than computational speed. That being said the code has
been thoroughly vectorized and optimized, and most expensive calculations are
performed through built-in `numpy`_ routines. Use across multiple processors can be
achieved though the built-in `multiprocessing` capabilities.

.. warning::

    Documentation of XICSRT is still in progress. Please get involved and help
    us improve the documentation!

Installation
------------

XICSRT can be simply installed using `pip`

.. code:: bash

    pip install xicsrt

Alternatively you can install from source using `setuptools`

.. code:: bash

    python setup.py install

Usage
-----

XICSRT is run by supplying a `config` dictionary to `raytrace()`.

.. code:: python

    results = xicsrt.raytrace(config)

To learn how format the input and interpret the output, try the :any:`examples`
or download the `XICSRT Tutorial`_.

Tutorial
--------

An `XICSRT Tutorial`_ presentation is available that introduces basic usage and
concepts.


Authors
-------

XICSRT development is coordinated by Novimir A. Pablant. A full list of
contributers can be found on the :doc:`pages/authors` page.


Citation
--------

If you use XICSRT for work leading to a publication, please use the following
citation:

\N. \A. Pablant, M. Bitter, P. C. Efthimion, L. Gao, K. W. Hill, B. F. Kraus, J. Kring, M. J. MacDonald, N. Ose, Y. Ping, M. B. Schneider, S. Stoupin, and Y. Yakusevitch, **"Design and expected performance of a variable-radii sinusoidal spiral x-ray spectrometer for the National Ignition Facility"**, Review of Scientific Instruments 92, 093904 (2021) https://doi.org/10.1063/5.0054329

A list of publications relating to XICSRT can be found at the
:doc:`userguide/list_of_publications` page.

License
-------

XICSRT is open source software released under the `MIT License`_. For
the full text of the licence see the :doc:`pages/license` page.
Please help improve XICSRT by contributing to the codebase.


.. toctree::
    :maxdepth: 1
    :caption: Contents:

    userguide/userguide
    apidoc/examples
    apidoc/xicsrt
    pages/development
    pages/authors
    pages/license

Indices and tables:
-------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

.. _xicsrt.readthedocs.org: https://xicsrt.readthedocs.org
.. _bitbucket.org/amicitas/xicsrt: https://bitbucket.org/amicitas/xicsrt
.. _github.com/PrincetonUniversity/xicsrt: https://github.com/PrincetonUniversity/xicsrt
.. _jupyter: https://jupyter.org/
.. _numpy: https://numpy.org/
.. _matplotlib: https://matplotlib.org/
.. _plotly: https://github.com/plotly
.. _XICSRT tutorial: https://drive.google.com/file/d/1ze1DPO_Cx8hJtj-eoli9cp2XAGZKcese/view?usp=sharing
.. _MIT License: https://opensource.org/licenses/MIT