XICSRT: Photon based raytracing in Python
=========================================

XICSRT is a general purpose, photon based, raytracing code intended
for both optical and x-ray raytracing.

**The best way to get started with XICSRT is with the :any:`examples`.**

| **Documentation:** `xicsrt.readthedocs.org`_
| **Git Repository:** `bitbucket.org/amicitas/xicsrt`_
| **Git Mirror:** `github.com/PrincetonUniversity/xicsrt`_

XICSRT provides a simple, extensible, optical and x-ray raytracing capability
in Python. Input is a single python dictionary (which can be saved as
a `json` file), output is a python dictionary (which can be saved as `hdf5`).

A few simple 2D and 3D visualization utilities are included in the project.
However, these are only meant to be used as examples. Interface, visualization,
and analysis are expected to be handled by the user or through external
wrapper projects. Than being said, XICSRT can be effectively used interactively
using a `jupyter`_ notebook and some familiarity with `numpy`_, `matplotlib`_ and
`plotly`_.

XICSRT is highly vectorized, and most expensive calculations
are performed through built-in `numpy` routines. A `multiprocessing`
module and a command line interface are included to facilitate use
across multiple processors. Overall though, the code has been written for
ease of extensibility, rather than computational speed.

.. warning::

    Documentation of XICSRT has only just begun. These docs are still very
    incomplete, and may be out of date or incorrect in places.

    If you are using XICSRT please help us improve the documentation!

.. include:: ../AUTHORS


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

XICSRT is run by supplying a `config` dictionary to `xicsrt.raytrace(config)`.

To learn how format the input, and interpret the output, see the: :any:`examples`.


License
-------
.. include:: ../LICENSE

.. toctree::
    :maxdepth: 1
    :caption: Contents:

    apidoc/xicsrt
    apidoc/examples
    development

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
