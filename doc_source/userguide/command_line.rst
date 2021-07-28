Command Line Interface
======================

A command line interface is available for xicsrt. For a standard installation
the command :code:`xicsrt` will be available. The command line interface can
also be invoked using :code:`python -m xicsrt`.

A saved `config` dictionary file is required to use the command line interface
(typially a `.json` file). Once defined we can simply pass this file to the
:code:`xicsrt` command.

.. code:: bash

    xicsrt config.json

To aid in saving of `config` dictionaries to `.json` files the helper function
:any:`xicsrt_io.save_config()` is available.

xicsrt
------

.. automodule:: xicsrt.__main__
   :noindex:
