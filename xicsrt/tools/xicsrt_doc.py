# -*- coding: utf-8 -*-
"""
.. Authors
    Novimir pablant <npablant@pppl.gov>

A set of tools to help with automatic API documentation of XICSRT.

Description
-----------
XICSRT uses sphinx for documentation, and API docs are based on the idea of
code self documentation though `python` doc strings. This module contains
a set of decorators and helper function to aid in self documentation.

The most important part of this module is the `@dochelper` decorator which
should be used for all element classes.

.. Note:
    Philosophy: Python help() should be just as readable as the Sphinx webpage.

Todo:
    - The config docstrings should all be indented follow the help() standard.
    - Would it be helpful to show which inherited class the options came from?
"""

import inspect


def dochelper(cls):
    """
    A functional wrapper for the DocHelper class. Intended to be used
    as a decorator.

    This decorator does the following:

    1. Adds a 'Configuration Options' section to the class docstring that
       contains all options defined in default_config and any class ancestors.

    """
    return DocHelper(cls)()

class DocHelper:
    """
    A class to help generate docstrings for XICSRT.

    This is expected to be used through the `@dochelper` class decorator.
    """

    def __init__(self, cls):
        self.cls = cls
        self.update_class_docstring(cls)

    def __call__(self):
        return self.cls

    def update_class_docstring(self, cls):
        #print(cls.__name__, cls.__qualname__)
        if cls.__doc__ == None:
            cls.__doc__ = ''

        cls.__doc__ += '\n'
        cls.__doc__ += 'Configuration Options:\n\n'
        #cls.__doc__ += '----------------------\n\n'

        for ancestor in inspect.getmro(cls):
            if hasattr(ancestor, 'default_config'):
                doc = ancestor.default_config.__doc__
                if doc is not None:
                    cls.__doc__ += inspect.getdoc(ancestor.default_config)
                    cls.__doc__ += '\n\n'
        return cls


