# -*- coding: utf-8 -*-
"""
.. Authors:
    Novimir pablant <npablant@pppl.gov>

A set of tools to help with automatic API documention of XICSRT.

Description
-----------
XICSRT uses sphinx for documentation, and API docs are based on the idea of
code self documentation though `python` doc strings. This module contains
a set of decorators and helper function to aid in self documentation.

Philosophy:
  Python help() should be just as readable as the Sphinx webpage.

Todo:
  - The config docstrings should all be indented follow the help() standard.
  - Would it be helpful to show which inherited class the options came from?
"""

import inspect

class DocHelper:
    """
    A class to help generate docstrings for XICSRT.

    This is expected to be used through one of the @_dochelper decorators.
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


def dochelper(cls):
    """
    A functional wrapper for the DocHelper class.
    """
    return DocHelper(cls)()

