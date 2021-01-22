# -*- coding: utf-8 -*-
"""
.. Authors
    Novimir Pablant <npablant@pppl.gov>

A set of tools to facilitate string handling in XICSRT.
"""

import numpy as np

def simplify_strings(value):
    """
    Recursively simplify strings in the given variable.

    This will do the following:

      1. Make all strings lower case.
    """
    if isinstance(value, str):
        value = str.lower(value)
    elif isinstance(value, dict):
        for key in value:
            value[key] = simplify_strings(value[key])
    elif isinstance(value, list):
        for ii in range(len(value)):
            value[ii] = simplify_strings(value[ii])
    elif isinstance(value, np.ndarray):
        for ii in range(len(value)):
            value[ii] = simplify_strings(value[ii])
    return value
