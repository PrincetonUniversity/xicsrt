# -*- coding: utf-8 -*-
"""
.. Authors
    Novimir Pablant <npablant@pppl.gov>

A set of miscellaneous tools used by XICSRT.
"""

import numpy as np



def _dict_to_numpy(obj):
    """
    Convert any numerical lists in a dictionary to numpy arrays.

    This function is specifically meant to help with saving and loading
    of config dictionaries and should not be considered a general tool.
    """
    for key in obj:
        if isinstance(obj[key], list):
            # Don't convert empty lists.
            if obj[key]:
                new = np.array(obj[key])
                # I don't want to convert unicode or object lists to numpy.
                # These are better left as lists to be dealt with later.
                if new.dtype.char != 'U' and new.dtype.char != 'O':
                    obj[key] = new
        elif isinstance(obj[key], dict):
            obj[key] = _dict_to_numpy(obj[key])
    return obj


def _dict_to_list(obj):
    """
    Convert any numpy arrays in a dictionary to lists.

    This function is specifically meant to help with saving and loading
    of config dictionaries and should not be considered a general tool.
    """
    for key in obj:
        if isinstance(obj[key], np.ndarray):
            obj[key] = obj[key].tolist()
        elif isinstance(obj[key], dict):
            obj[key] = _dict_to_list(obj[key])
    return obj