# -*- coding: utf-8 -*-
"""
.. Authors
    Novimir Pablant <npablant@pppl.gov>

A set of miscellaneous tools used by XICSRT.

Programming Notes
-----------------
Many of the functions have rather specific behavior for use within
xicsrt and may not have behavior that is well suited for general use outside
of their intended purposes.
"""

import numpy as np


def _convert_to_numpy(obj, inplace=False):
    """
    Convert any numerical lists in a dictionary to numpy arrays.

    This function is specifically meant to help with saving and loading
    of config dictionaries files and should not be considered a general tool.
    """
    if not inplace:
        obj = obj.copy()

    if isinstance(obj, dict):
        obj_keys = obj.keys()
    elif isinstance(obj, list):
        obj_keys = range(len(obj))
    else:
        raise TypeError('Object must be either a dict or a list.')

    for key in obj_keys:
        if isinstance(obj[key], list):
            # Don't convert empty lists.
            if obj[key]:
                new = np.array(obj[key])

                if new.dtype.char == 'U':
                    # Don't convert unicode lists to numpy.
                    # These cause issues with hdf5 export.
                    pass
                elif new.dtype.char == 'O':
                    obj[key] = _convert_to_numpy(obj[key])
                else:
                    obj[key] = new
        elif isinstance(obj[key], dict):
            obj[key] = _convert_to_numpy(obj[key])
    return obj


def _convert_from_numpy(obj, inplace=False):
    """
    Convert any numpy arrays in a dictionary to lists.

    This function is specifically meant to help with saving and loading
    of config dictionaries and should not be considered a general tool.
    """
    if not inplace:
        obj = obj.copy()

    if isinstance(obj, dict):
        obj_keys = obj.keys()
    elif isinstance(obj, list):
        obj_keys = range(len(obj))
    else:
        raise TypeError('Object must be either a dict or a list.')

    for key in obj_keys:
        if isinstance(obj[key], np.ndarray):
            obj[key] = obj[key].tolist()
        elif isinstance(obj[key], dict) or isinstance(obj[key], list):
            obj[key] = _convert_from_numpy(obj[key])
    return obj


def _debug_types(obj, level=0):
    """
    A debugging function to print data types within a nested data structure.
    """
    if isinstance(obj, dict):
        obj_keys = obj.keys()
    elif isinstance(obj, list):
        obj_keys = range(len(obj))
    else:
        raise TypeError('Object must be either a dict or a list.')

    for key in obj_keys:
        print(f"{' '*level} {key!s:15.15s} : {type(obj[key])}")
        if isinstance(obj[key], dict) or isinstance(obj[key], list):
            _debug_types(obj[key], level+1)


def _replace_strings(obj, old, new, inplace=False):
    """
    Recursively replaces all strings in the given object.

    This function is specifically meant to help with saving and loading
    of config dictionaries and should not be considered a general tool.
    """
    if not inplace:
        obj = obj.copy()

    if isinstance(obj, dict):
        obj_keys = obj.keys()
    elif isinstance(obj, list):
        obj_keys = range(len(obj))
    else:
        raise TypeError('Object must be either a dict or a list.')

    for key in obj_keys:
        if isinstance(obj[key], str):
            obj[key] = obj[key].replace(old, new)
        elif isinstance(obj[key], dict) or isinstance(obj[key], list):
            obj[key] = _replace_strings(obj[key], old, new)

    return obj