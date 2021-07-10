# -*- coding: utf-8 -*-
"""
Authors
-------
- Novimir A. Pablant <npblant@pppl.gov>
"""

import numpy as np
import logging
import copy

class RayArray(dict):
    """
    The base class for an Ray array.

    The RayArray object is essentially a dictionary of numpy arrays.
    Some convenience methods have been added.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if 'origin' in self and 'direction' in self:
            self.initialize()

    def initialize(self):
        """
        Initialize the ray array.
        This will ensure that all properties are present and of the correct type.
        """

        if not isinstance(self['origin'], np.ndarray):
            self['origin'] = np.array(self['origin'])
        if not isinstance(self['direction'], np.ndarray):
            self['direction'] = np.array(self['direction'])

        if not (('origin' in self) and ('direction' in self)):
            raise Exception('Cannot initialize, origin and direction must be present.')

        shape = self['origin'].shape
        if not 'mask' in self:
            self['mask'] = np.ones(shape[0], dtype=bool)
        if not 'wavelength' in self:
            self['wavelength'] = np.zeros(shape[0])

        if not isinstance(self['mask'], np.ndarray):
            self['mask'] = np.array(self['mask'])
        if not isinstance(self['wavelength'], np.ndarray):
            self['wavelength'] = np.array(self['wavelength'])

    def __getattribute__(self, key):
        """
        Setup shortcuts for the basic ray properties.
        """
        if key == 'O' or key == 'origin':
            return self['origin']
        elif key == 'D' or key == 'direction':
            return self['direction']
        elif key == 'W' or key == 'wavelength':
            return self['wavelength']
        elif key == 'M' or key == 'mask':
            return self['mask']
        else:
            return super().__getattribute__(key)

    def __setattr__(self, key, value):
        """
        Setup shortcuts for the basic ray properties.
        """

        if key == 'O' or key == 'origin':
            self['origin'] = value
        elif key == 'D' or key == 'direction':
            self['direction'] = value
        elif key == 'W' or key == 'wavelength':
            self['wavelength'] = value
        elif key == 'M' or key == 'mask':
            self['mask'] = value
        else:
            super().__setattr(key, value)

    def zeros(self, num):
        self['origin'] = np.zeros((num, 3))
        self['direction'] = np.zeros((num, 3))
        self['mask'] = np.zeros((num), dtype=bool)
        self['wavelength'] = np.zeros((num))

    def copy(self):
        ray_new = RayArray()
        for key in self:
            ray_new[key] = self[key].copy()
        return ray_new

    def extend(self, ray_in):
        for key in self:
            self[key] = np.concatenate((self[key], ray_in[key]))
