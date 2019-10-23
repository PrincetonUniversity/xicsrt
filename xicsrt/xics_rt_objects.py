# -*- coding: utf-8 -*-
"""
Authors
-------
- Novimir A. Pablant <npblant@pppl.gov>

Description
-----------
A set of base objects for xicsrt.
"""

import numpy as np
import scipy as sp
import xicsrt.tool

class RayArray(dict):
    """
    The base class for an Ray array.
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

    def copy(self):
        # Do this explicitly to avoid any unnecessary object creation.
        ray_new = GeometryObject(
            origin=self['origin'].copy()
            ,direction=self['direction'].copy()
            ,wavelength=self['wavelength'].copy()
            ,mask=self['mask'].copy()
            )

        return ray_new

class GeometryObject():
    """
    The base class for any geometrical objects used in XICSRT.
    """
    def __init__(self
            ,origin = None
            ,zaxis  = None
            ,xaxis  = None
            ):

        if origin is None:
            origin = np.array([0.0, 0.0, 0.0])
        if zaxis is None:
            zaxis = np.array([0.0, 0.0, 1.0])

        # Location with respect to the external coordinate system.
        self.origin = origin
        self.set_orientation(zaxis, xaxis)

    def __getattr__(self, key):
        """
        Setup shortcuts for the basic object properties.
        """
        if key == 'xaxis':
            return self.orientation[0,:]
        elif key == 'yaxis':
            return self.orientation[1,:]
        elif key == 'zaxis':
            return self.orientation[2, :]
        else:
            raise AttributeError()

    def set_orientation(self, zaxis, xaxis=None):
        if xaxis is None:
            xaxis = self.get_default_xaxis(zaxis)

        self.orientation = np.array([xaxis, np.cross(zaxis, xaxis), zaxis])

    def get_default_xaxis(self, zaxis):
        """
        Get the X-axis using a default definition.

        In order to fully define the orientation of a component both, a z-axis
        and an x-axis are expected.  For certain types of components the x-axis
        definition is unimportant and can be defined using a default definition.
        """

        xaxis = np.cross(np.array([0.0, 0.0, 1.0]), zaxis)
        if not np.all(xaxis == 0.0):
            xicsrt.math.normalize(xaxis)
        else:
            xaxis = np.array([1.0, 0.0, 0.0])

        return xaxis

    def ray_to_external(self, ray_local, copy=False):
        if copy:
            ray_external = ray_local.copy()
        else:
            ray_external = ray_local

        ray_external['origin'] = self.point_to_external(ray_external['origin'])
        ray_external['direction'] = self.vector_to_external(ray_external['direction'])
        return ray_external

    def ray_to_local(self, ray_external, copy=False):
        if copy:
            ray_local = ray_external.copy()
        else:
            ray_local = ray_external

        ray_local['origin'] = self.point_to_local(ray_local['origin'])
        ray_local['direction'] = self.vector_to_local(ray_local['direction'])
        return ray_local

    def point_to_external(self, point_local):
        return self.vector_to_external(point_local) + self.origin

    def point_to_local(self, point_external):
        return self.vector_to_local(point_external - self.origin)

    def vector_to_external(self, vector):
        vector = self.to_ndarray(vector)
        if vector.ndim == 2:
            vector[:] = np.einsum('ij,ki->kj', self.orientation, vector)
        elif vector.ndim == 1:
            vector[:] = np.einsum('ij,i->j', self.orientation, vector)
        else:
            raise Exception('vector.ndim must be 1 or 2')

        return vector

    def vector_to_local(self, vector):
        vector = self.to_ndarray(vector)
        if vector.ndim == 2:
            vector[:] = np.einsum('ji,ki->kj', self.orientation, vector)
        elif vector.ndim == 1:
            vector[:] = np.einsum('ji,i->j', self.orientation, vector)
        else:
            raise Exception('vector.ndim must be 1 or 2')

        return vector

    def aim_to_point(self, aim_point, xaxis=None):
        """
        Set the Z-Axis to aim at a particular point.
        """

        self.zaxis = aim_point - self.origin
        xicsrt.math.normalize(self.zaxis)

        if xaxis is not None:
            self.xaxis = xaxis
        else:
            self.set_default_xaxis()

    def to_ndarray(self, vector_in):
        if not isinstance(vector_in, np.ndarray):
            vector_in = np.array(vector_in, dtype=float)
        return vector_in

    def to_vector_array(self, vector_in):
        """
        Convert a vector to a numpy vector array (if needed).
        """
        vector_in = self.to_ndarray(vector_in)

        if vector_in.ndim < 2:
            return vector_in[None, :]
        else:
            return vector_in

class TraceObject(GeometryObject):
    """
    An object to use for raytracing.
    """

    def trace(self, ray):
        """
        Trace the given input ray in local coordinates.
        Individual components should reimplement this method.
        """

        return ray

class TraceLocalObject(TraceObject):
    """
    An object to use for ray tracing using local TraceObject coordinates.
    This approach can make the individual components simpler and easier to
    develop, however this also requires additional processing so it will
    slow things down.
    """

    def trace(self, ray_external):
        """
        Trace the given input ray through the system and return the output ray.
        """

        ray = self.ray_to_local(ray_external)
        ray = self.trace_local(ray)
        ray = self.ray_to_external(ray)

        return ray

    def trace_local(self, ray):
        """
        Trace the given input ray in local coordinates.
        Individual components should reimplement this method.
        """
        return ray
