# -*- coding: utf-8 -*-
"""
.. Authors:
    Novimir pablant <npablant@pppl.gov>

Define the :class:`ShapeObject` class.
"""

import numpy as np
from copy import deepcopy

from xicsrt.tools.xicsrt_doc import dochelper
from xicsrt.optics._TraceObject import TraceObject
from xicsrt.tools import xicsrt_math as xm


@dochelper
class ShapeObject(TraceObject):
    """
    The base class for intersections of rays with surfaces in XICSRT.

    This base class should be used to define intersections with various shapes
    such as planes, spheres and toroids.
    """

    def intersect(self, rays):
        """
        Calculate the location and normal of the surface at the ray
        intersections.

        Specific shape objects can reimplement this method, or alternatively
        reimplement the :meth:`intersect_location` and :meth:`intersect_normal`
        methods.

        Programming Notes
        -----------------
        Currently the expectation is that intersect has made copies of
        ray['origin'] and ray['mask'] before any calculations. This is done for
        two reasons: 1. provide more information for the interactions. 2. it is
        much easier to read and understand the code this way. From a memory
        efficiency standpoint it would be better to modify these arrays in place
        instead.
        """
        xloc, mask = self.intersect_location(rays)
        norm = self.intersect_normal(rays, xloc, mask)

        return xloc, norm, mask

    def intersect_location(self, rays):
        """
        Calculate the surface location at the ray intersections.

        This base-class just returns a copy of the ray origin.
        """
        xloc = rays['origin'].copy()
        mask = rays['mask'].copy()
        return xloc, mask

    def intersect_normal(self, xloc, mask):
        """
        Calculate the surface normal at the ray intersection locations.

        Normals are not defined for this base-class; an array of np.nan will
        always be returned.
        """
        norm = np.full(xloc.shape, np.nan, dtype=np.float64)
        return norm

    def location_from_distance(self, rays, dist, mask=None):
        """
        Calculate 3D locations given a distance along the rays.
        """
        if mask is None: mask = rays['mask']
        O = rays['origin']
        D = rays['direction']
        m = mask

        X = np.full(O.shape, np.nan, dtype=np.float64)
        X[m] = O[m] + D[m] * dist[m, np.newaxis]

        return X