# -*- coding: utf-8 -*-
"""
.. Authors:
    Novimir Pablant <npablant@pppl.gov>

Define the :class:`ShapePlane` class.
"""

import numpy as np
from copy import deepcopy


from xicsrt.tools.xicsrt_doc import dochelper
from xicsrt.optics._ShapeObject import ShapeObject
from xicsrt.tools import xicsrt_math as xm


@dochelper
class ShapePlane(ShapeObject):
    """
    A planar shape.
    This class defines intersections with a plane.
    """

    def intersect(self, rays):
        dist, mask = self.intersect_distance(rays)
        xloc = self.location_from_distance(rays, dist, mask)
        norm = self.intersect_normal(xloc, mask)

        return xloc, norm, mask

    def intersect_distance(self, rays):
        """
        Calculate the distance to an intersection with a plane.
        """

        O = rays['origin']
        D = rays['direction']
        m = rays['mask']

        if self.param['trace_local']:
            distance = np.full(m.shape, np.nan, dtype=np.float64)
            distance[m] = (np.dot((0 - O[m]), np.array([0., 0., 1.]))
                           / np.dot(D[m], np.array([0., 0., 1.])))
        else:
            distance = np.full(m.shape, np.nan, dtype=np.float64)
            distance[m] = (np.dot((self.param['origin'] - O[m]), self.param['zaxis'])
                           / np.dot(D[m], self.param['zaxis']))

        # Update the mask to only include positive distances.
        m &= (distance >= 0)

        return distance, m

    def intersect_normal(self, xloc, mask):
        """
        The planar optic is flat, so the normal direction is always the zaxis.
        """
        m = mask
        norm = np.full(xloc.shape, np.nan, dtype=np.float64)
        norm[m] = self.param['zaxis']
        return norm