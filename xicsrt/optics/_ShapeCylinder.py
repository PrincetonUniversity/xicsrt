# -*- coding: utf-8 -*-
"""
.. Authors:
   Conor Perks <cjperks@psfc.mit.edu>
   Novimir Pablant <npablant@pppl.gov>

Define the :class:`ShapeCylinder` class.
"""
import numpy as np

from xicsrt.tools.xicsrt_doc import dochelper
from xicsrt.tools import xicsrt_math as xm
from xicsrt.optics._ShapeObject import ShapeObject

@dochelper
class ShapeCylinder(ShapeObject):
    """
    A cylindrical shape.
    This class defines intersections with a cylinder.
    """

    def default_config(self):
        """
        radius : float (1.0)
          The radius of the cylinder.

        convex : bool (False)
          If True the optic will have a convex curvature, if False the surface
          will have a concave curvature (the default).
        """
        config = super().default_config()
        config['radius'] = 1.0
        config['convex'] = False
        return config

    def initialize(self):
        super().initialize()
        # Finds center location of the cylinder geometry (in global coordinates).
        if self.param['convex']:
            sign = -1
        else:
            sign = 1
        self.param['center'] = sign*self.param['radius'] * self.param['zaxis'] + self.param['origin']

    def intersect(self, rays):
        dist, mask = self.intersect_distance(rays)
        xloc = self.location_from_distance(rays, dist, mask)
        norm = self.intersect_normal(xloc, mask)

        return xloc, norm, mask

    def intersect_distance(self, rays):
        """
        Calulate the distance to the intersection of the rays with the
        cylindrical optic.

        This calculation is copied from:
        https://mrl.cs.nyu.edu/~dzorin/rend05/lecture2.pdf
        """

        # Create short variable names for readability.
        O = rays['origin']
        D = rays['direction']
        m = rays['mask']

        # Initializes arrays to store intersection variables
        distance = np.full(m.shape, np.nan, dtype=np.float64)

        # Create short variable names for readability.
        pa = self.param['center']
        va = self.param['xaxis']
        r = self.param['radius']

        # Calculate quadratic formula coefficients for t
        #
        # The cylinter axis parameterized as pa + t_axis*va
        # Below einsum is used to calculate dot products.

        # Length between ray origin and cylinder center
        dp = O-pa
        dot_Dva = np.einsum('ij,j->i', D, va)
        dot_dpva = np.einsum('ij,j->i', dp, va)
        A1 = D - dot_Dva[:,None]*va[None,:]
        B1 = dp - dot_dpva[:,None]*va[None,:]

        # Coefficients for f(t) = A*t**2 + B*t + C = 0
        A = np.einsum('ij,ij->i', A1, A1)
        B = 2*np.einsum('ij,ij->i', A1, B1)
        C = np.einsum('ij,ij->i', B1, B1)-r**2

        # Solves the quadratic formula for t
        dis = B**2-4*A*C

        # Update the mask to exclude rays that don't hit the cylinder.
        m[m] &= (dis[m] >= 0)

        t_0      = np.empty(m.shape, dtype=np.float64)
        t_1      = np.empty(m.shape, dtype=np.float64)
        t_0[m] = (-B[m] - np.sqrt(dis[m]))/(2*A[m])
        t_1[m] = (-B[m] + np.sqrt(dis[m]))/(2*A[m])

        # Distance traveled by the ray before hitting the optic surface with
        # the chosen curvature.
        if self.param['convex']:
            distance[m] = np.where(t_0[m] < t_1[m], t_0[m], t_1[m])
        else:
            distance[m] = np.where(t_0[m] > t_1[m], t_0[m], t_1[m])

        return distance, m

    def intersect_normal(self, xloc, mask):
        """
        Calculates the normal vector from the intersection with the cylinder surface
        """

        # Create short variable names for readability.
        m = mask
        pa = self.param['center']
        va = self.param['xaxis']

        # Initializes quantities
        norm = np.full(xloc.shape, np.nan, dtype=np.float64)
        pa_proj = np.full(xloc.shape, np.nan, dtype=np.float64)

        # A normal vector on a cylinder is orthogonal to the cylinder axis
        # As such, we need to project the 'center' of the cylinder onto the same
        # plane as the normal vector
        dummy = np.einsum('ij,j->i', pa-xloc[m], va)
        pa_proj[m] = pa - dummy[:,None]*va[None,:]

        # Calculates the normal vector from intersection point to cylinder center
        norm[m] = xm.normalize(pa_proj[m] - xloc[m])
        return norm
