# -*- coding: utf-8 -*-
"""
.. Authors:
    Novimir Pablant <npablant@pppl.gov>

Define the :class:`ShapeTorus` class.
"""
import numpy as np

from sys import float_info
from xicsrt.tools.xicsrt_doc import dochelper
from xicsrt.tools import xicsrt_math as xm
from xicsrt.tools import xicsrt_quartic as xq
from xicsrt.optics._ShapeObject import ShapeObject

@dochelper
class ShapeTorus(ShapeObject):
    """
    A Toroidal shape.
    This class defines intersections with a torus.
    """

    def default_config(self):
        """
        radius_major: float (1.0)
            The radius of curvature of the crystal in the toroidal (xaxis)
            direction. This is not the same as the geometric major radius of the
            axis of a toroid, which in our case would be r_major-r_minor.

        radius_minor: float (0.1)
            The radius of the curvature of the crystal in the poloidal (yaxis)
            direction. This is the same as the geometric minor radius of a
            toroid.
            
        convex : [bool, bool] ([False, False])
            Whether the major and minor curvature will be convex or concave.
            This has the effect of determining which of the four possible 
            intersections with the torus will be chosen. For a ray starting
            at infinity and passing through all four possible intersections
            with a torus, then the intersections correspond to:
              1 -> [True, True]
              2 -> [True, False]
              3 -> [False, False]
              4 -> [False, False]
        """

        config = super().default_config()
        config['radius_major']  = 1.0
        config['radius_minor']  = 0.2
        config['convex'] = [False, False]

        return config

    def initialize(self):
        super().initialize()
        
        r_minor = self.param['radius_minor']
        r_major = self.param['radius_major']

        if r_minor >= r_major:
            raise Exception(r'Cannot construct geometry with radius_major <= radius_minor.')

        # radius_major is the radius of curvature at the surface; here we find
        # the major and minor radii for the geometrical toroid construction
        # along with the geometrical center (in global coordinates).
        #
        # The indexing of the roots are specific to the algebraic quartic
        # solver in xicsrt_quartic.

        self.param['torus_minor'] = r_minor

        if np.all(self.param['convex'] == [False, False]):
            self.param['root_idx'] = 3
            self.param['torus_major'] = r_major - r_minor
            sign = 1
        elif np.all(self.param['convex'] == [False, True]):
            self.param['root_idx'] = 2
            self.param['torus_major'] = r_major + r_minor
            sign = 1
        elif np.all(self.param['convex'] == [True, False]):
            self.param['root_idx'] = 1
            self.param['torus_major'] = r_major + r_minor
            sign = -1
        elif np.all(self.param['convex'] == [True, True]):
            self.param['root_idx'] = 0
            self.param['torus_major'] = r_major - r_minor
            sign = -1
        else:
            raise Exception(f"Cannot be parse convex config option: {self.param['convex']}")

        self.param['center'] = self.param['origin'] + sign*r_major*self.param['zaxis']


    def intersect(self, rays):
        dist, mask = self.intersect_distance(rays)
        xloc = self.location_from_distance(rays, dist, mask)
        norm = self.intersect_normal(xloc, mask)

        return xloc, norm, mask


    def point_to_toroid(self, point_external, copy=False):
        """
        Convert a point into 'toroid' coordinates. In this coordinate system the
        origin is at the center of the geometrical torus.
        """
        return self.vector_to_local(point_external - self.param['center'], copy=copy)


    def intersect_distance(self, rays):
        """
        Calculate the distance to the intersection of the rays with the
        Toroidal optic.

        Programming Notes
        -----------------

        This calculation is done in 'torus' coordinates in which the center of
        the geometric torus is the origin.  The rays are converted to this
        coordinate system it the beginning of this method. Since we are only
        calculating the distance here, we don't need to convert back.
        """
        
        # Create short variable names for readability.
        O = rays['origin']
        D = rays['direction']
        m = rays['mask']

        r_minor = self.param['torus_minor']
        r_major = self.param['torus_major']

        O = self.point_to_toroid(O, copy=True)
        D = self.vector_to_local(D, copy=True)
        
        # defining resusable variables
        O_mag_sq = np.einsum('ij,ij->i', O, O)
        dot_OD = np.einsum('ij,ij->i', O, D)

        r_sq = r_major ** 2 + r_minor ** 2

        # The form of quartic equation to be solved is,
        # c0 ** t ^ 4 + c1 ** t ^ 3 + c2 ** t ^ 2 + c3 ** t + c4 = 0

        # Define the Axis of the toroid to be the y-axis.
        # 0 - x-axis, 1 - y-axis, 2 - z-axis
        axis_idx = 1
        ones = np.ones(len(m))

        # defining co-efficients
        # Coefficient of t^4
        c0 = ones**4
        # Coefficient of t^3
        c1 = 4 * ones**2 * dot_OD
        # Coefficient of t^2
        c2 = 4 * dot_OD**2 + 2 * O_mag_sq * ones**2 - 2 * r_sq * ones**2 + 4 * r_major**2 * D[:, axis_idx]**2
        # Coefficient of t^1
        c3 = 4 * dot_OD * (O_mag_sq - r_sq) + 8 * r_major**2 * D[:, axis_idx] * O[:, axis_idx]
        # Coefficient of t^0
        c4 = O_mag_sq**2 - 2 * r_sq * O_mag_sq + 4 * r_major**2 * O[:, axis_idx]**2 + (r_major**2 - r_minor**2)**2
    
        roots_0, roots_1, roots_2, roots_3 = xq.multi_quartic(c0, c1, c2, c3, c4)
        
        # neglecting complex & negative solution of the quartic equation    
        roots_0[roots_0.imag != 0] = np.nan
        roots_1[roots_1.imag != 0] = np.nan
        roots_2[roots_2.imag != 0] = np.nan
        roots_3[roots_3.imag != 0] = np.nan

        roots = np.stack((
            roots_0.real,
            roots_1.real,
            roots_2.real,
            roots_3.real), axis=1)

        # If we used xics_quartic.multi_quartic, then the roots will always
        # be in the expected order. If this solver gets replaced, some extra
        # logic may be necessary.
        # roots = np.sort(roots, axis=1)

        distances = roots[:, self.param['root_idx']]
        m[m] &= np.isfinite(distances[m]) & (distances[m] > 0.0)

        return distances, m
    

    def intersect_normal(self, xloc, mask):
        """
        Calculate the surface normal at each ray intersection point.

        Programming Notes
        -----------------

        This calculation is done in global coordinates.
        """
        m = mask
        normals = np.zeros(xloc.shape, dtype=np.float64)

        C = self.param['center']
        yaxis = np.cross(self.param['zaxis'], self.param['xaxis'])
        r_major     = self.param['torus_major']

        pt = xloc[m] - C

        # Projection of direction to geometrical center on xz plane.
        pt = pt - np.einsum('i,j->ij', np.einsum('ij,j->i', pt, yaxis), yaxis)
        pt_norm = xm.normalize(pt)

        # Location of the ray aiming point on toroidal axis.
        Q = C + r_major * pt_norm

        X_norm = (xloc[m] - Q)
        X_norm = xm.normalize(X_norm)

        normals[m] = X_norm
        
        return normals 
